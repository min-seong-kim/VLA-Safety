"""Read-only activation hooks and append-only, timestep-aligned feature storage."""

from collections import defaultdict
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch


class ActivationCapture:
    def __init__(self):
        self.handles = []
        self.regions = {}
        self.region_names = {}
        self.enabled = True
        self.reset(False)

    def reset(self, full_tokens=False):
        self.values = defaultdict(list)
        self.full = defaultdict(list)
        self.metadata = {}
        self.full_tokens = full_tokens

    def record(self, key, tensor, kind="vlm"):
        if not self.enabled:
            return
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            return
        if tensor.ndim == 2:
            tensor = tensor[:, None, :]
        if tensor.ndim != 3:
            raise ValueError(f"Unexpected activation shape: {key} {tensor.shape}")
        value = tensor.detach()
        if kind == "vlm" and value.shape[1] == 1:
            key = key.replace("vlm_", "autoregressive_")
            pooled = value[:, 0, :].unsqueeze(1)
        else:
            spans = self.regions.get(kind, [(0, value.shape[1])])
            chunks = []
            for start, stop in spans:
                if stop <= start or start >= value.shape[1]:
                    chunks.append(torch.zeros_like(value[:, 0, :], dtype=torch.float32))
                else:
                    chunks.append(value[:, start:min(stop, value.shape[1]), :].float().mean(dim=1))
            pooled = torch.stack(chunks, dim=1)
        self.values[key].append(pooled.to(torch.float16))
        if self.full_tokens:
            self.full[key].append(value.to(torch.float16))

    def layer_hooks(self, layers, prefix, kind="vlm", projections=False):
        for index, module in enumerate(layers):
            name = f"{prefix}/layer_{index:02d}"
            self.handles.append(module.register_forward_hook(
                lambda m, args, output, key=name, k=kind: self.record(key, output, k)))
            if index == 0:
                def embedding_hook(m, args, kwargs, key=prefix + "_input/layer_00", k=kind):
                    hidden = args[0] if args else kwargs["hidden_states"]
                    self.record(key, hidden, k)
                self.handles.append(module.register_forward_pre_hook(embedding_hook, with_kwargs=True))
            if projections:
                for name_part in ["k_proj", "v_proj"]:
                    projected = getattr(module.self_attn, name_part)
                    key = prefix.replace("hidden", name_part) + f"/layer_{index:02d}"
                    self.handles.append(projected.register_forward_hook(
                        lambda m, args, output, key=key, k=kind: self.record(key, output, k)))

    def vision_hooks(self, model):
        groups = defaultdict(list)
        for name, module in model.named_modules():
            if re.search(r"vision.*(?:encoder\.layers|blocks)\.\d+$", name):
                parent, index = name.rsplit(".", 1)
                groups[parent].append((int(index), module))
        for index, (parent, modules) in enumerate(sorted(groups.items())):
            self.layer_hooks([m for _, m in sorted(modules)], f"vision_{index}_hidden", "vision")
            self.metadata.setdefault("vision_modules", {})[str(index)] = parent
        self.vision_module_names = {str(i): k for i, k in enumerate(sorted(groups))}

    @staticmethod
    def combine(values):
        grouped = defaultdict(list)
        for key, tensors in values.items():
            prefix, layer = key.rsplit("/", 1)
            grouped[prefix].append((layer, torch.stack(tensors, dim=0)))
        result = {}
        for prefix, layers in grouped.items():
            # [forward_call, layer, batch_or_camera, region_or_token, hidden]
            stacked = torch.stack([v for _, v in sorted(layers)], dim=1)
            array = stacked.cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Non-finite extracted features: " + prefix)
            result[prefix] = array
        return result

    def extract(self):
        result = {"pooled/" + k: v for k, v in self.combine(self.values).items()}
        if self.full_tokens:
            result.update({"initial_full_tokens/" + k: v for k, v in self.combine(self.full).items()})
        return result


class TimestepFeatureWriter:
    def __init__(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        self.file = h5py.File(path, "w")
        self.file.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        self.file.attrs["committed_steps"] = 0
        self.file.attrs["axis_order"] = "timestep, forward_call, layer, batch_or_camera, region_or_token, hidden"

    def append(self, step, arrays, metadata):
        assert int(self.file.attrs["committed_steps"]) == step
        for name, array in arrays.items():
            array = np.asarray(array)
            if name.startswith("initial_full_tokens/"):
                assert step == 0
                self.file.create_dataset(name, data=array, compression="lzf", shuffle=True)
                continue
            if name not in self.file:
                parent, _, leaf = name.rpartition("/")
                group = self.file.require_group(parent) if parent else self.file
                group.create_dataset(leaf, shape=(0, *array.shape), maxshape=(None, *array.shape),
                                     dtype=array.dtype, chunks=(1, *array.shape), compression="lzf", shuffle=True)
            dataset = self.file[name]
            if dataset.shape[1:] != array.shape:
                raise ValueError(f"Feature shape changed for {name}: {dataset.shape} -> {array.shape}")
            dataset.resize(step + 1, axis=0)
            dataset[step] = array
        if "step_metadata" not in self.file:
            self.file.create_dataset("step_metadata", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        self.file["step_metadata"].resize(step + 1, axis=0)
        self.file["step_metadata"][step] = json.dumps(metadata, ensure_ascii=False)
        self.file.attrs["committed_steps"] = step + 1
        self.file.flush()

    def close(self):
        self.file.close()
