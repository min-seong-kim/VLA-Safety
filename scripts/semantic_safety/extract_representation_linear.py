"""Cache pooled residual and velocity features from identical offline inputs."""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch

from motor_lora import load_policy, make_input, tensor_batch
from representation_common import FLOW_STEPS, MODELS, ROOT, RUN, digest, feature_sites, immutable_json, milestone, samples
from rollout_common import write_json
from semantic_pairs_v6 import sha256
from v6_scoped_lora import install


def masked_mean(value, mask):
    assert value.shape[:2] == mask.shape and bool(mask.any(dim=1).all())
    return (value.float() * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)


class Capture:
    def __init__(self, model):
        self.model, self.enabled, self.handles = model, True, []
        self.original_prefix, self.original_denoise = model.embed_prefix, model.denoise_step
        self.reset()

        def prefix(*args, **kwargs):
            output = self.original_prefix(*args, **kwargs)
            if self.enabled:
                self.mask = output[1].detach()
            return output

        def denoise(*args, **kwargs):
            if self.enabled:
                self.flow += 1
            return self.original_denoise(*args, **kwargs)

        model.embed_prefix, model.denoise_step = prefix, denoise
        for index, layer in enumerate(model.paligemma_with_expert.paligemma.language_model.layers):
            self.handles.append(layer.register_forward_hook(
                lambda module, args, output, i=index: self.block(output, "vlm", i)))
        for index, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers):
            self.handles.append(layer.register_forward_hook(
                lambda module, args, output, i=index: self.block(output, "ae", i)))
        self.handles.append(model.action_out_proj.register_forward_pre_hook(
            lambda module, args: self.head(args[0], "velocity_pre")))
        self.handles.append(model.action_out_proj.register_forward_hook(
            lambda module, args, output: self.head(output, "velocity_post")))

    def reset(self):
        self.values, self.flow, self.mask = {}, 0, None

    def put(self, key, value):
        assert key not in self.values, f"Duplicate feature capture: {key}"
        self.values[key] = value.detach()

    def block(self, output, module, layer):
        if not self.enabled or (module == "ae" and self.flow not in FLOW_STEPS):
            return
        value = output[0] if isinstance(output, (tuple, list)) else output
        if module == "vlm":
            self.put(f"vlm/layer_{layer:02d}", masked_mean(value, self.mask))
        else:
            assert value.shape[1] == 10
            self.put(f"ae/flow_{self.flow:02d}/layer_{layer:02d}", value.float().mean(dim=1))

    def head(self, value, kind):
        if self.enabled and self.flow in FLOW_STEPS:
            assert value.shape[1] == 10
            self.put(f"{kind}/flow_{self.flow:02d}", value.float().mean(dim=1))

    def arrays(self):
        expected = dict(feature_sites())
        assert self.flow == 10 and set(self.values) == set(expected), (self.flow, sorted(self.values))
        result = {}
        for name, value in self.values.items():
            array = value.cpu().numpy()
            assert array.shape[1] == expected[name] and np.isfinite(array).all(), name
            result[name] = array
        return result


class Inputs:
    def __init__(self, policy):
        self.policy, self.files = policy, OrderedDict()

    def read(self, row):
        path = row["path"]
        if path not in self.files:
            if len(self.files) >= 20:
                _, old = self.files.popitem(last=False)
                old.close()
            self.files[path] = h5py.File(ROOT / path, "r")
        source = self.files[path]
        index = row["input_index"]
        arrays = [source["policy_inputs/" + name][index] for name in ("main_rgb_raw", "wrist_rgb_raw", "state")]
        seed = row["noise_seed"] + row["timestep"]
        hasher = hashlib.sha256()
        for value in arrays:
            hasher.update(value.tobytes())
        hasher.update(row["instruction"].encode())
        hasher.update(str(seed).encode())
        item = make_input(self.policy, *arrays, row["instruction"])
        noise = np.random.default_rng(seed).standard_normal((10, 32)).astype(np.float32)
        return item, noise, hasher.hexdigest()

    def batch(self, rows):
        loaded = [self.read(row) for row in rows]
        batch, _ = tensor_batch([value[0] for value in loaded])
        noise = torch.from_numpy(np.stack([value[1] for value in loaded])).cuda()
        return batch, noise, [value[2] for value in loaded]


def activate(model, architecture, scope):
    named = dict(model.named_parameters())
    path = ROOT / "checkpoints/pi05_libero_expansion_v5_lora_v1" / scope / "adapter.pt"
    with torch.no_grad():
        for name in architecture["trainable_parameter_names"]:
            named[name].zero_()
        if scope != "baseline":
            package = torch.load(path, map_location="cpu", weights_only=False)
            assert package["metadata"]["step"] == 16000 and package["metadata"]["lora"]["scope"] == scope
            for name, value in package["state_dict"].items():
                named[name].copy_(value)
    model.requires_grad_(False)
    model.eval()
    return {"scope": scope, "adapter_sha256": sha256(path) if scope != "baseline" else None,
            "base_checkpoint": "checkpoints/pi05_libero_pytorch", "all_parameters_frozen": True}


def extract(args):
    rows = samples(args.source, args.view, args.limit)
    if args.limit:
        assert args.out_root != RUN / "features", "Use a separate output root for smoke runs"
        rows = rows[:args.limit]
    folder = args.out_root / args.source / args.view
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder / "rows.json", rows)
    torch.set_num_threads(8)
    policy, cfg = load_policy("pi05_libero")
    model = policy._model
    assert cfg.model.action_horizon == 10 and cfg.model.action_dim == 32
    architecture = install(model, "all")
    capture, loader = Capture(model), Inputs(policy)
    baseline_check = None
    for scope in args.models:
        started = time.monotonic()
        meta = activate(model, architecture, scope)
        batch, noise, _ = loader.batch(rows[:min(4, len(rows))])
        capture.enabled = False
        with torch.inference_mode():
            expected = model.sample_actions("cuda", batch, noise=noise.clone(), num_steps=10)
        capture.enabled = True
        capture.reset()
        with torch.inference_mode():
            actual = model.sample_actions("cuda", batch, noise=noise.clone(), num_steps=10)
        arrays = capture.arrays()
        difference = float((actual - expected).abs().max())
        assert difference == 0., f"Hooks changed output: {difference}"
        if scope == "baseline":
            baseline_check = {key: value.copy() for key, value in arrays.items() if key.startswith("vlm/")}
        if scope == "ae" and baseline_check is not None:
            assert all(np.array_equal(arrays[key], value) for key, value in baseline_check.items())
        audit = {**meta, "hook_action_max_abs_difference": difference, "sites": len(arrays),
                 "flow_evaluations": capture.flow, "valid_prefix_tokens": capture.mask.sum(dim=1).tolist(),
                 "feature_dtype": "float32", "input_rows_digest": digest(rows)}
        write_json(folder / f"{scope}_hook_audit.json", audit)
        output = folder / f"{scope}.h5"
        with h5py.File(output, "a") as out:
            if "written" not in out:
                out.attrs.update(rows_digest=digest(rows), model_json=json.dumps(meta), complete=False)
                out.create_dataset("written", shape=(len(rows),), dtype=np.bool_)
                out.create_dataset("input_sha256", shape=(len(rows),), dtype="S64")
                for name, width in feature_sites():
                    out.create_dataset(name, shape=(len(rows), width), dtype="float32",
                        chunks=(min(16, len(rows)), width), compression="lzf", shuffle=True)
            assert out.attrs["rows_digest"] == digest(rows)
            assert json.loads(out.attrs["model_json"]) == meta
            # The initial states are identical across reference sources by collection audit.
            initial = args.out_root / "both/initial" / f"{scope}.h5"
            if args.view == "all" and initial.exists():
                initial_rows = json.loads(initial.with_name("rows.json").read_text())
                index = {row["episode_id"]: i for i, row in enumerate(initial_rows)}
                with h5py.File(initial, "r") as cached:
                    assert bool(cached.attrs["complete"])
                    written = out["written"][:]
                    targets = np.asarray([i for i, row in enumerate(rows) if row["timestep"] == 0 and not written[i]])
                    if len(targets):
                        old = [index[rows[i]["episode_id"]] for i in targets]
                        for name, _ in feature_sites():
                            out[name][targets] = cached[name][:][old]
                        out["input_sha256"][targets] = cached["input_sha256"][:][old]
                        out["written"][targets] = True
                    out.flush()
            # Preserve batch membership across interruptions, including partial commits.
            work_order = np.asarray([i for i, row in enumerate(rows)
                if not (args.view == "all" and initial.exists() and row["timestep"] == 0)])
            written = out["written"][:]
            for offset in range(0, len(work_order), args.batch_size):
                indices = work_order[offset:offset + args.batch_size]
                if written[indices].all():
                    continue
                batch, noise, hashes = loader.batch([rows[i] for i in indices])
                capture.reset()
                with torch.inference_mode():
                    model.sample_actions("cuda", batch, noise=noise, num_steps=10)
                for name, value in capture.arrays().items():
                    out[name][indices] = value
                out["input_sha256"][indices] = np.asarray(hashes, dtype="S64")
                out.flush()
                out["written"][indices] = True
                out.flush()
                written[indices] = True
                if offset % (args.batch_size * 20) == 0:
                    write_json(folder / f"{scope}_progress.json", {"completed": int(out["written"][:].sum()),
                        "total": len(rows), "seconds": time.monotonic() - started})
            assert out["written"][:].all()
            out.attrs["complete"] = True
        if scope != "baseline":
            with h5py.File(folder / "baseline.h5", "r") as baseline, h5py.File(output, "r") as current:
                assert np.array_equal(baseline["input_sha256"][:], current["input_sha256"][:])
                if scope == "ae":
                    assert all(np.array_equal(baseline[name][:], current[name][:])
                               for name, _ in feature_sites() if name.startswith("vlm/"))
                    audit["all_vlm_features_exact_to_baseline"] = True
                    write_json(folder / f"{scope}_hook_audit.json", audit)
        result = {"complete": True, "model": scope, "source": args.source, "view": args.view,
                  "inputs": len(rows), "seconds": time.monotonic() - started, "sites": len(arrays)}
        write_json(folder / f"{scope}_completion.json", result)
        print(json.dumps(result), flush=True)
    if not args.limit:
        milestone(f"features_{args.source}_{args.view}",
                  f"{args.source} {args.view}: 네 모델 representation 추출·검증 완료",
                  {"inputs_per_model": len(rows), "models": args.models, "sites": 78})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", choices=["both", "teacher"], required=True)
    parser.add_argument("--view", choices=["initial", "all"], required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-root", type=Path, default=RUN / "features")
    arguments = parser.parse_args()
    try:
        extract(arguments)
    except Exception:
        write_json(RUN / "experiment_error.json", {"stage": "feature_extraction", "source": arguments.source,
            "view": arguments.view, "error": traceback.format_exc()})
        from experiment_notifications import notify_progress
        if not arguments.limit:
            notify_progress(RUN)
        raise
