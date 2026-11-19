"""Measure which contextualized VLM positions are pooled, without changing the attention graph."""

from contextlib import ExitStack

import h5py
import numpy as np
import torch

from ae_output_nullspace_pilot import inputs, read
from contrast_initial_linear import digest
from extract_representation_linear import masked_mean
from modality_ablation import BATCH, COMMON, REFERENCE, RUN, VLM, load_policy
from motor_lora import frozen_prefix
from representation_common import immutable_json


def extract(name):
    rows = read(REFERENCE / "manifest.json")["rows"]
    policy = load_policy(name)
    net = policy._model
    with ExitStack() as stack:
        original = stack.enter_context(h5py.File(REFERENCE / "features" / (name + ".h5")))
        files = {}
        for condition in ("read_text", "read_image"):
            folder = RUN / "conditions" / condition
            (folder / "features").mkdir(exist_ok=True, parents=True)
            manifest = folder / "manifest.json"
            if not manifest.exists():
                manifest.symlink_to(REFERENCE / "manifest.json")
            f = stack.enter_context(h5py.File(folder / "features" / (name + ".h5"), "a"))
            signature = digest([original.attrs["signature"], condition, "intact_contextualized_pool_v1"])
            if VLM not in f:
                f.create_dataset(VLM, (len(rows), 2048), dtype="float32", compression="lzf")
                f.create_dataset("written", (len(rows),), dtype="bool")
                f.attrs.update(signature=signature, complete=False)
            assert signature == f.attrs["signature"]
            files[condition] = f
        hidden = []
        def save(module, args, output):
            hidden.append(output[0] if isinstance(output, (tuple, list)) else output)
        stack.callback(net.paligemma_with_expert.paligemma.language_model.layers[17].register_forward_hook(save).remove)
        with torch.inference_mode():
            for start in range(0, len(rows), BATCH):
                group = rows[start:start+BATCH]
                sl = slice(start, start+len(group))
                if all(f["written"][sl].all() for f in files.values()):
                    continue
                _, batch, hashes = inputs(policy, group)
                assert hashes == [h.decode() for h in original["input_sha256"][sl]]
                _, mask, _, _ = frozen_prefix(net, batch)
                assert len(hidden) == 1 and mask.shape[1]-batch.tokenized_prompt.shape[1] == 768
                states = hidden.pop()
                np.testing.assert_array_equal(masked_mean(states, mask).cpu().numpy(), original[VLM][sl])
                for condition, f in files.items():
                    keep = mask.clone()
                    if condition == "read_text":
                        keep[:, :768] = False
                    else:
                        keep[:, 768:] = False
                    f[VLM][sl] = masked_mean(states, keep).cpu().numpy()
                    f.flush()
                    f["written"][sl] = True
                    f.flush()
            for f in files.values():
                assert f["written"][:].all()
                f.attrs["complete"] = True
    print(name, "complete", flush=True)


if __name__ == "__main__":
    immutable_json(RUN / "pooling_control_protocol.json", dict(models=list(COMMON), site=VLM,
        study="supplementary readout control", arms=["read_text", "read_image"], unchanged_attention=True,
        original_reference="unmasked VLM pool must match the archived features bit for bit on every row",
        probes="same held-out folds, both the frozen and refitted probes", rollout_effect="readout only"))
    for m in COMMON:
        extract(m)
