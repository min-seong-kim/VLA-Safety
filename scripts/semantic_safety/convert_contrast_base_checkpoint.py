"""Convert the official base checkpoint, checking every missing/tied parameter."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "checkpoints/pi05_base_pytorch"


def main():
    if (DEST / "conversion_verified.json").exists():
        print("Verified base conversion already exists.")
        return
    assert not DEST.exists(), "Refusing to replace an existing unverified checkpoint"
    spec = importlib.util.spec_from_file_location("official_convert", ROOT / "openpi/examples/convert_jax_model_to_pytorch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from openpi.training import config

    original = torch.nn.Module.load_state_dict
    audits = []

    def checked(model, state, strict=True, **kwargs):
        result = original(model, state, strict=strict, **kwargs)
        if type(model).__name__ == "PI0Pytorch":
            assert not result.unexpected_keys, result.unexpected_keys
            # The official action policy uses transformer bodies and velocity heads,
            # never the two HF text-generation heads absent from the JAX policy.
            unused = {"paligemma_with_expert.paligemma.lm_head.weight",
                      "paligemma_with_expert.gemma_expert.lm_head.weight"}
            assert set(result.missing_keys) <= unused, result.missing_keys
            audits.append(dict(supplied_tensors=len(state), unused_text_generation_heads=result.missing_keys,
                               unexpected_keys=result.unexpected_keys, all_backbone_and_action_parameters_supplied=True))
        return result

    temporary = DEST.with_name(DEST.name + ".converting")
    assert not temporary.exists(), "Incomplete conversion needs inspection"
    torch.nn.Module.load_state_dict = checked
    torch.manual_seed(20261116)
    try:
        module.convert_pi0_checkpoint(str(ROOT / "checkpoints/pi05_base"), "bfloat16", str(temporary),
                                      config.get_config("pi05_libero").model)
    finally:
        torch.nn.Module.load_state_dict = original
    assert len(audits) == 1, audits
    receipt = dict(source="checkpoints/pi05_base", official_converter="openpi/examples/convert_jax_model_to_pytorch.py",
                   weights_are_base_not_LIBERO=True, policy_training=False, audits=audits,
                   controlled_inference=dict(action_horizon=10, discrete_state_input=False,
                                             reference="common LIBERO input/readout condition, not default base deployment"))
    digest = hashlib.sha256()
    with (temporary / "model.safetensors").open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""): digest.update(chunk)
    receipt["safetensors_sha256"] = digest.hexdigest()
    (temporary / "conversion_verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.rename(DEST)
    print(json.dumps(receipt))


if __name__ == "__main__": main()
