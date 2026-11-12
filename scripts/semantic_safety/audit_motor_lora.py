"""Verify adapter isolation, frozen-prefix parity, and the training objective."""

import argparse
import json
from pathlib import Path

import h5py
import torch

from motor_lora import (
    flow_loss, frozen_parameter_digest, frozen_prefix, install_lora,
    load_policy, make_input, tensor_batch,
)
from rollout_common import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/motor_lora_b_v1/preflight/lora_smoke.json"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(712)
    policy, _ = load_policy()
    model = policy._model
    with h5py.File(args.demo) as f:
        spec = json.loads(f.attrs["spec_json"])
        item = make_input(policy, f["observations/main_rgb_raw"][0], f["observations/wrist_rgb_raw"][0],
                          f["observations/state"][0], spec["instruction"], f["actions"][:10])
    obs, actions = tensor_batch([item])
    noise = model.sample_noise(actions.shape, "cuda")
    time = torch.tensor([.5], device="cuda")
    with torch.no_grad():
        hidden_before = frozen_prefix(model, obs)[3].clone()
        original = flow_loss(model, obs, actions, noise=noise, time=time)
        preprocess = model._preprocess_observation
        model._preprocess_observation = lambda observation, train=True: preprocess(observation, train=False)
        joint = model(obs, actions, noise=noise, time=time).mean()
        model._preprocess_observation = preprocess
    parity_difference = abs(original.item() - joint.item())
    assert parity_difference < max(.001, .01 * abs(joint.item())), (original, joint)
    meta = install_lora(model)
    before = frozen_parameter_digest(model)
    with torch.no_grad():
        initialized = flow_loss(model, obs, actions, noise=noise, time=time)
    assert torch.equal(original, initialized), (original, initialized)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for step in range(2):
        optimizer.zero_grad()
        loss = flow_loss(model, obs, actions, noise=noise, time=time)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        gradients = [float(p.grad.norm()) for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and any(value > 0 for value in gradients)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        losses.append(loss.item())
        print("AUDIT_STEP", step, loss.item(), flush=True)
    after = frozen_parameter_digest(model)
    with torch.no_grad():
        hidden_after = frozen_prefix(model, obs)[3]
    assert before == after
    assert torch.equal(hidden_before, hidden_after)
    report = {"status": "passed", "trainable_parameters": meta["trainable_parameters"],
              "expert_linear_modules": len(meta["target_modules"]), "zero_init_loss_exact": True,
              "all_frozen_parameter_hash_unchanged": True, "fixed_input_vlm_hidden_exact": True,
              "stock_joint_loss": joint.item(), "cached_prefix_loss": original.item(),
              "loss_absolute_difference": parity_difference, "gradient_isolation": True,
              "two_step_losses": losses, "frozen_parameter_sha256": before,
              "probe_trained": False, "smoke_adapter_discarded": True}
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
