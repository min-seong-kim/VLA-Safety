"""Train only action-expert LoRA on replay-validated neutral demonstrations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from motor_lora import (
    flow_loss, frozen_parameter_digest, frozen_prefix, install_lora, load_policy,
    make_input, save_adapter, tensor_batch,
)
from rollout_common import ROOT, write_json


class Demonstrations:
    def __init__(self, manifest, policy, seed, allowed_split="train"):
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.rows = [row for row in self.rows if row["accepted"] and row["split"] == allowed_split]
        if not self.rows:
            raise ValueError(f"No accepted {allowed_split} episodes in {manifest}")
        self.by_asset = {}
        for row in self.rows:
            self.by_asset.setdefault(row["asset"], []).append(row)
        self.assets = sorted(self.by_asset)
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(max_workers=4)

    def read(self, request):
        row, step = request
        with h5py.File(ROOT / row["path"], "r") as data:
            spec = json.loads(data.attrs["spec_json"])
            return make_input(self.policy, data["observations/main_rgb_raw"][step],
                              data["observations/wrist_rgb_raw"][step], data["observations/state"][step],
                              spec["instruction"], data["actions"][step:step + 10])

    def batch(self, size):
        requests = []
        for _ in range(size):
            rows = self.by_asset[self.assets[int(self.rng.integers(len(self.assets)))]]
            row = rows[int(self.rng.integers(len(rows)))]
            step = int(self.rng.integers(row["steps"] - 9))
            requests.append((row, step))
        return list(self.pool.map(self.read, requests))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=812)
    parser.add_argument("--calibration-smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "training.jsonl").exists():
        raise FileExistsError("Use an isolated output directory; existing training logs are preserved")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy, config = load_policy()
    model = policy._model
    lora = install_lora(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    data = Demonstrations(args.manifest, policy, args.seed,
                          allowed_split="calibration" if args.calibration_smoke else "train")
    fixed_items = data.batch(2)
    fixed_obs, _ = tensor_batch(fixed_items)
    before = frozen_parameter_digest(model)
    with torch.no_grad():
        hidden_before = frozen_prefix(model, fixed_obs)[3].cpu().clone()
    metadata = {"experiment": "B", "base_checkpoint": "pi05_libero_pytorch", "lora": lora,
                "seed": args.seed, "steps_planned": args.steps, "batch_size": args.batch_size,
                "learning_rate_peak": args.learning_rate, "optimizer": "AdamW", "weight_decay": .01,
                "gradient_clip": 1., "normalization": "unchanged original pi05_libero checkpoint statistics",
                "image_augmentation": False, "sampling": "uniform asset, episode, valid full 10-step chunk start",
                "loss": "official flow-matching target, frozen prefix KV, MSE over all 32 padded action dimensions",
                "training_episodes": len(data.rows), "training_assets": data.assets,
                "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "frozen_parameter_sha256_before": before, "probe_trained": False,
                "calibration_smoke_only": args.calibration_smoke,
                "student_privileged_state_inputs": False, "safety_labels_in_training": False}
    write_json(args.out / "config.json", metadata)
    started = time.monotonic()
    prefetch = ThreadPoolExecutor(max_workers=1)
    next_batch = prefetch.submit(data.batch, args.batch_size)
    warmup = min(100, max(1, args.steps // 10))
    with (args.out / "training.jsonl").open("a", buffering=1) as log:
        for step in range(1, args.steps + 1):
            items = next_batch.result()
            if step < args.steps:
                next_batch = prefetch.submit(data.batch, args.batch_size)
            observation, actions = tensor_batch(items)
            factor = min(1., step / warmup) * (.1 + .9 * .5 * (1 + math.cos(math.pi * max(0, step - warmup) /
                                                                                max(1, args.steps - warmup))))
            optimizer.param_groups[0]["lr"] = args.learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            loss = flow_loss(model, observation, actions)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            loss.backward()
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            row = {"step": step, "loss": loss.item(), "gradient_norm": float(norm),
                   "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - started}
            log.write(json.dumps(row) + "\n")
            if step == 1 or step % 20 == 0:
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                checkpoint = args.out / f"adapter_step_{step:05d}.pt"
                save_adapter(checkpoint, model, {**metadata, "step": step})
                write_json(args.out / "progress.json", {**row, "checkpoint": str(checkpoint)})
    after = frozen_parameter_digest(model)
    with torch.no_grad():
        hidden_after = frozen_prefix(model, fixed_obs)[3].cpu()
    audit = {"all_original_weights_unchanged": before == after,
             "fixed_input_vlm_hidden_exact": torch.equal(hidden_before, hidden_after),
             "frozen_parameter_sha256_after": after, "probe_trained": False,
             "trainable_parameters": lora["trainable_parameters"], "training_seconds": time.monotonic() - started}
    write_json(args.out / "freeze_audit.json", audit)
    assert audit["all_original_weights_unchanged"] and audit["fixed_input_vlm_hidden_exact"]
    print("TRAINING_COMPLETE", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
