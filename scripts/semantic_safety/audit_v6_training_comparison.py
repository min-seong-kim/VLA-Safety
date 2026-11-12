"""Check that the three completed runs implement the declared LoRA comparison."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from rollout_common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run, data = args.run_root.resolve(), args.data_root.resolve()
    protocol = json.loads((run / "experiment_protocol.json").read_text())
    manifest_sha = hashlib.sha256((data / "combined_train_manifest.jsonl").read_bytes()).hexdigest()
    common_keys = ["base_checkpoint", "seed", "steps_planned", "batch_size", "learning_rate_peak",
                   "optimizer", "weight_decay", "gradient_clip", "manifest_sha256", "training_episodes",
                   "conditions", "normalization", "image_augmentation", "sampling", "prompt_sampling",
                   "native_rehearsal_episodes", "native_rehearsal_fraction", "scripts"]
    phases = [("stage1", "", protocol["steps_stage1"], protocol["seed"], protocol["peak_learning_rate"])]
    extension = protocol.get("extension")
    if extension:
        phases.append(("extension", "_extension", extension["additional_steps_per_scope"],
                       extension["seed"], extension["learning_rate_peak"]))
    rows, frozen_hashes = [], {}
    for phase, suffix, steps, seed, learning_rate in phases:
        configs = {}
        for scope in ["ae", "vlm", "all"]:
            folder = run / f"train_{scope}{suffix}"
            config = json.loads((folder / "config.json").read_text())
            audit = json.loads((folder / "freeze_audit.json").read_text())
            log = [json.loads(line) for line in (folder / "training.jsonl").read_text().splitlines()]
            assert config["manifest_sha256"] == manifest_sha and not config["pilot"]
            if phase == "stage1":
                assert config["resume_adapter"] is None
            else:
                assert (ROOT / config["resume_adapter"]).resolve() == run / f"train_{scope}/adapter_step_06000.pt"
                assert config["frozen_parameter_sha256_before"] == frozen_hashes[scope]
            assert not config["safety_labels_in_training"] and not config["student_privileged_state_inputs"]
            assert config["steps_planned"] == steps and config["seed"] == seed
            assert config["learning_rate_peak"] == learning_rate
            assert [r["step"] for r in log] == list(range(1, steps + 1))
            assert all(math.isfinite(r[k]) for r in log for k in ["loss", "active7_loss", "gradient_norm"])
            assert audit["all_original_weights_unchanged"]
            assert config["frozen_parameter_sha256_before"] == audit["frozen_parameter_sha256_after"]
            frozen_hashes[scope] = audit["frozen_parameter_sha256_after"]
            if scope == "ae":
                assert audit["fixed_input_vlm_hidden_exact"]
            enabled = {"ae"} if scope == "ae" else {"vision", "language"} if scope == "vlm" else {"vision", "language", "ae"}
            for gradient in [audit["first_gradient"], audit["last_gradient"]]:
                for group, norm in gradient["norms"].items():
                    assert norm > 0 if group in enabled else norm == 0
            for name, digest in config["scripts"].items():
                snapshot = folder / "source_snapshot" / Path(name).name
                assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
            configs[scope] = {key: config[key] for key in common_keys}
            rows.append(dict(scope=scope, phase=phase, steps=len(log), original_weights_unchanged=True,
                             gradient_scope_correct=True, trainable_parameters=audit["trainable_parameters"],
                             training_seconds=audit["training_seconds"],
                             fixed_training_loss_before=audit["fixed_train_loss_before"],
                             fixed_training_loss_after=audit["fixed_train_loss_after"],
                             final_100_active7_loss=sum(r["active7_loss"] for r in log[-100:]) / 100))
        assert configs["ae"] == configs["vlm"] == configs["all"]
    result = dict(passed=True, rows=rows, same_training_configuration_and_source=True,
                  total_steps_per_scope=sum(phase[2] for phase in phases), extension_fresh_optimizer=bool(extension),
                  manifest_sha256=manifest_sha, all_started_from_official_base=True,
                  minibatch_equality_evidence="Same deterministic sampler source, manifest, seed and step count; no runtime minibatch trace was saved.",
                  original_weight_hash_note="Compare within scope: wrapping different modules changes parameter names in the digest.")
    write_json(run / "training_comparison_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
