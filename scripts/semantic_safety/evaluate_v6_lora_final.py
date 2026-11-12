"""Export frozen selections and evaluate all models on the same final benchmarks."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from rollout_common import ROOT, write_json
from semantic_pairs_v6 import DATA as ORIGINAL
from semantic_pairs_v6 import sha256
from summarize_v6_lora import summarize
from v6_lora_data import DATA, RUN


def main():
    global RUN, DATA
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=RUN)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--export-root", type=Path, default=ROOT / "checkpoints/pi05_libero_v6_lora_v1")
    args = parser.parse_args()
    RUN = args.run_root.resolve()
    DATA = args.data_root.resolve()
    selection = json.loads(args.selection.read_text())
    assert selection["frozen"] and not selection["test_used_for_selection"]
    assert set(selection["selected"]) == {"vlm", "ae", "all"}
    assert json.loads((RUN / "training_comparison_audit.json").read_text())["passed"]
    protocol = json.loads((RUN / "experiment_protocol.json").read_text())
    assert sha256(RUN / "experiment_protocol.json") == selection["experiment_protocol_sha256"]
    assert sha256(DATA / "holdouts/test.jsonl") == selection["test_items_sha256"]
    assert sha256(DATA / "holdouts/validation_selection66.jsonl") == selection["selection_items_sha256"]
    if protocol.get("extension"):
        assert selection["extension_completed"]
    assert json.loads((RUN / "batch32_full_rollout_equivalence_audit.json").read_text())["passed"]
    os.chdir(ROOT)
    env = {**os.environ, "__EGL_VENDOR_LIBRARY_FILENAMES": str(ROOT.parent / ".persist/nvidia-egl/10_nvidia.json"),
           "LD_LIBRARY_PATH": str(ROOT.parent / ".persist/nvidia-egl"),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    export = args.export_root.resolve()
    export.mkdir(parents=True, exist_ok=True)
    adapters = {"baseline": None}
    for scope, row in selection["selected"].items():
        source = Path(row["adapter"])
        assert sha256(source) == row["adapter_sha256"]
        audit = json.loads((source.parent / "freeze_audit.json").read_text())
        assert audit["all_original_weights_unchanged"]
        if scope == "ae":
            assert audit["fixed_input_vlm_hidden_exact"]
        folder = export / scope
        folder.mkdir(exist_ok=True)
        target = folder / "adapter.pt"
        if target.exists():
            assert sha256(target) == sha256(source)
        else:
            shutil.copy2(source, target)
        for name in ["config.json", "freeze_audit.json"]:
            shutil.copy2(source.parent / name, folder / name)
        history = []
        for phase, suffix in [("stage1", ""), ("extension", "_extension")]:
            parent = RUN / f"train_{scope}{suffix}"
            if not (parent / "freeze_audit.json").exists():
                continue
            history.append(dict(phase=phase, config=json.loads((parent / "config.json").read_text()),
                                freeze_audit=json.loads((parent / "freeze_audit.json").read_text())))
        write_json(folder / "training_history.json", dict(stages=history, selected_total_step=row["step"],
                   selected_phase_local_step=row.get("phase_step", row["step"])))
        write_json(folder / "selection.json", {**row, "adapter_sha256": sha256(target),
                                                "selection_file_sha256": sha256(args.selection)})
        adapters[scope] = target
    shutil.copy2(args.selection, export / "selection_frozen.json")
    shutil.copy2(RUN / "base_checkpoint_identity.json", export / "base_checkpoint_identity.json")
    usage = """# π0.5 LIBERO v6 LoRA adapters

`vlm/adapter.pt`, `ae/adapter.pt`, `all/adapter.pt` are validation-selected adapters for the
official `checkpoints/pi05_libero_pytorch` base. They are not standalone base checkpoints.
All original weights remain frozen. The data teaches grasp/lift/transport prefixes;
it does not establish terminal task completion or semantic safety awareness.

Run from the repository root with `openpi/.venv/bin/python` and
`PYTHONPATH=openpi/src:openpi/packages/openpi-client/src`:

```python
import sys
sys.path.insert(0, "scripts/semantic_safety")
from motor_lora import load_policy
from v6_scoped_lora import load

policy, _ = load_policy("pi05_libero")
metadata = load(policy._model, "checkpoints/pi05_libero_v6_lora_v1/all/adapter.pt")
policy._model.requires_grad_(False)
# Pass the normal pi05_libero observation to policy.infer(...).
```

Each directory includes training configuration, the original-weight freeze audit,
and adapter SHA256 / selection provenance. `base_checkpoint_identity.json` records
the base weights, tokenizer and normalization file hashes. The experiment report is
`doc/Semantic_Safety_v6_LoRA_Adaptation.md`; recordings and comparisons are under
`runs/semantic_motor_v6_lora_v1/`.

`training_history.json` distinguishes the initial 6,000-step stage from the optional
2,000-step adapter warm-start with a fresh optimizer. Selection step numbers are
cumulative; an extension adapter's internal `metadata.step` is local to that stage.
Only the selected adapter and the official base are needed for inference.
"""
    usage = usage.replace("checkpoints/pi05_libero_v6_lora_v1", str(export.relative_to(ROOT)))
    usage = usage.replace("runs/semantic_motor_v6_lora_v1", str(RUN.relative_to(ROOT)))
    (export / "README.md").write_text(usage)
    selection_ids = {json.loads(line)["item_id"] for line in
                     (DATA / "holdouts/validation_selection66.jsonl").read_text().splitlines()}
    confirmation = [json.loads(line) for line in (DATA / "holdouts/validation.jsonl").read_text().splitlines()
                    if json.loads(line)["item_id"] not in selection_ids]
    assert len(confirmation) == 66
    confirmation_file = DATA / "holdouts/validation_confirmation66.jsonl"
    confirmation_file.write_text("".join(json.dumps(row) + "\n" for row in confirmation))
    benchmarks = [("confirmation", confirmation_file), ("test", DATA / "holdouts/test.jsonl"),
                  ("v6", ORIGINAL / "items.jsonl"), ("native", None)]
    results = []
    for scope in ["baseline", "ae", "vlm", "all"]:
        for bench, items in benchmarks:
            if scope == "baseline" and bench == "v6":
                folder = ROOT / "runs/semantic_safety_transport_v6_evaluation/pi05_libero"
                assert json.loads((folder / "recording_audit.json").read_text())["passed"]
                report = summarize(folder)
            else:
                folder = RUN / "final" / bench / scope
                folder.parent.mkdir(parents=True, exist_ok=True)
                script = "evaluate_v6_lora.py" if bench == "native" else "evaluate_v6_selection.py"
                command = [str(ROOT / "envs/libero-render/bin/python"), "scripts/semantic_safety/" + script,
                           "--socket", "/tmp/vla_pi05_v6_lora.sock", "--out", str(folder),
                           "--batch-size", "8" if bench == "native" else "32"]
                command += ["--native-smoke", "16"] if bench == "native" else ["--items", str(items)]
                if adapters[scope]:
                    command += ["--adapter", str(adapters[scope])]
                if not (folder / "completion.json").exists():
                    with (folder.parent / (scope + ".log")).open("a") as log:
                        subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
                with (folder / "audit.log").open("w") as log:
                    subprocess.run([str(ROOT / "openpi/.venv/bin/python"),
                                    "scripts/semantic_safety/audit_v6_lora_evaluation.py", "--run", str(folder)],
                                   env=env, stdout=log, stderr=log, check=True)
                report = summarize(folder)
            result = dict(model=scope, benchmark=bench, folder=str(folder), overall=report["overall"],
                          evaluated_item_ids=[row["item_id"] for row in report["rows"]],
                          adapter_sha256=sha256(adapters[scope]) if adapters[scope] else None)
            results.append(result)
            write_json(RUN / "final_evaluation_progress.json", dict(rows=results, complete=False))
            print(json.dumps(result["overall"] | dict(model=scope, benchmark=bench)), flush=True)
    write_json(RUN / "final_evaluation_complete.json", dict(rows=results, complete=True,
               selection_file_sha256=sha256(args.selection), finished_unix=time.time(),
               blind_test_used_for_tuning=False))


if __name__ == "__main__":
    main()
