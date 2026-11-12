"""Evaluate the predeclared common warm-start stage without opening the test set."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from rollout_common import ROOT, write_json
from summarize_v6_lora import summarize


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run, data = args.run_root.resolve(), args.data_root.resolve()
    os.chdir(ROOT)
    while not (run / "validation_grid_complete.json").exists():
        time.sleep(20)
    archive = run / "validation_grid_stage1_complete.json"
    if not archive.exists():
        shutil.copy2(run / "validation_grid_complete.json", archive)
    previous = json.loads(archive.read_text())
    assert previous["complete"] and len(previous["rows"]) == 9
    assert not (run / "final/test").exists()
    grid = previous["rows"]
    env = {**os.environ, "__EGL_VENDOR_LIBRARY_FILENAMES": str(ROOT.parent / ".persist/nvidia-egl/10_nvidia.json"),
           "LD_LIBRARY_PATH": str(ROOT.parent / ".persist/nvidia-egl"),
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for scope in ["ae", "vlm", "all"]:
        for phase_step in [1000, 2000]:
            total_step = 6000 + phase_step
            adapter = run / f"train_{scope}_extension" / f"adapter_step_{phase_step:05d}.pt"
            while not adapter.exists():
                time.sleep(20)
            out = run / "validation" / f"{scope}_{total_step:05d}"
            if not (out / "completion.json").exists():
                with (out.parent / f"{scope}_{total_step:05d}.log").open("a") as log:
                    subprocess.run([str(ROOT / "envs/libero-render/bin/python"),
                                    "scripts/semantic_safety/evaluate_v6_selection.py",
                                    "--socket", "/tmp/vla_pi05_v6_lora.sock", "--items",
                                    str(data / "holdouts/validation_selection66.jsonl"),
                                    "--adapter", str(adapter), "--out", str(out), "--batch-size", "32"],
                                   env=env, stdout=log, stderr=log, check=True)
            with (out / "audit.log").open("w") as log:
                subprocess.run([str(ROOT / "openpi/.venv/bin/python"),
                                "scripts/semantic_safety/audit_v6_lora_evaluation.py", "--run", str(out)],
                               env=env, stdout=log, stderr=log, check=True)
            result = summarize(out)
            row = dict(scope=scope, step=total_step, phase="extension", phase_step=phase_step,
                       preceding_adapter_steps=6000, adapter=str(adapter), run=str(out), **result["overall"])
            grid.append(row)
            write_json(run / "validation_grid_progress.json", dict(rows=grid, extension=True,
                       selection_rule="transport count, sustained count, earliest total step"))
            print(json.dumps(row), flush=True)
    grid.sort(key=lambda row: (["ae", "vlm", "all"].index(row["scope"]), row["step"]))
    selected = {scope: max((row for row in grid if row["scope"] == scope),
                          key=lambda row: (row["transport_success"], row["sustained_transport_success"], -row["step"]))
                for scope in ["ae", "vlm", "all"]}
    result = dict(complete=True, rows=grid, selected=selected, test_used_for_selection=False,
                  selection_scenes=66, selection_rule="transport count, sustained count, earliest total step",
                  extension_completed=True, adapter_step_metadata_is_phase_local=True)
    write_json(run / "validation_grid_complete.json", result)
    write_json(run / "validation_extension_complete.json", result)


if __name__ == "__main__":
    main()
