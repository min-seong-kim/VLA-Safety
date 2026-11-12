"""Evaluate a predeclared checkpoint grid using only the 66 selection scenes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from rollout_common import ROOT, write_json
from v6_lora_data import DATA, RUN
from summarize_v6_lora import summarize


def main():
    global RUN, DATA
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root",type=Path,default=RUN)
    parser.add_argument("--data-root",type=Path,default=DATA)
    args=parser.parse_args();RUN=args.run_root.resolve();DATA=args.data_root.resolve()
    os.chdir(ROOT)
    assert json.loads((RUN/"batch32_full_rollout_equivalence_audit.json").read_text())["passed"]
    env={**os.environ,"__EGL_VENDOR_LIBRARY_FILENAMES":str(ROOT.parent/".persist/nvidia-egl/10_nvidia.json"),
         "LD_LIBRARY_PATH":str(ROOT.parent/".persist/nvidia-egl"),"OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1"}
    grid=[];selected={}
    for scope in ["ae","vlm","all"]:
        candidates=[]
        for step in [2000,4000,6000]:
            adapter=RUN/f"train_{scope}"/f"adapter_step_{step:05d}.pt"
            while not adapter.exists():time.sleep(20)
            out=RUN/"validation"/f"{scope}_{step:05d}";out.parent.mkdir(exist_ok=True)
            if not (out/"completion.json").exists():
                with (out.parent/f"{scope}_{step:05d}.log").open("a") as log:
                    subprocess.run([str(ROOT/"envs/libero-render/bin/python"),"scripts/semantic_safety/evaluate_v6_selection.py",
                                    "--socket","/tmp/vla_pi05_v6_lora.sock","--items",str(DATA/"holdouts/validation_selection66.jsonl"),
                                    "--adapter",str(adapter),"--out",str(out),"--batch-size","32"],env=env,stdout=log,stderr=log,check=True)
            with (out/"audit.log").open("w") as log:
                subprocess.run([str(ROOT/"openpi/.venv/bin/python"),"scripts/semantic_safety/audit_v6_lora_evaluation.py",
                                "--run",str(out)],env=env,stdout=log,stderr=log,check=True)
            report=summarize(out);row=dict(scope=scope,step=step,adapter=str(adapter),run=str(out),**report["overall"])
            candidates.append(row);grid.append(row)
            write_json(RUN/"validation_grid_progress.json",dict(rows=grid,selection_rule="transport count, sustained count, earliest step"))
            print(json.dumps(row),flush=True)
        selected[scope]=max(candidates,key=lambda r:(r["transport_success"],r["sustained_transport_success"],-r["step"]))
    write_json(RUN/"validation_grid_complete.json",dict(complete=True,rows=grid,selected=selected,
               test_used_for_selection=False,selection_scenes=66,selection_rule="transport count, sustained count, earliest step"))


if __name__=="__main__":main()
