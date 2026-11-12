"""Finish the three audited training runs, choose on validation, then open held-out tests."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from rollout_common import ROOT, write_json

DATA=ROOT/"data/semantic_expansion_v5_lora_v1"
RUN=ROOT/"runs/semantic_expansion_v5_lora_v1"
EXPORT=ROOT/"checkpoints/pi05_libero_expansion_v5_lora_v1"
SOCKET=Path("/tmp/expansion_v5_pi05_lora.sock")


def command(argv,log,label):
    write_json(RUN/"pipeline_status.json",dict(state="running",phase=label,updated_unix=time.time(),command=argv))
    with (RUN/log).open("a") as output:
        subprocess.run(argv,cwd=ROOT,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT,check=True)


def evaluate(label,*,items=None,native=None,adapter=None):
    folder=RUN/label
    if (folder/"completion.json").exists():
        result=json.loads((folder/"completion.json").read_text())
        assert result["complete"]
        return result
    args=[str(ROOT/"envs/libero-render/bin/python"),"scripts/semantic_safety/evaluate_expansion_v5_lora.py",
          "--socket",str(SOCKET),"--out",str(folder),"--batch-size","16"]
    if items:args += ["--items",str(items)]
    if native:args += ["--native",native]
    if adapter:args += ["--adapter",str(adapter)]
    command(args,label.replace("/","_")+".log","evaluate "+label)
    return json.loads((folder/"completion.json").read_text())


def train(scope,*,extension=False):
    folder=RUN/(f"train_{scope}_extension" if extension else f"train_{scope}")
    args=[str(ROOT/"openpi/.venv/bin/python"),"scripts/semantic_safety/train_expansion_v5_lora.py",
        "--manifest",str(DATA/"combined_train_manifest.jsonl"),"--out",str(folder),"--scope",scope,
        "--steps","8000" if extension else "16000","--batch-size","32",
        "--learning-rate","0.00002" if extension else "0.0002","--save-every","4000",
        "--seed","20271112" if extension else "20261112"]
    if extension:args += ["--init-adapter",str(RUN/f"train_{scope}/adapter_step_16000.pt")]
    command(args,folder.name+".log","train "+folder.name)
    audit=json.loads((folder/"freeze_audit.json").read_text())
    assert audit["all_original_weights_unchanged"]
    if scope=="ae":assert audit["fixed_input_vlm_hidden_exact"]
    return folder


def candidates(scope,folder,steps,start=0):
    results=[]
    for step in steps:
        adapter=folder/f"adapter_step_{step:05d}.pt"
        cumulative=start+step
        label=f"selection/{scope}_{cumulative:05d}"
        motor=evaluate(label+"/motor",items=DATA/"holdouts/selection.jsonl",adapter=adapter)
        native=evaluate(label+"/native",native="validation",adapter=adapter)
        results.append(dict(scope=scope,cumulative_step=cumulative,adapter=str(adapter),
            motor_successes=motor["successes"],motor_episodes=motor["episodes"],
            native_successes=native["successes"],native_episodes=native["episodes"]))
    return results


def select(rows,baseline_native):
    return max(rows,key=lambda r:(r["native_successes"]>=baseline_native-2,
        r["motor_successes"],r["native_successes"],-r["cumulative_step"]))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--allow-incomplete-coverage",action="store_true")
    args=parser.parse_args()
    if (RUN/"pause_requested.json").exists():
        print("PAUSED_BY_USER: use the explicit resume entry point",flush=True)
        return
    os.environ["PYTHONPATH"]=str(ROOT/"openpi/src")+":"+str(ROOT/"openpi/packages/openpi-client/src")
    os.environ["OPENBLAS_NUM_THREADS"]="1"
    os.environ["OMP_NUM_THREADS"]="8"
    os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"]=str(ROOT.parent/".persist/nvidia-egl/10_nvidia.json")
    os.environ["LD_LIBRARY_PATH"]=str(ROOT.parent/".persist/nvidia-egl")
    audit=json.loads((DATA/"training_data_audit.json").read_text())
    assert audit["passed"] and audit["all_accepted_demos_independently_replayed"]
    for key in ["observation_action_alignment_verified","finite_states_and_bounded_actions",
                "camera_shapes_dtypes_and_three_frames_per_episode_checked",
                "replayed_motor_success_and_physics_stability_verified","no_exact_train_holdout_layout_overlap"]:
        assert audit[key],key
    assert audit["minimum_episodes_among_covered_conditions"]>=32
    if not args.allow_incomplete_coverage:assert audit["conditions_with_demonstrations"]==audit["conditions_total"]
    assert json.loads((RUN/"training_preflight_complete.json").read_text())["passed"]
    command([str(ROOT/"openpi/.venv/bin/python"),"scripts/semantic_safety/check_expansion_v5_training_resume.py"],
            "verify_training_resume.log","verify training restart")
    assert SOCKET.exists(),"Start the audited local inference server first"
    baseline_native=evaluate("baseline_native_validation",native="validation")["successes"]
    evaluate("baseline_motor_selection",items=DATA/"holdouts/selection.jsonl")
    all_rows=[]
    for scope in ["ae","vlm","all"]:
        folder=train(scope)
        all_rows.extend(candidates(scope,folder,[4000,8000,12000,16000]))
        write_json(RUN/"checkpoint_candidates.json",all_rows)
    extend=False
    for scope in ["ae","vlm","all"]:
        rows=[r for r in all_rows if r["scope"]==scope]
        best=select(rows,baseline_native)
        previous=next(r for r in rows if r["cumulative_step"]==12000)
        extend |= best["cumulative_step"]==16000 and (best["motor_successes"]-previous["motor_successes"])/best["motor_episodes"]>=.02
    triggered=extend
    overrides_path=RUN/"execution_overrides.json"
    overrides=json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    extension_allowed=overrides.get("extension_enabled",True)
    assert isinstance(extension_allowed,bool)
    extend=triggered and extension_allowed
    write_json(RUN/"extension_decision.json",dict(extend_all_scopes=extend,test_accessed=False,
        validation_condition_met=triggered,extension_allowed=extension_allowed,
        rule="frozen experiment_protocol.json with explicit user override when present",
        user_override=overrides if not extension_allowed else None))
    if extend:
        for scope in ["ae","vlm","all"]:
            folder=train(scope,extension=True)
            all_rows.extend(candidates(scope,folder,[4000,8000],start=16000))
            write_json(RUN/"checkpoint_candidates.json",all_rows)
    selected={scope:select([r for r in all_rows if r["scope"]==scope],baseline_native) for scope in ["ae","vlm","all"]}
    write_json(RUN/"frozen_selection.json",dict(selected=selected,baseline_native_successes=baseline_native,
        test_scores_used_for_selection=False,frozen_before_test=True))
    for scope,row in selected.items():
        folder=EXPORT/scope;folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(row["adapter"],folder/"adapter.pt")
        write_json(folder/"selection.json",row)
    for scope in ["baseline","ae","vlm","all"]:
        adapter=None if scope=="baseline" else EXPORT/scope/"adapter.pt"
        evaluate(f"final/{scope}/motor",items=DATA/"holdouts/test.jsonl",adapter=adapter)
        evaluate(f"final/{scope}/native",native="test",adapter=adapter)
        evaluate(f"final/{scope}/confirmation",items=DATA/"holdouts/confirmation.jsonl",adapter=adapter)
    command([str(ROOT/"openpi/.venv/bin/python"),"scripts/semantic_safety/report_expansion_v5_lora.py"],
            "report.log","final report")
    write_json(RUN/"pipeline_status.json",dict(state="complete",updated_unix=time.time()))


if __name__=="__main__":
    try:
        main()
    except Exception:
        write_json(RUN/"pipeline_error.json",dict(error=traceback.format_exc(),updated_unix=time.time()))
        raise
