"""Resume the authorized benign curriculum, three LoRA scopes, and matched frozen probes."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import benign_adaptation_v1 as b
from experiment_notifications import notify_progress
from rollout_common import write_json

PYTHON=b.ROOT/"openpi/.venv/bin/python"
SIM=b.ROOT/"envs/libero-render/bin/python"
PROBE=b.ROOT/"envs/representation-probe/bin/python"
SOCKET=Path("/tmp/semantic_benign_adaptation_v1.sock")
SCRIPTS=b.ROOT/"scripts/semantic_safety"
SCOPES=["ae","vlm","all"]


def environment():
    env=os.environ.copy()
    env.update(PYTHONPATH=str(b.ROOT/"openpi/src")+":"+str(b.ROOT/"openpi/packages/openpi-client/src"),
               OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="8",JAX_PLATFORMS="cpu",WANDB_MODE="disabled",
               MUJOCO_GL="egl",PYOPENGL_PLATFORM="egl",
               __EGL_VENDOR_LIBRARY_FILENAMES=str(b.ROOT.parent/".persist/nvidia-egl/10_nvidia.json"),
               LD_LIBRARY_PATH=str(b.ROOT.parent/".persist/nvidia-egl"))
    return env


def command(argv,label):
    if (b.RUN/"pause_requested.json").exists():raise InterruptedError("User pause requested")
    write_json(b.RUN/"pipeline_status.json",dict(state="running",phase=label,updated_unix=time.time(),command=list(map(str,argv))))
    log=b.RUN/"logs"/(label.replace("/","_").replace(" ","_")+".log")
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("a") as stream:
        process=subprocess.Popen(list(map(str,argv)),cwd=b.ROOT,env=environment(),stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        write_json(b.RUN/"active_child.json",dict(pid=process.pid,phase=label,log=str(log),started_unix=time.time()))
        try:
            while process.poll() is None:
                if (b.RUN/"pause_requested.json").exists():
                    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30);raise InterruptedError("User pause requested")
                notify_progress(b.RUN)
                time.sleep(10)
            if process.returncode:
                with log.open("rb") as failure:
                    failure.seek(max(0,log.stat().st_size-12000))
                    details=failure.read().decode(errors="replace")
                raise RuntimeError(f"Stage {label} exited {process.returncode}:\n{details}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    notify_progress(b.RUN)


def evaluate(label,adapter=None,items=None,native=None):
    folder=b.RUN/label
    receipt=folder/"completion.json"
    if receipt.exists():
        value=json.loads(receipt.read_text());assert value["complete"];return value
    argv=[SIM,SCRIPTS/"evaluate_benign_adaptation_v1.py","--socket",SOCKET,"--out",folder,"--batch-size",12,"--horizon",460]
    if adapter:argv += ["--adapter",adapter]
    argv += ["--native",native] if native else ["--items",items]
    command(argv,"evaluate "+label)
    return json.loads(receipt.read_text())


def server():
    import psutil
    record=b.RUN/"inference_server_process.json"
    if record.exists():
        prior=json.loads(record.read_text())
        try:
            owned=psutil.Process(prior["pid"])
            if (owned.create_time()==prior.get("created_unix") and
                "benign_lora_server_v1.py" in " ".join(owned.cmdline()) and str(SOCKET) in owned.cmdline()):
                owned.terminate()
                try:owned.wait(timeout=20)
                except psutil.TimeoutExpired:owned.kill();owned.wait(timeout=10)
        except psutil.NoSuchProcess:pass
    if SOCKET.exists():SOCKET.unlink()
    stream=(b.RUN/"inference_server.log").open("a")
    process=subprocess.Popen([str(PYTHON),str(SCRIPTS/"benign_lora_server_v1.py"),"--socket",str(SOCKET)],
                             cwd=b.ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT)
    write_json(b.RUN/"inference_server_process.json",dict(pid=process.pid,created_unix=psutil.Process(process.pid).create_time(),
                                                         socket=str(SOCKET),started_unix=time.time()))
    for _ in range(180):
        if process.poll() is not None:raise RuntimeError("Inference server failed; inspect inference_server.log")
        if SOCKET.exists():return process,stream
        time.sleep(2)
    process.terminate();raise TimeoutError("Inference server startup timed out")


def stage(key,title,result):
    write_json(b.RUN/"milestones"/(key+".json"),dict(complete=True,title=title,result=result))
    notify_progress(b.RUN)


def adopt_active_child():
    """Wait for a verified surviving child before a restarted driver resumes stages."""
    import psutil
    path=b.RUN/"active_child.json"
    if not path.exists():return
    record=json.loads(path.read_text())
    try:
        process=psutil.Process(record["pid"])
        argv=process.cmdline()
        allowed={"collect_benign_adaptation_v1.py","train_benign_adaptation_v1.py",
                 "evaluate_benign_adaptation_v1.py","probe_benign_adaptation_v1.py","report_benign_adaptation_v1.py"}
        script=next((Path(arg).name for arg in argv[:3] if Path(arg).name in allowed),None)
        if script is None or Path(process.cwd())!=b.ROOT:return
        if abs(process.create_time()-record["started_unix"])>5:return
        if script.startswith(("train_","evaluate_")) and str(b.RUN) not in " ".join(argv):return
        write_json(b.RUN/"supervisor_status.json",dict(state="adopting_existing_child",child_pid=process.pid,
                   phase=record["phase"],updated_unix=time.time()))
        print("Adopting owned child",process.pid,record["phase"],flush=True)
        while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
            if (b.RUN/"pause_requested.json").exists():
                assert os.getpgid(process.pid)==process.pid
                os.killpg(process.pid,signal.SIGTERM)
                raise InterruptedError("User pause requested")
            notify_progress(b.RUN);time.sleep(10)
    except psutil.NoSuchProcess:return


def pipeline():
    command([SIM,SCRIPTS/"collect_benign_adaptation_v1.py","prepare"],"prepare disjoint benign curriculum")
    command([PROBE,SCRIPTS/"probe_benign_adaptation_v1.py","prepare"],"freeze original probe inputs and folds")
    command([SIM,SCRIPTS/"collect_benign_adaptation_v1.py","calibrate","--workers",6],"calibrate benign physical skills")
    calibration=json.loads((b.DATA/"collection/calibrate_summary.json").read_text())
    if not calibration["complete"]:
        command([SIM,SCRIPTS/"collect_benign_adaptation_v1.py","calibrate","--workers",4,"--profile-limit",300,
                 "--ids",",".join(map(str,calibration["failed"]))],"recover remaining training grasps")
    missing=[]
    for row in b.conditions():
        proof=json.loads((b.DATA/"collection/profiles"/f"c{row['condition_index']:03d}.json").read_text())
        if not proof["success"]:missing.append(row["condition_index"])
    assert not missing,f"Physical calibration incomplete: {missing}"
    stage("curriculum_validated","새 물체·안전한 조작 과제의 teacher 및 재실행 검증 완료",
          dict(conditions=len(b.conditions()),overlap_audit=str(b.DATA/"semantic_overlap_audit.json")))
    command([SIM,SCRIPTS/"collect_benign_adaptation_v1.py","collect","--workers",6,"--count",32],"collect benign demonstrations")
    command([SIM,SCRIPTS/"collect_benign_adaptation_v1.py","audit"],"audit complete training dataset")
    audit=json.loads((b.DATA/"training_data_audit.json").read_text());assert audit["passed"]
    observations=json.loads((b.RUN/"observation_replay_audit.json").read_text())
    assert observations["passed"] and observations["physical_signature"]==b.signature()
    preflight_manifest=b.DATA/"preflight_train_manifest.jsonl"
    if not preflight_manifest.exists():preflight_manifest=b.DATA/"combined_train_manifest.jsonl"
    for scope in SCOPES:
        command([PYTHON,SCRIPTS/"train_benign_adaptation_v1.py","--scope",scope,"--manifest",preflight_manifest,
                 "--out",b.RUN/("preflight_"+scope),"--steps",4,"--save-every",4,"--recovery-every",2,"--pilot"],"training preflight "+scope)
    stage("preflight","세 LoRA 범위의 입력·gradient·freeze 사전 검증 완료",dict(scopes=SCOPES))
    inference,stream=server()
    try:
        baseline_native=evaluate("selection/baseline/native",native="validation")
        evaluate("selection/baseline/motor",items=b.DATA/"holdouts/selection.jsonl")
        candidates=[]
        for scope in SCOPES:
            folder=b.RUN/("train_"+scope)
            command([PYTHON,SCRIPTS/"train_benign_adaptation_v1.py","--scope",scope,"--manifest",b.DATA/"combined_train_manifest.jsonl",
                     "--out",folder],"train "+scope)
            for step in [4000,8000,12000,16000]:
                adapter=folder/f"adapter_step_{step:05d}.pt"
                motor=evaluate(f"selection/{scope}_{step:05d}/motor",adapter=adapter,items=b.DATA/"holdouts/selection.jsonl")
                native=evaluate(f"selection/{scope}_{step:05d}/native",adapter=adapter,native="validation")
                candidates.append(dict(scope=scope,cumulative_step=step,adapter=str(adapter.relative_to(b.ROOT)),
                    adapter_sha256=b.sha(adapter),motor_successes=motor["successes"],motor_episodes=motor["episodes"],
                    native_successes=native["successes"],native_episodes=native["episodes"]))
                write_json(b.RUN/"checkpoint_candidates.json",candidates)
        chosen={}
        for scope in SCOPES:
            rows=[r for r in candidates if r["scope"]==scope]
            eligible=[r for r in rows if r["native_successes"]>=baseline_native["successes"]-2]
            chosen[scope]=max(eligible or rows,key=lambda r:(r["motor_successes"],r["native_successes"],-r["cumulative_step"]))
            chosen[scope]["retention_constraint_met"]=bool(eligible)
        selection=dict(selected=chosen,frozen_before_test=True,test_scores_used_for_selection=False,
                       safety_probe_scores_used_for_selection=False,baseline_native_validation=baseline_native)
        target=b.RUN/"frozen_selection.json"
        if target.exists():assert json.loads(target.read_text())==selection
        else:write_json(target,selection)
        notify_progress(b.RUN)
        export=b.ROOT/"checkpoints/pi05_libero_benign_adaptation_v1"
        for scope,row in chosen.items():
            folder=export/scope;folder.mkdir(parents=True,exist_ok=True)
            if not (folder/"adapter.pt").exists():shutil.copy2(b.ROOT/row["adapter"],folder/"adapter.pt")
            assert b.sha(folder/"adapter.pt")==row["adapter_sha256"]
            write_json(folder/"selection.json",row)
        for scope in ["baseline",*SCOPES]:
            adapter=None if scope=="baseline" else b.ROOT/chosen[scope]["adapter"]
            evaluate(f"final/{scope}/motor",adapter=adapter,items=b.DATA/"holdouts/test.jsonl")
            native=evaluate(f"final/{scope}/native",adapter=adapter,native="test")
            stage("motor_evaluation_"+scope,scope+" 새 조작 과제 및 LIBERO 유지 성능 평가 완료",native)
    finally:
        if inference.poll() is None:
            inference.terminate()
            try:inference.wait(timeout=20)
            except subprocess.TimeoutExpired:inference.kill();inference.wait()
        stream.close();SOCKET.unlink(missing_ok=True)
    for scope in SCOPES:
        command([PYTHON,SCRIPTS/"probe_benign_adaptation_v1.py","extract","--scope",scope,"--limit",16],"probe extraction smoke "+scope)
        command([PYTHON,SCRIPTS/"probe_benign_adaptation_v1.py","extract","--scope",scope],"extract frozen representations "+scope)
    command([PROBE,SCRIPTS/"probe_benign_adaptation_v1.py","probe","--workers",8],"fit matched linear probes")
    command([PROBE,SCRIPTS/"report_benign_adaptation_v1.py"],"verify and write comparative results")
    for name in ["pipeline_status.json","experiment_status.json"]:write_json(b.RUN/name,dict(state="complete",updated_unix=time.time()))
    notify_progress(b.RUN)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--supervise",action="store_true");args=parser.parse_args()
    b.RUN.mkdir(parents=True,exist_ok=True)
    with (b.RUN/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(b.RUN/"supervisor_process.json",dict(pid=os.getpid(),started_unix=time.time()))
        try:adopt_active_child()
        except InterruptedError:
            write_json(b.RUN/"supervisor_status.json",dict(state="paused",updated_unix=time.time()));return
        for attempt in range(1,7 if args.supervise else 2):
            try:
                write_json(b.RUN/"supervisor_status.json",dict(state="running",attempt=attempt,updated_unix=time.time()))
                if (b.RUN/"experiment_status.json").exists() and json.loads((b.RUN/"experiment_status.json").read_text()).get("state")=="complete":return
                pipeline();write_json(b.RUN/"supervisor_status.json",dict(state="complete",updated_unix=time.time()));return
            except InterruptedError:
                write_json(b.RUN/"supervisor_status.json",dict(state="paused",updated_unix=time.time()));notify_progress(b.RUN);return
            except Exception:
                error=dict(attempt=attempt,error=traceback.format_exc(),updated_unix=time.time())
                write_json(b.RUN/"pipeline_error.json",error)
                history=b.RUN/"automatic_recovery_history.json"
                rows=json.loads(history.read_text()) if history.exists() else []
                rows.append(error);write_json(history,rows);notify_progress(b.RUN)
                permanent=any(marker in error["error"] for marker in ["AssertionError","FileExistsError","FloatingPointError"])
                if not args.supervise or attempt>=6 or permanent:
                    write_json(b.RUN/"supervisor_status.json",dict(state="needs_attention",updated_unix=time.time()));notify_progress(b.RUN);raise
                archive=b.RUN/"resolved_errors";archive.mkdir(exist_ok=True)
                shutil.move(b.RUN/"pipeline_error.json",archive/f"attempt_{time.time_ns()}.json")
                time.sleep(min(60,attempt*10))


if __name__=="__main__":main()
