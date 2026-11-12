"""Adopt the owned v5 experiment, recover infrastructure failures, and enforce audit gates."""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import psutil

from rollout_common import ROOT, write_json
from experiment_notifications import notify_progress

RUN = ROOT / "runs/semantic_expansion_v5_lora_v1"
DATA = ROOT / "data/semantic_expansion_v5_lora_v1"
PYTHON = ROOT / "openpi/.venv/bin/python"
SOCKET = Path("/tmp/expansion_v5_pi05_lora.sock")
DRIVER = "run_expansion_v5_experiment.py"
SERVER = "v6_lora_server.py"


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def owned(pid, script):
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        command = process.cmdline()
        if not any(Path(arg).name == script for arg in command[1:3]):
            return None
        if Path(process.cwd()) != ROOT:
            return None
        if script == SERVER and str(SOCKET) not in command:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def find_owned(script):
    return [p for row in psutil.process_iter(["pid"]) if (p := owned(row.pid, script)) is not None]


def terminate_owned(process):
    # Verify the recorded root before signalling its private process group.
    script = SERVER if any(Path(arg).name == SERVER for arg in process.cmdline()[1:3]) else DRIVER
    checked = owned(process.pid, script)
    assert checked is not None and checked.create_time() == process.create_time()
    descendants = process.children(recursive=True)
    identities = [(p.pid, p.create_time()) for p in [*descendants, process]]
    for pid, created in reversed(identities):
        try:
            p = psutil.Process(pid)
            if p.create_time() == created:
                p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([process, *descendants], timeout=10)
    for p in alive:
        if (p.pid, p.create_time()) in identities:
            p.kill()


def phase_and_log():
    experiment = read_json(RUN / "experiment_status.json", {})
    phase = experiment.get("phase", "not_started")
    if phase == "three_scope_training_and_evaluation":
        pipeline = read_json(RUN / "pipeline_status.json", {})
        inner = pipeline.get("phase", "initializing_training")
        if inner.startswith("train "):
            log = RUN / (inner.split(" ", 1)[1] + ".log")
        elif inner.startswith("evaluate "):
            log = RUN / (inner.split(" ", 1)[1].replace("/", "_") + ".log")
        elif inner == "final report":
            log = RUN / "report.log"
        elif inner == "verify training restart":
            log = RUN / "verify_training_resume.log"
        else:
            log = RUN / (phase + ".log")
        return inner, log
    return phase, RUN / (phase + ".log")


def tail(path, size=20000):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size-size))
        return stream.read().decode(errors="replace")


def recovery_kind(phase, evidence):
    if "Start the audited local inference server first" in evidence:
        return "restart_owned_inference_and_resume"
    if phase.startswith("full_demonstrations_") and "AssertionError" in evidence:
        summary = read_json(DATA / "collection_r10/collect_summary.json", {})
        if summary.get("conditions") == 338 and not summary.get("complete", True):
            return "broader_training_grasp_search"
    if any(s in evidence for s in ["AssertionError", "Nonfinite", "non-finite", "FloatingPointError",
                                   "digest mismatch", "Frozen inputs changed", "No space left on device",
                                   "Input/output error", "Checksum", "signature changed"]):
        return "audit_or_data_failure"
    if any(s in evidence for s in ["CUDA", "cuda", "EGL", "ConnectionRefusedError", "BrokenPipeError",
                                   "EOFError", "ConnectionResetError", "server_missing"]):
        return "restart_owned_inference_and_resume"
    if "idle_timeout" in evidence or "SIGKILL" in evidence or "SIGTERM" in evidence or not evidence.strip():
        return "resume_interrupted_phase"
    return "retry_once_then_preserve_failure"


def archive_errors(attempt):
    archive = RUN / "automatic_recovery" / f"attempt_{attempt:03d}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["experiment_error.json", "pipeline_error.json", "experiment_driver.log"]:
        path = RUN / name
        if path.exists():
            path.rename(archive / name)
    return archive


def ensure_server(children):
    servers = find_owned(SERVER)
    assert len(servers) <= 1, "Multiple inference servers own the same experiment socket"
    if servers:
        return servers[0].pid
    SOCKET.unlink(missing_ok=True)
    environment = dict(os.environ,
        PYTHONPATH=str(ROOT/"openpi/src")+":"+str(ROOT/"openpi/packages/openpi-client/src"),
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="8")
    with (RUN / "inference_server.log").open("a") as log:
        child = subprocess.Popen([str(PYTHON), "scripts/semantic_safety/"+SERVER, "--socket", str(SOCKET)],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    children.append(child)
    write_json(RUN / "inference_server_process.json", dict(pid=child.pid, started_unix=time.time()))
    deadline = time.monotonic()+900
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("Inference server failed to start: " + tail(RUN/"inference_server.log", 4000))
        if SOCKET.exists():
            return child.pid
        time.sleep(5)
    raise TimeoutError("Inference server startup timed out")


def start_driver(args, children):
    assert not find_owned(DRIVER), "Refuse to launch a duplicate experiment driver"
    command = [str(PYTHON), "scripts/semantic_safety/"+DRIVER, "--workers", str(args.workers),
               "--episodes-per-condition", "32", "--recovery-workers", "24"]
    with (RUN / "experiment_driver.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children.append(child)
    (RUN / "experiment_driver.pid").write_text(str(child.pid)+"\n")
    return psutil.Process(child.pid)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--check", action="store_true", help="Inspect ownership and the next phase without starting anything")
    parser.add_argument("--resume",action="store_true",help="Explicitly resume a user-paused experiment")
    args = parser.parse_args()
    if args.check:
        drivers=find_owned(DRIVER)
        phase, log=phase_and_log()
        print(json.dumps(dict(drivers=[p.pid for p in drivers], servers=[p.pid for p in find_owned(SERVER)],
                              phase=phase, log=str(log), source="read-only ownership check")))
        return
    lock=(RUN/"supervisor.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pause=RUN/"pause_requested.json"
    if pause.exists():
        if not args.resume:
            notify_progress(RUN)
            print("PAUSED_BY_USER: no work started; explicit --resume is required",flush=True)
            return
        request=read_json(pause)
        archive=ROOT/request["archive"]
        archive.mkdir(parents=True,exist_ok=True)
        pause.rename(archive/("pause_request_resumed_"+str(time.time_ns())+".json"))
        write_json(RUN/"last_resume_request.json",dict(requested_unix=time.time(),
            original_pause=request,extension_override=read_json(RUN/"execution_overrides.json",{})))
    (RUN/"supervisor.pid").write_text(str(os.getpid())+"\n")
    history=read_json(RUN/"automatic_recovery_history.json", [])
    retries=Counter((r["phase"],r["kind"]) for r in history)
    children=[]
    drivers=find_owned(DRIVER)
    assert len(drivers)<=1,"Multiple experiment drivers are already running"
    driver=drivers[0] if drivers else None
    last_phase=None
    phase_seen=time.time()
    while True:
        children[:]=[child for child in children if child.poll() is None]
        notify_progress(RUN)
        experiment=read_json(RUN/"experiment_status.json", {})
        if experiment.get("state")=="complete":
            assert read_json(RUN/"pipeline_status.json", {}).get("state")=="complete"
            assert (ROOT/"doc/Semantic_Safety_Expansion_v5_LoRA_Adaptation.md").exists()
            assert all((ROOT/"checkpoints/pi05_libero_expansion_v5_lora_v1"/s/"adapter.pt").exists() for s in ["ae","vlm","all"])
            write_json(RUN/"supervisor_status.json",dict(state="complete",updated_unix=time.time(),restarts=len(history)))
            return
        phase,log=phase_and_log()
        if phase!=last_phase:
            phase_seen=time.time();last_phase=phase
        live=owned(driver.pid,DRIVER) if driver else None
        evidence=""
        if live:
            latest=max(phase_seen,log.stat().st_mtime if log.exists() else 0)
            timeout=(21600 if phase=="verify training restart" else 14400 if "freeze" in phase
                     else 7200 if "demonstrations" in phase else 2700)
            if time.time()-latest>timeout:
                evidence="idle_timeout: no phase output for "+str(timeout)+" seconds"
                terminate_owned(live);live=None
            elif phase.startswith("evaluate ") and not find_owned(SERVER):
                evidence="server_missing during evaluation"
                terminate_owned(live);live=None
        if not live:
            # A killed driver can leave its own descendants running. Only clean
            # the private session group when its surviving members are ours.
            if driver:
                orphans=[]
                for process in psutil.process_iter(["pid","cmdline","cwd"]):
                    try:
                        if os.getpgid(process.pid)==driver.pid and Path(process.info["cwd"] or "/")==ROOT:
                            orphans.append(process)
                            process.terminate()
                    except (ProcessLookupError,psutil.NoSuchProcess,psutil.AccessDenied):
                        pass
                _,survivors=psutil.wait_procs(orphans,timeout=10)
                for process in survivors:
                    try:
                        if os.getpgid(process.pid)==driver.pid:
                            process.kill()
                    except (ProcessLookupError,psutil.NoSuchProcess):
                        pass
            if driver is not None:
                evidence += tail(log)+tail(RUN/"experiment_driver.log",5000)
                for name in ["experiment_error.json","pipeline_error.json"]:
                    evidence += (RUN/name).read_text() if (RUN/name).exists() else ""
                kind=recovery_kind(phase,evidence)
                limit=0 if kind=="audit_or_data_failure" else 1 if kind=="retry_once_then_preserve_failure" else 3
                if retries[(phase,kind)]>=limit:
                    write_json(RUN/"supervisor_status.json",dict(state="needs_attention",phase=phase,kind=kind,
                        error=evidence[-20000:],updated_unix=time.time(),audit_gates_preserved=True))
                    notify_progress(RUN)
                    return
                attempt=len(history)+1
                archive=archive_errors(attempt)
                (archive/"phase_log_tail.txt").write_text(evidence)
                history.append(dict(attempt=attempt,phase=phase,kind=kind,updated_unix=time.time(),
                                    previous_pid=driver.pid,archive=str(archive.relative_to(ROOT))))
                retries[(phase,kind)]+=1
                write_json(RUN/"automatic_recovery_history.json",history)
                notify_progress(RUN)
                if kind=="restart_owned_inference_and_resume":
                    for server in find_owned(SERVER):
                        terminate_owned(server)
                print(json.dumps(history[-1]),flush=True)
            ensure_server(children)
            driver=start_driver(args,children)
            phase_seen=time.time()
        write_json(RUN/"supervisor_status.json",dict(state="monitoring",phase=phase,driver_pid=driver.pid,
            supervisor_pid=os.getpid(),updated_unix=time.time(),automatic_restarts=len(history),
            quality_gates_required=True,training_resume="full optimizer every 200 updates; same samples and noise"))
        time.sleep(args.interval)


if __name__=="__main__":
    try:
        main()
    except Exception:
        write_json(RUN/"supervisor_error.json",dict(error=traceback.format_exc(),updated_unix=time.time()))
        notify_progress(RUN)
        raise
