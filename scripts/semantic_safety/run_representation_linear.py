"""Resume the authorized two-reference, linear-only pipeline from durable receipts."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiment_notifications import notify_progress
from representation_common import MODELS, ROOT, RUN, TEACHER, now, prepare
from rollout_common import write_json

TRAIN = ROOT / "openpi/.venv/bin/python"
RENDER = ROOT / ".cache/conda-repro-validation/render/bin/python"
PROBE = ROOT / "envs/representation-probe/bin/python"


def complete(path):
    return path.exists() and json.loads(path.read_text()).get("complete") is True


def stage_complete(kind, source=None, view=None):
    if kind == "teacher":
        return complete(TEACHER / "completion.json")
    if kind == "features":
        return all(complete(RUN / "features" / source / view / f"{model}_completion.json") for model in MODELS)
    path = RUN / "probes" / source / view / "completion.json"
    return complete(path) and json.loads(path.read_text()).get("sites") == 312


def external_pid(script, required):
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().decode().split("\0")
            if (len(args) < 2 or Path(args[1]).name != script
                    or any(flag in args for flag in ("--limit", "--sites", "--episode-ids", "--out-root", "--output-root"))):
                continue
            if all(key in args and args[args.index(key) + 1] == value for key, value in required.items()):
                return int(path.parent.name)
        except (OSError, UnicodeDecodeError, IndexError):
            continue
    return None


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def launch(kind, source=None, view=None):
    if stage_complete(kind, source, view):
        return {"kind": kind, "source": source, "view": view, "already_complete": True}
    required = {} if kind == "teacher" else {"--source": source, "--view": view}
    if kind == "teacher":
        script, python, extra = "collect_representation_teacher.py", RENDER, ["--workers", "16", "--profile-limit", "1000"]
    elif kind == "features":
        script, python, extra = "extract_representation_linear.py", TRAIN, []
    else:
        script, python, extra = "probe_representation_linear.py", PROBE, ["--workers", "10", "--threads", "2"]
    command = [str(python), str(ROOT / "scripts/semantic_safety" / script)]
    command += [value for pair in required.items() for value in pair] + extra
    name = "_".join(str(value) for value in (kind, source, view) if value is not None)
    pid = external_pid(script, required)
    process = None
    if pid is None:
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        log = (RUN / "logs" / f"{name}.log").open("a")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        pid = process.pid
    write_json(RUN / "processes" / f"{name}.json", {"pid": pid, "command": command,
               "started_or_adopted_utc": now(), "adopted": process is None})
    return {"kind": kind, "source": source, "view": view, "pid": pid, "process": process, "name": name}


def wait(stage):
    retries = 0
    while not stage_complete(stage["kind"], stage["source"], stage["view"]):
        process = stage.get("process")
        running = process.poll() is None if process is not None else alive(stage["pid"])
        if not running:
            if retries >= 2:
                raise RuntimeError(f"Stage exited without complete receipt after retries: {stage.get('name')}")
            history_path = RUN / "automatic_recovery_history.json"
            history = json.loads(history_path.read_text()) if history_path.exists() else []
            history.append({"attempt": len(history) + 1, "stage": stage.get("name"),
                "state": "restarting_same_protocol_from_completed_receipts", "updated_utc": now()})
            write_json(history_path, history)
            stage = launch(stage["kind"], stage["source"], stage["view"])
            retries += 1
            notify_progress(RUN)
            continue
        write_json(RUN / "pipeline_status.json", {"state": "running", "waiting_for": stage.get("name"), "updated_utc": now()})
        time.sleep(20)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare()
    (RUN / "logs").mkdir(exist_ok=True)
    write_json(RUN / "supervisor_status.json", {"state": "running", "pid": os.getpid(), "updated_utc": now()})
    teacher = launch("teacher")
    wait(launch("features", "both", "initial"))
    initial_probe = launch("probes", "both", "initial")
    wait(launch("features", "both", "all"))
    both_probe = launch("probes", "both", "all")
    wait(teacher)
    wait(launch("features", "teacher", "all"))
    teacher_probe = launch("probes", "teacher", "all")
    for stage in (initial_probe, both_probe, teacher_probe):
        wait(stage)
    with (RUN / "logs/report.log").open("a") as log:
        subprocess.run([str(PROBE), "scripts/semantic_safety/report_representation_linear.py"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert complete(RUN / "report/summary.json")
    for name in ("pipeline_status.json", "experiment_status.json", "supervisor_status.json"):
        write_json(RUN / name, {"state": "complete", "completed_utc": now(), "report": str(RUN / "report/index.html")})
    notify_progress(RUN)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(RUN / "pipeline_error.json", {"error": traceback.format_exc(), "updated_utc": now()})
        write_json(RUN / "supervisor_status.json", {"state": "needs_attention", "updated_utc": now()})
        notify_progress(RUN)
        raise
