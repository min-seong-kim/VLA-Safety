"""Supervise only this recorded local batch, audit outputs, and finalize probe artifacts.

The supervisor adopts recorded PIDs, retries at most three times per model, and
preserves incomplete attempts. It never touches unrelated user processes.
"""

import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rollout_common import ROOT, write_json

RUN = ROOT / "runs/semantic_rollouts_v3"
SCRIPTS = ROOT / "scripts/semantic_safety"
MODELS = ["pi0_libero", "pi05_libero", "openvla_libero_spatial", "openvla_oft_libero_spatial"]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command_line(pid):
    try:
        status = Path(f"/proc/{pid}/stat").read_text().split()[2]
        if status == "Z":
            return ""
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def runner_alive(pid, model):
    command = command_line(pid)
    return "run_recorded_rollouts.py" in command and model in command and str(RUN / model) in command


def cleanup_owned_worker(parent_pid, model):
    socket = f"/tmp/vla_{parent_pid}_{model}.sock"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        command = command_line(pid)
        if "rollout_policy_worker.py" in command and socket in command:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 15
            while command_line(pid) and time.monotonic() < deadline:
                time.sleep(.25)
            if command_line(pid):
                os.kill(pid, signal.SIGKILL)
            print(json.dumps({"cleaned_owned_orphan_worker": pid, "model": model}), flush=True)
    Path(socket).unlink(missing_ok=True)


def collect_status(pids):
    groups = {}
    for model in MODELS:
        rows = []
        for path in (RUN / model / "episodes").glob("*/summary.json"):
            try:
                rows.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, FileNotFoundError):
                # A pre-existing worker may be finishing its summary; retry on the next poll.
                continue
        failures = list((RUN / model / "episodes").glob("*/error.json"))
        groups[model] = {"completed_episodes": len(rows), "planned_episodes": 72,
                         "recorded_control_timesteps": sum(r["steps"] for r in rows),
                         "current_attempt_errors": len(failures), "pid": pids[model],
                         "process_alive": runner_alive(pids[model], model),
                         "safe_episodes": sum(not r["task_is_unsafe"] for r in rows),
                         "unsafe_episodes": sum(r["task_is_unsafe"] for r in rows)}
    return {"updated_utc": utc(), "supervisor_pid": os.getpid(), "status": "running",
            "planned_episodes": 288, "models": groups,
            "completed_episodes": sum(g["completed_episodes"] for g in groups.values()),
            "recorded_control_timesteps": sum(g["recorded_control_timesteps"] for g in groups.values())}


def run_postprocess(script, extra=(), timeout=1800):
    command = [sys.executable, str(SCRIPTS / script), *extra]
    with (RUN / "postprocess.log").open("a") as log:
        log.write("\n" + utc() + " " + json.dumps(command) + "\n")
        log.flush()
        try:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return process.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as error:
            log.write("POSTPROCESS_FAILED " + repr(error) + "\n")
            return False


def main():
    lock = RUN / "supervisor.pid"
    if lock.exists():
        old = int(lock.read_text())
        if "supervise_semantic_rollouts.py" in command_line(old):
            raise RuntimeError(f"Supervisor already active: {old}")
    lock.write_text(str(os.getpid()))
    launch = json.loads((RUN / "launch.json").read_text())
    original = {r["model"]: r for r in launch["processes"]}
    pids = {m: original[m]["pid"] for m in MODELS}
    attempts = {m: 0 for m in MODELS}
    for restart in launch.get("restarts", []):
        pids[restart["model"]] = restart["pid"]
        attempts[restart["model"]] += 1
    children = []
    last_processed = -1
    last_audit = -1
    last_bundle = -1
    permanent = {}
    while True:
        # Reap child runners created by this supervisor.
        children = [child for child in children if child.poll() is None]
        status = collect_status(pids)
        for model, group in status["models"].items():
            if group["completed_episodes"] == 72 or group["process_alive"] or model in permanent:
                continue
            if attempts[model] >= 3:
                permanent[model] = "Stopped after three recorded restarts; inspect retained attempt logs"
                continue
            cleanup_owned_worker(pids[model], model)
            attempt = attempts[model] + 1
            archived = RUN / model / "incomplete_attempts" / f"before_restart_{attempt:02d}"
            for episode in (RUN / model / "episodes").glob("*"):
                if episode.is_dir() and not (episode / "summary.json").exists():
                    archived.mkdir(parents=True, exist_ok=True)
                    episode.rename(archived / episode.name)
            completion = RUN / model / "completion.json"
            if completion.exists():
                archived.mkdir(parents=True, exist_ok=True)
                completion.rename(archived / "completion.json")
            log_path = RUN / f"{model}_supervisor_retry{attempt}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(original[model]["command"], stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True, cwd=ROOT,
                                           env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"))
            children.append(process)
            pids[model] = process.pid
            attempts[model] = attempt
            launch.setdefault("restarts", []).append({"model": model, "pid": process.pid, "utc": utc(),
                                                       "reason": "recorded runner stopped before all 72 episodes completed",
                                                       "log": str(log_path.relative_to(ROOT))})
            write_json(RUN / "launch.json", launch)
            print(json.dumps({"restarted": model, "pid": process.pid, "attempt": attempt}), flush=True)
        status = collect_status(pids)
        status["restart_counts"] = attempts
        status["permanent_failures"] = permanent
        write_json(RUN / "status.json", status)
        completed = status["completed_episodes"]
        if completed != last_processed:
            run_postprocess("refresh_rollout_dashboard.py", ["--run-root", str(RUN)])
            last_processed = completed
        if completed - last_audit >= 8:
            run_postprocess("audit_recorded_rollouts.py", ["--run-root", str(RUN)])
            last_audit = completed
        if completed - last_bundle >= 24:
            run_postprocess("refresh_rollout_dashboard.py", ["--run-root", str(RUN), "--bundle"])
            last_bundle = completed
        print(json.dumps({"utc": utc(), "completed_episodes": completed,
                          "steps": status["recorded_control_timesteps"], "models": status["models"]}), flush=True)
        finished = all(g["completed_episodes"] == 72 or m in permanent for m, g in status["models"].items())
        if finished:
            status["status"] = "finalizing" if not permanent else "partial_failure"
            write_json(RUN / "status.json", status)
            audited = run_postprocess("audit_recorded_rollouts.py", ["--run-root", str(RUN)])
            exported = None
            if not permanent and audited:
                exported = True
                for model in MODELS:
                    targets = [("vlm_hidden", "last_prompt")]
                    if model.startswith("pi"):
                        targets.append(("action_expert_hidden", "first_action_token"))
                    for feature, region in targets:
                        output = RUN / "probe_matrices" / f"{model}__{feature}__{region}__t0_fold0.npz"
                        if output.exists() and output.with_suffix(".json").exists():
                            continue
                        exported &= run_postprocess("export_probe_matrix.py", ["--model", model, "--run-root", str(RUN),
                                                    "--feature", feature, "--region", region, "--out", str(output)])
            status["status"] = "complete" if not permanent and audited and exported else "needs_attention"
            status["final_audit_passed"] = audited
            status["probe_export_passed"] = exported
            status["finished_utc"] = utc()
            write_json(RUN / "status.json", status)
            run_postprocess("refresh_rollout_dashboard.py", ["--run-root", str(RUN), "--bundle"])
            write_json(RUN / "completion_report.json", status)
            print("FINAL", json.dumps(status), flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
