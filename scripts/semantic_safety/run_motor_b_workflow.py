"""Complete B training, validation selection, and final evaluation with local logs."""

import json
from pathlib import Path
import subprocess
import time

from rollout_common import ROOT, write_json


RUN = ROOT / "runs/motor_lora_b_v1"
DATA = ROOT / "data/motor_skill_b_v1/demonstrations"
SIM = ROOT / "envs/libero-render/bin/python"
POLICY = ROOT / "openpi/.venv/bin/python"
SCRIPTS = ROOT / "scripts/semantic_safety"


def launch(name, python, script, arguments):
    logs = RUN / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [str(python), str(SCRIPTS / script), *map(str, arguments)]
    log = (logs / f"{name}.log").open("a", buffering=1)
    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    with (RUN / "local_jobs.jsonl").open("a", buffering=1) as record:
        record.write(json.dumps({"name": name, "pid": process.pid, "command": command, "started_unix": time.time()}) + "\n")
    return process, log


def wait_for_socket(path, process):
    deadline = time.monotonic() + 600
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError("Policy server stopped before becoming ready")
        if time.monotonic() > deadline:
            raise TimeoutError(path)
        time.sleep(2)


def checked_wait(process, log):
    status = process.wait()
    log.close()
    if status != 0:
        raise RuntimeError(f"Process {process.pid} exited {status}")


def main():
    while not (DATA / "coverage.json").exists():
        time.sleep(5)
    coverage = json.loads((DATA / "coverage.json").read_text())
    assert coverage["complete"] and coverage["attempts"] == 648
    training = launch("training", POLICY, "train_motor_lora.py", [
        "--manifest", DATA / "manifest.jsonl", "--out", RUN / "training", "--steps", 3000,
        "--batch-size", 16, "--save-every", 500,
    ])
    socket = Path(f"/tmp/vla_motor_b_validation_{int(time.time())}.sock")
    server = launch("validation_policy", POLICY, "motor_policy_server.py", ["--socket", socket])
    wait_for_socket(socket, server[0])
    validation = launch("validation_B", SIM, "evaluate_motor_policy.py", [
        "--socket", socket, "--design", DATA / "frozen_design.json", "--out", RUN / "validation_B",
        "--commands", RUN / "validation_B_commands.json", "--batch-size", 6,
    ])
    checked_wait(*training)
    checked_wait(*validation)
    checked_wait(*server)
    audit = json.loads((RUN / "training/freeze_audit.json").read_text())
    assert audit["all_original_weights_unchanged"] and audit["fixed_input_vlm_hidden_exact"]
    candidates = []
    for step in (500, 1500, 3000):
        aggregate = json.loads((RUN / f"validation_B_{step:05d}/aggregate.json").read_text())
        candidates.append({"step": step, **aggregate})
    selected = max(candidates, key=lambda row: (row["successes"], row["lifts"], row["grasps"], -row["step"]))
    selection = {"selected_step": selected["step"], "adapter": selected["adapter"],
                 "rule": "validation successes, then lifts, then grasps, then earlier checkpoint",
                 "test_results_read_for_selection": False, "candidates": candidates}
    write_json(RUN / "selection.json", selection)
    commands = [
        {"adapter": selected["adapter"], "out": str(RUN / "test_B"), "split": "test", "kind": "motor"},
        {"adapter": selected["adapter"], "out": str(RUN / "native_B"), "kind": "native"},
        {"adapter": "original", "out": str(RUN / "transfer_A"), "kind": "transfer"},
        {"adapter": selected["adapter"], "out": str(RUN / "transfer_B"), "kind": "transfer"},
    ]
    write_json(RUN / "final_commands.json", commands)
    socket = Path(f"/tmp/vla_motor_b_final_{int(time.time())}.sock")
    server = launch("final_policy", POLICY, "motor_policy_server.py", ["--socket", socket])
    teacher = launch("teacher_test_ceiling", SIM, "audit_motor_physics.py", [
        "--design", DATA / "frozen_design.json", "--out", RUN / "teacher_test", "--mode", "teacher", "--workers", 4,
    ])
    wait_for_socket(socket, server[0])
    evaluation = launch("final_B_and_transfer", SIM, "evaluate_motor_policy.py", [
        "--socket", socket, "--design", DATA / "frozen_design.json", "--out", RUN / "test_B",
        "--commands", RUN / "final_commands.json", "--batch-size", 6,
    ])
    checked_wait(*evaluation)
    checked_wait(*server)
    checked_wait(*teacher)
    write_json(RUN / "workflow_completed.json", {"complete": True, "selected_step": selected["step"],
                                                  "probe_trained": False, "finished_unix": time.time()})


if __name__ == "__main__":
    main()
