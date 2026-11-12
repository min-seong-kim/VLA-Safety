"""Verify an actual interrupted LoRA run against an uninterrupted reference before main training."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

from rollout_common import ROOT, write_json

BASE=ROOT/"runs/semantic_expansion_v5_lora_v1/resume_smoke"
PYTHON=ROOT/"openpi/.venv/bin/python"
TRAINER=ROOT/"scripts/semantic_safety/train_expansion_v5_lora.py"


def current_source_check():
    current=hashlib.sha256(TRAINER.read_bytes()).hexdigest()
    for name in ["interrupted","reference"]:
        config=json.loads((BASE/name/"config.json").read_text())
        assert config["code_sha256"][TRAINER.name]==current
        assert config["lora"]["scope"]=="all" and config["steps_planned"]==8


def wait_for_existing():
    marker=BASE/"existing_controller.json"
    if not marker.exists():
        return
    info=json.loads(marker.read_text())
    started=time.monotonic()
    while True:
        try:
            process=psutil.Process(info["pid"])
            alive=(process.create_time()==info["create_time"] and process.status()!=psutil.STATUS_ZOMBIE)
        except psutil.NoSuchProcess:
            alive=False
        if not alive or (BASE/"audit.json").exists():
            return
        assert time.monotonic()-started<21600,"Existing resume verification did not finish within six hours"
        write_json(BASE/"verification_status.json",dict(state="waiting_for_existing_check",updated_unix=time.time()))
        time.sleep(30)


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    lock=(BASE/"verification.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    wait_for_existing()
    if (BASE/"audit.json").exists():
        assert json.loads((BASE/"audit.json").read_text())["passed"],"Resume comparison failed"
        current_source_check()
        print("INTERRUPTED_TRAINING_COMPARISON_ALREADY_PASSED",flush=True)
        return
    import torch

    # Do not compete with any surviving verification trainer after its controller exited.
    for process in psutil.process_iter(["cmdline"]):
        command=process.info["cmdline"] or []
        if str(BASE) in " ".join(command) and any(Path(arg).name==TRAINER.name for arg in command[1:3]):
            raise RuntimeError("A verification trainer is still running without its recorded controller")
    common=[str(PYTHON),str(TRAINER),"--manifest",str(ROOT/"data/semantic_expansion_v5_lora_v1/pilot_manifest.jsonl"),
            "--scope","all","--steps","8","--batch-size","4","--micro-batch-size","2",
            "--save-every","8","--recovery-every","2","--pilot"]
    folder=BASE/"interrupted"
    interrupted=BASE/"interruption_event.json"
    with (BASE/"interrupted.log").open("a") as output:
        command=common+["--out",str(folder)]
        if not interrupted.exists():
            assert not (folder/"config.json").exists(),"An unrecorded partial verification needs inspection"
            process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=output,
                stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                log=folder/"training.jsonl"
                if log.exists() and len(log.read_text().splitlines())>=3:
                    os.killpg(process.pid,signal.SIGTERM)
                    process.wait(timeout=20)
                    state=torch.load(folder/"training_state.pt",map_location="cpu",weights_only=False)
                    assert 0<state["step"]<8
                    write_json(interrupted,dict(signal="SIGTERM",durable_step=state["step"],
                        interrupted_pid=process.pid,updated_unix=time.time()))
                    break
                time.sleep(.05)
            assert interrupted.exists(),"Training exited before the intentional interruption"
        subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=output,stderr=subprocess.STDOUT,check=True)
    with (BASE/"reference.log").open("a") as output:
        subprocess.run(common+["--out",str(BASE/"reference")],cwd=ROOT,env=os.environ.copy(),
                       stdout=output,stderr=subprocess.STDOUT,check=True)
    current_source_check()
    adapters=[torch.load(BASE/name/"adapter_step_00008.pt",map_location="cpu",weights_only=False)["state_dict"]
              for name in ["interrupted","reference"]]
    a,b=adapters
    exact=all(torch.equal(a[n],b[n]) for n in a)
    logs=[[json.loads(line) for line in (BASE/name/"training.jsonl").read_text().splitlines()]
          for name in ["interrupted","reference"]]
    assert all([r["step"] for r in rows]==list(range(1,9)) for rows in logs)
    assert list((folder/"interrupted_logs").glob("*.jsonl")),"No interrupted log was preserved"
    lossdiff=max(abs(x["loss"]-y["loss"]) for x,y in zip(*logs))
    audit=dict(passed=exact and lossdiff==0.,intentional_process_termination=True,
        resumed_from_step=json.loads(interrupted.read_text())["durable_step"],full_optimizer_restored=True,
        adapter_tensors_bitwise_equal=exact,max_adapter_difference=max(float((a[n]-b[n]).abs().max()) for n in a),
        max_loss_difference=lossdiff,training_log_steps_unique=True,scope="all",steps=8,
        trainer_sha256=hashlib.sha256(TRAINER.read_bytes()).hexdigest())
    write_json(BASE/"audit.json",audit)
    assert audit["passed"],audit
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    main()
