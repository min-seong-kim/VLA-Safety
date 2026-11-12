"""Validate a recorded user pause and explicitly resume the preserved experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from rollout_common import ROOT, write_json
from supervise_expansion_v5_experiment import DRIVER, RUN, find_owned


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--check-only",action="store_true",help="Validate saved files without starting any process")
    args=parser.parse_args()
    assert (RUN/"pause_requested.json").exists(),"No recorded user pause; inspect current execution status first"
    assert not find_owned(DRIVER),"An experiment driver is already running"
    assert not find_owned("supervise_expansion_v5_experiment.py"),"An experiment supervisor is already running"
    state=json.loads((RUN/"resume_state.json").read_text())
    assert state["state"]=="paused"
    for relative,expected in state["files_sha256"].items():
        assert digest(ROOT/relative)==expected,"Saved file changed: "+relative
    ready=json.loads((RUN/"experiment_data_ready.json").read_text())
    for relative,expected in ready["fingerprint"].items():
        assert digest(ROOT/relative)==expected,"Frozen data definition changed: "+relative
    overrides=json.loads((RUN/"execution_overrides.json").read_text())
    assert overrides["extension_enabled"] is False,"The user deferred additional training; inspect any changed override"
    assert (ROOT/"openpi/.venv/bin/python").exists() and (ROOT/"envs/libero-render/bin/python").exists()
    assert (ROOT/"checkpoints/pi05_libero_pytorch").is_dir()
    print(json.dumps(dict(saved_files_verified=True,extension_enabled=False,
        models=[dict(scope=r["scope"],saved_step=r["saved_step"],target_step=r["target_step"]) for r in state["models"]],
        started=False if args.check_only else "launching")),flush=True)
    if args.check_only:
        return
    command=[str(ROOT/"openpi/.venv/bin/python"),"scripts/semantic_safety/supervise_expansion_v5_experiment.py",
             "--workers","64","--resume"]
    with (RUN/"supervisor.log").open("a") as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(2)
    assert process.poll() is None,"Supervisor exited; inspect supervisor.log"
    write_json(RUN/"explicit_resume_launch.json",dict(pid=process.pid,command=command,updated_unix=time.time(),
        requested_by="explicit invocation of resume_expansion_v5_experiment.py",extension_enabled=False))
    print("EXPERIMENT_RESUMED",process.pid,flush=True)


if __name__=="__main__":
    main()
