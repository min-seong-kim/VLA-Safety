"""Audit adapter identity, raw input alignment, actions, scores, and encoded videos."""

import argparse
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np

from rollout_common import ROOT, write_json
from semantic_pairs_v6 import TransportGate, sha256


def audit(out):
    run=json.loads((out/"run.json").read_text());model=json.loads((out/"model.json").read_text())
    rows=[json.loads(s) for s in (out/"episodes.jsonl").read_text().splitlines()]
    assert json.loads((out/"completion.json").read_text())["complete"]
    assert {r["item_id"] for r in rows}==set(run["item_ids"])
    assert model["adapter_sha256"]==run["adapter_sha256"]
    if run["adapter"]:assert sha256(Path(run["adapter"]))==run["adapter_sha256"]
    total=decisions=0
    for row in rows:
        folder=out/"episodes"/row["item_id"];steps=row["steps"]
        assert row["run_digest"]==run["digest"]
        events=[json.loads(s) for s in (folder/"events.jsonl").read_text().splitlines()]
        assert len(events)==steps and [e["control_step"] for e in events]==list(range(1,steps+1))
        with h5py.File(folder/"trajectory.h5") as data:
            item=json.loads(data.attrs["item_json"])
            assert len(data["observations/state"])==len(data["observations/sim_state"])==steps+1
            for key in ["observations/state","observations/sim_state","actions/raw","actions/executed","predicted_action_chunks"]:
                assert np.isfinite(data[key][:]).all()
            starts=data["chunk_start_steps"][:]
            assert np.array_equal(starts,np.arange(0,steps,10))
            assert np.array_equal(data["flow_noise_seeds"][:],starts+item["noise_seed"])
            raw=data["predicted_action_chunks"][:].reshape(-1,7)[:steps]
            assert np.array_equal(raw,data["actions/raw"][:])
            expected=raw.copy();expected[:,:6]=np.clip(expected[:,:6],-1,1);expected[:,6]=np.where(expected[:,6]>0,1.,-1.)
            assert np.array_equal(expected,data["actions/executed"][:])
            if item.get("kind")!="native":
                assert steps==run["horizon"]
                with np.load(ROOT/item["observation"]) as initial:
                    for camera,key in [("main_rgb_raw","image"),("wrist_rgb_raw","wrist_image")]:
                        assert np.array_equal(data["policy_inputs/"+camera][0],initial[key])
                    assert np.allclose(data["observations/state"][0],initial["state"],atol=1e-8,rtol=0)
                initial_metric=json.loads(data.attrs.get("initial_metric_json","{}"))
                start=initial_metric.get("actor_reference_position",item.get("validation",{}).get("actual_actor_reference_position"))
                if start is not None:
                    start=np.asarray(start)
                    gate=TransportGate(item,start);gate.update(start,False)
                    for event in events:
                        replay=gate.update(event["actor_reference_position"],event["grasped"])
                        for key in ["ever_grasped","ever_lifted_4cm","transport_success","sustained_transport_success"]:
                            assert replay[key]==event[key],(row["item_id"],event["control_step"],key)
                assert all(row[k]==events[-1][k] for k in ["ever_grasped","ever_lifted_4cm","transport_success","sustained_transport_success"])
                assert row["task_success"] is None and row["harm_observed"] is None
            decisions+=len(starts)
        info=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0", "-show_entries","stream=width,height,nb_read_frames","-of","json",str(folder/"views.mp4")]))["streams"][0]
        assert info["width"]==768 and info["height"]==432 and int(info["nb_read_frames"])==steps//row.get("video_stride",1)+1
        total+=steps
    result=dict(passed=True,episodes=len(rows),control_steps=total,policy_decisions=decisions,
                adapter_identity_checked=True,initial_inputs_match=True,finite_actions_states=True,
                video_frames_checked=True,inference=json.loads((out/"inference_audit.json").read_text()))
    write_json(out/"recording_audit.json",result);print(json.dumps(result))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--run",type=Path,required=True);audit(p.parse_args().run)
