"""Verify that recorded RGB/state really match independently replayed action times."""

import json

import h5py
import numpy as np

import benign_adaptation_v1 as b
from rollout_common import observation_payload, write_json


def main():
    rows=b.read_jsonl(b.DATA/"preflight_train_manifest.jsonl")
    chosen=[next(r for r in rows if r.get("condition_index")==i and r["data_type"]!="native_rehearsal")
            for i in [7,74,72,87]]
    evidence=[]
    for row in chosen:
        with h5py.File(b.ROOT/row["path"]) as source:
            spec=json.loads(source.attrs["spec_json"])
            actions=source["actions"][:]
            wrapper,obs,oracle=b.make_environment(spec,224)
            try:
                points={0,len(actions)//2,len(actions)}
                for step in range(len(actions)+1):
                    if step in points:
                        payload=observation_payload(obs)
                        error=float(np.max(np.abs(wrapper.env.sim.get_state().flatten()-source["observations/sim_state"][step])))
                        assert error<1e-6,(row["item_id"],step,error)
                        assert np.array_equal(payload["state"],source["observations/state"][step])
                        for key,name in [("image","main_rgb_raw"),("wrist_image","wrist_rgb_raw")]:
                            expected=source["observations/"+name][step]
                            difference=np.abs(payload[key].astype(np.int16)-expected.astype(np.int16))
                            record=dict(item_id=row["item_id"],control_step=step,camera=name,
                                        image_exact=bool(np.array_equal(payload[key],expected)),
                                        max_rgb_error=int(difference.max()),mean_rgb_error=float(difference.mean()),
                                        max_sim_state_error=error)
                            evidence.append(record)
                            assert record["image_exact"],record
                    if step<len(actions):obs,_,_,_=wrapper.env.step(actions[step])
            finally:wrapper.close()
    report=dict(passed=True,physical_signature=b.signature(),episodes=len(chosen),checks=evidence,
                independent_state_and_rgb_replay=True,all_sampled_pixels_bit_exact=True)
    write_json(b.RUN/"observation_replay_audit.json",report)
    print(json.dumps(dict(passed=True,episodes=len(chosen),camera_checks=len(evidence))),flush=True)


if __name__=="__main__":main()
