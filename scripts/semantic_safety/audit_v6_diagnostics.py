"""Audit action/state alignment and diagnostic predicates from saved episodes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from rollout_common import ROOT, write_json
from semantic_pairs_v6 import DATA, TransportGate, sha256
from semantic_v6_diagnostics import RUN, DiagnosticMetrics, items, prompt


def audit(out):
    run = json.loads((out/"run.json").read_text())
    assert json.loads((out/"completion.json").read_text())["complete"]
    assert sha256(DATA/"items.jsonl") == run["dataset_sha256"]
    assert sha256(DATA/"assets/manifest.json") == run["assets_sha256"]
    item_map = {i["item_id"]:i for i in items()}
    rows = [json.loads(x) for x in (out/"episodes.jsonl").read_text().splitlines()]
    assert {r["item_id"] for r in rows} == set(run["items"])
    steps = decisions = 0
    for row in rows:
        item = item_map[row["item_id"]]; folder = out/"episodes"/row["item_id"]
        assert row["run_digest"] == run["digest"] and row["steps"] == 360
        assert row["instruction"] == prompt(item, run["arm"])
        events = [json.loads(x) for x in (folder/"events.jsonl").read_text().splitlines()]
        assert [e["control_step"] for e in events] == list(range(1,361))
        held = run["arm"] == "held_transport_name"
        initial_path = RUN/"grasp_prefixes"/row["item_id"]/"prefix.npz" if held else ROOT/item["observation"]
        with h5py.File(folder/"trajectory.h5", "r") as data, np.load(initial_path) as initial:
            assert data["actions/executed"].shape == (360,7)
            assert len(data["observations/state"]) == len(data["observations/sim_state"]) == 361
            for name in ["actions/raw","actions/executed","observations/state","observations/sim_state","predicted_action_chunks"]:
                assert np.isfinite(data[name][:]).all()
            starts = data["chunk_start_steps"][:]
            assert np.array_equal(starts, np.arange(0,360,10))
            assert np.array_equal(data["flow_noise_seeds"][:], item["noise_seed"]+starts)
            for name, source in [("main_rgb_raw","image"),("wrist_rgb_raw","wrist_image")]:
                assert data["policy_inputs/"+name].shape == (36,384,384,3)
                assert np.array_equal(data["policy_inputs/"+name][0], initial[source])
            assert np.allclose(data["observations/state"][0], initial["state"], atol=1e-8, rtol=0)
            raw = data["predicted_action_chunks"][:].reshape(360,7)
            assert np.array_equal(raw, data["actions/raw"][:])
            expected = raw.copy(); expected[:,:6] = np.clip(expected[:,:6], -1, 1)
            expected[:,6] = np.where(expected[:,6] > 0, 1., -1.)
            assert np.array_equal(expected, data["actions/executed"][:])
            decisions += len(starts)
        gate = TransportGate(item, item["validation"]["actual_actor_reference_position"])
        extra = DiagnosticMetrics(item)
        if held:
            pref = json.loads((RUN/"grasp_prefixes"/row["item_id"]/"summary.json").read_text())
            pos = np.array(pref["final_metric"]["actor_reference_position"])
            gate.previous = pos.copy(); gate.previous_lifted_grasp = True
            first = extra.update(gate.update(pos, True), 0)
            assert first["lifted_4cm"] and first["carried_xy_m"] == 0
        else:
            first = extra.update(gate.update(gate.start, False), 0)
        assert not first["xy_only_transport_success"] and not first["transport_success"]
        keys = ["ever_grasped","ever_lifted_4cm","goal_hit_this_step","transport_success","sustained_transport_success",
                "xy_only_transport_success","xy_only_sustained_success","sustained_lift_10_steps"]
        for event in events:
            replay = extra.update(gate.update(event["actor_reference_position"],event["grasped"]),event["control_step"])
            assert all(bool(replay[k]) == event[k] for k in keys), (row["item_id"],event["control_step"])
            assert not event["transport_success"] or event["xy_only_transport_success"]
        summary_keys = [k for k in keys if k != "goal_hit_this_step"]
        assert all(row[k] == events[-1][k] for k in summary_keys)
        assert row["task_success"] is None and row["harm_observed"] is None
        steps += 360
    result = dict(passed=True,episodes=len(rows),control_steps=steps,policy_decisions=decisions,
                  initial_inputs_match=True,finite_states_actions=True,noise_seeds_matched=True,
                  diagnostic_predicates_replayed=True,source_labels_not_safety_evaluation=True,
                  inference=json.loads((out/"inference_audit.json").read_text()))
    write_json(out/"recording_audit.json",result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("--run",type=Path,required=True)
    audit(parser.parse_args().run)
