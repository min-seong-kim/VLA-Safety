"""Positive/negative geometry checks; these teleported states are NOT model rollouts."""

import json
from pathlib import Path

import numpy as np

from rollout_common import ROOT, bootstrap, write_json
from rollout_safety_env import DATA, safety_environment


def main():
    bootstrap()
    specs = [json.loads(line) for line in (DATA / "episodes.jsonl").read_text().splitlines()]
    results = []
    for spec in [s for s in specs if s["task_is_unsafe"]]:
        wrapper, obs, _, metadata, oracle = safety_environment(spec)
        env = wrapper.env
        saved = env.sim.get_state().flatten().copy()
        actor = oracle.roles["unsafe_actor"]
        target = oracle.roles["unsafe_target"]
        obj = env.get_object(actor)
        target_position = env.sim.data.body_xpos[oracle.body_ids[target]].copy()
        extent = np.asarray(oracle.manifest[oracle.props[target]["asset"]]["extent"])
        initial = oracle.evaluate(env, obs)
        assert not initial["risk_proxy"]
        detected = None
        mode = oracle.mode
        offsets = [(0, 0, extent[2] / 2), (0, 0, .01)]
        if mode == "pour":
            offsets = [(0, 0, extent[2] + .04)]
        elif mode == "flame":
            offsets = [(0, 0, extent[2] + .03)]
        elif mode == "access":
            offsets = [(0, 0, .10)]
        elif mode == "drop":
            offsets = [(0.60 - target_position[0], -target_position[1], .80 - target_position[2])]
        else:
            offsets += [(x, y, z) for x in [-.035, 0, .035] for y in [-.035, 0, .035]
                        for z in [.005, .02, max(.01, extent[2] - .005)]]
        for offset in offsets:
            position = target_position + offset
            quat = [np.sqrt(.5), np.sqrt(.5), 0, 0] if mode == "pour" else [1, 0, 0, 0]
            env.sim.set_state_from_flattened(saved)
            env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([position, quat]))
            env.sim.forward()
            oracle.last_positions[actor] = position - [0.02, 0, 0]
            measured = oracle.evaluate(env, obs)
            if measured["risk_proxy"]:
                detected = {"offset": list(offset), "metrics": measured}
                break
        env.sim.set_state_from_flattened(saved)
        env.sim.forward()
        negative_after_restore = oracle.evaluate(env, obs)["risk_proxy"]
        result = {"family_id": spec["family_id"], "initial_negative": not initial["risk_proxy"],
                  "positive_proxy_detectable": detected is not None, "restored_negative": not negative_after_restore,
                  "positive_test": detected, "not_a_model_rollout": True, "exclude_from_probe": True,
                  "test_scope": "geometry/oracle implementation only; no damage physics or trained-policy capability"}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "positive_test"}), flush=True)
        wrapper.close()
    write_json(ROOT / "runs/semantic_oracle_validation_v3.json", results)
    if any(not (r["initial_negative"] and r["positive_proxy_detectable"] and r["restored_negative"]) for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
