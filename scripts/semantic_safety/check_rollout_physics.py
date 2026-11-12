"""Measure passive scene drift with a stationary robot; excluded from model/probe data."""

import json

import numpy as np

from rollout_common import ROOT, bootstrap, write_json
from rollout_safety_env import DATA, safety_environment


def main():
    bootstrap()
    output = ROOT / "runs/semantic_rollout_physics_checks_v3"
    output.mkdir(exist_ok=True)
    specifications = [json.loads(line) for line in (DATA / "episodes.jsonl").read_text().splitlines()]
    reports = []
    for spec in (row for row in specifications if not row["task_is_unsafe"]):
        wrapper, obs, instruction, metadata, oracle = safety_environment(spec)
        env = wrapper.env
        names = [p["name"] for p in oracle.scene["props"] if p["movable"]]
        ids = [oracle.body_ids[name] for name in names]
        positions = [env.sim.data.body_xpos[ids].copy()]
        for step in range(160):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            positions.append(env.sim.data.body_xpos[ids].copy())
        positions = np.asarray(positions)
        drift = np.linalg.norm(positions - positions[0], axis=-1)
        maximum = {name: float(drift[:, index].max()) for index, name in enumerate(names)}
        flagged = [name for name, value in maximum.items() if value > .03]
        report = {"family_id": spec["family_id"], "scene_seed": spec["scene_seed"],
                  "control_steps": 160, "simulated_seconds": 8,
                  "command": [0.0] * 6 + [-1.0], "movable_objects": names,
                  "max_passive_displacement_m": maximum, "flagged_objects": flagged,
                  "flag_rule": "more than 3 cm displacement after standard initial settling under hold control",
                  "not_a_model_rollout": True, "exclude_from_probe": True}
        np.savez_compressed(output / (spec["family_id"] + ".npz"), positions=positions, object_names=names)
        write_json(output / (spec["family_id"] + ".json"), report)
        reports.append(report)
        print(json.dumps(report), flush=True)
        wrapper.close()
    write_json(output / "summary.json", {"scenes_checked": len(reports),
               "flagged_families": [r["family_id"] for r in reports if r["flagged_objects"]],
               "reports": reports, "test_scope": "passive scene stability, not policy task competence"})


if __name__ == "__main__":
    main()
