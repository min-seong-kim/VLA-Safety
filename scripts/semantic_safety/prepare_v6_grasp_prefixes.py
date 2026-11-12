"""Create and replay-verify physical grasp/lift prefixes for a conditional motor test."""

import argparse
import json
import math
import multiprocessing
from pathlib import Path
import traceback

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rollout_common import ROOT, bootstrap, observation_payload, write_json
from semantic_pairs_v6 import ASSETS, make_environment, sha256
from semantic_v6_diagnostics import RUN, assert_initial, check_sides, items


def profiles(item):
    extent = np.array(json.loads((ASSETS / "manifest.json").read_text())[item["actor_asset"]]["extent"])
    selected = json.loads((ROOT / "data/motor_skill_b_v1/profile_search/selected_profiles.json").read_text())
    base = dict(offset=[0., 0.], height=float(min(extent[2] * .5, .075)),
                yaw=-math.pi / 2 if extent[0] < extent[1] else 0.)
    candidates = []
    if item["actor_asset"] in selected:
        candidates.append({k: selected[item["actor_asset"]][k] for k in base})
    candidates += [base, {**base, "yaw": base["yaw"] + math.pi / 2}]
    if item["actor_asset"] == "kettle":
        # The v6 body cannot be grasped like a small cup; probe the handle.
        for x in [-.060, .060]:
            for yaw in [0., -math.pi / 2]:
                candidates.append(dict(offset=[x, 0.], height=.08, yaw=yaw))
    else:
        axis = int(np.argmax(extent[:2]))
        for sign in [1., -1.]:
            off = [0., 0.]; off[axis] = float(extent[axis] * .27 * sign)
            candidates.append({**base, "offset": off})
        candidates += [{**base, "height": max(.004, base["height"] - .012)},
                       {**base, "height": base["height"] + .014}]
    unique = {json.dumps(p, sort_keys=True): p for p in candidates}
    return list(unique.values())[:8]


class GraspTeacher:
    def __init__(self, env, oracle, profile):
        self.oracle = oracle; self.profile = profile
        self.stage = "approach"; self.stage_steps = 0
        self.orientation = Rotation.from_rotvec([0., 0., profile["yaw"]]).as_matrix() @ np.array(
            [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])

    def change(self, stage):
        self.stage = stage; self.stage_steps = 0

    def action(self, env, obs):
        current = np.asarray(obs["robot0_eef_pos"])
        actor = env.sim.data.body_xpos[self.oracle.actor]
        grasp = actor + np.array([*self.profile["offset"], self.profile["height"] - .0035])
        close = self.stage in {"close", "lift", "hold"}
        if self.stage == "approach":
            goal = grasp.copy(); goal[2] = max(1.08, grasp[2] + .12)
            if np.linalg.norm(current - goal) < .012 or self.stage_steps >= 60:
                self.change("descend")
        elif self.stage == "descend":
            goal = grasp
            if np.linalg.norm(current - goal) < .005 or self.stage_steps >= 60:
                self.grasp_position = current.copy(); self.change("close")
        elif self.stage == "close":
            goal = self.grasp_position
            if self.stage_steps >= 20:
                self.change("lift")
        else:
            goal = self.grasp_position.copy(); goal[2] += .085
            if self.stage == "lift" and (np.linalg.norm(current - goal) < .005 or self.stage_steps >= 70):
                self.change("hold")
        delta = np.clip((goal - current) * 12., -.45, .45)
        rotation = Rotation.from_matrix(self.orientation @ env.robots[0].controller.ee_ori_mat.T).as_rotvec()
        self.stage_steps += 1
        return np.r_[delta, np.clip(rotation * 1.5, -.7, .7), 1. if close else -1.].astype(np.float32)


def trial(item, profile):
    wrapper = None
    try:
        wrapper, obs, oracle = make_environment(item, 64); env = wrapper.env
        initial = oracle.reference_position(env).copy(); previous = initial.copy()
        teacher = GraspTeacher(env, oracle, profile); actions = []; stable = 0
        for step in range(270):
            action = teacher.action(env, obs); actions.append(action)
            obs, _, _, _ = env.step(action); metric = oracle.measure(env)
            position = oracle.reference_position(env)
            speed = float(np.linalg.norm(position - previous) * 20); previous = position.copy()
            valid = (metric["lifted_4cm"] and speed < .03 and np.linalg.norm(position[:2] - initial[:2]) <= .04
                     and not metric["transport_success"] and metric["distance_to_goal_xy_m"] > .1)
            stable = stable + 1 if valid and teacher.stage == "hold" else 0
            if stable >= 10:
                return dict(passed=True, steps=step+1, profile=profile, speed_m_s=speed,
                            actor_xy_displacement_m=float(np.linalg.norm(position[:2]-initial[:2])),
                            lift_m=float(position[2]-initial[2]), actions=np.asarray(actions))
        return dict(passed=False, steps=len(actions), profile=profile, final_stage=teacher.stage,
                    ever_grasped=bool(metric["ever_grasped"]), ever_lifted=bool(metric["ever_lifted_4cm"]))
    finally:
        if wrapper is not None: wrapper.close()


def process(item):
    bootstrap(); cv2.setNumThreads(1)
    folder = RUN / "grasp_prefixes" / item["item_id"]; folder.mkdir(parents=True, exist_ok=True)
    signature = dict(item_id=item["item_id"], code_sha256=sha256(__file__),
                     dataset_sha256=sha256(Path(__file__).with_name("semantic_pairs_v6.py")))
    path = folder / "summary.json"
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached["signature"] == signature
        return dict(item_id=item["item_id"], passed=cached["passed"], cached=True)
    attempts = []; wrapper = None
    try:
        for profile in profiles(item):
            result = trial(item, profile)
            attempts.append({k:v for k,v in result.items() if k != "actions"})
            if not result["passed"]: continue
            actions = result["actions"]
            # Reconstruct the accepted prefix from the frozen 384px scene.
            wrapper, obs, oracle = make_environment(item, 384); env = wrapper.env
            assert_initial(observation_payload(obs), item); pixels = check_sides(env, oracle, item)
            states = [env.sim.get_state().flatten().copy()]; events = []; held = 0
            previous = oracle.reference_position(env).copy()
            for step, action in enumerate(actions, 1):
                obs, _, _, _ = env.step(action)
                metric = {k:v.item() if isinstance(v,np.generic) else v for k,v in oracle.measure(env).items()}
                position = oracle.reference_position(env); speed = float(np.linalg.norm(position-previous)*20)
                previous = position.copy(); states.append(env.sim.get_state().flatten().copy())
                stable = (metric["lifted_4cm"] and speed < .03 and metric["displacement_xy_m"] <= .04
                          and not metric["transport_success"] and metric["distance_to_goal_xy_m"] > .1)
                held = held + 1 if stable else 0
                events.append(dict(step=step, speed_m_s=speed, **metric))
            assert held >= 10 and events[-1]["grasped"]
            payload = observation_payload(obs)
            np.savez_compressed(folder/"prefix.npz", actions=actions, sim_states=np.asarray(states), **payload)
            cv2.imwrite(str(folder/"held.png"), np.ascontiguousarray(payload["image"][::-1,::-1,::-1]))
            write_json(folder/"events.json", events)
            record = dict(item_id=item["item_id"], actor_asset=item["actor_asset"], passed=True,
                          signature=signature, attempts=attempts, selected_profile=profile,
                          prefix_steps=len(actions), stable_held_steps=held, image_x_centers=pixels,
                          action_replay_verified=True, privileged_teacher=True, teleport=False,
                          policy_used=False, final_metric=events[-1], prefix_sha256=sha256(folder/"prefix.npz"))
            write_json(path, record)
            return dict(item_id=item["item_id"], passed=True, tries=len(attempts), prefix_steps=len(actions))
        write_json(path, dict(item_id=item["item_id"], actor_asset=item["actor_asset"], passed=False,
                             signature=signature, attempts=attempts, reason="no validated grasp prefix in bounded profile search"))
        return dict(item_id=item["item_id"], passed=False, tries=len(attempts))
    except Exception:
        error = traceback.format_exc(); write_json(folder/"error.json", dict(traceback=error))
        return dict(item_id=item["item_id"], error=error)
    finally:
        if wrapper is not None: wrapper.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = items()
    if args.limit: rows = rows[:args.limit]
    with multiprocessing.get_context("spawn").Pool(args.workers, maxtasksperchild=8) as pool:
        results = list(pool.imap(process, rows))
        for result in results: print(json.dumps(result), flush=True)
    assert not any("error" in r for r in results), "Inspect prefix errors"
    write_json(RUN/"grasp_prefix_coverage.json", dict(attempted=len(rows), accepted=sum(r["passed"] for r in results),
               results=results, scope="feasibility-selected physical grasp states, not model successes or training data"))


if __name__ == "__main__":
    main()
