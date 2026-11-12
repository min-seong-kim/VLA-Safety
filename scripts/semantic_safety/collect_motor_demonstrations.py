"""Collect real closed-loop actions from a privileged scripted pick/place teacher."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import traceback

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from motor_skill_env import ASSETS, DATA, actor_assets, make_environment, make_spec, write_protocol
from rollout_common import bootstrap, state_vector, write_json


def default_profile(asset):
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    extent = np.asarray(manifest[asset]["extent"])
    yaw = -math.pi / 2 if extent[0] < extent[1] else 0.
    profile = {"offset": [0., 0.], "height": float(min(extent[2] * .5, .075)), "yaw": yaw,
            "max_translation_action": .7, "position_gain": 12., "close_steps": 16}
    if asset == "knife":
        profile.update(offset=[0., .055], height=.018)
    if asset == "pan":
        profile.update(offset=[-.07, 0.], height=.02, yaw=0.)
    if asset in {"bread", "moldy_bread"}:
        profile.update(offset=[.033, 0.], height=.018, yaw=0.)
    return profile


class ScriptedTeacher:
    def __init__(self, env, oracle, profile):
        self.oracle = oracle
        self.profile = profile
        self.stage = "approach"
        self.stage_steps = 0
        self.grasp_position = None
        self.relative_actor = None
        yaw = profile["yaw"]
        rz = Rotation.from_rotvec([0., 0., yaw]).as_matrix()
        self.orientation = rz @ np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])

    def change(self, stage):
        self.stage = stage
        self.stage_steps = 0

    def action(self, env, obs, metric):
        controller = env.robots[0].controller
        actor = env.sim.data.body_xpos[self.oracle.actor].copy()
        target = env.sim.data.body_xpos[self.oracle.target].copy()
        current = np.asarray(obs["robot0_eef_pos"])
        offset = np.asarray([*self.profile["offset"], self.profile["height"] - .0035])
        grasp_point = actor + offset
        stage = self.stage
        close = False
        if stage == "approach":
            goal = grasp_point.copy(); goal[2] = max(1.08, grasp_point[2] + .13)
            if np.linalg.norm(current - goal) < .016 or self.stage_steps >= 60:
                self.change("descend")
        elif stage == "descend":
            goal = grasp_point
            if np.linalg.norm(current - goal) < .006 or self.stage_steps >= 60:
                self.grasp_position = current.copy()
                self.change("close")
        elif stage == "close":
            goal = self.grasp_position
            close = True
            if self.stage_steps >= self.profile["close_steps"]:
                self.relative_actor = actor - current
                self.change("lift")
        elif stage == "lift":
            goal = self.grasp_position.copy(); goal[2] += .17
            close = True
            if np.linalg.norm(current - goal) < .012 or self.stage_steps >= 60:
                self.relative_actor = actor - current
                self.change("transport")
        elif stage == "transport":
            self.relative_actor = actor - current
            goal = target - self.relative_actor
            goal[2] = max(1.12, self.grasp_position[2] + .17)
            close = True
            if np.linalg.norm(current - goal) < .012 or self.stage_steps >= 80:
                self.change("lower")
        elif stage == "lower":
            self.relative_actor = actor - current
            goal = target - self.relative_actor
            goal[2] = target[2] + .045 - self.relative_actor[2]
            close = True
            if np.linalg.norm(current - goal) < .009 or self.stage_steps >= 60:
                self.release_position = current.copy()
                self.change("release")
        elif stage == "release":
            goal = self.release_position
            if self.stage_steps >= 16:
                self.change("retract")
        else:
            goal = self.release_position.copy(); goal[2] += .16
        delta = np.clip((goal - current) * self.profile["position_gain"],
                        -self.profile["max_translation_action"], self.profile["max_translation_action"])
        rotation = Rotation.from_matrix(self.orientation @ controller.ee_ori_mat.T).as_rotvec()
        rotation = np.clip(rotation * 1.5, -.7, .7)
        self.stage_steps += 1
        return np.r_[delta, rotation, 1. if close else -1.].astype(np.float32)


def record_teacher(spec, profile, folder, horizon=360, save_video=False):
    folder.mkdir(parents=True, exist_ok=True)
    wrapper, obs, oracle = make_environment(spec)
    env = wrapper.env
    teacher = ScriptedTeacher(env, oracle, profile)
    initial_state = env.sim.get_state().flatten().copy()
    xml = env.sim.model.get_xml()
    frames, wrists, states, sim_states, actions, metrics, stages = [], [], [], [], [], [], []
    metric = oracle.measure(env)
    started = time.monotonic()
    try:
        for step in range(horizon):
            frames.append(obs["agentview_image"].copy())
            wrists.append(obs["robot0_eye_in_hand_image"].copy())
            states.append(state_vector(obs))
            sim_states.append(env.sim.get_state().flatten().copy())
            action = teacher.action(env, obs, metric)
            stages.append(teacher.stage)
            actions.append(action)
            obs, _, _, _ = env.step(action)
            metric = oracle.measure(env)
            metrics.append({"step": step, "teacher_stage": teacher.stage, **metric})
            if metric["success"] and teacher.stage == "retract":
                break
        frames.append(obs["agentview_image"].copy())
        wrists.append(obs["robot0_eye_in_hand_image"].copy())
        states.append(state_vector(obs))
        sim_states.append(env.sim.get_state().flatten().copy())
        with h5py.File(folder / "trajectory.h5", "w") as f:
            f.attrs["spec_json"] = json.dumps(spec)
            f.attrs["teacher_profile_json"] = json.dumps(profile)
            f.attrs["success"] = bool(metric["success"])
            for key, value in {"observations/main_rgb_raw": frames, "observations/wrist_rgb_raw": wrists,
                               "observations/state": states, "observations/sim_state": sim_states,
                               "actions": actions, "initial_sim_state": initial_state}.items():
                f.create_dataset(key, data=np.asarray(value), compression="lzf", shuffle=True)
        (folder / "model.xml").write_text(xml)
        (folder / "events.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metrics))
        cv2.imwrite(str(folder / "initial.png"), np.ascontiguousarray(frames[0][::-1, ::-1, ::-1]))
        if save_video:
            out = cv2.VideoWriter(str(folder / "views.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
            for main, wrist in zip(frames, wrists):
                image = np.concatenate([main[::-1, ::-1], wrist[::-1, ::-1]], axis=1)
                out.write(np.ascontiguousarray(image[:, :, ::-1]))
            out.release()
        summary = {**spec, "teacher_profile": profile, "success": metric["success"], "steps": len(actions),
                   "ever_grasped": oracle.ever_grasped, "ever_lifted_4cm": oracle.ever_lifted,
                   "final_metric": metric, "final_teacher_stage": teacher.stage,
                   "wall_seconds": time.monotonic() - started, "demonstrator": "scripted_privileged_state_teacher",
                   "model_policy_used": False, "teleport_during_trajectory": False,
                   "action_replay_validation": "pending"}
        write_json(folder / "summary.json", summary)
        return summary
    finally:
        wrapper.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--assets", default="sugar,knife,cup,kettle,pan,bread")
    parser.add_argument("--split", default="calibration", choices=["calibration", "train", "validation", "test"])
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=360)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--out", type=Path, default=DATA / "teacher_calibration")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    bootstrap()
    write_protocol()
    profiles = json.loads(args.profile.read_text()) if args.profile else {}
    for asset in args.assets.split(","):
        for index in range(args.start, args.start + args.count):
            spec = make_spec(asset, args.split, index)
            profile = profiles.get(asset, default_profile(asset))
            folder = args.out / spec["episode_id"]
            if (folder / "summary.json").exists():
                continue
            try:
                summary = record_teacher(spec, profile, folder, args.horizon, args.video)
                print(json.dumps({key: summary[key] for key in ["episode_id", "success", "steps", "ever_grasped",
                                                                 "ever_lifted_4cm", "final_teacher_stage"]}), flush=True)
            except Exception:
                folder.mkdir(parents=True, exist_ok=True)
                write_json(folder / "error.json", {"spec": spec, "traceback": traceback.format_exc()})
                print(traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
