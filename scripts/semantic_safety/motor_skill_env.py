"""Neutral object transport environments for the frozen-VLM adaptation experiment.

Object poses are available to the scripted teacher and evaluator only. Policies
receive the same RGB/state/instruction interface as the original LIBERO runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rollout_common import ROOT, bootstrap, write_json


DATA = ROOT / "data/motor_skill_b_v1"
ASSETS = ROOT / "data/semantic_rollout_v3/assets"
DISPLAY_NAMES = {
    "alcohol_cleaner": "cleaner bottle", "basin": "ceramic bowl", "battery": "button battery",
    "bread": "bread", "cord": "cord", "cup": "mug", "foil": "foil ball",
    "frayed_cord": "cord with exposed wires", "kibble": "dog food", "moldy_bread": "green-spotted bread",
    "plastic_bowl": "plastic bowl", "power_bank": "power bank", "rice_cake": "rice cake",
    "sugar": "sugar cube", "toy": "toy turtle", "wood_bowl": "wooden bowl", "wrapper": "wrapper",
    "aerosol": "spray can",
}


def actor_assets() -> list[str]:
    scenes = [json.loads(line) for line in (ROOT / "data/semantic_rollout_v3/scenes.jsonl").read_text().splitlines()]
    return sorted({next(p["asset"] for p in s["props"] if p["name"] == s["roles"][r])
                   for s in scenes for r in ("safe_actor", "unsafe_actor")})


def make_spec(asset: str, split: str, index: int) -> dict:
    split_seed = {"calibration": 180000, "train": 280000, "validation": 380000, "test": 480000}[split]
    seed = split_seed + actor_assets().index(asset) * 1000 + index
    rng = np.random.default_rng(seed)
    side = 1 if index % 2 == 0 else -1
    actor_xy = [float(rng.uniform(-.09, .025)), float(side * rng.uniform(.075, .14))]
    target_xy = [float(rng.uniform(-.19, -.09)), float(-side * rng.uniform(.15, .24))]
    return {"episode_id": f"{asset}__{split}__{index:04d}", "asset": asset, "split": split,
            "seed": seed, "index": index, "actor_xy": actor_xy, "target_xy": target_xy,
            "instruction": f"Pick up the {DISPLAY_NAMES.get(asset, asset.replace('_', ' '))} and place it on the tray.",
            "task": "neutral_pick_and_place", "semantic_safety_label": None}


def prepare_bddl(spec: dict) -> Path:
    folder = DATA / "bddl"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{spec['episode_id']}.bddl"
    regions = []
    for name, point in [("actor", spec["actor_xy"]), ("tray", spec["target_xy"])]:
        x, y = point
        regions.append(f"({name}_start (:target main_table) (:ranges (({x-.00001} {y-.00001} {x+.00001} {y+.00001}))))")
    content = "\n".join([
        "(define (problem SemanticSafetyTabletop)", "(:domain robosuite)",
        "(:language Follow the supplied instruction)", "(:regions", *regions, ")",
        "(:fixtures main_table - table tray_1 - semantic_tray)",
        f"(:objects actor_1 - semantic_{spec['asset']})", "(:obj_of_interest actor_1 tray_1)",
        "(:init (On actor_1 main_table_actor_start) (On tray_1 main_table_tray_start))",
        "(:goal (And (On actor_1 main_table_tray_start)))", ")",
    ])
    if not path.exists() or path.read_text() != content:
        path.write_text(content)
    return path


class MotorOracle:
    """Require a physical grasp/lift followed by stable release on the target tray."""

    def __init__(self, env, spec, initial_obs):
        self.spec = spec
        self.actor = env.obj_body_id["actor_1"]
        self.target = env.obj_body_id["tray_1"]
        self.initial = env.sim.data.body_xpos[self.actor].copy()
        self.previous = self.initial.copy()
        self.ever_grasped = False
        self.ever_lifted = False
        self.stable_steps = 0
        self.actor_geoms = self._geoms(env, self.actor)
        self.target_geoms = self._geoms(env, self.target)
        manifest = json.loads((ASSETS / "manifest.json").read_text())
        self.target_extent = np.asarray(manifest["tray"]["extent"])

    @staticmethod
    def _geoms(env, root):
        result = set()
        for geom, body in enumerate(env.sim.model.geom_bodyid):
            cursor = int(body)
            while cursor and cursor != root:
                cursor = int(env.sim.model.body_parentid[cursor])
            if cursor == root and env.sim.model.geom_group[geom] == 0:
                result.add(geom)
        return result

    def measure(self, env):
        position = env.sim.data.body_xpos[self.actor].copy()
        target = env.sim.data.body_xpos[self.target].copy()
        rotation = env.sim.data.body_xmat[self.target].reshape(3, 3)
        local = rotation.T @ (position - target)
        gripper = env.robots[0].gripper
        object_geoms = env.get_object("actor_1").contact_geoms
        fingerpad_grasped = bool(env._check_grasp(gripper, object_geoms))
        grasped = bool(env._check_grasp([gripper.important_geoms["left_finger"],
                                       gripper.important_geoms["right_finger"]], object_geoms))
        lifted = bool(grasped and position[2] - self.initial[2] > .04)
        contact = any((c.geom1 in self.actor_geoms and c.geom2 in self.target_geoms) or
                      (c.geom2 in self.actor_geoms and c.geom1 in self.target_geoms)
                      for c in (env.sim.data.contact[i] for i in range(env.sim.data.ncon)) if c.dist <= .002)
        inside = bool(np.all(np.abs(local[:2]) < self.target_extent[:2] / 2 - .005))
        speed = float(np.linalg.norm(position - self.previous) * 20)
        self.ever_grasped |= grasped
        self.ever_lifted |= lifted
        placed = (self.ever_lifted and inside and contact and not grasped and speed < .03)
        self.stable_steps = self.stable_steps + 1 if placed else 0
        self.previous = position.copy()
        return {"grasped": grasped, "fingerpad_grasped": fingerpad_grasped,
                "lifted_4cm": bool(lifted), "ever_grasped": self.ever_grasped,
                "ever_lifted_4cm": self.ever_lifted, "actor_target_contact": bool(contact),
                "inside_target_xy": inside, "speed_m_s": speed, "stable_placement_steps": self.stable_steps,
                "success": self.stable_steps >= 10, "actor_position": position.tolist(),
                "target_position": target.tolist()}


def make_environment(spec, resolution=256):
    import random
    from libero.libero.envs import OffScreenRenderEnv
    from rollout_safety_env import register_assets

    register_assets()
    np.random.seed(spec["seed"])
    random.seed(spec["seed"])
    wrapper = OffScreenRenderEnv(bddl_file_name=str(prepare_bddl(spec)), camera_heights=resolution,
                                 camera_widths=resolution, hard_reset=False, ignore_done=True)
    wrapper.seed(spec["seed"])
    obs = wrapper.env.reset()
    for _ in range(10):
        obs, _, _, _ = wrapper.env.step([0.] * 6 + [-1.])
    oracle = MotorOracle(wrapper.env, spec, obs)
    return wrapper, obs, oracle


def write_protocol():
    write_json(DATA / "protocol.json", {
        "experiment": "B: frozen VLM, action-expert-only LoRA", "model": "pi05_libero",
        "teacher": "project-authored scripted Cartesian controller using simulator poses",
        "not_an_official_libero_oracle_policy": True,
        "privileged_inputs_used_by_teacher_only": ["actor/target poses", "collision geometry", "grasp/contact checks"],
        "student_inputs": ["agentview RGB", "wrist RGB", "neutral instruction"],
        "semantic_labels_or_probe_training": False,
        "split_unit": "complete episode / randomized initial layout",
        "splits": {"calibration": "teacher development only", "train": "successful expert trajectories",
                   "validation": "checkpoint selection", "test": "held-out final evaluation only"},
        "success": "bilateral full-finger contact, >4cm lift, released actor in tray with contact and speed<3cm/s for 10 consecutive steps",
        "grasp_metric": "both complete finger collision groups; also record the historical pad-only grasp",
        "historical_v3_results": "unchanged; new A and B runs share the new motor oracle",
        "demo_integrity": "actual env.step actions only; save reset state and validate action replay",
        "control": {"robot": "Panda", "controller": "OSC_POSE", "hz": 20},
        "source_assets": str(ASSETS.relative_to(ROOT)), "candidate_assets": actor_assets(),
        "training_scope": "neutral pick-and-place, not cutting/pouring/semantic refusal",
        "test_semantics": "original v3 safe tasks remain separate transfer evaluation",
    })


if __name__ == "__main__":
    write_protocol()
    print(DATA / "protocol.json")
