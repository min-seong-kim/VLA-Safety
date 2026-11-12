"""Render explicit safe/unsafe instruction pairs from LIBERO-Safety scenes.

Run with envs/libero-render/bin/python. No policy, actions, or training is run.
The two labels always share the exact same pre-action observation and state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "data/safety_sources/sources/libero_safety_code"
CONFIG = Path(__file__).with_name("libero_explicit_scenarios.json")
VIEWS = {
    "agentview": {"original": True},
    "left_15": {"yaw": -15, "lift": 0.0, "distance": 1.0},
    "right_15": {"yaw": 15, "lift": 0.0, "distance": 1.0},
    "elevated": {"yaw": 0, "lift": 0.15, "distance": 1.0},
    "closer": {"yaw": 0, "lift": 0.0, "distance": 0.85},
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba-libero"))
    config_dir = ROOT / "envs/libero-render/config"
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path[:0] = [str(CODE), str(CODE / "third_party/robosuite-1.4")]
    import yaml

    base = CODE / "libero/libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
        "assets": str(base / "assets"), "init_states": str(base / "init_files"),
        "datasets": str(ROOT / "data/safety_sources/raw/libero_safety"),
    }))


def camera_pose(sim, name):
    import numpy as np

    camera_id = sim.model.camera_name2id(name)
    return {"world_position": sim.data.cam_xpos[camera_id].copy(),
            "world_rotation": np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3).copy(),
            "fovy_degrees": float(sim.model.cam_fovy[camera_id])}


def set_camera_pose(sim, name, position, rotation):
    import numpy as np
    from scipy.spatial.transform import Rotation

    camera_id = sim.model.camera_name2id(name)
    body_id = sim.model.cam_bodyid[camera_id]
    body_rotation = np.asarray(sim.data.body_xmat[body_id]).reshape(3, 3)
    sim.model.cam_pos[camera_id] = body_rotation.T @ (position - sim.data.body_xpos[body_id])
    quat = Rotation.from_matrix(body_rotation.T @ rotation).as_quat()
    sim.model.cam_quat[camera_id] = quat[[3, 0, 1, 2]]
    sim.forward()


def view_pose(base, target, setting):
    import numpy as np
    from scipy.spatial.transform import Rotation

    if setting.get("original"):
        return base["world_position"], base["world_rotation"]
    offset = Rotation.from_euler("z", setting["yaw"], degrees=True).apply(base["world_position"] - target)
    position = target + offset * setting["distance"]
    position[2] += setting["lift"]
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0.0, 0.0, 1.0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return position, np.column_stack([right, up, backward])


def visibility(sim, segmentation, objects):
    import numpy as np
    import mujoco

    result = {}
    for name in objects:
        ids = [i for i, geom in enumerate(sim.model.geom_names) if geom and geom.startswith(name + "_")]
        if not ids:
            raise ValueError(f"No geometries for critical object: {name}")
        mask = (segmentation[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(segmentation[:, :, 1], ids)
        ys, xs = np.nonzero(mask)
        result[name] = {"pixels": int(mask.sum()),
                        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None}
    return result


def robot_state(observation):
    import numpy as np
    from scipy.spatial.transform import Rotation

    position = np.asarray(observation["robot0_eef_pos"])
    quaternion = np.asarray(observation["robot0_eef_quat"])
    gripper = np.asarray(observation["robot0_gripper_qpos"])
    state = np.concatenate([position, Rotation.from_quat(quaternion).as_rotvec(), gripper])
    return {"state": state.tolist(), "eef_position_world": position.tolist(),
            "eef_quaternion_xyzw": quaternion.tolist(), "gripper_qpos": gripper.tolist(),
            "joint_qpos": np.asarray(observation["robot0_joint_pos"]).tolist(),
            "state_layout": ["eef_position_world_xyz:3", "eef_rotation_vector_xyz:3", "gripper_qpos:2"]}


def generate_scene(scenario, args, out):
    import numpy as np
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.errors import RandomizationError

    scene_id = scenario["id"]
    source = CODE / "libero/libero/bddl_files/reasoning_safety" / scenario["source"]
    source_text = source.read_text()
    task_dir = out / "scenes" / scene_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "source.bddl").write_text(source_text)
    write_json(task_dir / "instructions.json", scenario)
    source_hash = sha256(source)
    constructor_seed = args.seed_base
    np.random.seed(constructor_seed)
    random.seed(constructor_seed)
    wrapper = OffScreenRenderEnv(
        bddl_file_name=str(source), camera_heights=args.resolution,
        camera_widths=args.resolution, camera_names=["agentview", "robot0_eye_in_hand"],
        hard_reset=False, initialization_noise=None, ignore_done=True,
    )
    env = wrapper.env
    rows, rejected, layout_records = [], [], []
    try:
        for layout_index in range(args.layouts):
            seed = args.seed_base + layout_index
            np.random.seed(seed)
            random.seed(seed)
            env.seed(seed)
            for attempt in range(20):
                try:
                    # The upstream wrapper's reset suppresses all exceptions; use the environment directly.
                    observation = env.reset()
                    break
                except RandomizationError:
                    if attempt == 19:
                        raise
            # Let the sampled objects settle while holding the arm pose and open gripper.
            hold_action = np.zeros(7)
            hold_action[-1] = -1.0
            settle_steps = scenario.get("settle_steps", args.settle_steps)
            for _ in range(settle_steps):
                env.step(hold_action)
            env.sim.forward()
            observation = env._get_observations(force_update=True)
            sim = env.sim
            physical_id = f"{scene_id}__s{seed}"
            folder = out / "observations" / physical_id
            folder.mkdir(parents=True, exist_ok=True)
            state = robot_state(observation)
            flat = sim.get_state().flatten()
            fingerprint = hashlib.sha256(
                np.round(np.concatenate([sim.data.qpos, sim.model.body_pos.ravel(), sim.model.body_quat.ravel()]), 8).tobytes()
            ).hexdigest()
            np.savez_compressed(folder / "sim_state.npz", state=flat, qpos=sim.data.qpos.copy(),
                                qvel=sim.data.qvel.copy(), body_pos=sim.model.body_pos.copy(),
                                body_quat=sim.model.body_quat.copy(), cam_pos=sim.model.cam_pos.copy(),
                                cam_quat=sim.model.cam_quat.copy())
            split = "train" if layout_index < int(args.layouts * .7) else (
                "validation" if layout_index < int(args.layouts * .85) else "test")
            merged_objects = {**env.objects_dict, **env.fixtures_dict}
            poses = {name: sim.data.body_xpos[sim.model.body_name2id(merged_objects[name].root_body)].copy()
                     for name in scenario["critical_objects"]}
            target = np.mean(list(poses.values()), axis=0)
            base = camera_pose(sim, "agentview")
            wrist = sim.render(width=args.resolution, height=args.resolution,
                               camera_name="robot0_eye_in_hand")[::-1].copy()
            Image.fromarray(wrist).save(folder / "wrist.png")
            layout_metadata = {"group_id": physical_id, "family_id": scene_id, "seed": seed,
                               "split": split, "state_fingerprint": fingerprint, "robot_state": state,
                               "critical_object_positions": {k: v.tolist() for k, v in poses.items()},
                               "source_bddl_sha256": source_hash,
                               "stage": "settled_before_any_policy_action", "policy_actions_executed": 0,
                               "initialization_hold_steps": settle_steps}
            write_json(folder / "layout.json", layout_metadata)
            layout_records.append(layout_metadata)
            for view in args.views.split(","):
                position, rotation = view_pose(base, target, VIEWS[view])
                set_camera_pose(sim, "agentview", position, rotation)
                rgb = sim.render(width=args.resolution, height=args.resolution, camera_name="agentview")[::-1].copy()
                segmentation = sim.render(width=args.resolution, height=args.resolution,
                                          camera_name="agentview", segmentation=True)[::-1].copy()
                visible = visibility(sim, segmentation, scenario["critical_objects"])
                camera = camera_pose(sim, "agentview")
                focal = args.resolution / (2 * math.tan(math.radians(camera["fovy_degrees"]) / 2))
                camera_info = {"view_id": view, "world_position": camera["world_position"].tolist(),
                               "world_rotation": camera["world_rotation"].tolist(),
                               "fovy_degrees": camera["fovy_degrees"],
                               "intrinsic_matrix": [[focal, 0, args.resolution / 2], [0, focal, args.resolution / 2], [0, 0, 1]],
                               "image_orientation": "top-left origin; raw OpenGL rows flipped vertically",
                               "resolution": [args.resolution, args.resolution], "visibility": visible}
                write_json(folder / f"{view}.camera.json", camera_info)
                image_path = folder / f"{view}.png"
                Image.fromarray(rgb).save(image_path)
                if any(v["pixels"] < args.min_visible_pixels for v in visible.values()):
                    rejected.append({"group_id": physical_id, "view_id": view, "reason": "critical_object_visibility",
                                     "visibility": visible, "image": str(image_path.relative_to(out))})
                    continue
                observation_id = physical_id + "__" + view
                for label in ["safe", "unsafe"]:
                    for wording, instruction in enumerate(scenario[label]):
                        rows.append({"id": f"{observation_id}__{label}_{wording}", "family_id": scene_id,
                                     "group_id": physical_id, "observation_id": observation_id, "seed": seed,
                                     "split": split, "view_id": view, "safety_label": label,
                                     "task_is_unsafe": label == "unsafe", "reference_type": "explicit",
                                     "instruction": instruction, "wording_id": wording,
                                     "image": str(image_path.relative_to(out)),
                                     "wrist_image": str((folder / "wrist.png").relative_to(out)),
                                     "state": state["state"], "state_layout": state["state_layout"],
                                     "sim_state_file": str((folder / "sim_state.npz").relative_to(out)),
                                     "camera_file": str((folder / f"{view}.camera.json").relative_to(out)),
                                     "image_sha256": sha256(image_path), "state_fingerprint": fingerprint,
                                     "annotation_only": {"hazard": scenario["hazard"], "rationale": scenario["rationale"],
                                                         "label_source": "authored_instruction_pair_on_published_scene",
                                                         "label_status": "research_v1_not_independently_human_annotated",
                                                         "executed_action_safety": None,
                                                         "source_bddl": str(source.relative_to(ROOT))}})
            set_camera_pose(sim, "agentview", base["world_position"], base["world_rotation"])
            print(f"RENDER {scene_id} layout={layout_index + 1}/{args.layouts} rows={len(rows)} rejected_views={len(rejected)}", flush=True)
        write_json(task_dir / "layouts.json", layout_records)
        write_json(task_dir / "rejected_views.json", rejected)
        (task_dir / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        return rows, rejected
    finally:
        wrapper.close()


def summarize(out, failures):
    rows, rejected = [], []
    for path in sorted((out / "scenes").glob("*/items.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    for path in sorted((out / "scenes").glob("*/rejected_views.json")):
        rejected.extend(json.loads(path.read_text()))
    (out / "items.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for split in ["train", "validation", "test"]:
        selected = [r for r in rows if r["split"] == split]
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    summary = {"records": len(rows), "families": dict(Counter(r["family_id"] for r in rows)),
               "physical_layouts": len({r["group_id"] for r in rows}),
               "primary_observations": len({r["observation_id"] for r in rows}),
               "labels": dict(Counter(r["safety_label"] for r in rows)),
               "splits": dict(Counter(r["split"] for r in rows)),
               "views": dict(Counter(r["view_id"] for r in rows)),
               "rejected_views": len(rejected), "generation_failures": failures,
               "scope": "Explicit pre-action potential-harm labels; no policy rollout or action ground truth.",
               "split_policy": "All cameras, labels and wordings from one physical layout stay together. This is within-family held-out-layout evaluation. Use family_id to hold out whole situations for stronger generalization tests."}
    write_json(out / "summary.json", summary)
    write_json(out / "rejected_views.json", rejected)
    display = {**summary, "generation_failures": [{k: v for k, v in f.items() if k != "traceback"} for f in failures]}
    print(json.dumps(display, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/libero_semantic_explicit_v1")
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--layouts", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=73000)
    parser.add_argument("--views", default=",".join(VIEWS))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--min-visible-pixels", type=int, default=40)
    parser.add_argument("--settle-steps", type=int, default=0,
                        help="Default pose-hold initialization steps; per-scene values override this.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.layouts < 1 or args.resolution < 64 or any(v not in VIEWS for v in args.views.split(",")):
        parser.error("Invalid layouts, resolution, or views")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    bootstrap()
    config = json.loads(CONFIG.read_text())
    selected = [s for s in config["scenarios"] if (
        args.scenes == "all" and s.get("enabled", True)) or s["id"] in args.scenes.split(",")]
    if not selected:
        parser.error("No scenarios selected")
    provenance = {"configuration": config, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "source_commit": json.loads((ROOT / "data/safety_sources/catalog/libero_safety_code.json").read_text())["commit"],
                  "asset_revision": json.loads((ROOT / "data/safety_sources/catalog/libero_safety_assets.json").read_text())["sha"],
                  "generator_sha256": sha256(Path(__file__)), "configuration_sha256": sha256(CONFIG)}
    write_json(args.out / "generation.json", provenance)
    failures = []
    for scenario in selected:
        target = args.out / "scenes" / scenario["id"] / "items.jsonl"
        if args.resume and target.exists():
            print("SKIP", scenario["id"], flush=True)
            continue
        if target.exists():
            raise FileExistsError(f"Use a fresh --out or --resume: {target}")
        try:
            generate_scene(scenario, args, args.out)
        except Exception as error:
            failures.append({"scene": scenario["id"], "error": str(error), "traceback": traceback.format_exc()})
            print(f"ERROR {scenario['id']}: {type(error).__name__}: {error}", flush=True)
    summarize(args.out, failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
