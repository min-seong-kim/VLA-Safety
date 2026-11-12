"""Reuse the actual LIBERO Long kitchen scene and its first official initial state."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rollout_common import ROOT, LIBERO, bootstrap, observation_payload, write_json

OUT = ROOT / "data/semantic_safety_observations_v5"
TASK = "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
BDDL = LIBERO / "libero/libero/bddl_files/libero_10" / (TASK + ".bddl")
INIT = LIBERO / "libero/libero/init_files/libero_10" / (TASK + ".pruned_init")
DEST = OUT / "native_reference"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference():
    DEST.mkdir(parents=True, exist_ok=True)
    path = DEST / "reference.json"
    if path.exists():
        data = json.loads(path.read_text())
        if data["source_bddl_sha256"] == sha(BDDL) and data["source_init_sha256"] == sha(INIT):
            return data
    bootstrap()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    from semantic_pairs_v4 import current_observation

    wrapper = OffScreenRenderEnv(bddl_file_name=str(BDDL), camera_heights=512, camera_widths=512,
                                camera_names=["agentview", "robot0_eye_in_hand"], hard_reset=False,
                                ignore_done=True, initialization_noise={"magnitude": 0., "type": "gaussian"})
    try:
        wrapper.reset()
        states = torch.load(INIT, weights_only=False)
        wrapper.set_init_state(np.asarray(states[0]))
        env = wrapper.env; sim = env.sim
        payload = observation_payload(current_observation(env))
        Image.fromarray(payload["image"][::-1, ::-1]).save(DEST / "agentview.png")
        (DEST / "model.xml").write_text(sim.model.get_xml())
        joints = {}
        for name in sim.model.joint_names:
            joints[name] = np.asarray(sim.data.get_joint_qpos(name)).tolist()
        objects = {}
        for name in ["microwave_1", "white_yellow_mug_1", "porcelain_mug_1"]:
            idx = env.obj_body_id[name]
            objects[name] = dict(world_position=sim.data.body_xpos[idx].tolist(),
                                 world_quat=sim.data.body_xquat[idx].tolist(),
                                 model_position=sim.model.body_pos[idx].tolist(),
                                 model_quat=sim.model.body_quat[idx].tolist())
        result = dict(suite="libero_10 (LIBERO Long)", task=TASK, initial_state_id=0,
                      source_bddl=str(BDDL.relative_to(ROOT)), source_bddl_sha256=sha(BDDL),
                      source_init=str(INIT.relative_to(ROOT)), source_init_sha256=sha(INIT),
                      joints=joints, objects=objects, robot_state=payload["state"].tolist(),
                      source_model_sha256=sha(DEST / "model.xml"), physics_steps=0,
                      table_height=.90)
        write_json(path, result)
        return result
    finally:
        wrapper.close()


def adapted_bddl(scene):
    actor = next(p for p in scene["props"] if p["role"] == "actor")
    source = BDDL.read_text()
    source = source.replace("LIBERO_Kitchen_Tabletop_Manipulation", "SemanticExpansionKitchenV5")
    source = source.replace("white_yellow_mug_1", actor["name"])
    source = source.replace(f"{actor['name']} - white_yellow_mug", f"{actor['name']} - expansion_{actor['asset']}")
    source = source.replace("(:language put the yellow and white mug in the microwave and close it)",
                            "(:language Follow the externally supplied instruction)")
    folder = OUT / "bddl"; folder.mkdir(exist_ok=True)
    path = folder / (scene["scene_id"] + ".bddl")
    path.write_text(source)
    return path


def make_native_environment(scene, resolution=512):
    from libero.libero.envs import OffScreenRenderEnv
    from semantic_pairs_v4 import current_observation

    ref = reference()
    wrapper = OffScreenRenderEnv(bddl_file_name=str(adapted_bddl(scene)), camera_heights=resolution,
                                camera_widths=resolution, camera_names=["agentview", "robot0_eye_in_hand"],
                                hard_reset=False, ignore_done=True,
                                initialization_noise={"magnitude": 0., "type": "gaussian"})
    env = wrapper.env; env.reset(); sim = env.sim
    for name in sim.model.joint_names:
        if name in ref["joints"]:
            sim.data.set_joint_qpos(name, ref["joints"][name])
    for name in ["microwave_1"]:
        idx = env.obj_body_id[name]
        sim.model.body_pos[idx] = ref["objects"][name]["model_position"]
        sim.model.body_quat[idx] = ref["objects"][name]["model_quat"]
    actor = next(p for p in scene["props"] if p["role"] == "actor")
    joint = env.objects_dict[actor["name"]].joints[0]
    sim.data.set_joint_qpos(joint, [*actor["xy"], .901 + actor["z_offset"], 1, 0, 0, 0])
    sim.data.qvel[:] = 0
    sim.forward()
    return wrapper, current_observation(env)
