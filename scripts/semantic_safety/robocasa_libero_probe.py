"""Import three RoboCasa rigid objects into canonical LIBERO and test physical grasping."""

import argparse
from copy import deepcopy
import hashlib
import html
import json
import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from robocasa_asset_import import DATA
from rollout_common import ROOT, bootstrap, write_json

RUN = ROOT / "runs/robocasa_libero_import_v1"
MODELS = {
    "bread": {"source": "bread/bread_0", "scale": [.7, .7, 1.], "kind": "bread"},
    "spray": {"source": "spray/spray_0", "scale": [1.75] * 3, "kind": "bottle"},
    "knife": {"source": "knife/knife_0", "scale": [1.2] * 3, "kind": "mesh:knife"},
}


def browser_video(path):
    encoder = shutil.which("ffmpeg")
    if encoder:
        temporary = path.with_name(path.stem + "_h264.mp4")
        subprocess.run([encoder, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)], check=True)
        temporary.replace(path)


def vector(values):
    return " ".join(f"{float(value):.12g}" for value in values)


def convert(key):
    import mujoco

    config = MODELS[key]
    source = DATA / "original/objaverse" / config["source"] / "model.xml"
    root = ET.parse(source).getroot()
    scale = np.asarray(config["scale"])
    defaults = {d.get("class"): d.find("geom").attrib for d in root.findall("default/default")}
    bbox = root.find(".//geom[@name='reg_bbox']")
    center = np.fromstring(bbox.get("pos"), sep=" ") * scale
    half = np.fromstring(bbox.get("size"), sep=" ") * scale
    for parent in list(root.iter()):
        for geom in list(parent.findall("geom")):
            if geom.get("name", "").startswith("reg_"):
                parent.remove(geom)
                continue
            attributes = {**defaults.get(geom.get("class"), {}), **geom.attrib}
            attributes.pop("class", None)
            geom.attrib.clear()
            geom.attrib.update(attributes)
            collision = geom.get("group", "0") == "0"
            geom.set("density", "100" if collision else "0")
            geom.set("solimp", "0.998 0.998 0.001")
            geom.set("solref", "0.001 2")
            geom.set("friction", "0.95 0.3 0.1")
    root.remove(root.find("default"))
    for element in root.iter():
        if element.get("pos") is not None:
            element.set("pos", vector(np.fromstring(element.get("pos"), sep=" ") * scale))
    for mesh in root.findall("asset/mesh"):
        mesh.set("scale", vector(np.fromstring(mesh.get("scale", "1 1 1"), sep=" ") * scale))
    for element in root.findall("asset/*"):
        if element.get("file"):
            path = (source.parent / element.get("file")).resolve()
            assert path.is_file(), path
            element.set("file", str(path))
    outer = root.find("worldbody/body")
    for name, position in [
        ("bottom_site", [*center[:2], center[2] - half[2]]),
        ("top_site", [*center[:2], center[2] + half[2]]),
        ("horizontal_radius_site", [np.linalg.norm(np.abs(center[:2]) + half[:2]), 0, 0]),
    ]:
        ET.SubElement(outer, "site", name=name, pos=vector(position), size="0.001", rgba="0 0 0 0")
    path = DATA / "converted" / (key + ".xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    # A temporary free joint checks that the imported collision body has valid inertia.
    dynamic = deepcopy(root)
    world = dynamic.find("worldbody")
    container = world.find("body")
    body_element = container.find("body[@name='object']")
    world.remove(container)
    world.append(body_element)
    ET.SubElement(body_element, "freejoint", name="probe_joint")
    compiled = mujoco.MjModel.from_xml_string(ET.tostring(dynamic, encoding="unicode"))
    body = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert compiled.body_mass[body] > 0 and (compiled.body_inertia[body] > 0).all()
    result = {"source_xml": str(source.relative_to(ROOT)), "converted_xml": str(path.relative_to(ROOT)),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "converted_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "official_category_scale": config["scale"], "bbox_extent_m": (2 * half).tolist(),
              "collision_geoms": int(np.sum(compiled.geom_group == 0)),
              "mass_kg": float(compiled.body_mass[body]), "compile_and_inertia_passed": True,
              "changes": ["Absolute mesh/texture paths", "Official category scaling",
                          "Invisible RoboCasa region annotations replaced by LIBERO placement sites",
                          "Default geom attributes inlined for legacy robosuite",
                          "RoboCasa runtime collision density/friction/solref; visual density zero"]}
    return result


def register():
    from libero.libero.envs.base_object import OBJECTS_DICT
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING, register_problem
    from libero.libero.envs.regions import REGION_SAMPLERS
    from libero.libero.envs.problems import libero_tabletop_manipulation  # noqa: F401
    from robosuite.models.objects import MujocoXMLObject

    for key in MODELS:
        def factory(name, joints="default", key=key, **kwargs):
            obj = MujocoXMLObject(str(DATA / "converted" / (key + ".xml")), name=name, joints=joints,
                                  obj_type="all", duplicate_collision_geoms=False)
            obj.category_name, obj.rotation, obj.rotation_axis = "rcimport_" + key, (0, 0), "z"
            obj.object_properties = {"vis_site_names": {}}
            return obj
        OBJECTS_DICT["rcimport_" + key] = factory
    if "robocasaimporttable" not in TASK_MAPPING:
        REGION_SAMPLERS["robocasaimporttable"] = REGION_SAMPLERS["libero_tabletop_manipulation"]

        @register_problem
        class RoboCasaImportTable(TASK_MAPPING["libero_tabletop_manipulation"]):
            def _check_success(self):
                return False


def make_environment(key, resolution=0):
    from libero.libero.envs.env_wrapper import ControlEnv

    register()
    random.seed(20261113)
    np.random.seed(20261113)
    path = DATA / "bddl" / (key + ".bddl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "(define (problem RoboCasaImportTable)", "(:domain robosuite)",
        f"(:language Pick up the {key} and move it across the table)",
        "(:regions (start (:target main_table) (:ranges ((-0.0601 -0.1801 -0.0599 -0.1799)))))",
        "(:fixtures main_table - table)", f"(:objects imported_1 - rcimport_{key})",
        "(:obj_of_interest imported_1)", "(:init (On imported_1 main_table_start))",
        "(:goal (And (On imported_1 main_table_start)))", ")", "",
    ]))
    wrapper = ControlEnv(bddl_file_name=str(path), camera_heights=max(resolution, 16),
                         camera_widths=max(resolution, 16), use_camera_obs=resolution > 0,
                         has_offscreen_renderer=resolution > 0, has_renderer=False, hard_reset=False,
                         ignore_done=True, initialization_noise={"magnitude": 0., "type": "gaussian"})
    env = wrapper.env
    env.reset()
    obj = env.objects_dict["imported_1"]
    # Place above the real table using the adapter's bottom offset, then settle physically.
    z = .90 - obj.bottom_offset[2] + .015
    env.sim.data.set_joint_qpos(obj.joints[0], [-.06, -.18, z, 1, 0, 0, 0])
    env.sim.data.qvel[:] = 0
    env.sim.forward()
    for _ in range(160):
        observation, _, _, _ = env.step([0.] * 6 + [-1.])
    assert np.isfinite(env.sim.get_state().flatten()).all()
    return wrapper, observation


class GraspOracle:
    def __init__(self, env):
        self.actor = env.obj_body_id["imported_1"]
        self.obj = env.objects_dict["imported_1"]
        self.initial = self.position(env).copy()
        self.goal = self.initial + [.06, .30, .17]
        self.working_point = np.zeros(3)
        self.ever_lifted, self.ever_grasped = False, False
        self.stable_steps = 0
        self.max_lift, self.max_held_xy = 0., 0.

    def position(self, env):
        return env.sim.data.body_xpos[self.actor].copy()

    def measure(self, env):
        gripper = env.robots[0].gripper
        grasped = bool(env._check_grasp([gripper.important_geoms["left_finger"],
                                       gripper.important_geoms["right_finger"]], self.obj.contact_geoms))
        displacement = self.position(env) - self.initial
        self.max_lift = max(self.max_lift, float(displacement[2]))
        self.ever_grasped |= grasped
        self.ever_lifted = bool(self.ever_lifted or (grasped and displacement[2] >= .06))
        if grasped:
            self.max_held_xy = max(self.max_held_xy, float(np.linalg.norm(displacement[:2])))
        success = bool(grasped and displacement[2] >= .06 and np.linalg.norm(displacement[:2]) >= .15)
        self.stable_steps = self.stable_steps + 1 if success else 0
        return {"ever_lifted": self.ever_lifted, "ever_grasped": self.ever_grasped,
                "grasped": grasped, "proxy_success": success, "stable_steps": self.stable_steps,
                "max_lift_m": self.max_lift, "max_held_displacement_xy_m": self.max_held_xy}


def profiles(key):
    if key == "knife":
        # The source Handle.006 visual mesh spans local y=-0.117..-0.035 m.
        # Prefer its center over the blade when validating physical manipulation.
        for height in (.3, .5, .7):
            yield {"height_fraction": height, "yaw": -math.pi / 2,
                   "offset": [0., -.33], "tilt": 0., "z_adjust": 0.}
    for height in (.65, .8, .45, .3):
        for yaw in (0., -math.pi / 2):
            yield {"height_fraction": height, "yaw": yaw, "offset": [0., 0.], "tilt": 0., "z_adjust": 0.}
    for offset in ([.3, 0], [-.3, 0], [0, .3], [0, -.3]):
        for yaw in (0., -math.pi / 2):
            yield {"height_fraction": .6, "yaw": yaw, "offset": offset, "tilt": 0., "z_adjust": .002}


def trial(key, profile, render=False):
    from expansion_v5_teacher import Teacher
    from expansion_v5_motor import body_bounds
    from PIL import Image
    import cv2

    wrapper, obs = make_environment(key, 384 if render else 0)
    try:
        env = wrapper.env
        oracle = GraspOracle(env)
        low, high = body_bounds(env, oracle.actor)
        assert low[2] > .88 and np.isfinite(env.sim.get_state().flatten()).all(), (low, high)
        initial_velocity = float(np.max(np.abs(env.sim.data.qvel)))
        initial = env.sim.get_state().flatten().copy()
        frames, actions = [], []
        if render:
            Image.fromarray(obs["agentview_image"][::-1]).save(RUN / (key + "_agentview.png"))
            Image.fromarray(obs["robot0_eye_in_hand_image"][::-1]).save(RUN / (key + "_wrist.png"))
        spec = {"skill": "give_proxy", "actor_kind": MODELS[key]["kind"],
                "props": [{"role": "target", "kind": "empty_space"}]}
        teacher = Teacher(env, oracle, spec, profile)
        metric = oracle.measure(env)
        stable = True
        for step in range(380):
            if render and step % 2 == 0:
                frames.append(obs["agentview_image"][::-1].copy())
            action = teacher.action(env, obs, metric)
            previous = env.sim.data.time
            obs, _, _, _ = env.step(action)
            actions.append(action)
            metric = oracle.measure(env)
            stable &= bool(env.sim.data.time > previous and np.isfinite(env.sim.data.qvel).all()
                           and np.max(np.abs(env.sim.data.qvel)) < 100)
            if not stable or metric["stable_steps"] >= 10 or teacher.failed:
                break
        result = {"passed": bool(stable and metric["stable_steps"] >= 10), "physics_stable": stable,
                  "steps": len(actions), "final_stage": teacher.stage, "metric": metric,
                  "profile": profile, "settled_bbox_m": [low.tolist(), high.tolist()],
                  "initial_max_abs_qvel": initial_velocity,
                  "final_position_m": oracle.position(env).tolist(),
                  "action_sha256": hashlib.sha256(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest(),
                  "physical_env_step_only": True, "teleport_during_manipulation": False}
        if render:
            Image.fromarray(obs["agentview_image"][::-1]).save(RUN / (key + "_final.png"))
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(str(RUN / (key + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"),
                                     10, (width, height))
            assert writer.isOpened(), "Video encoder unavailable"
            try:
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            browser_video(RUN / (key + ".mp4"))
            np.savez_compressed(RUN / (key + "_actions.npz"), initial_state=initial, actions=actions)
            (RUN / (key + "_model.xml")).write_text(env.sim.model.get_xml())
        return result
    finally:
        wrapper.close()


def replay(key, expected):
    recording = np.load(RUN / (key + "_actions.npz"))
    wrapper, _ = make_environment(key)
    try:
        env = wrapper.env
        initial_difference = float(np.max(np.abs(env.sim.get_state().flatten() - recording["initial_state"])))
        assert initial_difference == 0, initial_difference
        oracle = GraspOracle(env)
        stable = True
        for action in recording["actions"]:
            previous = env.sim.data.time
            env.step(action)
            metric = oracle.measure(env)
            stable &= bool(env.sim.data.time > previous and np.isfinite(env.sim.data.qvel).all()
                           and np.max(np.abs(env.sim.data.qvel)) < 100)
        difference = float(np.max(np.abs(oracle.position(env) - expected["final_position_m"])))
        return {"passed": bool(stable and metric["stable_steps"] >= 10 and difference < 1e-6),
                "initial_state_max_abs_difference": initial_difference,
                "final_position_max_abs_difference_m": difference, "metric": metric,
                "independent_environment": True, "steps": len(recording["actions"])}
    finally:
        wrapper.close()


def static_check(key):
    from PIL import Image
    from expansion_v5_motor import body_bounds

    wrapper, obs = make_environment(key, 512)
    try:
        env = wrapper.env
        body = env.obj_body_id["imported_1"]
        initial = env.sim.data.body_xpos[body].copy()
        stable = True
        for _ in range(200):
            previous = env.sim.data.time
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
            stable &= bool(env.sim.data.time > previous and np.isfinite(env.sim.data.qvel).all()
                           and np.max(np.abs(env.sim.data.qvel)) < 100)
        low, high = body_bounds(env, body)
        drift = float(np.linalg.norm(env.sim.data.body_xpos[body] - initial))
        result = {"passed": bool(stable and low[2] > .88 and drift < .005), "steps": 200,
                  "position_drift_m": drift, "bbox_world_m": [low.tolist(), high.tolist()],
                  "final_max_abs_qvel": float(np.max(np.abs(env.sim.data.qvel)))}
        Image.fromarray(obs["agentview_image"][::-1]).save(RUN / (key + "_static.png"))
        Image.fromarray(obs["robot0_eye_in_hand_image"][::-1]).save(RUN / (key + "_static_wrist.png"))
        return result
    finally:
        wrapper.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--objects", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--max-profiles", type=int, default=16)
    args = parser.parse_args()
    egl = ROOT.parent / ".persist/nvidia-egl"
    if (egl / "10_nvidia.json").exists():
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl / "10_nvidia.json")
        if os.environ.get("VLA_SAFETY_ROBOCASA_EGL_READY") != "1":
            # The dynamic loader reads LD_LIBRARY_PATH at process startup.
            os.environ["LD_LIBRARY_PATH"] = str(egl) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["VLA_SAFETY_ROBOCASA_EGL_READY"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], dict(os.environ))
    bootstrap()
    RUN.mkdir(parents=True, exist_ok=True)
    results = {key: json.loads(path.read_text()) for key in MODELS
               if (path := RUN / (key + "_result.json")).exists()}
    for key in args.objects:
        conversion = convert(key)
        static = static_check(key)
        assert static["passed"], static
        attempts, chosen, replay_check = [], None, None
        for profile in list(profiles(key))[:args.max_profiles]:
            result = trial(key, profile)
            attempts.append(result)
            if result["passed"]:
                chosen = trial(key, profile, render=True)
                assert chosen["passed"] and chosen["action_sha256"] == result["action_sha256"], chosen
                replay_check = replay(key, chosen)
                assert replay_check["passed"], replay_check
                break
        results[key] = {"conversion": conversion, "static": static, "grasp": chosen,
                        "replay": replay_check, "grasp_attempts": attempts}
        write_json(RUN / (key + "_result.json"), results[key])
        print(json.dumps({"object": key, "static": static["passed"], "grasp": bool(chosen),
                          "attempts": len(attempts)}), flush=True)
    write_json(RUN / "summary.json", {"complete": True, "objects": results,
               "scope": "Rigid mesh/texture/collision import and scripted physical manipulation; no VLA evaluation",
               "source": "Official RoboCasa object registry and official robocasa/robocasa-assets archive"})
    cards = []
    for key, value in results.items():
        cards.append(f"<section><h2>{html.escape(key)}</h2><img src='{key}_static.png'>"
                     + (f"<video controls src='{key}.mp4'></video>" if value["grasp"] else "<p>Grasp not verified</p>")
                     + "</section>")
    (RUN / "review.html").write_text("<!doctype html><meta charset='utf-8'><title>RoboCasa → LIBERO</title>"
        "<style>body{font-family:system-ui;margin:30px}section{display:inline-block;width:31%;vertical-align:top}"
        "img,video{width:100%}</style><h1>RoboCasa objects in canonical LIBERO</h1>" + "".join(cards))


if __name__ == "__main__":
    main()
