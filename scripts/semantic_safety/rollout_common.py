"""Canonical LIBERO bootstrap and trajectory utilities, isolated from the safety fork."""

from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
LIBERO = ROOT / "third_party/libero"
ROBOSUITE = ROOT / "third_party/robosuite_vanilla"


def bootstrap():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache/numba-rollout"))
    config = ROOT / "envs/libero-rollout-config"
    config.mkdir(exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    base = LIBERO / "libero/libero"
    import yaml

    content = yaml.safe_dump({
        "benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"), "assets": str(base / "assets"),
        "datasets": str(ROOT / "data"),
    })
    path = config / "config.yaml"
    if not path.exists() or path.read_text() != content:
        temporary = config / f"config.{os.getpid()}.tmp"
        temporary.write_text(content)
        temporary.replace(path)
    sys.path[:0] = [str(LIBERO), str(ROBOSUITE)]


def state_vector(obs):
    import numpy as np
    import math

    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3] ** 2)
    rot = np.zeros(3) if math.isclose(den, 0) else quat[:3] * 2 * math.acos(quat[3]) / den
    return np.concatenate((obs["robot0_eef_pos"], rot, obs["robot0_gripper_qpos"])).astype(np.float32)


def observation_payload(obs):
    import numpy as np

    return {"image": np.ascontiguousarray(obs["agentview_image"]),
            "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
            "state": state_vector(obs)}


def native_environment(task_id=0, initial_state_id=0, resolution=256):
    import numpy as np
    import random
    import torch
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    base = LIBERO / "libero/libero"
    source = base / "bddl_files" / task.problem_folder / task.bddl_file
    np.random.seed(0)
    random.seed(0)
    wrapper = OffScreenRenderEnv(bddl_file_name=str(source), camera_heights=resolution,
                                 camera_widths=resolution, hard_reset=False, ignore_done=True)
    wrapper.seed(0)
    wrapper.env.reset()
    init_file = base / "init_files" / task.problem_folder / task.init_states_file
    states = torch.load(init_file, weights_only=False)
    obs = wrapper.set_init_state(states[initial_state_id])
    for _ in range(10):
        obs, _, _, _ = wrapper.env.step([0.0] * 6 + [-1.0])
    return wrapper, obs, task.language, {
        "suite": "libero_spatial", "task_id": task_id, "initial_state_id": initial_state_id,
        "source_bddl": str(source.relative_to(ROOT)), "initialization_hold_steps": 10,
    }


def array_digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    bootstrap()
    import numpy as np
    from PIL import Image

    env, obs, instruction, metadata = native_environment()
    out = ROOT / "runs/semantic_rollout_preflight"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "native_observation.npz", **observation_payload(obs))
    Image.fromarray(obs["agentview_image"][::-1, ::-1]).save(out / "native_initial.png")
    controller = json.loads((ROBOSUITE / "robosuite/controllers/config/osc_pose.json").read_text())
    write_json(out / "native_initial.json", {"instruction": instruction, **metadata, "controller": controller})
    print(instruction, flush=True)
    env.close()
