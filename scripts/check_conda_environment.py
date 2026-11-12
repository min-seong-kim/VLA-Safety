"""Check imports and, optionally, a fixed policy input or an independent simulator replay."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/semantic_safety"))


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def train_smoke():
    import h5py
    import numpy as np
    import torch
    from motor_lora import load_policy

    data = ROOT / "data/semantic_expansion_v5_lora_v1/combined_train_manifest.jsonl"
    row = json.loads(data.open().readline())
    policy, _ = load_policy()
    with h5py.File(ROOT / row["path"]) as episode:
        value = {"observation/image": np.ascontiguousarray(episode["observations/main_rgb_raw"][0][::-1, ::-1]),
                 "observation/wrist_image": np.ascontiguousarray(episode["observations/wrist_rgb_raw"][0][::-1, ::-1]),
                 "observation/state": episode["observations/state"][0], "prompt": row["instruction"]}
    torch.manual_seed(20261112)
    with torch.inference_mode():
        actions = policy.infer(value)["actions"]
    assert np.isfinite(actions).all() and actions.shape == (10, 7), actions.shape
    return {"item_id": row["item_id"], "action_shape": list(actions.shape),
            "actions": actions.tolist(), "actions_sha256": digest(actions),
            "trained_or_updated_weights": False}


def render_smoke():
    import h5py
    import numpy as np
    from rollout_common import bootstrap

    bootstrap()
    from libero.libero import benchmark
    import robosuite
    from expansion_v5_motor import make_environment
    from evaluate_expansion_v5_lora import native_environment

    assert Path(benchmark.__file__).resolve().is_relative_to(ROOT / "third_party/libero")
    assert Path(robosuite.__file__).resolve().is_relative_to(ROOT / "third_party/robosuite_vanilla")
    data = ROOT / "data/semantic_expansion_v5_lora_v1/combined_train_manifest.jsonl"
    row = json.loads(data.open().readline())
    with h5py.File(ROOT / row["path"]) as episode:
        spec = json.loads(episode.attrs["spec_json"])
        actions = episode["actions"][:]
        expected = episode["observations/sim_state"][:]
    wrapper, observation, oracle = make_environment(spec, 224)
    try:
        main = observation["agentview_image"]
        wrist = observation["robot0_eye_in_hand_image"]
        assert main.shape == wrist.shape == (224, 224, 3) and main.std() > 0 and wrist.std() > 0
        result = {"main_rgb_sha256": digest(main), "wrist_rgb_sha256": digest(wrist)}
        error = float(np.max(np.abs(wrapper.env.sim.get_state().flatten() - expected[0])))
        oracle.measure(wrapper.env)
        for index, action in enumerate(actions):
            wrapper.env.step(action)
            metric = oracle.measure(wrapper.env)
            error = max(error, float(np.max(np.abs(wrapper.env.sim.get_state().flatten() - expected[index + 1]))))
        assert error < 1e-6 and metric["proxy_success"] and metric["stable_steps"] >= 6
        result.update(replay_steps=len(actions), max_sim_state_abs_error=error, replay_success=True)
    finally:
        wrapper.close()
    result["native_initial_states"] = {}
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        wrapper, observation, _ = native_environment({"suite": suite, "task_id": 0, "initial_state_id": 0}, 224)
        try:
            assert observation["agentview_image"].std() > 0
            result["native_initial_states"][suite] = {
                "state_sha256": digest(wrapper.env.sim.get_state().flatten()),
                "main_rgb_sha256": digest(observation["agentview_image"]),
                "wrist_rgb_sha256": digest(observation["robot0_eye_in_hand_image"])}
        finally:
            wrapper.close()
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--role", choices=["train", "render"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    names = (["torch", "torchvision", "transformers", "jax", "numpy", "h5py", "safetensors"]
             if args.role == "train" else ["torch", "mujoco", "numpy", "scipy", "numba", "h5py", "PyOpenGL"])
    result = {"role": args.role, "python": platform.python_version(), "prefix": sys.prefix,
              "versions": {name: importlib.metadata.version(name) for name in names}}
    if args.role == "train":
        import torch
        import torchvision
        import transformers
        import openpi
        from v6_scoped_lora import install, loss_matrix

        assert torch.__version__ == "2.7.1+cu128" and torch.version.cuda == "12.8"
        assert torch.cuda.is_available()
        value = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        assert torch.isfinite(value @ value).all()
        result["cuda"] = {"runtime": torch.version.cuda, "device": torch.cuda.get_device_name(), "matmul_passed": True}
    else:
        from rollout_common import bootstrap

        bootstrap()
        import mujoco
        import robosuite
        from libero.libero import benchmark

        assert mujoco.__version__ == "3.2.2"
        result["simulator_sources"] = {"robosuite": robosuite.__file__, "libero_benchmark": benchmark.__file__}
    if args.smoke:
        result["smoke"] = train_smoke() if args.role == "train" else render_smoke()
    result["passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "role": args.role, "smoke": args.smoke, "report": str(args.out)}))


if __name__ == "__main__":
    main()
