"""Execute local policies in canonical LIBERO and record aligned trajectories/features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
from multiprocessing.connection import Client

import h5py
import numpy as np

from rollout_common import ROOT, bootstrap, native_environment, observation_payload, write_json


def append_array(handle, key, value, index):
    value = np.asarray(value)
    if key not in handle:
        parent, _, leaf = key.rpartition("/")
        group = handle.require_group(parent) if parent else handle
        group.create_dataset(leaf, shape=(0, *value.shape), maxshape=(None, *value.shape),
                             chunks=(1, *value.shape), dtype=value.dtype, compression="lzf", shuffle=True)
    dataset = handle[key]
    dataset.resize(index + 1, axis=0)
    dataset[index] = value


def snapshot(handle, env, obs, index):
    payload = observation_payload(obs)
    values = {"main_rgb_raw": payload["image"], "wrist_rgb_raw": payload["wrist_image"],
              "state": payload["state"], "sim_state": env.sim.get_state().flatten(),
              "qpos": env.sim.data.qpos.copy(), "qvel": env.sim.data.qvel.copy(),
              "body_xpos": env.sim.data.body_xpos.copy(), "body_xquat": env.sim.data.body_xquat.copy(),
              "sim_time": np.asarray(env.sim.data.time)}
    for key, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite simulator observation: {key}, step {index}")
        append_array(handle, "observations/" + key, value, index)


def contact_snapshot(env):
    result = []
    for i in range(env.sim.data.ncon):
        contact = env.sim.data.contact[i]
        if contact.dist > 0.002:
            continue
        result.append({"geom1": int(contact.geom1), "geom2": int(contact.geom2),
                       "distance": float(contact.dist), "position": contact.pos.tolist()})
    return result


class PolicyClient:
    def __init__(self, model, run_dir):
        self.socket = Path(f"/tmp/vla_{os.getpid()}_{model}.sock")
        python = ROOT / ("openpi/.venv/bin/python" if model.startswith("pi") else
                         f"envs/{'openvla-oft-policy' if 'oft' in model else 'openvla-policy'}/bin/python")
        self.log = (run_dir / "policy_worker.log").open("a")
        env = dict(os.environ)
        # TensorFlow performs preprocessing on CPU; the policy itself uses PyTorch CUDA.
        env.update({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
        self.process = subprocess.Popen([str(python), str(ROOT / "scripts/semantic_safety/rollout_policy_worker.py"),
                                         "--model", model, "--socket", str(self.socket)],
                                        stdout=self.log, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + 600
        while not self.socket.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f"Policy worker exited; inspect {run_dir / 'policy_worker.log'}")
            if time.monotonic() > deadline:
                self.process.terminate()
                raise TimeoutError("Policy worker load exceeded 600 seconds")
            time.sleep(0.5)
        self.connection = Client(str(self.socket), family="AF_UNIX", authkey=b"vla-local-rollout-v2")
        self.metadata = self.connection.recv()["metadata"]

    def request(self, packet):
        self.connection.send(packet)
        if not self.connection.poll(300):
            raise TimeoutError("Policy inference response exceeded 300 seconds")
        result = self.connection.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        try:
            self.request({"op": "close"})
            self.connection.close()
            self.process.wait(timeout=20)
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            self.socket.unlink(missing_ok=True)
            self.log.close()


def run_episode(client, model, episode_dir, spec, horizon):
    import cv2

    started = time.time()
    episode_dir.mkdir(parents=True, exist_ok=False)
    wrapper = None
    video = None
    trajectory = None
    try:
        if spec["suite"] == "libero_spatial":
            wrapper, obs, instruction, metadata = native_environment(spec["task_id"], spec["initial_state_id"])
            oracle = None
        else:
            from rollout_safety_env import safety_environment

            wrapper, obs, instruction, metadata, oracle = safety_environment(spec)
        env = wrapper.env
        metadata.update(spec)
        metadata.update({"model": model, "instruction": instruction, "horizon": horizon,
                         "control_hz": 20, "replan_steps": 1, "executed_chunk_index": 0,
                         "observation_index": "observations[t] -> features[t] -> actions[t] -> observations[t+1]",
                         "initial_observation_sha256": hashlib.sha256(obs["agentview_image"].tobytes()).hexdigest(),
                         "initial_state_sha256": hashlib.sha256(env.sim.get_state().flatten().tobytes()).hexdigest(),
                         "body_names": list(env.sim.model.body_names), "geom_names": list(env.sim.model.geom_names),
                         "camera_names": list(env.sim.model.camera_names),
                         "camera_positions": env.sim.model.cam_pos.tolist(),
                         "camera_quaternions": env.sim.model.cam_quat.tolist(),
                         "harm_observed": None, "safety_refusal_inferred_from_failure": False})
        write_json(episode_dir / "episode.json", metadata)
        (episode_dir / "model.xml").write_text(env.sim.model.get_xml())
        trajectory = h5py.File(episode_dir / "trajectory.h5", "w")
        trajectory.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        trajectory.attrs["committed_steps"] = 0
        snapshot(trajectory, env, obs, 0)
        video = cv2.VideoWriter(str(episode_dir / "views.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
        if not video.isOpened():
            raise RuntimeError("Could not open MP4 encoder")

        def frame(observation):
            pair = np.concatenate([observation["agentview_image"][::-1, ::-1],
                                   observation["robot0_eye_in_hand_image"][::-1, ::-1]], axis=1)
            video.write(np.ascontiguousarray(pair[:, :, ::-1]))

        frame(obs)
        cv2.imwrite(str(episode_dir / "initial.png"), np.ascontiguousarray(obs["agentview_image"][::-1, ::-1, ::-1]))
        client.request({"op": "begin", "feature_path": str(episode_dir / "features.h5")})
        ever_success = False
        ever_risk = False
        first_risk = None
        inference_seconds = []
        with (episode_dir / "events.jsonl").open("w") as events:
            for step in range(horizon):
                result = client.request({"op": "infer", "observation": observation_payload(obs),
                                         "instruction": instruction, "seed": spec.get("noise_seed", 73000) + step,
                                         "step": step})
                action = result["action"]
                pre_contacts = contact_snapshot(env)
                obs, reward, done, info = env.step(action)
                measured = oracle.evaluate(env, obs) if oracle else {"task_success": bool(env._check_success()),
                                                                    "risk_proxy": None}
                ever_success |= measured["task_success"] is True
                if measured["risk_proxy"]:
                    ever_risk = True
                    if first_risk is None:
                        first_risk = step
                append_array(trajectory, "actions/submitted", action, step)
                append_array(trajectory, "actions/predicted_chunk", result["action_chunk"], step)
                append_array(trajectory, "actions/environment_clipped", np.clip(action, -1, 1), step)
                append_array(trajectory, "actions/reward", np.asarray(reward), step)
                snapshot(trajectory, env, obs, step + 1)
                event = {"step": step, "post_observation_index": step + 1, "environment_done": bool(done),
                         "pre_contacts": pre_contacts, "post_contacts": contact_snapshot(env), **measured,
                         "inference_seconds": result["inference_seconds"]}
                events.write(json.dumps(event) + "\n")
                events.flush()
                trajectory.attrs["committed_steps"] = step + 1
                trajectory.flush()
                inference_seconds.append(result["inference_seconds"])
                frame(obs)
                if oracle is None and ever_success:
                    break
                if (step + 1) % 25 == 0:
                    print(json.dumps({"episode": episode_dir.name, "step": step + 1,
                                      "risk_proxy": ever_risk, "task_success": ever_success}), flush=True)
        client.request({"op": "end"})
        summary = {"status": "complete", "episode_id": episode_dir.name, **spec,
                   "model": model, "steps": step + 1, "task_success": ever_success if oracle is None else
                   (ever_success if oracle.task_success_supported else None),
                   "risk_proxy_ever": ever_risk if oracle else None, "first_risk_step": first_risk,
                   "harm_observed": None, "stop_reason": "success" if oracle is None and ever_success else "horizon",
                   "wall_seconds": time.time() - started, "mean_inference_seconds": float(np.mean(inference_seconds)),
                   "feature_timestep_alignment": "pending post-run audit", "feature_file": "features.h5",
                   "initial_observation_sha256": metadata["initial_observation_sha256"]}
        write_json(episode_dir / "summary.json", summary)
        print("EPISODE_COMPLETE", json.dumps(summary), flush=True)
        return summary
    except Exception:
        write_json(episode_dir / "error.json", {"traceback": traceback.format_exc(), "spec": spec})
        try:
            client.request({"op": "end"})
        except Exception:
            pass
        raise
    finally:
        if video is not None:
            video.release()
        if trajectory is not None:
            trajectory.close()
        if wrapper is not None:
            wrapper.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=["native", "safety"], default="native")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--spec-file", type=Path)
    args = parser.parse_args()
    bootstrap()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.out_dir / "source_snapshot"
    source_dir.mkdir(exist_ok=True)
    for source in (ROOT / "scripts/semantic_safety").glob("*.py"):
        if not (source_dir / source.name).exists():
            shutil.copy2(source, source_dir / source.name)
    if args.suite == "native":
        specs = [{"episode_id": f"native_task{i % 10:02d}_init{i // 10:02d}", "suite": "libero_spatial",
                  "task_id": i % 10, "initial_state_id": i // 10, "noise_seed": 73000} for i in range(args.limit)]
    else:
        specs = [json.loads(line) for line in args.spec_file.read_text().splitlines()][:args.limit]
    write_json(args.out_dir / "run.json", {"model": args.model, "horizon": args.horizon, "specs": specs,
                                           "python": sys.executable, "pid": os.getpid(), "status": "running"})
    client = PolicyClient(args.model, args.out_dir)
    write_json(args.out_dir / "model.json", client.metadata)
    failures = []
    try:
        for spec in specs:
            episode_dir = args.out_dir / "episodes" / spec["episode_id"]
            if (episode_dir / "summary.json").exists():
                continue
            if episode_dir.exists():
                raise RuntimeError(f"Incomplete prior episode needs explicit recovery: {episode_dir}")
            try:
                summary = run_episode(client, args.model, episode_dir, spec, args.horizon)
                with (args.out_dir / "episodes.jsonl").open("a") as handle:
                    handle.write(json.dumps(summary) + "\n")
            except Exception:
                failure = {"episode_id": spec["episode_id"], "traceback": traceback.format_exc()}
                failures.append(failure)
                print("EPISODE_FAILED", json.dumps(failure), flush=True)
                # Stop on instrumentation/simulation errors, preserving the failed episode for diagnosis.
                break
    finally:
        client.close()
    write_json(args.out_dir / "completion.json", {"status": "complete" if not failures else "has_failures",
                                                  "requested_episodes": len(specs), "failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
