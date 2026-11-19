"""Paired native LIBERO rollouts, with a local render process per frozen initial state."""

import argparse
from contextlib import ExitStack
import hashlib
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import traceback

import numpy as np

from rollout_common import ROOT, observation_payload, write_json


def play(pipe, item):
    from evaluate_expansion_v5_lora import native_environment

    wrapper, obs, instruction = native_environment(item, resolution=224)
    try:
        env = wrapper.env
        assert not env._check_success()
        initial = observation_payload(obs)
        image_hash = hashlib.sha256(initial["image"].tobytes()).hexdigest()
        state_hash = hashlib.sha256(b"".join(initial[k].tobytes() for k in ("image", "wrist_image", "state"))).hexdigest()
        steps = 0
        success = unstable = False
        while steps < item["horizon"] and not (success or unstable):
            pipe.send(dict(observation=observation_payload(obs), instruction=instruction, seed=item["noise_seed"] + steps))
            raw = pipe.recv()
            assert raw.shape == (10, 7) and np.isfinite(raw).all()
            for action in raw:
                command = np.clip(action, -1., 1.)
                command[6] = 1. if action[6] > 0 else -1.
                prev = env.sim.data.time
                obs, _, _, _ = env.step(command)
                steps += 1
                unstable = (env.sim.data.time <= prev or not np.isfinite(env.sim.data.qvel).all()
                            or np.max(np.abs(env.sim.data.qvel)) > 100.)
                success = bool(env._check_success()) and not unstable
                if success or unstable or steps >= item["horizon"]:
                    break
        return dict(**item, instruction=instruction, steps=steps, success=success, physics_stable=not bool(unstable),
                    initial_main_sha256=image_hash, initial_observation_sha256=state_hash)
    finally:
        wrapper.close()


def render_worker(fd):
    import cv2
    cv2.setNumThreads(1)
    with Connection(fd) as pipe:
        try:
            while (item := pipe.recv()) is not None:
                pipe.send(dict(result=play(pipe, item)))
        except EOFError:
            pass
        except Exception:
            pipe.send(dict(error=traceback.format_exc()))


def panel():
    from ae_output_nullspace_pilot import read
    from modality_ablation import RUN

    proto = read(RUN / "protocol.json")["rollout"]
    return [dict(item_id=f"test_{suite}_{task:02d}_{init:02d}", suite=suite, task_id=task, initial_state_id=init,
                 noise_seed=72000000 + si*10000 + task*100 + init,
                 horizon=proto["libero_10_horizon"] if suite == "libero_10" else proto["horizon"])
            for si, suite in enumerate(proto["suites"]) for task in proto["task_ids"] for init in proto["initial_states"]]


def start_worker():
    local, remote = socket.socketpair()
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "NUMBA_NUM_THREADS": "1"}
    egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    if egl.is_file():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl)
    for key in ("PYTHONPATH",):
        env.pop(key, None)
    with remote:
        try:
            proc = subprocess.Popen([str(ROOT / "envs/libero-render/bin/python"), str(Path(__file__).resolve()),
                                     "--worker-fd", str(remote.fileno())], pass_fds=(remote.fileno(),), env=env)
        except BaseException:
            local.close()
            raise
    return Connection(local.detach()), proc


def stop(pair):
    pipe, proc = pair
    try:
        if proc.poll() is None:
            pipe.send(None)
    except (BrokenPipeError, OSError):
        pass
    pipe.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)


def recv(pipe):
    result = pipe.recv()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def simulate(policy, name, condition, workers, items, folder, size, digest):
    import torch
    from modality_ablation import STATE_BASE, forward, physical, state_spans
    from motor_lora import make_input, tensor_batch
    from openpi.models.tokenizer import PaligemmaTokenizer
    from ae_output_nullspace_pilot import read

    processor = PaligemmaTokenizer(200)._tokenizer if name == STATE_BASE else None
    pending = []
    for worker, spec in zip(workers, items, strict=True):
        receipt = folder / (spec["item_id"] + ".json")
        if receipt.exists():
            result = read(receipt)
            assert result["protocol_hash"] == digest and result["model"] == name and result["condition"] == condition
            continue
        worker[0].send(spec)
        pending.append((worker[0], spec))
    while pending:
        active, requests = [], []
        for pipe, spec in pending:
            message = recv(pipe)
            if "result" in message:
                result = message["result"]
                assert result["item_id"] == spec["item_id"] and result["horizon"] == spec["horizon"]
                write_json(folder / (spec["item_id"] + ".json"), dict(**result, model=name, condition=condition,
                                                                      protocol_hash=digest))
            else:
                active.append((pipe, spec))
                requests.append(message)
        pending = active
        if not requests:
            continue
        batch = requests + [requests[-1]] * (size - len(requests))
        values = []
        for entry in batch:
            obs = entry["observation"]
            values.append(make_input(policy, obs["image"], obs["wrist_image"], obs["state"], entry["instruction"]))
        spans = state_spans(values, [b["instruction"] for b in batch], processor) if processor else None
        horizon = policy._model.config.action_horizon
        noise = np.stack([np.random.default_rng(b["seed"]).standard_normal((horizon,32)).astype(np.float32) for b in batch])
        chunks, _ = forward(policy._model, tensor_batch(values)[0], torch.as_tensor(noise, device="cuda"), condition,
                            state_positions=spans)
        actions = np.asarray(physical(policy, values, chunks), dtype=np.float32)
        assert actions.shape == (size, horizon, 7) and np.isfinite(actions).all()
        for (pipe, spec), chunk in zip(active, actions):
            pipe.send(chunk[:10].copy())


def run(smoke=False):
    import gc
    import torch
    from modality_ablation import BATCH, COMMON, RUN, STATE_BASE, conditions, load_policy
    from ae_output_nullspace_pilot import read
    from experiment_notifications import notify_progress

    specs = panel()
    root = RUN / ("smoke/rollout" if smoke else "rollout")
    if smoke:
        # Smoke uses 16 episodes over two small, distinct packs; never read as the primary evaluation.
        specs = [dict(s, horizon=10) for s in specs]
        specs += [dict(s, item_id="repeat_"+s["item_id"]) for s in specs[:4]]
    digest = hashlib.sha256((RUN / "protocol.json").read_bytes()).hexdigest()
    size = BATCH
    with ExitStack() as stack:
        workers = []
        for _ in range(size):
            worker = start_worker()
            workers.append(worker)
            stack.callback(stop, worker)
        for name in (("libero",) if smoke else ("libero", "all", "ae", "vlm", "base", STATE_BASE)):
            policy = load_policy(name)
            for cond in (("intact",) if smoke else conditions(name)):
                folder = root / name / cond
                for start in range(0, len(specs), size):
                    group = specs[start:start+size]
                    simulate(policy, name, cond, workers[:len(group)], group, folder, size, digest)
                episodes = [read(folder / (s["item_id"] + ".json")) for s in specs]
                if not smoke:
                    for ep in episodes:
                        ref = read(root / "libero/intact" / (ep["item_id"] + ".json"))
                        assert ref["initial_observation_sha256"] == ep["initial_observation_sha256"]
                result = dict(complete=True, model=name, condition=cond, episodes=len(episodes),
                              successes=sum(e["success"] for e in episodes), unstable=sum(not e["physics_stable"] for e in episodes))
                write_json(folder / "completion.json", result)
                print(result, flush=True)
            del policy
            gc.collect()
            torch.cuda.empty_cache()
    if not smoke:
        write_json(RUN / "milestones/rollout.json", {"complete": True, "title": "마스킹 조건별 LIBERO 과제 평가 완료",
                                                     "result": {"episodes": len(specs)*28, "paired_tasks": len(specs)}})
        notify_progress(RUN)


def repeat_native_case():
    from ae_output_nullspace_pilot import read
    from modality_ablation import BATCH, RUN, load_policy

    # The AE-only archived run and the new intact run disagreed on LIBERO-10 task 0.
    # Replay it in the fixed-size inference batch used for the current paired comparison.
    spec = next(s for s in panel() if s["suite"]=="libero_10" and s["task_id"]==0)
    digest = hashlib.sha256((RUN/"protocol.json").read_bytes()).hexdigest()
    folder = RUN/"checks/native_ae_repeat"
    with ExitStack() as stack:
        worker = start_worker()
        stack.callback(stop, worker)
        simulate(load_policy("ae"), "ae", "intact", [worker], [spec], folder, BATCH, digest)
    rerun = read(folder/(spec["item_id"]+".json"))
    original = read(RUN/"rollout/ae/intact"/(spec["item_id"]+".json"))
    same = {key: original[key]==rerun[key] for key in ("initial_observation_sha256","success","steps","physics_stable")}
    write_json(folder/"audit.json", {"comparison":same, "original_steps":original["steps"], "repeat_steps":rerun["steps"],
                                     "original_success":original["success"], "repeat_success":rerun["success"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worker-fd", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeat-native-case", action="store_true")
    args = parser.parse_args()
    if args.worker_fd is not None:
        render_worker(args.worker_fd)
    elif args.repeat_native_case:
        repeat_native_case()
    else:
        run(args.smoke)
