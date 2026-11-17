"""Collect replay-verified teacher inputs on the unchanged 676 evaluation layouts."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np

from expansion_v5_motor import make_environment
from expansion_v5_teacher import Teacher, replay, rollout
from representation_common import BOTH, DATA, ROOT, RUN, TEACHER, milestone, read_rows
from rollout_common import observation_payload, write_json
from search_expansion_v5_recovery import proposals, refined_bread_profiles


def refined_profiles(spec, profile):
    rows = []
    if "bread" in spec["actor_kind"]:
        rows.extend(refined_bread_profiles(profile))
        for y in (-.12, .12, -.06, .06, -.20, .20):
            for yaw_delta in (0., .15, -.15):
                for height in (.6, .8, .4):
                    for x in (.38, .30, .46):
                        for dz in (.002, .006, -.002):
                            rows.append({**profile, "offset": [x, y], "height_fraction": height,
                                         "yaw": profile["yaw"] + yaw_delta, "z_adjust": dz})
    if spec["actor_kind"] in {"cloth", "paper"}:
        bases = [profile, {**profile, "offset": [-.45, 0], "yaw": np.pi / 2, "tilt": -1.5}]
        for base in bases:
            axis = int(np.argmax(np.abs(base["offset"])))
            sign = np.sign(base["offset"][axis])
            for height in (.5, .8, .2):
                for edge in (.40, .48, .35, .52):
                    for dz in (.003, .006, -.003, .010):
                        for tilt in (base["tilt"], np.sign(base["tilt"]) * 1.57):
                            offset = list(base["offset"])
                            offset[axis] = float(sign * edge)
                            rows.append({**base, "offset": offset, "height_fraction": height,
                                         "z_adjust": dz, "tilt": float(tilt)})
    if spec["actor_kind"] == "sealed_jar":
        for height in (.75, .9, .6, .4):
            for edge in (.30, .42, .48):
                for cross in (0., -.12, .12):
                    for tilt in (.4, .6, .8, 0.):
                        for dz in (.002, .006):
                            rows.append({**profile, "offset": [cross, -edge], "height_fraction": height,
                                         "tilt": tilt, "z_adjust": dz})
    return rows


def record(spec, profile, folder):
    wrapper, obs, oracle = make_environment(spec, 224)
    try:
        env = wrapper.env
        teacher = Teacher(env, oracle, spec, profile)
        metric = oracle.measure(env)
        assert not metric["proxy_success"]
        states = [env.sim.get_state().flatten().copy()]
        actions, starts, stages, images, wrists, robot = [], [], [], [], [], []
        unstable = False
        for step in range(460):
            if step % 10 == 0:
                payload = observation_payload(obs)
                starts.append(step)
                images.append(payload["image"].copy())
                wrists.append(payload["wrist_image"].copy())
                robot.append(payload["state"].copy())
            action = teacher.action(env, obs, metric)
            actions.append(action)
            stages.append(teacher.stage)
            previous = env.sim.data.time
            obs, _, _, _ = env.step(action)
            unstable = (env.sim.data.time <= previous or not np.isfinite(env.sim.data.qvel).all()
                        or float(np.max(np.abs(env.sim.data.qvel))) > 100.)
            metric = oracle.measure(env)
            states.append(env.sim.get_state().flatten().copy())
            if unstable or metric["stable_steps"] >= 10 or teacher.failed:
                break
        passed = bool(not unstable and metric["proxy_success"] and metric["stable_steps"] >= 6)
        assert passed, {"item": spec["item_id"], "metric": metric, "stage": teacher.stage}
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / "trajectory.partial.h5"
        with h5py.File(temporary, "w") as out:
            out.attrs.update(item_json=json.dumps(spec), spec_json=json.dumps(spec),
                profile_json=json.dumps(profile), instruction=spec["instruction"], teacher_success=passed,
                alignment="policy_inputs at control steps 0,10,... before teacher actions; all actions and sim states retained")
            arrays = {"policy_inputs/main_rgb_raw": images, "policy_inputs/wrist_rgb_raw": wrists,
                      "policy_inputs/state": robot, "observations/sim_state": states,
                      "actions": np.asarray(actions, dtype=np.float32)}
            for name, values in arrays.items():
                value = np.asarray(values)
                out.create_dataset(name, data=value, compression="lzf", shuffle=True,
                                   chunks=(1, *value.shape[1:]))
            out.create_dataset("chunk_start_steps", data=np.asarray(starts, dtype=np.int32))
            out.create_dataset("teacher_stage", data=np.asarray(stages, dtype="S16"))
        temporary.replace(folder / "trajectory.h5")
        (folder / "model.xml").write_text(env.sim.model.get_xml())
        result = {"passed": passed, "steps": len(actions), "decision_inputs": len(starts),
            "action_sha256": hashlib.sha256(np.asarray(actions, dtype=np.float32).tobytes()).hexdigest(),
            "final_metric": metric, "physical_actions_only": True, "teleport_during_recording": False}
        write_json(folder / "summary.json", result)
        return result
    finally:
        wrapper.close()


def collect(request):
    spec, limit = request
    folder = TEACHER / "episodes" / spec["item_id"]
    receipt_path = folder / "receipt.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old.get("accepted"):
            return old
    calibrated = json.loads((DATA / "collection_r10/profiles" / f"c{spec['condition_index']:03d}.json").read_text())
    assert calibrated["success"]
    profiles = [calibrated["profile"]]
    # Reuse grasp settings, never training trajectories or altered test layouts.
    for row in read_rows(DATA / "train_manifest.jsonl"):
        if row["condition_index"] == spec["condition_index"] and row.get("recovery_profile"):
            if row["recovery_profile"] not in profiles:
                profiles.append(row["recovery_profile"])
    profiles.extend(p for p in proposals(spec, calibrated["profile"]) if p not in profiles)
    profiles.extend(p for p in refined_profiles(spec, calibrated["profile"]) if p not in profiles)
    trials_path = folder / "trials.json"
    trials = json.loads(trials_path.read_text()) if trials_path.exists() else []
    chosen = None
    for profile in profiles[:limit]:
        prior = next((row for row in trials if row["profile"] == profile), None)
        if prior is None:
            result = rollout(spec, profile)
            prior = {"profile": profile, "result": result}
            trials.append(prior)
            write_json(trials_path, trials)
        if prior["result"]["passed"]:
            chosen = prior
            break
    result = {"episode_id": spec["item_id"], "accepted": False, "profile_trials": len(trials),
              "condition_index": spec["condition_index"], "source_split": "test", "layout_changed": False}
    if chosen:
        recorded = record(spec, chosen["profile"], folder)
        assert recorded["action_sha256"] == chosen["result"]["action_sha256"], "Rendered teacher changed actions"
        check = replay(folder)
        assert check["accepted"], check
        with h5py.File(folder / "trajectory.h5", "r") as teacher, h5py.File(
                BOTH / "episodes" / spec["item_id"] / "trajectory.h5", "r") as both:
            initial_equal = {name: np.array_equal(teacher["policy_inputs/" + name][0], both["policy_inputs/" + name][0])
                             for name in ("main_rgb_raw", "wrist_rgb_raw", "state")}
        assert all(initial_equal.values()), initial_equal
        result.update(accepted=True, replay_verified=True, initial_matches_both=initial_equal,
                      steps=recorded["steps"], decision_inputs=recorded["decision_inputs"],
                      trajectory_sha256=check["trajectory_sha256"], profile=chosen["profile"])
    write_json(receipt_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--profile-limit", type=int, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--episode-ids", help="Repair only completed rejected receipts; keep global progress intact")
    args = parser.parse_args()
    egl = ROOT.parent / ".persist/nvidia-egl"
    if (egl / "10_nvidia.json").exists():
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(egl / "10_nvidia.json"))
        os.environ["LD_LIBRARY_PATH"] = str(egl) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    specs = sorted(read_rows(DATA / "holdouts/test.jsonl"), key=lambda row: row["item_id"])
    suffix = ""
    if args.episode_ids:
        selected = set(args.episode_ids.split(","))
        specs = [spec for spec in specs if spec["item_id"] in selected]
        assert len(specs) == len(selected)
        suffix = "subset_" + hashlib.sha256(args.episode_ids.encode()).hexdigest()[:10] + "_"
    if args.limit:
        specs = specs[:args.limit]
    started = time.time()
    completed = []
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool:
        for result in pool.imap_unordered(collect, [(spec, args.profile_limit) for spec in specs]):
            completed.append(result)
            write_json(TEACHER / (suffix + "progress.json"), {"completed": len(completed), "total": len(specs),
                "accepted": sum(row["accepted"] for row in completed), "updated_unix": time.time()})
    summary = {"complete": all(row["accepted"] for row in completed), "episodes": len(completed),
        "accepted": sum(row["accepted"] for row in completed), "seconds": time.time() - started,
        "decision_inputs": sum(row.get("decision_inputs", 0) for row in completed),
        "failed_ids": [row["episode_id"] for row in completed if not row["accepted"]],
        "all_accepted_replayed": True, "same_initial_observations_as_both": True}
    write_json(TEACHER / ("smoke_completion.json" if args.limit else suffix + "completion.json"), summary)
    print(json.dumps(summary), flush=True)
    if not args.limit and not args.episode_ids:
        assert summary["complete"] and summary["episodes"] == 676, summary
        error = RUN / "experiment_error.json"
        if error.exists() and json.loads(error.read_text()).get("stage") == "teacher_collection":
            archive = RUN / "diagnostics" / f"teacher_recovered_{time.time_ns()}.json"
            archive.parent.mkdir(exist_ok=True)
            error.replace(archive)
        milestone("teacher_ready", "Teacher 평가 trajectory 수집 및 독립 replay 검증 완료", summary)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        write_json(RUN / "experiment_error.json", {"stage": "teacher_collection", "error": traceback.format_exc()})
        from experiment_notifications import notify_progress
        notify_progress(RUN)
        raise
