"""Run paired instruction and verified-grasp ablations with official pi05_libero."""

import argparse
import hashlib
import json
import multiprocessing
from multiprocessing.connection import Client
from pathlib import Path
import shutil
import time
import traceback

import cv2
import h5py
import numpy as np

from evaluate_semantic_pairs_v6 import AUTHKEY, Episode as RecordedEpisode, receive, request
from rollout_common import ROOT, bootstrap, observation_payload, write_json
from semantic_pairs_v6 import DATA, TransportGate, make_environment, sha256
from semantic_v6_diagnostics import ARMS, RUN, DiagnosticMetrics, assert_initial, check_sides, items, prompt


class CombinedOracle:
    def __init__(self, oracle, item):
        self.oracle = oracle
        self.extra = DiagnosticMetrics(item)
        self.step = -1

    def measure(self, env):
        self.step += 1
        return self.extra.update(self.oracle.measure(env), self.step)


class Episode(RecordedEpisode):
    def __init__(self, item, arm, folder, run_digest):
        self.item = item; self.arm = arm; self.folder = folder
        self.native = False; self.horizon = 360; self.chunk_steps = 10
        self.instruction = prompt(item, arm); self.held = arm == "held_transport_name"
        folder.mkdir(parents=True, exist_ok=True)
        self.wrapper, self.obs, oracle = make_environment(item, 384); self.env = self.wrapper.env
        assert_initial(observation_payload(self.obs), item)
        self.side_pixels = check_sides(self.env, oracle, item)
        self.prefix_metadata = None
        if self.held:
            prefix_folder = RUN / "grasp_prefixes" / item["item_id"]
            self.prefix_metadata = json.loads((prefix_folder / "summary.json").read_text())
            assert self.prefix_metadata["passed"] and self.prefix_metadata["action_replay_verified"]
            assert sha256(prefix_folder / "prefix.npz") == self.prefix_metadata["prefix_sha256"]
            with np.load(prefix_folder / "prefix.npz") as prefix:
                for action in prefix["actions"]:
                    self.obs, _, _, _ = self.env.step(action)
                assert np.allclose(self.env.sim.get_state().flatten(), prefix["sim_states"][-1], atol=1e-8, rtol=0)
                payload = observation_payload(self.obs)
                for key in ["image", "wrist_image"]:
                    assert np.array_equal(payload[key], prefix[key]), (item["item_id"], key)
            # Keep the original table-height/displacement origin, but credit no teacher carry.
            held_position = oracle.reference_position(self.env).copy()
            oracle.gate = TransportGate(item, item["validation"]["actual_actor_reference_position"])
            oracle.gate.previous = held_position
            oracle.gate.previous_lifted_grasp = True
        self.oracle = CombinedOracle(oracle, item)
        payload = observation_payload(self.obs)
        self.initial_image_sha = hashlib.sha256(payload["image"].tobytes()).hexdigest()
        (folder / "model.xml").write_text(self.env.sim.model.get_xml())
        self.data = h5py.File(folder / "trajectory.h5", "w")
        self.data.attrs["item_json"] = json.dumps(item, ensure_ascii=False)
        self.data.attrs["instruction"] = self.instruction
        self.data.attrs["arm"] = arm
        self.data.attrs["run_digest"] = run_digest
        self.data.attrs["state_is_model_input"] = False
        self.data.attrs["teacher_prefix_steps"] = self.prefix_metadata["prefix_steps"] if self.held else 0
        self.data.attrs["alignment"] = "post-handoff states[t], decision images at chunk start t -> actions[t] -> states[t+1]"
        self.events = (folder / "events.jsonl").open("w", buffering=1)
        self.writer = cv2.VideoWriter(str(folder / "views_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (768, 432))
        assert self.writer.isOpened()
        self.step = 0; self.started = time.monotonic(); self.run_digest = run_digest
        self.grasped = self.lifted = self.transport = self.sustained = self.native_success = False
        self.first_transport = None; self.max_carried = 0.; self.min_distance = float("inf")
        self.last_metric = self.oracle.measure(self.env)
        if self.held:
            assert self.last_metric["lifted_4cm"] and not self.last_metric["transport_success"]
            assert self.last_metric["carried_xy_m"] == 0
        self.record_state(); self.frame(); self.save_image("initial.png")

    def frame(self):
        image = np.zeros((432, 768, 3), dtype=np.uint8)
        image[:384] = np.concatenate([self.obs["agentview_image"][::-1,::-1],
                                     self.obs["robot0_eye_in_hand_image"][::-1,::-1]], axis=1)
        cv2.putText(image, self.instruction[:110], (7,403), cv2.FONT_HERSHEY_SIMPLEX, .36, (255,255,255), 1)
        xy = self.last_metric.get("xy_only_transport_success", False)
        caption = f"{self.arm} | step {self.step}/360 | lift={self.lifted} strict={self.transport} XY={xy}"
        cv2.putText(image, caption, (7,423), cv2.FONT_HERSHEY_SIMPLEX, .36, (220,230,240), 1)
        self.writer.write(np.ascontiguousarray(image[:,:,::-1]))

    def finish(self):
        result = super().finish()
        result.update(arm=self.arm, source_item_id=self.item["item_id"],
                      actor_asset=self.item["actor_asset"], destination_asset=self.item["destination_asset"],
                      initial_observation_matches_dataset=not self.held, scene_reset_matches_dataset=True,
                      initial_policy_input_matches_verified_prefix=self.held, source_safe_unsafe_labels_used_for_scoring=False,
                      image_x_centers=self.side_pixels, pregrasp_supplied_by_teacher=self.held,
                      prefix_steps=self.prefix_metadata["prefix_steps"] if self.held else 0,
                      **{key:self.last_metric[key] for key in ["xy_only_transport_success", "xy_only_sustained_success",
                                                              "first_xy_only_step", "sustained_lift_10_steps"]})
        write_json(self.folder / "summary.json", result)
        return result


def worker(connection):
    bootstrap(); cv2.setNumThreads(1); episode = None
    try:
        while True:
            message = connection.recv()
            if message["op"] == "close": break
            if message["op"] == "start":
                episode = Episode(message["item"], message["arm"], Path(message["folder"]), message["run_digest"])
                done = False
            else:
                done = episode.advance(message["actions"])
            if done:
                summary = episode.finish(); episode = None; connection.send(dict(done=True, summary=summary))
            else:
                connection.send(dict(done=False, step=episode.step, packet=episode.packet()))
    except Exception:
        error = traceback.format_exc()
        if episode: write_json(episode.folder / "error.json", dict(traceback=error))
        connection.send(dict(error=error))
    finally:
        if episode:
            episode.writer.release(); episode.data.close(); episode.events.close(); episode.wrapper.close()
        connection.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    selected = items(); excluded = []
    if args.arm == "held_transport_name":
        selected = []
        for item in items():
            prefix = RUN / "grasp_prefixes" / item["item_id"] / "summary.json"
            record = json.loads(prefix.read_text())
            (selected if record["passed"] else excluded).append(item)
    if args.limit:
        selected = [selected[i] for i in np.linspace(0, len(selected)-1, min(args.limit,len(selected)), dtype=int)]
    sources = [Path(__file__), Path(__file__).with_name("semantic_v6_diagnostics.py"),
               Path(__file__).with_name("evaluate_semantic_pairs_v6.py"), Path(__file__).with_name("semantic_pairs_v6.py"),
               Path(__file__).with_name("pi05_baseline_server.py")]
    run = dict(arm=args.arm, items=[i["item_id"] for i in selected], excluded_prefix_items=[i["item_id"] for i in excluded],
               dataset_sha256=sha256(DATA/"items.jsonl"), assets_sha256=sha256(DATA/"assets/manifest.json"),
               protocol_sha256=sha256(RUN/"protocol.json"), instruction_manifest_sha256=sha256(RUN/"instruction_manifest.json"),
               horizon=360, chunk_steps=10, batch_size=args.batch_size, additional_training=False,
               scripts={str(p.relative_to(ROOT)):sha256(p) for p in sources},
               prefix_sha256={i["item_id"]:sha256(RUN/"grasp_prefixes"/i["item_id"]/"prefix.npz") for i in selected}
               if args.arm == "held_transport_name" else None)
    digest = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest(); run["digest"] = digest
    if (args.out/"run.json").exists(): assert json.loads((args.out/"run.json").read_text())["digest"] == digest
    write_json(args.out/"run.json", run)
    snapshots = args.out/"source_snapshot"; snapshots.mkdir(exist_ok=True)
    for path in sources: shutil.copy2(path, snapshots/path.name)
    pending = []
    for item in selected:
        path = args.out/"episodes"/item["item_id"]/"summary.json"
        if path.exists(): assert json.loads(path.read_text())["run_digest"] == digest
        else: pending.append(item)
    workers = []; context = multiprocessing.get_context("spawn")
    for _ in range(min(args.batch_size, len(pending))):
        parent, child = context.Pipe(); process = context.Process(target=worker, args=(child,), daemon=True)
        process.start(); child.close(); workers.append((parent, process))
    completed = len(selected)-len(pending)
    try:
        with Client(str(args.socket), family="AF_UNIX", authkey=AUTHKEY) as connection:
            hello = receive(connection); assert hello["ready"]; write_json(args.out/"model.json", hello["metadata"])
            for offset in range(0, len(pending), args.batch_size):
                group = pending[offset:offset+args.batch_size]
                for (parent,_), item in zip(workers, group):
                    parent.send(dict(op="start", arm=args.arm, item=item, folder=str(args.out/"episodes"/item["item_id"]), run_digest=digest))
                active = {i:receive(workers[i][0]) for i in range(len(group))}
                while active:
                    indices = list(active)
                    response = request(connection, dict(op="predict", items=[active[i]["packet"] for i in indices]))
                    write_json(args.out/"inference_audit.json", response["reference_audit"])
                    for i, actions in zip(indices, response["actions"]): workers[i][0].send(dict(op="advance", actions=actions))
                    for i in indices:
                        state = receive(workers[i][0])
                        if state["done"]:
                            completed += 1; summary = state["summary"]; del active[i]
                            print(json.dumps(dict(completed=completed,total=len(selected),arm=args.arm,item=summary["item_id"],
                                                  lift=summary["ever_lifted_4cm"],strict=summary["transport_success"],
                                                  xy=summary["xy_only_transport_success"])),flush=True)
                        else: active[i] = state
                    write_json(args.out/"status.json", dict(completed=completed,total=len(selected),
                        active=[dict(item=group[i]["item_id"],step=state["step"]) for i,state in active.items()],updated_unix=time.time()))
            request(connection, dict(op="disconnect"))
        rows = [json.loads((args.out/"episodes"/i["item_id"]/"summary.json").read_text()) for i in selected]
        (args.out/"episodes.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
        write_json(args.out/"completion.json",dict(complete=True,episodes=len(rows),finished_unix=time.time()))
    finally:
        for parent, process in workers:
            if process.is_alive():
                try: parent.send(dict(op="close"))
                except (BrokenPipeError,EOFError): pass
                process.join(timeout=10)
                if process.is_alive(): process.terminate(); process.join(timeout=5)
            parent.close()


if __name__ == "__main__":
    main()
