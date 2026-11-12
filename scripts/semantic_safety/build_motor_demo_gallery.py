"""Export one accepted training demonstration per asset for visual review."""

import json
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np

from rollout_common import ROOT, write_json


def main():
    source = ROOT / "data/motor_skill_b_v1/demonstrations"
    destination = ROOT / "runs/motor_lora_b_v1/teacher_examples"
    destination.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines()]
    examples = {}
    for row in rows:
        if row["accepted"]:
            examples.setdefault(row["asset"], row)
    gallery = []
    for asset, row in sorted(examples.items()):
        output = destination / f"{asset}.mp4"
        with h5py.File(ROOT / row["path"]) as data:
            spec = json.loads(data.attrs["spec_json"])
            if not output.exists():
                temporary = destination / f"{asset}_raw.mp4"
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 304))
                for step in range(len(data["actions"]) + 1):
                    pair = np.concatenate([data["observations/main_rgb_raw"][step][::-1, ::-1],
                                           data["observations/wrist_rgb_raw"][step][::-1, ::-1]], axis=1)
                    if step == 0:
                        cv2.imwrite(str(destination / f"{asset}.png"), np.ascontiguousarray(pair[:, :, ::-1]))
                    frame = np.zeros((304, 512, 3), dtype=np.uint8)
                    frame[:256] = pair
                    cv2.putText(frame, spec["instruction"][:80], (5, 275), cv2.FONT_HERSHEY_SIMPLEX, .3,
                                (255, 255, 255), 1)
                    cv2.putText(frame, f"SCRIPTED TEACHER | {spec['episode_id']} | replay validated", (5, 294),
                                cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
                    writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
                writer.release()
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(temporary), "-c:v", "libx264",
                                "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "25", "-movflags", "+faststart",
                                str(output)], check=True)
                temporary.unlink()
        gallery.append({"asset": asset, "episode_id": row["episode_id"], "instruction": spec["instruction"],
                        "steps": row["steps"], "video": f"teacher_examples/{asset}.mp4",
                        "poster": f"teacher_examples/{asset}.png", "source_trajectory": row["path"]})
        print(asset, row["steps"], flush=True)
    write_json(destination / "gallery.json", gallery)


if __name__ == "__main__":
    main()
