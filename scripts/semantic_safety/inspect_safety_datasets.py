"""Read downloaded safety datasets and export auditable schema/sample inventories.

No dataset-provided Python is executed. Preview images are for inspection only;
original image bytes and robot observations remain unchanged in raw files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import zipfile

import numpy as np
from PIL import Image, ImageOps
import pyarrow.parquet as pq

from acquire_safety_datasets import DEFAULT_OUT, ROOT, save_json
from tfexample import decode_example, _masked_crc32c

sys.path.insert(0, str(ROOT / "envs/safety-inspection"))

try:
    import google_crc32c
except ImportError:
    google_crc32c = None


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def dump_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact(value):
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {str(k): compact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact(v) for v in value]
    if isinstance(value, np.ndarray):
        return compact(value.tolist())
    if isinstance(value, np.generic):
        return compact(value.item())
    return value


def save_preview(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as image:
        size = image.size
        preview = ImageOps.exif_transpose(image).convert("RGB")
        preview.thumbnail((768, 768))
        preview.save(path, quality=90)
    return {"path": relative(path), "original_size": size}


def bootstrap(out):
    inventories = []
    for archive in sorted((out / "raw").glob("*_code/source.tar.gz")):
        destination = out / "sources" / archive.parent.name
        files = []
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = Path(*Path(member.name).parts[1:])
                if not member.isfile() or member.size > 5_000_000:
                    continue
                if name.suffix not in {".py", ".md", ".json", ".bddl", ".yml", ".yaml", ".txt", ".cff", ".ipynb"} and name.name not in {"LICENSE", "SHA256SUMS"}:
                    continue
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError("Unsafe archive path")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    target.write_bytes(tar.extractfile(member).read())
                files.append(str(name))
        inventories.append({"archive": relative(archive), "extracted_text_files": files})
        print(f"SOURCE {archive.parent.name} text_files={len(files)}", flush=True)
    save_json(out / "inspection/source_inventory.json", inventories)

    episodes_path = out / "raw/libero_safety/meta/episodes.jsonl"
    if not episodes_path.exists():
        print("LIBERO episode metadata not downloaded yet", flush=True)
        return
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    by_task = defaultdict(list)
    for episode in episodes:
        by_task[tuple(episode["tasks"])].append(episode)
    selected = []
    for task, values in sorted(by_task.items()):
        selected.extend([values[0], values[len(values) // 2]])
    indices = {item["episode_index"] for item in selected}
    index = json.loads((out / "catalog/index.json").read_text())
    catalog = next(d for d in index["datasets"] if d["key"] == "libero_safety")
    files = []
    for item in catalog["files"]:
        path = Path(item["path"])
        if path.stem.startswith("episode_") and int(path.stem.split("_")[1]) in indices:
            files.append({**item, "dataset": "libero_safety"})
    save_json(out / "libero_sample_plan.json", {
        "selection": "first and middle episode per released task; all 15 task texts covered",
        "selected_episodes": selected, "files": files,
    })
    print(f"LIBERO plan episodes={len(selected)} files={len(files)} bytes={sum(f['size'] for f in files)}")


def parquet_inventory(out):
    results, normalized = [], []
    for directory in sorted((out / "raw").glob("emb*")):
        key = directory.name
        counts, image_counts, paired, schemas = Counter(), Counter(), [], {}
        total = 0
        for path in sorted(directory.glob("data/*.parquet")):
            file = pq.ParquetFile(path)
            split = path.stem.rsplit("-", 3)[0]
            schemas[split] = str(file.schema_arrow)
            row_number = 0
            samples = []
            for batch in file.iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    data = (row.get("image") or {}).get("bytes")
                    sha = hashlib.sha256(data).hexdigest() if data else None
                    image_counts[sha] += 1
                    label = row.get("type", row.get("Type"))
                    if label not in {"safe", "unsafe"}:
                        risk = row.get("risk", row.get("Risk"))
                        label = {"O": "unsafe", "X": "safe"}.get(risk)
                    counts[f"{split}/{label}"] += 1
                    item_id = str(row.get("id") or row.get("ID") or f"{path.stem}:{row_number}")
                    action = row.get("action", row.get("Action"))
                    if key == "embguardtest_v3" and split != "selective_risky":
                        group = "embguardtest_v3:scenario:" + item_id.rsplit("-", 1)[-1]
                    else:
                        group = key + ":image:" + (sha or f"missing:{path.stem}:{row_number}")
                    item = {"id": f"{key}:{path.stem}:{row_number}", "source_id": item_id,
                            "dataset": key, "source_split": split,
                            "safety_label": label, "instruction": action,
                            "observation_ref": {"file": relative(path), "row": row_number, "column": "image"},
                            "image_sha256": sha, "group_id": group,
                            "has_observation_bytes": bool(data),
                            "category": row.get("risk_type", row.get("Risk Type")),
                            "four_way_condition": None,
                            "four_way_status": "requires manual annotation; not inferred from split name",
                            "annotation_only": {k: v for k, v in row.items() if k not in {"image", "action", "Action"}}}
                    normalized.append(item)
                    if len(samples) < 3:
                        sample = compact(row)
                        if data:
                            sample["preview"] = save_preview(data, out / "inspection/previews" / key / f"{path.stem}_{row_number}.jpg")
                        samples.append(sample)
                    if key == "embguardtest_v3":
                        paired.append({"id": item_id, "split": split, "image_sha256": sha,
                                       "group_id": group, "action": action, "label": label})
                    row_number += 1
            total += row_number
            results.append({"dataset": key, "file": relative(path), "split": split,
                            "rows": file.metadata.num_rows, "schema": str(file.schema_arrow), "samples": samples})
        if total:
            summary = {"dataset": key, "rows": total, "counts": dict(counts),
                       "unique_image_hashes": len([k for k in image_counts if k is not None]),
                       "missing_images": image_counts.get(None, 0), "schemas": schemas}
            save_json(out / "inspection" / f"{key}_summary.json", summary)
            print(f"PARQUET {key} rows={total} labels={dict(counts)}", flush=True)
            if paired:
                save_json(out / "inspection/embguardtest_v3_pair_audit.json", paired)
    save_json(out / "inspection/parquet_files.json", results)
    dump_lines(out / "inspection/embodied_image_items.jsonl", normalized)


def crc(data):
    if google_crc32c is None:
        return _masked_crc32c(data)
    value = google_crc32c.value(data)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def tfrecords(path):
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            length = stream.read(8)
            if not length:
                return
            length_crc = stream.read(4)
            if len(length) != 8 or len(length_crc) != 4 or crc(length) != struct.unpack("<I", length_crc)[0]:
                raise ValueError("Invalid TFRecord length/CRC")
            size = struct.unpack("<Q", length)[0]
            if size > 512 * 1024 * 1024:
                raise ValueError("Oversized TFRecord record")
            record, checksum = stream.read(size), stream.read(4)
            if len(record) != size or len(checksum) != 4 or crc(record) != struct.unpack("<I", checksum)[0]:
                raise ValueError("Invalid TFRecord payload/CRC")
            yield offset, decode_example(record)


def asimov_inventory(out):
    summaries, items = [], []
    for path in sorted((out / "raw").glob("asimov*/0.1.0/*.tfrecord-*")):
        key = path.parents[1].name
        samples, fields, labels = [], {}, Counter()
        count = instruction_count = 0
        for offset, row in tfrecords(path):
            count += 1
            for name, value in row.items():
                fields.setdefault(name, {"type": type(value[0]).__name__ if value else "empty", "first_length": len(value)})
            if "instructions/instruction" in row:
                actions = row["instructions/instruction"]
                truths = row["instructions/undesirable_groundtruth_answer"]
                contexts = row["instructions/context"]
                if not len(actions) == len(truths) == len(contexts):
                    raise ValueError("ASIMOV ragged instruction-label mismatch")
                image_feature = "image" if "image" in row else "images"
                scene_images = row[image_feature]
                sha = hashlib.sha256(b"".join(hashlib.sha256(image).digest() for image in scene_images)).hexdigest()
                for index, (action, truth, context) in enumerate(zip(actions, truths, contexts)):
                    label = "unsafe" if truth else "safe"
                    labels[label] += 1
                    instruction_count += 1
                    items.append({"id": f"{key}:{count - 1}:{index}", "dataset": key,
                                  "instruction": action.decode(), "context": context.decode(),
                                  "safety_label": label, "label_semantics": "undesirable, broader than physical injury",
                                  "observation_ref": {"file": relative(path), "record_offset": offset,
                                                      "feature": image_feature, "image_count": len(scene_images)},
                                  "group_id": key + ":image:" + sha, "image_sha256": sha,
                                  "four_way_condition": None})
            if len(samples) < 3:
                sample = {k: compact(v[:2]) for k, v in row.items()}
                for image_key in ("image", "original_image", "images", "original_images", "video/frames"):
                    if row.get(image_key):
                        sample[image_key + "_preview"] = save_preview(row[image_key][0], out / "inspection/previews" / key / f"record{count - 1}_{image_key.replace('/', '_')}.jpg")
                        sample[image_key + "_count"] = len(row[image_key])
                samples.append(sample)
        summary = {"dataset": key, "file": relative(path), "records": count,
                   "instruction_count": instruction_count, "labels": dict(labels), "fields": fields, "samples": samples,
                   "crc32c_verified": True}
        summaries.append(summary)
        print(f"TFRECORD {key} records={count} instructions={instruction_count} labels={dict(labels)}", flush=True)
    save_json(out / "inspection/asimov_summary.json", summaries)
    dump_lines(out / "inspection/asimov_instruction_items.jsonl", items)


def rjb_inventory(out):
    records, files = [], []
    for path in sorted((out / "raw/robojailbench").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupt ZIP member: {bad}")
            names = archive.namelist()
            instruction_files = [n for n in names if n.endswith(("_instr.json", "_instr.jsonl"))]
            for name in instruction_files:
                raw = archive.read(name).decode()
                try:
                    content = json.loads(raw)
                except json.JSONDecodeError:
                    content = [json.loads(line) for line in raw.splitlines() if line.strip()]
                records.append({"archive": relative(path), "member": name, "content": content})
            images = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg")) and "__MACOSX" not in n]
            preview = save_preview(archive.read(images[0]), out / "inspection/previews/robojailbench" / (path.stem + ".jpg")) if images else None
            files.append({"file": relative(path), "members": len(names), "images": len(images),
                          "instruction_files": instruction_files, "sample_paths": names[:12], "preview": preview,
                          "zip_crc_verified": True})
    save_json(out / "inspection/robojailbench_files.json", files)
    save_json(out / "inspection/robojailbench_instructions.json", records)
    print(f"ROBOJAILBENCH archives={len(files)} instruction_files={len(records)}", flush=True)


def video_preview(path, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
                    "-frames:v", "1", "-vf", "scale=640:-2", "-threads", "1", "-y", str(output)],
                   check=True, timeout=60, capture_output=True)
    return relative(output)


def video_inventory(out):
    path = out / "raw/roboshackles/test/metadata.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        counts = Counter(row.get("category") for row in rows)
        samples = {}
        for row in rows:
            category = row.get("category")
            if category in samples:
                continue
            candidates = sorted((out / "raw/roboshackles/test" / str(category)).glob("*.mp4"))
            if not candidates:
                continue
            # Select the matching metadata row for the actual downloaded preview.
            video = candidates[0]
            video_name = str(video.relative_to(out / "raw/roboshackles/test"))
            matched = next((r for r in rows if r["file_name"] == video_name), None)
            samples[category] = {"row": matched, "video": relative(video),
                                 "preview": video_preview(video, out / "inspection/previews/roboshackles" / (category + ".jpg"))}
        save_json(out / "inspection/roboshackles_summary.json", {"rows": len(rows), "categories": dict(counts),
                  "keys": list(rows[0]), "samples": samples, "downloaded_videos": len(list(path.parent.glob("*/*.mp4")))})
        print(f"ROBOSHACKLES rows={len(rows)} categories={dict(counts)}", flush=True)
    hazard = []
    for path in sorted((out / "raw/hazardarena").glob("real/*/*/*/third_rgb_video.mp4")):
        parts = path.relative_to(out / "raw/hazardarena").parts
        hazard.append({"video": relative(path), "condition": parts[2], "task": parts[1],
                       "preview": video_preview(path, out / "inspection/previews/hazardarena" / f"{parts[1]}_{parts[2]}.jpg")})
    save_json(out / "inspection/hazardarena_video_samples.json", hazard)


def hdf_inventory(out):
    import h5py

    records = []
    for path in sorted((out / "raw/hazardarena").rglob("*.hdf5")):
        entry = {"file": relative(path), "groups": {}, "datasets": {}, "previews": []}
        try:
            handle = h5py.File(path, "r")
        except OSError as error:
            entry["open_error"] = str(error)
            records.append(entry)
            print(f"HDF5 UNREADABLE {path.relative_to(out / 'raw/hazardarena')}: {error}", flush=True)
            continue
        with handle:
            entry["attributes"] = compact(dict(handle.attrs))

            def visit(name, obj):
                if isinstance(obj, h5py.Group):
                    attrs = compact(dict(obj.attrs))
                    if attrs:
                        entry["groups"][name] = attrs
                elif isinstance(obj, h5py.Dataset):
                    description = {"shape": obj.shape, "dtype": str(obj.dtype), "attributes": compact(dict(obj.attrs))}
                    if obj.size < 40 or obj.dtype.kind in {"S", "U", "O"}:
                        description["sample"] = compact(obj[()] if obj.ndim == 0 else obj[:1])
                    entry["datasets"][name] = description
                    if obj.ndim in {4, 5} and obj.shape[-1] == 3 and obj.dtype == np.dtype("uint8") and obj.shape[0]:
                        cameras = range(obj.shape[1]) if obj.ndim == 5 else [None]
                        for camera in cameras:
                            preview_path = out / "inspection/previews/hazardarena_hdf" / ("_".join(path.parts[-4:-1]) + "_" + name.replace("/", "_") + f"_cam{camera}.jpg")
                            preview_path.parent.mkdir(parents=True, exist_ok=True)
                            image = Image.fromarray(obj[0, camera] if camera is not None else obj[0])
                            image.thumbnail((768, 768))
                            image.save(preview_path)
                            entry["previews"].append({"dataset": name, "frame": 0, "camera": camera, "path": relative(preview_path)})

            handle.visititems(visit)
        records.append(entry)
        print(f"HDF5 {path.relative_to(out / 'raw/hazardarena')} datasets={len(entry['datasets'])}", flush=True)
    save_json(out / "inspection/hazardarena_hdf5.json", records)


def libero_inventory(out):
    directory = out / "raw/libero_safety"
    info = json.loads((directory / "meta/info.json").read_text())
    tasks = [json.loads(line) for line in (directory / "meta/tasks.jsonl").read_text().splitlines()]
    episodes = []
    for path in sorted(directory.glob("data/*/*.parquet")):
        table = pq.read_table(path)
        episodes.append({"file": relative(path), "rows": len(table), "schema": str(table.schema),
                         "first_row": table.slice(0, 1).to_pylist()[0]})
    previews = []
    for path in sorted(directory.glob("videos/*/observation.image/*.mp4")):
        if len(previews) >= 15:
            break
        previews.append({"video": relative(path), "preview": video_preview(path, out / "inspection/previews/libero_safety" / (path.stem + ".jpg"))})
    save_json(out / "inspection/libero_safety_summary.json", {"info": info, "tasks": tasks, "sample_episodes": episodes, "previews": previews})
    print(f"LIBERO sampled_episodes={len(episodes)} released_tasks={len(tasks)}", flush=True)


def status(out):
    for path in sorted(out.glob("*_results.json")):
        obj = json.loads(path.read_text())
        by_key = defaultdict(lambda: {"files": 0, "bytes": 0})
        for row in obj["files"]:
            by_key[row["dataset"]]["files"] += 1
            by_key[row["dataset"]]["bytes"] += row["bytes"]
        print(path.name, json.dumps(by_key), "errors", obj["errors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "status", "parquet", "asimov", "robojailbench", "video", "hdf", "libero", "all"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    functions = {"bootstrap": bootstrap, "status": status, "parquet": parquet_inventory,
                 "asimov": asimov_inventory, "robojailbench": rjb_inventory,
                 "video": video_inventory, "hdf": hdf_inventory, "libero": libero_inventory}
    if args.command == "all":
        for key in ("parquet", "asimov", "robojailbench", "video", "hdf", "libero"):
            functions[key](args.out)
    else:
        functions[args.command](args.out)


if __name__ == "__main__":
    main()
