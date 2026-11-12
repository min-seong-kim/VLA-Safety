"""Build a portable HTML explorer for all acquired safety dataset records."""

from __future__ import annotations

import base64
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

from PIL import Image, ImageOps
import pyarrow.parquet as pq

from acquire_safety_datasets import DEFAULT_OUT, ROOT, save_json
from inspect_safety_datasets import compact, tfrecords


OUT = DEFAULT_OUT
INSPECT = OUT / "inspection"
TEMPLATES = Path(__file__).parent / "dashboard"


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Media:
    def __init__(self):
        self.images = {}
        self.cache = OUT / "dashboard_assets"
        self.cache.mkdir(exist_ok=True)
        self.errors = []

    def image(self, data):
        if not data:
            return None
        key = hashlib.sha256(data).hexdigest()
        if key not in self.images:
            with Image.open(io.BytesIO(data)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((384, 288))
                stream = io.BytesIO()
                image.save(stream, format="WEBP", quality=62, method=4)
            self.images[key] = "data:image/webp;base64," + base64.b64encode(stream.getvalue()).decode()
        return key

    def file(self, path):
        return self.image(path.read_bytes())

    def video_frame(self, path):
        key = hashlib.sha256(str(path.relative_to(ROOT)).encode()).hexdigest()
        target = self.cache / (key + ".jpg")
        if not target.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
                            "-frames:v", "1", "-vf", "scale=384:-2", "-threads", "1", "-q:v", "5", "-y", str(target)],
                           check=True, capture_output=True, timeout=45)
        return target

    def videos(self, paths):
        result = {}
        paths = sorted(set(paths))
        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = {pool.submit(self.video_frame, path): path for path in paths}
            for number, future in enumerate(as_completed(tasks), 1):
                path = tasks[future]
                try:
                    result[str(path)] = self.file(future.result())
                except Exception as error:
                    self.errors.append({"file": str(path.relative_to(ROOT)), "error": str(error)})
                    result[str(path)] = None
                if number % 200 == 0 or number == len(paths):
                    print(f"VIDEO THUMBNAILS {number}/{len(paths)}", flush=True)
        return result


def record(source, media=None, **extra):
    return {"id": source["id"], "dataset": source["dataset"], "instruction": source.get("instruction") or "",
            "label": source.get("safety_label"), "subset": source.get("source_split", source.get("subset", "")),
            "category": source.get("category") or "", "context": source.get("context", ""),
            "group": source.get("group_id", ""), "media": media or [], "media_names": [],
            "has_action": False, "kind": "image", "source": source, **extra}


def build_records(media):
    records = []
    emb = read_lines(INSPECT / "embodied_image_items.jsonl")
    by_file = defaultdict(dict)
    for row in emb:
        by_file[row["observation_ref"]["file"]][row["observation_ref"]["row"]] = row
    for filename, rows in by_file.items():
        number = 0
        for batch in pq.ParquetFile(ROOT / filename).iter_batches(batch_size=8, columns=["image"]):
            for item in batch.to_pylist():
                row = rows[number]
                img = media.image((item["image"] or {}).get("bytes"))
                records.append(record(row, [img] if img else [],
                                      annotation=row["annotation_only"],
                                      context=row["annotation_only"].get("situation", row["annotation_only"].get("Situation", "")),
                                      context_role="annotation", label_meaning="제안된 행동의 위험 여부"))
                number += 1
        print(f"EMBEDDED IMAGES {Path(filename).name} rows={number}", flush=True)

    asimov = read_lines(INSPECT / "asimov_instruction_items.jsonl")
    by_offset = defaultdict(list)
    for row in asimov:
        ref = row["observation_ref"]
        by_offset[(ref["file"], ref["record_offset"])].append(row)
    for filename in sorted((OUT / "raw").glob("asimov*/0.1.0/*.tfrecord-*")):
        key = filename.parents[1].name
        for index, (offset, row) in enumerate(tfrecords(filename)):
            if "instructions/instruction" in row:
                images = row.get("image", row.get("images", []))
                ids = [media.image(x) for x in images]
                for item in by_offset[(str(filename.relative_to(ROOT)), offset)]:
                    records.append(record(item, ids, label_meaning="undesirable / desirable; 신체 위해보다 넓은 의미"))
            else:
                source = {"id": f"{key}:{index}", "dataset": key,
                          "observation_ref": {"file": str(filename.relative_to(ROOT)), "record_offset": offset},
                          "annotation": {k: compact(v) for k, v in row.items() if k not in {"image", "video/frames"}}}
                if "video/frames" in row:
                    frames = row["video/frames"]
                    selected = sorted({0, len(frames) // 2, len(frames) - 1})
                    ids = [media.image(frames[i]) for i in selected]
                    names = [row["video/timestamps"][i].decode() for i in selected]
                    kind = "video_frames"
                else:
                    ids = [media.image(x) for x in row.get("image", [])]
                    names, kind = [], "image" if ids else "text"
                prompt = row.get("prompt", [b""])[0].decode()
                records.append(record(source, ids, instruction=prompt, media_names=names, kind=kind,
                                      annotation=source["annotation"],
                                      label_meaning="QA / 물체별 제약 / 영상 위험 annotation; task 이진 라벨 없음"))
        print(f"ASIMOV {key}", flush=True)

    rjb = read_lines(INSPECT / "robojailbench_items.jsonl")
    archives, rjb_media = {}, {}
    try:
        for row in rjb:
            ref = row["observation_ref"]
            if ref["member"] not in rjb_media:
                if ref["archive"] not in archives:
                    archives[ref["archive"]] = zipfile.ZipFile(ROOT / ref["archive"])
                rjb_media[ref["member"]] = media.image(archives[ref["archive"]].read(ref["member"]))
            records.append(record(row, [rjb_media[ref["member"]]], label_meaning="malicious / benign goal; 신체 위해 범위 재검토 필요"))
    finally:
        for archive in archives.values():
            archive.close()

    for row in read_lines(INSPECT / "hazardarena_episode_items.jsonl"):
        ids = [media.file(ROOT / x["path"]) for x in row["previews"]]
        names = ["camera " + str(x["camera"]) if x["camera"] is not None else x["dataset"] for x in row["previews"]]
        source_file = ROOT / row["observation_ref"]["file"]
        video = source_file.parent / "third_rgb_video.mp4"
        records.append(record(row, ids, has_action=True, subset=row["source_condition"],
                              category=row["id"].split(":")[-2], media_names=names,
                              video=str(video.relative_to(OUT)) if video.exists() else None,
                              annotation={"trajectory_behavior": row["trajectory_behavior"], "state_and_trajectory_fields": row["state_and_trajectory_fields"]},
                              label_meaning="요청 task의 위험성. unsafe_refuse 궤적은 저자가 거부 행동으로 설명함"))

    shackles = read_lines(OUT / "raw/roboshackles/test/metadata.jsonl")
    libero = read_json(INSPECT / "libero_safety_summary.json")
    video_paths = [OUT / "raw/roboshackles/test" / row["file_name"] for row in shackles]
    for item in libero["sample_episodes"]:
        path = Path(item["file"])
        video_paths.append(OUT / "raw/libero_safety/videos" / path.parent.name / "observation.image" / (path.stem + ".mp4"))
    thumbs = media.videos(video_paths)
    for row in shackles:
        video = OUT / "raw/roboshackles/test" / row["file_name"]
        source = {"id": "roboshackles:" + row["file_name"], "dataset": "roboshackles", "instruction": row["task_instruction"],
                  "category": row["category"], "metadata": row, "file": str(video.relative_to(ROOT))}
        img = thumbs[str(video)]
        records.append(record(source, [img] if img else [], kind="video", video=str(video.relative_to(OUT)),
                              label_meaning="위험 범주가 있는 영상; 별도 safe/unsafe 이진 필드 없음"))
    tasks = {row["task_index"]: row["task"] for row in libero["tasks"]}
    for row in libero["sample_episodes"]:
        path = Path(row["file"])
        video = OUT / "raw/libero_safety/videos" / path.parent.name / "observation.image" / (path.stem + ".mp4")
        source = {"id": "libero_safety:" + path.stem, "dataset": "libero_safety",
                  "instruction": tasks[row["first_row"]["task_index"]], "file": row["file"], "metadata": row}
        img = thumbs[str(video)]
        records.append(record(source, [img] if img else [], kind="video", has_action=True, video=str(video.relative_to(OUT)),
                              subset="task " + str(row["first_row"]["task_index"]),
                              label_meaning="학습 demonstration; semantic unsafe/safe 라벨 없음"))
    return records


def registry():
    definitions = [
        ("embguardtest_v3", "EMBGuardTest v3", "EMBGuard", "전체 · 329행", "Parquet", "image + action text", "task safe / unsafe", "위험 요소와 행동을 분리하는 첫 실험에 적합합니다.", "75개 장면 triple + 52개 selective pair. 127개 그룹을 유지해 분리하세요."),
        ("hazardarena", "HazardArena", "HazardArena", "공개 리뷰본 전체 · HDF5 45개", "HDF5 · MP4", "RGB + state + trajectory", "task SAFE / UNSAFE", "로봇 상태와 궤적까지 포함해 VLM–AE 분석에 연결할 수 있습니다.", "43개 판독 가능. real 15개는 instruction 없음. unsafe_refuse는 위험 지시의 거부 궤적입니다."),
        ("asimov_multimodal_auto_val", "ASIMOV Multimodal · Auto", "ASIMOV v1", "전체 · 장면 50 / 지시 786", "TFRecord", "image + context + instruction", "undesirable / desirable", "로봇 관측에서 지시의 의미적 위험을 살펴봅니다.", "undesirable은 물리적 위해보다 넓은 개념입니다. constitution과 jailbreak prompt는 입력에서 분리했습니다."),
        ("asimov_multimodal_manual_val", "ASIMOV Multimodal · Manual", "ASIMOV v1", "전체 · 장면 59 / 지시 159", "TFRecord", "image + context + instruction", "undesirable / desirable", "수동 작성된 다중모달 사례로 교차 평가할 수 있습니다.", "같은 장면의 지시들을 함께 묶고 로봇 카메라 분포와 구분하세요."),
        ("embhazard", "EMBHazard", "EMBGuard", "4 / 46 shard · 1,314행", "Parquet", "image + action text", "task safe / unsafe", "probe 학습 후보이며 현재 샘플은 657 safe / 657 unsafe입니다.", "전체 학습셋은 약 45.66 GB. 현재 확보분은 원본 shard 0·15·30·45입니다."),
        ("embguard_heldout", "EMBGuard · Heldout", "EMBGuard", "4개 scenario split 전체 · 565행", "Parquet", "image + action text", "task safe / unsafe", "별도 평가 후보입니다.", "이미지 null 2행이 있습니다. 중복 legacy safe/unsafe split은 받지 않았습니다."),
        ("embguardtest_legacy", "EMBGuardTest · Legacy", "EMBGuard", "전체 · 232행", "Parquet", "image + action text", "task safe / unsafe", "구버전 비교용 자료입니다.", "v3와 이미지·ID·split 구성이 다릅니다. 같은 버전의 결과로 합치지 마세요."),
        ("asimov_v2_constraints_with_rationale", "ASIMOV 2.0 · Constraints + rationale", "ASIMOV v2", "전체 · 164행", "TFRecord", "image + constraint prompt", "물체별 위반 bounding box", "위험한 물체를 장면에서 구별하는 보조 과제입니다.", "instruction-level 이진 라벨이 아닙니다. 다른 prompt variant와 장면 중복에 주의하세요."),
        ("asimov_v2_constraints_without_rationale", "ASIMOV 2.0 · Constraints", "ASIMOV v2", "전체 · 164행", "TFRecord", "image + constraint prompt", "물체별 위반 bounding box", "embodiment 제약을 반영한 물체 선택 과제입니다.", "with-rationale와 독립적인 328개 장면으로 간주하지 않습니다."),
        ("asimov_v2_injuries", "ASIMOV 2.0 · Injuries", "ASIMOV v2", "전체 · 319행", "TFRecord", "text prompt + answer", "위험 QA", "시각 입력 없이 위험 지식을 확인하는 자료입니다.", "text-only 데이터입니다. observation 이미지가 없는 것이 정상입니다."),
        ("asimov_v2_videos", "ASIMOV 2.0 · Videos", "ASIMOV v2", "전체 · 287레코드", "TFRecord", "frame sequence + answer", "영상 위험 / 개입 시점", "위험으로 전환되는 장면의 시간적 분석에 적합합니다.", "첫·중간·마지막 프레임을 제공합니다. 영상 전체 annotation을 첫 프레임의 task 라벨로 간주하지 않습니다."),
        ("robojailbench", "RoboJailBench", "RoboJailBench", "ZIP 13개 전체 · 지시 980개", "ZIP · JSONL", "image + paired goals", "malicious / benign", "같은 장면에서 지시를 바꾸는 대조 사례입니다.", "실제 ZIP에서 490쌍을 확인했습니다. 명시적 악의·충돌·사회적 위반을 연구 범위에 맞게 필터링하세요."),
        ("libero_safety", "LIBERO-Safety", "LIBERO-Safety", "전체 meta + 30 episode 샘플", "LeRobot · MP4 · Parquet", "RGB + state(8) + actions(7)", "이진 위험 라벨 없음", "학습 demonstration의 관측·action 구조를 확인합니다.", "원본 19,664 episode 중 15 task별 2개를 확보했습니다. reasoning_safety 평가 task와 구분해야 합니다."),
        ("roboshackles", "RoboShackles", "RoboShackles", "공개 test 전체 · 영상 1,200개", "MP4 · JSONL", "video + instruction", "위험 범주 6종", "사람 위해와 간접 위해의 영상 사례입니다.", "범주별 200개, instruction은 다국어입니다. 안전 대조군과 수치 action/state가 없습니다."),
        ("asimov_multimodal_robopair_val", "ASIMOV Multimodal · RoboPAIR", "ASIMOV v1", "전체 · 7레코드", "TFRecord", "multi-image + instruction", "undesirable", "jailbreak 관련 보조 자료로 보관했습니다.", "사용자의 주 연구 범위에서는 제외하는 편이 적절합니다. 여러 이미지가 들어 있습니다."),
    ]
    catalog = {d["key"]: d for d in read_json(OUT / "catalog/index.json")["datasets"]}
    sizes = Counter()
    for filename in ("download_plan_results.json", "libero_sample_plan_results.json"):
        for row in read_json(OUT / filename)["files"]:
            sizes[row["dataset"]] += row["bytes"]
    result = []
    for values in definitions:
        row = dict(zip(("id", "name", "family", "scope", "format", "input", "labels", "why", "note"), values))
        source = catalog[row["id"]]
        row.update(bytes=sizes[row["id"]], revision=source.get("revision", "GCS generation 고정"),
                   url=("https://huggingface.co/datasets/" + source["repo"] if source["kind"] == "huggingface"
                        else source.get("source", "https://asimov-benchmark.github.io/v2/" if "v2_" in row["id"] else "https://github.com/asimov-benchmark/code")))
        result.append(row)
    return result


def add_unpaired_rjb(records, media):
    """Preserve downloadable scenes that have no released instruction pair."""
    referenced = {r["source"]["observation_ref"]["member"] for r in records if r["dataset"] == "robojailbench"}
    added = 0
    for filename in sorted((OUT / "raw/robojailbench").glob("*.zip")):
        with zipfile.ZipFile(filename) as archive:
            for member in archive.namelist():
                if not member.lower().endswith((".png", ".jpg", ".jpeg")) or member in referenced or "__MACOSX" in member:
                    continue
                img = media.image(archive.read(member))
                path = Path(member)
                source = {"id": "robojailbench:unpaired:" + member, "dataset": "robojailbench",
                          "subset": path.parent.parent.name + " · 지시 없음", "safety_label": None,
                          "instruction": None, "instruction_missing": True,
                          "observation_ref": {"archive": str(filename.relative_to(ROOT)), "member": member},
                          "label_semantics": "raw scene without a matching released instruction"}
                records.append(record(source, [img], label_meaning="대응 instruction·이진 라벨이 없는 원본 장면"))
                referenced.add(member)
                added += 1
    print(f"ROBOJAILBENCH unpaired scenes added={added}", flush=True)


def optimize_images(images):
    def convert(item):
        key, uri = item
        if uri.startswith("data:image/webp"):
            return key, uri
        with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
            stream = io.BytesIO()
            image.save(stream, format="WEBP", quality=62, method=4)
        return key, "data:image/webp;base64," + base64.b64encode(stream.getvalue()).decode()
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(convert, images.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Reuse the existing embedded records and refresh templates")
    args = parser.parse_args()
    media = Media()
    if args.refresh:
        existing = (OUT / "dashboard.html").read_text()
        raw = existing.split('<script id="dataset-payload" type="application/json">', 1)[1].split('</script>', 1)[0]
        previous = json.loads(raw)
        records = previous["records"]
        media.images = optimize_images(previous["images"])
        media.errors = previous["media_errors"]
        del existing, raw, previous
    else:
        records = build_records(media)
    add_unpaired_rjb(records, media)
    datasets = registry()
    order = {row["id"]: i for i, row in enumerate(datasets)}
    records.sort(key=lambda row: order[row["dataset"]])
    assert len(records) == len({row["id"] for row in records})
    counts = Counter(row["dataset"] for row in records)
    for dataset in datasets:
        dataset["count"] = counts[dataset["id"]]
    quality = read_json(INSPECT / "quality_audit.json")
    payload = {"datasets": datasets, "records": records, "images": media.images, "quality": quality,
               "drafts": read_lines(INSPECT / "augmentation_proposals.jsonl"),
               "schemas": {"parquet": [{k:v for k,v in x.items() if k != "samples"} for x in read_json(INSPECT / "parquet_files.json")],
                           "asimov": [{k:v for k,v in x.items() if k != "samples"} for x in read_json(INSPECT / "asimov_summary.json")],
                           "libero": read_json(INSPECT / "libero_safety_summary.json")["info"]},
               "media_errors": media.errors, "created": "2026-11-09"}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("&", "\\u0026")
    html = (TEMPLATES / "index.html").read_text()
    html = html.replace("/*__STYLE__*/", (TEMPLATES / "style.css").read_text())
    html = html.replace("/*__APP__*/", (TEMPLATES / "app.js").read_text())
    html = html.replace("__PAYLOAD__", data)
    destination = OUT / "dashboard.html"
    destination.write_text(html)
    summary = {"html": str(destination.relative_to(ROOT)), "bytes": destination.stat().st_size,
               "records": len(records), "datasets": len(datasets), "embedded_images": len(media.images),
               "counts": dict(counts), "video_thumbnail_errors": media.errors}
    save_json(INSPECT / "dashboard_build.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
