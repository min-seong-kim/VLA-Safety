"""Validate, document, preview, and package a generated explicit-safety dataset."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def validate(folder, rows):
    checks = {}
    groups, observations, image_splits, state_splits = [defaultdict(set) for _ in range(4)]
    pairs = defaultdict(Counter)
    expected_images = {}
    referenced = set()
    camera_positions = defaultdict(set)
    for row in rows:
        assert len(row["state"]) == 8 and np.isfinite(row["state"]).all(), row["id"]
        groups[row["group_id"]].add(row["split"])
        observations[row["observation_id"]].add((row["image"], tuple(row["state"])))
        image_splits[row["image_sha256"]].add(row["split"])
        state_splits[row["state_fingerprint"]].add(row["split"])
        pairs[row["observation_id"]][row["safety_label"]] += 1
        expected_images[row["image"]] = row["image_sha256"]
        for key in ["image", "wrist_image", "sim_state_file", "camera_file"]:
            p = (folder / row[key]).resolve()
            assert p.is_relative_to(folder) and p.exists(), row[key]
            referenced.add(row[key])
    assert rows and len({r["id"] for r in rows}) == len(rows)
    assert all(len(x) == 1 for x in groups.values()), "Physical layout crosses splits"
    assert all(len(x) == 1 for x in observations.values()), "Pair has different input state/image"
    assert all(x == {"safe": 2, "unsafe": 2} for x in pairs.values()), "Unbalanced observation pair"
    assert all(len(x) == 1 for x in image_splits.values()), "Image hash crosses splits"
    assert all(len(x) == 1 for x in state_splits.values()), "Physical state crosses splits"
    for name, digest in expected_images.items():
        path = folder / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        with Image.open(path) as image:
            image.load()
            assert image.mode == "RGB" and image.size == (512, 512), name
    for name in sorted(referenced):
        if name.endswith(".camera.json"):
            camera = json.loads((folder / name).read_text())
            assert min(x["pixels"] for x in camera["visibility"].values()) >= 40, name
            camera_positions[Path(name).parent.name].add(tuple(camera["world_position"]))
    assert all(len(x) == 5 for x in camera_positions.values()), "Camera views are not all distinct"
    checks.update({"unique_ids": True, "readable_rgb_512_images": len(expected_images),
                   "state_dim_8_finite": True, "two_safe_two_unsafe_per_observation": True,
                   "paired_images_and_states_identical": True, "group_split_leakage": 0,
                   "exact_image_split_leakage": 0, "exact_state_split_leakage": 0,
                   "critical_object_visibility_passed": True, "five_distinct_camera_positions": True,
                   "independent_human_label_validation": False, "policy_rollouts_executed": 0})
    dump(folder / "validation.json", checks)
    return referenced


def write_protocols(folder, rows):
    families = sorted({r["family_id"] for r in rows})
    folds = []
    for index, test in enumerate(families):
        validation = families[(index + 1) % len(families)]
        training = [f for f in families if f not in [test, validation]]
        folds.append({"fold": index, "train_families": training,
                      "validation_families": [validation], "test_families": [test],
                      "counts": {"train": sum(r["family_id"] in training for r in rows),
                                 "validation": sum(r["family_id"] == validation for r in rows),
                                 "test": sum(r["family_id"] == test for r in rows)}})
    dump(folder / "leave_one_family_out.json", {
        "recommended_for": "Testing transfer beyond the same situation and fixed instruction templates.",
        "folds": folds, "grouping": "All layouts, cameras, labels and paraphrases of a family stay together."})
    keys = ["id", "instruction", "image", "wrist_image", "state", "state_layout"]
    (folder / "probe_inputs.jsonl").write_text("".join(
        json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n" for r in rows))


def preview(folder, rows):
    configuration = json.loads((folder / "generation.json").read_text())["configuration"]
    scenarios = {s["id"]: s for s in configuration["scenarios"]}
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group_id"], {"id": r["group_id"], "family": r["family_id"],
                                           "seed": r["seed"], "split": r["split"], "views": {},
                                           "safe": scenarios[r["family_id"]]["safe"],
                                           "unsafe": scenarios[r["family_id"]]["unsafe"]})
        if r["view_id"] in g["views"]:
            continue
        stream = io.BytesIO()
        with Image.open(folder / r["image"]) as im:
            im.thumbnail((448, 448))
            im.save(stream, "WEBP", quality=74)
        g["views"][r["view_id"]] = "data:image/webp;base64," + base64.b64encode(stream.getvalue()).decode()
    summary = json.loads((folder / "summary.json").read_text())
    payload = {"summary": summary, "groups": list(groups.values()),
               "titles": {k: v["title_ko"] for k, v in scenarios.items()}}
    source = Path(__file__).with_name("libero_explicit_preview.html").read_text()
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (folder / "preview.html").write_text(source.replace("__DATA__", data))


def documentation(folder):
    summary = json.loads((folder / "summary.json").read_text())
    content = f"""LIBERO Explicit Semantic Safety — v1

목적: 명시적 instruction을 수행하면 위험한지 VLA 내부 representation에서 읽어낼 수 있는지 분석.
이 자료는 LIBERO-Safety의 공개 장면을 새로 렌더링하고 안전/위험 지시를 연결한 연구용 파생 자료입니다.
공식 LIBERO-Safety의 새로운 릴리스나 공식 annotation은 아닙니다.

구성
- 상황 유형: {len(summary['families'])}
- 물리 배치: {summary['physical_layouts']}
- 주 카메라 관측: {summary['primary_observations']}
- 지시-관측 레코드: {summary['records']} ({summary['labels']})
- 각 관측은 safe 2개 + unsafe 2개 지시와 연결됩니다. 같은 이미지와 robot state를 공유합니다.
- 원본 agentview, 좌/우 15도, 높은 시점, 가까운 시점의 다섯 카메라를 사용했습니다.
- 별도로 배치당 손목 카메라 영상 1장을 저장했습니다. 데이터셋 크기 집계에는 주 관측만 셉니다.
- 원본 RGB PNG는 512×512입니다. 미리보기 HTML의 WebP만 축소했습니다.
- 카메라가 달라져도 독립적인 상황이 늘어나는 것은 아닙니다.

열기
- preview.html은 이미지를 내장하므로 HTML 파일만 PC로 내려받아 Chrome/Edge로 열 수 있습니다.
- items.jsonl: 모든 레코드와 annotation. 상대 경로의 기준은 이 데이터셋 폴더입니다.
- probe_inputs.jsonl: 이미지 경로, instruction, state 등 입력 필드만 남긴 인덱스입니다.
  id와 파일명은 관리용이며 tokenizer 입력에 넣지 마세요. 실제 모델 입력에는 이미지 픽셀,
  instruction, 필요 시 state만 사용하세요. safety_label, hazard, rationale은 입력에서 제외합니다.
- observations/: 실제 PNG, camera metadata, robot/simulator state.
- generation.json: 원본 코드/asset 고정 버전, 지시 템플릿, 생성 옵션.
- validation.json: 형식, 짝 구성, 관측/상태 중복, 분할 누출, 물체 가시성 검사.

라벨
- task_is_unsafe는 해당 지시를 수행할 때의 잠재적 신체 위해 또는 물건 손상 가능성입니다.
- 안전 라벨은 로봇의 실제 성공이나 모든 trajectory의 무충돌을 보증하는 라벨이 아닙니다.
- knife_edge는 즉시 상해가 아니라 칼 손잡이를 가장자리 밖에 두는 위험 배치입니다.
  직접적인 상해/손상 지시만 필요하면 이 family를 제외하세요.
- 지시는 명시적 물체 이름을 사용합니다. 위치 지시만으로 대상을 추론하는 implicit 조건은 없습니다.
- safe/unsafe 지시는 연구 목적으로 작성했으며 독립적인 사람의 주석 검증은 아직 하지 않았습니다.
- 원본 물리 모델에서 화재·파손·상해를 실제로 실행하거나 재현한 자료가 아닙니다.

분할
- train/validation/test.jsonl은 같은 상황 안에서 배치를 14/3/3개씩 나눕니다.
  동일 배치의 모든 카메라, safe/unsafe, 지시 변형은 반드시 함께 둡니다.
- 이 분할은 상황 및 지시 템플릿이 train/test에 반복됩니다. 새 위험 상황으로의 일반화 성능으로
  해석하면 안 됩니다. leave_one_family_out.json의 6개 fold를 주 일반화 평가로 권합니다.
- LDA/probe와 표준화/PCA 등 학습 가능한 전처리는 train에서만 fit하세요.
- 명시적 지시는 언어만으로도 위험을 판단할 수 있습니다. instruction-only baseline을 함께 보고,
  높은 probe 점수만으로 관측을 사용하거나 경고가 AE에서 사용된다고 결론 내리지 마세요.

생성 상태
- VLA 추론, feature 추출, probe 학습, 정책 rollout은 수행하지 않았습니다.
- 물컵 장면은 물 입자를 가라앉히기 위해 20회 초기 자세 유지 step 후 촬영했습니다.
- 나머지는 공개 장면의 reset 상태를 촬영했습니다. 이것은 action demonstration이 아닙니다.
- 초기 배치와 관측의 일관성이 부족한 bottle_edge, wood_candle은 제외했습니다.

재생성 (프로젝트 루트에서)
envs/libero-render/bin/python scripts/semantic_safety/generate_libero_explicit.py --out data/libero_semantic_explicit_v1_rebuild --layouts 20 --resolution 512
python3 scripts/semantic_safety/package_libero_explicit.py --data data/libero_semantic_explicit_v1_rebuild

출처
https://github.com/LIBERO-SAFETY/LIBERO-Safety
https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets
https://pypi.org/project/robosuite/1.4.0/
원본 코드와 개별 asset의 라이선스/출처 조건을 따릅니다. 본 ZIP에는 simulator assets를 포함하지 않습니다.
"""
    (folder / "README.txt").write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/libero_semantic_explicit_v1")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()
    folder = args.data.resolve()
    rows = [json.loads(x) for x in (folder / "items.jsonl").read_text().splitlines()]
    validate(folder, rows)
    write_protocols(folder, rows)
    preview(folder, rows)
    documentation(folder)
    provenance = folder / "provenance"
    provenance.mkdir(exist_ok=True)
    for name in ["generate_libero_explicit.py", "package_libero_explicit.py",
                 "libero_explicit_scenarios.json", "libero_explicit_preview.html"]:
        shutil.copy2(Path(__file__).with_name(name), provenance / name)
    for name in ["libero_extracted_assets.json", "robosuite_render_assets.json"]:
        shutil.copy2(ROOT / "data/safety_sources/inspection" / name, provenance / name)
    shutil.copy2(ROOT / "data/safety_sources/sources/libero_safety_code/LICENSE",
                 provenance / "LIBERO-Safety-LICENSE.txt")
    checksums = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + str(path.relative_to(folder)))
    (folder / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    if not args.no_zip:
        target = folder.with_suffix(".zip")
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, str(Path(folder.name) / path.relative_to(folder)))
        print("ZIP", target, target.stat().st_size)
    print("VALIDATION PASSED", len(rows), "records")
    print("HTML", folder / "preview.html", (folder / "preview.html").stat().st_size)


if __name__ == "__main__":
    main()
