"""Build and audit a semantic-diversity catalog; this does not render images.

Usage: python3 scripts/semantic_safety/build_synthesis_v2_catalog.py
Optional CPU lexical audit: add --lexical-audit (requires scikit-learn).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil

from synthesis_v2_scenarios import CATEGORIES, FAMILIES

ROOT = Path(__file__).resolve().parents[2]
STATUS = "authored_blueprint_not_rendered_not_independently_reviewed"
SOURCE_LINKS = [
    {"id": "hazardarena", "title": "HazardArena — taxonomy and matched twins",
     "url": "https://arxiv.org/html/2604.12447v1",
     "use": "Method attribution. The local revised PDF and pinned code have 51 inventory tasks; arXiv v1 has 40."},
    {"id": "autort", "title": "AutoRT — task generation and constitution-based critique",
     "url": "https://auto-rt.github.io/",
     "use": "Generation/critique workflow. Its deployment-specific ban on sharp objects is not our labeling rule."},
    {"id": "asimov2", "title": "ASIMOV-2.0 — injury/constraint-grounded generation",
     "url": "https://arxiv.org/html/2509.21651v1",
     "use": "Scenario grounding and independent review as a design reference; no external critic was run here."},
    {"id": "iso_scope", "title": "ISO 13482:2014 — public scope",
     "url": "https://www.iso.org/standard/53820.html",
     "use": ("Public scope only. No full-standard access, clause mapping, certification, "
             "or ISO-official taxonomy claimed.")},
]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates():
    pairs, rows = [], []
    seen_instructions = set()
    assert len({f["family_id"] for f in FAMILIES}) == len(FAMILIES)
    for family in FAMILIES:
        assert family["category"] in CATEGORIES
        for index, case in enumerate(family["cases"]):
            pair_id = f"{family['family_id']}__{index + 1:02d}"
            pair = {k: v for k, v in family.items() if k != "cases"}
            pair.update(case)
            pair.update({"pair_id": pair_id, "status": STATUS,
                         "reference_type": "explicit", "scene_instance_count": 0,
                         "label_source": "agent_authored_candidate",
                         "independent_human_review": False,
                         "observation_plan": {"environment": "LIBERO-compatible custom scene",
                                              "primary_view": "agentview", "canonical_layouts": 1,
                                              "primary_views_per_layout": 1,
                                              "same_observation_for_both_labels": True,
                                              "resolution": [512, 512]},
                         "evidence_location": "explicit_instruction_including_shared_context",
                         "visual_grounding_proven": False})
            assert case["safe_request"].split()[0] == case["unsafe_request"].split()[0], pair_id
            assert case["shared_context"] and len(case["entities"]) >= 3, pair_id
            pairs.append(pair)
            for label in ["safe", "unsafe"]:
                instruction = case["shared_context"] + " " + case[label + "_request"]
                assert not re.search(r"\b(unsafe|safely|dangerous|harmless)\b", instruction.lower()), pair_id
                assert instruction not in seen_instructions, (pair_id, label)
                seen_instructions.add(instruction)
                opaque_id = hashlib.sha256((pair_id + ":" + label).encode()).hexdigest()[:16]
                rows.append({"id": opaque_id, "pair_id": pair_id, "family_id": family["family_id"],
                             "mechanism_group": family["mechanism_group"], "category": family["category"],
                             "safety_label": label, "task_is_unsafe": label == "unsafe",
                             "instruction": instruction, "shared_context": case["shared_context"],
                             "request": case[label + "_request"], "status": STATUS,
                             "image": None, "robot_state": None,
                             "annotation_only": {"causal_rule_ko": family["causal_rule_ko"],
                                                 "label_source": "agent_authored_candidate",
                                                 "independent_human_review": False}})
    random.Random(26061109).shuffle(rows)
    return pairs, rows


def make_folds(pairs, key, count=6):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair[key]].append(pair["pair_id"])
    # Keep similar mechanisms together, even when their names or hazard categories differ.
    names = sorted(groups, key=lambda g: (-len(groups[g]), hashlib.sha256(g.encode()).hexdigest()))
    bins = [[] for _ in range(count)]
    sizes = [0] * count
    for name in names:
        index = min(range(count), key=lambda i: (sizes[i], i))
        bins[index].append(name)
        sizes[index] += len(groups[name])
    folds = []
    for index in range(count):
        test = set(bins[index])
        validation = set(bins[(index + 1) % count])
        train = set(groups) - test - validation
        fold = {"fold": index}
        for split, group_set in [("train", train), ("validation", validation), ("test", test)]:
            fold[split + "_groups"] = sorted(group_set)
            fold[split + "_pairs"] = sorted(p for g in group_set for p in groups[g])
            selected = [p for p in pairs if p["pair_id"] in fold[split + "_pairs"]]
            fold[split + "_category_counts"] = dict(Counter(p["category"] for p in selected))
        assert not (train & test or train & validation or validation & test)
        assert sum(len(fold[s + "_pairs"]) for s in ["train", "validation", "test"]) == len(pairs)
        folds.append(fold)
    assert Counter(p for f in folds for p in f["test_pairs"]) == Counter(p["pair_id"] for p in pairs)
    return {"grouping_key": key, "stage": "proposed_before_rendering_and_independent_review",
            "freeze_after": "Asset validation, scene inspection, and label review; rebuild if cases are excluded.",
            "claim": ("Held-out authored mechanism groups; groups are manually defined "
                      "and not independent natural laws."
                      if key == "mechanism_group" else
                      "Held-out situation families; related mechanisms may still occur in training."),
            "folds": folds}


def lexical_audit(rows, protocol):
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline

    outcomes = {}
    predictions = []
    for field in ["instruction", "request"]:
        all_y, all_p, fold_stats = [], [], []
        for fold in protocol["folds"]:
            train_ids, test_ids = set(fold["train_pairs"]), set(fold["test_pairs"])
            train = [r for r in rows if r["pair_id"] in train_ids]
            test = [r for r in rows if r["pair_id"] in test_ids]
            model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
                                  LogisticRegression(C=1.0, solver="liblinear", random_state=26061109))
            model.fit([r[field] for r in train], [int(r["task_is_unsafe"]) for r in train])
            y = np.array([int(r["task_is_unsafe"]) for r in test])
            probability = model.predict_proba([r[field] for r in test])[:, 1]
            all_y.extend(y.tolist())
            all_p.extend(probability.tolist())
            paired = defaultdict(dict)
            for row, value in zip(test, probability):
                paired[row["pair_id"]][row["safety_label"]] = float(value)
                predictions.append({"id": row["id"], "pair_id": row["pair_id"], "field": field,
                                    "fold": fold["fold"], "label": row["safety_label"],
                                    "probability_unsafe": float(value)})
            ordering = [float(p["unsafe"] > p["safe"]) + .5 * float(p["unsafe"] == p["safe"])
                        for p in paired.values()]
            fold_stats.append({"fold": fold["fold"], "train_records": len(train), "test_records": len(test),
                               "balanced_accuracy": float(balanced_accuracy_score(y, probability >= .5)),
                               "auroc": float(roc_auc_score(y, probability)),
                               "pair_ranking_accuracy": float(np.mean(ordering))})
        outcomes[field] = {"pooled_oof_balanced_accuracy": float(balanced_accuracy_score(all_y, np.array(all_p) >= .5)),
                           "pooled_oof_auroc": float(roc_auc_score(all_y, all_p)),
                           "mean_fold_auroc": float(np.mean([f["auroc"] for f in fold_stats])),
                           "folds": fold_stats}
    return {"status": "completed_cpu_lexical_diagnostic_only", "sklearn_version": sklearn.__version__,
            "model": "word unigram/bigram TF-IDF + logistic regression; C=1; fixed, not tuned",
            "protocol": "six mechanism-group holdout folds; vocabulary and IDF fitted only on training",
            "interpretation_ko": ("명시적 지시의 어휘 분리도 진단이다. 높은 값이 곧 허위 상관이라는 뜻도, "
                                  "낮은 값이 편향 없음이라는 뜻도 아니다. VLA representation probe 결과가 아니다."),
            "validation_used_for_hyperparameter_tuning": False, "results": outcomes}, predictions


def asset_requirements(pairs):
    requirements = defaultdict(list)
    for pair in pairs:
        for entity in pair["entities"]:
            requirements[entity].append(pair["pair_id"])
    return [{"entity": entity, "used_by_pairs": sorted(ids), "asset_binding": None,
             "status": "requires_explicit_asset_binding_and_visual_check"}
            for entity, ids in sorted(requirements.items())]


def write_review_sheet(path, pairs):
    columns = ["pair_id", "category", "family_id", "mechanism_group", "shared_context", "entities",
               "safe_request", "unsafe_request", "review_safe_label", "review_unsafe_label",
               "review_context_sufficiency", "review_visual_match", "review_comment"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for pair in pairs:
            row = {k: pair[k] for k in columns if k in pair}
            row["entities"] = " | ".join(pair["entities"])
            writer.writerow(row)


def read_source_inventory(out):
    path = ROOT / "data/safety_sources/sources/hazardarena_code/risk_task.json"
    inventory = json.loads(path.read_text())
    dump(out / "reference_hazardarena_inventory.json", {
        "source_file": str(path.relative_to(ROOT)), "sha256": digest(path),
        "note": "Reference inventory only. These are not generated observations or additional v2 examples.",
        "tasks": inventory})
    return {"tasks": len(inventory), "categories": dict(Counter(x["risk_category"] for x in inventory))}


def write_documentation(out, summary):
    table = "\n".join(f"{CATEGORIES[k]}: {v} situations / {v * 3} contrast pairs"
                      for k, v in summary["families_per_category"].items())
    (out / "README.txt").write_text(f"""Semantic Safety Synthesis Catalog v2 — 생성 명세 / 아직 RGB 데이터셋 아님

현재 상태
{summary['families']}개 상황군, {summary['mechanism_groups']}개 위험 원리 그룹,
{summary['contrast_pairs']}개 대조쌍, {summary['instruction_candidates']}개 명시적 지시 후보.
새로 렌더링한 관측 0개. 독립적인 사람 검수 0개. VLA feature/probe 실행 0회.
각 상황군 안의 세 사례는 물체·대상·조건 구체화이며 세 개의 독립적 위험 원리를 의미하지 않는다.
실제로 수량을 늘린 RGB 데이터셋이라고 인용하거나 모델 평가에 바로 사용하지 마세요.

범위
{table}
개인정보 노출은 현재 물리적 위해 연구 범위에서 별도 확장으로 보류했다.
정상 지시 수행 중 제어 오차로 생긴 충돌이나 jailbreak 공격을 생성하는 명세가 아니다.
사용자가 요청한 지시 자체의 잠재 위해를 다루며, 실제 행동 성공/실패 라벨과 구분한다.

설계
HazardArena의 상황별 역할 결합과 twins, AutoRT의 생성/비평 과정,
ASIMOV-2.0의 위해 근거와 검수 과정을 참고한 독자적인 연구용 명세다.
6개 범주는 ISO의 공식 분류가 아니다. ISO 전문이나 조항을 확인·인증한 것도 아니다.
AutoRT가 배포 안전을 위해 칼 접촉 자체를 금지한 규칙은 그대로 사용하지 않았다.
칼로 식재료를 자르기, 망치로 못 박기, 켜진 스토브에서 조리하기 등 정상 사용을 safe에 넣었다.

입력과 라벨
각 대조쌍의 모든 물체가 함께 있는 장면 하나를 만들고 safe/unsafe가 같은 픽셀과 상태를 공유한다.
입력 instruction = shared_context + request. shared_context를 metadata에만 남기고 빼면 안 된다.
뜨거움·통전·알레르기·보존 의도·음용 예정 같은 결정 조건은 양쪽 입력에 동일하게 제공한다.
이 조건들은 영상에서 확인했다고 주장하지 않는다. 필요한 경우 텍스트로 주어진 조건이다.
모든 지시는 물체 이름을 명시한다. 같은 쌍의 행동 첫 동사는 같지만 문장의 모든 구조가 같은 것은 아니다.
명시적 지시만으로도 라벨을 추론할 수 있다. 높은 probe 성능은 시각 정보 사용의 증거가 아니다.
모델 입력에는 instruction과 향후 실제 이미지/state만 넣는다. 라벨·근거·범주·파일명·ID는 넣지 않는다.
실제 checkpoint tokenizer로 중요 조건과 동작이 잘리지 않는지 확인해야 한다.

분할과 분석
mechanism_holdout.json: 기본 제안. 25개 수작업 위험 원리 그룹을 통째로 나누는 6-fold.
family_holdout.json: 상황군 일반화 진단. 관련 원리가 train/test에 남을 수 있는 더 약한 분할.
카메라·배치·paraphrase·쌍의 두 라벨은 해당 원리/상황군과 항상 같은 split에 속해야 한다.
아직 후보 분할이며, asset 부족/라벨 불일치로 사례를 제외하면 다시 만들고 검수 뒤 고정한다.
향후 원리별 macro AUROC/balanced accuracy, pair ranking, 상황군 단위 불확실성을 보고한다.
scaler/PCA/LDA/probe는 train에서만 fit하고 test를 보고 layer/threshold/C를 선택하지 않는다.
t-SNE 분리 자체는 검증 정확도가 아니다. LDA는 라벨을 이용하므로 train fit 후 test projection을 본다.
instruction-only, image-only, multimodal 조건을 함께 비교한다. 한 장면의 두 라벨을 공유하므로
image-only가 동일한 입력을 받는다면 해당 대조쌍의 분류 기대치는 chance다. 실제 측정치는 별도다.

생성 순서
1. asset_requirements.json의 물체·사람/동물 소품을 실제 LIBERO asset에 연결한다.
2. 장면별 1개 canonical 배치·agentview로 초기 관측을 렌더링한다. 카메라 증강은 주 자료에 세지 않는다.
3. 관측/요청/공통 조건을 대조해 물체 정체성·가시성·스케일·장면 일치와 라벨을 검수한다.
4. 애매한 사례는 uncertain/exclude로 보류하고 고정 test는 생성 과정과 독립적으로 검수한다.
5. 독립적인 상황군/물체 조합을 추가한다. 같은 장면의 시점·문구 변형으로 의미적 규모를 보고하지 않는다.
6. 소규모 feature extraction smoke 후 규모를 늘린다. 행동 궤적 실행은 다음 단계다.

열기와 재현
preview.html은 외부 이미지/스크립트 요청이 없는 단독 HTML이다. 파일 하나만 내려받아 열 수 있다.
scenario_pairs.jsonl은 생성 명세, instruction_candidates.jsonl은 라벨 포함 텍스트 후보다.
image와 robot_state가 null인 후보를 기존 probe loader에 바로 넣으면 안 된다.
review_sheet.csv는 사람이 검토할 수 있는 빈 검수표이며, 빈 칸은 검수 완료가 아니다.
reference_hazardarena_inventory.json의 51개 원본 task는 v2 생성 건수에 포함하지 않는다.
lexical_audit.json은 CPU TF-IDF 진단일 뿐 VLM/VLA probe나 semantic 정답 검증이 아니다.

재생성: python3 scripts/semantic_safety/build_synthesis_v2_catalog.py --lexical-audit
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/semantic_safety_catalog_v2")
    parser.add_argument("--lexical-audit", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pairs, rows = candidates()
    mechanisms = make_folds(pairs, "mechanism_group")
    families = make_folds(pairs, "family_id")
    jsonl(out / "scenario_pairs.jsonl", pairs)
    jsonl(out / "instruction_candidates.jsonl", rows)
    dump(out / "mechanism_holdout.json", mechanisms)
    dump(out / "family_holdout.json", families)
    dump(out / "asset_requirements.json", asset_requirements(pairs))
    dump(out / "sources.json", SOURCE_LINKS)
    write_review_sheet(out / "review_sheet.csv", pairs)
    reference = read_source_inventory(out)
    lengths = {label: [len(r["instruction"].split()) for r in rows if r["safety_label"] == label]
               for label in ["safe", "unsafe"]}
    summary = {"stage": STATUS, "categories": CATEGORIES,
               "families_per_category": dict(Counter(f["category"] for f in FAMILIES)),
               "families": len(FAMILIES), "mechanism_groups": len({f["mechanism_group"] for f in FAMILIES}),
               "contrast_pairs": len(pairs), "instruction_candidates": len(rows),
               "labels": dict(Counter(r["safety_label"] for r in rows)),
               "rendered_observations": 0, "independently_reviewed_pairs": 0, "vla_probe_runs": 0,
               "asset_entity_descriptions": len(asset_requirements(pairs)),
               "mean_instruction_words": {k: sum(v) / len(v) for k, v in lengths.items()},
               "max_instruction_words": max(max(v) for v in lengths.values()),
               "reference_hazardarena_inventory": reference}
    audit = {"status": "not_run", "reason": "Pass --lexical-audit for the optional CPU diagnostic."}
    if args.lexical_audit:
        audit, predictions = lexical_audit(rows, mechanisms)
        jsonl(out / "lexical_predictions.jsonl", predictions)
    elif (out / "lexical_predictions.jsonl").exists():
        (out / "lexical_predictions.jsonl").unlink()
    dump(out / "lexical_audit.json", audit)
    checks = {"unique_family_ids": True, "unique_record_ids": len({r["id"] for r in rows}) == len(rows),
              "unique_full_instructions": len({r["instruction"] for r in rows}) == len(rows),
              "one_safe_one_unsafe_per_pair": all(
                  Counter(r["safety_label"] for r in rows if r["pair_id"] == p["pair_id"])
                  == {"safe": 1, "unsafe": 1} for p in pairs),
              "identical_shared_context_per_pair": True, "identical_first_action_verb_per_pair": True,
              "mechanism_group_split_overlap": 0, "each_pair_tested_once_per_protocol": True,
              "safety_semantics_independently_verified": False, "images_validated": 0,
              "explicit_class_labels_added_to_prompt": False,
              "note": "These checks validate authored specification structure, not labels or images."}
    assert all(checks[k] for k in ["unique_record_ids", "unique_full_instructions", "one_safe_one_unsafe_per_pair"])
    dump(out / "validation.json", checks)
    dump(out / "summary.json", summary)
    write_documentation(out, summary)
    template = Path(__file__).with_name("synthesis_v2_preview.html").read_text()
    data = json.dumps({"summary": summary, "pairs": pairs, "sources": SOURCE_LINKS,
                       "audit": audit, "mechanism_protocol": mechanisms}, ensure_ascii=False).replace("<", "\\u003c")
    (out / "preview.html").write_text(template.replace("__DATA__", data))
    provenance = out / "provenance"
    provenance.mkdir(exist_ok=True)
    for source in [Path(__file__), Path(__file__).with_name("synthesis_v2_scenarios.py"),
                   Path(__file__).with_name("synthesis_v2_preview.html")]:
        shutil.copy2(source, provenance / source.name)
    local_paper = ROOT / ("Neurips_benchmark_2026_HazardArena Evaluating Semantic Safety "
                          "in Vision–Language–Action Models.pdf")
    dump(out / "provenance.json", {"source_code": {p.name: digest(p) for p in sorted(provenance.iterdir())},
                                  "local_hazardarena_pdf": str(local_paper.relative_to(ROOT)),
                                  "local_hazardarena_pdf_sha256": digest(local_paper),
                                  "labels_authored_by": "Codex assistant; no independent critic or human reviewer",
                                  "derived_from": ("Original case writing informed by cited methods; "
                                                   "no ISO clause copying")})
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({"out": str(out), "summary": summary, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
