"""Build a human-review catalog of authored semantic-safety setup proposals."""

import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "doc/semantic_safety_expansion_sources"
OUT = ROOT / "data/semantic_safety_catalog_v3"
CATEGORIES = {
    "body": "신체 위해·취약한 수신자",
    "food": "식품·섭취",
    "chemical": "화학물질·오염",
    "living_environment": "동물·식물·생활 생태계",
    "hygiene": "위생·의료용품 오염",
    "fire_heat": "화재·열·압력",
    "electrical": "전기·전자",
    "property": "재산·기록·정밀 부품 손상",
    "mechanical_access": "기계적 기능·접근·배수 방해",
}
SIDE_FIELDS = ["actor", "target", "actor_material", "actor_state", "target_state", "instruction_en"]
REFERENCES = [
    {"title": "CPSC: Button Cell and Coin Batteries", "url": "https://www.cpsc.gov/Safety-Education/Safety-Education-Centers/Button-Cell-Coin-Battery-Information-Center", "scope": "Small battery ingestion hazard; not validation of every proposed task."},
    {"title": "CDC: Cleaning and Disinfecting with Bleach", "url": "https://www.cdc.gov/hygiene/about/cleaning-and-disinfecting-with-bleach.html", "scope": "General household-cleaner incompatibility; no reaction recipes are specified."},
    {"title": "FDA: Dangers for Pets", "url": "https://www.fda.gov/animal-veterinary/animal-health-literacy/dangers-pets", "scope": "Examples of species-specific food and household exposures, not universal toxicity labels."},
    {"title": "FDA: Lovely Lilies and Curious Cats", "url": "https://www.fda.gov/animal-veterinary/animal-health-literacy/lovely-lilies-and-curious-cats-dangerous-combination", "scope": "Lily exposure in cats; not a statement about every plant or animal."},
    {"title": "FDA: Food Allergies", "url": "https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/food-allergies", "scope": "Recipient-specific allergens and cross-contact."},
    {"title": "Prior local inventory", "path": "doc/Semantic_Safety_Dataset_Inventory.md", "scope": "Previous authored scenario families and scope distinctions."},
]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalized(value):
    return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()


def first_verb(value):
    words = value.lower().split()
    while words and words[0] in {"fully", "gently", "slowly", "carefully", "firmly", "lightly", "briefly"}:
        words.pop(0)
    return words[0].strip(".,")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / name for name in ["body.jsonl", "food_chemical_environment.jsonl", "fire_electric_property.jsonl"]]
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["source_file"] = str(path.relative_to(ROOT))
            rows.append(row)
    assert len({r["local_id"] for r in rows}) == len(rows)
    prior = [json.loads(x) for x in (ROOT / "data/semantic_safety_transport_v6/items.jsonl").read_text().splitlines()]
    prior_requests = {normalized(x["instruction"]) for x in prior}
    exact = defaultdict(list)
    setups = defaultdict(list)
    records = []
    for row in rows:
        assert row["category"] in CATEGORIES, row["local_id"]
        for side in ["safe", "unsafe"]:
            assert all(isinstance(row[side][k], str) and row[side][k].strip() for k in SIDE_FIELDS), row["local_id"]
        assert row["safe"]["instruction_en"] != row["unsafe"]["instruction_en"] or row["safe"] != row["unsafe"]
        row["pair_id"] = "v3_" + row["local_id"]
        row["category_ko"] = CATEGORIES[row["category"]]
        row["review_status"] = "pending"
        row["review_notes"] = ""
        row["image"] = None
        row["trajectory"] = None
        row["label_status"] = "authored_candidate_not_human_validated"
        row["scene_status"] = "not_rendered"
        row["scoring_status"] = "deferred"
        row["verbs"] = {s: first_verb(row[s]["instruction_en"]) for s in ["safe", "unsafe"]}
        row["same_verb"] = row["verbs"]["safe"] == row["verbs"]["unsafe"]
        row["prior_v6_exact_request_sides"] = [s for s in ["safe", "unsafe"] if normalized(row[s]["instruction_en"]) in prior_requests]
        row["flags"] = []
        if not row["same_verb"]:
            row["flags"].append("verb_mismatch")
        if row["observability"] != "visual":
            row["flags"].append("context_required")
        if row["novelty"] == "instruction_variant":
            row["flags"].append("instruction_variant_not_new_setup")
        if row["prior_v6_exact_request_sides"]:
            row["flags"].append("prior_v6_request_overlap")
        if any(re.search(r"\b(safely|unsafe|dangerous|harmless)\b", row[s]["instruction_en"], re.I) for s in ["safe", "unsafe"]):
            row["flags"].append("explicit_label_word")
        exact_key = tuple(normalized(row[s]["instruction_en"]) for s in ["safe", "unsafe"])
        exact[exact_key].append(row["pair_id"])
        setup_key = tuple(normalized(row[s][k]) for s in ["safe", "unsafe"] for k in SIDE_FIELDS[:-1])
        setups[setup_key].append(row["pair_id"])
        row["setup_group"] = hashlib.sha256(json.dumps(setup_key).encode()).hexdigest()[:12]
        row["scene_pair_plan"] = "same_scene_instruction_candidate" if row["pair_axis"] == "instruction" else "paired_scene_candidate"
        for side in ["safe", "unsafe"]:
            request = row[side]["instruction_en"]
            context = row["shared_context_en"].strip()
            scene = row[side]
            state_context = (f"Object: {scene['actor']}. Material: {scene['actor_material']}. "
                             f"Object state: {scene['actor_state']}. Target: {scene['target']}. "
                             f"Target state: {scene['target_state']}.")
            scene["state_context_en"] = state_context
            records.append(dict(id=row["pair_id"] + "_" + side, pair_id=row["pair_id"],
                                candidate_label=side, instruction=" ".join([context, state_context, request]).strip(),
                                request=request, shared_context=context, state_context=state_context,
                                input_mode="explicit_state_text_draft_not_visual_only", image=None,
                                scene_spec=row[side], status="pending_human_review_not_rendered"))
        assert records[-2]["instruction"] != records[-1]["instruction"], row["pair_id"]
    duplicate_requests = [ids for ids in exact.values() if len(ids) > 1]
    repeated_setups = [ids for ids in setups.values() if len(ids) > 1]
    for row in rows:
        if any(row["pair_id"] in group for group in duplicate_requests):
            row["flags"].append("duplicate_request_pair_review_context")
        if any(row["pair_id"] in group for group in repeated_setups):
            row["flags"].append("shared_physical_setup")
    summary = dict(candidate_pairs=len(rows), instruction_records=len(records),
                   physical_setup_signatures=len(setups), authored_families=len({r["family_id"] for r in rows}),
                   authored_mechanism_groups=len({r["mechanism_group"] for r in rows}),
                   by_category=dict(Counter(r["category"] for r in rows)),
                   unsafe_verbs=dict(Counter(r["verbs"]["unsafe"] for r in rows)),
                   novelty=dict(Counter(r["novelty"] for r in rows)),
                   same_verb_pairs=sum(r["same_verb"] for r in rows),
                   identical_request_scene_pairs=sum(r["safe"]["instruction_en"] == r["unsafe"]["instruction_en"] for r in rows),
                   flags=dict(Counter(f for r in rows for f in r["flags"])),
                   exact_request_duplicate_groups=duplicate_requests, shared_setup_groups=repeated_setups,
                   images_rendered=0, trajectories_collected=0, human_reviewed=0,
                   note="Counts are authored proposals, not independent hazard principles or validated data. Setup signatures are only a syntactic duplicate check; semantic deduplication remains for review.")
    for name, data in [("scenario_pairs.jsonl", rows), ("instruction_candidates.jsonl", records)]:
        (OUT / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    save_json(OUT / "summary.json", summary)
    save_json(OUT / "references.json", REFERENCES)
    flat = []
    for r in rows:
        flat.append(dict(pair_id=r["pair_id"], category=r["category_ko"], family=r["family_id"],
                         mechanism_group=r["mechanism_group"], title=r["title_ko"], pair_axis=r["pair_axis"],
                         shared_context=r["shared_context_en"],
                         safe_actor=r["safe"]["actor"], safe_target=r["safe"]["target"], safe_instruction=r["safe"]["instruction_en"],
                         unsafe_actor=r["unsafe"]["actor"], unsafe_target=r["unsafe"]["target"], unsafe_instruction=r["unsafe"]["instruction_en"],
                         safe_state_context=r["safe"]["state_context_en"], unsafe_state_context=r["unsafe"]["state_context_en"],
                         changed_factor=r["changed_factor_ko"], rationale=r["rationale_ko"],
                         required_evidence=r["required_evidence_ko"], ambiguity=r["ambiguity_ko"],
                         novelty=r["novelty"], flags=";".join(r["flags"]), review_status="pending", review_notes=""))
    write_csv(OUT / "review_sheet.csv", flat)
    lines = ["# Semantic Safety 셋업 확장 후보 v3", "",
             f"검토 후보 **{len(rows)}쌍 / {len(records)}개 지시**. 실제 이미지·trajectory는 0개이며 모든 label은 작성자의 제안이다.", "",
             "[검색·검토 화면](../data/semantic_safety_catalog_v3/preview.html) · [CSV 검토표](../data/semantic_safety_catalog_v3/review_sheet.csv)", "",
             "## 확장 방향", "",
             "기존 여섯 범주에 동물·식물, 위생·의료용품 오염, 끼임·기계 기능·접근 방해를 더했다. "
             "물체 정체성, 대상, 재질, 작동 상태와 동작을 바꿔 후보를 작성했다. 채점 방식과 현재 모델의 동작 가능성은 선정 조건으로 사용하지 않았다.", "",
             "원본 v6 및 학습된 모델은 유지한다. 이 파일은 렌더링된 데이터셋이나 완료된 평가가 아니다. "
             "카테고리·family·mechanism 수는 작성 편의상의 분류이며 독립적인 위험 원리 수를 뜻하지 않는다.", "",
             "## 읽는 방법", "",
             "공통 조건은 안전·위험 양쪽에 똑같이 붙인다. 조건이 필요한 경우 모델 입력에서 빼면 label 근거가 사라질 수 있다. "
             "actor/target의 재질·상태는 앞으로 실제 장면이나 지시에서 확인 가능하게 구현해야 한다. "
             "현재 관측이 없으므로 모든 후보의 시각적 판별 가능성은 아직 미검증이다.", "",
             "특히 통전 여부 등 상태를 바꾸는 대조에서는 요청 문장이 같을 수 있다. "
             "`instruction_candidates.jsonl`의 full instruction에는 공통 조건뿐 아니라 각 장면의 재질·상태 설명도 붙였다. "
             "이 입력은 **명시적 상태 텍스트 버전의 초안**이며, 이미지에서 상태를 추론하는 실험의 입력과 구분해야 한다. "
             "실제 모델 입력을 만들 때는 이름·상태 중복과 tokenizer 길이를 다시 정리한다.", "",
             "도구와 대상 두 역할을 중심으로 쓰되 일부 후보는 사람·고정 설비·작동 중인 기계 같은 배경 조건이 필요하다. "
             "같은 장면에서 지시로 대상을 바꾸는 후보는 도구 1개와 대상 2개 이상이 필요할 수 있다. "
             "safe는 명시한 조건에서 의도된 사용이라는 후보 label이며 모든 환경에서 무조건 안전하다는 뜻이 아니다.", "",
             "단순 동사 교체나 같은 물체 조합의 반복은 별도 표시한다. 정규화된 요청 문자열 중복과 물체·상태 signature도 검사한다. "
             "자동 검사는 의미적 신규성·위해의 확실성·label 정확성을 보증하지 않는다. 최종 split은 사람 검토와 중복 통합 후 정한다.", "",
             "## 범주별 수", "", "| 범주 | 대조쌍 |", "|---|---:|"]
    for category, count in summary["by_category"].items():
        lines.append(f"| {CATEGORIES[category]} | {count} |")
    lines += ["", "## 전체 후보", ""]
    for category, title in CATEGORIES.items():
        lines += [f"### {title}", ""]
        for r in rows:
            if r["category"] != category:
                continue
            lines += [f"#### {r['pair_id']} — {r['title_ko']}", "",
                      f"- 분류: `{r['family_id']}` / `{r['mechanism_group']}` · 변화: {r['changed_factor_ko']}",
                      f"- 공통 조건: {r['shared_context_en'] or '(별도 텍스트 조건 없음; 장면 증거 필요)'}",
                      f"- **안전 후보:** {r['safe']['instruction_en']}",
                      f"- **위험 후보:** {r['unsafe']['instruction_en']}",
                      f"- 안전 장면 상태: {r['safe']['state_context_en']}",
                      f"- 위험 장면 상태: {r['unsafe']['state_context_en']}",
                      f"- 위해 근거: {r['rationale_ko']}",
                      f"- 필요한 관측/조건: {r['required_evidence_ko']}",
                      f"- 검토할 점: {r['ambiguity_ko'] or '대조쌍 label 및 장면 구현을 사용자 검토로 확정.'}", ""]
    (ROOT / "doc/Semantic_Safety_Setup_Expansion_v3.md").write_text("\n".join(lines))
    template = (ROOT / "scripts/semantic_safety/expansion_v3_preview.html").read_text()
    payload = json.dumps(dict(rows=rows, summary=summary, categories=CATEGORIES), ensure_ascii=False).replace("<", "\\u003c")
    (OUT / "preview.html").write_text(template.replace("__PAYLOAD__", payload))
    (OUT / "README.md").write_text(
        "# Semantic Safety 셋업 확장 후보 v3\n\n"
        f"{len(rows)}쌍, {len(records)}개 지시의 **검토용 생성 명세**입니다. 실제 관측·trajectory는 아직 없습니다.\n\n"
        "`preview.html`을 브라우저에서 열면 검색·범주·동사 필터, 승인/수정/보류/제외 및 메모 입력을 사용할 수 있습니다. "
        "검토 기록은 해당 브라우저에만 저장되므로 JSON 내보내기로 보관하세요. 다른 브라우저에서는 JSON 가져오기로 이어갈 수 있습니다. "
        "전체 목록은 `review_sheet.csv`, 구조화한 원본은 `scenario_pairs.jsonl`입니다.\n\n"
        "공통 조건은 양쪽 지시에 동일하게 붙입니다. 필요 조건을 metadata에만 숨긴 채 평가하면 안 됩니다. "
        "공통 조건 및 각 장면의 재질·상태 설명을 합친 명시적 텍스트 초안은 `instruction_candidates.jsonl`에 있습니다. "
        "시각 추론용 입력과 구분해야 하며, image가 null이므로 평가용 loader에 바로 넣지 마세요.\n\n"
        "instruction_variant, 같은 physical setup, 기존 v6 요청과 정확한 중복은 별도 표시합니다. "
        "새로운 의미적 사례인지, 위해가 명확한지, safe 동작이 정상 사용인지 검토해 최종 채택하세요.\n\n"
        "재생성: `openpi/.venv/bin/python scripts/semantic_safety/build_expansion_v3_catalog.py`\n"
        "빌드 스크립트는 파생 목록을 재생성합니다. 사람 검토 결과 JSON은 별도로 보관하며 source 목록과 자동 병합하지 않습니다.\n")
    provenance = OUT / "provenance"
    provenance.mkdir(exist_ok=True)
    for path in paths + [Path(__file__), ROOT / "scripts/semantic_safety/expansion_v3_preview.html"]:
        shutil.copy2(path, provenance / path.name)
    save_json(OUT / "structural_audit.json", dict(passed=True, pairs=len(rows), unique_ids=True,
                                                both_sides_present=True, paired_full_inputs_distinct=True,
                                                human_semantic_validation=False,
                                                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    with zipfile.ZipFile(OUT / "review_bundle.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ["preview.html", "README.md", "scenario_pairs.jsonl", "instruction_candidates.jsonl", "review_sheet.csv", "summary.json", "references.json"]:
            z.write(OUT / name, name)
        z.writestr("Full_Catalog.md", (ROOT / "doc/Semantic_Safety_Setup_Expansion_v3.md").read_text().replace("../data/semantic_safety_catalog_v3/", ""))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
