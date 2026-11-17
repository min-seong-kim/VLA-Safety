"""Audit frozen observations and package the offline human review; never fit models."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

from contrast_dataset_v1 import CATEGORIES, OUT, write_jsonl


def load(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    rows=load(OUT/"conditions.jsonl");groups=load(OUT/"contrast_sets.jsonl")
    scene_rows=load(OUT/"scenes.jsonl")
    reports={s["scene_id"]:json.loads((OUT/"observations"/s["scene_id"]/"validation.json").read_text()) for s in scene_rows}
    assert all(r.get("rendered") for r in reports.values()),"Unrendered scene"
    assert len({r["condition_id"] for r in rows})==len(rows)
    assert all(s["task_objects"]==2 for s in scene_rows)
    frequencies=defaultdict(Counter)
    # Count semantic conditions, never A/B duplicates, image layouts, or repeated controls.
    for r in rows:
        for identity in {r[k]["identity"] for k in ["actor","target"]}:
            frequencies[identity][r["label_name"]]+=1
    identity_names={r[role]["identity"]:r[role]["name_ko"] for r in rows for role in ["actor","target"]}
    identity_names.update(dict(bread="빵 (신선·곰팡이)",apple="사과 (신선·곰팡이)",banana="바나나 (신선·곰팡이)",
                              cookie="쿠키 (신선·곰팡이)",cheese="치즈 (신선·곰팡이)",rice="밥 (신선·곰팡이)",
                              cloth="면포 (청결·오염)",fan="팬 (작동·정지)",hotplate="가열판 (가열·상온)",
                              iron="다리미 (가열·상온)",hotmetal="금속 누름판 (온도 구분)",water="물병 (용도 구분)",kettle="주전자 (수온 구분)",
                              cable="전선 (통전 상태 구분)",gas="금속 가스통 계열 (압력 상태 구분)",
                              aerosol="에어로졸 금속통 계열 (압력 상태 구분)",vessel="압력 용기 계열 (압력 상태 구분)"))
    object_counts=[dict(identity=k,identity_name_ko=identity_names[k],safe=v["safe"],unsafe=v["unsafe"]) for k,v in sorted(frequencies.items())]
    one_sided={k for k,v in frequencies.items() if not v["safe"] or not v["unsafe"]}
    indexed={r["condition_id"]:r for r in rows}
    assert set(indexed)=={cid for g in groups for cid in g["condition_ids"]}
    collision_inputs=defaultdict(list)
    for r in rows:
        report=reports[r["scene_id"]]
        r["observation"]=report
        r["one_sided_objects"]=sorted({r[k]["identity"] for k in ["actor","target"]}&one_sided)
        r["visual_track_available"]=bool(r["visual_instruction_en"])
        if r["visual_instruction_en"]:
            key=(report["images"]["agentview"]["sha256"],report["images"]["wrist"]["sha256"],
                 report["robot_state_sha256"],r["visual_instruction_en"])
            collision_inputs[key].append(r)
    visual_collisions=[]
    for candidates in collision_inputs.values():
        if len({r["label"] for r in candidates})>1:
            visual_collisions.append([r["condition_id"] for r in candidates])
            for r in candidates:
                r["visual_track_available"]=False
                r["visual_track_exclusion_ko"]="반대 label 조건과 실제 카메라 입력·지시가 동일하여 B 트랙 보류; A의 사실 명시 조건은 유지"
                r["review_flags"].append("opposite_label_identical_visual_input")
    items=[]
    for r in rows:
        for track in ["A","B"]:
            if track=="B" and not r["visual_track_available"]:continue
            instruction=r["instruction_en"] if track=="A" else r["visual_instruction_en"]
            assert not re.search(r"\b(safe|unsafe|dangerous|harmless|safely)\b",instruction,re.I),instruction
            items.append(dict(item_id=r["condition_id"]+"_"+track,condition_id=r["condition_id"],
                              track=track,label=r["label"],set_ids=r["set_ids"],
                              instruction=instruction,scene_id=r["scene_id"],
                              observation_path=f"observations/{r['scene_id']}/initial_observation.npz",
                              semantic_status="candidate_pending_user_review"))
    contradictory=[]
    inputs=defaultdict(list)
    for item in items:
        report=reports[item["scene_id"]]
        key=(report["images"]["agentview"]["sha256"],report["images"]["wrist"]["sha256"],
             report["robot_state_sha256"],item["instruction"])
        inputs[key].append(item)
    for values in inputs.values():
        if len({r["label"] for r in values})>1:contradictory.append([r["item_id"] for r in values])
    assert not contradictory,contradictory
    track_bias={}
    for track in ["A","B"]:
        selected={item["condition_id"] for item in items if item["track"]==track}
        freq=defaultdict(Counter)
        for cid in selected:
            row=indexed[cid]
            for identity in {row[k]["identity"] for k in ["actor","target"]}:
                freq[identity][row["label_name"]]+=1
        track_bias[track]=[dict(identity=k,safe=v["safe"],unsafe=v["unsafe"]) for k,v in sorted(freq.items())]
    visual_single_class=[]
    for group in groups:
        selected=[indexed[cid] for cid in group["condition_ids"] if indexed[cid]["visual_track_available"]]
        if selected and len({r["label"] for r in selected})<2:visual_single_class.append(group["set_id"])
    summary=dict(contrast_sets=len(groups),semantic_conditions=len(rows),physical_scenes=len(reports),
                 track_A=sum(r["track"]=="A" for r in items),track_B=sum(r["track"]=="B" for r in items),
                 safe_conditions=sum(not r["label"] for r in rows),unsafe_conditions=sum(r["label"] for r in rows),
                 model_inputs=len(items),categories=len(CATEGORIES),
                 category_conditions=dict(Counter(r["category"] for r in rows)))
    audit=dict(rendered=len(reports),technical_pass=sum(r["technical_pass"] for r in reports.values()),
               technical_review_needed=[k for k,r in reports.items() if not r["technical_pass"]],
               opposite_label_identical_inputs=contradictory,visual_candidates_withheld=visual_collisions,
               semantic_labels_validated=False,object_identities=len(frequencies),
               both_label_identities=len(frequencies)-len(one_sided),one_sided_identities=len(one_sided),
               object_label_counts=object_counts,shared_controls=sum(len(r["set_ids"])>1 for r in rows),
               object_label_counts_by_track=track_bias,visual_sets_with_only_one_label=visual_single_class,
               training_executed=False,probe_fitting_executed=False,
               robot_state_hashes=len({r["robot_state_sha256"] for r in reports.values()}),
               split_note="Shared controls are deduplicated. Purge shared condition IDs from train/dev when testing related contrast sets.")
    fingerprint=hashlib.sha256((OUT/"conditions.jsonl").read_bytes())
    for scene_id,report in sorted(reports.items()):
        fingerprint.update((scene_id+json.dumps(report["images"],sort_keys=True)).encode())
    payload=dict(conditions=rows,groups=groups,categories=CATEGORIES,summary=summary,audit=audit,
                 dataset_sha256=fingerprint.hexdigest())
    template=Path(__file__).with_name("contrast_review_v1.html").read_text()
    encoded=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")
    (OUT/"preview.html").write_text(template.replace("__PAYLOAD__",encoded))
    write_jsonl(OUT/"review_conditions.jsonl",rows)
    write_jsonl(OUT/"items.jsonl",items)
    for name,value in [("summary.json",summary),("validation.json",audit)]:
        (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    with (OUT/"review_sheet.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["condition_id","set_ids","category","actor","target","label_candidate","instruction_A","instruction_B","B_available","rationale_ko","required_visual_cue_ko","flags","agentview","review_status","feedback"])
        for r in rows:
            writer.writerow([r["condition_id"],";".join(r["set_ids"]),r["category_ko"],r["actor"]["name_ko"],r["target"]["name_ko"],r["label_name"],
                             r["instruction_en"],r["visual_instruction_en"] or "",r["visual_track_available"],r["rationale_ko"],
                             r["visual_evidence_ko"] or "",";".join(r["review_flags"]),r["observation"]["images"]["agentview"]["path"],"pending",""])
    readme=["# 안전·위험 대조 데이터셋 검토 v1","",f"대조 묶음 {len(groups)}개, 중복 제거 의미 조건 {len(rows)}개, 실제 LIBERO 초기 장면 {len(reports)}개.",
            f"A 트랙 {summary['track_A']}개, B 트랙 후보 {summary['track_B']}개. 두 트랙은 같은 조건의 입력 방식이며 독립 상황으로 중복 집계하지 않는다.","",
            "`preview.html`을 브라우저에서 열어 지시문·실제 관찰·판단 근거를 검토한다. 조건별 유지/수정/제외와 메모를 저장하고 JSON으로 내보낼 수 있다.","",
            "Label은 정상적인 지시 이행에 따른 요청 수준의 위해 여부다. 판정 근거와 한국어 번역은 모델 입력에서 제외한다. A는 필요한 사실을 지시에 포함하고, B는 지정한 시각 단서를 관찰과 결합해야 한다.","",
            "## 범위와 검토할 점","",
            "- 조작 물체와 대상은 두 개다. 칼·톱 받침은 별도 고정 fixture다.",
            "- 초기 장면만 렌더링했다. 시연·정책 rollout·모델 학습·representation 분석은 수행하지 않았다.",
            "- 인체·생물은 기존 mesh 또는 도식적 proxy다. 실제 생체라는 의미와 시각적 정체성을 사람이 검토해야 한다.",
            "- B 트랙은 시각적 인식이 검증된 test set이 아니라 후보 목록이다. 반대 label과 입력이 같은 B 후보는 보류한다.",
            "- 모든 물체의 label 비율이 동일하지는 않다. `validation.json`의 물체별 출현 빈도를 함께 검토한다.",
            "- 공유 조건은 한 번만 저장했다. 이후 대조 묶음 단위 test에서는 해당 조건을 train/dev에서 제거해야 한다.",
            "- 의미 조건은 배치·문장 변형 수와 구분한다. 기존 v5 데이터는 변경하지 않았다.","",
            "## 재생성","","```bash",
            "envs/libero-render/bin/python scripts/semantic_safety/build_contrast_observations_v1.py prepare",
            "# Use the repository's NVIDIA EGL configuration when rendering.",
            "envs/libero-render/bin/python scripts/semantic_safety/build_contrast_observations_v1.py render --workers 4",
            "envs/representation-probe/bin/python scripts/semantic_safety/report_contrast_dataset_v1.py","```",""]
    (OUT/"README.md").write_text("\n".join(readme))
    # The review archive works offline; full simulation assets remain in the dataset folder.
    with zipfile.ZipFile(OUT/"observation_review.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for name in ["preview.html","README.md","conditions.jsonl","contrast_sets.jsonl","objects.jsonl","items.jsonl","review_sheet.csv","summary.json","validation.json","protocol.json"]:
            archive.write(OUT/name,name)
        for scene_id in reports:
            for name in ["agentview.png","wrist.png","overview.png","agentview.webp","wrist.webp","overview.webp","agentview_224.png","wrist_224.png"]:
                path=OUT/"observations"/scene_id/name
                archive.write(path,str(path.relative_to(OUT)))
    print(json.dumps(dict(summary=summary,audit={k:v for k,v in audit.items() if k not in {"object_label_counts","object_label_counts_by_track"}}),ensure_ascii=False))


if __name__=="__main__":main()
