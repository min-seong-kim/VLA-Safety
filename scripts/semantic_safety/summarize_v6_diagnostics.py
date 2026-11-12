"""Summarize paired v6 motor diagnostics without interpreting source safety labels as outcomes."""

from collections import Counter
import csv
import json
from pathlib import Path
import zipfile

from rollout_common import ROOT, write_json
from semantic_pairs_v6 import DATA, sha256
from semantic_v6_diagnostics import ARMS, BASELINE, RUN, items

LABELS = {
    "original": "원래 지시",
    "pickup_name": "물체 이름으로 집기",
    "pickup_position": "좌우 위치로 집기",
    "transport_name": "물체 이름으로 단순 운반",
    "transport_position": "좌우 위치로 단순 운반",
    "held_transport_name": "검증된 파지에서 단순 운반",
}
KEYS = ["ever_grasped", "ever_lifted_4cm", "sustained_lift_10_steps", "transport_success",
        "xy_only_transport_success", "xy_only_sustained_success"]


def counts(rows):
    return dict(n=len(rows), **{key:sum(bool(r[key]) for r in rows) for key in KEYS})


def comparison(left, right, metric):
    a = {r["item_id"]:r for r in left}; b = {r["item_id"]:r for r in right}
    common = sorted(a.keys() & b.keys())
    changed = {"gained":[], "lost":[], "both":[], "neither":[]}
    for item_id in common:
        first, second = bool(a[item_id][metric]), bool(b[item_id][metric])
        key = "both" if first and second else "gained" if second else "lost" if first else "neither"
        changed[key].append(item_id)
    return dict(metric=metric, n=len(common), left_success=sum(a[i][metric] for i in common),
                right_success=sum(b[i][metric] for i in common),
                transitions={k:len(v) for k,v in changed.items()}, item_ids=changed,
                independent_samples=False)


def main():
    protocol = json.loads((RUN/"protocol.json").read_text())
    assert sha256(DATA/"items.jsonl") == protocol["items_sha256"]
    assert sha256(DATA/"assets/manifest.json") == protocol["assets_sha256"]
    original = json.loads((RUN/"original_rescored.json").read_text())
    results = {"original":original}
    for arm in ARMS:
        folder = RUN/arm
        assert json.loads((folder/"completion.json").read_text())["complete"]
        assert json.loads((folder/"recording_audit.json").read_text())["passed"]
        results[arm] = [json.loads(line) for line in (folder/"episodes.jsonl").read_text().splitlines()]
    by_arm = {arm:counts(rows) for arm,rows in results.items()}
    pairs = {
        "pickup_position_vs_name": comparison(results["pickup_name"],results["pickup_position"],"ever_lifted_4cm"),
        "pickup_position_vs_name_sustained": comparison(results["pickup_name"],results["pickup_position"],"sustained_lift_10_steps"),
        "simple_transport_vs_original": comparison(original,results["transport_name"],"xy_only_transport_success"),
        "position_transport_vs_name": comparison(results["transport_name"],results["transport_position"],"xy_only_transport_success"),
        "verified_hold_vs_table": comparison(results["transport_name"],results["held_transport_name"],"xy_only_transport_success"),
        "verified_hold_vs_table_strict": comparison(results["transport_name"],results["held_transport_name"],"transport_success"),
    }
    item_map = {i["item_id"]:i for i in items()}
    actors = {}
    for actor in sorted({i["actor_asset"] for i in item_map.values()}):
        actors[actor] = {arm:counts([r for r in rows if item_map[r["item_id"]]["actor_asset"]==actor])
                         for arm,rows in results.items()}
    coverage = json.loads((RUN/"grasp_prefix_coverage.json").read_text())
    failed_prefixes = [json.loads((RUN/"grasp_prefixes"/r["item_id"]/"summary.json").read_text())
                       for r in coverage["results"] if not r["passed"]]
    summary = dict(protocol=protocol, by_arm=by_arm, comparisons=pairs, by_actor=actors,
                   prefix_coverage=dict(attempted=coverage["attempted"],accepted=coverage["accepted"],
                       failed_actor_counts=dict(Counter(r["actor_asset"] for r in failed_prefixes))),
                   new_policy_episodes=sum(len(results[a]) for a in ARMS), reused_original_episodes=len(original),
                   task_success=None, harm_observed=None, training_performed=False)
    write_json(RUN/"summary.json",summary)
    with (RUN/"arm_summary.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["arm","n",*KEYS])
        for arm,c in by_arm.items():writer.writerow([arm,c["n"],*[c[k] for k in KEYS]])
    with (RUN/"actor_summary.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["actor_asset","arm","n",*KEYS])
        for actor,arms in actors.items():
            for arm,c in arms.items():writer.writerow([actor,arm,c["n"],*[c[k] for k in KEYS]])
    videos = RUN/"videos";videos.mkdir(exist_ok=True)
    compact = []
    for arm,rows in results.items():
        target = BASELINE/"episodes" if arm=="original" else RUN/arm/"episodes"
        link = videos/arm
        if link.is_symlink(): assert link.resolve()==target.resolve()
        else: link.symlink_to(target, target_is_directory=True)
        for row in rows:
            compact.append({**{k:row.get(k) for k in ["item_id","pair_id","layout","task_is_unsafe","instruction",
                             "first_transport_step","first_xy_only_step",*KEYS]},
                            "arm":arm,"video":f"videos/{arm}/{row['item_id']}/views.mp4"})
    pair_catalog = [json.loads(line) for line in (DATA/"pairs.jsonl").read_text().splitlines()]
    payload = dict(labels=LABELS, summary=summary, episodes=compact,
                   pairs=[dict(pair_id=p["pair_id"],title_ko=p["title_ko"]) for p in pair_catalog])
    template = Path(__file__).with_name("semantic_v6_diagnostics_review.html").read_text()
    (RUN/"review.html").write_text(template.replace("__PAYLOAD__",json.dumps(payload,ensure_ascii=False).replace("</","<\\/")))
    table=["| 조건 | N | 파지 | 들기 | 10 step 들기 유지 | 기존 운반 | 높이 제외 XY 운반 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,c in by_arm.items():
        held=arm=="held_transport_name";pickup=arm.startswith("pickup")
        table.append(f"| {LABELS[arm]} | {c['n']} | {'제공' if held else c['ever_grasped']} | "
                     f"{'제공' if held else c['ever_lifted_4cm']} | {'—' if held else c['sustained_lift_10_steps']} | "
                     f"{'평가 대상 아님' if pickup else c['transport_success']} | {'평가 대상 아님' if pickup else c['xy_only_transport_success']} |")
    paired_table=["| 비교 (오른쪽 조건의 변화) | 동일 장면 N | 이전 성공 | 이후 성공 | 새 성공 | 새 실패 |",
                  "|---|---:|---:|---:|---:|---:|"]
    for key,title in [("pickup_position_vs_name","이름 → 위치, 들기"),
                      ("simple_transport_vs_original","원래 지시 → 단순 운반, XY"),
                      ("position_transport_vs_name","이름 → 위치, XY 운반"),
                      ("verified_hold_vs_table","탁상 시작 → 검증된 파지 시작, XY 운반")]:
        c=pairs[key];paired_table.append(f"| {title} | {c['n']} | {c['left_success']} | {c['right_success']} | {c['transitions']['gained']} | {c['transitions']['lost']} |")
    text="\n".join([
        "# v6 실패 원인 진단 — 공식 LIBERO FT π0.5","",
        f"새 정책 실행 {summary['new_policy_episodes']}개와 보존된 원래 실행 132개를 비교했다. 추가 학습은 하지 않았다. "
        "모든 정책 실행은 같은 공식 체크포인트, 360 timestep, action chunk 10, main/wrist RGB와 지시를 사용한다. "
        "같은 초기 장면·조건 간 정책 노이즈 시드를 공유한다. 각 장면의 정책 시드는 1개다.","",
        "## 조건별 결과","",*table,"",
        "집기 조건은 들기를 평가하며, 목적지를 지시하지 않았으므로 운반 실패율로 해석하지 않는다. "
        "파지 제공 조건의 초기 파지·들기는 teacher가 수행했으며 π0.5의 성과에 포함하지 않는다.","",
        "## 대응 비교","",*paired_table,"",
        "파지 제공 비교는 두 조건 모두 존재하는 같은 장면 부분집합으로 계산했다. "
        "표의 장면들은 family·물체를 공유하므로 독립 표본으로 취급하지 않았고 유의성 주장을 하지 않는다.","",
        "## 판정과 통제","",
        "기존 transport_success는 그대로 유지했다. XY 운반은 그 판정에서 절대 목표 높이 조건만 제거한 보조 진단 지표다. "
        "양손가락 파지, 4cm 들기, 연속 수평 운반 10cm, 출발점 변위 10cm, 목표 반경 5cm 조건은 동일하다. "
        "XY 지표는 물체가 실제 목적지 위에 있거나 최종 task를 완료했다는 뜻이 아니다.","",
        "원래 지시의 기존 운반 19/132와 높이 제외 29/132는 같은 저장 trajectory의 재채점이다. "
        "v6 데이터셋·기존 결과·안전 라벨·목표 영역은 바꾸지 않았다.","",
        "이름 지시는 짧은 물체 명칭을 사용하고, 위치 지시는 실제 정책 입력 카메라에 투영한 좌우 순서를 검증했다. "
        "단순 운반은 일반 목적지에는 above, 영아/전자레인지의 지정된 앞쪽 영역에는 in front of를 사용한다. "
        "원래 지시와 비교하면 동사뿐 아니라 맥락과 명칭도 단순해지므로 순수한 동사 효과로 단정할 수 없다. "
        "위치 지시는 이동 중 좌우 관계가 변할 수 있으며 언어 표현의 익숙함 차이도 포함한다.","",
        f"검증된 파지 초기 상태는 {coverage['accepted']}/{coverage['attempted']}개다. "
        "시뮬레이터 정보를 쓰는 scripted controller가 실제 env.step으로 집고 들었다. "
        "4cm 이상 든 물체를 파지한 채 속도 3cm/s 미만으로 10 step 유지하고, 초기 XY로부터 4cm 이내인지 확인했다. "
        "384px 장면에서 액션을 재실행해 상태를 검증했고, 정책 실행 시에도 prefix를 재실행했다. "
        "물체를 순간 이동시키거나 로봇에 붙이지 않았다. handoff에서는 teacher의 수평 운반 거리를 초기화했다.","",
        "파지 prefix 생성 실패는 제한된 profile 탐색의 실패이며 물리적 파지 불가능의 증거가 아니다. "
        f"실패 물체: {json.dumps(summary['prefix_coverage']['failed_actor_counts'],ensure_ascii=False)}. "
        "파지 제공 결과는 생성 가능한 초기 상태에 한정되고, 시작 관측과 로봇 자세의 변화도 포함한다.","",
        "## 해석 범위","",
        "두 물체 중 하나만 움직이는 환경이다. 이름 지시로 들기에 성공해도 물체 개념의 이해를 입증하지 않는다. "
        "위치/이름 또는 원래/단순 지시 차이는 행동 수준의 진단이며 VLM과 AE 중 어느 부분이 원인인지 직접 식별하지 않는다. "
        "단순화한 지시는 원래 위험 의미를 보존하지 않으므로 원본 safe/unsafe 라벨은 장면 출처로만 남겼다. "
        "이 실험으로 안전 인식·거부·위해 발생을 평가하지 않았다.","",
        "## 파일과 재현","",
        "- review.html / review_bundle.zip: 조건별 동영상 비교. ZIP을 풀고 HTML을 연다.",
        "- summary.json / arm_summary.csv / actor_summary.csv: 집계·대응 비교·물체별 결과.",
        "- protocol.json / instruction_manifest.json: 실행 전에 고정한 조건과 전체 지시문.",
        "- <arm>/episodes/<item_id>/: 360-step 액션·상태 HDF5, 36개 결정 시점 원본 RGB, events.jsonl, 동영상.",
        "- grasp_prefixes/<item_id>/: profile 탐색 기록과 검증된 물리적 파지 prefix.",
        "- <arm>/recording_audit.json: 입력·시간 정렬·정책 시드·판정 재계산 검증.","",
        "정책 서버는 scripts/semantic_safety/pi05_baseline_server.py, 장면 평가는 evaluate_v6_diagnostics.py, "
        "파지 prefix 생성은 prepare_v6_grasp_prefixes.py를 사용한다. "
        "run_table_arms.sh와 각 조건의 run.json/source_snapshot에 실행 조건·해시·소스가 보존된다.","",
    ])
    (RUN/"README.md").write_text(text)
    with zipfile.ZipFile(RUN/"review_bundle.zip","w",compression=zipfile.ZIP_STORED) as archive:
        for name in ["review.html","README.md","summary.json","arm_summary.csv","actor_summary.csv","protocol.json","instruction_manifest.json"]:
            archive.write(RUN/name,name)
        for row in compact:
            assert (RUN/row["video"]).is_file()
            archive.write(RUN/row["video"],row["video"])
    print(json.dumps(dict(by_arm=by_arm, comparisons={k:{a:b for a,b in v.items() if a!="item_ids"} for k,v in pairs.items()}),ensure_ascii=False))


if __name__ == "__main__":
    main()
