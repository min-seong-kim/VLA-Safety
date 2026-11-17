"""Report completed experiment stages to the user's existing Codex conversation."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "runs/experiment_reporting.json"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        # A producer may be replacing a status file. Inspect it on the next poll.
        return None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def inspect_milestones(run):
    """Inspect durable completion evidence; training steps/progress are not events."""
    events = {}

    def add(key, title, evidence, result=None):
        events[key] = {"id": key, "title": title, "evidence": str(evidence), "result": result}

    for path in sorted((run / "milestones").glob("*.json")):
        milestone = read_json(path) or {}
        if milestone.get("complete") is True and milestone.get("title"):
            add("milestone:" + path.stem, milestone["title"], path, milestone.get("result"))

    ready = read_json(run / "experiment_data_ready.json") or {}
    if ready.get("full_data_audit_passed") is True:
        add("data_ready", "학습 시연 준비 및 전체 검증 완료", run / "experiment_data_ready.json")

    for path in sorted(run.glob("train_*/freeze_audit.json")):
        audit = read_json(path) or {}
        config = read_json(path.parent / "config.json") or {}
        step = audit.get("completed_steps", 0)
        planned = config.get("steps_planned", 0)
        if planned > 0 and step >= planned and audit.get("all_original_weights_unchanged") is True:
            add(f"training:{path.parent.name}:{step}", f"{path.parent.name} 학습 및 freeze 검증 완료", path,
                {"completed_steps": step, "original_weights_unchanged": True})

    protocol = read_json(run / "experiment_protocol.json") or {}
    candidates = read_json(run / "checkpoint_candidates.json") or []
    required_steps = set(protocol.get("save_steps", []))
    for scope in protocol.get("scopes", []):
        rows = [row for row in candidates if row.get("scope") == scope]
        if required_steps and required_steps <= {row.get("cumulative_step") for row in rows}:
            add(f"validation:{scope}", f"{scope} 초기 체크포인트별 검증 완료",
                run / "checkpoint_candidates.json", rows)

    selected = read_json(run / "frozen_selection.json") or {}
    if selected.get("frozen_before_test") is True:
        add("selection_frozen", "최종 평가용 체크포인트 선택 완료", run / "frozen_selection.json",
            selected.get("selected"))

    for folder in sorted((run / "final").glob("*")):
        results = {kind: read_json(folder / kind / "completion.json") or {}
                   for kind in ["motor", "native", "confirmation"]}
        if all(value.get("complete") is True and value.get("episodes", 0) > 0 for value in results.values()):
            add(f"evaluation:{folder.name}", f"{folder.name} 최종 평가 완료", folder, results)

    for filename in ["pipeline_error.json", "experiment_error.json", "supervisor_error.json"]:
        error = read_json(run / filename)
        if error:
            digest = hashlib.sha256(json.dumps(error, sort_keys=True).encode()).hexdigest()[:16]
            add(f"error:{filename}:{digest}", "실험 오류 발생: 현재 복구 상태 확인 필요", run / filename)
    for row in read_json(run / "automatic_recovery_history.json") or []:
        add(f"recovery:{row['attempt']}", "실험 오류 후 자동 복구 시도", run / "automatic_recovery_history.json", row)

    status = read_json(run / "supervisor_status.json") or {}
    if status.get("state") == "needs_attention":
        add(f"attention:{status.get('updated_unix')}", "자동 복구 중단: 확인 필요", run / "supervisor_status.json")
    pause = read_json(run / "pause_requested.json")
    if pause:
        add(f"pause:{pause.get('requested_unix')}", "사용자 요청으로 실험 일시 중지", run / "pause_requested.json")
    experiment = read_json(run / "experiment_status.json") or {}
    pipeline = read_json(run / "pipeline_status.json") or {}
    if experiment.get("state") == pipeline.get("state") == "complete":
        add("experiment_complete", "전체 실험 및 최종 보고서 완료", run / "experiment_status.json")
    return events


def reported_event_ids(run):
    """Use foreground reports as well as callbacks to suppress stale notifications."""
    reported = set()
    folder = run / "reporting"
    for path in (folder / "reported").glob("*.json"):
        receipt = read_json(path) or {}
        if receipt.get("reported") is False:
            continue
        events = receipt.get("events")
        if events is None:
            packet = read_json(folder / "events" / path.name) or {}
            events = packet.get("events", [])
        reported.update(event if isinstance(event, str) else event["id"] for event in events)
    return reported


def acknowledge_report(run, event_ids):
    """Record an actual user-facing report without queueing an acknowledgement."""
    folder = run / "reporting"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        events = inspect_milestones(run)
        assert set(event_ids) <= events.keys(), "Only existing, verified events can be acknowledged"
        path = folder / "state.json"
        state = json.loads(path.read_text()) if path.exists() else {"events": {}}
        for event_id in event_ids:
            state["events"][event_id] = {"state": "reported", "updated_unix": time.time()}
        write_json(path, state)
        write_json(folder / "reported" / ("manual_" + uuid.uuid4().hex + ".json"), {
            "reported": True, "received_by": "foreground_report", "events": list(event_ids),
            "reported_in_active_turn": True, "updated_unix": time.time()})
    return {"state": "acknowledged_without_message", "events": len(event_ids)}


def report_once(run, *, config_path=CONFIG, initialize=False, sender=subprocess.run):
    """Queue each milestone at most once; record ambiguous delivery without resending."""
    config = read_json(config_path) or {}
    if not config.get("enabled"):
        return {"state": "disabled"}
    thread = str(uuid.UUID(config["thread_id"]))
    folder = run / "reporting"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "another_reporter_active"}
        state_path = folder / "state.json"
        # A malformed delivery ledger must never be replaced with an empty one.
        state = json.loads(state_path.read_text()) if state_path.exists() else {"events": {}}
        events = inspect_milestones(run)
        reported = reported_event_ids(run)
        for event_id in reported:
            state["events"][event_id] = {"state": "reported", "updated_unix": time.time()}
        # A final report already covers older completion and recovery milestones.
        terminal = all((read_json(run / name) or {}).get("state") == "complete"
                       for name in ("experiment_status.json", "pipeline_status.json"))
        if "experiment_complete" in reported and terminal:
            for event_id in events:
                if not event_id.startswith(("error:", "attention:", "pause:")):
                    state["events"].setdefault(event_id, {"state": "covered_by_final_report"})
        write_json(state_path, state)
        unseen = [value for key, value in events.items() if key not in state["events"]]
        if initialize:
            for event in unseen:
                state["events"][event["id"]] = {"state": "already_reported", "updated_unix": time.time()}
            write_json(state_path, state)
            return {"state": "initialized", "previously_reported_events": len(unseen)}
        if not unseen:
            write_json(folder / "pending.json", {"events": []})
            return {"state": "no_new_milestones"}
        mode = (read_json(folder / "delivery_mode.json") or {}).get(
            "mode", config.get("delivery_mode", "foreground"))
        if mode == "foreground":
            # The active assistant reports directly. Do not enqueue delayed duplicate turns.
            write_json(folder / "pending.json", {"events": unseen, "updated_unix": time.time()})
            return {"state": "deferred_to_foreground", "events": len(unseen)}
        assert mode == "background", f"Unknown notification delivery mode: {mode}"
        write_json(folder / "pending.json", {"events": []})
        packet_id = uuid.uuid4().hex
        packet_path = folder / "events" / (packet_id + ".json")
        write_json(packet_path, {"packet_id": packet_id, "run_dir": str(run), "thread_id": thread,
                                 "detected_unix": time.time(), "events": unseen})
        for event in unseen:
            state["events"][event["id"]] = {"state": "dispatching", "packet": str(packet_path)}
        write_json(state_path, state)
        autonomous_recovery = (read_json(run / "experiment_protocol.json") or {}).get("autonomous_recovery_authorized") is True
        recovery_instruction = (
            "사용자는 이 실행의 오류 수정과 동일 프로토콜의 재개·후속 단계 자동 진행을 승인했습니다. "
            "미해결 오류가 있으면 기존 실행을 복구하고 이어가세요. 실행 중인 감독 프로세스가 있으면 중복 실행하지 마세요. "
            "새 VLA 학습, LDA, 새 데이터 조건이나 실험 범위 확장은 시작하지 마세요. 이 메시지를 다시 큐에 넣지 마세요. "
        ) if autonomous_recovery else (
            "이 호출은 상태 보고만 요청하며, 새 학습·평가·추가 실험을 시작하거나 이 메시지를 다시 큐에 넣지 마세요. "
        )
        prompt = (
            "[사용자가 요청한 실험 단계 자동 보고] "
            f"이벤트 파일: {packet_path}\n"
            "사용자에게 말하기 전에 보고 이력을 먼저 확인하세요. 이미 보고한 packet 또는 event이거나, "
            "전체 완료를 이미 보고한 실험의 과거 단계이면 조용히 종료하세요. "
            "중복인 경우 진행 안내·수신 확인·기록 갱신 안내·최종 답변을 포함해 어떤 사용자 메시지도 남기지 마세요. "
            "이 파일과 근거 파일의 현재 상태를 확인하고, 새로 완료된 단계·핵심 결과·남은 단계를 한국어로 짧게 보고하세요. "
            "진행률만 반복 보고하지 마세요. 오류 이벤트는 현재 복구 여부를 확인해 함께 설명하세요. "
            "이미 보고한 packet이면 반복하지 마세요. 보고 전 reporting/reported/<packet_id>.json에 확인 기록을 남기세요. "
            + recovery_instruction +
            "이미 사용자가 중지/변경을 요청했다면 최신 지시를 따르세요."
        )
        command = [config.get("codex_command", "codex"), "queue", "--thread", thread,
                   "--message", prompt, "--resume"]
        try:
            result = sender(command, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
            # Queue acceptance and actual assistant receipt are separate facts.
            delivery = "queued" if result.returncode == 0 and "Queued message " in result.stdout else "queue_failed"
            if delivery == "queued" and result.stderr.strip():
                delivery = "queued_with_warning"
            receipt = {"state": delivery, "returncode": result.returncode,
                       "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            receipt = {"state": "delivery_unknown", "error": str(error)}
        receipt.update(packet_id=packet_id, updated_unix=time.time())
        write_json(folder / "dispatch" / (packet_id + ".json"), receipt)
        for event in unseen:
            state["events"][event["id"]].update(receipt)
        write_json(state_path, state)
        return receipt


def notify_progress(run):
    """Keep notification transport failures independent from experiment execution."""
    try:
        return report_once(run)
    except Exception:
        error = {"error": traceback.format_exc(), "updated_unix": time.time()}
        try:
            write_json(run / "reporting/watcher_error.json", error)
        except OSError:
            pass
        return {"state": "notification_error"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true", help="Mark existing milestones already reported; send nothing")
    parser.add_argument("--delivery-mode", choices=["foreground", "background"],
                        help="Foreground defers callbacks; background queues only unreported events")
    parser.add_argument("--acknowledge", nargs="+", metavar="EVENT_ID",
                        help="Record event IDs already reported directly; send no message")
    parser.add_argument("--watch", action="store_true", help="Watch until complete, paused, or requiring attention")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    assert run.is_dir(), "Run directory must already exist"
    assert args.interval >= 10, "Avoid frequent filesystem polling"
    if args.delivery_mode:
        write_json(run / "reporting/delivery_mode.json", {"mode": args.delivery_mode, "updated_unix": time.time()})
    if args.acknowledge:
        print(json.dumps(acknowledge_report(run, args.acknowledge)))
        return
    if args.initialize:
        print(json.dumps(report_once(run, initialize=True)))
        return
    while True:
        result = notify_progress(run)
        if result["state"] != "no_new_milestones":
            print(json.dumps(result, ensure_ascii=False), flush=True)
        supervisor = read_json(run / "supervisor_status.json") or {}
        experiment = read_json(run / "experiment_status.json") or {}
        if (not args.watch or experiment.get("state") == "complete"
                or supervisor.get("state") == "needs_attention" or (run / "pause_requested.json").exists()):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
