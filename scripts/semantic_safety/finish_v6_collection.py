"""Complete the 48-demo condition quota across all successful teacher families."""

import json
import multiprocessing
from pathlib import Path

from collect_v6_fast import process
from rollout_common import write_json
from v6_lora_data import DATA, conditions


def existing(tag, condition):
    folder = DATA / tag / "train" / f"c{condition:02d}"
    receipts = [json.loads(path.read_text()) for path in sorted(folder.glob("*/receipt.json"))]
    return receipts, [row for row in receipts if row["accepted"]]


def summarize_condition(tag, condition, reason):
    receipts, accepted = existing(tag, condition)
    summary = dict(condition=condition, split="train", asset=conditions()[condition]["actor_asset"],
                   accepted=accepted, attempted=len(receipts), requested=48, complete=len(accepted) >= 48,
                   stop_reason=reason)
    write_json(DATA / tag / "train" / f"c{condition:02d}" / "collection_summary.json", summary)
    return summary


def summarize_tag(tag, indices):
    summaries = [json.loads((DATA / tag / "train" / f"c{condition:02d}" / "collection_summary.json").read_text())
                 for condition in indices]
    rows = sorted([row for summary in summaries for row in summary["accepted"]], key=lambda row: row["item_id"])
    (DATA / tag / "train_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(DATA / tag / "train_summary.json", dict(episodes=len(rows), steps=sum(row["steps"] for row in rows),
               complete=all(summary["complete"] for summary in summaries), conditions=summaries,
               pooling_rule="48 successful demonstrations per condition across teacher families; bounded failures retained"))


def main():
    requests = []
    actions = []
    for condition in range(66):
        main_receipts, main_rows = existing("episodes", condition)
        other = sum(len(existing(tag, condition)[1]) for tag in ["pose_episodes", "front_episodes"])
        summary = DATA / "episodes/train" / f"c{condition:02d}" / "collection_summary.json"
        if summary.exists():
            continue
        if len(main_rows) + other >= 48:
            summarize_condition("episodes", condition, "pooled_success_quota_reached")
        else:
            target = max(1, 48 - other)
            requests.append((False, (condition, "train", target, 0, 96, "episodes")))
        actions.append(dict(condition=condition, existing_main=len(main_rows), other_teacher_successes=other,
                            remaining=max(0, 48 - len(main_rows) - other)))
    for condition in [4, 5, 6, 7, 50, 51]:
        summary = DATA / "front_episodes/train" / f"c{condition:02d}" / "collection_summary.json"
        if not summary.exists():
            assert sum(len(existing(tag, condition)[1]) for tag in ["episodes", "front_episodes", "pose_episodes"]) >= 48
            summarize_condition("front_episodes", condition, "pooled_success_quota_reached")
    summarize_tag("front_episodes", [4, 5, 6, 7, 50, 51])
    write_json(DATA / "collection_pooling_decision.json", dict(conditions=actions,
               source_trials_unchanged=True, accepted_demos_preserved=True, quota_per_condition=48,
               reason="Avoid collecting redundant demonstrations for conditions already covered by supplementary teachers."))
    if requests:
        with multiprocessing.get_context("spawn").Pool(min(12, len(requests)), maxtasksperchild=1) as pool:
            list(pool.imap_unordered(process, requests))
    summarize_tag("episodes", range(66))
    print("POOLED_COLLECTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
