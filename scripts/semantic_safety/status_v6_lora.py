"""Print current collection, training, and evaluation progress without changing runs."""

import argparse
import json
from pathlib import Path
import time

from v6_lora_data import DATA, RUN


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def training_state(folder, planned_steps):
    path = folder / "training.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines()[-101:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    config = read(folder / "config.json")
    state = dict(state="not_started", step=0, planned_steps=planned_steps)
    if config:
        state.update(state="initializing", planned_steps=config["steps_planned"])
    if rows:
        last = rows[-1]
        state.update(state="running", step=last["step"], elapsed_seconds=last["elapsed_seconds"],
                     latest_loss=last["loss"], seconds_since_log=time.time() - path.stat().st_mtime)
        if len(rows) > 10:
            rate = (last["elapsed_seconds"] - rows[0]["elapsed_seconds"]) / (last["step"] - rows[0]["step"])
            state.update(seconds_per_step=rate,
                         estimated_remaining_training_seconds=max(0, state["planned_steps"] - last["step"]) * rate)
    audit = read(folder / "freeze_audit.json")
    if audit:
        state.update(state="complete" if audit["all_original_weights_unchanged"] else "audit_failed",
                     original_weights_unchanged=audit["all_original_weights_unchanged"])
    return state


def main():
    global RUN, DATA
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root",type=Path,default=RUN)
    parser.add_argument("--data-root",type=Path,default=DATA)
    args=parser.parse_args();RUN=args.run_root.resolve();DATA=args.data_root.resolve()
    result = dict(unix_time=time.time(), run=str(RUN))
    dataset = read(DATA / "training_data_audit.json")
    if dataset:
        result["data"] = {key: dataset[key] for key in ["episodes", "control_steps", "conditions_with_demonstrations",
                                                       "missing_conditions", "collection_complete"]}
    else:
        result["data"] = dict(cached_successful_episodes=len(list((DATA / "training_cache").glob("*/audit.json"))),
                              collection_complete=False)
    result["rehearsal_merge"] = read(DATA / "rehearsal_merge_audit.json")
    result["training"] = {scope: training_state(RUN / f"train_{scope}", 6000) for scope in ["ae", "vlm", "all"]}
    extension = read(RUN / "extension_protocol.json")
    if extension:
        result["extension_training"] = {scope: training_state(RUN / f"train_{scope}_extension", 2000)
                                        for scope in ["ae", "vlm", "all"]}
        result["planned_total_steps_per_scope"] = extension["total_optimizer_updates_per_scope"]
        validation = read(RUN / "validation_extension_complete.json") or read(RUN / "validation_grid_progress.json")
    else:
        validation = read(RUN / "validation_grid_complete.json") or read(RUN / "validation_grid_progress.json")
    result["validation"] = validation
    final = read(RUN / "final_evaluation_complete.json") or read(RUN / "final_evaluation_progress.json")
    result["final_evaluation"] = None if final is None else dict(complete=final["complete"],
        rows=[{key: row[key] for key in ["model", "benchmark", "overall"]} for row in final["rows"]])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
