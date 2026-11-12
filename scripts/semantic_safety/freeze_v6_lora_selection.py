"""Freeze audited validation choices before running the untouched final test set."""

import argparse
import json
from pathlib import Path
import time

from rollout_common import write_json
from semantic_pairs_v6 import sha256


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    run, data = args.run_root.resolve(), args.data_root.resolve()
    destination = run / "selection_frozen.json"
    if destination.exists():
        raise FileExistsError("Selection is already frozen: " + str(destination))
    assert not (run / "final/test").exists(), "Final test must remain unopened until selection is frozen"
    protocol = json.loads((run / "experiment_protocol.json").read_text())
    comparison = json.loads((run / "training_comparison_audit.json").read_text())
    assert comparison["passed"]
    grid_path = run / "validation_grid_complete.json"
    grid = json.loads(grid_path.read_text())
    assert grid["complete"] and not grid["test_used_for_selection"]
    extension = protocol.get("extension")
    expected = extension["selection_total_steps"] if extension else protocol["selection_steps"]
    if extension:
        assert grid["extension_completed"]
        assert json.loads((run / "three_extensions_complete.json").read_text())["complete"]
        assert comparison["total_steps_per_scope"] == extension["total_optimizer_updates_per_scope"]
    selected = {}
    candidate_identities = []
    for scope in ["ae", "vlm", "all"]:
        rows = [row for row in grid["rows"] if row["scope"] == scope]
        assert sorted(row["step"] for row in rows) == expected
        for row in rows:
            folder = Path(row["run"])
            assert row["n"] == 66
            assert json.loads((folder / "recording_audit.json").read_text())["passed"]
            identity = json.loads((folder / "run.json").read_text())
            digest = sha256(Path(row["adapter"]))
            assert digest == identity["adapter_sha256"]
            candidate_identities.append(dict(scope=scope, step=row["step"], adapter_sha256=digest))
        best = max(rows, key=lambda row: (row["transport_success"], row["sustained_transport_success"], -row["step"]))
        assert best == grid["selected"][scope]
        selected[scope] = {**best, "adapter_sha256": sha256(Path(best["adapter"]))}
    result = dict(frozen=True, frozen_unix=time.time(), selected=selected,
                  extension_completed=bool(extension), test_used_for_selection=False,
                  selection_rule=grid["selection_rule"], selection_scenes=66,
                  total_training_steps_per_scope=comparison["total_steps_per_scope"],
                  validation_grid_sha256=sha256(grid_path),
                  training_comparison_audit_sha256=sha256(run / "training_comparison_audit.json"),
                  experiment_protocol_sha256=sha256(run / "experiment_protocol.json"),
                  selection_items_sha256=sha256(data / "holdouts/validation_selection66.jsonl"),
                  test_items_sha256=sha256(data / "holdouts/test.jsonl"),
                  candidate_adapter_identities=candidate_identities,
                  stopping_rule="One common 6000-step stage and one bounded 2000-step warm-start stage; final test is evaluation only.")
    write_json(destination, result)
    print(json.dumps({scope: dict(step=row["step"], transport=row["transport_success"],
                                 sustained=row["sustained_transport_success"])
                      for scope, row in selected.items()}, indent=2))


if __name__ == "__main__":
    main()
