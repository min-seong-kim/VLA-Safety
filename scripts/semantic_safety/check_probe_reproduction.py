"""Refit the original primary and benign-mix probes in scratch space and compare predictions."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import contrast_initial_linear as contrast
import mix_libero_benign_probes as mix


def scratch_inputs(reference, target):
    target.mkdir()
    for name in ("features", "manifest.json", "experiment_protocol.json"):
        (target / name).symlink_to((reference / name).resolve(strict=True))


def main():
    with TemporaryDirectory(prefix="vla-probe-replay-") as tmp:
        old, mixed = contrast.RUN, mix.RUN
        first, second = Path(tmp) / "original", Path(tmp) / "mixed"
        scratch_inputs(old, first)
        scratch_inputs(mixed, second)
        n, error = 0, 0.0
        with patch.object(contrast, "RUN", first):
            for track in ("A", "B"):
                for model in contrast.MODELS:
                    for site in mix.SITES:
                        contrast.probe_job((model, track, site))
                        relative = Path("probes") / track / model / site / "oof_predictions.npz"
                        with np.load(old / relative) as ref, np.load(first / relative) as got:
                            for key in ("fold", "label", "global_indices"):
                                np.testing.assert_array_equal(got[key], ref[key])
                            scores = got["decision_score"]
                            error = max(error, float(np.abs(scores - ref["decision_score"]).max()))
                            np.testing.assert_allclose(scores, ref["decision_score"], atol=1e-8, rtol=1e-8)
                            n += scores.size
            # Use the freshly refitted baseline folds for both the 0% control and calibration.
            with patch.object(mix, "RUN", second):
                mix.fit()
        hashes = {}
        for name in ("fold_metrics.csv", "summary.csv", "test_predictions.csv"):
            hashes[name] = hashlib.sha256((second / name).read_bytes()).hexdigest()
            expected = hashlib.sha256((mixed / name).read_bytes()).hexdigest()
            if hashes[name] != expected:
                raise ValueError(f"{name} differs at byte level. Expected {expected}, got {hashes[name]}")
        print(json.dumps(dict(original_test_logits=n, max_abs_original_logit_difference=error,
                              mixed_report_hashes=hashes, byte_identical=True), indent=2))


if __name__ == "__main__":
    main()
