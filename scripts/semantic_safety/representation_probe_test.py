"""Check family isolation and invariance to duplicated trajectory frames."""

from collections import Counter
import json
import unittest

import numpy as np

from probe_representation_linear import metrics, weights
from representation_common import RUN


class ProbeProtocolTests(unittest.TestCase):
    def test_each_family_is_held_out_exactly_once(self):
        folds = json.loads((RUN / "splits.json").read_text())["folds"]
        for part, repeats in (("test", 1), ("validation", 1), ("train", 3)):
            count = Counter(family for fold in folds for family in fold[part + "_families"])
            self.assertEqual(len(count), 35)
            self.assertEqual(set(count.values()), {repeats})
        for fold in folds:
            groups = [set(fold[part + "_families"]) for part in ("train", "validation", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_all_counterparts_and_layouts_share_family(self):
        rows = json.loads((RUN / "episodes.json").read_text())
        pairs = {}
        for row in rows:
            pairs.setdefault(row["pair_id"], []).append(row)
        self.assertEqual(len(pairs), 169)
        for group in pairs.values():
            self.assertEqual(len(group), 4)
            self.assertEqual(len({row["family_id"] for row in group}), 1)
            self.assertEqual(Counter(row["label"] for row in group), {0: 2, 1: 2})

    def test_duplicate_frames_do_not_increase_episode_weight_or_metric(self):
        original = [{"episode_id": f"e{i}", "family_id": f"f{i // 2}", "pair_group": f"p{i // 2}",
                     "label": i % 2} for i in range(4)]
        scores = [-1., 1., 0.2, -0.1]
        expanded, values = [], []
        for i, row in enumerate(original):
            for _ in range(13 if i == 0 else 1):
                expanded.append(row)
                values.append(scores[i])
        for rows in (original, expanded):
            w = weights(rows)
            for episode in original:
                total = sum(value for row, value in zip(rows, w) if row["episode_id"] == episode["episode_id"])
                self.assertAlmostEqual(total, 1.)
        before = metrics(original, np.asarray([row["label"] for row in original]), np.asarray(scores))
        after = metrics(expanded, np.asarray([row["label"] for row in expanded]), np.asarray(values))
        for key in ("AUROC", "balanced_accuracy", "episode_AUROC", "paired_episode_order_accuracy"):
            self.assertAlmostEqual(before[key], after[key])


if __name__ == "__main__":
    unittest.main()
