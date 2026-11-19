"""Regression checks for shared-control purging and fixed training weights."""

import unittest
import numpy as np

import split_comparison_v1 as c


def row(name, contrasts, observation, label=0, track="A", condition=None):
    return dict(item_id=name, condition_id=condition or name, contrast_ids=contrasts,
                family_ids=contrasts, observation_sha256=observation, label=label,
                track=track, group_id="fixed")


class SplitTests(unittest.TestCase):
    def test_observation_links_contrasts_transitively(self):
        rows = [row("a", ["a"], "same"), row("b", ["b"], "same"),
                row("shared", ["b", "c"], "other")]
        groups = c.connect(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(next(iter(groups.values()))), {"a", "b", "c"})

    def test_same_observation_different_instruction_purges_train(self):
        rows = [row("test", ["a"], "same"), row("train", ["c"], "same", 1),
                row("dev", ["b"], "dev"), row("other", ["d"], "other")]
        plan = c.materialize(rows, [["a"], ["b"], ["c"], ["d"], ["e"]], "contrast")[0]
        self.assertEqual(plan["indices"]["test"], [0])
        self.assertEqual(plan["indices"]["train"], [3])
        self.assertEqual(plan["excluded_shared_rows"]["train"], 1)

    def test_shared_condition_is_purged_across_tracks(self):
        rows = [row("testA", ["a"], "image1", condition="shared"),
                row("trainB", ["c"], "image2", track="B", condition="shared"),
                row("dev", ["b"], "dev"), row("other", ["d"], "other")]
        plan = c.materialize(rows, [["a"], ["b"], ["c"], ["d"], ["e"]], "contrast")[0]
        self.assertNotIn(1, plan["indices"]["train"])

    def test_shared_row_is_not_duplicated_in_test(self):
        rows = [row("shared", ["a", "b"], "same")]
        plan = c.materialize(rows, [["a", "b"], ["c"], ["d"], ["e"], ["f"]], "contrast")[0]
        self.assertEqual(plan["indices"]["test"], [0])

    def test_class_mass_and_fixed_weighting_unit(self):
        rows = [row("s1", ["a"], "a"), row("s2", ["b"], "b"), row("u", ["c"], "c", 1)]
        w = c.train_weights(rows)
        self.assertAlmostEqual(w[:2].sum(), w[2])
        for r in rows:
            r["family_ids"], r["contrast_ids"] = ["changed"], ["another"]
        np.testing.assert_array_equal(w, c.train_weights(rows))


if __name__ == "__main__":
    unittest.main()
