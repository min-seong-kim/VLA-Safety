"""Checks for milestone evidence and duplicate-free notification dispatch."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiment_notifications import inspect_milestones, report_once, write_json


class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        self.run.mkdir()
        self.config = Path(self.temporary.name) / "config.json"
        write_json(self.config, {"enabled": True, "thread_id": "00000000-0000-0000-0000-000000000001"})
        self.sent = []

    def sender(self, command, **kwargs):
        self.sent.append(command)
        return subprocess.CompletedProcess(command, 0, "Queued message test for thread test.\n", "")

    def complete_training(self, step=16000):
        write_json(self.run / "train_ae/config.json", {"steps_planned": 16000})
        write_json(self.run / "train_ae/freeze_audit.json",
                   {"completed_steps": step, "all_original_weights_unchanged": True})

    def test_progress_or_partial_training_does_not_report_completion(self):
        write_json(self.run / "pipeline_status.json", {"state": "running", "phase": "train train_ae"})
        self.complete_training(8000)
        self.assertEqual(inspect_milestones(self.run), {})
        self.complete_training()
        self.assertIn("training:train_ae:16000", inspect_milestones(self.run))

    def test_final_model_report_waits_for_all_three_evaluations(self):
        for kind in ["motor", "native"]:
            write_json(self.run / "final/ae" / kind / "completion.json",
                       {"complete": True, "episodes": 80, "successes": 78})
        self.assertNotIn("evaluation:ae", inspect_milestones(self.run))
        write_json(self.run / "final/ae/confirmation/completion.json",
                   {"complete": True, "episodes": 166, "successes": 75})
        self.assertIn("evaluation:ae", inspect_milestones(self.run))

    def test_accepted_event_is_not_queued_again_after_restart(self):
        self.complete_training()
        first = report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(first["state"], "queued")
        second = report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(second["state"], "no_new_milestones")
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0][-1], "--resume")
        self.assertNotIn("--model", self.sent[0])

    def test_historical_initialization_sends_nothing_but_new_events_are_sent(self):
        self.complete_training()
        result = report_once(self.run, config_path=self.config, initialize=True, sender=self.sender)
        self.assertEqual(result["previously_reported_events"], 1)
        self.assertEqual(self.sent, [])
        write_json(self.run / "pipeline_error.json", {"error": "test infrastructure failure"})
        report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(len(self.sent), 1)
        packet = json.loads(next((self.run / "reporting/events").glob("*.json")).read_text())
        self.assertEqual(len(packet["events"]), 1)
        self.assertTrue(packet["events"][0]["id"].startswith("error:"))

    def test_resume_warning_is_not_claimed_as_delivery_or_requeued(self):
        self.complete_training()

        def warning_sender(command, **kwargs):
            self.sent.append(command)
            return subprocess.CompletedProcess(command, 0, "Queued message test.\n", "Warning: resume failed")

        result = report_once(self.run, config_path=self.config, sender=warning_sender)
        self.assertEqual(result["state"], "queued_with_warning")
        report_once(self.run, config_path=self.config, sender=warning_sender)
        self.assertEqual(len(self.sent), 1)

    def test_broken_ledger_is_not_reset_and_timeout_is_not_retried(self):
        self.complete_training()
        ledger = self.run / "reporting/state.json"
        ledger.parent.mkdir()
        ledger.write_text("{broken")
        with self.assertRaises(json.JSONDecodeError):
            report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(self.sent, [])
        ledger.unlink()

        def timeout_sender(command, **kwargs):
            self.sent.append(command)
            raise subprocess.TimeoutExpired(command, 30)

        result = report_once(self.run, config_path=self.config, sender=timeout_sender)
        self.assertEqual(result["state"], "delivery_unknown")
        report_once(self.run, config_path=self.config, sender=timeout_sender)
        self.assertEqual(len(self.sent), 1)


if __name__ == "__main__":
    unittest.main()
