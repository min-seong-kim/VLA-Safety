"""Checks for milestone evidence and duplicate-free notification dispatch."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiment_notifications import acknowledge_report, inspect_milestones, report_once, write_json


class NotificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "run"
        self.run.mkdir()
        self.config = Path(self.temporary.name) / "config.json"
        write_json(self.config, {"enabled": True, "thread_id": "00000000-0000-0000-0000-000000000001",
                                 "delivery_mode": "background"})
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

    def test_generic_stage_requires_completion_and_is_not_requeued(self):
        path = self.run / "milestones/features.json"
        write_json(path, {"complete": False, "title": "features ready"})
        self.assertEqual(inspect_milestones(self.run), {})
        write_json(path, {"complete": True, "title": "features ready", "result": {"inputs": 16}})
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "queued")
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "no_new_milestones")
        self.assertEqual(len(self.sent), 1)

    def test_recovery_scope_requires_explicit_protocol_authorization(self):
        self.complete_training()
        write_json(self.run / "experiment_protocol.json", {"autonomous_recovery_authorized": True})
        report_once(self.run, config_path=self.config, sender=self.sender)
        prompt = self.sent[0][self.sent[0].index("--message") + 1]
        self.assertIn("동일 프로토콜", prompt)
        self.assertIn("중복 실행하지", prompt)
        self.assertIn("새 VLA 학습, LDA", prompt)

    def test_foreground_report_prevents_delayed_duplicate_callback(self):
        self.complete_training()
        write_json(self.run / "reporting/delivery_mode.json", {"mode": "foreground"})
        result = report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(result["state"], "deferred_to_foreground")
        self.assertEqual(self.sent, [])
        acknowledge_report(self.run, ["training:train_ae:16000"])
        write_json(self.run / "reporting/delivery_mode.json", {"mode": "background"})
        result = report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(result["state"], "no_new_milestones")
        self.assertEqual(self.sent, [])

    def test_default_mode_does_not_enqueue_while_assistant_is_active(self):
        write_json(self.config, {"enabled": True, "thread_id": "00000000-0000-0000-0000-000000000001"})
        self.complete_training()
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "deferred_to_foreground")
        self.assertEqual(self.sent, [])

    def test_unreported_foreground_event_is_sent_once_when_backgrounded(self):
        self.complete_training()
        write_json(self.run / "reporting/delivery_mode.json", {"mode": "foreground"})
        report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(self.sent, [])
        write_json(self.run / "reporting/delivery_mode.json", {"mode": "background"})
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "queued")
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "no_new_milestones")
        self.assertEqual(len(self.sent), 1)
        prompt = self.sent[0][self.sent[0].index("--message") + 1]
        self.assertIn("수신 확인", prompt)
        self.assertIn("어떤 사용자 메시지도 남기지 마세요", prompt)

    def test_report_receipt_survives_missing_dispatch_ledger(self):
        self.complete_training()
        write_json(self.run / "reporting/reported/manual.json", {
            "reported": True, "events": ["training:train_ae:16000"]})
        result = report_once(self.run, config_path=self.config, sender=self.sender)
        self.assertEqual(result["state"], "no_new_milestones")
        self.assertEqual(self.sent, [])

    def test_legacy_report_receipt_without_event_ids_still_suppresses_duplicate(self):
        self.complete_training()
        report_once(self.run, config_path=self.config, sender=self.sender)
        packet = next((self.run / "reporting/events").glob("*.json"))
        write_json(self.run / "reporting/reported" / packet.name, {"reported_in_active_turn": True})
        (self.run / "reporting/state.json").unlink()
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "no_new_milestones")
        self.assertEqual(len(self.sent), 1)

    def test_final_report_suppresses_old_stages_but_allows_new_errors(self):
        self.complete_training()
        for name in ("experiment_status.json", "pipeline_status.json"):
            write_json(self.run / name, {"state": "complete"})
        acknowledge_report(self.run, ["experiment_complete"])
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "no_new_milestones")
        self.assertEqual(self.sent, [])
        write_json(self.run / "pipeline_error.json", {"error": "New failure after completion"})
        self.assertEqual(report_once(self.run, config_path=self.config, sender=self.sender)["state"], "queued")
        packet = json.loads(next((self.run / "reporting/events").glob("*.json")).read_text())
        self.assertEqual(len(packet["events"]), 1)
        self.assertTrue(packet["events"][0]["id"].startswith("error:"))

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
