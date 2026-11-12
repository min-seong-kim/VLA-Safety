# Semantic Safety Experiment Operations

The user requests proactive Korean reports when a dataset, model training stage,
validation stage, final evaluation, or complete experiment finishes, and when an
error occurs. Avoid routine step-count updates and unchanged progress reports.

For long experiments, connect stage completion to `experiment_notifications.py`.
The v5 supervisor calls `notify_progress(RUN)` automatically. Other experiment
drivers must integrate this reporter or a watcher with matching completion
evidence before being left running in the background. Configuration lives in
`runs/experiment_reporting.json`; delivery targets the user's existing Codex
conversation through `codex queue --resume`. Never start a new experiment from
a reporting callback.

Distinguish queued messages from confirmed receipt. Inspect dispatch warnings
and errors; do not claim successful chat delivery solely from a completed job.
Use `--initialize` only for historical milestones already reported to the user.
The ledger prevents repeated messages after restarts. Do not retry ambiguous
queue delivery without checking whether the message was accepted.

The user deferred all optional 8,000-step extensions. Keep that override until
the user explicitly requests additional training.
