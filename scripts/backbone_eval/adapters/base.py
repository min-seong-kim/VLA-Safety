"""Common adapter contract.

A reader turns (image, task body) into ONE forward pass and returns, per example,
    h      : final-normed hidden state of the last valid prompt token, float32 [D]
    logp   : log_softmax over the model's real vocabulary of LMHead(h), float32 [V]
Gold labels are never an input. The same `h` feeds the likelihood readout and the linear probe.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np


@dataclasses.dataclass
class ForwardOut:
    h: np.ndarray  # [b, D] float32
    logp: np.ndarray  # [b, V] float32
    n_prompt_tokens: np.ndarray  # [b] text tokens incl. BOS and boundary token
    aux: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class CandidateCheck:
    label: str
    token_id: int | None
    pieces: list[str]
    ok: bool
    note: str = ""


class PrefixReader:
    name: str
    family: str
    attention: str  # human readable description of the attention actually used
    hidden_dim: int
    vocab_size: int
    max_text_tokens: int
    manifest: dict  # checkpoint revision, dtype, preprocessing, head, ...

    # --- text
    def wrap(self, body: str) -> str:
        raise NotImplementedError

    def prompt_ids(self, body: str) -> list[int]:
        raise NotImplementedError

    def candidate_checks(self, labels: list[str], sample_bodies: list[str]) -> list[CandidateCheck]:
        """Verify every candidate is exactly one token at the real answer boundary (see plan §5)."""
        raise NotImplementedError

    # --- forward
    def forward(self, pils: list, bodies: list[str]) -> ForwardOut:
        raise NotImplementedError

    # --- validation hooks (all optional, return dicts that go into validation_report.md)
    def check_head(self) -> dict:
        return {}

    def check_attention_direction(self) -> dict:
        return {}

    def param_fingerprint(self) -> str:
        return ""

    def describe(self) -> dict:
        d = dict(self.manifest)
        d.update(name=self.name, family=self.family, attention=self.attention, hidden_dim=self.hidden_dim,
                 vocab_size=self.vocab_size, max_text_tokens=self.max_text_tokens)
        return d


def summarize_checks(checks: list[CandidateCheck]) -> dict:
    return {c.label: {"token_id": c.token_id, "pieces": c.pieces, "ok": c.ok, "note": c.note} for c in checks}


def jdump(o) -> str:
    return json.dumps(o, ensure_ascii=False, default=str)
