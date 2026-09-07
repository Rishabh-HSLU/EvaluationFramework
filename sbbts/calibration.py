"""Beta calibration.

The paper (Section 4, Theorem 3.2) requires beta * Delta_t_i > 1 for every
training interval, and explicitly flags beta's value as an open calibration
question ("the model's behavior is influenced by the regularization
parameter beta, whose optimal selection currently lacks a systematic
criterion -- although practical guidelines recommend avoiding excessively
small values"). See scripts/reasoning.md for the full derivation, the
chosen safety factor, and the real-corpus value that superseded the
synthetic-corpus one this was first derived against.
"""

from __future__ import annotations

from scripts.sbbts_panel import SessionExample

from .data import corpus_min_delta_t


def derive_beta(examples: list[SessionExample], safety_factor: float = 2.0) -> float:
    """beta = safety_factor / min(Delta_t_i), comfortably above the 1/min(Delta_t_i) floor."""
    if safety_factor <= 1.0:
        raise ValueError("safety_factor must exceed 1.0 to keep beta * Delta_t_i > 1 with margin.")
    min_dt = corpus_min_delta_t(examples)
    return safety_factor / min_dt


def check_beta_constraint(examples: list[SessionExample], beta: float) -> bool:
    """True iff beta * Delta_t_i > 1 for every observed interval in the corpus."""
    min_dt = corpus_min_delta_t(examples)
    return beta * min_dt > 1.0
