"""Configuration for the SBBTS drift network and training loop.

Values are taken directly from Table 2 (Appendix C) of the SBBTS paper
unless noted otherwise. See scripts/reasoning.md for beta's derivation,
which the paper explicitly leaves as an open calibration question, and for
the batch_size / grad_accum_steps split (the real bucketing pipeline,
scripts/sbbts_batching.py, defaults to a micro-batch size of 64; the paper
wants an effective batch of 128).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SBBTSModelConfig:
    """Architecture hyperparameters (Appendix B, Table 2)."""

    state_dim: int = 1  # d: per-ticker log-return trajectories are scalar
    d_model: int = 128
    n_heads: int = 16
    n_layers: int = 1  # past-trajectory encoder depth


@dataclass(frozen=True)
class SBBTSTrainingConfig:
    """Training-loop hyperparameters (Algorithm 1, Table 2)."""

    K: int = 5  # outer iterations
    n_epoch: int = 1000  # max epochs per outer iteration, before early stopping
    batch_size: int = 128  # effective batch size (paper, Table 2)
    grad_accum_steps: int = 2  # micro_batch_size (64, real pipeline default) * this == batch_size
    lr: float = 1e-3
    patience: int = 15  # early-stopping patience, in epochs without improvement
    min_delta: float = 1e-3  # minimum validation-loss improvement to reset patience
    val_frac: float = 0.2  # fraction of sessions (by date) held out for validation
    safe_frac: float = 0.01  # fraction of each Delta_t_i kept clear of t_{i+1}
