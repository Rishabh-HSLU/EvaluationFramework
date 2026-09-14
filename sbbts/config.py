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

    # Encoder-layer internals. Appendix B says only "an encoder-only
    # architecture from [19] with one layer" and Table 2 lists no dropout, so
    # the paper is silent on all three of these. The reference implementation
    # (alexouadi/SBBTS, models/encoder_only.py) builds
    # nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    # and inherits every other argument from PyTorch; these are those inherited
    # values, written down here so they are an explicit choice rather than a
    # library default that could drift with a torch upgrade. The reference
    # applies the dropout consistently: model.train() for every training epoch,
    # model.eval() for validation and generation. The FFN blocks (time, state,
    # output) have no dropout in the paper, the reference, or here.
    encoder_dropout: float = 0.1
    encoder_dim_feedforward: int = 2048
    encoder_activation: str = "relu"


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

    # Numerical precision of the loss forward/backward. Only "fp32" is
    # accepted. bf16 autocast was tested and rejected for instability: an
    # 8-epoch bf16-vs-fp32 validation on a 2,048/512-example subsample (same
    # init, seed and validation draws) passed at max val-loss divergence
    # 1.0e-3, but two full runs then collapsed under bf16 at k=0 -- at epoch
    # 77 (two epochs after a mid-run fp32-to-bf16 switch) and at epoch 12
    # (from a cold bf16 start) -- while 75 fp32 epochs at the same objective
    # showed none. The validation observed ~4,000 optimizer steps; the
    # collapses arrived after ~40,000 and ~6,000, a rare-step tail event the
    # check could not see. See scripts/reasoning.md, "Precision". The field
    # remains part of a checkpoint's identity so a resume can never mix
    # precisions.
    precision: str = "fp32"
