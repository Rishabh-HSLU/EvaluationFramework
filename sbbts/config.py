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

    # Numerical precision of the loss forward/backward: "fp32", or "bf16" for
    # torch.autocast(bfloat16) around compute_loss (weights, gradients and the
    # Adam state stay fp32; no loss scaling is needed for bf16). Not a
    # hyperparameter of Algorithm 1, but part of a run's identity: a resume
    # refuses a checkpoint written under a different precision unless told
    # otherwise. bf16 was validated against fp32 on 2026-09-13 (same initial
    # weights, same RNG seed, same fixed validation draws, 2,048/512-example
    # subsample, 4 epochs at k=0 then 4 at k=1): max |val-loss divergence|
    # 1.0e-3 and max |train-loss divergence| 1.2e-3 over the 8 epochs, against
    # a median epoch-to-epoch validation movement of 5.9e-3 and a
    # seed-to-seed spread (sd) of 3.7e-3 -- i.e. below the run's own noise
    # floor. Measured speed-up on the A16: 2.1x per optimizer step.
    precision: str = "bf16"
