"""Training-loop tests: one step runs cleanly, loss overfits a tiny corpus.

Fixtures are directly-constructed `SessionExample` records fed through the
real `scripts.sbbts_batching.build_batches` -- not a standalone synthetic
corpus (see scripts/reasoning.md for why that was deleted).

Scope: this does not run the real K=5 x n_epoch=1000 schedule (see
scripts/reasoning.md and the task's scope boundary) -- only a single step
and a short single-outer-iteration overfit run, as explicit sanity checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import optim

from sbbts.calibration import derive_beta
from sbbts.config import SBBTSTrainingConfig
from sbbts.model import SBBTSDrift
from sbbts.training import (
    EarlyStopState,
    compute_loss,
    evaluate,
    make_validation_noise,
    train_outer_iteration,
    train_sbbts,
)
from scripts.sbbts_batching import build_batches
from scripts.sbbts_panel import GAP_DTYPE, VALUE_DTYPE, SessionExample

torch.manual_seed(0)


def _example(index: int, length: int) -> SessionExample:
    rng = np.random.default_rng(index)
    values = np.empty(length, dtype=VALUE_DTYPE)
    values[0] = 0.0
    values[1:] = np.cumsum(rng.normal(0.0, 1.0, size=length - 1))
    return SessionExample(
        ticker=f"T{index:04d}",
        session=pd.Timestamp("2026-01-05", tz="UTC"),
        values=values,
        gaps=rng.integers(1, 5, size=length).astype(GAP_DTYPE),
    )


def _tiny_model() -> SBBTSDrift:
    return SBBTSDrift(state_dim=1, d_model=16, n_heads=4, n_layers=1)


def _corpus_and_beta(lengths: list[int]) -> tuple[list[SessionExample], float]:
    examples = [_example(i, length) for i, length in enumerate(lengths)]
    beta = derive_beta(examples, safety_factor=2.0)
    return examples, beta


def test_one_training_step_runs_without_error_and_loss_is_finite() -> None:
    examples, beta = _corpus_and_beta([20, 25, 18, 30, 22, 28, 15, 24])
    batches = build_batches(examples, batch_size=8, n_buckets=2)
    model = _tiny_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for k in (0, 1):  # exercise both the identity-map path and the self-referential path
        optimizer.zero_grad()
        loss = compute_loss(model, batches[0], beta, k, safe_frac=0.01)
        assert torch.isfinite(loss)
        loss.backward()

        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert grad_norms, "expected at least one parameter to receive a gradient"
        assert all(torch.isfinite(torch.tensor(g)) for g in grad_norms)

        optimizer.step()


def test_loss_decreases_on_tiny_synthetic_overfit_case() -> None:
    examples, beta = _corpus_and_beta([20, 20, 20, 20])
    batches = build_batches(examples, batch_size=4, n_buckets=1)
    assert len(batches) == 1

    model = _tiny_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    config = SBBTSTrainingConfig(n_epoch=150, patience=150, min_delta=0.0, grad_accum_steps=1)

    logs = train_outer_iteration(model, optimizer, batches, [], beta, k=0, config=config)

    first_losses = [entry.train_loss for entry in logs[:5]]
    last_losses = [entry.train_loss for entry in logs[-5:]]
    assert sum(last_losses) / len(last_losses) < sum(first_losses) / len(first_losses)


def test_early_stopping_state_is_fresh_per_outer_iteration() -> None:
    """Patience is evaluated within one outer iteration only.

    A state that has already exhausted its patience is what k=0 might end
    with; k=1 must not inherit it. train_sbbts creates a fresh EarlyStopState
    per k, so every outer iteration starts at epoch 0 with a clean counter
    even when the previous one stopped early.
    """
    examples, beta = _corpus_and_beta([20, 25, 18, 30])
    batches = build_batches(examples, batch_size=4, n_buckets=1)
    model = _tiny_model()
    # patience=1 and a huge min_delta: no epoch ever counts as an improvement
    # after the first, so k=0 stops at epoch 2. If the counter leaked into
    # k=1, k=1 would stop after a single epoch (or before its first).
    config = SBBTSTrainingConfig(K=2, n_epoch=5, patience=1, min_delta=1e9, grad_accum_steps=1)
    logs = train_sbbts(model, batches, batches, beta, config)

    per_k = {k: [e.epoch for e in logs if e.outer_iter == k] for k in range(config.K)}
    assert per_k[0] == [0, 1]
    assert per_k[1] == [0, 1]  # started from scratch, not from an exhausted counter


def test_stale_state_from_another_iteration_would_stop_immediately() -> None:
    """Documents the failure mode the per-k reset prevents."""
    examples, beta = _corpus_and_beta([20, 25, 18, 30])
    batches = build_batches(examples, batch_size=4, n_buckets=1)
    model = _tiny_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    config = SBBTSTrainingConfig(n_epoch=5, patience=2, min_delta=0.0, grad_accum_steps=1)

    stale = EarlyStopState(best_monitored=-1e9, epochs_without_improve=1)  # from a "previous k"
    logs = train_outer_iteration(
        model, optimizer, batches, batches, beta, k=1, config=config, state=stale
    )
    assert len(logs) == 1  # one non-improvement on a meaningless best -> stop


def test_validation_loss_is_deterministic_with_fixed_noise() -> None:
    examples, beta = _corpus_and_beta([20, 25, 18, 30])
    batches = build_batches(examples, batch_size=4, n_buckets=1)
    model = _tiny_model()
    noise = make_validation_noise(batches, seed=7)

    fixed = [evaluate(model, batches, beta, k=1, safe_frac=0.01, noise=noise) for _ in range(3)]
    assert fixed[0] == fixed[1] == fixed[2]

    fresh = [evaluate(model, batches, beta, k=1, safe_frac=0.01) for _ in range(3)]
    assert len(set(fresh)) > 1

    again = make_validation_noise(batches, seed=7)
    assert all(torch.equal(a, b) for a, b in zip(noise.u_frac, again.u_frac, strict=True))
