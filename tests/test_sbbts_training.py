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
from sbbts.training import compute_loss, train_outer_iteration
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
