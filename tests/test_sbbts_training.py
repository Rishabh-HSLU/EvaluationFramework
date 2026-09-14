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
import pytest
import torch
from torch import optim

from sbbts.calibration import derive_beta
from sbbts.checkpoint import build_checkpoint, restore_training_state
from sbbts.config import SBBTSTrainingConfig
from sbbts.diagnostics import DegeneracyGate, TransformDiagnostics, transform_diagnostics
from sbbts.model import SBBTSDrift
from sbbts.training import (
    EarlyStopState,
    compute_loss,
    evaluate,
    freeze_transform,
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

    transform = freeze_transform(model)
    for k in (0, 1):  # exercise both the identity-map path and the frozen-transform path
        optimizer.zero_grad()
        loss = compute_loss(model, batches[0], beta, k, safe_frac=0.01, transform=transform)
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
        model,
        optimizer,
        batches,
        batches,
        beta,
        k=1,
        config=config,
        state=stale,
        transform=freeze_transform(model),
    )
    assert len(logs) == 1  # one non-improvement on a meaningless best -> stop


def test_validation_loss_is_deterministic_with_fixed_noise() -> None:
    examples, beta = _corpus_and_beta([20, 25, 18, 30])
    batches = build_batches(examples, batch_size=4, n_buckets=1)
    model = _tiny_model()
    noise = make_validation_noise(batches, seed=7)
    transform = freeze_transform(model)

    fixed = [
        evaluate(model, batches, beta, k=1, safe_frac=0.01, noise=noise, transform=transform)
        for _ in range(3)
    ]
    assert fixed[0] == fixed[1] == fixed[2]

    fresh = [
        evaluate(model, batches, beta, k=1, safe_frac=0.01, transform=transform) for _ in range(3)
    ]
    assert len(set(fresh)) > 1

    again = make_validation_noise(batches, seed=7)
    assert all(torch.equal(a, b) for a, b in zip(noise.u_frac, again.u_frac, strict=True))


def test_resume_from_checkpoint_matches_uninterrupted_run() -> None:
    """Two epochs, checkpoint, restore into fresh objects, two more == four straight."""
    import io

    examples, beta = _corpus_and_beta([20, 25, 18, 30, 22, 28])
    train_batches = build_batches(examples[:4], batch_size=2, n_buckets=1)
    val_batches = build_batches(examples[4:], batch_size=2, n_buckets=1)
    config = SBBTSTrainingConfig(n_epoch=4, patience=100, min_delta=0.0, grad_accum_steps=2)
    noise = make_validation_noise(val_batches, seed=3)

    def fresh():
        torch.manual_seed(11)
        model = _tiny_model()
        return model, optim.Adam(model.parameters(), lr=config.lr)

    torch.manual_seed(12)
    transform = freeze_transform(_tiny_model())  # theta_1, shared by every path

    # Uninterrupted reference.
    model_a, opt_a = fresh()
    torch.manual_seed(5)
    logs_a = train_outer_iteration(
        model_a,
        opt_a,
        train_batches,
        val_batches,
        beta,
        1,
        config,
        val_noise=noise,
        transform=transform,
    )

    # Interrupted after epoch index 1, checkpointed, resumed into new objects.
    model_b, opt_b = fresh()
    torch.manual_seed(5)
    captured = {}

    def on_checkpoint(log, state):
        if log.epoch == 1:
            payload = build_checkpoint(
                model_b, opt_b, outer_iter=1, epoch=1, outer_iters_completed=1, stop_state=state
            )
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            captured["bytes"] = buffer.getvalue()

    half = SBBTSTrainingConfig(n_epoch=2, patience=100, min_delta=0.0, grad_accum_steps=2)
    logs_b1 = train_outer_iteration(
        model_b,
        opt_b,
        train_batches,
        val_batches,
        beta,
        1,
        half,
        val_noise=noise,
        on_checkpoint=on_checkpoint,
        transform=transform,
    )

    model_c, opt_c = fresh()  # different weights until restored
    payload = torch.load(io.BytesIO(captured["bytes"]), weights_only=False)
    state = restore_training_state(payload, model_c, opt_c)
    logs_b2 = train_outer_iteration(
        model_c,
        opt_c,
        train_batches,
        val_batches,
        beta,
        1,
        config,
        val_noise=noise,
        state=state,
        start_epoch=payload["epoch"] + 1,
        transform=transform,
    )

    logs_b = logs_b1 + logs_b2
    assert [e.epoch for e in logs_b] == [0, 1, 2, 3]
    assert [e.train_loss for e in logs_b] == [e.train_loss for e in logs_a]
    assert [e.val_loss for e in logs_b] == [e.val_loss for e in logs_a]
    sa, sc = model_a.state_dict(), model_c.state_dict()
    assert all(torch.equal(sa[name], sc[name]) for name in sa)


def test_frozen_transform_gets_no_gradient_and_fixes_the_targets() -> None:
    """theta_k neither receives gradients nor moves, so k>=1 targets are fixed within the iteration."""
    examples, beta = _corpus_and_beta([20, 25, 18, 30])
    batches = build_batches(examples, batch_size=4, n_buckets=1)
    model = _tiny_model()
    transform = freeze_transform(model)
    before = {name: p.clone() for name, p in transform.state_dict().items()}
    assert all(not p.requires_grad for p in transform.parameters())

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    config = SBBTSTrainingConfig(n_epoch=3, patience=100, min_delta=0.0, grad_accum_steps=1)
    train_outer_iteration(model, optimizer, batches, [], beta, 1, config, transform=transform)

    after = transform.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    live = model.state_dict()
    assert any(not torch.equal(before[name], live[name]) for name in before)  # the live model moved

    # And k >= 1 without a transform is an error, never a silent self-transform.
    with pytest.raises(ValueError):
        compute_loss(model, batches[0], beta, 1, safe_frac=0.01)


def test_degeneracy_gate_trips_on_the_failed_run_signature_only() -> None:
    def diag(ratio, corr):
        return TransformDiagnostics(
            var_dx=1.0,
            var_dy=ratio,
            ratio=ratio,
            corr_send_bdx=corr,
            e_s_start2=0.05,
            e_s_drift_u0_2=0.05,
            n_intervals=1000,
        )

    gate = DegeneracyGate()
    assert gate.verdict(diag(0.44, 0.77))[0] == "abort"  # failed run, k=1
    assert gate.verdict(diag(0.38, 0.81))[0] == "abort"  # failed run, k=4
    assert gate.verdict(diag(0.78, 0.52))[0] == "warn"  # failed run's k=0 model: early warning
    assert gate.verdict(diag(2.5, 0.0))[0] == "abort"  # transform inflating the targets
    assert gate.verdict(diag(1.0, 0.1))[0] == "ok"
    assert gate.verdict(diag(0.5, 0.2))[0] == "warn"  # low ratio alone is not the joint signature
    assert gate.verdict(diag(float("nan"), 0.0))[0] == "abort"


def test_transform_diagnostics_identity_when_drift_is_zero() -> None:
    """A network with zero output is the identity transform: ratio 1, no correlation."""
    examples, beta = _corpus_and_beta([20, 25, 18, 30])
    batches = build_batches(examples, batch_size=4, n_buckets=1)
    model = _tiny_model()
    with torch.no_grad():
        model.output_proj.net[-1].weight.zero_()
        model.output_proj.net[-1].bias.zero_()
    d = transform_diagnostics(model, batches, beta, safe_frac=0.01)
    assert abs(d.ratio - 1.0) < 1e-6
    assert d.e_s_start2 == 0.0 and d.e_s_drift_u0_2 == 0.0
    assert d.n_intervals > 0
