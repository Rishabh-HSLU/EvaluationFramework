"""SBBTS training loop: Algorithm 1, adapted for irregular grids.

Loss (Eq. 4.1) and Brownian-bridge sampling (Eq. 4.2) are implemented
per-interval with each interval's own Delta_t_i, rather than assuming a
shared global time horizon T -- the paper's own experiments use a regular
grid (constant Delta_t) and its reference implementation hard-codes a
single T shared across all steps, which does not hold for our data
(missing minutes are dropped, producing irregular gaps).

Verified against the paper's reference implementation
(https://github.com/alexouadi/SBBTS, training/training_sbbts_dsbm.py):

  - Y^0 = identity (k == 0 skips the network entirely: Y_ti = X_ti).
  - For k > 0, the endpoint transform Y_ti = X_ti - s_theta(...)/beta is
    NOT detached from the autograd graph: the loss backpropagates jointly
    through the transport-map construction (steps 5-6 of Algorithm 1) and
    the drift prediction (step 8), all through the same live parameters.
    This is a self-referential/bootstrapped regression, not a frozen
    target network.
  - Both endpoints of interval i (Y_ti and Y_ti+1) condition on the SAME
    context Phi_theta(X_{t0:ti}) -- i.e. history only up to t_i, matching
    Algorithm 1 lines 5-6 using Phi^k_theta(X^b_{t0:ti}) for both.
  - A small time margin keeps the sampled bridge time away from t_{i+1},
    where the regression target (Y_ti+1 - Y_t)/(t_{i+1} - t) is undefined.
    The reference code uses a fixed absolute margin (safe_t=1e-2) because
    its experiments fix a shared Delta_t; we use a margin proportional to
    each interval's own Delta_t_i (`safe_frac`) since our Delta_t_i vary.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import optim

from scripts.sbbts_batching import PaddedBatch

from .config import SBBTSTrainingConfig
from .data import to_model_inputs
from .model import SBBTSDrift


@dataclass
class EpochLog:
    outer_iter: int
    epoch: int
    train_loss: float
    val_loss: float | None
    wall_time_sec: float


def compute_loss(
    model: SBBTSDrift,
    batch: PaddedBatch,
    beta: float,
    k: int,
    safe_frac: float = 0.01,
) -> torch.Tensor:
    """Eq. 4.1 loss on one batch, for outer iteration k.

    Only intervals where both endpoints are real observations contribute
    (`valid_mask` gates padded positions out of the loss, on top of already
    being masked out of the encoder's attention).
    """
    y, times, valid_mask = to_model_inputs(batch)
    if y.shape[1] < 2:
        raise ValueError("A batch needs at least 2 positions to form one training interval.")

    context_full = model.encode_history(y, times, valid_mask)  # (B, L, d_model)
    context = context_full[:, :-1, :]  # context for intervals i = 0..L-2, i.e. Phi_theta(X_{t0:ti})

    t_i = times[:, :-1]
    t_ip1 = times[:, 1:]
    x_i = y[:, :-1, :]
    x_ip1 = y[:, 1:, :]

    if k == 0:
        y_i, y_ip1 = x_i, x_ip1
    else:
        s_i = model.drift(t_i.unsqueeze(-1), x_i, context)
        s_ip1 = model.drift(t_ip1.unsqueeze(-1), x_ip1, context)
        y_i = x_i - s_i / beta
        y_ip1 = x_ip1 - s_ip1 / beta

    delta_t = (t_ip1 - t_i).clamp(min=1e-8)  # (B, L-1)
    span = (delta_t * (1.0 - safe_frac)).clamp(min=1e-8)
    u = torch.rand_like(delta_t) * span  # t ~ U([t_i, t_i + span]) subset of [t_i, t_{i+1})

    frac = (u / delta_t).unsqueeze(-1)
    mu_t = (1.0 - frac) * y_i + frac * y_ip1
    sigma2 = (u * (delta_t - u) / delta_t).clamp(min=0.0).unsqueeze(-1)
    y_t = mu_t + torch.sqrt(sigma2) * torch.randn_like(y_i)

    t_sample = (t_i + u).unsqueeze(-1)
    prediction = model.drift(t_sample, y_t, context)

    remaining = (delta_t - u).clamp(min=1e-8).unsqueeze(-1)
    target = (y_ip1 - y_t) / remaining

    per_interval_loss = ((prediction - target) ** 2).sum(dim=-1)  # (B, L-1)
    interval_mask = (valid_mask[:, :-1] & valid_mask[:, 1:]).to(per_interval_loss.dtype)
    denom = interval_mask.sum().clamp(min=1.0)
    return (per_interval_loss * interval_mask).sum() / denom


def evaluate(
    model: SBBTSDrift, batches: Sequence[PaddedBatch], beta: float, k: int, safe_frac: float
) -> float | None:
    if not batches:
        return None
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            total += compute_loss(model, batch, beta, k, safe_frac).item()
    return total / len(batches)


def train_outer_iteration(
    model: SBBTSDrift,
    optimizer: optim.Optimizer,
    train_batches: Sequence[PaddedBatch],
    val_batches: Sequence[PaddedBatch],
    beta: float,
    k: int,
    config: SBBTSTrainingConfig,
    on_epoch: Callable[[EpochLog], None] | None = None,
) -> list[EpochLog]:
    """Train to convergence for one outer iteration (Algorithm 1's inner `repeat`)."""
    best_monitored = None
    best_state = None
    epochs_without_improve = 0
    logs: list[EpochLog] = []

    accum_steps = max(1, config.grad_accum_steps)

    for epoch in range(config.n_epoch):
        start = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(train_batches):
            loss = compute_loss(model, batch, beta, k, config.safe_frac)
            (loss / accum_steps).backward()
            total_train_loss += loss.item()
            if (i + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        if len(train_batches) % accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        train_loss = total_train_loss / max(len(train_batches), 1)

        val_loss = evaluate(model, val_batches, beta, k, config.safe_frac)
        wall_time_sec = time.perf_counter() - start

        log = EpochLog(k, epoch, train_loss, val_loss, wall_time_sec)
        logs.append(log)
        if on_epoch is not None:
            on_epoch(log)

        monitored = val_loss if val_loss is not None else train_loss
        if best_monitored is None or monitored < best_monitored - config.min_delta:
            best_monitored = monitored
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return logs


def train_sbbts(
    model: SBBTSDrift,
    train_batches: Sequence[PaddedBatch],
    val_batches: Sequence[PaddedBatch],
    beta: float,
    config: SBBTSTrainingConfig,
    on_epoch: Callable[[EpochLog], None] | None = None,
) -> list[EpochLog]:
    """Algorithm 1 in full: K outer iterations, each trained to convergence.

    One Adam optimizer instance persists across all K outer iterations
    (matching the paper's reference implementation), so momentum is not
    reset when the outer-iteration index advances.

    Scope note: this entry point exists and is unit-tested for a single
    step and a tiny overfit case, but is deliberately not run to
    completion in this pass -- see the task's scope boundary.
    """
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    all_logs: list[EpochLog] = []
    for k in range(config.K):
        all_logs.extend(
            train_outer_iteration(
                model, optimizer, train_batches, val_batches, beta, k, config, on_epoch
            )
        )
    return all_logs


def make_progress_logger(
    config: SBBTSTrainingConfig, project_after: int = 3
) -> Callable[[EpochLog], None]:
    """Prints per-epoch wall-clock time, and a total-runtime projection once
    `project_after` epochs have completed.

    Exists so we know early if the full K x n_epoch schedule is
    impractical on this hardware, rather than discovering it hours in.
    """
    epoch_times: list[float] = []

    def _log(entry: EpochLog) -> None:
        epoch_times.append(entry.wall_time_sec)
        msg = (
            f"[outer {entry.outer_iter + 1}/{config.K}] epoch {entry.epoch + 1}: "
            f"train_loss={entry.train_loss:.6f}"
        )
        if entry.val_loss is not None:
            msg += f" val_loss={entry.val_loss:.6f}"
        msg += f" wall_time={entry.wall_time_sec:.3f}s"

        if len(epoch_times) == project_after:
            mean_epoch_time = sum(epoch_times) / len(epoch_times)
            total_epochs = config.K * config.n_epoch
            projected_hours = mean_epoch_time * total_epochs / 3600.0
            msg += (
                f"\n  -> mean epoch time over first {project_after} epochs: "
                f"{mean_epoch_time:.3f}s; projected total for K={config.K} x "
                f"n_epoch={config.n_epoch} ({total_epochs} epochs, no early "
                f"stopping applied to this estimate): {projected_hours:.2f} h"
            )
        print(msg)

    return _log
