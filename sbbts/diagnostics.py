"""Degeneracy diagnostics for the SBBTS endpoint transform, used as a gate.

The first full training run (2026-09-13, no-detach objective as in the
reference implementation) converged to a degenerate optimum: the endpoint
map s(t~_{i+1}, X_{i+1}, c_i) learned to be ~ 0.8 * beta * Delta X, so that
the auxiliary increments Delta Y = Delta X - Delta s / beta carried only
~40 % of the real increments' variance, while the bridge drift proper was
~ 0. On validation data the signature was (per outer-iteration
checkpoint; see scripts/reasoning.md, "Frozen endpoint transform"):

    k        Var(dY)/Var(dX)   corr(s_end, beta*dX)   E[s_start^2]
    0            0.78                 0.52                0.07
    1            0.44                 0.77                0.07
    2            0.36                 0.81                0.08
    4            0.38                 0.81                0.04

`transform_diagnostics` computes those quantities for any candidate
network treated as an endpoint transform; `DegeneracyGate` turns them
into an abort decision so a repeat is caught within an outer iteration,
not after eleven hours.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from scripts.sbbts_batching import PaddedBatch

from .data import to_model_inputs
from .model import SBBTSDrift


@dataclass(frozen=True)
class TransformDiagnostics:
    """Endpoint-transform statistics on real (validation) inputs, Delta_t = 1 intervals only."""

    var_dx: float  # Var(X_{i+1} - X_i)
    var_dy: float  # Var(Y_{i+1} - Y_i) under the candidate transform
    ratio: float  # var_dy / var_dx: 1 = transform leaves increments alone
    corr_send_bdx: float  # corr(s(t~_{i+1}, X_{i+1}, c_i), beta * dX): the absorption signature
    e_s_start2: float  # E[s(t_i, X_i, c_i)^2]: the map at the interval start
    e_s_drift_u0_2: float  # E[s(t_i, Y_i, c_i)^2]: the drift the sampler would apply at u = 0
    n_intervals: int

    def as_dict(self) -> dict:
        return {
            "var_dx": self.var_dx,
            "var_dy": self.var_dy,
            "ratio": self.ratio,
            "corr_send_bdx": self.corr_send_bdx,
            "e_s_start2": self.e_s_start2,
            "e_s_drift_u0_2": self.e_s_drift_u0_2,
            "n_intervals": self.n_intervals,
        }


@torch.no_grad()
def transform_diagnostics(
    candidate: SBBTSDrift,
    batches: Sequence[PaddedBatch],
    beta: float,
    safe_frac: float,
) -> TransformDiagnostics:
    """Evaluate `candidate` as the endpoint transform on `batches` (fp32, eval mode).

    Restricted to consecutive-minute intervals so Var(dX) is a single
    quantity rather than a mixture over gap lengths.
    """
    was_training = candidate.training
    candidate.eval()
    dx_all, dy_all, s_end_all, s_start_all, s_u0_all = [], [], [], [], []
    for batch in batches:
        y, times, mask = to_model_inputs(batch)
        context = candidate.encode_history(y, times, mask)[:, :-1, :]
        t_i, t_ip1 = times[:, :-1], times[:, 1:]
        delta_t = t_ip1 - t_i
        t_tilde = t_ip1 - safe_frac * delta_t
        x_i, x_ip1 = y[:, :-1, :], y[:, 1:, :]
        s_start = candidate.drift(t_i.unsqueeze(-1), x_i, context)
        s_end = candidate.drift(t_tilde.unsqueeze(-1), x_ip1, context)
        y_i = x_i - s_start / beta
        y_ip1 = x_ip1 - s_end / beta
        s_u0 = candidate.drift(t_i.unsqueeze(-1), y_i, context)
        keep = (mask[:, :-1] & mask[:, 1:] & (delta_t == 1)).unsqueeze(-1)
        dx_all.append((x_ip1 - x_i)[keep])
        dy_all.append((y_ip1 - y_i)[keep])
        s_end_all.append(s_end[keep])
        s_start_all.append(s_start[keep])
        s_u0_all.append(s_u0[keep])
    if was_training:
        candidate.train()
    dx = torch.cat(dx_all).double()
    dy = torch.cat(dy_all).double()
    s_end = torch.cat(s_end_all).double()
    s_start = torch.cat(s_start_all).double()
    s_u0 = torch.cat(s_u0_all).double()
    var_dx = float(dx.var())
    var_dy = float(dy.var())
    corr = float(torch.corrcoef(torch.stack([s_end, beta * dx]))[0, 1]) if len(dx) > 1 else 0.0
    return TransformDiagnostics(
        var_dx=var_dx,
        var_dy=var_dy,
        ratio=var_dy / var_dx if var_dx > 0 else float("nan"),
        corr_send_bdx=corr,
        e_s_start2=float((s_start**2).mean()),
        e_s_drift_u0_2=float((s_u0**2).mean()),
        n_intervals=int(dx.numel()),
    )


@dataclass(frozen=True)
class DegeneracyGate:
    """Abort rule on `TransformDiagnostics`.

    Abort when the failed run's joint signature appears -- the transform
    removing a large share of the increment variance while its endpoint
    value tracks beta * dX -- or when the transform inflates the targets
    instead (a frozen network extrapolated into noise). Thresholds sit
    between the healthy region (ratio ~ 1, |corr| small) and the failed
    run's k=1 state (0.44 / 0.77); the failed run's k=0 model (0.78 / 0.52)
    is inside the warning band but not the abort region, which is the
    intended early-warning behaviour.
    """

    abort_ratio_below: float = 0.6
    abort_corr_above: float = 0.6
    abort_ratio_above: float = 2.0
    warn_ratio_below: float = 0.7
    warn_corr_above: float = 0.5
    warn_ratio_above: float = 1.5

    def verdict(self, d: TransformDiagnostics) -> tuple[str, str]:
        """Return ("abort" | "warn" | "ok", message)."""
        if d.ratio != d.ratio:  # NaN
            return "abort", "Var(dY)/Var(dX) is NaN: non-finite transform output."
        if d.ratio < self.abort_ratio_below and d.corr_send_bdx > self.abort_corr_above:
            return "abort", (
                f"endpoint map is absorbing the increments: Var(dY)/Var(dX)={d.ratio:.3f} "
                f"< {self.abort_ratio_below} and corr(s_end, beta*dX)={d.corr_send_bdx:.3f} "
                f"> {self.abort_corr_above} (failed run: 0.44 / 0.77 at k=1)."
            )
        if d.ratio > self.abort_ratio_above:
            return "abort", (
                f"endpoint map is inflating the targets: Var(dY)/Var(dX)={d.ratio:.3f} "
                f"> {self.abort_ratio_above}."
            )
        warnings = []
        if d.ratio < self.warn_ratio_below:
            warnings.append(f"ratio {d.ratio:.3f} < {self.warn_ratio_below}")
        if d.corr_send_bdx > self.warn_corr_above:
            warnings.append(f"corr {d.corr_send_bdx:.3f} > {self.warn_corr_above}")
        if d.ratio > self.warn_ratio_above:
            warnings.append(f"ratio {d.ratio:.3f} > {self.warn_ratio_above}")
        if warnings:
            return "warn", "; ".join(warnings)
        return "ok", ""


@dataclass
class CollapseGate:
    """Abort rule on the training loss within one outer iteration.

    Both observed collapses (bf16, k=0, epochs 77 and 12) shared one
    signature: the training loss jumped ~ +0.5 above its running minimum
    within two epochs and then stayed flat there (3.80 +- 0.01) for the
    rest of the iteration, never recovering. Anything that rises
    `rise_above_min` or more above the iteration's running minimum and stays
    there for `persist_epochs` consecutive epochs is a collapse; ordinary
    epoch-to-epoch wobble is +-0.01 on the training loss.

    Reset at every outer iteration: the running minimum of k is not
    comparable to k+1's targets.
    """

    rise_above_min: float = 0.15
    persist_epochs: int = 3
    running_min: float = float("inf")
    consecutive_high: int = 0

    def update(self, train_loss: float) -> tuple[bool, str]:
        """Record an epoch's training loss; return (abort, message)."""
        if train_loss != train_loss:  # NaN
            return True, "training loss is NaN."
        if train_loss >= self.running_min + self.rise_above_min:
            self.consecutive_high += 1
        else:
            self.consecutive_high = 0
        self.running_min = min(self.running_min, train_loss)
        if self.consecutive_high >= self.persist_epochs:
            return True, (
                f"training loss {train_loss:.4f} has stayed >= {self.rise_above_min} above this "
                f"outer iteration's minimum ({self.running_min:.4f}) for {self.consecutive_high} "
                "consecutive epochs (collapse signature: bf16 runs 2026-09-13/14)."
            )
        return False, ""
