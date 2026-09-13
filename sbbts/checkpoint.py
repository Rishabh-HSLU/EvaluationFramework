"""Checkpoint payloads for resumable SBBTS training.

A payload holds everything `scripts.sbbts_train` needs to continue a run
after a kill without losing more than the epoch in flight: model and
optimizer state, the position in Algorithm 1 (outer-iteration index and
epoch index), the outer iteration's `EarlyStopState` (including the best
weights it will restore), and the torch RNG states so that the bridge-time,
Gaussian and dropout draws continue from where they were.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import optim

from .model import SBBTSDrift
from .training import EarlyStopState

FORMAT_VERSION = 1


def build_checkpoint(
    model: SBBTSDrift,
    optimizer: optim.Optimizer,
    *,
    outer_iter: int,
    epoch: int | None,
    outer_iters_completed: int,
    stop_state: EarlyStopState,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint payload.

    `epoch` is the last completed epoch of `outer_iter`, or None when the
    payload marks the end of an outer iteration (then
    `outer_iters_completed == outer_iter + 1` and the model already holds
    that iteration's best weights).
    """
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "outer_iter": outer_iter,
        "epoch": epoch,
        "outer_iters_completed": outer_iters_completed,
        "stop_state": dataclasses.asdict(stop_state),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Write atomically: a kill mid-write never leaves a truncated file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_checkpoint(path: Path, device: torch.device | str) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_training_state(
    payload: dict[str, Any], model: SBBTSDrift, optimizer: optim.Optimizer
) -> EarlyStopState:
    """Load model, optimizer, RNG streams and the early-stopping state from `payload`."""
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng_state"]])
    return EarlyStopState(**payload["stop_state"])
