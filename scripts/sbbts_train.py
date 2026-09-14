"""
Train the SBBTS drift network on the canonical real corpus, resumably.

Algorithm 1 at the paper configuration (`sbbts.config`), on the examples
``scripts.sbbts_panel`` wrote to ``data/sbbts/sbbts_training_examples.npz``,
batched by ``scripts.sbbts_batching`` at its defaults. Everything a resume
needs is checkpointed at the end of every epoch, so a kill costs at most the
epoch in flight:

    data/sbbts/checkpoints/latest.pt          most recent epoch (resume point)
    data/sbbts/checkpoints/epoch_k{k}_e{e}.pt the last three epochs
    data/sbbts/checkpoints/best.pt            best validation epoch of the
                                              outer iteration in progress
    data/sbbts/checkpoints/outer_k{k}.pt      weights at the end of each
                                              outer iteration (best epoch
                                              restored)

Each payload holds model, optimizer, torch CPU/CUDA RNG states, the
outer-iteration and epoch indices and the iteration's early-stopping state
(``sbbts.checkpoint``). Re-running the same command with ``latest.pt``
present continues from it; validation noise is re-derived from
(seed, outer iteration) so the resumed run validates on the same draws.

Progress is appended to ``data/sbbts/training_log.csv`` -- one row per
epoch with outer iteration, epoch, train/val loss, wall-clock and peak GPU
memory -- so it can be followed without attaching to the process.

Launch (survives a dropped terminal; stdout goes to the .log file):

    setsid nohup uv run python -m scripts.sbbts_train \\
        > data/sbbts/training.log 2>&1 < /dev/null &

``--subsample`` and ``--n-epoch`` exist only for smoke and resume tests and
are loudly labelled in the log; the paper run passes neither.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim

from sbbts.calibration import check_beta_constraint, derive_beta
from sbbts.checkpoint import (
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from sbbts.config import SBBTSModelConfig, SBBTSTrainingConfig
from sbbts.data import session_train_val_split
from sbbts.diagnostics import DegeneracyGate, TransformDiagnostics, transform_diagnostics
from sbbts.model import SBBTSDrift
from sbbts.training import (
    EarlyStopState,
    EpochLog,
    freeze_transform,
    make_validation_noise,
    train_outer_iteration,
)
from scripts.sbbts_batching import DEFAULT_BATCH_SIZE, DEFAULT_N_BUCKETS, PaddedBatch, build_batches
from scripts.sbbts_panel import EXAMPLES_OUTPUT_PATH, SBBTS_DIR, SessionExample

DEFAULT_CHECKPOINT_DIR = SBBTS_DIR / "checkpoints"
DEFAULT_LOG_CSV = SBBTS_DIR / "training_log.csv"
DEFAULT_DIAG_CSV = SBBTS_DIR / "diagnostics_log.csv"
KEEP_LAST_EPOCHS = 3
#: Endpoint-transform diagnostics run every this many epochs within an outer
#: iteration, plus at every outer-iteration boundary.
DIAG_EVERY_EPOCHS = 10
#: Every DIAG_BATCH_STRIDE-th validation batch, so the diagnostic spans all
#: length buckets at a fraction of a full validation pass.
DIAG_BATCH_STRIDE = 4
#: Identity of the training objective; a checkpoint written under a different
#: one is refused on resume.
TRANSFORM_MODE = "frozen-detached-ttilde"

DIAG_COLUMNS = [
    "timestamp",
    "outer_iter",
    "epoch",
    "subject",
    "var_dx",
    "var_dy",
    "ratio",
    "corr_send_bdx",
    "e_s_start2",
    "e_s_drift_u0_2",
    "n_intervals",
    "verdict",
    "message",
]


class DegeneracyAbort(RuntimeError):
    """Raised from the epoch callback when the gate trips; halts training cleanly."""


#: The pod is limited to 8 CPUs; torch must not spawn more intra-op threads.
MAX_THREADS = 8

LOG_COLUMNS = [
    "timestamp",
    "outer_iter",
    "epoch",
    "global_epoch",
    "train_loss",
    "val_loss",
    "best_val_this_outer",
    "epochs_without_improve",
    "wall_time_sec",
    "peak_gpu_alloc_mib",
    "peak_gpu_reserved_mib",
    "resumed",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_examples(path: Path) -> list[SessionExample]:
    """Read back the ragged (values, gaps, offsets) layout ``sbbts_panel.main`` writes."""
    z = np.load(path)
    values, gaps, offsets = z["values"], z["gaps"], z["offsets"]
    tickers, sessions = z["tickers"], z["sessions"]
    return [
        SessionExample(
            ticker=str(tickers[i]),
            session=pd.Timestamp(int(sessions[i]), tz="UTC"),
            values=values[offsets[i] : offsets[i + 1]],
            gaps=gaps[offsets[i] : offsets[i + 1]],
        )
        for i in range(len(offsets) - 1)
    ]


def to_device(batch: PaddedBatch, device: torch.device) -> PaddedBatch:
    return dataclasses.replace(
        batch,
        values=batch.values.to(device),
        gaps=batch.gaps.to(device),
        valid_mask=batch.valid_mask.to(device),
        lengths=batch.lengths.to(device),
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointWriter:
    """Rotates epoch checkpoints, keeps the best of the current outer iteration."""

    def __init__(self, directory: Path, model: SBBTSDrift, optimizer: optim.Optimizer, extra: dict):
        self.directory = directory
        self.model = model
        self.optimizer = optimizer
        self.extra = extra
        self.directory.mkdir(parents=True, exist_ok=True)

    def _write(self, name: str, **fields) -> Path:
        payload = build_checkpoint(self.model, self.optimizer, extra=self.extra, **fields)
        path = self.directory / name
        save_checkpoint(payload, path)
        return path

    def epoch(self, outer_iter: int, epoch: int, state: EarlyStopState, improved: bool) -> None:
        fields = dict(
            outer_iter=outer_iter,
            epoch=epoch,
            outer_iters_completed=outer_iter,
            stop_state=state,
        )
        self._write(f"epoch_k{outer_iter}_e{epoch}.pt", **fields)
        self._write("latest.pt", **fields)
        if improved:
            self._write("best.pt", **fields)
        self._prune()

    def outer_done(self, outer_iter: int, state: EarlyStopState) -> None:
        fields = dict(
            outer_iter=outer_iter,
            epoch=None,
            outer_iters_completed=outer_iter + 1,
            stop_state=state,
        )
        self._write(f"outer_k{outer_iter}.pt", **fields)
        self._write("latest.pt", **fields)

    def _prune(self) -> None:
        epoch_files = sorted(self.directory.glob("epoch_k*_e*.pt"), key=lambda p: p.stat().st_mtime)
        for stale in epoch_files[:-KEEP_LAST_EPOCHS]:
            stale.unlink()


def append_log_row(path: Path, row: dict, columns: list[str] = LOG_COLUMNS) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable SBBTS training at the paper config.")
    parser.add_argument("--examples", type=Path, default=EXAMPLES_OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-csv", type=Path, default=DEFAULT_LOG_CSV)
    parser.add_argument("--diag-csv", type=Path, default=DEFAULT_DIAG_CSV)
    parser.add_argument("--seed", type=int, default=0, help="init + noise seed")
    parser.add_argument("--threads", type=int, default=MAX_THREADS)
    parser.add_argument(
        "--accept-precision-change",
        action="store_true",
        help=(
            "resume a checkpoint whose training precision differs from the config's "
            "(a deliberate, logged switch); without it such a resume is refused"
        ),
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="TEST ONLY: train/val examples capped to this many each (0 = full corpus)",
    )
    parser.add_argument(
        "--n-epoch",
        type=int,
        default=0,
        help="TEST ONLY: override n_epoch per outer iteration (0 = config value)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(min(args.threads, MAX_THREADS))
    if not torch.cuda.is_available():
        sys.exit("CUDA is required for the paper-config run.")
    device = torch.device("cuda")

    model_cfg = SBBTSModelConfig()
    train_cfg = SBBTSTrainingConfig()
    test_mode = bool(args.subsample or args.n_epoch)
    if args.n_epoch:
        train_cfg = dataclasses.replace(train_cfg, n_epoch=args.n_epoch)
    if test_mode:
        log(
            f"*** TEST MODE: subsample={args.subsample or 'full'}, n_epoch={train_cfg.n_epoch} "
            "-- NOT the paper configuration ***"
        )

    log(
        f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}; threads={torch.get_num_threads()}"
    )
    log(f"model config: {dataclasses.asdict(model_cfg)}")
    log(f"training config: {dataclasses.asdict(train_cfg)}")
    gate = DegeneracyGate()
    log(
        f"endpoint transform: {TRANSFORM_MODE} -- for k >= 1, Y^k is built from the weights that "
        "ended outer iteration k-1, frozen and detached, at t~ = t_{i+1} - safe_frac*dt. "
        "NOTE: k >= 1 validation losses are therefore honest regressions onto fixed targets; "
        "expect them HIGHER than the no-detach run's 4.64 (which was target shrinkage). "
        "That is the correct outcome, not a failure."
    )
    log(
        f"degeneracy gate (validation, every {DIAG_EVERY_EPOCHS} epochs and at each outer "
        f"boundary): ABORT if Var(dY)/Var(dX) < {gate.abort_ratio_below} and corr(s_end, beta*dX) "
        f"> {gate.abort_corr_above}, or if Var(dY)/Var(dX) > {gate.abort_ratio_above}; WARN if ratio "
        f"< {gate.warn_ratio_below}, corr > {gate.warn_corr_above} or ratio > {gate.warn_ratio_above}."
    )
    log(
        f"precision: {train_cfg.precision} "
        + (
            "(torch.autocast bfloat16 around the loss forward/backward and validation; "
            "weights, gradients and Adam state fp32)"
            if train_cfg.precision == "bf16"
            else "(no autocast)"
        )
    )

    examples = load_examples(args.examples)
    log(
        f"examples: {len(examples):,} from {args.examples} "
        f"(sha256 {file_sha256(args.examples)[:16]}..., "
        f"mtime {datetime.fromtimestamp(args.examples.stat().st_mtime):%Y-%m-%d %H:%M})"
    )
    beta = derive_beta(examples)
    assert check_beta_constraint(examples, beta)
    log(f"beta = {beta}")

    train_examples, val_examples = session_train_val_split(examples, train_cfg.val_frac)
    if args.subsample:
        rng = np.random.default_rng(args.seed)
        pick = lambda pool, n: [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]  # noqa: E731
        train_examples = pick(train_examples, min(args.subsample, len(train_examples)))
        val_examples = pick(val_examples, min(args.subsample, len(val_examples)))
    train_batches = [
        to_device(b, device)
        for b in build_batches(train_examples, DEFAULT_BATCH_SIZE, DEFAULT_N_BUCKETS)
    ]
    val_batches = [
        to_device(b, device)
        for b in build_batches(val_examples, DEFAULT_BATCH_SIZE, DEFAULT_N_BUCKETS)
    ]
    log(
        f"split: {len(train_examples):,} train / {len(val_examples):,} val examples -> "
        f"{len(train_batches)} train micro-batches of {DEFAULT_BATCH_SIZE} "
        f"({-(-len(train_batches) // train_cfg.grad_accum_steps)} optimizer steps, "
        f"effective batch {DEFAULT_BATCH_SIZE * train_cfg.grad_accum_steps}) + "
        f"{len(val_batches)} val micro-batches per epoch"
    )

    torch.manual_seed(args.seed)
    model = SBBTSDrift(**dataclasses.asdict(model_cfg)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr)
    log(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ---- resume ---------------------------------------------------------------
    latest = args.checkpoint_dir / "latest.pt"
    k_start, epoch_start, state = 0, 0, EarlyStopState()
    resumed = False
    if latest.exists():
        payload = load_checkpoint(latest, device)
        saved_train_cfg = dict(payload.get("training_config") or {})
        # Checkpoints written before `precision` existed ran without autocast.
        saved_train_cfg.setdefault("precision", "fp32")
        precision_change = None
        if saved_train_cfg["precision"] != train_cfg.precision:
            if not args.accept_precision_change:
                sys.exit(
                    f"Refusing to resume {latest}: it was written at precision "
                    f"{saved_train_cfg['precision']!r} but the config says "
                    f"{train_cfg.precision!r}. Pass --accept-precision-change to switch "
                    "deliberately."
                )
            precision_change = (saved_train_cfg["precision"], train_cfg.precision)
            saved_train_cfg["precision"] = train_cfg.precision  # accepted; compare the rest
        mismatches = {
            name: (saved, current)
            for name, saved, current in (
                ("model_config", payload.get("model_config"), dataclasses.asdict(model_cfg)),
                ("training_config", saved_train_cfg, dataclasses.asdict(train_cfg)),
                ("test_mode", payload.get("test_mode"), test_mode),
                ("seed", payload.get("seed"), args.seed),
                ("transform", payload.get("transform"), TRANSFORM_MODE),
            )
            if saved != current
        }
        if mismatches:
            sys.exit(
                f"Refusing to resume {latest}: it was written under a different setup: {mismatches}"
            )
        state = restore_training_state(payload, model, optimizer)
        if precision_change is not None:
            log(
                f"PRECISION CHANGE accepted at this resume: {precision_change[0]} -> "
                f"{precision_change[1]}. Every checkpoint written from here on records "
                f"{precision_change[1]}, so a later resume will be held to it."
            )
        if payload["epoch"] is None:
            k_start, epoch_start, state = payload["outer_iters_completed"], 0, EarlyStopState()
        else:
            k_start, epoch_start = payload["outer_iter"], payload["epoch"] + 1
        resumed = True
        log(
            f"RESUMED from {latest}: outer_iter={payload['outer_iter']}, "
            f"last completed epoch={payload['epoch']}, "
            f"outer_iters_completed={payload['outer_iters_completed']} -> "
            f"continuing at k={k_start}, epoch={epoch_start}"
        )
    else:
        log(f"fresh start; checkpoints -> {args.checkpoint_dir}")

    extra = {
        "model_config": dataclasses.asdict(model_cfg),
        "training_config": dataclasses.asdict(train_cfg),
        "beta": beta,
        "seed": args.seed,
        "examples_path": str(args.examples),
        "test_mode": test_mode,
        "transform": TRANSFORM_MODE,
        "transform_state": None,  # theta_k's weights while an outer iteration k >= 1 is running
    }
    writer = CheckpointWriter(args.checkpoint_dir, model, optimizer, extra)
    diag_batches = val_batches[::DIAG_BATCH_STRIDE]

    def run_diagnostics(subject_model: SBBTSDrift, k: int, epoch: int | None, subject: str) -> None:
        """Compute, log, and gate the endpoint-transform diagnostics; raise on abort."""
        d: TransformDiagnostics = transform_diagnostics(
            subject_model, diag_batches, beta, train_cfg.safe_frac
        )
        verdict, message = gate.verdict(d)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "outer_iter": k,
            "epoch": "" if epoch is None else epoch,
            "subject": subject,
            **{key: (f"{v:.6f}" if isinstance(v, float) else v) for key, v in d.as_dict().items()},
            "verdict": verdict,
            "message": message,
        }
        append_log_row(args.diag_csv, row, DIAG_COLUMNS)
        log(
            f"diag[{subject}] k={k} epoch={epoch}: Var(dX)={d.var_dx:.3f} Var(dY)={d.var_dy:.3f} "
            f"ratio={d.ratio:.3f} corr(s_end,b*dX)={d.corr_send_bdx:.3f} "
            f"E[s_start^2]={d.e_s_start2:.3f} E[s_u0^2]={d.e_s_drift_u0_2:.3f} -> {verdict.upper()}"
            + (f": {message}" if message else "")
        )
        if verdict == "abort":
            raise DegeneracyAbort(f"[{subject}] k={k} epoch={epoch}: {message}")

    # Resume mid-iteration: theta_k comes back from the checkpoint.
    transform = None
    if latest.exists() and k_start >= 1 and epoch_start > 0:
        saved_transform = payload.get("transform_state")
        if saved_transform is None:
            sys.exit(
                f"Cannot resume k={k_start} mid-iteration: {latest} carries no frozen transform."
            )
        transform = freeze_transform(model)
        transform.load_state_dict(saved_transform)
        log(f"restored frozen transform theta_{k_start} from the checkpoint")
    global_epoch = [0]
    if args.log_csv.exists():
        with open(args.log_csv) as f:
            global_epoch[0] = max(0, sum(1 for _ in f) - 1)

    for k in range(k_start, train_cfg.K):
        if k != k_start:
            epoch_start, state = 0, EarlyStopState()
        if k >= 1 and (k != k_start or epoch_start == 0):
            # The weights that ended iteration k-1 (best epoch restored) become theta_k.
            transform = freeze_transform(model)
        extra["transform_state"] = None if transform is None else transform.state_dict()
        val_noise = make_validation_noise(
            val_batches, seed=args.seed + k, state_dim=model.state_dim
        )
        log(
            f"=== outer iteration {k + 1}/{train_cfg.K} (k={k}), starting at epoch {epoch_start} ==="
        )
        if k >= 1 and epoch_start == 0:
            log("NOTE: validation losses from here on regress onto frozen-transform targets.")
            run_diagnostics(transform, k, None, "frozen_transform_theta_k")
        torch.cuda.reset_peak_memory_stats()

        def on_checkpoint(entry: EpochLog, st: EarlyStopState, k=k) -> None:
            nonlocal resumed
            improved = st.best_epoch == entry.epoch
            writer.epoch(k, entry.epoch, st, improved)
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "outer_iter": k,
                "epoch": entry.epoch,
                "global_epoch": global_epoch[0],
                "train_loss": f"{entry.train_loss:.6f}",
                "val_loss": "" if entry.val_loss is None else f"{entry.val_loss:.6f}",
                "best_val_this_outer": f"{st.best_monitored:.6f}",
                "epochs_without_improve": st.epochs_without_improve,
                "wall_time_sec": f"{entry.wall_time_sec:.2f}",
                "peak_gpu_alloc_mib": f"{torch.cuda.max_memory_allocated() / 2**20:.0f}",
                "peak_gpu_reserved_mib": f"{torch.cuda.max_memory_reserved() / 2**20:.0f}",
                "resumed": int(resumed),
            }
            append_log_row(args.log_csv, row)
            global_epoch[0] += 1
            resumed = False
            log(
                f"k={k} epoch {entry.epoch + 1}/{train_cfg.n_epoch}: train={entry.train_loss:.6f} "
                f"val={entry.val_loss:.6f} best={st.best_monitored:.6f} "
                f"(no-improve {st.epochs_without_improve}/{train_cfg.patience}) "
                f"{entry.wall_time_sec:.1f}s peak {row['peak_gpu_alloc_mib']} MiB"
            )
            torch.cuda.reset_peak_memory_stats()
            if (entry.epoch + 1) % DIAG_EVERY_EPOCHS == 0:
                # The live weights are the candidate for theta_{k+1}: gate them.
                run_diagnostics(model, k, entry.epoch, "live_model_as_next_transform")

        started = time.perf_counter()
        try:
            logs = train_outer_iteration(
                model,
                optimizer,
                train_batches,
                val_batches,
                beta,
                k,
                train_cfg,
                val_noise=val_noise,
                state=state,
                start_epoch=epoch_start,
                on_checkpoint=on_checkpoint,
                transform=transform,
            )
        except DegeneracyAbort as exc:
            log(f"ABORTED by the degeneracy gate: {exc}")
            log("The last epoch's checkpoint is on disk; the run did not continue.")
            (args.checkpoint_dir / "ABORTED").write_text(f"{datetime.now().isoformat()} {exc}\n")
            sys.exit(2)
        # The best-restored weights are the candidate theta_{k+1}: gate them before moving on.
        run_diagnostics(model, k, None, "end_of_outer_iteration_best_weights")
        writer.outer_done(k, state)
        log(
            f"=== outer iteration k={k} done: {len(logs)} epochs this session in "
            f"{(time.perf_counter() - started) / 3600:.2f} h; best val {state.best_monitored:.6f} "
            f"at epoch {state.best_epoch}; weights restored to best; saved outer_k{k}.pt ==="
        )

    log("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
