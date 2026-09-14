"""
Generate the SBBTS synthetic panel from a trained checkpoint.

One path per (ticker, session) cell of the curated real panel is sampled
with ``sbbts.sampling.generate_paths`` on the full 390-bar session grid,
then ``scripts.sbbts_panel.assemble_price_panel`` rescales each path by its
ticker's training scale, anchors it on the session's first real print,
truncates early-close sessions and imposes real's NaN mask -- so the output
has the same clock, columns and missingness as ``gbm_prices.parquet`` and
``msv_prices.parquet``.

    uv run python -m scripts.sbbts_generate                 # full panel
    uv run python -m scripts.sbbts_generate --preview 512   # stats only, no file
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fineval.benchmark.config import CURATED_DIR
from sbbts.model import SBBTSDrift
from sbbts.sampling import generate_paths
from scripts.sbbts_panel import (
    REAL_PATH,
    SBBTS_DIR,
    SCALES_OUTPUT_PATH,
    SESSION_BARS,
    assemble_price_panel,
)

DEFAULT_CHECKPOINT = SBBTS_DIR / "checkpoints" / "outer_k4.pt"
OUTPUT_PATH = CURATED_DIR / "sbbts_prices.parquet"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the SBBTS synthetic price panel.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=0, help="Brownian-increment seed")
    parser.add_argument("--batch-paths", type=int, default=2048)
    parser.add_argument("--n-euler", type=int, default=50, help="N_pi, Euler steps per bar")
    parser.add_argument("--context-on", choices=("y", "x"), default="y")
    parser.add_argument("--retransform-start", action="store_true")
    parser.add_argument("--invert-iters", type=int, default=0)
    parser.add_argument(
        "--preview", type=int, default=0, help="sample this many paths, print stats, exit"
    )
    return parser


def load_model(checkpoint: Path, device: torch.device) -> tuple[SBBTSDrift, float, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SBBTSDrift(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, float(payload["beta"]), payload


def increment_stats(paths: torch.Tensor) -> dict:
    inc = paths[:, 1:, 0] - paths[:, :-1, 0]
    flat = inc.flatten().double()
    return {
        "finite_frac": float(torch.isfinite(flat).float().mean()),
        "inc_mean": float(flat.mean()),
        "inc_std": float(flat.std()),
        "inc_abs_p50": float(flat.abs().median()),
        "inc_abs_p99": float(flat.abs().quantile(0.99)),
        "inc_kurtosis_excess": float(((flat - flat.mean()) ** 4).mean() / flat.var() ** 2 - 3),
        "path_end_std": float(paths[:, -1, 0].double().std()),
        "lag1_autocorr": float(
            torch.corrcoef(torch.stack([inc[:, :-1].flatten(), inc[:, 1:].flatten()]))[0, 1]
        ),
        "abs_lag1_autocorr": float(
            torch.corrcoef(torch.stack([inc[:, :-1].abs().flatten(), inc[:, 1:].abs().flatten()]))[
                0, 1
            ]
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, beta, payload = load_model(args.checkpoint, device)
    log(
        f"checkpoint {args.checkpoint.name}: outer_iter={payload['outer_iter']} "
        f"best val={payload['stop_state']['best_monitored']:.6f} at epoch "
        f"{payload['stop_state']['best_epoch']}, precision={payload['training_config'].get('precision')}, "
        f"beta={beta}, model={payload['model_config']}"
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.preview:
        started = time.perf_counter()
        paths = generate_paths(
            model,
            args.preview,
            SESSION_BARS,
            beta,
            args.n_euler,
            args.context_on,
            generator,
            retransform_start=args.retransform_start,
            invert_iters=args.invert_iters,
            safe_frac=payload["training_config"]["safe_frac"],
        )
        log(
            f"preview: {args.preview} paths x {SESSION_BARS} bars in {time.perf_counter() - started:.1f}s"
        )
        log(f"standardized increment stats: {increment_stats(paths)}")
        return

    real = pd.read_parquet(REAL_PATH)
    scales_npz = np.load(SCALES_OUTPUT_PATH)
    scales = dict(zip(scales_npz["tickers"].tolist(), scales_npz["scales"].tolist(), strict=True))
    if set(scales) != set(real.columns):
        raise ValueError("Ticker scales do not match the curated real panel's columns.")
    sessions = pd.DatetimeIndex(real.index.normalize().unique())
    tickers = list(real.columns)
    path_tickers = [t for t in tickers for _ in sessions]
    path_sessions = [s for _ in tickers for s in sessions]
    n_paths = len(path_tickers)
    log(
        f"real panel {real.shape}; {len(tickers)} tickers x {len(sessions)} sessions = "
        f"{n_paths:,} paths of {SESSION_BARS} bars; N_pi={args.n_euler}, context_on={args.context_on}, "
        f"retransform_start={args.retransform_start}, invert_iters={args.invert_iters}, "
        f"seed={args.seed}, batch={args.batch_paths}"
    )

    chunks = []
    started = time.perf_counter()
    for start in range(0, n_paths, args.batch_paths):
        n = min(args.batch_paths, n_paths - start)
        chunks.append(
            generate_paths(
                model,
                n,
                SESSION_BARS,
                beta,
                args.n_euler,
                args.context_on,
                generator,
                retransform_start=args.retransform_start,
                invert_iters=args.invert_iters,
                safe_frac=payload["training_config"]["safe_frac"],
            ).cpu()
        )
        done = start + n
        elapsed = time.perf_counter() - started
        log(
            f"  {done:,}/{n_paths:,} paths, {elapsed / 60:.1f} min, eta {elapsed / done * (n_paths - done) / 60:.1f} min"
        )
    generated = torch.cat(chunks)
    log(f"generation done in {(time.perf_counter() - started) / 60:.1f} min")
    log(f"standardized increment stats (all paths): {increment_stats(generated)}")

    panel = assemble_price_panel(generated, path_tickers, path_sessions, scales, real)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output)
    log(f"wrote {args.output} shape={panel.shape}")


if __name__ == "__main__":
    main()
