"""Reproduce full-panel M2--M4 descriptive figures from the run #4 inputs."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fineval import config
from fineval.benchmark.config import build_metrics
from fineval.preprocessing import PreprocessingPipeline
from fineval.preprocessing.session_clock import overnight_masked_log_returns
from fineval.scripts.sensitivity.persistence import atomic_json, file_digest, numerical_environment

ROOT = Path(__file__).resolve().parents[1]
PANELS = {"Real": "real", "AIL": "ail", "GBM": "gbm", "MSV": "msv", "SBBTS": "sbbts"}
STYLES = {
    "Real": ("#202020", "-"),
    "AIL": ("#0072B2", "-"),
    "GBM": ("#E69F00", "--"),
    "MSV": ("#009E73", "-."),
    "SBBTS": ("#CC79A7", ":"),
}


def regime_features(metric, returns):
    """Canonical M4 operations with array pairing to avoid a large MultiIndex.

    Retain time-major/ticker-minor order, the same finite (vol, return) pairs,
    pandas qcut, and the metric's own rolling-volatility and GPD fit routines.
    Exact agreement with extract_features is tested on a nondegenerate panel.
    """
    vol = metric._rolling_vol(returns).reindex(returns.index).to_numpy().ravel()
    values = returns.to_numpy().ravel()
    finite = np.isfinite(vol) & np.isfinite(values)
    vol, values = vol[finite], values[finite]
    try:
        labels, edges = pd.qcut(
            vol, q=metric.n_regimes, labels=False, retbins=True, duplicates="raise"
        )
    except ValueError as exc:
        raise ValueError("M4 regimes cannot be identified; do not plot a partial profile") from exc
    rows = []
    for regime in range(metric.n_regimes):
        absolute = np.abs(values[labels == regime])
        threshold = np.quantile(absolute, 1 - metric.tail_fraction)
        excess = absolute[absolute > threshold] - threshold
        rows.append(
            {
                "regime": regime + 1,
                "xi": metric._fit_gpd_xi(excess),
                "vol_lower": edges[regime],
                "vol_upper": edges[regime + 1],
                "threshold": threshold,
                "observations": len(absolute),
                "exceedances": len(excess),
            }
        )
    return pd.DataFrame(rows)


def extract_panel(prices):
    pipeline = PreprocessingPipeline()
    returns = overnight_masked_log_returns(prices)
    returns, fff = pipeline._maybe_deseasonalize(returns)
    del prices
    gc.collect()
    metrics = {m.name: m for m in build_metrics()}
    m2, m3, m4 = (metrics[name] for name in ("M2", "M3", "M4"))
    acf = pd.DataFrame(
        {"lag_minutes": np.arange(1, m2.lag_max + 1), "acf": m2.extract_features(returns)}
    )
    raw, counts = [], []
    for scale in m3.scales:
        aggregated = m3._aggregate_scale(returns, scale)
        raw.append(m3._compute_kurtosis(aggregated))
        counts.append(len(aggregated))
    kurt = pd.DataFrame(
        {
            "horizon_minutes": m3.scales,
            "pearson_kurtosis": raw,
            "normalized_kurtosis": np.asarray(raw) / raw[0],
            "valid_aggregated_returns": counts,
        }
    )
    tails = regime_features(m4, returns)
    metadata = {
        "timestamps": len(returns),
        "tickers": returns.shape[1],
        "sessions": returns.index.normalize().nunique(),
        "fff_applied": fff is not None,
        "m3_common_support": m3.common_support,
        "m3_eligible_sessions": sum(
            len(group) - 1 >= m3.common_support
            for _, group in returns.groupby(returns.index.normalize())
        ),
    }
    return acf, kurt, tails, metadata


def render(directory):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    acf = pd.read_csv(directory / "m2_acf.csv", float_precision="round_trip")
    kurt = pd.read_csv(directory / "m3_kurtosis.csv", float_precision="round_trip")
    tails = pd.read_csv(directory / "m4_regime_xi.csv", float_precision="round_trip")
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, lower in zip(axes, (1, 60), strict=True):
        for label, (color, style) in STYLES.items():
            subset = acf.loc[(acf.panel == label) & (acf.lag_minutes >= lower)]
            ax.plot(
                subset.lag_minutes,
                subset.acf,
                color=color,
                linestyle=style,
                label=label,
                linewidth=1.8 if label == "Real" else 1.4,
            )
        ax.axhline(0, color="0.7", linewidth=0.7)
        ax.set(
            xlabel="Within-session lag (minutes)",
            ylabel="Mean ACF of absolute returns",
            xlim=(lower, 388),
        )
        ax.grid(alpha=0.18)
    axes[0].axvline(60, color="0.5", linestyle=":", linewidth=0.8)
    axes[0].set_title("(a) Full lag profile")
    axes[1].set_title("(b) M2 comparison window")
    axes[0].legend(frameon=False, fontsize=9)
    figures.append(("m2_absolute_return_acf", fig))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, column in zip(axes, ("pearson_kurtosis", "normalized_kurtosis"), strict=True):
        for label, (color, style) in STYLES.items():
            subset = kurt.loc[kurt.panel == label]
            ax.plot(
                subset.horizon_minutes,
                subset[column],
                marker="o",
                markersize=4,
                color=color,
                linestyle=style,
                label=label,
            )
        ax.set(xlabel="Aggregation horizon (minutes)", xticks=[1, 5, 15, 30])
        ax.grid(alpha=0.18)
    axes[0].set(ylabel="Pearson kurtosis", title="(a) Kurtosis across horizons")
    axes[0].axhline(3, color="0.6", linewidth=0.8, linestyle="--", label="Gaussian kurtosis = 3")
    axes[1].set(ylabel=r"$K(h)/K(1)$", title="(b) Normalized curve used by M3")
    axes[0].legend(frameon=False, fontsize=8)
    figures.append(("m3_kurtosis_aggregation", fig))
    fig, ax = plt.subplots(figsize=(6.8, 4), layout="constrained")
    for label, (color, style) in STYLES.items():
        subset = tails.loc[tails.panel == label]
        ax.plot(
            subset.regime,
            subset.xi,
            color=color,
            linestyle=style,
            marker="o",
            markersize=4,
            label=label,
        )
    ax.axhline(0, color="0.7", linewidth=0.7)
    ax.set(
        xlabel="Panel-specific lagged-volatility quintile",
        ylabel=r"GPD shape $\xi$",
        xticks=[1, 2, 3, 4, 5],
        xticklabels=["Q1\nCalmest", "Q2", "Q3", "Q4", "Q5\nMost turbulent"],
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3, fontsize=9)
    figures.append(("m4_xi_by_regime", fig))
    for name, fig in figures:
        fig.savefig(directory / (name + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(directory / (name + ".png"))
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/figures/run4")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Render the saved CSVs without loading price panels",
    )
    parser.add_argument("--panel", choices=PANELS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((ROOT / "results/run4/provenance/configuration.json").read_text())
    for name, expected in configuration["metric_parameters"].items():
        if hasattr(config, name) and not np.array_equal(getattr(config, name), expected):
            raise ValueError(f"Current configuration differs from run #4: {name}")
    records = directory / "panel_records"
    if args.panel:
        records.mkdir(exist_ok=True)
        # Bound this optional figure process without interrupting a benchmark.
        if sys.platform == "linux":
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
        expected = json.loads((ROOT / "results/run4/provenance/panel_digests.json").read_text())
        relative = f"data/curated/{PANELS[args.panel]}_prices.parquet"
        path = ROOT / relative
        if file_digest(path) != expected[relative]:
            raise ValueError(f"Run #4 panel identity differs: {relative}")
        print(f"Computing {args.panel}: full panel, canonical preprocessing", flush=True)
        prices = pd.read_parquet(path)
        acf, kurt, tails, metadata = extract_panel(prices)
        for name, frame in (("m2", acf), ("m3", kurt), ("m4", tails)):
            if not np.isfinite(frame.select_dtypes("number")).all().all():
                raise ValueError(f"Nonfinite {name} features for {args.panel}")
            frame.insert(0, "panel", args.panel)
            frame.to_csv(records / f"{args.panel}_{name}.csv", index=False)
        atomic_json(records / f"{args.panel}_metadata.json", metadata)
        print(f"Completed {args.panel}", flush=True)
        return
    if not args.plot_only:
        for label in PANELS:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.plot_metric_profiles",
                    "--panel",
                    label,
                    "--output-dir",
                    str(directory),
                ],
                cwd=ROOT,
                check=True,
            )
        for short, filename in (
            ("m2", "m2_acf.csv"),
            ("m3", "m3_kurtosis.csv"),
            ("m4", "m4_regime_xi.csv"),
        ):
            pd.concat(
                [
                    pd.read_csv(records / f"{label}_{short}.csv", float_precision="round_trip")
                    for label in PANELS
                ],
                ignore_index=True,
            ).to_csv(directory / filename, index=False)
        sources = [
            ROOT / "scripts/plot_metric_profiles.py",
            ROOT / "scripts/reproduce.py",
            *sorted((ROOT / "fineval").rglob("*.py")),
        ]
        atomic_json(
            directory / "provenance.json",
            {
                "scope": "In-sample full-panel empirical fidelity; descriptive features, not matched-bootstrap scores or confidence intervals",
                "randomness": "No sampling, no random seed, no retraining",
                "panels": json.loads(
                    (ROOT / "results/run4/provenance/panel_digests.json").read_text()
                ),
                "numerical_environment": numerical_environment(),
                "matplotlib": matplotlib.__version__,
                "command": "python -m scripts.reproduce profiles",
                "reference_metric_parameters": configuration["metric_parameters"],
                "source_sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in sources},
                "panel_metadata": {
                    label: json.loads((records / f"{label}_metadata.json").read_text())
                    for label in PANELS
                },
                "parameters": {
                    "m2_lags": [60, 388],
                    "m3_scales": [1, 5, 15, 30],
                    "m3_support": 360,
                    "m3_anchor": "close",
                    "m4_window": 60,
                    "m4_min_periods": 30,
                    "m4_regimes": 5,
                    "m4_tail_fraction": 0.05,
                    "m4_min_exceedances": 500,
                },
            },
        )
    render(directory)
    atomic_json(
        directory / "checksums.json",
        {
            "files": {
                str(p.relative_to(directory)): file_digest(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
                and p.name not in {"checksums.json", "job.json"}
                and p.suffix in (".csv", ".json", ".pdf", ".png")
            }
        },
    )
    print(f"Three figures and numerical source data saved to {directory}", flush=True)


if __name__ == "__main__":
    main()
