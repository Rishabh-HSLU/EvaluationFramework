"""Validate the approved grid and prepare its report and red manuscript update."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fineval.scripts.sensitivity.grid import GRID
from fineval.scripts.sensitivity.persistence import atomic_json, file_digest

SEEDS = (42, 1, 2, 3, 4)
GENERATORS = ("AIL", "GBM", "MSV", "SBBTS")


def validate_and_summarize(directory: Path) -> dict:
    metric_frames, aggregate_frames = [], []
    identities = []
    for seed in SEEDS:
        folder = directory / f"seed_{seed}"
        status = json.loads((folder / "status.json").read_text())
        if status["status"] != "completed" or len(status["completed_specs"]) != 19:
            raise ValueError(f"Seed {seed} did not complete all 19 specifications")
        snap = json.loads((folder / "grid_spec.json").read_text())
        identities.append(snap)
        m = pd.read_csv(folder / "sweep_metric_results.csv", float_precision="round_trip")
        a = pd.read_csv(folder / "sweep_aggregate_results.csv", float_precision="round_trip")
        if len(m) != 304 or len(a) != 76 or set(m.seed) != {seed}:
            raise ValueError(f"Incomplete result rows for seed {seed}")
        expected = {
            (s.spec_id, metric, g)
            for s in GRID
            for metric in ("M1", "M2", "M3", "M4")
            for g in GENERATORS
        }
        if set(zip(m.spec_id, m.metric, m.generator, strict=True)) != expected:
            raise ValueError(f"Metric identities differ for seed {seed}")
        if a.duplicated(["spec_id", "generator"]).any() or set(a.generator) != set(GENERATORS):
            raise ValueError(f"Aggregate identities differ for seed {seed}")
        if not np.isfinite(m[["score", "g_rr_mean", "g_sr_mean"]]).all().all():
            raise ValueError(f"Nonfinite scores in seed {seed}")
        if not (m.n_valid_draws.eq(100) & m.n_total_draws.eq(100)).all():
            raise ValueError(f"Some draws are invalid in seed {seed}; review before reporting")
        if not np.allclose(
            m.score, m.g_rr_mean / (m.g_rr_mean + m.g_sr_mean), rtol=2e-12, atol=2e-14
        ):
            raise ValueError("Saved scores disagree with the mean gap ratios")
        if not (a.k_used.eq(4) & a.k_total.eq(4) & a.min_valid_draws.eq(100)).all():
            raise ValueError("Incomplete aggregate metric coverage")
        calculated = (
            m.assign(logratio=np.abs(np.log(m.g_sr_mean / m.g_rr_mean)))
            .groupby(["spec_id", "generator"])
            .logratio.mean()
            .pipe(np.exp)
        )
        actual = a.set_index(["spec_id", "generator"]).G_dev.reindex(calculated.index)
        if not np.allclose(actual, calculated, rtol=2e-12, atol=2e-14):
            raise ValueError("Aggregate deviation does not reproduce")
        metric_frames.append(m)
        aggregate_frames.append(a)
    for snap in identities[1:]:
        for key in ("dataset_sha256", "source_sha256", "specs", "numerical_environment"):
            if snap[key] != identities[0][key]:
                raise ValueError(f"The seed runs used different {key}")
    m, a = (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(aggregate_frames, ignore_index=True),
    )
    primary = m.loc[m.spec_id == "primary", ["seed", "metric", "generator", "score"]]
    m = m.merge(
        primary.rename(columns={"score": "primary_score"}),
        on=["seed", "metric", "generator"],
        validate="many_to_one",
    )
    m["delta_score"] = m.score - m.primary_score
    m["point_reference_crossed"] = np.sign(m.score - 0.5) != np.sign(m.primary_score - 0.5)
    ap = a.loc[a.spec_id == "primary", ["seed", "generator", "G_dev"]]
    a = a.merge(
        ap.rename(columns={"G_dev": "primary_G_dev"}),
        on=["seed", "generator"],
        validate="many_to_one",
    )
    a["relative_change_percent"] = 100 * (a.G_dev / a.primary_G_dev - 1)
    summary = m.groupby(["spec_id", "metric", "generator"], as_index=False).agg(
        mean_score=("score", "mean"),
        seed_sd=("score", "std"),
        min_score=("score", "min"),
        max_score=("score", "max"),
        mean_delta=("delta_score", "mean"),
        min_delta=("delta_score", "min"),
        max_delta=("delta_score", "max"),
    )
    ag_summary = a.groupby(["spec_id", "generator"], as_index=False).agg(
        mean_G_dev=("G_dev", "mean"),
        seed_sd=("G_dev", "std"),
        min_G_dev=("G_dev", "min"),
        max_G_dev=("G_dev", "max"),
        mean_percent_change=("relative_change_percent", "mean"),
    )
    changes = {}
    for metric in ("M1", "M4"):
        specs = [s.spec_id for s in GRID if metric in s.affected_metrics]
        count = 0
        for seed in SEEDS:
            sub = m.loc[(m.seed == seed) & (m.metric == metric)]
            base_order = (
                sub.loc[sub.spec_id == "primary"]
                .sort_values("score", ascending=False, kind="stable")
                .generator.tolist()
            )
            for spec in specs:
                order = (
                    sub.loc[sub.spec_id == spec]
                    .sort_values("score", ascending=False, kind="stable")
                    .generator.tolist()
                )
                count += order != base_order
        changes[metric] = {"changed_orderings": count, "comparisons": len(specs) * len(SEEDS)}
    count = 0
    for seed in SEEDS:
        sub = a.loc[a.seed == seed]
        order = (
            sub.loc[sub.spec_id == "primary"].sort_values("G_dev", kind="stable").generator.tolist()
        )
        for spec in (s.spec_id for s in GRID if s.spec_id != "primary"):
            count += (
                sub.loc[sub.spec_id == spec].sort_values("G_dev", kind="stable").generator.tolist()
                != order
            )
    changes["aggregate"] = {"changed_orderings": count, "comparisons": 90}
    crossings = m.loc[
        (m.spec_id != "primary") & m.point_reference_crossed,
        ["seed", "spec_id", "metric", "generator", "primary_score", "score"],
    ]
    for name, frame in [
        ("metric_results.csv", m),
        ("aggregate_results.csv", a),
        ("metric_seed_summary.csv", summary),
        ("aggregate_seed_summary.csv", ag_summary),
        ("point_reference_crossings.csv", crossings),
    ]:
        frame.to_csv(directory / name, index=False)
    report = {
        "status": "validated",
        "seeds": list(SEEDS),
        "specifications": 19,
        "single_parameter_alternatives": 15,
        "two_parameter_corners": 3,
        "metric_rows": len(m),
        "aggregate_rows": len(a),
        "all_draws_valid": True,
        "ordering_changes": changes,
        "point_reference_crossings": len(crossings),
        "scope": "Paired point-estimate sensitivity on five seeds; no alternative-spec corpus intervals.",
        "input_sha256": identities[0]["dataset_sha256"],
        "source_sha256": identities[0]["source_sha256"],
    }
    atomic_json(directory / "validation.json", report)
    lines = [
        "# Five-seed sensitivity results",
        "",
        "19 frozen specifications; seeds 42, 1, 2, 3, 4; all four generators.",
        "B=100; 200 tickers per draw; fixed original run #4 panels.",
        "",
        "| Comparison | Changed complete orderings | Paired comparisons |",
        "|---|---:|---:|",
    ]
    for key, value in changes.items():
        lines.append(f"| {key} | {value['changed_orderings']} | {value['comparisons']} |")
    lines += [
        "",
        f"Point-reference crossings: {len(crossings)} metric/generator/spec/seed comparisons.",
        "These are point changes, not changes in confidence-interval containment.",
        "",
        "See metric_seed_summary.csv, aggregate_seed_summary.csv and the complete",
        "per-seed records. M2/M3 settings were fixed; copied unchanged results do not",
        "demonstrate sensitivity robustness for their own unvaried parameters.",
        "",
        "The analysis is exploratory. It documents the effects of the frozen choices",
        "without removing their AIL-informed selection history or testing all interactions.",
    ]
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def make_figures(directory: Path) -> None:
    m = pd.read_csv(directory / "metric_seed_summary.csv")
    a = pd.read_csv(directory / "aggregate_seed_summary.csv")
    specs = [s.spec_id for s in GRID]
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes, ("M1", "M4"), strict=True):
        values = (
            m.loc[m.metric == metric]
            .pivot(index="spec_id", columns="generator", values="mean_delta")
            .reindex(index=specs, columns=GENERATORS)
        )
        bound = max(0.001, float(np.abs(values.to_numpy()).max()))
        plot = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
        axis.set_xticks(range(4), GENERATORS)
        axis.set_yticks(range(19), specs, fontsize=7)
        axis.set_title(metric + ": mean change from same-seed primary")
        fig.colorbar(plot, ax=axis, label="Score change")
    fig.savefig(directory / "score_sensitivity.png", dpi=180)
    plt.close(fig)
    values = a.pivot(index="spec_id", columns="generator", values="mean_percent_change").reindex(
        index=specs, columns=GENERATORS
    )
    fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    bound = max(0.01, float(np.abs(values.to_numpy()).max()))
    plot = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-bound, vmax=bound)
    axis.set_xticks(range(4), GENERATORS)
    axis.set_yticks(range(19), specs, fontsize=8)
    axis.set_title("Mean percentage change in aggregate deviation")
    fig.colorbar(plot, ax=axis, label="Change from same-seed primary (%)")
    fig.savefig(directory / "aggregate_sensitivity.png", dpi=180)
    plt.close(fig)


def update_manuscript(directory: Path, workspace: Path, report: dict) -> None:
    previous = directory / "manuscript_update.json"
    if previous.exists():
        after = json.loads(previous.read_text())
        if not all(file_digest(workspace / name) == wanted for name, wanted in after.items()):
            raise ValueError(
                "Manuscript changed after the automatic update; review before rebuilding"
            )
        return
    guard = json.loads((directory / "manuscript_guard.json").read_text())
    for name, wanted in guard.items():
        if file_digest(workspace / name) != wanted:
            raise ValueError(
                f"Manuscript changed during the run; preserve edits and review: {name}"
            )
    manuscript = workspace / "manuscript"
    backup = workspace / "deliverables/manuscript/archive/before_sensitivity_update"
    backup.mkdir(parents=True, exist_ok=True)
    for name in guard:
        target = backup / Path(name).name
        shutil.copy2(workspace / name, target)
    c = report["ordering_changes"]
    section = r"""% SENSITIVITY-VALIDATED: fixed 19-specification grid, five paired seeds.
\subsection{\edited{Sensitivity analysis}}\label{sec:sensitivity}
{\color{red}
Several metric parameters were selected during development with AIL outputs
visible. We therefore evaluate the frozen 19-specification grid on the same
four generators and fixed panels, using 100 matched draws and 200 tickers
per draw under each of five root seeds (42, 1, 2, 3, and 4).
The grid includes the primary specification, 15 single-parameter alternatives,
and three selected two-parameter combinations.
It varies M1 tail sharpness and magnitude, and M4's volatility window,
minimum-valid fraction, tail fraction, and regime weights.
The minimum-period rule is retained when the window or valid fraction changes.
Within each seed the specifications share the same sampled ticker indices.

Across the examined alternatives, the complete score ordering changes in
@M1@ of 45 paired comparisons on M1 and @M4@ of 45 on M4.
The aggregate ordering changes in @AG@ of 90 paired comparisons.
There are @CROSS@ metric--generator--specification--seed comparisons in which
the point score changes its position relative to 0.5.
These counts describe point estimates, not confidence-interval coverage.
The five-seed means and ranges and the complete numerical grid accompany
the reference results; Appendix~\ref{app:sensitivity} shows the score changes.

Only the affected metric is recomputed under each alternative; unchanged
same-seed metric results are reused and the aggregate is recalculated.
Fast and full evaluation agree exactly on the nondegenerate preflight grid.
M2 and M3 parameters are held fixed, so their unchanged scores do not
demonstrate robustness to changes in those parameters.
The grid measures sensitivity within its specified ranges and selected
interactions. It does not establish robustness to every structural choice,
including the volatility-proxy family, or undo the AIL-informed selection
history. The primary specification is retained independently of these outcomes.
}
"""
    for marker, value in [
        ("@M1@", c["M1"]["changed_orderings"]),
        ("@M4@", c["M4"]["changed_orderings"]),
        ("@AG@", c["aggregate"]["changed_orderings"]),
        ("@CROSS@", report["point_reference_crossings"]),
    ]:
        section = section.replace(marker, str(value))
    p = manuscript / "sections/experiments.tex"
    text = p.read_text()
    start = text.index("% REVIEW-LUDO-13:")
    p.write_text(text[:start] + section)
    appendix = r"""\section{\edited{Sensitivity grid}}\label{app:sensitivity}
{\color{red}
Figure~\ref{fig:sensitivity-scores} shows the mean score changes from the
same-seed primary specification over five seeds.
Figure~\ref{fig:sensitivity-aggregate} reports percentage changes in
aggregate deviation. M1 changes concern tail weighting of standardized
marginal returns; M4 changes concern volatility conditioning and tail shape.
The full numerical settings, seed-level results, coverage diagnostics,
and summary tables are recorded with the repository's sensitivity results.
Ranges across seeds are numerical diagnostics, not corpus confidence intervals.
}
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/sensitivity_scores.png}
\caption{\edited{Mean change in M1 and M4 scores from the same-seed primary,
over five seeds. Unchanged rows reflect fixed metric settings.}}
\label{fig:sensitivity-scores}
\end{figure}
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/sensitivity_aggregate.png}
\caption{\edited{Mean percentage change in aggregate deviation from the
same-seed primary. Lower aggregate deviation indicates closer gap parity.}}
\label{fig:sensitivity-aggregate}
\end{figure}
"""
    (manuscript / "sections/sensitivity_appendix.tex").write_text(appendix)
    p = manuscript / "sections/appendix.tex"
    p.write_text(p.read_text() + "\n\\input{sections/sensitivity_appendix}\n")
    p = manuscript / "sections/discussion.tex"
    text = p.read_text()
    start = text.index("Some metric parameters were chosen during development")
    end = text.index("Finally, stylized-fact fidelity", start)
    text = (
        text[:start]
        + (
            "Some metric parameters were chosen during development with AIL\n"
            "outputs visible. The sensitivity analysis documents how scores and\n"
            "orderings respond to the frozen alternatives; its numerical results\n"
            "do not remove that selection history. The examined ranges and three\n"
            "selected interactions do not cover every structural metric choice.\n"
        )
        + text[end:]
    )
    p.write_text(text)
    p = manuscript / "sections/conclusion.tex"
    text = p.read_text().replace(
        "historical corpus; independent-period, repeated-fit, and parameter\n"
        "sensitivity evaluations remain necessary extensions.",
        "historical corpus. The frozen grid documents sensitivity to the\n"
        "examined settings; independent-period, repeated-fit, and broader\n"
        "structural sensitivity evaluations remain necessary extensions.",
    )
    p.write_text(text)
    (manuscript / "figures").mkdir(exist_ok=True)
    for source, dest in [
        ("score_sensitivity.png", "sensitivity_scores.png"),
        ("aggregate_sensitivity.png", "sensitivity_aggregate.png"),
    ]:
        shutil.copy2(directory / source, manuscript / "figures" / dest)
    names = [*guard, "manuscript/sections/sensitivity_appendix.tex"]
    atomic_json(
        directory / "manuscript_update.json",
        {name: file_digest(workspace / name) for name in names},
    )


def build_review(workspace: Path) -> None:
    manuscript = workspace / "manuscript"
    environment = os.environ.copy()
    fonts = workspace / "deliverables/build_dependencies/latin-modern/lm"
    if fonts.exists():
        environment["TEXMFHOME"] = str(fonts)
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-jobname=main",
        r"\pdfmapfile{+lm.map}\input{main.tex}",
    ]
    with (workspace / "deliverables/manuscript/sensitivity_build.log").open("w") as log:
        subprocess.run(
            command,
            cwd=manuscript,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            ["bibtex", "main"],
            cwd=manuscript,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
        for _ in range(2):
            subprocess.run(
                command,
                cwd=manuscript,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    out = workspace / "deliverables/manuscript"
    shutil.copy2(manuscript / "main.pdf", out / "revised_manuscript_red_review.pdf")
    shutil.copy2(manuscript / "main.pdf", out / "sensitivity_manuscript_red_review.pdf")
    shutil.copy2(
        manuscript / "main.pdf",
        out / "Benchmarking_Synthetic_Financial_Time_Series_Current_Progress.pdf",
    )
    with zipfile.ZipFile(
        out / "revised_manuscript.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for pattern in [
            "main.tex",
            "main.pdf",
            "references.bib",
            "*.bst",
            "elsarticle.cls",
            "elsarticle.dtx",
            "elsarticle.ins",
            "README.md",
            "sections/*.tex",
            "figures/*.png",
            "figures/*.pdf",
        ]:
            for path in manuscript.glob(pattern):
                archive.write(path, path.relative_to(manuscript))
    p = out / "README.md"
    p.write_text(
        p.read_text().replace(
            "run4_manuscript_red_review_20261016.pdf", "sensitivity_manuscript_red_review.pdf"
        )
    )
    p = workspace / "README.md"
    addition = (
        "## Sensitivity results\n\n"
        "The [validated five-seed grid](results/sensitivity/summary.md) contains\n"
        "all 19 frozen specifications, seed-level records, input/source provenance,\n"
        "numerical summaries and figures. Alternative-spec results are point\n"
        "estimates, not O=200 corpus intervals. Run #4 remains the primary reference.\n\n"
    )
    text = p.read_text()
    if "## Sensitivity results" not in text:
        p.write_text(text.replace("## Citation", addition + "## Citation"))
    p = workspace / "results/README.md"
    text = p.read_text()
    if "sensitivity/summary.md" not in text:
        p.write_text(
            text + "\nThe [five-seed sensitivity grid](sensitivity/summary.md)\n"
            "reports specification sensitivity separately from run #4.\n"
        )
    p = workspace / "docs/grid_spec_freeze.md"
    text = p.read_text()
    start = text.index("Status update,")
    end = text.index("Source of truth", start)
    p.write_text(
        text[:start]
        + (
            "Status update: the current 19-specification grid has completed for seeds\n"
            "42, 1, 2, 3 and 4, with all four generators and all five input hashes recorded.\n"
            "See [validated sensitivity results](../results/sensitivity/summary.md).\n"
            "The settings below are unchanged; the analysis remains exploratory.\n\n"
        )
        + text[end:]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_and_summarize(args.run_dir)
    make_figures(args.run_dir)
    update_manuscript(args.run_dir, args.workspace, report)
    build_review(args.workspace)
    paths = {
        str(p.relative_to(args.run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.run_dir.rglob("*")
        if p.is_file()
        and p.suffix in (".csv", ".json", ".png", ".md")
        and p.name
        not in {
            "checksums.json",
            "batch_status.json",
            "manuscript_update.json",
            "publication_update.json",
        }
    }
    atomic_json(args.run_dir / "checksums.json", {"files": paths})
    atomic_json(
        args.run_dir / "publication_update.json",
        {
            "status": "updated",
            "results_validated": True,
            "manuscript_updated_in_red": True,
            "pdf_built": True,
        },
    )
    print("Sensitivity results validated; manuscript, PDF, source ZIP and README updated.")


if __name__ == "__main__":
    main()
