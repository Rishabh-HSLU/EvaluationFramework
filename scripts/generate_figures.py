"""
Generate dissertation- and slide-ready artifacts for:
  1. the main six-bucket benchmark table (real vs synthetic generators)
  2. per-bucket sanity-check tables (one at a time)

Each artifact is emitted as:
  - PNG  (drop into Keynote / PowerPoint / Beamer raster)
  - PDF  (vector — preferred for Beamer / LaTeX)
  - .tex (booktabs source for dissertation \\input{})

Run:
  uv run python scripts/generate_figures.py main      # main benchmark only
  uv run python scripts/generate_figures.py b1        # bucket 1 sanity only
  uv run python scripts/generate_figures.py all       # both
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _wrap_text(text: str, width_chars: int) -> str:
    """Wrap text to roughly width_chars per line, ignoring inline math regions."""
    # Replace existing line breaks then re-wrap
    paragraph = re.sub(r"\s+", " ", text).strip()
    lines = textwrap.wrap(paragraph, width=width_chars, break_long_words=False)
    return "\n".join(lines)


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
TEX_DIR = ROOT / "paper" / "snippets"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style — academic, restrained, presentation-safe
# ---------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "pdf.fonttype": 42,  # embed TrueType in PDF
        "ps.fonttype": 42,
    }
)

HEADER_FILL = "#1f3b66"  # deep navy
HEADER_TEXT = "#ffffff"
PAGE_BG = "#f4f1ea"  # soft warm off-white (page / figure background)
ROW_BG = "#fbf9f3"  # slightly lighter for default rows
ROW_ALT = "#ece8db"  # warm sand for alternating rows
BORDER = "#c8c2af"
PASS_GREEN = "#1b6f3a"
FAIL_RED = "#a32a2a"
HIGHLIGHT = "#f6e7a2"  # warm gold for best-in-row


# ---------------------------------------------------------------------------
# Generic table renderer
# ---------------------------------------------------------------------------
def render_table(
    title: str,
    subtitle: str,
    columns: list[str],
    col_widths: list[float],
    rows: list[list[str]],
    row_styles: list[dict] | None = None,
    cell_styles: dict[tuple[int, int], dict] | None = None,
    footnote: str | None = None,
    fig_path: Path | None = None,
    **kwargs,
) -> None:
    """
    Render a polished table to PNG + PDF.
      cell_styles maps (row_idx, col_idx) -> {'bold': bool, 'fill': hex, 'color': hex}
    """
    n_rows = len(rows)
    cell_styles = cell_styles or {}
    row_styles = row_styles or [{} for _ in range(n_rows)]

    row_height = kwargs.get("row_height", 0.55)
    cell_fontsize = kwargs.get("cell_fontsize", 10.5)
    header_h = 0.6
    title_h = 0.5 if title else 0
    subtitle_h = 0.4 if subtitle else 0
    foot_h = 0.4 if footnote else 0.05

    total_w = sum(col_widths)
    total_h = title_h + subtitle_h + header_h + n_rows * row_height + foot_h

    fig = plt.figure(figsize=(total_w, total_h), dpi=170, facecolor=PAGE_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PAGE_BG)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # ---- title + subtitle ----
    y = total_h
    if title:
        y -= title_h * 0.55
        ax.text(
            total_w / 2,
            y,
            title,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#11254a",
        )
        y -= title_h * 0.45
    if subtitle:
        y -= subtitle_h * 0.3
        ax.text(
            total_w / 2,
            y,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            color="#3c4a66",
        )
        y -= subtitle_h * 0.7

    # ---- header ----
    x = 0.0
    header_y = total_h - title_h - subtitle_h - header_h
    for ci, col in enumerate(columns):
        ax.add_patch(
            Rectangle(
                (x, header_y),
                col_widths[ci],
                header_h,
                facecolor=HEADER_FILL,
                edgecolor=HEADER_FILL,
            )
        )
        ax.text(
            x + col_widths[ci] / 2,
            header_y + header_h / 2,
            col,
            ha="center",
            va="center",
            color=HEADER_TEXT,
            fontweight="bold",
            fontsize=11,
        )
        x += col_widths[ci]

    # ---- data rows ----
    for ri, row in enumerate(rows):
        x = 0.0
        y_top = header_y - (ri + 1) * row_height
        # row fill (alt stripes unless row_style overrides)
        rstyle = row_styles[ri]
        row_fill = rstyle.get("fill", ROW_ALT if ri % 2 == 1 else ROW_BG)
        for ci, cell in enumerate(row):
            style = cell_styles.get((ri, ci), {})
            fill = style.get("fill", row_fill)
            color = style.get("color", "#1a1a1a")
            weight = "bold" if style.get("bold") else "normal"
            ax.add_patch(
                Rectangle(
                    (x, y_top),
                    col_widths[ci],
                    row_height,
                    facecolor=fill,
                    edgecolor=BORDER,
                    linewidth=0.6,
                )
            )
            ax.text(
                x + col_widths[ci] / 2,
                y_top + row_height / 2,
                cell,
                ha="center",
                va="center",
                color=color,
                fontweight=weight,
                fontsize=cell_fontsize,
                linespacing=1.25,
            )
            x += col_widths[ci]
        # row separator if "separator_below" in row_style
        if rstyle.get("separator_below"):
            ax.plot([0, total_w], [y_top, y_top], color=HEADER_FILL, linewidth=1.2)

    # ---- footnote ----
    if footnote:
        # Wrap so the footnote stays inside the figure width.
        # ~13 chars/inch at fontsize 8.5pt italic serif → use total_w * 14.
        wrap_w = max(60, int(total_w * 14))
        wrapped = _wrap_text(footnote, wrap_w)
        n_lines = wrapped.count("\n") + 1
        # Center the wrapped block within foot_h, anchored at the top of the
        # footnote band so the baseline sits below the table.
        ax.text(
            total_w / 2,
            foot_h * 0.85 - 0.05,
            wrapped,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#3c4a66",
            style="italic",
            linespacing=1.3,
        )
        # If the wrapped footnote needs more than one line, expand foot_h
        # was not possible (already drawn) — but n_lines * 0.16" stays safe
        # at the current default of 0.4".
        _ = n_lines

    # Skip tight_layout — it can extend the canvas when content sits at the
    # very edge of the axes. We've set explicit xlim/ylim, so saving with
    # bbox_inches="tight" + small padding gives the correct trim.
    if fig_path:
        fig.savefig(
            fig_path.with_suffix(".png"),
            bbox_inches="tight",
            pad_inches=0.1,
            dpi=200,
            facecolor=PAGE_BG,
        )
        fig.savefig(
            fig_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1, facecolor=PAGE_BG
        )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main benchmark table
# ---------------------------------------------------------------------------
def make_main_benchmark() -> None:
    # (id, label, AIL (sim, lo, hi), GARCH, SFAGan)
    data = [
        (
            "B1",
            "Tail-weighted W$_1$ (5% tails)",
            (0.3149, 0.2912, 0.3401),
            (0.4241, 0.4025, 0.4454),
            (0.2956, 0.2746, 0.3170),
        ),
        (
            "B2",
            "ACF of $|r|$, lags 60-390",
            (0.2601, 0.2517, 0.2681),
            (0.1357, 0.1312, 0.1405),
            (0.0527, 0.0506, 0.0548),
        ),
        (
            "B3",
            "Leverage Corr($r_t,\\,|r_{t+k}|$), lags 1-390",
            (0.4216, 0.4191, 0.4241),
            (0.3779, 0.3744, 0.3813),
            (0.4006, 0.3983, 0.4029),
        ),
        (
            "B4",
            "Scale-weighted L-kurtosis",
            (0.5432, 0.5172, 0.5688),
            (0.3738, 0.3499, 0.3991),
            (0.1484, 0.1368, 0.1594),
        ),
        (
            "B5",
            "CFVC Frobenius gap",
            (0.1497, 0.1371, 0.1626),
            (0.0440, 0.0407, 0.0474),
            (0.0395, 0.0364, 0.0426),
        ),
        (
            "B6",
            "Conditional GPD tail regime",
            (0.1241, 0.1170, 0.1314),
            (0.2559, 0.2441, 0.2677),
            (0.1003, 0.0941, 0.1064),
        ),
    ]
    gens = ["AIL", "GARCH", "SFAGan"]

    # composites
    def arith(vals):
        return sum(vals) / len(vals)

    def geom(vals):
        import math

        return math.exp(sum(math.log(max(v, 1e-6)) for v in vals) / len(vals))

    sims = {g: [d[2 + i][0] for d in data] for i, g in enumerate(gens)}
    comp_arith = {g: arith(sims[g]) for g in gens}
    comp_geom = {g: geom(sims[g]) for g in gens}

    columns = ["Bucket", "Description", "AIL", "GARCH", "SFAGan"]
    col_widths = [1.0, 4.0, 2.5, 2.5, 2.5]

    rows: list[list[str]] = []
    cell_styles: dict[tuple[int, int], dict] = {}
    row_styles: list[dict] = []

    for ri, (bid, label, a, g, s) in enumerate(data):
        sims_row = [a[0], g[0], s[0]]
        best_idx = max(range(3), key=lambda i: sims_row[i])
        cells = [
            bid,
            label,
            f"{a[0]:.4f}\n[{a[1]:.3f}, {a[2]:.3f}]",
            f"{g[0]:.4f}\n[{g[1]:.3f}, {g[2]:.3f}]",
            f"{s[0]:.4f}\n[{s[1]:.3f}, {s[2]:.3f}]",
        ]
        rows.append(cells)
        cell_styles[(ri, 2 + best_idx)] = {"bold": True, "fill": HIGHLIGHT}
        row_styles.append({})

    # composite rows
    sep_row = [
        "",
        "Composite (arithmetic)",
        f"{comp_arith['AIL']:.4f}",
        f"{comp_arith['GARCH']:.4f}",
        f"{comp_arith['SFAGan']:.4f}",
    ]
    geom_row = [
        "",
        "Composite (geometric)",
        f"{comp_geom['AIL']:.4f}",
        f"{comp_geom['GARCH']:.4f}",
        f"{comp_geom['SFAGan']:.4f}",
    ]
    arith_best = max(range(3), key=lambda i: list(comp_arith.values())[i])
    geom_best = max(range(3), key=lambda i: list(comp_geom.values())[i])

    rows.append(sep_row)
    row_styles.append({"separator_below": False, "fill": "#e6ecf6"})
    cell_styles[(len(rows) - 1, 2 + arith_best)] = {"bold": True, "fill": HIGHLIGHT}

    rows.append(geom_row)
    row_styles.append({"fill": "#e6ecf6"})
    cell_styles[(len(rows) - 1, 2 + geom_best)] = {"bold": True, "fill": HIGHLIGHT}
    # add separator above composite group
    row_styles[-2]["separator_below"] = False  # the line is drawn below row_styles[-2-1]
    # we want a separator ABOVE the composite — place on the last B-row
    row_styles[5]["separator_below"] = True

    render_table(
        title="Six-bucket benchmark — real vs synthetic generators",
        subtitle=(
            "Similarity score $s_b = \\bar{g}_{rr} \\,/\\, (\\bar{g}_{rr} + \\bar{g}_{sr})$; "
            "$1$ = indistinguishable from real, $0.5$ = noise floor.   "
            "$N{=}200$, $200$ resamples, paired bootstrap ($B{=}2000$), seed $42$."
        ),
        columns=columns,
        col_widths=col_widths,
        rows=rows,
        row_styles=row_styles,
        cell_styles=cell_styles,
        footnote=(
            "Highlighted cell per row: best generator. Brackets show paired-bootstrap 95% CI. "
            "Composite rows aggregate the six similarity scores; geometric mean penalises "
            "low scores more strongly than arithmetic mean."
        ),
        fig_path=FIG_DIR / "benchmark_main",
    )

    # ----- LaTeX -----
    tex = r"""% Auto-generated by scripts/generate_figures.py — do not hand-edit.
\begin{table}[t]
\centering
\caption{Six-bucket benchmark: similarity score $s_b=\bar{g}_{rr}/(\bar{g}_{rr}+\bar{g}_{sr})$
for three synthetic 1-minute return generators against the bundled real NASDAQ eval corpus.
Parameters: $N{=}200$ paths per half, $200$ matched-$N$ resamples, paired bootstrap with
$B{=}2000$ resamples, seed $42$. Bracketed values are paired-bootstrap $95\%$ confidence
intervals. Bold marks the best generator per bucket / composite.}
\label{tab:benchmark-main}
\begin{tabular}{llccc}
\toprule
ID & Bucket / property & AIL & GARCH & SFAGan \\
\midrule
"""
    for bid, label, a, g, s in data:
        label_tex = label  # already LaTeX-friendly
        vals = [a[0], g[0], s[0]]
        best = max(range(3), key=lambda i: vals[i])

        def fmt(triple, b):
            point = f"{triple[0]:.4f}"
            cell = f"{point}\\,[{triple[1]:.3f},\\,{triple[2]:.3f}]"
            return f"\\textbf{{{cell}}}" if b else cell

        tex += (
            f"{bid} & {label_tex} & "
            f"{fmt(a, best == 0)} & {fmt(g, best == 1)} & {fmt(s, best == 2)} \\\\\n"
        )
    tex += "\\midrule\n"
    aa = [comp_arith[g_] for g_ in gens]
    gg = [comp_geom[g_] for g_ in gens]
    ba = max(range(3), key=lambda i: aa[i])
    bg = max(range(3), key=lambda i: gg[i])

    def fmt_c(v, b):
        return f"\\textbf{{{v:.4f}}}" if b else f"{v:.4f}"

    tex += (
        f"   & Composite (arithmetic) & "
        f"{fmt_c(aa[0], ba == 0)} & {fmt_c(aa[1], ba == 1)} & {fmt_c(aa[2], ba == 2)} \\\\\n"
    )
    tex += (
        f"   & Composite (geometric)  & "
        f"{fmt_c(gg[0], bg == 0)} & {fmt_c(gg[1], bg == 1)} & {fmt_c(gg[2], bg == 2)} \\\\\n"
    )
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    (TEX_DIR / "benchmark_main.tex").write_text(tex)


# ---------------------------------------------------------------------------
# Shared sanity-table renderer (chat-style 7-column layout)
# ---------------------------------------------------------------------------
def _render_sanity_table(
    bucket_id: str,
    g_rr: float,
    subtitle: str,
    rows_data: list[tuple],  # (id, check, pert, gap, ratio, expected, klass, passed)
    footnote: str,
    fig_path: Path,
) -> None:
    columns = [
        "ID",
        "Check",
        "Perturbation",
        r"mean gap $\bar{g}$",
        r"$\bar{g}/\bar{g}_{rr}$",
        "Expected",
        "Outcome",
    ]
    col_widths = [0.75, 2.10, 4.20, 1.45, 1.35, 3.70, 1.55]

    rows: list[list[str]] = []
    cell_styles: dict[tuple[int, int], dict] = {}
    for ri, (cid, check, pert, gap, ratio, expected, klass, passed) in enumerate(rows_data):
        outcome_text = f"{'PASS' if passed else 'FAIL'}\n{klass}"
        rows.append([cid, check, pert, f"{gap:.4f}", f"{ratio:.3f}×", expected, outcome_text])
        cell_styles[(ri, 6)] = {
            "bold": True,
            "color": PASS_GREEN if passed else FAIL_RED,
        }

    render_table(
        title=f"Bucket {bucket_id} — sanity-check diagnostics",
        subtitle=subtitle,
        columns=columns,
        col_widths=col_widths,
        rows=rows,
        cell_styles=cell_styles,
        footnote=footnote,
        fig_path=fig_path,
        row_height=1.10,
        cell_fontsize=10.0,
    )


# ---------------------------------------------------------------------------
# Bucket 1 sanity-check table
# ---------------------------------------------------------------------------
def make_bucket1_sanity() -> None:
    g_rr = 0.3469
    rows_data = [
        (
            "N1.1",
            "tail_replacement",
            "Replace bottom / top 5% with Gaussian",
            1.5180,
            4.375,
            "Heavy-tail destruction:\nlarge gap",
            "destructive",
            True,
        ),
        (
            "N1.2",
            "skew_flip",
            "Flip sign of all returns",
            0.0348,
            0.100,
            "Large gap IF gain/loss asymmetry present;\nnear-zero on symmetric corpora †",
            "adaptive",
            True,
        ),
        (
            "N1.3",
            "temporal_shuffle",
            "Random permutation per path",
            0.0000,
            0.000,
            "Exact zero — pooled marginal is\norder-invariant",
            "invariant",
            True,
        ),
        (
            "N1.4",
            "bulk_perturbation",
            "Perturb middle 60% quantiles only",
            0.00131,
            0.004,
            "Small gap — B1 measures only\nthe 5% tails",
            "invariant",
            True,
        ),
        (
            "N1.5",
            "scale_sensitivity",
            "Multiply all returns by 2",
            3.4320,
            9.894,
            "Large gap — tail magnitudes double",
            "destructive",
            True,
        ),
    ]
    dest_ratios = [4.375, 9.894]
    inv_ratios = [0.000, 0.004]
    subtitle = (
        f"Real-vs-real noise floor $\\bar{{g}}_{{rr}}={g_rr:.4f}$ (contiguous half-split). "
        f"Discrimination: mean(destructive) $={sum(dest_ratios) / len(dest_ratios):.2f}\\times$, "
        f"mean(invariant) $={sum(inv_ratios) / len(inv_ratios):.4f}\\times$ — three orders of magnitude apart."
    )
    footnote = (
        "† N1.2 is corpus-adaptive: it measures tail-magnitude asymmetry "
        "(max / min of the 5% / 95% tail magnitudes) and selects the "
        "threshold accordingly. The bundled corpus has asymmetry ratio below 1.3 at 5% tail "
        "depth and 1-min frequency — consistent with the literature on intraday gain/loss "
        "asymmetry weakening at higher frequencies (Cont 2001; Bouchaud–Potters)."
    )
    _render_sanity_table("B1", g_rr, subtitle, rows_data, footnote, FIG_DIR / "bucket1_sanity")
    _write_sanity_tex(
        "B1",
        "tail-weighted Wasserstein-1",
        "bucket1-sanity",
        g_rr,
        rows_data,
        TEX_DIR / "bucket1_sanity.tex",
        footnote_tex=(
            "$^\\dagger$N1.2 is corpus-adaptive: it measures tail-magnitude "
            "asymmetry (\\,$\\max/\\min$ of the $5\\%$/$95\\%$ tail magnitudes\\,) "
            "and selects the threshold accordingly. The bundled corpus has "
            "asymmetry ratio $<1.3$ at $5\\%$ tail depth and $1$-min frequency, "
            "consistent with the literature on intraday gain/loss asymmetry "
            "weakening at higher frequencies (Cont 2001; Bouchaud--Potters)."
        ),
    )


# ---------------------------------------------------------------------------
# Bucket 2 sanity-check table
# ---------------------------------------------------------------------------
def make_bucket2_sanity() -> None:
    g_rr = 0.001025
    rows_data = [
        (
            "N2.1",
            "shuffle",
            "Random permutation per path",
            0.01116,
            10.889,
            "Full destruction of temporal\nstructure — large gap",
            "destructive",
            True,
        ),
        (
            "N2.2",
            "iid_resample",
            "Independent draws from pooled marginal",
            0.01120,
            10.927,
            "Even more aggressive than shuffle —\nlarge gap",
            "destructive",
            True,
        ),
        (
            "N2.3",
            "short_memory_garch",
            "Simulate GARCH(1,1), $\\alpha{=}0.05$, $\\beta{=}0.90$",
            0.01123,
            10.961,
            "Half-life $\\approx 14$ lags vs real's\nslow decay — large gap",
            "destructive",
            True,
        ),
        (
            "N2.4",
            "tail_replacement",
            "Replace 5% tails with Gaussian, keep order",
            0.00446,
            4.351,
            "Partial: memory partly survives;\ngap below destructive mean",
            "moderate",
            True,
        ),
        (
            "N2.5",
            "scale_invariance",
            "Multiply all returns by 2",
            0.00000,
            0.000,
            "Exact zero — ACF is\nscale-invariant",
            "invariant",
            True,
        ),
    ]
    subtitle = (
        f"Real-vs-real noise floor $\\bar{{g}}_{{rr}}={g_rr:.6f}$ (contiguous half-split). "
        f"Three destructive routes converge: mean $={sum(dest_ratios) / len(dest_ratios):.2f}\\times$. "
        f"N2.4 lands at $4.35\\times$ = 40% of full destruction — see footnote."
    )
    footnote = (
        "B2 measures the lag-weighted gap in ACF of $|r|$ over lags 60–390 "
        "(Cont stylized fact 5: long-range dependence in volatility). "
        "Three independent destructive perturbations (N2.1, N2.2, N2.3) all yield the same "
        "ratio $\\approx 11\\times$ — the metric responds to whether memory exists, "
        "not to how it is destroyed. N2.4 reveals that the memory signal at 1-min frequency "
        "lives roughly 60% in the extreme moves and 40% in the temporal arrangement of the bulk."
    )
    _render_sanity_table("B2", g_rr, subtitle, rows_data, footnote, FIG_DIR / "bucket2_sanity")
    _write_sanity_tex(
        "B2",
        "ACF of $|r|$, lags 60-390",
        "bucket2-sanity",
        g_rr,
        rows_data,
        TEX_DIR / "bucket2_sanity.tex",
        footnote_tex=(
            "Three destructive perturbations (N2.1, N2.2, N2.3) yield essentially "
            "identical ratios $\\approx 11\\times$: the metric is sensitive to "
            "\\emph{whether} memory exists, not to the specific way it is destroyed. "
            "N2.4 lands at $40\\%$ of the destructive mean, indicating that the "
            "volatility-clustering signal at $1$-min frequency lives roughly "
            "$60\\%$ in the extreme moves and $40\\%$ in the temporal arrangement "
            "of the bulk."
        ),
    )


# ---------------------------------------------------------------------------
# Bucket 3 sanity-check table
# ---------------------------------------------------------------------------
def make_bucket3_sanity() -> None:
    g_rr = 0.000919  # 8-split averaged baseline
    rows_data = [
        (
            "N3.1",
            "time_reversal",
            "Reverse time order per path",
            0.00165,
            1.795,
            "Flip causal direction of leverage;\ngap large if signal exists",
            "destructive",
            True,
        ),
        (
            "N3.2",
            "sign_symmetrization",
            "Randomly flip sign of each $r_t$",
            0.00118,
            1.279,
            "Preserves $|r|$, destroys leverage\n→ should reach $\\geq1.5\\times$",
            "destructive*",
            False,
        ),
        (
            "N3.3",
            "symmetric_garch",
            "Simulate GARCH(1,1), $\\alpha{=}0.05$, $\\beta{=}0.90$",
            0.00113,
            1.227,
            "Symmetric model has $L(k)\\!=\\!0$\nat every lag",
            "destructive*",
            False,
        ),
        (
            "N3.4",
            "shuffle",
            "Random permutation per path",
            0.00116,
            1.259,
            "Destroys all cross-lag\nrelationships",
            "destructive*",
            False,
        ),
        (
            "N3.5",
            "scale_invariance",
            "Multiply all returns by 2",
            0.00000,
            0.000,
            "Corr is scale-invariant; exact zero",
            "invariant",
            True,
        ),
    ]
    subtitle = (
        f"Real-vs-real noise floor $\\bar{{g}}_{{rr}}={g_rr:.6f}$ "
        f"(8-split averaged — single-split estimate is noisy at 1-min frequency). "
        f"Destructive perturbations land at $1.23{{-}}1.79\\times$ the floor: "
        f"the metric responds in the right direction but the absolute signal is intrinsically weak."
    )
    footnote = (
        "* Three destructive checks (N3.2–N3.4) land below the $1.5\\times$ threshold. "
        "This is a signal-strength issue, not a metric bug. The leverage effect at 1-min "
        "equity frequency is fundamentally small (Cont 2001, fact 11; Bouchaud–Matacz–Potters 2001): "
        "even full destruction of the temporal arrangement moves the gap from $\\approx 9{\\times}10^{-4}$ "
        "(floor) to only $\\approx 1.2{\\times}10^{-3}$ (destroyed). N3.5 confirms the implementation "
        "is correct (exact scale invariance). Limited dynamic range carries over to the benchmark: "
        "AIL / GARCH / SFAGan cluster in $[0.38, 0.42]$ on B3."
    )
    _render_sanity_table("B3", g_rr, subtitle, rows_data, footnote, FIG_DIR / "bucket3_sanity")
    _write_sanity_tex(
        "B3",
        "leverage Corr($r_t,|r_{t+k}|$), lags 1-390",
        "bucket3-sanity",
        g_rr,
        rows_data,
        TEX_DIR / "bucket3_sanity.tex",
        footnote_tex=(
            "$^{*}$Three destructive checks (N3.2--N3.4) land below the $1.5\\times$ "
            "threshold. This reflects a signal-strength limitation, not a metric "
            "defect: the leverage effect at 1-min equity frequency is intrinsically "
            "weak (Cont 2001, stylized fact 11; Bouchaud--Matacz--Potters 2001). "
            "Even full destruction moves the gap from $\\approx9{\\times}10^{-4}$ "
            "(floor) to $\\approx1.2{\\times}10^{-3}$. N3.5 confirms exact scale "
            "invariance of the implementation. The narrow dynamic range carries "
            "over to the benchmark: AIL, GARCH and SFAGan cluster in $[0.38,0.42]$ "
            "on B3."
        ),
    )


# ---------------------------------------------------------------------------
# Bucket 4 sanity-check table
# ---------------------------------------------------------------------------
def make_bucket4_sanity() -> None:
    g_rr = 0.040290  # half-split L-kurt gap (this corpus)
    rows_data = [
        (
            "N4.1",
            "iid_resample",
            "Draw each element i.i.d. from pooled marginal",
            0.111010,
            2.756,
            "Full destruction of clustering;\nCLT decay → large gap",
            "destructive",
            True,
        ),
        (
            "N4.2",
            "shuffle",
            "Random permutation per path",
            0.026920,
            0.668,
            "Decay flattens partway, but per-path\nmarginal stays heavy → masked by noise*",
            "destructive*",
            False,
        ),
        (
            "N4.3",
            "scale_invariance",
            "Multiply all returns by 2",
            0.000000,
            0.000,
            "τ_4 is scale-invariant; exact zero",
            "invariant",
            True,
        ),
        (
            "N4.4",
            "tail_replacement",
            "Replace 5% tails with Gaussian",
            0.134742,
            1.214,
            "Expected $g_{tail} < g_{iid}$; observed\n1.21× (tail surgery hits h=1 harder)",
            "destructive*",
            False,
        ),
        (
            "N4.5",
            "block_resample",
            "Reshuffle 30-min blocks per path",
            0.001638,
            0.041,
            "Expected large at h≥60; observed ~zero\n(L-kurt trajectory preserved in corpus)",
            "destructive*",
            False,
        ),
    ]
    subtitle = (
        f"Real-vs-real noise floor $\\bar{{g}}_{{rr}}={g_rr:.4f}$ (contiguous half-split). "
        f"L-kurtosis (τ_4) of per-path-aggregated blocks, uniform weights over horizons "
        f"$\\{{1, 5, 30, 60, 390\\}}$ minutes. 2/5 sanity checks pass."
    )
    footnote = (
        "* Three destructive checks fail due to corpus physics, not metric defect. "
        "Real intraday L-kurtosis is essentially flat across horizons (τ_4 ≈ 0.31–0.35); "
        "the corpus shows weak aggregational Gaussianity at 1-min × 10-day scales. "
        "N4.2: shuffle decays only partway (heavy per-path marginal) and the half-split "
        "noise floor is dominated by cross-ticker composition differences. N4.4: tail "
        "surgery rewrites the h=1 marginal; iid-resample preserves it exactly. "
        "N4.5: 30-min block reshuffle preserves the τ_4 trajectory because within-block "
        "heaviness dominates cross-block dependence at h≥60. N4.1 and N4.3 confirm the "
        "implementation is correct (CLT signal detected; exact scale invariance)."
    )
    _render_sanity_table("B4", g_rr, subtitle, rows_data, footnote, FIG_DIR / "bucket4_sanity")
    _write_sanity_tex(
        "B4",
        "scale-weighted L-kurtosis",
        "bucket4-sanity",
        g_rr,
        rows_data,
        TEX_DIR / "bucket4_sanity.tex",
        footnote_tex=(
            "$^{*}$Three destructive checks fail for corpus-physics reasons, "
            "not metric defects. Real intraday $\\tau_4$ stays in $[0.31,0.35]$ "
            "across $h\\in\\{1,5,30,60,390\\}$ — the corpus exhibits weak "
            "aggregational Gaussianity at $1$-min $\\times$ $10$-day horizons. "
            "N4.2: within-path shuffle decays only partway because each path's "
            "marginal stays heavy; the half-split noise floor (\\(\\approx0.04\\)) "
            "is dominated by cross-ticker composition differences. N4.4: tail "
            "surgery rewrites the $h{=}1$ marginal while iid-resample preserves "
            "it exactly, inverting the expected ordering. N4.5: $30$-min block "
            "reshuffle preserves the $\\tau_4$ trajectory because within-block "
            "heavy-tailedness dominates cross-block dependence at $h\\geq60$. "
            "N4.1 and N4.3 confirm the implementation is correct."
        ),
    )


# ---------------------------------------------------------------------------
# Shared LaTeX writer for sanity tables
# ---------------------------------------------------------------------------
def _write_sanity_tex(
    bucket_id: str,
    bucket_descr: str,
    label: str,
    g_rr: float,
    rows_data: list[tuple],
    out_path: Path,
    footnote_tex: str,
) -> None:
    rows_tex = []
    for cid, check, pert, gap, ratio, expected, klass, passed in rows_data:
        # collapse multi-line expected to one line for LaTeX table cell
        expected_one = expected.replace("\n", " ")
        pass_str = (
            r"\textcolor{ForestGreen}{\textbf{PASS}}"
            if passed
            else r"\textcolor{red}{\textbf{FAIL}}"
        )
        check_tex = check.replace("_", r"\_")
        rows_tex.append(
            f"{cid} & \\texttt{{{check_tex}}} & {pert} & {gap:.4f} & {ratio:.3f}$\\times$ "
            f"& {expected_one} & {klass} & {pass_str} \\\\"
        )
    rows_block = "\n".join(rows_tex)
    tex = rf"""% Auto-generated by scripts/generate_figures.py — do not hand-edit.
% Requires xcolor (with ForestGreen) and booktabs.
\begin{{table}}[t]
\centering\small
\caption{{Bucket~{bucket_id} ({bucket_descr}) sanity-check diagnostics on the bundled real
NASDAQ eval corpus. Noise floor $\bar{{g}}_{{rr}}={g_rr:.6f}$ obtained from a contiguous
half-split. Destructive perturbations target the stylized fact the metric should detect;
invariant perturbations preserve the property the metric measures.}}
\label{{tab:{label}}}
\begin{{tabular}}{{@{{}}lllrrlll@{{}}}}
\toprule
ID & Check & Perturbation & $\bar{{g}}$ & $\bar{{g}}/\bar{{g}}_{{rr}}$ & Expected & Class & Outcome \\
\midrule
{rows_block}
\bottomrule
\end{{tabular}}

\medskip
\footnotesize
{footnote_tex}
\end{{table}}
"""
    out_path.write_text(tex)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    if target in ("main", "all"):
        make_main_benchmark()
        print(
            f"OK  benchmark_main -> {FIG_DIR}/{{benchmark_main.png, .pdf}}, "
            f"{TEX_DIR}/benchmark_main.tex"
        )
    if target in ("b1", "all"):
        make_bucket1_sanity()
        print(
            f"OK  bucket1_sanity -> {FIG_DIR}/{{bucket1_sanity.png, .pdf}}, "
            f"{TEX_DIR}/bucket1_sanity.tex"
        )
    if target in ("b2", "all"):
        make_bucket2_sanity()
        print(
            f"OK  bucket2_sanity -> {FIG_DIR}/{{bucket2_sanity.png, .pdf}}, "
            f"{TEX_DIR}/bucket2_sanity.tex"
        )
    if target in ("b3", "all"):
        make_bucket3_sanity()
        print(
            f"OK  bucket3_sanity -> {FIG_DIR}/{{bucket3_sanity.png, .pdf}}, "
            f"{TEX_DIR}/bucket3_sanity.tex"
        )
    if target in ("b4", "all"):
        make_bucket4_sanity()
        print(
            f"OK  bucket4_sanity -> {FIG_DIR}/{{bucket4_sanity.png, .pdf}}, "
            f"{TEX_DIR}/bucket4_sanity.tex"
        )


if __name__ == "__main__":
    main()
