#!/usr/bin/env python3
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qss_common import QSS_WORK, RESULTS, check_budget, connect, log, write_run

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#777777"


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight", "figure.constrained_layout.use": True,
    })


def panel_labels(axes):
    for label, axis in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
        axis.text(-0.13, 1.04, label, transform=axis.transAxes, fontsize=8,
                  fontweight="bold", va="bottom")


def save(fig, name):
    paths = []
    for suffix in ("pdf", "png"):
        path = RESULTS / f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        if path.stat().st_size < 10_000:
            raise ValueError(f"expected {path} >=10,000 bytes, got {path.stat().st_size}")
        paths.append(path)
    plt.close(fig)
    return paths


def forest(axis, frame, labels, color=BLUE, xlabel="Adjusted difference in five-year citations"):
    y = np.arange(len(frame))[::-1]
    axis.axvline(0, color="black", lw=0.6)
    axis.errorbar(frame.estimate, y,
                  xerr=np.vstack([frame.estimate - frame.ci_low, frame.ci_high - frame.estimate]),
                  fmt="o", color=color, ecolor=color, capsize=2)
    axis.set_yticks(y, labels)
    axis.set_xlabel(xlabel)


def measurement_figure(con):
    scope = con.execute("""
        SELECT * FROM read_parquet(?) USING SAMPLE 100000 ROWS (reservoir, 20260902)
        WHERE semantic_title_n>=100
    """, [str(QSS_WORK / "journal_year_scope.parquet")]).df()
    names = con.execute("""
        SELECT journal_id,any_value(journal_name) AS journal_name
        FROM read_parquet(?) GROUP BY journal_id
    """, [str(QSS_WORK / "focal_base.parquet")]).df()
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.25))
    complete = scope[(scope.semantic_title_half0_n >= 50) & (scope.semantic_title_half1_n >= 50)]
    axes[0].scatter(complete.semantic_title_half0, complete.semantic_title_half1,
                    s=3, alpha=0.18, color=BLUE, linewidths=0)
    axes[0].set(xlabel="Hash half 1 similarity", ylabel="Hash half 2 similarity",
                title="Split-half reliability")
    valid = scope.dropna(subset=["reference_field_hhi"])
    axes[1].scatter(valid.semantic_title_similarity, valid.reference_field_hhi,
                    s=3, alpha=0.18, color=GREEN, linewidths=0)
    axes[1].set(xlabel="Title-embedding similarity", ylabel="Reference-field HHI",
                title="Independent measurement")
    representative = scope.merge(names, on="journal_id").dropna(subset=["journal_name"])
    quantiles = representative.semantic_title_similarity.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    selected = pd.concat([
        representative.iloc[(representative.semantic_title_similarity - value).abs().argsort()[:1]]
        for value in quantiles
    ]).sort_values("semantic_title_similarity")
    labels = [str(value)[:38] for value in selected.journal_name]
    axes[2].barh(np.arange(len(selected)), selected.semantic_title_similarity, color=ORANGE)
    axes[2].set_yticks(np.arange(len(selected)), labels)
    axes[2].set(xlabel="Title-embedding similarity", title="Representative journal-years")
    panel_labels(axes)
    return save(fig, "figure1_measurement")


def diagnostics_figure(con, estimates, balance):
    sample = con.execute("""
        SELECT treatment,propensity,common_support FROM read_parquet(?)
        USING SAMPLE 500000 ROWS (reservoir, 20260902) WHERE treatment IS NOT NULL
    """, [str(QSS_WORK / "analysis_dataset.parquet")]).df()
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.25))
    bins = np.linspace(0, 1, 41)
    for arm, label, color in [(0, "Broad", BLUE), (1, "Specialized", ORANGE)]:
        axes[0].hist(sample.loc[sample.treatment.eq(arm), "propensity"], bins=bins,
                     density=True, histtype="step", color=color, label=label)
    axes[0].axvspan(0, 0.05, color=GRAY, alpha=0.12)
    axes[0].axvspan(0.95, 1, color=GRAY, alpha=0.12)
    axes[0].set(xlabel="Out-of-fold propensity", ylabel="Density", title="Common support")
    axes[0].legend()
    b = balance[balance.exposure.eq("semantic_title")]
    raw = b[b.stage.eq("raw")].set_index("covariate").smd.abs().sort_values(ascending=False).head(12)
    weighted = b[b.stage.eq("weighted")].set_index("covariate").reindex(raw.index).smd.abs()
    y = np.arange(len(raw))[::-1]
    axes[1].scatter(raw, y, color=GRAY, s=12, label="Raw")
    axes[1].scatter(weighted, y, color=GREEN, s=12, label="Weighted")
    axes[1].axvline(0.10, color="black", lw=0.6, ls="--")
    axes[1].set_yticks(y, raw.index)
    axes[1].set(xlabel="Absolute standardized mean difference", title="Covariate balance")
    axes[1].legend()
    uptake = estimates[(estimates.exposure.eq("semantic_title")) &
                       (estimates.scale.eq("absolute")) &
                       (estimates.outcome.isin(["total_citations", "any_citation"]))]
    forest(axes[2], uptake, ["Total citations", "Any citation"],
           xlabel="Adjusted mean or probability difference")
    axes[2].set_title("Overall uptake")
    panel_labels(axes)
    return save(fig, "figure2_diagnostics")


def decomposition_figure(estimates):
    order = ["within_subfield", "cross_subfield", "cross_field", "unclassified",
             "cross_field_minus_within_subfield"]
    frame = estimates[(estimates.exposure.eq("semantic_title")) &
                      (estimates.scale.eq("absolute"))].set_index("outcome").loc[order].reset_index()
    labels = ["Within subfield", "Different subfield, same field", "Different field",
              "Unclassified", "Cross-field minus within-subfield"]
    fig, axis = plt.subplots(figsize=(3.50, 2.55))
    forest(axis, frame, labels, ORANGE)
    axis.set_title("Routing of citation attention")
    return save(fig, "figure3_decomposition")


def heterogeneity_figure(estimates):
    frame = estimates[(estimates.rq.eq("RQ3")) &
                      (estimates.estimand.eq("subgroup common-support ATE"))].copy()
    frame["quartile"] = frame.population.str.extract(r"(\d)$")[0].astype(int)
    frame = frame.sort_values("quartile")
    fig, axis = plt.subplots(figsize=(3.50, 2.55))
    axis.axhline(0, color="black", lw=0.6)
    axis.errorbar(frame.quartile, frame.estimate,
                  yerr=np.vstack([frame.estimate - frame.ci_low, frame.ci_high - frame.estimate]),
                  fmt="o-", color=BLUE, capsize=2)
    axis.set_xticks([1, 2, 3, 4])
    axis.set(xlabel="Paper reference-field entropy quartile",
             ylabel="Adjusted cross-field citation difference",
             title="Venue-paper fit")
    return save(fig, "figure4_interdisciplinarity")


def main():
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    estimates_path = RESULTS / "causal_estimates.csv"
    balance_path = RESULTS / "balance.csv"
    if not estimates_path.is_file() or not balance_path.is_file():
        raise FileNotFoundError("expected completed causal estimates and balance outputs")
    estimates = pd.read_csv(estimates_path)
    balance = pd.read_csv(balance_path)
    if estimates.empty or balance.empty:
        raise ValueError(f"expected nonempty figure inputs, got estimates={len(estimates)}, balance={len(balance)}")
    style()
    con = connect("50GB", 8)
    paths = measurement_figure(con)
    paths += diagnostics_figure(con, estimates, balance)
    paths += decomposition_figure(estimates)
    paths += heterogeneity_figure(estimates)
    write_run("figures", "complete", {"figures": 4, "files": len(paths)})
    log("figures complete " + ", ".join(path.name for path in paths))


if __name__ == "__main__":
    main()
