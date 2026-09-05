#!/usr/bin/env python3
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qss_v3_common import RESULTS, V2_WORK, check_budget, connect, log

FIGURES = RESULTS / "figures"
SKILL_SCRIPTS = Path("/Users/haining/.codex/skills/nature-visualizer/scripts")
if SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(SKILL_SCRIPTS))
    from nature_style import add_panel_labels, apply_style
else:
    def apply_style(*args, **kwargs):
        mpl.rcParams.update({
            "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
            "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.linewidth": 0.5, "lines.linewidth": 0.8,
            "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
            "figure.constrained_layout.use": True,
        })

    def add_panel_labels(axes, **kwargs):
        for label, axis in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(axes)):
            axis.text(-0.14, 1.03, label, transform=axis.transAxes,
                      fontsize=8, fontweight="bold", va="bottom")

CORAL = "#E64B35"
SKY = "#4DBBD5"
TEAL = "#00A087"
NAVY = "#3C5488"
GRAY = "#777777"


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path, facecolor="white", dpi=300, bbox_inches="tight")
        if path.stat().st_size < 10_000:
            raise ValueError(f"expected {path} >=10,000 bytes, got {path.stat().st_size}")
        paths.append(path)
    plt.close(fig)
    return paths


def forest(axis, frame, labels=None, color=NAVY, reference=0):
    frame = frame.reset_index(drop=True)
    y = np.arange(len(frame))[::-1]
    axis.axvline(reference, color="black", lw=0.6)
    axis.errorbar(frame.estimate, y,
                  xerr=np.vstack([frame.estimate - frame.ci_low,
                                  frame.ci_high - frame.estimate]),
                  fmt="o", ms=3.2, color=color, ecolor=color, capsize=1.5, lw=0.8)
    axis.set_yticks(y, labels if labels is not None else frame.level.astype(str))


def figure1_measurement(con):
    scope = con.execute("""
      SELECT semantic_title_similarity,semantic_title_half0,semantic_title_half1,
             semantic_title_n,semantic_title_half0_n,semantic_title_half1_n
      FROM read_parquet(?) USING SAMPLE 150000 ROWS (reservoir, 20260902)
      WHERE semantic_title_n>=100
    """, [str(V2_WORK / "journal_year_scope.parquet")]).df()
    split = scope[(scope.semantic_title_half0_n >= 50) & (scope.semantic_title_half1_n >= 50)]
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.25))
    axes[0].hist(scope.semantic_title_similarity.dropna(), bins=45,
                 color=NAVY, alpha=0.88, edgecolor="white", linewidth=0.2)
    axes[0].set(xlabel="Mean pairwise title similarity", ylabel="Journal-years",
                title="Venue specialization")
    axes[1].hexbin(split.semantic_title_half0, split.semantic_title_half1,
                   gridsize=45, mincnt=1, cmap="Blues", linewidths=0)
    limits = [min(split.semantic_title_half0.min(), split.semantic_title_half1.min()),
              max(split.semantic_title_half0.max(), split.semantic_title_half1.max())]
    axes[1].plot(limits, limits, color="black", lw=0.6, ls="--")
    axes[1].set(xlabel="Hash half 1", ylabel="Hash half 2",
                title="Split-half reliability = 0.985")
    axes[2].axis("off")
    axes[2].text(0.02, 0.82, "Exposure", weight="bold", transform=axes[2].transAxes)
    axes[2].text(0.02, 0.68, "Journal titles in years t−3 to t−1", transform=axes[2].transAxes)
    axes[2].annotate("", xy=(0.50, 0.55), xytext=(0.05, 0.55),
                     arrowprops={"arrowstyle": "->", "lw": 0.8}, xycoords="axes fraction")
    axes[2].text(0.02, 0.40, "Focal papers", weight="bold", transform=axes[2].transAxes)
    axes[2].text(0.02, 0.26, "2015–2020", transform=axes[2].transAxes)
    axes[2].annotate("", xy=(0.95, 0.55), xytext=(0.52, 0.55),
                     arrowprops={"arrowstyle": "->", "lw": 0.8}, xycoords="axes fraction")
    axes[2].text(0.57, 0.82, "Outcome", weight="bold", transform=axes[2].transAxes)
    axes[2].text(0.57, 0.68, "Citations in next 60 months", transform=axes[2].transAxes)
    axes[2].text(0.57, 0.40, "Near / intermediate / far", transform=axes[2].transAxes)
    axes[2].text(0.57, 0.26, "Venue-free Qwen3 taxonomy", transform=axes[2].transAxes)
    add_panel_labels(axes, venue="nature")
    return save(fig, "figure1_measurement_design")


def figure2_main(estimates):
    absolute = estimates[(estimates.analysis == "primary") &
                         estimates.outcome.isin(["total_citations", "near", "intermediate", "far"])]
    absolute = absolute.set_index("outcome").loc[["total_citations", "near", "intermediate", "far"]]
    labels = ["All external citations", "Near fields", "Intermediate fields", "Far fields"]
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.65), gridspec_kw={"width_ratios": [1.05, 1]})
    x = np.arange(len(absolute))
    width = 0.34
    axes[0].bar(x - width / 2, absolute.mean_broad, width, color=SKY, label="Broad journal")
    axes[0].bar(x + width / 2, absolute.mean_specialized, width, color=CORAL,
                label="Specialized journal")
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set(ylabel="Adjusted citations per paper", title="Adjusted marginal means")
    axes[0].legend(frameon=False)
    forest(axes[1], absolute.reset_index(), labels, CORAL)
    axes[1].set(xlabel="Specialized minus broad (citations per paper)",
                title="Differences in citation counts")
    theta = estimates[(estimates.analysis == "primary") &
                      (estimates.outcome == "far_to_near_routing")].iloc[0]
    axes[1].text(0.02, 0.04,
                 f"Far-to-near ratio: {100 * (np.exp(theta.estimate) - 1):.1f}%\n"
                 f"95% CI {100 * (np.exp(theta.ci_low) - 1):.1f} to "
                 f"{100 * (np.exp(theta.ci_high) - 1):.1f}%",
                 transform=axes[1].transAxes, va="bottom")
    add_panel_labels(axes, venue="nature")
    return save(fig, "figure2_main_results")


def modifier_panel(axis, data, test, title, labels):
    frame = data[(data.test == test) & (data.status == "estimated")].sort_values("order")
    forest(axis, frame, labels, TEAL)
    axis.set(xlabel="Change in log far-to-near citation ratio", title=title)


def figure3_modifiers(subgroups):
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.50))
    quartiles = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    modifier_panel(axes[0], subgroups, "paper_venue_fit", "Paper reference breadth", quartiles)
    modifier_panel(axes[1], subgroups, "author_audience_breadth", "Lead-author research breadth", quartiles)
    modifier_panel(axes[2], subgroups, "author_publication_experience", "Prior publication experience", quartiles)
    add_panel_labels(axes, venue="nature")
    return save(fig, "figure3_modifiers")


def figure4_generality(subgroups, labels):
    domains = subgroups[(subgroups.test == "semantic_domain") &
                        (subgroups.status == "estimated")].copy().sort_values("estimate")
    label_map = labels.set_index("qwen_macro").representative_journals.to_dict()
    domain_labels = [f"D{int(row.level):02d}: {str(label_map.get(int(row.level), ''))[:34]}"
                     for row in domains.itertuples()]
    years = subgroups[(subgroups.test == "publication_year") &
                      (subgroups.status == "estimated")].sort_values("level")
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 5.60), gridspec_kw={"width_ratios": [1.45, 1]})
    forest(axes[0], domains, domain_labels, NAVY)
    axes[0].set(xlabel="Change in log far-to-near citation ratio",
                title="Venue-free semantic domains")
    x = years.level.astype(int).to_numpy()
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].errorbar(x, years.estimate,
                     yerr=np.vstack([years.estimate - years.ci_low,
                                     years.ci_high - years.estimate]),
                     fmt="o-", color=CORAL, capsize=2)
    axes[1].set_xticks(x)
    axes[1].set(xlabel="Publication year", ylabel="Change in log far-to-near citation ratio",
                title="Stability across focal cohorts")
    add_panel_labels(axes, venue="nature")
    return save(fig, "figure4_domains_time")


def main():
    check_budget()
    estimates = pd.read_csv(RESULTS / "dirty_estimates.csv")
    subgroups = pd.read_csv(RESULTS / "subgroup_estimates.csv")
    labels = pd.read_csv(RESULTS / "macro_labels.csv")
    if estimates.empty or subgroups.empty or labels.empty:
        raise ValueError(f"expected nonempty figure inputs, got {len(estimates)}, "
                         f"{len(subgroups)}, {len(labels)}")
    apply_style("nature", n=4, mark="fill")
    con = connect("50GB", 8)
    paths = figure1_measurement(con)
    paths += figure2_main(estimates)
    paths += figure3_modifiers(subgroups)
    paths += figure4_generality(subgroups, labels)
    if len(paths) != 8:
        raise ValueError(f"expected 8 figure files, got {len(paths)}")
    log("article figures complete " + ", ".join(path.name for path in paths))


if __name__ == "__main__":
    main()
