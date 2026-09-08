#!/usr/bin/env python3
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from qss_article_figures import CORAL, INK, LIGHT_GRAY, MID_GRAY, WHITE, style
from qss_common import SEED
from qss_v3_common import ARTIFACTS, RESULTS, V2_WORK, V3_WORK, check_budget, connect, log, write_run

ANALYSIS_ALL = V2_WORK / "analysis_dataset.parquet"
QWEN = V2_WORK / "qwen3_semantics.parquet"
ANALYSIS = V3_WORK / "analysis_dataset.parquet"
SCORES = V3_WORK / "routing_scores.parquet"
PAPER_POINTS = V3_WORK / "hierarchy_papers.parquet"
JOURNALS = RESULTS / "hierarchy_journals.csv"
AREAS = RESULTS / "hierarchy_areas.csv"
EDGES_OUT = RESULTS / "hierarchy_edges.csv"
FIGURE = RESULTS / "figures/figure2_main_results"
YEAR_SOURCE = RESULTS / "figure2_years.csv"
PCS = [f"qpc{i:02d}" for i in range(1, 33)]
GROUPS = {0: "Broader-scope journals", 1: "Narrower-scope journals",
          2: "Middle 50% (not compared)"}
COLORS = {0: "#2386B8", 1: CORAL, 2: "#858C96"}
AREA_LABEL_OFFSETS = {
    4: (-8, 9), 7: (-5, 11), 8: (-8, -10), 18: (9, 8), 30: (-10, 8), 31: (-8, -11)
}
CASES = {
    ("Biotechnology Letters", 4): (-64, 7),
    ("Protein Expression and Purification", 4): (18, -10),
    ("Journal of Materials Science", 7): (-70, 8),
    ("Journal of Solid State Electrochemistry", 7): (18, -8),
}
EXPECTED = {0: 3_268_625, 1: 4_349_037, 2: 7_499_589}


def result_row(table, outcome):
    rows = table.loc[(table.analysis.eq("primary")) & table.outcome.eq(outcome)]
    if len(rows) != 1:
        raise ValueError(f"expected one primary {outcome} row, got {len(rows)}")
    return rows.iloc[0]


def draw_primary_results(axis, estimates):
    axis.axis("off")
    axis.set(xlim=(0, 1), ylim=(0, 1))
    other = result_row(estimates, "far")
    same = result_row(estimates, "near")
    routing = result_row(estimates, "far_to_near_routing")
    total = result_row(estimates, "total_citations")
    any_other = result_row(estimates, "any_far")

    axis.text(0.02, 0.98, "Where citations accumulated", fontsize=7,
              fontweight="bold", va="top")
    columns = [(0.48, COLORS[0], "Broader-scope\njournals"),
               (0.80, COLORS[1], "Narrower-scope\njournals")]
    for x0, color, label in columns:
        axis.text(x0, 0.86, label, ha="center", va="center", fontsize=5.7,
                  fontweight="bold", color=color)
    rows = [(0.70, "From other\nresearch areas", other),
            (0.52, "From the\nsame topic", same)]
    for y0, label, row in rows:
        axis.text(0.03, y0, label, va="center", fontsize=5.4, color=INK)
        for x0, color, value in ((0.48, COLORS[0], row.mean_broad),
                                 (0.80, COLORS[1], row.mean_specialized)):
            axis.add_patch(FancyBboxPatch((x0 - 0.115, y0 - 0.065), 0.23, 0.13,
                           boxstyle="round,pad=0.008,rounding_size=0.018",
                           facecolor=color, edgecolor="none", alpha=0.10))
            axis.text(x0, y0 + 0.010, f"{value:.2f}", ha="center", va="center",
                      fontsize=9, fontweight="bold", color=color)
            axis.text(x0, y0 - 0.037, "citations per paper", ha="center", va="center",
                      fontsize=4.7, color=MID_GRAY)

    axis.text(0.03, 0.355, "Other area ÷\nsame topic", fontsize=5.4, va="center")
    axis.text(0.48, 0.355, f"{other.mean_broad:.2f} ÷ {same.mean_broad:.2f} = "
              f"{routing.mean_broad:.2f}", ha="center", va="center", fontsize=6.2,
              fontweight="bold", color=COLORS[0])
    axis.text(0.80, 0.355, f"{other.mean_specialized:.2f} ÷ {same.mean_specialized:.2f} = "
              f"{routing.mean_specialized:.2f}", ha="center", va="center", fontsize=6.2,
              fontweight="bold", color=COLORS[1])
    axis.plot([0.48, 0.48, 0.80, 0.80], [0.285, 0.265, 0.265, 0.285], color=INK, lw=0.6)
    ratio = np.exp(routing.estimate)
    lower = 100 * (1 - np.exp(routing.ci_high))
    upper = 100 * (1 - np.exp(routing.ci_low))
    axis.text(0.64, 0.215, f"{routing.mean_specialized:.2f} ÷ "
              f"{routing.mean_broad:.2f} = {ratio:.3f}", ha="center", fontsize=5.2,
              color=MID_GRAY)
    axis.text(0.64, 0.145, f"{100 * (1 - ratio):.1f}% lower", ha="center",
              fontsize=10, fontweight="bold", color=CORAL)
    axis.text(0.64, 0.095, f"95% CI, {lower:.1f}–{upper:.1f}% lower", ha="center",
              fontsize=5.2, color=MID_GRAY)
    axis.text(0.02, 0.025,
              f"Overall citations: {total.mean_broad:.2f} vs {total.mean_specialized:.2f}; "
              f"difference {total.estimate:.2f} (95% CI {total.ci_low:.2f} to {total.ci_high:.2f}).\n"
              f"Any other-area citation: {100 * any_other.mean_broad:.1f}% vs "
              f"{100 * any_other.mean_specialized:.1f}%.",
              fontsize=4.6, color=MID_GRAY, va="bottom", linespacing=1.25)


def draw_year_results(ratio_axis, effect_axis, years, tests):
    if list(years.level.astype(int)) != list(range(2015, 2021)):
        raise ValueError(f"expected publication years 2015-2020, got {years.level.tolist()}")
    if not (years.estimate < 0).all():
        raise ValueError("expected all six annual routing contrasts to be negative")
    y = np.arange(len(years))[::-1]
    broad = years.far_near_broad.to_numpy()
    narrow = years.far_near_specialized.to_numpy()
    for yy, left, right in zip(y, narrow, broad):
        ratio_axis.plot([left, right], [yy, yy], color="#C8CCD1", lw=1.0, zorder=1)
    ratio_axis.scatter(broad, y, s=16, color=COLORS[0], edgecolor=WHITE, lw=0.35,
                       zorder=3, label="broader")
    ratio_axis.scatter(narrow, y, s=16, color=COLORS[1], edgecolor=WHITE, lw=0.35,
                       zorder=3, label="narrower")
    for yy, b, n in zip(y, broad, narrow):
        ratio_axis.text(b + 0.010, yy, f"{b:.2f}", va="center", ha="left",
                        fontsize=4.5, color=COLORS[0])
        ratio_axis.text(n - 0.010, yy, f"{n:.2f}", va="center", ha="right",
                        fontsize=4.5, color=COLORS[1])
    ratio_axis.set(yticks=y, yticklabels=years.level.astype(int), xlim=(1.17, 1.64),
                   xlabel="Other-area citations per same-topic citation")
    ratio_axis.tick_params(axis="y", length=0)
    ratio_axis.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
                      fontsize=4.7, handletextpad=0.25, columnspacing=0.8)
    ratio_axis.spines[["top", "right", "left"]].set_visible(False)

    point = 100 * np.expm1(years.estimate.to_numpy())
    low = 100 * np.expm1(years.ci_low.to_numpy())
    high = 100 * np.expm1(years.ci_high.to_numpy())
    effect_axis.axvline(0, color=INK, lw=0.5)
    effect_axis.errorbar(point, y, xerr=[point - low, high - point], fmt="o",
                         ms=3.2, color=CORAL, ecolor=CORAL, elinewidth=0.75,
                         capsize=1.4, zorder=2)
    for yy, value in zip(y, point):
        effect_axis.text(-24.5, yy, f"{value:.1f}%", ha="left", va="center",
                         fontsize=4.5, color=INK)
    effect_axis.set(yticks=[], xlim=(-25, 13), xlabel="Narrower vs broader (%)")
    effect_axis.spines[["top", "right", "left"]].set_visible(False)
    global_p = float(tests.loc[tests.test.eq("publication_year_global"), "p_value"].iloc[0])
    trend_p = float(tests.loc[tests.test.eq("publication_year_linear_trend"), "p_value"].iloc[0])
    effect_axis.text(0.98, -0.37, f"year heterogeneity P={global_p:.3f}; trend P={trend_p:.3f}",
                     transform=effect_axis.transAxes, ha="right", fontsize=4.5,
                     color=MID_GRAY)


def layer_xy(x, y, bounds, base, height=0.245):
    x0, x1, y0, y1 = bounds
    return 0.08 + 0.86 * np.clip((np.asarray(x) - x0) / (x1 - x0), 0, 1), \
        base + height * np.clip((np.asarray(y) - y0) / (y1 - y0), 0, 1)


def main():
    check_budget()
    for path in (ANALYSIS_ALL, QWEN, ANALYSIS, SCORES,
                 RESULTS / "network_edges.csv", RESULTS / "network_nodes.csv",
                 RESULTS / "network_metrics.csv", RESULTS / "macro_labels.csv"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"expected nonempty input at {path}")
    con = connect()
    pc_sql = ",".join(f"a.{p}" for p in PCS)
    count_rows = con.execute(f"""
      SELECT coalesce(treatment,2)::INTEGER scope_group,count(*) n
      FROM read_parquet('{ANALYSIS_ALL}') WHERE NOT focal_ood GROUP BY 1 ORDER BY 1
    """).fetchall()
    full_counts = {int(group): int(n) for group, n in count_rows}
    if full_counts != EXPECTED or sum(full_counts.values()) != 15_117_251:
        raise ValueError(f"expected full paper counts {EXPECTED}, got {full_counts}")

    sample = con.execute(f"""
      WITH ranked AS (
        SELECT a.id,q.qwen_macro,a.journal_id,coalesce(a.treatment,2)::INTEGER scope_group,
          {pc_sql},row_number() OVER (PARTITION BY q.qwen_macro
            ORDER BY hash(a.id || '|hierarchy|{SEED}')) rn
        FROM read_parquet('{ANALYSIS_ALL}') a JOIN read_parquet('{QWEN}') q USING(id)
        WHERE NOT a.focal_ood
      ) SELECT * EXCLUDE(rn) FROM ranked WHERE rn<=6000
    """).df()
    macro_counts = sample.groupby("qwen_macro").size()
    if len(sample) != 192_000 or len(macro_counts) != 32 or not (macro_counts == 6000).all() \
            or set(sample.scope_group) != {0, 1, 2}:
        raise ValueError(f"expected 192000 papers, 6000 per area, and three groups; got "
                         f"rows={len(sample)} areas={len(macro_counts)} groups={sample.scope_group.unique()}")
    x = sample[PCS].to_numpy(dtype=np.float32)
    y = sample.qwen_macro.to_numpy(dtype=np.int32)
    if not np.isfinite(x).all():
        raise ValueError("expected finite Qwen3 principal components")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.08, metric="euclidean",
                        target_metric="categorical", target_weight=0.15,
                        random_state=SEED, transform_seed=SEED, n_jobs=1)
    xy = reducer.fit_transform(x, y=y)
    if xy.shape != (192_000, 2) or not np.isfinite(xy).all():
        raise ValueError(f"expected finite UMAP coordinates (192000,2), got {xy.shape}")
    sample[["umap_x", "umap_y"]] = xy
    sample[["id", "qwen_macro", "journal_id", "scope_group", "umap_x", "umap_y"]].to_parquet(
        PAPER_POINTS, index=False)

    means = ",".join(f"avg({p}) AS {p}" for p in PCS)
    case_names = ",".join("'" + name.replace("'", "''") + "'" for name, _ in CASES)
    journals = con.execute(f"""
      WITH base AS (
        SELECT a.journal_id,a.journal_name,s.qwen_macro,s.treatment,{pc_sql},
          CASE WHEN s.treatment=1 THEN (a.far-s.propensity*s.psi_far_1)/(1-s.propensity)
               ELSE (a.far-(1-s.propensity)*s.psi_far_0)/s.propensity END m_far,
          CASE WHEN s.treatment=1 THEN (a.near-s.propensity*s.psi_near_1)/(1-s.propensity)
               ELSE (a.near-(1-s.propensity)*s.psi_near_0)/s.propensity END m_near
        FROM read_parquet('{SCORES}') s JOIN read_parquet('{ANALYSIS}') a USING(id)
      ), grouped AS (
        SELECT journal_id,any_value(journal_name) journal_name,qwen_macro,treatment,count(*) n,
          avg(m_far)/avg(m_near) reach,{means} FROM base GROUP BY journal_id,qwen_macro,treatment
        HAVING count(*)>=500 AND avg(m_near)>0
      ), ranked AS (
        SELECT *,row_number() OVER(PARTITION BY treatment ORDER BY n DESC,journal_id,qwen_macro) rank
        FROM grouped
      ) SELECT * EXCLUDE(rank) FROM ranked WHERE rank<=35 OR journal_name IN ({case_names})
    """).df()
    arm_counts = journals.treatment.value_counts().to_dict()
    if len(journals) < 70 or min(arm_counts.values()) < 35:
        raise ValueError(f"expected >=35 journal markers per arm, got {arm_counts}")
    journals[["umap_x", "umap_y"]] = reducer.transform(journals[PCS].to_numpy(dtype=np.float32))
    if not np.isfinite(journals[["umap_x", "umap_y", "reach"]]).all().all():
        raise ValueError("expected finite journal coordinates and reach")
    journals.drop(columns=PCS).to_csv(JOURNALS, index=False)

    labels = pd.read_csv(RESULTS / "macro_labels.csv")
    labels.loc[labels.qwen_macro.eq(18), "display_label"] = "Mixed records"
    areas = sample.groupby("qwen_macro")[["umap_x", "umap_y"]].median().reset_index()
    areas = areas.merge(pd.read_csv(RESULTS / "network_nodes.csv").drop(columns=["mds_x", "mds_y"]),
                        on="qwen_macro", validate="one_to_one")
    areas = areas.merge(labels[["qwen_macro", "display_label"]], on="qwen_macro", validate="one_to_one")
    areas.to_csv(AREAS, index=False)

    all_edges = pd.read_csv(RESULTS / "network_edges.csv")
    cross = all_edges[all_edges.source_macro.ne(all_edges.target_macro)].copy()
    cross = cross.sort_values("pooled_standardized_share", ascending=False)
    cross["pooled_fraction"] = cross.pooled_standardized_share / cross.pooled_standardized_share.sum()
    cross["cumulative_fraction"] = cross.pooled_fraction.cumsum()
    selected = cross[cross.cumulative_fraction.sub(cross.pooled_fraction).lt(0.40)].copy()
    backbone_coverage = float(selected.pooled_fraction.sum())
    if not 0.40 <= backbone_coverage < 0.42 or len(selected) < 40:
        raise ValueError(f"expected treatment-blind citation backbone near 40%, got "
                         f"edges={len(selected)} coverage={backbone_coverage}")
    selected.to_csv(EDGES_OUT, index=False)

    xpad, ypad = np.ptp(xy[:, 0]) * 0.035, np.ptp(xy[:, 1]) * 0.035
    bounds = (xy[:, 0].min() - xpad, xy[:, 0].max() + xpad,
              xy[:, 1].min() - ypad, xy[:, 1].max() + ypad)
    paper_base, journal_base, citation_base, layer_h = 0.035, 0.365, 0.695, 0.245
    px, py = layer_xy(sample.umap_x, sample.umap_y, bounds, paper_base, layer_h)
    jx, jy = layer_xy(journals.umap_x, journals.umap_y, bounds, journal_base, layer_h)
    axx, ayy = layer_xy(areas.umap_x, areas.umap_y, bounds, citation_base, layer_h)
    area_xy = dict(zip(areas.qwen_macro.astype(int), zip(axx, ayy)))

    estimates = pd.read_csv(RESULTS / "dirty_estimates.csv")
    years = pd.read_csv(RESULTS / "subgroup_estimates.csv")
    years = years.loc[years.test.eq("publication_year")].sort_values("order").copy()
    tests = pd.read_csv(RESULTS / "subgroup_tests.csv")
    years.assign(
        ratio_change_percent=100 * np.expm1(years.estimate),
        ratio_ci_low_percent=100 * np.expm1(years.ci_low),
        ratio_ci_high_percent=100 * np.expm1(years.ci_high),
    ).to_csv(YEAR_SOURCE, index=False)

    style()
    fig = plt.figure(figsize=(183 / 25.4, 160 / 25.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.48, 1.0], left=0.04, right=0.985,
                           bottom=0.07, top=0.96, wspace=0.12)
    axis = fig.add_subplot(grid[0, 0])
    right = grid[0, 1].subgridspec(2, 1, height_ratios=[1.22, 0.78], hspace=0.24)
    result_axis = fig.add_subplot(right[0, 0])
    year_grid = right[1, 0].subgridspec(1, 2, width_ratios=[1.30, 0.85], wspace=0.27)
    ratio_axis = fig.add_subplot(year_grid[0, 0])
    effect_axis = fig.add_subplot(year_grid[0, 1])

    for macro in (4, 7, 8, 12):
        row = areas.loc[areas.qwen_macro.eq(macro)].iloc[0]
        xx, y0 = layer_xy([row.umap_x], [row.umap_y], bounds, paper_base, layer_h)
        _, y1 = layer_xy([row.umap_x], [row.umap_y], bounds, citation_base, layer_h)
        axis.plot([xx[0], xx[0]], [y0[0], y1[0]], color="#BFC4CB", lw=0.35,
                  ls=(0, (2, 3)), alpha=0.42, zorder=0)

    point_style = {2: (0.14, 0.10), 0: (0.20, 0.19), 1: (0.18, 0.13)}
    for group in (2, 0, 1):
        mask = sample.scope_group.eq(group).to_numpy()
        size, alpha = point_style[group]
        axis.scatter(px[mask], py[mask], s=size,
                     color=COLORS[group], alpha=alpha,
                     linewidths=0, rasterized=True, zorder=1)
    for group in (0, 1, 2):
        axis.scatter([], [], s=16, color=COLORS[group], alpha=0.9,
                     label=f"{GROUPS[group]}  {full_counts[group] / 1e6:.2f}m")
    axis.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.99, paper_base + 0.005),
                fontsize=5.2, handletextpad=0.25, borderaxespad=0)

    halo_lo, halo_hi = np.quantile(np.log1p(journals.n), [0.05, 0.95])
    halo = 35 + 170 * np.clip((np.log1p(journals.n) - halo_lo) / (halo_hi - halo_lo), 0, 1)
    journal_colors = np.where(journals.treatment.eq(1), CORAL, COLORS[0])
    axis.scatter(jx, jy, s=halo, c=journal_colors, alpha=0.12, linewidths=0, zorder=2)
    axis.scatter(jx, jy, s=13, c=journal_colors, edgecolors=WHITE, linewidths=0.35, zorder=3)
    for idx, row in journals.iterrows():
        key = (row.journal_name, int(row.qwen_macro))
        if key in CASES:
            axis.annotate(row.journal_name, (jx[idx], jy[idx]), xytext=CASES[key],
                          textcoords="offset points", fontsize=4.8, color=INK,
                          arrowprops={"arrowstyle": "-", "lw": 0.3, "color": MID_GRAY})

    pooled_max = selected.pooled_standardized_share.max()
    for row in selected.itertuples():
        start, end = area_xy[int(row.source_macro)], area_xy[int(row.target_macro)]
        axis.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.10",
                       arrowstyle="-|>", mutation_scale=3.2,
                       linewidth=0.12 + 0.65 * row.pooled_standardized_share / pooled_max,
                       color=LIGHT_GRAY, alpha=0.46, shrinkA=3, shrinkB=3, zorder=1))
    node_sizes = 9 + 105 * areas.source_share / areas.source_share.max()
    axis.scatter(axx, ayy, s=node_sizes, facecolor=WHITE, edgecolor=INK, linewidth=0.42, zorder=4)
    for row in areas.nlargest(6, "source_share").itertuples():
        index = areas.index[areas.qwen_macro.eq(row.qwen_macro)][0]
        dx, dy = AREA_LABEL_OFFSETS[int(row.qwen_macro)]
        axis.annotate(row.display_label, (axx[index], ayy[index]), xytext=(dx, dy),
                      textcoords="offset points", ha="left" if dx > 0 else "right",
                      fontsize=4.7, color=INK,
                      bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.88, "pad": 0.5})

    axis.text(0.01, citation_base + layer_h, "3  CITATION ORIGINS", fontsize=6.5,
              fontweight="bold", va="top")
    axis.text(0.01, journal_base + layer_h, "2  JOURNALS", fontsize=6.5,
              fontweight="bold", va="top")
    axis.text(0.01, paper_base + layer_h, "1  PAPERS", fontsize=6.5,
              fontweight="bold", va="top")
    axis.text(0.01, citation_base + layer_h - 0.025, "Where later citations came from", fontsize=5.2, color=MID_GRAY)
    axis.text(0.01, journal_base + layer_h - 0.025, "Where papers were published", fontsize=5.2, color=MID_GRAY)
    axis.text(0.01, paper_base + layer_h - 0.025, "What each paper studied", fontsize=5.2, color=MID_GRAY)
    axis.text(0.01, paper_base + layer_h - 0.043, "192,000-paper display sample", fontsize=4.6,
              color=MID_GRAY)
    axis.annotate("", xy=(-0.015, 0.93), xytext=(-0.015, 0.07),
                  arrowprops={"arrowstyle": "-|>", "lw": 0.55, "color": MID_GRAY})
    axis.text(-0.045, 0.50, "publication and follow-up", rotation=90, ha="center", va="center",
              fontsize=5.0, color=MID_GRAY)
    axis.set(xlim=(-0.07, 1.01), ylim=(0, 1), xticks=[], yticks=[])
    for spine in axis.spines.values():
        spine.set_visible(False)

    draw_primary_results(result_axis, estimates)
    draw_year_results(ratio_axis, effect_axis, years, tests)
    ratio_axis.text(-0.19, 1.17, "Across publication years", transform=ratio_axis.transAxes,
                    fontsize=7, fontweight="bold", va="bottom")
    fig.text(0.012, 0.965, "a", fontsize=8, fontweight="bold")
    fig.text(0.623, 0.965, "b", fontsize=8, fontweight="bold")
    fig.text(0.623, 0.425, "c", fontsize=8, fontweight="bold")
    fig.savefig(FIGURE.with_suffix(".pdf"), dpi=300, facecolor=WHITE)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=300, facecolor=WHITE)
    plt.close(fig)

    outputs = [PAPER_POINTS, JOURNALS, AREAS, EDGES_OUT, YEAR_SOURCE,
               FIGURE.with_suffix(".pdf"), FIGURE.with_suffix(".png")]
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"expected nonempty output at {path}")
    free = shutil.disk_usage(V3_WORK).free
    write_run("hierarchy", {"eligible_papers": sum(full_counts.values()), "shown_papers": len(sample),
              "journal_markers": len(journals), "research_areas": len(areas),
              "citation_backbone_edges": len(selected)},
              {"eligible_papers_by_scope_group": full_counts, "backbone_flow_coverage": backbone_coverage,
               "umap_supervision": "qwen_macro only", "group_free_bytes": free})
    check_budget()
    log(f"hierarchy complete eligible={sum(full_counts.values()):,} shown={len(sample):,} "
        f"journals={len(journals)} edges={len(selected)} coverage={backbone_coverage:.3f} free={free:,}")


if __name__ == "__main__":
    main()
