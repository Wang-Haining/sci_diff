#!/usr/bin/env python3
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.patches import FancyArrowPatch

from qss_article_figures import CORAL, INK, LIGHT_GRAY, MID_GRAY, SKY, WHITE, style
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
PCS = [f"qpc{i:02d}" for i in range(1, 33)]
GROUPS = {0: "Broader quartile", 1: "Narrower quartile", 2: "Middle 50%"}
COLORS = {0: SKY, 1: CORAL, 2: "#B9BEC7"}
CASES = {
    ("Biotechnology Letters", 4): (-64, 7),
    ("Protein Expression and Purification", 4): (18, -10),
    ("Journal of Materials Science", 7): (-70, 8),
    ("Journal of Solid State Electrochemistry", 7): (18, -8),
}
EXPECTED = {0: 3_268_625, 1: 4_349_037, 2: 7_499_589}


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

    style()
    fig = plt.figure(figsize=(183 / 25.4, 145 / 25.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.15, 0.72], left=0.045, right=0.98,
                           bottom=0.06, top=0.95, wspace=0.18)
    axis = fig.add_subplot(grid[0, 0])
    metric_grid = grid[0, 1].subgridspec(5, 1, height_ratios=[0.55, 1, 1, 1, 1.25], hspace=0.78)
    title_axis = fig.add_subplot(metric_grid[0, 0]); title_axis.axis("off")
    metric_axes = [fig.add_subplot(metric_grid[i, 0]) for i in (1, 2, 3)]
    legend_axis = fig.add_subplot(metric_grid[4, 0]); legend_axis.axis("off")
    title_axis.text(0, 0.98, "Network-wide evidence", fontsize=7, fontweight="bold", va="top")

    for macro in (4, 7, 8, 12):
        row = areas.loc[areas.qwen_macro.eq(macro)].iloc[0]
        xx, y0 = layer_xy([row.umap_x], [row.umap_y], bounds, paper_base, layer_h)
        _, y1 = layer_xy([row.umap_x], [row.umap_y], bounds, citation_base, layer_h)
        axis.plot([xx[0], xx[0]], [y0[0], y1[0]], color="#BFC4CB", lw=0.35,
                  ls=(0, (2, 3)), alpha=0.42, zorder=0)

    for group in (2, 0, 1):
        mask = sample.scope_group.eq(group).to_numpy()
        axis.scatter(px[mask], py[mask], s=0.16 if group == 2 else 0.20,
                     color=COLORS[group], alpha=0.07 if group == 2 else 0.16,
                     linewidths=0, rasterized=True, zorder=1)
    for group in (0, 1, 2):
        axis.scatter([], [], s=16, color=COLORS[group], alpha=0.9,
                     label=f"{GROUPS[group]}  {full_counts[group] / 1e6:.2f}m")
    axis.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.99, paper_base + 0.005),
                fontsize=5.2, handletextpad=0.25, borderaxespad=0)

    halo_lo, halo_hi = np.quantile(np.log(journals.reach), [0.05, 0.95])
    halo = 35 + 170 * np.clip((np.log(journals.reach) - halo_lo) / (halo_hi - halo_lo), 0, 1)
    journal_colors = np.where(journals.treatment.eq(1), CORAL, SKY)
    axis.scatter(jx, jy, s=halo, c=journal_colors, alpha=0.12, linewidths=0, zorder=2)
    axis.scatter(jx, jy, s=13, c=journal_colors, edgecolors=WHITE, linewidths=0.35, zorder=3)
    for idx, row in journals.iterrows():
        key = (row.journal_name, int(row.qwen_macro))
        if key in CASES:
            axis.annotate(row.journal_name, (jx[idx], jy[idx]), xytext=CASES[key],
                          textcoords="offset points", fontsize=4.8, color=INK,
                          arrowprops={"arrowstyle": "-", "lw": 0.3, "color": MID_GRAY})

    pooled_max = selected.pooled_standardized_share.max()
    shift_max = selected.standardized_share_difference.abs().max()
    for row in selected.itertuples():
        start, end = area_xy[int(row.source_macro)], area_xy[int(row.target_macro)]
        axis.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.10",
                       arrowstyle="-", linewidth=0.12 + 0.65 * row.pooled_standardized_share / pooled_max,
                       color=LIGHT_GRAY, alpha=0.36, shrinkA=2, shrinkB=2, zorder=1))
    for row in selected.itertuples():
        start, end = area_xy[int(row.source_macro)], area_xy[int(row.target_macro)]
        increase = row.standardized_share_difference >= 0
        axis.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.10",
                       arrowstyle="-|>", mutation_scale=3.2,
                       linewidth=0.15 + 0.70 * abs(row.standardized_share_difference) / shift_max,
                       color=CORAL if increase else SKY, alpha=0.56,
                       linestyle="-" if increase else "--", shrinkA=3, shrinkB=3, zorder=2))
    node_sizes = 9 + 105 * areas.source_share / areas.source_share.max()
    axis.scatter(axx, ayy, s=node_sizes, facecolor=WHITE, edgecolor=INK, linewidth=0.42, zorder=4)
    for row in areas.nlargest(6, "source_share").itertuples():
        index = areas.index[areas.qwen_macro.eq(row.qwen_macro)][0]
        dx = 7 if axx[index] < 0.52 else -7
        axis.annotate(row.display_label, (axx[index], ayy[index]), xytext=(dx, 5),
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
    axis.annotate("", xy=(-0.015, 0.93), xytext=(-0.015, 0.07),
                  arrowprops={"arrowstyle": "-|>", "lw": 0.55, "color": MID_GRAY})
    axis.text(-0.045, 0.50, "publication and follow-up", rotation=90, ha="center", va="center",
              fontsize=5.0, color=MID_GRAY)
    axis.set(xlim=(-0.07, 1.01), ylim=(0, 1), xticks=[], yticks=[])
    for spine in axis.spines.values():
        spine.set_visible(False)

    metrics = pd.read_csv(RESULTS / "network_metrics.csv").set_index("metric")
    metric_specs = [("directed_modularity", "Within-area retention", "higher"),
                    ("audience_participation", "Diversity of citing areas", "lower"),
                    ("semantic_span", "Mean semantic distance", "lower")]
    for metric_axis, (metric, label, direction) in zip(metric_axes, metric_specs):
        row = metrics.loc[metric]
        point = 100 * row.contrast_specialized_minus_broad
        low, high = 100 * row.bootstrap_ci_low, 100 * row.bootstrap_ci_high
        metric_axis.axvline(0, color=INK, lw=0.5)
        metric_axis.errorbar(point, 0, xerr=[[point - low], [high - point]], fmt="o",
                             ms=3.2, color=CORAL, capsize=1.5, lw=0.8)
        metric_axis.set(yticks=[], title=label)
        metric_axis.set_xlabel("Narrower − broader (×100)", fontsize=5)
        metric_axis.text(0.98, 0.08, f"{direction}: {point:+.2f}", transform=metric_axis.transAxes,
                         ha="right", fontsize=5.0, color=MID_GRAY)
    legend_axis.plot([], [], color=CORAL, lw=1, label="relatively more for narrower-scope papers")
    legend_axis.plot([], [], color=SKY, lw=1, ls="--", label="relatively more for broader-scope papers")
    legend_axis.scatter([], [], s=45, facecolor=CORAL, alpha=0.15, edgecolor="none",
                        label="larger journal halo = broader citation reach")
    legend_axis.legend(frameon=False, loc="upper left", fontsize=5.2, handlelength=1.5)
    fig.text(0.012, 0.965, "a", fontsize=8, fontweight="bold")
    fig.text(0.757, 0.965, "b", fontsize=8, fontweight="bold")
    fig.savefig(RESULTS / "figures/figure_semantic_hierarchy.pdf", dpi=300, facecolor=WHITE)
    fig.savefig(RESULTS / "figures/figure_semantic_hierarchy.png", dpi=300, facecolor=WHITE)
    plt.close(fig)

    outputs = [PAPER_POINTS, JOURNALS, AREAS, EDGES_OUT,
               RESULTS / "figures/figure_semantic_hierarchy.pdf",
               RESULTS / "figures/figure_semantic_hierarchy.png"]
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
