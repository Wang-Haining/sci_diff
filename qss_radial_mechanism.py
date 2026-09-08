#!/usr/bin/env python3
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Wedge
from scipy.cluster.hierarchy import leaves_list, linkage, to_tree
from scipy.spatial.distance import squareform

from qss_article_figures import CORAL, INK, LIGHT_GRAY, MID_GRAY, WHITE, style
from qss_network import TAXONOMY, load_taxonomy
from qss_v3_common import ARTIFACTS, RESULTS, V2_WORK, check_budget, connect, log, write_run

ANALYSIS = V2_WORK / "analysis_dataset.parquet"
QWEN = V2_WORK / "qwen3_semantics.parquet"
LEAF_SCOPE = RESULTS / "radial_leaf_scope.csv"
AREA_ORDER = RESULTS / "radial_area_order.csv"
FLOW = RESULTS / "radial_citation_backbone.csv"
FIGURE = RESULTS / "figures/figure_radial_mechanism"
BLUE = "#2386B8"
GROUP_COLORS = {0: BLUE, 2: "#B7BBC2", 1: CORAL}
EXPECTED_GROUPS = {0: 3_268_625, 1: 4_349_037, 2: 7_499_589}
SHORT_LABELS = {
    0: "Economics & policy", 1: "Earth & environment", 2: "Reproduction & metabolism",
    3: "Electrical engineering", 4: "Plant & microbial biology", 5: "Energy engineering",
    6: "Clinical diagnostics", 7: "Electrochemical materials", 8: "Humanities & politics",
    9: "Cell & neural biology", 10: "Social behavior & violence", 11: "Mental health & cognition",
    12: "Computing & networks", 13: "Marine & paleoscience", 14: "Immune disease",
    15: "Cardiovascular medicine", 16: "Public health & care", 17: "Surgery",
    18: "Mixed records", 19: "Synthetic chemistry", 20: "Civil engineering",
    21: "Metallurgy & alloys", 22: "Mathematics & physics", 23: "Ecology & taxonomy",
    24: "Astronomy & imaging", 25: "Orthopedics & sports", 26: "Climate & agriculture",
    27: "Oncology", 28: "Education & language", 29: "Electronic materials",
    30: "Drug discovery", 31: "Chronic & infectious disease",
}


def xy(radius, angle):
    return radius * np.cos(angle), radius * np.sin(angle)


def curved_edge(axis, angle_a, angle_b, color, width, alpha, zorder):
    start = xy(0.50, angle_a)
    end = xy(0.50, angle_b)
    control_a = xy(0.16, angle_a)
    control_b = xy(0.16, angle_b)
    path = Path([start, control_a, control_b, end],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    axis.add_patch(PathPatch(path, facecolor="none", edgecolor=color,
                             lw=width, alpha=alpha, zorder=zorder))


def tree_nodes(root, macro_angles, max_distance):
    positions = {}

    def place(node):
        if node.is_leaf():
            positions[node.id] = (0.52, macro_angles[node.id])
            return [macro_angles[node.id]]
        angles = place(node.left) + place(node.right)
        radius = 0.08 + 0.40 * (1 - node.dist / max_distance)
        positions[node.id] = (radius, float(np.mean(angles)))
        return angles

    place(root)
    return positions


def draw_audience(axis, x0, y0, color, journal, same, other, ratio):
    axis.add_patch(FancyBboxPatch((x0 - 0.19, y0 + 0.13), 0.38, 0.085,
                                  boxstyle="round,pad=0.012,rounding_size=0.035",
                                  facecolor=WHITE, edgecolor=color, lw=1.0))
    axis.text(x0, y0 + 0.172, journal, ha="center", va="center", fontsize=5.8,
              fontweight="bold", color=INK)
    angles = np.linspace(np.pi * 0.10, np.pi * 0.90, 7)
    for index, angle in enumerate(angles):
        radius = 0.20
        target = (x0 + radius * np.cos(angle), y0 + radius * np.sin(angle) - 0.11)
        axis.plot([x0, target[0]], [y0 + 0.13, target[1]], color=color,
                  lw=0.45 if index < 4 else 0.28, alpha=0.58)
        axis.add_patch(Circle(target, 0.011, facecolor=(color if index < 4 else WHITE),
                              edgecolor=color, lw=0.45))
    axis.text(x0, y0 - 0.03, f"{other:.2f} from other areas", ha="center", fontsize=5.5)
    axis.text(x0, y0 - 0.085, f"{same:.2f} from the same topic", ha="center", fontsize=5.5,
              color=MID_GRAY)
    axis.text(x0, y0 - 0.15, f"{ratio:.2f} : 1", ha="center", fontsize=8,
              fontweight="bold", color=color)


def main():
    check_budget()
    for path in (ANALYSIS, QWEN, TAXONOMY, RESULTS / "network_nodes.csv",
                 RESULTS / "network_edges.csv", RESULTS / "dirty_estimates.csv"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"expected nonempty input at {path}")
    leaf_to_macro, macro_centers, _, _ = load_taxonomy()
    bundle = np.load(TAXONOMY)
    leaf_centers = bundle["leaf_centers"].astype(np.float64)
    leaf_centers /= np.linalg.norm(leaf_centers, axis=1, keepdims=True)
    if leaf_centers.shape != (1000, 768) or leaf_to_macro.shape != (1000,):
        raise ValueError(f"unexpected taxonomy shapes {leaf_centers.shape}, {leaf_to_macro.shape}")

    con = connect()
    counts = con.execute(f"""
      SELECT q.qwen_macro,q.qwen_leaf,coalesce(a.treatment,2)::INTEGER scope_group,count(*) n
      FROM read_parquet('{ANALYSIS}') a JOIN read_parquet('{QWEN}') q USING(id)
      WHERE NOT a.focal_ood AND NOT q.qwen_ood GROUP BY ALL ORDER BY 1,2,3
    """).df()
    totals = counts.groupby("scope_group").n.sum().astype(int).to_dict()
    if totals != EXPECTED_GROUPS or counts.qwen_leaf.nunique() != 998:
        raise ValueError(f"expected groups={EXPECTED_GROUPS} and 998 observed leaves, got "
                         f"groups={totals} leaves={counts.qwen_leaf.nunique()}")
    index = pd.MultiIndex.from_product([range(1000), (0, 2, 1)],
                                       names=["qwen_leaf", "scope_group"])
    complete = counts.set_index(["qwen_leaf", "scope_group"]).n.reindex(index, fill_value=0)
    wide = complete.unstack().reset_index()
    wide.columns = ["qwen_leaf", "broader", "middle", "narrower"]
    wide["qwen_macro"] = leaf_to_macro
    wide["total"] = wide[["broader", "middle", "narrower"]].sum(axis=1)

    macro_distance = np.clip(1 - macro_centers @ macro_centers.T, 0, 2)
    np.fill_diagonal(macro_distance, 0)
    hierarchy = linkage(squareform(macro_distance, checks=True), method="average",
                        optimal_ordering=True)
    macro_order = leaves_list(hierarchy).astype(int)
    ordered_leaves = []
    for macro in macro_order:
        members = np.flatnonzero(leaf_to_macro == macro)
        centered = leaf_centers[members] - leaf_centers[members].mean(axis=0)
        direction = np.linalg.svd(centered, full_matrices=False)[2][0]
        score = centered @ direction
        if score[np.argmax(np.abs(score))] < 0:
            score *= -1
        ordered_leaves.extend(members[np.argsort(score)].tolist())
    if len(ordered_leaves) != 1000 or len(set(ordered_leaves)) != 1000:
        raise ValueError("expected a unique order for all 1,000 topic leaves")

    gap = 0.010
    opening = np.deg2rad(50)
    leaf_width = (2 * np.pi - opening - 32 * gap) / 1000
    cursor = np.pi / 2 - opening / 2
    leaf_angles, area_rows = {}, []
    for macro in macro_order:
        members = [leaf for leaf in ordered_leaves if leaf_to_macro[leaf] == macro]
        end = cursor
        for leaf in members:
            leaf_angles[leaf] = cursor - leaf_width / 2
            cursor -= leaf_width
        start = cursor
        label = SHORT_LABELS[macro]
        area_rows.append({"qwen_macro": macro, "display_label": label, "start": start,
                          "end": end, "angle": (start + end) / 2, "leaves": len(members),
                          "papers": int(wide.loc[wide.qwen_macro.eq(macro), "total"].sum())})
        cursor -= gap
    areas = pd.DataFrame(area_rows)
    wide["angle"] = wide.qwen_leaf.map(leaf_angles)
    wide.to_csv(LEAF_SCOPE, index=False)
    areas.to_csv(AREA_ORDER, index=False)

    edges = pd.read_csv(RESULTS / "network_edges.csv")
    cross = edges.loc[edges.source_macro.ne(edges.target_macro)].copy()
    cross = cross.sort_values("pooled_standardized_share", ascending=False)
    cross["fraction"] = cross.pooled_standardized_share / cross.pooled_standardized_share.sum()
    cross["cumulative"] = cross.fraction.cumsum()
    selected = cross.loc[cross.cumulative.sub(cross.fraction).lt(0.30)].copy()
    coverage = float(selected.fraction.sum())
    if not 0.30 <= coverage < 0.32 or not 25 <= len(selected) <= 60:
        raise ValueError(f"unexpected circular backbone edges={len(selected)} coverage={coverage}")
    selected.to_csv(FLOW, index=False)

    style()
    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.72, 0.78], left=0.025, right=0.98,
                           bottom=0.035, top=0.97, wspace=0.04)
    wheel = fig.add_subplot(grid[0, 0]); mechanism = fig.add_subplot(grid[0, 1])
    wheel.set_aspect("equal"); wheel.axis("off"); mechanism.axis("off")

    macro_angles = dict(zip(areas.qwen_macro.astype(int), areas.angle))
    root, node_list = to_tree(hierarchy, rd=True)
    positions = tree_nodes(root, macro_angles, float(hierarchy[:, 2].max()))
    for node_id in range(32, 63):
        node = node_list[node_id]
        for child in (node.left, node.right):
            wheel.plot(*zip(xy(*positions[node_id]), xy(*positions[child.id])),
                       color="#D6DADF", lw=0.35, zorder=0)
    for macro in range(32):
        mx, my = xy(*positions[macro])
        members = np.flatnonzero(leaf_to_macro == macro)
        for leaf in members:
            lx, ly = xy(0.67, leaf_angles[int(leaf)])
            wheel.plot([mx, lx], [my, ly], color="#E4E6E9", lw=0.16, zorder=0)

    node_table = pd.read_csv(RESULTS / "network_nodes.csv").set_index("qwen_macro")
    max_shift = node_table.self_retention_difference.abs().max()
    for row in areas.itertuples():
        shift = node_table.loc[row.qwen_macro, "self_retention_difference"]
        color = CORAL if shift >= 0 else BLUE
        wheel.add_patch(Wedge((0, 0), 0.545, np.degrees(row.start), np.degrees(row.end),
                              width=0.010 + 0.020 * abs(shift) / max_shift,
                              facecolor=color, edgecolor="none", alpha=0.72, zorder=2))
        wheel.add_patch(Circle(xy(0.52, row.angle), 0.009, facecolor=WHITE,
                               edgecolor=INK, lw=0.35, zorder=4))

    edge_scale = selected.pooled_standardized_share.max()
    shift_scale = selected.standardized_share_difference.abs().max()
    for row in selected.itertuples():
        a, b = macro_angles[int(row.source_macro)], macro_angles[int(row.target_macro)]
        curved_edge(wheel, a, b, LIGHT_GRAY,
                    0.15 + 1.2 * row.pooled_standardized_share / edge_scale, 0.38, 1)
        color = CORAL if row.standardized_share_difference >= 0 else BLUE
        curved_edge(wheel, a, b, color,
                    0.18 + 0.9 * abs(row.standardized_share_difference) / shift_scale, 0.60, 2)

    max_volume = np.log1p(wide.total).max()
    for row in wide.itertuples():
        angle = leaf_angles[row.qwen_leaf]
        half = leaf_width * 0.43
        wheel.add_patch(Wedge((0, 0), 0.735 + 0.065 * np.log1p(row.total) / max_volume,
                              np.degrees(angle - half), np.degrees(angle + half),
                              width=0.065 * np.log1p(row.total) / max_volume,
                              facecolor="#6F747B", edgecolor="none", alpha=0.80))
        total = row.total
        radius = 0.815
        if total:
            for group, value in ((0, row.broader), (2, row.middle), (1, row.narrower)):
                width = 0.080 * value / total
                if width:
                    wheel.add_patch(Wedge((0, 0), radius + width,
                                          np.degrees(angle - half), np.degrees(angle + half),
                                          width=width, facecolor=GROUP_COLORS[group], edgecolor="none"))
                radius += width

    for row in areas.itertuples():
        wheel.add_patch(Wedge((0, 0), 0.704, np.degrees(row.start), np.degrees(row.end),
                              width=0.010, facecolor=("#B7BBC2" if row.qwen_macro == 18 else INK),
                              edgecolor="none", alpha=0.82))
        tx, ty = xy(0.945, row.angle)
        degrees = (np.degrees(row.angle) + 360) % 360
        rotation = degrees
        align = "left"
        if 90 < degrees < 270:
            rotation += 180; align = "right"
        wheel.text(tx, ty, row.display_label, rotation=rotation, rotation_mode="anchor",
                   ha=align, va="center", fontsize=5.0,
                   color=MID_GRAY if row.qwen_macro == 18 else INK)
    wheel.add_patch(Circle((0, 0), 0.125, facecolor=WHITE, edgecolor=INK, lw=0.45, zorder=5))
    wheel.text(0, 0.035, "OpenAlex articles", ha="center", fontsize=6.2,
               fontweight="bold", zorder=6)
    wheel.text(0, -0.005, "15.1 million papers", ha="center", fontsize=5.0,
               color=MID_GRAY, zorder=6)
    wheel.text(0, -0.045, "32 areas · 1,000 topics", ha="center", fontsize=5.0,
               color=MID_GRAY, zorder=6)
    wheel.text(0, 1.04, "A semantic hierarchy of scientific work", fontsize=7,
               fontweight="bold", ha="center")
    wheel.set(xlim=(-1.34, 1.24), ylim=(-1.28, 1.20))

    estimates = pd.read_csv(RESULTS / "dirty_estimates.csv")
    routing = estimates.loc[(estimates.analysis.eq("primary")) &
                            (estimates.outcome.eq("far_to_near_routing"))].iloc[0]
    same = estimates.loc[(estimates.analysis.eq("primary")) & estimates.outcome.eq("near")].iloc[0]
    other = estimates.loc[(estimates.analysis.eq("primary")) & estimates.outcome.eq("far")].iloc[0]
    mechanism.text(0.02, 0.96, "Journal scope as an audience filter", fontsize=7,
                   fontweight="bold", va="top")
    mechanism.plot([0.04, 0.12], [0.885, 0.885], color="#6F747B", lw=3)
    mechanism.text(0.14, 0.885, "paper volume", va="center", fontsize=5.0)
    for x0, color in zip((0.47, 0.51, 0.55), (BLUE, "#B7BBC2", CORAL)):
        mechanism.add_patch(Circle((x0, 0.885), 0.010, facecolor=color, edgecolor="none"))
    mechanism.text(0.58, 0.885, "broader · middle · narrower", va="center", fontsize=5.0)
    mechanism.plot([0.04, 0.08], [0.845, 0.845], color=BLUE, lw=0.8)
    mechanism.plot([0.085, 0.125], [0.845, 0.845], color=CORAL, lw=0.8)
    mechanism.text(0.14, 0.845, "relative citation flows", va="center", fontsize=5.0)
    mechanism.text(0.50, 0.805, "Comparable published content", ha="center", fontsize=5.6,
                   color=MID_GRAY)
    mechanism.add_patch(Circle((0.50, 0.755), 0.025, facecolor="#EFEFF1", edgecolor=INK, lw=0.45))
    mechanism.add_patch(FancyArrowPatch((0.48, 0.73), (0.27, 0.65), arrowstyle="-|>",
                                        mutation_scale=5, lw=0.45, color=MID_GRAY))
    mechanism.add_patch(FancyArrowPatch((0.52, 0.73), (0.73, 0.65), arrowstyle="-|>",
                                        mutation_scale=5, lw=0.45, color=MID_GRAY))
    draw_audience(mechanism, 0.25, 0.44, BLUE, "Broader-scope journal",
                  same.mean_broad, other.mean_broad, routing.mean_broad)
    draw_audience(mechanism, 0.75, 0.44, CORAL, "Narrower-scope journal",
                  same.mean_specialized, other.mean_specialized, routing.mean_specialized)
    mechanism.plot([0.25, 0.25, 0.75, 0.75], [0.235, 0.215, 0.215, 0.235],
                   color=INK, lw=0.65)
    mechanism.text(0.50, 0.145, "Adjusted ratio: 9.2% lower", ha="center",
                   fontsize=8.2, fontweight="bold", color=INK)
    ratio_low = 100 * (1 - np.exp(routing.ci_high))
    ratio_high = 100 * (1 - np.exp(routing.ci_low))
    mechanism.text(0.50, 0.105, f"95% CI, {ratio_low:.1f}–{ratio_high:.1f}% lower", ha="center",
                   fontsize=5.2, color=MID_GRAY)
    mechanism.text(0.50, 0.067, "other-area citations relative to same-topic citations",
                   ha="center", fontsize=5.3)
    mechanism.text(0.50, 0.020, "Total-citation difference was imprecise.",
                   ha="center", fontsize=5.2, color=MID_GRAY)
    mechanism.set(xlim=(0, 1), ylim=(0, 1))
    fig.text(0.012, 0.965, "a", fontsize=8, fontweight="bold")
    fig.text(0.695, 0.965, "b", fontsize=8, fontweight="bold")
    fig.savefig(f"{FIGURE}.pdf", dpi=300, facecolor=WHITE)
    fig.savefig(f"{FIGURE}.png", dpi=300, facecolor=WHITE)
    plt.close(fig)

    outputs = [LEAF_SCOPE, AREA_ORDER, FLOW, FIGURE.with_suffix(".pdf"), FIGURE.with_suffix(".png")]
    for path in outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"expected nonempty output at {path}")
    free = shutil.disk_usage(V2_WORK).free
    write_run("radial_mechanism", {"eligible_papers": sum(totals.values()),
              "taxonomy_leaves": 1000, "observed_leaves": counts.qwen_leaf.nunique(),
              "research_areas": len(areas), "citation_backbone_edges": len(selected)},
              {"eligible_papers_by_scope_group": totals, "backbone_flow_coverage": coverage,
               "layout_inputs": "frozen Qwen3 centroids only", "group_free_bytes": free})
    check_budget()
    log(f"radial mechanism complete papers={sum(totals.values()):,} leaves=1000 "
        f"areas={len(areas)} edges={len(selected)} coverage={coverage:.3f} free={free:,}")


if __name__ == "__main__":
    main()
