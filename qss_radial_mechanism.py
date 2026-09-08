#!/usr/bin/env python3
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import Circle, PathPatch, Wedge
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
FIGURE = RESULTS / "figures/figure3_network"
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


def draw_network_metrics(panel, metrics):
    panel.axis("off")
    panel.set(xlim=(0, 1), ylim=(0, 1))
    panel.text(0.02, 0.98, "The same pattern across the network", fontsize=7,
               fontweight="bold", va="top")
    panel.scatter([0.05, 0.26], [0.905, 0.905], s=18, color=[BLUE, CORAL],
                  edgecolor=WHITE, lw=0.35)
    panel.text(0.085, 0.905, "broader-scope", va="center", fontsize=4.9)
    panel.text(0.295, 0.905, "narrower-scope", va="center", fontsize=4.9)
    panel.text(0.82, 0.905, "difference (95% CI)", va="center", ha="center",
               fontsize=4.9, color=MID_GRAY)

    specs = [
        ("directed_modularity", "Within-area concentration",
         "more citations stay in the paper's area"),
        ("audience_participation", "Breadth of citing areas",
         "citations come from fewer research areas"),
        ("semantic_span", "Title-content distance",
         "citing papers are closer in subject"),
    ]
    for y0, (metric, title, interpretation) in zip((0.70, 0.43, 0.16), specs):
        row = metrics.loc[metric]
        panel.text(0.02, y0 + 0.105, title, fontsize=6.0, fontweight="bold", va="bottom")
        panel.text(0.02, y0 + 0.072, interpretation, fontsize=4.8, color=MID_GRAY, va="bottom")

        value_axis = panel.inset_axes([0.03, y0 - 0.015, 0.55, 0.075])
        values = np.array([row.broad, row.specialized], dtype=float)
        padding = max(np.ptp(values) * 0.9, max(abs(values)) * 0.015, 0.002)
        value_axis.plot(values, [0, 0], color="#C8CCD1", lw=1.2, zorder=1)
        value_axis.scatter(values, [0, 0], s=22, color=[BLUE, CORAL], edgecolor=WHITE,
                           lw=0.4, zorder=2)
        value_axis.text(values[0], -0.26, f"{values[0]:.4f}", color=BLUE,
                        ha="center", va="top", fontsize=4.6)
        value_axis.text(values[1], 0.26, f"{values[1]:.4f}", color=CORAL,
                        ha="center", va="bottom", fontsize=4.6)
        value_axis.set(xlim=(values.min() - padding, values.max() + padding), ylim=(-0.5, 0.5),
                       xticks=[], yticks=[])
        for spine in value_axis.spines.values():
            spine.set_visible(False)

        effect_axis = panel.inset_axes([0.67, y0 - 0.015, 0.31, 0.075])
        point = 100 * row.contrast_specialized_minus_broad
        low, high = 100 * row.ci_low, 100 * row.ci_high
        effect_axis.axvline(0, color=INK, lw=0.45)
        effect_axis.errorbar(point, 0, xerr=[[point - low], [high - point]], fmt="o",
                             color=CORAL, ms=3.0, capsize=1.4, lw=0.75)
        span = max(abs(low), abs(high)) * 1.35
        effect_axis.set(xlim=(-span, span), ylim=(-0.5, 0.5), yticks=[])
        effect_axis.tick_params(axis="x", labelsize=4.2, length=2)
        effect_axis.spines[["top", "right", "left"]].set_visible(False)
        panel.text(0.825, y0 - 0.055, f"{point:+.3f} ({low:+.3f}, {high:+.3f}) ×100",
                   ha="center", fontsize=4.5, color=MID_GRAY)
    panel.text(0.02, 0.015,
               "All three contrasts kept the same direction when each research area was omitted in turn.",
               fontsize=4.7, color=MID_GRAY, va="bottom", wrap=True)


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
    wheel.text(0, -0.045, "31 research areas + mixed records", ha="center", fontsize=5.0,
               color=MID_GRAY, zorder=6)
    wheel.text(0, 1.04, "How scientific work is organized and cited", fontsize=7,
               fontweight="bold", ha="center")
    wheel.set(xlim=(-1.34, 1.24), ylim=(-1.28, 1.20))

    metrics = pd.read_csv(RESULTS / "network_metrics.csv").set_index("metric")
    expected_metrics = {"directed_modularity", "audience_participation", "semantic_span"}
    if set(metrics.index) != expected_metrics:
        raise ValueError(f"expected network metrics {expected_metrics}, got {set(metrics.index)}")
    draw_network_metrics(mechanism, metrics)
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
