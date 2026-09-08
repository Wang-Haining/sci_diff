#!/usr/bin/env python3
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from qss_common import SEED
from qss_v3_common import (
    ARTIFACTS,
    RESULTS,
    V2_WORK,
    V3_WORK,
    check_budget,
    connect,
    log,
)

FIGURES = RESULTS / "figures"
SOURCE_DATA = FIGURES / "source_data"
SCOPE = V2_WORK / "journal_year_scope.parquet"
SCORES = V3_WORK / "routing_scores.parquet"
V2_ARTIFACTS = ARTIFACTS.parent / "qss_v2"
CASE_SELECTION = RESULTS / "case_selection.csv"
CORRIDORS = RESULTS / "journal_corridors.csv"

CORAL = "#E64B35"
SKY = "#4DBBD5"
TEAL = "#00A087"
NAVY = "#3C5488"
GOLD = "#F39B7F"
INK = "#252525"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D9D9D9"
PALE_GRAY = "#ECECEC"
WHITE = "#FFFFFF"

MAIN_WIDTH = 183 / 25.4
REPO = Path(__file__).resolve().parent
EXPECTED_SUBGROUPS = {
    "paper_venue_fit": [1, 2, 3, 4],
    "author_audience_breadth": [1, 2, 3, 4],
    "author_publication_experience": [1, 2, 3, 4],
    "semantic_domain": list(range(32)),
    "publication_year": list(range(2015, 2021)),
}
EXPECTED_TESTS = {
    "paper_venue_fit_global",
    "paper_venue_fit_q4_minus_q1",
    "author_audience_breadth_global",
    "author_audience_breadth_q4_minus_q1",
    "author_publication_experience_global",
    "author_publication_experience_q4_minus_q1",
    "semantic_domain_global",
    "publication_year_global",
    "publication_year_linear_trend",
    "paper_venue_fit_continuous",
}
FIGURE_NAMES = [
    "figure1_measurement_design", "figure2_main_results",
    "figure3_network", "figure4_boundaries_modifiers",
    "extended_data_figure1_cohort_coverage", "extended_data_figure2_diagnostics",
    "extended_data_figure3_sensitivities", "extended_data_figure4_heterogeneity",
]
NODE_LABEL_OFFSETS = {
    4: (0, 22), 7: (13, 0), 8: (-8, 10), 9: (-17, -7),
    12: (17, 8), 18: (-28, -8), 30: (28, -12), 31: (-15, 6),
}
SOURCE_FILES = [
    "SourceData_Figure1.csv", "SourceData_Figure2.csv",
    "SourceData_Figure2_Years.csv", "SourceData_Hierarchy_Papers.csv",
    "SourceData_Hierarchy_Areas.csv", "SourceData_Hierarchy_Edges.csv",
    "SourceData_Hierarchy_Journals.csv",
    "SourceData_Figure3_nodes.csv", "SourceData_Figure3_edges.csv",
    "SourceData_Figure3_metrics.csv",
    "SourceData_Radial_AreaOrder.csv", "SourceData_Radial_CitationBackbone.csv",
    "SourceData_Radial_LeafScope.csv",
    "SourceData_Figure4_estimates.csv", "SourceData_Figure4_same_author.csv",
    "SourceData_ED1_cohort_coverage.csv", "SourceData_ED2_balance.csv",
    "SourceData_ED2_propensity_candidates.csv", "SourceData_ED2_propensity_bins.csv",
    "SourceData_ED3_network_edges.csv", "SourceData_ED3_lodo.csv",
    "SourceData_ED3_sensitivities.csv", "SourceData_ED4_corridors.csv",
    "SourceData_ED4_nodes.csv",
]


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.titleweight": "normal",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
    })


def panel_label(axis, label, x=-0.16, y=1.05):
    axis.text(x, y, label, transform=axis.transAxes, fontsize=8,
              fontweight="bold", va="bottom", ha="left", clip_on=False)


def node_labels(axis, nodes, top_n, fontsize=5):
    for row in nodes.nlargest(top_n, "source_share").itertuples():
        offset = NODE_LABEL_OFFSETS.get(int(row.qwen_macro), (0, 0))
        axis.annotate("\n".join(textwrap.wrap(row.display_label, 17)),
                      (row.mds_x, row.mds_y), xytext=offset,
                      textcoords="offset points", ha="center", va="center",
                      fontsize=fontsize, linespacing=0.9, zorder=4,
                      bbox={"boxstyle": "round,pad=0.12", "facecolor": WHITE,
                            "edgecolor": "none", "alpha": 0.82},
                      arrowprops=({"arrowstyle": "-", "color": MID_GRAY, "lw": 0.3}
                                  if offset != (0, 0) else None))


def require_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"expected nonempty input at {path}")


def require_columns(frame, name, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if frame.empty or missing:
        raise ValueError(f"expected nonempty {name} with {columns}, "
                         f"got rows={len(frame)} missing={missing}")


def require_finite(frame, name, columns):
    require_columns(frame, name, columns)
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    bad = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
    if bad:
        raise ValueError(f"expected finite {name} columns {columns}, got bad={bad}")


def load_json(path):
    require_file(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"expected completed manifest at {path}, got {payload.get('status')}")
    return payload


def load_csv(path, columns):
    require_file(path)
    frame = pd.read_csv(path)
    require_columns(frame, path.name, columns)
    return frame


def normalize_level(value):
    number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"expected integer subgroup level, got {value}")
    return int(number)


def validate_subgroups(estimates, tests, labels):
    require_finite(estimates, "subgroup estimates",
                   ["order", "estimate", "ci_low", "ci_high", "n", "n_broad",
                    "n_specialized", "journals", "far_near_broad", "far_near_specialized"])
    if set(estimates.test) != set(EXPECTED_SUBGROUPS):
        raise ValueError(f"unsupported subgroup tests: got {sorted(set(estimates.test))}")
    if set(estimates.status) != {"estimated"}:
        raise ValueError(f"expected all prespecified subgroup rows estimated, got "
                         f"{estimates.status.value_counts().to_dict()}")
    actual = {(row.test, normalize_level(row.level)) for row in estimates.itertuples()}
    expected = {(test, level) for test, levels in EXPECTED_SUBGROUPS.items() for level in levels}
    if actual != expected or len(estimates) != 50:
        raise ValueError(f"subgroup contract failed: expected 50 fixed rows, got {len(estimates)}; "
                         f"missing={sorted(expected - actual)[:5]} extra={sorted(actual - expected)[:5]}")
    if (estimates.ci_low > estimates.estimate).any() or (estimates.ci_high < estimates.estimate).any():
        raise ValueError("subgroup confidence interval does not contain its estimate")

    if set(tests.test) != EXPECTED_TESTS or set(tests.status) != {"estimated"}:
        raise ValueError(f"subgroup-test contract failed: got tests={sorted(set(tests.test))} "
                         f"status={tests.status.value_counts().to_dict()}")
    for row in tests.to_dict("records"):
        columns = ["estimate", "p_value"]
        columns += ["df"] if row["test"].endswith("_global") else ["se", "ci_low", "ci_high"]
        require_finite(pd.DataFrame([row]), row["test"], columns)

    require_finite(labels, "macro labels", ["qwen_macro", "n", "journals"])
    if len(labels) != 32 or labels.qwen_macro.nunique() != 32 \
            or set(labels.qwen_macro.astype(int)) != set(range(32)) \
            or labels[["display_label", "representative_journals"]].isna().any().any() \
            or labels.display_label.duplicated().any():
        raise ValueError(f"expected 32 complete macro labels, got rows={len(labels)}")


def bool_column(values, name):
    if values.dtype == bool:
        return values
    mapped = values.astype(str).str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(f"expected boolean {name}, got {values[mapped.isna()].unique()[:5]}")
    return mapped.astype(bool)


def validate_network(nodes, edges, metrics, lodo):
    require_finite(nodes, "network nodes",
                   ["qwen_macro", "mds_x", "mds_y", "source_share", "focal_broad",
                    "focal_specialized", "self_retention_broad",
                    "self_retention_specialized", "self_retention_difference"])
    require_finite(edges, "network edges",
                   ["source_macro", "target_macro", "raw_edges_broad",
                    "raw_edges_specialized", "row_share_broad", "row_share_specialized",
                    "row_share_difference", "standardized_share_broad",
                    "standardized_share_specialized", "standardized_share_difference",
                    "pooled_standardized_share"])
    require_finite(metrics, "network metrics",
                   ["broad", "specialized", "contrast_specialized_minus_broad",
                    "ci_low", "ci_high", "bootstrap_ci_low", "bootstrap_ci_high",
                    "null_broad", "null_specialized"])
    require_finite(lodo, "network leave-one-domain-out",
                   ["omitted_source_macro", "broad", "specialized",
                    "contrast_specialized_minus_broad"])
    if (len(nodes), len(edges), len(metrics), len(lodo)) != (32, 1024, 3, 96):
        raise ValueError("expected network dimensions 32/1024/3/96, got "
                         f"{len(nodes)}/{len(edges)}/{len(metrics)}/{len(lodo)}")
    macros = set(range(32))
    if set(nodes.qwen_macro.astype(int)) != macros or nodes.qwen_macro.nunique() != 32:
        raise ValueError("expected one node for every frozen macrodomain")
    pairs = edges[["source_macro", "target_macro"]].astype(int)
    if len(pairs.drop_duplicates()) != 1024 \
            or set(pairs.source_macro) != macros or set(pairs.target_macro) != macros:
        raise ValueError("expected exactly one edge cell for every 32x32 domain pair")
    expected_metrics = {"directed_modularity", "audience_participation", "semantic_span"}
    if set(metrics.metric) != expected_metrics or metrics.metric.nunique() != 3:
        raise ValueError(f"expected three fixed network metrics, got {metrics.metric.tolist()}")
    lodo_pairs = {(row.metric, int(row.omitted_source_macro)) for row in lodo.itertuples()}
    if lodo_pairs != {(metric, domain) for metric in expected_metrics for domain in macros}:
        raise ValueError("expected 32 leave-one-source-domain rows for each network metric")
    edges["plot_edge"] = bool_column(edges.plot_edge, "plot_edge")
    if not edges.plot_edge.any() or edges.loc[edges.source_macro.eq(edges.target_macro), "plot_edge"].any():
        raise ValueError("treatment-blind plot_edge must select non-diagonal edges only")
    for arm in ("broad", "specialized"):
        sums = edges.groupby("source_macro")[f"row_share_{arm}"].sum().to_numpy()
        if not np.allclose(sums, 1, atol=1e-8) \
                or not np.isclose(edges[f"standardized_share_{arm}"].sum(), 1, atol=1e-8):
            raise ValueError(f"network flow conservation failed for {arm}")
    if not np.isclose(nodes.source_share.sum(), 1, atol=1e-8):
        raise ValueError(f"expected node source shares to sum to 1, got {nodes.source_share.sum()}")
    expected_difference = (edges.standardized_share_specialized
                           - edges.standardized_share_broad)
    if not np.allclose(edges.standardized_share_difference, expected_difference, atol=1e-12):
        raise ValueError("standardized network differences do not reconcile")


def validate_corridors(selection, corridors):
    required = [
        "case_rank", "qwen_macro", "display_label", "broad_id", "narrow_id",
        "broad_name", "narrow_name", "broad_scope", "narrow_scope", "broad_n",
        "narrow_n", "shared_n", "overlap", "routing_change_percent",
        "routing_ci_low_percent", "routing_ci_high_percent", "subgroup_n", "journals",
    ]
    require_finite(corridors, "journal corridors", [
        "case_rank", "qwen_macro", "broad_scope", "narrow_scope", "broad_n", "narrow_n",
        "shared_n", "overlap", "routing_change_percent", "routing_ci_low_percent",
        "routing_ci_high_percent", "subgroup_n", "journals",
    ])
    require_columns(corridors, "journal corridors", required)
    require_columns(selection, "case selection", required[:13])
    if len(corridors) != 4 or corridors.qwen_macro.nunique() != 4 \
            or set(corridors.case_rank.astype(int)) != {1, 2, 3, 4}:
        raise ValueError(f"expected four ranked journal corridors, got rows={len(corridors)}")
    pre_outcome = required[:13]
    selected = selection[pre_outcome].sort_values("case_rank").reset_index(drop=True)
    joined = corridors[pre_outcome].sort_values("case_rank").reset_index(drop=True)
    text_columns = ["display_label", "broad_id", "narrow_id", "broad_name", "narrow_name"]
    numeric_columns = [column for column in pre_outcome if column not in text_columns]
    if not selected[text_columns].astype(str).equals(joined[text_columns].astype(str)) \
            or not np.allclose(selected[numeric_columns].to_numpy(dtype=float),
                               joined[numeric_columns].to_numpy(dtype=float),
                               rtol=0, atol=1e-12):
        raise ValueError("journal corridors do not reproduce the outcome-blind case selection")
    if corridors[["display_label", "broad_name", "narrow_name"]].isna().any().any() \
            or (corridors[["display_label", "broad_name", "narrow_name"]]
                .astype(str).apply(lambda column: column.str.strip().eq("")).any().any()):
        raise ValueError("journal corridor reader labels must be nonempty")
    if (corridors.routing_ci_low_percent > corridors.routing_change_percent).any() \
            or (corridors.routing_ci_high_percent < corridors.routing_change_percent).any():
        raise ValueError("journal corridor confidence interval does not contain estimate")
    if (corridors.narrow_scope <= corridors.broad_scope).any():
        raise ValueError("expected each narrower-scope journal example to have the higher scope score")


def read_inputs():
    estimates = load_csv(
        RESULTS / "dirty_estimates.csv",
        ["analysis", "outcome", "mean_broad", "mean_specialized", "estimate",
         "ci_low", "ci_high", "bootstrap_ci_low", "bootstrap_ci_high", "n", "journals"],
    )
    require_finite(estimates, "dirty estimates",
                   ["mean_broad", "mean_specialized", "estimate", "ci_low", "ci_high",
                    "bootstrap_ci_low", "bootstrap_ci_high", "n", "journals"])
    subgroups = load_csv(
        RESULTS / "subgroup_estimates.csv",
        ["test", "level", "order", "status", "estimate", "ci_low", "ci_high",
         "far_near_broad", "far_near_specialized", "n", "n_broad", "n_specialized", "journals"],
    )
    tests = load_csv(RESULTS / "subgroup_tests.csv", ["test", "modifier", "status"])
    labels = load_csv(RESULTS / "macro_labels.csv",
                      ["qwen_macro", "display_label", "n", "journals",
                       "representative_journals"])
    nodes = load_csv(RESULTS / "network_nodes.csv",
                     ["qwen_macro", "mds_x", "mds_y", "source_share",
                      "representative_journals"])
    edges = load_csv(RESULTS / "network_edges.csv",
                     ["source_macro", "target_macro", "standardized_share_difference",
                      "plot_edge"])
    metrics = load_csv(RESULTS / "network_metrics.csv",
                       ["metric", "broad", "specialized", "contrast_specialized_minus_broad",
                        "ci_low", "ci_high", "null_broad", "null_specialized"])
    lodo = load_csv(RESULTS / "network_leave_one_domain_out.csv",
                    ["omitted_source_macro", "metric", "contrast_specialized_minus_broad"])
    same_journal = load_csv(
        RESULTS / "same_journal_sensitivity.csv",
        ["estimand", "estimate", "ci_low", "ci_high", "bootstrap_ci_low", "bootstrap_ci_high"],
    )
    dynamics = load_csv(
        RESULTS / "citation_dynamics.csv",
        ["horizon_months", "outcome", "specialized_minus_broad", "ci_low", "ci_high"],
    )
    same_author = load_csv(
        RESULTS / "same_author_sensitivity.csv",
        ["author_role", "strata", "authors", "papers", "theta",
         "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_draws"],
    )
    selection = load_csv(
        CASE_SELECTION,
        ["case_rank", "qwen_macro", "display_label", "broad_id", "narrow_id",
         "broad_name", "narrow_name", "broad_scope", "narrow_scope", "broad_n",
         "narrow_n", "shared_n", "overlap"],
    )
    corridors = load_csv(
        CORRIDORS,
        ["case_rank", "qwen_macro", "display_label", "broad_id", "narrow_id",
         "broad_name", "narrow_name", "broad_scope", "narrow_scope", "broad_n",
         "narrow_n", "shared_n", "overlap", "routing_change_percent",
         "routing_ci_low_percent", "routing_ci_high_percent", "subgroup_n", "journals"],
    )
    require_finite(same_journal, "same-journal sensitivity",
                   ["estimate", "ci_low", "ci_high", "bootstrap_ci_low", "bootstrap_ci_high"])
    require_finite(dynamics, "citation dynamics",
                   ["horizon_months", "specialized_minus_broad", "ci_low", "ci_high"])
    require_finite(same_author, "same-author sensitivity",
                   ["strata", "authors", "papers", "theta",
                    "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_draws"])
    if set(same_journal.estimand) != {"external", "inclusive", "inclusive_minus_external"} \
            or len(dynamics) != 15 or set(same_author.author_role) != {"first", "last"}:
        raise ValueError("unexpected sensitivity-analysis rows")
    validate_subgroups(subgroups, tests, labels)
    nodes = nodes.merge(labels[["qwen_macro", "display_label"]], on="qwen_macro",
                        validate="one_to_one")
    validate_network(nodes, edges, metrics, lodo)
    validate_corridors(selection, corridors)
    return (estimates, subgroups, tests, labels, nodes, edges, metrics, lodo,
            same_journal, dynamics, same_author, selection, corridors)


def estimate_row(estimates, analysis, outcome):
    row = estimates[(estimates.analysis == analysis) & (estimates.outcome == outcome)]
    if len(row) != 1:
        raise ValueError(f"expected one estimate for {analysis}/{outcome}, got {len(row)}")
    return row.iloc[0]


def forest(axis, frame, labels, colors=None, reference=0, transform=None, markers=None):
    values = frame.estimate.to_numpy(dtype=float)
    low = frame.ci_low.to_numpy(dtype=float)
    high = frame.ci_high.to_numpy(dtype=float)
    if transform is not None:
        values, low, high = transform(values), transform(low), transform(high)
    if not np.isfinite(np.r_[values, low, high]).all() \
            or np.any(low > values) or np.any(values > high):
        raise ValueError("invalid forest-plot estimate or confidence interval")
    y = np.arange(len(frame))[::-1]
    axis.axvline(reference, color=INK, lw=0.6, zorder=0)
    colors = colors or [NAVY] * len(frame)
    markers = markers or ["o"] * len(frame)
    for index in range(len(frame)):
        axis.errorbar(values[index], y[index],
                      xerr=[[values[index] - low[index]], [high[index] - values[index]]],
                      fmt=markers[index], ms=3.5, color=colors[index], ecolor=colors[index],
                      capsize=1.5, lw=0.8, markeredgewidth=0.5, zorder=2)
    axis.set_yticks(y, labels)
    return values, low, high


def save(fig, name):
    paths = []
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{name}.{suffix}"
        fig.savefig(path, facecolor=WHITE, dpi=300)
        if path.stat().st_size < 10_000:
            raise ValueError(f"expected {path} >=10,000 bytes, got {path.stat().st_size}")
        paths.append(path)
    plt.close(fig)
    return paths


def source_writer(commit):
    records = []

    def write(filename, frame, figure, panel, origin):
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"expected complete nonempty source data {filename}, "
                             f"rows={len(frame)} missing={int(frame.isna().sum().sum())}")
        path = SOURCE_DATA / filename
        frame.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            "figure/panel": f"{figure}/{panel}",
            "source_file": filename,
            "originating_artifact": origin,
            "rows": len(frame),
            "git_commit": commit,
            "sha256": digest,
        })
        return path

    return write, records


def measurement_data(con, nodes, exposure_run):
    require_file(SCOPE)
    scope = con.execute("""
      SELECT semantic_title_similarity,semantic_title_half0,semantic_title_half1,
             semantic_title_n,semantic_title_half0_n,semantic_title_half1_n
      FROM read_parquet(?) WHERE semantic_title_n>=100
        AND isfinite(semantic_title_similarity)
    """, [str(SCOPE)]).df()
    require_finite(scope, "journal-year scope", ["semantic_title_similarity"])
    split = scope[(scope.semantic_title_half0_n >= 50) & (scope.semantic_title_half1_n >= 50)].copy()
    require_finite(split, "split-half scope", ["semantic_title_half0", "semantic_title_half1"])
    reliability = exposure_run["extra"]["reliability"]
    if reliability.get("split_n") != len(split) or not 0.7 <= reliability.get("spearman_brown", 0) <= 1:
        raise ValueError(f"split-half manifest mismatch: rows={len(split)} reliability={reliability}")

    counts, edges = np.histogram(scope.semantic_title_similarity, bins=40)
    rows = [{"panel": "a", "mark": "timeline", "item": label,
             "x": x, "y": 0.5, "value": x}
            for label, x in (("history starts", -36), ("time zero", 0),
                             ("follow-up ends", 60))]
    rows += [{"panel": "b", "mark": "histogram", "item": f"bin {index}",
              "x": (edges[index] + edges[index + 1]) / 2, "y": int(count),
              "value": edges[index + 1] - edges[index]}
             for index, count in enumerate(counts)]
    x_edges = np.linspace(split.semantic_title_half0.min(), split.semantic_title_half0.max(), 33)
    y_edges = np.linspace(split.semantic_title_half1.min(), split.semantic_title_half1.max(), 33)
    density, _, _ = np.histogram2d(
        split.semantic_title_half0, split.semantic_title_half1, bins=[x_edges, y_edges],
    )
    rows += [{"panel": "b", "mark": "split-half bin", "item": f"cell {i}-{j}",
              "x": (x_edges[i] + x_edges[i + 1]) / 2,
              "y": (y_edges[j] + y_edges[j + 1]) / 2, "value": int(density[i, j])}
             for i, j in np.argwhere(density > 0)]
    rows.append({"panel": "b", "mark": "reliability", "item": "Spearman-Brown",
                 "x": reliability["spearman_brown"], "y": reliability["split_n"],
                 "value": reliability["pearson"]})
    for label, column, extra_filter, sign in (
        ("Title versus title-and-abstract", "semantic_abstract_similarity",
         "AND semantic_abstract_reliable", 1),
        ("Title versus reference-field HHI", "reference_field_hhi", "", 1),
        ("Title versus negative reference-field entropy", "reference_field_entropy", "", -1),
        ("Title versus topic HHI", "topic_hhi", "", 1),
        ("Title versus negative topic entropy", "topic_entropy", "", -1),
    ):
        pair = con.execute(f"""
          SELECT semantic_title_similarity AS title_score, {column} AS comparison
          FROM read_parquet(?)
          WHERE semantic_reliable AND {column} IS NOT NULL {extra_filter}
        """, [str(SCOPE)]).df()
        rho = sign * pair.title_score.rank(method="average").corr(
            pair.comparison.rank(method="average"))
        if len(pair) < 50_000 or not np.isfinite(rho):
            raise ValueError(f"invalid scope validity {label}: n={len(pair)} rho={rho}")
        rows.append({"panel": "b", "mark": "convergent validity", "item": label,
                     "x": float(rho), "y": int(len(pair)), "value": int(sign)})
    rows += [{"panel": "c", "mark": "choice-set schematic", "item": label,
              "x": x, "y": y, "value": arm}
             for label, x, y, arm in (("broad", 0.18, 0.34, 0), ("broad", 0.30, 0.68, 0),
                                      ("middle excluded", 0.49, 0.48, 2),
                                      ("specialized", 0.70, 0.66, 1),
                                      ("specialized", 0.82, 0.32, 1))]
    rows += [{"panel": "d", "mark": "named research area", "item": r.display_label,
              "x": r.mds_x, "y": r.mds_y, "value": r.source_share}
             for r in nodes.itertuples()]
    return scope, split, pd.DataFrame(rows)


def figure1(con, nodes, exposure_run):
    scope, split, source = measurement_data(con, nodes, exposure_run)
    fig, axes = plt.subplots(1, 4, figsize=(MAIN_WIDTH, 2.30),
                             gridspec_kw={"width_ratios": [1.05, 1.25, 1.05, 1.05]},
                             constrained_layout=True)

    axis = axes[0]
    axis.axis("off")
    axis.plot([0.08, 0.92], [0.53, 0.53], color=INK, lw=0.8, transform=axis.transAxes)
    for x, color, label in ((0.13, NAVY, "t−3 to t−1\nmeasure journal focus"),
                            (0.50, CORAL, "t\npublish"),
                            (0.87, TEAL, "t to t+60 mo\ncount citations")):
        axis.scatter(x, 0.53, s=42, color=color, edgecolor=WHITE, lw=0.6,
                     transform=axis.transAxes, zorder=3)
        axis.text(x, 0.40 if x != 0.50 else 0.69, label, ha="center", va="center",
                  transform=axis.transAxes, fontsize=6)
    axis.text(0.50, 0.07, "No focal or future information\nenters journal specialization",
              ha="center", transform=axis.transAxes, color=MID_GRAY, fontsize=6)
    axis.set_title("Study timeline", pad=3)

    axis = axes[1]
    axis.hist(scope.semantic_title_similarity, bins=40, color=NAVY,
              edgecolor=WHITE, linewidth=0.15)
    axis.set(xlabel="Similarity among recent papers", ylabel="Journal-years",
             title="Journal specialization")
    inset = axis.inset_axes([0.49, 0.48, 0.49, 0.48])
    validity = source[source.mark.eq("convergent validity")].set_index("item")
    checks = [
        ("Split half", exposure_run["extra"]["reliability"]["spearman_brown"], NAVY),
        ("Title + abstract", validity.loc["Title versus title-and-abstract", "x"], TEAL),
        ("Reference HHI", validity.loc["Title versus reference-field HHI", "x"], GOLD),
        ("Reference entropy", validity.loc["Title versus negative reference-field entropy", "x"], GOLD),
    ]
    y = np.arange(len(checks))[::-1]
    inset.hlines(y, 0, [item[1] for item in checks], color=LIGHT_GRAY, lw=0.7)
    inset.scatter([item[1] for item in checks], y,
                  color=[item[2] for item in checks], s=12, zorder=2)
    inset.set(xlim=(0, 1), xticks=[0, 0.5, 1], yticks=y,
              yticklabels=[item[0] for item in checks])
    inset.set_facecolor(WHITE)
    inset.tick_params(axis="both", labelsize=5.5, pad=1)
    for label in inset.get_yticklabels():
        label.set_bbox({"facecolor": WHITE, "edgecolor": "none", "pad": 0.2})
    inset.spines[["top", "right"]].set_visible(False)
    inset.set_xlabel("Agreement", fontsize=5.5, labelpad=1)

    axis = axes[2]
    axis.axis("off")
    axis.add_patch(plt.Circle((0.50, 0.50), 0.36, transform=axis.transAxes,
                              color=PALE_GRAY, zorder=0))
    for x, y, color, marker in ((0.18, 0.34, SKY, "o"), (0.30, 0.68, SKY, "o"),
                                (0.49, 0.48, LIGHT_GRAY, "x"),
                                (0.70, 0.66, CORAL, "s"), (0.82, 0.32, CORAL, "s")):
        styling = {"color": MID_GRAY} if marker == "x" else {
            "color": color, "edgecolor": INK,
        }
        axis.scatter(x, y, s=35, marker=marker, linewidth=0.4,
                     transform=axis.transAxes, zorder=2, **styling)
    axis.text(0.12, 0.14, "broader", ha="center", color=SKY,
              transform=axis.transAxes, fontsize=6)
    axis.text(0.88, 0.14, "narrower", ha="center", color=CORAL,
              transform=axis.transAxes, fontsize=6)
    axis.text(0.49, 0.39, "middle 50%\nnot analysed", ha="center", va="top",
              color=MID_GRAY, transform=axis.transAxes, fontsize=5)
    axis.text(0.50, 0.89, "similar titles, same year", ha="center",
              transform=axis.transAxes, fontsize=6)
    axis.set_title("Comparison groups", pad=3)

    axis = axes[3]
    size = 8 + 140 * nodes.source_share.to_numpy() / nodes.source_share.max()
    axis.scatter(nodes.mds_x, nodes.mds_y, s=size, color=WHITE, edgecolor=NAVY, lw=0.55)
    node_labels(axis, nodes, top_n=8, fontsize=5.5)
    axis.set(xticks=[], yticks=[], title="Research areas from titles")
    axis.set_aspect("equal", adjustable="datalim")
    for spine in axis.spines.values():
        spine.set_visible(False)
    for label, axis in zip("abcd", axes):
        panel_label(axis, label, x=-0.28, y=1.07)
    return source, save(fig, "figure1_measurement_design")


def percent_ratio(values):
    return 100 * np.expm1(np.asarray(values, dtype=float))


def figure2(estimates):
    order = ["total_citations", "near", "intermediate", "far"]
    absolute = pd.DataFrame([estimate_row(estimates, "primary", name) for name in order])
    labels = ["All included", "Same topic", "Same area,\nother topic", "Other area"]
    routing = estimate_row(estimates, "primary", "far_to_near_routing")
    any_far = estimate_row(estimates, "primary", "any_far")
    near = estimate_row(estimates, "primary", "near")
    distant = estimate_row(estimates, "primary", "far")
    share_broad = 100 * distant.mean_broad / (near.mean_broad + distant.mean_broad)
    share_specialized = 100 * distant.mean_specialized / (
        near.mean_specialized + distant.mean_specialized
    )
    source = pd.concat([absolute, pd.DataFrame([routing, any_far])], ignore_index=True)

    fig, axes = plt.subplots(1, 4, figsize=(MAIN_WIDTH, 2.45), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.25, 1.0, 1.0]})
    y = np.arange(4)[::-1]
    for index, row in enumerate(absolute.itertuples()):
        axes[0].plot([row.mean_broad, row.mean_specialized], [y[index], y[index]],
                     color=LIGHT_GRAY, lw=1.1, zorder=0)
    axes[0].scatter(absolute.mean_broad, y, color=SKY, s=20, label="Broader scope", zorder=2)
    axes[0].scatter(absolute.mean_specialized, y, color=CORAL, marker="s", s=18,
                    label="Narrower scope", zorder=2)
    axes[0].set_yticks(y, labels)
    axes[0].set(xlabel="Adjusted citations per paper", title="Adjusted counts")
    axes[0].legend(frameon=False, loc="lower right", handletextpad=0.3)

    forest(axes[1], absolute, labels, colors=[MID_GRAY, MID_GRAY, GOLD, CORAL])
    axes[1].set(xlabel="Narrower minus broader", title="Difference")
    effect, low, high = forest(
        axes[2], pd.DataFrame([routing]), ["Other area / same topic"], colors=[CORAL],
        transform=percent_ratio,
    )
    axes[2].set(xlabel="Ratio change (%)", title="Relative citation mix")
    axes[2].text(0.04, 0.08,
                 f"broader {routing.mean_broad:.2f}\nnarrower {routing.mean_specialized:.2f}\n"
                 f"other-area share {share_broad:.1f}% → {share_specialized:.1f}%",
                 transform=axes[2].transAxes, fontsize=6)
    axes[2].annotate(f"{effect[0]:.1f}%\n[{low[0]:.1f}, {high[0]:.1f}]",
                     (effect[0], 0), xytext=(4, 12), textcoords="offset points", fontsize=6)

    forest(axes[3], pd.DataFrame([any_far]), ["Any other-area citation"], colors=[CORAL],
           transform=lambda values: 100 * np.asarray(values, dtype=float))
    axes[3].set(xlabel="Difference (pp)", title="Any other-area\ncitation")
    axes[3].text(0.04, 0.07,
                 f"broader {100 * any_far.mean_broad:.1f}%\n"
                 f"narrower {100 * any_far.mean_specialized:.1f}%\n"
                 f"95% CI {100 * any_far.ci_low:.2f}, {100 * any_far.ci_high:.2f}",
                 transform=axes[3].transAxes, fontsize=6)
    for label, axis in zip("abcd", axes):
        panel_label(axis, label)
    return source, save(fig, "figure2_main_results")


def subgroup_frame(subgroups, test):
    frame = subgroups[subgroups.test.eq(test)].copy().sort_values("order")
    if len(frame) != len(EXPECTED_SUBGROUPS[test]):
        raise ValueError(f"expected {len(EXPECTED_SUBGROUPS[test])} rows for {test}, got {len(frame)}")
    return frame


def normalized_figure3_source(estimates):
    rows = []
    for analysis, outcome, label in (
        ("primary", "far_to_near_routing", "Later citation distribution"),
        ("primary", "reference_routing", "Published-reference distribution"),
        ("reference_adjusted", "far_to_near_routing", "After reference adjustment"),
    ):
        row = estimate_row(estimates, analysis, outcome)
        rows.append({"evidence": "overall", "test": analysis, "level": label,
                     "estimate": row.estimate, "ci_low": row.ci_low, "ci_high": row.ci_high,
                     "mean_broad": row.mean_broad, "mean_specialized": row.mean_specialized,
                     "n": int(row.n), "journals": int(row.journals)})
    return pd.DataFrame(rows)


def figure3(estimates, same_author):
    primary = estimate_row(estimates, "primary", "far_to_near_routing")
    reference = estimate_row(estimates, "primary", "reference_routing")
    adjusted = estimate_row(estimates, "reference_adjusted", "far_to_near_routing")
    source = normalized_figure3_source(estimates)

    fig, axes = plt.subplots(1, 2, figsize=(MAIN_WIDTH, 2.65), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.22, 1.0]})
    forest(axes[0], pd.DataFrame([primary, reference, adjusted]),
           ["Later citations", "Final references", "After reference adjustment"],
           colors=[CORAL, TEAL, NAVY],
           transform=percent_ratio)
    axes[0].set(xlabel="Other-area / same-topic ratio change (%)",
                title="References and adjusted comparison")

    author = same_author.rename(columns={"theta": "estimate",
                                         "bootstrap_ci_low": "ci_low",
                                         "bootstrap_ci_high": "ci_high"})
    author = author.set_index("author_role").loc[["first", "last"]].reset_index()
    forest(axes[1], author, ["Same first author", "Same last author"],
           colors=[TEAL, NAVY], markers=["o", "s"], transform=percent_ratio)
    axes[1].set(xlabel="Other-area / same-topic ratio change (%)",
                title="Within-author comparisons")
    for label, axis in zip("ab", axes):
        panel_label(axis, label, x=-0.20, y=1.07)
    return source, save(fig, "figure4_boundaries_modifiers")


def curved_edge(axis, start, end, color, width, alpha, dashed=False, arrow=False):
    patch = FancyArrowPatch(
        start, end, connectionstyle="arc3,rad=0.11",
        arrowstyle="-|>" if arrow else "-", mutation_scale=6,
        linewidth=width, color=color, alpha=alpha,
        linestyle="--" if dashed else "-", shrinkA=4, shrinkB=4,
        capstyle="round", joinstyle="round", zorder=1 if not arrow else 2,
    )
    axis.add_patch(patch)


def metric_axis(axis, row, title):
    point = 100 * float(row.contrast_specialized_minus_broad)
    low, high = 100 * float(row.bootstrap_ci_low), 100 * float(row.bootstrap_ci_high)
    null = 100 * float(row.null_specialized - row.null_broad)
    axis.axvline(0, color=INK, lw=0.5)
    axis.errorbar(point, 0, xerr=[[point - low], [high - point]], fmt="o",
                  ms=3, color=CORAL, capsize=1.5, lw=0.8)
    axis.scatter(null, -0.18, marker="x", s=14, color=MID_GRAY, linewidth=0.7)
    axis.set_yticks([])
    axis.set_title(title, fontsize=6, pad=2)


def lodo_axis(axis, values, full, title):
    x = 100 * values.contrast_specialized_minus_broad.to_numpy(dtype=float)
    axis.axvline(0, color=INK, lw=0.5)
    axis.scatter(x, np.linspace(-0.16, 0.16, len(x)), s=8, facecolor=WHITE,
                 edgecolor=NAVY, linewidth=0.45, alpha=0.9)
    axis.axvline(100 * full, color=CORAL, lw=1.0)
    axis.set_yticks([])
    axis.set_title(title, fontsize=6, pad=2)


def draw_network(axis, nodes, edges):
    coordinates = nodes.set_index("qwen_macro")[["mds_x", "mds_y"]]
    selected = edges[edges.plot_edge].copy()
    pooled_max = selected.pooled_standardized_share.max()
    shift_max = selected.standardized_share_difference.abs().max()
    if pooled_max <= 0 or shift_max <= 0:
        raise ValueError(f"expected positive plotted edge scales, got {pooled_max}, {shift_max}")
    for row in selected.itertuples():
        start = tuple(coordinates.loc[int(row.source_macro)])
        end = tuple(coordinates.loc[int(row.target_macro)])
        curved_edge(axis, start, end, LIGHT_GRAY,
                    0.35 + 1.25 * row.pooled_standardized_share / pooled_max, 0.65)
    for row in selected.itertuples():
        start = tuple(coordinates.loc[int(row.source_macro)])
        end = tuple(coordinates.loc[int(row.target_macro)])
        increased = row.standardized_share_difference >= 0
        curved_edge(axis, start, end, CORAL if increased else SKY,
                    0.25 + 1.35 * abs(row.standardized_share_difference) / shift_max,
                    0.78, dashed=not increased, arrow=True)
    size = 13 + 215 * nodes.source_share.to_numpy() / nodes.source_share.max()
    axis.scatter(nodes.mds_x, nodes.mds_y, s=size, color=WHITE,
                 edgecolor=INK, linewidth=0.55, zorder=3)
    node_labels(axis, nodes, top_n=8, fontsize=5.5)
    axis.set(xticks=[], yticks=[])
    axis.set_title("How citations moved between research areas", pad=10)
    axis.set_aspect("equal", adjustable="datalim")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.plot([], [], color=CORAL, lw=1, label="higher for narrower-scope papers")
    axis.plot([], [], color=SKY, lw=1, ls="--", label="lower for narrower-scope papers")
    axis.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.24),
                ncol=1, fontsize=5.5, handlelength=1.5)


def figure3_network(nodes, edges, metrics):
    fig = plt.figure(figsize=(MAIN_WIDTH, 3.75))
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.42, 1],
        left=0.055, right=0.97, bottom=0.24, top=0.84, wspace=0.34,
    )
    network_axis = fig.add_subplot(grid[0, 0])
    metric_grid = grid[0, 1].subgridspec(3, 1, hspace=1.05)
    metric_axes = [fig.add_subplot(metric_grid[index, 0]) for index in range(3)]

    draw_network(network_axis, nodes, edges)
    panel_label(network_axis, "a", x=-0.07)

    ordered_metrics = ["directed_modularity", "audience_participation", "semantic_span"]
    short = ["Within-area\nretention", "Diversity of\nciting areas",
             "Mean semantic\ndistance"]
    indexed = metrics.set_index("metric")
    for axis, metric, title in zip(metric_axes, ordered_metrics, short):
        metric_axis(axis, indexed.loc[metric], title)
    fig.text(0.585, 0.93, "b", fontsize=8, fontweight="bold")
    fig.text(0.78, 0.93, "Three views of citation concentration",
             ha="center", fontsize=7)
    metric_axes[-1].set_xlabel("Narrower − broader (×100)", fontsize=5.5)
    return save(fig, "figure3_network")


def ed1_data(v2_dirty, v3_prepare, v3_analyze, network_run):
    counts = v3_prepare["counts"]
    extra = v3_analyze["extra"]
    network = network_run["extra"]["exclusions"]
    rows = [
        {"panel": "a", "arm": "all", "measure": "Eligible focal papers",
         "value": counts["candidate_focal"], "unit": "papers"},
        {"panel": "a", "arm": "all", "measure": "Subgroup and network sample",
         "value": network_run["counts"]["support"], "unit": "papers"},
    ]
    for arm, code in (("Broader scope", "0"), ("Narrower scope", "1")):
        rows += [
            {"panel": "b", "arm": arm, "measure": "Focal paper in distribution",
             "value": 1 - v2_dirty["extra"]["focal_ood_rates"][code],
             "unit": "proportion"},
            {"panel": "b", "arm": arm, "measure": "Citing-paper classification",
             "value": extra["citing_coverage"][code], "unit": "proportion"},
            {"panel": "b", "arm": arm, "measure": "Reference classification",
             "value": extra["reference_coverage"][code], "unit": "proportion"},
        ]
    focal_date = {int(row[0]): float(row[3]) for row in v3_prepare["extra"]["focal_date_qc"]}
    citing_date = {int(row[0]): float(row[2]) for row in v3_prepare["extra"]["citing_january_1_by_arm"]}
    for arm, code in (("Broader scope", 0), ("Narrower scope", 1)):
        rows += [
            {"panel": "c", "arm": arm, "measure": "Focal paper dated 1 January",
             "value": focal_date[code], "unit": "proportion"},
            {"panel": "c", "arm": arm, "measure": "Citing paper dated 1 January",
             "value": citing_date[code], "unit": "proportion"},
        ]
    labels = {
        "excluded_external": "Same journal/author",
        "external_unclassified": "Area unavailable",
        "eligible": "Included",
    }
    for measure in ("excluded_external", "external_unclassified", "eligible"):
        rows.append({"panel": "d", "arm": "all", "measure": labels[measure],
                     "value": network[measure], "unit": "citation edges"})
    frame = pd.DataFrame(rows)
    if int(frame.query("panel == 'd'").value.sum()) != int(network["support_edges"]):
        raise ValueError("network exclusion source data does not sum to support edges")
    return frame


def extended_data1(frame):
    fig, axes = plt.subplots(2, 2, figsize=(MAIN_WIDTH, 4.55), constrained_layout=True)
    axis = axes[0, 0]
    flow = frame[frame.panel.eq("a")]
    bars = axis.barh([1, 0], flow.value / 1e6, color=[MID_GRAY, NAVY])
    axis.set_yticks([1, 0], flow.measure)
    axis.set(xlabel="Papers (millions)",
             title="Half of eligible papers entered the analysis sample")
    for bar, value in zip(bars, flow.value):
        axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                  f"  {value / 1e6:.2f}m", va="center", fontsize=6)

    axis = axes[0, 1]
    coverage = frame[frame.panel.eq("b")]
    measures = ["Focal paper in distribution", "Citing-paper classification",
                "Reference classification"]
    x = np.arange(len(measures))
    for offset, (arm, color) in zip((-0.17, 0.17),
                                    (("Broader scope", SKY), ("Narrower scope", CORAL))):
        values = coverage[coverage.arm.eq(arm)].set_index("measure").loc[measures, "value"]
        axis.bar(x + offset, 100 * values, 0.32, color=color, label=arm)
    axis.axhline(80, color=INK, lw=0.6, ls="--")
    axis.set_xticks(x, ["Focal", "Citing", "References"])
    axis.set(ylabel="Classified or in distribution (%)", title="Text coverage was high in both groups")
    axis.legend(frameon=False, loc="lower left")

    axis = axes[1, 0]
    dates = frame[frame.panel.eq("c")]
    x = np.arange(2)
    for offset, (arm, color) in zip((-0.17, 0.17),
                                    (("Broader scope", SKY), ("Narrower scope", CORAL))):
        values = dates[dates.arm.eq(arm)].set_index("measure").loc[
            ["Focal paper dated 1 January", "Citing paper dated 1 January"], "value"]
        axis.bar(x + offset, 100 * values, 0.32, color=color, label=arm)
    axis.set_xticks(x, ["Focal", "Citing"])
    axis.set(ylabel="Dated 1 January (%)", title="January 1 date shares by group")

    axis = axes[1, 1]
    decomposition = frame[frame.panel.eq("d")]
    left = 0
    for row, color in zip(decomposition.itertuples(), [LIGHT_GRAY, GOLD, TEAL]):
        axis.barh([0], [row.value / 1e6], left=left, color=color, label=row.measure)
        left += row.value / 1e6
    axis.set_yticks([])
    axis.set(xlabel="Citation links (millions)", title="Every citation link was accounted for")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    for label, axis in zip("abcd", axes.ravel()):
        panel_label(axis, label, x=-0.12)
    return save(fig, "extended_data_figure1_cohort_coverage")


def propensity_bins(con):
    require_file(SCORES)
    bins = con.execute("""
      SELECT treatment,
             greatest(0,least(39,floor(propensity*40)::INTEGER)) AS bin,
             count(*) AS n
      FROM read_parquet(?) GROUP BY ALL ORDER BY treatment,bin
    """, [str(SCORES)]).df()
    require_finite(bins, "propensity bins", ["treatment", "bin", "n"])
    grid = pd.MultiIndex.from_product([[0, 1], range(40)], names=["treatment", "bin"]).to_frame(index=False)
    bins = grid.merge(bins, on=["treatment", "bin"], how="left", validate="one_to_one")
    bins["n"] = bins.n.fillna(0).astype(int)
    bins["arm"] = bins.treatment.map({0: "Broader scope", 1: "Narrower scope"})
    bins["bin_left"] = bins.bin / 40
    bins["bin_right"] = (bins.bin + 1) / 40
    totals = bins.groupby("treatment").n.transform("sum")
    bins["density"] = bins.n / totals / 0.025
    if (totals <= 0).any() or bins.isna().any().any():
        raise ValueError("propensity-bin construction failed")
    return bins[["treatment", "arm", "bin", "bin_left", "bin_right", "n", "density"]]


def diagnostics_data():
    balance_columns = ["candidate", "stage", "covariate", "mean_broad", "mean_specialized", "smd"]
    primary_balance = load_csv(RESULTS / "balance.csv", balance_columns)
    downstream_balance = load_csv(RESULTS / "downstream_balance.csv", balance_columns)
    require_finite(primary_balance, "primary balance", ["mean_broad", "mean_specialized", "smd"])
    require_finite(downstream_balance, "downstream balance", ["mean_broad", "mean_specialized", "smd"])
    primary_balance["source_artifact"] = "results/qss_v3/balance.csv"
    downstream_balance["source_artifact"] = "results/qss_v3/downstream_balance.csv"
    balance = pd.concat([primary_balance, downstream_balance], ignore_index=True)

    candidate_columns = ["candidate", "num_leaves", "support", "support_n",
                         "max_weighted_abs_smd", "ess_broad", "ess_specialized",
                         "weight_p50", "weight_p95", "weight_p99", "weight_max",
                         "best_iterations"]
    primary_candidates = load_csv(RESULTS / "propensity_candidates.csv", candidate_columns)
    downstream_candidate = load_csv(RESULTS / "downstream_propensity.csv", candidate_columns)
    finite = candidate_columns[1:-1]
    require_finite(primary_candidates, "primary propensity diagnostics", finite)
    require_finite(downstream_candidate, "downstream propensity diagnostics", finite)
    primary_candidates["source_artifact"] = "results/qss_v3/propensity_candidates.csv"
    downstream_candidate["source_artifact"] = "results/qss_v3/downstream_propensity.csv"
    candidates = pd.concat([primary_candidates, downstream_candidate], ignore_index=True)
    if candidates.candidate.duplicated().any() or balance.isna().any().any() \
            or candidates.isna().any().any():
        raise ValueError("expected complete unique balance and propensity diagnostics")
    return balance, candidates


def extended_data2(bins, balance, candidates):
    fig, axes = plt.subplots(2, 2, figsize=(MAIN_WIDTH, 4.70), constrained_layout=True)
    axis = axes[0, 0]
    for arm, color in (("Broader scope", SKY), ("Narrower scope", CORAL)):
        frame = bins[bins.arm.eq(arm)]
        axis.step((frame.bin_left + frame.bin_right) / 2, frame.density,
                  where="mid", color=color, label=arm)
    axis.set(xlabel="Propensity within support", ylabel="Density",
             title="Propensity overlap")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    y = np.arange(len(candidates))[::-1]
    candidate_labels = {
        "primary_leaves_63": "63-leaf propensity model (selected)",
        "primary_leaves_255": "255-leaf propensity model",
        "reference_adjusted": "Reference-adjusted model",
        "downstream_deterministic": "Rerun for subgroup and network analyses",
    }
    names = candidates.candidate.map(candidate_labels).tolist()
    axis.scatter(100 * candidates.support, y, color=NAVY, s=20)
    for yi, row in zip(y, candidates.itertuples()):
        axis.text(100 * row.support + 0.8, yi,
                  f"ESS {min(row.ess_broad, row.ess_specialized) / 1e3:.0f}k",
                  va="center", fontsize=5.5)
    axis.set_yticks(y, names)
    axis.set(xlabel="Retained (%)", title="Support and effective size")

    axis = axes[1, 0]
    selected = balance[(balance.candidate == "primary_leaves_63") &
                       balance.stage.isin(["raw", "weighted"])]
    weighted = selected[selected.stage.eq("weighted")].set_index("covariate")
    top = weighted.smd.abs().nlargest(12).index
    raw = selected[selected.stage.eq("raw")].set_index("covariate").reindex(top)
    weighted = weighted.reindex(top)
    if len(top) != 12 or raw.smd.isna().any() or weighted.smd.isna().any():
        raise ValueError("raw/weighted balance rows do not align")
    y = np.arange(len(top))[::-1]
    axis.scatter(raw.smd.abs(), y, color=MID_GRAY, s=14, label="Raw")
    axis.scatter(weighted.smd.abs(), y, color=TEAL, marker="s", s=14, label="Weighted")
    for index in range(len(top)):
        axis.plot([abs(raw.smd.iloc[index]), abs(weighted.smd.iloc[index])], [y[index], y[index]],
                  color=LIGHT_GRAY, lw=0.7, zorder=0)
    axis.axvline(0.10, color=INK, lw=0.6, ls="--")
    balance_labels = {
        "lead_prior_venue_specialization": "Lead prior venue scope",
        "log1p_prior_prestige": "Prior journal prestige",
        "log1p_institution_mean_prior_citations": "Institution mean citations",
        "log1p_institution_mean_prior_works": "Institution mean papers",
        "log1p_institution_max_prior_citations": "Institution max citations",
        "log1p_institution_max_prior_works": "Institution max papers",
        "lead_country=__MISSING__": "Lead-author country missing",
        "choice_prevalence": "Choice-set treated share",
        "lead_prior_venue_specialization__missing": "Lead prior scope missing",
        "lead_prior_venue_missing": "Lead prior venue missing",
        "qpc07": "Paper-content component 7",
        "qpc14": "Paper-content component 14",
        "semantic_cluster=479": "Text-defined content-group indicator",
    }
    axis.set_yticks(y, [balance_labels.get(str(name), str(name)) for name in top])
    axis.set(xlabel="Absolute standardized mean difference",
             title="Largest residual imbalances")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    y = np.arange(len(candidates))[::-1]
    for yi, row in zip(y, candidates.itertuples()):
        axis.plot([row.weight_p50, row.weight_p99], [yi, yi], color=LIGHT_GRAY, lw=1.0)
        axis.scatter(row.weight_p50, yi, color=SKY, s=13)
        axis.scatter(row.weight_p95, yi, color=TEAL, marker="s", s=13)
        axis.scatter(row.weight_p99, yi, color=CORAL, marker="^", s=15)
    axis.set_xscale("log")
    axis.set_xlim(0.9, 22)
    axis.set_xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.set_yticks(y, names)
    axis.set(xlabel="Inverse-probability weight", title="Weight distribution")
    axis.legend(handles=[
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=SKY,
                   markeredgecolor=SKY, label="50th"),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor=TEAL,
                   markeredgecolor=TEAL, label="95th"),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor=CORAL,
                   markeredgecolor=CORAL, label="99th"),
    ], frameon=False, ncol=3, loc="upper right")
    for label, axis in zip("abcd", axes.ravel()):
        panel_label(axis, label, x=-0.12)
    return save(fig, "extended_data_figure2_diagnostics")


def sensitivity_data(estimates, dynamics):
    primary = estimate_row(estimates, "primary", "far_to_near_routing")
    winsor = estimate_row(estimates, "primary", "far_to_near_routing_winsorized")
    rows = []

    def add(panel, item, statistic, value, origin):
        rows.append({"panel": panel, "item": item, "statistic": statistic,
                     "value": float(value), "origin": origin})

    for item, row in (("Primary", primary), ("99.9% winsorized", winsor)):
        for statistic in ("estimate", "ci_low", "ci_high", "bootstrap_ci_low", "bootstrap_ci_high"):
            add("c", item, statistic, row[statistic],
                "results/qss_v3/dirty_estimates.csv")
    any_far = estimate_row(estimates, "primary", "any_far")
    for statistic in ("estimate", "ci_low", "ci_high"):
        add("d", "Cross-fitted AIPW", statistic, any_far[statistic],
            "results/qss_v3/dirty_estimates.csv")
    ipw_any = dynamics[(dynamics.horizon_months == 60) & dynamics.outcome.eq("any_distant")]
    if len(ipw_any) != 1:
        raise ValueError(f"expected one 60-month IPW any-distant row, got {len(ipw_any)}")
    ipw_any = ipw_any.iloc[0]
    for source, statistic in (("specialized_minus_broad", "estimate"),
                              ("ci_low", "ci_low"), ("ci_high", "ci_high")):
        add("d", "Fixed-support IPW", statistic, ipw_any[source],
            "results/qss_v3/citation_dynamics.csv")
    return pd.DataFrame(rows)


def extended_data3(nodes, edges, metrics, lodo, estimates, dynamics):
    primary = estimate_row(estimates, "primary", "far_to_near_routing")
    winsor = estimate_row(estimates, "primary", "far_to_near_routing_winsorized")
    fig = plt.figure(figsize=(MAIN_WIDTH, 5.45))
    grid = fig.add_gridspec(2, 2, left=0.105, right=0.96, bottom=0.09, top=0.94,
                           wspace=0.42, hspace=0.63, height_ratios=[1.25, 0.75])
    heat_axis = fig.add_subplot(grid[0, 0])
    lodo_grid = grid[0, 1].subgridspec(1, 3, wspace=0.42)
    lodo_axes = [fig.add_subplot(lodo_grid[0, index]) for index in range(3)]
    winsor_axis = fig.add_subplot(grid[1, 0])
    model_axis = fig.add_subplot(grid[1, 1])

    matrix = edges.pivot(index="source_macro", columns="target_macro",
                         values="standardized_share_difference").sort_index().sort_index(axis=1)
    limit = float(np.abs(matrix.to_numpy()).max())
    if limit <= 0:
        raise ValueError(f"expected nonzero network difference matrix, got limit={limit}")
    flow_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "flow_difference", [SKY, WHITE, CORAL],
    )
    image = heat_axis.imshow(matrix, cmap=flow_cmap, vmin=-limit, vmax=limit,
                             interpolation="nearest", aspect="equal")
    heat_ticks = np.arange(0, 32, 4)
    names = nodes.set_index("qwen_macro").display_label
    heat_labels = [textwrap.fill(names.loc[int(tick)], 18) for tick in heat_ticks]
    heat_axis.set(xticks=heat_ticks, yticks=heat_ticks,
                  xlabel="Citing paper area", ylabel="Focal paper area",
                  title="All 1,024 links between research areas")
    heat_axis.set_xticklabels(heat_labels, rotation=50, ha="right", fontsize=5)
    heat_axis.set_yticklabels(heat_labels, fontsize=5)
    colorbar = fig.colorbar(image, ax=heat_axis, fraction=0.045, pad=0.03)
    colorbar.ax.set_title("Δ share", fontsize=6, pad=3)
    colorbar.ax.tick_params(labelsize=5.5)

    ordered_metrics = ["directed_modularity", "audience_participation", "semantic_span"]
    short = ["Within-area\nretention", "Diversity of\nciting areas", "Mean semantic\ndistance"]
    indexed = metrics.set_index("metric")
    for axis, metric, title in zip(lodo_axes, ordered_metrics, short):
        values = lodo[lodo.metric.eq(metric)].sort_values("omitted_source_macro")
        lodo_axis(axis, values, indexed.loc[metric, "contrast_specialized_minus_broad"], title)
        axis.set_xlabel("Narrower − broader\n(×100)", fontsize=5.5)
    lodo_axes[0].text(-0.32, 1.18, "b", transform=lodo_axes[0].transAxes,
                      fontsize=8, fontweight="bold")
    lodo_axes[1].text(0.5, 1.18, "Direction after omitting one research area",
                      transform=lodo_axes[1].transAxes, ha="center", fontsize=7)

    forest(winsor_axis, pd.DataFrame([primary, winsor]), ["Raw counts", "99.9% winsorized"],
           colors=[CORAL, NAVY], transform=percent_ratio)
    winsor_axis.set(xlabel="Other-area / same-topic ratio change (%)",
                    title="High-citation papers do not explain the result")

    any_far = estimate_row(estimates, "primary", "any_far")
    ipw_any = dynamics[(dynamics.horizon_months == 60) & dynamics.outcome.eq("any_distant")]
    if len(ipw_any) != 1:
        raise ValueError(f"expected one 60-month IPW any-distant row, got {len(ipw_any)}")
    ipw_any = ipw_any.rename(columns={"specialized_minus_broad": "estimate"})
    any_models = pd.concat([pd.DataFrame([any_far]), ipw_any], ignore_index=True)
    forest(model_axis, any_models, ["Cross-fitted AIPW", "Fixed-support IPW"],
           colors=[CORAL, NAVY], transform=lambda values: 100 * np.asarray(values, dtype=float))
    model_axis.set(xlabel="Narrower minus broader (percentage points)",
                   title="Any citation from another area at 60 months")
    panel_label(heat_axis, "a", x=-0.16)
    panel_label(winsor_axis, "c", x=-0.14)
    panel_label(model_axis, "d", x=-0.14)
    return save(fig, "extended_data_figure3_sensitivities")


def tidy_tests(tests):
    identifier = {"test", "modifier", "status"}
    rows = []
    for row in tests.to_dict("records"):
        for statistic, value in row.items():
            if statistic in identifier or pd.isna(value):
                continue
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"nonfinite subgroup-test statistic {row['test']}/{statistic}")
            rows.append({"test": row["test"], "modifier": row["modifier"],
                         "status": row["status"], "statistic": statistic, "value": number})
    frame = pd.DataFrame(rows)
    if set(frame.test) != EXPECTED_TESTS:
        raise ValueError("tidy subgroup tests lost one or more prespecified tests")
    return frame


def extended_data4(corridors, nodes):
    corridors = corridors.sort_values("case_rank")
    scope_low = float(corridors[["broad_scope", "narrow_scope"]].min().min())
    scope_high = float(corridors[["broad_scope", "narrow_scope"]].max().max())
    scope_pad = max(0.002, 0.08 * (scope_high - scope_low))
    effect_low = min(-1.0, float(corridors.routing_ci_low_percent.min()))
    effect_high = max(1.0, float(corridors.routing_ci_high_percent.max()))
    effect_pad = 0.08 * (effect_high - effect_low)
    fig, axes = plt.subplots(2, 2, figsize=(MAIN_WIDTH, 5.65))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.97,
                        wspace=0.12, hspace=0.20)
    node_size = 5 + 55 * nodes.source_share.to_numpy() / nodes.source_share.max()

    for label, axis, row in zip("abcd", axes.ravel(), corridors.itertuples()):
        axis.set_axis_off()
        axis.add_patch(FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98, transform=axis.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.02", facecolor="#FAFAFA",
            edgecolor=LIGHT_GRAY, linewidth=0.7, clip_on=False,
        ))
        axis.text(0.05, 0.92, textwrap.fill(str(row.display_label), 42), transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="top")
        axis.text(0.05, 0.81, "Journals selected without citation outcomes",
                  transform=axis.transAxes, fontsize=5.5, color=MID_GRAY, va="top")
        panel_label(axis, label, x=0.00, y=0.99)

        axis.text(0.08, 0.73, "Broader: " + textwrap.shorten(str(row.broad_name), 45),
                  transform=axis.transAxes, fontsize=5.5, color=SKY, va="top")
        axis.text(0.08, 0.66, "Narrower: " + textwrap.shorten(str(row.narrow_name), 43),
                  transform=axis.transAxes, fontsize=5.5, color=CORAL, va="top")
        scope_axis = axis.inset_axes([0.08, 0.39, 0.55, 0.17])
        scope_axis.plot([row.broad_scope, row.narrow_scope], [0, 0], color=LIGHT_GRAY, lw=1.3)
        scope_axis.scatter(row.broad_scope, 0, color=SKY, s=22, zorder=2)
        scope_axis.scatter(row.narrow_scope, 0, color=CORAL, marker="s", s=21, zorder=2)
        scope_axis.set_yticks([])
        scope_axis.set_xlim(scope_low - scope_pad, scope_high + scope_pad)
        scope_axis.set_ylim(-0.4, 0.4)
        scope_axis.set_xlabel("Venue-free journal scope score", fontsize=5.5)
        scope_axis.tick_params(axis="both", labelsize=5)

        map_axis = axis.inset_axes([0.70, 0.48, 0.23, 0.27])
        map_axis.scatter(nodes.mds_x, nodes.mds_y, s=node_size, color=PALE_GRAY,
                         edgecolor=LIGHT_GRAY, linewidth=0.25)
        selected = nodes[nodes.qwen_macro.eq(int(row.qwen_macro))]
        if len(selected) != 1:
            raise ValueError(f"expected one map node for corridor {row.qwen_macro}, got {len(selected)}")
        map_axis.scatter(selected.mds_x, selected.mds_y, s=42, color=CORAL,
                         edgecolor=INK, linewidth=0.45, zorder=3)
        map_axis.set(xticks=[], yticks=[], title="Location in frozen text map")
        map_axis.set_aspect("equal", adjustable="datalim")
        for spine in map_axis.spines.values():
            spine.set_visible(False)

        effect_axis = axis.inset_axes([0.69, 0.14, 0.25, 0.20])
        effect_axis.axvline(0, color=INK, lw=0.5)
        effect_axis.errorbar(
            row.routing_change_percent, 0,
            xerr=[[row.routing_change_percent - row.routing_ci_low_percent],
                  [row.routing_ci_high_percent - row.routing_change_percent]],
            fmt="o", color=CORAL, ms=3.5, capsize=1.5, lw=0.8,
        )
        effect_axis.set_xlim(effect_low - effect_pad, effect_high + effect_pad)
        effect_axis.set_yticks([])
        effect_axis.set_title("Area-wide estimate\n(all journals)", fontsize=5.7, pad=2)
        effect_axis.set_xlabel("Ratio change (%)", fontsize=5.5)
        effect_axis.tick_params(axis="x", labelsize=5)

        axis.text(0.08, 0.14,
                  f"Shared content-cell support: {int(row.shared_n):,}\n"
                  f"Overlap score: {row.overlap:.2f}",
                  transform=axis.transAxes, fontsize=5.5, va="bottom", linespacing=1.35)
        axis.text(0.08, 0.05, "The journals illustrate scope; the estimate uses all journals in this area.",
                  transform=axis.transAxes, fontsize=5.3, color=MID_GRAY, va="bottom")
    return save(fig, "extended_data_figure4_heterogeneity")


def main():
    check_budget()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    protected_redesigns = {"figure2_main_results", "figure3_network"}
    for name in [name for name in FIGURE_NAMES if name not in protected_redesigns] + [
            "figure3_modifiers", "figure4_domains_time",
            "figure3_boundaries_modifiers", "figure4_network"]:
        for suffix in ("pdf", "png"):
            path = FIGURES / f"{name}.{suffix}"
            if path.exists():
                path.unlink()
    stale_sources = [
        "SourceData_Figure3_estimates.csv", "SourceData_Figure3_tests.csv",
        "SourceData_Figure4_nodes.csv", "SourceData_Figure4_edges.csv",
        "SourceData_Figure4_metrics.csv", "SourceData_Figure4_lodo.csv",
        "SourceData_Figure4_tests.csv", "SourceData_Figure3_lodo.csv",
        "SourceData_ED4_subgroups.csv", "SourceData_ED4_tests.csv",
        "SourceData_ED4_domain_labels.csv", "SourceData_Figure3_web.csv",
    ]
    for name in SOURCE_FILES + stale_sources + ["source_data_manifest.csv"]:
        path = SOURCE_DATA / name
        if path.exists():
            path.unlink()
    style()

    (estimates, subgroups, tests, labels, nodes, edges, metrics, lodo,
     same_journal, dynamics, same_author, selection, corridors) = read_inputs()
    manifests = {
        "v2_exposure": load_json(V2_ARTIFACTS / "run_exposure.json"),
        "v2_dirty": load_json(V2_ARTIFACTS / "run_dirty_analyze.json"),
        "v3_prepare": load_json(ARTIFACTS / "run_prepare.json"),
        "v3_analyze": load_json(ARTIFACTS / "run_analyze.json"),
        "downstream": load_json(ARTIFACTS / "run_downstream.json"),
        "network": load_json(ARTIFACTS / "run_network.json"),
        "cases": load_json(ARTIFACTS / "run_cases.json"),
    }
    actual_selection_hash = hashlib.sha256(CASE_SELECTION.read_bytes()).hexdigest()
    cases_extra = manifests["cases"].get("extra", {})
    if cases_extra.get("selection_sha256") != actual_selection_hash \
            or cases_extra.get("selection_used_pre_outcome_columns_only") is not True:
        raise ValueError("case-selection hash or outcome-blind manifest assertion failed")
    network_counts = manifests["network"]["counts"]
    expected_network_counts = {
        "nodes": 32, "edge_cells": 1024, "bootstrap_draws": 500,
        "leave_one_domain_out_rows": 96,
    }
    actual_network_counts = {name: network_counts.get(name) for name in expected_network_counts}
    if actual_network_counts != expected_network_counts:
        raise ValueError(f"network manifest dimension mismatch: {actual_network_counts}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip()
    write_source, source_records = source_writer(commit)
    con = connect("50GB", 8)

    figure1_source, paths = figure1(con, nodes, manifests["v2_exposure"])
    write_source("SourceData_Figure1.csv", figure1_source, "Figure 1", "a-d",
                 "qss_v2/journal_year_scope.parquet;artifacts/qss_v2/run_exposure.json;"
                 "results/qss_v3/network_nodes.csv")
    figure2_source = pd.concat([
        pd.DataFrame([estimate_row(estimates, "primary", name) for name in
                      ("total_citations", "near", "intermediate", "far")]),
        pd.DataFrame([estimate_row(estimates, "primary", "far_to_near_routing"),
                      estimate_row(estimates, "primary", "any_far")]),
    ], ignore_index=True)
    write_source("SourceData_Figure2.csv", figure2_source, "Figure 2", "a-d",
                 "results/qss_v3/dirty_estimates.csv")
    figure3_source, new_paths = figure3(estimates, same_author)
    paths += new_paths
    write_source("SourceData_Figure4_estimates.csv", figure3_source, "Figure 4", "a",
                 "results/qss_v3/dirty_estimates.csv")
    write_source("SourceData_Figure4_same_author.csv", same_author,
                 "Figure 4", "b", "results/qss_v3/same_author_sensitivity.csv")
    domain_names = labels.set_index("qwen_macro").display_label
    source_nodes = nodes.rename(columns={"qwen_macro": "internal_domain_id"})
    source_edges = edges.rename(columns={
        "source_macro": "source_internal_domain_id",
        "target_macro": "citing_internal_domain_id",
    }).assign(
        source_domain=edges.source_macro.map(domain_names),
        citing_domain=edges.target_macro.map(domain_names),
    )
    source_lodo = lodo.rename(columns={
        "omitted_source_macro": "omitted_internal_domain_id",
    }).assign(
        omitted_source_domain=lodo.omitted_source_macro.map(domain_names),
    )
    if source_edges[["source_domain", "citing_domain"]].isna().any().any() \
            or source_lodo.omitted_source_domain.isna().any():
        raise ValueError("reader-facing network Source Data lost a domain name")
    write_source("SourceData_Figure3_nodes.csv", source_nodes, "Figure 3", "a",
                 "results/qss_v3/network_nodes.csv")
    write_source("SourceData_Figure3_edges.csv", source_edges, "Figure 3", "a",
                 "results/qss_v3/network_edges.csv")
    write_source("SourceData_Figure3_metrics.csv", metrics, "Figure 3", "b",
                 "results/qss_v3/network_metrics.csv")
    redesigned_sources = [
        ("SourceData_Figure2_Years.csv", "figure2_years.csv", "Figure 2", "c"),
        ("SourceData_Hierarchy_Papers.csv", "hierarchy_papers.csv", "Figure 2", "a"),
        ("SourceData_Hierarchy_Areas.csv", "hierarchy_areas.csv", "Figure 2", "a"),
        ("SourceData_Hierarchy_Edges.csv", "hierarchy_edges.csv", "Figure 2", "a"),
        ("SourceData_Hierarchy_Journals.csv", "hierarchy_journals.csv", "Figure 2", "a"),
        ("SourceData_Radial_AreaOrder.csv", "radial_area_order.csv", "Figure 3", "a"),
        ("SourceData_Radial_CitationBackbone.csv", "radial_citation_backbone.csv", "Figure 3", "a"),
        ("SourceData_Radial_LeafScope.csv", "radial_leaf_scope.csv", "Figure 3", "a"),
    ]
    for source_name, result_name, figure_name, panel_name in redesigned_sources:
        write_source(source_name, pd.read_csv(RESULTS / result_name), figure_name, panel_name,
                     f"results/qss_v3/{result_name}")
    ed1 = ed1_data(
        manifests["v2_dirty"], manifests["v3_prepare"], manifests["v3_analyze"],
        manifests["network"],
    )
    paths += extended_data1(ed1)
    write_source("SourceData_ED1_cohort_coverage.csv", ed1, "Extended Data Figure 1", "a-d",
                 "artifacts/qss_v2/run_dirty_analyze.json;artifacts/qss_v3/run_prepare.json;"
                 "artifacts/qss_v3/run_analyze.json;artifacts/qss_v3/run_network.json")
    bins = propensity_bins(con)
    balance, candidates = diagnostics_data()
    paths += extended_data2(bins, balance, candidates)
    write_source("SourceData_ED2_balance.csv", balance, "Extended Data Figure 2", "c",
                 "results/qss_v3/balance.csv;results/qss_v3/downstream_balance.csv")
    write_source("SourceData_ED2_propensity_candidates.csv", candidates,
                 "Extended Data Figure 2", "b,d",
                 "results/qss_v3/propensity_candidates.csv;results/qss_v3/downstream_propensity.csv")
    write_source("SourceData_ED2_propensity_bins.csv", bins, "Extended Data Figure 2", "a",
                 "qss_v3/routing_scores.parquet")
    sensitivity = sensitivity_data(estimates, dynamics)
    paths += extended_data3(nodes, edges, metrics, lodo, estimates, dynamics)
    write_source("SourceData_ED3_network_edges.csv", source_edges,
                 "Extended Data Figure 3", "a", "results/qss_v3/network_edges.csv")
    write_source("SourceData_ED3_lodo.csv", source_lodo,
                 "Extended Data Figure 3", "b",
                 "results/qss_v3/network_leave_one_domain_out.csv")
    write_source("SourceData_ED3_sensitivities.csv", sensitivity,
                 "Extended Data Figure 3", "c-d",
                 "results/qss_v3/dirty_estimates.csv;results/qss_v3/citation_dynamics.csv")
    paths += extended_data4(corridors, nodes)
    corridor_columns = [
        "case_rank", "qwen_macro", "display_label", "broad_id", "narrow_id",
        "broad_name", "narrow_name", "broad_scope", "narrow_scope", "broad_n",
        "narrow_n", "shared_n", "overlap", "routing_change_percent",
        "routing_ci_low_percent", "routing_ci_high_percent", "subgroup_n", "journals",
    ]
    write_source("SourceData_ED4_corridors.csv", corridors[corridor_columns],
                 "Extended Data Figure 4", "a-d",
                 "results/qss_v3/case_selection.csv;results/qss_v3/journal_corridors.csv")
    write_source("SourceData_ED4_nodes.csv", source_nodes[
        ["internal_domain_id", "display_label", "mds_x", "mds_y", "source_share"]
    ], "Extended Data Figure 4", "a-d", "results/qss_v3/network_nodes.csv")

    manifest = pd.DataFrame(source_records).sort_values(["figure/panel", "source_file"])
    if len(manifest) != 16 or manifest.sha256.str.fullmatch(r"[0-9a-f]{64}").sum() != 16:
        raise ValueError(f"expected 16 hashed source-data files, got {len(manifest)}")
    manifest.to_csv(SOURCE_DATA / "source_data_manifest.csv", index=False)
    if len(paths) != 16 or len(list(FIGURES.glob("*.pdf"))) != 8 \
            or len(list(FIGURES.glob("*.png"))) != 8:
        raise ValueError(f"expected 8 PDF and 8 PNG figures, got paths={len(paths)}")
    check_budget()
    log(f"article figures complete files={len(paths)} source_files={len(source_records)} "
        f"source_rows={manifest.rows.sum():,} seed={SEED} commit={commit}")


if __name__ == "__main__":
    main()
