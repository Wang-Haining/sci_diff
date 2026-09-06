#!/usr/bin/env python3
import json
import shutil

import numpy as np
import pandas as pd

from qss_common import GROUP_ROOT, SEED
from qss_v3_common import (
    ARTIFACTS,
    RESULTS,
    V2_STAGED,
    V2_WORK,
    V3_WORK,
    check_budget,
    connect,
    log,
    path_glob,
    tree_bytes,
    validate_snapshot,
    write_run,
)

SCORES = V3_WORK / "routing_scores.parquet"
ANALYSIS = V3_WORK / "analysis_dataset.parquet"
EDGES = V3_WORK / "citation_edges"
CITING = V3_WORK / "citing_metadata"
CANDIDATE = V3_WORK / "candidate_focal.parquet"
QWEN_V2 = V2_WORK / "qwen3_semantics.parquet"
QWEN_V3 = V3_WORK / "qwen3_semantics"
TAXONOMY = ARTIFACTS.parent / "qss_v2/qwen3_taxonomy.npz"
OUTCOME_RUN = ARTIFACTS.parent / "qss_v2/run_outcome_embed.json"
PREPARE_RUN = ARTIFACTS / "run_prepare.json"
ANALYZE_RUN = ARTIFACTS / "run_analyze.json"
DOWNSTREAM_RUN = ARTIFACTS / "run_downstream.json"
EMBED_RUN = ARTIFACTS / "run_embed.json"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
MACROS = 32
LEAVES = 1_000
BOOTSTRAPS = 500
OUTPUTS = {
    "nodes": RESULTS / "network_nodes.csv",
    "edges": RESULTS / "network_edges.csv",
    "metrics": RESULTS / "network_metrics.csv",
    "lodo": RESULTS / "network_leave_one_domain_out.csv",
    "dynamics": RESULTS / "citation_dynamics.csv",
    "same_journal": RESULTS / "same_journal_sensitivity.csv",
    "score_diagnostics": RESULTS / "round2_score_diagnostics.csv",
}


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"expected manifest at {path}")
    return json.loads(path.read_text())


def load_taxonomy():
    outcome = load_json(OUTCOME_RUN)
    prepare = load_json(PREPARE_RUN)
    embed = load_json(EMBED_RUN)
    analyze = load_json(ANALYZE_RUN)
    downstream = load_json(DOWNSTREAM_RUN)
    if outcome["extra"].get("model_commit") != MODEL_REVISION:
        raise ValueError(f"expected Qwen revision {MODEL_REVISION}, got "
                         f"{outcome['extra'].get('model_commit')}")
    if prepare["extra"].get("window") != "[publication_date, publication_date + 60 months)":
        raise ValueError(f"unexpected citation window: {prepare['extra'].get('window')}")
    if embed["extra"].get("qwen3_model_commit") != MODEL_REVISION \
            or embed["extra"].get("embedding_dimension") != 768:
        raise ValueError(f"unexpected v3 Qwen embedding contract: {embed['extra']}")
    if analyze["counts"].get("support") != 3_818_173:
        raise ValueError(f"expected primary support 3,818,173, got "
                         f"{analyze['counts'].get('support')}")
    qwen_qc = analyze["extra"].get("combined_qwen_qc")
    if not isinstance(qwen_qc, list) or len(qwen_qc) != 2 \
            or qwen_qc[0] <= 0 or qwen_qc[0] != qwen_qc[1]:
        raise ValueError(f"expected successful unique combined Qwen QC, got {qwen_qc}")
    if not downstream["extra"].get("deterministic_lightgbm") \
            or downstream["counts"].get("support") != downstream["counts"].get("routing_scores"):
        raise ValueError(f"invalid deterministic downstream manifest: {downstream}")
    bundle = np.load(TAXONOMY)
    centers = bundle["leaf_centers"].astype(np.float64)
    leaf_to_macro = bundle["leaf_to_macro"].astype(np.int16)
    macro_centers = bundle["macro_centers"].astype(np.float64)
    if centers.shape != (LEAVES, 768) or leaf_to_macro.shape != (LEAVES,):
        raise ValueError(f"unexpected leaf taxonomy shapes: {centers.shape}, "
                         f"{leaf_to_macro.shape}")
    if macro_centers.shape != (MACROS, 768) or set(leaf_to_macro) != set(range(MACROS)):
        raise ValueError(f"unexpected macro taxonomy: centers={macro_centers.shape}, "
                         f"labels={np.unique(leaf_to_macro)}")
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    macro_centers /= np.linalg.norm(macro_centers, axis=1, keepdims=True)
    distances = np.clip(1 - centers @ centers.T, 0, 2)
    np.fill_diagonal(distances, 0)
    return leaf_to_macro, macro_centers, distances, int(qwen_qc[0])


def classical_mds(centers):
    distances = np.clip(1 - centers @ centers.T, 0, 2)
    centering = np.eye(MACROS) - np.ones((MACROS, MACROS)) / MACROS
    gram = -0.5 * centering @ np.square(distances) @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    if np.any(values[order] <= 0):
        raise ValueError(f"expected two positive MDS eigenvalues, got {values[order]}")
    coordinates = vectors[:, order] * np.sqrt(values[order])
    for column in range(2):
        anchor = np.argmax(np.abs(coordinates[:, column]))
        if coordinates[anchor, column] < 0:
            coordinates[:, column] *= -1
    return coordinates


def validate_inputs(con, leaf_to_macro, expected_qwen):
    if not SCORES.is_file():
        raise FileNotFoundError(f"expected completed downstream scores at {SCORES}")
    downstream = load_json(DOWNSTREAM_RUN)
    score_qc = con.execute("""
      SELECT count(*),count(DISTINCT id),count(DISTINCT qwen_macro),
             min(propensity),max(propensity),min(treatment),max(treatment)
      FROM read_parquet(?)
    """, [str(SCORES)]).fetchone()
    if score_qc[0] != downstream["counts"].get("routing_scores") \
            or SCORES.stat().st_size != downstream["extra"].get("score_bytes"):
        raise ValueError(f"routing scores do not match downstream manifest: "
                         f"rows={score_qc[0]} bytes={SCORES.stat().st_size}")
    if score_qc[0] != score_qc[1] or score_qc[2] != MACROS or score_qc[5:] != (0, 1):
        raise ValueError(f"routing score QC failed: {score_qc}")
    if not (0.05 <= score_qc[3] <= score_qc[4] <= 0.95):
        raise ValueError(f"expected propensity in [0.05,0.95], got {score_qc[3:5]}")

    expected_edges = load_json(PREPARE_RUN)["counts"]["citation_edges"]
    edge_qc = con.execute("""
      SELECT count(*) AS n,
             count(DISTINCT struct_pack(citing_id:=citing_id,cited_id:=cited_id)) AS unique_n
      FROM read_parquet(?)
    """, [path_glob(EDGES)]).fetchone()
    if edge_qc != (expected_edges, expected_edges):
        raise ValueError(f"expected {expected_edges:,} unique citation edges, got {edge_qc}")
    expected_citing = load_json(PREPARE_RUN)["counts"]["citing_metadata"]
    citing_qc = con.execute("""
      SELECT count(*),count(DISTINCT id) FROM read_parquet(?)
    """, [path_glob(CITING)]).fetchone()
    if citing_qc != (expected_citing, expected_citing):
        raise ValueError(f"expected {expected_citing:,} unique citing records, got {citing_qc}")
    outside = con.execute("""
      SELECT count(*) FROM read_parquet(?) e JOIN read_parquet(?) f ON e.cited_id=f.id
      WHERE e.citing_date<f.publication_date
         OR e.citing_date>=f.publication_date+INTERVAL 60 MONTH
    """, [path_glob(EDGES), str(CANDIDATE)]).fetchone()[0]
    if outside:
        raise ValueError(f"expected all citations inside rolling 60-month window, got {outside}")

    mapping = pd.DataFrame({
        "qwen_leaf": np.arange(LEAVES, dtype=np.int16),
        "frozen_macro": leaf_to_macro,
    })
    con.register("leaf_macro", mapping)
    con.execute(f"""
      CREATE OR REPLACE TEMP VIEW qwen_all AS
      SELECT id,qwen_leaf,qwen_macro,qwen_ood FROM read_parquet('{QWEN_V2}')
      UNION ALL
      SELECT id,qwen_leaf,qwen_macro,qwen_ood FROM read_parquet('{path_glob(QWEN_V3)}')
    """)
    live_qwen_qc = con.execute(
        "SELECT count(*),count(DISTINCT id) FROM qwen_all",
    ).fetchone()
    if live_qwen_qc != (expected_qwen, expected_qwen):
        raise ValueError(f"expected {expected_qwen:,} complete unique Qwen semantics, "
                         f"got {live_qwen_qc}")
    con.execute("""
      CREATE TEMP TABLE network_journals AS
      SELECT journal_id,row_number() OVER (ORDER BY journal_id)-1 AS journal_code
      FROM (SELECT DISTINCT journal_id FROM read_parquet(?))
    """, [str(SCORES)])
    con.execute("""
      CREATE TEMP TABLE network_support AS
      SELECT s.id,j.journal_code,s.journal_id,s.treatment,s.propensity,
             s.qwen_macro AS source_macro,q.qwen_leaf AS source_leaf,c.author_ids,
             c.publication_date AS focal_date,
             CASE WHEN s.treatment=1 THEN 1.0/s.propensity
                  ELSE 1.0/(1.0-s.propensity) END AS ipw
      FROM read_parquet(?) s
      JOIN network_journals j USING (journal_id)
      JOIN read_parquet(?) c USING (id)
      JOIN qwen_all q USING (id)
      JOIN leaf_macro m USING (qwen_leaf)
      WHERE NOT q.qwen_ood AND q.qwen_macro=m.frozen_macro
        AND s.qwen_macro=m.frozen_macro
    """, [str(SCORES), str(CANDIDATE)])
    support_qc = con.execute("""
      SELECT count(*),count(DISTINCT id),count(DISTINCT journal_code),
             count(DISTINCT source_macro) FROM network_support
    """).fetchone()
    if support_qc[0] != score_qc[0] or support_qc[0] != support_qc[1] \
            or support_qc[3] != MACROS:
        raise ValueError(f"network support QC failed: scores={score_qc[0]}, got={support_qc}")
    log(f"network inputs support={support_qc[0]:,} journals={support_qc[2]:,} "
        f"edges={edge_qc[0]:,}")
    return score_qc[0], support_qc[2], edge_qc[0]


def build_eligible_edges(con):
    con.execute("""
      CREATE TEMP TABLE network_joined_edges AS
      SELECT s.id,s.journal_code,s.treatment,s.source_macro,s.source_leaf,s.ipw,
             s.focal_date,e.citing_date,
             qc.qwen_leaf AS target_leaf,m.frozen_macro AS target_macro,
             c.journal_id=s.journal_id AS same_journal,
             COALESCE(list_has_any(c.author_ids,s.author_ids),false) AS shared_author,
             COALESCE(c.language='en' AND qc.id IS NOT NULL AND NOT qc.qwen_ood
               AND qc.qwen_macro=m.frozen_macro,false) AS classified
      FROM read_parquet(?) e
      JOIN network_support s ON e.cited_id=s.id
      JOIN read_parquet(?) c ON e.citing_id=c.id
      LEFT JOIN qwen_all qc ON c.id=qc.id
      LEFT JOIN leaf_macro m ON qc.qwen_leaf=m.qwen_leaf
    """, [path_glob(EDGES), path_glob(CITING)])
    decomposition = con.execute("""
      SELECT count(*) AS support_edges,
             count(*) FILTER (WHERE same_journal IS NULL) AS missing_citing_journal,
             count(*) FILTER (WHERE same_journal) AS same_journal,
             count(*) FILTER (WHERE shared_author) AS shared_author,
             count(*) FILTER (WHERE same_journal IS DISTINCT FROM false OR shared_author)
               AS excluded_external,
             count(*) FILTER (WHERE same_journal=false AND NOT shared_author) AS external,
             count(*) FILTER (WHERE same_journal=false AND NOT shared_author AND NOT classified)
               AS unclassified,
             count(*) FILTER (WHERE same_journal=false AND NOT shared_author AND classified)
               AS eligible
      FROM network_joined_edges
    """).fetchone()
    if decomposition[0] != decomposition[4] + decomposition[5] \
            or decomposition[5] != decomposition[6] + decomposition[7] \
            or decomposition[7] <= 0:
        raise ValueError(f"citation exclusion decomposition failed: {decomposition}")
    con.execute("""
      CREATE TEMP TABLE network_eligible_edges AS
      SELECT id,journal_code,treatment,source_macro,target_macro,source_leaf,target_leaf,
             ipw,focal_date,citing_date
      FROM network_joined_edges
      WHERE same_journal=false AND NOT shared_author AND classified
    """)
    log("network eligible citations " + " ".join(
        f"{name}={value:,}" for name, value in zip(
            ["support", "missing_journal", "same_journal", "shared_author",
             "excluded_external", "external", "unclassified", "eligible"], decomposition,
        )
    ))
    return decomposition


def score_diagnostics(con, journal_n, multipliers):
    reader = con.execute("""
      WITH scored AS (
        SELECT j.journal_code,s.treatment,a.january_1,
               s.psi_near_0,s.psi_near_1,s.psi_far_0,s.psi_far_1,
          CASE WHEN s.treatment=1 THEN s.psi_near_0
               ELSE (a.near-(1-s.propensity)*s.psi_near_0)/s.propensity END AS m_near_0,
          CASE WHEN s.treatment=0 THEN s.psi_near_1
               ELSE (a.near-s.propensity*s.psi_near_1)/(1-s.propensity) END AS m_near_1,
          CASE WHEN s.treatment=1 THEN s.psi_far_0
               ELSE (a.far-(1-s.propensity)*s.psi_far_0)/s.propensity END AS m_far_0,
          CASE WHEN s.treatment=0 THEN s.psi_far_1
               ELSE (a.far-s.propensity*s.psi_far_1)/(1-s.propensity) END AS m_far_1
        FROM read_parquet(?) s JOIN read_parquet(?) a USING (id)
        JOIN network_journals j USING (journal_id)
      ), stacked AS (
        SELECT *,0 AS subset FROM scored
        UNION ALL SELECT *,1 FROM scored WHERE NOT january_1
      )
      SELECT journal_code,subset,count(*) AS n,
             count(*) FILTER (WHERE treatment=0) AS broad_n,
             count(*) FILTER (WHERE treatment=1) AS specialized_n,
             sum(psi_near_0),sum(psi_near_1),sum(psi_far_0),sum(psi_far_1),
             sum(m_near_0),sum(m_near_1),sum(m_far_0),sum(m_far_1)
      FROM stacked GROUP BY ALL ORDER BY journal_code,subset
    """, [str(SCORES), str(ANALYSIS)]).fetchall()
    values = np.zeros((journal_n, 2, 11), dtype=np.float64)
    for row in reader:
        values[row[0], row[1]] = row[2:]
    totals = values.sum(axis=0)
    if int(totals[0, 0]) != load_json(DOWNSTREAM_RUN)["counts"]["support"] \
            or not 0 < totals[1, 0] < totals[0, 0]:
        raise ValueError(f"score diagnostic population QC failed: {totals[:, :3]}")
    jan_rates = (totals[0, 1:3] - totals[1, 1:3]) / totals[0, 1:3]
    output = []
    for subset, population in enumerate(("all_support", "exclude_january_1")):
        for estimator, start in (("aipw", 3), ("outcome_model_only", 7)):
            n = totals[subset, 0]
            means = totals[subset, start:start + 4] / n
            if np.any(means <= 0) or not np.isfinite(means).all():
                raise ValueError(f"invalid {population} {estimator} means: {means}")
            theta = np.log(means[3]) - np.log(means[1]) \
                - np.log(means[2]) + np.log(means[0])
            centered = values[:, subset, start:start + 4] \
                - values[:, subset, 0][:, None] * means
            influence = centered[:, 3] / means[3] - centered[:, 1] / means[1] \
                - centered[:, 2] / means[2] + centered[:, 0] / means[0]
            groups = int(np.count_nonzero(values[:, subset, 0]))
            se = np.sqrt((groups / (groups - 1)) * np.square(influence).sum() / n ** 2)
            draws = multipliers @ values[:, subset]
            draw_means = draws[:, start:start + 4] / draws[:, [0]]
            draw_theta = np.log(draw_means[:, 3]) - np.log(draw_means[:, 1]) \
                - np.log(draw_means[:, 2]) + np.log(draw_means[:, 0])
            output.append({
                "population": population, "estimator": estimator, "n": int(n),
                "journals": groups, "broad_n": int(totals[subset, 1]),
                "specialized_n": int(totals[subset, 2]), "near_broad": means[0],
                "near_specialized": means[1], "distant_broad": means[2],
                "distant_specialized": means[3], "estimate": theta,
                "relative_change": np.exp(theta) - 1, "se": se,
                "ci_low": theta - 1.96 * se, "ci_high": theta + 1.96 * se,
                "bootstrap_ci_low": np.quantile(draw_theta, 0.025),
                "bootstrap_ci_high": np.quantile(draw_theta, 0.975),
                "january_1_rate_broad": jan_rates[0],
                "january_1_rate_specialized": jan_rates[1],
            })
    frame = pd.DataFrame(output)
    expected = load_json(DOWNSTREAM_RUN)["extra"]["reproduced_theta"]
    actual = frame.loc[(frame.population == "all_support")
                       & (frame.estimator == "aipw"), "estimate"].iloc[0]
    if not np.isclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError(f"expected downstream theta {expected}, got {actual}")
    log(f"score diagnostics rows={len(frame)} non-Jan-1={int(totals[1, 0]):,}")
    return frame


def same_journal_sensitivity(con, journal_n, multipliers):
    reader = con.execute("""
      WITH focal AS (
        SELECT journal_code,treatment,sum(ipw) AS denominator
        FROM network_support GROUP BY ALL
      ), citations AS (
        SELECT journal_code,treatment,
          sum(ipw) FILTER (WHERE same_journal=false AND classified
                            AND source_leaf=target_leaf) AS external_near,
          sum(ipw) FILTER (WHERE same_journal=false AND classified
                            AND source_leaf<>target_leaf AND source_macro=target_macro)
                            AS external_intermediate,
          sum(ipw) FILTER (WHERE same_journal=false AND classified
                            AND source_macro<>target_macro) AS external_distant,
          sum(ipw) FILTER (WHERE same_journal AND classified
                            AND source_leaf=target_leaf) AS same_near,
          sum(ipw) FILTER (WHERE same_journal AND classified
                            AND source_leaf<>target_leaf AND source_macro=target_macro)
                            AS same_intermediate,
          sum(ipw) FILTER (WHERE same_journal AND classified
                            AND source_macro<>target_macro) AS same_distant,
          count(*) FILTER (WHERE same_journal AND NOT shared_author) AS same_nonself,
          count(*) FILTER (WHERE same_journal AND NOT shared_author AND NOT classified)
                            AS same_unclassified
        FROM network_joined_edges WHERE NOT shared_author GROUP BY ALL
      )
      SELECT f.*,c.* EXCLUDE(journal_code,treatment)
      FROM focal f LEFT JOIN citations c USING (journal_code,treatment)
      ORDER BY journal_code,treatment
    """).fetch_record_batch(100_000)
    values = np.zeros((journal_n, 2, 9), dtype=np.float64)
    rows = 0
    for batch in reader:
        columns = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(11)]
        for row in zip(*columns):
            journal, arm = row[:2]
            values[journal, arm] = np.nan_to_num(row[2:], nan=0.0)
        rows += len(batch)
    total = values.sum(axis=0)
    same_nonself = int(total[:, 7].sum())
    same_unclassified = int(total[:, 8].sum())
    if rows <= 0 or same_nonself != 1_888_865 or same_unclassified >= same_nonself:
        raise ValueError(f"same-journal edge QC failed rows={rows} nonself={same_nonself} "
                         f"unclassified={same_unclassified}")

    def estimates(array):
        means = array[..., 1:7] / array[..., [0]]
        external = np.log(means[..., 2]) - np.log(means[..., 0])
        inclusive = np.log(means[..., 2] + means[..., 5]) \
            - np.log(means[..., 0] + means[..., 3])
        return means, np.stack([
            external[..., 1] - external[..., 0],
            inclusive[..., 1] - inclusive[..., 0],
            (inclusive[..., 1] - inclusive[..., 0])
            - (external[..., 1] - external[..., 0]),
        ], axis=-1)

    means, base = estimates(total)
    draw_values = (multipliers @ values.reshape(journal_n, -1)).reshape(
        BOOTSTRAPS, 2, 9,
    )
    _, draws = estimates(draw_values)
    output = []
    for index, name in enumerate(("external", "inclusive", "inclusive_minus_external")):
        se = float(draws[:, index].std(ddof=1))
        output.append({
            "estimand": name, "estimate": base[index], "se": se,
            "ci_low": base[index] - 1.96 * se, "ci_high": base[index] + 1.96 * se,
            "bootstrap_ci_low": np.quantile(draws[:, index], 0.025),
            "bootstrap_ci_high": np.quantile(draws[:, index], 0.975),
            "near_broad": means[0, 0] + (means[0, 3] if index else 0),
            "near_specialized": means[1, 0] + (means[1, 3] if index else 0),
            "distant_broad": means[0, 2] + (means[0, 5] if index else 0),
            "distant_specialized": means[1, 2] + (means[1, 5] if index else 0),
            "same_journal_nonself_edges": same_nonself,
            "same_journal_unclassified_edges": same_unclassified,
        })
    con.execute("DROP TABLE network_joined_edges")
    frame = pd.DataFrame(output)
    log(f"same-journal sensitivity external={base[0]:.6f} inclusive={base[1]:.6f} "
        f"delta={base[2]:.6f} edges={same_nonself:,}")
    return frame


def citation_dynamics(con, journal_n, multipliers):
    horizons = pd.DataFrame({"horizon_months": [12, 24, 36, 48, 60]})
    con.register("citation_horizons", horizons)
    reader = con.execute("""
      WITH paper AS (
        SELECT s.journal_code,s.treatment,s.id,s.ipw,h.horizon_months,
               count(e.citing_date) FILTER (
                 WHERE e.target_macro<>e.source_macro
                   AND e.citing_date<s.focal_date+h.horizon_months*INTERVAL 1 MONTH
               ) AS distant
        FROM network_support s CROSS JOIN citation_horizons h
        LEFT JOIN network_eligible_edges e ON s.id=e.id
        GROUP BY ALL
      )
      SELECT journal_code,treatment,horizon_months,sum(ipw) AS denominator,
             sum(ipw*distant) AS distant,
             sum(ipw*(distant>0)::INTEGER) AS any_distant,
             sum(ipw*greatest(distant-1,0)) AS additional_after_first
      FROM paper GROUP BY ALL ORDER BY journal_code,treatment,horizon_months
    """).fetch_record_batch(100_000)
    values = np.zeros((journal_n, 2, len(horizons), 4), dtype=np.float64)
    horizon_index = {value: index for index, value in enumerate(horizons.horizon_months)}
    rows = 0
    for batch in reader:
        columns = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(7)]
        for journal, arm, horizon, denominator, distant, any_distant, additional in zip(*columns):
            values[journal, arm, horizon_index[horizon]] = (
                denominator, distant, any_distant, additional,
            )
        rows += len(batch)
    if rows <= 0 or np.any(values.sum(axis=0)[..., 0] <= 0):
        raise ValueError(f"invalid citation-dynamics aggregation rows={rows}")

    def means(array):
        return array[..., 1:] / array[..., [0]]

    base = means(values.sum(axis=0))
    draws = means((multipliers @ values.reshape(journal_n, -1)).reshape(
        BOOTSTRAPS, 2, len(horizons), 4,
    ))
    if not np.allclose(base[..., 0], base[..., 1] + base[..., 2], atol=1e-10) \
            or not np.allclose(draws[..., 0], draws[..., 1] + draws[..., 2], atol=1e-10):
        raise ValueError("expected distant = any distant + additional-after-first exactly")
    output = []
    names = ["distant_citations", "any_distant", "additional_after_first"]
    for horizon_index_, horizon in enumerate(horizons.horizon_months):
        for outcome_index, outcome in enumerate(names):
            contrast = base[1, horizon_index_, outcome_index] - base[0, horizon_index_, outcome_index]
            contrast_draws = (draws[:, 1, horizon_index_, outcome_index]
                              - draws[:, 0, horizon_index_, outcome_index])
            se = float(contrast_draws.std(ddof=1))
            output.append({
                "horizon_months": int(horizon), "outcome": outcome,
                "broad": base[0, horizon_index_, outcome_index],
                "specialized": base[1, horizon_index_, outcome_index],
                "specialized_minus_broad": contrast, "se": se,
                "ci_low": contrast - 1.96 * se, "ci_high": contrast + 1.96 * se,
                "bootstrap_ci_low": np.quantile(contrast_draws, 0.025),
                "bootstrap_ci_high": np.quantile(contrast_draws, 0.975),
            })
    frame = pd.DataFrame(output)
    log(f"citation dynamics rows={len(frame)} endpoint="
        f"{frame.loc[(frame.horizon_months==60) & (frame.outcome=='distant_citations'), 'specialized_minus_broad'].iloc[0]:.6f}")
    return frame


def fill_dense(con, journal_n, distances):
    focal_n = np.zeros((journal_n, 2, MACROS), dtype=np.float64)
    focal_ipw = np.zeros_like(focal_n)
    focal = con.execute("""
      SELECT journal_code,treatment,source_macro,count(*) AS n,sum(ipw) AS ipw
      FROM network_support GROUP BY ALL
    """).fetchall()
    for journal, arm, source, n, weight in focal:
        focal_n[journal, arm, source] = n
        focal_ipw[journal, arm, source] = weight

    distance_table = pd.DataFrame({
        "source_leaf": np.repeat(np.arange(LEAVES, dtype=np.int16), LEAVES),
        "target_leaf": np.tile(np.arange(LEAVES, dtype=np.int16), LEAVES),
        "semantic_distance": distances.astype(np.float32).ravel(),
    })
    con.register("leaf_distances", distance_table)
    weighted = np.zeros((journal_n, 2, MACROS, MACROS), dtype=np.float64)
    distance_sum = np.zeros_like(weighted)
    raw = np.zeros((2, MACROS, MACROS), dtype=np.int64)
    reader = con.execute("""
      SELECT e.journal_code,e.treatment,e.source_macro,e.target_macro,
             count(*) AS raw_edges,sum(e.ipw) AS weighted_edges,
             sum(e.ipw*d.semantic_distance) AS weighted_distance
      FROM network_eligible_edges e JOIN leaf_distances d USING (source_leaf,target_leaf)
      GROUP BY ALL ORDER BY e.journal_code,e.treatment,e.source_macro,e.target_macro
    """).fetch_record_batch(100_000)
    rows = 0
    for batch in reader:
        columns = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(7)]
        for journal, arm, source, target, n, weight, span in zip(*columns):
            weighted[journal, arm, source, target] = weight
            distance_sum[journal, arm, source, target] = span
            raw[arm, source, target] += n
        rows += len(batch)
    if rows <= 0 or raw.sum() <= 0:
        raise ValueError(f"expected nonempty journal-domain aggregation, got rows={rows}")
    log(f"network aggregate rows={rows:,} eligible={raw.sum():,}")
    return focal_n, focal_ipw, raw, weighted, distance_sum


def leaf_nulls(con, pi, row_totals, distances):
    output = []
    for column in ("source_leaf", "target_leaf"):
        values = np.zeros((2, MACROS, LEAVES), dtype=np.float64)
        rows = con.execute(f"""
          SELECT treatment,source_macro,{column},sum(ipw)
          FROM network_eligible_edges GROUP BY ALL
        """).fetchall()
        for arm, source, leaf, weight in rows:
            values[arm, source, leaf] = weight
        output.append(values)
    source, target = output
    nulls = np.empty(2)
    for arm in (0, 1):
        out_leaf = (pi[:, None] * source[arm] / row_totals[arm, :, None]).sum(axis=0)
        in_leaf = (pi[:, None] * target[arm] / row_totals[arm, :, None]).sum(axis=0)
        if not np.allclose([out_leaf.sum(), in_leaf.sum()], 1, atol=1e-9):
            raise ValueError(f"leaf configuration margins do not sum to one for arm={arm}: "
                             f"{out_leaf.sum()}, {in_leaf.sum()}")
        nulls[arm] = out_leaf @ distances @ in_leaf
    return nulls


def network_values(focal, edges, distance_sum, keep=None):
    if focal.ndim == 2:
        focal = focal[None, ...]
        edges = edges[None, ...]
        distance_sum = distance_sum[None, ...]
    draws = len(focal)
    if keep is None:
        keep = np.ones(MACROS, dtype=bool)
    focal = focal.copy()
    focal[:, :, ~keep] = 0
    pi = focal.sum(axis=1)
    pi /= pi.sum(axis=1, keepdims=True)
    totals = edges.sum(axis=3)
    if np.any(totals[:, :, keep] <= 0):
        bad = np.argwhere(totals[:, :, keep] <= 0)[0]
        raise ValueError(f"zero citation row in network draw={bad[0]} arm={bad[1]}")
    probabilities = np.divide(
        edges, totals[..., None], out=np.zeros_like(edges), where=totals[..., None] > 0,
    )
    w = pi[:, None, :, None] * probabilities
    if not np.allclose(pi.sum(axis=1), 1, atol=1e-10) \
            or not np.allclose(probabilities[:, :, keep].sum(axis=3), 1, atol=1e-10) \
            or not np.allclose(w.sum(axis=(2, 3)), 1, atol=1e-10):
        raise ValueError("network flow conservation failed")
    incoming = w.sum(axis=2)
    modularity = np.trace(w, axis1=2, axis2=3) - (pi[:, None, :] * incoming).sum(axis=2)
    participation = (pi[:, None, :] * (MACROS / (MACROS - 1))
                     * (1 - np.square(probabilities).sum(axis=3))).sum(axis=2)
    source_span = np.divide(
        distance_sum.sum(axis=3), totals, out=np.zeros_like(totals), where=totals > 0,
    )
    span = (pi[:, None, :] * source_span).sum(axis=2)
    values = np.stack([modularity, participation, span], axis=2)
    return values[0] if draws == 1 else values


def metric_table(base, draws, nulls):
    names = ["directed_modularity", "audience_participation", "semantic_span"]
    directions = ["higher", "lower", "lower"]
    rows = []
    for index, (name, direction) in enumerate(zip(names, directions)):
        contrast = base[1, index] - base[0, index]
        contrast_draws = draws[:, 1, index] - draws[:, 0, index]
        se = float(contrast_draws.std(ddof=1))
        rows.append({
            "metric": name, "expected_specialized_direction": direction,
            "broad": base[0, index], "specialized": base[1, index],
            "contrast_specialized_minus_broad": contrast, "se": se,
            "ci_low": contrast - 1.96 * se, "ci_high": contrast + 1.96 * se,
            "bootstrap_ci_low": np.quantile(contrast_draws, 0.025),
            "bootstrap_ci_high": np.quantile(contrast_draws, 0.975),
            "null_broad": nulls[0, index], "null_specialized": nulls[1, index],
            "direction_concordant": contrast > 0 if direction == "higher" else contrast < 0,
        })
    return pd.DataFrame(rows)


def output_edges(raw, weighted, focal_ipw, pi):
    totals = weighted.sum(axis=2)
    probabilities = weighted / totals[:, :, None]
    standardized = pi[None, :, None] * probabilities
    pooled = standardized.mean(axis=0)
    off_diagonal = ~np.eye(MACROS, dtype=bool)
    candidates = np.flatnonzero(off_diagonal.ravel())
    values = pooled.ravel()[candidates]
    order = candidates[np.lexsort((candidates, -values))]
    cumulative = np.cumsum(pooled.ravel()[order])
    cross_total = pooled[off_diagonal].sum()
    selected_n = int(np.searchsorted(cumulative, 0.5 * cross_total, side="left") + 1)
    selected = np.zeros(MACROS * MACROS, dtype=bool)
    selected[order[:selected_n]] = True
    rows = []
    for source in range(MACROS):
        for target in range(MACROS):
            rows.append({
                "source_macro": source, "target_macro": target,
                "raw_edges_broad": raw[0, source, target],
                "raw_edges_specialized": raw[1, source, target],
                "ipw_edges_broad": weighted[0, source, target],
                "ipw_edges_specialized": weighted[1, source, target],
                "rate_broad": weighted[0, source, target] / focal_ipw[0, source],
                "rate_specialized": weighted[1, source, target] / focal_ipw[1, source],
                "row_share_broad": probabilities[0, source, target],
                "row_share_specialized": probabilities[1, source, target],
                "row_share_difference": probabilities[1, source, target]
                                        - probabilities[0, source, target],
                "standardized_share_broad": standardized[0, source, target],
                "standardized_share_specialized": standardized[1, source, target],
                "standardized_share_difference": standardized[1, source, target]
                                                 - standardized[0, source, target],
                "pooled_standardized_share": pooled[source, target],
                "plot_edge": bool(selected[source * MACROS + target]),
            })
    frame = pd.DataFrame(rows)
    plotted_mass = frame.loc[frame.plot_edge, "pooled_standardized_share"].sum()
    if plotted_mass < 0.5 * cross_total or frame.plot_edge.sum() != selected_n:
        raise ValueError(f"treatment-blind edge selection failed: selected={selected_n}, "
                         f"mass={plotted_mass}, cross={cross_total}")
    return frame


def main():
    validate_snapshot()
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    leaf_to_macro, macro_centers, distances, expected_qwen = load_taxonomy()
    con = connect("650GB", 32)
    support_n, journal_n, raw_edge_n = validate_inputs(con, leaf_to_macro, expected_qwen)
    decomposition = build_eligible_edges(con)
    focal_n_j, focal_ipw_j, raw, weighted_j, distance_j = fill_dense(
        con, journal_n, distances,
    )
    if raw.sum() != decomposition[7]:
        raise ValueError(f"eligible-edge reconciliation failed: {raw.sum()} != {decomposition[7]}")

    focal_n = focal_n_j.sum(axis=0)
    focal_ipw = focal_ipw_j.sum(axis=0)
    weighted = weighted_j.sum(axis=0)
    distance_sum = distance_j.sum(axis=0)
    pi = focal_n.sum(axis=0) / focal_n.sum()
    if np.any(focal_n <= 0) or not np.isclose(pi.sum(), 1):
        raise ValueError(f"invalid focal domain distribution: min={focal_n.min()} sum={pi.sum()}")
    row_totals = weighted.sum(axis=2)
    if np.any(row_totals <= 0):
        raise ValueError(f"expected positive citation flow for all arm-domain rows, got "
                         f"min={row_totals.min()}")

    base = network_values(focal_n, weighted, distance_sum)
    rng = np.random.default_rng(SEED)
    multipliers = rng.poisson(1, size=(BOOTSTRAPS, journal_n)).astype(np.float64)
    diagnostics = score_diagnostics(con, journal_n, multipliers)
    dynamics = citation_dynamics(con, journal_n, multipliers)
    same_journal = same_journal_sensitivity(con, journal_n, multipliers)
    focal_draws = (multipliers @ focal_n_j.reshape(journal_n, -1)).reshape(
        BOOTSTRAPS, 2, MACROS,
    )
    weighted_draws = (multipliers @ weighted_j.reshape(journal_n, -1)).reshape(
        BOOTSTRAPS, 2, MACROS, MACROS,
    )
    distance_draws = (multipliers @ distance_j.reshape(journal_n, -1)).reshape(
        BOOTSTRAPS, 2, MACROS, MACROS,
    )
    draws = network_values(focal_draws, weighted_draws, distance_draws)
    if draws.shape != (BOOTSTRAPS, 2, 3) or not np.isfinite(draws).all():
        raise ValueError(f"network bootstrap QC failed: shape={draws.shape} "
                         f"finite={np.isfinite(draws).all()}")

    leaf_span_null = leaf_nulls(con, pi, row_totals, distances)
    probabilities = weighted / row_totals[:, :, None]
    incoming = (pi[None, :, None] * probabilities).sum(axis=1)
    participation_null = (MACROS / (MACROS - 1)) * (1 - np.square(incoming).sum(axis=1))
    nulls = np.column_stack([np.zeros(2), participation_null, leaf_span_null])
    metrics = metric_table(base, draws, nulls)

    lodo = []
    names = metrics.metric.tolist()
    for omitted in range(MACROS):
        keep = np.arange(MACROS) != omitted
        values = network_values(focal_n, weighted, distance_sum, keep)
        for index, name in enumerate(names):
            contrast = values[1, index] - values[0, index]
            lodo.append({
                "omitted_source_macro": omitted, "metric": name,
                "broad": values[0, index], "specialized": values[1, index],
                "contrast_specialized_minus_broad": contrast,
                "direction_concordant": contrast > 0 if index == 0 else contrast < 0,
            })
    lodo = pd.DataFrame(lodo)

    edge_output = output_edges(raw, weighted, focal_ipw, pi)
    coordinates = classical_mds(macro_centers)
    if not np.allclose(
        coordinates, classical_mds(macro_centers), rtol=0, atol=1e-12,
    ):
        raise ValueError("expected deterministic MDS coordinates")
    labels = pd.read_csv(RESULTS / "macro_labels.csv")
    if set(labels.qwen_macro) != set(range(MACROS)):
        raise ValueError(f"expected labels for 32 macrodomains, got {len(labels)}")
    diagonal = edge_output[edge_output.source_macro == edge_output.target_macro].set_index(
        "source_macro",
    )
    nodes = pd.DataFrame({
        "qwen_macro": np.arange(MACROS), "mds_x": coordinates[:, 0],
        "mds_y": coordinates[:, 1], "source_share": pi,
        "focal_broad": focal_n[0], "focal_specialized": focal_n[1],
        "self_retention_broad": diagonal.row_share_broad,
        "self_retention_specialized": diagonal.row_share_specialized,
        "self_retention_difference": diagonal.row_share_difference,
    }).merge(labels[["qwen_macro", "representative_journals"]], on="qwen_macro", validate="one_to_one")

    frames = {
        "nodes": nodes, "edges": edge_output, "metrics": metrics, "lodo": lodo,
        "dynamics": dynamics, "same_journal": same_journal,
        "score_diagnostics": diagnostics,
    }
    for name, frame in frames.items():
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"expected complete nonempty {name}, got rows={len(frame)} "
                             f"missing={int(frame.isna().sum().sum())}")
        frame.to_csv(OUTPUTS[name], index=False)
    output_metadata = {
        name: {"path": str(OUTPUTS[name].relative_to(RESULTS.parent.parent)),
               "rows": len(frame), "columns": frame.columns.tolist(),
               "bytes": OUTPUTS[name].stat().st_size}
        for name, frame in frames.items()
    }
    output_bytes = sum(item["bytes"] for item in output_metadata.values())
    if output_bytes >= 50_000_000:
        raise ValueError(f"expected network outputs <50MB, got {output_bytes:,}")

    concordant = int(metrics.direction_concordant.sum())
    run = write_run("network", {
        "support": support_n, "journals": journal_n, "raw_citation_edges": raw_edge_n,
        "support_citation_edges": decomposition[0], "eligible_edges": decomposition[7],
        "nodes": len(nodes), "edge_cells": len(edge_output),
        "bootstrap_draws": BOOTSTRAPS, "leave_one_domain_out_rows": len(lodo),
        "citation_dynamics_rows": len(dynamics),
        "same_journal_sensitivity_rows": len(same_journal),
        "score_diagnostic_rows": len(diagnostics),
    }, {
        "model_commit": MODEL_REVISION,
        "window": "[publication_date, publication_date + 60 months)",
        "edge_direction": "focal/cited macrodomain to citing-paper macrodomain",
        "support_source": "routing_scores.parquet from deterministic downstream refit",
        "standardization": "IPW destination shares within source macrodomain, common empirical source distribution",
        "semantic_span": "cosine distance between frozen Qwen3 leaf centroids",
        "layout": "classical MDS of 1-cosine macro-centroid distances; standard MDS squares dissimilarities in the double-centering step",
        "display_edges": "off-diagonal edges selected without arm labels to reach 50% of pooled cross-domain standardized mass",
        "bootstrap": "journal-cluster Poisson(1), shared across arms and metrics",
        "citation_dynamics": "Hajek IPW cumulative distant, any distant, and additional-after-first at 12-month intervals on fixed downstream support",
        "same_journal_sensitivity": "Hajek IPW external versus all non-shared-author citations on fixed downstream support",
        "score_diagnostics": "No-refit AIPW and reconstructed outcome-model predictions, with a fixed-score exclusion of January 1 focal dates",
        "score_diagnostic_estimates": {
            f"{row.population}:{row.estimator}": float(row.estimate)
            for row in diagnostics.itertuples()
        },
        "reference_version_limitation": "OpenAlex exposes one work-level referenced_works list; location version tags do not carry separate reference lists, so preprint-versus-journal reference change is not identifiable from this snapshot",
        "configuration_null": "directed source/destination independence; semantic-span null preserves standardized leaf margins",
        "exclusions": dict(zip(
            ["support_edges", "missing_citing_journal", "same_journal", "shared_author",
             "excluded_external", "external", "external_unclassified", "eligible"],
            [int(value) for value in decomposition],
        )),
        "metrics_concordant": concordant, "metrics_total": 3,
        "outputs": output_metadata, "output_bytes": output_bytes,
        "persistent_bytes": tree_bytes(V2_WORK) + tree_bytes(V2_STAGED) + tree_bytes(V3_WORK),
        "group_free_bytes": shutil.disk_usage(GROUP_ROOT).free,
    })
    check_budget()
    log(f"network complete support={support_n:,} eligible_edges={decomposition[7]:,} "
        f"concordant={concordant}/3 output_bytes={output_bytes:,} "
        f"commit={run['git_commit']}")


if __name__ == "__main__":
    main()
