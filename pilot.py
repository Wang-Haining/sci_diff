#!/usr/bin/env python3
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SNAPSHOT_DATE = "2026-06-26"
GROUP_ROOT = Path("/home/group/jasonclark")
SNAPSHOT = GROUP_ROOT / "g91p721/openalex" / SNAPSHOT_DATE / "data/parquet"
WORK = GROUP_ROOT / "g91p721/sci_diff/work"
TMP = GROUP_ROOT / "g91p721/sci_diff/tmp"
RESULTS = Path(__file__).resolve().parent / "results"
WORKS = str(SNAPSHOT / "works/updated_date=*/*.parquet")
MIN_FREE = 1_500_000_000_000
RAW_CAP = 800_000_000_000
WORK_CAP = 200_000_000_000
OUTCOMES = ["total_citations", "within_subfield", "cross_subfield", "cross_field"]
COVARIATES = ["authors_count", "reference_count", "countries_count", "institutions_count", "is_oa"]
CELLS = ["topic_id", "prestige_quintile"]
SEED = 20260902


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def free_bytes():
    return shutil.disk_usage(GROUP_ROOT).free


def work_bytes():
    return sum(p.stat().st_size for p in WORK.glob("*.parquet"))


def check_budget():
    free = free_bytes()
    used = work_bytes()
    if free < MIN_FREE:
        raise RuntimeError(f"expected group free space >= {MIN_FREE}, got {free}")
    if used > WORK_CAP:
        raise RuntimeError(f"expected persistent work <= {WORK_CAP}, got {used}")
    log(f"storage persistent={used:,} bytes free={free:,} bytes")


def validate_snapshot():
    manifest_path = SNAPSHOT / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"expected snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("date") != SNAPSHOT_DATE:
        raise ValueError(f"expected snapshot date {SNAPSHOT_DATE}, got {manifest.get('date')}")
    expected_bytes = manifest["meta"]["content_length"]
    if expected_bytes > RAW_CAP:
        raise ValueError(f"expected raw bytes <= {RAW_CAP}, got {expected_bytes}")
    files = [f for entity in manifest["entities"] for f in entity["files"]]
    actual_bytes = 0
    actual_rows = 0
    for item in files:
        rel = item["url"].split("/data/parquet/", 1)[1]
        path = SNAPSHOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest file {path}")
        size = path.stat().st_size
        if size != item["meta"]["content_length"]:
            raise ValueError(f"expected {path} bytes={item['meta']['content_length']}, got {size}")
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != item["meta"]["record_count"]:
            raise ValueError(f"expected {path} rows={item['meta']['record_count']}, got {rows}")
        actual_bytes += size
        actual_rows += rows
    if actual_bytes != expected_bytes or actual_rows != manifest["meta"]["record_count"]:
        raise ValueError(f"snapshot totals differ: bytes={actual_bytes}, rows={actual_rows}")
    RESULTS.mkdir(exist_ok=True)
    shutil.copy2(manifest_path, RESULTS / "openalex_manifest.json")
    log(f"snapshot verified files={len(files):,} bytes={actual_bytes:,} rows={actual_rows:,}")
    return manifest


def materialize(con, name, sql):
    check_budget()
    path = WORK / name
    if path.exists():
        path.unlink()
    escaped = str(path).replace("'", "''")
    log(f"building {name}")
    con.execute(f"COPY ({sql}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
    if rows <= 0:
        raise ValueError(f"expected {name} rows > 0, got {rows}")
    check_budget()
    log(f"built {name} rows={rows:,} bytes={path.stat().st_size:,}")
    return path, rows


def weighted_stats(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.isnan(values).any() or np.isnan(weights).any() or weights.sum() <= 0:
        raise ValueError("weighted statistic received missing values or non-positive weights")
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return mean, var


def smd_rows(frame, weights, measure, stage):
    rows = []
    treatment = frame["treatment"].to_numpy()
    for covariate in COVARIATES:
        values = frame[covariate].to_numpy(dtype=float)
        m0, v0 = weighted_stats(values[treatment == 0], weights[treatment == 0])
        m1, v1 = weighted_stats(values[treatment == 1], weights[treatment == 1])
        denom = math.sqrt((v0 + v1) / 2)
        rows.append({"measure": measure, "stage": stage, "covariate": covariate,
                     "mean_broad": m0, "mean_specialized": m1,
                     "smd": (m1 - m0) / denom if denom else 0.0})
    return rows


def analyze_measure(df, measure, treatment_col, rng):
    d = df[df[treatment_col].notna()].copy()
    d["treatment"] = d[treatment_col].astype("int8")
    if set(d["treatment"].unique()) != {0, 1}:
        raise ValueError(f"expected {measure} treatments {{0,1}}, got {sorted(d['treatment'].unique())}")
    counts = d.groupby(CELLS + ["treatment"], observed=True).agg(
        papers=("id", "size"), journals=("journal_id", "nunique")).reset_index()
    paper_wide = counts.pivot(index=CELLS, columns="treatment", values="papers").reindex(columns=[0, 1])
    journal_wide = counts.pivot(index=CELLS, columns="treatment", values="journals").reindex(columns=[0, 1])
    valid = paper_wide.index[paper_wide.notna().all(axis=1) & journal_wide.notna().all(axis=1) &
                             (paper_wide[0] >= 20) & (paper_wide[1] >= 20) &
                             (journal_wide[0] >= 2) & (journal_wide[1] >= 2)]
    key_index = pd.MultiIndex.from_frame(d[CELLS])
    kept = d[key_index.isin(valid)].copy()
    if kept.empty:
        raise ValueError(f"expected common-support rows for {measure}, got 0")
    cell_codes, _ = pd.factorize(pd.MultiIndex.from_frame(kept[CELLS]), sort=True)
    kept["cell_code"] = cell_codes
    n_cells = int(cell_codes.max()) + 1
    treatment = kept["treatment"].to_numpy(dtype=int)
    arm_index = cell_codes * 2 + treatment
    arm_n = np.bincount(arm_index, minlength=n_cells * 2).reshape(n_cells, 2)
    cell_weight = 2 * arm_n[:, 0] * arm_n[:, 1] / arm_n.sum(axis=1)
    paper_weight = cell_weight[cell_codes] / arm_n[cell_codes, treatment]
    balance = smd_rows(d, np.ones(len(d)), measure, "raw")
    balance += smd_rows(kept, paper_weight, measure, "standardized")
    group_cols = ["journal_id"] + CELLS + ["treatment"]
    agg = kept.groupby(group_cols, observed=True).agg(
        n=("id", "size"), **{f"sum_{y}": (y, "sum") for y in OUTCOMES}).reset_index()
    journal_codes, journals = pd.factorize(agg["journal_id"], sort=True)
    agg_cell_codes, _ = pd.factorize(pd.MultiIndex.from_frame(agg[CELLS]), sort=True)
    agg_arm_index = agg_cell_codes * 2 + agg["treatment"].to_numpy(dtype=int)
    agg_n = agg["n"].to_numpy(dtype=float)
    sums = agg[[f"sum_{y}" for y in OUTCOMES]].to_numpy(dtype=float)

    def estimates(cluster_multiplicity):
        mult = cluster_multiplicity[journal_codes]
        ns = np.bincount(agg_arm_index, weights=agg_n * mult,
                         minlength=n_cells * 2).reshape(n_cells, 2)
        present = (ns[:, 0] > 0) & (ns[:, 1] > 0)
        weights = cell_weight[present]
        if not present.any() or weights.sum() <= 0:
            return None
        result = []
        for j in range(len(OUTCOMES)):
            ys = np.bincount(agg_arm_index, weights=sums[:, j] * mult,
                             minlength=n_cells * 2).reshape(n_cells, 2)
            means = ys[present] / ns[present]
            result.append((np.average(means[:, 0], weights=weights),
                           np.average(means[:, 1], weights=weights)))
        return np.asarray(result)

    point = estimates(np.ones(len(journals)))
    if point is None:
        raise ValueError(f"expected observed common-support estimates for {measure}, got none")
    boot = np.empty((200, len(OUTCOMES)))
    for b in range(200):
        for _ in range(100):
            sampled = rng.integers(0, len(journals), size=len(journals))
            multiplicity = np.bincount(sampled, minlength=len(journals))
            pair = estimates(multiplicity)
            if pair is not None:
                break
        if pair is None:
            raise RuntimeError(f"failed to draw a valid {measure} cluster bootstrap after 100 attempts")
        boot[b] = pair[:, 1] - pair[:, 0]
    coverage = len(kept) / len(d)
    rows = []
    for j, outcome in enumerate(OUTCOMES):
        raw = d.groupby("treatment")[outcome].mean()
        rows.append({"measure": measure, "estimation": "raw", "outcome": outcome,
                     "n_papers": len(d), "n_journals": d["journal_id"].nunique(),
                     "support_papers": len(kept), "support_journals": kept["journal_id"].nunique(),
                     "coverage": coverage, "mean_broad": raw[0], "mean_specialized": raw[1],
                     "contrast": raw[1] - raw[0], "ci_low": np.nan, "ci_high": np.nan})
        rows.append({"measure": measure, "estimation": "standardized", "outcome": outcome,
                     "n_papers": len(d), "n_journals": d["journal_id"].nunique(),
                     "support_papers": len(kept), "support_journals": kept["journal_id"].nunique(),
                     "coverage": coverage, "mean_broad": point[j, 0], "mean_specialized": point[j, 1],
                     "contrast": point[j, 1] - point[j, 0],
                     "ci_low": np.quantile(boot[:, j], 0.025),
                     "ci_high": np.quantile(boot[:, j], 0.975)})
    return rows, balance


def markdown_table(frame, columns):
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in frame[columns].iterrows():
        vals = [f"{row[c]:.3f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, rule] + body)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    RESULTS.mkdir(exist_ok=True)
    manifest = validate_snapshot()
    check_budget()
    con = duckdb.connect()
    con.execute("SET threads=32")
    con.execute("SET memory_limit='200GB'")
    con.execute(f"SET temp_directory='{TMP}'")
    con.execute("SET max_temp_directory_size='400GB'")
    con.execute("SET preserve_insertion_order=false")
    works = WORKS.replace("'", "''")
    try:
        scope, n_scope = materialize(con, "journal_scope.parquet", f"""
            WITH history AS (
                SELECT primary_location.source.id AS journal_id,
                       primary_location.source.display_name AS journal_name,
                       primary_topic.id AS topic_id,
                       COALESCE((SELECT sum(x.cited_by_count) FROM unnest(counts_by_year) t(x)
                                 WHERE x.year <= 2019), 0)::DOUBLE / (2020-publication_year) AS annual_cites
                FROM read_parquet('{works}')
                WHERE publication_year BETWEEN 2017 AND 2019 AND type='article'
                  AND NOT COALESCE(is_xpac, false) AND NOT COALESCE(is_retracted, false)
                  AND primary_location.is_published AND primary_location.source.type='journal'
                  AND primary_location.source.id IS NOT NULL AND primary_topic.id IS NOT NULL
            ), topic_counts AS (
                SELECT journal_id, topic_id, count(*) AS n_topic FROM history GROUP BY ALL
            ), totals AS (
                SELECT journal_id, any_value(journal_name) AS journal_name, count(*) AS n_history,
                       avg(annual_cites) AS prior_prestige FROM history GROUP BY journal_id HAVING count(*) >= 50
            )
            SELECT t.journal_id, t.journal_name, t.n_history, t.prior_prestige,
                   sum(power(c.n_topic::DOUBLE/t.n_history, 2)) AS hhi,
                   -sum((c.n_topic::DOUBLE/t.n_history)*ln(c.n_topic::DOUBLE/t.n_history)) AS entropy,
                   2017 AS history_start, 2019 AS history_end
            FROM totals t JOIN topic_counts c USING (journal_id)
            GROUP BY ALL
        """)
        focal, n_focal = materialize(con, "focal_2020.parquet", f"""
            WITH base AS (
                SELECT w.id, w.primary_location.source.id AS journal_id,
                       w.primary_location.source.display_name AS journal_name,
                       w.primary_topic.id AS topic_id, w.primary_topic.display_name AS topic_name,
                       w.primary_topic.subfield.id AS subfield_id,
                       w.primary_topic.subfield.display_name AS subfield_name,
                       w.primary_topic.field.id AS field_id, w.primary_topic.field.display_name AS field_name,
                       s.hhi, -s.entropy AS neg_entropy, s.prior_prestige,
                       w.authors_count, w.referenced_works_count AS reference_count,
                       w.countries_distinct_count AS countries_count,
                       w.institutions_distinct_count AS institutions_count,
                       COALESCE(w.open_access.is_oa, false)::INTEGER AS is_oa,
                       COALESCE((SELECT sum(x.cited_by_count) FROM unnest(w.counts_by_year) t(x)
                                 WHERE x.year BETWEEN 2020 AND 2024), 0) AS qc_citations
                FROM read_parquet('{works}') w JOIN read_parquet('{scope}') s
                  ON w.primary_location.source.id=s.journal_id
                WHERE w.publication_year=2020 AND w.type='article'
                  AND NOT COALESCE(w.is_xpac, false) AND NOT COALESCE(w.is_retracted, false)
                  AND w.primary_location.is_published AND w.primary_location.source.type='journal'
                  AND w.primary_topic.id IS NOT NULL
            ), journal_subfield AS (
                SELECT DISTINCT subfield_id, journal_id, hhi, neg_entropy, prior_prestige FROM base
            ), ranked AS (
                SELECT *, ntile(4) OVER (PARTITION BY subfield_id ORDER BY hhi, journal_id) AS hhi_q,
                          ntile(4) OVER (PARTITION BY subfield_id ORDER BY neg_entropy, journal_id) AS entropy_q,
                          ntile(5) OVER (PARTITION BY subfield_id ORDER BY prior_prestige, journal_id) AS prestige_quintile
                FROM journal_subfield
            )
            SELECT b.*, CASE WHEN r.hhi_q=1 THEN 0 WHEN r.hhi_q=4 THEN 1 END AS hhi_treatment,
                   CASE WHEN r.entropy_q=1 THEN 0 WHEN r.entropy_q=4 THEN 1 END AS entropy_treatment,
                   r.prestige_quintile
            FROM base b JOIN ranked r USING (subfield_id, journal_id, hhi, neg_entropy, prior_prestige)
            WHERE r.hhi_q IN (1,4) OR r.entropy_q IN (1,4)
        """)
        edges, n_edges = materialize(con, "citation_edges_2020_2024.parquet", f"""
            SELECT c.id AS citing_id, u.cited_id, c.publication_year AS citing_year,
                   c.primary_topic.subfield.id AS citing_subfield_id,
                   c.primary_topic.field.id AS citing_field_id, count(*) AS multiplicity
            FROM read_parquet('{works}') c CROSS JOIN unnest(c.referenced_works) u(cited_id)
            JOIN read_parquet('{focal}') f ON u.cited_id=f.id
            WHERE c.publication_year BETWEEN 2020 AND 2024 AND c.type='article'
              AND NOT COALESCE(c.is_xpac, false) AND NOT COALESCE(c.is_retracted, false)
              AND c.primary_location.is_published AND c.primary_location.source.type='journal'
            GROUP BY ALL
        """)
        analysis, n_analysis = materialize(con, "analysis_dataset.parquet", f"""
            WITH labeled AS (
                SELECT e.*, f.subfield_id, f.field_id,
                       CASE WHEN e.citing_subfield_id IS NULL OR e.citing_field_id IS NULL THEN 'unclassified'
                            WHEN e.citing_subfield_id=f.subfield_id THEN 'within_subfield'
                            WHEN e.citing_field_id=f.field_id THEN 'cross_subfield'
                            ELSE 'cross_field' END AS citation_class
                FROM read_parquet('{edges}') e JOIN read_parquet('{focal}') f ON e.cited_id=f.id
            )
            SELECT f.*, count(l.citing_id) AS total_citations,
                   count(l.citing_id) FILTER (WHERE l.citation_class='within_subfield') AS within_subfield,
                   count(l.citing_id) FILTER (WHERE l.citation_class='cross_subfield') AS cross_subfield,
                   count(l.citing_id) FILTER (WHERE l.citation_class='cross_field') AS cross_field,
                   count(l.citing_id) FILTER (WHERE l.citation_class='unclassified') AS unclassified
            FROM read_parquet('{focal}') f LEFT JOIN labeled l ON f.id=l.cited_id GROUP BY ALL
        """)
        scope_qc = con.execute("SELECT min(n_history), min(hhi), max(hhi), min(history_start), max(history_end) FROM read_parquet(?)", [str(scope)]).fetchone()
        if scope_qc[0] < 50 or not (0 < scope_qc[1] <= scope_qc[2] <= 1) or scope_qc[3:] != (2017, 2019):
            raise ValueError(f"journal scope QC failed: {scope_qc}")
        edge_qc = con.execute("SELECT min(citing_year), max(citing_year), sum(multiplicity-1) FROM read_parquet(?)", [str(edges)]).fetchone()
        if not (2020 <= edge_qc[0] <= edge_qc[1] <= 2024):
            raise ValueError(f"citation year QC failed: {edge_qc}")
        bad_decomp = con.execute("SELECT count(*) FROM read_parquet(?) WHERE total_citations != within_subfield+cross_subfield+cross_field+unclassified", [str(analysis)]).fetchone()[0]
        if bad_decomp:
            raise ValueError(f"expected zero decomposition failures, got {bad_decomp}")
        df = con.execute("SELECT * FROM read_parquet(?)", [str(analysis)]).df()
        if df[COVARIATES + OUTCOMES].isna().any().any():
            raise ValueError("analysis covariates/outcomes contain missing values")
        scope_df = con.execute("SELECT hhi, -entropy AS neg_entropy FROM read_parquet(?)", [str(scope)]).df()
        scope_corr = scope_df.corr(method="spearman").iloc[0, 1]
        rng = np.random.default_rng(SEED)
        summary_rows, balance_rows = [], []
        for measure, treatment in [("hhi", "hhi_treatment"), ("entropy", "entropy_treatment")]:
            rows, balance = analyze_measure(df, measure, treatment, rng)
            summary_rows += rows
            balance_rows += balance
        summary = pd.DataFrame(summary_rows)
        balance = pd.DataFrame(balance_rows)
        summary.to_csv(RESULTS / "pilot_summary.csv", index=False)
        balance.to_csv(RESULTS / "balance.csv", index=False)
        std = summary[summary.estimation.eq("standardized")]
        cross = std[std.outcome.eq("cross_field")].set_index("measure")
        within = std[std.outcome.eq("within_subfield")].set_index("measure")
        measures = ["hhi", "entropy"]
        go = all(cross.loc[m, "contrast"] < within.loc[m, "contrast"] and
                 cross.loc[m, "ci_high"] < 0 and cross.loc[m, "coverage"] >= 0.5 for m in measures)
        promising = all(cross.loc[m, "contrast"] < 0 for m in measures)
        decision = "GO" if go else "PROMISING BUT NEEDS MORE IDENTIFICATION" if promising else "NO-GO"
        main_cols = ["measure", "estimation", "outcome", "mean_broad", "mean_specialized", "contrast", "ci_low", "ci_high", "coverage"]
        report = [decision, "", "# OpenAlex journal-specialization dirty pilot", "",
                  f"- Snapshot: {manifest['date']} ({manifest['meta']['record_count']:,} records)",
                  f"- Journal scopes: {n_scope:,}; focal papers: {n_focal:,}; citation edges: {n_edges:,}",
                  f"- Analysis papers: {n_analysis:,}; duplicate reference entries removed: {int(edge_qc[2] or 0):,}",
                  f"- HHI vs negative-entropy Spearman correlation: {scope_corr:.3f}", "",
                  "## Complete outcome decomposition", "", markdown_table(summary, main_cols), "",
                  "## Interpretation boundary", "",
                  "This is an exploratory observational pilot, not a causal estimate. OpenAlex topics may use journal information; residual manuscript sorting remains; and citing works are restricted to non-XPAC, non-retracted journal articles from 2020-2024.", ""]
        (RESULTS / "pilot_report.md").write_text("\n".join(report))
        log(f"decision={decision} scope_corr={scope_corr:.3f} duplicate_refs={int(edge_qc[2] or 0):,}")
        log(f"summary scopes={n_scope:,} focal={n_focal:,} edges={n_edges:,} analysis={n_analysis:,}")
    finally:
        con.close()
        if TMP.exists():
            shutil.rmtree(TMP)
    check_budget()


if __name__ == "__main__":
    main()
