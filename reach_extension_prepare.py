#!/usr/bin/env python3
"""Build additive 60-month reach outcomes on the saved downstream support."""
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from qss_common import GROUP_ROOT, QSS_TMP, REPO, SEED, TMP_CAP
from qss_network import load_taxonomy, validate_inputs
from qss_v3_common import (
    ARTIFACTS, RESULTS, V2_STAGED, V2_WORK, V3_WORK, check_budget, connect,
    log, path_glob, tree_bytes, validate_snapshot,
)

WORK = V3_WORK / "reach_extension_v1"
TEMP = QSS_TMP / "reach_extension_v1/prepare"
OUTPUT = RESULTS.parent / "reach_extension_v1"
RUN = ARTIFACTS.parent / "reach_extension_v1/run_prepare.json"
SCORES = V3_WORK / "routing_scores.parquet"
ANALYSIS = V3_WORK / "analysis_dataset.parquet"
CLASSIFIED = WORK / "classified_edges_60.parquet"
OUTCOMES = WORK / "outcomes_60.parquet"
COVERAGE = OUTPUT / "w1_coverage_60.csv"
CORRELATIONS = OUTPUT / "w3_scope_correlations.csv"
LABELS = RESULTS / "hierarchy_areas.csv"
VARIANTS = ("all32", "named31")
KS = (3, 5, 10)


def require_new(path):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite extension artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rarefaction_term(k):
    # One area's chance of appearing in k draws without replacement.
    absent = " * ".join(f"((total-n-{j})::DOUBLE/(total-{j}))" for j in range(k))
    return f"CASE WHEN total<{k} THEN NULL WHEN total-n<{k} THEN 1.0 ELSE 1.0-({absent}) END"


def portfolio_sql(table):
    rare = ", ".join(f"sum({rarefaction_term(k)}) AS rarefied_macros_k{k}" for k in KS)
    return f"""WITH p AS (
      SELECT *,sum(n) OVER (PARTITION BY focal_id) AS total FROM {table}
    ) SELECT focal_id,sum(-(n::DOUBLE/total)*ln(n::DOUBLE/total)) AS macro_entropy,
      {rare} FROM p GROUP BY focal_id"""


def write_parquet(con, path, query):
    check_storage()
    require_new(path)
    con.execute(f"COPY ({query}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()[0]
    if n <= 0:
        raise ValueError(f"expected nonempty {path}, got {n} rows")
    check_storage()
    log(f"wrote {path.name}: rows={n:,} bytes={path.stat().st_size:,}")
    return n


def check_storage():
    check_budget()
    spill = tree_bytes(QSS_TMP) - tree_bytes(V2_STAGED)
    if spill > TMP_CAP:
        raise RuntimeError(f"expected all QSS spill <= {TMP_CAP}, got {spill}")
    return spill


def scope_correlations(con):
    scope = con.execute(f"""SELECT journal_id,focal_year,history_n,prior_prestige,semantic_title_similarity
      FROM read_parquet('{V2_WORK / 'journal_year_scope.parquet'}')
      WHERE focal_year BETWEEN 2015 AND 2020 AND history_n>=100
        AND semantic_title_similarity IS NOT NULL""").df()
    if scope.empty or scope.duplicated(["journal_id", "focal_year"]).any() \
            or not np.isfinite(scope[["history_n", "prior_prestige", "semantic_title_similarity"]]).all().all():
        raise ValueError(f"invalid scored journal-year correlation inputs: rows={len(scope)}")
    rows = []
    for year in ("all", *range(2015, 2021)):
        d = scope if year == "all" else scope[scope.focal_year.eq(year)]
        for feature in ("history_n", "prior_prestige"):
            rho = d.semantic_title_similarity.corr(d[feature], method="spearman")
            if not np.isfinite(rho):
                raise ValueError(f"undefined scope correlation year={year} feature={feature} rows={len(d)}")
            rows.append({"publication_year": year, "feature": feature, "rho": float(rho),
                         "journal_years": len(d), "journals": d.journal_id.nunique(),
                         "unit": "unique journal-year", "weighting": "unweighted", "minimum_history_n": 100})
    require_new(CORRELATIONS)
    result = pd.DataFrame(rows)
    result.to_csv(CORRELATIONS, index=False)
    log(f"scope correlations complete: scored_journal_years={len(scope):,} journals={scope.journal_id.nunique():,}")
    print(result.to_string(index=False), flush=True)
    return result


def main():
    for path in (CLASSIFIED, OUTCOMES, COVERAGE, CORRELATIONS, RUN):
        if path.exists():
            raise FileExistsError(f"extension output already exists: {path}")
    validate_snapshot()
    check_storage()
    labels = pd.read_csv(LABELS)
    mixed = labels.loc[labels.display_label.eq("Mixed records"), "qwen_macro"]
    if len(mixed) != 1 or labels.qwen_macro.nunique() != 32:
        raise ValueError(f"expected one Mixed records label among 32 areas, got {mixed.tolist()}")
    mixed_id = int(mixed.iloc[0])
    leaf_to_macro, _, distances, expected_qwen = load_taxonomy()
    con = connect()
    TEMP.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{TEMP}'")
    con.execute("SET max_temp_directory_size='180GB'")
    correlations = scope_correlations(con)
    support_n, journals, raw_edges = validate_inputs(con, leaf_to_macro, expected_qwen)
    if (support_n, journals) != (3_827_491, 20_215):
        raise ValueError(f"fixed downstream support changed: papers={support_n}, journals={journals}")
    con.execute(f"""CREATE TEMP TABLE external_edges AS
      SELECT s.id AS focal_id,e.citing_id,s.journal_id,s.treatment,
        s.source_leaf,s.source_macro,qc.qwen_leaf AS target_leaf,qc.qwen_macro AS target_macro,
        s.focal_date,e.citing_date,
        COALESCE(c.language='en',false) AS english,
        qc.id IS NOT NULL AS label_present,COALESCE(qc.qwen_ood,false) AS target_ood,
        COALESCE(c.language='en' AND qc.id IS NOT NULL AND NOT qc.qwen_ood
          AND qc.qwen_macro=m.frozen_macro,false) AS classified
      FROM read_parquet('{path_glob(V3_WORK / 'citation_edges')}') e
      JOIN network_support s ON e.cited_id=s.id
      JOIN read_parquet('{path_glob(V3_WORK / 'citing_metadata')}') c ON e.citing_id=c.id
      LEFT JOIN qwen_all qc ON c.id=qc.id
      LEFT JOIN leaf_macro m ON qc.qwen_leaf=m.qwen_leaf
      WHERE (c.journal_id=s.journal_id) IS FALSE
        AND NOT COALESCE(list_has_any(c.author_ids,s.author_ids),false)
    """)
    edge_qc = con.execute("""SELECT count(*),count(DISTINCT(focal_id,citing_id)),
      count(*) FILTER (WHERE citing_date<focal_date OR citing_date>=focal_date+INTERVAL 60 MONTH)
      FROM external_edges""").fetchone()
    if edge_qc[0] <= 0 or edge_qc[0] != edge_qc[1] or edge_qc[2] != 0:
        raise ValueError(f"external-edge uniqueness/window check failed: {edge_qc}")
    first_stage = con.execute(f"""SELECT treatment,count(*) AS external_citations,
      count(*) FILTER (WHERE NOT english) AS nonenglish_citations,
      count(*) FILTER (WHERE english AND NOT label_present) AS english_missing_label_citations,
      count(*) FILTER (WHERE english AND label_present AND target_ood) AS english_ood_citations,
      count(*) FILTER (WHERE classified AND target_macro={mixed_id}) AS classified_mixed_origin_citations
      FROM external_edges GROUP BY treatment ORDER BY treatment""").df()
    con.register("leaf_distances", pd.DataFrame({
        "source_leaf": np.repeat(np.arange(1000), 1000),
        "target_leaf": np.tile(np.arange(1000), 1000),
        "semantic_distance": distances.astype(np.float32).ravel(),
    }))
    con.execute(f"""CREATE TEMP TABLE classified_distances AS
      SELECT e.*,d.semantic_distance,e.target_macro<>{mixed_id} AS named31_eligible
      FROM external_edges e JOIN leaf_distances d USING (source_leaf,target_leaf)
      WHERE e.classified""")
    cuts = con.execute("""SELECT quantile_cont(semantic_distance,[0.25,0.5,0.75])
      FROM classified_distances WHERE source_leaf<>target_leaf""").fetchone()[0]
    cuts = [float(x) for x in cuts]
    if not np.isfinite(cuts).all() or not np.all(np.diff(cuts) > 0):
        raise ValueError(f"expected three finite increasing distance cuts, got {cuts}")
    bin_sql = "CASE WHEN source_leaf=target_leaf THEN 0 ELSE 1 + " + " + ".join(
        f"(semantic_distance>={cut!r})::INTEGER" for cut in cuts) + " END"
    classified_n = write_parquet(con, CLASSIFIED,
        f"SELECT *,({bin_sql})::UTINYINT AS distance_bin FROM classified_distances")
    con.execute(f"CREATE TEMP VIEW classified_edges AS SELECT * FROM read_parquet('{CLASSIFIED}')")
    classified_qc = con.execute("""SELECT count(DISTINCT(focal_id,citing_id)),
      count(*) FILTER (WHERE distance_bin NOT BETWEEN 0 AND 4
        OR (distance_bin=0)<>(source_leaf=target_leaf)
        OR NOT isfinite(semantic_distance)) FROM classified_edges""").fetchone()
    if classified_qc != (classified_n, 0):
        raise ValueError(f"classified edge/distance check failed: {classified_qc}")
    con.execute("CREATE TEMP TABLE totals AS SELECT focal_id,count(*) AS total_citations "
                "FROM external_edges GROUP BY focal_id")
    selections = ["s.id", "s.journal_id", "c.publication_year", "s.treatment", "s.propensity",
                  "s.source_macro", "s.source_leaf", "COALESCE(t.total_citations,0) AS total_citations"]
    joins = []
    count_names = ("classified_citations", "near", "intermediate", "far", "n_macros_cited",
                   "n_macros_other", "n_leaves_cited") + tuple(f"distance_bin{k}" for k in range(5))
    for variant in VARIANTS:
        where = "true" if variant == "all32" else "named31_eligible"
        bins = ", ".join(f"count(*) FILTER (WHERE distance_bin={k}) AS distance_bin{k}" for k in range(5))
        con.execute(f"""CREATE TEMP TABLE counts_{variant} AS SELECT focal_id,
          count(*) AS classified_citations,
          count(*) FILTER (WHERE source_leaf=target_leaf) AS near,
          count(*) FILTER (WHERE source_leaf<>target_leaf AND source_macro=target_macro) AS intermediate,
          count(*) FILTER (WHERE source_macro<>target_macro) AS far,
          count(DISTINCT target_macro) AS n_macros_cited,
          count(DISTINCT target_macro) FILTER (WHERE source_macro<>target_macro) AS n_macros_other,
          count(DISTINCT target_leaf) AS n_leaves_cited,{bins}
          FROM classified_edges WHERE {where} GROUP BY focal_id""")
        con.execute(f"""CREATE TEMP TABLE macro_counts_{variant} AS
          SELECT focal_id,target_macro,count(*) AS n FROM classified_edges
          WHERE {where} GROUP BY focal_id,target_macro""")
        con.execute(f"CREATE TEMP TABLE portfolio_{variant} AS " + portfolio_sql(f"macro_counts_{variant}"))
        selections += [f"COALESCE(g_{variant}.{name},0) AS {name}_{variant}" for name in count_names]
        selections += [
            f"COALESCE(t.total_citations,0)-COALESCE(g_{variant}.classified_citations,0) AS unclassified_{variant}",
            f"(COALESCE(g_{variant}.far,0)>0)::UTINYINT AS any_far_{variant}",
            f"p_{variant}.macro_entropy AS macro_entropy_{variant}",
        ] + [f"p_{variant}.rarefied_macros_k{k} AS rarefied_macros_k{k}_{variant}" for k in KS]
        joins += [f"LEFT JOIN counts_{variant} g_{variant} ON s.id=g_{variant}.focal_id",
                  f"LEFT JOIN portfolio_{variant} p_{variant} ON s.id=p_{variant}.focal_id"]
    outcome_n = write_parquet(con, OUTCOMES, f"""SELECT {', '.join(selections)} FROM network_support s
      JOIN read_parquet('{V3_WORK / 'candidate_focal.parquet'}') c ON s.id=c.id
      LEFT JOIN totals t ON s.id=t.focal_id {' '.join(joins)}""")
    if outcome_n != support_n:
        raise ValueError(f"expected {support_n} outcome rows including uncited papers, got {outcome_n}")
    con.execute(f"CREATE TEMP VIEW outcomes AS SELECT * FROM read_parquet('{OUTCOMES}')")
    matched, bad = con.execute(f"""SELECT count(*),count(*) FILTER (WHERE
        o.total_citations<>a.total_citations OR o.near_all32<>a.near
        OR o.intermediate_all32<>a.intermediate OR o.far_all32<>a.far
        OR o.unclassified_all32<>a.unclassified OR o.any_far_all32<>a.any_far)
      FROM outcomes o JOIN read_parquet('{ANALYSIS}') a USING (id)""").fetchone()
    if matched != support_n or bad or con.execute("SELECT count(DISTINCT id) FROM outcomes").fetchone()[0] != support_n:
        raise ValueError(f"raw-outcome/unique-ID reconciliation failed: matched={matched}, changed={bad}")
    coverage = []
    for variant in VARIANTS:
        errors = [f"total_citations<>near_{variant}+intermediate_{variant}+far_{variant}+unclassified_{variant}",
                  f"classified_citations_{variant}<>" + "+".join(f"distance_bin{k}_{variant}" for k in range(5)),
                  f"(macro_entropy_{variant} IS NULL)<>(classified_citations_{variant}=0)",
                  f"NOT isfinite(macro_entropy_{variant})",
                  f"macro_entropy_{variant} < -1e-10 OR macro_entropy_{variant}>ln(32)+1e-10",
                  f"n_macros_cited_{variant}>classified_citations_{variant}",
                  f"n_macros_other_{variant}>n_macros_cited_{variant}",
                  f"n_leaves_cited_{variant}>classified_citations_{variant}"]
        for k in KS:
            errors += [f"(rarefied_macros_k{k}_{variant} IS NULL)<>(classified_citations_{variant}<{k})",
                       f"NOT isfinite(rarefied_macros_k{k}_{variant})",
                       f"rarefied_macros_k{k}_{variant}<1-1e-8 OR rarefied_macros_k{k}_{variant}>{k}+1e-8"]
        failures = con.execute("SELECT count(*) FROM outcomes WHERE " + " OR ".join(f"({x})" for x in errors)).fetchone()[0]
        if failures:
            raise ValueError(f"{variant} decomposition/conditional-outcome checks failed: rows={failures}")
        eligible = ", ".join(f"count(rarefied_macros_k{k}_{variant}) AS rarefied_k{k}_eligible_n" for k in KS)
        frame = con.execute(f"""SELECT treatment,count(*) AS papers,count(DISTINCT journal_id) AS journals,
          sum(total_citations) AS total_citations,sum(classified_citations_{variant}) AS classified_citations,
          count(*) FILTER (WHERE total_citations=0) AS uncited_papers,
          count(*) FILTER (WHERE classified_citations_{variant}=0) AS no_classified_citations_papers,
          count(macro_entropy_{variant}) AS entropy_eligible_n,{eligible}
          FROM outcomes GROUP BY treatment ORDER BY treatment""").df()
        frame.insert(0, "variant", variant)
        coverage.append(frame)
    coverage = pd.concat(coverage, ignore_index=True)
    coverage = coverage.merge(first_stage, on="treatment", validate="many_to_one")
    if not coverage.total_citations.eq(coverage.external_citations).all():
        raise ValueError("per-arm external citation accounting changed during outcome aggregation")
    coverage["classification_coverage"] = coverage.classified_citations / coverage.total_citations
    for name in ("entropy", *(f"rarefied_k{k}" for k in KS)):
        coverage[f"{name}_eligible_fraction"] = coverage[f"{name}_eligible_n"] / coverage.papers
    for name in ("nonenglish", "english_missing_label", "english_ood", "classified_mixed_origin"):
        coverage[f"{name}_fraction_of_external_citations"] = coverage[f"{name}_citations"] / coverage.external_citations
    require_new(COVERAGE)
    coverage.to_csv(COVERAGE, index=False)
    spill_bytes = check_storage()
    run = {
        "design": "reach_extension_v1", "stage": "prepare", "status": "complete",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "snapshot_date": "2026-06-26", "support_source": str(SCORES), "support_sha256": digest(SCORES),
        "support_n": support_n, "journals": journals, "raw_edges": raw_edges,
        "external_edges": edge_qc[0], "classified_edges_all32": classified_n,
        "outcome_rows": outcome_n, "mixed_id": mixed_id, "labels_sha256": digest(LABELS),
        "window": "[publication_date, publication_date + 60 months)",
        "distance_quartile_cuts": cuts, "distance_cut_population": "all32 classified non-same-leaf edges",
        "distance_cut_weighting": "unweighted pooled edges; same cuts in both variants",
        "coverage_rates_denominator": "external citations after same-journal/shared-author exclusion; eligibility fractions use all fixed-support papers in the arm",
        "named31_definition": "retain all focal papers; Mixed citing origins become unclassified; total unchanged",
        "entropy_population": "conditional on >=1 classified citation; otherwise NULL",
        "rarefaction_population": "conditional on >=k classified citations; otherwise NULL; not a causal selected-population effect",
        "packages": {name: importlib.metadata.version(name) for name in ("duckdb", "numpy", "pandas", "scipy")},
        "input_manifests_sha256": {name: digest(ARTIFACTS / name) for name in ("run_prepare.json", "run_embed.json", "run_downstream.json")},
        "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in (OUTCOMES, CLASSIFIED, COVERAGE, CORRELATIONS)},
        "persistent_bytes": tree_bytes(V2_WORK) + tree_bytes(V3_WORK) + tree_bytes(V2_STAGED),
        "spill_bytes": spill_bytes, "temp_directory": str(TEMP), "stage_spill_limit": "180GB",
        "free_bytes": shutil.disk_usage(GROUP_ROOT).free,
    }
    require_new(RUN)
    RUN.write_text(json.dumps(run, indent=2) + "\n")
    log(f"extension prepare complete: papers={support_n:,} journals={journals:,} classified_edges={classified_n:,} "
        f"quartile_cuts={cuts} persistent_bytes={run['persistent_bytes']:,} free_bytes={run['free_bytes']:,}")
    print(coverage.to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
