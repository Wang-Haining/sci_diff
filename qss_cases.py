#!/usr/bin/env python3
from pathlib import Path

from qss_common import file_sha256, reset_output
from qss_v3_common import RESULTS, V3_WORK, check_budget, connect, log, validate_snapshot, write_run

ANALYSIS = V3_WORK / "analysis_dataset.parquet"
SCORES = V3_WORK / "routing_scores.parquet"
LABELS = RESULTS / "macro_labels.csv"
SUBGROUPS = RESULTS / "subgroup_estimates.csv"
SELECTION = RESULTS / "case_selection.csv"
CASES = RESULTS / "journal_corridors.csv"

BASE = f"""
SELECT s.id,s.qwen_macro,s.journal_id,a.journal_name,s.publication_year,
       s.choice_set_id,s.treatment,a.semantic_title_similarity,a.prior_prestige
FROM read_parquet('{SCORES}') s JOIN read_parquet('{ANALYSIS}') a USING(id)
JOIN read_csv_auto('{LABELS}') l USING(qwen_macro)
WHERE l.display_label<>'Editorial & miscellaneous records'
"""

SELECT_CASES = f"""
WITH base AS ({BASE}),
areas AS (
 SELECT qwen_macro,count(*) area_n,count(*) FILTER(WHERE treatment=0) area_broad_n,
  count(*) FILTER(WHERE treatment=1) area_narrow_n,
  count(DISTINCT journal_id) FILTER(WHERE treatment=0) area_broad_journals,
  count(DISTINCT journal_id) FILTER(WHERE treatment=1) area_narrow_journals,
  count(DISTINCT publication_year) area_years,
  stddev_pop(ln(1+prior_prestige)) area_prestige_sd
 FROM base GROUP BY qwen_macro
), journal0 AS (
 SELECT qwen_macro,journal_id,any_value(journal_name) journal_name,count(*) n,
  count(DISTINCT publication_year) publication_years,avg(treatment) treatment_rate,
  avg(semantic_title_similarity) scope_score,avg(ln(1+prior_prestige)) log_prestige
 FROM base GROUP BY qwen_macro,journal_id
), journals AS (
 SELECT *,CASE WHEN treatment_rate<=0.1 THEN 0 WHEN treatment_rate>=0.9 THEN 1 END arm
 FROM journal0 WHERE n>=500 AND publication_years>=3
  AND (treatment_rate<=0.1 OR treatment_rate>=0.9)
), eligible AS (
 SELECT * EXCLUDE(arm_rank) FROM (
  SELECT *,row_number() OVER(PARTITION BY qwen_macro,arm ORDER BY n DESC,journal_id) arm_rank
  FROM journals
 ) WHERE arm_rank<=25
), cells AS (
 SELECT b.qwen_macro,b.journal_id,e.arm,b.choice_set_id,count(*) cell_n
 FROM base b JOIN eligible e USING(qwen_macro,journal_id)
 GROUP BY b.qwen_macro,b.journal_id,e.arm,b.choice_set_id
), shared AS (
 SELECT b.qwen_macro,b.journal_id broad_id,n.journal_id narrow_id,
  sum(least(b.cell_n,n.cell_n)) shared_n
 FROM cells b JOIN cells n ON b.qwen_macro=n.qwen_macro
  AND b.choice_set_id=n.choice_set_id AND b.arm=0 AND n.arm=1
 GROUP BY b.qwen_macro,b.journal_id,n.journal_id
), pairs AS (
 SELECT x.*,2.0*x.shared_n/(b.n+n.n) overlap,b.n broad_n,n.n narrow_n,
  b.publication_years broad_years,n.publication_years narrow_years,
  b.treatment_rate broad_treatment_rate,n.treatment_rate narrow_treatment_rate,
  b.scope_score broad_scope,n.scope_score narrow_scope,
  b.log_prestige broad_log_prestige,n.log_prestige narrow_log_prestige,
  b.journal_name broad_name,n.journal_name narrow_name,
  abs(b.log_prestige-n.log_prestige)/nullif(a.area_prestige_sd,0) prestige_gap,
  a.* EXCLUDE(qwen_macro)
 FROM shared x JOIN eligible b ON x.qwen_macro=b.qwen_macro AND x.broad_id=b.journal_id
 JOIN eligible n ON x.qwen_macro=n.qwen_macro AND x.narrow_id=n.journal_id
 JOIN areas a ON x.qwen_macro=a.qwen_macro WHERE x.shared_n>=250
), ranked AS (
 SELECT *,row_number() OVER(PARTITION BY qwen_macro ORDER BY overlap DESC,shared_n DESC,
  prestige_gap,broad_id,narrow_id) pair_rank
 FROM pairs WHERE overlap>=0.20 AND prestige_gap<=0.50 AND area_broad_n>=5000
  AND area_narrow_n>=5000 AND area_broad_journals>=10 AND area_narrow_journals>=10
  AND area_years=6
), chosen AS (
 SELECT * FROM ranked WHERE pair_rank=1
), final AS (
 SELECT *,row_number() OVER(ORDER BY area_n DESC,qwen_macro) case_rank FROM chosen
)
SELECT f.case_rank,l.display_label,f.* EXCLUDE(case_rank,pair_rank)
FROM final f JOIN read_csv_auto('{LABELS}') l USING(qwen_macro)
WHERE f.case_rank<=8 ORDER BY f.case_rank
"""

JOIN_OUTCOMES = f"""
WITH selected AS (SELECT * FROM read_csv_auto('{SELECTION}')),
means AS (
 SELECT s.qwen_macro,count(*) score_n,count(DISTINCT s.journal_id) score_journals,
  avg(psi_near_0) near_broad,avg(psi_near_1) near_narrow,
  avg(psi_far_0) far_broad,avg(psi_far_1) far_narrow
 FROM read_parquet('{SCORES}') s JOIN selected c USING(qwen_macro) GROUP BY s.qwen_macro
), subgroup AS (
 SELECT CAST(level AS INTEGER) qwen_macro,estimate theta,ci_low theta_ci_low,
  ci_high theta_ci_high,far_near_broad,far_near_specialized,n subgroup_n,journals
 FROM read_csv_auto('{SUBGROUPS}') WHERE test='semantic_domain' AND status='estimated'
)
SELECT c.*,m.score_n,m.score_journals,100*m.near_broad near_per_100_broad,
 100*m.near_narrow near_per_100_narrow,100*m.far_broad other_area_per_100_broad,
 100*m.far_narrow other_area_per_100_narrow,
 100*(m.far_narrow-m.far_broad) other_area_difference_per_100,
 g.subgroup_n,g.journals,g.far_near_broad,g.far_near_specialized,
 g.theta,g.theta_ci_low,g.theta_ci_high,
 100*(exp(g.theta)-1) routing_change_percent,
 100*(exp(g.theta_ci_low)-1) routing_ci_low_percent,
 100*(exp(g.theta_ci_high)-1) routing_ci_high_percent
FROM selected c JOIN means m USING(qwen_macro) JOIN subgroup g USING(qwen_macro)
ORDER BY c.case_rank
"""


def copy_csv(con, path: Path, query: str):
    reset_output(path)
    con.execute(f"COPY ({query}) TO '{path}' (HEADER, DELIMITER ',')")


def main():
    validate_snapshot()
    check_budget()
    for path in (ANALYSIS, SCORES, LABELS, SUBGROUPS):
        if not path.is_file():
            raise FileNotFoundError(f"expected input at {path}")
    if any(word in SELECT_CASES.lower() for word in ("psi_", "total_citations", "any_far")):
        raise ValueError("case selection query contains an outcome column")
    con = connect("200GB", 32)
    mismatch = con.execute(f"""SELECT count(*) FROM read_parquet('{SCORES}') s
        JOIN read_parquet('{ANALYSIS}') a USING(id) WHERE s.journal_id<>a.journal_id
        OR s.publication_year<>a.publication_year OR s.treatment<>a.treatment""").fetchone()[0]
    if mismatch:
        raise ValueError(f"expected score/baseline agreement, got mismatches={mismatch}")
    copy_csv(con, SELECTION, SELECT_CASES)
    qc = con.execute(f"SELECT count(*),count(DISTINCT qwen_macro),min(shared_n),min(overlap),"
                     f"max(prestige_gap),max(broad_treatment_rate),min(narrow_treatment_rate) "
                     f"FROM read_csv_auto('{SELECTION}')").fetchone()
    if qc[0:2] != (8, 8) or qc[2] < 250 or qc[3] < 0.20 or qc[4] > 0.50 \
            or qc[5] > 0.10 or qc[6] < 0.90:
        raise ValueError(f"expected eight unique eligible cases, got {qc}")
    selection_sha256 = file_sha256(SELECTION)
    copy_csv(con, CASES, JOIN_OUTCOMES)
    out = con.execute(f"SELECT count(*),count(DISTINCT qwen_macro),min(score_n),"
                      f"max(abs(score_n-subgroup_n)) "
                      f"FROM read_csv_auto('{CASES}')").fetchone()
    if out[0:2] != (8, 8) or out[2] <= 0 or out[3] != 0:
        raise ValueError(f"case outcome join failed: {out}")
    run = write_run("cases", {"selected_cases": qc[0], "case_rows": out[0]},
                    {"selection_sha256": selection_sha256,
                     "selection_used_pre_outcome_columns_only": True})
    check_budget()
    log(f"cases complete rows={out[0]} selection_sha256={selection_sha256} "
        f"commit={run['git_commit']}")


if __name__ == "__main__":
    main()
