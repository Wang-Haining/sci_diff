#!/usr/bin/env python3
import math
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

from qss_common import GROUP_ROOT, SEED, SNAPSHOT
from qss_v3_analyze import BASE_NUMERIC, CATEGORICAL, HEAVY, fit_propensity, fold_features
from qss_v3_common import (
    RESULTS, V2_WORK, V3_WORK, check_budget, connect, copy_query,
    log, tree_bytes, validate_snapshot, write_run,
)

ANALYSIS = V3_WORK / "analysis_dataset.parquet"
NEWS_DATA = GROUP_ROOT / "g91p721/sciscinet/v2_linkage_probe/sciscinet_newsfeed_metadata.parquet"
SCISCI_PAPERS = GROUP_ROOT / "g91p721/sciscinet/v2_linkage_probe/sciscinet_papers.parquet"
NEWS_ANALYSIS = V3_WORK / "news_analysis_dataset.parquet"
WORKS = SNAPSHOT / "works/**/*.parquet"
YEARS = (2018, 2019)
OUTCOMES = ("any_web_5cy", "web_pages_5cy", "web_pages_winsorized")


def build_news_analysis(con):
    for path in (NEWS_DATA, SCISCI_PAPERS):
        if not path.is_file():
            raise FileNotFoundError(f"expected SciSciNet input at {path}")
    paper_qc = con.execute(f"""
      SELECT count(*),count(DISTINCT paperid) FROM read_parquet('{SCISCI_PAPERS}')
    """).fetchone()
    if paper_qc != (249_803_279, 249_803_279):
        raise ValueError(f"unexpected SciSciNet paper universe: {paper_qc}")
    news_qc = con.execute(f"""
      SELECT count(*) AS rows,count(try_cast(timestamp AS TIMESTAMPTZ)) AS dated,
             count(DISTINCT (paperid,subj_id)) AS paper_pages,
             count(DISTINCT newsfeed_id) AS newsfeed_ids,
             count(*) FILTER (WHERE subj_id IS NULL OR trim(subj_id)='') AS missing_urls,
             CAST(min(try_cast(timestamp AS TIMESTAMPTZ)) AS VARCHAR),
             CAST(max(try_cast(timestamp AS TIMESTAMPTZ)) AS VARCHAR)
      FROM read_parquet('{NEWS_DATA}')
    """).fetchone()
    if news_qc[0] != news_qc[1] or news_qc[0] <= news_qc[2] or news_qc[4] != 0 \
            or news_qc[5][:10] != "2017-04-05" or news_qc[6][:10] != "2025-02-14":
        raise ValueError(f"unexpected SciSciNet news coverage: {news_qc}")

    rows = copy_query(con, NEWS_ANALYSIS, f"""
      WITH focal0 AS (
        SELECT a.*,regexp_extract(a.id,'W[0-9]+$') AS paperid,w.doi,
               lower(regexp_replace(trim(w.doi),
                 '^(https?://(dx\\.)?doi\\.org/|doi:[ ]*)','')) AS doi_norm
        FROM read_parquet('{ANALYSIS}') a
        LEFT JOIN (
          SELECT id,doi FROM read_parquet('{WORKS}',union_by_name=true)
          WHERE publication_year BETWEEN 2018 AND 2019
        ) w USING (id)
        WHERE a.publication_year BETWEEN 2018 AND 2019
      ), focal AS (
        SELECT *,count(*) OVER (PARTITION BY doi_norm) AS focal_doi_n FROM focal0
      ), sci AS (
        SELECT paperid,doi,lower(regexp_replace(trim(doi),
          '^(https?://(dx\\.)?doi\\.org/|doi:[ ]*)','')) AS doi_norm
        FROM read_parquet('{SCISCI_PAPERS}')
      ), focal_dois AS (
        SELECT DISTINCT doi_norm FROM focal WHERE doi_norm IS NOT NULL AND doi_norm<>''
      ), sci_doi AS (
        SELECT s.doi_norm,min(s.paperid) AS paperid,count(DISTINCT s.paperid) AS paper_n
        FROM sci s JOIN focal_dois f USING (doi_norm) GROUP BY s.doi_norm
      ), mapped AS (
        SELECT f.*,d.doi_norm AS direct_scisci_doi_norm,
          CASE WHEN d.paperid IS NOT NULL AND f.doi_norm IS NOT NULL AND f.doi_norm<>''
                 THEN d.paperid
               WHEN f.doi_norm IS NOT NULL AND f.doi_norm<>'' AND f.focal_doi_n=1
                    AND u.paper_n=1 THEN u.paperid END AS scisci_paperid,
          CASE WHEN d.paperid IS NOT NULL AND f.doi_norm IS NOT NULL AND f.doi_norm<>''
                 THEN 'direct_id'
               WHEN f.doi_norm IS NULL OR f.doi_norm='' THEN 'missing_openalex_doi'
               WHEN f.focal_doi_n<>1 THEN 'ambiguous_openalex_doi'
               WHEN u.paper_n>1 THEN 'ambiguous_scisci_doi'
               WHEN u.paper_n=1 THEN 'doi_rescue' ELSE 'unmatched' END AS link_method
        FROM focal f LEFT JOIN sci d ON f.paperid=d.paperid
        LEFT JOIN sci_doi u ON f.doi_norm=u.doi_norm
      ), pages AS (
        SELECT n.paperid,n.subj_id,
               min(try_cast(n.timestamp AS TIMESTAMPTZ)) AS first_seen,
               lower(regexp_replace(regexp_extract(n.subj_id,'^https?://([^/]+)',1),
                                    '^www\\.','')) AS host
        FROM read_parquet('{NEWS_DATA}') n
        GROUP BY n.paperid,n.subj_id
      ), linked AS (
        SELECT f.id,p.subj_id,p.host
        FROM mapped f JOIN pages p ON f.scisci_paperid=p.paperid
        WHERE f.scisci_paperid IS NOT NULL
          AND p.first_seen>=make_date(f.publication_year,1,1)
          AND p.first_seen<make_date(f.publication_year+5,1,1)
      ), totals AS (
        SELECT id,count(*) AS web_pages_5cy FROM linked GROUP BY id
      )
      SELECT f.*,COALESCE(t.web_pages_5cy,0) AS web_pages_5cy,
             (COALESCE(t.web_pages_5cy,0)>0)::UTINYINT AS any_web_5cy
      FROM mapped f LEFT JOIN totals t USING (id)
    """)
    duplicate_ids = con.execute(
        "SELECT count(*)-count(DISTINCT id) FROM read_parquet(?)", [str(NEWS_ANALYSIS)],
    ).fetchone()[0]
    if duplicate_ids or rows <= 0:
        raise ValueError(f"news analysis IDs failed: rows={rows} duplicates={duplicate_ids}")
    duplicate_links = con.execute(
        "SELECT count(scisci_paperid)-count(DISTINCT scisci_paperid) "
        "FROM read_parquet(?)", [str(NEWS_ANALYSIS)],
    ).fetchone()[0]
    if duplicate_links:
        raise ValueError(f"canonical SciSciNet paper IDs are not unique: duplicates={duplicate_links}")
    coverage = con.execute("""
      SELECT publication_year AS period,treatment,count(*) AS eligible,
             count(doi_norm) FILTER (WHERE doi_norm<>'') AS doi_n,
             count(*) FILTER (WHERE link_method='direct_id') AS direct_n,
             count(*) FILTER (WHERE link_method='direct_id' AND
               (direct_scisci_doi_norm IS NULL OR direct_scisci_doi_norm=''))
               AS direct_missing_scisci_doi_n,
             count(*) FILTER (WHERE link_method='direct_id' AND doi_norm<>direct_scisci_doi_norm)
               AS direct_doi_mismatch_n,
             count(*) FILTER (WHERE link_method='doi_rescue') AS rescued_n,
             count(*) FILTER (WHERE link_method='missing_openalex_doi') AS missing_doi_n,
             count(*) FILTER (WHERE link_method LIKE 'ambiguous%') AS ambiguous_n,
             count(*) FILTER (WHERE link_method='unmatched') AS unmatched_n,
             count(scisci_paperid) AS linked_n,
             sum(web_pages_5cy) AS pages,count(*) FILTER (WHERE any_web_5cy=1) AS mentioned
      FROM read_parquet(?) GROUP BY publication_year,treatment
      ORDER BY publication_year,treatment
    """, [str(NEWS_ANALYSIS)]).fetchdf()
    if len(coverage) != 4 or set(coverage.treatment) != {0, 1} or (coverage.doi_n <= 0).any():
        raise ValueError(f"invalid DOI coverage: {coverage.to_dict('records')}")
    count_columns = [
        "eligible", "doi_n", "direct_n", "direct_missing_scisci_doi_n",
        "direct_doi_mismatch_n", "rescued_n", "missing_doi_n", "ambiguous_n",
        "unmatched_n", "linked_n", "pages", "mentioned",
    ]
    overall = coverage.groupby("treatment", as_index=False)[count_columns].sum()
    overall.insert(0, "period", "all")
    coverage["period"] = coverage.period.astype(str)
    coverage = pd.concat([overall, coverage], ignore_index=True)
    coverage["doi_share"] = coverage.doi_n / coverage.eligible
    coverage["linkage_share_among_doi"] = coverage.linked_n / coverage.doi_n
    top_hosts = con.execute(f"""
      WITH focal AS (
        SELECT scisci_paperid paperid,publication_date,publication_year
        FROM read_parquet('{NEWS_ANALYSIS}') WHERE scisci_paperid IS NOT NULL
      ), pages AS (
        SELECT paperid,subj_id,min(try_cast(timestamp AS TIMESTAMPTZ)) first_seen,
               lower(regexp_replace(regexp_extract(subj_id,'^https?://([^/]+)',1),
                                    '^www\\.','')) host
        FROM read_parquet('{NEWS_DATA}') GROUP BY paperid,subj_id
      )
      SELECT host,count(*) pages FROM focal JOIN pages USING (paperid)
      WHERE first_seen>=make_date(publication_year,1,1)
        AND first_seen<make_date(publication_year+5,1,1)
        AND host<>'' GROUP BY host ORDER BY pages DESC,host LIMIT 3
    """).fetchdf()
    if len(top_hosts) != 3 or top_hosts.host.nunique() != 3:
        raise ValueError(f"expected three unique top hosts, got {top_hosts.to_dict('records')}")
    timing_qc = con.execute(f"""
      WITH focal AS (
        SELECT scisci_paperid paperid,publication_date,publication_year
        FROM read_parquet('{NEWS_ANALYSIS}') WHERE scisci_paperid IS NOT NULL
      ), pages AS (
        SELECT paperid,subj_id,min(try_cast(timestamp AS TIMESTAMPTZ)) first_seen
        FROM read_parquet('{NEWS_DATA}') GROUP BY paperid,subj_id
      )
      SELECT count(*) FILTER (WHERE first_seen<make_date(publication_year,1,1)) AS before_window,
             count(*) FILTER (WHERE first_seen>=make_date(publication_year,1,1)
                AND first_seen<publication_date) AS before_recorded_publication,
             count(*) FILTER (WHERE first_seen>=make_date(publication_year,1,1) AND
                first_seen<make_date(publication_year+5,1,1)) AS included,
             count(*) FILTER (WHERE first_seen>=make_date(publication_year+5,1,1)) AS after_window
      FROM focal JOIN pages USING (paperid)
    """).fetchone()
    return news_qc, coverage, top_hosts, timing_qc


def load_frame(con):
    frame = con.execute(
        "SELECT * EXCLUDE(doi,doi_norm,direct_scisci_doi_norm,paperid,focal_doi_n) "
        "FROM read_parquet(?) "
        "WHERE scisci_paperid IS NOT NULL ORDER BY id",
        [str(NEWS_ANALYSIS)],
    ).df()
    if frame.empty or frame.id.nunique() != len(frame) or set(frame.publication_year) != set(YEARS):
        raise ValueError(f"invalid DOI analysis frame rows={len(frame)} ids={frame.id.nunique()} "
                         f"years={sorted(frame.publication_year.unique())}")
    for name in HEAVY:
        if (frame[name] < 0).any():
            raise ValueError(f"expected nonnegative {name}")
        frame[f"log1p_{name}"] = np.log1p(frame[name].to_numpy(dtype=float)).astype(np.float32)
    for name in CATEGORICAL:
        frame[name] = frame[name].astype("category")
    frame["treatment"] = frame.treatment.astype(np.int8)
    cap = float(frame.web_pages_5cy.quantile(0.999))
    if cap <= 0:
        raise ValueError(f"expected positive web-page winsorization cap, got {cap}")
    frame["web_pages_winsorized"] = frame.web_pages_5cy.clip(upper=cap)
    return frame


def cluster_interval(signal, codes, estimate, multipliers, groups):
    centered = signal - estimate
    sums = np.bincount(codes, weights=centered, minlength=groups)
    n = len(signal)
    active_groups = np.unique(codes).size
    if active_groups < 2:
        raise ValueError(f"expected at least two journal clusters, got {active_groups}")
    se = math.sqrt((active_groups / (active_groups - 1)) * np.square(sums).sum() / n ** 2)
    draws = estimate + multipliers @ sums / n
    return se, estimate - 1.96 * se, estimate + 1.96 * se, \
        float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def fit_news_outcomes(frame, support, propensity):
    treatment = frame.treatment.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy()
    predictions = {name: [np.full(len(frame), np.nan, dtype=np.float32),
                          np.full(len(frame), np.nan, dtype=np.float32)] for name in OUTCOMES}
    diagnostics = []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        prevalence = frame.loc[train].groupby(
            "choice_set_id", observed=True,
        ).treatment.agg(["sum", "count"])
        prevalence = (prevalence["sum"] + 0.5) / (prevalence["count"] + 1)
        x_train = fold_features(frame, train, BASE_NUMERIC, prevalence)
        x_test = fold_features(frame, test, BASE_NUMERIC, prevalence)
        train_treatment = treatment[train]
        bucket = frame.loc[train, "early_stop_bucket"].to_numpy()
        for outcome in OUTCOMES:
            binary = outcome.startswith("any_")
            y = frame.loc[train, outcome].to_numpy(dtype=float)
            for arm in (0, 1):
                fit = (train_treatment == arm) & (bucket != 0)
                valid = (train_treatment == arm) & (bucket == 0)
                model_class = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
                metric = "binary_logloss" if binary else "poisson"
                model = model_class(
                    objective="binary" if binary else "poisson", n_estimators=3000,
                    learning_rate=0.05, num_leaves=255, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    random_state=SEED + 100 * fold + arm, n_jobs=32, verbosity=-1,
                    deterministic=True, force_col_wise=True,
                )
                model.fit(
                    x_train.iloc[fit], y[fit], eval_set=[(x_train.iloc[valid], y[valid])],
                    eval_metric=metric, callbacks=[lgb.early_stopping(50, verbose=False)],
                    categorical_feature="auto",
                )
                pred = model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1] \
                    if binary else model.predict(x_test, num_iteration=model.best_iteration_)
                predictions[outcome][arm][test] = pred
                diagnostics.append({"fold": fold, "outcome": outcome, "arm": arm,
                                    "objective": "binary" if binary else "poisson",
                                    "best_iteration": int(model.best_iteration_),
                                    "validation_loss": float(model.best_score_["valid_0"][metric])})
                log(f"news outcome fold={fold} name={outcome} arm={arm} "
                    f"best_iteration={model.best_iteration_}")
    if any(not np.isfinite(predictions[name][arm]).all()
           for name in OUTCOMES for arm in (0, 1)):
        raise ValueError("nonfinite news outcome predictions")

    d = frame.loc[support].reset_index(drop=True)
    a = d.treatment.to_numpy(dtype=np.int8)
    p = propensity[support].astype(float)
    codes, journals = pd.factorize(d.journal_id, sort=True)
    multipliers = np.random.default_rng(SEED).standard_normal((500, len(journals)))
    rows = []
    for outcome in OUTCOMES:
        y = d[outcome].to_numpy(dtype=float)
        m0 = predictions[outcome][0][support].astype(float)
        m1 = predictions[outcome][1][support].astype(float)
        psi0 = m0 + (1 - a) * (y - m0) / (1 - p)
        psi1 = m1 + a * (y - m1) / p
        for year in ("all",) + YEARS:
            mask = np.ones(len(d), dtype=bool) if year == "all" else d.publication_year.eq(year).to_numpy()
            mu0, mu1 = psi0[mask].mean(), psi1[mask].mean()
            difference = (psi1 - psi0)[mask]
            absolute = cluster_interval(
                difference, codes[mask], float(difference.mean()), multipliers,
                len(journals),
            )
            log_ratio = math.log(mu1 / mu0)
            ratio_signal = (psi1[mask] - mu1) / mu1 - (psi0[mask] - mu0) / mu0
            relative = cluster_interval(
                log_ratio + ratio_signal, codes[mask], log_ratio, multipliers,
                len(journals),
            )
            rows.extend([
                {"outcome": outcome, "period": year, "scale": "absolute_difference",
                 "mean_broad": mu0, "mean_narrower": mu1, "estimate": difference.mean(),
                 "se": absolute[0], "ci_low": absolute[1], "ci_high": absolute[2],
                 "bootstrap_ci_low": absolute[3], "bootstrap_ci_high": absolute[4],
                 "n": int(mask.sum()), "journals": int(d.loc[mask, "journal_id"].nunique())},
                {"outcome": outcome, "period": year, "scale": "log_mean_ratio",
                 "mean_broad": mu0, "mean_narrower": mu1, "estimate": log_ratio,
                 "se": relative[0], "ci_low": relative[1], "ci_high": relative[2],
                 "bootstrap_ci_low": relative[3], "bootstrap_ci_high": relative[4],
                 "n": int(mask.sum()), "journals": int(d.loc[mask, "journal_id"].nunique())},
            ])
    return pd.DataFrame(rows), pd.DataFrame(diagnostics), d, p


def source_sensitivity(con, d, propensity, top_hosts):
    ids = d[["id", "scisci_paperid", "publication_date", "publication_year", "treatment"]].copy()
    ids = ids.rename(columns={"scisci_paperid": "paperid"})
    ids["ipw"] = np.where(ids.treatment.eq(1), 1 / propensity, 1 / (1 - propensity))
    con.register("supported", ids)
    escaped = [host.replace("'", "''") for host in top_hosts.host]
    count_terms = [
        "count(p.subj_id) AS all_pages",
        "count(p.subj_id) FILTER (WHERE NOT (p.host='wikipedia.org' "
        "OR p.host LIKE '%.wikipedia.org' OR p.host='slideshare.net' "
        "OR p.host LIKE '%.slideshare.net')) AS no_wikipedia_slideshare_pages",
    ]
    for index in range(1, 4):
        exclusions = ",".join(f"'{host}'" for host in escaped[:index])
        count_terms.append(
            f"count(p.subj_id) FILTER (WHERE p.host NOT IN ({exclusions})) "
            f"AS no_top{index}_pages"
        )
    counts_sql = ",".join(count_terms)
    table = con.execute(f"""
      WITH pages AS (
        SELECT n.paperid,n.subj_id,min(try_cast(n.timestamp AS TIMESTAMPTZ)) first_seen,
               lower(regexp_replace(regexp_extract(n.subj_id,'^https?://([^/]+)',1),
                                    '^www\\.','')) host
        FROM read_parquet('{NEWS_DATA}') n GROUP BY n.paperid,n.subj_id
      ), counts AS (
        SELECT s.id,{counts_sql}
        FROM supported s LEFT JOIN pages p ON s.paperid=p.paperid
          AND p.first_seen>=make_date(s.publication_year,1,1)
          AND p.first_seen<make_date(s.publication_year+5,1,1)
        GROUP BY s.id
      )
      SELECT s.*,c.* EXCLUDE(id) FROM supported s JOIN counts c USING (id)
    """).fetchdf()
    rows = []
    for outcome in ("all_pages", "no_wikipedia_slideshare_pages",
                    "no_top1_pages", "no_top2_pages", "no_top3_pages"):
        for form, values in (("count", table[outcome]), ("any", table[outcome].gt(0).astype(int))):
            means = []
            for arm in (0, 1):
                mask = table.treatment.eq(arm)
                means.append(float(np.average(values[mask], weights=table.loc[mask, "ipw"])))
            rows.append({"outcome": outcome, "form": form, "mean_broad": means[0],
                         "mean_narrower": means[1], "difference": means[1] - means[0],
                         "log_mean_ratio": math.log(means[1] / means[0])})
    return pd.DataFrame(rows)


def main():
    validate_snapshot()
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    news_qc, doi_coverage, top_hosts, timing_qc = build_news_analysis(con)
    frame = load_frame(con)
    candidates = []
    for leaves in (63, 255):
        candidates.append((leaves,) + fit_propensity(
            frame, BASE_NUMERIC, leaves, f"news_leaves_{leaves}", deterministic=True,
        ))
    difference = abs(candidates[0][5]["max_weighted_abs_smd"]
                     - candidates[1][5]["max_weighted_abs_smd"])
    chosen = candidates[0] if difference <= 0.005 else min(
        candidates, key=lambda item: item[5]["max_weighted_abs_smd"],
    )
    leaves, propensity, _, support, balance, diagnostic = chosen
    if support.mean() < 0.50:
        log(f"news support below promotion gate: {support.mean():.4%}")
    estimates, outcome_diagnostics, supported, p = fit_news_outcomes(
        frame, support, propensity,
    )
    source_checks = source_sensitivity(con, supported, p, top_hosts)
    primary_names = ["any_web_5cy", "web_pages_5cy"]
    overall = estimates[(estimates.period.eq("all")) &
                        estimates.scale.eq("absolute_difference") &
                        estimates.outcome.isin(primary_names)]
    years = estimates[(estimates.period.ne("all")) &
                      estimates.scale.eq("absolute_difference") &
                      estimates.outcome.isin(primary_names)]
    primary_pass = len(overall) == 2 and (overall.estimate < 0).all() and \
        (overall.ci_high < 0).all() and (overall.bootstrap_ci_high < 0).all()
    year_pass = len(years) == 4 and (years.estimate < 0).all()
    source_pass = (source_checks.loc[source_checks.outcome.ne("all_pages"), "difference"] < 0).all()
    overall_coverage = doi_coverage.loc[doi_coverage.period.eq("all")].sort_values("treatment")
    linkage = overall_coverage.linkage_share_among_doi.to_numpy(dtype=float)
    linkage_pass = bool(linkage.min() >= 0.95 and abs(linkage[1] - linkage[0]) <= 0.02)
    arm_n = frame.treatment.value_counts()
    ess_pass = bool(diagnostic["ess_broad"] >= 0.20 * arm_n[0] and
                    diagnostic["ess_specialized"] >= 0.20 * arm_n[1])
    promote = bool(primary_pass and year_pass and source_pass and linkage_pass
                   and ess_pass and support.mean() >= 0.50)

    estimates.to_csv(RESULTS / "news_estimates.csv", index=False)
    pd.concat([item[4] for item in candidates], ignore_index=True).to_csv(
        RESULTS / "news_balance.csv", index=False,
    )
    pd.DataFrame([item[5] for item in candidates]).to_csv(
        RESULTS / "news_propensity.csv", index=False,
    )
    outcome_diagnostics.to_csv(RESULTS / "news_outcome_diagnostics.csv", index=False)
    doi_coverage.to_csv(RESULTS / "news_doi_coverage.csv", index=False)
    top_hosts.to_csv(RESULTS / "news_top_hosts.csv", index=False)
    source_checks.to_csv(RESULTS / "news_source_sensitivity.csv", index=False)
    gates = pd.DataFrame([
        {"gate": "primary_any_and_count_negative_ci", "passed": bool(primary_pass)},
        {"gate": "both_years_same_direction", "passed": bool(year_pass)},
        {"gate": "top_host_exclusion_same_direction", "passed": bool(source_pass)},
        {"gate": "sciscinet_linkage_at_least_95_percent_and_within_2pp",
         "passed": linkage_pass},
        {"gate": "each_arm_ess_at_least_20_percent", "passed": ess_pass},
        {"gate": "common_support_at_least_50_percent", "passed": bool(support.mean() >= 0.50)},
        {"gate": "promote_to_main_text", "passed": promote},
    ])
    gates.to_csv(RESULTS / "news_gates.csv", index=False)
    run = write_run("news", {
        "eligible_2018_2019": int(overall_coverage.eligible.sum()),
        "sciscinet_linked_doi_observable": len(frame), "support": int(support.sum()),
        "journals": int(supported.journal_id.nunique()),
        "tracked_pages": int(overall_coverage.pages.sum()),
    }, {
        "window": "[January 1 of publication_year, January 1 of publication_year + 5)",
        "newsfeed_coverage": [news_qc[5], news_qc[6]],
        "newsfeed_rows": news_qc[0], "unique_paper_pages": news_qc[2],
        "unique_newsfeed_ids": news_qc[3],
        "deduplicated_rows": news_qc[0] - news_qc[2],
        "timing_qc": {"before_window": timing_qc[0],
                      "before_recorded_publication_in_window": timing_qc[1],
                      "included": timing_qc[2], "after_window": timing_qc[3]},
        "selected_propensity_leaves": leaves, "propensity": diagnostic,
        "top_hosts": top_hosts.to_dict("records"), "promotion_gates": gates.to_dict("records"),
        "promote_to_main_text": promote,
        "persistent_bytes": tree_bytes(V2_WORK) + tree_bytes(V3_WORK),
        "group_free_bytes": shutil.disk_usage(GROUP_ROOT).free,
    })
    check_budget()
    for row in overall.sort_values("outcome").itertuples():
        log(f"news result outcome={row.outcome} broad={row.mean_broad:.6g} "
            f"narrower={row.mean_narrower:.6g} difference={row.estimate:.6g} "
            f"ci=[{row.ci_low:.6g},{row.ci_high:.6g}] "
            f"bootstrap=[{row.bootstrap_ci_low:.6g},{row.bootstrap_ci_high:.6g}]")
    log(f"news complete support={support.sum():,} journals={supported.journal_id.nunique():,} "
        f"max_smd={diagnostic['max_weighted_abs_smd']:.6g} "
        f"linkage={linkage.tolist()} promote={promote} commit={run['git_commit']}")


if __name__ == "__main__":
    main()
