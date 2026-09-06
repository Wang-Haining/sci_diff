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
NEWS_ANALYSIS = V3_WORK / "news_analysis_dataset.parquet"
WORKS = SNAPSHOT / "works/**/*.parquet"
YEARS = (2018, 2019, 2020)
OUTCOMES = ("any_web_24m", "web_pages_24m")


def build_news_analysis(con):
    if not NEWS_DATA.is_file():
        raise FileNotFoundError(f"expected SciSciNet news data at {NEWS_DATA}")
    news_qc = con.execute(f"""
      SELECT count(*) AS rows,count(try_cast(timestamp AS TIMESTAMPTZ)) AS dated,
             min(try_cast(timestamp AS TIMESTAMPTZ)),max(try_cast(timestamp AS TIMESTAMPTZ))
      FROM read_parquet('{NEWS_DATA}')
    """).fetchone()
    if news_qc[0] != news_qc[1] or str(news_qc[2])[:10] != "2017-04-05" \
            or str(news_qc[3])[:10] != "2025-02-14":
        raise ValueError(f"unexpected SciSciNet news coverage: {news_qc}")

    rows = copy_query(con, NEWS_ANALYSIS, f"""
      WITH focal AS (
        SELECT a.*,regexp_extract(a.id,'W[0-9]+$') AS paperid,w.doi
        FROM read_parquet('{ANALYSIS}') a
        LEFT JOIN (
          SELECT id,doi FROM read_parquet('{WORKS}',union_by_name=true)
          WHERE publication_year BETWEEN 2018 AND 2020
        ) w USING (id)
        WHERE a.publication_year BETWEEN 2018 AND 2020
      ), pages AS (
        SELECT n.paperid,n.subj_id,
               min(try_cast(n.timestamp AS TIMESTAMPTZ)) AS first_seen,
               lower(regexp_replace(regexp_extract(n.subj_id,'^https?://([^/]+)',1),
                                    '^www\\.','')) AS host
        FROM read_parquet('{NEWS_DATA}') n
        GROUP BY n.paperid,n.subj_id
      ), linked AS (
        SELECT f.paperid,p.subj_id,p.host
        FROM focal f JOIN pages p USING (paperid)
        WHERE f.doi IS NOT NULL
          AND p.first_seen>=f.publication_date
          AND p.first_seen<f.publication_date+INTERVAL 24 MONTH
      ), totals AS (
        SELECT paperid,count(*) AS web_pages_24m FROM linked GROUP BY paperid
      )
      SELECT f.*,COALESCE(t.web_pages_24m,0) AS web_pages_24m,
             (COALESCE(t.web_pages_24m,0)>0)::UTINYINT AS any_web_24m
      FROM focal f LEFT JOIN totals t USING (paperid)
    """)
    duplicate_ids = con.execute(
        "SELECT count(*)-count(DISTINCT id) FROM read_parquet(?)", [str(NEWS_ANALYSIS)],
    ).fetchone()[0]
    if duplicate_ids or rows <= 0:
        raise ValueError(f"news analysis IDs failed: rows={rows} duplicates={duplicate_ids}")
    coverage = con.execute("""
      SELECT treatment,count(*) AS eligible,count(doi) AS doi_n,
             avg((doi IS NOT NULL)::INT) AS doi_share,
             sum(web_pages_24m) AS pages,count(*) FILTER (WHERE any_web_24m=1) AS mentioned
      FROM read_parquet(?) GROUP BY treatment ORDER BY treatment
    """, [str(NEWS_ANALYSIS)]).fetchdf()
    if list(coverage.treatment) != [0, 1] or (coverage.doi_n <= 0).any():
        raise ValueError(f"invalid DOI coverage: {coverage.to_dict('records')}")
    top_hosts = con.execute(f"""
      WITH focal AS (
        SELECT regexp_extract(id,'W[0-9]+$') paperid,publication_date
        FROM read_parquet('{NEWS_ANALYSIS}') WHERE doi IS NOT NULL
      ), pages AS (
        SELECT paperid,subj_id,min(try_cast(timestamp AS TIMESTAMPTZ)) first_seen,
               lower(regexp_replace(regexp_extract(subj_id,'^https?://([^/]+)',1),
                                    '^www\\.','')) host
        FROM read_parquet('{NEWS_DATA}') GROUP BY paperid,subj_id
      )
      SELECT host,count(*) pages FROM focal JOIN pages USING (paperid)
      WHERE first_seen>=publication_date AND first_seen<publication_date+INTERVAL 24 MONTH
        AND host<>'' GROUP BY host ORDER BY pages DESC,host LIMIT 3
    """).fetchdf()
    if len(top_hosts) != 3 or top_hosts.host.nunique() != 3:
        raise ValueError(f"expected three unique top hosts, got {top_hosts.to_dict('records')}")
    return news_qc, coverage, top_hosts


def load_frame(con):
    frame = con.execute(
        "SELECT * EXCLUDE(doi,paperid) FROM read_parquet(?) WHERE doi IS NOT NULL ORDER BY id",
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
    return frame


def cluster_interval(signal, codes, estimate, multipliers, groups):
    centered = signal - estimate
    sums = np.bincount(codes, weights=centered, minlength=groups)
    n = len(signal)
    se = math.sqrt((groups / (groups - 1)) * np.square(sums).sum() / n ** 2)
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
    ids = d[["id", "publication_date", "treatment"]].copy()
    ids["paperid"] = ids.id.str.extract(r"(W[0-9]+)$", expand=False)
    ids["ipw"] = np.where(ids.treatment.eq(1), 1 / propensity, 1 / (1 - propensity))
    con.register("supported", ids)
    escaped = [host.replace("'", "''") for host in top_hosts.host]
    count_terms = ["count(p.subj_id) AS all_pages"]
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
          AND p.first_seen>=s.publication_date
          AND p.first_seen<s.publication_date+INTERVAL 24 MONTH
        GROUP BY s.id
      )
      SELECT s.*,c.all_pages,c.no_top3_pages FROM supported s JOIN counts c USING (id)
    """).fetchdf()
    rows = []
    for outcome in ("all_pages", "no_top1_pages", "no_top2_pages", "no_top3_pages"):
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
    news_qc, doi_coverage, top_hosts = build_news_analysis(con)
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
        raise ValueError(f"expected news support >=50%, got {support.mean():.4%}")
    estimates, outcome_diagnostics, supported, p = fit_news_outcomes(
        frame, support, propensity,
    )
    source_checks = source_sensitivity(con, supported, p, top_hosts)
    overall = estimates[(estimates.period.eq("all")) &
                        estimates.scale.eq("absolute_difference")]
    years = estimates[(estimates.period.ne("all")) &
                      estimates.scale.eq("absolute_difference")]
    primary_pass = len(overall) == 2 and (overall.estimate < 0).all() and \
        (overall.ci_high < 0).all()
    year_pass = len(years) == 6 and (years.estimate < 0).all()
    source_pass = (source_checks.difference < 0).all()
    promote = bool(primary_pass and year_pass and source_pass and support.mean() >= 0.50)

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
        {"gate": "all_three_years_same_direction", "passed": bool(year_pass)},
        {"gate": "top_host_exclusion_same_direction", "passed": bool(source_pass)},
        {"gate": "common_support_at_least_50_percent", "passed": bool(support.mean() >= 0.50)},
        {"gate": "promote_to_main_text", "passed": promote},
    ])
    gates.to_csv(RESULTS / "news_gates.csv", index=False)
    run = write_run("news", {
        "eligible_2018_2020": int(doi_coverage.eligible.sum()),
        "doi_observable": len(frame), "support": int(support.sum()),
        "journals": int(supported.journal_id.nunique()), "tracked_pages": int(doi_coverage.pages.sum()),
    }, {
        "window": "[publication_date, publication_date + 24 months)",
        "newsfeed_coverage": [str(news_qc[2]), str(news_qc[3])],
        "selected_propensity_leaves": leaves, "propensity": diagnostic,
        "top_hosts": top_hosts.to_dict("records"), "promotion_gates": gates.to_dict("records"),
        "promote_to_main_text": promote,
        "persistent_bytes": tree_bytes(V2_WORK) + tree_bytes(V3_WORK),
        "group_free_bytes": shutil.disk_usage(GROUP_ROOT).free,
    })
    check_budget()
    log(f"news complete support={support.sum():,} journals={supported.journal_id.nunique():,} "
        f"promote={promote} commit={run['git_commit']}")


if __name__ == "__main__":
    main()
