#!/usr/bin/env python3
import json
import math
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qss_common import EMBED_DIM, GROUP_ROOT, SEED
from qss_v3_common import (
    ARTIFACTS, RESULTS, V2_WORK, V3_WORK, check_budget, connect, copy_query,
    log, path_glob, reset_output, tree_bytes, validate_snapshot, write_run,
)

CANDIDATE = V3_WORK / "candidate_focal.parquet"
EDGES = V3_WORK / "citation_edges"
CITING = V3_WORK / "citing_metadata"
REFERENCE_EDGES = V3_WORK / "reference_edges"
LEADS = V3_WORK / "lead_authors.parquet"
PRIOR = V3_WORK / "author_prior_papers"
AUTHOR_YEAR = V3_WORK / "author_year_routing.parquet"
AUTHOR_FEATURES = V3_WORK / "author_routing.parquet"
REFERENCE_FEATURES = V3_WORK / "reference_routing.parquet"
OUTCOMES = V3_WORK / "citation_outcomes.parquet"
ANALYSIS = V3_WORK / "analysis_dataset.parquet"

PC = [f"pc{i:02d}" for i in range(1, 33)]
QPC = [f"qpc{i:02d}" for i in range(1, 33)]
HEAVY = [
    "reference_count", "classified_n", "author_mean_prior_works",
    "author_max_prior_works", "author_mean_prior_citations",
    "author_max_prior_citations", "institution_mean_prior_works",
    "institution_max_prior_works", "institution_mean_prior_citations",
    "institution_max_prior_citations", "history_n", "prior_prestige",
]
BASE_NUMERIC = PC + QPC + [
    "reference_fields", "reference_entropy", "authors_count", "countries_count",
    "institutions_count", "international", "prior_oa_share", "prior_english_share",
    "lead_prior_venue_specialization", "lead_prior_embedding_breadth",
    "lead_prior_venue_contributors", "lead_prior_breadth_contributors",
    "lead_prior_venue_missing", "lead_prior_breadth_missing",
] + [f"log1p_{name}" for name in HEAVY]
CATEGORICAL = ["publication_year", "lead_country", "semantic_cluster", "choice_set_id"]
PRIMARY_OUTCOMES = [
    "total_citations", "near", "intermediate", "far", "unclassified", "any_far",
    "near_winsorized", "far_winsorized", "ref_near", "ref_far",
]
QWEN_V2 = V2_WORK / "qwen3_semantics.parquet"
QWEN_V3 = V3_WORK / "qwen3_semantics"
SPECTER_V2 = V2_WORK / "embeddings_title"
SPECTER_V3 = V3_WORK / "specter_embeddings"
SCOPE = V2_WORK / "journal_year_scope.parquet"


def vectors(column):
    values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    return np.asarray(values.values, dtype=np.float32).reshape(-1, EMBED_DIM)


def build_author_year(con):
    reset_output(AUTHOR_YEAR)
    schema = pa.schema([
        ("author_id", pa.string()), ("focal_year", pa.int32()),
        ("prior_papers", pa.int64()), ("venue_papers", pa.int64()),
        ("prior_venue_specialization", pa.float64()),
        ("prior_embedding_breadth", pa.float64()),
    ])
    writer = pq.ParquetWriter(AUTHOR_YEAR, schema, compression="zstd")
    query = f"""
      WITH embedding AS (
        SELECT id,embedding FROM read_parquet('{path_glob(SPECTER_V2)}')
        UNION ALL
        SELECT id,embedding FROM read_parquet('{path_glob(SPECTER_V3)}')
      )
      SELECT p.author_id,p.focal_year,p.prior_id,s.semantic_title_similarity,e.embedding
      FROM read_parquet('{path_glob(PRIOR)}') p
      JOIN embedding e ON p.prior_id=e.id
      LEFT JOIN read_parquet('{SCOPE}') s
        ON p.focal_year=s.focal_year AND p.journal_id=s.journal_id
       AND s.semantic_reliable
      ORDER BY p.author_id,p.focal_year,p.prior_id
    """
    reader = con.execute(query).fetch_record_batch(100_000)
    pending = None
    output = []
    row_count = input_count = 0

    def emit(value):
        nonlocal row_count
        author, year, total, squares, n, spec_sum, spec_n = value
        cosine = (float(total @ total) - squares) / (n * (n - 1)) if n > 1 else None
        output.append({
            "author_id": author, "focal_year": year, "prior_papers": n,
            "venue_papers": spec_n,
            "prior_venue_specialization": spec_sum / spec_n if spec_n else None,
            "prior_embedding_breadth": 1 - cosine if cosine is not None else None,
        })
        row_count += 1
        if len(output) == 100_000:
            writer.write_table(pa.Table.from_pylist(output, schema=schema))
            output.clear()

    for batch in reader:
        authors = batch.column("author_id").to_numpy(zero_copy_only=False)
        years = batch.column("focal_year").to_numpy(zero_copy_only=False)
        matrix = vectors(batch.column("embedding"))
        spec = batch.column("semantic_title_similarity").to_numpy(
            zero_copy_only=False,
        ).astype(float)
        starts = np.r_[0, np.flatnonzero(
            (authors[1:] != authors[:-1]) | (years[1:] != years[:-1])
        ) + 1]
        ends = np.r_[starts[1:], len(authors)]
        sums = np.add.reduceat(matrix.astype(np.float64), starts, axis=0)
        squares = np.add.reduceat(np.einsum("ij,ij->i", matrix, matrix), starts)
        spec_sum = np.add.reduceat(np.nan_to_num(spec, nan=0.0), starts)
        spec_n = np.add.reduceat(np.isfinite(spec).astype(np.int64), starts)
        for start, end, total, square, score_sum, score_n in zip(
            starts, ends, sums, squares, spec_sum, spec_n,
        ):
            value = [str(authors[start]), int(years[start]), total, float(square),
                     int(end - start), float(score_sum), int(score_n)]
            if pending is not None and pending[:2] == value[:2]:
                pending[2] += value[2]
                pending[3] += value[3]
                pending[4] += value[4]
                pending[5] += value[5]
                pending[6] += value[6]
            else:
                if pending is not None:
                    emit(pending)
                pending = value
        input_count += len(batch)
        if input_count % 5_000_000 < 100_000:
            log(f"author embedding aggregation rows={input_count:,} groups={row_count:,}")
    if pending is not None:
        emit(pending)
    if output:
        writer.write_table(pa.Table.from_pylist(output, schema=schema))
    writer.close()
    prior_n = con.execute(
        "SELECT count(*) FROM read_parquet(?)", [path_glob(PRIOR)],
    ).fetchone()[0]
    if input_count != prior_n or row_count == 0:
        raise ValueError(f"author embedding coverage failed: prior={prior_n}, "
                         f"embedded={input_count}, groups={row_count}")
    log(f"built author-year routing rows={row_count:,} prior_papers={input_count:,}")
    return row_count


def build_structural_features(con):
    author_n = build_author_year(con)
    feature_n = copy_query(con, AUTHOR_FEATURES, f"""
      SELECT l.focal_id,
             avg(a.prior_venue_specialization) FILTER
               (WHERE a.prior_venue_specialization IS NOT NULL)
               AS lead_prior_venue_specialization,
             avg(a.prior_embedding_breadth) FILTER
               (WHERE a.prior_embedding_breadth IS NOT NULL)
               AS lead_prior_embedding_breadth,
             count(a.prior_venue_specialization) AS lead_prior_venue_contributors,
             count(a.prior_embedding_breadth) AS lead_prior_breadth_contributors
      FROM read_parquet('{LEADS}') l
      LEFT JOIN read_parquet('{AUTHOR_YEAR}') a
        ON l.author_id=a.author_id AND l.focal_year=a.focal_year
      GROUP BY l.focal_id
    """)
    con.execute(f"""
      CREATE OR REPLACE TEMP VIEW qwen_all AS
      SELECT id,qwen_leaf,qwen_macro,qwen_ood FROM read_parquet('{QWEN_V2}')
      UNION ALL
      SELECT id,qwen_leaf,qwen_macro,qwen_ood FROM read_parquet('{path_glob(QWEN_V3)}')
    """)
    qwen_qc = con.execute(
        "SELECT count(*),count(DISTINCT id) FROM qwen_all",
    ).fetchone()
    if qwen_qc[0] != qwen_qc[1]:
        raise ValueError(f"combined Qwen semantics contain duplicates: {qwen_qc}")
    reference_n = copy_query(con, REFERENCE_FEATURES, f"""
      WITH labeled AS (
        SELECT r.focal_id,
          CASE WHEN qr.id IS NULL OR qr.qwen_ood OR qf.qwen_ood THEN 'unclassified'
               WHEN qr.qwen_leaf=qf.qwen_leaf THEN 'near'
               WHEN qr.qwen_macro=qf.qwen_macro THEN 'intermediate'
               ELSE 'far' END AS class
        FROM read_parquet('{path_glob(REFERENCE_EDGES)}') r
        JOIN read_parquet('{CANDIDATE}') f ON r.focal_id=f.id
        JOIN qwen_all qf ON f.id=qf.id
        LEFT JOIN qwen_all qr ON r.ref_id=qr.id
      ), totals AS (
        SELECT focal_id,count(*) AS ref_total,
               count(*) FILTER (WHERE class='near') AS ref_near,
               count(*) FILTER (WHERE class='intermediate') AS ref_intermediate,
               count(*) FILTER (WHERE class='far') AS ref_far,
               count(*) FILTER (WHERE class='unclassified') AS ref_unclassified
        FROM labeled GROUP BY focal_id
      )
      SELECT f.id,
             COALESCE(t.ref_total,0) AS ref_total,
             COALESCE(t.ref_near,0) AS ref_near,
             COALESCE(t.ref_intermediate,0) AS ref_intermediate,
             COALESCE(t.ref_far,0) AS ref_far,
             COALESCE(t.ref_unclassified,0) AS ref_unclassified,
             ln((COALESCE(t.ref_far,0)+0.5)/(COALESCE(t.ref_near,0)+0.5)) AS ref_log_ratio,
             COALESCE(t.ref_near+t.ref_intermediate+t.ref_far,0) AS ref_classified,
             CASE WHEN COALESCE(t.ref_total,0)>0
               THEN t.ref_unclassified::DOUBLE/t.ref_total ELSE 0 END AS ref_unclassified_share
      FROM read_parquet('{CANDIDATE}') f LEFT JOIN totals t ON f.id=t.focal_id
    """)
    return author_n, feature_n, reference_n, qwen_qc


def build_outcomes(con):
    outcome_n = copy_query(con, OUTCOMES, f"""
      WITH labeled AS (
        SELECT e.cited_id,c.id AS citing_id,c.journal_id=f.journal_id AS same_journal,
               COALESCE(list_has_any(c.author_ids,f.author_ids),false) AS shared_author,
          CASE WHEN c.language<>'en' OR c.language IS NULL OR qc.id IS NULL
                       OR qc.qwen_ood OR qf.qwen_ood THEN 'unclassified'
               WHEN qc.qwen_leaf=qf.qwen_leaf THEN 'near'
               WHEN qc.qwen_macro=qf.qwen_macro THEN 'intermediate'
               ELSE 'far' END AS class
        FROM read_parquet('{path_glob(EDGES)}') e
        JOIN read_parquet('{CANDIDATE}') f ON e.cited_id=f.id
        JOIN read_parquet('{path_glob(CITING)}') c ON e.citing_id=c.id
        JOIN qwen_all qf ON f.id=qf.id
        LEFT JOIN qwen_all qc ON c.id=qc.id
      ), external AS (
        SELECT * FROM labeled WHERE NOT same_journal AND NOT shared_author
      ), totals AS (
        SELECT cited_id,count(*) AS total_citations,
               count(*) FILTER (WHERE class='near') AS near,
               count(*) FILTER (WHERE class='intermediate') AS intermediate,
               count(*) FILTER (WHERE class='far') AS far,
               count(*) FILTER (WHERE class='unclassified') AS unclassified
        FROM external GROUP BY cited_id
      ), exclusions AS (
        SELECT cited_id,count(*) FILTER (WHERE same_journal) AS same_journal_citations,
               count(*) FILTER (WHERE shared_author) AS shared_author_citations
        FROM labeled GROUP BY cited_id
      )
      SELECT f.id,COALESCE(t.total_citations,0) AS total_citations,
             COALESCE(t.near,0) AS near,COALESCE(t.intermediate,0) AS intermediate,
             COALESCE(t.far,0) AS far,COALESCE(t.unclassified,0) AS unclassified,
             (COALESCE(t.far,0)>0)::UTINYINT AS any_far,
             COALESCE(x.same_journal_citations,0) AS same_journal_citations,
             COALESCE(x.shared_author_citations,0) AS shared_author_citations
      FROM read_parquet('{CANDIDATE}') f
      LEFT JOIN totals t ON f.id=t.cited_id LEFT JOIN exclusions x ON f.id=x.cited_id
    """)
    bad = con.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE "
        "total_citations<>near+intermediate+far+unclassified", [str(OUTCOMES)],
    ).fetchone()[0]
    if bad:
        raise ValueError(f"expected exact citation decomposition, got failures={bad}")
    return outcome_n


def build_analysis(con):
    return copy_query(con, ANALYSIS, f"""
      SELECT c.* EXCLUDE(referenced_works,author_ids,first_author_id,last_author_id),
             a.lead_prior_venue_specialization,a.lead_prior_embedding_breadth,
             COALESCE(a.lead_prior_venue_contributors,0) AS lead_prior_venue_contributors,
             COALESCE(a.lead_prior_breadth_contributors,0) AS lead_prior_breadth_contributors,
             (a.lead_prior_venue_specialization IS NULL)::UTINYINT AS lead_prior_venue_missing,
             (a.lead_prior_embedding_breadth IS NULL)::UTINYINT AS lead_prior_breadth_missing,
             r.* EXCLUDE(id),o.* EXCLUDE(id)
      FROM read_parquet('{CANDIDATE}') c
      LEFT JOIN read_parquet('{AUTHOR_FEATURES}') a ON c.id=a.focal_id
      JOIN read_parquet('{REFERENCE_FEATURES}') r USING (id)
      JOIN read_parquet('{OUTCOMES}') o USING (id)
    """)


def weighted_stats(values, weights):
    finite = np.isfinite(values)
    if not finite.any() or weights[finite].sum() <= 0:
        raise ValueError("weighted statistic has no finite positive-weight values")
    mean = np.average(values[finite], weights=weights[finite])
    var = np.average((values[finite] - mean) ** 2, weights=weights[finite])
    return mean, var


def balance_table(frame, treatment, weights, numeric, label):
    rows = []
    for stage, stage_weights in (("raw", np.ones(len(frame))), ("weighted", weights)):
        for name in numeric:
            values = frame[name].to_numpy(dtype=float)
            m0, v0 = weighted_stats(values[treatment == 0], stage_weights[treatment == 0])
            m1, v1 = weighted_stats(values[treatment == 1], stage_weights[treatment == 1])
            scale = math.sqrt((v0 + v1) / 2)
            rows.append({"candidate": label, "stage": stage, "covariate": name,
                         "mean_broad": m0, "mean_specialized": m1,
                         "smd": (m1 - m0) / scale if scale else 0.0})
            if np.isnan(values).any():
                missing = np.isnan(values).astype(float)
                z0, q0 = weighted_stats(missing[treatment == 0], stage_weights[treatment == 0])
                z1, q1 = weighted_stats(missing[treatment == 1], stage_weights[treatment == 1])
                scale = math.sqrt((q0 + q1) / 2)
                rows.append({"candidate": label, "stage": stage,
                             "covariate": name + "__missing", "mean_broad": z0,
                             "mean_specialized": z1,
                             "smd": (z1 - z0) / scale if scale else 0.0})
        for name in CATEGORICAL:
            table = pd.DataFrame({"level": frame[name].astype(object), "arm": treatment,
                                  "weight": stage_weights}).fillna({"level": "__MISSING__"})
            grouped = table.groupby(["level", "arm"], observed=False).weight.sum().unstack(fill_value=0)
            for arm in (0, 1):
                if arm not in grouped:
                    grouped[arm] = 0.0
            p0 = grouped[0] / grouped[0].sum()
            p1 = grouped[1] / grouped[1].sum()
            scale = np.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2)
            smd = (p1 - p0).divide(scale.where(scale.gt(0)), fill_value=0).fillna(0)
            rows.extend({"candidate": label, "stage": stage,
                         "covariate": f"{name}={level}", "mean_broad": p0[level],
                         "mean_specialized": p1[level], "smd": smd[level]}
                        for level in grouped.index)
    return pd.DataFrame(rows)


def fold_features(frame, mask, numeric, prevalence):
    columns = numeric + CATEGORICAL
    x = frame.loc[mask, columns].copy()
    x["choice_prevalence"] = frame.loc[mask, "choice_set_id"].map(prevalence)
    if x.choice_prevalence.isna().any():
        missing = int(x.choice_prevalence.isna().sum())
        raise ValueError(f"choice-set prevalence missing for {missing} rows")
    for name in CATEGORICAL:
        x[name] = x[name].astype("category")
    return x


def raw_scale_balance(frame, treatment, weights, label):
    rows = []
    for stage, stage_weights in (("raw_scale_raw", np.ones(len(frame))),
                                 ("raw_scale_weighted", weights)):
        for name in HEAVY:
            values = frame[name].to_numpy(dtype=float)
            m0, v0 = weighted_stats(values[treatment == 0], stage_weights[treatment == 0])
            m1, v1 = weighted_stats(values[treatment == 1], stage_weights[treatment == 1])
            scale = math.sqrt((v0 + v1) / 2)
            rows.append({"candidate": label, "stage": stage, "covariate": name,
                         "mean_broad": m0, "mean_specialized": m1,
                         "smd": (m1 - m0) / scale if scale else 0.0})
    return pd.DataFrame(rows)


def fit_propensity(frame, numeric, leaves, label, deterministic=False):
    treatment = frame.treatment.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy()
    propensity = np.full(len(frame), np.nan, dtype=np.float32)
    oof_prevalence = np.full(len(frame), np.nan, dtype=np.float32)
    iterations = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        counts = frame.loc[train].groupby("choice_set_id", observed=True).treatment.agg(["sum", "count"])
        prevalence = (counts["sum"] + 0.5) / (counts["count"] + 1)
        fit = train & frame.early_stop_bucket.ne(0).to_numpy()
        valid = train & frame.early_stop_bucket.eq(0).to_numpy()
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=3000, learning_rate=0.05,
            num_leaves=leaves, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=SEED + fold,
            n_jobs=32, verbosity=-1, deterministic=deterministic,
            force_col_wise=deterministic,
        )
        x_fit = fold_features(frame, fit, numeric, prevalence)
        x_valid = fold_features(frame, valid, numeric, prevalence)
        x_test = fold_features(frame, test, numeric, prevalence)
        model.fit(
            x_fit, treatment[fit], eval_set=[(x_valid, treatment[valid])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature="auto",
        )
        propensity[test] = model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1]
        oof_prevalence[test] = x_test.choice_prevalence.to_numpy(dtype=np.float32)
        iterations.append(int(model.best_iteration_))
        log(f"propensity {label} fold={fold} best_iteration={model.best_iteration_}")
    if not np.isfinite(propensity).all() or not np.isfinite(oof_prevalence).all():
        raise ValueError(f"nonfinite OOF propensity/prevalence for {label}")
    support = (propensity >= 0.05) & (propensity <= 0.95)
    if not support.any():
        raise ValueError(f"no common-support rows for {label}")
    d = frame.loc[support].reset_index(drop=True)
    d["choice_prevalence"] = oof_prevalence[support]
    a = treatment[support]
    p = propensity[support].astype(float)
    weights = a / p + (1 - a) / (1 - p)
    balance = balance_table(d, a, weights, numeric + ["choice_prevalence"], label)
    balance = pd.concat([
        balance, raw_scale_balance(d, a, weights, label),
    ], ignore_index=True)
    ess = {}
    for arm in (0, 1):
        values = weights[a == arm]
        ess[arm] = float(values.sum() ** 2 / np.square(values).sum())
    diagnostic = {
        "candidate": label, "num_leaves": leaves,
        "support": float(support.mean()), "support_n": int(support.sum()),
        "max_weighted_abs_smd": float(
            balance.loc[balance.stage.eq("weighted"), "smd"].abs().max()
        ),
        "ess_broad": ess[0], "ess_specialized": ess[1],
        "weight_p50": float(np.quantile(weights, 0.5)),
        "weight_p95": float(np.quantile(weights, 0.95)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_max": float(weights.max()),
        "best_iterations": json.dumps(iterations),
    }
    return propensity, oof_prevalence, support, balance, diagnostic


def fit_outcomes(frame, numeric, outcomes, support, propensity, label):
    treatment = frame.treatment.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy()
    predictions = {name: [np.full(len(frame), np.nan, dtype=np.float32),
                          np.full(len(frame), np.nan, dtype=np.float32)] for name in outcomes}
    diagnostics = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        counts = frame.loc[train].groupby("choice_set_id", observed=True).treatment.agg(["sum", "count"])
        prevalence = (counts["sum"] + 0.5) / (counts["count"] + 1)
        x_train = fold_features(frame, train, numeric, prevalence)
        x_test = fold_features(frame, test, numeric, prevalence)
        train_treatment = treatment[train]
        train_bucket = frame.loc[train, "early_stop_bucket"].to_numpy()
        for outcome in outcomes:
            binary = outcome.startswith("any_")
            train_outcome = frame.loc[train, outcome].to_numpy()
            for arm in (0, 1):
                fit = (train_treatment == arm) & (train_bucket != 0)
                valid = (train_treatment == arm) & (train_bucket == 0)
                model_class = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
                model = model_class(
                    objective="binary" if binary else "poisson", n_estimators=3000,
                    learning_rate=0.05, num_leaves=255, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    random_state=SEED + 100 * fold + arm, n_jobs=32, verbosity=-1,
                )
                model.fit(
                    x_train.iloc[fit], train_outcome[fit],
                    eval_set=[(x_train.iloc[valid], train_outcome[valid])],
                    eval_metric="binary_logloss" if binary else "poisson",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                    categorical_feature="auto",
                )
                if binary:
                    pred = model.predict_proba(x_test, num_iteration=model.best_iteration_)[:, 1]
                else:
                    pred = model.predict(x_test, num_iteration=model.best_iteration_)
                predictions[outcome][arm][test] = pred
                diagnostics.append({
                    "analysis": label, "fold": fold, "outcome": outcome, "arm": arm,
                    "objective": "binary" if binary else "poisson",
                    "best_iteration": int(model.best_iteration_),
                    "validation_loss": float(model.best_score_["valid_0"][
                        "binary_logloss" if binary else "poisson"
                    ]),
                })
                log(f"outcome {label} fold={fold} name={outcome} arm={arm} "
                    f"best_iteration={model.best_iteration_}")
    if any(not np.isfinite(value[arm]).all()
           for value in predictions.values() for arm in (0, 1)):
        raise ValueError(f"nonfinite outcome predictions for {label}")
    return estimates_from_predictions(
        frame, outcomes, support, propensity, predictions, label,
    ), pd.DataFrame(diagnostics)


def clustered_interval(signal, codes, estimate, multipliers):
    centered = signal - estimate
    sums = np.bincount(codes, weights=centered)
    n, groups = len(signal), len(sums)
    se = math.sqrt((groups / (groups - 1)) * np.sum(sums ** 2) / n ** 2)
    draws = estimate + multipliers @ sums / n
    return se, estimate - 1.96 * se, estimate + 1.96 * se, \
        float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def estimates_from_predictions(frame, outcomes, support, propensity, predictions, label):
    d = frame.loc[support].reset_index(drop=True)
    a = d.treatment.to_numpy(dtype=np.int8)
    p = propensity[support].astype(float)
    codes, journals = pd.factorize(d.journal_id, sort=True)
    multipliers = np.random.default_rng(SEED).standard_normal((500, len(journals)))
    rows, signals = [], {}
    for outcome in outcomes:
        y = d[outcome].to_numpy(dtype=float)
        m0 = predictions[outcome][0][support].astype(float)
        m1 = predictions[outcome][1][support].astype(float)
        psi0 = m0 + (1 - a) * (y - m0) / (1 - p)
        psi1 = m1 + a * (y - m1) / p
        mu0, mu1 = psi0.mean(), psi1.mean()
        signal = psi1 - psi0
        estimate = signal.mean()
        interval = clustered_interval(signal, codes, estimate, multipliers)
        rows.append({
            "analysis": label, "outcome": outcome, "scale": "absolute",
            "mean_broad": mu0, "mean_specialized": mu1, "estimate": estimate,
            "se": interval[0], "ci_low": interval[1], "ci_high": interval[2],
            "bootstrap_ci_low": interval[3], "bootstrap_ci_high": interval[4],
            "n": len(d), "journals": len(journals), "support": float(support.mean()),
        })
        signals[outcome] = (psi0, psi1)
    for near, far, name in (
        ("near", "far", "far_to_near_routing"),
        ("near_winsorized", "far_winsorized", "far_to_near_routing_winsorized"),
        ("ref_near", "ref_far", "reference_routing"),
    ):
        if near not in signals or far not in signals:
            continue
        near0, near1 = signals[near]
        far0, far1 = signals[far]
        means = [far1.mean(), near1.mean(), far0.mean(), near0.mean()]
        if min(means) <= 0:
            raise ValueError(f"nonpositive routing mean for {name}: {means}")
        estimate = math.log(means[0]) - math.log(means[1]) - math.log(means[2]) + math.log(means[3])
        influence = ((far1 - means[0]) / means[0] - (near1 - means[1]) / means[1]
                     - (far0 - means[2]) / means[2] + (near0 - means[3]) / means[3])
        interval = clustered_interval(estimate + influence, codes, estimate, multipliers)
        rows.append({
            "analysis": label, "outcome": name, "scale": "log_ratio_of_mean_ratios",
            "mean_broad": means[2] / means[3],
            "mean_specialized": means[0] / means[1], "estimate": estimate,
            "se": interval[0], "ci_low": interval[1], "ci_high": interval[2],
            "bootstrap_ci_low": interval[3], "bootstrap_ci_high": interval[4],
            "n": len(d), "journals": len(journals), "support": float(support.mean()),
        })
    return pd.DataFrame(rows)


def main():
    validate_snapshot()
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    prepare = json.loads((ARTIFACTS / "run_prepare.json").read_text())
    embed = json.loads((ARTIFACTS / "run_embed.json").read_text())
    if prepare["status"] != "complete" or embed["status"] != "complete":
        raise RuntimeError(f"expected completed prepare/embed runs, got {prepare}, {embed}")
    con = connect()
    author_year_n, author_n, reference_n, qwen_qc = build_structural_features(con)
    outcome_n = build_outcomes(con)
    analysis_n = build_analysis(con)
    frame = con.execute("SELECT * FROM read_parquet(?)", [str(ANALYSIS)]).df()
    if len(frame) != analysis_n or set(frame.treatment.unique()) != {0, 1}:
        raise ValueError(f"analysis cohort QC failed: rows={len(frame)} treatment={frame.treatment.unique()}")
    for name in HEAVY:
        if (frame[name] < 0).any():
            raise ValueError(f"expected nonnegative {name}")
        frame[f"log1p_{name}"] = np.log1p(frame[name].to_numpy(dtype=float)).astype(np.float32)
    for name in CATEGORICAL:
        frame[name] = frame[name].astype("category")
    frame["treatment"] = frame.treatment.astype(np.int8)
    caps = {name: float(frame[name].quantile(0.999)) for name in ("near", "far")}
    frame["near_winsorized"] = frame.near.clip(upper=caps["near"])
    frame["far_winsorized"] = frame.far.clip(upper=caps["far"])

    candidates = []
    for leaves in (63, 255):
        candidates.append((leaves,) + fit_propensity(
            frame, BASE_NUMERIC, leaves, f"primary_leaves_{leaves}",
        ))
    smd_difference = abs(candidates[0][5]["max_weighted_abs_smd"]
                         - candidates[1][5]["max_weighted_abs_smd"])
    chosen = candidates[0] if smd_difference <= 0.005 else min(
        candidates, key=lambda item: item[5]["max_weighted_abs_smd"],
    )
    leaves, propensity, prevalence, support, balance, diagnostic = chosen
    all_balance = pd.concat([item[4] for item in candidates], ignore_index=True)
    propensity_diagnostics = pd.DataFrame([item[5] for item in candidates])
    if support.mean() < 0.50:
        raise ValueError(f"expected common support >=0.50, got {support.mean()}")
    primary_estimates, outcome_diagnostics = fit_outcomes(
        frame, BASE_NUMERIC, PRIMARY_OUTCOMES, support, propensity, "primary",
    )

    sensitivity_numeric = BASE_NUMERIC + [
        "ref_log_ratio", "ref_classified", "ref_unclassified_share",
    ]
    sens_propensity, _, sens_support, sens_balance, sens_diagnostic = fit_propensity(
        frame, sensitivity_numeric, leaves, "reference_adjusted",
    )[:5]
    sensitivity_estimates, sensitivity_outcome_diagnostics = fit_outcomes(
        frame, sensitivity_numeric, ["near", "far"], sens_support,
        sens_propensity, "reference_adjusted",
    )
    all_balance = pd.concat([all_balance, sens_balance], ignore_index=True)
    propensity_diagnostics = pd.concat([
        propensity_diagnostics, pd.DataFrame([sens_diagnostic]),
    ], ignore_index=True)
    outcome_diagnostics = pd.concat([
        outcome_diagnostics, sensitivity_outcome_diagnostics,
    ], ignore_index=True)
    estimates = pd.concat([primary_estimates, sensitivity_estimates], ignore_index=True)

    primary_theta = estimates[(estimates.analysis.eq("primary")) &
                              estimates.outcome.eq("far_to_near_routing")].iloc[0]
    theta_ref = estimates[(estimates.analysis.eq("primary")) &
                          estimates.outcome.eq("reference_routing")].iloc[0]
    max_smd = float(diagnostic["max_weighted_abs_smd"])
    wording = "effect" if max_smd < 0.10 else "association"
    ratio = float(theta_ref.estimate / primary_theta.estimate)
    total_flow = float(frame.total_citations.sum())
    top_cut = float(frame.total_citations.quantile(0.999))
    top_share = float(frame.loc[frame.total_citations.ge(top_cut), "total_citations"].sum() / total_flow)
    reference_coverage = {}
    citing_coverage = {}
    for arm in (0, 1):
        block = frame[frame.treatment.eq(arm)]
        reference_coverage[arm] = float(block.ref_classified.sum() / block.ref_total.sum())
        citing_coverage[arm] = float((block.near + block.intermediate + block.far).sum()
                                     / block.total_citations.sum())

    estimates.to_csv(RESULTS / "dirty_estimates.csv", index=False)
    all_balance.to_csv(RESULTS / "balance.csv", index=False)
    propensity_diagnostics.to_csv(RESULTS / "propensity_candidates.csv", index=False)
    outcome_diagnostics.to_csv(RESULTS / "outcome_diagnostics.csv", index=False)
    gates = pd.DataFrame([{
        "gate": "weighted_max_abs_smd", "value": max_smd,
        "threshold": "<0.10", "passed": max_smd < 0.10,
    }])
    gates.to_csv(RESULTS / "gates.csv", index=False)
    report = [
        wording.upper(), "", "# QSS v3 dirty pilot", "",
        f"- theta: {primary_theta.estimate:.6f} (95% CI {primary_theta.ci_low:.6f} "
        f"to {primary_theta.ci_high:.6f}; bootstrap {primary_theta.bootstrap_ci_low:.6f} "
        f"to {primary_theta.bootstrap_ci_high:.6f})",
        f"- theta_ref: {theta_ref.estimate:.6f}; theta_ref/theta: {ratio:.6f}",
        f"- Rows: {len(frame):,}; support: {int(support.sum()):,} ({support.mean():.4%})",
        f"- Journals in support: {frame.loc[support, 'journal_id'].nunique():,}",
        f"- Selected propensity leaves: {leaves}; max weighted |SMD|: {max_smd:.6f}",
        f"- Reference coverage broad/specialized: {reference_coverage[0]:.4%} / "
        f"{reference_coverage[1]:.4%}",
        f"- Citing coverage broad/specialized: {citing_coverage[0]:.4%} / "
        f"{citing_coverage[1]:.4%}",
        f"- January 1 focal dates broad/specialized: "
        f"{prepare['extra']['focal_date_qc'][0][3]:.4%} / "
        f"{prepare['extra']['focal_date_qc'][1][3]:.4%}",
        f"- Top 0.1% threshold: {top_cut:.0f}; citation contribution: {top_share:.4%}",
        f"- Persistent bytes: {tree_bytes(V2_WORK) + tree_bytes(V3_WORK):,}; "
        f"group free bytes: {shutil.disk_usage(GROUP_ROOT).free:,}", "",
        "All estimates and diagnostics are in the accompanying CSV files.", "",
    ]
    (RESULTS / "dirty_report.md").write_text("\n".join(report))
    write_run("analyze", {
        "author_year": author_year_n, "author_features": author_n,
        "reference_features": reference_n, "citation_outcomes": outcome_n,
        "analysis": analysis_n, "support": int(support.sum()),
    }, {
        "selected_propensity_leaves": leaves, "max_weighted_smd": max_smd,
        "wording": wording, "theta": float(primary_theta.estimate),
        "theta_ref": float(theta_ref.estimate), "theta_ref_over_theta": ratio,
        "reference_coverage": reference_coverage, "citing_coverage": citing_coverage,
        "winsor_caps": caps, "top_0_1_percent_flow_share": top_share,
        "combined_qwen_qc": qwen_qc,
    })
    check_budget()
    log(f"v3 complete theta={primary_theta.estimate:.6f} theta_ref={theta_ref.estimate:.6f} "
        f"max_smd={max_smd:.6f} wording={wording}")


if __name__ == "__main__":
    main()
