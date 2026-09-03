#!/usr/bin/env python3
import json
import math
import shutil
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from qss_common import (
    ARTIFACTS, EMBED_DIM, GROUP_ROOT, QSS_WORK, RESULTS, SECTION_B_FROZEN, SEED,
    STAGED_INPUT, check_budget, connect, copy_query, log, reset_output, tree_bytes,
    validate_snapshot, write_run,
)

TITLE = QSS_WORK / "embeddings_title"
TITLE_ABSTRACT = QSS_WORK / "embeddings_title_abstract"
EMBED_INPUT = STAGED_INPUT
FOCAL = QSS_WORK / "focal_base.parquet"
EDGES = QSS_WORK / "citation_edges.parquet"
CITING = QSS_WORK / "citing_metadata.parquet"
SCOPE = QSS_WORK / "journal_year_scope.parquet"
SEMANTICS = QSS_WORK / "focal_semantics.parquet"
QWEN = QSS_WORK / "qwen3_semantics.parquet"
OUTCOMES = QSS_WORK / "citation_outcomes.parquet"
UNESTIMATED = QSS_WORK / "analysis_unestimated.parquet"
ANALYSIS = QSS_WORK / "analysis_dataset.parquet"

PC_COLS = [f"pc{i:02d}" for i in range(1, 33)]
QPC_COLS = [f"qpc{i:02d}" for i in range(1, 33)]
NUMERIC_COVARIATES = PC_COLS + QPC_COLS + [
    "reference_count",
    "classified_n", "reference_fields", "reference_entropy", "authors_count",
    "countries_count", "institutions_count", "international",
    "author_mean_prior_works", "author_max_prior_works",
    "author_mean_prior_citations", "author_max_prior_citations",
    "institution_mean_prior_works", "institution_max_prior_works",
    "institution_mean_prior_citations", "institution_max_prior_citations",
    "history_n", "prior_oa_share", "prior_english_share", "prior_prestige",
]
CATEGORICAL_COVARIATES = ["publication_year", "publication_month", "semantic_cluster", "lead_country"]
COVARIATES = NUMERIC_COVARIATES + CATEGORICAL_COVARIATES
OUTCOME_COLS = [
    "total_citations", "near", "intermediate", "far", "unclassified", "any_far",
]


def path_glob(path):
    return str(path / "*.parquet") if path.is_dir() else str(path)


def vectors(column):
    values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    return np.asarray(values.values, dtype=np.float32).reshape(-1, EMBED_DIM)


def scope_scores(con, embedding_path, prefix):
    query = f"""
        SELECT h.id,h.history_journal_id AS journal_id,h.history_publication_year AS publication_year,
               hash(h.id)%2 AS half,e.embedding
        FROM read_parquet('{path_glob(EMBED_INPUT)}') h
        JOIN read_parquet('{path_glob(embedding_path)}') e USING (id)
        WHERE h.is_history AND h.history_journal_id IS NOT NULL
        ORDER BY h.history_journal_id,h.history_publication_year,half
    """
    reader = con.execute(query).fetch_record_batch(100_000)
    annual = {}
    for batch in reader:
        journals = np.asarray(batch.column("journal_id"))
        years = np.asarray(batch.column("publication_year"))
        halves = np.asarray(batch.column("half"))
        matrix = vectors(batch.column("embedding"))
        boundaries = np.r_[0, np.flatnonzero(
            (journals[1:] != journals[:-1]) | (years[1:] != years[:-1]) |
            (halves[1:] != halves[:-1])
        ) + 1]
        sums = np.add.reduceat(matrix.astype(np.float64), boundaries, axis=0)
        sumsq = np.add.reduceat(np.einsum("ij,ij->i", matrix, matrix), boundaries)
        ends = np.r_[boundaries[1:], len(matrix)]
        for start, end, total, squares in zip(boundaries, ends, sums, sumsq):
            key = (str(journals[start]), int(years[start]), int(halves[start]))
            if key in annual:
                annual[key][0] += total
                annual[key][1] += float(squares)
                annual[key][2] += int(end - start)
            else:
                annual[key] = [total, float(squares), int(end - start)]
    if not annual:
        raise ValueError(f"expected annual {prefix} embeddings, got 0")

    def cosine_score(parts):
        total = sum((part[0] for part in parts), np.zeros(EMBED_DIM))
        squares = sum(part[1] for part in parts)
        n = sum(part[2] for part in parts)
        return ((total @ total - squares) / (n * (n - 1)), n) if n > 1 else (np.nan, n)

    by_journal = defaultdict(dict)
    for (journal, year, half), value in annual.items():
        by_journal[journal][(year, half)] = value
    rows = []
    for journal, values in sorted(by_journal.items()):
        for focal_year in range(2015, 2021):
            selected = [value for (year, _), value in values.items()
                        if focal_year - 3 <= year <= focal_year - 1]
            score, n = cosine_score(selected)
            halves = []
            half_ns = []
            for half in (0, 1):
                half_score, half_n = cosine_score([value for (year, part), value in values.items()
                                                   if part == half and focal_year - 3 <= year <= focal_year - 1])
                halves.append(half_score)
                half_ns.append(half_n)
            rows.append({
                "journal_id": journal, "focal_year": focal_year,
                f"{prefix}_similarity": score, f"{prefix}_n": n,
                f"{prefix}_half0": halves[0], f"{prefix}_half1": halves[1],
                f"{prefix}_half0_n": half_ns[0], f"{prefix}_half1_n": half_ns[1],
            })
    frame = pd.DataFrame(rows)
    complete = frame[(frame[f"{prefix}_half0_n"] >= 50) & (frame[f"{prefix}_half1_n"] >= 50)].dropna()
    if len(complete) < 2:
        raise ValueError(f"expected >=2 split-half journal-years for {prefix}, got {len(complete)}")
    pearson = complete[[f"{prefix}_half0", f"{prefix}_half1"]].corr().iloc[0, 1]
    corrected = 2 * pearson / (1 + pearson)
    rank_r = spearmanr(complete[f"{prefix}_half0"], complete[f"{prefix}_half1"]).statistic
    log(f"{prefix} journal-years={len(frame):,} split_n={len(complete):,} "
        f"pearson={pearson:.3f} spearman_brown={corrected:.3f} spearman={rank_r:.3f}")
    return frame, {"pearson": pearson, "spearman_brown": corrected,
                   "spearman": rank_r, "split_n": len(complete)}


def build_scope(con):
    primary, reliability = scope_scores(con, TITLE, "semantic_title")
    replication, abstract_reliability = scope_scores(con, TITLE_ABSTRACT, "semantic_abstract")
    baseline = con.execute(
        "SELECT * FROM read_parquet(?)", [str(QSS_WORK / "journal_year_baseline.parquet")]
    ).df()
    reference = con.execute(
        "SELECT * FROM read_parquet(?)", [str(QSS_WORK / "journal_year_reference_scope.parquet")]
    ).df()
    frame = baseline.merge(primary, on=["journal_id", "focal_year"], how="left", validate="one_to_one")
    frame = frame.merge(replication, on=["journal_id", "focal_year"], how="left", validate="one_to_one")
    frame = frame.merge(reference, on=["journal_id", "focal_year"], how="left", validate="one_to_one")
    frame["semantic_reliable"] = frame.semantic_title_n.ge(100)
    frame["semantic_abstract_reliable"] = frame.semantic_abstract_n.ge(100)
    reset_output(SCOPE)
    frame.to_parquet(SCOPE, index=False, compression="zstd")
    if frame.semantic_reliable.sum() == 0:
        raise ValueError("expected reliable journal-year scope rows, got 0")
    return len(frame), reliability, abstract_reliability


def embedding_sample(con):
    table = con.execute(f"""
        SELECT e.id,e.embedding FROM read_parquet('{path_glob(TITLE)}') e
        JOIN read_parquet('{path_glob(FOCAL)}') f USING (id)
        ORDER BY hash(e.id) LIMIT 1000000
    """).fetch_arrow_table()
    if table.num_rows != 1_000_000:
        raise ValueError(f"expected 1,000,000 semantic training papers, got {table.num_rows}")
    return vectors(table["embedding"])


def build_semantics(con):
    sample = embedding_sample(con)
    pca = PCA(n_components=32, svd_solver="randomized", random_state=SEED).fit(sample)
    kmeans = MiniBatchKMeans(
        n_clusters=1000, batch_size=8192, max_iter=100, n_init=3, random_state=SEED,
    ).fit(sample)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARTIFACTS / "semantic_models.npz", pca_components=pca.components_,
                        pca_mean=pca.mean_, cluster_centers=kmeans.cluster_centers_)
    reset_output(SEMANTICS)
    schema = pa.schema([("id", pa.string()), ("semantic_cluster", pa.int16())] +
                       [(name, pa.float32()) for name in PC_COLS])
    writer = pq.ParquetWriter(SEMANTICS, schema, compression="zstd")
    reader = con.execute(f"""
        SELECT e.id,e.embedding FROM read_parquet('{path_glob(TITLE)}') e
        JOIN read_parquet('{path_glob(FOCAL)}') f USING (id)
    """).fetch_record_batch(100_000)
    n = 0
    for batch in reader:
        matrix = vectors(batch.column("embedding"))
        transformed = pca.transform(matrix).astype("float32")
        data = {"id": batch.column("id"),
                "semantic_cluster": pa.array(kmeans.predict(matrix), type=pa.int16())}
        data.update({name: pa.array(transformed[:, i]) for i, name in enumerate(PC_COLS)})
        writer.write_table(pa.table(data, schema=schema))
        n += len(matrix)
        if n % 1_000_000 < 100_000:
            log(f"focal semantics rows={n:,}")
    writer.close()
    expected = con.execute("SELECT count(*) FROM read_parquet(?)", [str(FOCAL)]).fetchone()[0]
    if n != expected:
        raise ValueError(f"expected focal semantic rows={expected}, got {n}")
    return n, float(pca.explained_variance_ratio_.sum()), float(kmeans.inertia_)


def build_outcomes(con):
    return copy_query(con, OUTCOMES, f"""
        WITH labeled AS (
          SELECT e.citing_id,e.cited_id,
                 c.journal_id=f.journal_id AS same_journal,
                 COALESCE(list_has_any(c.author_ids,f.author_ids),false) AS shared_author,
                 CASE WHEN c.language<>'en' OR c.language IS NULL OR qc.id IS NULL
                                OR qc.qwen_ood OR qf.qwen_ood
                           THEN 'unclassified'
                      WHEN qc.qwen_leaf=qf.qwen_leaf THEN 'near'
                      WHEN qc.qwen_macro=qf.qwen_macro THEN 'intermediate'
                      ELSE 'far' END AS citation_class,
                 CASE WHEN c.language<>'en' OR c.language IS NULL THEN 'language'
                      WHEN qc.id IS NULL THEN 'missing_title'
                      WHEN qc.qwen_ood OR qf.qwen_ood THEN 'ood'
                      ELSE NULL END AS unclassified_reason
          FROM read_parquet('{path_glob(EDGES)}') e
          JOIN read_parquet('{path_glob(FOCAL)}') f ON e.cited_id=f.id
          JOIN read_parquet('{path_glob(CITING)}') c ON e.citing_id=c.id
          JOIN read_parquet('{path_glob(QWEN)}') qf ON f.id=qf.id
          LEFT JOIN read_parquet('{path_glob(QWEN)}') qc ON c.id=qc.id
        ), external AS (
          SELECT * FROM labeled WHERE NOT same_journal AND NOT shared_author
        ), totals AS (
          SELECT cited_id, count(*) AS total_citations,
                 count(*) FILTER (WHERE citation_class='near') AS near,
                 count(*) FILTER (WHERE citation_class='intermediate') AS intermediate,
                 count(*) FILTER (WHERE citation_class='far') AS far,
                 count(*) FILTER (WHERE citation_class='unclassified') AS unclassified,
                 count(*) FILTER (WHERE unclassified_reason='language') AS unclassified_language,
                 count(*) FILTER (WHERE unclassified_reason='missing_title') AS unclassified_missing_title,
                 count(*) FILTER (WHERE unclassified_reason='ood') AS unclassified_ood
          FROM external GROUP BY cited_id
        ), all_edges AS (
          SELECT cited_id,count(*) AS all_unique_citations,
                 count(*) FILTER (WHERE same_journal) AS same_journal_citations,
                 count(*) FILTER (WHERE shared_author) AS shared_author_citations
          FROM labeled GROUP BY cited_id
        )
        SELECT f.id,
               COALESCE(t.total_citations,0) AS total_citations,
               COALESCE(t.near,0) AS near,
               COALESCE(t.intermediate,0) AS intermediate,
               COALESCE(t.far,0) AS far,
               COALESCE(t.unclassified,0) AS unclassified,
               COALESCE(t.unclassified_language,0) AS unclassified_language,
               COALESCE(t.unclassified_missing_title,0) AS unclassified_missing_title,
               COALESCE(t.unclassified_ood,0) AS unclassified_ood,
               COALESCE(a.all_unique_citations,0) AS all_unique_citations,
               COALESCE(a.same_journal_citations,0) AS same_journal_citations,
               COALESCE(a.shared_author_citations,0) AS shared_author_citations,
               (COALESCE(t.far,0)>0)::UTINYINT AS any_far
        FROM read_parquet('{path_glob(FOCAL)}') f
        LEFT JOIN totals t ON f.id=t.cited_id LEFT JOIN all_edges a ON f.id=a.cited_id
    """)


def build_unestimated(con):
    pc = ",".join(f"s.{name}" for name in PC_COLS)
    qpc = ",".join(f"q.{name}" for name in QPC_COLS)
    query = f"""
        WITH candidate AS (
          SELECT f.id,f.publication_year,f.publication_month,
                 f.journal_id,f.journal_name,f.lead_country,
                 f.reference_count,f.authors_count,f.countries_count,f.institutions_count,
                 s.semantic_cluster,{pc},q.qwen_ood AS focal_ood,{qpc},
                 r.classified_n,r.reference_fields,
                 r.reference_entropy,r.reference_hhi,
                 COALESCE(a.author_mean_prior_works,0) AS author_mean_prior_works,
                 COALESCE(a.author_max_prior_works,0) AS author_max_prior_works,
                 COALESCE(a.author_mean_prior_citations,0) AS author_mean_prior_citations,
                 COALESCE(a.author_max_prior_citations,0) AS author_max_prior_citations,
                 COALESCE(i.institution_mean_prior_works,0) AS institution_mean_prior_works,
                 COALESCE(i.institution_max_prior_works,0) AS institution_max_prior_works,
                 COALESCE(i.institution_mean_prior_citations,0) AS institution_mean_prior_citations,
                 COALESCE(i.institution_max_prior_citations,0) AS institution_max_prior_citations,
                 (f.countries_count>1)::UTINYINT AS international,
                 j.history_n,j.prior_oa_share,j.prior_english_share,j.prior_prestige,
                 j.semantic_title_similarity,
                 o.* EXCLUDE(id), hash(f.journal_id)%5 AS fold
          FROM read_parquet('{path_glob(FOCAL)}') f
          JOIN read_parquet('{path_glob(SEMANTICS)}') s USING (id)
          JOIN read_parquet('{path_glob(QWEN)}') q USING (id)
          JOIN read_parquet('{path_glob(SCOPE)}') j
            ON f.journal_id=j.journal_id AND f.publication_year=j.focal_year
          JOIN read_parquet('{path_glob(QSS_WORK / 'focal_reference_metrics.parquet')}') r USING (id)
          LEFT JOIN read_parquet('{path_glob(QSS_WORK / 'focal_author_history.parquet')}') a USING (id)
          LEFT JOIN read_parquet('{path_glob(QSS_WORK / 'focal_institution_history.parquet')}') i USING (id)
          JOIN read_parquet('{path_glob(OUTCOMES)}') o USING (id)
          WHERE j.semantic_reliable AND f.work_type='article'
        ), choices AS (
          SELECT semantic_cluster,publication_year,journal_id,
                 any_value(semantic_title_similarity) AS semantic_title_similarity
          FROM candidate GROUP BY ALL
        ), semantic_ranks AS (
          SELECT *, ntile(4) OVER (PARTITION BY semantic_cluster,publication_year
                                  ORDER BY semantic_title_similarity,journal_id) AS semantic_q
          FROM choices
        ), assigned AS (
          SELECT c.*,CASE WHEN semantic_q=1 THEN 0 WHEN semantic_q=4 THEN 1 END AS treatment
          FROM candidate c JOIN semantic_ranks USING (semantic_cluster,publication_year,journal_id)
        ), arm_counts AS (
          SELECT semantic_cluster,publication_year,treatment,count(*) AS papers,
                 count(DISTINCT journal_id) AS journals
          FROM assigned WHERE treatment IS NOT NULL GROUP BY ALL
        ), valid AS (
          SELECT semantic_cluster,publication_year FROM arm_counts GROUP BY ALL
          HAVING count(*)=2 AND min(papers)>=20 AND min(journals)>=2
        )
        SELECT a.*
        FROM assigned a JOIN valid USING (semantic_cluster,publication_year)
    """
    return copy_query(con, UNESTIMATED, query)


def weighted_moments(values, weights):
    finite = np.isfinite(values)
    if finite.sum() == 0 or weights[finite].sum() <= 0:
        raise ValueError("weighted moments received no finite positive-weight observations")
    mean = np.average(values[finite], weights=weights[finite])
    variance = np.average((values[finite] - mean) ** 2, weights=weights[finite])
    return mean, variance


def balance_table(frame, treatment, weights, exposure):
    rows = []
    for stage, stage_weights in [("raw", np.ones(len(frame))), ("weighted", weights)]:
        for covariate in NUMERIC_COVARIATES:
            values = frame[covariate].to_numpy(dtype=float)
            m0, v0 = weighted_moments(values[treatment == 0], stage_weights[treatment == 0])
            m1, v1 = weighted_moments(values[treatment == 1], stage_weights[treatment == 1])
            scale = math.sqrt((v0 + v1) / 2)
            rows.append({"exposure": exposure, "stage": stage, "covariate": covariate,
                         "mean_broad": m0, "mean_specialized": m1,
                         "smd": (m1 - m0) / scale if scale else 0.0})
            if np.isnan(values).any():
                missing = np.isnan(values).astype(float)
                q0, z0 = weighted_moments(missing[treatment == 0], stage_weights[treatment == 0])
                q1, z1 = weighted_moments(missing[treatment == 1], stage_weights[treatment == 1])
                scale = math.sqrt((z0 + z1) / 2)
                rows.append({"exposure": exposure, "stage": stage,
                             "covariate": covariate + "__missing", "mean_broad": q0,
                             "mean_specialized": q1, "smd": (q1 - q0) / scale if scale else 0.0})
        for covariate in CATEGORICAL_COVARIATES:
            series = frame[covariate]
            levels = list(pd.unique(series.dropna()))
            if series.isna().any():
                levels.append(None)
            for level in levels:
                values = (series.isna() if level is None else series.eq(level)).to_numpy(dtype=float)
                m0, v0 = weighted_moments(values[treatment == 0], stage_weights[treatment == 0])
                m1, v1 = weighted_moments(values[treatment == 1], stage_weights[treatment == 1])
                scale = math.sqrt((v0 + v1) / 2)
                label = "missing" if level is None else str(level)
                rows.append({"exposure": exposure, "stage": stage,
                             "covariate": f"{covariate}={label}", "mean_broad": m0,
                             "mean_specialized": m1, "smd": (m1 - m0) / scale if scale else 0.0})
    return rows


def clustered_interval(signal, codes, journals, estimate, multipliers):
    centered = signal - estimate
    sums = np.bincount(codes, weights=centered)
    n = len(signal)
    g = len(journals)
    if g < 2:
        raise ValueError(f"expected >=2 journal clusters, got {g}")
    se = math.sqrt((g / (g - 1)) * np.sum(sums ** 2) / (n ** 2))
    draws = estimate + multipliers @ sums / n
    return se, estimate - 1.96 * se, estimate + 1.96 * se, \
        float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def fit_exposure(frame, treatment_col, outcomes, exposure, rng):
    treatment = frame[treatment_col].to_numpy(dtype=np.int8)
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError(f"expected {exposure} arms 0 and 1, got {np.unique(treatment)}")
    x = frame[COVARIATES].copy()
    for column in CATEGORICAL_COVARIATES:
        x[column] = x[column].astype("category")
    propensity = np.empty(len(frame), dtype=np.float32)
    predictions = {outcome: [np.empty(len(frame), dtype=np.float32),
                             np.empty(len(frame), dtype=np.float32)] for outcome in outcomes}
    folds = frame.fold.to_numpy()
    for fold in range(5):
        train = folds != fold
        test = ~train
        if test.sum() == 0:
            raise ValueError(f"expected held-out rows for fold {fold}, got 0")
        classifier = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED + fold, n_jobs=32, verbosity=-1,
        )
        classifier.fit(x.loc[train], treatment[train], categorical_feature="auto")
        propensity[test] = classifier.predict_proba(x.loc[test])[:, 1]
        for outcome in outcomes:
            binary = outcome.startswith("any_")
            for arm in (0, 1):
                arm_train = train & (treatment == arm)
                model_class = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
                model = model_class(
                    objective="binary" if binary else "poisson", n_estimators=300,
                    learning_rate=0.05, num_leaves=31, min_child_samples=100,
                    subsample=0.8, colsample_bytree=0.8, random_state=SEED + 10 * fold + arm,
                    n_jobs=32, verbosity=-1,
                )
                model.fit(x.loc[arm_train], frame.loc[arm_train, outcome], categorical_feature="auto")
                pred = model.predict_proba(x.loc[test])[:, 1] if binary else model.predict(x.loc[test])
                predictions[outcome][arm][test] = pred
        log(f"{exposure} cross-fit fold={fold} train={train.sum():,} test={test.sum():,}")
    support = np.isfinite(propensity) & (propensity >= 0.05) & (propensity <= 0.95)
    retention = float(support.mean())
    if support.sum() == 0:
        raise ValueError(f"expected {exposure} common support rows, got 0")
    d = frame.loc[support].reset_index(drop=True)
    a = treatment[support]
    p = propensity[support].astype(float)
    ipw = a / p + (1 - a) / (1 - p)
    balance = balance_table(d, a, ipw, exposure)
    journal_codes, journals = pd.factorize(d.journal_id, sort=True)
    multipliers = rng.standard_normal((500, len(journals)))
    rows = []
    arm_signals = {}
    for outcome in outcomes:
        y = d[outcome].to_numpy(dtype=float)
        m0 = predictions[outcome][0][support].astype(float)
        m1 = predictions[outcome][1][support].astype(float)
        psi0 = m0 + (1 - a) * (y - m0) / (1 - p)
        psi1 = m1 + a * (y - m1) / p
        mu0, mu1 = psi0.mean(), psi1.mean()
        diff_signal = psi1 - psi0
        difference = diff_signal.mean()
        se, low, high, boot_low, boot_high = clustered_interval(
            diff_signal, journal_codes, journals, difference, multipliers,
        )
        rq = "RQ1" if outcome == "total_citations" else "RQ2"
        rows.append({"rq": rq,
                     "estimand": "common-support ATE", "population": exposure,
                     "exposure": exposure, "outcome": outcome, "scale": "absolute",
                     "mean_broad": mu0, "mean_specialized": mu1, "estimate": difference,
                     "se": se, "ci_low": low, "ci_high": high,
                     "bootstrap_ci_low": boot_low, "bootstrap_ci_high": boot_high,
                     "n": len(d), "journals": d.journal_id.nunique(), "support": retention})
        if mu0 > 0:
            ratio = mu1 / mu0
            ratio_signal = (psi1 - mu1) / mu0 - (mu1 / mu0 ** 2) * (psi0 - mu0) + ratio
            se, low, high, boot_low, boot_high = clustered_interval(
                100 * (ratio_signal - 1), journal_codes, journals,
                100 * (ratio - 1), multipliers,
            )
            rows.append({"rq": rq,
                         "estimand": "common-support ATE", "population": exposure,
                         "exposure": exposure, "outcome": outcome, "scale": "relative_percent",
                         "mean_broad": mu0, "mean_specialized": mu1,
                         "estimate": 100 * (ratio - 1), "se": se, "ci_low": low,
                         "ci_high": high, "bootstrap_ci_low": boot_low,
                         "bootstrap_ci_high": boot_high, "n": len(d),
                         "journals": d.journal_id.nunique(), "support": retention})
        arm_signals[outcome] = (psi0, psi1)
    for near, far, label in [
        ("near", "far", "far_to_near_routing"),
        ("near_winsorized", "far_winsorized", "far_to_near_routing_winsorized"),
    ]:
        near0, near1 = arm_signals[near]
        far0, far1 = arm_signals[far]
        means = [far1.mean(), near1.mean(), far0.mean(), near0.mean()]
        if min(means) <= 0:
            raise ValueError(f"expected positive routing means for {label}, got {means}")
        estimate = math.log(means[0]) - math.log(means[1]) - math.log(means[2]) + math.log(means[3])
        influence = ((far1 - means[0]) / means[0] - (near1 - means[1]) / means[1]
                     - (far0 - means[2]) / means[2] + (near0 - means[3]) / means[3])
        signal = estimate + influence
        se, low, high, boot_low, boot_high = clustered_interval(
            signal, journal_codes, journals, estimate, multipliers,
        )
        rows.append({
            "rq": "RQ2", "estimand": "common-support log routing ratio",
            "population": exposure, "exposure": exposure, "outcome": label,
            "scale": "log_ratio_of_mean_ratios", "mean_broad": means[2] / means[3],
            "mean_specialized": means[0] / means[1], "estimate": estimate,
            "se": se, "ci_low": low, "ci_high": high,
            "bootstrap_ci_low": boot_low, "bootstrap_ci_high": boot_high,
            "n": len(d), "journals": len(journals), "support": retention,
        })
    return rows, balance, propensity, support, ipw, d, arm_signals


def main():
    if not SECTION_B_FROZEN:
        raise RuntimeError("Section B outcome and inference contract is not frozen")
    validate_snapshot()
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    exposure_run = json.loads((ARTIFACTS / "run_exposure.json").read_text())
    outcome_run = json.loads((ARTIFACTS / "run_outcome_embed.json").read_text())
    if exposure_run["status"] != "complete" or not exposure_run["extra"]["measurement_gate"]:
        raise RuntimeError(f"expected completed Section A measurement gate, got {exposure_run}")
    if outcome_run["status"] != "complete" or outcome_run["extra"]["model_commit"] != \
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3":
        raise RuntimeError(f"expected pinned completed Qwen3 stage, got {outcome_run}")
    con = connect()
    edge_qc = con.execute(f"""
        SELECT count(*),count(DISTINCT (citing_id,cited_id))
        FROM read_parquet('{path_glob(EDGES)}')
    """).fetchone()
    if edge_qc[0] != edge_qc[1]:
        raise ValueError(f"expected unique citation edges, got {edge_qc}")
    window_errors = con.execute(f"""
        SELECT count(*) FROM read_parquet('{path_glob(EDGES)}') e
        JOIN read_parquet('{path_glob(FOCAL)}') f ON e.cited_id=f.id
        WHERE e.citing_year NOT BETWEEN f.publication_year AND f.publication_year+4
    """).fetchone()[0]
    if window_errors:
        raise ValueError(f"expected zero citation-window violations, got {window_errors}")
    outcomes_n = build_outcomes(con)
    bad_decomposition = con.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE total_citations<>near+intermediate+far+unclassified",
        [str(OUTCOMES)],
    ).fetchone()[0]
    if bad_decomposition:
        raise ValueError(f"expected zero citation decomposition failures, got {bad_decomposition}")
    analysis_n = build_unestimated(con)
    frame = con.execute("SELECT * FROM read_parquet(?)", [str(UNESTIMATED)]).df()
    if len(frame) != analysis_n:
        raise ValueError(f"expected analysis rows={analysis_n}, got {len(frame)}")
    required = NUMERIC_COVARIATES + ["publication_year", "publication_month", "semantic_cluster"] + OUTCOME_COLS
    if frame[required].isna().any().any():
        missing = frame[required].isna().sum()
        raise ValueError(f"analysis variables contain missing values: {missing[missing.gt(0)].to_dict()}")
    extreme = frame[frame.treatment.notna()].copy().reset_index(drop=True)
    extreme["treatment"] = extreme.treatment.astype("int8")
    ood_rates = {int(arm): float(rate) for arm, rate in
                 extreme.groupby("treatment").focal_ood.mean().items()}
    if set(ood_rates) != {0, 1}:
        raise ValueError(f"expected focal OOD rates for both arms, got {ood_rates}")
    primary_frame = extreme[~extreme.focal_ood].copy().reset_index(drop=True)
    primary_frame["treatment"] = primary_frame.treatment.astype("int8")
    caps = {name: float(primary_frame[name].quantile(0.999)) for name in ("near", "far")}
    for name in ("near", "far"):
        primary_frame[f"{name}_winsorized"] = primary_frame[name].clip(upper=caps[name])
    modeled_outcomes = OUTCOME_COLS + ["near_winsorized", "far_winsorized"]
    estimates, balance, propensity, support, ipw, supported, _ = fit_exposure(
        primary_frame, "treatment", modeled_outcomes, "semantic_title", np.random.default_rng(SEED),
    )
    coverage = {}
    audience_qc = {}
    for arm in (0, 1):
        arm_frame = primary_frame[primary_frame.treatment.eq(arm)]
        total = float(arm_frame.total_citations.sum())
        if total <= 0:
            raise ValueError(f"expected positive external citations in arm {arm}, got {total}")
        coverage[arm] = float((arm_frame.near + arm_frame.intermediate + arm_frame.far).sum() / total)
        audience_qc[arm] = {
            "english": float(1 - arm_frame.unclassified_language.sum() / total),
            "missing_title": float(arm_frame.unclassified_missing_title.sum() / total),
            "ood": float(arm_frame.unclassified_ood.sum() / total),
        }
    top_cut = float(primary_frame.total_citations.quantile(0.999))
    total_flow = float(primary_frame.total_citations.sum())
    top_share = float(primary_frame.loc[primary_frame.total_citations.ge(top_cut), "total_citations"].sum() / total_flow)
    primary_frame["propensity"] = propensity
    primary_frame["common_support"] = support
    primary_frame["ipw"] = np.nan
    primary_frame.loc[support, "ipw"] = ipw.astype(np.float32)
    frame = frame.merge(primary_frame[["id", "propensity", "common_support", "ipw"]],
                        on="id", how="left", validate="one_to_one")
    reset_output(ANALYSIS)
    frame.to_parquet(ANALYSIS, index=False, compression="zstd")
    estimates = pd.DataFrame(estimates)
    balance = pd.DataFrame(balance)
    max_smd = float(balance[balance.stage.eq("weighted")].smd.abs().max())
    routing = estimates.loc[estimates.outcome.eq("far_to_near_routing")].iloc[0]
    gate_rows = [
        {"gate": "section_a_measurement", "value": float(exposure_run["extra"]["reliability"]["spearman_brown"]),
         "threshold": ">=0.70", "passed": exposure_run["extra"]["reliability"]["spearman_brown"] >= 0.70},
        {"gate": "common_support", "value": float(support.mean()), "threshold": ">=0.50",
         "passed": float(support.mean()) >= 0.50},
        {"gate": "weighted_max_abs_smd", "value": max_smd, "threshold": "<0.10", "passed": max_smd < 0.10},
        {"gate": "focal_ood_arm_difference", "value": abs(ood_rates[1] - ood_rates[0]),
         "threshold": "<=0.02", "passed": abs(ood_rates[1] - ood_rates[0]) <= 0.02},
        {"gate": "broad_citing_coverage", "value": coverage[0], "threshold": ">=0.80", "passed": coverage[0] >= 0.80},
        {"gate": "specialized_citing_coverage", "value": coverage[1], "threshold": ">=0.80", "passed": coverage[1] >= 0.80},
        {"gate": "primary_direction", "value": float(routing.estimate), "threshold": "<0",
         "passed": bool(routing.estimate < 0)},
    ]
    for row in gate_rows:
        row["passed"] = bool(row["passed"])
    inferential_ok = all(row["passed"] for row in gate_rows[:-1])
    wording = "exploratory_effect" if inferential_ok else "exploratory_association"
    estimates["diagnostic_status"] = wording
    estimates.to_csv(RESULTS / "dirty_estimates.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(RESULTS / "dirty_gates.csv", index=False)
    near = estimates[(estimates.outcome.eq("near")) & estimates.scale.eq("absolute")].iloc[0]
    far = estimates[(estimates.outcome.eq("far")) & estimates.scale.eq("absolute")].iloc[0]
    report = [
        "EXPLORATORY EFFECT ESTIMATE" if inferential_ok else "EXPLORATORY ASSOCIATION ONLY",
        "", "# QSS v2 dirty Qwen3 report", "", "## Primary routing estimate", "",
        f"- theta: {routing.estimate:.6f} (95% CI {routing.ci_low:.6f} to {routing.ci_high:.6f}; "
        f"multiplier bootstrap {routing.bootstrap_ci_low:.6f} to {routing.bootstrap_ci_high:.6f})",
        f"- Broad far/near marginal mean ratio: {routing.mean_broad:.6f}",
        f"- Specialized far/near marginal mean ratio: {routing.mean_specialized:.6f}",
        f"- Near means, broad/specialized: {near.mean_broad:.6f} / {near.mean_specialized:.6f}",
        f"- Far means, broad/specialized: {far.mean_broad:.6f} / {far.mean_specialized:.6f}",
        "", "## Counts and diagnostics", "",
        f"- Candidate rows: {analysis_n:,}", f"- Extreme-arm rows before focal OOD exclusion: {len(extreme):,}",
        f"- Primary rows after focal OOD exclusion: {len(primary_frame):,}",
        f"- Common-support rows: {int(support.sum()):,}", f"- Journals: {supported.journal_id.nunique():,}",
        f"- Focal OOD, broad/specialized: {ood_rates[0]:.4%} / {ood_rates[1]:.4%}",
        f"- Classified citing flow, broad/specialized: {coverage[0]:.4%} / {coverage[1]:.4%}",
        f"- English citing flow, broad/specialized: {audience_qc[0]['english']:.4%} / "
        f"{audience_qc[1]['english']:.4%}",
        f"- English missing-title flow, broad/specialized: {audience_qc[0]['missing_title']:.4%} / "
        f"{audience_qc[1]['missing_title']:.4%}",
        f"- English OOD flow, broad/specialized: {audience_qc[0]['ood']:.4%} / "
        f"{audience_qc[1]['ood']:.4%}",
        f"- Maximum weighted absolute SMD: {max_smd:.6f}",
        f"- Top 0.1% threshold: {top_cut:.0f} citations; observed contribution: {top_share:.4%}",
        "", "## Gates", "",
        *[f"- {row['gate']}: {'PASS' if row['passed'] else 'FAIL'} ({row['value']}; {row['threshold']})"
          for row in gate_rows],
        "", "This is the prespecified dirty direction-finding run, not the human-validated final QSS analysis.", "",
    ]
    (RESULTS / "dirty_report.md").write_text("\n".join(report))
    free = shutil.disk_usage(GROUP_ROOT).free
    write_run("dirty_analyze", "complete", {
        "citation_edges": edge_qc[0], "citation_outcomes": outcomes_n, "analysis": analysis_n,
        "extreme_arm": len(extreme), "primary_after_focal_ood": len(primary_frame),
        "support": int(support.sum()), "journals": int(supported.journal_id.nunique()),
    }, {
        "qwen3_model_commit": outcome_run["extra"]["model_commit"], "winsor_caps": caps,
        "focal_ood_rates": ood_rates, "citing_coverage": coverage, "audience_qc": audience_qc,
        "top_0_1_percent_threshold": top_cut, "top_0_1_percent_flow_share": top_share,
        "max_weighted_smd": max_smd, "gates": gate_rows, "wording": wording,
        "qss_work_bytes": tree_bytes(QSS_WORK), "group_free_bytes": free,
    })
    check_budget()
    log(f"dirty analysis complete rows={analysis_n:,} support={support.sum():,} "
        f"theta={routing.estimate:.6f} max_smd={max_smd:.6f} wording={wording} "
        f"work={tree_bytes(QSS_WORK):,} free={free:,}")


if __name__ == "__main__":
    main()
