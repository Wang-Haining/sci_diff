#!/usr/bin/env python3
import math
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
    ARTIFACTS, EMBED_DIM, QSS_WORK, RESULTS, SECTION_B_FROZEN, SEED, STAGED_INPUT, check_budget,
    connect, copy_query, log, reset_output, validate_snapshot, write_run,
)

TITLE = QSS_WORK / "embeddings_title"
TITLE_ABSTRACT = QSS_WORK / "embeddings_title_abstract"
EMBED_INPUT = STAGED_INPUT
FOCAL = QSS_WORK / "focal_base.parquet"
EDGES = QSS_WORK / "citation_edges"
CITING = QSS_WORK / "citing_metadata"
SCOPE = QSS_WORK / "journal_year_scope.parquet"
SEMANTICS = QSS_WORK / "focal_semantics.parquet"
OUTCOMES = QSS_WORK / "citation_outcomes.parquet"
UNESTIMATED = QSS_WORK / "analysis_unestimated.parquet"
ANALYSIS = QSS_WORK / "analysis_dataset.parquet"

PC_COLS = [f"pc{i:02d}" for i in range(1, 33)]
NUMERIC_COVARIATES = PC_COLS + [
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
    "total_citations", "within_subfield", "cross_subfield", "cross_field",
    "unclassified", "any_citation", "any_cross_field", "total_citations_2y",
    "cross_field_2y", "semantic_diffusion_mass", "all_unique_citations",
]
PRIMARY_OUTCOMES = [
    "total_citations", "within_subfield", "cross_subfield", "cross_field",
    "unclassified", "any_citation", "any_cross_field", "total_citations_2y",
    "cross_field_2y", "semantic_diffusion_mass", "all_unique_citations",
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
          SELECT e.citing_id,e.cited_id,e.citing_year,
                 e.citing_year<=f.publication_year+1 AS two_year,
                 c.journal_id=f.journal_id AS same_journal,
                 COALESCE(list_has_any(c.author_ids,f.author_ids),false) AS shared_author,
                 CASE WHEN c.subfield_id IS NULL OR f.subfield_id IS NULL OR
                                c.field_id IS NULL OR f.field_id IS NULL THEN 'unclassified'
                      WHEN c.subfield_id=f.subfield_id THEN 'within_subfield'
                      WHEN c.field_id=f.field_id THEN 'cross_subfield'
                      ELSE 'cross_field' END AS citation_class,
                 CASE WHEN ce.id IS NOT NULL THEN greatest(0,1-list_inner_product(fe.embedding,ce.embedding)) END AS semantic_distance
          FROM read_parquet('{path_glob(EDGES)}') e
          JOIN read_parquet('{path_glob(FOCAL)}') f ON e.cited_id=f.id
          JOIN read_parquet('{path_glob(CITING)}') c ON e.citing_id=c.id
          JOIN read_parquet('{path_glob(TITLE)}') fe ON f.id=fe.id
          LEFT JOIN read_parquet('{path_glob(TITLE)}') ce ON c.id=ce.id
        ), external AS (
          SELECT * FROM labeled WHERE NOT same_journal AND NOT shared_author
        ), totals AS (
          SELECT cited_id, count(*) AS total_citations,
                 count(*) FILTER (WHERE citation_class='within_subfield') AS within_subfield,
                 count(*) FILTER (WHERE citation_class='cross_subfield') AS cross_subfield,
                 count(*) FILTER (WHERE citation_class='cross_field') AS cross_field,
                 count(*) FILTER (WHERE citation_class='unclassified') AS unclassified,
                 count(*) FILTER (WHERE two_year) AS total_citations_2y,
                 count(*) FILTER (WHERE two_year AND citation_class='cross_field') AS cross_field_2y,
                 sum(semantic_distance) AS semantic_diffusion_mass,
                 avg(semantic_distance) AS mean_semantic_distance,
                 count(semantic_distance) AS semantic_citing_n
          FROM external GROUP BY cited_id
        ), all_edges AS (
          SELECT cited_id,count(*) AS all_unique_citations,
                 count(*) FILTER (WHERE same_journal) AS same_journal_citations,
                 count(*) FILTER (WHERE shared_author) AS shared_author_citations
          FROM labeled GROUP BY cited_id
        )
        SELECT f.id,
               COALESCE(t.total_citations,0) AS total_citations,
               COALESCE(t.within_subfield,0) AS within_subfield,
               COALESCE(t.cross_subfield,0) AS cross_subfield,
               COALESCE(t.cross_field,0) AS cross_field,
               COALESCE(t.unclassified,0) AS unclassified,
               COALESCE(t.total_citations_2y,0) AS total_citations_2y,
               COALESCE(t.cross_field_2y,0) AS cross_field_2y,
               COALESCE(t.semantic_diffusion_mass,0) AS semantic_diffusion_mass,
               t.mean_semantic_distance,t.semantic_citing_n,
               COALESCE(a.all_unique_citations,0) AS all_unique_citations,
               COALESCE(a.same_journal_citations,0) AS same_journal_citations,
               COALESCE(a.shared_author_citations,0) AS shared_author_citations,
               (COALESCE(t.total_citations,0)>0)::UTINYINT AS any_citation,
               (COALESCE(t.cross_field,0)>0)::UTINYINT AS any_cross_field
        FROM read_parquet('{path_glob(FOCAL)}') f
        LEFT JOIN totals t ON f.id=t.cited_id LEFT JOIN all_edges a ON f.id=a.cited_id
    """)


def build_unestimated(con):
    pc = ",".join(f"s.{name}" for name in PC_COLS)
    query = f"""
        WITH candidate AS (
          SELECT f.id,f.work_type,f.has_abstract,f.publication_year,f.publication_month,
                 f.journal_id,f.journal_name,f.lead_country,
                 f.topic_id,f.subfield_id,f.field_id,f.reference_count,f.authors_count,
                 f.countries_count,f.institutions_count,s.semantic_cluster,{pc},
                 r.classified_n,r.reference_fields,
                 r.reference_entropy,r.reference_hhi,
                 COALESCE(a.known_authors,0) AS known_authors,
                 COALESCE(a.author_mean_prior_works,0) AS author_mean_prior_works,
                 COALESCE(a.author_max_prior_works,0) AS author_max_prior_works,
                 COALESCE(a.author_mean_prior_citations,0) AS author_mean_prior_citations,
                 COALESCE(a.author_max_prior_citations,0) AS author_max_prior_citations,
                 COALESCE(i.known_institutions,0) AS known_institutions,
                 COALESCE(i.institution_mean_prior_works,0) AS institution_mean_prior_works,
                 COALESCE(i.institution_max_prior_works,0) AS institution_max_prior_works,
                 COALESCE(i.institution_mean_prior_citations,0) AS institution_mean_prior_citations,
                 COALESCE(i.institution_max_prior_citations,0) AS institution_max_prior_citations,
                 (f.countries_count>1)::UTINYINT AS international,
                 j.history_n,j.prior_oa_share,j.prior_english_share,j.prior_prestige,
                 j.topic_hhi,j.topic_entropy,j.semantic_title_similarity,j.semantic_title_n,
                 j.semantic_abstract_similarity,j.semantic_abstract_n,
                 j.reference_field_hhi,j.reference_field_entropy,
                 o.* EXCLUDE(id), hash(f.journal_id)%5 AS fold
          FROM read_parquet('{path_glob(FOCAL)}') f
          JOIN read_parquet('{path_glob(SEMANTICS)}') s USING (id)
          JOIN read_parquet('{path_glob(SCOPE)}') j
            ON f.journal_id=j.journal_id AND f.publication_year=j.focal_year
          JOIN read_parquet('{path_glob(QSS_WORK / 'focal_reference_metrics.parquet')}') r USING (id)
          LEFT JOIN read_parquet('{path_glob(QSS_WORK / 'focal_author_history.parquet')}') a USING (id)
          LEFT JOIN read_parquet('{path_glob(QSS_WORK / 'focal_institution_history.parquet')}') i USING (id)
          JOIN read_parquet('{path_glob(OUTCOMES)}') o USING (id)
          WHERE j.semantic_reliable
        ), choices AS (
          SELECT semantic_cluster,publication_year,journal_id,
                 any_value(semantic_title_similarity) AS semantic_title_similarity,
                 any_value(semantic_abstract_similarity) AS semantic_abstract_similarity,
                 any_value(semantic_abstract_n) AS semantic_abstract_n,
                 any_value(reference_field_hhi) AS reference_field_hhi,
                 any_value(topic_hhi) AS topic_hhi,any_value(topic_entropy) AS topic_entropy
          FROM candidate WHERE work_type='article' GROUP BY ALL
        ), semantic_ranks AS (
          SELECT *, ntile(4) OVER (PARTITION BY semantic_cluster,publication_year
                                  ORDER BY semantic_title_similarity,journal_id) AS semantic_q
          FROM choices
        ), reference_ranks AS (
          SELECT semantic_cluster,publication_year,journal_id,
                 ntile(4) OVER (PARTITION BY semantic_cluster,publication_year
                                ORDER BY reference_field_hhi,journal_id) AS reference_q
          FROM choices WHERE reference_field_hhi IS NOT NULL
        ), abstract_ranks AS (
          SELECT semantic_cluster,publication_year,journal_id,
                 ntile(4) OVER (PARTITION BY semantic_cluster,publication_year
                                ORDER BY semantic_abstract_similarity,journal_id) AS abstract_q
          FROM choices WHERE semantic_abstract_n>=100
        ), topic_ranks AS (
          SELECT semantic_cluster,publication_year,journal_id,
                 ntile(4) OVER (PARTITION BY semantic_cluster,publication_year
                                ORDER BY topic_hhi,journal_id) AS topic_hhi_q,
                 ntile(4) OVER (PARTITION BY semantic_cluster,publication_year
                                ORDER BY topic_entropy,journal_id) AS topic_entropy_q
          FROM choices WHERE topic_hhi IS NOT NULL AND topic_entropy IS NOT NULL
        ), assigned AS (
          SELECT c.*,CASE WHEN semantic_q=1 THEN 0 WHEN semantic_q=4 THEN 1 END AS treatment,
                     CASE WHEN reference_field_hhi IS NOT NULL AND reference_q=1 THEN 0
                          WHEN reference_field_hhi IS NOT NULL AND reference_q=4 THEN 1 END AS reference_treatment,
                     CASE WHEN abstract_q=1 THEN 0 WHEN abstract_q=4 THEN 1 END AS abstract_treatment,
                     CASE WHEN topic_hhi_q=1 THEN 0 WHEN topic_hhi_q=4 THEN 1 END AS topic_hhi_treatment,
                     CASE WHEN topic_entropy_q=1 THEN 1 WHEN topic_entropy_q=4 THEN 0 END AS topic_entropy_treatment
          FROM candidate c JOIN semantic_ranks USING (semantic_cluster,publication_year,journal_id)
          LEFT JOIN reference_ranks USING (semantic_cluster,publication_year,journal_id)
          LEFT JOIN abstract_ranks USING (semantic_cluster,publication_year,journal_id)
          LEFT JOIN topic_ranks USING (semantic_cluster,publication_year,journal_id)
        ), arm_counts AS (
          SELECT semantic_cluster,publication_year,treatment,count(*) AS papers,
                 count(DISTINCT journal_id) AS journals
          FROM assigned WHERE treatment IS NOT NULL AND work_type='article' GROUP BY ALL
        ), valid AS (
          SELECT semantic_cluster,publication_year FROM arm_counts GROUP BY ALL
          HAVING count(*)=2 AND min(papers)>=20 AND min(journals)>=2
        ), ref_arm_counts AS (
          SELECT semantic_cluster,publication_year,reference_treatment,count(*) AS papers,
                 count(DISTINCT journal_id) AS journals
          FROM assigned WHERE reference_treatment IS NOT NULL AND work_type='article' GROUP BY ALL
        ), ref_valid AS (
          SELECT semantic_cluster,publication_year FROM ref_arm_counts GROUP BY ALL
          HAVING count(*)=2 AND min(papers)>=20 AND min(journals)>=2
        ), abstract_arm_counts AS (
          SELECT semantic_cluster,publication_year,abstract_treatment,count(*) AS papers,
                 count(DISTINCT journal_id) AS journals FROM assigned
          WHERE abstract_treatment IS NOT NULL AND work_type='article' AND has_abstract GROUP BY ALL
        ), abstract_valid AS (
          SELECT semantic_cluster,publication_year FROM abstract_arm_counts GROUP BY ALL
          HAVING count(*)=2 AND min(papers)>=20 AND min(journals)>=2
        ), topic_hhi_arm_counts AS (
          SELECT semantic_cluster,publication_year,topic_hhi_treatment,count(*) AS papers,
                 count(DISTINCT journal_id) AS journals FROM assigned
          WHERE topic_hhi_treatment IS NOT NULL AND work_type='article' GROUP BY ALL
        ), topic_hhi_valid AS (
          SELECT semantic_cluster,publication_year FROM topic_hhi_arm_counts GROUP BY ALL
          HAVING count(*)=2 AND min(papers)>=20 AND min(journals)>=2
        ), topic_entropy_arm_counts AS (
          SELECT semantic_cluster,publication_year,topic_entropy_treatment,count(*) AS papers,
                 count(DISTINCT journal_id) AS journals FROM assigned
          WHERE topic_entropy_treatment IS NOT NULL AND work_type='article' GROUP BY ALL
        ), topic_entropy_valid AS (
          SELECT semantic_cluster,publication_year FROM topic_entropy_arm_counts GROUP BY ALL
          HAVING count(*)=2 AND min(papers)>=20 AND min(journals)>=2
        )
        SELECT a.* EXCLUDE(reference_treatment,abstract_treatment,topic_hhi_treatment,topic_entropy_treatment),
               CASE WHEN rv.semantic_cluster IS NOT NULL THEN a.reference_treatment END AS reference_treatment,
               CASE WHEN av.semantic_cluster IS NOT NULL AND a.has_abstract THEN a.abstract_treatment END AS abstract_treatment,
               CASE WHEN hv.semantic_cluster IS NOT NULL THEN a.topic_hhi_treatment END AS topic_hhi_treatment,
               CASE WHEN ev.semantic_cluster IS NOT NULL THEN a.topic_entropy_treatment END AS topic_entropy_treatment
        FROM assigned a JOIN valid USING (semantic_cluster,publication_year)
        LEFT JOIN ref_valid rv USING (semantic_cluster,publication_year)
        LEFT JOIN abstract_valid av USING (semantic_cluster,publication_year)
        LEFT JOIN topic_hhi_valid hv USING (semantic_cluster,publication_year)
        LEFT JOIN topic_entropy_valid ev USING (semantic_cluster,publication_year)
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


def clustered_interval(signal, journal_ids, estimate, rng):
    codes, unique = pd.factorize(journal_ids, sort=True)
    centered = signal - estimate
    sums = np.bincount(codes, weights=centered)
    n = len(signal)
    g = len(unique)
    if g < 2:
        raise ValueError(f"expected >=2 journal clusters, got {g}")
    se = math.sqrt((g / (g - 1)) * np.sum(sums ** 2) / (n ** 2))
    draws = estimate + rng.standard_normal((500, g)) @ sums / n
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
    rows = []
    signals = {}
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
            diff_signal, d.journal_id, difference, rng,
        )
        rq = "RQ1" if outcome in ("total_citations", "total_citations_2y",
                                  "all_unique_citations", "any_citation") else "RQ2"
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
                100 * (ratio_signal - 1), d.journal_id, 100 * (ratio - 1), rng,
            )
            rows.append({"rq": rq,
                         "estimand": "common-support ATE", "population": exposure,
                         "exposure": exposure, "outcome": outcome, "scale": "relative_percent",
                         "mean_broad": mu0, "mean_specialized": mu1,
                         "estimate": 100 * (ratio - 1), "se": se, "ci_low": low,
                         "ci_high": high, "bootstrap_ci_low": boot_low,
                         "bootstrap_ci_high": boot_high, "n": len(d),
                         "journals": d.journal_id.nunique(), "support": retention})
        signals[outcome] = diff_signal
    contrast = signals["cross_field"] - signals["within_subfield"]
    estimate = contrast.mean()
    se, low, high, boot_low, boot_high = clustered_interval(contrast, d.journal_id, estimate, rng)
    rows.append({"rq": "RQ2", "estimand": "contrast of common-support ATEs",
                 "population": exposure, "exposure": exposure,
                 "outcome": "cross_field_minus_within_subfield", "scale": "absolute",
                 "mean_broad": np.nan, "mean_specialized": np.nan, "estimate": estimate,
                 "se": se, "ci_low": low, "ci_high": high,
                 "bootstrap_ci_low": boot_low, "bootstrap_ci_high": boot_high,
                 "n": len(d), "journals": d.journal_id.nunique(), "support": retention})
    return rows, balance, propensity, support, ipw, d, signals


def h3_rows(data, cross_signal, rng):
    values = data.reference_entropy.to_numpy(dtype=float)
    quartile = np.empty(len(data), dtype=np.int8)
    for year in sorted(data.publication_year.unique()):
        mask = data.publication_year.eq(year).to_numpy()
        cuts = np.quantile(values[mask], [0.25, 0.5, 0.75])
        quartile[mask] = np.searchsorted(cuts, values[mask], side="right") + 1
    rows = []
    for q in range(1, 5):
        mask = quartile == q
        estimate = cross_signal[mask].mean()
        se, low, high, boot_low, boot_high = clustered_interval(
            cross_signal[mask], data.loc[mask, "journal_id"], estimate, rng,
        )
        rows.append({"rq": "RQ3", "estimand": "subgroup common-support ATE",
                     "population": f"reference entropy quartile {q}", "exposure": "semantic_title",
                     "outcome": "cross_field", "scale": "absolute", "mean_broad": np.nan,
                     "mean_specialized": np.nan, "estimate": estimate, "se": se,
                     "ci_low": low, "ci_high": high, "bootstrap_ci_low": boot_low,
                     "bootstrap_ci_high": boot_high, "n": int(mask.sum()),
                     "journals": data.loc[mask, "journal_id"].nunique(), "support": 1.0})
    z = (values - values.mean()) / values.std()
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.linalg.solve(x.T @ x, x.T @ cross_signal)
    residual = cross_signal - x @ beta
    codes, journals = pd.factorize(data.journal_id, sort=True)
    scores = np.zeros((len(journals), 2))
    np.add.at(scores, codes, x * residual[:, None])
    bread = np.linalg.inv(x.T @ x)
    covariance = bread @ (scores.T @ scores) @ bread * len(journals) / (len(journals) - 1)
    se = math.sqrt(covariance[1, 1])
    rows.append({"rq": "RQ3", "estimand": "linear effect modification",
                 "population": "semantic-title common support", "exposure": "semantic_title",
                 "outcome": "cross_field", "scale": "per SD reference entropy",
                 "mean_broad": np.nan, "mean_specialized": np.nan, "estimate": beta[1],
                 "se": se, "ci_low": beta[1] - 1.96 * se, "ci_high": beta[1] + 1.96 * se,
                 "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan,
                 "n": len(data), "journals": len(journals), "support": 1.0})
    return rows


def residualize(values, weights, groups):
    result = values.astype(float).copy()
    for _ in range(100):
        before = result.copy()
        for group in groups:
            numerator = np.bincount(group, weights=result * weights)
            denominator = np.bincount(group, weights=weights)
            result -= (numerator / denominator)[group]
        if np.max(np.abs(result - before)) < 1e-10:
            break
    return result


def h4_row(frame):
    cells = frame.groupby(["journal_id", "semantic_cluster", "publication_year"], observed=True).agg(
        papers=("id", "size"), cross_field=("cross_field", "mean"),
        specialization=("semantic_title_similarity", "first"),
    ).reset_index()
    cells["specialization"] = (cells.specialization - cells.specialization.mean()) / cells.specialization.std()
    journal, journals = pd.factorize(cells.journal_id, sort=True)
    cluster_year, _ = pd.factorize(pd.MultiIndex.from_frame(cells[["semantic_cluster", "publication_year"]]), sort=True)
    weights = cells.papers.to_numpy(dtype=float)
    x = residualize(cells.specialization.to_numpy(), weights, [journal, cluster_year])
    y = residualize(cells.cross_field.to_numpy(), weights, [journal, cluster_year])
    denominator = np.sum(weights * x * x)
    if denominator <= 0:
        raise ValueError(f"expected positive within-journal specialization variance, got {denominator}")
    estimate = np.sum(weights * x * y) / denominator
    residual = y - estimate * x
    score = np.bincount(journal, weights=weights * x * residual)
    se = math.sqrt(len(journals) / (len(journals) - 1) * np.sum(score ** 2) / denominator ** 2)
    return {"rq": "RQ4", "estimand": "within-journal two-way fixed-effects association",
            "population": "journal-cluster-year cells", "exposure": "semantic_title_continuous",
            "outcome": "cross_field", "scale": "per within-sample SD", "mean_broad": np.nan,
            "mean_specialized": np.nan, "estimate": estimate, "se": se,
            "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
            "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan, "n": len(cells),
            "journals": len(journals), "support": 1.0}


def main():
    if not SECTION_B_FROZEN:
        raise RuntimeError("Section B outcome and inference contract is not frozen")
    validate_snapshot()
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    scope_n, reliability, abstract_reliability = build_scope(con)
    reset_output(TITLE_ABSTRACT)
    reset_output(EMBED_INPUT)
    check_budget()
    semantics_n, pca_variance, cluster_inertia = build_semantics(con)
    outcomes_n = build_outcomes(con)
    bad_decomposition = con.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE total_citations<>within_subfield+cross_subfield+cross_field+unclassified",
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
    rng = np.random.default_rng(SEED)
    primary_frame = frame[frame.treatment.notna() & frame.work_type.eq("article")].copy().reset_index(drop=True)
    primary_frame["treatment"] = primary_frame.treatment.astype("int8")
    estimates, balance, propensity, support, ipw, supported, signals = fit_exposure(
        primary_frame, "treatment", PRIMARY_OUTCOMES, "semantic_title", rng,
    )
    reference_frame = frame[frame.reference_treatment.notna() & frame.work_type.eq("article")].copy().reset_index(drop=True)
    reference_frame["reference_treatment"] = reference_frame.reference_treatment.astype("int8")
    ref_estimates, ref_balance, _, _, _, _, _ = fit_exposure(
        reference_frame, "reference_treatment", ["within_subfield", "cross_field"],
        "reference_field", rng,
    )
    estimates += ref_estimates
    balance += ref_balance
    sensitivity_counts = {}
    for treatment_col, exposure in [
        ("abstract_treatment", "semantic_title_abstract"),
        ("topic_hhi_treatment", "openalex_topic_hhi"),
        ("topic_entropy_treatment", "openalex_topic_entropy"),
    ]:
        sensitivity = frame[frame[treatment_col].notna() & frame.work_type.eq("article")].copy().reset_index(drop=True)
        sensitivity[treatment_col] = sensitivity[treatment_col].astype("int8")
        more_estimates, more_balance, _, _, _, _, _ = fit_exposure(
            sensitivity, treatment_col, ["within_subfield", "cross_field"], exposure, rng,
        )
        estimates += more_estimates
        balance += more_balance
        sensitivity_counts[exposure] = len(sensitivity)
    review_frame = frame[frame.treatment.notna() & frame.work_type.eq("review")].copy().reset_index(drop=True)
    review_frame["treatment"] = review_frame.treatment.astype("int8")
    review_estimates, review_balance, _, _, _, _, _ = fit_exposure(
        review_frame, "treatment", ["total_citations", "within_subfield", "cross_field"],
        "semantic_title_reviews", rng,
    )
    estimates += review_estimates
    balance += review_balance
    estimates += h3_rows(supported, signals["cross_field"], rng)
    estimates.append(h4_row(frame[frame.work_type.eq("article")]))

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
    max_smd = balance[balance.stage.eq("weighted")].groupby("exposure").smd.apply(lambda x: x.abs().max())
    primary = estimates[(estimates.exposure == "semantic_title") & (estimates.scale == "absolute")]
    reference = estimates[(estimates.exposure == "reference_field") & (estimates.scale == "absolute")]
    primary_cross = primary.loc[primary.outcome.eq("cross_field")].iloc[0]
    primary_contrast = primary.loc[primary.outcome.eq("cross_field_minus_within_subfield")].iloc[0]
    ref_cross = reference.loc[reference.outcome.eq("cross_field")].iloc[0]
    ref_contrast = reference.loc[reference.outcome.eq("cross_field_minus_within_subfield")].iloc[0]
    gate = {
        "causal_measurement_reliability": reliability["spearman_brown"] >= 0.70,
        "causal_common_support": primary_cross.support >= 0.50,
        "causal_covariate_balance": max_smd["semantic_title"] < 0.10,
        "causal_independent_measurement": (ref_cross.support >= 0.50 and
                                            max_smd["reference_field"] < 0.10),
        "claim_primary_direction": primary_cross.estimate < 0 and primary_contrast.estimate < 0,
        "claim_independent_replication": ref_cross.estimate < 0 and ref_contrast.estimate < 0,
    }
    causal_wording = all(value for key, value in gate.items() if key.startswith("causal_"))
    audience_claim = causal_wording and all(
        value for key, value in gate.items() if key.startswith("claim_")
    )
    estimates["diagnostic_status"] = "causal" if causal_wording else "association_only"
    estimates.to_csv(RESULTS / "causal_estimates.csv", index=False)
    balance.to_csv(RESULTS / "balance.csv", index=False)
    pd.DataFrame([{"gate": key, "passed": value} for key, value in gate.items()]).to_csv(
        RESULTS / "gates.csv", index=False,
    )
    for transient in (
        FOCAL, EDGES, CITING, SEMANTICS, OUTCOMES, UNESTIMATED, TITLE,
        QSS_WORK / "focal_reference_metrics.parquet",
        QSS_WORK / "focal_author_history.parquet",
        QSS_WORK / "focal_institution_history.parquet",
        QSS_WORK / "journal_year_baseline.parquet",
        QSS_WORK / "journal_year_reference_scope.parquet",
    ):
        reset_output(transient)
        log(f"removed consumed intermediate {transient}")
    write_run("analyze", "complete", {
        "journal_year_scope": scope_n, "focal_semantics": semantics_n,
        "citation_outcomes": outcomes_n, "analysis": analysis_n,
        "primary_articles": len(primary_frame), "reviews": len(review_frame),
        **sensitivity_counts,
        "support": int(support.sum()), "journals": int(frame.journal_id.nunique()),
    }, {"reliability": reliability, "abstract_reliability": abstract_reliability,
        "pca_variance": pca_variance, "cluster_inertia": cluster_inertia,
        "max_weighted_smd": max_smd.to_dict(), "gates": gate,
        "wording": "causal" if causal_wording else "association",
        "audience_segmentation_claim": audience_claim})
    check_budget()
    log(f"analysis complete rows={analysis_n:,} support={support.sum():,} "
        f"max_smd={max_smd.to_dict()} wording={'causal' if causal_wording else 'association'} "
        f"audience_claim={audience_claim}")


if __name__ == "__main__":
    main()
