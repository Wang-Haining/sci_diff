#!/usr/bin/env python3
import json
from pathlib import Path

from qss_common import (
    FOCAL_YEARS, QSS_TMP, QSS_WORK, REPO, SNAPSHOT, check_budget, connect,
    copy_query, log, validate_snapshot, write_run,
)

WORKS = str(SNAPSHOT / "works/updated_date=*/*.parquet")
AUTHORS = str(SNAPSHOT / "authors/updated_date=*/*.parquet")
INSTITUTIONS = str(SNAPSHOT / "institutions/updated_date=*/*.parquet")
PILOT_FILES = [
    REPO / "results/pilot_report.md", REPO / "results/pilot_summary.csv",
    REPO / "results/balance.csv", REPO / "results/openalex_manifest.json",
]


def parquet(path):
    return str(path / "*.parquet") if path.is_dir() else str(path)


def validate_frozen_pilot():
    missing = [str(path) for path in PILOT_FILES if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"expected nonempty frozen pilot files, got missing={missing}")
    decision = PILOT_FILES[0].read_text().splitlines()[0]
    if decision != "GO":
        raise ValueError(f"expected frozen pilot decision GO, got {decision}")
    diagnostics = {"status": "frozen_not_rerun", "decision": decision}
    log(f"validated separate frozen pilot artifacts: {diagnostics}")
    return diagnostics


def main():
    validate_snapshot()
    pilot_reproduction = validate_frozen_pilot()
    QSS_WORK.mkdir(parents=True, exist_ok=True)
    QSS_TMP.mkdir(parents=True, exist_ok=True)
    check_budget()
    con = connect()
    works = WORKS.replace("'", "''")
    authors = AUTHORS.replace("'", "''")
    institutions = INSTITUTIONS.replace("'", "''")
    counts = {}

    eligible = """
        NOT COALESCE(is_xpac, false)
        AND NOT COALESCE(is_retracted, false)
        AND primary_location.is_published
        AND primary_location.source.type='journal'
        AND primary_location.source.id IS NOT NULL
        AND title IS NOT NULL AND trim(title)<>''
    """
    list_expr = """
        list_filter(list_distinct(list_transform(authorships, a -> a.author.id)), x -> x IS NOT NULL)
    """.strip()
    inst_expr = """
        list_filter(list_distinct(flatten(list_transform(authorships,
          a -> list_transform(a.institutions, i -> i.id)))), x -> x IS NOT NULL)
    """.strip()
    country_expr = """
        list_filter(list_distinct(flatten(list_transform(authorships, a -> a.countries))),
                    x -> x IS NOT NULL)
    """.strip()
    lead_country_expr = """
        list_extract(list_filter(flatten(list_transform(authorships, a -> a.countries)),
                                 x -> x IS NOT NULL), 1)
    """.strip()

    focal = QSS_WORK / "focal_base.parquet"
    counts["focal"] = copy_query(con, focal, f"""
        SELECT id, type AS work_type, title,
               (abstract_inverted_index IS NOT NULL AND trim(abstract_inverted_index) NOT IN ('','{{}}')) AS has_abstract,
               publication_year,
               month(publication_date)::UTINYINT AS publication_month,
               primary_location.source.id AS journal_id,
               primary_location.source.display_name AS journal_name,
               primary_topic.id AS topic_id, primary_topic.subfield.id AS subfield_id,
               primary_topic.field.id AS field_id, referenced_works,
               referenced_works_count AS reference_count, authors_count,
               countries_distinct_count AS countries_count,
               institutions_distinct_count AS institutions_count,
               {list_expr} AS author_ids, {inst_expr} AS institution_ids,
               {country_expr} AS country_codes, {lead_country_expr} AS lead_country
        FROM read_parquet('{works}')
        WHERE publication_year BETWEEN 2015 AND 2020 AND language='en'
          AND type IN ('article','review') AND {eligible}
    """)

    history = QSS_WORK / "history_base.parquet"
    counts["history"] = copy_query(con, history, f"""
        SELECT id, CASE WHEN language='en' THEN title END AS title,
               CASE WHEN language='en' THEN abstract_inverted_index END AS abstract_inverted_index,
               publication_year,
               primary_location.source.id AS journal_id,
               primary_topic.id AS topic_id, primary_topic.field.id AS field_id,
               referenced_works, COALESCE(open_access.is_oa, false) AS is_oa,
               language, counts_by_year
        FROM read_parquet('{works}')
        WHERE publication_year BETWEEN 2012 AND 2019 AND type='article' AND {eligible}
    """)

    reference_ids = QSS_WORK / "reference_ids.parquet"
    counts["reference_ids"] = copy_query(con, reference_ids, f"""
        SELECT DISTINCT ref_id FROM (
          SELECT unnest(list_distinct(referenced_works)) AS ref_id FROM read_parquet('{parquet(focal)}')
          UNION ALL
          SELECT unnest(list_distinct(referenced_works)) AS ref_id FROM read_parquet('{parquet(history)}')
        ) WHERE ref_id IS NOT NULL
    """, per_thread=True)

    lookup = QSS_WORK / "reference_field_lookup.parquet"
    counts["reference_lookup"] = copy_query(con, lookup, f"""
        SELECT w.id, w.primary_topic.id AS topic_id,
               w.primary_topic.subfield.id AS subfield_id,
               w.primary_topic.field.id AS field_id
        FROM read_parquet('{works}') w
        JOIN read_parquet('{parquet(reference_ids)}') r ON w.id=r.ref_id
    """, per_thread=True)

    focal_ref = QSS_WORK / "focal_reference_metrics.parquet"
    counts["focal_reference_metrics"] = copy_query(con, focal_ref, f"""
        WITH edges AS (
          SELECT f.id, f.reference_count, l.field_id
          FROM read_parquet('{parquet(focal)}') f
          LEFT JOIN unnest(list_distinct(f.referenced_works)) u(ref_id) ON true
          LEFT JOIN read_parquet('{parquet(lookup)}') l ON u.ref_id=l.id
        ), n AS (
          SELECT id, reference_count, count(field_id) AS classified_n,
                 count(DISTINCT field_id) AS reference_fields
          FROM edges GROUP BY ALL
        ), c AS (
          SELECT id, field_id, count(*) AS field_n FROM edges
          WHERE field_id IS NOT NULL GROUP BY ALL
        ), h AS (
          SELECT c.id,
                 -sum((field_n::DOUBLE/n.classified_n)*ln(field_n::DOUBLE/n.classified_n)) AS reference_entropy,
                 CASE WHEN n.classified_n>1 THEN
                   sum(field_n*(field_n-1))::DOUBLE/(n.classified_n*(n.classified_n-1)) END AS reference_hhi
          FROM c JOIN n USING (id) GROUP BY c.id, n.classified_n
        )
        SELECT n.id, n.reference_count, n.classified_n, n.reference_fields,
               COALESCE(h.reference_entropy, 0) AS reference_entropy,
               h.reference_hhi
        FROM n LEFT JOIN h USING (id)
    """)

    journal = QSS_WORK / "journal_year_baseline.parquet"
    years = ",".join(f"({y})" for y in FOCAL_YEARS)
    counts["journal_year_baseline"] = copy_query(con, journal, f"""
        WITH focal_years(focal_year) AS (VALUES {years}), h AS (
          SELECT y.focal_year, w.journal_id, w.id, w.topic_id, w.is_oa, w.language,
                 COALESCE((SELECT sum(c.cited_by_count) FROM unnest(w.counts_by_year) q(c)
                           WHERE c.year<=y.focal_year-1), 0) AS prior_citations
          FROM focal_years y JOIN read_parquet('{parquet(history)}') w
            ON w.publication_year BETWEEN y.focal_year-3 AND y.focal_year-1
        ), totals AS (
          SELECT focal_year, journal_id, count(*) AS history_n,
                 avg(is_oa::INTEGER) AS prior_oa_share,
                 avg((language='en')::INTEGER) AS prior_english_share,
                 avg(prior_citations) AS prior_prestige
          FROM h GROUP BY ALL
        ), tc AS (
          SELECT focal_year, journal_id, topic_id, count(*) AS n_topic
          FROM h WHERE topic_id IS NOT NULL GROUP BY ALL
        ), tz AS (
          SELECT *, sum(n_topic) OVER (PARTITION BY focal_year,journal_id) AS topic_n FROM tc
        ), topic_scope AS (
          SELECT focal_year, journal_id,
                 sum(n_topic*(n_topic-1))::DOUBLE/(max(topic_n)*(max(topic_n)-1)) AS topic_hhi,
                 -sum((n_topic::DOUBLE/topic_n)*ln(n_topic::DOUBLE/topic_n)) AS topic_entropy
          FROM tz GROUP BY focal_year, journal_id
        )
        SELECT t.*, s.topic_hhi, s.topic_entropy
        FROM totals t LEFT JOIN topic_scope s USING (focal_year,journal_id)
    """)

    journal_ref = QSS_WORK / "journal_year_reference_scope.parquet"
    counts["journal_year_reference_scope"] = copy_query(con, journal_ref, f"""
        WITH focal_years(focal_year) AS (VALUES {years}), e AS (
          SELECT y.focal_year, h.journal_id, l.field_id, count(*) AS n_field
          FROM focal_years y JOIN read_parquet('{parquet(history)}') h
            ON h.publication_year BETWEEN y.focal_year-3 AND y.focal_year-1
          CROSS JOIN unnest(list_distinct(h.referenced_works)) u(ref_id)
          JOIN read_parquet('{parquet(lookup)}') l ON u.ref_id=l.id
          WHERE l.field_id IS NOT NULL GROUP BY ALL
        ), z AS (
          SELECT *, sum(n_field) OVER (PARTITION BY focal_year,journal_id) AS n_refs
          FROM e
        )
        SELECT focal_year, journal_id, max(n_refs) AS classified_references,
               sum(n_field*(n_field-1))::DOUBLE/(max(n_refs)*(max(n_refs)-1)) AS reference_field_hhi,
               -sum((n_field::DOUBLE/n_refs)*ln(n_field::DOUBLE/n_refs)) AS reference_field_entropy
        FROM z GROUP BY focal_year,journal_id
    """)

    author_stats = QSS_WORK / "focal_author_history.parquet"
    counts["focal_author_history"] = copy_query(con, author_stats, f"""
        WITH wanted AS (
          SELECT f.id, f.publication_year, unnest(f.author_ids) AS author_id
          FROM read_parquet('{parquet(focal)}') f
        ), a AS (
          SELECT w.id, w.author_id,
                 COALESCE((SELECT sum(c.works_count) FROM unnest(x.counts_by_year) q(c)
                           WHERE c.year<=w.publication_year-1),0) AS works,
                 COALESCE((SELECT sum(c.cited_by_count) FROM unnest(x.counts_by_year) q(c)
                           WHERE c.year<=w.publication_year-1),0) AS citations
          FROM wanted w LEFT JOIN read_parquet('{authors}') x ON w.author_id=x.id
        )
        SELECT id, count(author_id) AS known_authors,
               avg(works) AS author_mean_prior_works, max(works) AS author_max_prior_works,
               avg(citations) AS author_mean_prior_citations,
               max(citations) AS author_max_prior_citations
        FROM a GROUP BY id
    """)

    inst_stats = QSS_WORK / "focal_institution_history.parquet"
    counts["focal_institution_history"] = copy_query(con, inst_stats, f"""
        WITH wanted AS (
          SELECT f.id, f.publication_year, unnest(f.institution_ids) AS institution_id
          FROM read_parquet('{parquet(focal)}') f
        ), i AS (
          SELECT w.id, w.institution_id,
                 COALESCE((SELECT sum(c.works_count) FROM unnest(x.counts_by_year) q(c)
                           WHERE c.year<=w.publication_year-1),0) AS works,
                 COALESCE((SELECT sum(c.cited_by_count) FROM unnest(x.counts_by_year) q(c)
                           WHERE c.year<=w.publication_year-1),0) AS citations
          FROM wanted w LEFT JOIN read_parquet('{institutions}') x ON w.institution_id=x.id
        )
        SELECT id, count(institution_id) AS known_institutions,
               avg(works) AS institution_mean_prior_works,
               max(works) AS institution_max_prior_works,
               avg(citations) AS institution_mean_prior_citations,
               max(citations) AS institution_max_prior_citations
        FROM i GROUP BY id
    """)

    edges = QSS_WORK / "citation_edges.parquet"
    counts["citation_edges"] = copy_query(con, edges, f"""
        SELECT c.id AS citing_id, f.id AS cited_id, c.publication_year AS citing_year,
               count(*) AS source_entries
        FROM read_parquet('{works}') c
        CROSS JOIN unnest(c.referenced_works) u(cited_id)
        JOIN read_parquet('{parquet(focal)}') f ON u.cited_id=f.id
          AND c.publication_year BETWEEN f.publication_year AND f.publication_year+4
        WHERE c.publication_year BETWEEN 2015 AND 2024 AND c.type='article'
          AND NOT COALESCE(c.is_xpac, false) AND NOT COALESCE(c.is_retracted, false)
          AND c.primary_location.is_published AND c.primary_location.source.type='journal'
          AND c.primary_location.source.id IS NOT NULL
        GROUP BY c.id,f.id,c.publication_year
    """, per_thread=True)

    citing = QSS_WORK / "citing_metadata.parquet"
    counts["citing_metadata"] = copy_query(con, citing, f"""
        WITH ids AS (SELECT DISTINCT citing_id FROM read_parquet('{parquet(edges)}'))
        SELECT c.id, c.title, c.language, c.publication_year,
               c.primary_location.source.id AS journal_id,
               c.primary_topic.subfield.id AS subfield_id,
               c.primary_topic.field.id AS field_id, {list_expr} AS author_ids
        FROM read_parquet('{works}') c JOIN ids ON c.id=ids.citing_id
    """, per_thread=True)

    embed_input = QSS_TMP / "embedding_input"
    counts["embedding_input"] = copy_query(con, embed_input, f"""
        SELECT id, any_value(title) AS title,
               any_value(abstract_inverted_index) FILTER (WHERE role='history') AS abstract_inverted_index,
               any_value(journal_id) FILTER (WHERE role='history') AS history_journal_id,
               any_value(publication_year) FILTER (WHERE role='history') AS history_publication_year,
               bool_or(role='history') AS is_history, bool_or(role='focal') AS is_focal,
               bool_or(role='citing') AS is_citing
        FROM (
          SELECT id,title,abstract_inverted_index,journal_id,publication_year,'history' AS role
          FROM read_parquet('{parquet(history)}')
          WHERE title IS NOT NULL
          UNION ALL
          SELECT id,title,NULL,journal_id,publication_year,'focal' FROM read_parquet('{parquet(focal)}')
          UNION ALL
          SELECT id,title,NULL,journal_id,publication_year,'citing' FROM read_parquet('{parquet(citing)}')
          WHERE language='en' AND title IS NOT NULL AND trim(title)<>''
        ) GROUP BY id
    """, per_thread=True)

    qc = con.execute(f"""
        SELECT min(publication_year), max(publication_year),
               count(*) FILTER (WHERE list_count(author_ids)<>authors_count)
        FROM read_parquet('{parquet(focal)}')
    """).fetchone()
    if qc[0:2] != (2015, 2020):
        raise ValueError(f"expected focal years 2015..2020, got {qc[0:2]}")
    duplicate_edges = con.execute(f"""
        SELECT count(*)-count(DISTINCT (citing_id,cited_id))
        FROM read_parquet('{parquet(edges)}')
    """).fetchone()[0]
    if duplicate_edges:
        raise ValueError(f"expected unique citation edges, got duplicates={duplicate_edges}")
    windows = con.execute(f"""
        SELECT count(*) FROM read_parquet('{parquet(edges)}') e
        JOIN read_parquet('{parquet(focal)}') f ON e.cited_id=f.id
        WHERE e.citing_year NOT BETWEEN f.publication_year AND f.publication_year+4
    """).fetchone()[0]
    if windows:
        raise ValueError(f"expected zero citation-window violations, got {windows}")
    for transient in (history, reference_ids, lookup):
        reset_output(transient)
        log(f"removed consumed intermediate {transient}")
    write_run("prepare", "complete", counts, {
        "author_count_mismatches": qc[2], "pilot_reproduction": pilot_reproduction,
    })
    check_budget()
    log("prepare complete " + json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
