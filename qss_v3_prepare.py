#!/usr/bin/env python3
import json

from qss_common import SNAPSHOT
from qss_v3_common import (
    V2_WORK, V3_WORK, check_budget, connect, copy_query, log, path_glob,
    validate_snapshot, write_run,
)

WORKS = str(SNAPSHOT / "works/updated_date=*/*.parquet")
CANDIDATE = V3_WORK / "candidate_focal.parquet"
EDGES = V3_WORK / "citation_edges"
CITING = V3_WORK / "citing_metadata"
REFERENCE_EDGES = V3_WORK / "reference_edges"
LEADS = V3_WORK / "lead_authors.parquet"
PRIOR = V3_WORK / "author_prior_papers"
QWEN_INPUT = V3_WORK / "qwen_input"
SPECTER_INPUT = V3_WORK / "specter_input"


def main():
    validate_snapshot()
    V3_WORK.mkdir(parents=True, exist_ok=True)
    check_budget()
    con = connect()
    works = WORKS.replace("'", "''")
    v2_analysis = path_glob(V2_WORK / "analysis_unestimated.parquet")
    v2_qwen = path_glob(V2_WORK / "qwen3_semantics.parquet")
    v2_specter = path_glob(V2_WORK / "embeddings_title")
    counts = {}

    first = """
      list_extract(list_transform(list_filter(w.authorships,
        x -> x.author_position='first' AND x.author.id IS NOT NULL),
        x -> x.author.id), 1)
    """.strip()
    last = """
      list_extract(list_transform(list_filter(w.authorships,
        x -> x.author_position='last' AND x.author.id IS NOT NULL),
        x -> x.author.id), 1)
    """.strip()
    author_ids = """
      list_filter(list_distinct(list_transform(w.authorships, x -> x.author.id)),
                  x -> x IS NOT NULL)
    """.strip()

    counts["candidate_focal"] = copy_query(con, CANDIDATE, f"""
      SELECT a.* EXCLUDE(publication_month,total_citations,near,intermediate,far,
                         unclassified,unclassified_language,
                         unclassified_missing_title,unclassified_ood,
                         all_unique_citations,same_journal_citations,
                         shared_author_citations,any_far),
             w.publication_date,w.referenced_works,{author_ids} AS author_ids,
             {first} AS first_author_id,{last} AS last_author_id,
             (month(w.publication_date)=1 AND day(w.publication_date)=1) AS january_1,
             (a.publication_year*1000+a.semantic_cluster)::INTEGER AS choice_set_id,
             (hash(a.id)%10)::UTINYINT AS early_stop_bucket
      FROM read_parquet('{v2_analysis}') a
      JOIN read_parquet('{works}') w USING (id)
      WHERE a.treatment IN (0,1) AND NOT a.focal_ood
    """)
    date_qc = con.execute(f"""
      SELECT treatment,count(*) AS n,
             count(*) FILTER (WHERE publication_date IS NULL) AS missing,
             avg(january_1::INTEGER) AS january_1_rate,
             count(*) FILTER (WHERE year(publication_date)<>publication_year) AS year_mismatch
      FROM read_parquet('{CANDIDATE}') GROUP BY treatment ORDER BY treatment
    """).fetchall()
    if len(date_qc) != 2 or any(row[2] or row[4] for row in date_qc):
        raise ValueError(f"focal publication date QC failed: {date_qc}")

    eligible = """
      c.publication_year BETWEEN 2015 AND 2025 AND c.type='article'
      AND NOT COALESCE(c.is_xpac,false) AND NOT COALESCE(c.is_retracted,false)
      AND c.primary_location.is_published
      AND c.primary_location.source.type='journal'
      AND c.primary_location.source.id IS NOT NULL
    """
    counts["citation_edges"] = copy_query(con, EDGES, f"""
      SELECT c.id AS citing_id,f.id AS cited_id,c.publication_date AS citing_date,
             c.publication_year AS citing_year
      FROM read_parquet('{works}') c
      CROSS JOIN unnest(list_distinct(c.referenced_works)) u(cited_id)
      JOIN read_parquet('{CANDIDATE}') f ON u.cited_id=f.id
        AND c.publication_date>=f.publication_date
        AND c.publication_date<f.publication_date+INTERVAL 60 MONTH
      WHERE {eligible}
      GROUP BY ALL
    """, per_thread=True)
    edge_qc = con.execute(f"""
      SELECT count(*),count(DISTINCT (citing_id,cited_id))
      FROM read_parquet('{path_glob(EDGES)}')
    """).fetchone()
    window_errors = con.execute(f"""
      SELECT count(*) FROM read_parquet('{path_glob(EDGES)}') e
      JOIN read_parquet('{CANDIDATE}') f ON e.cited_id=f.id
      WHERE e.citing_date<f.publication_date
         OR e.citing_date>=f.publication_date+INTERVAL 60 MONTH
    """).fetchone()[0]
    if edge_qc[0] != edge_qc[1] or window_errors:
        raise ValueError(f"citation edge QC failed: edges={edge_qc} windows={window_errors}")

    counts["citing_metadata"] = copy_query(con, CITING, f"""
      WITH ids AS (SELECT DISTINCT citing_id FROM read_parquet('{path_glob(EDGES)}'))
      SELECT c.id,c.title,c.language,c.publication_date,c.publication_year,
             c.primary_location.source.id AS journal_id,{author_ids} AS author_ids
      FROM read_parquet('{works}') c JOIN ids ON c.id=ids.citing_id
    """, per_thread=True)

    counts["reference_edges"] = copy_query(con, REFERENCE_EDGES, f"""
      SELECT f.id AS focal_id,u.ref_id
      FROM read_parquet('{CANDIDATE}') f
      CROSS JOIN unnest(list_distinct(f.referenced_works)) u(ref_id)
      WHERE u.ref_id IS NOT NULL GROUP BY ALL
    """, per_thread=True)
    ref_qc = con.execute(f"""
      SELECT count(*),count(DISTINCT (focal_id,ref_id))
      FROM read_parquet('{path_glob(REFERENCE_EDGES)}')
    """).fetchone()
    if ref_qc[0] != ref_qc[1]:
        raise ValueError(f"reference edge QC failed: {ref_qc}")

    counts["lead_authors"] = copy_query(con, LEADS, f"""
      SELECT id AS focal_id,publication_year AS focal_year,first_author_id AS author_id
      FROM read_parquet('{CANDIDATE}') WHERE first_author_id IS NOT NULL
      UNION ALL
      SELECT id,publication_year,last_author_id
      FROM read_parquet('{CANDIDATE}')
      WHERE last_author_id IS NOT NULL AND last_author_id<>first_author_id
    """)
    counts["author_prior_papers"] = copy_query(con, PRIOR, f"""
      WITH author_year AS (
        SELECT DISTINCT author_id,focal_year FROM read_parquet('{LEADS}')
      )
      SELECT DISTINCT y.author_id,y.focal_year,w.id AS prior_id,
             w.publication_year AS prior_year,
             w.primary_location.source.id AS journal_id,w.title
      FROM read_parquet('{works}') w
      CROSS JOIN unnest(w.authorships) u(authorship)
      JOIN author_year y ON authorship.author.id=y.author_id
        AND w.publication_year BETWEEN y.focal_year-5 AND y.focal_year-1
      WHERE w.publication_year BETWEEN 2010 AND 2019 AND w.language='en'
        AND w.type='article' AND w.title IS NOT NULL AND trim(w.title)<>''
        AND NOT COALESCE(w.is_xpac,false) AND NOT COALESCE(w.is_retracted,false)
        AND w.primary_location.is_published
        AND w.primary_location.source.type='journal'
        AND w.primary_location.source.id IS NOT NULL
    """, per_thread=True)

    counts["qwen_input"] = copy_query(con, QWEN_INPUT, f"""
      WITH needed AS (
        SELECT ref_id AS id FROM read_parquet('{path_glob(REFERENCE_EDGES)}')
        UNION
        SELECT citing_id FROM read_parquet('{path_glob(EDGES)}')
      ), missing AS (
        SELECT n.id FROM needed n ANTI JOIN read_parquet('{v2_qwen}') q USING (id)
      )
      SELECT w.id,any_value(w.title) AS title
      FROM read_parquet('{works}') w JOIN missing m USING (id)
      WHERE w.language='en' AND w.title IS NOT NULL AND trim(w.title)<>''
      GROUP BY w.id
    """, per_thread=True)
    counts["specter_input"] = copy_query(con, SPECTER_INPUT, f"""
      WITH needed AS (
        SELECT DISTINCT prior_id AS id,any_value(title) AS title
        FROM read_parquet('{path_glob(PRIOR)}') GROUP BY prior_id
      )
      SELECT n.* FROM needed n ANTI JOIN read_parquet('{v2_specter}') e USING (id)
    """, per_thread=True)

    reference_coverage = con.execute(f"""
      SELECT f.treatment,count(*) AS edges,count(q.id) AS existing_qwen,
             count(q.id)::DOUBLE/count(*) AS coverage
      FROM read_parquet('{path_glob(REFERENCE_EDGES)}') r
      JOIN read_parquet('{CANDIDATE}') f ON r.focal_id=f.id
      LEFT JOIN read_parquet('{v2_qwen}') q ON r.ref_id=q.id
      GROUP BY f.treatment ORDER BY f.treatment
    """).fetchall()
    citing_january_1 = con.execute(f"""
      SELECT f.treatment,count(*) AS edges,
             avg((month(e.citing_date)=1 AND day(e.citing_date)=1)::INTEGER) AS rate
      FROM read_parquet('{path_glob(EDGES)}') e
      JOIN read_parquet('{CANDIDATE}') f ON e.cited_id=f.id
      GROUP BY f.treatment ORDER BY f.treatment
    """).fetchall()
    write_run("prepare", counts, {
        "focal_date_qc": date_qc,
        "citing_january_1_by_arm": citing_january_1,
        "initial_reference_qwen_coverage": reference_coverage,
        "window": "[publication_date, publication_date + 60 months)",
        "author_history_years": "t-5:t-1",
        "author_far_near_history": "not added in dirty v3",
    })
    check_budget()
    log(f"v3 prepare complete counts={json.dumps(counts, sort_keys=True)}")


if __name__ == "__main__":
    main()
