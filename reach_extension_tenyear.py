#!/usr/bin/env python3
"""Ten-year citation inputs for the saved 2015 comparison sample; no refitting."""
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from qss_common import (
    GROUP_ROOT, MIN_FREE, QSS_TMP, REPO, SEED, SNAPSHOT, STAGED_INPUT,
    TMP_CAP, WORK_CAP, file_sha256, git_head, log, tree_bytes, validate_snapshot,
)
from qss_v3_common import V2_WORK, V3_WORK, connect, path_glob

BASE = V3_WORK / "reach_extension_v1/tenyear"
ARTIFACTS = REPO / "artifacts/reach_extension_v1"
SCORES = V3_WORK / "routing_scores.parquet"
CANDIDATE = V3_WORK / "candidate_focal.parquet"
OLD_OUTCOMES = V3_WORK / "citation_outcomes.parquet"
OLD_EDGES = V3_WORK / "citation_edges"
TAXONOMY = REPO / "artifacts/qss_v2/qwen3_taxonomy.npz"
AREAS = REPO / "results/qss_v3/hierarchy_areas.csv"
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
OUTCOMES = ("total_citations", "near", "intermediate", "far", "unclassified", "any_far")
WINDOWS = {"60": "NOT late", "120": "true", "late60_120": "late"}


def budget():
    persistent = tree_bytes(V2_WORK) + tree_bytes(V3_WORK) + tree_bytes(STAGED_INPUT)
    spill = tree_bytes(QSS_TMP) - tree_bytes(STAGED_INPUT)
    free = shutil.disk_usage(GROUP_ROOT).free
    if persistent > WORK_CAP or spill > TMP_CAP or free < MIN_FREE:
        raise RuntimeError(f"storage gate failed: persistent={persistent}, spill={spill}, free={free}")
    return {"persistent_bytes": persistent, "spill_bytes": spill, "free_bytes": free}


def new_target(path):
    if path.exists() or path.with_name(path.name + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite output or unfinished output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".partial")


def save(con, name, sql, empty=False):
    budget()
    target = BASE / name
    pending = new_target(target)
    con.execute(f"COPY ({sql}) TO '{pending}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 8192)")
    n = con.execute("SELECT count(*) FROM read_parquet(?)", [str(pending)]).fetchone()[0]
    if not empty and n == 0:
        raise ValueError(f"expected nonempty {name}, got {n}")
    budget()
    pending.rename(target)
    log(f"built {name}: rows={n:,} bytes={target.stat().st_size:,}")
    return n


def write_manifest(stage, counts, extra):
    path = ARTIFACTS / f"run_tenyear_{stage}.json"
    pending = new_target(path)
    sources = (SCORES, OLD_OUTCOMES, TAXONOMY, AREAS, SNAPSHOT / "manifest.json",
               REPO / "artifacts/qss_v2/run_outcome_embed.json", REPO / "artifacts/qss_v3/run_embed.json")
    payload = {"design": "reach_extension_v1", "stage": stage, "status": "complete",
               "snapshot_date": "2026-06-26", "seed": SEED, "git_commit": git_head(),
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "counts": counts,
               "frozen_inputs": {str(p): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                                 for p in sources},
               "outputs": {str(p): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                           for p in sorted(BASE.rglob("*.parquet"))},
               "storage": budget(), "extra": extra}
    with pending.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    pending.rename(path)
    log(f"tenyear {stage} complete: {json.dumps(counts, sort_keys=True)}; storage={payload['storage']}")


def prior_manifest(stage):
    path = ARTIFACTS / f"run_tenyear_{stage}.json"
    run = json.loads(path.read_text())
    if run["status"] != "complete" or run["snapshot_date"] != "2026-06-26":
        raise ValueError(f"invalid predecessor manifest: {path}")
    for name, info in (run["frozen_inputs"] | run["outputs"]).items():
        if file_sha256(Path(name)) != info["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    return run


def qwen_view(con, include_new=False):
    paths = [V2_WORK / "qwen3_semantics.parquet", V3_WORK / "qwen3_semantics"]
    if include_new:
        paths.append(BASE / "qwen3_semantics")
    con.execute("CREATE TEMP VIEW qwen_all AS " + " UNION ALL ".join(
        f"SELECT id,qwen_leaf,qwen_macro,qwen_ood FROM read_parquet('{path_glob(p)}')" for p in paths))
    qc = con.execute("SELECT count(*),count(DISTINCT id),count(*) FILTER (WHERE "
                     "qwen_leaf IS NULL OR qwen_macro IS NULL OR qwen_ood IS NULL "
                     "OR qwen_leaf NOT BETWEEN 0 AND 999 OR qwen_macro NOT BETWEEN 0 AND 31) "
                     "FROM qwen_all").fetchone()
    if qc[0] != qc[1] or qc[2]:
        raise ValueError(f"Qwen label QC failed: {qc}")


def prepare():
    if BASE.exists():
        raise FileExistsError(f"ten-year directory already exists: {BASE}")
    validate_snapshot()
    budget()
    for name, key in (("qss_v2/run_outcome_embed.json", "model_commit"),
                      ("qss_v3/run_embed.json", "qwen3_model_commit")):
        run = json.loads((REPO / "artifacts" / name).read_text())
        if run["extra"][key] != QWEN_REVISION or run["extra"]["embedding_dimension"] != 768:
            raise ValueError(f"Qwen revision/dimension mismatch: {name}")
    con = connect("220GB", 32)
    temp = QSS_TMP / "reach_extension_v1/tenyear_prepare"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp}'")
    con.execute("SET max_temp_directory_size='180GB'")
    qwen_view(con)
    qc = con.execute(f"SELECT count(*),count(DISTINCT id),count(DISTINCT journal_id) "
                     f"FROM read_parquet('{SCORES}')").fetchone()
    if qc != (3_827_491, 3_827_491, 20_215):
        raise ValueError(f"saved comparison sample changed: {qc}")
    counts = {"focal": save(con, "focal_2015.parquet", f"""
      SELECT s.id,s.journal_id,s.publication_year,s.treatment,s.propensity,s.choice_set_id,
             s.qwen_macro,q.qwen_leaf,q.qwen_ood,f.publication_date,f.author_ids,f.fold
      FROM read_parquet('{SCORES}') s JOIN read_parquet('{CANDIDATE}') f USING (id)
      JOIN qwen_all q USING (id) WHERE s.publication_year=2015
        AND s.journal_id=f.journal_id AND s.treatment=f.treatment AND s.qwen_macro=q.qwen_macro
    """)}
    focal = BASE / "focal_2015.parquet"
    qc = con.execute(f"SELECT count(*),count(DISTINCT id),count(DISTINCT journal_id),"
                     f"count(*) FILTER (WHERE publication_date IS NULL OR year(publication_date)<>2015 "
                     f"OR qwen_ood OR propensity NOT BETWEEN 0.05 AND 0.95) FROM read_parquet('{focal}')").fetchone()
    if qc != (596_758, 596_758, 11_635, 0):
        raise ValueError(f"2015 focal QC failed: {qc}")
    works = str(SNAPSHOT / "works/updated_date=*/*.parquet")
    counts["edges"] = save(con, "citation_edges.parquet", f"""
      SELECT c.id AS citing_id,f.id AS cited_id,c.publication_date AS citing_date,
             c.publication_year AS citing_year,
             c.publication_date>=f.publication_date+INTERVAL 60 MONTH AS late
      FROM read_parquet('{works}') c
      CROSS JOIN unnest(list_distinct(c.referenced_works)) u(cited_id)
      JOIN read_parquet('{focal}') f ON u.cited_id=f.id
        AND c.publication_date>=f.publication_date
        AND c.publication_date<f.publication_date+INTERVAL 120 MONTH
      WHERE c.publication_year BETWEEN 2015 AND 2025 AND c.type='article'
        AND NOT COALESCE(c.is_xpac,false) AND NOT COALESCE(c.is_retracted,false)
        AND c.primary_location.is_published AND c.primary_location.source.type='journal'
        AND c.primary_location.source.id IS NOT NULL GROUP BY ALL
    """)
    edge_qc = con.execute(f"""
      SELECT count(*),count(DISTINCT (e.citing_id,e.cited_id)),
        count(*) FILTER (WHERE e.citing_date<f.publication_date
          OR e.citing_date>=f.publication_date+INTERVAL 120 MONTH
          OR e.late<>(e.citing_date>=f.publication_date+INTERVAL 60 MONTH)),
        count(*) FILTER (WHERE NOT late),count(*) FILTER (WHERE late)
      FROM read_parquet('{BASE}/citation_edges.parquet') e
      JOIN read_parquet('{focal}') f ON e.cited_id=f.id
    """).fetchone()
    if edge_qc[0] != edge_qc[1] or edge_qc[2]:
        raise ValueError(f"ten-year edge uniqueness/window failure: {edge_qc}")
    edge_replay = con.execute(f"""
      WITH old AS (SELECT e.citing_id,e.cited_id,e.citing_date,e.citing_year
          FROM read_parquet('{path_glob(OLD_EDGES)}') e
          JOIN read_parquet('{focal}') f ON e.cited_id=f.id),
      new AS (SELECT citing_id,cited_id,citing_date,citing_year
          FROM read_parquet('{BASE}/citation_edges.parquet') WHERE NOT late)
      SELECT count(*) FROM ((SELECT * FROM old EXCEPT SELECT * FROM new)
                       UNION ALL (SELECT * FROM new EXCEPT SELECT * FROM old))
    """).fetchone()[0]
    if edge_replay:
        raise ValueError(f"original 60-month edge set mismatch: {edge_replay}")
    counts["citing"] = save(con, "citing_metadata.parquet", f"""
      WITH ids AS (SELECT DISTINCT citing_id FROM read_parquet('{BASE}/citation_edges.parquet'))
      SELECT c.id,c.title,c.language,c.publication_date,c.publication_year,
             c.primary_location.source.id AS journal_id,
             list_filter(list_distinct(list_transform(c.authorships,x->x.author.id)),
                         x->x IS NOT NULL) AS author_ids
      FROM read_parquet('{works}') c JOIN ids ON c.id=ids.citing_id
    """)
    meta_qc = con.execute(f"SELECT count(*),count(DISTINCT id),count(*) FILTER (WHERE journal_id IS NULL) "
                          f"FROM read_parquet('{BASE}/citing_metadata.parquet')").fetchone()
    if meta_qc != (counts["citing"], counts["citing"], 0):
        raise ValueError(f"citing metadata uniqueness/journal QC failed: {meta_qc}")
    counts["missing_qwen"] = save(con, "qwen_missing.parquet", f"""
      SELECT c.id,c.title FROM read_parquet('{BASE}/citing_metadata.parquet') c
      ANTI JOIN qwen_all q USING (id)
      WHERE c.language='en' AND c.title IS NOT NULL AND trim(c.title)<>'' ORDER BY c.id
    """, empty=True)
    write_manifest("prepare", counts, {"edge_qc": edge_qc, "old_60_edge_mismatches": edge_replay,
                                       "windows": "[publication_date, +60 months), [+60, +120 months)",
                                       "focal_population": "saved 2015 support, including uncited papers"})


def outcome_query(mixed):
    sums, names = [], []
    for window, condition in WINDOWS.items():
        for suffix, class_col in (("", "class"), ("_no_mixed", "class_no_mixed")):
            for outcome in OUTCOMES:
                name = f"{outcome}_{window}{suffix}"
                where = condition if outcome == "total_citations" else f"({condition}) AND {class_col}='{outcome if outcome != 'any_far' else 'far'}'"
                aggregate = f"count(*) FILTER (WHERE {where})"
                sums.append(f"({aggregate}>0)::UTINYINT AS {name}" if outcome == "any_far" else f"{aggregate} AS {name}")
                names.append(name)
    return f"""
      WITH labeled AS (
        SELECT e.*,c.journal_id=f.journal_id AS same_journal,
          COALESCE(list_has_any(c.author_ids,f.author_ids),false) AS shared_author,q.qwen_macro,
          CASE WHEN c.language<>'en' OR c.language IS NULL OR q.id IS NULL
                    OR q.qwen_ood OR f.qwen_ood THEN 'unclassified'
               WHEN q.qwen_leaf=f.qwen_leaf THEN 'near'
               WHEN q.qwen_macro=f.qwen_macro THEN 'intermediate' ELSE 'far' END AS class
        FROM read_parquet('{BASE}/citation_edges.parquet') e
        JOIN read_parquet('{BASE}/focal_2015.parquet') f ON e.cited_id=f.id
        JOIN read_parquet('{BASE}/citing_metadata.parquet') c ON e.citing_id=c.id
        LEFT JOIN qwen_all q ON c.id=q.id
      ), external AS (
        SELECT *,CASE WHEN qwen_macro={mixed} THEN 'unclassified' ELSE class END AS class_no_mixed
        FROM labeled WHERE NOT same_journal AND NOT shared_author
      ), totals AS (SELECT cited_id,{','.join(sums)} FROM external GROUP BY cited_id)
      SELECT f.id,f.journal_id,f.treatment,f.propensity,f.fold,f.qwen_macro,
             {','.join(f'COALESCE(t.{n},0) AS {n}' for n in names)}
      FROM read_parquet('{BASE}/focal_2015.parquet') f LEFT JOIN totals t ON f.id=t.cited_id
    """


def outcomes():
    prep, embedded = prior_manifest("prepare"), prior_manifest("embed")
    con = connect("220GB", 32)
    temp = QSS_TMP / "reach_extension_v1/tenyear_outcomes"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp}'")
    con.execute("SET max_temp_directory_size='180GB'")
    qwen_view(con, include_new=True)
    with AREAS.open() as stream:
        mixed = [int(r["qwen_macro"]) for r in csv.DictReader(stream) if r["display_label"] == "Mixed records"]
    if len(mixed) != 1:
        raise ValueError(f"expected one Mixed records area, got {mixed}")
    n = save(con, "outcomes_2015.parquet", outcome_query(mixed[0]))
    path = BASE / "outcomes_2015.parquet"
    qc = con.execute(f"SELECT count(*),count(DISTINCT id) FROM read_parquet('{path}')").fetchone()
    if qc != (596_758, 596_758) or n != prep["counts"]["focal"]:
        raise ValueError(f"outcome population mismatch: {qc}")
    checks = [f"a.{name}_60<>b.{name}" for name in OUTCOMES]
    replay = con.execute(f"SELECT count(*),count(*) FILTER (WHERE {' OR '.join(checks)}) "
                         f"FROM read_parquet('{path}') a JOIN read_parquet('{OLD_OUTCOMES}') b USING (id)").fetchone()
    if replay != (n, 0):
        raise ValueError(f"original 60-month counts did not replay exactly: {replay}")
    checks = []
    for suffix in ("", "_no_mixed"):
        for window in WINDOWS:
            checks.append(f"total_citations_{window}{suffix}<>near_{window}{suffix}+intermediate_{window}{suffix}+far_{window}{suffix}+unclassified_{window}{suffix}")
            checks.append(f"any_far_{window}{suffix}<>(far_{window}{suffix}>0)")
        for name in OUTCOMES[:-1]:
            checks.append(f"{name}_120{suffix}<>{name}_60{suffix}+{name}_late60_120{suffix}")
    checks += [f"total_citations_{w}<>total_citations_{w}_no_mixed" for w in WINDOWS]
    checks += [f"any_far_120{s}<>(any_far_60{s} OR any_far_late60_120{s})" for s in ("", "_no_mixed")]
    bad = con.execute(f"SELECT count(*) FROM read_parquet('{path}') WHERE {' OR '.join(checks)}").fetchone()[0]
    if bad:
        raise ValueError(f"window addition/outcome decomposition failures: {bad}")
    counts = {"focal": n, "replay_mismatches": replay[1], "missing_qwen_encoded": embedded["counts"]["labels"]}
    totals = con.execute(f"SELECT sum(total_citations_60),sum(total_citations_120),sum(total_citations_late60_120),"
                         f"count(*) FILTER (WHERE total_citations_120=0) FROM read_parquet('{path}')").fetchone()
    write_manifest("outcomes", counts, {"citation_totals_60_120_late_uncited": totals,
                   "mixed_macro": mixed[0], "no_mixed": "Mixed citing origins are unclassified; Mixed focals retained"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "outcomes"):
        raise SystemExit("usage: reach_extension_tenyear.py prepare|outcomes")
    {"prepare": prepare, "outcomes": outcomes}[sys.argv[1]]()
