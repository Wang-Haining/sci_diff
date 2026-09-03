#!/usr/bin/env python3
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb

SNAPSHOT_DATE = "2026-06-26"
GROUP_ROOT = Path("/home/group/jasonclark")
SNAPSHOT = GROUP_ROOT / "g91p721/openalex" / SNAPSHOT_DATE / "data/parquet"
QSS_WORK = GROUP_ROOT / "g91p721/sci_diff/qss_v1"
QSS_TMP = GROUP_ROOT / "g91p721/sci_diff/qss_tmp"
REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results" / "qss_v1"
ARTIFACTS = REPO / "artifacts" / "qss_v1"
SEED = 20260902
MIN_FREE = 1_500_000_000_000
WORK_CAP = 200_000_000_000
TMP_CAP = 400_000_000_000
RAW_BYTES = 783_555_949_110
RAW_RECORDS = 649_096_577
RAW_FILES = 5_202
FOCAL_YEARS = tuple(range(2015, 2021))
HISTORY_START = min(FOCAL_YEARS) - 3
HISTORY_END = max(FOCAL_YEARS) - 1
EMBED_DIM = 768


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def tree_bytes(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def check_budget():
    used = tree_bytes(QSS_WORK)
    temporary = tree_bytes(QSS_TMP)
    free = shutil.disk_usage(GROUP_ROOT).free
    if used > WORK_CAP:
        raise RuntimeError(f"expected qss work <= {WORK_CAP}, got {used}")
    if free < MIN_FREE:
        raise RuntimeError(f"expected group free >= {MIN_FREE}, got {free}")
    if temporary > TMP_CAP:
        raise RuntimeError(f"expected qss temp <= {TMP_CAP}, got {temporary}")
    log(f"storage qss_work={used:,} bytes qss_temp={temporary:,} bytes group_free={free:,} bytes")


def validate_snapshot():
    path = SNAPSHOT / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"expected manifest at {path}")
    manifest = json.loads(path.read_text())
    files = [x for entity in manifest["entities"] for x in entity["files"]]
    got = (manifest.get("date"), manifest["meta"]["content_length"],
           manifest["meta"]["record_count"], len(files))
    expected = (SNAPSHOT_DATE, RAW_BYTES, RAW_RECORDS, RAW_FILES)
    if got != expected:
        raise ValueError(f"expected manifest {expected}, got {got}")
    return manifest


def connect(memory_limit="700GB", threads=32):
    QSS_TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{QSS_TMP}'")
    con.execute("SET max_temp_directory_size='400GB'")
    con.execute("SET preserve_insertion_order=false")
    return con


def reset_output(path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_query(con, path, sql, per_thread=False):
    check_budget()
    reset_output(path)
    escaped = str(path).replace("'", "''")
    if per_thread:
        path.mkdir(parents=True)
        con.execute(f"COPY ({sql}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD, PER_THREAD_OUTPUT true)")
        pattern = str(path / "*.parquet")
    else:
        con.execute(f"COPY ({sql}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        pattern = str(path)
    rows = con.execute("SELECT count(*) FROM read_parquet(?)", [pattern]).fetchone()[0]
    if rows <= 0:
        raise ValueError(f"expected {path} rows > 0, got {rows}")
    check_budget()
    log(f"built {path.name} rows={rows:,} bytes={tree_bytes(path):,}")
    return rows


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_run(stage, status, counts=None, extra=None):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "design": "qss_v1",
        "stage": stage,
        "status": status,
        "snapshot_date": SNAPSHOT_DATE,
        "seed": SEED,
        "git_commit": git_head(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {name: package_version(name) for name in (
            "duckdb", "numpy", "pandas", "pyarrow", "scikit-learn", "scipy",
            "lightgbm", "torch", "transformers", "adapters", "matplotlib",
        )},
        "counts": counts or {},
        "extra": extra or {},
    }
    (ARTIFACTS / f"run_{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
