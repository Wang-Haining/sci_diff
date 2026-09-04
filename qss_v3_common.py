#!/usr/bin/env python3
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from qss_common import (
    GROUP_ROOT, MIN_FREE, QSS_TMP, QSS_WORK as V2_WORK, REPO, SEED, SNAPSHOT,
    TMP_CAP, WORK_CAP, log, reset_output, tree_bytes, validate_snapshot,
)

V3_WORK = GROUP_ROOT / "g91p721/sci_diff/qss_v3"
V3_TMP = QSS_TMP / "v3"
V2_STAGED = QSS_TMP / "embedding_input"
RESULTS = REPO / "results/qss_v3"
ARTIFACTS = REPO / "artifacts/qss_v3"


def path_glob(path):
    return str(path / "*.parquet") if path.is_dir() else str(path)


def check_budget():
    persistent = tree_bytes(V2_WORK) + tree_bytes(V2_STAGED) + tree_bytes(V3_WORK)
    temporary = tree_bytes(V3_TMP)
    free = shutil.disk_usage(GROUP_ROOT).free
    if persistent > WORK_CAP:
        raise RuntimeError(f"expected combined QSS persistent <= {WORK_CAP}, got {persistent}")
    if temporary > TMP_CAP:
        raise RuntimeError(f"expected qss_v3 spill <= {TMP_CAP}, got {temporary}")
    if free < MIN_FREE:
        raise RuntimeError(f"expected group free >= {MIN_FREE}, got {free}")
    log(f"storage persistent={persistent:,} spill={temporary:,} free={free:,}")


def connect(memory_limit="650GB", threads=32):
    V3_TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{V3_TMP}'")
    con.execute("SET max_temp_directory_size='400GB'")
    con.execute("SET preserve_insertion_order=false")
    return con


def copy_query(con, path, sql, per_thread=False):
    check_budget()
    reset_output(path)
    escaped = str(path).replace("'", "''")
    if per_thread:
        path.mkdir(parents=True)
        con.execute(f"COPY ({sql}) TO '{escaped}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, PER_THREAD_OUTPUT true)")
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


def write_run(stage, counts=None, extra=None):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ("duckdb", "numpy", "pandas", "pyarrow", "scikit-learn",
                 "lightgbm", "torch", "transformers", "adapters"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "design": "qss_v3", "stage": stage, "status": "complete",
        "snapshot_date": "2026-06-26", "seed": SEED,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packages": packages, "counts": counts or {}, "extra": extra or {},
    }
    (ARTIFACTS / f"run_{stage}.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
