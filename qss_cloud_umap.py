#!/usr/bin/env python3
"""Paper-level UMAP point cloud for Figure 2.

A proportional random sample of the 7,617,662 eligible two-arm papers is embedded
with unsupervised UMAP on the 32 frozen Qwen3 title components. Each sampled paper
carries its research area and whether it entered the deterministic comparison
sample (broad / narrow) or fell outside common support. The output is a small
parquet used only for display.
"""
import numpy as np
import pandas as pd
import umap

from qss_common import SEED
from qss_v3_common import RESULTS, V2_WORK, V3_WORK, connect, log, path_glob, write_run

ANALYSIS = V3_WORK / "analysis_dataset.parquet"
SCORES = V3_WORK / "routing_scores.parquet"
QWEN_V2 = V2_WORK / "qwen3_semantics.parquet"
QWEN_V3 = V3_WORK / "qwen3_semantics"
OUT = V3_WORK / "cloud_umap_papers.parquet"
SAMPLE_FRACTION = 0.16   # ~1.22 million papers
EXPECTED_ELIGIBLE = 7_617_662


def main():
    con = connect()
    qcols = ",".join(f"q.qpc{i:02d}" for i in range(1, 33))
    frame = con.execute(f"""
      WITH qwen AS (
        SELECT * FROM read_parquet('{QWEN_V2}')
        UNION ALL
        SELECT * FROM read_parquet('{path_glob(QWEN_V3)}')
      ),
      base AS (
        SELECT a.id, a.treatment, q.qwen_macro, q.qwen_leaf, {qcols},
               CASE WHEN s.id IS NULL THEN 'excluded'
                    WHEN a.treatment = 1 THEN 'narrow' ELSE 'broad' END AS status,
               hash(a.id || '|{SEED}') AS h
        FROM read_parquet('{ANALYSIS}') a
        JOIN qwen q USING (id)
        LEFT JOIN read_parquet('{SCORES}') s USING (id)
      )
      SELECT * EXCLUDE (h), (SELECT count(*) FROM base) AS n_total
      FROM base WHERE (h % 10000) < {int(SAMPLE_FRACTION * 10000)}
    """).df()
    n_total = int(frame.n_total.iloc[0])
    if n_total != EXPECTED_ELIGIBLE:
        raise ValueError(f"expected {EXPECTED_ELIGIBLE:,} eligible papers, got {n_total:,}")
    frame = frame.drop(columns="n_total")
    log(f"sampled {len(frame):,} papers; status counts {frame.status.value_counts().to_dict()}")

    X = frame[[f"qpc{i:02d}" for i in range(1, 33)]].to_numpy(dtype=np.float32)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.08, metric="euclidean",
                        n_components=2, random_state=None, low_memory=True, verbose=True)
    coords = reducer.fit_transform(X)
    out = frame[["id", "qwen_macro", "qwen_leaf", "status"]].copy()
    out["umap_x"] = coords[:, 0].astype(np.float32)
    out["umap_y"] = coords[:, 1].astype(np.float32)
    out.to_parquet(OUT, index=False, compression="zstd")
    # Display-level aggregate (research-area medians) for labels.
    medians = out.groupby("qwen_macro")[["umap_x", "umap_y"]].median().reset_index()
    medians["n_sampled"] = out.groupby("qwen_macro").size().values
    medians.to_csv(RESULTS / "cloud_umap_area_medians.csv", index=False)
    write_run("cloud_umap", {"sampled": len(out), "eligible": n_total},
              {"sample_fraction": SAMPLE_FRACTION,
               "umap": {"n_neighbors": 30, "min_dist": 0.08, "metric": "euclidean",
                        "input": "qpc01-qpc32", "random_state": "unseeded (parallel)"},
               "status_counts": out.status.value_counts().to_dict()})
    log(f"cloud umap complete rows={len(out):,} -> {OUT}")


if __name__ == "__main__":
    main()
