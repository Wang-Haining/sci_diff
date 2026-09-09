#!/usr/bin/env python3
"""Leaf-level counts and layout for the Figure 2 manuscript cloud.

Top layer of the hierarchy figure: the 7,617,662 eligible two-arm papers,
split into those outside the deterministic common support (gray) and those
inside it by journal group (broad / narrow). Counts are aggregated to the
1,000 frozen Qwen3 leaf topics; the leaf centers are laid out with UMAP so the
figure never needs paper-level coordinates.
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
TAXONOMY = "artifacts/qss_v2/qwen3_taxonomy.npz"
EXPECTED_ELIGIBLE = 7_617_662
EXPECTED_SUPPORT = 3_827_491


def main():
    con = connect()
    counts = con.execute(f"""
      WITH qwen AS (
        SELECT id, qwen_leaf, qwen_macro FROM read_parquet('{QWEN_V2}')
        UNION ALL
        SELECT id, qwen_leaf, qwen_macro FROM read_parquet('{path_glob(QWEN_V3)}')
      ),
      base AS (
        SELECT a.id, a.treatment, q.qwen_leaf, q.qwen_macro,
               s.id IS NOT NULL AS in_support
        FROM read_parquet('{ANALYSIS}') a
        JOIN qwen q USING (id)
        LEFT JOIN read_parquet('{SCORES}') s USING (id)
      )
      SELECT qwen_leaf, qwen_macro,
             count(*) FILTER (WHERE NOT in_support) AS n_excluded,
             count(*) FILTER (WHERE in_support AND treatment = 0) AS n_broad,
             count(*) FILTER (WHERE in_support AND treatment = 1) AS n_narrow
      FROM base GROUP BY 1, 2 ORDER BY 1
    """).df()
    total = int(counts[["n_excluded", "n_broad", "n_narrow"]].to_numpy().sum())
    support = int(counts[["n_broad", "n_narrow"]].to_numpy().sum())
    if total != EXPECTED_ELIGIBLE or support != EXPECTED_SUPPORT:
        raise ValueError(f"expected {EXPECTED_ELIGIBLE:,} eligible and {EXPECTED_SUPPORT:,} "
                         f"support papers, got {total:,} and {support:,}")
    if counts.qwen_leaf.nunique() != len(counts):
        raise ValueError("a leaf maps to more than one macrocluster")

    tax = np.load(TAXONOMY)
    centers = tax["leaf_centers"]
    layout = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine",
                       random_state=SEED).fit_transform(centers)
    leaves = pd.DataFrame({"qwen_leaf": np.arange(len(centers)),
                           "leaf_macro": tax["leaf_to_macro"].astype(int),
                           "umap_x": layout[:, 0], "umap_y": layout[:, 1]})
    out = leaves.merge(counts, on="qwen_leaf", how="left")
    for col in ("n_excluded", "n_broad", "n_narrow"):
        out[col] = out[col].fillna(0).astype(int)
    if (out.qwen_macro.dropna().astype(int) != out.loc[out.qwen_macro.notna(), "leaf_macro"]).any():
        raise ValueError("leaf-to-macro map disagrees between taxonomy and semantics")
    out = out.drop(columns="qwen_macro").rename(columns={"leaf_macro": "qwen_macro"})
    out.to_csv(RESULTS / "cloud_leaf_counts.csv", index=False)

    groups = con.execute(f"""
      SELECT qwen_macro, treatment, count(*) AS n_papers, count(DISTINCT journal_id) AS n_journals
      FROM read_parquet('{SCORES}') GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    groups.to_csv(RESULTS / "cloud_journal_groups.csv", index=False)

    write_run("cloud", {"eligible": total, "support": support, "leaves": len(out),
                        "journal_groups": len(groups)},
              {"umap": {"n_neighbors": 15, "min_dist": 0.05, "metric": "cosine"}})
    log(f"cloud complete eligible={total:,} support={support:,} leaves={len(out)}")


if __name__ == "__main__":
    main()
