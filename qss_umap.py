#!/usr/bin/env python3
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from qss_common import SEED
from qss_v3_common import ARTIFACTS, RESULTS, V3_WORK, check_budget, connect, log, write_run

ANALYSIS = V3_WORK / "analysis_dataset.parquet"
SCORES = V3_WORK / "routing_scores.parquet"
LABELS = RESULTS / "macro_labels.csv"
POINTS = V3_WORK / "umap_papers.parquet"
JOURNALS = RESULTS / "umap_journals.csv"
FIGURES = RESULTS / "figures"
PCS = [f"qpc{i:02d}" for i in range(1, 33)]
CASE_NAMES = {
    "Biotechnology Letters", "Protein Expression and Purification",
    "Journal of Materials Science", "Journal of Solid State Electrochemistry",
}
BLUE, CORAL, INK, CLOUD = "#4DBBD5", "#E64B35", "#252525", "#506784"


def main():
    check_budget()
    for path in (ANALYSIS, SCORES, LABELS):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"expected nonempty input at {path}")
    con = connect()
    pc_sql = ",".join(f"a.{x}" for x in PCS)
    sample = con.execute(f"""
      WITH ranked AS (
        SELECT s.id,s.qwen_macro,a.journal_id,a.journal_name,a.treatment,{pc_sql},
          row_number() OVER (PARTITION BY s.qwen_macro
            ORDER BY hash(s.id || '|{SEED}')) AS rn
        FROM read_parquet('{SCORES}') s JOIN read_parquet('{ANALYSIS}') a USING(id)
      ) SELECT * EXCLUDE(rn) FROM ranked WHERE rn<=6000
    """).df()
    counts = sample.groupby("qwen_macro").size()
    if len(sample) != 192_000 or len(counts) != 32 or not (counts == 6000).all():
        raise ValueError(f"expected 6000 papers in each of 32 areas, got {counts.to_dict()}")

    x = sample[PCS].to_numpy(dtype=np.float32)
    y = sample.qwen_macro.to_numpy(dtype=np.int32)
    if not np.isfinite(x).all():
        raise ValueError("expected finite Qwen3 principal components")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.08, metric="euclidean",
                        target_metric="categorical", target_weight=0.15,
                        random_state=SEED, transform_seed=SEED, n_jobs=1)
    xy = reducer.fit_transform(x, y=y)
    if xy.shape != (192_000, 2) or not np.isfinite(xy).all():
        raise ValueError(f"expected finite UMAP coordinates (192000,2), got {xy.shape}")
    sample[["umap_x", "umap_y"]] = xy
    POINTS.parent.mkdir(parents=True, exist_ok=True)
    sample[["id", "qwen_macro", "journal_id", "treatment", "umap_x", "umap_y"]].to_parquet(
        POINTS, index=False)

    means = ",".join(f"avg({p}) AS {p}" for p in PCS)
    case_sql = ",".join("'" + name.replace("'", "''") + "'" for name in CASE_NAMES)
    journals = con.execute(f"""
      WITH base AS (
        SELECT a.journal_id,a.journal_name,s.qwen_macro,s.treatment,a.semantic_title_similarity,
          {pc_sql},
          CASE WHEN s.treatment=1 THEN (a.far-s.propensity*s.psi_far_1)/(1-s.propensity)
               ELSE (a.far-(1-s.propensity)*s.psi_far_0)/s.propensity END AS m_far,
          CASE WHEN s.treatment=1 THEN (a.near-s.propensity*s.psi_near_1)/(1-s.propensity)
               ELSE (a.near-(1-s.propensity)*s.psi_near_0)/s.propensity END AS m_near
        FROM read_parquet('{SCORES}') s JOIN read_parquet('{ANALYSIS}') a USING(id)
      ), grouped AS (
        SELECT journal_id,any_value(journal_name) journal_name,qwen_macro,treatment,count(*) n,
          avg(semantic_title_similarity) scope_score,avg(m_far)/avg(m_near) reach,{means}
        FROM base GROUP BY journal_id,qwen_macro,treatment HAVING count(*)>=500 AND avg(m_near)>0
      ), ranked AS (
        SELECT *,row_number() OVER
          (PARTITION BY treatment ORDER BY n DESC,journal_id,qwen_macro) arm_rank FROM grouped
      ) SELECT * EXCLUDE(arm_rank) FROM ranked
        WHERE arm_rank<=35 OR journal_name IN ({case_sql})
    """).df()
    arm_counts = journals.treatment.value_counts().to_dict()
    if len(journals) < 70 or set(journals.treatment) != {0, 1} \
            or min(arm_counts.values()) < 35 or not CASE_NAMES.issubset(set(journals.journal_name)):
        raise ValueError(f"expected >=35 journal markers per arm and four cases, got {arm_counts}")
    journals[["umap_x", "umap_y"]] = reducer.transform(
        journals[PCS].to_numpy(dtype=np.float32))
    if not np.isfinite(journals[["umap_x", "umap_y", "reach"]]).all().all():
        raise ValueError("expected finite journal coordinates and reach")
    JOURNALS.parent.mkdir(parents=True, exist_ok=True)
    journals.drop(columns=PCS).to_csv(JOURNALS, index=False)

    labels = pd.read_csv(LABELS)
    labels.loc[labels.qwen_macro == 18, "display_label"] = "Mixed records"
    centers = sample.groupby("qwen_macro")[["umap_x", "umap_y"]].median().reset_index()
    centers = centers.merge(labels[["qwen_macro", "display_label", "n"]], on="qwen_macro")
    show = set(centers.nlargest(8, "n").qwen_macro) | {4, 7, 8, 12}

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 86 / 25.4))
    for axis in axes:
        axis.scatter(sample.umap_x, sample.umap_y, s=0.22, c=CLOUD, alpha=0.035,
                     linewidths=0, rasterized=True)
        axis.set(xticks=[], yticks=[])
        for spine in axis.spines.values():
            spine.set_visible(False)
    for row in centers[centers.qwen_macro.isin(show)].itertuples():
        axes[0].text(row.umap_x, row.umap_y, row.display_label, ha="center", va="center",
                     fontsize=5.4, color=INK,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2})
    axes[0].set_title("The semantic landscape of scientific papers", loc="left", fontweight="bold")

    lo, hi = np.quantile(np.log(journals.reach), [0.05, 0.95])
    halo = 45 + 250 * np.clip((np.log(journals.reach) - lo) / (hi - lo), 0, 1)
    colors = np.where(journals.treatment.eq(1), CORAL, BLUE)
    axes[1].scatter(journals.umap_x, journals.umap_y, s=halo, c=colors, alpha=0.12,
                    linewidths=0, zorder=2)
    axes[1].scatter(journals.umap_x, journals.umap_y, s=17, c=colors, edgecolors="white",
                    linewidths=0.45, zorder=3)
    for row in journals[journals.journal_name.isin(CASE_NAMES)].itertuples():
        axes[1].annotate(row.journal_name, (row.umap_x, row.umap_y), xytext=(5, 5),
                         textcoords="offset points", fontsize=5.2, color=INK,
                         arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#777777"})
    axes[1].scatter([], [], s=22, c=BLUE, label="Broader-scope journal")
    axes[1].scatter([], [], s=22, c=CORAL, label="Narrower-scope journal")
    axes[1].legend(frameon=False, loc="lower left", handletextpad=0.3)
    axes[1].set_title("Where journals sit—and how widely their papers travel", loc="left", fontweight="bold")
    axes[1].text(0.99, 0.01, "Larger halo = more citations from other research areas\nrelative to citations from the same topic",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=5.5, color="#555555")
    for i, axis in enumerate(axes):
        axis.text(-0.04, 1.03, "ab"[i], transform=axis.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "figure_semantic_landscape"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    for path in (POINTS, JOURNALS, stem.with_suffix(".pdf"), stem.with_suffix(".png")):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"expected nonempty output at {path}")
    free = shutil.disk_usage(V3_WORK).free
    write_run("umap", {"papers": len(sample), "journal_markers": len(journals), "areas": 32},
              {"supervision": "qwen_macro only", "target_weight": 0.15,
               "paper_sample_per_area": 6000, "group_free_bytes": free})
    check_budget()
    log(f"UMAP complete papers={len(sample):,} journals={len(journals)} areas=32 free={free:,}")


if __name__ == "__main__":
    main()
