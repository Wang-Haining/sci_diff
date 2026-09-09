#!/usr/bin/env python3
"""Enrichment of citation flow by title-content distance (Figure 4a).

Same eligible external citation edges, same IPW standardization, same journal
multiplier weights as qss_network.py. Each edge is placed in a bin of the cosine
distance between the cited paper's leaf-topic center and the citing paper's
leaf-topic center (1,000 frozen Qwen3 leaves). Within each journal group the
edge weight in a bin is expressed as a share of the source area's flow, then
standardized to the common source-area distribution pi. The enrichment is
log2(narrower share / broader share) per bin.
"""
import numpy as np
import pandas as pd

from qss_common import SEED
from qss_network import (BOOTSTRAPS, LEAVES, MACROS, build_eligible_edges, connect,
                         load_taxonomy, validate_inputs)
from qss_v3_common import RESULTS, log, write_run

N_BINS = 16   # distance bins beyond the same-leaf bin


def main():
    leaf_to_macro, macro_centers, distances, expected_qwen = load_taxonomy()
    con = connect("650GB", 32)
    support_n, journal_n, raw_edge_n = validate_inputs(con, leaf_to_macro, expected_qwen)
    decomposition = build_eligible_edges(con)

    table = pd.DataFrame({
        "source_leaf": np.repeat(np.arange(LEAVES, dtype=np.int16), LEAVES),
        "target_leaf": np.tile(np.arange(LEAVES, dtype=np.int16), LEAVES),
        "semantic_distance": distances.astype(np.float32).ravel(),
    })
    con.register("leaf_distances", table)
    con.execute("""
      CREATE TEMP TABLE edge_dist AS
      SELECT e.journal_code, e.treatment, e.source_macro, e.target_macro, e.ipw,
             d.semantic_distance, e.source_leaf = e.target_leaf AS same_leaf
      FROM network_eligible_edges e JOIN leaf_distances d USING (source_leaf, target_leaf)
    """)
    # Bin edges: same-leaf pairs form bin 0; other pairs are cut at pooled IPW-weighted
    # quantiles of distance so every bin carries comparable weight.
    qs = np.linspace(0, 1, N_BINS + 1)[1:-1]
    cuts = con.execute(f"""
      SELECT {", ".join(f"quantile_cont(semantic_distance, {q}) " for q in qs)}
      FROM edge_dist WHERE NOT same_leaf
    """).fetchone()
    cuts = np.array(cuts, dtype=float)
    con.register("cuts", pd.DataFrame({"cut": cuts, "k": np.arange(1, N_BINS)}))
    agg = con.execute("""
      SELECT journal_code, treatment, source_macro,
             CASE WHEN same_leaf THEN 0
                  ELSE 1 + (SELECT count(*) FROM cuts WHERE cut <= semantic_distance) END AS bin,
             count(*) AS raw_edges, sum(ipw) AS w
      FROM edge_dist GROUP BY ALL
    """).df()
    n_bins = N_BINS + 1
    if agg.bin.min() != 0 or agg.bin.max() != n_bins - 1:
        raise ValueError(f"unexpected bin range {agg.bin.min()}..{agg.bin.max()}")
    if int(agg.raw_edges.sum()) != decomposition[7]:
        raise ValueError("eligible-edge reconciliation failed")

    W = np.zeros((journal_n, 2, MACROS, n_bins))
    W[agg.journal_code.to_numpy(), agg.treatment.to_numpy(), agg.source_macro.to_numpy(),
      agg.bin.to_numpy()] = agg.w.to_numpy()
    focal = con.execute("""
      SELECT journal_code, treatment, source_macro, count(*) AS n FROM network_support GROUP BY ALL
    """).df()
    F = np.zeros((journal_n, 2, MACROS))
    F[focal.journal_code.to_numpy(), focal.treatment.to_numpy(), focal.source_macro.to_numpy()] = focal.n.to_numpy()

    def shares(Wj, Fj):
        # Wj: (..., 2, MACROS, n_bins); Fj: (..., 2, MACROS)
        pi = Fj.sum(axis=-2) / Fj.sum(axis=(-2, -1), keepdims=True)[..., 0]  # (..., MACROS)
        row = Wj.sum(axis=-1, keepdims=True)
        P = np.divide(Wj, row, out=np.zeros_like(Wj), where=row > 0)
        return (pi[..., None, :, None] * P).sum(axis=-2)                       # (..., 2, n_bins)

    base = shares(W.sum(axis=0), F.sum(axis=0))
    rng = np.random.default_rng(SEED)
    mult = rng.poisson(1, size=(BOOTSTRAPS, journal_n)).astype(float)
    Wd = (mult @ W.reshape(journal_n, -1)).reshape(BOOTSTRAPS, 2, MACROS, n_bins)
    Fd = (mult @ F.reshape(journal_n, -1)).reshape(BOOTSTRAPS, 2, MACROS)
    draws = shares(Wd, Fd)
    ratio = np.log2(base[1] / base[0])
    ratio_d = np.log2(draws[:, 1, :] / draws[:, 0, :])
    lo, hi = np.quantile(ratio_d, [0.025, 0.975], axis=0)
    edges_lo = np.concatenate([[0.0, 0.0], cuts])
    edges_hi = np.concatenate([[0.0], cuts, [float(distances.max())]])
    raw_by_bin = agg.groupby(["treatment", "bin"]).raw_edges.sum().unstack(fill_value=0)
    out = pd.DataFrame({
        "bin": np.arange(n_bins), "distance_lo": edges_lo, "distance_hi": edges_hi,
        "same_topic": [True] + [False] * N_BINS,
        "share_broad": base[0], "share_narrow": base[1],
        "log2_ratio_narrow_over_broad": ratio, "ci_low": lo, "ci_high": hi,
        "raw_edges_broad": raw_by_bin.loc[0].reindex(range(n_bins), fill_value=0).to_numpy(),
        "raw_edges_narrow": raw_by_bin.loc[1].reindex(range(n_bins), fill_value=0).to_numpy(),
    })
    out.to_csv(RESULTS / "distance_enrichment.csv", index=False)
    # Coarse reconciliation: same leaf / same macro / other macro.
    coarse = con.execute("""
      SELECT treatment, CASE WHEN same_leaf THEN 'same_topic'
                             WHEN source_macro = target_macro THEN 'same_area_other_topic'
                             ELSE 'other_area' END AS category, sum(ipw) AS w, count(*) AS n
      FROM edge_dist GROUP BY ALL ORDER BY 1, 2
    """).df()
    coarse.to_csv(RESULTS / "distance_enrichment_coarse.csv", index=False)
    write_run("distance_enrichment", {"support": support_n, "journals": journal_n,
                                      "eligible_edges": int(decomposition[7]), "bins": n_bins},
              {"bin_cuts": cuts.tolist(), "log2_ratio": ratio.tolist(),
               "monotone_decreasing_beyond_same_topic": bool(np.all(np.diff(ratio[1:]) <= 0))})
    log(f"distance enrichment complete: log2 ratios {np.round(ratio, 3).tolist()}")


if __name__ == "__main__":
    main()
