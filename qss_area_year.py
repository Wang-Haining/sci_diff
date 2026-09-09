#!/usr/bin/env python3
"""Research-area x publication-year cells of the routing contrast.

Reuses the saved deterministic paper-level AIPW scores (routing_scores.parquet)
and the same estimator, journal-clustered inference, multiplier weights, and
minimum-size rule as qss_downstream.py. Exploratory: 32 x 6 cells.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from qss_common import SEED
from qss_downstream import SCORES, interval_from_influence, routing_components
from qss_v3_common import RESULTS, log, write_run

EXPECTED_ROWS = 3_827_491
MIN_ARM = 5_000
MIN_JOURNALS = 50


def benjamini_hochberg(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def main():
    d = pd.read_parquet(SCORES)
    if len(d) != EXPECTED_ROWS or d.id.nunique() != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} unique score rows, got {len(d):,}")
    labels = pd.read_csv(RESULTS / "macro_labels.csv")[["qwen_macro", "display_label"]]
    codes, journals = pd.factorize(d.journal_id, sort=True)
    multipliers = np.random.default_rng(SEED).standard_normal((500, len(journals)))
    theta_all, _, _ = routing_components(d, np.ones(len(d), dtype=bool))
    rows = []
    for macro in range(32):
        for year in range(2015, 2021):
            mask = (d.qwen_macro.eq(macro) & d.publication_year.eq(year)).to_numpy()
            n0 = int(((d.treatment == 0).to_numpy() & mask).sum())
            n1 = int(((d.treatment == 1).to_numpy() & mask).sum())
            nj = int(d.loc[mask, "journal_id"].nunique())
            row = {"qwen_macro": macro, "publication_year": year, "n": int(mask.sum()),
                   "n_broad": n0, "n_specialized": n1, "journals": nj}
            if min(n0, n1) < MIN_ARM or nj < MIN_JOURNALS:
                row["status"] = "not_estimable"
                rows.append(row)
                continue
            theta, means, influence = routing_components(d, mask)
            se, lo, hi, blo, bhi, _ = interval_from_influence(
                theta, influence, codes[mask], multipliers, len(journals))
            row.update({"status": "estimated", "estimate": theta, "se": se,
                        "ci_low": lo, "ci_high": hi,
                        "bootstrap_ci_low": blo, "bootstrap_ci_high": bhi,
                        "far_near_broad": means[2] / means[3],
                        "far_near_specialized": means[0] / means[1]})
            rows.append(row)
    out = pd.DataFrame(rows).merge(labels, on="qwen_macro", how="left")
    est = out.status.eq("estimated")
    out.loc[est, "p_value"] = 2 * norm.sf(np.abs(out.loc[est, "estimate"] / out.loc[est, "se"]))
    out.loc[est, "q_value_bh"] = benjamini_hochberg(out.loc[est, "p_value"])
    out.to_csv(RESULTS / "area_year_estimates.csv", index=False)
    summary = {
        "cells": int(len(out)), "estimated": int(est.sum()),
        "negative": int((out.loc[est, "estimate"] < 0).sum()),
        "ci_excludes_zero": int(((out.loc[est, "ci_high"] < 0) | (out.loc[est, "ci_low"] > 0)).sum()),
        "ci_excludes_zero_negative": int((out.loc[est, "ci_high"] < 0).sum()),
        "bh_q_below_0.05": int((out.loc[est, "q_value_bh"] < 0.05).sum()),
        "overall_theta": theta_all,
    }
    write_run("area_year", {"rows": EXPECTED_ROWS, "journals": len(journals)}, summary)
    log(f"area x year complete: {summary}")


if __name__ == "__main__":
    main()
