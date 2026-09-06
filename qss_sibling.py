#!/usr/bin/env python3
import json
import math
import subprocess
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from qss_v3_common import ARTIFACTS, RESULTS, V3_WORK, check_budget

SEED = 20260902
CANDIDATE = V3_WORK / "candidate_focal.parquet"
ANALYSIS = V3_WORK / "analysis_dataset.parquet"
SCORES = V3_WORK / "routing_scores.parquet"
OUTPUT = RESULTS / "same_author_sensitivity.csv"


def estimate(frame, outcome, weights=None):
    y0 = frame[f"{outcome}0"].to_numpy(float)
    y1 = frame[f"{outcome}1"].to_numpy(float)
    n0 = frame.n0.to_numpy(float)
    n1 = frame.n1.to_numpy(float)
    weights = np.ones(len(frame)) if weights is None else weights

    def score(beta):
        odds = np.exp(beta)
        probability = n1 * odds / (n0 + n1 * odds)
        return np.sum(weights * (y1 - (y0 + y1) * probability))

    return brentq(score, -10, 10)


def analyze_role(frame, role):
    data = frame[frame.author_role.eq(role)].reset_index(drop=True)
    authors, codes = np.unique(data.author_id, return_inverse=True)
    far = estimate(data, "far")
    near = estimate(data, "near")
    theta = far - near
    rng = np.random.default_rng(SEED + (0 if role == "first" else 1))
    draws = []
    for _ in range(500):
        weights = rng.poisson(1, len(authors))[codes]
        draws.append(estimate(data, "far", weights) - estimate(data, "near", weights))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "author_role": role, "strata": len(data), "authors": len(authors),
        "papers": int((data.n0 + data.n1).sum()),
        "maximum_pairs": int(np.minimum(data.n0, data.n1).sum()),
        "beta_far": far, "beta_near": near, "theta": theta,
        "ratio_change": math.expm1(theta), "bootstrap_ci_low": low,
        "bootstrap_ci_high": high, "bootstrap_draws": 500,
    }


def main():
    check_budget()
    con = duckdb.connect()
    con.execute("SET threads=32")
    query = """
      WITH authors AS (
        SELECT id,publication_year,semantic_cluster,first_author_id author_id,'first' author_role
        FROM read_parquet(?) WHERE first_author_id IS NOT NULL
        UNION ALL
        SELECT id,publication_year,semantic_cluster,last_author_id,'last'
        FROM read_parquet(?) WHERE last_author_id IS NOT NULL AND last_author_id<>first_author_id
      ), eligible AS (
        SELECT a.*,d.treatment,d.journal_id,d.near,d.far
        FROM authors a JOIN read_parquet(?) d USING (id)
        JOIN read_parquet(?) s USING (id)
      )
      SELECT author_role,author_id,publication_year,semantic_cluster,
        count(*) FILTER (WHERE treatment=0) n0,count(*) FILTER (WHERE treatment=1) n1,
        sum(near) FILTER (WHERE treatment=0) near0,sum(near) FILTER (WHERE treatment=1) near1,
        sum(far) FILTER (WHERE treatment=0) far0,sum(far) FILTER (WHERE treatment=1) far1,
        count(DISTINCT journal_id) journals
      FROM eligible GROUP BY ALL HAVING n0>0 AND n1>0
    """
    frame = con.execute(query, [str(CANDIDATE), str(CANDIDATE),
                                str(ANALYSIS), str(SCORES)]).df()
    if len(frame) != 50_297 or frame.journals.min() < 2:
        raise ValueError(f"same-author strata QC failed: rows={len(frame)} min_journals={frame.journals.min()}")
    results = pd.DataFrame(analyze_role(frame, role) for role in ("first", "last"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    run = {"design": "qss_v3", "stage": "same_author", "status": "complete",
           "seed": SEED, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "strata": len(frame), "results": results.to_dict("records")}
    (ARTIFACTS / "run_same_author.json").write_text(json.dumps(run, indent=2) + "\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
