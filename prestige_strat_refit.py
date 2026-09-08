#!/usr/bin/env python3
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prestige_strat import OUTPUT, frozen_primary_guard
from qss_common import REPO, SEED, validate_snapshot
from qss_downstream import (
    aipw_scores, fit_propensity, fit_routing_predictions, interval_from_influence,
    load_frame, routing_components,
)
from qss_v3_analyze import BASE_NUMERIC, CATEGORICAL
from qss_v3_common import check_budget, connect


B = 500
NEW_CATEGORY = "choice_prestige_stratum"


def attach_prestige_strata(con, frame):
    con.register("analysis_ids", frame[["id"]])
    mapping = con.execute("""
      WITH journal_rows AS (
        SELECT choice_set_id,journal_id,
          any_value(prior_prestige) prior_prestige,
          any_value(treatment) treatment,
          count(DISTINCT prior_prestige) n_prestige,
          count(DISTINCT treatment) n_treatment
        FROM analysis_ids i JOIN read_parquet(
          '/home/group/jasonclark/g91p721/sci_diff/qss_v3/analysis_dataset.parquet'
        ) a USING (id)
        GROUP BY choice_set_id,journal_id
      ), quartiles AS (
        SELECT *,ntile(4) OVER (
          PARTITION BY choice_set_id ORDER BY prior_prestige,journal_id
        ) prestige_quartile
        FROM journal_rows WHERE n_prestige=1 AND n_treatment=1
      )
      SELECT choice_set_id,journal_id,treatment,prestige_quartile
      FROM quartiles
    """).df()
    expected = frame[["choice_set_id", "journal_id", "treatment"]].drop_duplicates()
    if len(mapping) != len(expected):
        raise ValueError(f"prestige mapping mismatch: mapping={len(mapping)} "
                         f"expected={len(expected)}")
    frame["_row_order"] = np.arange(len(frame))
    frame = frame.merge(mapping, on=["choice_set_id", "journal_id", "treatment"],
                        how="left", validate="many_to_one", sort=False)
    frame = frame.sort_values("_row_order", kind="mergesort").drop(columns="_row_order")
    if frame.prestige_quartile.isna().any():
        raise ValueError(f"missing prestige quartiles={frame.prestige_quartile.isna().sum()}")
    choice = frame.choice_set_id.astype(int).to_numpy()
    quartile = frame.prestige_quartile.to_numpy(dtype=int)
    frame[NEW_CATEGORY] = pd.Categorical(choice * 4 + quartile)
    if NEW_CATEGORY not in CATEGORICAL:
        CATEGORICAL.append(NEW_CATEGORY)
    return frame


def update_report(row):
    path = OUTPUT / "report.md"
    text = path.read_text()
    old = "Required by the frozen rule and not yet run."
    if old not in text:
        raise ValueError("expected pending Task 4 sentence in report")
    if row["status"] == "gate_failed":
        replacement = (
            f"The refit failed its acceptance gate: support retention was "
            f"{100*row['support']:.1f}% and the weighted prestige |SMD| was "
            f"{row['prestige_abs_smd']:.3f}. No Task 4 outcome estimate was made."
        )
    else:
        replacement = (
            f"The prestige-stratified propensity refit passed its gate "
            f"(support {100*row['support']:.1f}%; weighted prestige |SMD| "
            f"{row['prestige_abs_smd']:.3f}). Its AIPW routing contrast was "
            f"theta={row['theta']:.4f} (500-draw journal-cluster bootstrap 95% CI "
            f"{row['bootstrap_ci_low']:.4f} to {row['bootstrap_ci_high']:.4f}; "
            f"N={row['n']:,}, journals={row['journals']:,})."
        )
    path.write_text(text.replace(old, replacement))


def main():
    frozen_primary_guard()
    validate_snapshot()
    check_budget()
    t2 = pd.read_csv(OUTPUT / "t2_standardized_theta.csv").iloc[0]
    if t2.decision not in ("SURVIVES", "ATTENUATED"):
        raise ValueError(f"Task 4 is not permitted after decision={t2.decision}")
    con = connect()
    frame = attach_prestige_strata(con, load_frame(con))
    propensity, _, support, balance, diagnostic = fit_propensity(
        frame, BASE_NUMERIC, 63, "t4_prestige_stratum", deterministic=True,
    )
    prestige = balance[(balance.stage == "weighted") &
                       (balance.covariate == "log1p_prior_prestige")]
    if len(prestige) != 1:
        raise ValueError(f"expected one T4 prestige balance row, got {len(prestige)}")
    prestige_smd = float(prestige.iloc[0].smd)
    gate = diagnostic["support"] >= 0.45 and abs(prestige_smd) <= 0.10
    row = {
        "status": "accepted" if gate else "gate_failed",
        "support": diagnostic["support"], "support_n": diagnostic["support_n"],
        "prestige_smd": prestige_smd, "prestige_abs_smd": abs(prestige_smd),
        "max_weighted_abs_smd": diagnostic["max_weighted_abs_smd"],
        "ess_broad": diagnostic["ess_broad"],
        "ess_narrow": diagnostic["ess_specialized"],
        "best_iterations": diagnostic["best_iterations"],
    }
    balance.to_csv(OUTPUT / "t4_balance.csv", index=False)
    if gate:
        predictions, diagnostics = fit_routing_predictions(frame, BASE_NUMERIC)
        diagnostics.to_csv(OUTPUT / "t4_outcome_diagnostics.csv", index=False)
        d = aipw_scores(frame, support, propensity, predictions)
        mask = np.ones(len(d), dtype=bool)
        theta, means, influence = routing_components(d, mask)
        codes, journals = pd.factorize(d.journal_id, sort=True)
        multipliers = np.random.default_rng(SEED).standard_normal((B, len(journals)))
        interval = interval_from_influence(
            theta, influence, codes, multipliers, len(journals),
        )
        row.update({
            "theta": theta, "se": interval[0], "ci_low": interval[1],
            "ci_high": interval[2], "bootstrap_ci_low": interval[3],
            "bootstrap_ci_high": interval[4], "n": len(d),
            "journals": len(journals), "mean_far_narrow": means[0],
            "mean_near_narrow": means[1], "mean_far_broad": means[2],
            "mean_near_broad": means[3],
        })
    pd.DataFrame([row]).to_csv(OUTPUT / "t4_aipw_prestige_strat.csv", index=False)
    update_report(row)
    run = {
        "stage": "prestige_strat_t4", "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "seed": SEED, "bootstrap_draws": B, "gate_passed": gate,
        "diagnostic": diagnostic, "result": row,
        "group_free_bytes": shutil.disk_usage("/home/group/jasonclark").free,
    }
    (OUTPUT / "run_t4.json").write_text(json.dumps(run, indent=2) + "\n")
    check_budget()
    if gate:
        print(f"T4 complete theta={row['theta']:.6f} "
              f"bootstrap_CI=({row['bootstrap_ci_low']:.6f},"
              f"{row['bootstrap_ci_high']:.6f}) support={row['support']:.3f} "
              f"prestige_abs_smd={row['prestige_abs_smd']:.3f}", flush=True)
    else:
        print(f"T4 gate failed support={row['support']:.3f} "
              f"prestige_abs_smd={row['prestige_abs_smd']:.3f}", flush=True)


if __name__ == "__main__":
    main()
