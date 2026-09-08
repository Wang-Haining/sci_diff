#!/usr/bin/env python3
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qss_common import REPO, SEED, validate_snapshot
from qss_downstream import estimate_groups, heterogeneity_test, trend_test
from qss_v3_common import V3_WORK, check_budget, connect


ANALYSIS = V3_WORK / "analysis_dataset.parquet"
SCORES = V3_WORK / "routing_scores.parquet"
OUTPUT = REPO / "outputs/prestige_strat"
DECISION = REPO / "reports/prestige_strat_decision.md"
PRIMARY_RESULTS = REPO / "results/qss_v3/dirty_estimates.csv"
PRIMARY_RUN = REPO / "artifacts/qss_v3/run_analyze.json"
BALANCE = REPO / "results/qss_v3/balance.csv"
EXPECTED_THETA = -0.09674843896193808
EXPECTED_CI = (-0.144336408333361, -0.04916046959051515)
EXPECTED_N = 3_818_173
EXPECTED_JOURNALS = 20_203
EXPECTED_ANALYSIS = 7_617_662
EXPECTED_SCORES = 3_827_491
B = 500
FUNDING_QUARTILE_SOURCE = (
    "/Users/haining/Documents/Codex/2026-09-06/"
    "thread-nsf-nih-formalize-sciscinet-open/summarize_boundaries.py"
)


def frozen_primary_guard():
    required = [DECISION, PRIMARY_RESULTS, PRIMARY_RUN, BALANCE, ANALYSIS, SCORES]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    result = pd.read_csv(PRIMARY_RESULTS)
    row = result[(result.analysis == "primary") &
                 (result.outcome == "far_to_near_routing")]
    if len(row) != 1:
        raise ValueError(f"expected one frozen primary routing row, got {len(row)}")
    row = row.iloc[0]
    got = (float(row.estimate), float(row.ci_low), float(row.ci_high),
           int(row.n), int(row.journals))
    expected = (EXPECTED_THETA, *EXPECTED_CI, EXPECTED_N, EXPECTED_JOURNALS)
    if got != expected:
        raise ValueError(f"frozen primary mismatch: expected={expected}, got={got}")
    run = json.loads(PRIMARY_RUN.read_text())
    run_got = (run["counts"]["analysis"], run["counts"]["support"],
               run["extra"]["theta"])
    run_expected = (EXPECTED_ANALYSIS, EXPECTED_N, EXPECTED_THETA)
    if run_got != run_expected:
        raise ValueError(f"frozen run mismatch: expected={run_expected}, got={run_got}")
    print(f"frozen primary reproduced theta={got[0]:.10f} "
          f"CI=({got[1]:.10f},{got[2]:.10f}) N={got[3]:,} "
          f"journals={got[4]:,}", flush=True)


def prepare_tables(con):
    con.execute(f"CREATE TEMP VIEW analysis AS SELECT * FROM read_parquet('{ANALYSIS}')")
    con.execute(f"CREATE TEMP VIEW scores AS SELECT * FROM read_parquet('{SCORES}')")
    analysis_qc = con.execute("""
      SELECT count(*),count(DISTINCT id),count(DISTINCT journal_id)
      FROM analysis
    """).fetchone()
    score_qc = con.execute("""
      SELECT count(*),count(DISTINCT id),count(DISTINCT journal_id)
      FROM scores
    """).fetchone()
    if analysis_qc[:2] != (EXPECTED_ANALYSIS, EXPECTED_ANALYSIS):
        raise ValueError(f"analysis QC mismatch: {analysis_qc}")
    if score_qc[:2] != (EXPECTED_SCORES, EXPECTED_SCORES):
        raise ValueError(f"routing-score QC mismatch: {score_qc}")
    invalid = con.execute("""
      SELECT count(*) FROM analysis
      WHERE prior_prestige IS NULL OR prior_prestige<0
         OR semantic_title_similarity IS NULL OR treatment NOT IN (0,1)
         OR choice_set_id IS NULL OR journal_id IS NULL
    """).fetchone()[0]
    if invalid:
        raise ValueError(f"invalid scope/prestige rows={invalid}")

    # Verbatim funding-repository rule: distinct journals within choice set,
    # ordered by prior prestige and journal ID, then reattached to papers.
    con.execute("""
      CREATE TEMP TABLE prestige_journals AS
      SELECT choice_set_id,journal_id,
        any_value(prior_prestige) prior_prestige,
        any_value(treatment) treatment,
        count(DISTINCT prior_prestige) n_prestige,
        count(DISTINCT treatment) n_treatment
      FROM analysis GROUP BY choice_set_id,journal_id
    """)
    bad = con.execute("""
      SELECT count(*) FROM prestige_journals
      WHERE n_prestige<>1 OR n_treatment<>1
    """).fetchone()[0]
    if bad:
        raise ValueError(f"choice-set journals with nonunique prestige/treatment={bad}")
    con.execute("""
      CREATE TEMP TABLE prestige_quartiles AS
      SELECT *,ntile(4) OVER (
        PARTITION BY choice_set_id ORDER BY prior_prestige,journal_id
      ) prestige_quartile
      FROM prestige_journals WHERE n_prestige=1 AND n_treatment=1
    """)
    con.execute("""
      CREATE TEMP TABLE comparison AS
      SELECT a.id,a.publication_year,a.journal_id,a.choice_set_id,a.treatment,
        a.semantic_title_similarity,a.prior_prestige,a.near,a.far,
        q.prestige_quartile,s.propensity,
        s.psi_near_0,s.psi_near_1,s.psi_far_0,s.psi_far_1
      FROM scores s JOIN analysis a USING (id,journal_id,choice_set_id,treatment)
      JOIN prestige_quartiles q USING (choice_set_id,journal_id,treatment)
    """)
    comparison_qc = con.execute("""
      SELECT count(*),count(DISTINCT id),count(DISTINCT journal_id),
        count(DISTINCT prestige_quartile)
      FROM comparison
    """).fetchone()
    if comparison_qc != (EXPECTED_SCORES, EXPECTED_SCORES,
                         score_qc[2], 4):
        raise ValueError(f"comparison join QC mismatch: {comparison_qc}")
    return analysis_qc, comparison_qc


def weighted_spearman(table):
    rows = []
    for sample, d in table.groupby("sample", observed=True):
        d = d.copy()
        group = d.groupby("choice_set_id", observed=True)
        d["rank_scope"] = group.semantic_title_similarity.rank(method="average")
        d["rank_prestige"] = group.log1p_prior_prestige.rank(method="average")
        moments = d.assign(
            wx=d.n * d.rank_scope, wy=d.n * d.rank_prestige,
        ).groupby("choice_set_id", observed=True).agg(
            weight=("n", "sum"), wx=("wx", "sum"), wy=("wy", "sum"),
            journals=("journal_id", "nunique"),
        )
        d = d.join((moments.wx / moments.weight).rename("mean_x"),
                   on="choice_set_id")
        d = d.join((moments.wy / moments.weight).rename("mean_y"),
                   on="choice_set_id")
        d["cov"] = d.n * (d.rank_scope - d.mean_x) * (d.rank_prestige - d.mean_y)
        d["var_x"] = d.n * np.square(d.rank_scope - d.mean_x)
        d["var_y"] = d.n * np.square(d.rank_prestige - d.mean_y)
        sums = d.groupby("choice_set_id", observed=True)[["cov", "var_x", "var_y"]].sum()
        valid = (sums.var_x > 0) & (sums.var_y > 0)
        rho = sums.loc[valid, "cov"] / np.sqrt(
            sums.loc[valid, "var_x"] * sums.loc[valid, "var_y"])
        pooled = sums.loc[valid, "cov"].sum() / math.sqrt(
            sums.loc[valid, "var_x"].sum() * sums.loc[valid, "var_y"].sum())
        rows.append({
            "record_type": "within_choice_set_spearman", "sample": sample,
            "sets": int(len(rho)), "median": float(rho.median()),
            "q25": float(rho.quantile(0.25)), "q75": float(rho.quantile(0.75)),
            "pooled": float(pooled),
        })
    return rows


def task1(con):
    journal_years = con.execute("""
      WITH all_rows AS (
        SELECT 'eligible' sample,publication_year,journal_id,
          any_value(semantic_title_similarity) semantic_title_similarity,
          any_value(prior_prestige) prior_prestige,
          count(DISTINCT semantic_title_similarity) n_scope,
          count(DISTINCT prior_prestige) n_prestige
        FROM analysis GROUP BY publication_year,journal_id
        UNION ALL
        SELECT 'comparison' sample,publication_year,journal_id,
          any_value(semantic_title_similarity),any_value(prior_prestige),
          count(DISTINCT semantic_title_similarity),count(DISTINCT prior_prestige)
        FROM comparison GROUP BY publication_year,journal_id
      ) SELECT * FROM all_rows
    """).df()
    if ((journal_years.n_scope != 1) | (journal_years.n_prestige != 1)).any():
        raise ValueError("journal-year scope/prestige was not unique")
    journal_years["log1p_prior_prestige"] = np.log1p(journal_years.prior_prestige)
    output = []
    for sample, d in journal_years.groupby("sample", observed=True):
        output.append({
            "record_type": "journal_year_spearman", "sample": sample,
            "journal_years": len(d),
            "rho": float(d.semantic_title_similarity.corr(
                d.log1p_prior_prestige, method="spearman")),
        })

    choice_journals = con.execute("""
      SELECT 'eligible' sample,choice_set_id,journal_id,
        any_value(semantic_title_similarity) semantic_title_similarity,
        ln(1+any_value(prior_prestige)) log1p_prior_prestige,count(*) n
      FROM analysis GROUP BY choice_set_id,journal_id
      UNION ALL
      SELECT 'comparison',choice_set_id,journal_id,
        any_value(semantic_title_similarity),ln(1+any_value(prior_prestige)),count(*)
      FROM comparison GROUP BY choice_set_id,journal_id
    """).df()
    output.extend(weighted_spearman(choice_journals))

    quartiles = con.execute("""
      SELECT 'eligible' sample,a.treatment,q.prestige_quartile,count(*) n,
        count(DISTINCT a.journal_id) journals,sum(a.near) near_events,
        sum(a.far) far_events
      FROM analysis a JOIN prestige_quartiles q
        USING (choice_set_id,journal_id,treatment)
      GROUP BY a.treatment,q.prestige_quartile
      UNION ALL
      SELECT 'comparison',treatment,prestige_quartile,count(*),
        count(DISTINCT journal_id),sum(near),sum(far)
      FROM comparison GROUP BY treatment,prestige_quartile
    """).df()
    quartiles["row_proportion"] = quartiles.n / quartiles.groupby(
        ["sample", "treatment"], observed=True).n.transform("sum")
    for row in quartiles.to_dict("records"):
        output.append({"record_type": "prestige_quartile_distribution", **row})

    balance = pd.read_csv(BALANCE)
    prestige = balance[(balance.candidate == "primary_leaves_63") &
                       (balance.stage == "weighted") &
                       (balance.covariate == "log1p_prior_prestige")]
    if len(prestige) != 1:
        raise ValueError(f"expected one weighted prestige SMD, got {len(prestige)}")
    output.append({
        "record_type": "current_aipw_weighted_smd", "sample": "primary",
        "smd": float(prestige.iloc[0].smd),
        "abs_smd": abs(float(prestige.iloc[0].smd)),
    })

    con.execute("""
      CREATE TEMP TABLE journal_year_ventiles AS
      WITH jy AS (
        SELECT publication_year,journal_id,
          any_value(semantic_title_similarity) semantic_title_similarity,
          any_value(prior_prestige) prior_prestige
        FROM analysis GROUP BY publication_year,journal_id
      ) SELECT *,
        ntile(20) OVER (ORDER BY semantic_title_similarity,publication_year,journal_id)
          scope_ventile,
        ntile(20) OVER (ORDER BY prior_prestige,publication_year,journal_id)
          prestige_ventile
      FROM jy
    """)
    ventiles = con.execute("""
      WITH joined AS (
        SELECT 'eligible' sample,a.treatment,v.scope_ventile,v.prestige_ventile
        FROM analysis a JOIN journal_year_ventiles v USING (publication_year,journal_id)
        UNION ALL
        SELECT 'comparison',a.treatment,v.scope_ventile,v.prestige_ventile
        FROM comparison a JOIN journal_year_ventiles v USING (publication_year,journal_id)
      ) SELECT sample,treatment,scope_ventile,prestige_ventile,count(*) n
      FROM joined GROUP BY ALL
    """).df()
    ventiles["share"] = ventiles.n / ventiles.groupby(
        ["sample", "treatment"], observed=True).n.transform("sum")
    for row in ventiles.to_dict("records"):
        output.append({"record_type": "joint_ventile", **row})
    t1 = pd.DataFrame(output)
    t1.to_csv(OUTPUT / "t1_scope_prestige.csv", index=False)
    draw_ventiles(ventiles)
    return t1


def draw_ventiles(ventiles):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True,
                             sharex=True, sharey=True)
    vmax = float((100 * ventiles.share).quantile(0.99))
    image = None
    for row, sample in enumerate(("eligible", "comparison")):
        for col, arm in enumerate((0, 1)):
            axis = axes[row, col]
            d = ventiles[(ventiles.sample == sample) & (ventiles.treatment == arm)]
            matrix = d.pivot(index="prestige_ventile", columns="scope_ventile",
                             values="share").reindex(index=range(1, 21),
                                                     columns=range(1, 21), fill_value=0)
            image = axis.imshow(100 * matrix, origin="lower", aspect="auto",
                                cmap="magma", vmin=0, vmax=vmax)
            axis.set_title(("Broader-scope" if arm == 0 else "Narrower-scope") +
                           f" | {sample}", fontsize=9)
            axis.set_xlabel("Journal-scope ventile")
            axis.set_ylabel("Prior-prestige ventile")
            axis.set_xticks([0, 9, 19], [1, 10, 20])
            axis.set_yticks([0, 9, 19], [1, 10, 20])
    colorbar = fig.colorbar(image, ax=axes, shrink=0.75, pad=0.02)
    colorbar.set_label("Papers in arm (%)")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"prestige_joint_ventiles.{suffix}", dpi=300)
    plt.close(fig)


def standardized_routing(journal_rows):
    strata = ["choice_set_id", "prestige_quartile"]
    cell = journal_rows.groupby(strata + ["treatment"], observed=True).agg(
        n=("n", "sum"), near_events=("near_events", "sum"),
        far_events=("far_events", "sum"),
    ).reset_index()
    n_wide = cell.pivot(index=strata, columns="treatment", values="n")
    near_wide = cell.pivot(index=strata, columns="treatment", values="near_events")
    far_wide = cell.pivot(index=strata, columns="treatment", values="far_events")
    if n_wide.isna().any().any() or (n_wide.min(axis=1) < 20).any():
        raise ValueError("T2 retained a missing or thin treatment cell")
    stratum_n = n_wide.sum(axis=1)
    total = float(stratum_n.sum())
    risks = {}
    means = {}
    for outcome, wide in (("near", near_wide), ("far", far_wide)):
        for arm in (0, 1):
            risks[(outcome, arm)] = wide[arm] / n_wide[arm]
            means[(outcome, arm)] = float(
                (stratum_n / total * risks[(outcome, arm)]).sum())

    stats = pd.DataFrame({"stratum_n": stratum_n, "n_0": n_wide[0],
                          "n_1": n_wide[1]})
    for key, values in risks.items():
        stats[f"risk_{key[0]}_{key[1]}"] = values
    d = journal_rows.merge(stats, left_on=strata, right_index=True)
    component_order = [("far", 1), ("near", 1), ("far", 0), ("near", 0)]
    signals = []
    for outcome, arm in component_order:
        risk = d[f"risk_{outcome}_{arm}"]
        target = d.n * (risk - means[(outcome, arm)])
        observed = d[f"{outcome}_events"]
        outcome_part = np.where(
            d.treatment.eq(arm),
            d.stratum_n / d[f"n_{arm}"] * (observed - d.n * risk), 0.0,
        )
        signals.append(target + outcome_part)
    journals = np.sort(d.journal_id.unique())
    cluster = pd.DataFrame({"journal_id": d.journal_id})
    for index, signal in enumerate(signals):
        cluster[f"signal_{index}"] = signal
    cluster = cluster.groupby("journal_id", observed=True).sum().reindex(
        journals, fill_value=0)
    multipliers = np.random.default_rng(SEED).standard_normal((B, len(journals)))
    mean_vector = np.array([means[key] for key in component_order])
    draws = mean_vector + multipliers @ cluster.to_numpy() / total
    if (draws <= 0).any():
        raise ValueError("nonpositive T2 marginal-mean bootstrap draw")
    theta = (math.log(mean_vector[0]) - math.log(mean_vector[1]) -
             math.log(mean_vector[2]) + math.log(mean_vector[3]))
    theta_draws = (np.log(draws[:, 0]) - np.log(draws[:, 1]) -
                   np.log(draws[:, 2]) + np.log(draws[:, 3]))
    ci = np.quantile(theta_draws, [0.025, 0.975])
    raw = cell.groupby("treatment", observed=True)[
        ["n", "near_events", "far_events"]].sum()
    raw_near = raw.near_events / raw.n
    raw_far = raw.far_events / raw.n
    raw_theta = (math.log(raw_far[1]) - math.log(raw_near[1]) -
                 math.log(raw_far[0]) + math.log(raw_near[0]))
    ess = {arm: total ** 2 / float((np.square(stratum_n) / n_wide[arm]).sum())
           for arm in (0, 1)}
    return {
        "theta": theta, "ci_low": float(ci[0]), "ci_high": float(ci[1]),
        "bootstrap_se": float(theta_draws.std(ddof=1)),
        "mean_near_broad": means[("near", 0)],
        "mean_far_broad": means[("far", 0)],
        "mean_near_narrow": means[("near", 1)],
        "mean_far_narrow": means[("far", 1)],
        "unstratified_raw_theta_same_set": raw_theta,
        "standardized_minus_raw_theta": theta - raw_theta,
        "ess_broad": ess[0], "ess_narrow": ess[1],
        "journals": len(journals), "effective_strata": len(n_wide),
        "retained_n": int(total), "n_broad": int(raw.loc[0, "n"]),
        "n_narrow": int(raw.loc[1, "n"]),
        "near_events_broad": int(raw.loc[0, "near_events"]),
        "near_events_narrow": int(raw.loc[1, "near_events"]),
        "far_events_broad": int(raw.loc[0, "far_events"]),
        "far_events_narrow": int(raw.loc[1, "far_events"]),
    }


def task2(con):
    cells = con.execute("""
      SELECT choice_set_id,prestige_quartile,treatment,count(*) n,
        count(DISTINCT journal_id) journals,sum(near) near_events,sum(far) far_events
      FROM comparison GROUP BY ALL
    """).df()
    status = cells.pivot(index=["choice_set_id", "prestige_quartile"],
                         columns="treatment", values="n")
    keep = status.notna().all(axis=1) & (status.min(axis=1) >= 20)
    eligible_keys = status.index[keep]
    cells = cells.set_index(["choice_set_id", "prestige_quartile"])
    cells["retained"] = cells.index.isin(eligible_keys)
    cells.reset_index().to_csv(OUTPUT / "t2_strata.csv", index=False)
    con.register("t2_keys", eligible_keys.to_frame(index=False))
    journal_rows = con.execute("""
      SELECT c.choice_set_id,c.prestige_quartile,c.treatment,c.journal_id,
        count(*) n,sum(c.near) near_events,sum(c.far) far_events
      FROM comparison c JOIN t2_keys k USING (choice_set_id,prestige_quartile)
      GROUP BY ALL
    """).df()
    result = standardized_routing(journal_rows)
    result.update({
        "comparison_sample_n": EXPECTED_SCORES,
        "retained_share": result["retained_n"] / EXPECTED_SCORES,
        "candidate_strata": len(status), "dropped_strata": int((~keep).sum()),
        "thin_or_single_arm_papers": EXPECTED_SCORES - result["retained_n"],
        "primary_theta": EXPECTED_THETA,
        "standardized_minus_primary_theta": result["theta"] - EXPECTED_THETA,
    })
    if result["ci_low"] <= 0 <= result["ci_high"]:
        branch = "FAILS"
    elif abs(result["theta"]) >= 0.048:
        branch = "SURVIVES"
    else:
        branch = "ATTENUATED"
    result["decision"] = branch
    pd.DataFrame([result]).to_csv(OUTPUT / "t2_standardized_theta.csv", index=False)
    return result


def task3(con):
    d = con.execute("""
      SELECT journal_id,treatment,prestige_quartile,near,far,
        psi_near_0,psi_near_1,psi_far_0,psi_far_1
      FROM comparison
    """).df()
    codes, journals = pd.factorize(d.journal_id, sort=True)
    multipliers = np.random.default_rng(SEED).standard_normal((B, len(journals)))
    estimates, draws = estimate_groups(
        d, "prestige_quartile", [1, 2, 3, 4], "prior_prestige",
        codes, multipliers, len(journals),
    )
    for quartile in range(1, 5):
        mask = d.prestige_quartile.eq(quartile)
        row = estimates.level.eq(quartile)
        for arm, label in ((0, "broad"), (1, "narrow")):
            arm_mask = mask & d.treatment.eq(arm)
            estimates.loc[row, f"journals_{label}"] = d.loc[
                arm_mask, "journal_id"].nunique()
            estimates.loc[row, f"near_events_{label}"] = d.loc[arm_mask, "near"].sum()
            estimates.loc[row, f"far_events_{label}"] = d.loc[arm_mask, "far"].sum()
    estimates["row_type"] = "quartile_estimate"
    tests = pd.DataFrame([
        heterogeneity_test(draws, "prior_prestige_global", "prestige_quartile"),
        trend_test(draws, [1, 2, 3, 4], "prior_prestige_linear_trend",
                   "prestige_quartile"),
    ])
    tests["row_type"] = "test"
    output = pd.concat([estimates, tests], ignore_index=True, sort=False)
    output.to_csv(OUTPUT / "t3_theta_by_prestige_q.csv", index=False)
    return output


def write_report(t1, t2, t3, counts):
    jy = t1[t1.record_type == "journal_year_spearman"].set_index("sample")
    within = t1[t1.record_type == "within_choice_set_spearman"].set_index("sample")
    smd = t1[t1.record_type == "current_aipw_weighted_smd"].iloc[0]
    quartiles = t3[t3.row_type == "quartile_estimate"]
    lines = [
        "# Prestige-stratified routing contrast",
        "",
        f"## Decision: {t2['decision']}",
        "",
        f"Direct standardization within comparison set x prior-prestige quartile "
        f"gave theta = {t2['theta']:.4f} (500-draw journal-cluster bootstrap "
        f"95% CI {t2['ci_low']:.4f} to {t2['ci_high']:.4f}). The unstratified "
        f"raw contrast on the same retained papers was {t2['unstratified_raw_theta_same_set']:.4f}; "
        f"the standardized-minus-raw change was {t2['standardized_minus_raw_theta']:.4f}.",
        "",
        "The branch follows the rule frozen in `reports/prestige_strat_decision.md`; "
        "no secondary result determines it.",
        "",
        "## T1. Scope-prestige dependence",
        "",
        f"Across distinct journal-years, unweighted Spearman rho was "
        f"{jy.loc['eligible','rho']:.3f} in the eligible cohort and "
        f"{jy.loc['comparison','rho']:.3f} in the saved comparison sample. "
        f"Within comparison sets, the paper-weighted rho had median "
        f"{within.loc['comparison','median']:.3f} (IQR "
        f"{within.loc['comparison','q25']:.3f} to {within.loc['comparison','q75']:.3f}) "
        f"and pooled rho {within.loc['comparison','pooled']:.3f}. The current "
        f"AIPW-weighted SMD for log(1 + prior prestige) was {smd.smd:.3f} "
        f"(|SMD| {smd.abs_smd:.3f}).",
        "",
        "The joint ventile plot is `prestige_joint_ventiles.pdf`; the 2 x 4 arm "
        "tables and all ventile cells are in `t1_scope_prestige.csv`.",
        "",
        "## T2. Direct standardization",
        "",
        f"The analysis retained {t2['retained_n']:,} papers "
        f"({100*t2['retained_share']:.1f}% of the saved deterministic comparison "
        f"sample) in {t2['effective_strata']:,} strata and {t2['journals']:,} journals. "
        f"Standardized means were near={t2['mean_near_broad']:.3f}, "
        f"far={t2['mean_far_broad']:.3f} for broader-scope journals and "
        f"near={t2['mean_near_narrow']:.3f}, far={t2['mean_far_narrow']:.3f} "
        f"for narrower-scope journals.",
        "",
        "## T3. Deterministic AIPW scores by prestige quartile",
        "",
        "| Prestige quartile | theta | 95% CI | N broad | N narrow |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in quartiles.iterrows():
        lines.append(
            f"| {int(row.level)} | {row.estimate:.4f} | {row.ci_low:.4f} to "
            f"{row.ci_high:.4f} | {int(row.n_broad):,} | {int(row.n_specialized):,} |"
        )
    global_test = t3[t3.test == "prior_prestige_global"].iloc[0]
    trend = t3[t3.test == "prior_prestige_linear_trend"].iloc[0]
    lines.extend([
        "",
        f"Global heterogeneity p={global_test.p_value:.3g}; linear-trend estimate "
        f"per quartile={trend.estimate:.4f} (p={trend.p_value:.3g}).",
        "",
        "## T4. Prestige-balanced AIPW refit",
        "",
        ("Required by the frozen rule and not yet run." if t2["decision"] != "FAILS"
         else "Not run because Task 2 reached FAILS."),
        "",
        "## Provenance",
        "",
        f"Frozen primary exact guard: theta={EXPECTED_THETA:.10f}, "
        f"CI=({EXPECTED_CI[0]:.10f}, {EXPECTED_CI[1]:.10f}), "
        f"N={EXPECTED_N:,}, journals={EXPECTED_JOURNALS:,}. T1-T3 use the saved "
        f"deterministic paper-level comparison scores (N={EXPECTED_SCORES:,}); these "
        "diagnostics do not replace the frozen primary analysis.",
        "",
        f"Prestige-quartile source copied verbatim from `{FUNDING_QUARTILE_SOURCE}`, "
        "function `stratified_counts`: quartiles over distinct journals within each "
        "comparison set, followed by reattachment to papers.",
        "",
        f"Analysis rows={counts[0][0]:,}; comparison rows={counts[1][0]:,}; "
        f"bootstrap draws={B}; seed={SEED}.",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n")


def main():
    frozen_primary_guard()
    validate_snapshot()
    check_budget()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    con = connect()
    counts = prepare_tables(con)
    t1 = task1(con)
    t2 = task2(con)
    t3 = task3(con)
    write_report(t1, t2, t3, counts)
    run = {
        "stage": "prestige_strat_t1_t3", "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "seed": SEED, "bootstrap_draws": B, "decision": t2["decision"],
        "counts": {"analysis": counts[0][0], "comparison": counts[1][0],
                   "retained_t2": t2["retained_n"],
                   "effective_strata_t2": t2["effective_strata"]},
        "theta_t2": t2["theta"], "ci_t2": [t2["ci_low"], t2["ci_high"]],
        "frozen_primary": {"theta": EXPECTED_THETA, "ci": list(EXPECTED_CI),
                           "n": EXPECTED_N, "journals": EXPECTED_JOURNALS},
        "funding_quartile_source": FUNDING_QUARTILE_SOURCE,
        "group_free_bytes": shutil.disk_usage("/home/group/jasonclark").free,
    }
    (OUTPUT / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    check_budget()
    print(f"prestige stratification complete decision={t2['decision']} "
          f"theta={t2['theta']:.6f} CI=({t2['ci_low']:.6f},"
          f"{t2['ci_high']:.6f}) retained={t2['retained_n']:,}", flush=True)


if __name__ == "__main__":
    main()
