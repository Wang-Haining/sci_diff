#!/usr/bin/env python3
import json
import math
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import chi2

from qss_common import GROUP_ROOT, SEED
from qss_v3_analyze import BASE_NUMERIC, CATEGORICAL, HEAVY, fold_features, fit_propensity
from qss_v3_common import (
    ARTIFACTS, RESULTS, V2_WORK, V3_WORK, check_budget, connect, log,
    path_glob, tree_bytes, validate_snapshot, write_run,
)

ANALYSIS = V3_WORK / "analysis_dataset.parquet"
QWEN_V2 = V2_WORK / "qwen3_semantics.parquet"
QWEN_V3 = V3_WORK / "qwen3_semantics"
SCORES = V3_WORK / "routing_scores.parquet"
EXPECTED_SUPPORT = 3_818_173
EXPECTED_THETA = -0.09674843896193808
OUTCOMES = ("near", "far")


def load_frame(con):
    frame = con.execute(f"""
      WITH qwen AS (
        SELECT id,qwen_macro FROM read_parquet('{QWEN_V2}')
        UNION ALL
        SELECT id,qwen_macro FROM read_parquet('{path_glob(QWEN_V3)}')
      )
      SELECT a.*,q.qwen_macro
      FROM read_parquet('{ANALYSIS}') a JOIN qwen q USING (id)
    """).df()
    if len(frame) != 7_617_662 or frame.id.nunique() != len(frame):
        raise ValueError(f"expected 7,617,662 unique rows, got rows={len(frame)} "
                         f"ids={frame.id.nunique()}")
    if frame.qwen_macro.isna().any() or frame.qwen_macro.nunique() != 32:
        raise ValueError(f"expected 32 complete macroclusters, got "
                         f"missing={frame.qwen_macro.isna().sum()} "
                         f"clusters={frame.qwen_macro.nunique()}")
    for name in HEAVY:
        if (frame[name] < 0).any():
            raise ValueError(f"expected nonnegative {name}")
        frame[f"log1p_{name}"] = np.log1p(frame[name].to_numpy(dtype=float)).astype(np.float32)
    for name in CATEGORICAL:
        frame[name] = frame[name].astype("category")
    frame["treatment"] = frame.treatment.astype(np.int8)
    return frame


def fit_routing_predictions(frame, numeric):
    treatment = frame.treatment.to_numpy(dtype=np.int8)
    folds = frame.fold.to_numpy()
    predictions = {name: [np.full(len(frame), np.nan, dtype=np.float32),
                          np.full(len(frame), np.nan, dtype=np.float32)]
                   for name in OUTCOMES}
    diagnostics = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        counts = frame.loc[train].groupby(
            "choice_set_id", observed=True,
        ).treatment.agg(["sum", "count"])
        prevalence = (counts["sum"] + 0.5) / (counts["count"] + 1)
        x_train = fold_features(frame, train, numeric, prevalence)
        x_test = fold_features(frame, test, numeric, prevalence)
        train_treatment = treatment[train]
        bucket = frame.loc[train, "early_stop_bucket"].to_numpy()
        for outcome in OUTCOMES:
            y = frame.loc[train, outcome].to_numpy(dtype=float)
            for arm in (0, 1):
                fit = (train_treatment == arm) & (bucket != 0)
                valid = (train_treatment == arm) & (bucket == 0)
                model = lgb.LGBMRegressor(
                    objective="poisson", n_estimators=3000, learning_rate=0.05,
                    num_leaves=255, min_child_samples=100, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8,
                    random_state=SEED + 100 * fold + arm, n_jobs=32, verbosity=-1,
                )
                model.fit(
                    x_train.iloc[fit], y[fit],
                    eval_set=[(x_train.iloc[valid], y[valid])], eval_metric="poisson",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                    categorical_feature="auto",
                )
                predictions[outcome][arm][test] = model.predict(
                    x_test, num_iteration=model.best_iteration_,
                )
                diagnostics.append({
                    "fold": fold, "outcome": outcome, "arm": arm,
                    "best_iteration": int(model.best_iteration_),
                    "validation_loss": float(model.best_score_["valid_0"]["poisson"]),
                })
                log(f"downstream outcome fold={fold} name={outcome} arm={arm} "
                    f"best_iteration={model.best_iteration_}")
    if any(not np.isfinite(predictions[name][arm]).all()
           for name in OUTCOMES for arm in (0, 1)):
        raise ValueError("expected finite near/far predictions for every row")
    return predictions, pd.DataFrame(diagnostics)


def aipw_scores(frame, support, propensity, predictions):
    d = frame.loc[support].reset_index(drop=True)
    a = d.treatment.to_numpy(dtype=np.int8)
    p = propensity[support].astype(float)
    for outcome in OUTCOMES:
        y = d[outcome].to_numpy(dtype=float)
        m0 = predictions[outcome][0][support].astype(float)
        m1 = predictions[outcome][1][support].astype(float)
        d[f"psi_{outcome}_0"] = m0 + (1 - a) * (y - m0) / (1 - p)
        d[f"psi_{outcome}_1"] = m1 + a * (y - m1) / p
    d["propensity"] = p.astype(np.float32)
    return d


def routing_components(d, mask):
    names = ["psi_far_1", "psi_near_1", "psi_far_0", "psi_near_0"]
    values = [d.loc[mask, name].to_numpy(dtype=float) for name in names]
    means = np.array([value.mean() for value in values])
    if len(values[0]) == 0 or np.min(means) <= 0:
        raise ValueError(f"invalid routing group rows={len(values[0])} means={means}")
    theta = math.log(means[0]) - math.log(means[1]) - math.log(means[2]) + math.log(means[3])
    influence = ((values[0] - means[0]) / means[0]
                 - (values[1] - means[1]) / means[1]
                 - (values[2] - means[2]) / means[2]
                 + (values[3] - means[3]) / means[3])
    return theta, means, influence


def interval_from_influence(theta, influence, codes, multipliers, groups):
    sums = np.bincount(codes, weights=influence, minlength=groups)
    n = len(influence)
    se = math.sqrt((groups / (groups - 1)) * np.square(sums).sum() / n ** 2)
    draws = theta + multipliers @ sums / n
    return se, theta - 1.96 * se, theta + 1.96 * se, \
        float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), draws


def add_quartiles(d, value, output, within_choice=False):
    result = np.zeros(len(d), dtype=np.int8)
    valid = d[value].notna().to_numpy()
    block = d.loc[valid, ["id", value] + (["choice_set_id"] if within_choice else [])].copy()
    sort = (["choice_set_id"] if within_choice else []) + [value, "id"]
    block = block.sort_values(sort, kind="mergesort")
    if within_choice:
        rank = block.groupby("choice_set_id", observed=True).cumcount()
        size = block.groupby("choice_set_id", observed=True)[value].transform("size")
    else:
        rank = np.arange(len(block))
        size = len(block)
    block[output] = np.minimum(4, np.floor(4 * rank / size).astype(int) + 1)
    result[block.index] = block[output].to_numpy(dtype=np.int8)
    d[output] = result
    counts = d.loc[d[output].gt(0), output].value_counts()
    if set(counts.index) != {1, 2, 3, 4}:
        raise ValueError(f"expected four {output} groups, got {counts.to_dict()}")


def estimate_groups(d, modifier, levels, test, codes, multipliers, groups):
    rows, draws = [], {}
    for order, level in enumerate(levels, start=1):
        mask = d[modifier].eq(level).to_numpy()
        n0 = int(((d.treatment == 0).to_numpy() & mask).sum())
        n1 = int(((d.treatment == 1).to_numpy() & mask).sum())
        journals = int(d.loc[mask, "journal_id"].nunique())
        if min(n0, n1) < 5_000 or journals < 50:
            rows.append({"test": test, "modifier": modifier, "level": level,
                         "order": order, "status": "not_estimable", "n": int(mask.sum()),
                         "n_broad": n0, "n_specialized": n1, "journals": journals})
            continue
        theta, means, influence = routing_components(d, mask)
        interval = interval_from_influence(
            theta, influence, codes[mask], multipliers, groups,
        )
        rows.append({
            "test": test, "modifier": modifier, "level": level, "order": order,
            "status": "estimated", "estimate": theta, "se": interval[0],
            "ci_low": interval[1], "ci_high": interval[2],
            "bootstrap_ci_low": interval[3], "bootstrap_ci_high": interval[4],
            "far_near_broad": means[2] / means[3],
            "far_near_specialized": means[0] / means[1],
            "n": int(mask.sum()), "n_broad": n0, "n_specialized": n1,
            "journals": journals,
        })
        draws[str(level)] = interval[5]
    return pd.DataFrame(rows), draws


def contrast_test(draws, high, low, name, modifier):
    if str(high) not in draws or str(low) not in draws:
        return {"test": name, "modifier": modifier, "status": "not_estimable"}
    values = draws[str(high)] - draws[str(low)]
    estimate = float(values.mean())
    se = float(values.std(ddof=1))
    z = estimate / se
    return {"test": name, "modifier": modifier, "status": "estimated",
            "estimate": estimate, "se": se,
            "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
            "bootstrap_ci_low": float(np.quantile(values, 0.025)),
            "bootstrap_ci_high": float(np.quantile(values, 0.975)),
            "p_value": float(math.erfc(abs(z) / math.sqrt(2)))}


def heterogeneity_test(draws, name, modifier):
    keys = sorted(draws, key=lambda value: float(value))
    if len(keys) < 2:
        return {"test": name, "modifier": modifier, "status": "not_estimable"}
    matrix = np.column_stack([draws[key] for key in keys])
    estimates = matrix.mean(axis=0)
    contrasts = estimates[1:] - estimates[0]
    covariance = np.cov(matrix[:, 1:] - matrix[:, [0]], rowvar=False)
    covariance = np.atleast_2d(covariance)
    statistic = float(contrasts @ np.linalg.pinv(covariance) @ contrasts)
    return {"test": name, "modifier": modifier, "status": "estimated",
            "estimate": statistic, "df": len(contrasts),
            "p_value": float(chi2.sf(statistic, len(contrasts)))}


def trend_test(draws, levels, name, modifier):
    keys = [str(level) for level in levels if str(level) in draws]
    x = np.array([float(key) for key in keys])
    matrix = np.column_stack([draws[key] for key in keys])
    slopes = ((matrix - matrix.mean(axis=1, keepdims=True)) @ (x - x.mean())
              / np.square(x - x.mean()).sum())
    estimate = float(slopes.mean())
    se = float(slopes.std(ddof=1))
    return {"test": name, "modifier": modifier, "status": "estimated",
            "estimate": estimate, "se": se,
            "ci_low": estimate - 1.96 * se, "ci_high": estimate + 1.96 * se,
            "bootstrap_ci_low": float(np.quantile(slopes, 0.025)),
            "bootstrap_ci_high": float(np.quantile(slopes, 0.975)),
            "p_value": float(math.erfc(abs(estimate / se) / math.sqrt(2)))}


def continuous_test(d, value, name, codes, multipliers, groups):
    mask = d[value].notna().to_numpy()
    theta, _, influence = routing_components(d, mask)
    x = d.loc[mask, value].to_numpy(dtype=float)
    x = (x - x.mean()) / x.std()
    design = np.column_stack([np.ones(len(x)), x])
    signal = theta + influence
    beta = np.linalg.solve(design.T @ design, design.T @ signal)
    residual = signal - design @ beta
    bread = np.linalg.inv(design.T @ design / len(x))
    contributions = (design * residual[:, None]) @ bread.T
    journal_sums = np.zeros((groups, 2))
    np.add.at(journal_sums, codes[mask], contributions)
    slope_draws = beta[1] + multipliers @ journal_sums[:, 1] / len(x)
    se = math.sqrt((groups / (groups - 1)) * np.square(journal_sums[:, 1]).sum() / len(x) ** 2)
    return {"test": name, "modifier": value, "status": "estimated",
            "estimate": float(beta[1]), "se": se,
            "ci_low": float(beta[1] - 1.96 * se), "ci_high": float(beta[1] + 1.96 * se),
            "bootstrap_ci_low": float(np.quantile(slope_draws, 0.025)),
            "bootstrap_ci_high": float(np.quantile(slope_draws, 0.975)),
            "p_value": float(math.erfc(abs(beta[1] / se) / math.sqrt(2))),
            "n": int(mask.sum())}


def main():
    validate_snapshot()
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    frame = load_frame(con)
    propensity, prevalence, support, balance, diagnostic = fit_propensity(
        frame, BASE_NUMERIC, 63, "downstream_reproduction",
    )
    if int(support.sum()) != EXPECTED_SUPPORT:
        raise ValueError(f"expected support={EXPECTED_SUPPORT:,}, got {support.sum():,}")
    predictions, outcome_diagnostics = fit_routing_predictions(frame, BASE_NUMERIC)
    d = aipw_scores(frame, support, propensity, predictions)
    theta, means, _ = routing_components(d, np.ones(len(d), dtype=bool))
    if abs(theta - EXPECTED_THETA) > 0.0005:
        raise ValueError(f"expected theta within 0.0005 of {EXPECTED_THETA}, got {theta}")

    add_quartiles(d, "reference_entropy", "reference_entropy_quartile", True)
    add_quartiles(d, "lead_prior_embedding_breadth", "author_breadth_quartile")
    add_quartiles(d, "author_mean_prior_works", "author_works_quartile")
    means_by_choice = d.groupby("choice_set_id", observed=True).reference_entropy.transform("mean")
    sd_by_choice = d.groupby("choice_set_id", observed=True).reference_entropy.transform("std")
    d["reference_entropy_z"] = ((d.reference_entropy - means_by_choice) / sd_by_choice).fillna(0)

    codes, journals = pd.factorize(d.journal_id, sort=True)
    multipliers = np.random.default_rng(SEED).standard_normal((500, len(journals)))
    specifications = [
        ("reference_entropy_quartile", [1, 2, 3, 4], "paper_venue_fit"),
        ("author_breadth_quartile", [1, 2, 3, 4], "author_audience_breadth"),
        ("author_works_quartile", [1, 2, 3, 4], "author_publication_experience"),
        ("qwen_macro", list(range(32)), "semantic_domain"),
        ("publication_year", list(range(2015, 2021)), "publication_year"),
    ]
    estimate_tables, tests = [], []
    all_draws = {}
    for modifier, levels, test in specifications:
        table, draws = estimate_groups(
            d, modifier, levels, test, codes, multipliers, len(journals),
        )
        estimate_tables.append(table)
        all_draws[modifier] = draws
        tests.append(heterogeneity_test(draws, f"{test}_global", modifier))
        if levels == [1, 2, 3, 4]:
            tests.append(contrast_test(draws, 4, 1, f"{test}_q4_minus_q1", modifier))
    tests.append(trend_test(
        all_draws["publication_year"], range(2015, 2021),
        "publication_year_linear_trend", "publication_year",
    ))
    tests.append(continuous_test(
        d, "reference_entropy_z", "paper_venue_fit_continuous",
        codes, multipliers, len(journals),
    ))

    score_columns = [
        "id", "journal_id", "publication_year", "qwen_macro", "treatment",
        "propensity", "choice_set_id", "reference_entropy", "reference_entropy_z",
        "reference_entropy_quartile", "lead_prior_embedding_breadth",
        "author_breadth_quartile", "author_mean_prior_works", "author_works_quartile",
        "psi_near_0", "psi_near_1", "psi_far_0", "psi_far_1",
    ]
    d[score_columns].to_parquet(SCORES, index=False, compression="zstd")
    score_rows = con.execute("SELECT count(*),count(DISTINCT id) FROM read_parquet(?)",
                             [str(SCORES)]).fetchone()
    if score_rows != (EXPECTED_SUPPORT, EXPECTED_SUPPORT):
        raise ValueError(f"routing score QC failed: {score_rows}")

    subgroup_estimates = pd.concat(estimate_tables, ignore_index=True)
    subgroup_tests = pd.DataFrame(tests)
    subgroup_estimates.to_csv(RESULTS / "subgroup_estimates.csv", index=False)
    subgroup_tests.to_csv(RESULTS / "subgroup_tests.csv", index=False)
    balance.to_csv(RESULTS / "downstream_balance.csv", index=False)
    outcome_diagnostics.to_csv(RESULTS / "downstream_outcome_diagnostics.csv", index=False)
    pd.DataFrame([diagnostic]).to_csv(RESULTS / "downstream_propensity.csv", index=False)

    macro_labels = d.groupby("qwen_macro", observed=True).agg(
        n=("id", "size"), journals=("journal_id", "nunique"),
    ).reset_index()
    representatives = (d.groupby(["qwen_macro", "journal_name"], observed=True).size()
                       .rename("papers").reset_index()
                       .sort_values(["qwen_macro", "papers"], ascending=[True, False])
                       .groupby("qwen_macro").head(3)
                       .groupby("qwen_macro").journal_name.apply(lambda x: "; ".join(x.astype(str))))
    macro_labels["representative_journals"] = macro_labels.qwen_macro.map(representatives)
    macro_labels.to_csv(RESULTS / "macro_labels.csv", index=False)

    run = write_run("downstream", {
        "analysis": len(frame), "support": len(d), "journals": len(journals),
        "subgroup_estimates": len(subgroup_estimates), "subgroup_tests": len(subgroup_tests),
        "routing_scores": score_rows[0],
    }, {
        "reproduced_theta": theta, "expected_theta": EXPECTED_THETA,
        "theta_delta": theta - EXPECTED_THETA, "far_near_means": means.tolist(),
        "propensity": diagnostic, "score_bytes": SCORES.stat().st_size,
        "persistent_bytes": tree_bytes(V2_WORK) + tree_bytes(V3_WORK),
        "group_free_bytes": shutil.disk_usage(GROUP_ROOT).free,
    })
    check_budget()
    log(f"downstream complete theta={theta:.6f} support={len(d):,} "
        f"estimates={len(subgroup_estimates)} tests={len(subgroup_tests)} "
        f"commit={run['git_commit']}")


if __name__ == "__main__":
    main()
