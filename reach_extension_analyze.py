#!/usr/bin/env python3
"""Audience breadth and follow-up extensions on the saved comparison sample."""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from qss_common import REPO, SEED, QSS_TMP, STAGED_INPUT, TMP_CAP, tree_bytes
from qss_v3_common import V3_WORK, check_budget, connect, log, validate_snapshot
from qss_v3_analyze import BASE_NUMERIC, CATEGORICAL, HEAVY, fold_features

WORK = V3_WORK / "reach_extension_v1"
RESULTS = REPO / "results/reach_extension_v1"
ARTIFACTS = REPO / "artifacts/reach_extension_v1"
SCORES = V3_WORK / "routing_scores.parquet"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def budget():
    check_budget()
    spill = tree_bytes(QSS_TMP) - tree_bytes(STAGED_INPUT)
    if spill > TMP_CAP:
        raise RuntimeError(f"combined spill exceeds {TMP_CAP}: {spill}")


def fit_score(frame, outcome, directory):
    """Reuse fixed propensity/folds; persist OOF means and both arm scores."""
    y = frame[outcome].to_numpy(dtype=float)
    a = frame.treatment.to_numpy(dtype=np.int8)
    p = frame.propensity.to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError(f"invalid nonnegative outcome {outcome}")
    prediction = np.full((len(frame), 2), np.nan, dtype=float)
    diagnostics = []
    binary = outcome.startswith("any_")
    for fold in range(5):
        train = frame.fold.ne(fold).to_numpy()
        test = ~train
        counts = frame.loc[train].groupby("choice_set_id", observed=True).treatment.agg(["sum", "count"])
        prevalence = (counts["sum"] + 0.5) / (counts["count"] + 1)
        x_train = fold_features(frame, train, BASE_NUMERIC, prevalence)
        x_test = fold_features(frame, test, BASE_NUMERIC, prevalence)
        bucket = frame.loc[train, "early_stop_bucket"].to_numpy()
        for arm in (0, 1):
            fit = (a[train] == arm) & (bucket != 0)
            valid = (a[train] == arm) & (bucket == 0)
            if not fit.any() or not valid.any() or not test.any():
                raise ValueError(f"empty fit/validation/test: {outcome}, fold={fold}, arm={arm}")
            if y[train][fit].sum() == 0 or (binary and np.unique(y[train][fit]).size != 2):
                raise ValueError(f"degenerate outcome: {outcome}, fold={fold}, arm={arm}")
            cls = lgb.LGBMClassifier if binary else lgb.LGBMRegressor
            objective = "binary" if binary else "poisson"
            metric = "binary_logloss" if binary else "poisson"
            model = cls(objective=objective, n_estimators=3000, learning_rate=0.05,
                        num_leaves=255, min_child_samples=100, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8,
                        random_state=SEED + 100 * fold + arm, n_jobs=32,
                        verbosity=-1, deterministic=True, force_col_wise=True)
            model.fit(x_train.iloc[fit], y[train][fit],
                      eval_set=[(x_train.iloc[valid], y[train][valid])], eval_metric=metric,
                      callbacks=[lgb.early_stopping(50, verbose=False)], categorical_feature="auto")
            predicted = (model.predict_proba(x_test)[:, 1] if binary else model.predict(x_test))
            prediction[test, arm] = predicted
            model.booster_.save_model(str(directory / f"{outcome}_fold{fold}_arm{arm}.txt"))
            diagnostics.append(dict(outcome=outcome, fold=fold, arm=arm, objective=objective,
                                    best_iteration=int(model.best_iteration_),
                                    validation_loss=float(model.best_score_["valid_0"][metric])))
            log(f"extension outcome={outcome} fold={fold} arm={arm} trees={model.best_iteration_}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"nonfinite OOF predictions: {outcome}")
    psi = prediction.copy()
    psi[:, 0] += (1 - a) * (y - prediction[:, 0]) / (1 - p)
    psi[:, 1] += a * (y - prediction[:, 1]) / p
    pd.DataFrame(dict(id=frame.id, m0=prediction[:, 0], m1=prediction[:, 1],
                      psi0=psi[:, 0], psi1=psi[:, 1])).to_parquet(
        directory / f"{outcome}.parquet", index=False, compression="zstd")
    budget()
    return psi, diagnostics


def summarize(signal, mask, codes, multipliers):
    values = signal[mask]
    estimate = float(values.mean())
    groups = np.unique(codes[mask]).size
    if len(values) < 2 or groups < 2 or not np.isfinite(values).all():
        raise ValueError(f"invalid interval inputs: n={len(values)}, journals={groups}")
    sums = np.bincount(codes[mask], weights=values - estimate, minlength=multipliers.shape[1])
    se = math.sqrt(groups / (groups - 1) * (sums @ sums)) / len(values)
    draws = estimate + multipliers @ sums / len(values)
    return dict(estimate=estimate, se=se, ci_low=estimate - 1.96 * se,
                ci_high=estimate + 1.96 * se, bootstrap_ci_low=float(np.quantile(draws, .025)),
                bootstrap_ci_high=float(np.quantile(draws, .975)),
                n=int(mask.sum()), journals=int(groups))


def log_ratio_signal(numerator, denominator):
    """Delta-method score for log(mean numerator / mean denominator)."""
    mu_num, mu_den = float(numerator.mean()), float(denominator.mean())
    if min(mu_num, mu_den) <= 0:
        raise ValueError(f"nonpositive marginal mean: numerator={mu_num}, denominator={mu_den}")
    theta = math.log(mu_num / mu_den)
    return theta + (numerator - mu_num) / mu_num - (denominator - mu_den) / mu_den


def conditional_diagnostics(frame, names, codes, multipliers):
    """Descriptive IPW means conditional on observed classified-citation eligibility."""
    a, p = frame.treatment.to_numpy(), frame.propensity.to_numpy(dtype=float)
    rows = []
    for name in names:
        y = frame[name].to_numpy(dtype=float)
        eligible = np.isfinite(y)
        scores, means = [], []
        for arm in (0, 1):
            w = eligible * (a == arm) / (p if arm else 1 - p)
            if w.sum() <= 0:
                raise ValueError(f"no eligible positive weight: {name}, arm={arm}")
            selected = eligible & (a == arm)
            mean = float(np.dot(w[eligible], y[eligible]) / w.sum())
            contribution = np.zeros(len(frame))
            contribution[eligible] = len(frame) * w[eligible] * (y[eligible] - mean) / w.sum()
            scores.append(mean + contribution)
            means.append(mean)
            rows.append(dict(outcome=name, statistic="eligibility", arm=arm,
                             eligible_n=int(selected.sum()), arm_n=int((a == arm).sum()),
                             eligibility_fraction=float(selected.sum() / (a == arm).sum())))
        interval = summarize(scores[1] - scores[0], np.ones(len(frame), bool), codes, multipliers)
        rows.append(dict(outcome=name, statistic="conditional_IPW_contrast",
                         mean_broad=means[0], mean_narrow=means[1], **interval))
    return pd.DataFrame(rows)


def main(mode):
    if mode not in ("60", "tenyear"):
        raise ValueError(f"expected mode 60 or tenyear, got {mode}")
    validate_snapshot()
    budget()
    outcome_path = WORK / ("outcomes_60.parquet" if mode == "60" else "tenyear/outcomes_2015.parquet")
    output = WORK / f"scores_{mode}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite extension scores: {output}")
    output.mkdir(parents=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source_hashes = {str(path): digest(path) for path in (SCORES, outcome_path, V3_WORK / "analysis_dataset.parquet")}
    con = connect()
    temp = QSS_TMP / "reach_extension_v1" / f"analysis_{mode}"
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{temp}'")
    con.execute("SET max_temp_directory_size='180GB'")
    columns = list(dict.fromkeys(["id", "journal_id", "publication_year", "treatment", "fold",
                                "early_stop_bucket", "choice_set_id", "history_n"] +
                               [x for x in BASE_NUMERIC if not x.startswith("log1p_")] + HEAVY + CATEGORICAL))
    query = f"""SELECT {','.join('a.' + name for name in columns)},s.propensity
      FROM read_parquet('{V3_WORK / 'analysis_dataset.parquet'}') a
      JOIN read_parquet('{SCORES}') s USING(id)
      {'WHERE a.publication_year=2015' if mode == 'tenyear' else ''} ORDER BY a.id"""
    frame = con.execute(query).df()
    outcomes = con.execute("SELECT * FROM read_parquet(?) ORDER BY id", [str(outcome_path)]).df()
    if not frame.id.equals(outcomes.id) or frame.id.nunique() != len(frame):
        raise ValueError(f"outcome/support ID mismatch: baseline={len(frame)} outcomes={len(outcomes)}")
    if mode == "60" and (len(frame), frame.journal_id.nunique()) != (3_827_491, 20_215):
        raise ValueError(f"fixed support mismatch: n={len(frame)}, journals={frame.journal_id.nunique()}")
    if mode == "tenyear" and set(frame.publication_year) != {2015}:
        raise ValueError("ten-year analysis included a non-2015 paper")
    if not frame.propensity.between(.05 - 1e-7, .95 + 1e-7).all():
        raise ValueError("saved propensity violates support")
    if frame.groupby("journal_id").fold.nunique().max() != 1 or set(frame.fold) != set(range(5)):
        raise ValueError("journal-grouped fold membership invalid")
    for name in HEAVY:
        if (frame[name] < 0).any():
            raise ValueError(f"negative baseline feature: {name}")
        frame[f"log1p_{name}"] = np.log1p(frame[name].to_numpy(dtype=float)).astype(np.float32)
    for name in CATEGORICAL:
        frame[name] = frame[name].astype("category")
    for name in outcomes:
        if name not in frame:
            frame[name] = outcomes[name].to_numpy()
    names = [name for name in outcomes if name.startswith(
        ("near_", "far_", "intermediate_", "unclassified_", "any_far_", "distance_bin", "n_macros_", "n_leaves_"))]
    names += (["total_citations"] if mode == "60" else
              [name for name in outcomes if name.startswith("total_citations_") and not name.endswith("_no_mixed")])
    caps = {}
    if mode == "60":
        caps["total_citations"] = float(frame.total_citations.quantile(.999))
        frame["total_citations_winsorized"] = frame.total_citations.clip(upper=caps["total_citations"])
        names.append("total_citations_winsorized")
    names = sorted(set(names))
    manifest = dict(stage=f"analyze_{mode}", status="running", commit=commit, seed=SEED,
                    snapshot_date="2026-06-26", support="saved downstream deterministic",
                    propensity="reused unchanged", outcome_training="fixed saved support, journal-grouped cross-fitting",
                    n=len(frame), journals=int(frame.journal_id.nunique()), outcomes=names,
                    inputs=source_hashes, winsorization_caps=caps,
                    multiplier="500 shared Gaussian journal multipliers", models_directory=str(output),
                    started_utc=datetime.now(timezone.utc).isoformat())
    manifest_path = ARTIFACTS / f"run_analyze_{mode}.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite run manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    frame[["id", "journal_id", "treatment", "propensity", "fold", "early_stop_bucket"]].to_parquet(
        output / "sample.parquet", index=False, compression="zstd")
    codes, journals = pd.factorize(frame.journal_id, sort=True)
    multipliers = np.random.default_rng(SEED).standard_normal((500, len(journals)))
    np.savez_compressed(output / "journal_multipliers.npz", journals=journals.to_numpy(dtype=str), multipliers=multipliers)
    all_mask = np.ones(len(frame), bool)
    rows, diagnostics, scores = [], [], {}
    for name in names:
        equivalent = next((old for old in scores if old.startswith("any_") == name.startswith("any_")
                           and np.array_equal(frame[name].to_numpy(), frame[old].to_numpy())), None)
        if equivalent is None:
            score, diag = fit_score(frame, name, output)
        else:
            score, diag = scores[equivalent], []
            manifest.setdefault("identical_outcome_score_reuse", {})[name] = equivalent
            log(f"identical observed endpoint: {name} uses {equivalent} model/scores")
        scores[name] = score
        diagnostics.extend(diag)
        means = score.mean(axis=0)
        rows.append(dict(outcome=name, scale="absolute", population="all", mean_broad=means[0], mean_narrow=means[1],
                         **summarize(score[:, 1] - score[:, 0], all_mask, codes, multipliers)))
        # AIPW can yield a nonpositive mean for a rare outcome: keep its absolute result, not a fake ratio.
        if min(means) > 0:
            signal = log_ratio_signal(score[:, 1], score[:, 0])
            rows.append(dict(outcome=name, scale="log_mean_ratio", population="all", mean_broad=means[0], mean_narrow=means[1],
                             **summarize(signal, all_mask, codes, multipliers)))
        else:
            log(f"relative scale undefined: outcome={name}, marginal means={means.tolist()}")
        pd.DataFrame(rows).to_csv(RESULTS / f"estimates_{mode}.csv", index=False)
        pd.DataFrame(diagnostics).to_csv(RESULTS / f"outcome_diagnostics_{mode}.csv", index=False)

    routing = {}
    for near in [name for name in names if name.startswith("near_")]:
        suffix = near.removeprefix("near_")
        far = "far_" + suffix
        if far not in scores:
            raise ValueError(f"missing paired far score for {near}")
        s0 = log_ratio_signal(scores[far][:, 0], scores[near][:, 0])
        s1 = log_ratio_signal(scores[far][:, 1], scores[near][:, 1])
        routing[suffix] = s1 - s0
        rows.append(dict(outcome="far_to_near_" + suffix, scale="log_ratio_of_mean_ratios", population="all",
                         mean_broad=float(scores[far][:, 0].mean()/scores[near][:, 0].mean()),
                         mean_narrow=float(scores[far][:, 1].mean()/scores[near][:, 1].mean()),
                         **summarize(routing[suffix], all_mask, codes, multipliers)))
        # Closed composition from the same component arm means, not a separately fitted total.
        components = ["near_" + suffix, "intermediate_" + suffix, "far_" + suffix]
        denominator = sum(scores[x] for x in components)
        valid_composition = all(np.isfinite(scores[x].mean(axis=0)).all() and
                                (scores[x].mean(axis=0) >= 0).all() for x in components)
        if not valid_composition:
            rows.append(dict(outcome="composition_"+suffix, scale="classified_share_difference",
                             population="all", status="undefined_negative_or_nonfinite_component_mean"))
            log(f"classified composition undefined for {suffix}; absolute estimates retained")
        for name in components:
            if not valid_composition:
                continue
            shares, share_scores = [], []
            for arm in (0, 1):
                mu, den = scores[name][:, arm].mean(), denominator[:, arm].mean()
                if den <= 0:
                    raise ValueError(f"nonpositive classified mean: {suffix}, arm={arm}, mean={den}")
                shares.append(mu/den)
                share_scores.append(mu/den + (scores[name][:, arm] - (mu/den)*denominator[:, arm])/den)
            rows.append(dict(outcome=name, scale="classified_share_difference", population="all",
                             mean_broad=shares[0], mean_narrow=shares[1],
                             **summarize(share_scores[1]-share_scores[0], all_mask, codes, multipliers)))
        if mode == "60":
            jy = frame[["journal_id", "publication_year", "history_n"]].drop_duplicates()
            cuts = np.quantile(jy.history_n, [1/3, 2/3])
            levels = np.searchsorted(cuts, frame.history_n.to_numpy(), side="left") + 1
            for level in (1, 2, 3):
                mask = levels == level
                signal = np.zeros(len(frame))
                signal[mask] = (log_ratio_signal(scores[far][mask, 1], scores[near][mask, 1]) -
                                log_ratio_signal(scores[far][mask, 0], scores[near][mask, 0]))
                rows.append(dict(outcome="far_to_near_"+suffix, scale="log_ratio_of_mean_ratios",
                                 population=f"history_volume_tertile_{level}", volume_cut_low=cuts[0], volume_cut_high=cuts[1],
                                 **summarize(signal, mask, codes, multipliers)))
    if mode == "60":
        for suffix in ("all32", "named31"):
            bins = [f"distance_bin{i}_{suffix}" for i in range(5)]
            denominator = sum(scores[x] for x in bins)
            if any(not np.isfinite(scores[x].mean(axis=0)).all() or
                   (scores[x].mean(axis=0) < 0).any() for x in bins):
                rows.append(dict(outcome="distance_composition_"+suffix, scale="distance_share_difference",
                                 population="all", status="undefined_negative_or_nonfinite_component_mean"))
                log(f"distance composition undefined for {suffix}; absolute estimates retained")
                continue
            if np.any(denominator.mean(axis=0) <= 0):
                raise ValueError(f"nonpositive distance-bin denominator: {suffix}")
            for name in bins:
                shares, sig = [], []
                for arm in (0, 1):
                    den = denominator[:, arm].mean()
                    share = scores[name][:, arm].mean()/den
                    shares.append(share)
                    sig.append(share+(scores[name][:, arm]-share*denominator[:, arm])/den)
                rows.append(dict(outcome=name, scale="distance_share_difference", population="all",
                                 mean_broad=shares[0], mean_narrow=shares[1],
                                 **summarize(sig[1]-sig[0], all_mask, codes, multipliers)))
        conditional = [x for x in outcomes if x.startswith(("macro_entropy_", "rarefied_macros_"))]
        conditional_diagnostics(frame, conditional, codes, multipliers).to_csv(
            RESULTS / "conditional_breadth.csv", index=False)
    else:
        for suffix in ("", "_no_mixed"):
            signal = routing["120"+suffix]-routing["60"+suffix]
            rows.append(dict(outcome="theta_120_minus_60"+suffix, scale="paired_log_ratio_contrast", population="all",
                             **summarize(signal, all_mask, codes, multipliers)))
    pd.DataFrame(rows).to_csv(RESULTS / f"estimates_{mode}.csv", index=False)
    if any(digest(Path(path)) != value for path, value in source_hashes.items()):
        raise RuntimeError("frozen input changed during analysis")
    manifest.update(status="complete", estimate_rows=len(rows), model_fits=len(diagnostics),
                    finished_utc=datetime.now(timezone.utc).isoformat(), output_bytes=tree_bytes(output))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    budget()
    log(f"extension {mode} COMPLETE papers={len(frame):,} journals={len(journals):,} estimates={len(rows)} models={len(diagnostics)}")
    print(pd.DataFrame(rows).query("scale == 'log_ratio_of_mean_ratios'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
