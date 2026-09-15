#!/usr/bin/env python3
"""No-refit signal screening; not a replacement for the frozen AIPW analysis."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from reach_extension_analyze import summarize, log_ratio_signal, digest
from reach_extension_tenyear import budget, connect, git_head, prior_manifest, REPO, SEED, V3_WORK


def main():
    prior_manifest("outcomes")
    root = V3_WORK / "reach_extension_v1"
    out = REPO / "results/reach_extension_pilot_v1"
    out.mkdir(exist_ok=False)
    con = connect("32GB", 4)
    runs, inputs = {}, {}
    for mode, relative, expected in (("60", "outcomes_60.parquet", (3827491, 20215)),
                                    ("tenyear", "tenyear/outcomes_2015.parquet", (596758, 11635))):
        path = root / relative
        inputs[str(path)] = digest(path)
        frame = con.execute("SELECT * FROM read_parquet(?) ORDER BY id", [str(path)]).df()
        saved_path = V3_WORK / "routing_scores.parquet"
        inputs[str(saved_path)] = digest(saved_path)
        saved = con.execute("SELECT * FROM read_parquet(?) " +
            ("WHERE publication_year=2015 " if mode == "tenyear" else "") + "ORDER BY id", [str(saved_path)]).df()
        for col in ("id", "journal_id", "treatment", "propensity"):
            assert np.array_equal(frame[col], saved[col]), f"saved support mismatch: {mode}/{col}"
        assert (len(frame), frame.journal_id.nunique()) == expected, (mode, frame.shape)
        assert frame.id.nunique() == len(frame) and frame.propensity.between(.05-1e-7, .95+1e-7).all()
        a, p = frame.treatment.to_numpy(), frame.propensity.to_numpy(dtype=float)
        w = np.column_stack(((1-a)/(1-p), a/p))
        codes, journals = pd.factorize(frame.journal_id, sort=True)
        multipliers = np.random.default_rng(SEED).standard_normal((500, len(journals)))
        mask, rows, scores = np.ones(len(frame), bool), [], {}

        def record(name, scale, signal, means):
            rows.append(dict(outcome=name, scale=scale, mean_broad=float(means[0]),
                mean_narrow=float(means[1]), **summarize(signal, mask, codes, multipliers)))

        names = [x for x in frame if x.startswith(("total_citations", "near_", "far_", "intermediate_",
                 "unclassified_", "any_far_", "distance_bin", "n_macros_", "n_leaves_"))]
        for name in names:
            y = frame[name].to_numpy(dtype=float)
            assert np.isfinite(y).all() and (y >= 0).all(), name
            means = (w * y[:, None]).sum(axis=0) / w.sum(axis=0)
            score = means + w * (y[:, None] - means) / w.mean(axis=0)
            scores[name] = score
            record(name, "IPW_absolute", score[:, 1]-score[:, 0], means)
            if min(means) > 0:
                record(name, "IPW_log_mean_ratio", log_ratio_signal(score[:, 1], score[:, 0]), means)
        routing = {}
        for near in [x for x in names if x.startswith("near_")]:
            suffix = near.removeprefix("near_")
            components = [scores[k+suffix] for k in ("near_", "intermediate_", "far_", "unclassified_")]
            total = scores["total_citations" if mode == "60" else "total_citations_"+suffix]
            assert np.allclose(sum(components), total, rtol=1e-10, atol=1e-8), suffix
            far = scores["far_"+suffix]
            routing[suffix] = log_ratio_signal(far[:, 1], scores[near][:, 1])-log_ratio_signal(far[:, 0], scores[near][:, 0])
            record("far_to_near_"+suffix, "IPW_log_ratio_of_ratios", routing[suffix], far.mean(axis=0)/scores[near].mean(axis=0))
        if mode == "tenyear":
            for suffix in ("", "_no_mixed"):
                assert np.array_equal(frame["total_citations_60"+suffix]+frame["total_citations_late60_120"+suffix], frame["total_citations_120"+suffix])
                rows.append(dict(outcome="theta_120_minus_60"+suffix, scale="IPW_paired_change",
                    theta_60=routing["60"+suffix].mean(), theta_120=routing["120"+suffix].mean(),
                    **summarize(routing["120"+suffix]-routing["60"+suffix], mask, codes, multipliers)))
        else:
            for suffix in ("all32", "named31"):
                bins = ["distance_bin"+str(i)+"_"+suffix for i in range(5)]
                denominator = sum(scores[x] for x in bins)
                for name in bins:
                    shares = scores[name].mean(axis=0)/denominator.mean(axis=0)
                    psi = shares+(scores[name]-shares*denominator)/denominator.mean(axis=0)
                    record(name, "IPW_distance_share_difference", psi[:, 1]-psi[:, 0], shares)
            old = [saved[f"psi_{y}_{arm}"].to_numpy() for y in ("near", "far") for arm in (0, 1)]
            signal = log_ratio_signal(old[3], old[1])-log_ratio_signal(old[2], old[0])
            assert abs(signal.mean()-(-0.08245270348646583)) < 1e-7, signal.mean()
            record("saved_downstream_AIPW", "reference_only_log_ratio_of_ratios", signal, [old[2].mean()/old[0].mean(), old[3].mean()/old[1].mean()])
        result = pd.DataFrame(rows)
        result.to_csv(out / f"estimates_{mode}.csv", index=False)
        runs[mode] = dict(n=len(frame), journals=len(journals), arm_n=[int((a==i).sum()) for i in (0,1)],
                         arm_ess=(w.sum(axis=0)**2/(w*w).sum(axis=0)).tolist(), result_rows=len(rows))
        print(mode, runs[mode], flush=True)
        print(result[result.scale.str.contains("ratios|paired_change")].to_string(index=False), flush=True)
        budget()
    assert all(digest(Path(k)) == v for k,v in inputs.items()), "input changed"
    run = dict(status="complete", estimator="Hajek IPW screening; zero new model fits", seed=SEED,
        commit=git_head(), snapshot="2026-06-26", uncertainty="journal IF and 500 shared Gaussian multipliers; saved weights held fixed",
        manuscript_use=False, automatic_full_restart=False, inputs=inputs, populations=runs, storage=budget())
    (out / "run_pilot.json").write_text(json.dumps(run, indent=2)+"\n")
    print("PILOT COMPLETE; original analyses unchanged; user review before further fitting", flush=True)


if __name__ == "__main__":
    main()
