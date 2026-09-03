#!/usr/bin/env python3
import json

import pandas as pd

from qss_common import ARTIFACTS, RESULTS, log, write_run


def estimate_line(row):
    return (f"{row.outcome} ({row.scale}): {row.estimate:.4f} "
            f"(95% CI {row.ci_low:.4f} to {row.ci_high:.4f}; "
            f"multiplier bootstrap {row.bootstrap_ci_low:.4f} to {row.bootstrap_ci_high:.4f})")


def main():
    estimates = pd.read_csv(RESULTS / "causal_estimates.csv")
    gates = pd.read_csv(RESULTS / "gates.csv")
    run = json.loads((ARTIFACTS / "run_analyze.json").read_text())
    if estimates.empty or gates.empty:
        raise ValueError(f"expected nonempty estimates and gates, got {len(estimates)}, {len(gates)}")
    causal = run["extra"]["wording"] == "causal"
    audience = bool(run["extra"]["audience_segmentation_claim"])
    decision = ("AUDIENCE-SEGMENTATION CLAIM SUPPORTED" if audience else
                "CAUSAL ESTIMATE; PRIMARY CLAIM NOT SUPPORTED" if causal else "ASSOCIATION ONLY")
    primary = estimates[(estimates.exposure.eq("semantic_title")) &
                        (estimates.scale.eq("absolute"))]
    replication = estimates[(estimates.exposure.eq("reference_field")) &
                            (estimates.scale.eq("absolute"))]
    text = [decision, "", "# QSS qss_v1 analysis report", "",
            "## Frozen question", "",
            "Among papers with comparable content, authorship, institutions, and feasible "
            "journal choice sets, how does publication in a specialized rather than broad "
            "journal change five-year disciplinary reach?", "", "## Cohort and diagnostics", "",
            f"- Analysis rows: {run['counts']['analysis']:,}",
            f"- Primary common-support rows: {run['counts']['support']:,}",
            f"- Journals: {run['counts']['journals']:,}",
            f"- Split-half reliability: {run['extra']['reliability']['spearman_brown']:.3f}",
            f"- Maximum weighted SMD: {run['extra']['max_weighted_smd']}", "",
            "## Primary estimates", ""]
    text += [f"- {estimate_line(row)}" for _, row in primary.iterrows()]
    text += ["", "## Independent reference-field exposure replication", ""]
    text += [f"- {estimate_line(row)}" for _, row in replication.iterrows()]
    text += ["", "## Prespecified gates", ""]
    text += [f"- {row.gate}: {'PASS' if row.passed else 'FAIL'}" for _, row in gates.iterrows()]
    text += ["", "All null, contrary, and nonevaluable estimates remain in `causal_estimates.csv`. "
             "The decision controls wording only and did not select an estimator.", ""]
    path = RESULTS / "study_report.md"
    path.write_text("\n".join(text))
    write_run("summary", "complete", {"estimate_rows": len(estimates), "gate_rows": len(gates)},
              {"decision": decision})
    log(f"summary complete decision={decision} estimate_rows={len(estimates)}")


if __name__ == "__main__":
    main()
