#!/usr/bin/env python3
from qss_analyze import build_scope, build_semantics
from qss_common import RESULTS, check_budget, connect, log, validate_snapshot, write_run


def main():
    validate_snapshot()
    check_budget()
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = connect()
    scope_n, reliability, abstract_reliability = build_scope(con)
    semantics_n, pca_variance, cluster_inertia = build_semantics(con)
    write_run("exposure", "complete", {
        "journal_year_scope": scope_n,
        "focal_semantics": semantics_n,
    }, {
        "reliability": reliability,
        "abstract_reliability": abstract_reliability,
        "pca_variance": pca_variance,
        "cluster_inertia": cluster_inertia,
        "measurement_gate": reliability["spearman_brown"] >= 0.70,
        "section_b_touched": False,
    })
    check_budget()
    log(f"exposure complete scope={scope_n:,} focal_semantics={semantics_n:,} "
        f"reliability={reliability['spearman_brown']:.3f}; Section B not touched")


if __name__ == "__main__":
    main()
