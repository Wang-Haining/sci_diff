# Prestige-stratified routing contrast: decision rule

This rule was written before inspecting any prestige-stratified routing estimate.

Frozen primary result: theta = -0.0967 (95% CI -0.1443 to -0.0492), N = 3,818,173 papers in 20,203 journals.

- **SURVIVES:** the Task 2 confidence interval excludes zero and |theta| is at least 0.048. Interpretation: Holding prior journal prestige fixed within comparison sets leaves a substantial negative routing contrast; prior prestige does not account for the primary signal in this diagnostic.
- **ATTENUATED:** the Task 2 confidence interval excludes zero and |theta| is below 0.048. Interpretation: Holding prior journal prestige fixed within comparison sets leaves a statistically distinguishable but less than half-sized routing contrast; prior prestige accounts for a substantial share of the primary signal.
- **FAILS:** the Task 2 confidence interval includes zero. Interpretation: After holding prior journal prestige fixed within comparison sets, the routing contrast is compatible with zero; the primary signal does not survive this prestige diagnostic.

No other outcome will determine the branch. Task 4 will run only after a SURVIVES or ATTENUATED result.

Prestige quartiles will follow the sibling funding analysis verbatim: within each comparison set, distinct journals are ordered by prior prestige (with journal ID as the deterministic tie-breaker), divided by `ntile(4)`, and then reattached to papers. Source: `/Users/haining/Documents/Codex/2026-09-06/thread-nsf-nih-formalize-sciscinet-open/summarize_boundaries.py`, function `stratified_counts`.
