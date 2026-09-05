# QSS v3 dirty pilot

**Snapshot:** OpenAlex Parquet 2026-06-26

**Seed:** 20260902

## Question and estimand

Among papers with comparable content, authorship, and feasible journal choice
sets, how does publication in a specialized rather than broad journal change the
disciplinary routing of citations over the next 60 months?

The exposure and outcome taxonomy stay as in qss_v2:

- journal specialization is based on prior SPECTER2 title similarity;
- specialized and broad are the top and bottom quartiles within
  `semantic_cluster x publication_year` choice sets;
- Qwen3 classifies citations as near, intermediate, far, or unclassified;
- shared-author self-citations and same-journal citations are excluded; and
- focal OOD papers are excluded.

The primary estimand is

`theta = log(mu_far,1 / mu_near,1) - log(mu_far,0 / mu_near,0)`

in the empirical common-support population. Secondary outcomes are total, near,
intermediate, far, unclassified, any-far, and the 99.9%-winsorized routing ratio.

## Follow-up

Use one rolling window for every focal paper:

`[publication_date, publication_date + 60 months)`.

When OpenAlex knows only the year and records January 1, this reduces exactly to
the old `t:t+4` calendar-year window. Report the January 1 proportion by treatment
arm; do not create a separate date-quality branch. Rebuild qss_v3 citation edges
through 2025 so late-2020 focal papers have complete follow-up. Publication month
is not an adjustment variable. Publication year remains in the model.

## Baseline adjustment

Keep the qss_v2 baseline covariates except publication month. Add:

- full `choice_set_id` as a categorical feature;
- out-of-fold choice-set treatment prevalence, calculated within each journal-
  grouped fold from its training rows only as
  `(n_specialized + 0.5) / (n_total + 1)`;
- first/last-author prior-venue specialization; and
- first/last-author prior-paper embedding breadth.

Heavy-tailed counts enter the models as `log1p`: reference and classified-reference
counts, author and institution prior works and citations, journal history N, and
prior journal prestige. Save both raw- and model-scale balance diagnostics.

Author history uses `t-5:t-1`. Author position comes from
`authorships.author_position`, never list order. Prior-venue specialization is the
mean specialization of journals used by the author before the focal paper.
Prior-paper breadth is `1 - unbiased mean pairwise SPECTER2 cosine` for authors
with at least two prior papers. The primary paper-level value is the mean of the
first and last author values, counting a single author once. Missing values stay
missing and receive missingness and contributor-count features. Add prior author
far/near citation history if compute and the existing storage limits allow it.

## Reference routing

Classify each focal paper's references with the same Qwen3 taxonomy. If existing
Qwen3 embeddings cover less than 90% of reference edges in either arm, embed the
missing referenced works first.

Report the audience-alignment diagnostic

`theta_ref = log(mu_ref_far,1 / mu_ref_near,1) - log(mu_ref_far,0 / mu_ref_near,0)`

and `theta_ref / theta`. Reference routing is not in the primary adjustment set.
A sensitivity adds paper-level `log((ref_far + 0.5)/(ref_near + 0.5))`, classified
reference count, and unclassified-reference share. If `theta_ref` is clearly
negative, preprint matching can be considered later; it is not part of this run.

## Propensity model

Use the same five journal-grouped cross-fitting folds. Compare two LightGBM
propensity models with `num_leaves` equal to 63 or 255. Both use learning rate
0.05, minimum child samples 100, row subsampling 0.8, column subsampling 0.8, and
at most 3,000 trees.

Within each fold, deterministically reserve 10% of its training rows for early
stopping with patience 50, then predict only the held-out cross-fitting fold.
Choose the candidate with the smaller maximum weighted absolute SMD; if the two
maxima differ by at most 0.005, choose 63 leaves. Save for both candidates:
selected tree counts, support, arm sizes, ESS, all weighted SMDs, and weight
percentiles. Common support remains propensity `[0.05, 0.95]` with at least 50%
retention.

The categorical features are publication year, lead country, semantic cluster,
and choice-set ID. Journal ID, publication month, focal OA, current or future
journal features, and citation outcomes are excluded.

## Outcome models and inference

Use one LightGBM profile for every arm-specific outcome nuisance model:
`num_leaves=255`, learning rate 0.05, minimum child samples 100, row subsampling
0.8, column subsampling 0.8, and at most 3,000 trees. Use the same deterministic
10% training-row validation split and patience-50 early stopping. Count outcomes
use `objective="poisson"` and Poisson deviance; binary outcomes use
`objective="binary"` and log loss. Record selected tree counts and validation
losses.

Estimate all marginal means with five-fold AIPW. Construct the CI for `theta`
from the joint paper-level influence function of the four near/far arm means.
Journal-cluster analytic intervals and 500 multiplier-bootstrap draws use the
same journal weights across outcomes.

Report every residual imbalance and show whether the outcome-adjusted estimate
changes across the prespecified sensitivity analyses. SMD is a diagnostic for
measured covariates, not an identification condition. Causal interpretation
rests on conditional exchangeability, positivity, consistency, and the stated
measurement assumptions. Report support, ESS, weight tails, focal OOD difference,
citing classification coverage, January 1 dates, `theta_ref`, and all null or
contrary outcomes.

## Checks and outputs

Fail on a wrong snapshot or model revision, duplicate embeddings or citation
edges, invalid embedding shape or norm, post-time-zero covariates, incomplete
60-month extraction, or failure of
`total = near + intermediate + far + unclassified`.

Write qss_v3 artifacts without overwriting qss_v2:

- rebuilt citation edges and the analysis dataset;
- author and reference routing features;
- diagnostics for both propensity candidates;
- outcome early-stopping diagnostics;
- estimates, balance, report, and run manifest; and
- the 99.9%-winsorized result and top-0.1% citation contribution.
