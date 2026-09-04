# QSS v3: outcome-informed design repair

**Design version:** qss_v3

**Status:** FROZEN 2026-09-04, before any v3 model or outcome run

**Snapshot:** OpenAlex Parquet 2026-06-26

**Seed:** 20260902

## What is and is not confirmatory

The qss_v2 dirty outcome has already been observed. Its primary estimate was
`theta = -0.107877` (95% CI `-0.151184` to `-0.064570`), with weighted maximum
absolute SMD `0.144141`. QSS v3 is therefore an **outcome-informed design repair**,
not a pristine preregistration or an independent confirmation. The dirty result
will remain reported as the pre-repair estimate.

This document freezes all v3 decisions before changing propensity, date-window,
author-history, reference-routing, or outcome-model code. QSS v2 artifacts and
code are not overwritten. Except for the prespecified outcome-nuisance prediction
loss below, no future citation outcome, routing estimate, or significance result
may be used to select among the rules below.

## Scientific question and unchanged estimand

The question remains: among papers with comparable content, authorship, and
feasible journal choice sets, how does publication in a specialized rather than
broad journal change the disciplinary routing of later citations?

The exposure, extreme-quartile contrast, venue-free Qwen3 taxonomy, external
citation exclusions, and primary routing estimand remain unchanged from qss_v2:

`theta = log(mu_far,1 / mu_near,1) - log(mu_far,0 / mu_near,0)`.

The treatment arm is specialized (`1`) and the comparator is broad (`0`). The
primary estimand is for the empirical common-support population. Secondary
outcomes remain total, near, intermediate, far, unclassified, any-far, and the
99.9%-winsorized routing ratio. The qss_v2 Qwen3 model revision, taxonomy,
near/intermediate/far/OOD rules, focal eligibility, self-citation exclusion, and
same-journal exclusion are unchanged.

## Analysis order and outcome firewall

The stages run in this order:

1. date-quality audit and follow-up-window decision;
2. first/last-author baseline feature construction;
3. reference-embedding coverage audit and reference-routing construction;
4. outcome-blind propensity candidate comparison;
5. outcome-nuisance comparison using only prespecified out-of-fold prediction loss;
6. freeze the selected candidate IDs and population in a run manifest;
7. estimate the primary and secondary effects once.

Stages 1-4 may not calculate or expose a v3 citation outcome or `theta`. Stage 5
may use outcome values only to calculate the losses defined below; it may not
calculate an AIPW estimate, arm marginal mean, treatment contrast, or routing
ratio. All candidate diagnostics are retained, including failed candidates.

## Choice-set representation

The frozen choice set is `semantic_cluster x publication_year`, with the same
1,000 SPECTER2 clusters and years 2015-2020 as qss_v2. It enters the propensity
model in two forms:

- the full choice-set ID as a categorical variable; and
- its out-of-fold treated-paper prevalence as a numeric variable.

For fold `k`, prevalence is calculated only from papers whose journals are in the
other four folds, as `(n_specialized + 0.5) / (n_total + 1)`. A held-out choice set
absent from its training folds is a data error and stops the run. The prevalence
supplements rather than replaces choice-set identity. Folds remain grouped by
journal, and the same fold assignment is used for all nuisance models.

## Publication dates and the five-year clock

Publication month is removed from the primary adjustment set in every branch. It
may reflect journal processing and publication timing after venue choice and is
therefore not treated as a baseline confounder. Publication year remains adjusted.

Before any v3 outcome construction, scan the source snapshot through 2025 and
report by treatment arm:

- missing or invalid `publication_date`;
- dates whose year disagrees with `publication_year`;
- January 1 dates, conservatively treated as year-only/imputed dates;
- focal papers with day-resolved dates; and
- incoming citation edges whose citing papers have day-resolved dates.

Use a rolling 60-month outcome window only if, in each arm, at least 90% of focal
papers and at least 90% of their incoming citation edges have valid non-January-1
dates, and each corresponding absolute arm difference is at most 2 percentage
points. The rolling window is `[focal publication_date, focal publication_date +
60 months)`. Assert that the citing date precedes the snapshot and that the full
window is administratively observable.

If any date gate fails, retain the qss_v2 calendar-year window `t:t+4` for every
paper. The branch is chosen from date counts alone, is written to the manifest,
and cannot be revisited after outcomes are summarized. Failure of the rolling-date
gate is not itself a causal-wording failure; undisclosed or mixed clocks are.
If the rolling branch is selected, rebuild v3 citation edges through 2025 because
the frozen qss_v2 edge file ends at each paper's `t+4`; never mix the two edge sets.

## Author audience-history covariates

All author variables use only information available before the focal paper. The
history window is the five publication years `t-5:t-1`. Author positions are read
from the raw OpenAlex `authorships.author_position` field; list order is never used
to infer first or last author. A single author is counted once.

Two inexpensive variables are mandatory:

1. **Prior-venue specialization.** For each author, average the focal-year
   `journal_year_scope` specialization score over that author's prior journal
   papers. This evaluates the author's recent venue portfolio using journal scope
   known before the focal paper and does not use the focal venue choice.
2. **Prior-paper embedding breadth.** For each author with at least two prior
   papers, calculate `1 - unbiased mean pairwise cosine` across venue-free
   SPECTER2 title embeddings of those papers.

To keep the five-year window identical across focal cohorts, 2010-2011 prior-paper
titles not present in qss_v2 are embedded with the same pinned SPECTER2 revision;
the history window is never silently shortened for the 2015-2016 cohorts.

The primary aggregation is the mean of the first- and last-author values among
those observed. A single author contributes once. Each construct also carries
the number of contributing lead authors and a missingness indicator; missing
values remain missing for LightGBM and are never filled with zero. The all-author
mean is a prespecified sensitivity analysis. All author-history features, raw and
aggregated, must assert a latest source year no later than `t-1`.

The more expensive variable—authors' prior far/near citation routing—is not needed
to start the primary v3 repair. It is added as an extended sensitivity only if the
required pre-focal citing works are already classified or can be embedded in one
bounded job while persistent storage remains at most 200 GB, spill at most 400 GB,
and group free space at least 1.5 TB. Its citations must occur before the focal
paper's time zero. Failure of this feasibility condition is reported, not worked
around by changing paths or budgets.

## Reference routing: diagnostic and sensitivity only

Each focal paper's references are classified against its Qwen3 leaf and macrocluster
using the frozen qss_v2 taxonomy. Before calculation, report arm-specific coverage
of reference edges by existing Qwen3 embeddings. If either arm is below 90%, embed
the missing referenced works before continuing, subject to the unchanged storage
gates. Citation and reference edges remain unique pairs.

The audience-alignment diagnostic applies the same adjusted log ratio-of-means to
pre-publication reference counts:

`theta_ref = log(mu_ref_far,1 / mu_ref_near,1) - log(mu_ref_far,0 / mu_ref_near,0)`.

This is not called a strict negative-control outcome because peer review and
revision may alter the published reference list. Its interpretation is asymmetric:

- `theta_ref` is considered near zero only when its 95% CI contains zero and
  `abs(theta_ref) <= 0.05`; this supports, but does not prove, adequate control of
  pre-existing audience alignment;
- a negative `theta_ref` cannot distinguish content/audience confounding from
  venue-induced narrowing of references; and
- if the upper 95% confidence limit is below zero, preprint matching becomes a
  conditional follow-up, not part of the present v3 run.

Reference routing is excluded from the primary adjustment set because the final
published reference list may partly mediate a venue effect. A prespecified
sensitivity adds paper-level `log((ref_far + 0.5)/(ref_near + 0.5))`, classified
reference count, and unclassified-reference share. The unadjusted and adjusted
versions are both reported; neither replaces the primary after results are seen.

## Propensity repair and selection

All propensity candidates use the same journal-grouped five folds, covariate set,
rows, treatment, and random seed. Journal ID, publication month, focal OA, current
or post-publication journal features, and citation outcomes are prohibited.

The categorical inputs are publication year, lead country, semantic cluster, and
full choice-set ID. The numeric inputs include both frozen embedding spaces,
baseline paper/reference/team/institution features, prior journal features, the
two mandatory author audience-history constructs, and out-of-fold choice-set
prevalence. The following heavy-tailed variables are replaced by `log1p` versions:
reference counts, classified reference counts, author and institution prior works
and citations, journal history N, and prior journal prestige. Raw-scale SMDs are
still reported; balance gates use the scales actually supplied to the model.

The fixed LightGBM candidate grid is:

- `num_leaves` in `{63, 255}`;
- `n_estimators` in `{1000, 3000}`;
- all four combinations, with learning rate `0.05`, minimum child samples `100`,
  row subsampling `0.8`, and column subsampling `0.8`.

For every candidate, retain out-of-fold propensity, support retention, arm sizes,
arm-specific effective sample size, weighted SMDs, and weight percentiles through
the maximum. Empirical support remains `[0.05, 0.95]`. A candidate is eligible only
if overall retention is at least 50% and the ESS is at least 25% of retained papers
in each arm.

Selection is lexicographic and outcome-blind: minimize maximum weighted absolute
SMD; if candidates are within 0.005, maximize the smaller arm's ESS fraction; if
within 0.01, minimize pooled p99 inverse-probability weight; if still tied, choose
fewer leaves and then fewer trees. No citation outcome or `theta` enters selection.
If the selected candidate has any weighted absolute SMD at or above 0.10, retain
the estimates but do not use causal wording; do not add another candidate.

## Outcome nuisance selection

After the propensity choice is frozen, compare exactly two arm-specific LightGBM
outcome profiles:

- compact: `num_leaves=63`, `n_estimators=1000`, `learning_rate=0.05`;
- flexible: `num_leaves=255`, `n_estimators=3000`, `learning_rate=0.03`.

All other tree settings and journal-grouped folds match the propensity stage.
Selection occurs separately for each outcome and treatment arm using pooled
out-of-fold Poisson deviance for counts and log loss for binary outcomes. If loss
differs by less than 1%, use the compact profile. Candidate predictions and losses
are retained. This stage may inspect prediction loss only; it may not calculate a
treatment contrast or routing ratio. The selected profile IDs are frozen before
the single AIPW run.

## Estimation, uncertainty, and diagnostics

The estimator remains journal-grouped five-fold AIPW. The four arm means for near
and far citations produce `theta` through their joint paper-level influence
functions. Journal-cluster analytic intervals and 500 multiplier-bootstrap draws
reuse the same journal weights across every outcome. Report absolute marginal
means, relative changes, top-0.1% citation contribution, and the prespecified
99.9%-winsorized routing ratio.

Required data assertions remain: exact snapshot and model revisions; unique
embeddings and citation/reference edges; 768 dimensions and unit norms; no
post-time-zero covariates; exact follow-up branch; and exact decomposition
`total = near + intermediate + far + unclassified`.

Inferential gates are:

- Section A measurement reliability at least 0.70;
- common-support retention at least 50%;
- selected propensity ESS at least 25% of retained papers in each arm;
- every weighted absolute SMD below 0.10 on its modeling scale;
- focal OOD arm difference at most 2 percentage points;
- citing-flow classification coverage at least 80% in each arm; and
- `theta_ref` near zero under the rule above.

Data, leakage, time-window, or decomposition failures stop the run. If all gates
pass, the manuscript may use carefully qualified **estimated effect** language
while disclosing that v3 repaired an outcome-informed dirty analysis. If balance
or `theta_ref` fails, the same estimates are retained and described as
**associations**. Sibling comparisons remain a stated identification limitation,
not an unreported requirement or a post hoc rescue.

## Frozen outputs

V3 writes to new `qss_v3` paths and never overwrites qss_v2. At minimum it retains:

- `date_quality.csv` and the selected follow-up branch;
- `author_routing.parquet` and `reference_routing.parquet`;
- all propensity candidate balance, ESS, and weight diagnostics;
- all outcome-nuisance candidate losses;
- the selected-candidate manifest with Git commit and seed;
- `analysis_dataset.parquet`, estimates, gates, and report; and
- the unchanged dirty estimate beside the v3 estimate in the final report.

No OVB analysis, multilingual model comparison, sibling design, event study,
preprint matching, or manuscript drafting is part of this run unless the frozen
conditional trigger above is met. Storage paths and budgets remain those already
approved for qss_v2.
