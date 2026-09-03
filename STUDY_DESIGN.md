# Estimating the specialization effect of journals on the disciplinary reach of science

**Design version:** qss_v2
**Status:** DIRTY QWEN3 SECTION B FROZEN 2026-09-03
**Snapshot:** OpenAlex Parquet 2026-06-26
**Seed:** 20260902

## Two-stage freeze boundary

**Section A — exposure and structural data: FROZEN 2026-09-03.** This section
includes the snapshot and storage gates; focal and history windows; SPECTER2
title embeddings at the pinned revisions; journal-year specialization and
split-half reliability; 1,000 venue-free choice-set clusters; citation-edge
extraction, deduplication and timing; and baseline covariates other than PCs from
the independent outcome encoder. These artifacts may be produced immediately.

**Section B — dirty outcome and inference: FROZEN 2026-09-03.** This exploratory
run uses `Qwen/Qwen3-Embedding-0.6B` at commit
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, without a prompt. The first 768
Matryoshka dimensions are unit normalized. The taxonomy uses the 1,000,000
lowest BLAKE2b hashes of `20260902|OpenAlex work ID` among English 2012-2014
history titles for training and the next 250,000 for held-out calibration.

The dirty run deliberately does not wait for human validation and does not compare
encoders. It is labeled exploratory throughout. Human validation, omitted-variable
bounds, multilingual replication, sibling comparisons and event studies are deferred;
none may be added after seeing this run and described as prespecified.

## Story and contribution

The 2020 pilot found no stable loss of total or within-subfield citations after
topic-by-prestige standardization, but it found fewer cross-field citations for
papers in specialized journals. The paper therefore tests an attention-routing
claim, not the older claim that specialty journals merely have lower impact.

The contribution is a paper-level causal estimand for a property of the venue:
among manuscripts with comparable content, authors, institutions, and feasible
journal choice sets, how would five-year disciplinary reach differ if publication
occurred in a highly specialized rather than broad journal?

## Research questions

**Main question.** Among scientific papers with comparable content, authorship,
and feasible journal choice sets, what is the effect of publishing in a more
specialized rather than broad journal on the five-year disciplinary reach of
subsequent citations?

1. Does specialization change total five-year uptake or the probability of any
   citation?
2. Does specialization reduce cross-field citations more than within-subfield
   citations? This is the sole primary hypothesis.
3. Is the cross-field effect more negative for papers with more interdisciplinary
   reference portfolios?
4. Within a journal, do changes in specialization track changes in cross-field
   reach for otherwise comparable papers?

The dirty primary null is
`log(mu_far,1 / mu_near,1) - log(mu_far,0 / mu_near,0) = 0`; the directional
alternative is less than zero. A null total-citation effect does not refute it.

## Target-trial emulation

- **Eligibility:** English-language OpenAlex works published in 2015-2020 with
  type `article`, a nonempty title, a published primary journal location and
  source ID, and neither XPAC nor retracted status. Reviews are excluded from the
  main cohort and analyzed separately.
- **Time zero:** publication year of the focal paper. Month is adjusted for but
  the citation clock uses complete calendar years for consistency across sources.
- **Treatment history:** journal publications in years `t-3` through `t-1`; the
  focal paper never contributes to its exposure.
- **Treatment:** journal-years in the top specialization quartile within the
  focal paper's venue-free semantic-cluster by publication-year choice set.
- **Comparator:** bottom-quartile journal-years in the same choice set. The middle
  50% is excluded from the primary binary contrast.
- **Follow-up:** incoming citations from journal articles published in `t` through
  `t+4`.
- **Primary outcome:** external five-year citations classified by the venue-free
  Qwen3 taxonomy as near, intermediate, far or unclassified.
- **Primary estimand:** average treatment effect among papers retained in empirical
  common support, defined by out-of-fold propensity scores in `[0.05, 0.95]`.
- **Effect scale:** log ratio of far-to-near marginal mean citation rates.
- **Clustering:** journal-level uncertainty throughout.

The causal interpretation requires consistency, conditional exchangeability,
positivity, and negligible interference. Failure of the prespecified measurement,
overlap, or balance gates automatically changes manuscript wording from effect to
association; it does not trigger estimator shopping.

## Causal graph

The adjustment graph is fixed as follows. `C` denotes manuscript content and
reference profile; `A` prior author history; `I` prior institution history; `J`
prior journal prestige, size, language, and OA share; `S` journal specialization;
`M` post-publication access and audience routing; and `Y` citation reach.

```text
C,A,I,J ──> S ──> M ──> Y
│ │ │ │      └────────> Y
└─┴─┴─┴────────────────> Y
```

`C`, `A`, `I`, and `J` are measured no later than `t-1` and adjusted. Focal-paper
OA and other post-publication visibility variables belong to `M` and are not
adjusted in the primary model. Residual manuscript quality and editorial selection
remain possible unmeasured common causes of `S` and `Y`; the target-trial language
does not make their absence empirically testable.

## Exposure measurement

The primary score is the mean pairwise cosine similarity of unit-normalized
SPECTER2 embeddings of titles from the journal's prior three years. For `N`
vectors `z`, it is calculated without pair sampling as

`(||sum(z)||^2 - N) / (N * (N - 1))`.

This estimator is not mechanically inflated by journal size and never receives
journal name. Primary journal-years require at least 100 historical papers.

Independent checks are:

1. SPECTER2 title-plus-abstract similarity among papers with abstracts;
2. unbiased Simpson concentration of fields represented in historical references;
3. OpenAlex primary-topic HHI and negative entropy as a transparent, potentially
   journal-contaminated baseline only.

The split-half gate uses the Spearman-Brown-corrected Pearson correlation between
hash-defined halves with at least 50 papers per half. Title-only embeddings also
define 1,000 venue-free paper clusters. Clusters of 500
and 2,000 and a 50-paper history threshold are sensitivity analyses. Split-half
reliability of the primary score must be at least 0.70.

## Outcomes and covariates (dirty Section B)

One thousand Qwen3 leaf clusters are fitted to the pre-2015 training sample, then
their paper-weighted centers are grouped into 32 macroclusters. Same leaf is `near`;
different leaf in the same macrocluster is `intermediate`; different macrocluster
is `far`. Non-English, missing-title and OOD citing papers are `unclassified`.
Macrocluster-specific held-out P99 squared distance to the assigned leaf center is
the OOD cutoff. Focal OOD papers are excluded from the primary estimate.

The exact external-citation decomposition is total = near + intermediate + far +
unclassified after excluding same-journal and shared-author citations. Secondary
outcomes are each component, any far citation and a 99.9%-winsorized routing ratio.

Adjustment variables are fixed at or before publication: the existing 32 SPECTER2
PCs, 32 Qwen3 PCs, semantic choice-set cluster, year and month, reference profile,
team/country/institution composition, prior author and institution history, and
the journal's prior size, prestige, English share and OA share. Journal ID,
OpenAlex field/topic labels, focal OA and post-publication journal features are
excluded. Published titles are an imperfect baseline-content proxy; preprint
matching is not performed.

## Estimation and gates (dirty Section B)

Five journal-grouped folds provide out-of-fold nuisance predictions. LightGBM
models treatment and arm-specific outcomes; AIPW estimates the four correlated
near/far arm means. The primary log routing ratio uses their joint paper-level
influence function. Journal-cluster intervals and all 500 multiplier-bootstrap
draws use the same journal weights across outcomes.

Required gates are:

- exact 2026-06-26 manifest, exposure windows, citation windows, and unique edges;
- split-half semantic reliability at least 0.70;
- at least two journals and 20 papers per treatment arm in each retained choice set;
- at least 50% of eligible extreme-arm papers retained in propensity support;
- all weighted absolute standardized mean differences below 0.10;
- exact citation decomposition with missing classifications retained;
- focal OOD exclusion-rate difference between arms at most 0.02;
- classified citing-flow coverage at least 0.80 in each arm.

Failure of a data, leakage, time-window or decomposition assertion stops the job.
Failure of an inferential gate retains all estimates but changes wording to
`exploratory association`. Even if every gate passes, this dirty run is not the
human-validated final QSS analysis.

## Data contracts and reporting

`journal_year_scope.parquet` remains the frozen Section A exposure contract.
`qwen3_semantics.parquet` contains leaf/macrocluster assignment, OOD status and 32
Qwen3 PCs. `analysis_dataset.parquet` contains baseline variables, treatment,
propensity, weights and outcomes. Exploratory outputs are `dirty_report.md`,
`dirty_estimates.csv`, `dirty_gates.csv` and stage run manifests; they never
overwrite a later final analysis.

Every stage writes a `run.json` with code commit, input snapshot, packages, seed,
row counts, sizes, and status. The 2020 pilot remains a separate frozen benchmark.
All positive, null, contrary, and nonevaluable results are retained.
