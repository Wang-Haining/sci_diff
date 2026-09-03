# Estimating the specialization effect of journals on the disciplinary reach of science

**Design version:** qss_v1
**Status:** FROZEN
**Frozen:** 2026-09-02
**Snapshot:** OpenAlex Parquet 2026-06-26
**Seed:** 20260902

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

The primary null is
`ATE_cross_field - ATE_within_subfield = 0`; the directional alternative is less
than zero. A null total-citation effect does not refute the primary hypothesis.

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
- **Primary outcome:** external cross-field citations per paper, excluding citing
  papers sharing an author or journal with the focal paper.
- **Primary estimand:** average treatment effect among papers retained in empirical
  common support, defined by out-of-fold propensity scores in `[0.05, 0.95]`.
- **Effect scales:** adjusted mean difference, adjusted mean ratio, and risk
  difference for any cross-field citation.
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

## Outcomes and covariates

All categorical citation outcomes retain an `unclassified` category. The complete
decomposition is total, within-subfield, different-subfield/same-field,
different-field, and unclassified. Two-year versions, any cross-field citation,
and estimates that retain journal and author self-citations are secondary.

Semantic reach is checked independently using cosine distance between focal and
citing-paper SPECTER2 embeddings. Uncited papers receive zero semantic diffusion
mass; mean citing distance is explicitly conditional on at least one citation.

Adjustment variables are fixed at or before publication: 32 embedding principal
components, semantic cluster, year and month, reference count and reference-field
entropy, author and institution counts, countries and international collaboration,
author and institution works/citations through `t-1`, and the journal's historical
size, annualized citation prestige, English share, and OA share. Focal-paper OA is
not adjusted in the primary model because it may lie on the venue-to-reach path.

## Estimation and gates

Five journal-grouped folds provide out-of-fold nuisance predictions. LightGBM
models the treatment probability and arm-specific outcomes; AIPW estimates arm
means and contrasts. Journal-cluster influence-function intervals are primary and
500 journal multiplier-bootstrap draws are the verification interval.

Paper interdisciplinarity is reference-field entropy, examined by prespecified
quartiles and a continuous modifier. Within-journal triangulation uses aggregated
journal-by-semantic-cluster-by-year cells, alternating-projection fixed effects for
journal and cluster-year, and journal-clustered uncertainty.

Required gates are:

- exact 2026-06-26 manifest, exposure windows, citation windows, and unique edges;
- split-half semantic reliability at least 0.70;
- at least two journals and 20 papers per treatment arm in each retained choice set;
- at least 50% of eligible extreme-arm papers retained in propensity support;
- all weighted absolute standardized mean differences below 0.10;
- exact citation decomposition with missing classifications retained;
- the primary estimate and contrast-of-effects replicated by at least one
  independent exposure or outcome definition before claiming audience segmentation.

## Data contracts and reporting

`journal_year_scope.parquet` contains journal ID, focal year, history N, three
scope measures, split-half scores, prior prestige, prior OA share, and reliability
flags. `analysis_dataset.parquet` contains the paper ID, baseline variables,
treatment, fold, propensity, weights, and outcomes. `causal_estimates.csv` contains
RQ, population, exposure, outcome, effect scale, estimate, interval, support, and
gate status.

Every stage writes a `run.json` with code commit, input snapshot, packages, seed,
row counts, sizes, and status. The 2020 pilot remains a separate frozen benchmark.
All positive, null, contrary, and nonevaluable results are retained.
