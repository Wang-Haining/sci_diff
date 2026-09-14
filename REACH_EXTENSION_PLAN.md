# Audience breadth and follow-up extension

Approved 2026-09-14. These analyses extend, and do not replace, the frozen primary results.

## Fixed comparison

- Use the saved deterministic comparison sample: 3,827,491 papers and 20,215 journals.
- Reuse its saved propensity unchanged and the original journal-grouped five-fold assignments.
- Fit new outcome models on this fixed support, using the existing baseline variables, Poisson/binary objectives, and deterministic early-stopped LightGBM profile. No new propensity selection.
- The original 3,818,173-paper primary support was not persisted. No extension is described as a reconstruction of that fit.
- Save sample IDs, predictions, arm scores, fitted outcome models, and shared journal multipliers for all new estimates.
- Snapshot remains 2026-06-26. Existing raw data, inputs, models, and results are read-only.

## Questions and outputs

1. **Audience breadth.** Estimate differences in the number of distinct citing areas, other areas, and topics, alongside the citation counts and any-other-area probability on the same sample. Uncited papers have zero distinct areas/topics. Shannon entropy is undefined with zero classified citations. Expected area richness in 3, 5, and 10 citations is computed analytically without replacement; it is a conditional, descriptive diagnostic, with arm-specific eligibility fractions and fixed-IPW means, not a causal estimate for all papers.
2. **Distance and counts.** Same-topic citations form bin 0. Four other bins use pooled, unweighted non-same-topic edge-distance quartiles on this sample. Report AIPW counts, relative counts, and shares from the component marginal means. These shares are not means of individual paper shares and do not use source-area standardization. The existing fine-distance graphic uses different bins and source-area-standardized IPW shares.
3. **Size and prestige.** First report Spearman correlations on unique scored journal-years, with journal scope versus prior article volume and versus prior prestige. Then report routing contrasts within prior-volume tertiles, with cutpoints based on unique journal-years in the comparison population.
4. **Total citations.** Report the total mean ratio with a joint influence-function interval and the 99.9% winsorized version. Select the cap from the pooled fixed-support outcome distribution and record it.
5. **Time.** Restrict to 2015 papers in the saved support. Report 0–60, 0–120, and 60–120 months using the same papers, baseline adjustment, and propensity. Reconcile the rebuilt first-five-year counts exactly to the saved outcomes. Report the five-year citation stock; do not adjust for it in the main comparison because it may lie on the pathway from publication to later citation. Stock-conditioned analysis is not part of this initial run.

Both 32-area and named-31-area citation-origin definitions are retained. The named-area version recodes Mixed-record citing origins as unclassified, without dropping Mixed focal papers or changing total citations. The two definitions share distance cutpoints. All endpoints use the existing language, out-of-distribution, shared-author, same-journal, and publication-date rules.

## Inference and integrity

- Seed 20260902; 500 Gaussian journal multipliers shared across all endpoints within each analysis. Same draws for the paired 60-versus-120-month contrast.
- Every citation pair is unique. Windows are half-open and defined by calendar-month offsets. Counts add to total, and first plus second five-year counts add to ten-year counts; the any-citation indicators combine by logical OR, not addition.
- Joint delta-method scores propagate all arm-mean uncertainty for ratios and shares. A nonpositive adjusted mean leaves an absolute estimate interpretable but makes its log ratio undefined; this is recorded, not replaced with a constant.
- Classification coverage and rarefaction eligibility are reported by arm. Conditional diagnostics are not described as evidence that journal scope has no effect on entry into another field.
- Related but nonidentical endpoints are not independent replications. Absence of significance is not equivalence. No result determines a publication destination or a merger decision.
- W6/W7 are not authorized for this run. No new model training, taxonomy, preprint study changes, or manuscript-result replacement.

## Storage and execution

New persistent data: `qss_v3/reach_extension_v1/`. New aggregate results: `results/reach_extension_v1/`; run manifests: `artifacts/reach_extension_v1/`.

Combined qss_v2, qss_v3, and staged embedding input must remain at or below 200 GB; combined spill at or below 400 GB; group free space at least 1.5 TB. No personal-storage fallback. New output paths fail if already present.

Prepare produces breadth/distance outcomes and the 2015 extended citation inputs. The 60-month estimator can run while missing ten-year citing titles are encoded with the existing pinned Qwen3 model and taxonomy. A second estimator fits the three follow-up windows after encoding and count reconciliation. Result synchronization waits for both estimators. No equivalent job may be submitted twice.

## Reconciliation before new results

The earlier distance implementation correctly used unweighted edge quantiles; weighted-bin wording in its comment, manuscript, and figure axis was wrong. Correcting that wording does not change any result. A broad-to-narrow distance trend is not established by comparing the three categorical count estimates with a differently standardized distance-share curve. The new aligned estimates address that comparison directly.

Reach and the preprint study share frozen Qwen and SPECTER measurements, but differ in paper versions, samples, time windows, citation denominators, and adjustment. A five-to-ten-year trend does not by itself resolve these differences or identify the preprint mechanism outside its observed population.
