GO

# OpenAlex journal-specialization dirty pilot

- Snapshot: 2026-06-26 (649,096,577 records)
- Journal scopes: 39,202; focal papers: 2,502,770; citation edges: 23,743,646
- Analysis papers: 2,502,770; duplicate reference entries removed: 0
- HHI vs negative-entropy Spearman correlation: 0.958

## Complete outcome decomposition

| measure | estimation | outcome | mean_broad | mean_specialized | contrast | ci_low | ci_high | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hhi | raw | total_citations | 9.835 | 8.790 | -1.045 | nan | nan | 0.735 |
| hhi | standardized | total_citations | 11.798 | 11.016 | -0.782 | -2.243 | 0.203 | 0.735 |
| hhi | raw | within_subfield | 4.587 | 4.938 | 0.351 | nan | nan | 0.735 |
| hhi | standardized | within_subfield | 5.653 | 5.914 | 0.260 | -0.571 | 0.661 | 0.735 |
| hhi | raw | cross_subfield | 1.486 | 1.266 | -0.220 | nan | nan | 0.735 |
| hhi | standardized | cross_subfield | 1.857 | 1.596 | -0.262 | -0.520 | -0.064 | 0.735 |
| hhi | raw | cross_field | 3.761 | 2.585 | -1.176 | nan | nan | 0.735 |
| hhi | standardized | cross_field | 4.287 | 3.506 | -0.781 | -1.196 | -0.419 | 0.735 |
| entropy | raw | total_citations | 10.435 | 8.440 | -1.995 | nan | nan | 0.716 |
| entropy | standardized | total_citations | 11.449 | 10.774 | -0.675 | -1.728 | 0.207 | 0.716 |
| entropy | raw | within_subfield | 4.991 | 4.825 | -0.166 | nan | nan | 0.716 |
| entropy | standardized | within_subfield | 5.667 | 5.959 | 0.293 | -0.361 | 0.662 | 0.716 |
| entropy | raw | cross_subfield | 1.512 | 1.194 | -0.318 | nan | nan | 0.716 |
| entropy | standardized | cross_subfield | 1.763 | 1.523 | -0.239 | -0.413 | -0.078 | 0.716 |
| entropy | raw | cross_field | 3.931 | 2.420 | -1.511 | nan | nan | 0.716 |
| entropy | standardized | cross_field | 4.020 | 3.292 | -0.728 | -1.037 | -0.422 | 0.716 |

## Interpretation boundary

This is an exploratory observational pilot, not a causal estimate. OpenAlex topics may use journal information; residual manuscript sorting remains; and citing works are restricted to non-XPAC, non-retracted journal articles from 2020-2024.
