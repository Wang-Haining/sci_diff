EXPLORATORY ASSOCIATION ONLY

# QSS v2 dirty Qwen3 report

## Primary routing estimate

- theta: -0.107877 (95% CI -0.151184 to -0.064570; multiplier bootstrap -0.149404 to -0.066571)
- Broad far/near marginal mean ratio: 1.421266
- Specialized far/near marginal mean ratio: 1.275924
- Near means, broad/specialized: 1.943976 / 1.798653
- Far means, broad/specialized: 2.762907 / 2.294944

## Counts and diagnostics

- Candidate rows: 15,233,734
- Extreme-arm rows before focal OOD exclusion: 7,683,322
- Primary rows after focal OOD exclusion: 7,617,662
- Common-support rows: 5,729,600
- Journals: 20,486
- Focal OOD, broad/specialized: 0.6054% / 1.0410%
- Classified citing flow, broad/specialized: 96.5624% / 97.1990%
- English citing flow, broad/specialized: 96.9880% / 97.5967%
- English missing-title flow, broad/specialized: 0.0021% / 0.0024%
- English OOD flow, broad/specialized: 0.4235% / 0.3953%
- Maximum weighted absolute SMD: 0.144141
- Top 0.1% threshold: 234 citations; observed contribution: 7.5617%

## Gates

- section_a_measurement: PASS (0.9850569378571297; >=0.70)
- common_support: PASS (0.7521467872951044; >=0.50)
- weighted_max_abs_smd: FAIL (0.14414084338772695; <0.10)
- focal_ood_arm_difference: PASS (0.004355693154527033; <=0.02)
- broad_citing_coverage: PASS (0.9656242126636789; >=0.80)
- specialized_citing_coverage: PASS (0.9719896505677165; >=0.80)
- primary_direction: PASS (-0.10787737463558111; <0)

This is the prespecified dirty direction-finding run, not the human-validated final QSS analysis.
