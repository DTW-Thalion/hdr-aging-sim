# Full Pipeline Report

Generated: 2026-04-09T01:55:29Z

## Step 1: Load mechanistic evidence
- Axes: ['I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B']
- Active J entries: 39
- Excluded entries: 3
- Calibration scalar: 0.062534

## Step 2-3: Stability summary (age 65)
- alpha(A): -0.002149
- Recovery time: 465.3 days
- Damping ratio: 1.0000

## Step 4: Age trajectory

| Age | alpha | Recovery (d) | Damping | beta(I,M) | Stable |
|-----|-------|-------------|---------|-----------|--------|
| 30 | -0.005538 | 180.6 | 1.0000 | 0.023784 | True |
| 40 | -0.003924 | 254.9 | 1.0000 | 0.012113 | True |
| 50 | -0.003001 | 333.2 | 1.0000 | 0.007303 | True |
| 60 | -0.002388 | 418.8 | 1.0000 | 0.004861 | True |
| 70 | -0.001944 | 514.3 | 1.0000 | 0.003450 | True |
| 80 | -0.001624 | 615.6 | 1.0000 | 0.002558 | True |

## Step 5: Sensitivity analysis (N=1000, prior_scale=0.12)
- Elapsed: 0.8s
- Monotone fraction: 0.9980
- Age 30: stable=100.0%, alpha_mean=-0.005538
- Age 50: stable=100.0%, alpha_mean=-0.003001
- Age 80: stable=100.0%, alpha_mean=-0.001638

## Step 6: Synthetic ELSA-like cohort (N=2000)
- Generated in 3.4s
- Persons: 2000, Visits: 4
- Biomarkers: ['log_CRP', 'HbA1c_BMI', 'grip_strength']

## Step 7: Tier-1 pipeline
- lambda_max(Gamma_change): 1.529 -> 1.647 -> 2.134 -> 2.048
- Increases with age: False
- SWDS means: 0.510 -> 0.561 -> 0.777 -> 0.826
- Increases with age: True
- Primacy Pi: 1.000 -> 1.276 -> 1.253 -> 1.294

## Step 8: Single-intervention ranking (age 70)

| Rank | Intervention | delta-alpha | % change | n_J |
|------|-------------|------------|----------|-----|
| 1 | nad_precursors | -0.0000398 | -2.0% | 3 |
| 2 | exercise_resistance | -0.0000021 | -0.1% | 17 |
| 3 | mitoq | -0.0000011 | -0.1% | 2 |
| 4 | rapamycin | -0.0000010 | -0.1% | 1 |
| 5 | canakinumab | -0.0000006 | -0.0% | 1 |
| 6 | anakinra | -0.0000006 | -0.0% | 1 |
| 7 | colchicine | -0.0000003 | -0.0% | 2 |
| 8 | pioglitazone | -0.0000003 | -0.0% | 1 |
| 9 | circadian_hygiene | -0.0000000 | -0.0% | 2 |
| 10 | senolytic_dq | -0.0000000 | -0.0% | 1 |

## Step 9: R6 2x2x2 factorial

| Arm | alpha | SWDS |
|-----|-------|------|
| control | -0.001624 | 0.8490 |
| circadian_hygiene | -0.001624 | 0.7134 |
| exercise_resistance | -0.001610 | 0.6412 |
| exercise_resistance+circadian_hygiene | -0.001610 | 0.6770 |
| colchicine | -0.001624 | 0.7563 |
| colchicine+circadian_hygiene | -0.001624 | 0.7141 |
| colchicine+exercise_resistance | -0.001611 | 0.6664 |
| colchicine+exercise_resistance+circadian_hygiene | -0.001611 | 0.6405 |

### Main effects (delta-SWDS)
- colchicine: -0.0258
- exercise_resistance: -0.1020
- circadian_hygiene: -0.0420

### 2-way interactions
- colchicine*exercise_resistance: +0.0403 (antagonism)
- colchicine*circadian_hygiene: +0.0158 (antagonism)
- exercise_resistance*circadian_hygiene: +0.0938 (antagonism)

## Step 10: ABC self-consistency test
- Observed summary: lambda_max_trend_slope=0.0106, off_diag_sign_concordance=0.6000, pi_slope=0.0145, swds_p75=0.7553
- Proposals: 500
- Accepted: 25 (5.0%)
- Tolerance: 1.0753
- Elapsed: 1.0s
- Entries tightened: 30/39
- Entries shifted: 5/39
- Mean posterior/prior std ratio: 0.761
- Sign concordance: 1.000 (p=0.029 vs null 0.497)

## Summary

Total elapsed: 5.3s

| Step | Status |
|------|--------|
| 1. Load evidence | 39 entries |
| 2-3. Stability | alpha=-0.002149 |
| 4. Age trajectory | 6 ages, all stable |
| 5. Sensitivity | monotone=0.998 |
| 6. Synthetic cohort | N=2000 in 3.4s |
| 7. Tier-1 | Gamma_change increases: False |
| 8. Interventions | top: nad_precursors |
| 9. Factorial | 8 arms, synergy: False |
| 10. ABC | concordance=1.000 |
