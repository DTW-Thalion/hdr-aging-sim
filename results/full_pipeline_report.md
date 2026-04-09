# Full Pipeline Report

Generated: 2026-04-09T12:03:13Z

## Step 1: Load mechanistic evidence
- Axes: ['I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B']
- Active J entries: 39
- Excluded entries: 3
- Calibration scalar: 0.034514

## Step 2-3: Stability summary (age 65)
- alpha(A): -0.000356
- Recovery time: 2807.1 days
- Damping ratio: 1.0000

## Step 4: Age trajectory

| Age | alpha | Recovery (d) | Damping | beta(I,M) | Stable |
|-----|-------|-------------|---------|-----------|--------|
| 30 | -0.000993 | 1006.8 | 1.0000 | 0.357135 | True |
| 40 | -0.000699 | 1431.4 | 1.0000 | 0.182205 | True |
| 50 | -0.000526 | 1900.9 | 1.0000 | 0.110215 | True |
| 60 | -0.000406 | 2464.1 | 1.0000 | 0.073772 | True |
| 70 | -0.000311 | 3214.4 | 1.0000 | 0.052810 | True |
| 80 | -0.000230 | 4353.5 | 1.0000 | 0.039656 | True |

## Step 5: Sensitivity analysis (N=1000, prior_scale=0.12)
- Elapsed: 0.7s
- Monotone fraction: 1.0000
- Age 30: stable=100.0%, alpha_mean=-0.000993
- Age 50: stable=100.0%, alpha_mean=-0.000526
- Age 80: stable=100.0%, alpha_mean=-0.000230

## Step 6: Synthetic ELSA-like cohort (N=2000)
- Generated in 3.2s
- Persons: 2000, Visits: 4
- Biomarkers: ['log_CRP', 'HbA1c_BMI', 'grip_strength']

## Step 7: Tier-1 pipeline
- lambda_max(Gamma_change): 0.708 -> 0.788 -> 0.966 -> 0.942
- Increases with age: False
- SWDS means: 0.275 -> 0.302 -> 0.347 -> 0.394
- Increases with age: True
- Primacy Pi: 1.000 -> 0.495 -> 0.650 -> 0.628

## Step 8: Single-intervention ranking (age 70)

| Rank | Intervention | delta-alpha | % change | n_J |
|------|-------------|------------|----------|-----|
| 1 | nad_precursors | -0.0000008 | -0.2% | 3 |
| 2 | canakinumab | -0.0000002 | -0.1% | 1 |
| 3 | anakinra | -0.0000002 | -0.1% | 1 |
| 4 | mitoq | -0.0000002 | -0.1% | 2 |
| 5 | colchicine | -0.0000002 | -0.0% | 2 |
| 6 | exercise_resistance | -0.0000001 | -0.0% | 17 |
| 7 | senolytic_dq | -0.0000001 | -0.0% | 1 |
| 8 | metformin | -0.0000001 | -0.0% | 5 |
| 9 | pioglitazone | -0.0000000 | -0.0% | 1 |
| 10 | semaglutide | -0.0000000 | -0.0% | 2 |

## Step 9: R6 2x2x2 factorial

| Arm | alpha | SWDS |
|-----|-------|------|
| control | -0.000230 | 0.3637 |
| circadian_hygiene | -0.000230 | 0.3284 |
| exercise_resistance | -0.000230 | 0.3298 |
| exercise_resistance+circadian_hygiene | -0.000230 | 0.3092 |
| colchicine | -0.000230 | 0.3340 |
| colchicine+circadian_hygiene | -0.000230 | 0.3371 |
| colchicine+exercise_resistance | -0.000230 | 0.2989 |
| colchicine+exercise_resistance+circadian_hygiene | -0.000230 | 0.2876 |

### Main effects (delta-SWDS)
- colchicine: -0.0184
- exercise_resistance: -0.0344
- circadian_hygiene: -0.0161

### 2-way interactions
- colchicine*exercise_resistance: -0.0158 (synergy)
- colchicine*circadian_hygiene: +0.0238 (antagonism)
- exercise_resistance*circadian_hygiene: +0.0002 (antagonism)

## Step 10: ABC self-consistency test
- Observed summary: lambda_max_trend_slope=0.0032, off_diag_sign_concordance=0.6000, pi_slope=-0.0086, swds_p75=0.3611
- Proposals: 500
- Accepted: 25 (5.0%)
- Tolerance: 1.3306
- Elapsed: 0.9s
- Entries tightened: 35/39
- Entries shifted: 7/39
- Mean posterior/prior std ratio: 0.737
- Sign concordance: 1.000 (p=0.029 vs null 0.497)

## Summary

Total elapsed: 4.9s

| Step | Status |
|------|--------|
| 1. Load evidence | 39 entries |
| 2-3. Stability | alpha=-0.000356 |
| 4. Age trajectory | 6 ages, all stable |
| 5. Sensitivity | monotone=1.000 |
| 6. Synthetic cohort | N=2000 in 3.2s |
| 7. Tier-1 | Gamma_change increases: False |
| 8. Interventions | top: nad_precursors |
| 9. Factorial | 8 arms, synergy: True |
| 10. ABC | concordance=1.000 |
