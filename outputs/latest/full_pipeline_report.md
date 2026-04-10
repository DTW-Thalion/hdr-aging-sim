# Full Pipeline Report

Generated: 2026-04-10T00:51:14Z

## Step 1: Load mechanistic evidence
- All axes: ['I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B']
- Fast axes (dynamical): ['I', 'M', 'mito', 'P', 'C', 'N', 'F']
- Quasi-static axes (forcing): ['E', 'B']
- Fast subsystem dimension: 7
- Active J entries: 39
- Excluded entries: 3
- Calibration scalar: 0.271655

## Step 2-3: Stability summary (age 65)
- alpha(A_fast): -0.021218  (fast 7-axis subsystem)
- alpha(A_full): 0.000764  (full 9-axis, E/B-dominated)
- Recovery time: 47.1 days
- Damping ratio: 1.0000
- Equilibrium shift (E/B forcing): 0.4350

## Step 4: Age trajectory

| Age | alpha | Recovery (d) | Damping | beta(I,M) | Stable |
|-----|-------|-------------|---------|-----------|--------|
| 30 | -0.120000 | 8.3 | 1.0000 | 1.428099 | True |
| 40 | -0.062560 | 16.0 | 1.0000 | 0.673202 | True |
| 50 | -0.039450 | 25.3 | 1.0000 | 0.390376 | True |
| 60 | -0.026149 | 38.2 | 1.0000 | 0.254264 | True |
| 70 | -0.017000 | 58.8 | 1.0000 | 0.178362 | True |
| 80 | -0.010035 | 99.6 | 1.0000 | 0.131673 | True |

## Step 5: Sensitivity analysis (N=1000, prior_scale=0.12)
- Elapsed: 1.1s
- Monotone fraction: 1.0000
- Age 30: stable=100.0%, alpha_mean=-0.120016
- Age 50: stable=100.0%, alpha_mean=-0.039475
- Age 80: stable=100.0%, alpha_mean=-0.010138

## Step 6: Synthetic ELSA-like cohort (N=2000)
- Generated in 4.6s
- Persons: 2000, Visits: 4
- Biomarkers: ['log_CRP', 'HbA1c_BMI', 'grip_strength']

## Step 7: Tier-1 pipeline
- lambda_max(Gamma_change): 0.534 -> 0.753 -> 1.090 -> 1.325
- Increases with age: True
- SWDS means: 0.214 -> 0.340 -> 1.017 -> 1.913
- Increases with age: True
- Primacy Pi: 1.000 -> 1.844 -> 1.625 -> 1.496

## Step 8: Single-intervention ranking (age 70)

| Rank | Intervention | delta-alpha | % change | n_J |
|------|-------------|------------|----------|-----|
| 1 | exercise_resistance | -0.0046998 | -27.6% | 19 |
| 2 | metformin | -0.0001342 | -0.8% | 6 |
| 3 | anti_tnf | -0.0000927 | -0.5% | 3 |
| 4 | pioglitazone | -0.0000736 | -0.4% | 1 |
| 5 | semaglutide | -0.0000562 | -0.3% | 2 |
| 6 | empagliflozin | -0.0000353 | -0.2% | 2 |
| 7 | tocilizumab | -0.0000113 | -0.1% | 1 |
| 8 | senolytic_dq | +0.0000000 | +0.0% | 1 |
| 9 | teriparatide | +0.0000000 | +0.0% | 1 |
| 10 | denosumab | +0.0000000 | +0.0% | 4 |

## Step 9: R6 2x2x2 factorial

| Arm | alpha | SWDS |
|-----|-------|------|
| control | -0.010035 | 0.4466 |
| circadian_hygiene | -0.009983 | 0.4770 |
| exercise_resistance | -0.014430 | 0.2730 |
| exercise_resistance+circadian_hygiene | -0.014344 | 0.3098 |
| colchicine | -0.009838 | 0.4302 |
| colchicine+circadian_hygiene | -0.009795 | 0.4556 |
| colchicine+exercise_resistance | -0.014138 | 0.3142 |
| colchicine+exercise_resistance+circadian_hygiene | -0.014068 | 0.3283 |

### Main effects (delta-SWDS)
- colchicine: +0.0054
- exercise_resistance: -0.1460
- circadian_hygiene: +0.0266

### 2-way interactions
- colchicine*exercise_resistance: +0.0488 (antagonism)
- colchicine*circadian_hygiene: -0.0138 (synergy)
- exercise_resistance*circadian_hygiene: -0.0025 (synergy)

## Step 10: ABC self-consistency test
- Observed summary: lambda_max_trend_slope=0.0178, off_diag_sign_concordance=0.8000, pi_slope=0.0079, swds_p75=0.3144
- Proposals: 500
- Accepted: 25 (5.0%)
- Tolerance: 0.9226
- Elapsed: 1.1s
- Entries tightened: 23/26
- Entries shifted: 4/26
- Mean posterior/prior std ratio: 0.716
- Sign concordance: 1.000 (p=0.029 vs null 0.497)

## Summary

Total elapsed: 7.0s

| Step | Status |
|------|--------|
| 1. Load evidence | 39 entries |
| 2-3. Stability | alpha=-0.021218 |
| 4. Age trajectory | 6 ages, all stable |
| 5. Sensitivity | monotone=1.000 |
| 6. Synthetic cohort | N=2000 in 4.6s |
| 7. Tier-1 | Gamma_change increases: True |
| 8. Interventions | top: exercise_resistance |
| 9. Factorial | 8 arms, synergy: True |
| 10. ABC | concordance=1.000 |
