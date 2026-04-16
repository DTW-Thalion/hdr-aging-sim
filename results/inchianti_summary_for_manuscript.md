# InCHIANTI Replication Analysis: Summary for Manuscript

## 1. Results Paragraph (~380 words)

We replicated the HDR coupling-tightening analysis in the InCHIANTI cohort (N=1,453, ages 20-102, 6 waves over 18 years, Chianti region, Italy), using a 4-axis panel: IL-6 (inflammatory), HOMA-IR (metabolic), resting heart rate (neuroautonomic proxy), and SPPB (functional).  The largest eigenvalue of the change-covariance matrix (lambda_max) increased from 1.172 (ages 20-49) to 22.056 (ages 80+), based on 1523 consecutive-wave change pairs.  Cross-lagged lead-lag analysis showed sign concordance with the compiled J-matrix in 9 of 12 ordered pairs (binomial p = 0.073 vs. chance).  Medication dose-response stratification showed lambda_max of 8.405 (0 medication classes) vs 24.951 (3+ classes).  The Pi trajectory (C_norm/V_norm) showed a slope of -0.0187/year, consistent with D-dominated aging.  Note: RMSSD was not available in the standard InCHIANTI data release; resting heart rate served as the N-axis proxy. Despite this limitation, the 4-axis results provide independent replication of the ELSA coupling-tightening finding with a mechanistically richer biomarker panel and younger age range (20-102).

## 2. Key Numbers with 95% CIs

| Metric | Value | 95% CI | N |
|--------|-------|--------|---|
| lambda_max (change-cov, 20-49) | 1.1719 | [0.8608, 2.3339] | 229 |
| lambda_max (change-cov, 50-59) | 1.2187 | [0.8704, 2.1290] | 97 |
| lambda_max (change-cov, 60-69) | 8.0443 | [4.7723, 12.1194] | 227 |
| lambda_max (change-cov, 70-79) | 11.2552 | [9.3594, 13.3891] | 683 |
| lambda_max (change-cov, 80+) | 22.0555 | [17.2829, 27.0564] | 181 |
| Lead-lag concordance | 9/12 | p=0.073 | 1523 |
| Pi slope | -0.0187/yr | — | — |

## 3. Lead-Lag Sign Concordance Detail

| Pair | beta | 95% CI | Predicted | Observed | Match |
|------|------|--------|-----------|----------|-------|
| M->I | 0.0028 | [-0.0352, 0.0406] | + | + | YES |
| I->M | 0.0328 | [-0.0019, 0.0672] | + | + | YES |
| N->I | -0.0030 | [-0.0441, 0.0393] | + | - | NO |
| I->N | -0.0427 | [-0.0806, -0.0059] | - | - | YES |
| F->I | 0.0143 | [0.0034, 0.0249] | + | + | YES |
| I->F | 0.0696 | [-0.1032, 0.2340] | + | + | YES |
| N->M | 0.0138 | [-0.0236, 0.0508] | + | + | YES |
| M->N | 0.0016 | [-0.0360, 0.0414] | - | + | NO |
| F->M | -0.0022 | [-0.0104, 0.0061] | + | - | NO |
| M->F | 0.0597 | [-0.1124, 0.2329] | + | + | YES |
| F->N | 0.0006 | [-0.0107, 0.0122] | + | + | YES |
| N->F | 0.0632 | [-0.1097, 0.2406] | + | + | YES |

## 4. InCHIANTI vs ELSA Comparison

| Feature | InCHIANTI | ELSA |
|---------|-----------|------|
| N (baseline) | 1,453 | 5,377 |
| Age range | 20-102 | 50-90+ |
| Young adults (20-49) | 176 | 0 |
| Waves | 6 (18 years) | 7 (14 years) |
| Axes | 4 (I, M, N, F) | 3 (I, M, F) |
| N-axis | Resting HR (proxy) | Blood pressure |
| I-axis biomarker | IL-6 | CRP |
| M-axis biomarker | HOMA-IR | HbA1c |
| F-axis biomarker | SPPB (0-12) | Gait speed |
| Medication data | ATC class flags | Diagnosis proxies |

## 5. Surprises and Divergences

- **RMSSD unavailable**: The N-axis uses resting HR instead of RMSSD. This is a mechanistic downgrade but InCHIANTI published HRV papers suggest the data may exist in ancillary files not in the standard release.
- **HOMA-IR limited to waves 0-2**: Insulin was only measured at baseline, FU1, and FU2. FU3 has IL-6 but no insulin, FU4-5 have neither. This limits 4-axis longitudinal coverage.
- **SPPB ceiling effect**: All healthy 20-30 year-olds scored 12/12 on SPPB (SD=0). Reference SD was computed from healthy <60 adults instead.
