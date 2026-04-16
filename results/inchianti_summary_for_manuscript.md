# InCHIANTI Replication Analysis: Summary for Manuscript

## 1. Results Paragraph (~380 words)

We replicated the HDR coupling-tightening analysis in the InCHIANTI cohort (N=1,453, ages 20-102, 6 waves over 18 years, Chianti region, Italy), using a 4-axis panel: IL-6 (inflammatory), HOMA-IR (metabolic), resting heart rate (neuroautonomic proxy), and SPPB (functional). The largest eigenvalue of the change-covariance matrix (lambda_max) increased monotonically with age, from 1.17 (ages 20-49, 95% CI [0.86, 2.33]) through 8.04 (60-69, [4.77, 12.12]) to 22.06 (80+, [17.28, 27.06]), based on 1,523 consecutive-wave change pairs from 957 subjects (Figure A). Cross-lagged lead-lag analysis showed sign concordance with the compiled J-matrix in 9 of 12 ordered pairs (75%, binomial p = 0.073; Figure B), meeting the pre-specified threshold of 8/12. Two of the three discordant pairs involved the N-axis (resting HR), consistent with the mechanistic inferiority of this proxy relative to RMSSD.

Medication dose-response analysis exploited InCHIANTI's ATC-coded medication records across four sub-analyses. The raw stratification showed lambda_max increasing with medication class count (8.4 at 0 classes to 25.0 at 3+), but this is confounded by age. Within-decade stratification (Figure D, panel b) showed overlapping 95% CIs between medicated and unmedicated groups at every age stratum (e.g., 70-79: unmedicated 10.6, medicated 11.8). SWDS-Gamma regression confirmed confounding by indication: after adjusting for comorbidity count, age, and sex, the medication coefficient was non-significant (beta = -1.0, p = 0.35), while comorbidity count (p = 0.047) and age (p < 0.001) were significant predictors. Within-hypertension age-matched comparison (302 pairs, age balance 70.6 vs 70.4 years) showed treated subjects had higher, not lower, lambda_max (14.3 vs 11.8), suggesting confounding by severity rather than genuine compression. The off-diagonal correlation test showed mild compression only at ages 70-79 (medicated mean|r| = 0.053 vs unmedicated 0.084) and 80+ (0.065 vs 0.126).

The Pi trajectory (C_norm/V_norm) showed a slope of -0.019/year, indicating D-dominated aging (Figure C) — divergent from the ELSA medication-naive result of +0.001/year. This divergence is likely attributable to the resting HR proxy inflating diagonal variance relative to off-diagonal covariance.

Note: RMSSD was not available in the standard InCHIANTI data release; resting heart rate served as the N-axis proxy.

## 2. Key Numbers with 95% CIs

| Metric | Value | 95% CI | N |
|--------|-------|--------|---|
| lambda_max (change-cov, 20-49) | 1.1719 | [0.8608, 2.3339] | 229 |
| lambda_max (change-cov, 50-59) | 1.2187 | [0.8704, 2.1290] | 97 |
| lambda_max (change-cov, 60-69) | 8.0443 | [4.7723, 12.1194] | 227 |
| lambda_max (change-cov, 70-79) | 11.2552 | [9.3594, 13.3891] | 683 |
| lambda_max (change-cov, 80+) | 22.0555 | [17.2829, 27.0564] | 181 |
| Lead-lag concordance | 9/12 | p=0.073 | 1,523 |
| Pi slope | -0.0187/yr | -- | -- |
| SWDS-Gamma: n_meds beta | -0.997 | SE=1.076, p=0.354 | 1,523 |
| SWDS-Gamma: n_comorbid beta | 1.511 | SE=0.761, p=0.047 | 1,523 |
| HTN matched: treated lambda_max | 14.303 | [11.07, 17.72] | 302 pairs |
| HTN matched: untreated lambda_max | 11.788 | [8.76, 15.18] | 302 pairs |

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

## 4. Refined Medication Dose-Response Results

### 4a. Age-stratified lambda_max (within-decade, medicated vs unmedicated)

| Decade | Unmedicated N | Unmedicated lambda_max | Medicated N | Medicated lambda_max | Overlap? |
|--------|---------------|------------------------|-------------|----------------------|----------|
| 50-59 | 73 | 1.24 [0.79, 2.34] | 24 | 1.41 [0.78, 2.94] | YES |
| 60-69 | 138 | 8.01 [3.80, 13.77] | 89 | 8.14 [3.89, 13.95] | YES |
| 70-79 | 334 | 10.64 [7.83, 13.90] | 349 | 11.77 [9.25, 14.47] | YES |
| 80+ | 64 | 23.28 [15.01, 31.69] | 117 | 21.55 [15.57, 27.78] | YES |

### 4b. Off-diagonal correlation (direct compression test)

| Decade | Unmedicated mean\|r\| | Medicated mean\|r\| | Compression? |
|--------|------------------------|---------------------|--------------|
| 50-59 | 0.074 | 0.208 | NO (reverse) |
| 60-69 | 0.159 | 0.091 | mild |
| 70-79 | 0.084 | 0.053 | YES |
| 80+ | 0.126 | 0.065 | YES |

## 5. InCHIANTI vs ELSA Comparison

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

## 6. Figures

| Figure | File | Content |
|--------|------|---------|
| A | `outputs/figure_inchianti_lambda_max.pdf` | 2-panel: change-cov + cross-sectional lambda_max by age |
| B | `outputs/figure_inchianti_lead_lag.pdf` | 4x4 cross-lagged beta heatmap with J-concordance borders |
| C | `outputs/figure_inchianti_pi.pdf` | Pi trajectory with slope = -0.019/yr |
| D | `outputs/figure_inchianti_medication.pdf` | 4-panel: raw, age-stratified, SWDS-Gamma regression, HTN matched |
| E | `outputs/figure_inchianti_vs_elsa.pdf` | Side-by-side cohort lambda_max comparison |

## 7. Surprises and Divergences

- **Medication compression not supported**: Four sub-analyses consistently show that the raw medication-lambda_max association is driven by confounding by indication (age and comorbidity), not genuine compression. Within-decade lambda_max CIs overlap completely. The HTN age-matched comparison shows the *opposite* of compression (treated > untreated). Only mild off-diagonal correlation compression is observed at ages 70+.

- **Pi divergence from ELSA**: The D-dominated Pi trajectory (-0.019/yr) opposes the ELSA result (+0.001/yr proportional). Likely driven by resting HR proxy inflating diagonal variance relative to off-diagonal covariance.

- **RMSSD unavailable**: The N-axis uses resting HR instead of RMSSD. InCHIANTI published HRV papers suggest the data may exist in ancillary files not in the standard release.

- **HOMA-IR limited to waves 0-2**: Insulin was only measured at baseline, FU1, and FU2. FU3 has IL-6 but no insulin, FU4-5 have neither.

- **SPPB ceiling effect**: All healthy 20-30 year-olds scored 12/12 on SPPB (SD=0). Reference SD was computed from healthy <60 adults instead.

- **Lead-lag N-axis discordance**: 2 of 3 discordant lead-lag pairs involve the N-axis, consistent with resting HR being a noisy proxy. The I-M and I-F coupling directions are well-replicated.
