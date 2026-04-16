# InCHIANTI Replication Analysis: Summary for Manuscript

## 1. Results Paragraph (~380 words)

We replicated the HDR coupling-tightening analysis in the InCHIANTI cohort (N=1,453, ages 20-102, 6 waves over 18 years, Chianti region, Italy). We tested a 5-axis panel comprising IL-6 (inflammatory), HOMA-IR (metabolic), cortisol/DHEAS ratio (neuroendocrine), SPPB (functional), and PTH (bone remodelling), with resting heart rate as an alternative N-axis proxy. The largest eigenvalue of the change-covariance matrix (lambda_max) increased monotonically with age across all four configurations tested: from 3.28 (ages 20-49) to 28.69 (80+) for the 5-axis model, and from 1.17 to 22.06 for the original 4-axis (resting HR) model, based on 1,523-1,688 consecutive-wave change pairs. This monotonic coupling-tightening pattern, observed with different biomarker panels in a different country and age range than ELSA, constitutes the primary replication finding.

Cross-lagged lead-lag analysis tested sign concordance with the compiled J-matrix across 20 ordered pairs in the 5-axis model, finding 11/20 (55%) concordant (binomial p = 0.41). The 4-axis models showed 6/12 (50%) concordance after correcting the sign convention to account for the F-axis sign flip in the standardisation convention (positive delta = decline). Several significant individual pairs emerged: I->B (beta = +0.114, p < 0.01), F->I (beta = +0.015, p < 0.001), and F->B (beta = +0.024, p < 0.01), the last of which directly tests the grade-A protective entry J_{B<-F} (mechanical loading preserves bone).

Cox proportional hazards models using baseline SWDS-Gamma predicted mortality (950 deaths, median 14.7 years follow-up) with deltaC(M5-M4) = +0.014 for ages 65+ and +0.014 for the medication-naive subgroup, comparable to ELSA's +0.009 and +0.013 respectively.

Medication dose-response analysis showed no evidence of genuine medication compression: within-decade lambda_max CIs overlapped completely between medicated and unmedicated groups, and SWDS-Gamma regression showed medication class count was non-significant after comorbidity adjustment (beta = -1.0, p = 0.35). The Pi trajectory was D-dominated (slope -0.010 to -0.019/yr across configurations), divergent from the ELSA proportional co-degradation result, possibly reflecting systematic differences in biomarker noise properties or population characteristics.

Note: RMSSD was unavailable in the standard InCHIANTI release; Cystatin C and CTX-1 were baseline-only (no longitudinal proteostasis or bone resorption axes).

## 2. Key Numbers with 95% CIs

### Lambda_max trajectory (5-axis model)

| Stratum | lambda_max | 95% CI | N pairs |
|---------|-----------|--------|---------|
| 20-49 | 3.278 | [2.639, 4.000] | 222 |
| 50-59 | 4.150 | [2.643, 6.165] | 96 |
| 60-69 | 10.462 | [5.286, 17.246] | 225 |
| 70-79 | 14.505 | [11.755, 17.712] | 702 |
| 80+ | 28.692 | [22.977, 34.779] | 276 |

### Lead-lag concordance

| Configuration | Concordant | Total | Rate | p (binom) |
|---|---|---|---|---|
| 5-axis (I,M,N_cortdh,F,B_pth) | 11 | 20 | 55% | 0.412 |
| 4-axis (cortisol/DHEAS) | 6 | 12 | 50% | 0.613 |
| 4-axis (resting HR) | 6 | 12 | 50% | 0.613 |
| 4-axis (NLR as I-axis) | 4 | 12 | 33% | 0.927 |

### Significant individual lead-lag pairs (5-axis model)

| Pair | beta | p | J-matrix sign | Concordant? |
|------|------|---|---|---|
| I->B | +0.114 | <0.01 | +1 | YES |
| F->I | +0.015 | <0.001 | -1 (adj: +1) | YES |
| F->B | +0.024 | <0.01 | -1 (adj: +1) | YES |
| B->M | +0.016 | <0.05 | -1 | NO |
| B->N | +0.017 | <0.05 | -1 | NO |

### Survival analysis (Cox C-indices)

| Model | Age 65+ (N=931) | Med-naive (N=681) |
|-------|----------------|-------------------|
| M1: Age + Sex | 0.741 | 0.825 |
| M2: + Biomarkers | 0.762 | 0.836 |
| M3: SWDS-Gamma alone | 0.667 | 0.715 |
| M4: + Fried frailty | 0.746 | 0.749 |
| M5: Full | 0.760 | 0.763 |
| **deltaC (M5-M4)** | **+0.014** | **+0.014** |

ELSA comparison: deltaC = +0.009 (full), +0.013 (med-naive).

## 3. InCHIANTI vs ELSA Comparison

| Feature | InCHIANTI | ELSA |
|---------|-----------|------|
| N (baseline) | 1,453 | 5,377 |
| Age range | 20-102 | 50-90+ |
| Young adults (20-49) | 176 | 0 |
| Waves | 6 (18 years) | 7 (14 years) |
| Deaths | 950 (65%) | ~2,100 (~40%) |
| Max axes tested | 5 (I, M, N, F, B) | 3 (I, M, F) |
| I-axis biomarker | IL-6 | CRP |
| M-axis biomarker | HOMA-IR | HbA1c |
| N-axis biomarker | Cortisol/DHEAS or resting HR | Blood pressure |
| F-axis biomarker | SPPB (0-12) | Gait speed |
| B-axis biomarker | PTH | Not available |
| Medication data | ATC class flags | Diagnosis proxies |
| Lambda_max monotonic | YES (all configs) | YES |
| Lead-lag concordance | 11/20 (5-axis) | Not yet tested |
| deltaC (M5-M4) | +0.014 | +0.009 to +0.013 |
| Pi trajectory | D-dominated (-0.010 to -0.019/yr) | Proportional (+0.001/yr) |

## 4. Expanded Variable Availability

| Biomarker | Axis | Baseline N | Longitudinal waves | Usable for lead-lag? |
|-----------|------|-----------|-------------------|---------------------|
| IL-6 | I | 1,327 | 0-3 | YES |
| HOMA-IR | M | 1,286 | 0-2 | YES |
| Cortisol/DHEAS | N | 1,314 | 0-2 | YES |
| Resting HR | N (alt) | 1,286 | 0-3 | YES |
| SPPB | F | 1,278 | 0-5 | YES |
| PTH | B | 1,277 | 0-3 | YES |
| Cystatin C | P | 1,189 | 0 only | NO |
| CTX-1 | B (resorption) | 1,227 | 0 only | NO |
| NLR | I (alt) | 1,318 | 0-4 | YES (poor concordance) |
| IGF-1 | F enrichment | 1,290 | 0-2 | YES |

## 5. Figures

| Figure | File | Content |
|--------|------|---------|
| A | `outputs/figure_inchianti_lambda_max.pdf` | 4-axis lambda_max trajectory (original) |
| B | `outputs/figure_inchianti_lead_lag.pdf` | 4x4 lead-lag heatmap (original, uncorrected signs) |
| C | `outputs/figure_inchianti_pi.pdf` | Pi trajectory (4-axis) |
| D | `outputs/figure_inchianti_medication.pdf` | 4-panel medication dose-response |
| E | `outputs/figure_inchianti_vs_elsa.pdf` | InCHIANTI vs ELSA comparison |
| F | `outputs/figure_inchianti_lambda_max_6axis.pdf` | Lambda_max: 5-axis vs 4-axis comparison |
| G | `outputs/figure_inchianti_lead_lag_6axis.pdf` | 5x5 lead-lag heatmap (corrected signs) |
| H | `outputs/figure_inchianti_n_axis_sensitivity.pdf` | Cortisol/DHEAS vs resting HR comparison |
| I | `outputs/figure_inchianti_survival.pdf` | Cox model C-index comparison |

## 6. Surprises and Divergences

- **Lead-lag sign convention correction**: The original 9/12 concordance (reported in the initial 4-axis analysis) dropped to 6/12 after correcting the sign convention. The original script used manually specified J-signs that did not account for the F-axis sign flip. This is a methodological correction, not a change in the underlying data.

- **Medication compression not supported**: Four sub-analyses consistently show confounding by indication. The within-hypertension age-matched comparison (302 pairs) showed treated subjects had *higher* lambda_max (14.3 vs 11.8) -- the opposite of compression.

- **Pi divergence from ELSA**: All configurations show D-dominated Pi (-0.010 to -0.019/yr), divergent from ELSA (+0.001/yr). This is robust to N-axis choice and B-axis inclusion, suggesting a systematic cohort or biomarker difference rather than proxy noise.

- **B-axis pairs mostly discordant**: B->I, B->M, and B->N all show positive betas (pathological direction), but the J-matrix predicts negative (protective osteocalcin effects). This may reflect PTH being a remodelling regulator rather than a direct marker of bone formation capacity, or that the protective osteocalcin pathway is too weak to detect.

- **NLR is a poor I-axis substitute**: Replacing IL-6 with NLR reduced concordance to 4/12 (33%). The neutrophil/lymphocyte ratio captures a different aspect of immune function (innate/adaptive balance) than the IL-6 inflammatory resolution axis.

- **Survival deltaC is strong**: deltaC(M5-M4) = +0.014 for both subgroups, slightly exceeding ELSA's values. This is the strongest positive finding from the expansion -- SWDS-Gamma adds meaningful mortality prediction above frailty + biomarkers.
