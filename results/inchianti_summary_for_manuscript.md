# InCHIANTI Replication Analysis: Summary for Manuscript

## 1. Results Paragraph (~380 words)

We replicated the HDR coupling-tightening analysis in the InCHIANTI cohort (N=1,453, ages 20-102, 6 waves over 18 years, Chianti region, Italy). We tested a 5-axis panel comprising IL-6 (inflammatory), HOMA-IR (metabolic), cortisol/DHEAS ratio (neuroendocrine), SPPB (functional), and PTH (bone remodelling), with resting heart rate as an alternative N-axis proxy. The largest eigenvalue of the change-covariance matrix (lambda_max) increased monotonically with age across all four configurations tested: from 3.28 (ages 20-49) to 28.69 (80+) for the 5-axis model, and from 1.17 to 22.06 for the original 4-axis (resting HR) model, based on 1,523-1,688 consecutive-wave change pairs. This monotonic coupling-tightening pattern, observed with different biomarker panels in a different country and age range than ELSA, constitutes the primary replication finding.

Cross-lagged lead-lag analysis tested sign concordance with the compiled J-matrix. Under Convention B (biological direction: all pairs predict positive beta in delta-space, capturing co-decline), the 4-axis resting HR model showed 9/12 (75%) concordance (binomial p = 0.073); under Convention A (naive J sign), 8/12 (67%). The 5-axis model achieved 12/20 (60%, Conv B) or 7/20 (35%, Conv A). A 3-convention audit (including transition-matrix Phi signs) established that Convention B is the appropriate comparison for cross-lagged regressions in declining systems. Several individual pairs were significant: F->I (beta = +0.015, p < 0.001), I->B (beta = +0.114, p < 0.01), and F->B (beta = +0.024, p < 0.01), the last directly testing the grade-A protective entry J_{B<-F} (mechanical loading preserves bone).

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

### Lead-lag concordance (3-convention audit)

| Configuration | Conv A (naive J) | Conv B (biological) | Conv C (Phi) |
|---|---|---|---|
| 4-axis (resting HR) | 8/12 (67%, p=0.19) | **9/12 (75%, p=0.073)** | 6/12 (50%) |
| 4-axis (cortisol/DHEAS) | 6/12 (50%) | 7/12 (58%) | 6/12 (50%) |
| 5-axis (I,M,N_cortdh,F,B_pth) | 7/20 (35%) | 12/20 (60%) | -- |
| 4-axis (NLR as I-axis) | 6/12 (50%) | 7/12 (58%) | 6/12 (50%) |

**Recommended: Convention B** — see `results/lead_lag_concordance_audit.md` for full rationale.

### Significant individual lead-lag pairs (5-axis model)

| Pair | beta | p | J-matrix sign | Concordant? |
|------|------|---|---|---|
| I->B | +0.114 | <0.01 | +1 | YES |
| F->I | +0.015 | <0.001 | -1 (adj: +1) | YES |
| F->B | +0.024 | <0.01 | -1 (adj: +1) | YES |
| B->M | +0.016 | <0.05 | -1 | NO |
| B->N | +0.017 | <0.05 | -1 | NO |

### Lead-lag FDR correction (two-cohort, Convention B)

Source: [`results/lead_lag_fdr_combined.json`](lead_lag_fdr_combined.json)
(Benjamini-Hochberg FDR applied independently within each cohort, produced
by `scripts/elsa_lead_lag.py`).

**ELSA (3-axis, N = 10,849 consecutive-wave pairs, 6,245 subjects):
4 / 6 pairs survive FDR < 0.05.**

| Pair | beta | p (raw) | q (FDR) | FDR<0.05 |
|------|-----:|--------:|--------:|:--------:|
| I→M | +0.0646 | 3.2e-20 | 1.9e-19 | ✓ |
| M→I | +0.0276 | 1.0e-4  | 3.0e-4  | ✓ |
| F→I | +0.0251 | 2.3e-3  | 4.6e-3  | ✓ |
| I→F | +0.0137 | 4.2e-3  | 6.3e-3  | ✓ |
| F→M | −0.0140 | 0.043   | 0.051   | ✗ (discordant) |
| M→F | +0.0074 | 0.071   | 0.071   | ✗ |

**InCHIANTI (4-axis, N = 1,523 consecutive-wave pairs): 0 / 12 pairs
survive FDR < 0.05** (strongest: F→I raw p = 0.0085 → q = 0.102; I→M
q = 0.123). The InCHIANTI per-pair significance story dissolves under
multiple-comparison correction; the defensible manuscript claim is
*directional concordance* (9/12 Conv-B concordant, binomial p = 0.073),
not individual-pair significance. ELSA carries the per-pair FDR-level
evidence (I→M, M→I, F→I, I→F all survive FDR<0.05).

### Lambda_max age-trajectory null tests

Source: [`results/lambda_max_null_tests.json`](lambda_max_null_tests.json)
(produced by `scripts/lambda_max_null_tests.py`, seed 42).

| Null | Observed ratio | Null median [95%] | p | Interpretation |
|------|---------------:|-------------------|--:|----------------|
| InCHIANTI age-permutation (n=1000) | 18.82×       | 1.54 [1.17, 2.32] | <0.001 | Amplification is age-specific, not a marginal artefact |
| InCHIANTI random-panel (all C(8,4)=70) | 18.82× (HDR) | 2.07 [1.19, 5.01] | <0.001 | HDR 4-axis selection matters: ~9× the median random panel |
| ELSA age-permutation (n=1000) | 1.24×           | 1.10 [1.03, 1.26] | 0.032  | Significant but marginal; ELSA magnitude supports direction only |

**Univariate-variance control (InCHIANTI; counterintuitive):** the ratio
lambda_max / max-single-axis-variance is 1.008 at ages 20-49, 1.018 at
50-59, then collapses to ~1.001 at 60-69, 70-79, and 80+. At old ages the
multivariate lambda_max is essentially the biggest individual-axis variance
— coupling contribution to lambda_max growth is negligible and the 19×
amplification is driven by single-axis variance growth (primarily F/SPPB
decline). The manuscript should note this caveat when attributing
lambda_max amplification to "coupling tightening".

### Survival analysis (Cox C-indices)

Source of truth: [`results/inchianti_cox_frozen.json`](inchianti_cox_frozen.json)
(age65+; regenerated by `scripts/inchianti_survival.py`). Event counts: 706
deaths / 931 at risk (age65+); 338 / 681 (med-naive).

| Model | Age 65+ (N=931) | Med-naive (N=681) |
|-------|----------------|-------------------|
| M1: Age + Sex | 0.741 | 0.825 |
| M2: + Biomarkers | 0.762 | 0.836 |
| M3: SWDS-Gamma alone | 0.667 | 0.715 |
| M4: + Fried frailty | 0.746 | 0.749 |
| M4a: M4 + biomarkers only (no SWDS-Gamma) | 0.760 | 0.763 |
| M4b: M4 + SWDS-Gamma only (no biomarkers) | 0.753 | 0.759 |
| M5: Full | 0.760 | 0.763 |
| **deltaC (M5-M4)** | **+0.014** | **+0.014** |
| deltaC (M4a-M4) biomarkers alone | +0.014 | +0.014 |
| deltaC (M4b-M4) SWDS-Gamma alone | +0.007 | +0.010 |

Interpretation: biomarkers alone carry the dominant incremental signal
above Fried frailty; SWDS-Gamma alone captures roughly half of that; the
combined M5-M4 does not exceed the biomarker-only increment, indicating
overlap between the coupling-weighted score and raw biomarker information.

ELSA comparison (source: [`results/elsa_cox_frozen.json`](elsa_cox_frozen.json)):
deltaC(M5-M4) = +0.009 (full, N=5,431; Mahalanobis and z-sum benchmarks both
also +0.009, so SWDS-Gamma adds no measurable signal over isotropic distance
in ELSA) and +0.013 (med-naive, N=3,233).

#### Bootstrap 95% confidence intervals for deltaC

Source: [`results/bootstrap_delta_c.json`](bootstrap_delta_c.json)
(event-stratified paired bootstrap, n=2000 resamples, produced by
`scripts/run_bootstrap_delta_c.py`).

| Comparison | deltaC | 95% CI | p | N / events |
|------------|-------:|--------|---:|-----------|
| InCHIANTI M5 - M4             | +0.0140 | [+0.0082, +0.0226] | <0.001  | 923 / 698  |
| InCHIANTI M4a - M4 (biomarkers alone) | +0.0141 | [+0.0080, +0.0227] | <0.001  | 923 / 698  |
| InCHIANTI M4b - M4 (SWDS-Gamma alone) | +0.0071 | [+0.0024, +0.0133] | 0.001   | 923 / 698  |
| ELSA M5 - M4 (full matched)   | +0.0087 | [+0.0030, +0.0191] | 0.0005  | 5,431 / 1,122 |
| ELSA M5 - M4 (med-naive)      | +0.0131 | [+0.0032, +0.0319] | 0.005   | 3,233 / 618 |

All five 95% CIs **exclude zero** (significant vs null), but **none exclude
0.01** (the Pencina/Steyerberg prognostic-biomarker threshold). SWDS-Gamma
adds signal above Fried frailty / Rockwood FI, but cannot be declared to
reliably clear the 0.01 threshold in either cohort.

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

## 6. Linearity Robustness Check (Supplementary Note 10)

Source: [`results/nonlinearity_test.json`](nonlinearity_test.json)
(regenerated by `scripts/inchianti_nonlinearity_test.py`).

Each of the 12 ordered-pair cross-lagged regressions was re-fit with three
nonlinear augmentations on the same N=1,523 triplets:

- **M1** adds a quadratic-in-predictor term (x_i²)
- **M2** adds a predictor × autoregressor interaction (x_i · x_j)
- **M3** adds both terms jointly

All models use HC3 heteroscedasticity-robust standard errors. Bonferroni
threshold for the 12×2 = 24 nonlinear-term tests is p < 0.0021.

| Test | Surviving Bonferroni | Min p | Threshold |
|------|----------------------|-------|-----------|
| Quadratic term (β_quad) | **0/12** | 0.09 | p<0.0021 |
| Interaction term (η) | **0/12** | 0.039 | p<0.0021 |
| Joint F-test M3 vs M0 | **0/12** | 0.050 | p<0.0042 (12-test Bonf) |
| Residual regression on x_i² (R²) | max 0.0007 | — | — |

The smallest nonlinear-term p-value across all 24 tests is 0.039
(N→M interaction), which does not survive correction. Residual
regressions of the linear-model residuals on x_i² yield R² ≤ 0.0007
across all pairs — no residual curvature is detectable. Conclusion:
the linear OU specification is adequate within the observed biomarker
range; behaviour far from equilibrium is not probed by this in-sample
test and remains a separate open question.

## 7. Surprises and Divergences

- **Lead-lag sign convention audited**: A 3-convention comparison established that Convention B (biological direction, 9/12 = 75%) is the correct comparison for cross-lagged regressions in declining systems. An intermediate "correction" to 6/12 was itself erroneous (double-counted the F-axis sign flip). The audited result restores the original 9/12 (Conv B) or 8/12 (Conv A, naive J sign). Convention C (Phi transition matrix) is uninformative at the 3-year InCHIANTI visit interval.

- **Medication compression not supported**: Four sub-analyses consistently show confounding by indication. The within-hypertension age-matched comparison (302 pairs) showed treated subjects had *higher* lambda_max (14.3 vs 11.8) -- the opposite of compression.

- **Pi divergence from ELSA**: All configurations show D-dominated Pi (-0.010 to -0.019/yr), divergent from ELSA (+0.001/yr). This is robust to N-axis choice and B-axis inclusion, suggesting a systematic cohort or biomarker difference rather than proxy noise.

- **B-axis pairs mostly discordant**: B->I, B->M, and B->N all show positive betas (pathological direction), but the J-matrix predicts negative (protective osteocalcin effects). This may reflect PTH being a remodelling regulator rather than a direct marker of bone formation capacity, or that the protective osteocalcin pathway is too weak to detect.

- **NLR is a poor I-axis substitute**: Replacing IL-6 with NLR reduced concordance to 4/12 (33%). The neutrophil/lymphocyte ratio captures a different aspect of immune function (innate/adaptive balance) than the IL-6 inflammatory resolution axis.

- **Survival deltaC, with decomposition**: deltaC(M5-M4) = +0.014 for both subgroups, slightly exceeding ELSA's values. The M4a/M4b decomposition clarifies the attribution: biomarkers alone contribute +0.014 above Fried frailty, while SWDS-Gamma alone contributes +0.007-0.010. Combined M5-M4 does not exceed biomarker-only M4a-M4, so the coupling-weighted score overlaps substantially with raw biomarker information rather than adding an independent signal.

- **deltaC significance, with caveat (added 2026-04-19)**: Event-stratified paired bootstrap (n=2000, `results/bootstrap_delta_c.json`) gives 95% CIs for deltaC(M5-M4) that exclude zero in all three cohorts (InCHIANTI age65+ +0.014 [+0.008,+0.023]; ELSA full +0.009 [+0.003,+0.019]; ELSA med-naive +0.013 [+0.003,+0.032]) but **none exclude 0.01**. The Pencina/Steyerberg prognostic-biomarker threshold is not reliably cleared by SWDS-Gamma above frailty-adjusted models.

- **Lead-lag under FDR (added 2026-04-19)**: Benjamini-Hochberg FDR within each cohort (`results/lead_lag_fdr_combined.json`) shows ELSA retains 4/6 pairs (I→M q=1.9e-19, M→I, F→I, I→F) but **InCHIANTI retains 0/12 pairs**. The InCHIANTI claim is *directional concordance* (9/12 Conv-B, binomial p=0.073), not per-pair significance. ELSA carries the FDR-level evidence.

- **Lambda_max single-axis dominance (added 2026-04-19)**: Null-test analysis (`results/lambda_max_null_tests.json`) shows InCHIANTI's 19× amplification is statistically real (age-permutation p<0.001; random-panel p<0.001). However, the ratio lambda_max / max-individual-variance collapses to ~1.001 at ages 70+, meaning the biggest eigenvalue is essentially dominated by single-axis (F/SPPB) variance at old ages — coupling contribution to lambda_max growth is negligible. The manuscript should distinguish "lambda_max amplifies with age" (true) from "coupling tightens with age" (not supported by this control).
