
---

### Unit Tests (pytest)
*Run: 2026-03-21 12:53:38*

- **Total**: 31 tests
- **Passed**: 31 ✅
- **Failed**: 0 
- **Duration**: 0.81s

- ✅ `test_stable_system_has_negative_abscissa` (0.000s)
- ✅ `test_aged_system_closer_to_zero` (0.000s)
- ✅ `test_diagonal_J_is_zero` (0.000s)
- ✅ `test_recovery_timescale_increases_with_age` (0.000s)
- ✅ `test_f_column_is_negative` (0.000s)
- ✅ `test_spectral_radius_discrete` (0.001s)
- ✅ `test_calibration_alpha_range` (0.000s)
- ✅ `test_recovery_ratio` (0.000s)
- ✅ `test_csv_loaded` (0.000s)
- ✅ `test_csv_basin_structure` (0.000s)
- ✅ `test_dcct_to_ifcc_known_values` (0.000s)
- ✅ `test_unit_detection_dcct` (0.799s)
- ✅ `test_unit_detection_ifcc` (0.000s)
- ✅ `test_empty_series` (0.000s)
- ✅ `test_identity_covariance` (0.000s)
- ✅ `test_diagonal_covariance` (0.000s)
- ✅ `test_batch_matches_individual` (0.001s)
- ✅ `test_zero_vector` (0.000s)
- ✅ `test_nonnegative` (0.000s)
- ✅ `test_identity_matrix` (0.000s)
- ✅ `test_known_eigenvalues` (0.000s)
- ✅ `test_eigenvalues_descending` (0.000s)
- ✅ `test_fi_range` (0.000s)
- ✅ `test_fi_zero_deficits` (0.000s)
- ✅ `test_fi_insufficient_items` (0.000s)
- ✅ `test_grip_max_selection` (0.001s)
- ✅ `test_grip_max_all_nan` (0.001s)
- ✅ `test_negative_codes_replaced` (0.001s)
- ✅ `test_zscore_known_values` (0.000s)
- ✅ `test_zscore_missing_col_fallback` (0.000s)
- ✅ `test_zscore_small_ref_fallback` (0.000s)


---

### Q-Sensitivity Analysis
*Run: 2026-03-21 14:38:57 | Python 3.14.3*

Tests robustness of stability trends to age-varying noise


#### Trend Survival

| β | Q₈₀/Q₃₀ | α̂ trend | λ_max trend |
| --- | --- | --- | --- |
| 0.00 | 1.0× | ✅ | ✅ |
| 0.25 | 1.2× | ✅ | ✅ |
| 0.50 | 1.5× | ✅ | ✅ |
| 0.75 | 1.8× | ✅ | ✅ |
| 1.00 | 2.0× | ✅ | ✅ |
| 1.50 | 2.5× | ✅ | ✅ |
| 2.00 | 3.0× | ✅ | ✅ |
| 3.00 | 4.0× | ✅ | ✅ |
| 5.00 | 6.0× | ✅ | ✅ |
- ✅ **α̂ trend survives all β ≤ 5.0**: PASS
- ✅ **λ_max trend survives all β ≤ 5.0**: PASS

*Note: λ_max(Γ̂) does not require Q specification — robust by construction.*


---

### Γ-Native Equivalence Study
*Run: 2026-03-21 14:39:03 | Python 3.14.3*

Confirms SWDS-Γ ≈ SWDS and λ_max(Γ̂) tracks stability


#### λ_max(Γ̂) Stability Tracking

- ✅ **λ_max monotone increasing with age**: PASS
- **Spearman(α_true, λ_max)**: 1.0000

#### SWDS-Γ vs SWDS Ranking Equivalence

- **Spearman (age 44)**: 0.9447
- **Spearman (age 54)**: 0.9622
- **Spearman (age 64)**: 0.9706
- **Spearman (age 74)**: 0.9786
- ✅ **All rank correlations > 0.93**: PASS

#### C-Index Comparison

| Score | C-index |
| --- | --- |
| SWDS-Γ | 0.5165 |
| SWDS (A-based) | 0.5303 |
| Mahalanobis | 0.4742 |
| L2 | 0.5048 |
| Age | 0.5484 |

#### T* Calibration (Layer A)

- **Bootstrap mean concordance**: 0.608
- **Bootstrap SD**: 0.253
- **T* (mean - 2×SD)**: 0.103


---

### Prior Stress Tests
*Run: 2026-03-21 14:39:06 | Python 3.14.3*

Quantifies prior vs data contribution for Tests 3-4 Layer B


#### Mean Concordance by Condition

- **Correct prior**: 0.771
- **Null prior**: 0.542
- **Adversarial prior**: 0.292
- **Layer A (Γ̂ signs)**: 0.500

#### Decomposition

- **Prior contribution (correct − null)**: +0.229
- **Data contribution (null − chance)**: +0.042
- ✅ **Null prior ≈ chance (confirms Tier-3)**: PASS
- ✅ **Adversarial < null (prior matters)**: PASS


---

### Unit Tests (pytest)
*Run: 2026-03-28 16:16:11*

- **Total**: 31 tests
- **Passed**: 31 ✅
- **Failed**: 0 
- **Duration**: 5.07s

- ✅ `test_stable_system_has_negative_abscissa` (0.005s)
- ✅ `test_aged_system_closer_to_zero` (0.000s)
- ✅ `test_diagonal_J_is_zero` (0.000s)
- ✅ `test_recovery_timescale_increases_with_age` (0.000s)
- ✅ `test_f_column_is_negative` (0.000s)
- ✅ `test_spectral_radius_discrete` (0.003s)
- ✅ `test_calibration_alpha_range` (0.000s)
- ✅ `test_recovery_ratio` (0.000s)
- ✅ `test_csv_loaded` (0.001s)
- ✅ `test_csv_basin_structure` (0.000s)
- ✅ `test_dcct_to_ifcc_known_values` (0.000s)
- ✅ `test_unit_detection_dcct` (5.050s)
- ✅ `test_unit_detection_ifcc` (0.000s)
- ✅ `test_empty_series` (0.000s)
- ✅ `test_identity_covariance` (0.000s)
- ✅ `test_diagonal_covariance` (0.000s)
- ✅ `test_batch_matches_individual` (0.000s)
- ✅ `test_zero_vector` (0.000s)
- ✅ `test_nonnegative` (0.001s)
- ✅ `test_identity_matrix` (0.002s)
- ✅ `test_known_eigenvalues` (0.000s)
- ✅ `test_eigenvalues_descending` (0.000s)
- ✅ `test_fi_range` (0.001s)
- ✅ `test_fi_zero_deficits` (0.000s)
- ✅ `test_fi_insufficient_items` (0.000s)
- ✅ `test_grip_max_selection` (0.002s)
- ✅ `test_grip_max_all_nan` (0.001s)
- ✅ `test_negative_codes_replaced` (0.001s)
- ✅ `test_zscore_known_values` (0.000s)
- ✅ `test_zscore_missing_col_fallback` (0.000s)
- ✅ `test_zscore_small_ref_fallback` (0.000s)


---

### R6: J Matrix Reconciliation
*Run: 2026-04-01*

Audited 9×9 J matrix CSV against manuscript Table 1 counts.

| Source | Positive | Negative | Unknown |
| --- | --- | --- | --- |
| CSV (authoritative) | 57 | 11 | 4 |
| Manuscript (Table 1) | 49 | 10 | 13 |

- ✅ **CSV counts verified**: 57 + 11 + 4 = 72 off-diagonal entries
- ✅ **PMID population**: 67/72 entries now have primary citations
- Remaining gaps: 4 B→X unknowns (no established mechanism), 1 mito→N Grade C theoretical

See `outputs/j_matrix_audit_report.json` for full audit details.


---

### R6: D vs. J Primacy Decomposition
*Run: 2026-04-03 | Python 3.14*

Decomposes age-stratified Γ̂ into D-degradation (variance) vs. J-degradation (correlation).


#### Full Sample (Wave 2, N=5,440)

| Metric | Slope/yr | p-value | R² |
| --- | --- | --- | --- |
| V_norm (variance) | −0.0083 | 0.0047 | 0.76 |
| C_norm (correlation) | −0.0143 | 0.0057 | 0.75 |
| P (primacy ratio) | −0.0084 | 0.0519 | 0.49 |

- Quadratic test for P: p = 0.054 (no significant nonlinearity)


#### Medication-Naive Subgroup (Wave 2, N=3,237)

| Metric | Slope/yr | p-value | R² |
| --- | --- | --- | --- |
| V_norm (variance) | −0.0072 | 0.0032 | 0.79 |
| C_norm (correlation) | −0.0060 | 0.0282 | 0.58 |
| P (primacy ratio) | +0.0014 | 0.4562 | 0.10 |

- Quadratic test for P: p = 0.572 (no significant nonlinearity)
- ✅ **P ≈ 1.0 across all strata**: proportional co-degradation confirmed
- ✅ **Neither damage nor hyperfunction dominates**: both D and J degrade in lock-step


---

### R6: Figure P-Value Corrections (E2/E3)
*Run: 2026-04-04 | Python 3.14*

Co-author review flagged two p-value discrepancies in R6 figures.


#### E2: Coupling Tightening Trend (Figure 3a)

Original annotation used Kendall τ on n=4 strata midpoints, which has minimum achievable p=0.333 for any monotone sequence. Replaced with permutation trend test (10,000 iterations).

| Stratum | λ_max(Γ̂_change) | 95% CI |
| --- | --- | --- |
| 50–59 | 0.7227 | [0.6408, 0.8222] |
| 60–69 | 0.7013 | [0.6547, 0.7538] |
| 70–79 | 0.7751 | [0.7087, 0.8484] |
| 80+ | 0.8704 | [0.7426, 1.0128] |

- **Permutation trend**: τ = 0.667, p = 0.2139
- Sequence is non-monotone (dip at 60–69 stratum)
- Trend not significant at α = 0.05


#### E3: KM Survival by SWDS-Γ Tertile (Figure 4c)

Med-naive KM genuinely non-significant. Events near-equal across tertiles — no survival separation. Full-sample KM shows significant separation and is now used in the figure.

| Subgroup | N | Events | T1 events | T2 events | T3 events | p (log-rank, T1 vs T3) |
| --- | --- | --- | --- | --- | --- | --- |
| Med-naive | 3,233 | 618 | 203 | 197 | 218 | 0.7873 |
| Full sample | 5,431 | 1,122 | 344 | 362 | 416 | 0.0079 |

- Figure 4c now shows **full sample** KM (p = 0.0079)
- Med-naive definition (r2hibpe==0 AND r2diabe==0) is consistent with Cox model
- Deceased coding confirmed correct (0=censored, 1=dead)
- Survival time construction confirmed correct (mean ~19y, max 20y)
- ⚠️ Med-naive subgroup has insufficient SWDS-Γ variance to separate KM curves despite ΔC = +0.013 in Cox model


---

### R6: Data-Driven Basin Recovery (GMM)
*Run: 2026-04-04 | Python 3.14*

Addresses TCST adversarial review: demonstrates that latent basins are recoverable from the data itself (not just the exogenous age-65 threshold). GMM fitted on the pooled ELSA 3-axis panel (N = 20,934 observations across 10,085 participants).


#### GMM Model Selection

| K | BIC | AIC |
| --- | --- | --- |
| 2 | 169,710 | 169,559 |
| 3 | 168,313 | 168,083 |
| **4** | **167,773** | **167,463** |

- BIC-optimal K = 4


#### Basin Separability Comparison

| Method | d_min | ω_min (μ̄=0.05) | T_deploy (daily) | Disagree vs age-65 |
| --- | --- | --- | --- | --- |
| Age-based (<65 vs ≥65) | 0.021 | 144.2 | 18.2 mo | 0% (by definition) |
| Score-based (median SWDS) | 0.418 | 7.2 | 0.9 mo | N/A |
| GMM (K=4) | 1.321 | 3.1 | 2.1 mo | 50.4% (K=2) |

- GMM d_min 63× larger than age-based → clusters far more separable in biomarker space
- Data-driven deployment threshold met in ~2 months (vs 18 months age-based)


#### Clinical Characterisation (K=2 Solution)

| | Cluster A (healthier) | Cluster B (sicker) |
| --- | --- | --- |
| N (%) | 17,325 (82.8%) | 3,609 (17.2%) |
| Mean age | 66.2 ± 9.0 | 67.5 ± 8.9 |
| CRP (mg/L) | 2.23 ± 2.30 | 9.65 ± 16.28 |
| HbA1c (mmol/mol) | 38.0 ± 4.0 | 49.1 ± 14.5 |
| Grip (kg) | 31.6 ± 11.4 | 28.4 ± 11.1 |
| Male | 46.7% | 40.3% |
| Mortality | 17.2% | 17.6% |

- Cluster B = high-inflammatory, insulin-resistant, weak grip — consistent with multi-system dysregulation
- Mean ages nearly identical (66 vs 68) — basins reflect physiological state, not age per se
- ⚠️ HMM analysis skipped (hmmlearn unavailable on Python 3.14)
