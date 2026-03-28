
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

