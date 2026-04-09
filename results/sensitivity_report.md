# Sensitivity Analysis Report

Generated: 2026-04-09T01:19:09Z
MC draws: 10000
Total elapsed: 7.1s

## 1. Monte Carlo Analysis

### 1.1 Stability by Age

| Age | Stable % | alpha mean | alpha [5%, 95%] | Recovery (d) | beta_IM mean | beta_IM [5%] |
|-----|----------|------------|-----------------|--------------|-------------|-------------|
| 30 | 100.0 | -0.005538 | [-0.005543, -0.005533] | 180.6 | 0.023784 | 0.023781 |
| 40 | 100.0 | -0.003924 | [-0.003937, -0.003910] | 254.9 | 0.012113 | 0.012108 |
| 50 | 100.0 | -0.003001 | [-0.003028, -0.002971] | 333.3 | 0.007303 | 0.007296 |
| 60 | 100.0 | -0.002387 | [-0.002440, -0.002331] | 418.9 | 0.004861 | 0.004853 |
| 70 | 100.0 | -0.001946 | [-0.002046, -0.001844] | 514.5 | 0.003450 | 0.003439 |
| 80 | 100.0 | -0.001631 | [-0.001837, -0.001440] | 616.6 | 0.002558 | 0.002545 |

### 1.2 Monotone alpha Ordering

Fraction of draws with monotonically increasing alpha across ages: **99.9%**

## 2. Entry Sensitivity Ranking (alpha, OAT)

| Rank | Entry | Importance | delta | Prior sigma |
|------|-------|-----------|-------|-------------|
| 1 | J_mito_E | 0.000252 | 0.000453 | 0.0375 |
| 2 | J_F_B | 0.000208 | 0.000416 | 0.1500 |
| 3 | J_E_mito | 0.000159 | 0.000318 | 0.0300 |
| 4 | J_I_E | 0.000117 | 0.000196 | 0.1200 |
| 5 | J_F_E | 0.000100 | 0.000199 | 0.0800 |
| 6 | J_M_B | 0.000100 | 0.000166 | 0.1200 |
| 7 | J_I_F | 0.000045 | 0.000075 | 0.1200 |
| 8 | J_mito_M | 0.000043 | 0.000072 | 0.1200 |
| 9 | J_I_B | 0.000034 | 0.000069 | 0.0800 |
| 10 | J_B_F | 0.000027 | 0.000045 | 0.0450 |
| 11 | J_M_mito | 0.000023 | 0.000039 | 0.1200 |
| 12 | J_P_E | 0.000022 | 0.000036 | 0.0450 |
| 13 | J_F_mito | 0.000020 | 0.000040 | 0.1500 |
| 14 | J_mito_P | 0.000016 | 0.000026 | 0.1200 |
| 15 | J_E_I | 0.000012 | 0.000019 | 0.0450 |

### 2.1 Bifurcation Margin Sensitivity (top 10)

| Rank | Entry | Importance | delta |
|------|-------|-----------|-------|
| 1 | J_I_M | 0.000033 | 0.000065 |
| 2 | J_M_I | 0.000033 | 0.000065 |
| 3 | J_I_mito | 0.000000 | 0.000000 |
| 4 | J_I_B | 0.000000 | 0.000000 |
| 5 | J_M_B | 0.000000 | 0.000000 |
| 6 | J_E_mito | 0.000000 | 0.000000 |
| 7 | J_mito_I | 0.000000 | 0.000000 |
| 8 | J_mito_E | 0.000000 | 0.000000 |
| 9 | J_P_mito | 0.000000 | 0.000000 |
| 10 | J_C_I | 0.000000 | 0.000000 |

## 3. Stress Tests

### 3.1 Concordance Analysis

| Scenario | Concordance | n_agree / n_total |
|----------|-------------|-------------------|
| correct_prior | 0.795 | 31/39 |
| null_prior | 0.641 | 25/39 |
| adversarial_prior | 0.179 | 7/39 |

Ordering: correct (0.795) > null (0.641) > adversarial (0.179): **PASS**

### 3.2 Confidence Grade Ablation

- Entries ablated: 16 (non-informative / R6-only)
- alpha_full: -0.002149
- alpha_ablated: -0.002132
- |delta_alpha| / |alpha|: 0.0080
- Stable after ablation: True

### 3.3 Exclusion Impact

- Excluded entries added back: 3 (['J_mito_B', 'J_B_M', 'J_B_N'])
- alpha_without: -0.002149
- alpha_with: -0.002187
- |delta_alpha| / |alpha|: 0.0175
- Exclusion safe (<5%): **True**
- Mean |delta_SWDS|: 0.047720

### 3.4 Decomposition-Informed vs Uniform Prior

- Informative entries: 23
- std(alpha) narrow: 0.000574
- std(alpha) wide (2x sigma): 0.001372
- std reduction: 58.1%
- CI90 narrow: 0.001910
- CI90 wide: 0.004205
- CI90 reduction: 54.6%

## 4. Acceptance Criteria Summary

- [PASS] **stability_gt99_all_ages**: True
- [PASS] **monotone_alpha_gt99**: True
- [PASS] **im_loop_dominant**: True
- [PASS] **exclusion_safe_lt5pct**: True
- [PASS] **concordance_ordering**: True
