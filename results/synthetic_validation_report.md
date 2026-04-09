# Synthetic Cohort Validation Report

Generated: 2026-04-09T01:31:29Z
Total elapsed: 33.8s

Design: N=5000, ages 50-90, 4 visits at 4-year intervals, ELSA 3-axis (I, M, F)

## Condition: clean
Generation time: 10.6s

### lambda_max(Gamma_change) by age stratum

| Age | N | lambda_max | trace |
|-----|---|-----------|-------|
| 50-60 | 2282 | 1.4210 | 2.1436 |
| 60-70 | 3827 | 1.6704 | 2.4745 |
| 70-80 | 3728 | 2.0259 | 2.9422 |
| 80-90 | 3686 | 2.1595 | 3.1135 |

### lambda_max(Gamma_cross) by age stratum

| Age | N | lambda_max |
|-----|---|-----------|
| 50-60 | 1287 | 0.7123 |
| 60-70 | 1258 | 0.8579 |
| 70-80 | 1215 | 0.9609 |
| 80-90 | 1240 | 1.0917 |

### SWDS-Gamma by age stratum

| Age | N | mean | std | median |
|-----|---|------|-----|--------|
| 50-60 | 1287 | 0.5431 | 0.6627 | 0.2889 |
| 60-70 | 1258 | 0.6487 | 0.8442 | 0.3435 |
| 70-80 | 1215 | 0.7275 | 0.9177 | 0.3981 |
| 80-90 | 1240 | 0.8196 | 0.9705 | 0.4640 |

### Primacy ratio

| Age | V_norm | C_norm | Pi |
|-----|--------|--------|-----|
| 50-60 | 1.000 | 1.000 | 1.000 |
| 60-70 | 1.184 | 1.963 | 1.658 |
| 70-80 | 1.347 | 2.141 | 1.589 |
| 80-90 | 1.515 | 2.559 | 1.690 |

## Condition: survivorship
Generation time: 7.1s

### lambda_max(Gamma_change) by age stratum

| Age | N | lambda_max | trace |
|-----|---|-----------|-------|
| 50-60 | 1623 | 1.4012 | 2.1168 |
| 60-70 | 2308 | 1.6124 | 2.3975 |
| 70-80 | 1782 | 1.7174 | 2.6165 |
| 80-90 | 1260 | 1.8371 | 2.8159 |

### lambda_max(Gamma_cross) by age stratum

| Age | N | lambda_max |
|-----|---|-----------|
| 50-60 | 1000 | 0.5956 |
| 60-70 | 863 | 0.6706 |
| 70-80 | 639 | 0.6849 |
| 80-90 | 452 | 0.5231 |

### SWDS-Gamma by age stratum

| Age | N | mean | std | median |
|-----|---|------|-----|--------|
| 50-60 | 1000 | 0.4466 | 0.5479 | 0.2337 |
| 60-70 | 863 | 0.5039 | 0.6692 | 0.2697 |
| 70-80 | 639 | 0.5178 | 0.6951 | 0.2652 |
| 80-90 | 452 | 0.4147 | 0.4980 | 0.2267 |

### Primacy ratio

| Age | V_norm | C_norm | Pi |
|-----|--------|--------|-----|
| 50-60 | 1.000 | 1.000 | 1.000 |
| 60-70 | 1.136 | 1.497 | 1.318 |
| 70-80 | 1.201 | 2.363 | 1.967 |
| 80-90 | 1.009 | 1.703 | 1.688 |

## Condition: medication
Generation time: 9.6s

### lambda_max(Gamma_change) by age stratum

| Age | N | lambda_max | trace |
|-----|---|-----------|-------|
| 50-60 | 2206 | 1.5506 | 2.2083 |
| 60-70 | 3718 | 1.6310 | 2.4039 |
| 70-80 | 3730 | 1.9164 | 2.7912 |
| 80-90 | 3793 | 2.0582 | 2.9318 |

### lambda_max(Gamma_cross) by age stratum

| Age | N | lambda_max |
|-----|---|-----------|
| 50-60 | 1230 | 0.7055 |
| 60-70 | 1238 | 0.8258 |
| 70-80 | 1255 | 1.0579 |
| 80-90 | 1277 | 0.9905 |

### SWDS-Gamma by age stratum

| Age | N | mean | std | median |
|-----|---|------|-----|--------|
| 50-60 | 1230 | 0.5378 | 0.6881 | 0.2721 |
| 60-70 | 1238 | 0.6296 | 0.7993 | 0.3345 |
| 70-80 | 1255 | 0.7932 | 1.0241 | 0.4105 |
| 80-90 | 1277 | 0.7455 | 0.9918 | 0.3847 |

### Primacy ratio

| Age | V_norm | C_norm | Pi |
|-----|--------|--------|-----|
| 50-60 | 1.000 | 1.000 | 1.000 |
| 60-70 | 1.175 | 1.026 | 0.873 |
| 70-80 | 1.427 | 1.694 | 1.187 |
| 80-90 | 1.383 | 1.749 | 1.264 |

## Condition: both
Generation time: 6.5s

### lambda_max(Gamma_change) by age stratum

| Age | N | lambda_max | trace |
|-----|---|-----------|-------|
| 50-60 | 1596 | 1.3239 | 2.0102 |
| 60-70 | 2324 | 1.5133 | 2.2446 |
| 70-80 | 1933 | 1.7318 | 2.5277 |
| 80-90 | 1410 | 1.8040 | 2.6495 |

### lambda_max(Gamma_cross) by age stratum

| Age | N | lambda_max |
|-----|---|-----------|
| 50-60 | 1003 | 0.6518 |
| 60-70 | 860 | 0.6035 |
| 70-80 | 683 | 0.6911 |
| 80-90 | 474 | 0.5872 |

### SWDS-Gamma by age stratum

| Age | N | mean | std | median |
|-----|---|------|-----|--------|
| 50-60 | 1003 | 0.4929 | 0.6605 | 0.2456 |
| 60-70 | 860 | 0.4643 | 0.5729 | 0.2517 |
| 70-80 | 683 | 0.5266 | 0.6690 | 0.2707 |
| 80-90 | 474 | 0.4481 | 0.5622 | 0.2468 |

### Primacy ratio

| Age | V_norm | C_norm | Pi |
|-----|--------|--------|-----|
| 50-60 | 1.000 | 1.000 | 1.000 |
| 60-70 | 0.976 | 1.193 | 1.222 |
| 70-80 | 1.097 | 1.202 | 1.095 |
| 80-90 | 0.976 | 1.813 | 1.858 |

## Acceptance Criteria

- [PASS] **gamma_change_increases_clean**
- [PASS] **gamma_cross_decreases_survivorship**
- [PASS] **swds_increases_with_age**
- [PASS] **medication_compresses_correlations**
- [PASS] **swds_full_weaker_than_naive**

Medication correlation reduction: 2.2%
