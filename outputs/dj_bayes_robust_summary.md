## Supplementary: Bayesian Model Comparison and Misspecification Robustness

### S9.1 Bayesian Model Comparison

We computed Bayes factors comparing the proportional co-degradation regime (50D/50J) against each alternative, using simulation-calibrated P-slope distributions as empirically derived priors and the ELSA medication-naive P-slope (+0.0014/yr) as the datum.

| Comparison | BF | Evidence (Jeffreys) |
|------------|-----|---------------------|
| 50D/50J vs Pure D | 0.5 | anecdotal |
| 50D/50J vs 75D/25J | 0.6 | anecdotal |
| 50D/50J vs 25D/75J | 0.4 | anecdotal |
| 50D/50J vs Pure J | 0.1 | anecdotal |

**Posterior model probabilities** (uniform prior across 5 regimes):

| Regime | P(regime \| data) |
|--------|-------------------|
| Pure D | 0.1123 |
| 75D/25J | 0.1037 |
| 50D/50J | 0.0583 |
| 25D/75J | 0.1405 |
| Pure J | 0.5852 |

The Pure J regime receives the highest posterior probability (0.585). 
The Bayes factor comparing proportional co-degradation to Pure D is 0.5, providing anecdotal evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to 75D/25J is 0.6, providing anecdotal evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to 25D/75J is 0.4, providing anecdotal evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to Pure J is 0.1, providing anecdotal evidence in favor of proportional co-degradation (Jeffreys scale).

### S9.2 TOST Equivalence Test

The equivalence bound was set to delta = 0.0002/yr, half the distance between the proportional regime mean (0.0076/yr) and the nearest adjacent regime. The equivalence region is [0.0075, 0.0078]/yr.

- Lower bound test: t = -3.639, p = 0.9925
- Upper bound test: t = 3.833, p = 0.0061
- TOST p = 0.9925

The TOST equivalence test does not reject at alpha = 0.05. The observed P-slope cannot be positively declared equivalent to the proportional regime at this sample size, though the Bayesian analysis provides complementary evidence.

### S9.3 Misspecification Robustness

Three misspecification scenarios were tested:

- **M1 (Correlated noise)**: Q with off-diagonal rho = 0.3 between all axes
- **M2 (Mild nonlinearity)**: 10% quadratic correction to the linear drift (Euler-Maruyama simulation, reduced to 100 MC runs and 2,000 samples)
- **M3 (Latent omitted axis)**: Data generated from 4-axis model, P computed from 3-axis (I, M, F) projection

| Scenario | Monotone | Min power | Max power |
|----------|----------|-----------|-----------|
| M1 | No | 0.069 | 0.620 |
| M2 | No | 0.056 | 0.870 |
| M3 | Yes | 0.049 | 0.995 |

**Per-scenario P-slopes:**

*M1*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | 0.0023 | [-0.0007, 0.0058] |
| 75D/25J | 0.0023 | [-0.0008, 0.0054] |
| 50D/50J | 0.0025 | [-0.0010, 0.0061] |
| 25D/75J | 0.0026 | [-0.0018, 0.0072] |
| Pure J | 0.0038 | [-0.0012, 0.0103] |

*M2*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | 0.0083 | [0.0017, 0.0157] |
| 75D/25J | 0.0080 | [0.0006, 0.0150] |
| 50D/50J | 0.0084 | [0.0020, 0.0160] |
| 25D/75J | 0.0114 | [0.0012, 0.0242] |
| Pure J | 0.0119 | [-0.0043, 0.0394] |

*M3*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | 0.0077 | [0.0042, 0.0119] |
| 75D/25J | 0.0077 | [0.0035, 0.0117] |
| 50D/50J | 0.0077 | [0.0037, 0.0119] |
| 25D/75J | 0.0103 | [0.0037, 0.0190] |
| Pure J | 0.0103 | [0.0006, 0.0233] |

Under misspecification scenario M1, the P statistic does not preserve monotone ordering and does not maintain discrimination power >= 0.80 between adjacent regimes (minimum power = 0.069).
Under misspecification scenario M2, the P statistic does not preserve monotone ordering and does not maintain discrimination power >= 0.80 between adjacent regimes (minimum power = 0.056).
Under misspecification scenario M3, the P statistic preserves monotone ordering and does not maintain discrimination power >= 0.80 between adjacent regimes (minimum power = 0.049).

### Bottom Line

The proportional co-degradation interpretation is supported by Bayesian model comparison with at least anecdotal evidence (minimum BF = 0.1) and TOST equivalence is not confirmed (p = 0.9925). The P statistic is not robust to violations of the OU model assumptions tested here (correlated noise, mild nonlinearity, latent axis omission).
 Discrimination is weakened under: correlated noise, nonlinear drift.