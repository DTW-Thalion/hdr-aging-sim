## Supplementary: Bayesian Model Comparison and Misspecification Robustness

### S9.1 Bayesian Model Comparison

We computed Bayes factors comparing the proportional co-degradation regime (50D/50J) against each alternative, using simulation-calibrated P-slope distributions as empirically derived priors and the ELSA medication-naive P-slope (+0.0014/yr) as the datum.

| Comparison | BF | Evidence (Jeffreys) |
|------------|-----|---------------------|
| 50D/50J vs Pure D | 5.9 | substantial |
| 50D/50J vs 75D/25J | 1.5 | anecdotal |
| 50D/50J vs 25D/75J | 3.8 | substantial |
| 50D/50J vs Pure J | 428.6 | decisive |

**Posterior model probabilities** (uniform prior across 5 regimes):

| Regime | P(regime \| data) |
|--------|-------------------|
| Pure D | 0.0807 |
| 75D/25J | 0.3200 |
| 50D/50J | 0.4743 |
| 25D/75J | 0.1238 |
| Pure J | 0.0011 |

The 50D/50J regime receives the highest posterior probability (0.474). 
The Bayes factor comparing proportional co-degradation to Pure D is 5.9, providing substantial evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to 75D/25J is 1.5, providing anecdotal evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to 25D/75J is 3.8, providing substantial evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to Pure J is 428.6, providing decisive evidence in favor of proportional co-degradation (Jeffreys scale).

### S9.2 TOST Equivalence Test

The equivalence bound was set to delta = 0.0036/yr, half the distance between the proportional regime mean (0.0054/yr) and the nearest adjacent regime. The equivalence region is [0.0018, 0.0090]/yr.

- Lower bound test: t = -0.265, p = 0.5992
- Upper bound test: t = 4.555, p = 0.0030
- TOST p = 0.5992

The TOST equivalence test does not reject at alpha = 0.05. The observed P-slope cannot be positively declared equivalent to the proportional regime at this sample size, though the Bayesian analysis provides complementary evidence.

### S9.3 Misspecification Robustness

Three misspecification scenarios were tested:

- **M1 (Correlated noise)**: Q with off-diagonal rho = 0.3 between all axes
- **M2 (Mild nonlinearity)**: 10% quadratic correction to the linear drift (Euler-Maruyama simulation, reduced to 100 MC runs and 2,000 samples)
- **M3 (Latent omitted axis)**: Data generated from 4-axis model, P computed from 3-axis (I, M, F) projection

| Scenario | Monotone | Min power | Max power |
|----------|----------|-----------|-----------|
| M1 | No | 0.835 | 1.000 |
| M2 | Yes | 0.370 | 1.000 |
| M3 | Yes | 0.085 | 1.000 |

**Per-scenario P-slopes:**

*M1*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | 0.0028 | [-0.0015, 0.0077] |
| 75D/25J | 0.0063 | [0.0015, 0.0116] |
| 50D/50J | 0.0032 | [-0.0001, 0.0071] |
| 25D/75J | 0.0045 | [0.0003, 0.0094] |
| Pure J | 0.0160 | [0.0099, 0.0223] |

*M2*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | -0.0023 | [-0.0078, 0.0043] |
| 75D/25J | -0.0012 | [-0.0071, 0.0053] |
| 50D/50J | 0.0108 | [0.0032, 0.0177] |
| 25D/75J | 0.0135 | [-0.0008, 0.0353] |
| Pure J | 0.0361 | [0.0098, 0.0838] |

*M3*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | -0.0029 | [-0.0060, 0.0001] |
| 75D/25J | -0.0027 | [-0.0059, 0.0012] |
| 50D/50J | 0.0052 | [0.0008, 0.0098] |
| 25D/75J | 0.0129 | [0.0043, 0.0256] |
| Pure J | 0.0307 | [0.0166, 0.0480] |

Under misspecification scenario M1, the P statistic does not preserve monotone ordering and maintains discrimination power >= 0.80 between adjacent regimes (minimum power = 0.835).
Under misspecification scenario M2, the P statistic preserves monotone ordering and does not maintain discrimination power >= 0.80 between adjacent regimes (minimum power = 0.370).
Under misspecification scenario M3, the P statistic preserves monotone ordering and does not maintain discrimination power >= 0.80 between adjacent regimes (minimum power = 0.085).

### Bottom Line

The proportional co-degradation interpretation is supported by Bayesian model comparison with at least anecdotal evidence (minimum BF = 1.5) and TOST equivalence is not confirmed (p = 0.5992). The P statistic is partially robust to violations of the OU model assumptions tested here (correlated noise, mild nonlinearity, latent axis omission).
 Discrimination is weakened under: correlated noise.