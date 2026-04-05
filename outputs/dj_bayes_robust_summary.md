## Supplementary: Bayesian Model Comparison and Misspecification Robustness

### S9.1 Bayesian Model Comparison

We computed Bayes factors comparing the proportional co-degradation regime (50D/50J) against each alternative, using simulation-calibrated P-slope distributions as empirically derived priors and the ELSA medication-naive P-slope (+0.0014/yr) as the datum.

| Comparison | BF | Evidence (Jeffreys) |
|------------|-----|---------------------|
| 50D/50J vs Pure D | 4.7 | substantial |
| 50D/50J vs 75D/25J | 1.4 | anecdotal |
| 50D/50J vs 25D/75J | 3.6 | substantial |
| 50D/50J vs Pure J | 386.2 | decisive |

**Posterior model probabilities** (uniform prior across 5 regimes):

| Regime | P(regime \| data) |
|--------|-------------------|
| Pure D | 0.0977 |
| 75D/25J | 0.3186 |
| 50D/50J | 0.4560 |
| 25D/75J | 0.1265 |
| Pure J | 0.0012 |

The 50D/50J regime receives the highest posterior probability (0.456). 
The Bayes factor comparing proportional co-degradation to Pure D is 4.7, providing substantial evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to 75D/25J is 1.4, providing anecdotal evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to 25D/75J is 3.6, providing substantial evidence in favor of proportional co-degradation (Jeffreys scale).
The Bayes factor comparing proportional co-degradation to Pure J is 386.2, providing decisive evidence in favor of proportional co-degradation (Jeffreys scale).

### S9.2 TOST Equivalence Test

The equivalence bound was set to delta = 0.0036/yr, half the distance between the proportional regime mean (0.0054/yr) and the nearest adjacent regime. The equivalence region is [0.0019, 0.0090]/yr.

- Lower bound test: t = -0.295, p = 0.6100
- Upper bound test: t = 4.554, p = 0.0030
- TOST p = 0.6100

The TOST equivalence test does not reject at alpha = 0.05. The observed P-slope cannot be positively declared equivalent to the proportional regime at this sample size, though the Bayesian analysis provides complementary evidence.

### S9.3 Misspecification Robustness

Three misspecification scenarios were tested:

- **M1 (Correlated noise)**: Q with off-diagonal rho = 0.3 between all axes
- **M2 (Mild nonlinearity)**: 10% quadratic correction to the linear drift (Euler-Maruyama simulation, reduced to 100 MC runs and 2,000 samples)
- **M3 (Latent omitted axis)**: Data generated from 4-axis model, P computed from 3-axis (I, M, F) projection

| Scenario | Monotone | Min power | Max power |
|----------|----------|-----------|-----------|
| M1 | No | 0.838 | 1.000 |
| M2 | Yes | 0.336 | 1.000 |
| M3 | Yes | 0.083 | 1.000 |

**Per-scenario P-slopes:**

*M1*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | 0.0028 | [-0.0015, 0.0072] |
| 75D/25J | 0.0063 | [0.0011, 0.0123] |
| 50D/50J | 0.0032 | [-0.0000, 0.0071] |
| 25D/75J | 0.0046 | [0.0001, 0.0096] |
| Pure J | 0.0159 | [0.0100, 0.0227] |

*M2*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | -0.0023 | [-0.0067, 0.0042] |
| 75D/25J | -0.0011 | [-0.0067, 0.0054] |
| 50D/50J | 0.0109 | [0.0027, 0.0185] |
| 25D/75J | 0.0135 | [-0.0008, 0.0370] |
| Pure J | 0.0360 | [0.0099, 0.0817] |

*M3*:

| Regime | P-slope mean | 95% CI |
|--------|-------------|--------|
| Pure D | -0.0029 | [-0.0062, 0.0003] |
| 75D/25J | -0.0026 | [-0.0062, 0.0013] |
| 50D/50J | 0.0052 | [0.0012, 0.0100] |
| 25D/75J | 0.0129 | [0.0040, 0.0256] |
| Pure J | 0.0308 | [0.0160, 0.0484] |

Under misspecification scenario M1, the P statistic does not preserve monotone ordering and maintains discrimination power >= 0.80 between adjacent regimes (minimum power = 0.838).
Under misspecification scenario M2, the P statistic preserves monotone ordering and does not maintain discrimination power >= 0.80 between adjacent regimes (minimum power = 0.336).
Under misspecification scenario M3, the P statistic preserves monotone ordering and does not maintain discrimination power >= 0.80 between adjacent regimes (minimum power = 0.083).

### Bottom Line

The proportional co-degradation interpretation is supported by Bayesian model comparison with at least anecdotal evidence (minimum BF = 1.4) and TOST equivalence is not confirmed (p = 0.6100). The P statistic is partially robust to violations of the OU model assumptions tested here (correlated noise, mild nonlinearity, latent axis omission).
 Discrimination is weakened under: correlated noise.