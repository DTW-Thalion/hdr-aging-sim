## Supplementary: P Statistic Simulation Validation

### Design

We validated the discrimination power of the primacy ratio P = C_norm / V_norm under realistic strong-coupling conditions (epsilon ~ 0.9-3.6) matching the HDR parameterization. Five degradation regimes were simulated (Pure D, 75D/25J, 50D/50J, 25D/75J, Pure J) with the total spectral-abscissa drift alpha(A) held constant across regimes via binary-search calibration. Each regime was tested under four confound conditions (clean, survivorship bias, medication compression, both) at two dimensionalities (3-axis and 4-axis).

Configuration: N = 5000 samples per stratum, 200 Monte Carlo runs per scenario, 7 age strata (50-55 to 80-85), Q = I (identity diffusion).

### Reference Alpha Trajectory

| Age | alpha(A) |
|-----|----------|
| 52 | -0.2424 |
| 58 | -0.2263 |
| 62 | -0.2119 |
| 68 | -0.1990 |
| 72 | -0.1874 |
| 78 | -0.1769 |
| 82 | -0.1640 |

### Results: P-Slope by Regime and Confound (3-axis)

| Regime | Clean | Survivorship | Medication | Both |
|--------|-------|--------------|------------|------|
| Pure D | -0.0035 +/- 0.0012 | 0.0026 +/- 0.0018 | 0.0017 +/- 0.0017 | 0.0075 +/- 0.0022 |
| 75D/25J | -0.0035 +/- 0.0013 | 0.0023 +/- 0.0019 | 0.0015 +/- 0.0015 | 0.0074 +/- 0.0021 |
| 50D/50J | -0.0035 +/- 0.0012 | 0.0022 +/- 0.0018 | 0.0014 +/- 0.0014 | 0.0076 +/- 0.0021 |
| 25D/75J | 0.0005 +/- 0.0021 | 0.0053 +/- 0.0029 | 0.0064 +/- 0.0026 | 0.0110 +/- 0.0038 |
| Pure J | 0.0001 +/- 0.0031 | 0.0051 +/- 0.0048 | 0.0057 +/- 0.0040 | 0.0102 +/- 0.0057 |

### Power Analysis

| Pair | Clean | Both |
|------|-------|------|
| Pure D vs 75D/25J | 0.047 | 0.051 |
| 75D/25J vs 50D/50J | 0.047 | 0.081 |
| 50D/50J vs 25D/75J | 1.000 | 1.000 |
| 25D/75J vs Pure J | 0.146 | 0.139 |

### Minimum Detectable Effect

- 3-axis, 50/50 + Both confounds: MDE = 0.0004/yr
- 4-axis, 50/50 + Both confounds: MDE = 0.0006/yr
- ELSA observed P-slope: +0.0014/yr

### Monotone Ordering

- 3-axis, Clean: Broken
- 3-axis, Both:  Broken
- 4-axis, Clean: Broken
- 4-axis, Both:  Broken

### Conclusion

The simulation study reveals limitations of the primacy ratio P under strong coupling. Further investigation is needed.