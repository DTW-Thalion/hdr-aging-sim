# Tau Registry Comparison: Legacy vs V2 (Literature-Calibrated)

## Recovery Time Constants (days)

| Axis | Legacy tau(30) | Legacy tau(80) | V2 tau(25) | V2 tau(80) | V2 tau(120) | Trajectory | PMID |
|------|---------------|---------------|-----------|-----------|------------|------------|------|
| I    |       7.000 |      25.000 |   4.000 |  17.000 |   45.000 | gompertz   | 27467771 |
| M    |       0.100 |       0.300 |   0.080 |   0.210 |    0.350 | piecewise-linear | 18268070 |
| E    |    1000.000 |    1500.000 | 500.000 | 2000.000 | 5000.000 | piecewise-exp | 15509558 |
| mito |       1.000 |       3.000 |  36.000 |  57.000 |   65.000 | saturating-exp | 8986817 |
| P    |       0.500 |       2.000 |   1.500 |   3.000 |    4.000 | piecewise-linear | 24437518 |
| C    |       1.000 |       3.000 |   6.000 |  10.000 |   18.000 | piecewise-linear | 1557592 |
| N    |       0.010 |       0.040 |   0.003 |   0.005 |    0.008 | piecewise-linear | 29581219 |
| F    |       8.000 |      42.000 |   2.000 |   3.500 |    6.000 | piecewise-gompertz | 9252485 |
| B    |      90.000 |     120.000 | 135.000 | 250.000 |  500.000 | piecewise-linear | 3213608 |

## Key Changes

| Axis | Change Factor (tau_25/tau_30) | Notes |
|------|------------------------------|-------|
| I    |                     0.57 | 0.57× change |
| M    |                     0.80 | ~0.80× (minor adjustment) |
| E    |                     0.50 | 0.50× change |
| mito |                    36.00 | **36× increase** — order-of-magnitude correction |
| P    |                     3.00 | 3.00× change |
| C    |                     6.00 | **6× increase** — order-of-magnitude correction |
| N    |                     0.30 | 0.30× change |
| F    |                     0.25 | **0.25× decrease** — order-of-magnitude correction |
| B    |                     1.50 | 1.50× change |

## Spectral Abscissa Trajectory (4-axis: I, M, N, F)

Legacy calibration scalar: c = 0.063892
V2 calibration scalar: c = 14.591633

| Age | alpha (Legacy) | alpha (V2) | Pyrkov Target | Legacy Stable? | V2 Stable? |
|-----|---------------|-----------|--------------|----------------|------------|
|  25 |   -0.1339 | -0.1340 |      -0.1340 | Yes            | Yes        |
|  30 |   -0.1339 | -0.0121 |      -0.1108 | Yes            | Yes        |
|  40 |   -0.0809 | 0.2632 |      -0.0758 | Yes            | NO         |
|  50 |   -0.0583 | 0.5928 |      -0.0518 | Yes            | NO         |
|  60 |   -0.0457 | 1.0042 |      -0.0354 | Yes            | NO         |
|  70 |   -0.0375 | 1.5434 |      -0.0242 | Yes            | NO         |
|  80 |   -0.0318 | 2.2828 |      -0.0166 | Yes            | NO         |
|  90 |   -0.0318 | 3.3903 |      -0.0113 | Yes            | NO         |
| 100 |   -0.0318 | 4.9782 |      -0.0078 | Yes            | NO         |
| 110 |   -0.0318 | 15.4393 |      -0.0053 | Yes            | NO         |
| 120 |   -0.0318 | 26.0737 |      -0.0036 | Yes            | NO         |

## J Matrix Fill Comparison

- Original (no imputation): 42/72 nonzero (58% fill)
- Imputed (tier defaults): 68/72 nonzero (94% fill)
- Unknown sign entries (remain 0): 4/72
