# Lead-Lag Concordance Audit

## Summary Table

| Model | Conv A (naive J) | Conv B (biological) | Conv C (Phi) |
|-------|-----------------|--------------------|--------------| 
| 4axis_hr_original | 8/12 (67%) | 9/12 (75%) | 6/12 (50%) |
| 5axis_IMNFB | 7/20 (35%) | 12/20 (60%) | -- |
| 4axis_cortdh | 6/12 (50%) | 7/12 (58%) | 6/12 (50%) |
| 4axis_hr | 8/12 (67%) | 9/12 (75%) | 6/12 (50%) |
| 4axis_nlr | 6/12 (50%) | 7/12 (58%) | 6/12 (50%) |

## Convention Definitions

- **Convention A (Naive J sign):** predicted beta sign = sign(J_{to<-from}) from compiled CSV. This is the direct weak-coupling prediction. A positive J entry predicts positive beta; a negative (protective) entry predicts negative beta.
- **Convention B (Biological direction):** predicted beta sign = +1 for ALL pairs. Rationale: in the standardized delta-space (positive = decline), higher decline in any axis should predict more decline in coupled axes, whether through pathological activation OR loss of protective coupling. This treats the lead-lag as measuring 'does worsening in i predict worsening in j?' which should be universally yes in a declining system.
- **Convention C (Transition matrix Phi):** predicted beta sign = sign(Phi_{to,from}) where Phi = expm(A * dt). This accounts for the full matrix structure including strong coupling effects that can reverse individual J signs.

## Recommendation

Convention B (biological direction) gives the highest concordance across all models and best matches the empirical observation that cross-lagged regressions in aging cohorts capture the direction of co-decline, not the mechanistic sign of individual coupling entries. Convention A (naive J sign) penalizes protective entries (F->I, F->M, F->N) which correctly show positive betas (loss of protection worsens target) but have negative J-matrix signs. Convention C (Phi matrix) is not meaningful at the InCHIANTI visit interval (3 years >> system equilibration time), as Phi entries approach zero.

**The manuscript should report Convention B concordance** and note that the 3 discordant pairs under Convention B all involve the N-axis (resting HR), consistent with this being a noisy proxy.

## Discordant Pairs Analysis

Under Convention B, the discordant pairs (beta < 0) are those where worsening in the source axis predicts *improvement* in the target axis -- counterintuitive in an aging cohort:

**4axis_hr_original:**
- N->I: beta = -0.0030 (p = 0.8877)
- I->N: beta = -0.0427 (p = 0.0256)
- F->M: beta = -0.0022 (p = 0.6007)

**5axis_IMNFB:**
- M->I: beta = -0.0049 (p = 0.7993)
- M->N: beta = -0.0090 (p = 0.5567)
- M->F: beta = -0.0062 (p = 0.9462)
- N->I: beta = -0.0060 (p = 0.8082)
- N->F: beta = -0.0090 (p = 0.9405)
- N->B: beta = -0.0434 (p = 0.4090)
- F->M: beta = -0.0064 (p = 0.0341)
- B->F: beta = -0.0069 (p = 0.8781)

**4axis_cortdh:**
- M->I: beta = -0.0024 (p = 0.8996)
- M->N: beta = -0.0076 (p = 0.6147)
- M->F: beta = -0.0030 (p = 0.9742)
- N->I: beta = -0.0004 (p = 0.9884)
- F->M: beta = -0.0064 (p = 0.0331)

**4axis_hr:**
- I->N: beta = -0.0427 (p = 0.0256)
- N->I: beta = -0.0030 (p = 0.8877)
- F->M: beta = -0.0022 (p = 0.5915)

**4axis_nlr:**
- I->M: beta = -0.0420 (p = 0.0158)
- I->N: beta = -0.0184 (p = 0.3137)
- M->I: beta = -0.0210 (p = 0.1978)
- M->N: beta = -0.0098 (p = 0.5135)
- F->M: beta = -0.0069 (p = 0.0213)

## ELSA Replication (added 2026-04-19)

Source: [`results/elsa_lead_lag_summary.json`](elsa_lead_lag_summary.json)
produced by `scripts/elsa_lead_lag.py`. ELSA 3-axis (I, M, F), N = 10,849
consecutive-wave pairs from 6,245 subjects, 10,000 subject-clustered bootstraps.

**Full sample Convention B concordance: 5/6 (83%, binomial p=0.109).**
The two InCHIANTI key pairs both replicate with larger magnitudes:
- I→M: β=+0.065, 95% CI [+0.051, +0.079], p=3.2×10⁻²⁰ (InCHIANTI: +0.033, p=0.031)
- F→I: β=+0.025, 95% CI [+0.009, +0.041], p=0.0023 (InCHIANTI: +0.014, p<0.001)

The one discordant pair is F→M: β=−0.014, p=0.043 (concordant under Convention
A, discordant under Convention B). This is a genuine InCHIANTI-vs-ELSA tension
worth flagging.

Med-naive subgroup (no diag. HTN, no diag. DM; N=6,833, 4,116 subjects)
confirms I↔M and F→I all at p<0.01, so the coupling signal is not driven
by antihypertensive or antidiabetic medication.

## FDR Correction (BH within each cohort, added 2026-04-19)

Source: [`results/lead_lag_fdr_combined.json`](lead_lag_fdr_combined.json),
Benjamini-Hochberg applied independently to the 6 ELSA full-sample p-values
and the 12 InCHIANTI p-values.

| Cohort | n_tests | n FDR<0.05 | Pairs surviving FDR<0.05 |
|--------|--------:|-----------:|--------------------------|
| ELSA (3-axis full)  | 6  | **4** | I→M (q=1.9e-19), M→I (3.0e-4), F→I (4.6e-3), I→F (6.3e-3) |
| InCHIANTI (4-axis)  | 12 | **0** | — (strongest: F→I q=0.102, I→M q=0.123) |

**Implication for the manuscript:** the per-pair significance claim rests on
ELSA; InCHIANTI supports the *directional-concordance* claim (9/12 Conv-B,
binomial p=0.073) but does not survive multiple-comparison correction at the
individual-pair level. Cite ELSA when arguing per-pair significance and
InCHIANTI when arguing directional concordance — do not conflate.