# Exploratory: Disease-Specific Coupling Modification Tests

Self-contained exploratory analyses testing HDR predictions about how
pharmaceutical interventions and endogenous F-axis state modify cross-axis
coupling strength. Results inform whether a standalone paper on
disease-specific coupling modification is warranted, or whether selected
findings integrate into the main manuscript.

## Architecture

Every script in this folder imports only from `_utils.py` (local) and
standard scientific-Python libraries. No imports from `src/hdr_sim`.
The entire folder can be extracted into a standalone repo without
dependency surgery.

```
exploratory/disease_coupling/
  _utils.py              # Panel loading + cross-lagged regression (OLS/HC3)
  test_a_statins.py      # Test A: Statins and I→M
  test_b_metformin.py    # Test B: Antidiabetics and D-vs-J discrimination
  test_c_exercise.py     # Test C: SPPB tertile as coupling modifier
  test_d_nsaid.py        # Test D: NSAIDs and I→* row
  test_e_bone.py         # Test E: Bone-axis (5-axis) coupling
  run_all.py             # Runs A–E and writes results/summary.json
  results/               # JSON outputs
  figures/               # PDF outputs
```

Run everything: `python exploratory/disease_coupling/run_all.py`

Data dependency: `data/inchianti_panel.parquet` (already built by the
main pipeline). Test E additionally reads PTH from the raw SAS files at
`~/Downloads/inCHIANTI/InCHIANTI_CD_Share` (waves 0–3).

## Method notes

- **Cross-lagged regression.** For each consecutive wave pair (w, w+1):
  `Δ_target(w+1) ~ β·Δ_source(w) + γ·Δ_target(w) + δ·age_w` (+ group
  interaction where relevant). OLS with HC3 robust SEs.
- **Sign convention.** All axis deltas are z-scored so *positive = decline*
  (SPPB is sign-flipped for this reason). Every HDR-predicted
  pathological coupling therefore has β > 0 (Convention B).
- **Age matching.** Nearest-neighbor 1:1 within ±5 y, without replacement.
- **Interaction test.** A single regression with `Δ_source × group`
  interaction term; reported alongside the per-group fits.
- **Multiple testing.** Raw p and Bonferroni-corrected p reported within
  each test. No cross-test correction.
- **Minimum N.** Any group with fewer than 30 lag-pairs is skipped.

## Results summary

| Test | Prediction | Outcome | Key numbers |
|------|-----------|---------|-------------|
| **A — Statins → I→M** | Statin users show weaker β_{I→M} | **Not supported.** | β_nostatin=+0.024, β_statin=+0.050, interaction p=0.75 (unadj) / 0.52 (matched). Bonferroni p=1.00. N_statin=110 pairs. |
| **B — Antidiabetics D-vs-J** | M→I weakens; I→M unchanged | **Not supported** (low power, both NS). | I→M interaction p=0.93; M→I interaction p=0.35. N_antidm=111 (I→M) / 154 (M→I) pairs. |
| **C — SPPB tertile → all pairs** | Pathological couplings weaker in high-SPPB; F→I more protective | **Directionally consistent; not significant.** | 9/12 couplings weaker in high-SPPB. F→I stronger-protective = YES. 0/12 trend tests survive Bonferroni. Raw trend p<0.05 only for I→F (p=0.026). |
| **D — NSAIDs → I→ row** | I→M, I→N, I→F all attenuate | **Not supported by interaction test.** | Bonferroni p = [0.24, 1.00, 1.00]. Marginal sign flip on I→M (β=−0.07 vs +0.03, raw interaction p=0.08) — descriptive only. |
| **E — Bone coupling (5-axis)** | F→B protective, I→B inflammatory | **F→B supported; I→B NS.** | F→B β=+0.025, p=0.003 (Bonferroni p=0.007, N=2409). I→B β=+0.046, p=0.20 (NS). Bisphosphonate N=38 pairs → interaction underpowered. |

The single robust positive finding is **Test E F→B coupling**: a full-sample
significant cross-lagged association in the predicted direction (SPPB at
wave w → PTH at wave w+1), survivng Bonferroni correction across the two
primary 5-axis bone tests. This validates the 5-axis HDR extension on the
F→B arm. I→B is directionally consistent but under-powered.

Tests A, B, and D are all limited by small user-group sizes (N_users ≈
100–250 lag-pairs) and substantial confounding-by-indication. None yield
clean evidence for drug-specific coupling modification.

Test C's descriptive pattern (9/12 couplings weaker in high-SPPB, F→I
stronger-protective) aligns with the HDR prediction that the F→ row is
predominantly protective, but none of the 12 individual trend tests
survives multiple-testing correction. SPPB's ceiling effect also drives
some degenerate tertiles (notably the F→B high-SPPB fit in Test E, where
the high-SPPB group is almost entirely at SPPB=12 and Δ_F ≈ 0).

## Recommendation

- **Test E F→B** is publication-quality on its own and supports the
  5-axis extension currently sketched in the main manuscript.
- **Tests A–D** do not rise to the bar for a standalone coupling-modification
  paper as currently powered. A larger cohort (e.g. ELSA for statin
  users; NHANES for NSAID users) would be needed to raise N per treatment
  group above ~500 lag-pairs before these tests are informative.
- **Test C**'s descriptive consistency (9/12 weaker + F→I stronger-protective)
  could be mentioned as exploratory support for the F-axis protective
  role, but should not be framed as confirmed.

## Reproducibility

- InCHIANTI data root: `~/Downloads/inCHIANTI/InCHIANTI_CD_Share` (unchanged
  from the main pipeline).
- Youthful reference: healthy subjects aged 20–30 at baseline (N≈50–55
  per axis after dx exclusion); SPPB reference widens to healthy <60 to
  sidestep the SPPB=12 ceiling.
- Random seed for age-matching: 0.
- All outputs are deterministic given the parquet panel and raw SAS files.
