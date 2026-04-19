# Exploratory: Disease-Specific Coupling Modification — ELSA

Cross-cohort replication of the InCHIANTI exploratory coupling tests
([`../disease_coupling/`](../disease_coupling/)) using ELSA's 4 nurse
waves (2, 4, 6, 8).

## Architecture

Self-contained; does not import from `src/hdr_sim`. Medication flags
come from the raw ELSA core-wave .tab files (not shipped), reduced to a
small parquet by [`extract_meds_from_raw.py`](extract_meds_from_raw.py)
which runs once against `~/Downloads/ELISA Study 5050_V1.zip`.

```
exploratory/disease_coupling_elsa/
  _utils.py               # Panel loading + cross-lagged regression
  extract_meds_from_raw.py# One-shot raw→parquet medication extractor
  test_a_statins.py       # Test A: prescribed statins and I→M
  test_b_metformin.py     # Test B: antidiabetics and D-vs-J
  test_c_exercise.py      # Test C: grip tertile as coupling modifier
  test_d_nsaid.py         # Test D: NSAIDs/aspirin proxies and I→ row
  run_all.py              # Runs A–D and writes results/summary.json
  results/                # JSON outputs
  figures/                # PDF outputs
```

Run everything: `python exploratory/disease_coupling_elsa/run_all.py`

### Data dependencies

- `data/elsa/elsa_nurse_biomarkers_consolidated.tab`
- `data/elsa/elsa_supplementary_variables.tab`
- `data/elsa/gh_elsa_h_hdr_subset.tab`
- `data/elsa/elsa_med_flags_waves_2_4_6_8.parquet`
  (produced by `extract_meds_from_raw.py` from the raw UKDA-5050 zip)

### Axes

| Axis | Biomarkers |
|------|------------|
| I | log(CRP) |
| M | z-mean of HbA1c, chol/HDL ratio, log(triglycerides) |
| N | z-mean of SBP, DBP, pulse |
| F | grip strength (sign-flipped — walking speed not in consolidated file) |

Reference group for z-scoring: age 50–55 at wave 2 (N≈966). ELSA has no
age-20–30 reference group (cohort starts at age 50).

### Medication flags

Extracted from raw core files via `extract_meds_from_raw.py`:

| Flag | Source variable | Available waves | N_users (range/wave) |
|------|-----------------|-----------------|----------------------|
| `med_statin` | `hechmd` (prescribed cholesterol meds) | 4, 6, 8 | 1976–2405 |
| `med_statin_otc` | `statins` (OTC only) | 6, 8 | 51–98 |
| `med_antidm` | `hemdb` | 2, 4, 6, 8 | 474–779 |
| `med_antihtn` | `hemda` | 2, 4, 6, 8 | 1372–3345 |
| `med_nsaid` | `hepmed` (OA pain meds, proxy) | 4, 6 | 448–480 |
| `med_aspirin` | `hehrtmd` (blood-thinners) | 4 | 1293 |
| `med_bisphos` | `heostec` (osteoporosis meds) | 2, 6, 8 | 341–398 |
| `med_insulin` | `heins` | 2, 4, 6, 8 | 156–244 |

## Results summary

| Test | Prediction | Outcome | Key numbers |
|------|-----------|---------|-------------|
| **A — Statins → I→M** | Statin users show weaker β_{I→M} | **Opposite direction, but NS after adjustment.** | β_nostatin=+0.47, β_statin=+0.94 (unadjusted — confounded by indication). Interaction p=0.11 (unadj) / 0.15 (age-matched). Bonferroni p=0.32. N_statin=1575 lag-pairs. |
| **B — Antidiabetics D-vs-J** | M→I weakens; I→M unchanged | **Not supported** (both interactions NS). | I→M interaction p=0.13; M→I interaction p=0.70. |
| **C — Grip tertile → all pairs** | Pathological couplings weaker in high-grip; F→I more protective | **Directionally robust; trend NS after correction.** | **10/12** couplings weaker in high-grip (InCHIANTI: 9/12). F→I stronger-protective = YES. Raw trend p<0.05 for M→I (p=0.039) and I→N (p=0.019); none survive Bonferroni. |
| **D — NSAIDs → I→ row** | I→M, I→N, I→F attenuated | **Not supported with either proxy.** | OA-pain-med proxy (`hepmed`, N≈300 lag-pairs): all Bonferroni p=1.00. Aspirin proxy (`hehrtmd`, wave-4 only, N≈430 pairs): I→F raw p=0.038 (aspirin users have STRONGER I→F, opposite of prediction), Bonferroni p=0.11. |
| **E — Bone coupling** | F→B protective, I→B inflammatory | **Skipped: no PTH in ELSA.** | — |

## Cross-cohort consistency with InCHIANTI

| Test | InCHIANTI | ELSA | Consistent? |
|------|-----------|------|-------------|
| A (Statins → I→M) | β_statin > β_nostatin, interaction p=0.75 | β_statin > β_nostatin, interaction p=0.11 | Both NS; ELSA closer to significance but still null |
| B (Antidiabetics D-vs-J) | Both interactions NS | Both interactions NS | Consistent |
| C (F-axis tertile, N pairs weaker in high-F) | 9/12 | 10/12 | **Consistent directional pattern, ELSA slightly stronger** |
| D (NSAIDs → I→ row) | All NS (Bonferroni) | All NS (Bonferroni) | Consistent |

**The single directional finding reinforced by ELSA is Test C:** in both
cohorts, high-function participants (SPPB tertile in InCHIANTI, grip
tertile in ELSA) show uniformly weaker pathological cross-axis coupling
(9/12 and 10/12 respectively). Neither cohort's trend p-values survive
Bonferroni correction, but the direction-of-effect concordance across
12 independent pairs is striking.

Test A, B, D null findings **replicate** the InCHIANTI nulls at greater
statistical power (ELSA prescribed-statin N ≈ 20× the InCHIANTI statin
N), reinforcing the conclusion that the HDR coupling-modification
prediction does not find robust support for drug-specific interventions
in these observational cohorts.

## Caveats

- Confounding-by-indication is severe for statins, antidiabetics, and
  NSAIDs. Statin users in ELSA have higher baseline Δ_I, Δ_M, and Δ_N
  even within age-matched pairs. A full IV or target-trial emulation is
  beyond the scope of this exploratory file.
- `med_nsaid` = `hepmed` restricts to OA patients only (those asked the
  question), biasing the control group toward people without chronic
  pain. This reduces the NSAID interpretation.
- ELSA's grip-strength tertile is a weaker physical-function proxy than
  InCHIANTI's SPPB (no chair-stand or balance components).
- ELSA's F-axis composite excludes walking speed (not in consolidated
  biomarker file). Grip strength alone is the operational F measure
  here, which may underpower F→* couplings.
