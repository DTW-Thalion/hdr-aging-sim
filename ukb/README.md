# HDR UK Biobank Analysis Package

## Documentation

Short answers are in this README. For depth, see [docs/](docs/):

- [SETUP.md](docs/SETUP.md) — detailed setup, data prep, first-run verification
- [FIELD_MAPPING.md](docs/FIELD_MAPPING.md) — canonical UKB field ID reference
- [METHODS.md](docs/METHODS.md) — statistical methods, formulas, references
- [INTERPRETATION.md](docs/INTERPRETATION.md) — how to read each output
- [OUTPUTS.md](docs/OUTPUTS.md) — catalogue of every file the pipeline writes
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — error-by-error fixes

## Purpose

This package runs the Homeodynamic Remediation (HDR) framework's empirical
validation pipeline on UK Biobank data, extending the InCHIANTI (4-axis,
N=1,453) and ELSA (3-axis, N=6,420) analyses to 5–6 axes with N≈500,000.
The pipeline reproduces every headline test from the HDR manuscript on a
larger cohort and adds two new axes (circadian, bone) that were not
estimable in InCHIANTI or ELSA.

HDR treats aging as erosion of a stability eigenvalue on a low-dimensional
physiological manifold. The key empirical signatures are:

- **λ_max of the change-covariance grows with age** (manifold softening)
- **Cross-lagged coefficients match compiled J-matrix signs** (directional coupling)
- **SWDS-Γ adds mortality-prediction value beyond biomarkers and frailty** (ΔC > 0)
- **Π = C_norm / V_norm decomposition shifts toward D-dominated / J-dominated regimes with age**

All four tests are run for each feasible axis tier.

## Prerequisites

- Python 3.9 or later
- UK Biobank data access (approved application) with baseline + imaging data
- Packages in `requirements.txt`
- (Optional) Raw accelerometry CWA files or pre-computed summary fields
- (Optional) DXA BMD fields from the imaging assessment

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit config.yaml to point to your data files and check field IDs
#    (see "Field ID verification" below for your export format).

# 3. Dry-run test on 1,000 participants to confirm field mapping
python step1_extract.py --test

# 4. Run the full pipeline
python run_all.py

# 5. Read the summary
cat results/ukb_hdr_summary.md
```

The full pipeline takes 2–4 hours on a modern workstation (Step 6 bootstrap is the bottleneck).

## What each step does

1. **`step1_extract.py`** — Reads the UK Biobank export, auto-detects its format
   (wide field-encoded vs. melted/long), resolves field IDs per-instance, applies
   transformations (log-CRP, log-cystatin, HbA1c unit detection, grip max), and
   writes a harmonised long-format panel (`eid × instance`) to Parquet.

2. **`step2_qc.py`** — Sample description: N by instance, biomarker completeness,
   distributions, outlier counts, age/sex/ethnicity breakdown, medication
   prevalence, mortality summary, longitudinal coverage per tier. Writes
   `ukb_qc_report.md` and `ukb_qc_stats.json`.

3. **`step3_cross_sectional.py`** — Age-stratified Γ̂ with bootstrap 95% CI for
   λ_max, per-axis variance, pairwise correlations, univariate control ratio,
   J-sign concordance, Π decomposition. With N≈500K this is dramatically more
   precise than InCHIANTI / ELSA.

4. **`step4_longitudinal.py`** — Within-person visit-pair change-covariance
   (baseline → imaging, ~4–14 yr interval) with age-stratified λ_max trajectory,
   Kendall τ trend test, medication-stratified analysis, Π decomposition. The
   core HDR test.

5. **`step5_lead_lag.py`** — Cross-lagged regression for every ordered axis pair.
   OLS β with subject-clustered bootstrap CIs, BH-FDR within tier, sign
   concordance against compiled J-matrix, subgroup analyses.

6. **`step6_mortality.py`** — Cox models M1–M5 (age+sex → frailty → biomarkers →
   SWDS-Γ → full) with matched complete-case sample, 2,000-resample bootstrap
   for ΔC CIs. Additional comparators (Mahalanobis, z-sum). Subgroup analyses by
   medication status, sex, age, ethnicity.

7. **`step7_null_tests.py`** — Age-permutation null for λ_max age trend,
   random-biomarker-panel null (500 panels sampled from ~25 non-HDR UKB
   biomarkers), axis-substitution null, univariate control. Replicates
   Supplementary Note 11 on larger data.

8. **`step8_circadian.py`** — (Optional) Computes circadian biomarker from UKB
   summary accelerometry fields (Tier 1) or raw CWA files (Tier 2). Produces
   interdaily stability (IS), intradaily variability (IV), relative amplitude
   (RA). Skipped if accelerometry is not configured.

## Expected runtime

| Step | Typical time (N=500K) |
|------|----------------------|
| 1 — extract          | 10 min |
| 2 — QC               | 5 min  |
| 3 — cross-sectional  | 5 min (per tier) |
| 4 — longitudinal     | 20 min (per tier) |
| 5 — lead-lag         | 10 min (per tier) |
| 6 — mortality        | 60 min (dominated by bootstrap) |
| 7 — null tests       | 15 min (per tier) |
| 8 — circadian        | 10 min (summary) – days (raw CWA) |

Run times scale roughly linearly in N for cross-sectional work and
quadratically for bootstrap steps.

## Output files

All outputs land in `results/` (configurable in `config.yaml`).

**Panel / QC**

- `ukb_panel_long.parquet` — harmonised long-format panel (one row per eid × instance)
- `ukb_panel_summary.json` — generation metadata, column dictionary
- `ukb_qc_report.md` — human-readable QC
- `ukb_qc_stats.json` — machine-readable QC

**Cross-sectional (per tier)**

- `ukb_cross_sectional_{tier}.json`
- `figure_ukb_cross_sectional_{tier}.pdf`

**Longitudinal (per tier)**

- `ukb_longitudinal_{tier}.json`
- `figure_ukb_longitudinal_{tier}.pdf`

**Lead-lag (per tier)**

- `ukb_lead_lag_{tier}.json`
- `figure_ukb_lead_lag_{tier}.pdf`

**Mortality (per tier)**

- `ukb_mortality_{tier}.json`
- `figure_ukb_mortality_{tier}.pdf`

**Null tests (per tier)**

- `ukb_null_tests_{tier}.json`
- `figure_ukb_null_tests_{tier}.pdf`

**Summary**

- `ukb_hdr_summary.md` — headline numbers, comparison table vs. InCHIANTI / ELSA

## Interpreting results

- **λ_max monotonic increase with age, p_trend < 0.001, age-permutation p < 0.05**
  confirms stability erosion (HDR's strongest prediction).

- **λ_max / max(σ²_i) > 1.2** indicates genuine multi-axis coupling contributes
  to the eigenvalue, not just a single-axis variance artefact. InCHIANTI shows
  ~1.0 here (F-axis dominated); UKB with a broader axis set should resolve this.

- **Sign concordance > 70% with binomial p < 0.05** replicates the directional
  predictions from the compiled J matrix.

- **ΔC(M5 − M4) ≥ 0.01** with a lower 95% CI bound clearly above 0 meets the
  pre-specified "clinically meaningful" threshold from the manuscript.

- **ΔC(M5 − M4a) > 0** is the strongest statement: SWDS-Γ carries information
  beyond raw biomarkers alone. In InCHIANTI, M4a ≈ M5 (both C = 0.760);
  UKB's statistical power should definitively resolve this.

- **Π shifting from D-dominated (young) toward J-dominated (old)** implies
  coupling is doing more of the variance-shaping as stability erodes.

- **Lead-lag FDR-significant with correct sign** for I→M, F→I replicates
  InCHIANTI / ELSA. Novel predictions unlocked by tier 3–4 (I→C, F→B, C→N,
  P→I) become testable for the first time.

## Troubleshooting

**Field ID naming conventions.** The config uses the bare UKB field ID (e.g.
`30710`). The extraction script searches for columns matching
`f.30710.0.0` / `p30710_i0` / `30710-0.0` and any common variant. If your
export uses a format we don't recognise, edit `step1_extract.py`:
`_resolve_field_column()`.

**Missing fields.** If a configured field is entirely absent, the script prints
a clear warning and drops the corresponding axis (higher tiers will be skipped
automatically). Do not silently rename fields in the config — edit the axis
entry to `null` if the field is unavailable.

**Memory.** UKB data with 500K participants × ~50 columns fits in ~16 GB RAM
as float64. If you are memory-constrained, set `chunk_size` in `config.yaml`
to 50000 and the extraction runs in chunks.

**HbA1c units.** The extraction auto-detects DCCT (%) vs IFCC (mmol/mol). If
the auto-detection is wrong (e.g. a tiny slice of data with different
conventions), force the unit by editing `_convert_hba1c` in
`step1_extract.py`.

**Withdrawal handling.** UKB ships a withdrawal list with each data refresh.
Set `withdrawal_path` in `config.yaml` to the current text file (one eid per
line) before running; extraction excludes those eids.

**Accelerometry.** If only UKB summary fields are available, step 8 computes a
circadian proxy from fields 90001–90060 (Tier 1). For raw CWA files, install
[`actipy`](https://github.com/OxWearables/actipy) or
[`accelerometer`](https://github.com/OxWearables/biobankAccelerometerAnalysis)
and set `accel_raw_dir`. Pre-processing 100K raw recordings takes days on a
single workstation; consider batch/HPC processing.

**DXA availability.** The imaging subset with DXA is ~18K participants. Tier
4 is underpowered if N < 5,000 for any subanalysis — the summary report flags
this automatically.

## Data privacy

This package **never writes individual-level data to the results directory**.
All outputs are aggregate statistics, bootstrap distributions, and figures
suitable for external sharing (within UKB's approved-collaborator rules).
The extracted panel (`ukb_panel_long.parquet`) is individual-level and stays
inside the UKB environment.

## Code quality

- Every public function has a docstring and type hints on its signature.
- Every output file has a header comment with generation date, script, and
  a SHA-1 hash of the effective config.
- `tqdm` progress bars on any loop > 100 iterations.
- `try/except` around each step in `run_all.py` — if one step fails, the
  others still run on whatever data is available.
- `np.random.default_rng(seed)` is used everywhere; no global seeding.
- Intermediate results are saved after each step so the pipeline can be
  resumed from the middle.

## Reference

Todd Whitney et al., *Homeodynamic Remediation: a 9-axis framework for aging*
(manuscript). This package is the companion UK Biobank validation suite.

Issues / questions: open an issue on the [hdr-aging-sim](https://github.com/DTW-Thalion/hdr-aging-sim) repository.
