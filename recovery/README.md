# HDR Recovery Dynamics Analysis

Tests the core HDR (Homeodynamic Regulation) prediction that recovery
timescales (τ) lengthen with age, by fitting exponential recovery curves to
serial in-hospital biomarkers across four physiological axes (I, M, N, renal).

This is the first analysis in the HDR program that targets the
**perturbation-recovery** mechanism directly, rather than secular drift in
between-visit covariance. Acute hospitalisation is a measurable multi-axis
perturbation; serial labs/vitals trace the recovery trajectory at hourly to
daily resolution.

## Datasets

- **MIMIC-IV** (primary): ~94K ICU stays at BIDMC (2008–2019). Free with
  PhysioNet credentialing + CITI training. https://physionet.org/content/mimiciv/
- **ISARIC** (optional): 945K hospitalised COVID patients across 76 countries.
  Access via IDDO: https://www.iddo.org/covid19/data-sharing/accessing-data

## Pipeline

| Step | Script | What it does |
|------|--------|---|
| 1  | `step1_extract_mimic.py`    | Extract & harmonise MIMIC-IV into a time-indexed biomarker panel |
| 1b | `step1b_extract_isaric.py`  | Same for ISARIC (optional) |
| 2  | `step2_define_episodes.py`  | Identify peak-and-recovery episodes per admission |
| 3  | `step3_fit_recovery.py`     | Fit exponential recovery curves; estimate τ̂ per axis per episode |
| 4  | `step4_age_dependence.py`   | Primary test: τ̂ vs age, per axis and multivariate |
| 5  | `step5_cross_axis_coupling.py` | Peak-to-τ̂ cross-lagged regression; daily co-recovery |
| 6  | `step6_predict_outcomes.py` | Mortality / LOS prediction: τ̂ vs admission biomarkers vs SOFA |
| 7  | `step7_figures.py`          | Publication figures |
| ―  | `run_all.py`                | Master driver |

`hdr_core.py` bundles shared utilities (axis definitions, exponential model,
bootstrap, age stratification).

## Configuration

Edit `config.yaml`:
- Set `mimic_dir` to the local MIMIC-IV root (or `isaric_file` for ISARIC).
- Adjust biomarker `itemid`s if a derivative MIMIC release uses different IDs.
- Tune `episodes.min_measurements_per_axis` (default 4) for a stricter cohort.

## Running

```bash
pip install -r requirements.txt
python run_all.py --config config.yaml
```

Each step writes intermediate parquet/json into `results/`. Steps after 1
read those intermediates; you can rerun any step independently.

## Output

```
results/
├── mimic_biomarker_panel.parquet     # step 1
├── recovery_episodes.parquet         # step 2
├── recovery_fits.parquet             # step 3
├── recovery_fit_summary.json         # step 3
├── tau_vs_age.json                   # step 4
├── cross_axis_coupling.json          # step 5
├── outcome_prediction.json           # step 6
├── figure_*.pdf                      # step 7
└── recovery_analysis_summary.md      # run_all
```

## Memory note

`labevents.csv.gz` (~6 GB compressed) and `chartevents.csv.gz` (much larger)
are streamed in chunks and filtered by `itemid` on read — never loaded into
RAM in full. Plan for ~32 GB RAM and ~50 GB free disk for intermediates.

## What this analysis would prove

| Result | Implication |
|---|---|
| τ̂ slope vs age β > 0 (significant) | D-degradation (recovery slowing) validated mechanistically, not just statistically |
| Cross-axis peak→τ̂ signs match J-matrix | Coupling matrix captures real perturbation-response physiology |
| ΔC(M3 − M2) > 0, ΔC(M7 − M6) > 0    | Recovery dynamics carry prognostic info beyond admission severity |
