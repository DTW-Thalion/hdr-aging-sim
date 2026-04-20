# Outputs Catalogue

Every file the pipeline writes, where it comes from, and what its
fields mean. All files land under `config.yaml → output_dir` (default
`./results/`).

Files are grouped below by pipeline step.

---

## Step 1 — data extraction

### `ukb_panel_long.parquet`

The harmonised long-format panel. One row per `(eid, instance)`.

| Column            | Type   | Meaning                                              |
|-------------------|--------|------------------------------------------------------|
| eid               | int    | Participant ID                                       |
| instance          | int    | Assessment instance (0 = baseline, 2 = imaging, …)   |
| age               | float  | Age at this instance (years)                         |
| sex               | int    | 0 = female, 1 = male                                 |
| ethnicity         | int    | UKB ethnicity code                                   |
| smoking           | int    | 0 never, 1 previous, 2 current                       |
| townsend          | float  | Townsend deprivation index                           |
| crp               | float  | Raw CRP (mg/L)                                       |
| hba1c             | float  | HbA1c (IFCC mmol/mol)                                |
| grip_max          | float  | max(grip_left, grip_right) in kg                     |
| pulse_rate        | float  | Resting pulse (bpm)                                  |
| sbp, dbp          | float  | Blood pressure (mmHg)                                |
| cystatin_c        | float  | Raw cystatin C (mg/L)                                |
| bmd               | float  | Bone mineral density (first available: heel or DXA)  |
| bmi               | float  | BMI                                                  |
| homa_ir           | float  | If insulin available, glucose × insulin / 22.5       |
| accel_overall     | float  | Field 90012 pass-through                             |
| sleep_duration    | float  | Field 1160 pass-through                              |
| delta_I           | float  | log(clip(crp, 0.01))                                 |
| delta_M           | float  | HbA1c (or glucose fallback)                          |
| delta_F           | float  | -grip_max (sign-flipped)                             |
| delta_N           | float  | pulse_rate                                           |
| delta_C           | float  | Populated by step 8; otherwise NaN                   |
| delta_B           | float  | -bmd (sign-flipped)                                  |
| delta_P           | float  | log(clip(cystatin_c, 0.01))                          |
| med_statin        | bool   | Any coded statin use                                 |
| med_antihtn       | bool   | Any coded antihypertensive use                       |
| med_insulin       | bool   | Any coded insulin use                                |
| med_hrt           | bool   | Any coded HRT use (women only)                       |
| n_med_classes     | int    | Sum of statin + antihtn + insulin                    |
| co_diabetes       | bool   | Self-reported diabetes                               |
| co_hypertension   | bool   | Self-reported hypertension (code 4 in field 6150)    |
| co_heart          | bool   | Self-reported heart attack / angina / stroke         |
| co_cancer         | bool   | Self-reported cancer                                 |
| comorbidity_count | int    | Sum of co_* flags                                    |
| frailty_index     | float  | Rockwood FI in [0, 1]                                |
| baseline_date     | date   | Date of instance-0 assessment                        |
| death_date        | date   | Date of death (NaT if alive)                         |
| time_years        | float  | (death or censor date − baseline) / 365.25           |
| event             | float  | 1 if died during follow-up, else 0                   |

**Individual-level.** Do NOT share this file outside the UKB environment.

### `ukb_panel_summary.json`

Metadata about the extracted panel.

```json
{
  "script": "step1_extract.py",
  "config_hash": "abc123...",
  "test_mode": false,
  "n_rows": 600000,
  "n_eids": 500000,
  "instances": [0, 2],
  "columns": ["eid", "instance", "age", ...],
  "n_by_instance": {"0": 500000, "2": 100000},
  "axis_completeness": {"delta_I": 475000, "delta_M": 480000, ...},
  "n_deaths": 30000
}
```

---

## Step 2 — QC

### `ukb_qc_report.md`

Human-readable QC report. Sections:

- Sample size by instance
- Axis completeness (fraction non-missing) by instance
- Age distribution (mean, SD, median, min, max) by instance
- Medication prevalence by age stratum at baseline
- Longitudinal coverage per tier (N with ≥2 instances complete)
- Mortality summary (total deaths, median follow-up, by age stratum)

### `ukb_qc_stats.json`

Machine-readable version of the above. Structure:

```json
{
  "script": "step2_qc.py",
  "config_hash": "...",
  "n_by_instance": {...},
  "completeness_by_instance": {"0": {"I": 0.94, "M": 0.95, ...}, ...},
  "distributions_by_instance": {
    "0": {"crp": {"n": ..., "mean": ..., "sd": ..., "iqr_lo": ..., "iqr_hi": ..., "outliers_5sd": ...}, ...}
  },
  "age_summary": {"0": {...}, "2": {...}},
  "age_decade_counts": [{"instance": 0, "_age_decade": 40, "n": ...}, ...],
  "medication_prevalence_baseline": [{"stratum": "40-49", "n": ..., "statin": ..., ...}],
  "longitudinal_coverage_by_tier": {"tier1": ..., "tier2": ..., ...},
  "n_deaths": 30000,
  "median_follow_up_years": 10.2,
  "mortality_by_age_stratum": [...]
}
```

---

## Step 3 — cross-sectional (per tier)

### `ukb_cross_sectional_{tier}.json`

```json
{
  "tier": "tier2",
  "axes": ["I", "M", "F", "N"],
  "axis_cols": ["delta_I", "delta_M", "delta_F", "delta_N"],
  "youthful_reference_age": [40, 49],
  "reference_mu_sd": {"delta_I": {"mean": ..., "sd": ...}, ...},
  "per_stratum": [
    {
      "stratum": "40-49",
      "n": 50000,
      "lambda_max": 2.45,
      "lambda_max_ci": [2.42, 2.48],
      "trace": 4.1,
      "kappa": 3.2,
      "per_axis_variance": {"I": ..., "M": ..., ...},
      "mean_abs_corr": 0.15,
      "univar_ratio": 1.05,
      "Pi": {"V_norm": ..., "C_norm": ..., "Pi": 0.32},
      "sign_concordance": {
        "concordance": 0.75,
        "n_agree": 9,
        "n_total": 12,
        "n_excluded": 0,
        "binomial_p": 0.073,
        "pair_details": [...]
      }
    },
    ...
  ],
  "trend_kendall": {"tau": 0.8, "p_value": 0.01},
  "lambda_max_ratio": 1.45
}
```

### `figure_ukb_cross_sectional_{tier}.pdf`

2×2 panel figure:

- Top-left: λ_max by stratum with bootstrap 95% CI
- Top-right: univariate control ratio by stratum
- Bottom-left: mean |off-diagonal correlation| by stratum
- Bottom-right: Π by stratum

Suptitle reports the λ_max ratio, Kendall τ, and trend p-value.

---

## Step 4 — longitudinal (per tier)

### `ukb_longitudinal_{tier}.json`

```json
{
  "tier": "tier2",
  "axes": [...],
  "n_pairs": 100000,
  "per_stratum": [
    {
      "stratum": "40-49",
      "n": 8000,
      "lambda_max": ...,
      "lambda_max_ci": [...],
      "trace": ..., "kappa": ...,
      "univar_ratio": ...,
      "Pi": {...},
      "sign_concordance": {...}
    },
    ...
  ],
  "trend_kendall": {"tau": ..., "p_value": ...},
  "permutation_trend": {
    "observed_ratio": 1.8,
    "p_value": 0.001,
    "n_perm": 1000,
    "null_mean": 1.1,
    "null_p95": 1.4
  },
  "lambda_max_ratio": 1.8,
  "medication_stratified": {
    "medication_naive": [...],
    "on_medication": [...]
  }
}
```

### `figure_ukb_longitudinal_{tier}.pdf`

2×2 panels:

- λ_max trajectory (all + med-naive + on-med overlaid)
- Univariate control ratio
- Π trajectory
- N pairs per stratum

---

## Step 5 — lead-lag (per tier)

### `ukb_lead_lag_{tier}.json`

```json
{
  "tier": "tier2",
  "axes": [...],
  "n_triplets": 100000,
  "main": {
    "n_pairs": 12,
    "n_tested_vs_J": 12,
    "n_concordant": 10,
    "concordance_rate": 0.83,
    "binomial_p": 0.019,
    "pairs": [
      {
        "from_axis": "I",
        "to_axis": "M",
        "pair": "I->M",
        "predicted_sign": 1,
        "observed_sign": 1,
        "concordant": true,
        "beta": 0.042,
        "ci_lower": 0.032,
        "ci_upper": 0.052,
        "p_value": 3.1e-12,
        "n": 100000,
        "q_fdr": 8e-12
      },
      ...
    ]
  },
  "subgroups": {
    "medication_naive": {"n_concordant": ..., ...},
    "male": {...},
    "female": {...}
  }
}
```

### `figure_ukb_lead_lag_{tier}.pdf`

β heatmap (to-axis × from-axis), RdBu colormap, annotated cells with
β values and `*` for FDR-significant (q < 0.05). Title gives
concordance rate and binomial p.

---

## Step 6 — mortality (per tier)

### `ukb_mortality_{tier}.json`

```json
{
  "tier": "tier2",
  "axes": [...],
  "n_matched": 300000,
  "n_events": 20000,
  "biomarker_cols": ["log_crp", "hba1c", "grip_max", "pulse_rate", "log_cystatin_c", "bmi"],
  "models": {
    "M1_age_sex":      {"C": 0.721, "N": 300000, "covs": [...]},
    "M2_biomarkers":   {"C": 0.758, "N": 300000, "covs": [...]},
    "M3_swds":         {"C": 0.690, "N": 300000, "covs": ["log_swds"]},
    "M3a_mahalanobis": {"C": ...,   ...},
    "M3b_z_sum":       {"C": ...,   ...},
    "M4_frailty":      {"C": 0.748, ...},
    "M4a_frailty_bio": {"C": 0.762, ...},
    "M4b_frailty_swds":{"C": 0.755, ...},
    "M5_full":         {"C": 0.765, ...}
  },
  "delta_bootstraps": {
    "delta_C_M5_M4":  {"mean": 0.017, "ci_lower": 0.013, "ci_upper": 0.021, "n_boot": 2000},
    "delta_C_M5_M4a": {"mean": 0.003, "ci_lower": 0.001, "ci_upper": 0.005, "n_boot": 2000},
    "delta_C_M4b_M4": {"mean": 0.007, "ci_lower": 0.004, "ci_upper": 0.011, "n_boot": 2000}
  },
  "subgroups": {
    "medication_naive": {"n": ..., "events": ..., "C_M4": ..., "C_M4a": ..., "C_M5": ..., "delta_C_M5_M4": ..., "delta_C_M5_M4a": ...},
    "male":     {...},
    "female":   {...},
    "age_40_49":{...},
    "age_50_59":{...},
    ...
  }
}
```

### `figure_ukb_mortality_{tier}.pdf`

Horizontal bar chart of C-indices across M1 → M5 with numeric
annotations. Suptitle includes matched N, events, and the headline
ΔC(M5 − M4a) with 95% CI.

---

## Step 7 — null tests (per tier)

### `ukb_null_tests_{tier}.json`

```json
{
  "tier": "tier2",
  "axes": [...],
  "observed_ratio": 1.8,
  "age_permutation_null": {
    "observed_ratio": 1.8,
    "p_value": 0.001,
    "n_perm": 1000,
    "null_mean": 1.1,
    "null_p95": 1.4
  },
  "random_panel_null": {
    "n_panels": 500,
    "k": 4,
    "observed_ratio": 1.8,
    "p_value": 0.02,
    "null_mean": 1.3,
    "null_median": 1.25,
    "null_p95": 1.6,
    "null_ratios": [1.12, 1.34, ...]
  },
  "univariate_control": [
    {"stratum": "40-49", "n": 50000, "lambda_max": ..., "max_axis_variance": ..., "ratio": 1.05},
    ...
  ],
  "axis_substitution": {
    "M→bmi": 1.35,
    "N→sbp": 1.62,
    ...
  }
}
```

### `figure_ukb_null_tests_{tier}.pdf`

2×2 panels:

- Age-permutation null — vertical lines for observed and null-p95
- Random-panel null — histogram of null ratios with observed in red
- Univariate control trajectory (by stratum)
- Axis-substitution bar chart with observed as reference line

---

## Step 8 — circadian (optional)

### `ukb_circadian_proxy.parquet`

| Column            | Meaning                                   |
|-------------------|-------------------------------------------|
| eid               | Participant ID                            |
| circadian_score   | Composite HDR circadian axis value        |
| day_night_ratio   | Tier 1 component                          |
| overall_accel     | Tier 1 component                          |
| moderate_frac     | Tier 1 component                          |
| IS, IV, RA, L5_onset | Tier 2 components (raw CWA)            |

**Individual-level.** Do NOT share.

### `ukb_circadian_qc.json`

```json
{
  "script": "step8_circadian.py",
  "config_hash": "...",
  "tier": "1",
  "n_participants": 85000,
  "score_mean": 0.0,
  "score_sd": 0.98,
  "missing_pct": 0.05
}
```

Step 8 also **mutates `ukb_panel_long.parquet`** by writing the
circadian_score into `delta_C` for baseline-instance rows.

---

## Final summary

### `ukb_hdr_summary.md`

One-page Markdown summary written by `run_all.py`. Sections:

- **Sample** — N per instance, deaths, follow-up, longitudinal
  coverage per tier
- **Per-tier results** — cross-sectional + longitudinal λ_max, lead-lag
  concordance, mortality ΔC, null-test p-values
- **Comparison table** — UKB vs. InCHIANTI vs. ELSA on the headline
  metrics (N, axes, λ_max ratio, ΔC(M5−M4), I→M p)
- **Step log** — status and duration per step

This is the file a manuscript author reads first. Every number in it
is reproducible from the per-step JSON outputs.

---

## Retention guidance

What to keep:

- **All JSON files** — small, aggregate, publication-ready
- **All PDF figures** — small, publication-ready
- **`ukb_hdr_summary.md`** — the narrative

What NOT to share:

- **`ukb_panel_long.parquet`** — individual-level
- **`ukb_circadian_proxy.parquet`** — individual-level

When archiving a run for a manuscript submission, ZIP the `results/`
directory *excluding* the two parquet files. Every provenance artefact
is already preserved in the config-hash headers on each JSON.
