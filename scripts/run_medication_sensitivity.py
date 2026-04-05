#!/usr/bin/env python3
"""
Corrected Medication Sensitivity Analysis (R5)
===============================================
Fixes 5 divergence points identified by diagnose_delta_c.py, then runs
all medication sensitivity Cox models on matched samples.

Fixes applied:
  1a. Matched-sample Cox: ALL 5 models run on the SAME N
  1b. Covariates: 3-axis M2 uses exactly R4 covariates (no sysval)
  1c. NaN handling: hemda/hemdb excluded from Cox adjustment covariates
  1d. SWDS-Γ: uses cross-sectional stratum covariance (not within-person)
  1e. Survival time: matches R4/diagnostic construction exactly

Usage:
    python scripts/run_medication_sensitivity.py

Outputs:
    outputs/elsa_medication_sensitivity_corrected.json
"""

import json
import os
import sys
import warnings

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
DATA_DIR = os.path.join(ROOT, 'data', 'elsa')
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from hdr_sim.estimation import compute_swds_gamma, gamma_stability_proxy

SEED = 42
np.random.seed(SEED)

NURSE_WAVES = [2, 4, 6, 8, 11]
NURSE_WAVE_YEARS = {2: 2004, 4: 2008, 6: 2012, 8: 2016, 11: 2023}
AGE_STRATA = [(50, 59), (60, 69), (70, 79), (80, 90)]
GRIP_COLS = ['mmgsd1', 'mmgsd2', 'mmgsd3']

FILE_PATTERNS = {
    'harmonised': 'gh_elsa_h_hdr_subset',
    'supplement': 'elsa_supplementary_variables',
    'eol': 'h_elsa_eol_a2',
    'nurse_consolidated': 'elsa_nurse_biomarkers_consolidated',
    'nurse_w2': 'wave_2_nurse_data',
    'nurse_w4': 'wave_4_nurse_data',
    'nurse_w6': 'wave_6_elsa_nurse_data',
    'nurse_w8w9': 'elsa_nurse_w8w9',
    'nurse_w11': 'wave_11_elsa_nurse_data',
}

ADL_ITEMS = ['headldr', 'headlwa', 'headlba', 'headlea', 'headlbe', 'headlwc']
IADL_ITEMS = ['headlda', 'headlpr', 'headlsh', 'headlph', 'headlco',
              'headlme', 'headlho', 'headlmo']
MOBILITY_ITEMS = ['hemobwa', 'hemobsi', 'hemobch', 'hemobcs', 'hemobcl',
                  'hemobst', 'hemobre', 'hemobpu', 'hemobli', 'hemobpi']
CONDITION_VARS_LONG = ['diabetes', 'highbp', 'heart', 'stroke', 'lung', 'arthritis']

# R4 reference values for validation
R4_JSON = {
    'M1_c': 0.6018951673299953,
    'M2_c': 0.6132838631989543,
    'M3_c': 0.603372054250296,
    'M4_c': 0.6092545265046148,
    'M5_c': 0.6160020904162192,
    'delta_c': 0.0067475639116044706,
    'M1_n': 5431, 'M1_events': 1122,
    'M2_n': 4818, 'M2_events': 988,
    'M4_n': 5431, 'M4_events': 1122,
    'M5_n': 4818, 'M5_events': 988,
}


# ============================================================================
# Data loading helpers (identical to diagnostic)
# ============================================================================
def find_file(pattern, data_dir=DATA_DIR):
    for f in os.listdir(data_dir):
        if f.lower().endswith('.tab') and pattern.lower() in f.lower():
            return os.path.join(data_dir, f)
    return None


def load_tab(path):
    df = pd.read_csv(path, sep='\t', low_memory=False, na_values=[' ', '', 'NA'])
    df.columns = [c.lower() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith('string'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def filter_missing(df, exclude_cols=None):
    exclude = set(exclude_cols or [])
    for col in df.columns:
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] < 0, col] = np.nan
    return df


def detect_and_convert_hba1c(series):
    valid = series.dropna()
    if len(valid) == 0:
        return series, 'unknown'
    if valid.median() < 15:
        return (series - 2.15) * 10.929, 'DCCT'
    return series.copy(), 'IFCC'


def zscore_vs_ref(series, ref_df, col_name):
    ref_vals = ref_df[col_name].dropna() if col_name in ref_df.columns else pd.Series(dtype=float)
    if len(ref_vals) > 10:
        mu, sigma = ref_vals.mean(), ref_vals.std()
    else:
        mu, sigma = series.mean(), series.std()
    sigma = max(sigma, 1e-6)
    return (series - mu) / sigma


def compute_rockwood_fi(row):
    deficits = 0
    total = 0

    def check(val, is_deficit):
        nonlocal deficits, total
        if pd.notna(val):
            total += 1
            if is_deficit(val):
                deficits += 1

    for col in ADL_ITEMS:
        check(row.get(col), lambda v: v == 1)
    for col in IADL_ITEMS:
        check(row.get(col), lambda v: v == 1)
    for col in MOBILITY_ITEMS:
        check(row.get(col), lambda v: v == 1)
    check(row.get('hehelf'), lambda v: v >= 4)
    check(row.get('hemda'), lambda v: v == 1)
    check(row.get('hemdb'), lambda v: v == 1)
    for cond in CONDITION_VARS_LONG:
        check(row.get(cond), lambda v: v == 1)
    check(row.get('cesd'), lambda v: v >= 4)
    if total < 10:
        return np.nan
    return deficits / total


def harmonised_to_long(harm, waves=[2, 4, 6, 8]):
    harm = harm.copy()
    harm.columns = [c.lower() for c in harm.columns]
    static_cols = ['idauniq']
    for sc in ['ragender', 'raeduc_e', 'raeducl']:
        if sc in harm.columns:
            static_cols.append(sc)

    records = []
    for w in waves:
        wave_data = harm[static_cols].copy()
        wave_data['wave'] = w
        col_mapping = {
            f'r{w}agey': 'age',
            f'r{w}iwy': 'interview_year',
            f'r{w}iwstat': 'iwstat',
            f'r{w}cesd': 'cesd',
            f'r{w}shlt': 'self_health',
            f'r{w}diabe': 'diabetes',
            f'r{w}hibpe': 'highbp',
            f'r{w}hearte': 'heart',
            f'r{w}stroke': 'stroke',
            f'r{w}lunge': 'lung',
            f'r{w}arthre': 'arthritis',
            f'r{w}smokev': 'smoking',
            f'r{w}drink': 'alcohol',
            f'r{w}adla': 'adl_count',
            f'r{w}iadla': 'iadl_count',
            f'r{w}walkra': 'walk_speed',
            f'r{w}wghtsft': 'weight_shift',
            f'r{w}rgrip1': 'grip1',
            f'r{w}rgrip2': 'grip2',
            f'r{w}mbmi': 'bmi_harmonised',
            f'inw{w}n': 'had_nurse',
        }
        for src, dst in col_mapping.items():
            if src in harm.columns:
                wave_data[dst] = harm[src].values
            else:
                wave_data[dst] = np.nan
        records.append(wave_data)

    result = pd.concat(records, ignore_index=True)
    result = result.rename(columns={'ragender': 'sex'})
    return result


# ============================================================================
# Data loading & panel construction
# ============================================================================
def load_all_data():
    """Load all ELSA files, build merged panel with 3-axis and 4-axis vars."""
    print("=" * 70)
    print("CORRECTED MEDICATION SENSITIVITY PIPELINE (R5)")
    print("=" * 70)

    # Load files
    files = {}
    for key, pattern in FILE_PATTERNS.items():
        path = find_file(pattern)
        if path:
            files[key] = load_tab(path)
            print(f"  {key:25s}: {os.path.basename(path)} — "
                  f"{files[key].shape[0]:>7,} rows")
        elif not (key.startswith('nurse_w') and 'nurse_consolidated' in files):
            print(f"  {key:25s}: NOT FOUND")

    assert 'harmonised' in files, "Harmonised file required"
    assert 'nurse_consolidated' in files or any(
        k.startswith('nurse_w') for k in files), "Nurse data required"

    # Extract nurse biomarkers
    target_cols = ['idauniq', 'hscrp', 'hba1c', 'bmival', 'sysval', 'diaval',
                   'pulval', 'chol', 'hdl', 'ldl', 'trig', 'cfib', 'hgb',
                   'mmgsd1', 'mmgsd2', 'mmgsd3', 'mmgsn1', 'mmgsn2', 'mmgsn3',
                   'wbc', 'igf1', 'htval', 'wtval', 'wstval', 'mmcrsa', 'mmcrav',
                   'hemda', 'hemdb', 'hemdab', 'statins', 'statina']

    if 'nurse_consolidated' in files:
        df = files['nurse_consolidated'].copy()
        df = filter_missing(df, exclude_cols=['idauniq', 'wave'])
        available = [c for c in target_cols if c in df.columns]
        panel = df[available + ['wave']].copy()

        if 'bmi' in df.columns and 'bmival' not in panel.columns:
            panel['bmival'] = pd.to_numeric(df['bmi'], errors='coerce')
            panel.loc[panel['bmival'] < 0, 'bmival'] = np.nan

        for w in sorted(panel['wave'].unique()):
            w_mask = panel['wave'] == w
            if 'hba1c' in panel.columns and panel.loc[w_mask, 'hba1c'].notna().any():
                converted, unit = detect_and_convert_hba1c(panel.loc[w_mask, 'hba1c'])
                panel.loc[w_mask, 'hba1c'] = converted
                print(f"  Wave {int(w)}: HbA1c {unit}")

        grip_dom = [c for c in GRIP_COLS if c in panel.columns]
        panel['grip_max'] = panel[grip_dom].max(axis=1) if grip_dom else np.nan
    else:
        raise NotImplementedError("Legacy nurse file path not implemented — use consolidated")

    panel = panel.sort_values(['idauniq', 'wave', 'hscrp'], na_position='last')
    panel = panel.drop_duplicates(subset=['idauniq', 'wave'], keep='first')

    # Harmonised → long
    harm = files['harmonised'].copy()
    harm.columns = [c.lower() for c in harm.columns]
    harm = filter_missing(harm, exclude_cols=['idauniq'])
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])

    # Mortality
    mort = harm[['idauniq']].copy()
    mort['deceased'] = 0
    mort['death_wave'] = np.nan
    mort['death_age'] = np.nan
    mort['death_year'] = np.nan

    for w in range(3, 11):
        col = f'r{w}iwstat'
        if col in harm.columns:
            is_dead = (harm[col] == 4)
            newly_dead = is_dead & (mort['deceased'] == 0)
            mort.loc[newly_dead, 'deceased'] = 1
            mort.loc[newly_dead, 'death_wave'] = w

    if 'eol' in files:
        eol = files['eol'].copy()
        eol.columns = [c.lower() for c in eol.columns]
        eol = filter_missing(eol, exclude_cols=['idauniq'])
        if 'radage' in eol.columns:
            eol_sub = eol[['idauniq', 'radage', 'raxyear']].dropna(subset=['radage'])
            eol_sub = eol_sub.rename(columns={'radage': 'eol_death_age',
                                               'raxyear': 'eol_death_year'})
            mort = mort.merge(eol_sub, on='idauniq', how='left')
            has_eol = mort['eol_death_age'].notna()
            mort.loc[has_eol, 'deceased'] = 1
            mort.loc[has_eol, 'death_age'] = mort.loc[has_eol, 'eol_death_age']
            mort.loc[has_eol, 'death_year'] = mort.loc[has_eol, 'eol_death_year']
            mort = mort.drop(columns=['eol_death_age', 'eol_death_year'])

    n_dead = (mort['deceased'] == 1).sum()
    print(f"  Mortality: {n_dead:,} deceased")

    # Supplementary
    supp = None
    if 'supplement' in files:
        supp = files['supplement'].copy()
        supp.columns = [c.lower() for c in supp.columns]
        supp = filter_missing(supp, exclude_cols=['idauniq', 'wave'])

    # Merge
    merged = panel.merge(harm_long, on=['idauniq', 'wave'], how='left')

    # Wave 11 age estimation
    w11_mask = (merged['wave'] == 11) & merged['age'].isna()
    if w11_mask.any():
        w8_ages = harm_long[harm_long['wave'] == 8][['idauniq', 'age']].rename(
            columns={'age': 'age_w8'})
        merged = merged.merge(w8_ages, on='idauniq', how='left')
        can_fill = w11_mask & merged['age_w8'].notna()
        year_diff = NURSE_WAVE_YEARS.get(11, 2023) - NURSE_WAVE_YEARS.get(8, 2016)
        merged.loc[can_fill, 'age'] = merged.loc[can_fill, 'age_w8'] + year_diff
        merged = merged.drop(columns=['age_w8'], errors='ignore')

    merged['interview_year'] = merged['interview_year'].fillna(
        merged['wave'].map(NURSE_WAVE_YEARS))
    merged = merged.merge(mort, on='idauniq', how='left')
    if supp is not None:
        merged = merged.merge(supp, on=['idauniq', 'wave'], how='left')

    merged = merged[(merged['age'] >= 50) & (merged['age'] <= 90)]

    # BMI fill from harmonised
    if 'bmi_harmonised' in merged.columns:
        mask = merged['bmival'].isna() & merged['bmi_harmonised'].notna()
        merged.loc[mask, 'bmival'] = merged.loc[mask, 'bmi_harmonised']

    # Reference subgroup (Wave 2, age 50-55)
    ref_mask = (merged['wave'] == 2) & (merged['age'] >= 50) & (merged['age'] <= 55)
    ref = merged.loc[ref_mask]

    # --- 3-axis variables ---
    merged['log_crp'] = np.log(merged['hscrp'].clip(lower=0.01))
    ref_log_crp = np.log(ref['hscrp'].clip(lower=0.01).dropna())
    if len(ref_log_crp) > 10:
        crp_mean, crp_std = ref_log_crp.mean(), ref_log_crp.std()
    else:
        crp_mean, crp_std = merged['log_crp'].mean(), merged['log_crp'].std()
    crp_std = max(crp_std, 1e-6)
    merged['dx_I'] = (merged['log_crp'] - crp_mean) / crp_std

    for var, label in [('hba1c', 'hba1c_z'), ('bmival', 'bmi_z')]:
        ref_vals = ref[var].dropna() if var in ref.columns else pd.Series(dtype=float)
        if len(ref_vals) > 10:
            mu, sigma = ref_vals.mean(), ref_vals.std()
        else:
            mu, sigma = merged[var].mean(), merged[var].std()
        sigma = max(sigma, 1e-6)
        merged[label] = (merged[var] - mu) / sigma

    merged['dx_M'] = (merged['hba1c_z'] + merged['bmi_z']) / np.sqrt(2)

    ref_grip = ref['grip_max'].dropna() if 'grip_max' in ref.columns else pd.Series(dtype=float)
    if len(ref_grip) > 10:
        grip_mean, grip_std = ref_grip.mean(), ref_grip.std()
    else:
        grip_mean, grip_std = merged['grip_max'].mean(), merged['grip_max'].std()
    grip_std = max(grip_std, 1e-6)
    merged['dx_F'] = -(merged['grip_max'] - grip_mean) / grip_std

    merged['complete_3axis'] = (
        merged['dx_I'].notna() & merged['dx_M'].notna() & merged['dx_F'].notna()
    )

    # --- 4-axis variables ---
    merged['cfib_z'] = zscore_vs_ref(
        merged.get('cfib', pd.Series(dtype=float, index=merged.index)), ref, 'cfib')
    merged['dx_I_4'] = merged[['dx_I', 'cfib_z']].mean(axis=1, skipna=True)
    merged.loc[merged['dx_I_4'].isna() & merged['dx_I'].notna(), 'dx_I_4'] = \
        merged.loc[merged['dx_I_4'].isna() & merged['dx_I'].notna(), 'dx_I']

    if 'chol' in merged.columns and 'hdl' in merged.columns:
        merged['chol_hdl_ratio'] = merged['chol'] / merged['hdl'].clip(lower=0.1)
        merged['chol_hdl_ratio_z'] = zscore_vs_ref(merged['chol_hdl_ratio'], ref, 'chol_hdl_ratio')
    else:
        merged['chol_hdl_ratio_z'] = np.nan

    if 'trig' in merged.columns:
        merged['log_trig'] = np.log(merged['trig'].clip(lower=0.01))
        merged['log_trig_z'] = zscore_vs_ref(merged['log_trig'], ref, 'log_trig')
    else:
        merged['log_trig_z'] = np.nan

    merged['dx_M_4'] = merged[['hba1c_z', 'chol_hdl_ratio_z', 'log_trig_z']].mean(
        axis=1, skipna=True)

    for var in ['sysval', 'diaval', 'pulval']:
        if var in merged.columns:
            merged[f'{var}_z'] = zscore_vs_ref(merged[var], ref, var)
        else:
            merged[f'{var}_z'] = np.nan

    merged['dx_N_4'] = merged[['sysval_z', 'diaval_z', 'pulval_z']].mean(
        axis=1, skipna=True)

    # Pulse-only N axis variant
    merged['dx_N_pulse'] = merged['pulval_z'].copy()

    merged['grip_z_rev'] = -zscore_vs_ref(merged['grip_max'], ref, 'grip_max')
    if 'walk_speed' in merged.columns:
        merged['walk_z_rev'] = -zscore_vs_ref(merged['walk_speed'], ref, 'walk_speed')
    else:
        merged['walk_z_rev'] = np.nan
    merged['dx_F_4'] = merged[['grip_z_rev', 'walk_z_rev']].mean(axis=1, skipna=True)

    merged['complete_4axis'] = (
        merged['dx_I_4'].notna() & merged['dx_M_4'].notna() &
        merged['dx_N_4'].notna() & merged['dx_F_4'].notna()
    )

    # --- Med-robust M axis: exclude chol/trig (statin-affected) ---
    merged['dx_M_robust'] = merged['hba1c_z'].copy()  # HbA1c only (less statin-affected)

    print(f"\n  Panel: {len(merged):,} person-visits, "
          f"{merged['idauniq'].nunique():,} unique people")
    print(f"  Complete 3-axis: {merged['complete_3axis'].sum():,}")
    print(f"  Complete 4-axis: {merged['complete_4axis'].sum():,}")

    return merged, harm, files


# ============================================================================
# FIX 1d: SWDS-Γ using CROSS-SECTIONAL stratum covariance
# ============================================================================
def compute_swds_cross_sectional(merged, axes, complete_col):
    """
    Compute SWDS-Γ using cross-sectional stratum-level covariance.

    This matches the R4 definition: Γ̂_stratum is the sample covariance
    of the CROSS-SECTIONAL data within each age stratum (not within-person
    residuals).

    FIX 1d: The refactored pipeline used within-person residual covariance.
    The R4 pipeline uses plain cross-sectional covariance within strata.
    """
    complete = merged[merged[complete_col]].copy()

    # Build cross-sectional Γ̂ per age stratum from all complete visits
    gamma_lookup = {}
    for (lo, hi) in AGE_STRATA:
        stratum = complete[(complete['age'] >= lo) & (complete['age'] < hi)]
        if len(stratum) >= 20:
            X = stratum[axes].dropna().values
            if len(X) >= 20:
                gamma_lookup[(lo, hi)] = np.cov(X.T)

    # Global fallback
    X_all = complete[axes].dropna().values
    Gamma_global = np.cov(X_all.T) if len(X_all) > len(axes) else np.eye(len(axes))

    def get_gamma(age):
        for (lo, hi), G in gamma_lookup.items():
            if lo <= age < hi:
                return G
        return Gamma_global

    scores = []
    for idx, row in complete.iterrows():
        dx = row[axes].values.astype(float)
        if np.any(np.isnan(dx)):
            scores.append(np.nan)
            continue
        G = get_gamma(row['age'])
        scores.append(compute_swds_gamma(dx, G))

    complete['swds_gamma'] = scores
    return complete


# ============================================================================
# FIX 1e: Survival time construction (matches R4/diagnostic exactly)
# ============================================================================
def build_survival_baseline(merged):
    """
    Construct baseline (wave 2) survival data matching R4 logic exactly.

    FIX 1e: Uses the same survival time construction as the diagnostic:
    - Baseline year from interview_year (default 2004 for wave 2)
    - Last contact year defaults to 2024
    - Time = death_year - baseline_year (deceased) or
             last_contact_year - baseline_year (censored)
    """
    baseline = merged[merged['wave'] == 2].copy()
    baseline['deceased'] = baseline['deceased'].fillna(0).astype(int)

    baseline_year = baseline['interview_year'].fillna(NURSE_WAVE_YEARS.get(2, 2004))

    # Last contact: default 2024 (matches diagnostic)
    baseline['last_contact_year'] = 2024

    baseline['time'] = np.where(
        baseline['deceased'] == 1,
        baseline['death_year'].fillna(baseline['last_contact_year']) - baseline_year,
        baseline['last_contact_year'] - baseline_year
    )

    baseline = baseline[baseline['time'] > 0].copy()
    return baseline


# ============================================================================
# FIX 1a + 1b + 1c: Matched-sample Cox models with correct covariates
# ============================================================================
def run_matched_cox(baseline, axes, swds_col='swds_gamma', model_label='3-axis'):
    """
    Run 5 nested Cox models on a MATCHED sample (same N for all models).

    FIX 1a: Define one sample as intersection of all non-missing values for
            ALL variables used in ANY of the 5 models. Run all 5 on this sample.

    FIX 1b: 3-axis M2 uses EXACTLY R4 covariates:
            age, sex, log_crp, hba1c, grip_max, bmival, smoking, diabetes, highbp
            4-axis M2 adds the 4-axis-specific biomarkers.
            NO sysval in 3-axis M2.

    FIX 1c: hemda, hemdb are NOT included as adjustment covariates.
            They are used only for subgroup definition (med-naive analysis).
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    # Define model covariates based on axis set
    base_covs = ['age', 'sex']
    # FIX 1c: Only smoking, diabetes, highbp as adjustments (NO hemda, hemdb)
    adj_covs = ['smoking', 'diabetes', 'highbp']

    if model_label.startswith('3-axis'):
        # FIX 1b: EXACTLY R4 biomarkers for 3-axis (NO sysval)
        bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival']
    else:
        # 4-axis: include axis-relevant biomarkers
        bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival', 'sysval']

    models = {
        'M1: Age + Sex':     base_covs + adj_covs,
        'M2: + Biomarkers':  base_covs + bio_covs + adj_covs,
        'M3: + SWDS-G':      base_covs + [swds_col] + adj_covs,
        'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
        'M5: Full':          base_covs + bio_covs + [swds_col, 'rockwood_fi'] + adj_covs,
    }

    # FIX 1a: Collect ALL variables needed across ALL models
    all_vars = set()
    for covs in models.values():
        all_vars.update(covs)
    all_vars.update(['time', 'deceased'])

    # Filter to columns that exist
    all_vars = [v for v in all_vars if v in baseline.columns]

    # Define the MATCHED sample: all rows with no NaN in ANY required variable
    matched = baseline[all_vars].dropna().copy()

    # Drop near-zero-variance covariates (e.g., diabetes/highbp in med-naive subgroup)
    dropped_covs = []
    for c in list(adj_covs):
        if c in matched.columns and matched[c].std() < 0.01:
            dropped_covs.append(c)
    if dropped_covs:
        print(f"  Dropping near-zero-variance covariates: {dropped_covs}")
        # Remove from all model definitions
        for name in models:
            models[name] = [v for v in models[name] if v not in dropped_covs]
        # Recollect all_vars after dropping
        all_vars_new = set()
        for covs in models.values():
            all_vars_new.update(covs)
        all_vars_new.update(['time', 'deceased'])
        all_vars_new = [v for v in all_vars_new if v in baseline.columns]
        matched = baseline[all_vars_new].dropna().copy()

    n_matched = len(matched)
    n_events = int(matched['deceased'].sum())

    print(f"\n  [{model_label}] Matched sample: N={n_matched:,}, events={n_events:,}")
    print(f"  [{model_label}] Biomarker audit:")
    print(f"    base_covs = {base_covs}")
    print(f"    bio_covs  = {bio_covs}")
    print(f"    adj_covs  = {[c for c in adj_covs if c not in dropped_covs]}")
    print(f"    swds_col  = {swds_col}")

    if n_matched < 50 or n_events < 10:
        print(f"  SKIP: insufficient matched sample")
        return None

    results = {}
    for name, covs in models.items():
        available_covs = [c for c in covs if c in matched.columns]
        surv_data = matched[available_covs + ['time', 'deceased']].copy()

        try:
            # Check for inf/nan in data
            if surv_data.isin([np.inf, -np.inf]).any().any():
                surv_data = surv_data.replace([np.inf, -np.inf], np.nan).dropna()

            cph = CoxPHFitter()
            cph.fit(surv_data, duration_col='time', event_col='deceased')

            # Use fitted model's own concordance_index_ (standard Harrell's C
            # computed on the full partial hazard, which is the authoritative value)
            c_idx = cph.concordance_index_

            # Also compute via lifelines.utils for comparison
            c_idx_util = concordance_index(
                surv_data['time'],
                -cph.predict_partial_hazard(surv_data),
                surv_data['deceased']
            )

            results[name] = {
                'c_index': c_idx,
                'c_index_util': c_idx_util,
                'n': len(surv_data),
                'events': int(surv_data['deceased'].sum()),
                'covariates': available_covs,
            }

            # Diagnostic printing
            print(f"    {name}:")
            print(f"      Covariates: {available_covs}")
            print(f"      N={len(surv_data):,}, events={int(surv_data['deceased'].sum()):,}")
            print(f"      C-index (model.concordance_index_): {c_idx:.6f}")
            print(f"      C-index (lifelines.utils):          {c_idx_util:.6f}")
            if abs(c_idx - c_idx_util) > 1e-6:
                print(f"      ** DISCREPANCY: {c_idx - c_idx_util:+.6f}")

            # Print M1 coefficients for sanity check
            if 'M1' in name:
                print(f"      M1 coefficients (HR):")
                for cov_name in available_covs:
                    coef = cph.params_[cov_name]
                    hr = np.exp(coef)
                    print(f"        {cov_name}: coef={coef:.4f}, HR={hr:.4f}")

        except Exception as e:
            # Retry with penalizer for convergence issues
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(surv_data, duration_col='time', event_col='deceased')
                c_idx = cph.concordance_index_
                c_idx_util = concordance_index(
                    surv_data['time'],
                    -cph.predict_partial_hazard(surv_data),
                    surv_data['deceased']
                )
                results[name] = {
                    'c_index': c_idx,
                    'c_index_util': c_idx_util,
                    'n': len(surv_data),
                    'events': int(surv_data['deceased'].sum()),
                    'covariates': available_covs,
                    'penalized': True,
                }
                print(f"    {name}: C={c_idx:.6f} (N={len(surv_data):,}, "
                      f"events={int(surv_data['deceased'].sum()):,}) [penalized]")
                print(f"      C-index (model): {c_idx:.6f}, C-index (util): {c_idx_util:.6f}")
            except Exception as e2:
                print(f"    {name}: FAILED — {e2}")
                results[name] = {'c_index': np.nan, 'n': 0, 'events': 0}

    # Delta C
    m5_c = results.get('M5: Full', {}).get('c_index', np.nan)
    m4_c = results.get('M4: + Rockwood FI', {}).get('c_index', np.nan)
    dc = m5_c - m4_c if not (np.isnan(m5_c) or np.isnan(m4_c)) else np.nan
    results['_delta_c'] = dc
    results['_n_matched'] = n_matched
    results['_n_events'] = n_events

    if not np.isnan(dc):
        print(f"    DC(M5-M4) = {dc:+.6f}")

    return results


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    merged, harm, files = load_all_data()

    # ======================================================================
    # Step 1: Compute SWDS-Γ (cross-sectional, FIX 1d) and FI
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Compute SWDS-Γ (cross-sectional) and Rockwood FI")
    print("=" * 70)

    # 3-axis SWDS-Γ
    axes_3 = ['dx_I', 'dx_M', 'dx_F']
    complete_3 = compute_swds_cross_sectional(merged, axes_3, 'complete_3axis')
    print(f"  3-axis SWDS-Γ: {complete_3['swds_gamma'].notna().sum():,} scores")

    # 4-axis SWDS-Γ
    axes_4 = ['dx_I_4', 'dx_M_4', 'dx_N_4', 'dx_F_4']
    complete_4_mask = (
        merged['dx_I_4'].notna() & merged['dx_M_4'].notna() &
        merged['dx_N_4'].notna() & merged['dx_F_4'].notna()
    )
    complete_4 = compute_swds_cross_sectional(merged, axes_4, 'complete_4axis')
    complete_4 = complete_4.rename(columns={'swds_gamma': 'swds_gamma_4'})
    print(f"  4-axis SWDS-Γ: {complete_4['swds_gamma_4'].notna().sum():,} scores")

    # Rockwood FI
    supp_cols_present = any(col in complete_3.columns for col in ADL_ITEMS)
    if supp_cols_present:
        complete_3['rockwood_fi'] = complete_3.apply(compute_rockwood_fi, axis=1)
        print(f"  Rockwood FI (3-axis): {complete_3['rockwood_fi'].notna().sum():,}")

    if supp_cols_present:
        complete_4['rockwood_fi'] = complete_4.apply(compute_rockwood_fi, axis=1)
        print(f"  Rockwood FI (4-axis): {complete_4['rockwood_fi'].notna().sum():,}")

    # ======================================================================
    # Step 2: Build baseline survival data (FIX 1e)
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Build baseline survival data")
    print("=" * 70)

    baseline_3 = build_survival_baseline(complete_3)
    print(f"  3-axis baseline (wave 2): N={len(baseline_3):,}, "
          f"events={baseline_3['deceased'].sum():,}, "
          f"median FU={baseline_3['time'].median():.1f}y")

    baseline_4 = build_survival_baseline(complete_4)
    print(f"  4-axis baseline (wave 2): N={len(baseline_4):,}, "
          f"events={baseline_4['deceased'].sum():,}, "
          f"median FU={baseline_4['time'].median():.1f}y")

    # ======================================================================
    # Step 3: Validate 3-axis against R4
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Validate 3-axis model against R4 (matched samples)")
    print("=" * 70)

    cox_3axis = run_matched_cox(baseline_3, axes_3,
                                swds_col='swds_gamma', model_label='3-axis')

    if cox_3axis:
        dc_3 = cox_3axis['_delta_c']
        print(f"\n  R4 reference:  DC = +{R4_JSON['delta_c']:.7f}")
        print(f"  R5 (matched):  DC = {dc_3:+.7f}")
        print(f"  Note: R4 used unmatched samples; R5 uses matched (same N for all models)")
        if abs(dc_3) < 0.015:
            print(f"  PASS: 3-axis DC is in expected range (<0.015)")
        else:
            print(f"  WARNING: 3-axis DC is unexpectedly large — investigate")

    # ======================================================================
    # Step 4: Run all medication sensitivity analyses
    # ======================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Medication Sensitivity Analyses")
    print("=" * 70)

    all_results = {}

    # --- Model 1: 3-axis full sample ---
    print("\n--- Model 1: 3-axis (full sample) ---")
    all_results['3-axis (full sample)'] = cox_3axis

    # --- Model 2: 4-axis original (full sample) ---
    print("\n--- Model 2: 4-axis original (full sample) ---")
    cox_4axis = run_matched_cox(baseline_4, axes_4,
                                swds_col='swds_gamma_4', model_label='4-axis')
    all_results['4-axis (full sample)'] = cox_4axis

    # --- Model 3: 4-axis pulse-only N (full sample) ---
    print("\n--- Model 3: 4-axis pulse-only N ---")
    # Replace dx_N_4 with pulse-only
    baseline_4p = baseline_4.copy()
    baseline_4p['dx_N_4'] = baseline_4p['dx_N_pulse']
    # Recompute SWDS for pulse-only variant
    axes_4p = ['dx_I_4', 'dx_M_4', 'dx_N_4', 'dx_F_4']
    complete_mask = (
        baseline_4p['dx_I_4'].notna() & baseline_4p['dx_M_4'].notna() &
        baseline_4p['dx_N_4'].notna() & baseline_4p['dx_F_4'].notna()
    )
    baseline_4p = baseline_4p[complete_mask].copy()

    # Recompute SWDS-Γ for this variant
    gamma_lookup_4p = {}
    for (lo, hi) in AGE_STRATA:
        stratum = baseline_4p[(baseline_4p['age'] >= lo) & (baseline_4p['age'] < hi)]
        X = stratum[axes_4p].dropna().values
        if len(X) >= 20:
            gamma_lookup_4p[(lo, hi)] = np.cov(X.T)
    X_all_4p = baseline_4p[axes_4p].dropna().values
    Gamma_global_4p = np.cov(X_all_4p.T) if len(X_all_4p) > 4 else np.eye(4)

    def get_gamma_4p(age):
        for (lo, hi), G in gamma_lookup_4p.items():
            if lo <= age < hi:
                return G
        return Gamma_global_4p

    swds_4p = []
    for idx, row in baseline_4p.iterrows():
        dx = row[axes_4p].values.astype(float)
        if np.any(np.isnan(dx)):
            swds_4p.append(np.nan)
        else:
            swds_4p.append(compute_swds_gamma(dx, get_gamma_4p(row['age'])))
    baseline_4p['swds_gamma_4'] = swds_4p

    cox_4pulse = run_matched_cox(baseline_4p, axes_4p,
                                 swds_col='swds_gamma_4', model_label='4-axis-pulse-N')
    all_results['4-axis pulse-only N'] = cox_4pulse

    # --- Models 4 & 5: Med-naive subgroups ---
    # Define med-naive: no baseline hypertension treatment AND no baseline diabetes treatment
    # Using r2hibpe and r2diabe from harmonised (near-complete coverage)
    print("\n--- Defining medication-naive subgroup ---")
    harm_clean = harm.copy()

    # Get baseline hibpe and diabe (wave 2) from harmonised
    hibpe_col = 'r2hibpe'
    diabe_col = 'r2diabe'

    if hibpe_col in harm_clean.columns and diabe_col in harm_clean.columns:
        med_naive_ids = harm_clean[
            (harm_clean[hibpe_col] == 0) & (harm_clean[diabe_col] == 0)
        ]['idauniq'].values
        print(f"  Med-naive (no hibpe, no diabe at baseline): {len(med_naive_ids):,} people")
    else:
        # Fallback: use hemda/hemdb from nurse data
        print(f"  WARNING: r2hibpe/r2diabe not in harmonised, using hemda/hemdb fallback")
        baseline_nurse = merged[(merged['wave'] == 2)]
        med_naive_ids = baseline_nurse[
            (baseline_nurse.get('hemda', pd.Series(dtype=float)) != 1) &
            (baseline_nurse.get('hemdb', pd.Series(dtype=float)) != 1)
        ]['idauniq'].values
        print(f"  Med-naive (no hemda, no hemdb at wave 2): {len(med_naive_ids):,} people")

    # 4-axis med-naive
    print("\n--- Model 4: 4-axis med-naive ---")
    baseline_4_naive = baseline_4[baseline_4['idauniq'].isin(med_naive_ids)].copy()
    print(f"  4-axis med-naive baseline: N={len(baseline_4_naive):,}")
    cox_4naive = run_matched_cox(baseline_4_naive, axes_4,
                                 swds_col='swds_gamma_4', model_label='4-axis-med-naive')
    all_results['4-axis med-naive'] = cox_4naive

    # 3-axis med-naive
    print("\n--- Model 5: 3-axis med-naive ---")
    baseline_3_naive = baseline_3[baseline_3['idauniq'].isin(med_naive_ids)].copy()
    print(f"  3-axis med-naive baseline: N={len(baseline_3_naive):,}")
    cox_3naive = run_matched_cox(baseline_3_naive, axes_3,
                                 swds_col='swds_gamma', model_label='3-axis-med-naive')
    all_results['3-axis med-naive'] = cox_3naive

    # --- Model 6: 4-axis med-robust ---
    print("\n--- Model 6: 4-axis med-robust ---")
    # Med-robust: dx_M_robust (HbA1c only), dx_N_pulse (pulse only)
    baseline_4r = baseline_4.copy()
    baseline_4r['dx_M_4'] = baseline_4r['dx_M_robust']
    baseline_4r['dx_N_4'] = baseline_4r['dx_N_pulse']
    axes_4r = ['dx_I_4', 'dx_M_4', 'dx_N_4', 'dx_F_4']
    complete_mask_r = (
        baseline_4r['dx_I_4'].notna() & baseline_4r['dx_M_4'].notna() &
        baseline_4r['dx_N_4'].notna() & baseline_4r['dx_F_4'].notna()
    )
    baseline_4r = baseline_4r[complete_mask_r].copy()

    # Recompute SWDS-Γ for med-robust variant
    gamma_lookup_4r = {}
    for (lo, hi) in AGE_STRATA:
        stratum = baseline_4r[(baseline_4r['age'] >= lo) & (baseline_4r['age'] < hi)]
        X = stratum[axes_4r].dropna().values
        if len(X) >= 20:
            gamma_lookup_4r[(lo, hi)] = np.cov(X.T)
    X_all_4r = baseline_4r[axes_4r].dropna().values
    Gamma_global_4r = np.cov(X_all_4r.T) if len(X_all_4r) > 4 else np.eye(4)

    def get_gamma_4r(age):
        for (lo, hi), G in gamma_lookup_4r.items():
            if lo <= age < hi:
                return G
        return Gamma_global_4r

    swds_4r = []
    for idx, row in baseline_4r.iterrows():
        dx = row[axes_4r].values.astype(float)
        if np.any(np.isnan(dx)):
            swds_4r.append(np.nan)
        else:
            swds_4r.append(compute_swds_gamma(dx, get_gamma_4r(row['age'])))
    baseline_4r['swds_gamma_4'] = swds_4r

    cox_4robust = run_matched_cox(baseline_4r, axes_4r,
                                  swds_col='swds_gamma_4', model_label='4-axis-med-robust')
    all_results['4-axis med-robust'] = cox_4robust

    # ======================================================================
    # Step 5: Output
    # ======================================================================
    print("\n" + "=" * 70)
    print("CORRECTED MEDICATION SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"{'Model':<28s} {'N(Cox)':>8s} {'Events':>8s} {'DC(M5-M4)':>12s} {'Threshold':>10s}")
    print("-" * 70)

    threshold = 0.01
    summary_rows = []
    for label, res in all_results.items():
        if res is None:
            print(f"{label:<28s} {'N/A':>8s} {'N/A':>8s} {'N/A':>12s} {'N/A':>10s}")
            continue
        n = res['_n_matched']
        ev = res['_n_events']
        dc = res['_delta_c']
        passes = dc >= threshold if not np.isnan(dc) else False
        mark = 'PASS' if passes else 'FAIL'
        print(f"{label:<28s} {n:>8,} {ev:>8,} {dc:>+12.4f} {mark:>10s}")
        summary_rows.append({
            'model': label,
            'n_cox': n,
            'events': ev,
            'delta_c': dc,
            'passes_threshold': passes,
        })

    print("-" * 70)
    print(f"Threshold: DC >= {threshold}")
    print("All models within each row use matched samples (same N for all 5 Cox models).")

    # Detailed C-indices
    print(f"\n{'=' * 70}")
    print("DETAILED C-INDICES (M1–M5) PER MODEL")
    print("=" * 70)

    model_names = ['M1: Age + Sex', 'M2: + Biomarkers', 'M3: + SWDS-G',
                   'M4: + Rockwood FI', 'M5: Full']
    header = f"{'Model':<28s}"
    for m in model_names:
        header += f" {m.split(':')[0]:>8s}"
    header += f" {'DC(M5-M4)':>10s}"
    print(header)
    print("-" * (28 + 8 * len(model_names) + 12))

    for label, res in all_results.items():
        if res is None:
            continue
        row_str = f"{label:<28s}"
        for m in model_names:
            c = res.get(m, {}).get('c_index', np.nan)
            if np.isnan(c):
                row_str += f" {'N/A':>8s}"
            else:
                row_str += f" {c:>8.4f}"
        dc = res.get('_delta_c', np.nan)
        row_str += f" {dc:>+10.4f}" if not np.isnan(dc) else f" {'N/A':>10s}"
        print(row_str)

    # ======================================================================
    # Save JSON
    # ======================================================================
    output = {
        'description': 'Corrected medication sensitivity analysis (R5)',
        'fixes_applied': [
            '1a: Matched-sample Cox (same N for all 5 models)',
            '1b: 3-axis M2 uses exactly R4 covariates (no sysval)',
            '1c: hemda/hemdb excluded from Cox adjustment covariates',
            '1d: SWDS-Gamma uses cross-sectional stratum covariance',
            '1e: Survival time matches R4/diagnostic construction',
        ],
        'r4_reference': {
            'delta_c': R4_JSON['delta_c'],
            'note': 'R4 used unmatched samples (different N per model)',
        },
        'threshold': threshold,
        'models': {},
    }

    for label, res in all_results.items():
        if res is None:
            output['models'][label] = {'status': 'insufficient_data'}
            continue

        model_data = {
            'n_cox': res['_n_matched'],
            'events': res['_n_events'],
            'delta_c_m5_m4': float(res['_delta_c']) if not np.isnan(res['_delta_c']) else None,
            'passes_threshold': bool(res.get('_delta_c', 0) >= threshold),
            'c_indices': {},
        }
        for m in model_names:
            if m in res:
                model_data['c_indices'][m] = {
                    'c_index': float(res[m]['c_index']) if not np.isnan(res[m].get('c_index', np.nan)) else None,
                    'n': res[m].get('n', 0),
                    'events': res[m].get('events', 0),
                }
        output['models'][label] = model_data

    json_path = os.path.join(OUTPUT_DIR, 'elsa_medication_sensitivity_corrected.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Decision guidance
    print(f"\n{'=' * 70}")
    print("DECISION GUIDANCE")
    print("=" * 70)

    dc_3_full = all_results.get('3-axis (full sample)', {})
    dc_4_naive = all_results.get('4-axis med-naive', {})

    if dc_3_full:
        dc3 = dc_3_full.get('_delta_c', np.nan)
        if not np.isnan(dc3):
            if dc3 > 0.01:
                print(f"  WARNING: 3-axis DC = {dc3:+.4f} > 0.01 — something may still be wrong")
            else:
                print(f"  3-axis DC = {dc3:+.4f} (expected ~+0.0065 to +0.0067)")

    if dc_4_naive:
        dc4n = dc_4_naive.get('_delta_c', np.nan)
        if not np.isnan(dc4n):
            if dc4n >= 0.01:
                print(f"  4-axis med-naive DC = {dc4n:+.4f} >= 0.01 -> Nature Aging viable")
            elif dc4n >= 0.005:
                print(f"  4-axis med-naive DC = {dc4n:+.4f} — marginal (0.005-0.01)")
            else:
                print(f"  4-axis med-naive DC = {dc4n:+.4f} < 0.005 -> FI dominates -> Cell Systems")


if __name__ == '__main__':
    main()
