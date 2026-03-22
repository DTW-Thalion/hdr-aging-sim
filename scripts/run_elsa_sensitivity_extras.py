#!/usr/bin/env python3
"""
ELSA Sensitivity Extras (O1, O2, O3)
=====================================
Three supplementary sensitivity analyses for the ELSA validation:

  O3: Alternative youthful reference sensitivity (age 55-60 vs 50-55)
  O1: Calibration plots for M4 and M5 (predicted vs observed 5-year mortality)
  O2: Frailty-transition secondary endpoint (incident FI > 0.25)

Reuses the corrected pipeline infrastructure from run_medication_sensitivity.py.

Usage:
    python scripts/run_elsa_sensitivity_extras.py

Outputs:
    outputs/figure_elsa_calibration.pdf
    outputs/elsa_frailty_transition.json
    outputs/elsa_sensitivity_extras.json
"""

import json
import os
import sys
import warnings
import zipfile

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
DATA_DIR = os.path.join(ROOT, 'data', 'elsa')
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from hdr_sim.estimation import compute_swds_gamma

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
}

ADL_ITEMS = ['headldr', 'headlwa', 'headlba', 'headlea', 'headlbe', 'headlwc']
IADL_ITEMS = ['headlda', 'headlpr', 'headlsh', 'headlph', 'headlco',
              'headlme', 'headlho', 'headlmo']
MOBILITY_ITEMS = ['hemobwa', 'hemobsi', 'hemobch', 'hemobcs', 'hemobcl',
                  'hemobst', 'hemobre', 'hemobpu', 'hemobli', 'hemobpi']
CONDITION_VARS_LONG = ['diabetes', 'highbp', 'heart', 'stroke', 'lung', 'arthritis']


# ===========================================================================
# Data loading helpers (shared with run_medication_sensitivity.py)
# ===========================================================================
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


# ===========================================================================
# Full data loading & panel construction
# ===========================================================================
def load_all_data():
    """Load all ELSA files, build merged panel with 3-axis variables."""
    print("=" * 70)
    print("ELSA SENSITIVITY EXTRAS — DATA LOADING")
    print("=" * 70)

    files = {}
    for key, pattern in FILE_PATTERNS.items():
        path = find_file(pattern)
        if path:
            files[key] = load_tab(path)
            print(f"  {key:25s}: {os.path.basename(path)} — "
                  f"{files[key].shape[0]:>7,} rows")
        else:
            print(f"  {key:25s}: NOT FOUND")

    assert 'harmonised' in files, "Harmonised file required"
    assert 'nurse_consolidated' in files, "Nurse consolidated data required"

    # Extract nurse biomarkers
    target_cols = ['idauniq', 'hscrp', 'hba1c', 'bmival', 'sysval', 'diaval',
                   'pulval', 'chol', 'hdl', 'ldl', 'trig', 'cfib', 'hgb',
                   'mmgsd1', 'mmgsd2', 'mmgsd3', 'mmgsn1', 'mmgsn2', 'mmgsn3',
                   'wbc', 'igf1', 'htval', 'wtval', 'wstval', 'mmcrsa', 'mmcrav',
                   'hemda', 'hemdb', 'hemdab', 'statins', 'statina']

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

    grip_dom = [c for c in GRIP_COLS if c in panel.columns]
    panel['grip_max'] = panel[grip_dom].max(axis=1) if grip_dom else np.nan

    panel = panel.sort_values(['idauniq', 'wave', 'hscrp'], na_position='last')
    panel = panel.drop_duplicates(subset=['idauniq', 'wave'], keep='first')

    # Harmonised -> long
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

    print(f"\n  Panel: {len(merged):,} person-visits, "
          f"{merged['idauniq'].nunique():,} unique people")

    return merged, harm, files


# ===========================================================================
# Z-score computation against a specific reference subgroup
# ===========================================================================
def compute_3axis_zscores(merged, ref):
    """Compute 3-axis z-scores against a given reference subgroup."""
    out = merged.copy()

    # dx_I: log(CRP) z-scored
    out['log_crp'] = np.log(out['hscrp'].clip(lower=0.01))
    ref_log_crp = np.log(ref['hscrp'].clip(lower=0.01).dropna())
    if len(ref_log_crp) > 10:
        crp_mean, crp_std = ref_log_crp.mean(), ref_log_crp.std()
    else:
        crp_mean, crp_std = out['log_crp'].mean(), out['log_crp'].std()
    crp_std = max(crp_std, 1e-6)
    out['dx_I'] = (out['log_crp'] - crp_mean) / crp_std

    # dx_M: (hba1c_z + bmi_z) / sqrt(2)
    for var, label in [('hba1c', 'hba1c_z'), ('bmival', 'bmi_z')]:
        ref_vals = ref[var].dropna() if var in ref.columns else pd.Series(dtype=float)
        if len(ref_vals) > 10:
            mu, sigma = ref_vals.mean(), ref_vals.std()
        else:
            mu, sigma = out[var].mean(), out[var].std()
        sigma = max(sigma, 1e-6)
        out[label] = (out[var] - mu) / sigma

    out['dx_M'] = (out['hba1c_z'] + out['bmi_z']) / np.sqrt(2)

    # dx_F: -grip z-score (higher = worse)
    ref_grip = ref['grip_max'].dropna() if 'grip_max' in ref.columns else pd.Series(dtype=float)
    if len(ref_grip) > 10:
        grip_mean, grip_std = ref_grip.mean(), ref_grip.std()
    else:
        grip_mean, grip_std = out['grip_max'].mean(), out['grip_max'].std()
    grip_std = max(grip_std, 1e-6)
    out['dx_F'] = -(out['grip_max'] - grip_mean) / grip_std

    out['complete_3axis'] = (
        out['dx_I'].notna() & out['dx_M'].notna() & out['dx_F'].notna()
    )
    return out


# ===========================================================================
# SWDS-Γ computation (cross-sectional stratum covariance — matches R5 fix 1d)
# ===========================================================================
def compute_swds_cross_sectional(df, axes, complete_col):
    """Compute SWDS-Γ using cross-sectional stratum-level covariance."""
    complete = df[df[complete_col]].copy()

    gamma_lookup = {}
    for (lo, hi) in AGE_STRATA:
        stratum = complete[(complete['age'] >= lo) & (complete['age'] < hi)]
        if len(stratum) >= 20:
            X = stratum[axes].dropna().values
            if len(X) >= 20:
                gamma_lookup[(lo, hi)] = np.cov(X.T)

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


# ===========================================================================
# Survival time construction (matches R4/R5 exactly)
# ===========================================================================
def build_survival_baseline(df):
    """Construct baseline (wave 2) survival data matching R4 logic."""
    baseline = df[df['wave'] == 2].copy()
    baseline['deceased'] = baseline['deceased'].fillna(0).astype(int)
    baseline_year = baseline['interview_year'].fillna(NURSE_WAVE_YEARS.get(2, 2004))
    baseline['last_contact_year'] = 2024
    baseline['time'] = np.where(
        baseline['deceased'] == 1,
        baseline['death_year'].fillna(baseline['last_contact_year']) - baseline_year,
        baseline['last_contact_year'] - baseline_year
    )
    baseline = baseline[baseline['time'] > 0].copy()
    return baseline


# ===========================================================================
# Matched-sample Cox runner (same as corrected pipeline)
# ===========================================================================
def run_matched_cox(baseline, axes, swds_col='swds_gamma', model_label='3-axis',
                    custom_models=None):
    """Run 5 nested Cox models on a matched sample (same N for all)."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    base_covs = ['age', 'sex']
    adj_covs = ['smoking', 'diabetes', 'highbp']
    bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival']

    if custom_models is not None:
        models = custom_models
    else:
        models = {
            'M1: Age + Sex':     base_covs + adj_covs,
            'M2: + Biomarkers':  base_covs + bio_covs + adj_covs,
            'M3: + SWDS-G':      base_covs + [swds_col] + adj_covs,
            'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
            'M5: Full':          base_covs + bio_covs + [swds_col, 'rockwood_fi'] + adj_covs,
        }

    all_vars = set()
    for covs in models.values():
        all_vars.update(covs)
    all_vars.update(['time', 'deceased'])
    all_vars = [v for v in all_vars if v in baseline.columns]
    matched = baseline[all_vars].dropna().copy()

    # Drop near-zero-variance covariates
    dropped_covs = []
    for c in list(adj_covs):
        if c in matched.columns and matched[c].std() < 0.01:
            dropped_covs.append(c)
    if dropped_covs:
        print(f"  Dropping near-zero-variance covariates: {dropped_covs}")
        for name in models:
            models[name] = [v for v in models[name] if v not in dropped_covs]
        all_vars_new = set()
        for covs in models.values():
            all_vars_new.update(covs)
        all_vars_new.update(['time', 'deceased'])
        all_vars_new = [v for v in all_vars_new if v in baseline.columns]
        matched = baseline[all_vars_new].dropna().copy()

    n_matched = len(matched)
    n_events = int(matched['deceased'].sum())
    print(f"\n  [{model_label}] Matched sample: N={n_matched:,}, events={n_events:,}")

    if n_matched < 50 or n_events < 10:
        print(f"  SKIP: insufficient matched sample")
        return None, None

    results = {}
    fitted_models = {}
    for name, covs in models.items():
        available_covs = [c for c in covs if c in matched.columns]
        surv_data = matched[available_covs + ['time', 'deceased']].copy()
        if surv_data.isin([np.inf, -np.inf]).any().any():
            surv_data = surv_data.replace([np.inf, -np.inf], np.nan).dropna()

        try:
            cph = CoxPHFitter()
            cph.fit(surv_data, duration_col='time', event_col='deceased')
            c_idx = concordance_index(
                surv_data['time'],
                -cph.predict_partial_hazard(surv_data),
                surv_data['deceased']
            )
            results[name] = {
                'c_index': c_idx,
                'n': len(surv_data),
                'events': int(surv_data['deceased'].sum()),
                'covariates': available_covs,
            }
            fitted_models[name] = (cph, surv_data)
            print(f"    {name}: C={c_idx:.6f} (N={len(surv_data):,}, "
                  f"events={int(surv_data['deceased'].sum()):,})")
        except Exception as e:
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(surv_data, duration_col='time', event_col='deceased')
                c_idx = concordance_index(
                    surv_data['time'],
                    -cph.predict_partial_hazard(surv_data),
                    surv_data['deceased']
                )
                results[name] = {
                    'c_index': c_idx,
                    'n': len(surv_data),
                    'events': int(surv_data['deceased'].sum()),
                    'covariates': available_covs,
                    'penalized': True,
                }
                fitted_models[name] = (cph, surv_data)
                print(f"    {name}: C={c_idx:.6f} [penalized]")
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

    return results, fitted_models


# ===========================================================================
# O3: ALTERNATIVE YOUTHFUL REFERENCE SENSITIVITY
# ===========================================================================
def run_o3_reference_sensitivity(merged):
    """
    Re-run the 3-axis SWDS-Γ Cox models using age 55-60 at wave 2 as
    the youthful reference (instead of primary age 50-55).
    """
    print("\n" + "=" * 70)
    print("O3: ALTERNATIVE YOUTHFUL REFERENCE SENSITIVITY")
    print("=" * 70)

    axes_3 = ['dx_I', 'dx_M', 'dx_F']

    # --- Primary reference (50-55) ---
    print("\n  --- Primary reference: age 50-55 at wave 2 ---")
    ref_primary = merged[(merged['wave'] == 2) &
                         (merged['age'] >= 50) & (merged['age'] <= 55)]
    n_ref_primary = len(ref_primary)
    print(f"  N reference (primary): {n_ref_primary:,}")

    merged_primary = compute_3axis_zscores(merged, ref_primary)
    # Rockwood FI
    merged_primary['rockwood_fi'] = merged_primary.apply(compute_rockwood_fi, axis=1)
    complete_primary = compute_swds_cross_sectional(
        merged_primary, axes_3, 'complete_3axis')
    complete_primary['rockwood_fi'] = merged_primary.loc[
        complete_primary.index, 'rockwood_fi']
    baseline_primary = build_survival_baseline(complete_primary)
    cox_primary, _ = run_matched_cox(baseline_primary, axes_3,
                                     model_label='3-axis (ref 50-55)')

    # --- Alternative reference (55-60) ---
    print("\n  --- Alternative reference: age 55-60 at wave 2 ---")
    ref_alt = merged[(merged['wave'] == 2) &
                     (merged['age'] >= 55) & (merged['age'] <= 60)]
    n_ref_alt = len(ref_alt)
    print(f"  N reference (alternative): {n_ref_alt:,}")

    merged_alt = compute_3axis_zscores(merged, ref_alt)
    merged_alt['rockwood_fi'] = merged_alt.apply(compute_rockwood_fi, axis=1)
    complete_alt = compute_swds_cross_sectional(
        merged_alt, axes_3, 'complete_3axis')
    complete_alt['rockwood_fi'] = merged_alt.loc[complete_alt.index, 'rockwood_fi']
    baseline_alt = build_survival_baseline(complete_alt)
    cox_alt, _ = run_matched_cox(baseline_alt, axes_3,
                                 model_label='3-axis (ref 55-60)')

    # Extract results
    dc_primary = cox_primary['_delta_c'] if cox_primary else np.nan
    dc_alt = cox_alt['_delta_c'] if cox_alt else np.nan
    m5_primary = cox_primary.get('M5: Full', {}).get('c_index', np.nan) if cox_primary else np.nan
    m5_alt = cox_alt.get('M5: Full', {}).get('c_index', np.nan) if cox_alt else np.nan

    diff = abs(dc_primary - dc_alt) if not (np.isnan(dc_primary) or np.isnan(dc_alt)) else np.nan
    robust = diff < 0.002 if not np.isnan(diff) else None
    sensitive = diff > 0.005 if not np.isnan(diff) else None

    if robust is not None:
        status = "ROBUST" if robust else ("SENSITIVE" if sensitive else "MODERATE")
    else:
        status = "UNKNOWN"

    print(f"\n  YOUTHFUL REFERENCE SENSITIVITY")
    print(f"  {'':30s} {'Primary (50-55)':>18s} {'Alternative (55-60)':>20s}")
    print(f"  {'N reference:':<30s} {n_ref_primary:>18,} {n_ref_alt:>20,}")
    print(f"  {'DC(M5-M4):':<30s} {dc_primary:>+18.4f} {dc_alt:>+20.4f}")
    print(f"  {'M5 C-index:':<30s} {m5_primary:>18.4f} {m5_alt:>20.4f}")
    print(f"  Difference in DC: {diff:.4f} [{status}]")

    return {
        'n_ref_primary': int(n_ref_primary),
        'n_ref_alt': int(n_ref_alt),
        'dc_primary': float(dc_primary) if not np.isnan(dc_primary) else None,
        'dc_alt': float(dc_alt) if not np.isnan(dc_alt) else None,
        'm5_c_primary': float(m5_primary) if not np.isnan(m5_primary) else None,
        'm5_c_alt': float(m5_alt) if not np.isnan(m5_alt) else None,
        'dc_difference': float(diff) if not np.isnan(diff) else None,
        'status': status,
    }


# ===========================================================================
# O1: CALIBRATION PLOT
# ===========================================================================
def run_o1_calibration(merged):
    """
    Produce calibration plots for M4 and M5 on the 3-axis matched sample.
    Uses 5-year predicted survival probability vs observed 5-year mortality.
    """
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index

    print("\n" + "=" * 70)
    print("O1: CALIBRATION PLOT (M4 and M5)")
    print("=" * 70)

    axes_3 = ['dx_I', 'dx_M', 'dx_F']

    # Build data with primary reference
    ref = merged[(merged['wave'] == 2) &
                 (merged['age'] >= 50) & (merged['age'] <= 55)]
    merged_z = compute_3axis_zscores(merged, ref)
    merged_z['rockwood_fi'] = merged_z.apply(compute_rockwood_fi, axis=1)
    complete = compute_swds_cross_sectional(merged_z, axes_3, 'complete_3axis')
    complete['rockwood_fi'] = merged_z.loc[complete.index, 'rockwood_fi']
    baseline = build_survival_baseline(complete)

    base_covs = ['age', 'sex']
    adj_covs = ['smoking', 'diabetes', 'highbp']
    bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival']

    models_spec = {
        'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
        'M5: Full': base_covs + bio_covs + ['swds_gamma', 'rockwood_fi'] + adj_covs,
    }

    # Build matched sample for both M4 and M5
    all_vars = set()
    for covs in models_spec.values():
        all_vars.update(covs)
    all_vars.update(['time', 'deceased'])
    all_vars = [v for v in all_vars if v in baseline.columns]
    matched = baseline[all_vars].dropna().copy()

    # Drop near-zero-variance
    for c in list(adj_covs):
        if c in matched.columns and matched[c].std() < 0.01:
            for name in models_spec:
                models_spec[name] = [v for v in models_spec[name] if v != c]

    n_matched = len(matched)
    n_events = int(matched['deceased'].sum())
    print(f"  Matched sample: N={n_matched:,}, events={n_events:,}")

    T_HORIZON = 5.0  # 5-year calibration
    cal_results = {}

    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 6))

    for idx, (model_name, covs) in enumerate(models_spec.items()):
        available_covs = [c for c in covs if c in matched.columns]
        surv_data = matched[available_covs + ['time', 'deceased']].copy()
        if surv_data.isin([np.inf, -np.inf]).any().any():
            surv_data = surv_data.replace([np.inf, -np.inf], np.nan).dropna()

        try:
            cph = CoxPHFitter()
            cph.fit(surv_data, duration_col='time', event_col='deceased')
        except Exception:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(surv_data, duration_col='time', event_col='deceased')

        # Predict 5-year survival probability
        pred_surv = cph.predict_survival_function(surv_data, times=[T_HORIZON])
        pred_5yr_mort = 1.0 - pred_surv.iloc[0].values  # predicted 5-year mortality

        # Bin into deciles of predicted risk
        surv_data = surv_data.copy()
        surv_data['pred_mort'] = pred_5yr_mort
        surv_data['risk_decile'] = pd.qcut(
            surv_data['pred_mort'], 10, labels=False, duplicates='drop')

        # For each decile, compute observed 5-year mortality via KM
        calibration_data = []
        for d in sorted(surv_data['risk_decile'].unique()):
            sub = surv_data[surv_data['risk_decile'] == d]
            mean_pred = sub['pred_mort'].mean()

            # Observed: KM estimate at 5 years
            kmf = KaplanMeierFitter()
            kmf.fit(sub['time'], event_observed=sub['deceased'])
            # KM survival at T_HORIZON
            km_timeline = kmf.survival_function_at_times([T_HORIZON])
            observed_mort = 1.0 - km_timeline.iloc[0]

            calibration_data.append({
                'decile': int(d),
                'n': len(sub),
                'mean_predicted': float(mean_pred),
                'observed_km': float(observed_mort),
            })

        cal_df = pd.DataFrame(calibration_data)

        # Hosmer-Lemeshow style chi-squared
        hl_chi2 = 0.0
        for _, row in cal_df.iterrows():
            n_d = row['n']
            obs = row['observed_km']
            exp = row['mean_predicted']
            if exp > 0 and exp < 1:
                hl_chi2 += n_d * (obs - exp) ** 2 / (exp * (1 - exp))

        from scipy.stats import chi2
        hl_df = len(cal_df) - 2
        hl_p = 1.0 - chi2.cdf(hl_chi2, hl_df) if hl_df > 0 else np.nan

        cal_results[model_name] = {
            'calibration': calibration_data,
            'hl_chi2': float(hl_chi2),
            'hl_df': int(hl_df),
            'hl_p': float(hl_p) if not np.isnan(hl_p) else None,
        }

        short_name = model_name.split(':')[0]
        print(f"  {model_name}: HL chi2={hl_chi2:.2f}, df={hl_df}, p={hl_p:.4f}")

        # Plot
        ax = axes_plot[idx]
        lo = min(cal_df['mean_predicted'].min(), cal_df['observed_km'].min()) * 0.9
        hi = max(cal_df['mean_predicted'].max(), cal_df['observed_km'].max()) * 1.1
        lo = max(lo, 0)
        hi = min(hi, 1)
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='Perfect calibration')
        ax.scatter(cal_df['mean_predicted'], cal_df['observed_km'],
                   s=80, zorder=5, c='steelblue', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Mean predicted 5-year mortality (decile)')
        ax.set_ylabel('Observed 5-year mortality (KM)')
        ax.set_title(f'Calibration: {short_name}\n'
                     f'HL $\\chi^2$={hl_chi2:.1f}, p={hl_p:.3f}')
        ax.legend(fontsize=8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    plt.tight_layout()
    pdf_path = os.path.join(OUTPUT_DIR, 'figure_elsa_calibration.pdf')
    plt.savefig(pdf_path, dpi=150)
    png_path = os.path.join(OUTPUT_DIR, 'figure_elsa_calibration.png')
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")

    return cal_results


# ===========================================================================
# O2: FRAILTY-TRANSITION SECONDARY ENDPOINT
# ===========================================================================
def run_o2_frailty_transition(merged):
    """
    Construct incident frailty (FI > 0.25) as a secondary outcome.
    Run nested Cox models — excluding FI from predictors (circular).
    """
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index

    print("\n" + "=" * 70)
    print("O2: FRAILTY-TRANSITION SECONDARY ENDPOINT")
    print("=" * 70)

    axes_3 = ['dx_I', 'dx_M', 'dx_F']

    # Build z-scores with primary reference
    ref = merged[(merged['wave'] == 2) &
                 (merged['age'] >= 50) & (merged['age'] <= 55)]
    merged_z = compute_3axis_zscores(merged, ref)

    # Rockwood FI for all waves
    merged_z['rockwood_fi'] = merged_z.apply(compute_rockwood_fi, axis=1)

    # SWDS-Γ
    complete = compute_swds_cross_sectional(merged_z, axes_3, 'complete_3axis')
    complete['rockwood_fi'] = merged_z.loc[complete.index, 'rockwood_fi']

    # Baseline: wave 2, complete 3-axis, with FI ≤ 0.25 (non-frail)
    baseline = complete[(complete['wave'] == 2)].copy()
    non_frail = baseline[
        baseline['rockwood_fi'].notna() & (baseline['rockwood_fi'] <= 0.25)
    ].copy()
    non_frail_ids = set(non_frail['idauniq'].values)

    n_at_risk = len(non_frail_ids)
    print(f"  Non-frail at baseline (FI <= 0.25): {n_at_risk:,}")

    # Determine baseline age for each person
    baseline_ages = non_frail.set_index('idauniq')['age'].to_dict()

    # For each non-frail person, find first wave where FI > 0.25
    frailty_events = []
    for pid in non_frail_ids:
        person_future = merged_z[
            (merged_z['idauniq'] == pid) & (merged_z['wave'] > 2)
        ].sort_values('wave')

        event_found = False
        for _, row in person_future.iterrows():
            fi = row.get('rockwood_fi', np.nan)
            if pd.notna(fi) and fi > 0.25:
                frailty_events.append({
                    'idauniq': pid,
                    'frailty_event': 1,
                    'frailty_wave': int(row['wave']),
                    'time_to_frailty': row['age'] - baseline_ages.get(pid, row['age']),
                })
                event_found = True
                break

        if not event_found:
            # Censored: use last observed wave
            if len(person_future) > 0:
                last = person_future.iloc[-1]
                t = last['age'] - baseline_ages.get(pid, last['age'])
            else:
                t = 0
            frailty_events.append({
                'idauniq': pid,
                'frailty_event': 0,
                'frailty_wave': int(last['wave']) if len(person_future) > 0 else 2,
                'time_to_frailty': max(t, 0),
            })

    frailty_df = pd.DataFrame(frailty_events)
    frailty_df = frailty_df[frailty_df['time_to_frailty'] > 0].copy()

    n_events = int(frailty_df['frailty_event'].sum())
    n_total = len(frailty_df)
    pct = 100 * n_events / n_total if n_total > 0 else 0
    median_fu = frailty_df['time_to_frailty'].median()

    print(f"  N at risk (with follow-up > 0): {n_total:,}")
    print(f"  Events (incident frailty): {n_events:,} ({pct:.1f}%)")
    print(f"  Median follow-up: {median_fu:.1f} years")

    # Merge baseline covariates from non_frail
    covs_to_keep = [
        'idauniq', 'age', 'sex', 'smoking', 'diabetes', 'highbp',
        'log_crp', 'hba1c', 'grip_max', 'bmival', 'swds_gamma',
        'adl_count', 'iadl_count',
    ]
    # Add condition counts
    for cond in CONDITION_VARS_LONG:
        if cond in non_frail.columns:
            covs_to_keep.append(cond)

    available_covs = [c for c in covs_to_keep if c in non_frail.columns]
    baseline_covs = non_frail[available_covs].copy()

    frailty_merged = frailty_df.merge(baseline_covs, on='idauniq', how='left')

    # Create deficit composites (replacing FI to avoid circularity)
    cond_cols = [c for c in CONDITION_VARS_LONG if c in frailty_merged.columns]
    frailty_merged['chronic_count'] = frailty_merged[cond_cols].sum(axis=1)

    # Ensure adl_count and iadl_count exist
    for col in ['adl_count', 'iadl_count']:
        if col not in frailty_merged.columns:
            frailty_merged[col] = 0

    frailty_merged['deficit_composite'] = (
        frailty_merged['adl_count'].fillna(0) +
        frailty_merged['iadl_count'].fillna(0) +
        frailty_merged['chronic_count'].fillna(0)
    )

    # Define models (NO rockwood_fi — that defines the outcome)
    base_covs = ['age', 'sex']
    adj_covs = ['smoking', 'diabetes', 'highbp']
    # M4 uses deficit items instead of FI
    deficit_covs = ['adl_count', 'iadl_count', 'chronic_count']

    models = {
        'M1: Age + Sex':           base_covs + adj_covs,
        'M3: + SWDS-G':            base_covs + ['swds_gamma'] + adj_covs,
        'M4: + Deficits':          base_covs + deficit_covs + adj_covs,
        'M5: + SWDS-G + Deficits': base_covs + deficit_covs + ['swds_gamma'] + adj_covs,
    }

    # Build matched sample — deduplicate columns first
    all_vars = set()
    for covs in models.values():
        all_vars.update(covs)
    all_vars.update(['time_to_frailty', 'frailty_event'])
    all_vars = sorted(set(v for v in all_vars if v in frailty_merged.columns))
    # Guard against duplicate column names in frailty_merged
    fm = frailty_merged.loc[:, ~frailty_merged.columns.duplicated()]
    matched = fm[all_vars].dropna().copy()

    # Drop near-zero-variance
    for c in list(adj_covs) + deficit_covs:
        if c in matched.columns:
            col_std = matched[c].std()
            if isinstance(col_std, (int, float)) and col_std < 0.01:
                print(f"  Dropping near-zero-variance: {c}")
                for name in models:
                    models[name] = [v for v in models[name] if v != c]

    n_matched = len(matched)
    n_matched_events = int(matched['frailty_event'].sum())
    print(f"\n  Matched sample: N={n_matched:,}, events={n_matched_events:,}")

    if n_matched < 50 or n_matched_events < 10:
        print("  SKIP: insufficient sample for frailty transition analysis")
        return {
            'n_at_risk': int(n_at_risk),
            'n_events': n_events,
            'event_rate_pct': float(pct),
            'median_fu': float(median_fu),
            'status': 'insufficient_sample',
        }

    results = {}
    for name, covs in models.items():
        available = [c for c in covs if c in matched.columns]
        surv_data = matched[available + ['time_to_frailty', 'frailty_event']].copy()
        if surv_data.isin([np.inf, -np.inf]).any().any():
            surv_data = surv_data.replace([np.inf, -np.inf], np.nan).dropna()

        try:
            cph = CoxPHFitter()
            cph.fit(surv_data, duration_col='time_to_frailty',
                    event_col='frailty_event')
            c_idx = concordance_index(
                surv_data['time_to_frailty'],
                -cph.predict_partial_hazard(surv_data),
                surv_data['frailty_event']
            )
            results[name] = {'c_index': c_idx, 'n': len(surv_data),
                             'events': int(surv_data['frailty_event'].sum())}
            print(f"    {name}: C={c_idx:.4f} "
                  f"(N={len(surv_data):,}, events={int(surv_data['frailty_event'].sum()):,})")
        except Exception as e:
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(surv_data, duration_col='time_to_frailty',
                        event_col='frailty_event')
                c_idx = concordance_index(
                    surv_data['time_to_frailty'],
                    -cph.predict_partial_hazard(surv_data),
                    surv_data['frailty_event']
                )
                results[name] = {'c_index': c_idx, 'n': len(surv_data),
                                 'events': int(surv_data['frailty_event'].sum()),
                                 'penalized': True}
                print(f"    {name}: C={c_idx:.4f} [penalized]")
            except Exception as e2:
                print(f"    {name}: FAILED — {e2}")
                results[name] = {'c_index': np.nan, 'n': 0, 'events': 0}

    # Delta C (M5 vs M4)
    m5_c = results.get('M5: + SWDS-G + Deficits', {}).get('c_index', np.nan)
    m4_c = results.get('M4: + Deficits', {}).get('c_index', np.nan)
    dc = m5_c - m4_c if not (np.isnan(m5_c) or np.isnan(m4_c)) else np.nan

    print(f"\n  FRAILTY-TRANSITION ANALYSIS (FI > 0.25)")
    print(f"  {'Model':<28s} {'C-index':>10s} {'DC vs M4':>10s}")
    for name in models:
        c = results.get(name, {}).get('c_index', np.nan)
        if name == 'M4: + Deficits':
            dc_str = '(reference)'
        elif not np.isnan(c) and not np.isnan(m4_c):
            dc_str = f"{c - m4_c:+.4f}"
        else:
            dc_str = 'N/A'
        print(f"  {name:<28s} {c:>10.4f} {dc_str:>10s}")

    if not np.isnan(dc):
        print(f"\n  SWDS-Γ ΔC for incident frailty: {dc:+.4f}")

    output = {
        'n_at_risk': int(n_at_risk),
        'n_events': n_events,
        'event_rate_pct': float(pct),
        'median_fu': float(median_fu),
        'n_matched': n_matched,
        'n_matched_events': n_matched_events,
        'models': {},
        'delta_c_m5_m4': float(dc) if not np.isnan(dc) else None,
    }
    for name in models:
        r = results.get(name, {})
        output['models'][name] = {
            'c_index': float(r.get('c_index', np.nan)) if not np.isnan(r.get('c_index', np.nan)) else None,
            'n': r.get('n', 0),
            'events': r.get('events', 0),
        }

    # Save frailty-specific JSON
    frailty_json_path = os.path.join(OUTPUT_DIR, 'elsa_frailty_transition.json')
    with open(frailty_json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {frailty_json_path}")

    return output


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    merged, harm, files = load_all_data()

    # ===== O3: Reference Sensitivity =====
    o3_results = run_o3_reference_sensitivity(merged)

    # ===== O1: Calibration Plots =====
    o1_results = run_o1_calibration(merged)

    # ===== O2: Frailty Transition =====
    o2_results = run_o2_frailty_transition(merged)

    # ===================================================================
    # COMBINED SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("ELSA SENSITIVITY ANALYSES — SUMMARY")
    print("=" * 70)

    # O3
    dc_p = o3_results.get('dc_primary')
    dc_a = o3_results.get('dc_alt')
    dc_diff = o3_results.get('dc_difference')
    status_o3 = o3_results.get('status', 'UNKNOWN')
    print(f"\n  1. Youthful Reference Sensitivity:")
    print(f"     Primary (50-55):     DC = {dc_p:+.4f}" if dc_p is not None else
          f"     Primary (50-55):     DC = N/A")
    print(f"     Alternative (55-60): DC = {dc_a:+.4f}" if dc_a is not None else
          f"     Alternative (55-60): DC = N/A")
    print(f"     Difference: {dc_diff:.4f} [{status_o3}]" if dc_diff is not None else
          f"     Difference: N/A")

    # O1
    m4_cal = o1_results.get('M4: + Rockwood FI', {})
    m5_cal = o1_results.get('M5: Full', {})
    m5_hl = m5_cal.get('hl_chi2', np.nan)
    m5_p = m5_cal.get('hl_p', np.nan)
    m4_hl = m4_cal.get('hl_chi2', np.nan)
    m4_p = m4_cal.get('hl_p', np.nan)
    print(f"\n  2. Calibration:")
    print(f"     M5 Hosmer-Lemeshow chi2: {m5_hl:.2f} (p={m5_p:.4f})"
          if m5_p is not None else f"     M5 HL: N/A")
    print(f"     M4 Hosmer-Lemeshow chi2: {m4_hl:.2f} (p={m4_p:.4f})"
          if m4_p is not None else f"     M4 HL: N/A")
    print(f"     Figure: outputs/figure_elsa_calibration.pdf")

    # O2
    n_risk = o2_results.get('n_at_risk', 0)
    n_ev = o2_results.get('n_events', 0)
    dc_frailty = o2_results.get('delta_c_m5_m4')
    above_below = "ABOVE" if (dc_frailty is not None and dc_frailty >= 0.01) else "BELOW"
    print(f"\n  3. Frailty Transition:")
    print(f"     N at risk: {n_risk:,} | Events: {n_ev:,}")
    print(f"     SWDS-Gamma DC for incident frailty: "
          f"{dc_frailty:+.4f} [{above_below} 0.01]" if dc_frailty is not None else
          f"     SWDS-Gamma DC: N/A")

    # ===================================================================
    # Save combined JSON
    # ===================================================================
    combined = {
        'description': 'ELSA sensitivity extras (O1, O2, O3)',
        'o3_reference_sensitivity': o3_results,
        'o1_calibration': o1_results,
        'o2_frailty_transition': o2_results,
    }
    json_path = os.path.join(OUTPUT_DIR, 'elsa_sensitivity_extras.json')
    with open(json_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ===================================================================
    # Package outputs into ZIP
    # ===================================================================
    zip_files = [
        'figure_elsa_calibration.pdf',
        'figure_elsa_calibration.png',
        'elsa_frailty_transition.json',
        'elsa_sensitivity_extras.json',
    ]
    zip_path = os.path.join(OUTPUT_DIR, 'elsa_sensitivity_extras.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in zip_files:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)
                print(f"  + {fname}")
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n  Packaged -> {zip_path} ({size_mb:.2f} MB)")

    print("\n  DONE.")


if __name__ == '__main__':
    main()
