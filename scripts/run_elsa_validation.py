#!/usr/bin/env python3
"""
Phase 3: ELSA Cohort Validation Pipeline

Executes the Γ-native stability analysis on longitudinal data from the
English Longitudinal Study of Ageing (ELSA), Waves 2–11.

Requires ELSA data files in data/elsa/ (see data/elsa/README.md for access).

Usage:
    python scripts/run_elsa_validation.py

Outputs:
    outputs/figure_elsa_validation.pdf  — 6-panel validation figure
    (console)                           — summary statistics
"""

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hdr_sim.estimation import (
    compute_swds_gamma,
    compute_swds_gamma_batch,
    gamma_stability_proxy,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, 'data', 'elsa')
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# Nurse visit waves and approximate interview years
NURSE_WAVES = [2, 4, 6, 8, 11]
NURSE_WAVE_YEARS = {2: 2004, 4: 2008, 6: 2012, 8: 2016, 11: 2023}

# Age strata for cross-sectional analysis
AGE_STRATA = [(50, 59), (60, 69), (70, 79), (80, 90)]
AGE_STRATA_LABELS = ['50–59', '60–69', '70–79', '80+']

# 3-axis biomarker columns from nurse files
BIOMARKER_COLS = {
    'hscrp': 'CRP (mg/L)',
    'hba1c': 'HbA1c',
    'bmival': 'BMI (kg/m²)',
}
GRIP_COLS = ['mmgsd1', 'mmgsd2', 'mmgsd3']

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
FILE_PATTERNS = {
    'harmonised': 'gh_elsa_h_hdr_subset',
    'supplement': 'elsa_supplementary_variables',
    'eol': 'h_elsa_eol_a2',
    'nurse_w2': 'wave_2_nurse_data',
    'nurse_w4': 'wave_4_nurse_data',
    'nurse_w6': 'wave_6_elsa_nurse_data',
    # nurse_w8 SKIPPED — subset of w8w9 (half sample only, 3525 rows)
    'nurse_w8w9': 'elsa_nurse_w8w9',
    'nurse_w11': 'wave_11_elsa_nurse_data',
}


def find_file(pattern, data_dir=DATA_DIR):
    """Find a TAB file matching the pattern in data_dir."""
    for f in os.listdir(data_dir):
        if f.lower().endswith('.tab') and pattern.lower() in f.lower():
            return os.path.join(data_dir, f)
    return None


def load_tab(path):
    """Load ELSA tab file with proper missing-value and type handling."""
    df = pd.read_csv(path, sep='\t', low_memory=False,
                     na_values=[' ', '', 'NA'])
    df.columns = [c.lower() for c in df.columns]
    # Force numeric conversion on all columns (ELSA TAB files may store
    # numeric values as strings with spaces for missing)
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith('string'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def filter_missing(df, exclude_cols=None):
    """Replace ELSA missing codes (< 0) with NaN for numeric columns."""
    exclude = set(exclude_cols or [])
    for col in df.columns:
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] < 0, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# HbA1c unit conversion
# ---------------------------------------------------------------------------
def detect_and_convert_hba1c(series):
    """
    Auto-detect HbA1c units and convert to IFCC (mmol/mol).

    DCCT (%) values typically 4–14; IFCC (mmol/mol) values typically 20–130.
    Rule: median < 15 → DCCT; median >= 15 → already IFCC.
    Conversion: IFCC = (DCCT - 2.15) × 10.929
    """
    valid = series.dropna()
    if len(valid) == 0:
        return series, 'unknown'
    median_val = valid.median()
    if median_val < 15:
        # DCCT → convert to IFCC
        converted = (series - 2.15) * 10.929
        return converted, 'DCCT'
    else:
        return series.copy(), 'IFCC'


# ---------------------------------------------------------------------------
# Frailty instruments
# ---------------------------------------------------------------------------
# Individual ADL items from supplementary file
ADL_ITEMS = ['headldr', 'headlwa', 'headlba', 'headlea', 'headlbe', 'headlwc']
IADL_ITEMS = ['headlda', 'headlpr', 'headlsh', 'headlph', 'headlco',
              'headlme', 'headlho', 'headlmo']
MOBILITY_ITEMS = ['hemobwa', 'hemobsi', 'hemobch', 'hemobcs', 'hemobcl',
                  'hemobst', 'hemobre', 'hemobpu', 'hemobli', 'hemobpi']
CONDITION_VARS_WIDE = ['diabe', 'hibpe', 'hearte', 'stroke', 'lunge', 'arthre']
# Canonical names used in long format (from harmonised_to_long)
CONDITION_VARS_LONG = ['diabetes', 'highbp', 'heart', 'stroke', 'lung', 'arthritis']


def compute_rockwood_fi(row):
    """
    Compute Rockwood Frailty Index from ~35 deficit items.

    Args:
        row: a merged DataFrame row containing both harmonised (long-format
             canonical names) and supplementary columns.

    Sources:
        - Supplementary file: ADL, IADL, mobility, self-rated health, medications
        - Harmonised file: chronic conditions, CES-D (via canonical names)
    """
    deficits = 0
    total = 0

    def check(val, is_deficit):
        nonlocal deficits, total
        if pd.notna(val):
            total += 1
            if is_deficit(val):
                deficits += 1

    # --- From supplementary file (merged into row) ---
    for col in ADL_ITEMS:
        check(row.get(col), lambda v: v == 1)
    for col in IADL_ITEMS:
        check(row.get(col), lambda v: v == 1)
    for col in MOBILITY_ITEMS:
        check(row.get(col), lambda v: v == 1)

    # Self-rated health: 1=excellent ... 5=poor; deficit if >=4
    check(row.get('hehelf'), lambda v: v >= 4)

    # Medications
    check(row.get('hemda'), lambda v: v == 1)  # BP meds
    check(row.get('hemdb'), lambda v: v == 1)  # diabetes meds

    # --- From harmonised file (canonical long-format names) ---
    for cond in CONDITION_VARS_LONG:
        check(row.get(cond), lambda v: v == 1)

    # Depression: CES-D >= 4 of 8
    check(row.get('cesd'), lambda v: v >= 4)

    if total < 10:
        return np.nan
    return deficits / total


def compute_fried_phenotype(row):
    """
    Construct Fried frailty phenotype (0–5 criteria) from merged row.
    0 = robust, 1–2 = pre-frail, 3+ = frail.
    Uses canonical long-format column names.
    """
    criteria = 0
    n_valid = 0

    # 1. Weight loss
    wghtsft = row.get('weight_shift')
    if pd.notna(wghtsft):
        n_valid += 1
        if wghtsft == 1:
            criteria += 1

    # 2. Exhaustion: CES-D items psceda or pscedh (from supplementary)
    dep = row.get('psceda')
    nogo = row.get('pscedh')
    if pd.notna(dep) or pd.notna(nogo):
        n_valid += 1
        if (pd.notna(dep) and dep == 1) or (pd.notna(nogo) and nogo == 1):
            criteria += 1

    # 3. Low physical activity: heactb <= 1
    pa = row.get('heactb')
    if pd.notna(pa):
        n_valid += 1
        if pa <= 1:
            criteria += 1

    # 4. Slow walking speed: below sex-specific 20th percentile
    walk = row.get('walk_speed')
    if pd.notna(walk) and walk > 0:
        n_valid += 1

    # 5. Weak grip: below sex-specific 20th percentile
    grip = row.get('grip1')
    if pd.notna(grip) and grip > 0:
        n_valid += 1

    if n_valid < 3:
        return np.nan
    return criteria


# ---------------------------------------------------------------------------
# Step 1: Load all files
# ---------------------------------------------------------------------------
def load_all_files():
    """Load and return all ELSA data files."""
    files = {}
    print("=" * 70)
    print("ELSA Cohort Validation Pipeline — Phase 3")
    print("=" * 70)
    print("\nStep 1: Loading data files")
    print("-" * 40)

    for key, pattern in FILE_PATTERNS.items():
        path = find_file(pattern)
        if path:
            df = load_tab(path)
            files[key] = df
            print(f"  {key:15s}: {os.path.basename(path)} — "
                  f"{df.shape[0]:,} rows × {df.shape[1]:,} cols")
        else:
            print(f"  {key:15s}: NOT FOUND (looked for '{pattern}')")

    assert 'harmonised' in files, \
        "Harmonised subset (gh_elsa_h_hdr_subset.tab) is required"

    nurse_keys = [k for k in files if k.startswith('nurse_')]
    assert len(nurse_keys) >= 2, \
        f"Need >=2 nurse files, found {len(nurse_keys)}"

    return files


# ---------------------------------------------------------------------------
# Step 2: Extract and merge biomarkers
# ---------------------------------------------------------------------------
def extract_nurse_biomarkers(files):
    """
    Extract biomarkers from nurse files and merge into a long-format panel.
    Returns DataFrame with columns: idauniq, wave, hscrp, hba1c, bmival,
    grip_max, sysval, diaval, chol, hdl, ldl, trig, cfib, hgb.
    """
    print("\nStep 2: Extracting nurse biomarkers")
    print("-" * 40)

    # Map nurse file keys to wave numbers
    # nurse_w8 SKIPPED — it's a subset of w8w9 (half sample only)
    nurse_wave_map = {
        'nurse_w2': 2,
        'nurse_w4': 4,
        'nurse_w6': 6,
        'nurse_w8w9': 8,  # Combined W8+W9, treated as single wave 8 timepoint
        'nurse_w11': 11,
    }

    target_cols = ['idauniq', 'hscrp', 'hba1c', 'bmival', 'sysval', 'diaval',
                   'chol', 'hdl', 'ldl', 'trig', 'cfib', 'hgb',
                   'mmgsd1', 'mmgsd2', 'mmgsd3',
                   'mmgsn1', 'mmgsn2', 'mmgsn3',
                   'wbc', 'igf1', 'htval', 'wtval']

    all_waves = []
    hba1c_units = {}

    for key, wave in nurse_wave_map.items():
        if key not in files:
            continue

        df = files[key].copy()
        df.columns = [c.lower() for c in df.columns]

        # Select available target columns
        available = [c for c in target_cols if c in df.columns]
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            print(f"  {key} (wave {wave}): missing cols: "
                  f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

        sub = df[available].copy()
        sub = filter_missing(sub, exclude_cols=['idauniq'])

        # Map alternate BMI column name (W8/9 may use 'bmi' not 'bmival')
        if 'bmi' in df.columns and 'bmival' not in sub.columns:
            sub['bmival'] = pd.to_numeric(df['bmi'], errors='coerce')
            sub.loc[sub['bmival'] < 0, 'bmival'] = np.nan

        # HbA1c unit detection and conversion
        if 'hba1c' in sub.columns:
            sub['hba1c'], unit = detect_and_convert_hba1c(sub['hba1c'])
            hba1c_units[wave] = unit
            print(f"  {key} (wave {wave}): HbA1c detected as {unit}, "
                  f"N={sub['hba1c'].notna().sum():,}")

        # Compute grip_max (max of dominant hand trials)
        grip_dom = [c for c in GRIP_COLS if c in sub.columns]
        if grip_dom:
            sub['grip_max'] = sub[grip_dom].max(axis=1)
        else:
            sub['grip_max'] = np.nan

        sub['wave'] = wave

        # If w8w9 and w8 both loaded, prefer w8w9 (it's the merged file)
        all_waves.append(sub)

        n_complete = sub[['hscrp', 'hba1c', 'grip_max']].dropna().shape[0]
        print(f"  {key} (wave {wave}): {sub.shape[0]:,} people, "
              f"{n_complete:,} with complete 3-axis biomarkers")

    # Concatenate all waves; if w8 and w8w9 overlap, deduplicate
    panel = pd.concat(all_waves, ignore_index=True)

    # Deduplicate: keep one row per (idauniq, wave), preferring non-NaN
    panel = panel.sort_values(['idauniq', 'wave', 'hscrp'],
                              na_position='last')
    panel = panel.drop_duplicates(subset=['idauniq', 'wave'], keep='first')

    print(f"\n  Combined panel: {panel.shape[0]:,} person-visits, "
          f"{panel['idauniq'].nunique():,} unique people")
    print(f"  HbA1c units by wave: {hba1c_units}")

    return panel, hba1c_units


def prepare_harmonised(files):
    """Load and clean harmonised file, print summary, return cleaned DataFrame."""
    print("\nStep 2b: Preparing harmonised file")
    print("-" * 40)

    harm = files['harmonised'].copy()
    harm.columns = [c.lower() for c in harm.columns]
    harm = filter_missing(harm, exclude_cols=['idauniq'])

    if 'ragender' in harm.columns:
        n_male = (harm['ragender'] == 1).sum()
        n_female = (harm['ragender'] == 2).sum()
        print(f"  Sex: {n_male:,} male, {n_female:,} female")
    print(f"  Harmonised: {harm.shape[0]:,} people, {harm.shape[1]:,} columns")
    return harm


def extract_mortality(files, harm):
    """
    Construct mortality outcomes from harmonised iwstat + EOL file.

    Primary source: harmonised r{w}iwstat == 4 (deceased by wave w).
    Supplementary: EOL file for precise death age/year.
    """
    print("\nStep 2c: Constructing mortality outcomes")
    print("-" * 40)

    mort = harm[['idauniq']].copy()
    mort['deceased'] = 0
    mort['death_wave'] = np.nan
    mort['death_age'] = np.nan
    mort['death_year'] = np.nan

    # Method 1: From harmonised iwstat (value 4 = deceased by that wave)
    for w in range(3, 11):
        col = f'r{w}iwstat'
        if col in harm.columns:
            is_dead = (harm[col] == 4)
            # Only mark if not already marked at earlier wave
            newly_dead = is_dead & (mort['deceased'] == 0)
            mort.loc[newly_dead, 'deceased'] = 1
            mort.loc[newly_dead, 'death_wave'] = w

    # Method 2: From EOL file (more precise)
    if 'eol' in files:
        eol = files['eol'].copy()
        eol.columns = [c.lower() for c in eol.columns]
        eol = filter_missing(eol, exclude_cols=['idauniq'])

        if 'radage' in eol.columns:
            eol_sub = eol[['idauniq', 'radage', 'raxyear']].dropna(
                subset=['radage'])
            eol_sub = eol_sub.rename(columns={
                'radage': 'eol_death_age',
                'raxyear': 'eol_death_year'
            })
            mort = mort.merge(eol_sub, on='idauniq', how='left')

            # Use EOL data where available (more precise)
            has_eol = mort['eol_death_age'].notna()
            mort.loc[has_eol, 'deceased'] = 1
            mort.loc[has_eol, 'death_age'] = mort.loc[has_eol, 'eol_death_age']
            mort.loc[has_eol, 'death_year'] = mort.loc[has_eol, 'eol_death_year']
            mort = mort.drop(columns=['eol_death_age', 'eol_death_year'])

    n_dead = (mort['deceased'] == 1).sum()
    n_alive = (mort['deceased'] == 0).sum()
    print(f"  Deceased: {n_dead:,}, Alive/Censored: {n_alive:,}")

    return mort


def extract_supplementary(files):
    """Extract supplementary variables (long format: idauniq × wave)."""
    print("\nStep 2d: Loading supplementary file")
    print("-" * 40)

    if 'supplement' not in files:
        print("  Supplementary file not found — skipping individual ADL/mobility")
        return None

    supp = files['supplement'].copy()
    supp.columns = [c.lower() for c in supp.columns]
    supp = filter_missing(supp, exclude_cols=['idauniq', 'wave'])

    print(f"  Supplementary: {supp.shape[0]:,} rows, "
          f"{supp['idauniq'].nunique():,} unique people, "
          f"waves: {sorted(supp['wave'].unique())}")
    return supp


# ---------------------------------------------------------------------------
# Step 2e: Convert harmonised to long format
# ---------------------------------------------------------------------------
def harmonised_to_long(harm, waves=[2, 4, 6, 8]):
    """Convert wide harmonised file to long format for merging."""
    harm = harm.copy()
    harm.columns = [c.lower() for c in harm.columns]

    # Time-invariant columns
    static_cols = ['idauniq']
    for sc in ['ragender', 'raeduc_e', 'raeducl']:
        if sc in harm.columns:
            static_cols.append(sc)

    records = []
    for w in waves:
        wave_data = harm[static_cols].copy()
        wave_data['wave'] = w

        # Map wave-specific columns to canonical names
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


# ---------------------------------------------------------------------------
# Step 3: Build analysis panel
# ---------------------------------------------------------------------------
def build_analysis_panel(panel, harm_long, mort, supp):
    """
    Merge nurse biomarkers with demographics (long), mortality, supplementary.
    Construct 3-axis model variables and SWDS-Γ.
    """
    print("\nStep 3: Building analysis panel")
    print("-" * 40)

    # Merge nurse biomarkers with harmonised long-format demographics
    merged = panel.merge(harm_long, on=['idauniq', 'wave'], how='left')

    # For wave 11, age may not be in harmonised (only goes to w10)
    # Estimate from earlier waves via the wide file
    w11_mask = (merged['wave'] == 11) & merged['age'].isna()
    if w11_mask.any():
        # Get person ages from wave 8 in harm_long
        w8_ages = harm_long[harm_long['wave'] == 8][['idauniq', 'age']].rename(
            columns={'age': 'age_w8'})
        merged = merged.merge(w8_ages, on='idauniq', how='left')
        can_fill = w11_mask & merged['age_w8'].notna()
        year_diff = NURSE_WAVE_YEARS.get(11, 2023) - NURSE_WAVE_YEARS.get(8, 2016)
        merged.loc[can_fill, 'age'] = merged.loc[can_fill, 'age_w8'] + year_diff
        merged = merged.drop(columns=['age_w8'], errors='ignore')

    # Fill interview_year NaN with known wave midpoints
    merged['interview_year'] = merged['interview_year'].fillna(
        merged['wave'].map(NURSE_WAVE_YEARS))

    # Merge mortality
    merged = merged.merge(mort, on='idauniq', how='left')

    # Merge supplementary (long format, by idauniq + wave)
    if supp is not None:
        merged = merged.merge(supp, on=['idauniq', 'wave'], how='left')

    # Filter: age 50–90
    pre_filter = len(merged)
    merged = merged[(merged['age'] >= 50) & (merged['age'] <= 90)]
    print(f"  After age 50–90 filter: {len(merged):,} person-visits "
          f"(dropped {pre_filter - len(merged):,})")

    # Fill missing nurse bmival from harmonised bmi_harmonised
    if 'bmi_harmonised' in merged.columns:
        mask = merged['bmival'].isna() & merged['bmi_harmonised'].notna()
        n_filled = mask.sum()
        merged.loc[mask, 'bmival'] = merged.loc[mask, 'bmi_harmonised']
        if n_filled > 0:
            # Per-wave breakdown
            for w in sorted(merged['wave'].unique()):
                w_filled = ((merged['wave'] == w) & mask).sum()
                if w_filled > 0:
                    print(f"  Wave {w}: filled {w_filled:,} missing bmival "
                          f"from harmonised bmi_harmonised")

    # --- Construct 3-axis model ---
    print("\n  Constructing 3-axis model (I, M, F)...")

    # Reference: youngest age stratum (50–55) at wave 2 for z-score reference
    ref_mask = (merged['wave'] == 2) & (merged['age'] >= 50) & (merged['age'] <= 55)
    ref = merged.loc[ref_mask]

    # Δx_I: z-score of log(CRP)
    merged['log_crp'] = np.log(merged['hscrp'].clip(lower=0.01))
    ref_log_crp = np.log(ref['hscrp'].clip(lower=0.01).dropna())
    if len(ref_log_crp) > 10:
        crp_mean, crp_std = ref_log_crp.mean(), ref_log_crp.std()
    else:
        crp_mean = merged['log_crp'].mean()
        crp_std = merged['log_crp'].std()
    crp_std = max(crp_std, 1e-6)
    merged['dx_I'] = (merged['log_crp'] - crp_mean) / crp_std

    # Δx_M: z-score composite of HbA1c (IFCC) and BMI
    for var, label in [('hba1c', 'hba1c_z'), ('bmival', 'bmi_z')]:
        ref_vals = ref[var].dropna() if var in ref.columns else pd.Series(dtype=float)
        if len(ref_vals) > 10:
            mu, sigma = ref_vals.mean(), ref_vals.std()
        else:
            mu = merged[var].mean()
            sigma = merged[var].std()
        sigma = max(sigma, 1e-6)
        merged[label] = (merged[var] - mu) / sigma

    merged['dx_M'] = (merged['hba1c_z'] + merged['bmi_z']) / np.sqrt(2)

    # Δx_F: reversed z-score of max grip strength (higher grip = lower deficit)
    ref_grip = ref['grip_max'].dropna() if 'grip_max' in ref.columns else pd.Series(dtype=float)
    if len(ref_grip) > 10:
        grip_mean, grip_std = ref_grip.mean(), ref_grip.std()
    else:
        grip_mean = merged['grip_max'].mean()
        grip_std = merged['grip_max'].std()
    grip_std = max(grip_std, 1e-6)
    merged['dx_F'] = -(merged['grip_max'] - grip_mean) / grip_std  # reversed

    # Flag completeness
    merged['complete_3axis'] = (
        merged['dx_I'].notna() &
        merged['dx_M'].notna() &
        merged['dx_F'].notna()
    )

    n_complete = merged['complete_3axis'].sum()
    print(f"  Complete 3-axis observations: {n_complete:,} / {len(merged):,}")

    # People with >=2 complete visits (for longitudinal analysis)
    visit_counts = merged.loc[merged['complete_3axis']].groupby('idauniq').size()
    longitudinal_ids = visit_counts[visit_counts >= 2].index
    merged['in_longitudinal'] = merged['idauniq'].isin(longitudinal_ids)
    n_long = merged['in_longitudinal'].sum()
    n_long_people = len(longitudinal_ids)
    print(f"  Longitudinal panel (>=2 visits): {n_long_people:,} people, "
          f"{n_long:,} visits")

    return merged


# ---------------------------------------------------------------------------
# Step 4: Cross-sectional Γ̂ analysis
# ---------------------------------------------------------------------------
def cross_sectional_gamma(merged):
    """
    Compute cross-sectional Γ̂ per age stratum per wave.
    Returns dict of results.
    """
    print("\nStep 4: Cross-sectional Γ̂ analysis")
    print("-" * 40)

    axes = ['dx_I', 'dx_M', 'dx_F']
    results = []

    for wave in sorted(merged['wave'].unique()):
        wave_data = merged[(merged['wave'] == wave) & merged['complete_3axis']]
        if len(wave_data) < 20:
            continue

        for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
            stratum = wave_data[(wave_data['age'] >= lo) & (wave_data['age'] < hi)]
            if len(stratum) < 20:
                continue

            X = stratum[axes].values
            Gamma_hat = np.cov(X.T)

            proxy = gamma_stability_proxy(Gamma_hat)

            results.append({
                'wave': wave,
                'age_group': label,
                'age_lo': lo,
                'age_hi': hi,
                'age_mid': (lo + hi) / 2,
                'n': len(stratum),
                'lambda_max': proxy['lambda_max'],
                'lambda_min': proxy['lambda_min'],
                'kappa': proxy['kappa'],
                'trace': proxy['trace'],
                'Gamma_hat': Gamma_hat,
            })

            print(f"  Wave {wave:2d}, {label}: N={len(stratum):5,}, "
                  f"λ_max={proxy['lambda_max']:.4f}, "
                  f"κ={proxy['kappa']:.2f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 5: Within-person Γ̂ analysis
# ---------------------------------------------------------------------------
def within_person_gamma(merged):
    """
    Compute within-person covariance Γ̂_within per age stratum.
    Uses within-person residuals (removes individual fixed effects).
    """
    print("\nStep 5: Within-person Γ̂ analysis (key result)")
    print("-" * 40)

    axes = ['dx_I', 'dx_M', 'dx_F']
    long_data = merged[merged['in_longitudinal'] & merged['complete_3axis']].copy()

    # Compute within-person means
    person_means = long_data.groupby('idauniq')[axes].mean()
    person_means.columns = [f'{c}_mean' for c in axes]

    long_data = long_data.merge(person_means, on='idauniq', how='left')

    # Within-person residuals
    for ax in axes:
        long_data[f'{ax}_resid'] = long_data[ax] - long_data[f'{ax}_mean']

    resid_axes = [f'{ax}_resid' for ax in axes]

    # Determine baseline age (age at first visit)
    first_ages = long_data.groupby('idauniq')['age'].min().rename('baseline_age')
    long_data = long_data.merge(first_ages, on='idauniq', how='left')

    results = []
    for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
        stratum = long_data[
            (long_data['baseline_age'] >= lo) & (long_data['baseline_age'] < hi)
        ]
        if len(stratum) < 20:
            print(f"  {label}: insufficient data (N={len(stratum)})")
            continue

        X_resid = stratum[resid_axes].values
        Gamma_within = np.cov(X_resid.T)

        proxy = gamma_stability_proxy(Gamma_within)

        results.append({
            'age_group': label,
            'age_lo': lo,
            'age_hi': hi,
            'age_mid': (lo + hi) / 2,
            'n_visits': len(stratum),
            'n_people': stratum['idauniq'].nunique(),
            'lambda_max': proxy['lambda_max'],
            'lambda_min': proxy['lambda_min'],
            'kappa': proxy['kappa'],
            'trace': proxy['trace'],
            'Gamma_within': Gamma_within,
        })

        print(f"  {label}: {stratum['idauniq'].nunique():,} people, "
              f"{len(stratum):,} visits, "
              f"λ_max(Γ_within)={proxy['lambda_max']:.4f}")

    results_df = pd.DataFrame(results)

    # Test monotone trend
    if len(results_df) >= 3:
        tau, p = stats.kendalltau(results_df['age_mid'], results_df['lambda_max'])
        print(f"\n  Kendall τ (age vs λ_max within): τ={tau:.3f}, p={p:.4f}")
        if tau > 0 and p < 0.05:
            print("  ✓ λ_max(Γ_within) increases with age (stability erosion confirmed)")
        elif tau > 0:
            print("  ~ λ_max(Γ_within) trend positive but not significant")
        else:
            print("  ✗ λ_max(Γ_within) does not increase with age")

    return results_df


# ---------------------------------------------------------------------------
# Step 5b: Visit-pair within-person analysis
# ---------------------------------------------------------------------------
def visit_pair_gamma(merged):
    """
    Alternative within-person approach: consecutive visit-pair residuals.
    Δ(Δx) = Δx(wave k+1) - Δx(wave k) for each individual.
    """
    print("\nStep 5b: Visit-pair Δ(Δx) analysis")
    print("-" * 40)

    axes = ['dx_I', 'dx_M', 'dx_F']
    long_data = merged[merged['in_longitudinal'] & merged['complete_3axis']].copy()
    long_data = long_data.sort_values(['idauniq', 'wave'])

    # Compute consecutive differences
    diffs = []
    for uid, group in long_data.groupby('idauniq'):
        if len(group) < 2:
            continue
        for i in range(len(group) - 1):
            row_a = group.iloc[i]
            row_b = group.iloc[i + 1]
            diff = {ax: row_b[ax] - row_a[ax] for ax in axes}
            diff['idauniq'] = uid
            diff['age_mid'] = (row_a['age'] + row_b['age']) / 2
            diff['wave_from'] = row_a['wave']
            diff['wave_to'] = row_b['wave']
            diffs.append(diff)

    if not diffs:
        print("  No visit pairs found")
        return pd.DataFrame()

    diff_df = pd.DataFrame(diffs)
    print(f"  {len(diff_df):,} visit-pairs from {diff_df['idauniq'].nunique():,} people")

    results = []
    for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
        stratum = diff_df[(diff_df['age_mid'] >= lo) & (diff_df['age_mid'] < hi)]
        if len(stratum) < 20:
            continue
        X = stratum[axes].values
        Gamma_change = np.cov(X.T)
        proxy = gamma_stability_proxy(Gamma_change)
        results.append({
            'age_group': label,
            'age_mid': (lo + hi) / 2,
            'n_pairs': len(stratum),
            'lambda_max': proxy['lambda_max'],
        })
        print(f"  {label}: {len(stratum):,} pairs, "
              f"λ_max(Γ_change)={proxy['lambda_max']:.4f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 6: Compute individual SWDS-Γ
# ---------------------------------------------------------------------------
def compute_individual_swds(merged, within_results):
    """Compute SWDS-Γ for each individual at each visit."""
    print("\nStep 6: Computing individual SWDS-Γ scores")
    print("-" * 40)

    axes = ['dx_I', 'dx_M', 'dx_F']
    complete = merged[merged['complete_3axis']].copy()

    # Build stratum covariance lookup
    gamma_lookup = {}
    for _, row in within_results.iterrows():
        gamma_lookup[(row['age_lo'], row['age_hi'])] = row['Gamma_within']

    # Fallback: use cross-sectional covariance from the whole sample
    X_all = complete[axes].values
    Gamma_global = np.cov(X_all.T)

    def get_gamma(age):
        for (lo, hi), G in gamma_lookup.items():
            if lo <= age < hi:
                return G
        return Gamma_global

    # Compute SWDS-Γ per person-visit
    swds_scores = []
    for idx, row in complete.iterrows():
        dx = row[axes].values.astype(float)
        if np.any(np.isnan(dx)):
            swds_scores.append(np.nan)
            continue
        G = get_gamma(row['age'])
        score = compute_swds_gamma(dx, G)
        swds_scores.append(score)

    complete = complete.copy()
    complete['swds_gamma'] = swds_scores

    print(f"  SWDS-Γ computed for {complete['swds_gamma'].notna().sum():,} observations")
    print(f"  SWDS-Γ distribution: "
          f"mean={complete['swds_gamma'].mean():.3f}, "
          f"median={complete['swds_gamma'].median():.3f}, "
          f"SD={complete['swds_gamma'].std():.3f}")

    return complete


# ---------------------------------------------------------------------------
# Step 7: Rockwood FI computation
# ---------------------------------------------------------------------------
def compute_frailty_indices(complete):
    """Compute Rockwood FI for each person-visit.
    All needed columns (harmonised + supplementary) are already merged into complete.
    """
    print("\nStep 7: Computing Rockwood Frailty Index")
    print("-" * 40)

    # Check if supplementary columns are present
    supp_cols_present = any(col in complete.columns for col in ADL_ITEMS)
    if not supp_cols_present:
        print("  Supplementary columns not found in merged data — cannot compute full FI")
        complete = complete.copy()
        complete['rockwood_fi'] = np.nan
        return complete

    complete = complete.copy()
    complete['rockwood_fi'] = complete.apply(compute_rockwood_fi, axis=1)

    n_valid = complete['rockwood_fi'].notna().sum()
    print(f"  Rockwood FI computed for {n_valid:,} / {len(complete):,} observations")
    if n_valid > 0:
        print(f"  FI distribution: "
              f"mean={complete['rockwood_fi'].mean():.3f}, "
              f"median={complete['rockwood_fi'].median():.3f}, "
              f"SD={complete['rockwood_fi'].std():.3f}")

    return complete


# ---------------------------------------------------------------------------
# Step 8: Cox proportional hazards (5 nested models)
# ---------------------------------------------------------------------------
def run_cox_models(complete, mort, demo):
    """
    Run 5 nested Cox models for mortality prediction.
    Returns dict of model results.
    """
    print("\nStep 8: Cox proportional hazards models")
    print("-" * 40)

    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("  lifelines not installed — skipping Cox models")
        print("  Install with: pip install lifelines")
        return None

    # Use baseline (wave 2) observations only for survival analysis
    baseline = complete[complete['wave'] == 2].copy()
    if len(baseline) < 50:
        # Try earliest available wave
        earliest_wave = complete['wave'].min()
        baseline = complete[complete['wave'] == earliest_wave].copy()
        print(f"  Using wave {earliest_wave} as baseline (wave 2 had <50)")

    # Merge mortality
    baseline = baseline.merge(
        mort[['idauniq', 'deceased', 'death_year', 'death_age']],
        on='idauniq', how='left', suffixes=('', '_mort')
    )

    # Handle duplicate mortality columns from prior merge
    for col in ['deceased', 'death_year', 'death_age']:
        mort_col = f'{col}_mort'
        if mort_col in baseline.columns:
            baseline[col] = baseline[mort_col].fillna(baseline.get(col, np.nan))
            baseline = baseline.drop(columns=[mort_col])

    baseline['deceased'] = baseline['deceased'].fillna(0).astype(int)

    # Compute survival time
    baseline_year = baseline['interview_year'].fillna(
        NURSE_WAVE_YEARS.get(int(baseline['wave'].iloc[0]), 2004))

    # Find last known alive year
    last_iwy_cols = [f'iwy_w{w}' for w in range(10, 0, -1)
                     if f'iwy_w{w}' in baseline.columns]
    if last_iwy_cols:
        baseline['last_contact_year'] = baseline[last_iwy_cols].max(axis=1)
    else:
        baseline['last_contact_year'] = 2024

    baseline['time'] = np.where(
        baseline['deceased'] == 1,
        baseline['death_year'].fillna(baseline['last_contact_year']) - baseline_year,
        baseline['last_contact_year'] - baseline_year
    )

    # Filter: positive follow-up time
    baseline = baseline[baseline['time'] > 0].copy()

    print(f"  Baseline sample: {len(baseline):,}")
    print(f"  Events (deaths): {baseline['deceased'].sum():,}")
    print(f"  Median follow-up: {baseline['time'].median():.1f} years")

    if len(baseline) < 50 or baseline['deceased'].sum() < 10:
        print("  Insufficient events for Cox models")
        return None

    # Covariates now use canonical long-format names from harmonised_to_long
    # 'smoking', 'diabetes', 'highbp' are already columns in baseline
    # 'hemda', 'hemdb' come from supplementary merge

    # Base covariates
    base_covs = ['age', 'sex']

    # Individual biomarkers
    bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival', 'sysval']

    # Adjustment covariates — only include those with >=50% coverage
    adj_candidates = ['smoking', 'diabetes', 'highbp', 'hemda', 'hemdb']
    adj_covs = []
    for c in adj_candidates:
        if c in baseline.columns and baseline[c].notna().mean() > 0.5:
            adj_covs.append(c)
    if adj_covs:
        print(f"  Adjustment covariates with >50% coverage: {adj_covs}")
    else:
        print("  No adjustment covariates with >50% coverage")

    # 5 nested models
    models = {
        'M1: Age + Sex': base_covs + adj_covs,
        'M2: + Biomarkers': base_covs + bio_covs + adj_covs,
        'M3: + SWDS-Γ': base_covs + ['swds_gamma'] + adj_covs,
        'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
        'M5: Full': base_covs + bio_covs + ['swds_gamma', 'rockwood_fi'] + adj_covs,
    }

    results = {}
    for name, covs in models.items():
        all_covs = [c for c in covs if c in baseline.columns]

        # Prepare data (drop rows with NaN in required covariates)
        surv_data = baseline[all_covs + ['time', 'deceased']].dropna().copy()

        if len(surv_data) < 50 or surv_data['deceased'].sum() < 10:
            print(f"  {name}: insufficient data (N={len(surv_data)}, "
                  f"events={surv_data['deceased'].sum()})")
            results[name] = {'c_index': np.nan, 'n': len(surv_data)}
            continue

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
                'aic': cph.AIC_partial_,
                'model': cph,
            }

            print(f"  {name}: C-index={c_idx:.4f}, N={len(surv_data):,}, "
                  f"events={int(surv_data['deceased'].sum()):,}")

        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            results[name] = {'c_index': np.nan, 'n': len(surv_data)}

    # Delta C and LRT
    if (results.get('M5: Full', {}).get('c_index') and
            results.get('M4: + Rockwood FI', {}).get('c_index')):
        dc = (results['M5: Full']['c_index'] -
              results['M4: + Rockwood FI']['c_index'])
        print(f"\n  ΔC(M5 vs M4) = {dc:+.4f}")
        if dc >= 0.01:
            print("  ✓ SWDS-Γ adds >=0.01 C-index over Rockwood FI")
        else:
            print(f"  ~ SWDS-Γ adds {dc:+.4f} C-index (below 0.01 threshold)")

    return results


# ---------------------------------------------------------------------------
# Step 9: Figures
# ---------------------------------------------------------------------------
def make_figures(cross_results, within_results, visit_pair_results,
                 complete, cox_results):
    """Generate 6-panel figure."""
    print("\nStep 9: Generating figures")
    print("-" * 40)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ELSA Cohort Validation — Γ-Native Pipeline (Phase 3)',
                 fontsize=14, fontweight='bold')

    # Panel (a): Cross-sectional λ_max by age stratum
    ax = axes[0, 0]
    if len(cross_results) > 0:
        # Average across waves per age group
        cs_agg = cross_results.groupby('age_group').agg(
            lambda_max_mean=('lambda_max', 'mean'),
            lambda_max_se=('lambda_max', 'sem'),
            age_mid=('age_mid', 'first'),
        ).reset_index()
        cs_agg = cs_agg.sort_values('age_mid')

        ax.errorbar(cs_agg['age_mid'], cs_agg['lambda_max_mean'],
                     yerr=cs_agg['lambda_max_se'],
                     fmt='o-', capsize=4, color='steelblue', linewidth=2)
        ax.set_xlabel('Age stratum midpoint')
        ax.set_ylabel('λ_max(Γ̂)')
        ax.set_title('(a) Cross-sectional λ_max(Γ̂)')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (b): Within-person λ_max by age stratum
    ax = axes[0, 1]
    if len(within_results) > 0:
        wr = within_results.sort_values('age_mid')
        ax.plot(wr['age_mid'], wr['lambda_max'], 'o-',
                color='darkred', linewidth=2, markersize=8)
        ax.set_xlabel('Baseline age stratum midpoint')
        ax.set_ylabel('λ_max(Γ̂_within)')
        ax.set_title('(b) Within-person λ_max(Γ̂_within)')

        # Add N annotation
        for _, row in wr.iterrows():
            ax.annotate(f"N={row['n_people']:,}",
                        (row['age_mid'], row['lambda_max']),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha='center')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (c): SWDS-Γ distribution by age stratum
    ax = axes[0, 2]
    if complete is not None and 'swds_gamma' in complete.columns:
        for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
            sub = complete[(complete['age'] >= lo) & (complete['age'] < hi)
                           & complete['swds_gamma'].notna()]
            if len(sub) > 10:
                ax.hist(sub['swds_gamma'], bins=30, alpha=0.5, label=label,
                        density=True)
        ax.set_xlabel('SWDS-Γ')
        ax.set_ylabel('Density')
        ax.set_title('(c) SWDS-Γ by age stratum')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (d): SWDS-Γ vs age scatter
    ax = axes[1, 0]
    if complete is not None and 'swds_gamma' in complete.columns:
        valid = complete[complete['swds_gamma'].notna()]
        ax.scatter(valid['age'], valid['swds_gamma'], alpha=0.1, s=5,
                   color='grey')

        # Stratum means
        for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
            sub = valid[(valid['age'] >= lo) & (valid['age'] < hi)]
            if len(sub) > 0:
                ax.plot((lo + hi) / 2, sub['swds_gamma'].mean(), 'ro',
                        markersize=10, zorder=5)

        ax.set_xlabel('Age')
        ax.set_ylabel('SWDS-Γ')
        ax.set_title('(d) SWDS-Γ vs age')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (e): C-index comparison
    ax = axes[1, 1]
    if cox_results:
        model_names = list(cox_results.keys())
        c_indices = [cox_results[m].get('c_index', np.nan) for m in model_names]
        valid_mask = [not np.isnan(c) for c in c_indices]
        model_names = [m for m, v in zip(model_names, valid_mask) if v]
        c_indices = [c for c, v in zip(c_indices, valid_mask) if v]

        if model_names:
            short_names = [m.split(':')[0] for m in model_names]
            bars = ax.barh(range(len(short_names)), c_indices,
                           color='steelblue', alpha=0.8)
            ax.set_yticks(range(len(short_names)))
            ax.set_yticklabels(short_names, fontsize=9)
            ax.set_xlabel('C-index')
            ax.set_title('(e) Cox model C-indices')
            ax.set_xlim(0.5, max(c_indices) + 0.05 if c_indices else 1.0)
            ax.axvline(0.5, color='grey', linestyle='--', alpha=0.5)

            for i, (bar, c) in enumerate(zip(bars, c_indices)):
                ax.text(c + 0.005, i, f'{c:.3f}', va='center', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Cox results', ha='center', va='center',
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'lifelines not installed', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (f): Kaplan-Meier by SWDS-Γ tertile
    ax = axes[1, 2]
    try:
        from lifelines import KaplanMeierFitter

        if (complete is not None and 'swds_gamma' in complete.columns
                and 'time' in complete.columns):
            surv_data = complete[
                complete['swds_gamma'].notna() &
                complete['time'].notna() &
                (complete['time'] > 0)
            ].copy()

            if len(surv_data) > 50 and surv_data['deceased'].sum() > 10:
                tertiles = pd.qcut(surv_data['swds_gamma'], 3,
                                   labels=['T1 (low)', 'T2 (mid)', 'T3 (high)'])
                surv_data['tertile'] = tertiles

                colors = ['forestgreen', 'orange', 'crimson']
                kmf = KaplanMeierFitter()
                for i, (label, color) in enumerate(
                        zip(['T1 (low)', 'T2 (mid)', 'T3 (high)'], colors)):
                    mask = surv_data['tertile'] == label
                    if mask.sum() > 5:
                        kmf.fit(surv_data.loc[mask, 'time'],
                                surv_data.loc[mask, 'deceased'],
                                label=label)
                        kmf.plot_survival_function(ax=ax, color=color,
                                                   linewidth=2)

                ax.set_xlabel('Years from baseline')
                ax.set_ylabel('Survival probability')
                ax.set_title('(f) KM by SWDS-Γ tertile')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'Insufficient events', ha='center',
                        va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No survival data', ha='center', va='center',
                    transform=ax.transAxes)
    except ImportError:
        ax.text(0.5, 0.5, 'lifelines not installed', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(OUTPUT_DIR, 'figure_elsa_validation.pdf')
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Step 10: Summary statistics
# ---------------------------------------------------------------------------
def print_summary(merged, cross_results, within_results, complete, cox_results):
    """Print comprehensive summary block."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # N per wave
    print("\n--- Sample sizes per wave ---")
    for wave in sorted(merged['wave'].unique()):
        w_data = merged[merged['wave'] == wave]
        n_total = len(w_data)
        n_complete = w_data['complete_3axis'].sum()
        print(f"  Wave {wave:2d}: {n_total:6,} total, "
              f"{n_complete:6,} with complete 3-axis biomarkers")

    # Longitudinal panel
    long = merged[merged['in_longitudinal']]
    print(f"\n  Longitudinal panel (>=2 complete visits): "
          f"{long['idauniq'].nunique():,} people")

    # Biomarker distributions per wave
    print("\n--- Biomarker distributions (mean ± SD) ---")
    for wave in sorted(merged['wave'].unique()):
        w_data = merged[merged['wave'] == wave]
        vals = []
        for col, label in [('hscrp', 'CRP'), ('hba1c', 'HbA1c(IFCC)'),
                           ('grip_max', 'Grip'), ('bmival', 'BMI')]:
            if col in w_data.columns:
                v = w_data[col].dropna()
                if len(v) > 0:
                    vals.append(f"{label}={v.mean():.1f}±{v.std():.1f}")
        if vals:
            print(f"  Wave {wave:2d}: {', '.join(vals)}")

    # λ_max trends
    if len(cross_results) > 0:
        print("\n--- Cross-sectional λ_max(Γ̂) trend ---")
        cs_agg = cross_results.groupby('age_group')['lambda_max'].mean()
        for ag in AGE_STRATA_LABELS:
            if ag in cs_agg.index:
                print(f"  {ag}: λ_max = {cs_agg[ag]:.4f}")

    if len(within_results) > 0:
        print("\n--- Within-person λ_max(Γ̂_within) trend ---")
        for _, row in within_results.sort_values('age_mid').iterrows():
            print(f"  {row['age_group']}: λ_max = {row['lambda_max']:.4f} "
                  f"(N={row['n_people']:,})")

    # SWDS-Γ
    if complete is not None and 'swds_gamma' in complete.columns:
        print("\n--- SWDS-Γ distribution ---")
        s = complete['swds_gamma'].dropna()
        print(f"  N = {len(s):,}")
        print(f"  Mean = {s.mean():.4f}, Median = {s.median():.4f}, "
              f"SD = {s.std():.4f}")
        print(f"  Range: [{s.min():.4f}, {s.max():.4f}]")

    # Cox results
    if cox_results:
        print("\n--- Cox model C-indices ---")
        for name, res in cox_results.items():
            c = res.get('c_index', np.nan)
            n = res.get('n', 0)
            ev = res.get('events', 0)
            if not np.isnan(c):
                print(f"  {name}: C = {c:.4f} (N={n:,}, events={ev:,})")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct_survival_data(harm, eol, baseline_wave=2):
    """
    Construct survival time and event indicator for Cox models.

    Primary source: harmonised iwstat (4 = deceased)
    Supplementary: EOL file for precise death age/year
    """
    print("\nStep 8a: Constructing survival data")
    print("-" * 40)

    baseline_year_col = f'r{baseline_wave}iwy'
    baseline_age_col = f'r{baseline_wave}agey'

    records = []
    for _, row in harm.iterrows():
        pid = row['idauniq']

        baseline_year = row.get(baseline_year_col)
        baseline_age = row.get(baseline_age_col)
        if pd.isna(baseline_year) or pd.isna(baseline_age):
            continue
        if baseline_year <= 0 or baseline_age <= 0:
            continue

        event = 0
        last_year = baseline_year

        for w in range(baseline_wave + 1, 11):
            iwstat_col = f'r{w}iwstat'
            iwy_col = f'r{w}iwy'

            iwstat = row.get(iwstat_col)
            iwy = row.get(iwy_col)

            if pd.isna(iwstat):
                continue

            if iwstat == 4:  # deceased
                event = 1
                if pd.notna(iwy) and iwy > 0:
                    last_year = iwy
                break
            elif iwstat == 1:  # alive, interviewed
                if pd.notna(iwy) and iwy > 0:
                    last_year = iwy

        time_years = last_year - baseline_year
        if time_years <= 0:
            time_years = 0.5  # minimum follow-up for baseline-only

        records.append({
            'idauniq': pid,
            'time_years': time_years,
            'event': event,
            'age_at_baseline': baseline_age,
        })

    surv = pd.DataFrame(records)

    # Supplement with EOL for precise death year where available
    if eol is not None and 'raxyear' in eol.columns:
        eol_info = eol[['idauniq', 'radage', 'raxyear']].copy()
        eol_info = eol_info[eol_info['idauniq'] > 0]
        surv = surv.merge(eol_info, on='idauniq', how='left')

        has_eol = surv['raxyear'].notna() & (surv['raxyear'] > 0) & (surv['event'] == 1)
        if has_eol.any():
            # Compute baseline year for each person
            bly = harm.set_index('idauniq')[baseline_year_col]
            surv['_bly'] = surv['idauniq'].map(bly)
            refine_mask = has_eol & surv['_bly'].notna()
            surv.loc[refine_mask, 'time_years'] = (
                surv.loc[refine_mask, 'raxyear'] - surv.loc[refine_mask, '_bly'])
            print(f"  Refined {refine_mask.sum()} death times from EOL file")
            surv = surv.drop(columns=['_bly'], errors='ignore')

        surv = surv.drop(columns=['radage', 'raxyear'], errors='ignore')

    # Ensure positive follow-up
    surv = surv[surv['time_years'] > 0]

    print(f"  Survival data: {len(surv):,} persons, {surv['event'].sum():,} events "
          f"({surv['event'].mean()*100:.1f}%), "
          f"median follow-up {surv['time_years'].median():.1f} years")

    return surv


# ---------------------------------------------------------------------------
# Data quality diagnostics (Patch 7)
# ---------------------------------------------------------------------------
def print_data_quality(merged):
    """Print data quality diagnostics after all merges."""
    print("\n" + "=" * 70)
    print("DATA QUALITY DIAGNOSTICS")
    print("=" * 70)

    # Axis completeness by wave
    for w in sorted(merged['wave'].unique()):
        wdata = merged[merged['wave'] == w]
        n_total = len(wdata)
        n_crp = wdata['hscrp'].notna().sum()
        n_hba1c = wdata['hba1c'].notna().sum()
        n_grip = wdata['grip_max'].notna().sum()
        n_bmi = wdata['bmival'].notna().sum()
        n_3axis = wdata['complete_3axis'].sum() if 'complete_3axis' in wdata.columns else 0
        print(f"  Wave {w}: N={n_total:,}, CRP={n_crp:,}, HbA1c={n_hba1c:,}, "
              f"grip={n_grip:,}, BMI={n_bmi:,}, complete 3-axis={n_3axis:,}")

    # Longitudinal panel
    if 'complete_3axis' in merged.columns:
        complete_visits = merged[merged['complete_3axis']].groupby('idauniq')['wave'].nunique()
        for nv in [1, 2, 3, 4]:
            print(f"  Persons with >={nv} complete waves: {(complete_visits >= nv).sum():,}")


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

    # Step 1: Load
    files = load_all_files()

    # Step 2: Extract
    panel, hba1c_units = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)

    # Step 2e: Convert harmonised to long format
    print("\nStep 2e: Converting harmonised to long format")
    print("-" * 40)
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    print(f"  Long format: {harm_long.shape[0]:,} rows "
          f"({harm_long['idauniq'].nunique():,} people x "
          f"{harm_long['wave'].nunique()} waves)")

    # Step 3: Build panel
    merged = build_analysis_panel(panel, harm_long, mort, supp)

    # Data quality diagnostics
    print_data_quality(merged)

    # Step 4: Cross-sectional Gamma
    cross_results = cross_sectional_gamma(merged)

    # Step 5: Within-person Gamma
    within_results = within_person_gamma(merged)

    # Step 5b: Visit-pair analysis
    visit_pair_results = visit_pair_gamma(merged)

    # Step 6: Individual SWDS-Gamma
    complete = compute_individual_swds(merged, within_results)

    # Step 7: Rockwood FI
    complete = compute_frailty_indices(complete)

    # Step 8a: Construct proper survival data from harmonised iwstat
    eol = files.get('eol')
    if eol is not None:
        eol_clean = eol.copy()
        eol_clean.columns = [c.lower() for c in eol_clean.columns]
        eol_clean = filter_missing(eol_clean, exclude_cols=['idauniq'])
    else:
        eol_clean = None
    surv = construct_survival_data(harm, eol_clean, baseline_wave=2)

    # Merge survival data into complete
    complete = complete.merge(surv[['idauniq', 'time_years', 'event']],
                              on='idauniq', how='left')
    complete['time'] = complete['time_years']

    # Step 8b: Cox models
    cox_results = run_cox_models(complete, mort, harm)

    # Step 9: Figures
    make_figures(cross_results, within_results, visit_pair_results,
                 complete, cox_results)

    # Step 10: Summary
    print_summary(merged, cross_results, within_results, complete, cox_results)


if __name__ == '__main__':
    main()
