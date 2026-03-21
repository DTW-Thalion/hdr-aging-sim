#!/usr/bin/env python3
"""
Phase 3: ELSA Cohort Validation Pipeline

Executes the Γ-native stability analysis on longitudinal data from the
English Longitudinal Study of Ageing (ELSA), Waves 2–11.

Runs BOTH a 3-axis model (I, M, F) and a 4-axis model (I, M, N, F)
with enriched biomarker composites, plus benchmark comparisons.

Requires ELSA data files in data/elsa/ (see data/elsa/README.md for access).

Usage:
    python scripts/run_elsa_validation.py

Outputs:
    outputs/figure_elsa_validation_3axis.pdf  — 6-panel validation figure (3-axis)
    outputs/figure_elsa_validation_4axis.pdf  — 6-panel validation figure (4-axis)
    outputs/figure_elsa_4axis_comparison.pdf  — 3-axis vs 4-axis comparison
    outputs/elsa_4axis_results.json           — machine-readable results ledger
    (console)                                 — summary statistics
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

# Bootstrap configuration
N_BOOTSTRAP = 1000

# Model definitions
MODELS = {
    '3-axis': {
        'axes': ['dx_I', 'dx_M', 'dx_F'],
        'complete_col': 'complete_3axis',
        'label': 'I, M, F (3-axis)',
    },
    '4-axis': {
        'axes': ['dx_I_4', 'dx_M_4', 'dx_N_4', 'dx_F_4'],
        'complete_col': 'complete_4axis',
        'label': 'I, M, N, F (4-axis)',
    },
}

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
FILE_PATTERNS = {
    'harmonised': 'gh_elsa_h_hdr_subset',
    'supplement': 'elsa_supplementary_variables',
    'eol': 'h_elsa_eol_a2',
    'nurse_consolidated': 'elsa_nurse_biomarkers_consolidated',
    # Legacy nurse file patterns (fallback if consolidated not found)
    'nurse_w2': 'wave_2_nurse_data',
    'nurse_w4': 'wave_4_nurse_data',
    'nurse_w6': 'wave_6_elsa_nurse_data',
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
# Z-score helper
# ---------------------------------------------------------------------------
def zscore_vs_ref(series, ref_df, col_name):
    """Z-score a series against the youthful reference subgroup."""
    ref_vals = ref_df[col_name].dropna() if col_name in ref_df.columns else pd.Series(dtype=float)
    if len(ref_vals) > 10:
        mu, sigma = ref_vals.mean(), ref_vals.std()
    else:
        mu = series.mean()
        sigma = series.std()
    sigma = max(sigma, 1e-6)
    return (series - mu) / sigma


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
    print("ELSA Cohort Validation Pipeline — Phase 3 (3-axis + 4-axis)")
    print("=" * 70)
    print("\nStep 1: Loading data files")
    print("-" * 40)

    for key, pattern in FILE_PATTERNS.items():
        path = find_file(pattern)
        if path:
            df = load_tab(path)
            files[key] = df
            print(f"  {key:20s}: {os.path.basename(path)} — "
                  f"{df.shape[0]:,} rows × {df.shape[1]:,} cols")
        else:
            # Only warn for required files, not legacy nurse files when
            # consolidated is present
            if key.startswith('nurse_w') and 'nurse_consolidated' in files:
                pass  # consolidated available, legacy not needed
            elif key == 'nurse_consolidated':
                print(f"  {key:20s}: NOT FOUND (will try legacy nurse files)")
            else:
                print(f"  {key:20s}: NOT FOUND (looked for '{pattern}')")

    assert 'harmonised' in files, \
        "Harmonised subset (gh_elsa_h_hdr_subset.tab) is required"

    # Check we have nurse data (consolidated or legacy)
    if 'nurse_consolidated' in files:
        print("\n  Using consolidated nurse file (all waves)")
    else:
        nurse_keys = [k for k in files if k.startswith('nurse_w')]
        assert len(nurse_keys) >= 2, \
            f"Need consolidated nurse file or >=2 legacy nurse files, found {len(nurse_keys)}"
        print(f"\n  Using {len(nurse_keys)} legacy nurse files")

    return files


# ---------------------------------------------------------------------------
# Step 2: Extract and merge biomarkers
# ---------------------------------------------------------------------------
def extract_nurse_biomarkers(files):
    """
    Extract biomarkers from nurse files and merge into a long-format panel.
    Supports both the consolidated file and legacy per-wave files.
    Returns DataFrame with columns: idauniq, wave, hscrp, hba1c, bmival,
    grip_max, sysval, diaval, pulval, chol, hdl, ldl, trig, cfib, hgb,
    wstval, mmcrsa, mmcrav, wbc, igf1, hemda, hemdb, hemdab.
    """
    print("\nStep 2: Extracting nurse biomarkers")
    print("-" * 40)

    target_cols = ['idauniq', 'hscrp', 'hba1c', 'bmival', 'sysval', 'diaval',
                   'pulval', 'chol', 'hdl', 'ldl', 'trig', 'cfib', 'hgb',
                   'mmgsd1', 'mmgsd2', 'mmgsd3',
                   'mmgsn1', 'mmgsn2', 'mmgsn3',
                   'wbc', 'igf1', 'htval', 'wtval',
                   'wstval', 'mmcrsa', 'mmcrav',
                   'hemda', 'hemdb', 'hemdab',
                   'statins', 'statina']

    hba1c_units = {}

    if 'nurse_consolidated' in files:
        # --- Consolidated file path ---
        df = files['nurse_consolidated'].copy()
        df = filter_missing(df, exclude_cols=['idauniq', 'wave'])

        # Select available target columns + wave
        available = [c for c in target_cols if c in df.columns]
        missing_cols = [c for c in target_cols if c not in df.columns and c != 'idauniq']
        if missing_cols:
            print(f"  Consolidated file missing cols: "
                  f"{', '.join(missing_cols[:8])}{'...' if len(missing_cols) > 8 else ''}")

        panel = df[available + ['wave']].copy()

        # Map alternate BMI column name
        if 'bmi' in df.columns and 'bmival' not in panel.columns:
            panel['bmival'] = pd.to_numeric(df['bmi'], errors='coerce')
            panel.loc[panel['bmival'] < 0, 'bmival'] = np.nan

        # HbA1c unit detection and conversion per wave
        for w in sorted(panel['wave'].unique()):
            w_mask = panel['wave'] == w
            if 'hba1c' in panel.columns and panel.loc[w_mask, 'hba1c'].notna().any():
                converted, unit = detect_and_convert_hba1c(panel.loc[w_mask, 'hba1c'])
                panel.loc[w_mask, 'hba1c'] = converted
                hba1c_units[int(w)] = unit
                n_valid = panel.loc[w_mask, 'hba1c'].notna().sum()
                print(f"  Wave {int(w)}: HbA1c detected as {unit}, N={n_valid:,}")

        # Compute grip_max (max of dominant hand trials)
        grip_dom = [c for c in GRIP_COLS if c in panel.columns]
        if grip_dom:
            panel['grip_max'] = panel[grip_dom].max(axis=1)
        else:
            panel['grip_max'] = np.nan

        # Report per-wave completeness
        for w in sorted(panel['wave'].unique()):
            wdata = panel[panel['wave'] == w]
            n_complete = wdata[['hscrp', 'hba1c', 'grip_max']].dropna().shape[0]
            print(f"  Wave {int(w)}: {wdata.shape[0]:,} people, "
                  f"{n_complete:,} with complete 3-axis biomarkers")

    else:
        # --- Legacy per-wave file path ---
        nurse_wave_map = {
            'nurse_w2': 2,
            'nurse_w4': 4,
            'nurse_w6': 6,
            'nurse_w8w9': 8,
            'nurse_w11': 11,
        }

        all_waves = []
        for key, wave in nurse_wave_map.items():
            if key not in files:
                continue

            df = files[key].copy()
            df.columns = [c.lower() for c in df.columns]

            available = [c for c in target_cols if c in df.columns]
            missing = [c for c in target_cols if c not in df.columns]
            if missing:
                print(f"  {key} (wave {wave}): missing cols: "
                      f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

            sub = df[available].copy()
            sub = filter_missing(sub, exclude_cols=['idauniq'])

            if 'bmi' in df.columns and 'bmival' not in sub.columns:
                sub['bmival'] = pd.to_numeric(df['bmi'], errors='coerce')
                sub.loc[sub['bmival'] < 0, 'bmival'] = np.nan

            if 'hba1c' in sub.columns:
                sub['hba1c'], unit = detect_and_convert_hba1c(sub['hba1c'])
                hba1c_units[wave] = unit
                print(f"  {key} (wave {wave}): HbA1c detected as {unit}, "
                      f"N={sub['hba1c'].notna().sum():,}")

            grip_dom = [c for c in GRIP_COLS if c in sub.columns]
            if grip_dom:
                sub['grip_max'] = sub[grip_dom].max(axis=1)
            else:
                sub['grip_max'] = np.nan

            sub['wave'] = wave
            all_waves.append(sub)

            n_complete = sub[['hscrp', 'hba1c', 'grip_max']].dropna().shape[0]
            print(f"  {key} (wave {wave}): {sub.shape[0]:,} people, "
                  f"{n_complete:,} with complete 3-axis biomarkers")

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
    Construct BOTH 3-axis and 4-axis model variables.
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
            for w in sorted(merged['wave'].unique()):
                w_filled = ((merged['wave'] == w) & mask).sum()
                if w_filled > 0:
                    print(f"  Wave {w}: filled {w_filled:,} missing bmival "
                          f"from harmonised bmi_harmonised")

    # --- Reference subgroup for z-scoring ---
    ref_mask = (merged['wave'] == 2) & (merged['age'] >= 50) & (merged['age'] <= 55)
    ref = merged.loc[ref_mask]

    # =====================================================================
    # 3-AXIS MODEL (I, M, F) — preserved from original
    # =====================================================================
    print("\n  Constructing 3-axis model (I, M, F)...")

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

    # =====================================================================
    # 4-AXIS MODEL (I, M, N, F) — NEW
    # =====================================================================
    print("\n  Constructing 4-axis model (I, M, N, F)...")

    # Δx_I (4-axis): log(CRP) + fibrinogen composite
    merged['cfib_z'] = zscore_vs_ref(merged.get('cfib', pd.Series(dtype=float)), ref, 'cfib')
    merged['dx_I_4'] = merged[['dx_I', 'cfib_z']].mean(axis=1, skipna=True)
    # Fall back to dx_I if fibrinogen missing
    merged.loc[merged['dx_I_4'].isna() & merged['dx_I'].notna(), 'dx_I_4'] = \
        merged.loc[merged['dx_I_4'].isna() & merged['dx_I'].notna(), 'dx_I']

    # Δx_M (4-axis): HbA1c + total/HDL ratio + triglycerides composite
    if 'chol' in merged.columns and 'hdl' in merged.columns:
        merged['chol_hdl_ratio'] = merged['chol'] / merged['hdl'].clip(lower=0.1)
        merged['chol_hdl_ratio_z'] = zscore_vs_ref(merged['chol_hdl_ratio'], ref, 'chol_hdl_ratio')
    else:
        merged['chol_hdl_ratio'] = np.nan
        merged['chol_hdl_ratio_z'] = np.nan

    if 'trig' in merged.columns:
        merged['log_trig'] = np.log(merged['trig'].clip(lower=0.01))
        merged['log_trig_z'] = zscore_vs_ref(merged['log_trig'], ref, 'log_trig')
    else:
        merged['log_trig'] = np.nan
        merged['log_trig_z'] = np.nan

    # Waist circumference (cap at 180 to remove 999.9 error codes)
    if 'wstval' in merged.columns:
        n_wst_flagged = (merged['wstval'] > 180).sum()
        merged.loc[merged['wstval'] > 180, 'wstval'] = np.nan
        if n_wst_flagged > 0:
            print(f"  Flagged {n_wst_flagged} wstval > 180 → NaN (error codes)")
        merged['wstval_z'] = zscore_vs_ref(merged['wstval'], ref, 'wstval')
    else:
        merged['wstval_z'] = np.nan

    # Composite: average of available z-scores (HbA1c_z already exists from 3-axis)
    m_components = ['hba1c_z', 'chol_hdl_ratio_z', 'log_trig_z']
    merged['dx_M_4'] = merged[m_components].mean(axis=1, skipna=True)

    # Δx_N (4-axis): SBP + DBP + pulse composite
    if 'sysval' in merged.columns:
        merged['sysval_z'] = zscore_vs_ref(merged['sysval'], ref, 'sysval')
    else:
        merged['sysval_z'] = np.nan
    if 'diaval' in merged.columns:
        merged['diaval_z'] = zscore_vs_ref(merged['diaval'], ref, 'diaval')
    else:
        merged['diaval_z'] = np.nan
    if 'pulval' in merged.columns:
        merged['pulval_z'] = zscore_vs_ref(merged['pulval'], ref, 'pulval')
    else:
        merged['pulval_z'] = np.nan

    n_components = ['sysval_z', 'diaval_z', 'pulval_z']
    merged['dx_N_4'] = merged[n_components].mean(axis=1, skipna=True)

    # Δx_F (4-axis): grip strength + gait speed composite
    # Grip: REVERSE coded (higher grip = healthier, so negate z-score)
    merged['grip_z_rev'] = -zscore_vs_ref(merged['grip_max'], ref, 'grip_max')
    # Gait speed: REVERSE coded (higher speed = healthier)
    if 'walk_speed' in merged.columns:
        merged['walk_z_rev'] = -zscore_vs_ref(merged['walk_speed'], ref, 'walk_speed')
    else:
        merged['walk_z_rev'] = np.nan
    f_components = ['grip_z_rev', 'walk_z_rev']
    merged['dx_F_4'] = merged[f_components].mean(axis=1, skipna=True)

    # Complete 4-axis flag
    merged['complete_4axis'] = (
        merged['dx_I_4'].notna() &
        merged['dx_M_4'].notna() &
        merged['dx_N_4'].notna() &
        merged['dx_F_4'].notna()
    )

    n_complete_4 = merged['complete_4axis'].sum()
    n_persons_4 = merged.loc[merged['complete_4axis'], 'idauniq'].nunique()
    print(f"  4-axis complete: {n_complete_4:,} person-visits, "
          f"{n_persons_4:,} unique persons")

    # Per-wave 4-axis completeness
    for w in sorted(merged['wave'].unique()):
        wdata = merged[merged['wave'] == w]
        n4 = wdata['complete_4axis'].sum()
        print(f"    Wave {w}: {n4:,} complete 4-axis")

    # People with >=2 complete visits (for longitudinal analysis)
    visit_counts = merged.loc[merged['complete_3axis']].groupby('idauniq').size()
    longitudinal_ids = visit_counts[visit_counts >= 2].index
    merged['in_longitudinal'] = merged['idauniq'].isin(longitudinal_ids)
    n_long = merged['in_longitudinal'].sum()
    n_long_people = len(longitudinal_ids)
    print(f"  Longitudinal panel (>=2 visits): {n_long_people:,} people, "
          f"{n_long:,} visits")

    # 4-axis longitudinal
    visit_counts_4 = merged.loc[merged['complete_4axis']].groupby('idauniq').size()
    longitudinal_ids_4 = visit_counts_4[visit_counts_4 >= 2].index
    merged['in_longitudinal_4'] = merged['idauniq'].isin(longitudinal_ids_4)
    n_long_4 = merged['in_longitudinal_4'].sum()
    n_long_people_4 = len(longitudinal_ids_4)
    print(f"  4-axis longitudinal (>=2 visits): {n_long_people_4:,} people, "
          f"{n_long_4:,} visits")

    return merged


# ---------------------------------------------------------------------------
# Step 4: Cross-sectional Γ̂ analysis (parameterized)
# ---------------------------------------------------------------------------
def cross_sectional_gamma(merged, model_key='3-axis'):
    """
    Compute cross-sectional Γ̂ per age stratum per wave.
    Returns DataFrame of results.
    """
    model = MODELS[model_key]
    axes = model['axes']
    complete_col = model['complete_col']
    print(f"\nStep 4: Cross-sectional Γ̂ analysis ({model['label']})")
    print("-" * 40)

    results = []

    for wave in sorted(merged['wave'].unique()):
        wave_data = merged[(merged['wave'] == wave) & merged[complete_col]]
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
# Step 5: Within-person Γ̂ analysis (parameterized)
# ---------------------------------------------------------------------------
def within_person_gamma(merged, model_key='3-axis'):
    """
    Compute within-person covariance Γ̂_within per age stratum.
    Uses within-person residuals (removes individual fixed effects).
    """
    model = MODELS[model_key]
    axes = model['axes']
    complete_col = model['complete_col']
    in_long_col = 'in_longitudinal' if model_key == '3-axis' else 'in_longitudinal_4'

    print(f"\nStep 5: Within-person Γ̂ analysis ({model['label']})")
    print("-" * 40)

    long_data = merged[merged[in_long_col] & merged[complete_col]].copy()

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
# Step 5b: Visit-pair within-person analysis (parameterized + bootstrap)
# ---------------------------------------------------------------------------
def visit_pair_gamma(merged, model_key='3-axis', n_bootstrap=N_BOOTSTRAP):
    """
    Visit-pair Δ(Δx) analysis with bootstrap CIs and monotone-trend test.
    Δ(Δx) = Δx(wave k+1) - Δx(wave k) for each individual.
    """
    model = MODELS[model_key]
    axes = model['axes']
    complete_col = model['complete_col']
    in_long_col = 'in_longitudinal' if model_key == '3-axis' else 'in_longitudinal_4'

    print(f"\nStep 5b: Visit-pair Δ(Δx) analysis ({model['label']})")
    print("-" * 40)

    long_data = merged[merged[in_long_col] & merged[complete_col]].copy()
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

        # Full off-diagonal structure
        offdiag_info = {}
        n_axes = len(axes)
        for i in range(n_axes):
            for j in range(i + 1, n_axes):
                pair_label = f'{axes[i]}_{axes[j]}'
                offdiag_info[pair_label] = Gamma_change[i, j]

        # Bootstrap CI for λ_max
        boot_lambdas = []
        stratum_ids = stratum['idauniq'].unique()
        rng = np.random.RandomState(SEED)
        for _ in range(n_bootstrap):
            boot_ids = rng.choice(stratum_ids, size=len(stratum_ids), replace=True)
            boot_data = pd.concat([stratum[stratum['idauniq'] == uid] for uid in boot_ids],
                                  ignore_index=True)
            if len(boot_data) < 10:
                continue
            X_boot = boot_data[axes].values
            G_boot = np.cov(X_boot.T)
            bp = gamma_stability_proxy(G_boot)
            boot_lambdas.append(bp['lambda_max'])

        boot_lambdas = np.array(boot_lambdas)
        ci_lo = np.percentile(boot_lambdas, 2.5) if len(boot_lambdas) > 0 else np.nan
        ci_hi = np.percentile(boot_lambdas, 97.5) if len(boot_lambdas) > 0 else np.nan

        results.append({
            'age_group': label,
            'age_lo': lo,
            'age_hi': hi,
            'age_mid': (lo + hi) / 2,
            'n_pairs': len(stratum),
            'lambda_max': proxy['lambda_max'],
            'lambda_max_ci_lo': ci_lo,
            'lambda_max_ci_hi': ci_hi,
            'Gamma_change': Gamma_change,
            **offdiag_info,
        })
        print(f"  {label}: {len(stratum):,} pairs, "
              f"λ_max(Γ_change)={proxy['lambda_max']:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]")

    results_df = pd.DataFrame(results)

    # Jonckheere-Terpstra test for monotone trend
    if len(results_df) >= 3:
        # Use Kendall τ as approximation (scipy doesn't have JT directly)
        tau, p = stats.kendalltau(results_df['age_mid'], results_df['lambda_max'])
        print(f"\n  Monotone trend test (Kendall τ): τ={tau:.3f}, p={p:.4f}")

    # Report off-diagonal structure
    if len(results_df) > 0:
        print("\n  Off-diagonal Γ̂_change structure (largest age increase):")
        offdiag_cols = [c for c in results_df.columns if '_dx_' in c or c.startswith('dx_')]
        offdiag_cols = [c for c in results_df.columns
                        if c not in ['age_group', 'age_lo', 'age_hi', 'age_mid',
                                     'n_pairs', 'lambda_max', 'lambda_max_ci_lo',
                                     'lambda_max_ci_hi', 'Gamma_change']]
        for col in offdiag_cols:
            vals = results_df[col].values
            if len(vals) >= 2:
                change = vals[-1] - vals[0]
                print(f"    {col}: youngest={vals[0]:.4f}, oldest={vals[-1]:.4f}, "
                      f"Δ={change:+.4f}")

    return results_df


# ---------------------------------------------------------------------------
# Step 6: Compute individual SWDS-Γ (parameterized)
# ---------------------------------------------------------------------------
def compute_individual_swds(merged, within_results, model_key='3-axis'):
    """Compute SWDS-Γ for each individual at each visit."""
    model = MODELS[model_key]
    axes = model['axes']
    complete_col = model['complete_col']
    score_col = f'swds_gamma{"" if model_key == "3-axis" else "_4"}'

    print(f"\nStep 6: Computing individual SWDS-Γ scores ({model['label']})")
    print("-" * 40)

    complete = merged[merged[complete_col]].copy()

    # Build stratum covariance lookup
    gamma_lookup = {}
    if len(within_results) > 0 and 'Gamma_within' in within_results.columns:
        for _, row in within_results.iterrows():
            gamma_lookup[(row['age_lo'], row['age_hi'])] = row['Gamma_within']

    # Fallback: use cross-sectional covariance from the whole sample
    X_all = complete[axes].dropna().values
    if len(X_all) > len(axes):
        Gamma_global = np.cov(X_all.T)
    else:
        Gamma_global = np.eye(len(axes))

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

    complete[score_col] = swds_scores

    print(f"  SWDS-Γ computed for {complete[score_col].notna().sum():,} observations")
    print(f"  SWDS-Γ distribution: "
          f"mean={complete[score_col].mean():.3f}, "
          f"median={complete[score_col].median():.3f}, "
          f"SD={complete[score_col].std():.3f}")

    return complete


# ---------------------------------------------------------------------------
# Benchmark scores (R1): Mahalanobis, z-sum
# ---------------------------------------------------------------------------
def compute_benchmark_scores(complete, within_results, model_key='4-axis'):
    """
    Compute benchmark scores for comparison with SWDS-Γ:
      - Mahalanobis distance: Δx^T Γ̂^{-1} Δx (weights by INVERSE covariance)
      - Z-scored sum: ||Δx||²/n (unweighted — no eigenvalue structure)
    """
    model = MODELS[model_key]
    axes = model['axes']
    complete_col = model['complete_col']
    n_axes = len(axes)

    print(f"\nStep 6b: Computing benchmark scores ({model['label']})")
    print("-" * 40)

    sub = complete[complete[complete_col]].copy()

    # Build stratum covariance lookup
    gamma_lookup = {}
    if len(within_results) > 0 and 'Gamma_within' in within_results.columns:
        for _, row in within_results.iterrows():
            gamma_lookup[(row['age_lo'], row['age_hi'])] = row['Gamma_within']

    X_all = sub[axes].dropna().values
    if len(X_all) > n_axes:
        Gamma_global = np.cov(X_all.T)
    else:
        Gamma_global = np.eye(n_axes)

    def get_gamma(age):
        for (lo, hi), G in gamma_lookup.items():
            if lo <= age < hi:
                return G
        return Gamma_global

    mahal_scores = []
    zsum_scores = []
    for idx, row in sub.iterrows():
        dx = row[axes].values.astype(float)
        if np.any(np.isnan(dx)):
            mahal_scores.append(np.nan)
            zsum_scores.append(np.nan)
            continue

        G = get_gamma(row['age'])

        # Z-sum: ||Δx||²/n (unweighted)
        zsum = np.dot(dx, dx) / n_axes
        zsum_scores.append(zsum)

        # Mahalanobis: Δx^T Γ̂^{-1} Δx
        try:
            G_inv = np.linalg.inv(G)
            mahal = dx @ G_inv @ dx
        except np.linalg.LinAlgError:
            mahal = np.nan
        mahal_scores.append(mahal)

    sub['mahalanobis'] = mahal_scores
    sub['z_sum'] = zsum_scores

    for name, col in [('Mahalanobis', 'mahalanobis'), ('Z-sum', 'z_sum')]:
        s = sub[col].dropna()
        if len(s) > 0:
            print(f"  {name}: mean={s.mean():.3f}, median={s.median():.3f}, "
                  f"SD={s.std():.3f}")

    return sub


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
# Step 8: Cox proportional hazards (parameterized, with benchmarks)
# ---------------------------------------------------------------------------
def run_cox_models(complete, mort, demo, model_key='3-axis',
                   include_benchmarks=False):
    """
    Run nested Cox models for mortality prediction.
    Returns dict of model results.
    """
    model = MODELS[model_key]
    swds_col = f'swds_gamma{"" if model_key == "3-axis" else "_4"}'

    print(f"\nStep 8: Cox proportional hazards models ({model['label']})")
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
        f'M3: + SWDS-Γ': base_covs + [swds_col] + adj_covs,
        'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
        'M5: Full': base_covs + bio_covs + [swds_col, 'rockwood_fi'] + adj_covs,
    }

    # Add benchmark models if requested
    if include_benchmarks:
        models['M3a: + Mahalanobis'] = base_covs + ['mahalanobis'] + adj_covs
        models['M3b: + Z-sum'] = base_covs + ['z_sum'] + adj_covs
        models['M3c: + SWDS-Γ'] = base_covs + [swds_col] + adj_covs

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

    # Delta C: M5 vs M4
    if (results.get('M5: Full', {}).get('c_index') and
            results.get('M4: + Rockwood FI', {}).get('c_index')):
        dc = (results['M5: Full']['c_index'] -
              results['M4: + Rockwood FI']['c_index'])
        print(f"\n  ΔC(M5 vs M4) = {dc:+.4f}")
        if dc >= 0.01:
            print("  ✓ SWDS-Γ adds >=0.01 C-index over Rockwood FI")
        else:
            print(f"  ~ SWDS-Γ adds {dc:+.4f} C-index (below 0.01 threshold)")
        results['_delta_c_m5_m4'] = dc

    # Benchmark comparisons
    if include_benchmarks:
        print("\n  --- Benchmark C-index comparison ---")
        for m in ['M3a: + Mahalanobis', 'M3b: + Z-sum', 'M3c: + SWDS-Γ']:
            c = results.get(m, {}).get('c_index', np.nan)
            if not np.isnan(c):
                print(f"  {m}: C = {c:.4f}")

    return results


# ---------------------------------------------------------------------------
# Medication stratification analysis (4-axis)
# ---------------------------------------------------------------------------
def medication_stratification(merged, model_key='4-axis'):
    """
    Stratified analysis by medication use for the 4-axis model:
      (a) Γ̂_change separately for antihypertensive vs non
      (b) N-axis variance by statin use
      (c) Medication-stable subgroup
    """
    model = MODELS[model_key]
    axes = model['axes']
    complete_col = model['complete_col']
    in_long_col = 'in_longitudinal_4' if model_key == '4-axis' else 'in_longitudinal'

    print(f"\nStep 8c: Medication stratification ({model['label']})")
    print("-" * 40)

    long_data = merged[merged[in_long_col] & merged[complete_col]].copy()
    long_data = long_data.sort_values(['idauniq', 'wave'])

    # --- (a) Stratified Γ̂_change by antihypertensive use ---
    if 'hemda' in long_data.columns:
        for med_status, label in [(0, 'No antihypertensives'), (1, 'On antihypertensives')]:
            sub = long_data[long_data['hemda'] == med_status]
            diffs = []
            for uid, group in sub.groupby('idauniq'):
                if len(group) < 2:
                    continue
                for i in range(len(group) - 1):
                    row_a = group.iloc[i]
                    row_b = group.iloc[i + 1]
                    diff = {ax: row_b[ax] - row_a[ax] for ax in axes}
                    diff['age_mid'] = (row_a['age'] + row_b['age']) / 2
                    diffs.append(diff)

            if len(diffs) >= 20:
                diff_df = pd.DataFrame(diffs)
                X = diff_df[axes].values
                G_change = np.cov(X.T)
                proxy = gamma_stability_proxy(G_change)
                print(f"  {label}: {len(diffs)} pairs, "
                      f"λ_max(Γ_change)={proxy['lambda_max']:.4f}")

                # N-axis variance (Γ_N,N) — if 4-axis, N is index 2
                if model_key == '4-axis':
                    n_idx = 2  # dx_N_4 is 3rd axis
                    print(f"    Γ̂_N,N = {G_change[n_idx, n_idx]:.4f}")
            else:
                print(f"  {label}: insufficient pairs ({len(diffs)})")
    else:
        print("  hemda column not available — skipping antihypertensive stratification")

    # --- (b) N-axis variance by statin use ---
    statin_col = None
    for candidate in ['statins', 'statina']:
        if candidate in long_data.columns and long_data[candidate].notna().sum() > 50:
            statin_col = candidate
            break

    if statin_col:
        print(f"\n  Statin stratification (using {statin_col}):")
        for status, label in [(0, 'Non-statin'), (1, 'Statin users')]:
            sub = long_data[long_data[statin_col] == status]
            if len(sub) >= 20 and model_key == '4-axis':
                X = sub[axes].values
                G = np.cov(X.T)
                n_idx = 2  # dx_N_4
                print(f"    {label}: N={len(sub)}, Γ̂_N,N = {G[n_idx, n_idx]:.4f}")
    else:
        print("  Statin column not available — skipping statin stratification")

    # --- (c) Medication-stable subgroup ---
    if 'hemda' in long_data.columns:
        person_hemda = long_data.groupby('idauniq')['hemda'].agg(['nunique', 'count'])
        stable = person_hemda[(person_hemda['nunique'] == 1) & (person_hemda['count'] >= 2)]
        n_stable = len(stable)
        n_total = person_hemda[person_hemda['count'] >= 2].shape[0]
        print(f"\n  Medication-stable subgroup: {n_stable:,} / {n_total:,} "
              f"persons with stable hemda across waves")


# ---------------------------------------------------------------------------
# Step 9: Figures
# ---------------------------------------------------------------------------
def make_figures(cross_results, within_results, visit_pair_results,
                 complete, cox_results, model_key='3-axis'):
    """Generate 6-panel figure for a single model."""
    model = MODELS[model_key]
    score_col = f'swds_gamma{"" if model_key == "3-axis" else "_4"}'
    suffix = '3axis' if model_key == '3-axis' else '4axis'

    print(f"\nStep 9: Generating figures ({model['label']})")
    print("-" * 40)

    fig, axes_arr = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'ELSA Cohort Validation — {model["label"]}',
                 fontsize=14, fontweight='bold')

    # Panel (a): Cross-sectional λ_max by age stratum
    ax = axes_arr[0, 0]
    if len(cross_results) > 0:
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
    ax = axes_arr[0, 1]
    if len(within_results) > 0:
        wr = within_results.sort_values('age_mid')
        ax.plot(wr['age_mid'], wr['lambda_max'], 'o-',
                color='darkred', linewidth=2, markersize=8)
        ax.set_xlabel('Baseline age stratum midpoint')
        ax.set_ylabel('λ_max(Γ̂_within)')
        ax.set_title('(b) Within-person λ_max(Γ̂_within)')

        for _, row in wr.iterrows():
            ax.annotate(f"N={row['n_people']:,}",
                        (row['age_mid'], row['lambda_max']),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha='center')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (c): SWDS-Γ distribution by age stratum
    ax = axes_arr[0, 2]
    if complete is not None and score_col in complete.columns:
        for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
            sub = complete[(complete['age'] >= lo) & (complete['age'] < hi)
                           & complete[score_col].notna()]
            if len(sub) > 10:
                ax.hist(sub[score_col], bins=30, alpha=0.5, label=label,
                        density=True)
        ax.set_xlabel('SWDS-Γ')
        ax.set_ylabel('Density')
        ax.set_title('(c) SWDS-Γ by age stratum')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (d): SWDS-Γ vs age scatter
    ax = axes_arr[1, 0]
    if complete is not None and score_col in complete.columns:
        valid = complete[complete[score_col].notna()]
        ax.scatter(valid['age'], valid[score_col], alpha=0.1, s=5,
                   color='grey')

        for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
            sub = valid[(valid['age'] >= lo) & (valid['age'] < hi)]
            if len(sub) > 0:
                ax.plot((lo + hi) / 2, sub[score_col].mean(), 'ro',
                        markersize=10, zorder=5)

        ax.set_xlabel('Age')
        ax.set_ylabel('SWDS-Γ')
        ax.set_title('(d) SWDS-Γ vs age')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (e): C-index comparison
    ax = axes_arr[1, 1]
    if cox_results:
        model_names = [k for k in cox_results.keys() if not k.startswith('_')]
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
    ax = axes_arr[1, 2]
    try:
        from lifelines import KaplanMeierFitter

        if (complete is not None and score_col in complete.columns
                and 'time' in complete.columns):
            surv_data = complete[
                complete[score_col].notna() &
                complete['time'].notna() &
                (complete['time'] > 0)
            ].copy()

            if len(surv_data) > 50 and surv_data['deceased'].sum() > 10:
                tertiles = pd.qcut(surv_data[score_col], 3,
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

    output_path = os.path.join(OUTPUT_DIR, f'figure_elsa_validation_{suffix}.pdf')
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {output_path}")


def make_comparison_figure(results_3, results_4):
    """Generate comparison figure: 3-axis vs 4-axis λ_max trends and ΔC."""
    print("\nStep 9b: Generating comparison figure")
    print("-" * 40)

    fig, axes_arr = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('3-axis vs 4-axis Model Comparison',
                 fontsize=14, fontweight='bold')

    # Panel (a): visit-pair λ_max trends
    ax = axes_arr[0]
    for label, res, color in [('3-axis', results_3.get('visit_pair', pd.DataFrame()),
                                'steelblue'),
                               ('4-axis', results_4.get('visit_pair', pd.DataFrame()),
                                'darkred')]:
        if len(res) > 0:
            r = res.sort_values('age_mid')
            ax.plot(r['age_mid'], r['lambda_max'], 'o-', color=color,
                    linewidth=2, markersize=8, label=label)
            if 'lambda_max_ci_lo' in r.columns:
                ax.fill_between(r['age_mid'], r['lambda_max_ci_lo'],
                               r['lambda_max_ci_hi'], alpha=0.2, color=color)
    ax.set_xlabel('Age stratum midpoint')
    ax.set_ylabel('λ_max(Γ̂_change)')
    ax.set_title('(a) Visit-pair λ_max trends')
    ax.legend()

    # Panel (b): C-index comparison
    ax = axes_arr[1]
    cox_3 = results_3.get('cox', {})
    cox_4 = results_4.get('cox', {})
    model_labels = ['M1', 'M2', 'M3', 'M4', 'M5']
    for label_set, cox_res, color, offset in [('3-axis', cox_3, 'steelblue', -0.15),
                                               ('4-axis', cox_4, 'darkred', 0.15)]:
        c_vals = []
        for ml in model_labels:
            found = False
            for k, v in (cox_res or {}).items():
                if k.startswith(ml) and not k.startswith('_'):
                    c_vals.append(v.get('c_index', np.nan))
                    found = True
                    break
            if not found:
                c_vals.append(np.nan)

        valid = [(i, c) for i, c in enumerate(c_vals) if not np.isnan(c)]
        if valid:
            x_pos = [v[0] + offset for v in valid]
            ax.bar(x_pos, [v[1] for v in valid], width=0.25, color=color,
                   alpha=0.8, label=label_set)

    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels(model_labels)
    ax.set_ylabel('C-index')
    ax.set_title('(b) Cox model C-indices')
    ax.legend()
    ax.set_ylim(0.5, None)

    # Panel (c): ΔC comparison
    ax = axes_arr[2]
    dc_3 = cox_3.get('_delta_c_m5_m4', np.nan) if cox_3 else np.nan
    dc_4 = cox_4.get('_delta_c_m5_m4', np.nan) if cox_4 else np.nan

    labels = []
    values = []
    colors = []
    if not np.isnan(dc_3):
        labels.append('3-axis')
        values.append(dc_3)
        colors.append('steelblue')
    if not np.isnan(dc_4):
        labels.append('4-axis')
        values.append(dc_4)
        colors.append('darkred')

    if values:
        ax.bar(labels, values, color=colors, alpha=0.8)
        ax.axhline(0.01, color='green', linestyle='--', alpha=0.7,
                   label='≥0.01 threshold')
        ax.axhline(0, color='grey', linestyle='-', alpha=0.3)
        ax.set_ylabel('ΔC (M5 vs M4)')
        ax.set_title('(c) ΔC: SWDS-Γ over Rockwood FI')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No ΔC data', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = os.path.join(OUTPUT_DIR, 'figure_elsa_4axis_comparison.pdf')
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Step 10: Summary statistics
# ---------------------------------------------------------------------------
def print_summary(merged, cross_results, within_results, complete, cox_results,
                  model_key='3-axis'):
    """Print comprehensive summary block for a model."""
    model = MODELS[model_key]
    complete_col = model['complete_col']
    score_col = f'swds_gamma{"" if model_key == "3-axis" else "_4"}'

    print(f"\n{'=' * 70}")
    print(f"SUMMARY STATISTICS — {model['label']}")
    print("=" * 70)

    # N per wave
    print("\n--- Sample sizes per wave ---")
    for wave in sorted(merged['wave'].unique()):
        w_data = merged[merged['wave'] == wave]
        n_total = len(w_data)
        n_complete = w_data[complete_col].sum()
        print(f"  Wave {wave:2d}: {n_total:6,} total, "
              f"{n_complete:6,} with complete {model['label']} biomarkers")

    # Longitudinal panel
    in_long_col = 'in_longitudinal' if model_key == '3-axis' else 'in_longitudinal_4'
    long = merged[merged[in_long_col]]
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
        print(f"\n--- Cross-sectional λ_max(Γ̂) trend ---")
        cs_agg = cross_results.groupby('age_group')['lambda_max'].mean()
        for ag in AGE_STRATA_LABELS:
            if ag in cs_agg.index:
                print(f"  {ag}: λ_max = {cs_agg[ag]:.4f}")

    if len(within_results) > 0:
        print(f"\n--- Within-person λ_max(Γ̂_within) trend ---")
        for _, row in within_results.sort_values('age_mid').iterrows():
            print(f"  {row['age_group']}: λ_max = {row['lambda_max']:.4f} "
                  f"(N={row['n_people']:,})")

    # SWDS-Γ
    if complete is not None and score_col in complete.columns:
        print(f"\n--- SWDS-Γ distribution ---")
        s = complete[score_col].dropna()
        print(f"  N = {len(s):,}")
        print(f"  Mean = {s.mean():.4f}, Median = {s.median():.4f}, "
              f"SD = {s.std():.4f}")
        print(f"  Range: [{s.min():.4f}, {s.max():.4f}]")

    # Cox results
    if cox_results:
        print(f"\n--- Cox model C-indices ---")
        for name, res in cox_results.items():
            if name.startswith('_'):
                continue
            c = res.get('c_index', np.nan)
            n = res.get('n', 0)
            ev = res.get('events', 0)
            if not np.isnan(c):
                print(f"  {name}: C = {c:.4f} (N={n:,}, events={ev:,})")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# 4-axis data quality diagnostics
# ---------------------------------------------------------------------------
def print_4axis_diagnostics(merged):
    """Print data quality diagnostics specific to the 4-axis model."""
    print("\n" + "=" * 70)
    print("4-AXIS DATA QUALITY DIAGNOSTICS")
    print("=" * 70)

    # Per-wave N for each axis
    print("\n--- Per-wave N for each 4-axis component ---")
    for w in sorted(merged['wave'].unique()):
        wdata = merged[merged['wave'] == w]
        n_i = wdata['dx_I_4'].notna().sum()
        n_m = wdata['dx_M_4'].notna().sum()
        n_n = wdata['dx_N_4'].notna().sum()
        n_f = wdata['dx_F_4'].notna().sum()
        n_all = wdata['complete_4axis'].sum()
        print(f"  Wave {w}: I={n_i:,}, M={n_m:,}, N={n_n:,}, F={n_f:,}, "
              f"complete={n_all:,}")

    # Correlation matrix of 4-axis scores
    four_axes = ['dx_I_4', 'dx_M_4', 'dx_N_4', 'dx_F_4']
    complete = merged[merged['complete_4axis']]
    if len(complete) > 20:
        corr = complete[four_axes].corr()
        print("\n--- Correlation matrix of 4-axis scores ---")
        print(corr.to_string(float_format=lambda x: f'{x:.3f}'))

        # Flag high correlations
        for i in range(len(four_axes)):
            for j in range(i + 1, len(four_axes)):
                r = corr.iloc[i, j]
                if abs(r) > 0.8:
                    print(f"  ⚠ High correlation: {four_axes[i]} vs {four_axes[j]} "
                          f"r = {r:.3f}")

    # Overlap: 4-axis-complete that are also 3-axis-complete
    both_complete = (merged['complete_4axis'] & merged['complete_3axis']).sum()
    four_complete = merged['complete_4axis'].sum()
    three_complete = merged['complete_3axis'].sum()
    if four_complete > 0:
        pct = 100 * both_complete / four_complete
        print(f"\n--- Overlap ---")
        print(f"  4-axis-complete observations: {four_complete:,}")
        print(f"  3-axis-complete observations: {three_complete:,}")
        print(f"  Both complete: {both_complete:,} ({pct:.1f}% of 4-axis)")
        pct_of_3 = 100 * both_complete / three_complete if three_complete > 0 else 0
        print(f"  3-axis that are also 4-axis: {pct_of_3:.1f}%")

    # Flag extreme triglycerides
    if 'trig' in merged.columns:
        n_extreme_trig = (merged['trig'] > 20).sum()
        if n_extreme_trig > 0:
            print(f"\n  ⚠ Triglycerides > 20: {n_extreme_trig} observations "
                  f"(extreme but possibly genuine)")


# ---------------------------------------------------------------------------
# Data quality diagnostics (original)
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


# ---------------------------------------------------------------------------
# JSON results output
# ---------------------------------------------------------------------------
def write_results_json(results_3, results_4, output_path):
    """Write machine-readable results ledger."""
    print(f"\nWriting results to {output_path}")

    def safe_val(v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v) if not np.isnan(v) else None
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, pd.DataFrame):
            return None
        return v

    def extract_cox(cox_res):
        if not cox_res:
            return {}
        out = {}
        for k, v in cox_res.items():
            if k.startswith('_'):
                out[k] = safe_val(v)
            elif isinstance(v, dict):
                out[k] = {kk: safe_val(vv) for kk, vv in v.items()
                          if kk != 'model'}
        return out

    def extract_df(df, exclude_cols=None):
        if df is None or len(df) == 0:
            return []
        exclude = set(exclude_cols or [])
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                if col in exclude:
                    continue
                record[col] = safe_val(row[col])
            records.append(record)
        return records

    ledger = {
        '3-axis': {
            'cross_sectional': extract_df(results_3.get('cross'),
                                           exclude_cols=['Gamma_hat']),
            'within_person': extract_df(results_3.get('within'),
                                         exclude_cols=['Gamma_within']),
            'visit_pair': extract_df(results_3.get('visit_pair'),
                                      exclude_cols=['Gamma_change']),
            'cox_models': extract_cox(results_3.get('cox')),
        },
        '4-axis': {
            'cross_sectional': extract_df(results_4.get('cross'),
                                           exclude_cols=['Gamma_hat']),
            'within_person': extract_df(results_4.get('within'),
                                         exclude_cols=['Gamma_within']),
            'visit_pair': extract_df(results_4.get('visit_pair'),
                                      exclude_cols=['Gamma_change']),
            'cox_models': extract_cox(results_4.get('cox')),
        },
    }

    # Prominent ΔC comparison
    dc_3 = results_3.get('cox', {}).get('_delta_c_m5_m4', None) if results_3.get('cox') else None
    dc_4 = results_4.get('cox', {}).get('_delta_c_m5_m4', None) if results_4.get('cox') else None
    ledger['critical_result'] = {
        'delta_c_3axis': safe_val(dc_3),
        'delta_c_4axis': safe_val(dc_4),
        'threshold': 0.01,
        '3axis_exceeds_threshold': dc_3 is not None and dc_3 >= 0.01,
        '4axis_exceeds_threshold': dc_4 is not None and dc_4 >= 0.01,
    }

    with open(output_path, 'w') as f:
        json.dump(ledger, f, indent=2)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main: construct_survival_data
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
# Main
# ---------------------------------------------------------------------------
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

    # Step 3: Build panel (constructs BOTH 3-axis and 4-axis)
    merged = build_analysis_panel(panel, harm_long, mort, supp)

    # Data quality diagnostics
    print_data_quality(merged)
    print_4axis_diagnostics(merged)

    # Step 8a: Construct proper survival data from harmonised iwstat
    eol = files.get('eol')
    if eol is not None:
        eol_clean = eol.copy()
        eol_clean.columns = [c.lower() for c in eol_clean.columns]
        eol_clean = filter_missing(eol_clean, exclude_cols=['idauniq'])
    else:
        eol_clean = None
    surv = construct_survival_data(harm, eol_clean, baseline_wave=2)

    # =====================================================================
    # RUN ANALYSES FOR BOTH MODELS
    # =====================================================================
    results_3 = {}
    results_4 = {}

    for model_key, results_dict in [('3-axis', results_3), ('4-axis', results_4)]:
        print(f"\n{'#' * 70}")
        print(f"# RUNNING {MODELS[model_key]['label'].upper()} ANALYSIS")
        print(f"{'#' * 70}")

        # Step 4: Cross-sectional Gamma
        cross_results = cross_sectional_gamma(merged, model_key=model_key)
        results_dict['cross'] = cross_results

        # Step 5: Within-person Gamma
        within_results = within_person_gamma(merged, model_key=model_key)
        results_dict['within'] = within_results

        # Step 5b: Visit-pair analysis (with bootstrap)
        visit_pair_results = visit_pair_gamma(merged, model_key=model_key)
        results_dict['visit_pair'] = visit_pair_results

        # Step 6: Individual SWDS-Gamma
        complete = compute_individual_swds(merged, within_results,
                                           model_key=model_key)

        # Step 7: Rockwood FI (only compute once)
        if 'rockwood_fi' not in complete.columns:
            complete = compute_frailty_indices(complete)

        # Merge survival data
        if 'time_years' not in complete.columns:
            complete = complete.merge(surv[['idauniq', 'time_years', 'event']],
                                      on='idauniq', how='left')
            complete['time'] = complete['time_years']
            complete['deceased'] = complete['event']

        # Step 6b: Benchmark scores (4-axis only)
        if model_key == '4-axis':
            complete = compute_benchmark_scores(complete, within_results,
                                                 model_key=model_key)

        results_dict['complete'] = complete

        # Step 8: Cox models
        include_benchmarks = (model_key == '4-axis')
        cox_results = run_cox_models(complete, mort, harm,
                                      model_key=model_key,
                                      include_benchmarks=include_benchmarks)
        results_dict['cox'] = cox_results

        # Step 9: Figures
        make_figures(cross_results, within_results, visit_pair_results,
                     complete, cox_results, model_key=model_key)

        # Step 10: Summary
        print_summary(merged, cross_results, within_results, complete,
                      cox_results, model_key=model_key)

    # Medication stratification (4-axis only)
    medication_stratification(merged, model_key='4-axis')

    # Comparison figure
    make_comparison_figure(results_3, results_4)

    # =====================================================================
    # CRITICAL RESULT
    # =====================================================================
    print("\n" + "=" * 70)
    print("CRITICAL RESULT: ΔC COMPARISON")
    print("=" * 70)

    dc_3 = results_3.get('cox', {}).get('_delta_c_m5_m4', np.nan) if results_3.get('cox') else np.nan
    dc_4 = results_4.get('cox', {}).get('_delta_c_m5_m4', np.nan) if results_4.get('cox') else np.nan

    print(f"\n  3-axis ΔC(M5 vs M4) = {dc_3:+.4f}" if not np.isnan(dc_3)
          else "\n  3-axis ΔC: not available")
    print(f"  4-axis ΔC(M5 vs M4) = {dc_4:+.4f}" if not np.isnan(dc_4)
          else "  4-axis ΔC: not available")
    print(f"  Prespecified threshold: ≥0.01")

    if not np.isnan(dc_4):
        if dc_4 >= 0.01:
            print(f"\n  ✓ 4-axis SWDS-Γ EXCEEDS the ≥0.01 threshold (ΔC = {dc_4:+.4f})")
        else:
            print(f"\n  ✗ 4-axis SWDS-Γ does NOT exceed the ≥0.01 threshold (ΔC = {dc_4:+.4f})")

    if not np.isnan(dc_3) and not np.isnan(dc_4):
        improvement = dc_4 - dc_3
        print(f"  4-axis improvement over 3-axis: {improvement:+.4f}")

    print("=" * 70)

    # JSON output
    json_path = os.path.join(OUTPUT_DIR, 'elsa_4axis_results.json')
    write_results_json(results_3, results_4, json_path)


if __name__ == '__main__':
    main()
