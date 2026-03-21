#!/usr/bin/env python3
"""
Diagnostic script: 3-Axis ΔC Discrepancy Investigation
=======================================================
R4 manuscript reports ΔC(M5 − M4) = +0.0067 for the 3-axis model.
A refactored pipeline reportedly yields +0.0108 — a 61% increase.

This script is READ-ONLY: it loads data, replicates every pipeline step
with verbose diagnostics, and prints a side-by-side comparison against
the R4 numbers.  It does NOT modify any pipeline code or data files.

Usage:
    python scripts/diagnose_delta_c.py
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

# Suppress non-critical warnings for clean output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

# Data files are gitignored and live only in the main repo, not worktrees.
# Detect worktree and resolve to main repo's data directory.
DATA_DIR = os.path.join(ROOT, 'data', 'elsa')

def _has_data_files(d):
    """Check if directory has actual ELSA tab files (not just .gitignore)."""
    if not os.path.isdir(d):
        return False
    return any(f.endswith('.tab') for f in os.listdir(d))

if not _has_data_files(DATA_DIR):
    # Worktree structure: ROOT/.claude/worktrees/<name> → main repo is ROOT/../../..
    _main_data = os.path.normpath(os.path.join(ROOT, '..', '..', '..', 'data', 'elsa'))
    if _has_data_files(_main_data):
        DATA_DIR = _main_data
    else:
        # Try git to find the main worktree
        import subprocess
        try:
            result = subprocess.run(['git', 'worktree', 'list', '--porcelain'],
                                    capture_output=True, text=True, cwd=ROOT)
            for line in result.stdout.splitlines():
                if line.startswith('worktree ') and '.claude/worktrees' not in line:
                    _main = line.split(' ', 1)[1].strip()
                    _candidate = os.path.join(_main, 'data', 'elsa')
                    if _has_data_files(_candidate):
                        DATA_DIR = _candidate
                        break
        except Exception:
            pass

from hdr_sim.estimation import compute_swds_gamma, gamma_stability_proxy

# ---------------------------------------------------------------------------
# R4 reference numbers (Appendix H, Table 10)
# ---------------------------------------------------------------------------
R4 = {
    'N_baseline':     5431,
    'events':         1122,
    'event_rate':     0.180,   # 18.0%
    'median_fu':      20.0,    # years
    'fi_mean':        0.12,
    'fi_sd':          0.14,
    'M1_c':           0.602,
    'M2_c':           0.613,
    'M3_c':           0.603,
    'M4_c':           0.609,
    'M5_c':           0.616,
    'delta_c':        0.0067,
}

# Precise values from the R4 JSON ledger
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

# ---------------------------------------------------------------------------
# Helpers (exact copies from run_elsa_validation.py)
# ---------------------------------------------------------------------------
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


def find_file(pattern, data_dir=DATA_DIR):
    for f in os.listdir(data_dir):
        if f.lower().endswith('.tab') and pattern.lower() in f.lower():
            return os.path.join(data_dir, f)
    return None


def load_tab(path):
    df = pd.read_csv(path, sep='\t', low_memory=False,
                     na_values=[' ', '', 'NA'])
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
    median_val = valid.median()
    if median_val < 15:
        converted = (series - 2.15) * 10.929
        return converted, 'DCCT'
    else:
        return series.copy(), 'IFCC'


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


def zscore_vs_ref(series, ref_df, col_name):
    ref_vals = ref_df[col_name].dropna() if col_name in ref_df.columns else pd.Series(dtype=float)
    if len(ref_vals) > 10:
        mu, sigma = ref_vals.mean(), ref_vals.std()
    else:
        mu = series.mean()
        sigma = series.std()
    sigma = max(sigma, 1e-6)
    return (series - mu) / sigma


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
# MAIN DIAGNOSTIC
# ============================================================================
def main():
    SEED = 42
    np.random.seed(SEED)

    print("=" * 72)
    print("DIAGNOSTIC: 3-Axis ΔC Discrepancy Investigation")
    print("R4 = +0.0067, Reported New = +0.0108")
    print("=" * 72)

    # ------------------------------------------------------------------
    # STEP 1: Load files
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 1: Load data files")
    print("=" * 72)

    files = {}
    for key, pattern in FILE_PATTERNS.items():
        path = find_file(pattern)
        if path:
            df = load_tab(path)
            files[key] = df
            print(f"  {key:25s}: {os.path.basename(path)} — "
                  f"{df.shape[0]:>7,} rows × {df.shape[1]:>3,} cols")
        else:
            if key.startswith('nurse_w') and 'nurse_consolidated' in files:
                pass
            else:
                print(f"  {key:25s}: NOT FOUND")

    has_consolidated = 'nurse_consolidated' in files
    has_legacy = any(k.startswith('nurse_w') and k != 'nurse_consolidated'
                     for k in files)
    print(f"\n  Data source: {'Consolidated' if has_consolidated else 'Legacy per-wave'}")
    print(f"  Legacy nurse files available: {has_legacy}")

    # ------------------------------------------------------------------
    # STEP 2: Extract biomarkers (consolidated path)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2: Extract nurse biomarkers")
    print("=" * 72)

    target_cols = ['idauniq', 'hscrp', 'hba1c', 'bmival', 'sysval', 'diaval',
                   'pulval', 'chol', 'hdl', 'ldl', 'trig', 'cfib', 'hgb',
                   'mmgsd1', 'mmgsd2', 'mmgsd3',
                   'mmgsn1', 'mmgsn2', 'mmgsn3',
                   'wbc', 'igf1', 'htval', 'wtval',
                   'wstval', 'mmcrsa', 'mmcrav',
                   'hemda', 'hemdb', 'hemdab',
                   'statins', 'statina']

    if has_consolidated:
        df = files['nurse_consolidated'].copy()
        df = filter_missing(df, exclude_cols=['idauniq', 'wave'])

        available = [c for c in target_cols if c in df.columns]
        missing_cols = [c for c in target_cols if c not in df.columns and c != 'idauniq']
        print(f"  Available cols: {len(available)}/{len(target_cols)}")
        if missing_cols:
            print(f"  Missing cols: {', '.join(missing_cols)}")

        panel = df[available + ['wave']].copy()

        if 'bmi' in df.columns and 'bmival' not in panel.columns:
            panel['bmival'] = pd.to_numeric(df['bmi'], errors='coerce')
            panel.loc[panel['bmival'] < 0, 'bmival'] = np.nan

        # HbA1c conversion per wave
        hba1c_units = {}
        print("\n  HbA1c unit detection per wave:")
        for w in sorted(panel['wave'].unique()):
            w_mask = panel['wave'] == w
            if 'hba1c' in panel.columns and panel.loc[w_mask, 'hba1c'].notna().any():
                raw = panel.loc[w_mask, 'hba1c'].copy()
                converted, unit = detect_and_convert_hba1c(raw)
                panel.loc[w_mask, 'hba1c'] = converted
                hba1c_units[int(w)] = unit
                valid = panel.loc[w_mask, 'hba1c'].dropna()
                print(f"    Wave {int(w):2d}: unit={unit:4s}, N_valid={len(valid):>5,}, "
                      f"median={valid.median():>6.1f}, "
                      f"min={valid.min():>6.1f}, max={valid.max():>6.1f}")

        # Grip strength
        grip_dom = [c for c in GRIP_COLS if c in panel.columns]
        print(f"\n  Grip columns in consolidated file: {grip_dom}")
        if grip_dom:
            panel['grip_max'] = panel[grip_dom].max(axis=1)
            print(f"  grip_max computed as max of {grip_dom}")
        else:
            panel['grip_max'] = np.nan
            print("  WARNING: No grip trial columns (mmgsd1-3) — grip_max is all NaN!")
            # Check for alternative grip columns
            grip_alts = [c for c in df.columns if 'grip' in c.lower() or 'mmgs' in c.lower()]
            print(f"  Alternate grip-related columns: {grip_alts}")

    else:
        print("  ERROR: No consolidated file; legacy path not implemented in diagnostic")
        print("  (This is itself a potential discrepancy source)")
        return

    # Deduplicate
    pre_dedup = len(panel)
    panel = panel.sort_values(['idauniq', 'wave', 'hscrp'], na_position='last')
    panel = panel.drop_duplicates(subset=['idauniq', 'wave'], keep='first')
    n_dedup = pre_dedup - len(panel)
    print(f"\n  Deduplication: {pre_dedup:,} → {len(panel):,} (removed {n_dedup:,})")

    # Per-wave summary
    print("\n  Per-wave panel summary:")
    for w in sorted(panel['wave'].unique()):
        wdata = panel[panel['wave'] == w]
        n_crp = wdata['hscrp'].notna().sum()
        n_hba1c = wdata['hba1c'].notna().sum()
        n_bmi = wdata['bmival'].notna().sum()
        n_grip = wdata['grip_max'].notna().sum()
        n_3axis = (wdata['hscrp'].notna() & wdata['hba1c'].notna() &
                   wdata['bmival'].notna() & wdata['grip_max'].notna()).sum()
        print(f"    Wave {int(w):2d}: N={len(wdata):>5,}, "
              f"CRP={n_crp:>5,}, HbA1c={n_hba1c:>5,}, "
              f"BMI={n_bmi:>5,}, Grip={n_grip:>5,}, "
              f"All4={n_3axis:>5,}")

    # ------------------------------------------------------------------
    # STEP 2b: Harmonised → long format
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2b: Harmonised to long format + merge")
    print("=" * 72)

    harm = files['harmonised'].copy()
    harm.columns = [c.lower() for c in harm.columns]
    harm = filter_missing(harm, exclude_cols=['idauniq'])
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    print(f"  Harmonised long: {len(harm_long):,} rows, "
          f"{harm_long['idauniq'].nunique():,} people")

    # ------------------------------------------------------------------
    # STEP 2c: Mortality
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 2c: Mortality construction")
    print("=" * 72)

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
            print(f"  EOL file enrichment: {has_eol.sum():,} deaths from EOL")

    n_dead = (mort['deceased'] == 1).sum()
    n_alive = (mort['deceased'] == 0).sum()
    print(f"  Total deceased: {n_dead:,}, alive/censored: {n_alive:,}")

    # ------------------------------------------------------------------
    # STEP 2d: Supplementary
    # ------------------------------------------------------------------
    supp = None
    if 'supplement' in files:
        supp = files['supplement'].copy()
        supp.columns = [c.lower() for c in supp.columns]
        supp = filter_missing(supp, exclude_cols=['idauniq', 'wave'])
        print(f"  Supplementary: {supp.shape[0]:,} rows")

    # ------------------------------------------------------------------
    # STEP 3: Build merged panel + 3-axis model
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 3: Build analysis panel + 3-axis model variables")
    print("=" * 72)

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

    # Mortality merge
    merged = merged.merge(mort, on='idauniq', how='left')

    # Supplementary merge
    if supp is not None:
        merged = merged.merge(supp, on=['idauniq', 'wave'], how='left')

    # Age filter
    pre_filter = len(merged)
    merged = merged[(merged['age'] >= 50) & (merged['age'] <= 90)]
    print(f"  Age 50-90 filter: {pre_filter:,} → {len(merged):,} "
          f"(dropped {pre_filter - len(merged):,})")

    # BMI fill from harmonised
    if 'bmi_harmonised' in merged.columns:
        mask = merged['bmival'].isna() & merged['bmi_harmonised'].notna()
        n_filled = mask.sum()
        merged.loc[mask, 'bmival'] = merged.loc[mask, 'bmi_harmonised']
        print(f"  BMI filled from harmonised: {n_filled:,}")
        for w in sorted(merged['wave'].unique()):
            w_filled = ((merged['wave'] == w) & mask).sum()
            if w_filled > 0:
                print(f"    Wave {int(w)}: {w_filled:,} BMI filled")

    # --- Reference subgroup ---
    ref_mask = (merged['wave'] == 2) & (merged['age'] >= 50) & (merged['age'] <= 55)
    ref = merged.loc[ref_mask]
    print(f"\n  Reference subgroup (W2, age 50-55): N={len(ref):,}")

    # --- dx_I ---
    merged['log_crp'] = np.log(merged['hscrp'].clip(lower=0.01))
    ref_log_crp = np.log(ref['hscrp'].clip(lower=0.01).dropna())
    if len(ref_log_crp) > 10:
        crp_mean, crp_std = ref_log_crp.mean(), ref_log_crp.std()
    else:
        crp_mean = merged['log_crp'].mean()
        crp_std = merged['log_crp'].std()
    crp_std = max(crp_std, 1e-6)
    merged['dx_I'] = (merged['log_crp'] - crp_mean) / crp_std
    print(f"  dx_I: ref_mean={crp_mean:.4f}, ref_std={crp_std:.4f}")

    # --- dx_M ---
    for var, label in [('hba1c', 'hba1c_z'), ('bmival', 'bmi_z')]:
        ref_vals = ref[var].dropna() if var in ref.columns else pd.Series(dtype=float)
        if len(ref_vals) > 10:
            mu, sigma = ref_vals.mean(), ref_vals.std()
        else:
            mu = merged[var].mean()
            sigma = merged[var].std()
        sigma = max(sigma, 1e-6)
        merged[label] = (merged[var] - mu) / sigma
        print(f"  {label}: ref_mean={mu:.4f}, ref_std={sigma:.4f}, "
              f"N_ref={len(ref_vals):,}")

    merged['dx_M'] = (merged['hba1c_z'] + merged['bmi_z']) / np.sqrt(2)

    # --- dx_F ---
    ref_grip = ref['grip_max'].dropna() if 'grip_max' in ref.columns else pd.Series(dtype=float)
    if len(ref_grip) > 10:
        grip_mean, grip_std = ref_grip.mean(), ref_grip.std()
    else:
        grip_mean = merged['grip_max'].mean()
        grip_std = merged['grip_max'].std()
    grip_std = max(grip_std, 1e-6)
    merged['dx_F'] = -(merged['grip_max'] - grip_mean) / grip_std
    print(f"  dx_F: ref_mean={grip_mean:.4f}, ref_std={grip_std:.4f}, "
          f"N_ref={len(ref_grip):,}")

    # --- Completeness ---
    merged['complete_3axis'] = (
        merged['dx_I'].notna() &
        merged['dx_M'].notna() &
        merged['dx_F'].notna()
    )

    print(f"\n  DIAGNOSTIC: 3-axis sample composition")
    print(f"    Total person-visits: {len(merged):,}")
    print(f"    Complete 3-axis: {merged['complete_3axis'].sum():,}")
    print(f"    Unique persons with ≥1 complete: "
          f"{merged.loc[merged['complete_3axis'], 'idauniq'].nunique():,}")

    visit_counts = merged.loc[merged['complete_3axis']].groupby('idauniq').size()
    n_ge2 = (visit_counts >= 2).sum()
    print(f"    Unique persons with ≥2 complete: {n_ge2:,}")

    for w in sorted(merged['wave'].unique()):
        wdata = merged[(merged['wave'] == w) & merged['complete_3axis']]
        print(f"    Wave {int(w):2d}: {len(wdata):,} complete 3-axis")

    # ------------------------------------------------------------------
    # STEP 4: Within-person Γ̂ (needed for SWDS-Γ computation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 4: Within-person covariance (for SWDS-Γ)")
    print("=" * 72)

    axes = ['dx_I', 'dx_M', 'dx_F']
    long_ids = visit_counts[visit_counts >= 2].index
    merged['in_longitudinal'] = merged['idauniq'].isin(long_ids)

    long_data = merged[merged['in_longitudinal'] & merged['complete_3axis']].copy()
    person_means = long_data.groupby('idauniq')[axes].mean()
    person_means.columns = [f'{c}_mean' for c in axes]
    long_data = long_data.merge(person_means, on='idauniq', how='left')
    for ax in axes:
        long_data[f'{ax}_resid'] = long_data[ax] - long_data[f'{ax}_mean']
    resid_axes = [f'{ax}_resid' for ax in axes]
    first_ages = long_data.groupby('idauniq')['age'].min().rename('baseline_age')
    long_data = long_data.merge(first_ages, on='idauniq', how='left')

    gamma_lookup = {}
    for (lo, hi), label in zip(AGE_STRATA, ['50-59', '60-69', '70-79', '80+']):
        stratum = long_data[
            (long_data['baseline_age'] >= lo) & (long_data['baseline_age'] < hi)]
        if len(stratum) >= 20:
            X_resid = stratum[resid_axes].values
            G_within = np.cov(X_resid.T)
            gamma_lookup[(lo, hi)] = G_within
            proxy = gamma_stability_proxy(G_within)
            print(f"  {label}: {stratum['idauniq'].nunique():,} people, "
                  f"λ_max={proxy['lambda_max']:.4f}")

    # ------------------------------------------------------------------
    # STEP 5: Compute individual SWDS-Γ
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 5: Compute SWDS-Γ scores")
    print("=" * 72)

    complete = merged[merged['complete_3axis']].copy()
    X_all = complete[axes].dropna().values
    Gamma_global = np.cov(X_all.T) if len(X_all) > 3 else np.eye(3)

    def get_gamma(age):
        for (lo, hi), G in gamma_lookup.items():
            if lo <= age < hi:
                return G
        return Gamma_global

    swds_scores = []
    for idx, row in complete.iterrows():
        dx = row[axes].values.astype(float)
        if np.any(np.isnan(dx)):
            swds_scores.append(np.nan)
            continue
        G = get_gamma(row['age'])
        score = compute_swds_gamma(dx, G)
        swds_scores.append(score)

    complete['swds_gamma'] = swds_scores
    print(f"  SWDS-Γ computed for {complete['swds_gamma'].notna().sum():,}")
    print(f"  Distribution: mean={complete['swds_gamma'].mean():.4f}, "
          f"SD={complete['swds_gamma'].std():.4f}")

    # ------------------------------------------------------------------
    # STEP 6: Rockwood FI
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 6: Rockwood Frailty Index")
    print("=" * 72)

    supp_cols_present = any(col in complete.columns for col in ADL_ITEMS)
    if supp_cols_present:
        complete['rockwood_fi'] = complete.apply(compute_rockwood_fi, axis=1)
        n_fi = complete['rockwood_fi'].notna().sum()
        fi_vals = complete['rockwood_fi'].dropna()
        print(f"  FI computed: {n_fi:,} / {len(complete):,}")
        if len(fi_vals) > 0:
            print(f"  FI distribution: mean={fi_vals.mean():.4f}, "
                  f"SD={fi_vals.std():.4f}, median={fi_vals.median():.4f}")
    else:
        complete['rockwood_fi'] = np.nan
        print("  WARNING: Supplementary columns missing — FI is all NaN")

    # ------------------------------------------------------------------
    # STEP 7: Cox proportional hazards
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 7: Cox proportional hazards models (3-axis)")
    print("=" * 72)

    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("  ERROR: lifelines not installed. pip install lifelines")
        return

    # Baseline = wave 2
    baseline = complete[complete['wave'] == 2].copy()
    print(f"  Wave 2 complete 3-axis: {len(baseline):,}")

    # Merge mortality (may have been merged already, handle carefully)
    if 'deceased' not in baseline.columns or baseline['deceased'].isna().all():
        baseline = baseline.merge(
            mort[['idauniq', 'deceased', 'death_year', 'death_age']],
            on='idauniq', how='left', suffixes=('', '_mort'))
        for col in ['deceased', 'death_year', 'death_age']:
            mort_col = f'{col}_mort'
            if mort_col in baseline.columns:
                baseline[col] = baseline[mort_col].fillna(baseline.get(col, np.nan))
                baseline = baseline.drop(columns=[mort_col])

    baseline['deceased'] = baseline['deceased'].fillna(0).astype(int)

    # Compute survival time (replicating pipeline logic exactly)
    baseline_year = baseline['interview_year'].fillna(
        NURSE_WAVE_YEARS.get(2, 2004))

    # Last contact year: pipeline uses iwy columns or defaults to 2024
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

    # Filter positive follow-up
    pre_time = len(baseline)
    baseline = baseline[baseline['time'] > 0].copy()
    print(f"  After time > 0 filter: {len(baseline):,} (dropped {pre_time - len(baseline):,})")

    print(f"\n  DIAGNOSTIC: Baseline Cox sample")
    print(f"    N:             {len(baseline):,}")
    print(f"    Events:        {baseline['deceased'].sum():,}")
    event_rate = baseline['deceased'].mean()
    print(f"    Event rate:    {event_rate:.1%}")
    print(f"    Median FU:     {baseline['time'].median():.1f} years")
    print(f"    Mean FU:       {baseline['time'].mean():.1f} years")
    print(f"    Min FU:        {baseline['time'].min():.1f} years")
    print(f"    Max FU:        {baseline['time'].max():.1f} years")

    # Variable distributions at baseline
    print(f"\n  DIAGNOSTIC: Baseline variable distributions")
    for var, label in [('dx_I', 'dx_I'), ('dx_M', 'dx_M'), ('dx_F', 'dx_F'),
                       ('swds_gamma', 'SWDS-Γ'), ('rockwood_fi', 'FI'),
                       ('log_crp', 'log_CRP'), ('hba1c', 'HbA1c'),
                       ('grip_max', 'Grip'), ('bmival', 'BMI')]:
        if var in baseline.columns:
            vals = baseline[var].dropna()
            print(f"    {label:10s}: N={len(vals):>5,}, "
                  f"mean={vals.mean():>8.4f}, SD={vals.std():>8.4f}, "
                  f"median={vals.median():>8.4f}")

    # --- Run nested models ---
    base_covs = ['age', 'sex']
    bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival', 'sysval']
    adj_candidates = ['smoking', 'diabetes', 'highbp', 'hemda', 'hemdb']
    adj_covs = []
    for c in adj_candidates:
        if c in baseline.columns and baseline[c].notna().mean() > 0.5:
            adj_covs.append(c)
    print(f"\n  Adjustment covariates (>50% coverage): {adj_covs}")

    models = {
        'M1: Age + Sex':     base_covs + adj_covs,
        'M2: + Biomarkers':  base_covs + bio_covs + adj_covs,
        'M3: + SWDS-Γ':      base_covs + ['swds_gamma'] + adj_covs,
        'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
        'M5: Full':          base_covs + bio_covs + ['swds_gamma', 'rockwood_fi'] + adj_covs,
    }

    results = {}
    for name, covs in models.items():
        all_covs = [c for c in covs if c in baseline.columns]
        surv_data = baseline[all_covs + ['time', 'deceased']].dropna().copy()

        if len(surv_data) < 50 or surv_data['deceased'].sum() < 10:
            print(f"  {name}: SKIP (N={len(surv_data)}, events={surv_data['deceased'].sum()})")
            results[name] = {'c_index': np.nan, 'n': len(surv_data), 'events': 0}
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
                'covariates': all_covs,
            }
            print(f"  {name}: C={c_idx:.6f}, N={len(surv_data):,}, "
                  f"events={int(surv_data['deceased'].sum()):,}, "
                  f"covs={all_covs}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            results[name] = {'c_index': np.nan, 'n': len(surv_data), 'events': 0}

    # Delta C
    m5_c = results.get('M5: Full', {}).get('c_index', np.nan)
    m4_c = results.get('M4: + Rockwood FI', {}).get('c_index', np.nan)
    if not (np.isnan(m5_c) or np.isnan(m4_c)):
        dc = m5_c - m4_c
    else:
        dc = np.nan

    # ------------------------------------------------------------------
    # STEP 8: Side-by-side comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 8: Side-by-side comparison with R4 manuscript")
    print("=" * 72)

    def fmt(v, prec=4):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '     N/A'
        return f'{v:>{prec + 5}.{prec}f}'

    def fmt_int(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '    N/A'
        return f'{int(v):>7,}'

    new_n = len(baseline)
    new_events = int(baseline['deceased'].sum())
    new_rate = baseline['deceased'].mean()
    new_fu = baseline['time'].median()
    fi_base = baseline['rockwood_fi'].dropna()
    new_fi_mean = fi_base.mean() if len(fi_base) > 0 else np.nan
    new_fi_sd = fi_base.std() if len(fi_base) > 0 else np.nan

    header = f"{'':30s} {'R4 manuscript':>14s} {'This run':>14s} {'Difference':>14s}"
    print(header)
    print("-" * len(header))

    rows = [
        ('N (baseline, Cox)',
         fmt_int(R4['N_baseline']),
         fmt_int(new_n),
         fmt_int(new_n - R4['N_baseline'])),
        ('Events',
         fmt_int(R4['events']),
         fmt_int(new_events),
         fmt_int(new_events - R4['events'])),
        ('Event rate',
         f"  {R4['event_rate']:.1%}",
         f"  {new_rate:.1%}",
         f"  {new_rate - R4['event_rate']:+.1%}"),
        ('Median FU (years)',
         f"  {R4['median_fu']:.1f}",
         f"  {new_fu:.1f}",
         f"  {new_fu - R4['median_fu']:+.1f}"),
        ('FI mean ± SD',
         f"  {R4['fi_mean']:.2f} ± {R4['fi_sd']:.2f}",
         f"  {new_fi_mean:.4f} ± {new_fi_sd:.4f}" if not np.isnan(new_fi_mean) else '  N/A',
         ''),
    ]

    for label, r4_val, new_val, diff in rows:
        print(f"  {label:28s} {r4_val:>14s} {new_val:>14s} {diff:>14s}")

    print()
    model_keys = [
        ('M1: Age + Sex',     'M1_c'),
        ('M2: + Biomarkers',  'M2_c'),
        ('M3: + SWDS-Γ',      'M3_c'),
        ('M4: + Rockwood FI', 'M4_c'),
        ('M5: Full',          'M5_c'),
    ]

    for mname, r4_key in model_keys:
        r4_c = R4_JSON[r4_key]
        new_c = results.get(mname, {}).get('c_index', np.nan)
        r4_n = R4_JSON.get(f'{r4_key[:-2]}_n', '')
        new_n_m = results.get(mname, {}).get('n', '')
        diff = new_c - r4_c if not np.isnan(new_c) else np.nan
        print(f"  {mname:28s}  C={r4_c:.6f} (N={r4_n})  "
              f"C={new_c:.6f} (N={new_n_m})  "
              f"ΔC={diff:+.6f}" if not np.isnan(diff) else
              f"  {mname:28s}  C={r4_c:.6f}  C=N/A")

    print()
    print(f"  {'ΔC (M5-M4)':28s}  {R4_JSON['delta_c']:+.7f}         "
          f"{dc:+.7f}" if not np.isnan(dc) else
          f"  {'ΔC (M5-M4)':28s}  {R4_JSON['delta_c']:+.7f}         N/A")

    # ------------------------------------------------------------------
    # STEP 9: Detailed divergence analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STEP 9: Divergence analysis")
    print("=" * 72)

    # 9a: N difference between M1/M4 (which use full sample) and M2/M5 (need biomarkers)
    m1_n = results.get('M1: Age + Sex', {}).get('n', 0)
    m2_n = results.get('M2: + Biomarkers', {}).get('n', 0)
    m4_n = results.get('M4: + Rockwood FI', {}).get('n', 0)
    m5_n = results.get('M5: Full', {}).get('n', 0)
    print(f"\n  9a: Sample size by model")
    print(f"    M1 (age+sex) N = {m1_n:,}    (R4: {R4_JSON['M1_n']:,}, diff: {m1_n - R4_JSON['M1_n']:+,})")
    print(f"    M2 (bio)     N = {m2_n:,}    (R4: {R4_JSON['M2_n']:,}, diff: {m2_n - R4_JSON['M2_n']:+,})")
    print(f"    M4 (FI)      N = {m4_n:,}    (R4: {R4_JSON['M4_n']:,}, diff: {m4_n - R4_JSON['M4_n']:+,})")
    print(f"    M5 (full)    N = {m5_n:,}    (R4: {R4_JSON['M5_n']:,}, diff: {m5_n - R4_JSON['M5_n']:+,})")

    print(f"\n    KEY OBSERVATION: M5 and M4 have DIFFERENT N values")
    print(f"    M4 includes all complete 3-axis + FI; M5 additionally requires biomarkers")
    print(f"    ΔC(M5-M4) compares C-indices from DIFFERENT samples!")

    # 9b: HbA1c edge cases
    print(f"\n  9b: HbA1c conversion diagnostics")
    for w in sorted(merged['wave'].unique()):
        wdata = merged[merged['wave'] == w]
        hba1c_valid = wdata['hba1c'].dropna()
        if len(hba1c_valid) > 0:
            # Check for suspicious values
            n_low = (hba1c_valid < 10).sum()   # Very low IFCC
            n_mid = ((hba1c_valid >= 10) & (hba1c_valid < 20)).sum()  # Boundary zone
            n_high = (hba1c_valid > 130).sum()  # Very high IFCC
            print(f"    Wave {int(w):2d}: N={len(hba1c_valid):>5,}, "
                  f"median={hba1c_valid.median():.1f}, "
                  f"<10={n_low}, 10-20={n_mid}, >130={n_high}")

    # 9c: Grip strength source comparison
    print(f"\n  9c: Grip strength diagnostics")
    for w in sorted(merged['wave'].unique()):
        wdata = merged[merged['wave'] == w]
        grip_valid = wdata['grip_max'].dropna()
        if len(grip_valid) > 0:
            print(f"    Wave {int(w):2d}: N={len(grip_valid):>5,}, "
                  f"mean={grip_valid.mean():.1f}, "
                  f"median={grip_valid.median():.1f}, "
                  f"min={grip_valid.min():.1f}, max={grip_valid.max():.1f}")

    # 9d: Survival time diagnostics
    print(f"\n  9d: Survival time construction")
    dead = baseline[baseline['deceased'] == 1]
    alive = baseline[baseline['deceased'] == 0]
    print(f"    Deceased: N={len(dead):,}, mean_time={dead['time'].mean():.1f}y, "
          f"median={dead['time'].median():.1f}y")
    print(f"    Censored: N={len(alive):,}, mean_time={alive['time'].mean():.1f}y, "
          f"median={alive['time'].median():.1f}y")

    # Check last_contact_year distribution
    print(f"    last_contact_year: mean={baseline['last_contact_year'].mean():.1f}, "
          f"mode={baseline['last_contact_year'].mode().iloc[0]:.0f}")
    print(f"    death_year (if deceased): "
          f"mean={dead['death_year'].dropna().mean():.1f}" if len(dead['death_year'].dropna()) > 0 else
          "    death_year: no valid values")

    # 9e: Covariate availability
    print(f"\n  9e: Covariate availability at baseline (Wave 2)")
    for var in ['age', 'sex', 'smoking', 'diabetes', 'highbp',
                'hemda', 'hemdb', 'sysval', 'log_crp', 'hba1c',
                'grip_max', 'bmival', 'rockwood_fi', 'swds_gamma']:
        if var in baseline.columns:
            coverage = baseline[var].notna().mean()
            print(f"    {var:15s}: {coverage:>6.1%} coverage "
                  f"({baseline[var].notna().sum():,} / {len(baseline):,})")

    # ------------------------------------------------------------------
    # STEP 10: Diagnosis
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("DISCREPANCY DIAGNOSIS")
    print("=" * 72)

    # Check if this run reproduces R4
    if not np.isnan(dc):
        dc_diff = abs(dc - R4_JSON['delta_c'])
        reproduces_r4 = dc_diff < 0.0005

        if reproduces_r4:
            print(f"""
  This diagnostic run REPRODUCES the R4 result.
    R4  ΔC = {R4_JSON['delta_c']:+.7f}
    Now ΔC = {dc:+.7f}
    Absolute difference: {dc_diff:.7f}

  If the "new pipeline" reports +0.0108, the discrepancy is NOT in the
  data loading or variable construction — it must be in a code difference
  between this pipeline and the "new" one.

  LIKELY CAUSES to investigate in the refactored pipeline:
    1. Different covariate set in M4 or M5 (adj_covs differ)
    2. Different handling of NaN rows → different effective N per model
    3. Survival time construction differs (last_contact_year, death_year)
    4. SWDS-Γ computed with cross-sectional Γ̂ instead of within-person
    5. FI construction uses different variable set
    6. M5 and M4 run on DIFFERENT sample sizes (N_M5 ≠ N_M4)
       — ΔC = C(M5, N_M5) - C(M4, N_M4) is NOT a clean comparison

  CRITICAL STRUCTURAL ISSUE:
    M4 (age+sex+FI) uses N={m4_n} — all rows with valid age, sex, FI
    M5 (full) uses N={m5_n} — only rows with ALL biomarkers + FI + SWDS
    The sample SHRINKS by {m4_n - m5_n} people when adding biomarkers.
    If M5 N ≠ M4 N, ΔC conflates model improvement with sample selection.
""")
        else:
            print(f"""
  This diagnostic run does NOT reproduce the R4 result.
    R4  ΔC = {R4_JSON['delta_c']:+.7f}
    Now ΔC = {dc:+.7f}
    Difference: {dc - R4_JSON['delta_c']:+.7f}

  The discrepancy arises in the data loading or variable construction
  within THIS diagnostic script relative to the original R4 pipeline.

  Check:
    1. Whether the original R4 used legacy nurse files vs consolidated
    2. Whether the consolidated file was updated since R4
    3. Whether the R4 JSON was produced by a different code version
""")

        # Additional: compute ΔC on matched samples
        print("  ADDITIONAL: ΔC on matched samples")
        m5_covs = results.get('M5: Full', {}).get('covariates', [])
        m4_covs = results.get('M4: + Rockwood FI', {}).get('covariates', [])
        all_needed = list(set(m5_covs + m4_covs))
        matched = baseline[all_needed + ['time', 'deceased']].dropna().copy()
        print(f"    Matched sample (all M4+M5 covariates non-missing): N={len(matched):,}, "
              f"events={int(matched['deceased'].sum()):,}")

        if len(matched) >= 50 and matched['deceased'].sum() >= 10:
            # Re-run M4 and M5 on matched sample
            for mname, covs_key in [('M4: + Rockwood FI', m4_covs),
                                     ('M5: Full', m5_covs)]:
                mc = [c for c in covs_key if c in matched.columns]
                surv_data = matched[mc + ['time', 'deceased']].dropna().copy()
                try:
                    cph = CoxPHFitter()
                    cph.fit(surv_data, duration_col='time', event_col='deceased')
                    c_idx = concordance_index(
                        surv_data['time'],
                        -cph.predict_partial_hazard(surv_data),
                        surv_data['deceased']
                    )
                    print(f"    {mname} (matched): C={c_idx:.6f}, N={len(surv_data):,}")
                except Exception as e:
                    print(f"    {mname} (matched): FAILED — {e}")

            # Matched ΔC
            try:
                m4_matched = baseline[m4_covs + ['time', 'deceased']].dropna()
                m5_matched = baseline[m5_covs + ['time', 'deceased']].dropna()

                # Get intersection of IDs
                common_data = matched.copy()

                cph4 = CoxPHFitter()
                cph4.fit(common_data[[c for c in m4_covs if c in common_data.columns] + ['time', 'deceased']].dropna(),
                         duration_col='time', event_col='deceased')
                sd4 = common_data[[c for c in m4_covs if c in common_data.columns] + ['time', 'deceased']].dropna()
                c4 = concordance_index(sd4['time'], -cph4.predict_partial_hazard(sd4), sd4['deceased'])

                cph5 = CoxPHFitter()
                cph5.fit(common_data[[c for c in m5_covs if c in common_data.columns] + ['time', 'deceased']].dropna(),
                         duration_col='time', event_col='deceased')
                sd5 = common_data[[c for c in m5_covs if c in common_data.columns] + ['time', 'deceased']].dropna()
                c5 = concordance_index(sd5['time'], -cph5.predict_partial_hazard(sd5), sd5['deceased'])

                dc_matched = c5 - c4
                print(f"\n    ΔC (matched sample): {dc_matched:+.7f}")
                print(f"    ΔC (original, unmatched): {dc:+.7f}")
                print(f"    Difference due to sample selection: {dc - dc_matched:+.7f}")
            except Exception as e:
                print(f"    Matched ΔC computation failed: {e}")

    else:
        print("  ΔC could not be computed — check model failures above.")

    print("\n" + "=" * 72)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 72)


if __name__ == '__main__':
    main()
