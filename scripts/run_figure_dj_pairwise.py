#!/usr/bin/env python3
"""
Supplementary Figure 4: Pairwise Variance and Correlation Growth
=================================================================

4-panel figure showing individual axis variances and pairwise absolute
correlations across age strata, comparing full sample vs medication-naive
subgroup.

If ELSA data is available in data/elsa/, uses real data.
Otherwise generates synthetic illustration data from the HDR model
(clearly labelled).

Outputs:
    outputs/figure_dj_pairwise.pdf / .png

Usage:
    python scripts/run_figure_dj_pairwise.py
"""

import os
import sys

import numpy as np
from scipy import stats
from scipy.linalg import solve_continuous_lyapunov
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from hdr_sim.aging_params import tau_of_age, J_of_age, AXIS_COLORS
from hdr_sim.dynamics import build_A, spectral_abscissa
from hdr_sim.estimation import stationary_covariance
from hdr_sim.plotting import setup_style, add_panel_label, save_figure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 1000
AGE_STRATA = [(50, 59), (60, 69), (70, 79), (80, 90)]
AGE_STRATA_LABELS = ['50\u201359', '60\u201369', '70\u201379', '80+']
AGE_MIDS = [54.5, 64.5, 74.5, 84.5]

# 3-axis model: I=0, M=1, F=3 (drop N)
IDX_3_IN_4 = [0, 1, 3]
AXIS_LABELS_3 = ['I', 'M', 'F']
AXIS_COLORS_3 = [AXIS_COLORS[0], AXIS_COLORS[1], AXIS_COLORS[3]]

# Medication parameters for confound simulation
MED_FRACTION_BY_AGE = {54.5: 0.20, 64.5: 0.40, 74.5: 0.60, 84.5: 0.70}
MED_COMPRESSION = 0.6

ELSA_DIR = os.path.join(ROOT, 'data', 'elsa')


# ---------------------------------------------------------------------------
# ELSA data loading (simplified from run_elsa_validation.py)
# ---------------------------------------------------------------------------

def check_elsa_available():
    """Check whether ELSA STATA/TAB files are present."""
    if not os.path.isdir(ELSA_DIR):
        return False
    for f in os.listdir(ELSA_DIR):
        if f.lower().endswith('.tab') and 'nurse' in f.lower():
            return True
    return False


def load_elsa_data():
    """
    Load ELSA data following run_elsa_validation.py patterns.
    Returns (data_by_stratum_full, data_by_stratum_naive) where each is a dict
    mapping stratum label -> N×3 array of (I, M, F) z-scores.
    """
    import pandas as pd

    def find_file(pattern):
        for f in os.listdir(ELSA_DIR):
            if f.lower().endswith('.tab') and pattern.lower() in f.lower():
                return os.path.join(ELSA_DIR, f)
        return None

    def load_tab(path):
        df = pd.read_csv(path, sep='\t', low_memory=False, na_values=[' ', '', 'NA'])
        df.columns = [c.lower() for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def filter_missing(df, exclude_cols=None):
        exclude = set(exclude_cols or [])
        for col in df.columns:
            if col not in exclude and pd.api.types.is_numeric_dtype(df[col]):
                df.loc[df[col] < 0, col] = np.nan
        return df

    # Load nurse data
    nurse_path = find_file('nurse_biomarkers_consolidated')
    if nurse_path is None:
        nurse_path = find_file('wave_2_nurse')
    if nurse_path is None:
        raise FileNotFoundError("No nurse data found in ELSA directory")

    nurse = load_tab(nurse_path)
    nurse = filter_missing(nurse, exclude_cols=['idauniq', 'wave'])

    # Load harmonised
    harm_path = find_file('gh_elsa_h_hdr_subset')
    if harm_path is None:
        raise FileNotFoundError("No harmonised file found")
    harm = load_tab(harm_path)
    harm = filter_missing(harm, exclude_cols=['idauniq'])

    # Grip max
    grip_cols = [c for c in ['mmgsd1', 'mmgsd2', 'mmgsd3'] if c in nurse.columns]
    if grip_cols:
        nurse['grip_max'] = nurse[grip_cols].max(axis=1)
    else:
        nurse['grip_max'] = np.nan

    # HbA1c unit conversion (DCCT -> IFCC if median < 15)
    if 'hba1c' in nurse.columns:
        for w in nurse['wave'].dropna().unique():
            mask = nurse['wave'] == w
            vals = nurse.loc[mask, 'hba1c'].dropna()
            if len(vals) > 0 and vals.median() < 15:
                nurse.loc[mask, 'hba1c'] = (nurse.loc[mask, 'hba1c'] - 2.15) * 10.929

    # Z-scores against youngest stratum reference
    # Get age from harmonised (use wave-specific age columns)
    # Merge nurse + harmonised on idauniq
    if 'wave' in nurse.columns:
        # Build age lookup from harmonised
        age_cols = {}
        for w in [2, 4, 6, 8, 11]:
            for c in harm.columns:
                if f'r{w}agey' in c or f'r{w}age_y' in c:
                    age_cols[w] = c
                    break

        # Simpler: use ragey_e if available
        if 'ragey_e' in harm.columns:
            harm_age = harm[['idauniq', 'ragey_e']].copy()
            harm_age.rename(columns={'ragey_e': 'age_baseline'}, inplace=True)
        else:
            # Fallback: any age column
            age_col = [c for c in harm.columns if 'age' in c.lower()]
            if age_col:
                harm_age = harm[['idauniq', age_col[0]]].copy()
                harm_age.rename(columns={age_col[0]: 'age_baseline'}, inplace=True)
            else:
                harm_age = harm[['idauniq']].copy()
                harm_age['age_baseline'] = np.nan

        merged = nurse.merge(harm_age, on='idauniq', how='left')

        # Approximate age at each wave
        wave_year = {2: 2004, 4: 2008, 6: 2012, 8: 2016, 11: 2023}
        if 'age_baseline' in merged.columns:
            merged['age_at_wave'] = merged.apply(
                lambda r: r['age_baseline'] + (wave_year.get(r.get('wave', 2), 2004) - 2004)
                if pd.notna(r.get('age_baseline')) else np.nan,
                axis=1,
            )
        else:
            merged['age_at_wave'] = np.nan
    else:
        merged = nurse.copy()
        merged['age_at_wave'] = np.nan

    # Medication naive: no BP meds (hemda) and no diabetes meds (hemdb)
    merged['med_naive'] = True
    for col in ['hemda', 'hemdb']:
        if col in merged.columns:
            merged.loc[merged[col] == 1, 'med_naive'] = False

    # Build z-scores vs youngest stratum
    ref_mask = (merged['age_at_wave'] >= 50) & (merged['age_at_wave'] < 60)
    ref = merged[ref_mask]

    for biomarker, col in [('dx_I', 'hscrp'), ('dx_M', 'hba1c'), ('dx_F', 'grip_max')]:
        if col in merged.columns:
            ref_vals = ref[col].dropna()
            mu = ref_vals.mean() if len(ref_vals) > 10 else merged[col].mean()
            sigma = max(ref_vals.std() if len(ref_vals) > 10 else merged[col].std(), 1e-6)
            merged[biomarker] = (merged[col] - mu) / sigma
            # Flip F so higher = worse
            if biomarker == 'dx_F':
                merged[biomarker] = -merged[biomarker]

    # Stratify
    data_full = {}
    data_naive = {}
    axes_cols = ['dx_I', 'dx_M', 'dx_F']

    for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
        mask = (merged['age_at_wave'] >= lo) & (merged['age_at_wave'] < hi + 1)
        sub = merged[mask].dropna(subset=axes_cols)
        data_full[label] = sub[axes_cols].values

        sub_naive = sub[sub['med_naive']]
        data_naive[label] = sub_naive[axes_cols].values

    return data_full, data_naive


# ---------------------------------------------------------------------------
# Synthetic data generation (model-based illustration)
# ---------------------------------------------------------------------------

def generate_synthetic_data():
    """
    Generate synthetic cross-sectional data from the HDR model to mimic
    the expected pairwise patterns. Returns same structure as load_elsa_data.
    """
    rng = np.random.default_rng(SEED)
    N_per_stratum = 2000
    Q = np.eye(3)

    data_full = {}
    data_naive = {}

    for age_mid, label in zip(AGE_MIDS, AGE_STRATA_LABELS):
        tau_4 = tau_of_age(age_mid)
        J_4 = J_of_age(age_mid)
        # 3-axis slice
        tau = tau_4[IDX_3_IN_4]
        J = J_4[np.ix_(IDX_3_IN_4, IDX_3_IN_4)]
        A = build_A(tau, J)

        alpha = spectral_abscissa(A)
        if alpha >= 0:
            # Nudge to stable
            A -= (alpha + 0.01) * np.eye(3)

        Gamma = stationary_covariance(A, Q)

        # Full sample with medication compression
        X_full = rng.multivariate_normal(np.zeros(3), Gamma, size=N_per_stratum)
        med_frac = MED_FRACTION_BY_AGE.get(age_mid, 0.0)
        medicated = rng.random(N_per_stratum) < med_frac
        X_med = X_full.copy()
        X_med[medicated, 0] *= np.sqrt(MED_COMPRESSION)  # I
        X_med[medicated, 1] *= np.sqrt(MED_COMPRESSION)  # M

        data_full[label] = X_med
        data_naive[label] = X_full[~medicated]

    return data_full, data_naive


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_variance(X, axis_idx, n_boot=N_BOOTSTRAP, rng=None):
    """Bootstrap the variance of a single axis."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    N = len(X)
    if N < 10:
        return np.nan, np.nan, np.nan
    vals = X[:, axis_idx]
    boot = np.array([np.var(rng.choice(vals, N, replace=True)) for _ in range(n_boot)])
    return np.mean(boot), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def bootstrap_abs_corr(X, i, j, n_boot=N_BOOTSTRAP, rng=None):
    """Bootstrap the absolute correlation between axes i and j."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    N = len(X)
    if N < 30:
        return np.nan, np.nan, np.nan

    def abs_corr(data):
        r = np.corrcoef(data[:, i], data[:, j])[0, 1]
        return abs(r)

    boot = np.array([abs_corr(X[rng.choice(N, N, replace=True)]) for _ in range(n_boot)])
    return np.mean(boot), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure(data_full, data_naive, is_synthetic=False):
    """Create the 4-panel figure."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    rng = np.random.default_rng(SEED)
    x_pos = np.arange(len(AGE_STRATA_LABELS))

    suptitle = ''
    if is_synthetic:
        suptitle = 'Synthetic illustration (ELSA data not available)'

    # --- Panel (a): Individual axis variances ---
    ax = axes[0, 0]
    for ax_idx, (ax_label, ax_color) in enumerate(zip(AXIS_LABELS_3, AXIS_COLORS_3)):
        # Full sample (grey)
        means_f, lo_f, hi_f = [], [], []
        # Naive (colour)
        means_n, lo_n, hi_n = [], [], []

        for label in AGE_STRATA_LABELS:
            m, l, h = bootstrap_variance(data_full[label], ax_idx, rng=rng)
            means_f.append(m); lo_f.append(l); hi_f.append(h)
            m, l, h = bootstrap_variance(data_naive[label], ax_idx, rng=rng)
            means_n.append(m); lo_n.append(l); hi_n.append(h)

        offset = (ax_idx - 1) * 0.12
        # Full in grey
        errs_f = [np.array(means_f) - np.array(lo_f), np.array(hi_f) - np.array(means_f)]
        ax.errorbar(x_pos + offset - 0.03, means_f, yerr=errs_f,
                    fmt='s', color='#bdc3c7', markersize=5, capsize=3,
                    linewidth=1, alpha=0.6)
        # Naive in colour
        errs_n = [np.array(means_n) - np.array(lo_n), np.array(hi_n) - np.array(means_n)]
        ax.errorbar(x_pos + offset + 0.03, means_n, yerr=errs_n,
                    fmt='o-', color=ax_color, markersize=5, capsize=3,
                    linewidth=1.5, label=f'\u0393_{{{ax_label}{ax_label}}} (naive)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(AGE_STRATA_LABELS)
    ax.set_xlabel('Age stratum')
    ax.set_ylabel('Variance')
    ax.set_title('Individual axis variances')
    ax.legend(fontsize=7, loc='upper left')
    add_panel_label(ax, '(a)')

    # --- Panels (b), (c), (d): Pairwise |r| ---
    pairs = [(0, 1, 'I', 'M'), (0, 2, 'I', 'F'), (1, 2, 'M', 'F')]
    panel_labels = ['(b)', '(c)', '(d)']
    panel_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]

    for (i, j, li, lj), p_label, ax in zip(pairs, panel_labels, panel_axes):
        means_f, lo_f, hi_f = [], [], []
        means_n, lo_n, hi_n = [], [], []

        for label in AGE_STRATA_LABELS:
            m, l, h = bootstrap_abs_corr(data_full[label], i, j, rng=rng)
            means_f.append(m); lo_f.append(l); hi_f.append(h)
            m, l, h = bootstrap_abs_corr(data_naive[label], i, j, rng=rng)
            means_n.append(m); lo_n.append(l); hi_n.append(h)

        errs_f = [np.array(means_f) - np.array(lo_f), np.array(hi_f) - np.array(means_f)]
        errs_n = [np.array(means_n) - np.array(lo_n), np.array(hi_n) - np.array(means_n)]

        ax.errorbar(x_pos - 0.03, means_f, yerr=errs_f,
                    fmt='s', color='#bdc3c7', markersize=5, capsize=3,
                    linewidth=1, alpha=0.6, label='Full sample')
        ax.errorbar(x_pos + 0.03, means_n, yerr=errs_n,
                    fmt='o-', color=AXIS_COLORS_3[i], markersize=5, capsize=3,
                    linewidth=1.5, label='Medication-naive')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(AGE_STRATA_LABELS)
        ax.set_xlabel('Age stratum')
        ax.set_ylabel(f'|r({li}, {lj})|')
        ax.set_title(f'|r({li}, {lj})| vs age')
        ax.legend(fontsize=7)
        add_panel_label(ax, p_label)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontstyle='italic', y=1.01)

    fig.tight_layout()
    save_figure(fig, 'figure_dj_pairwise', OUTPUT_DIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating Supplementary Figure 4: pairwise variance/correlation growth ...")

    if check_elsa_available():
        print("  ELSA data found — loading real data")
        try:
            data_full, data_naive = load_elsa_data()
            is_synthetic = False
            for label in AGE_STRATA_LABELS:
                print(f"  Stratum {label}: full N={len(data_full[label])}, "
                      f"naive N={len(data_naive[label])}")
        except Exception as e:
            print(f"  Warning: ELSA loading failed ({e})")
            print("  Falling back to synthetic data")
            data_full, data_naive = generate_synthetic_data()
            is_synthetic = True
    else:
        print("  ELSA data not found — generating synthetic illustration")
        data_full, data_naive = generate_synthetic_data()
        is_synthetic = True

    plot_figure(data_full, data_naive, is_synthetic=is_synthetic)
    print("Done.")


if __name__ == '__main__':
    main()
