#!/usr/bin/env python3
"""
Medication Sensitivity Analysis for ELSA 4-Axis Model

Diagnoses the medication-compression confound that caused ΔC = −0.0007
in the original 4-axis model, and tests whether SWDS-Γ recovers its
value when medication effects are addressed.

Three strategies:
  1. Pulse-only N axis (removes BP medication confound)
  2. Medication-naive subgroup (no hypertension/diabetes at baseline)
  3. Medication-robust composite (biomarkers minimally affected by meds)

Usage:
    python scripts/run_elsa_medication_sensitivity.py

Outputs:
    outputs/elsa_medication_sensitivity.json  — all numerical results
    outputs/figure_elsa_medication_sensitivity.pdf — 4-panel comparison figure
"""

import json
import os
import sys
import warnings
import importlib.util

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
    gamma_stability_proxy,
)

# Import functions from run_elsa_validation.py
_val_spec = importlib.util.spec_from_file_location(
    "run_elsa_validation",
    os.path.join(ROOT, 'scripts', 'run_elsa_validation.py'))
_val_mod = importlib.util.module_from_spec(_val_spec)
_val_spec.loader.exec_module(_val_mod)

# Re-use data loading & panel construction from validation script
load_all_files = _val_mod.load_all_files
extract_nurse_biomarkers = _val_mod.extract_nurse_biomarkers
prepare_harmonised = _val_mod.prepare_harmonised
extract_mortality = _val_mod.extract_mortality
extract_supplementary = _val_mod.extract_supplementary
harmonised_to_long = _val_mod.harmonised_to_long
build_analysis_panel = _val_mod.build_analysis_panel
construct_survival_data = _val_mod.construct_survival_data
filter_missing = _val_mod.filter_missing
compute_rockwood_fi = _val_mod.compute_rockwood_fi
zscore_vs_ref = _val_mod.zscore_vs_ref
cross_sectional_gamma = _val_mod.cross_sectional_gamma
within_person_gamma = _val_mod.within_person_gamma
visit_pair_gamma = _val_mod.visit_pair_gamma
compute_individual_swds = _val_mod.compute_individual_swds
compute_frailty_indices = _val_mod.compute_frailty_indices

MODELS = _val_mod.MODELS
AGE_STRATA = _val_mod.AGE_STRATA
AGE_STRATA_LABELS = _val_mod.AGE_STRATA_LABELS
NURSE_WAVE_YEARS = _val_mod.NURSE_WAVE_YEARS
N_BOOTSTRAP = _val_mod.N_BOOTSTRAP

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)


# =========================================================================
# Reusable pipeline functions (parameterised by axis columns)
# =========================================================================

def register_model(name, axes, complete_col, label):
    """Register a custom model variant in the MODELS dict."""
    MODELS[name] = {
        'axes': axes,
        'complete_col': complete_col,
        'label': label,
    }


def run_cox_models_custom(complete, surv, model_key, swds_col):
    """
    Run nested Cox models M1-M5 for a given model variant.
    Returns dict with C-indices and _delta_c_m5_m4.
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("  lifelines not installed — skipping Cox models")
        return None

    # Merge survival data
    if 'time_years' not in complete.columns:
        complete = complete.merge(
            surv[['idauniq', 'time_years', 'event']],
            on='idauniq', how='left')
        complete['time'] = complete['time_years']
        complete['deceased'] = complete['event']

    # Use wave 2 baseline
    baseline = complete[complete['wave'] == 2].copy()
    if len(baseline) < 50:
        earliest = complete['wave'].min()
        baseline = complete[complete['wave'] == earliest].copy()

    # Ensure survival columns
    if 'time' not in baseline.columns and 'time_years' in baseline.columns:
        baseline['time'] = baseline['time_years']
    if 'deceased' not in baseline.columns and 'event' in baseline.columns:
        baseline['deceased'] = baseline['event']

    baseline['deceased'] = baseline['deceased'].fillna(0).astype(int)
    baseline = baseline[baseline.get('time', pd.Series(dtype=float)).notna()
                        & (baseline.get('time', pd.Series(0)) > 0)].copy()

    if len(baseline) < 50 or baseline['deceased'].sum() < 10:
        print(f"  Insufficient data for Cox models (N={len(baseline)}, "
              f"events={baseline['deceased'].sum()})")
        return None

    print(f"  Cox baseline: N={len(baseline):,}, "
          f"events={baseline['deceased'].sum():,}")

    base_covs = ['age', 'sex']
    bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival', 'sysval']
    adj_candidates = ['smoking', 'diabetes', 'highbp', 'hemda', 'hemdb']
    adj_covs = [c for c in adj_candidates
                if c in baseline.columns and baseline[c].notna().mean() > 0.5]

    models_spec = {
        'M1: Age + Sex': base_covs + adj_covs,
        'M2: + Biomarkers': base_covs + bio_covs + adj_covs,
        'M3: + SWDS-Γ': base_covs + [swds_col] + adj_covs,
        'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
        'M5: Full': base_covs + bio_covs + [swds_col, 'rockwood_fi'] + adj_covs,
    }

    results = {}
    for name, covs in models_spec.items():
        all_covs = [c for c in covs if c in baseline.columns]
        surv_data = baseline[all_covs + ['time', 'deceased']].dropna().copy()

        # Replace inf/-inf with NaN and re-drop
        surv_data = surv_data.replace([np.inf, -np.inf], np.nan).dropna()

        # Drop constant columns (zero variance — causes Cox convergence failure)
        const_cols = [c for c in all_covs
                      if c in surv_data.columns and surv_data[c].std() < 1e-10]
        if const_cols:
            surv_data = surv_data.drop(columns=const_cols)
            all_covs = [c for c in all_covs if c not in const_cols]

        if len(surv_data) < 50 or surv_data['deceased'].sum() < 10:
            print(f"  {name}: insufficient data (N={len(surv_data)})")
            results[name] = {'c_index': np.nan, 'n': len(surv_data), 'events': 0}
            continue

        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(surv_data, duration_col='time', event_col='deceased')
            c_idx = concordance_index(
                surv_data['time'],
                -cph.predict_partial_hazard(surv_data),
                surv_data['deceased'])
            results[name] = {
                'c_index': c_idx,
                'n': len(surv_data),
                'events': int(surv_data['deceased'].sum()),
                'aic': cph.AIC_partial_,
            }
            print(f"  {name}: C={c_idx:.4f}, N={len(surv_data):,}, "
                  f"events={int(surv_data['deceased'].sum()):,}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            results[name] = {'c_index': np.nan, 'n': len(surv_data), 'events': 0}

    # Delta C
    c_m5 = results.get('M5: Full', {}).get('c_index')
    c_m4 = results.get('M4: + Rockwood FI', {}).get('c_index')
    if c_m5 and c_m4 and not np.isnan(c_m5) and not np.isnan(c_m4):
        dc = c_m5 - c_m4
        results['_delta_c_m5_m4'] = dc
        print(f"\n  ΔC(M5 vs M4) = {dc:+.4f}")
        if dc >= 0.01:
            print("  ✓ SWDS-Γ adds ≥0.01 C-index over Rockwood FI")
        else:
            print(f"  ~ SWDS-Γ adds {dc:+.4f} (below 0.01 threshold)")
    else:
        results['_delta_c_m5_m4'] = np.nan

    return results


def run_full_pipeline(merged, surv, model_key, swds_col_name):
    """
    Run the full Γ analysis + SWDS + Cox pipeline for a model variant.
    Returns dict with cross, within, visit_pair, complete, cox results.
    """
    res = {}

    # Cross-sectional Γ
    res['cross'] = cross_sectional_gamma(merged, model_key=model_key)

    # Within-person Γ
    res['within'] = within_person_gamma(merged, model_key=model_key)

    # Visit-pair Γ_change with bootstrap
    res['visit_pair'] = visit_pair_gamma(merged, model_key=model_key,
                                         n_bootstrap=N_BOOTSTRAP)

    # Individual SWDS-Γ
    complete = compute_individual_swds(merged, res['within'],
                                       model_key=model_key)

    # Rockwood FI
    if 'rockwood_fi' not in complete.columns:
        complete = compute_frailty_indices(complete)

    # Merge survival data
    if 'time_years' not in complete.columns:
        complete = complete.merge(
            surv[['idauniq', 'time_years', 'event']],
            on='idauniq', how='left')
        complete['time'] = complete['time_years']
        complete['deceased'] = complete['event']

    res['complete'] = complete

    # Cox models
    res['cox'] = run_cox_models_custom(complete, surv, model_key, swds_col_name)

    return res


# =========================================================================
# Strategy 1: Pulse-Only N Axis
# =========================================================================

def build_pulse_only_variant(merged):
    """
    Create 4-axis variant where N axis = pulse only (not SBP/DBP).
    Pulse is minimally affected by most antihypertensives (except beta-blockers).
    """
    print("\n" + "=" * 70)
    print("STRATEGY 1: PULSE-ONLY N AXIS")
    print("=" * 70)

    # dx_N_pulse: pulse-only neuroendocrine axis
    # pulval_z already computed in build_analysis_panel
    merged['dx_N_pulse'] = merged['pulval_z'].copy()

    # Complete-case flag: I_4, M_4, N_pulse, F_4
    merged['complete_pulse'] = (
        merged['dx_I_4'].notna() &
        merged['dx_M_4'].notna() &
        merged['dx_N_pulse'].notna() &
        merged['dx_F_4'].notna()
    )

    n_complete = merged['complete_pulse'].sum()
    n_persons = merged.loc[merged['complete_pulse'], 'idauniq'].nunique()
    print(f"  Pulse-only complete: {n_complete:,} person-visits, "
          f"{n_persons:,} unique persons")

    # Longitudinal flag
    vc = merged.loc[merged['complete_pulse']].groupby('idauniq').size()
    long_ids = vc[vc >= 2].index
    merged['in_longitudinal_pulse'] = merged['idauniq'].isin(long_ids)
    n_long = len(long_ids)
    print(f"  Longitudinal (≥2 visits): {n_long:,} persons")

    # Register model
    register_model(
        'pulse-only',
        axes=['dx_I_4', 'dx_M_4', 'dx_N_pulse', 'dx_F_4'],
        complete_col='complete_pulse',
        label='I, M, N(pulse), F',
    )

    return merged


# =========================================================================
# Strategy 2: Medication-Naive Subgroup
# =========================================================================

def build_medication_naive_subgroup(merged):
    """
    Restrict to individuals with NO diagnosed hypertension AND NO diabetes
    at baseline (wave 2).
    """
    print("\n" + "=" * 70)
    print("STRATEGY 2: MEDICATION-NAIVE SUBGROUP")
    print("=" * 70)

    # Baseline conditions from wave 2
    baseline = merged[merged['wave'] == 2][['idauniq', 'diabetes', 'highbp']].copy()
    baseline = baseline.rename(columns={'diabetes': 'bl_diabetes', 'highbp': 'bl_highbp'})

    # Medication-naive: no diabetes AND no hypertension at baseline
    baseline['med_naive'] = (
        (baseline['bl_diabetes'] != 1) &
        (baseline['bl_highbp'] != 1)
    )

    # Report prevalence
    n_total = len(baseline)
    n_diab = (baseline['bl_diabetes'] == 1).sum()
    n_hbp = (baseline['bl_highbp'] == 1).sum()
    n_naive = baseline['med_naive'].sum()
    print(f"  Baseline (wave 2): {n_total:,} persons")
    print(f"  Diabetes at baseline: {n_diab:,} ({100*n_diab/max(n_total,1):.1f}%)")
    print(f"  Hypertension at baseline: {n_hbp:,} ({100*n_hbp/max(n_total,1):.1f}%)")
    print(f"  Medication-naive (neither): {n_naive:,} ({100*n_naive/max(n_total,1):.1f}%)")

    # Merge flag back to full panel
    merged = merged.merge(baseline[['idauniq', 'med_naive']], on='idauniq', how='left')
    merged['med_naive'] = merged['med_naive'].fillna(False)

    naive_panel = merged[merged['med_naive']].copy()
    n_naive_complete_3 = naive_panel['complete_3axis'].sum() if 'complete_3axis' in naive_panel.columns else 0
    n_naive_complete_4 = naive_panel['complete_4axis'].sum() if 'complete_4axis' in naive_panel.columns else 0
    print(f"\n  Naive subgroup panel: {len(naive_panel):,} person-visits")
    print(f"  3-axis complete: {n_naive_complete_3:,}")
    print(f"  4-axis complete: {n_naive_complete_4:,}")

    # Rebuild longitudinal flags for naive subgroup
    for model_key, cc, lc in [
        ('3-axis', 'complete_3axis', 'in_longitudinal'),
        ('4-axis', 'complete_4axis', 'in_longitudinal_4'),
    ]:
        if cc in naive_panel.columns:
            vc = naive_panel.loc[naive_panel[cc]].groupby('idauniq').size()
            long_ids = vc[vc >= 2].index
            naive_panel[lc] = naive_panel['idauniq'].isin(long_ids)
            n_long = len(long_ids)
            print(f"  {model_key} longitudinal (≥2 visits): {n_long:,} persons")

    return merged, naive_panel


# =========================================================================
# Strategy 3: Medication-Robust Composite
# =========================================================================

def build_medication_robust_variant(merged):
    """
    Create medication-robust axis composites:
      I: log(CRP) + fibrinogen (unchanged)
      M: HbA1c + log(triglycerides) + waist (drop total chol, HDL)
      N: pulse only (not BP)
      F: grip + gait (unchanged)
    """
    print("\n" + "=" * 70)
    print("STRATEGY 3: MEDICATION-ROBUST COMPOSITE")
    print("=" * 70)

    # I axis: same as original 4-axis (CRP + fibrinogen not medication targets)
    # dx_I_4 already exists — reuse
    print("  I axis: dx_I_4 (log(CRP) + fibrinogen) — unchanged")

    # M axis: HbA1c + log(triglycerides) + waist circumference
    # DROP: total cholesterol (statin target), HDL (statin-affected)
    # KEEP: HbA1c, log(triglycerides), waist circumference
    m_robust_cols = ['hba1c_z', 'log_trig_z']
    has_waist = 'wstval_z' in merged.columns and merged['wstval_z'].notna().any()
    if has_waist:
        m_robust_cols.append('wstval_z')
        print(f"  M axis: HbA1c + log(trig) + waist circumference")
    else:
        print(f"  M axis: HbA1c + log(trig) (waist not available)")

    merged['dx_M_robust'] = merged[m_robust_cols].mean(axis=1, skipna=True)

    # N axis: pulse only (from Strategy 1)
    # dx_N_pulse already computed
    print("  N axis: pulse only (dx_N_pulse)")

    # F axis: same as original (grip + gait not pharmacologically managed)
    print("  F axis: dx_F_4 (grip + gait) — unchanged")

    # Complete-case flag
    merged['complete_robust'] = (
        merged['dx_I_4'].notna() &
        merged['dx_M_robust'].notna() &
        merged['dx_N_pulse'].notna() &
        merged['dx_F_4'].notna()
    )

    n_complete = merged['complete_robust'].sum()
    n_persons = merged.loc[merged['complete_robust'], 'idauniq'].nunique()
    print(f"\n  Robust complete: {n_complete:,} person-visits, "
          f"{n_persons:,} unique persons")

    # Longitudinal flag
    vc = merged.loc[merged['complete_robust']].groupby('idauniq').size()
    long_ids = vc[vc >= 2].index
    merged['in_longitudinal_robust'] = merged['idauniq'].isin(long_ids)
    print(f"  Longitudinal (≥2 visits): {len(long_ids):,} persons")

    # Register model
    register_model(
        'med-robust',
        axes=['dx_I_4', 'dx_M_robust', 'dx_N_pulse', 'dx_F_4'],
        complete_col='complete_robust',
        label='I, M(robust), N(pulse), F',
    )

    return merged


# =========================================================================
# N-axis variance diagnostic
# =========================================================================

def n_axis_variance_diagnostic(merged):
    """
    Print variance comparison for N-axis biomarkers:
    SBP and pulse variance in full sample vs medication-free subgroup.
    """
    print("\n" + "-" * 70)
    print("N-AXIS VARIANCE DIAGNOSTIC")
    print("-" * 70)

    # Identify people NOT on antihypertensives at their observation
    has_hemda = 'hemda' in merged.columns and merged['hemda'].notna().sum() > 50
    if not has_hemda:
        # Try to get hemda from highbp as proxy (diagnosed hypertension)
        if 'highbp' in merged.columns and merged['highbp'].notna().sum() > 50:
            print("  hemda not available, using highbp (diagnosed hypertension) as proxy")
            merged['_bp_med_proxy'] = merged['highbp']
            bp_col = '_bp_med_proxy'
        else:
            print("  Neither hemda nor highbp available — cannot stratify by BP medication")
            return {}
    else:
        bp_col = 'hemda'

    all_data = merged[merged['complete_4axis']].copy()
    no_bp_meds = all_data[all_data[bp_col] != 1]

    result = {}
    for var, label in [('sysval', 'SBP'), ('diaval', 'DBP'), ('pulval', 'Pulse')]:
        if var not in all_data.columns:
            continue
        var_all = all_data[var].dropna().var()
        var_no_meds = no_bp_meds[var].dropna().var()
        n_all = all_data[var].notna().sum()
        n_no_meds = no_bp_meds[var].notna().sum()

        ratio = var_no_meds / var_all if var_all > 0 else np.nan
        print(f"  {label:6s} variance (all):        {var_all:8.2f}  (N={n_all:,})")
        print(f"  {label:6s} variance (no BP meds):  {var_no_meds:8.2f}  (N={n_no_meds:,})")
        print(f"  {label:6s} ratio (no-meds/all):    {ratio:.3f}")
        if label in ('SBP', 'DBP'):
            expected = "HIGHER" if ratio > 1.0 else "lower"
            print(f"    → {expected} without meds (expected: HIGHER — medication compresses)")
        elif label == 'Pulse':
            expected = "similar" if 0.8 < ratio < 1.2 else "different"
            print(f"    → {expected} (expected: similar — less affected by most antihypertensives)")
        print()
        result[f'{label}_var_all'] = float(var_all)
        result[f'{label}_var_no_meds'] = float(var_no_meds)
        result[f'{label}_ratio'] = float(ratio)

    return result


# =========================================================================
# Correlation matrix comparison
# =========================================================================

def axis_correlation_comparison(merged, naive_panel):
    """
    Compare 4-axis correlation matrices: full sample vs med-naive.
    """
    print("\n" + "-" * 70)
    print("4-AXIS CORRELATION MATRIX COMPARISON")
    print("-" * 70)

    four_axes = ['dx_I_4', 'dx_M_4', 'dx_N_4', 'dx_F_4']
    axis_labels = ['I', 'M', 'N', 'F']

    corr_full = None
    corr_naive = None

    full_complete = merged[merged['complete_4axis']]
    if len(full_complete) > 20:
        corr_full = full_complete[four_axes].corr()
        print("\n  Full sample correlation matrix:")
        display = corr_full.copy()
        display.index = axis_labels
        display.columns = axis_labels
        print(display.to_string(float_format=lambda x: f'{x:+.3f}'))

    naive_complete = naive_panel[naive_panel['complete_4axis']] if 'complete_4axis' in naive_panel.columns else pd.DataFrame()
    if len(naive_complete) > 20:
        corr_naive = naive_complete[four_axes].corr()
        print("\n  Medication-naive correlation matrix:")
        display = corr_naive.copy()
        display.index = axis_labels
        display.columns = axis_labels
        print(display.to_string(float_format=lambda x: f'{x:+.3f}'))

    if corr_full is not None and corr_naive is not None:
        diff = corr_naive - corr_full
        print("\n  Difference (naive − full):")
        display = diff.copy()
        display.index = axis_labels
        display.columns = axis_labels
        print(display.to_string(float_format=lambda x: f'{x:+.3f}'))

    return corr_full, corr_naive


# =========================================================================
# Interaction test: SWDS-Γ × medication status
# =========================================================================

def medication_interaction_test(merged, surv):
    """
    Test whether SWDS-Γ × medication status predicts mortality differently.
    Only run if med-naive ΔC ≥ 0.01.
    """
    print("\n  Testing SWDS-Γ × medication-status interaction...")

    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("  lifelines not installed — skipping interaction test")
        return None

    if 'med_naive' not in merged.columns or 'swds_gamma_4' not in merged.columns:
        print("  Required columns not available")
        return None

    baseline = merged[(merged['wave'] == 2) & merged['complete_4axis']].copy()
    baseline = baseline.merge(surv[['idauniq', 'time_years', 'event']],
                              on='idauniq', how='left')
    baseline['time'] = baseline['time_years']
    baseline['deceased'] = baseline['event'].fillna(0).astype(int)
    baseline = baseline[baseline['time'].notna() & (baseline['time'] > 0)]

    if len(baseline) < 100:
        print("  Insufficient sample for interaction test")
        return None

    # Create interaction term
    baseline['med_naive_int'] = baseline['med_naive'].astype(float)
    baseline['swds_x_naive'] = baseline['swds_gamma_4'] * baseline['med_naive_int']

    covs = ['age', 'sex', 'swds_gamma_4', 'med_naive_int', 'swds_x_naive',
            'rockwood_fi']
    covs = [c for c in covs if c in baseline.columns]
    surv_data = baseline[covs + ['time', 'deceased']].dropna()

    if len(surv_data) < 50 or surv_data['deceased'].sum() < 10:
        print("  Insufficient data after dropna")
        return None

    try:
        cph = CoxPHFitter()
        cph.fit(surv_data, duration_col='time', event_col='deceased')
        interaction_p = cph.summary.loc['swds_x_naive', 'p'] if 'swds_x_naive' in cph.summary.index else np.nan
        interaction_hr = np.exp(cph.summary.loc['swds_x_naive', 'coef']) if 'swds_x_naive' in cph.summary.index else np.nan
        print(f"  Interaction HR = {interaction_hr:.3f}, p = {interaction_p:.4f}")
        return {'interaction_hr': float(interaction_hr),
                'interaction_p': float(interaction_p)}
    except Exception as e:
        print(f"  Interaction test failed: {e}")
        return None


# =========================================================================
# Figure generation
# =========================================================================

def make_medication_sensitivity_figure(all_results, corr_full, corr_naive):
    """
    Generate 4-panel comparison figure:
      (a) λ_max(Γ_change) trend: original vs pulse-only vs med-naive vs med-robust
      (b) ΔC comparison bar chart
      (c) 4-axis correlation matrix: full vs med-naive (side by side)
      (d) Γ_change off-diagonal structure: full vs med-naive
    """
    print("\nGenerating medication sensitivity figure...")

    fig = plt.figure(figsize=(16, 14))

    # ---- Panel (a): λ_max trends ----
    ax_a = fig.add_subplot(2, 2, 1)
    colors = {
        '3-axis (original)': 'steelblue',
        '4-axis (original)': 'darkred',
        '4-axis pulse-only N': 'darkorange',
        '4-axis med-naive': 'forestgreen',
        '4-axis med-robust': 'purple',
    }
    for label, res in all_results.items():
        vp = res.get('visit_pair')
        if vp is None or len(vp) == 0:
            continue
        vp = vp.sort_values('age_mid')
        color = colors.get(label, 'grey')
        ax_a.plot(vp['age_mid'], vp['lambda_max'], 'o-',
                  color=color, linewidth=2, markersize=6, label=label)
        if 'lambda_max_ci_lo' in vp.columns:
            ax_a.fill_between(vp['age_mid'],
                              vp['lambda_max_ci_lo'],
                              vp['lambda_max_ci_hi'],
                              alpha=0.15, color=color)
    ax_a.set_xlabel('Age stratum midpoint', fontsize=11)
    ax_a.set_ylabel('λ_max(Γ̂_change)', fontsize=11)
    ax_a.set_title('(a) Visit-pair λ_max(Γ̂_change) trend', fontsize=12,
                    fontweight='bold')
    ax_a.legend(fontsize=8, loc='upper left')

    # ---- Panel (b): ΔC bar chart ----
    ax_b = fig.add_subplot(2, 2, 2)
    dc_labels = []
    dc_values = []
    dc_colors = []
    dc_ns = []
    for label, res in all_results.items():
        cox = res.get('cox')
        if cox is None:
            continue
        dc = cox.get('_delta_c_m5_m4', np.nan)
        if np.isnan(dc):
            continue
        dc_labels.append(label.replace('4-axis ', '').replace('3-axis ', '3ax '))
        dc_values.append(dc)
        dc_colors.append(colors.get(label, 'grey'))
        # Get N from M5
        m5 = cox.get('M5: Full', {})
        dc_ns.append(m5.get('n', 0))

    if dc_values:
        y_pos = range(len(dc_labels))
        bars = ax_b.barh(y_pos, dc_values, color=dc_colors, alpha=0.8)
        ax_b.set_yticks(list(y_pos))
        ax_b.set_yticklabels(dc_labels, fontsize=9)
        ax_b.set_xlabel('ΔC (M5 − M4)', fontsize=11)
        ax_b.set_title('(b) ΔC: SWDS-Γ increment over Rockwood FI',
                        fontsize=12, fontweight='bold')
        ax_b.axvline(0.01, color='green', linestyle='--', alpha=0.7,
                     label='≥0.01 threshold')
        ax_b.axvline(0, color='grey', linestyle='-', alpha=0.3)
        ax_b.legend(fontsize=8)
        for i, (bar, dc, n) in enumerate(zip(bars, dc_values, dc_ns)):
            ax_b.text(dc + 0.001 if dc >= 0 else dc - 0.001,
                      i, f'{dc:+.4f} (N={n:,})',
                      va='center', fontsize=8,
                      ha='left' if dc >= 0 else 'right')

    # ---- Panel (c): Correlation matrices side by side ----
    ax_c1 = fig.add_subplot(2, 2, 3)
    axis_labels = ['I', 'M', 'N', 'F']

    if corr_full is not None:
        im1 = ax_c1.imshow(corr_full.values, cmap='RdBu_r', vmin=-1, vmax=1,
                            aspect='equal')
        ax_c1.set_xticks(range(4))
        ax_c1.set_yticks(range(4))
        ax_c1.set_xticklabels(axis_labels)
        ax_c1.set_yticklabels(axis_labels)
        ax_c1.set_title('(c-i) Full sample correlations', fontsize=11,
                         fontweight='bold')
        for i in range(4):
            for j in range(4):
                ax_c1.text(j, i, f'{corr_full.values[i, j]:.2f}',
                           ha='center', va='center', fontsize=10,
                           color='white' if abs(corr_full.values[i, j]) > 0.5 else 'black')
        plt.colorbar(im1, ax=ax_c1, fraction=0.046, pad=0.04)
    else:
        ax_c1.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax_c1.transAxes)
        ax_c1.set_title('(c-i) Full sample correlations', fontsize=11)

    # ---- Panel (d): Correlation matrix for med-naive ----
    ax_c2 = fig.add_subplot(2, 2, 4)
    if corr_naive is not None:
        im2 = ax_c2.imshow(corr_naive.values, cmap='RdBu_r', vmin=-1, vmax=1,
                            aspect='equal')
        ax_c2.set_xticks(range(4))
        ax_c2.set_yticks(range(4))
        ax_c2.set_xticklabels(axis_labels)
        ax_c2.set_yticklabels(axis_labels)
        ax_c2.set_title('(c-ii) Med-naive correlations', fontsize=11,
                         fontweight='bold')
        for i in range(4):
            for j in range(4):
                ax_c2.text(j, i, f'{corr_naive.values[i, j]:.2f}',
                           ha='center', va='center', fontsize=10,
                           color='white' if abs(corr_naive.values[i, j]) > 0.5 else 'black')
        plt.colorbar(im2, ax=ax_c2, fraction=0.046, pad=0.04)
    else:
        ax_c2.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax_c2.transAxes)
        ax_c2.set_title('(c-ii) Med-naive correlations', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Medication Sensitivity Analysis — ELSA 4-Axis Model',
                 fontsize=14, fontweight='bold')

    output_path = os.path.join(OUTPUT_DIR, 'figure_elsa_medication_sensitivity.pdf')
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path


# =========================================================================
# JSON output
# =========================================================================

def write_sensitivity_json(all_results, variance_diag, corr_full, corr_naive,
                           interaction_result, output_path):
    """Write machine-readable results ledger."""

    def safe_val(v):
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v) if not np.isnan(v) else None
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, pd.DataFrame):
            return None
        if isinstance(v, bool):
            return v
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

    def extract_vp(vp_df):
        if vp_df is None or len(vp_df) == 0:
            return []
        records = []
        for _, row in vp_df.iterrows():
            rec = {}
            for col in vp_df.columns:
                if col == 'Gamma_change':
                    rec[col] = row[col].tolist() if isinstance(row[col], np.ndarray) else None
                else:
                    rec[col] = safe_val(row[col])
            records.append(rec)
        return records

    ledger = {}

    # Per-model results
    for label, res in all_results.items():
        key = label.replace(' ', '_').replace('(', '').replace(')', '')
        entry = {
            'visit_pair': extract_vp(res.get('visit_pair')),
            'cox_models': extract_cox(res.get('cox')),
        }
        # Extract N complete
        complete = res.get('complete')
        if complete is not None:
            entry['n_complete'] = int(len(complete))
        ledger[key] = entry

    # Variance diagnostic
    ledger['n_axis_variance_diagnostic'] = variance_diag

    # Correlation matrices
    if corr_full is not None:
        ledger['correlation_matrix_full'] = corr_full.values.tolist()
    if corr_naive is not None:
        ledger['correlation_matrix_naive'] = corr_naive.values.tolist()

    # Interaction test
    if interaction_result:
        ledger['interaction_test'] = interaction_result

    # Summary ΔC table
    summary = {}
    for label, res in all_results.items():
        cox = res.get('cox')
        if cox:
            dc = cox.get('_delta_c_m5_m4', None)
            summary[label] = {
                'delta_c': safe_val(dc) if dc is not None else None,
                'exceeds_threshold': bool(dc is not None and not np.isnan(dc) and dc >= 0.01),
            }
    ledger['delta_c_summary'] = summary

    with open(output_path, 'w') as f:
        json.dump(ledger, f, indent=2, default=str)
    print(f"  Saved: {output_path}")


# =========================================================================
# Summary table
# =========================================================================

def print_summary_table(all_results):
    """Print the final medication sensitivity comparison table."""
    print("\n" + "=" * 70)
    print("MEDICATION SENSITIVITY ANALYSIS — SUMMARY")
    print("=" * 70)

    header = f"{'Model':<28s} {'ΔC(M5-M4)':>10s}   {'λ_max trend':>12s}   {'N complete':>10s}"
    print(header)
    print("-" * 70)

    for label, res in all_results.items():
        cox = res.get('cox')
        dc = cox.get('_delta_c_m5_m4', np.nan) if cox else np.nan
        dc_str = f"{dc:+.4f}" if not np.isnan(dc) else "N/A"

        # λ_max monotone trend
        vp = res.get('visit_pair')
        if vp is not None and len(vp) >= 3:
            tau, p = stats.kendalltau(vp['age_mid'], vp['lambda_max'])
            if tau > 0 and p < 0.05:
                trend_str = "monotone ✓"
            elif tau > 0:
                trend_str = f"positive (p={p:.2f})"
            else:
                trend_str = f"non-mono (τ={tau:.2f})"
        else:
            trend_str = "insufficient"

        # N complete
        complete = res.get('complete')
        n_complete = len(complete) if complete is not None else 0

        print(f"  {label:<26s} {dc_str:>10s}   {trend_str:>12s}   {n_complete:>10,}")

    print("-" * 70)
    print(f"  {'Threshold: ΔC ≥ 0.01':>40s}")

    # Full C-index comparison table
    print("\n" + "=" * 70)
    print("C-INDEX COMPARISON TABLE")
    print("=" * 70)

    # Collect all model labels and their cox results
    model_labels_ordered = ['M1: Age + Sex', 'M3: + SWDS-Γ',
                            'M4: + Rockwood FI', 'M5: Full']

    # Header
    col_labels = list(all_results.keys())
    col_short = [l.replace('4-axis ', '').replace('3-axis ', '3ax ')[:16]
                 for l in col_labels]

    header = f"{'Model':<20s}" + "".join(f"  {c:>16s}" for c in col_short)
    print(header)
    print("-" * (20 + 18 * len(col_short)))

    for ml in model_labels_ordered:
        row = f"  {ml:<18s}"
        for label in col_labels:
            cox = all_results[label].get('cox')
            if cox:
                c = cox.get(ml, {}).get('c_index', np.nan)
                row += f"  {c:>16.4f}" if not np.isnan(c) else f"  {'N/A':>16s}"
            else:
                row += f"  {'N/A':>16s}"
        print(row)

    # ΔC row
    row = f"  {'ΔC (M5-M4)':<18s}"
    for label in col_labels:
        cox = all_results[label].get('cox')
        dc = cox.get('_delta_c_m5_m4', np.nan) if cox else np.nan
        row += f"  {dc:>+16.4f}" if not np.isnan(dc) else f"  {'N/A':>16s}"
    print(row)
    print("=" * 70)


# =========================================================================
# Main
# =========================================================================

def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

    print("=" * 70)
    print("MEDICATION SENSITIVITY ANALYSIS")
    print("ELSA 4-Axis SWDS-Γ Model")
    print("=" * 70)

    # ---- Load data (reuse validation pipeline) ----
    print("\n--- Loading data ---")
    files = load_all_files()
    panel, hba1c_units = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)

    print("\nConverting harmonised to long format...")
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    print(f"  Long format: {harm_long.shape[0]:,} rows")

    # Build analysis panel (constructs 3-axis and 4-axis)
    merged = build_analysis_panel(panel, harm_long, mort, supp)

    # Survival data
    eol = files.get('eol')
    if eol is not None:
        eol_clean = eol.copy()
        eol_clean.columns = [c.lower() for c in eol_clean.columns]
        eol_clean = filter_missing(eol_clean, exclude_cols=['idauniq'])
    else:
        eol_clean = None
    surv = construct_survival_data(harm, eol_clean, baseline_wave=2)

    # Collect all results
    all_results = {}

    # ====================================================================
    # BASELINE: Run original 3-axis and 4-axis for comparison
    # ====================================================================
    print("\n" + "#" * 70)
    print("# BASELINE: ORIGINAL 3-AXIS MODEL")
    print("#" * 70)
    res_3ax = run_full_pipeline(merged, surv, '3-axis', 'swds_gamma')
    all_results['3-axis (original)'] = res_3ax

    print("\n" + "#" * 70)
    print("# BASELINE: ORIGINAL 4-AXIS MODEL")
    print("#" * 70)
    res_4ax = run_full_pipeline(merged, surv, '4-axis', 'swds_gamma_4')
    all_results['4-axis (original)'] = res_4ax

    # ====================================================================
    # STRATEGY 1: Pulse-Only N Axis
    # ====================================================================
    print("\n" + "#" * 70)
    print("# STRATEGY 1: PULSE-ONLY N AXIS")
    print("#" * 70)
    merged = build_pulse_only_variant(merged)

    # Need to add longitudinal column mapping for the pulse-only model
    # The within_person_gamma and visit_pair_gamma functions look for
    # 'in_longitudinal' or 'in_longitudinal_4'; for custom models we need
    # to set the right column.
    # Patch: temporarily add the column with the expected name
    # The code uses 'in_longitudinal_4' for non-3-axis models
    orig_in_long_4 = merged['in_longitudinal_4'].copy() if 'in_longitudinal_4' in merged.columns else None
    merged['in_longitudinal_4'] = merged['in_longitudinal_pulse']

    res_pulse = run_full_pipeline(merged, surv, 'pulse-only', 'swds_gamma_4')
    all_results['4-axis pulse-only N'] = res_pulse

    # Restore original longitudinal flag
    if orig_in_long_4 is not None:
        merged['in_longitudinal_4'] = orig_in_long_4

    # ====================================================================
    # STRATEGY 2: Medication-Naive Subgroup
    # ====================================================================
    print("\n" + "#" * 70)
    print("# STRATEGY 2: MEDICATION-NAIVE SUBGROUP")
    print("#" * 70)
    merged, naive_panel = build_medication_naive_subgroup(merged)

    # Run ORIGINAL 4-axis on naive subgroup
    print("\n  Running original 4-axis on medication-naive subgroup...")
    # The naive_panel needs its own longitudinal flags computed above
    orig_in_long_4_naive = naive_panel.get('in_longitudinal_4')

    res_naive_4ax = run_full_pipeline(naive_panel, surv, '4-axis', 'swds_gamma_4')
    all_results['4-axis med-naive'] = res_naive_4ax

    # Also run 3-axis on naive subgroup for comparison
    print("\n  Running 3-axis on medication-naive subgroup...")
    res_naive_3ax = run_full_pipeline(naive_panel, surv, '3-axis', 'swds_gamma')
    all_results['3-axis med-naive'] = res_naive_3ax

    # ====================================================================
    # STRATEGY 3: Medication-Robust Composite
    # ====================================================================
    print("\n" + "#" * 70)
    print("# STRATEGY 3: MEDICATION-ROBUST COMPOSITE")
    print("#" * 70)
    merged = build_medication_robust_variant(merged)

    # Patch longitudinal flag for robust model
    orig_in_long_4_2 = merged['in_longitudinal_4'].copy() if 'in_longitudinal_4' in merged.columns else None
    merged['in_longitudinal_4'] = merged['in_longitudinal_robust']

    res_robust = run_full_pipeline(merged, surv, 'med-robust', 'swds_gamma_4')
    all_results['4-axis med-robust'] = res_robust

    if orig_in_long_4_2 is not None:
        merged['in_longitudinal_4'] = orig_in_long_4_2

    # ====================================================================
    # DIAGNOSTICS
    # ====================================================================

    # N-axis variance diagnostic
    variance_diag = n_axis_variance_diagnostic(merged)

    # Correlation matrix comparison
    corr_full, corr_naive = axis_correlation_comparison(merged, naive_panel)

    # Interaction test (only if med-naive ΔC ≥ 0.01)
    interaction_result = None
    naive_cox = res_naive_4ax.get('cox')
    naive_dc = naive_cox.get('_delta_c_m5_m4', 0) if naive_cox else 0
    if naive_dc and not np.isnan(naive_dc) and naive_dc >= 0.01:
        print("\n  Med-naive ΔC ≥ 0.01 → running interaction test...")
        # Need swds_gamma_4 in full merged for the interaction test
        complete_4 = res_4ax.get('complete')
        if complete_4 is not None and 'swds_gamma_4' in complete_4.columns:
            # Merge swds_gamma_4 back to merged
            swds_map = complete_4[['idauniq', 'wave', 'swds_gamma_4']].drop_duplicates()
            if 'swds_gamma_4' not in merged.columns:
                merged = merged.merge(swds_map, on=['idauniq', 'wave'], how='left')
        interaction_result = medication_interaction_test(merged, surv)
    else:
        print(f"\n  Med-naive ΔC = {naive_dc if naive_dc else 'N/A'} "
              f"(below 0.01 → skipping interaction test)")

    # Med-naive subgroup comparison table
    print("\n" + "=" * 70)
    print("MED-NAIVE SUBGROUP COMPARISON TABLE")
    print("=" * 70)
    print(f"{'':28s} {'Full sample':>24s}   {'Med-naive subgroup':>24s}")
    print(f"{'':28s} {'3-axis':>12s} {'4-axis':>12s}   {'3-axis':>12s} {'4-axis':>12s}")
    print("-" * 80)

    model_order = ['M1: Age + Sex', 'M3: + SWDS-Γ', 'M4: + Rockwood FI', 'M5: Full']
    for ml in model_order:
        row = f"  {ml:<26s}"
        for res_key in ['3-axis (original)', '4-axis (original)',
                        '3-axis med-naive', '4-axis med-naive']:
            cox = all_results.get(res_key, {}).get('cox')
            if cox:
                c = cox.get(ml, {}).get('c_index', np.nan)
                row += f"  {c:>10.4f}" if not np.isnan(c) else f"  {'N/A':>10s}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)

    # ΔC row
    row = f"  {'ΔC (M5-M4)':<26s}"
    for res_key in ['3-axis (original)', '4-axis (original)',
                    '3-axis med-naive', '4-axis med-naive']:
        cox = all_results.get(res_key, {}).get('cox')
        dc = cox.get('_delta_c_m5_m4', np.nan) if cox else np.nan
        row += f"  {dc:>+10.4f}" if not np.isnan(dc) else f"  {'N/A':>10s}"
    print(row)
    print("=" * 80)

    # ====================================================================
    # SUMMARY TABLE
    # ====================================================================
    print_summary_table(all_results)

    # ====================================================================
    # FIGURE
    # ====================================================================
    make_medication_sensitivity_figure(all_results, corr_full, corr_naive)

    # ====================================================================
    # JSON OUTPUT
    # ====================================================================
    json_path = os.path.join(OUTPUT_DIR, 'elsa_medication_sensitivity.json')
    write_sensitivity_json(all_results, variance_diag, corr_full, corr_naive,
                           interaction_result, json_path)

    # ====================================================================
    # KEY DIAGNOSTIC CONCLUSIONS
    # ====================================================================
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTIC CONCLUSIONS")
    print("=" * 70)

    for label, res in all_results.items():
        cox = res.get('cox')
        if not cox:
            continue
        dc = cox.get('_delta_c_m5_m4', np.nan)
        if np.isnan(dc):
            continue
        status = "✓ EXCEEDS" if dc >= 0.01 else "✗ below"
        print(f"  {label:<28s}: ΔC = {dc:+.4f}  {status} 0.01 threshold")

    print()
    # Interpret
    naive_4_dc = all_results.get('4-axis med-naive', {}).get('cox', {}).get('_delta_c_m5_m4', np.nan)
    robust_dc = all_results.get('4-axis med-robust', {}).get('cox', {}).get('_delta_c_m5_m4', np.nan)
    pulse_dc = all_results.get('4-axis pulse-only N', {}).get('cox', {}).get('_delta_c_m5_m4', np.nan)

    if not np.isnan(naive_4_dc) and naive_4_dc >= 0.01:
        print("  HEADLINE: SWDS-Γ works when medication confound is absent.")
        print("  The medication-naive subgroup shows ΔC ≥ 0.01.")
        if not np.isnan(robust_dc) and robust_dc >= 0.01:
            print("  The medication-robust composite also recovers the effect")
            print("  → practical solution available for the general population.")
    elif not np.isnan(robust_dc) and robust_dc >= 0.01:
        print("  The medication-robust composite recovers ΔC ≥ 0.01.")
        print("  → biomarker selection matters more than subgroup restriction.")
    elif not np.isnan(pulse_dc) and pulse_dc >= 0.01:
        print("  Pulse-only N axis recovers ΔC ≥ 0.01.")
        print("  → BP medication is specifically the N-axis problem.")
    else:
        print("  None of the three strategies clear ΔC ≥ 0.01.")
        print("  The FI genuinely dominates at ELSA's biomarker resolution.")
        print("  Paper contribution: framework + covariance trend + medication insight,")
        print("  not the clinical-increment claim.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
