#!/usr/bin/env python3
"""
Figure 4: Mortality Prediction (two-cohort, R6)

Three-panel figure showing SWDS-Gamma mortality prediction across:
  InCHIANTI (primary):   4-axis {I, M, N, F}, Fried frailty, N=923, 698 deaths
                         (706 deaths in pre-matched N=931 age65+ subset;
                         8 frailty-incomplete subjects dropped for matched-N)
  ELSA (replication):    3-axis {I, M, F},    Rockwood FI,   N=5,431, 1,122 deaths

Panels:
  (a) Nested Cox C-indices (M1-M5) for both cohorts, dual y-axis because
      InCHIANTI (~0.66-0.76) and ELSA (~0.60-0.62) live on different scales.
  (b) Incremental Delta-C over M4 with clinical-relevance threshold at 0.01:
      InCHIANTI +0.014 (clears) vs ELSA +0.009 (below). InCHIANTI
      decomposition (biomarkers alone +0.014, SWDS-Gamma alone +0.007) is
      annotated below the main bars.
  (c) Kaplan-Meier survival curves by SWDS-Gamma tertile in the InCHIANTI
      matched age65+ baseline sample (N=923). Log-rank p and per-tertile N
      are printed in the panel.

Panels (a) and (b) render from frozen Cox JSONs (single source of truth):
  results/inchianti_cox_frozen.json  (matched N=923)
  results/elsa_cox_frozen.json       (matched N=5,431)

Panel (c) reads results/inchianti_baseline_matched.csv written by
scripts/inchianti_survival.py (same matched age65+ sample as the frozen JSON).

A runtime audit compares the live ELSA Cox run against the frozen ELSA JSON
and the live inchianti_survival_analysis.json against the frozen InCHIANTI
JSON, flagging drift > 5e-4.

Usage:
    python scripts/inchianti_survival.py          # regenerate InCHIANTI ledger + CSV
    python scripts/run_figure_mortality_prediction.py

Outputs:
    outputs/figure_mortality_prediction.pdf
    outputs/figure_mortality_prediction.png
"""

import importlib.util
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
from scipy import stats

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hdr_sim.estimation import compute_swds_gamma, gamma_stability_proxy
from hdr_sim.plotting import setup_style, add_panel_label, save_figure

# Import reusable functions from run_elsa_validation.py
_val_spec = importlib.util.spec_from_file_location(
    "run_elsa_validation",
    os.path.join(ROOT, 'scripts', 'run_elsa_validation.py'))
_val_mod = importlib.util.module_from_spec(_val_spec)
_val_spec.loader.exec_module(_val_mod)

load_all_files = _val_mod.load_all_files
extract_nurse_biomarkers = _val_mod.extract_nurse_biomarkers
prepare_harmonised = _val_mod.prepare_harmonised
extract_mortality = _val_mod.extract_mortality
extract_supplementary = _val_mod.extract_supplementary
harmonised_to_long = _val_mod.harmonised_to_long
build_analysis_panel = _val_mod.build_analysis_panel
construct_survival_data = _val_mod.construct_survival_data
within_person_gamma = _val_mod.within_person_gamma
compute_individual_swds = _val_mod.compute_individual_swds
compute_frailty_indices = _val_mod.compute_frailty_indices
compute_benchmark_scores = _val_mod.compute_benchmark_scores
filter_missing = _val_mod.filter_missing
zscore_vs_ref = _val_mod.zscore_vs_ref

AGE_STRATA = _val_mod.AGE_STRATA
AGE_STRATA_LABELS = _val_mod.AGE_STRATA_LABELS
NURSE_WAVE_YEARS = _val_mod.NURSE_WAVE_YEARS

# Import medication sensitivity functions
_med_spec = importlib.util.spec_from_file_location(
    "run_medication_sensitivity",
    os.path.join(ROOT, 'scripts', 'run_medication_sensitivity.py'))
_med_mod = importlib.util.module_from_spec(_med_spec)
_med_spec.loader.exec_module(_med_mod)

compute_swds_cross_sectional = _med_mod.compute_swds_cross_sectional
build_survival_baseline = _med_mod.build_survival_baseline
run_matched_cox = _med_mod.run_matched_cox
compute_rockwood_fi = _med_mod.compute_rockwood_fi

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 2026
np.random.seed(SEED)


def load_data():
    """Load and build the merged ELSA panel plus survival data."""
    print("=" * 70)
    print("FIGURE: MORTALITY PREDICTION")
    print("=" * 70)

    files = load_all_files()
    panel, _ = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    merged = build_analysis_panel(panel, harm_long, mort, supp)

    eol = files.get('eol')
    if eol is not None:
        eol_clean = eol.copy()
        eol_clean.columns = [c.lower() for c in eol_clean.columns]
        eol_clean = filter_missing(eol_clean, exclude_cols=['idauniq'])
    else:
        eol_clean = None

    return merged, harm, files


def compute_scores_and_cox(merged, harm):
    """Compute SWDS-Gamma, benchmarks, and run Cox models for full and med-naive."""
    axes_3 = ['dx_I', 'dx_M', 'dx_F']

    # --- Biomarker audit: print exact columns used ---
    print("\n--- Biomarker Audit (mortality prediction, 3-axis) ---")
    print(f"  Axes: {axes_3}")
    print(f"  Cox M2 bio_covs: ['log_crp', 'hba1c', 'grip_max', 'bmival']")
    print(f"  SWDS-Gamma uses: {axes_3} (cross-sectional stratum covariance)")
    for col in axes_3 + ['log_crp', 'hba1c', 'grip_max', 'bmival', 'swds_gamma']:
        if col in merged.columns:
            n_valid = merged[col].notna().sum()
            print(f"    {col}: {n_valid:,} non-null values")

    # --- SWDS-Gamma (cross-sectional, matching R5 pipeline) ---
    print("\n--- Computing SWDS-Gamma (cross-sectional) ---")
    complete_3 = compute_swds_cross_sectional(merged, axes_3, 'complete_3axis')

    # --- Rockwood FI ---
    complete_3['rockwood_fi'] = complete_3.apply(compute_rockwood_fi, axis=1)

    # --- Benchmark scores ---
    # Compute Mahalanobis and z-sum for full sample
    print("\n--- Computing benchmark scores ---")
    gamma_lookup = {}
    for (lo, hi) in AGE_STRATA:
        stratum = complete_3[(complete_3['age'] >= lo) & (complete_3['age'] < hi)]
        X = stratum[axes_3].dropna().values
        if len(X) >= 20:
            gamma_lookup[(lo, hi)] = np.cov(X.T)
    X_all = complete_3[axes_3].dropna().values
    Gamma_global = np.cov(X_all.T) if len(X_all) > 3 else np.eye(3)

    def get_gamma(age):
        for (lo, hi), G in gamma_lookup.items():
            if lo <= age < hi:
                return G
        return Gamma_global

    mahal_scores = []
    zsum_scores = []
    for _, row in complete_3.iterrows():
        dx = row[axes_3].values.astype(float)
        if np.any(np.isnan(dx)):
            mahal_scores.append(np.nan)
            zsum_scores.append(np.nan)
            continue
        G = get_gamma(row['age'])
        zsum_scores.append(np.dot(dx, dx) / len(axes_3))
        try:
            G_inv = np.linalg.inv(G)
            mahal_scores.append(dx @ G_inv @ dx)
        except np.linalg.LinAlgError:
            mahal_scores.append(np.nan)
    complete_3['mahalanobis'] = mahal_scores
    complete_3['z_sum'] = zsum_scores

    # --- Survival baseline ---
    print("\n--- Building survival baseline ---")
    baseline_full = build_survival_baseline(complete_3)
    print(f"  Full sample baseline: N={len(baseline_full):,}, "
          f"events={baseline_full['deceased'].sum():,}")

    # --- Med-naive subgroup ---
    print("\n--- Defining medication-naive subgroup ---")
    harm_clean = harm.copy()
    harm_clean.columns = [c.lower() for c in harm_clean.columns]
    hibpe_col = 'r2hibpe'
    diabe_col = 'r2diabe'
    if hibpe_col in harm_clean.columns and diabe_col in harm_clean.columns:
        med_naive_ids = harm_clean[
            (harm_clean[hibpe_col] == 0) & (harm_clean[diabe_col] == 0)
        ]['idauniq'].values
    else:
        # Fallback
        baseline_nurse = merged[merged['wave'] == 2]
        med_naive_ids = baseline_nurse[
            (baseline_nurse.get('hemda', pd.Series(dtype=float)) != 1) &
            (baseline_nurse.get('hemdb', pd.Series(dtype=float)) != 1)
        ]['idauniq'].values
    print(f"  Med-naive IDs: {len(med_naive_ids):,}")

    baseline_naive = baseline_full[baseline_full['idauniq'].isin(med_naive_ids)].copy()
    print(f"  Med-naive baseline: N={len(baseline_naive):,}, "
          f"events={baseline_naive['deceased'].sum():,}")

    # --- Cox models ---
    print("\n--- Running Cox models (full sample, 3-axis) ---")
    cox_full = run_matched_cox(baseline_full, axes_3,
                               swds_col='swds_gamma', model_label='3-axis')

    print("\n--- Running Cox models (med-naive, 3-axis) ---")
    cox_naive = run_matched_cox(baseline_naive, axes_3,
                                swds_col='swds_gamma', model_label='3-axis-med-naive')

    # --- Benchmark Cox models (full sample only) ---
    print("\n--- Running benchmark Cox models (full sample) ---")
    cox_benchmarks = run_benchmark_cox(baseline_full)

    return {
        'cox_full': cox_full,
        'cox_naive': cox_naive,
        'cox_benchmarks': cox_benchmarks,
        'baseline_full': baseline_full,
        'baseline_naive': baseline_naive,
        'complete_3': complete_3,
    }


def run_benchmark_cox(baseline):
    """Run Cox models replacing SWDS-Gamma with Mahalanobis/z-sum."""
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        return {}

    base_covs = ['age', 'sex']
    adj_covs = ['smoking', 'diabetes', 'highbp']
    bio_covs = ['log_crp', 'hba1c', 'grip_max', 'bmival']

    results = {}
    for score_name, score_col in [('Mahalanobis', 'mahalanobis'),
                                   ('z-sum', 'z_sum')]:
        models = {
            'M4: + Rockwood FI': base_covs + ['rockwood_fi'] + adj_covs,
            'M5_bench: Full': base_covs + bio_covs + [score_col, 'rockwood_fi'] + adj_covs,
        }

        all_vars = set()
        for covs in models.values():
            all_vars.update(covs)
        all_vars.update(['time', 'deceased'])
        all_vars = [v for v in all_vars if v in baseline.columns]

        matched = baseline[all_vars].dropna().copy()
        # Drop near-zero-variance
        dropped = [c for c in adj_covs if c in matched.columns and matched[c].std() < 0.01]
        if dropped:
            for name in models:
                models[name] = [v for v in models[name] if v not in dropped]
            all_vars_new = set()
            for covs in models.values():
                all_vars_new.update(covs)
            all_vars_new.update(['time', 'deceased'])
            all_vars_new = [v for v in all_vars_new if v in baseline.columns]
            matched = baseline[all_vars_new].dropna().copy()

        if len(matched) < 50:
            continue

        c_m4 = np.nan
        c_m5 = np.nan
        for name, covs in models.items():
            avail = [c for c in covs if c in matched.columns]
            surv_data = matched[avail + ['time', 'deceased']].copy()
            surv_data = surv_data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(surv_data) < 50 or surv_data['deceased'].sum() < 10:
                continue
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(surv_data, duration_col='time', event_col='deceased')
                c_idx = concordance_index(
                    surv_data['time'],
                    -cph.predict_partial_hazard(surv_data),
                    surv_data['deceased'])
                if 'M4' in name:
                    c_m4 = c_idx
                else:
                    c_m5 = c_idx
            except Exception:
                pass

        dc = c_m5 - c_m4 if not (np.isnan(c_m5) or np.isnan(c_m4)) else np.nan
        results[score_name] = {'delta_c': dc, 'c_m5': c_m5, 'c_m4': c_m4}
        print(f"  {score_name}: DC = {dc:+.4f}" if not np.isnan(dc) else f"  {score_name}: DC = N/A")

    return results


FROZEN_ELSA_PATH = os.path.join(ROOT, 'results', 'elsa_cox_frozen.json')
FROZEN_INCHIANTI_PATH = os.path.join(ROOT, 'results', 'inchianti_cox_frozen.json')


def _load_frozen(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _audit_against_frozen(cox_res, frozen, key_map, label):
    """Print audit line per model: frozen vs freshly-computed (should match to 3 d.p.)."""
    print(f"\n  AUDIT [{label}] frozen JSON vs live pipeline:")
    for mkey, live_key in key_map.items():
        frozen_c = frozen['models'].get(mkey, {}).get('C', None)
        live_c = cox_res.get(live_key, {}).get('c_index', None) if cox_res else None
        if frozen_c is None or live_c is None:
            print(f"    {mkey}: frozen={frozen_c}, live={live_c} (skip)")
            continue
        drift = abs(frozen_c - live_c)
        status = 'OK' if drift < 5e-4 else 'DRIFT'
        print(f"    {mkey}: frozen={frozen_c:.4f}, live={live_c:.4f}, "
              f"|delta|={drift:.4f} [{status}]")


def _audit_inchianti_against_frozen(frozen_inch, label):
    """Compare live results/inchianti_survival_analysis.json (age65+) vs frozen."""
    ledger_path = os.path.join(ROOT, 'results',
                                'inchianti_survival_analysis.json')
    print(f"\n  AUDIT [{label}] frozen JSON vs live ledger ({ledger_path}):")
    if not os.path.exists(ledger_path):
        print(f"    ledger missing — run scripts/inchianti_survival.py first")
        return
    with open(ledger_path, 'r', encoding='utf-8') as f:
        live = json.load(f)
    age65 = live.get('age65+', {})
    key_map = {
        'M1': 'M1_age_sex',
        'M2': 'M2_biomarkers',
        'M3': 'M3_swds',
        'M4': 'M4_frailty',
        'M4a': 'M4a_frailty_biomarkers',
        'M4b': 'M4b_frailty_swds',
        'M5': 'M5_full',
    }
    for mkey, live_key in key_map.items():
        frozen_c = frozen_inch['models'].get(mkey, {}).get('C', None)
        live_entry = age65.get(live_key, {})
        live_c = live_entry.get('C') if isinstance(live_entry, dict) else None
        if frozen_c is None or live_c is None:
            print(f"    {mkey}: frozen={frozen_c}, live={live_c} (skip)")
            continue
        drift = abs(frozen_c - live_c)
        status = 'OK' if drift < 5e-4 else 'DRIFT'
        print(f"    {mkey}: frozen={frozen_c:.4f}, live={live_c:.4f}, "
              f"|delta|={drift:.4f} [{status}]")
    live_N = age65.get('N')
    live_events = age65.get('n_events')
    print(f"    sample: frozen N={frozen_inch['N']} events="
          f"{frozen_inch['n_events']}   live N={live_N} events={live_events}")


INCHIANTI_BASELINE_CSV = os.path.join(ROOT, 'results',
                                       'inchianti_baseline_matched.csv')


def make_figure(results):
    """Three-panel two-cohort mortality prediction figure.

    Panel (a) and (b) render C-indices and delta-C values from frozen JSONs
    (`results/elsa_cox_frozen.json`, `results/inchianti_cox_frozen.json`)
    which are the single source of truth. Panel (c) reads the matched
    InCHIANTI baseline (`results/inchianti_baseline_matched.csv`, N=923)
    written by scripts/inchianti_survival.py and plots KM by SWDS-Gamma
    tertile.

    Live-vs-frozen audits are run for both cohorts; drift > 5e-4 flags.
    """
    setup_style()

    cox_full = results['cox_full']

    # -- Single source of truth: frozen JSON per cohort --
    frozen_elsa = _load_frozen(FROZEN_ELSA_PATH)
    frozen_inch = _load_frozen(FROZEN_INCHIANTI_PATH)

    # Live-vs-frozen audits
    elsa_key_map = {
        'M1': 'M1: Age + Sex',
        'M2': 'M2: + Biomarkers',
        'M3': 'M3: + SWDS-G',
        'M4': 'M4: + Rockwood FI',
        'M5': 'M5: Full',
    }
    _audit_against_frozen(cox_full, frozen_elsa, elsa_key_map,
                          f'ELSA matched N={frozen_elsa["N"]}, '
                          f'events={frozen_elsa["n_events"]}')
    _audit_inchianti_against_frozen(
        frozen_inch,
        f'InCHIANTI age65+ matched N={frozen_inch["N"]}, '
        f'events={frozen_inch["n_events"]}')

    # ---- Figure layout ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle(
        f'Two-cohort mortality prediction  |  '
        f'InCHIANTI age65+ N={frozen_inch["N"]:,} '
        f'(events={frozen_inch["n_events"]:,}, Fried frailty)   \u00b7   '
        f'ELSA N={frozen_elsa["N"]:,} '
        f'(events={frozen_elsa["n_events"]:,}, Rockwood FI)',
        fontsize=9.5, y=0.995)

    # ================================================================
    # Panel (a): Dual-y grouped C-index bars — InCHIANTI + ELSA
    # ================================================================
    ax_inch = axes[0]
    add_panel_label(ax_inch, '(a)')
    ax_elsa = ax_inch.twinx()

    model_labels = ['M1', 'M2', 'M3', 'M4', 'M5']
    c_inch = [frozen_inch['models'][m]['C'] for m in model_labels]
    c_elsa = [frozen_elsa['models'][m]['C'] for m in model_labels]

    x = np.arange(len(model_labels))
    width = 0.38

    col_inch = '#b8437a'
    col_elsa = 'steelblue'

    bars_i = ax_inch.bar(
        x - width / 2, c_inch, width,
        label=f'InCHIANTI age65+ (N={frozen_inch["N"]:,}, '
              f'{frozen_inch["n_events"]:,} deaths)',
        color=col_inch, alpha=0.88, zorder=3)
    bars_e = ax_elsa.bar(
        x + width / 2, c_elsa, width,
        label=f'ELSA (N={frozen_elsa["N"]:,}, '
              f'{frozen_elsa["n_events"]:,} deaths)',
        color=col_elsa, alpha=0.88, zorder=3)

    ax_inch.set_xticks(x)
    ax_inch.set_xticklabels(model_labels, fontsize=9)
    ax_inch.set_xlabel('Model')
    ax_inch.set_ylabel('InCHIANTI C-index (Harrell)', color=col_inch)
    ax_elsa.set_ylabel('ELSA C-index (Harrell)', color=col_elsa)
    ax_inch.tick_params(axis='y', labelcolor=col_inch)
    ax_elsa.tick_params(axis='y', labelcolor=col_elsa)
    ax_inch.set_title('Nested Cox C-indices \u2014 both cohorts')
    # Leave headroom for the legend above the bars.
    ax_inch.set_ylim(0.60, 0.82)
    ax_elsa.set_ylim(0.55, 0.66)

    # Merge legends from twin axes
    lines = [bars_i, bars_e]
    labs = [b.get_label() for b in lines]
    ax_inch.legend(lines, labs, fontsize=7, loc='upper center',
                   bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)

    for bars, vals, offset, ax_ref in [
            (bars_i, c_inch, -width / 2, ax_inch),
            (bars_e, c_elsa, width / 2, ax_elsa)]:
        for i, c in enumerate(vals):
            ax_ref.text(i + offset, c + 0.002, f'{c:.3f}',
                        ha='center', va='bottom', fontsize=6.5)

    print("\n  Panel (a) C-indices (frozen, both cohorts):")
    for ml, ci, ce in zip(model_labels, c_inch, c_elsa):
        print(f"    {ml}: InCHIANTI-65+={ci:.4f}, ELSA={ce:.4f}")

    # ================================================================
    # Panel (b): ΔC over M4 with threshold + InCHIANTI decomposition
    # ================================================================
    ax = axes[1]
    add_panel_label(ax, '(b)')

    dc_inch = frozen_inch['delta_C']['M5_minus_M4']
    dc_elsa = frozen_elsa['delta_C']['M5_minus_M4']
    threshold = 0.01

    labels = [f'InCHIANTI\n(M5-M4)', f'ELSA\n(M5-M4)']
    values = [dc_inch, dc_elsa]
    # Clears threshold -> signal color; below -> muted
    colors = ['#2ca02c' if v >= threshold else '#c44e52' for v in values]

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.88,
                  zorder=3, width=0.55)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'$\Delta C$ (relative to M4)')
    ax.set_title(r'$\Delta C$ over M4 + InCHIANTI decomposition')
    ax.axhline(threshold, color='darkred', linestyle='--', linewidth=1.2,
               alpha=0.85,
               label=f'Clinical-relevance threshold (+{threshold:.2f})',
               zorder=2)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.3, zorder=1)

    # Leave plenty of headroom for the decomposition box + legend + bar labels.
    y_top = 0.025
    y_bot = -0.003
    ax.set_ylim(y_bot, y_top)

    for i, v in enumerate(values):
        status = 'clears' if v >= threshold else 'below'
        ax.text(i, v + 0.0006, f'{v:+.3f}\n({status})',
                ha='center', va='bottom', fontsize=8)

    # --- Decomposition annotation: biomarkers-alone vs SWDS-alone (InCHIANTI) ---
    dc_bio = frozen_inch['delta_C']['M4a_minus_M4']
    dc_swds = frozen_inch['delta_C']['M4b_minus_M4']
    decomp_text = (
        'InCHIANTI decomposition over M4:\n'
        f'  biomarkers alone (M4a-M4) = {dc_bio:+.3f}\n'
        f'  SWDS-$\\Gamma$ alone (M4b-M4) = {dc_swds:+.3f}')
    ax.text(0.98, 0.98, decomp_text, transform=ax.transAxes,
            fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#f5f5f5',
                      edgecolor='#cccccc', linewidth=0.7, alpha=0.95))

    ax.legend(fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.13), frameon=False)

    print("\n  Panel (b) deltaC values (frozen):")
    print(f"    InCHIANTI M5-M4 = {dc_inch:+.4f} "
          f"({'clears' if dc_inch >= threshold else 'below'} threshold)")
    print(f"    ELSA M5-M4      = {dc_elsa:+.4f} "
          f"({'clears' if dc_elsa >= threshold else 'below'} threshold)")
    print(f"    InCHIANTI M4a-M4 (biomarkers alone) = {dc_bio:+.4f}")
    print(f"    InCHIANTI M4b-M4 (SWDS-Gamma alone) = {dc_swds:+.4f}")

    # ================================================================
    # Panel (c): KM by SWDS-Gamma tertile — InCHIANTI matched age65+
    # ================================================================
    ax = axes[2]
    add_panel_label(ax, '(c)')

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        if not os.path.exists(INCHIANTI_BASELINE_CSV):
            raise FileNotFoundError(
                f"Missing {INCHIANTI_BASELINE_CSV} — run "
                f"scripts/inchianti_survival.py first.")

        inch_base = pd.read_csv(INCHIANTI_BASELINE_CSV)
        print(f"\n  Loaded InCHIANTI matched baseline: "
              f"N={len(inch_base):,}, deaths={int(inch_base['event'].sum()):,}")

        surv = inch_base.dropna(subset=['swds_gamma', 'time_years',
                                         'event']).copy()
        surv = surv[surv['time_years'] > 0]
        surv['tertile'] = pd.qcut(
            surv['swds_gamma'], 3,
            labels=['T1 (low)', 'T2 (mid)', 'T3 (high)'])

        tertile_summary = {}
        for tl in ['T1 (low)', 'T2 (mid)', 'T3 (high)']:
            tsub = surv[surv['tertile'] == tl]
            tertile_summary[tl] = (len(tsub), int(tsub['event'].sum()),
                                    float(tsub['swds_gamma'].median()))
            print(f"    {tl}: N={len(tsub):,}, "
                  f"deaths={int(tsub['event'].sum()):,}, "
                  f"median SWDS-G={tsub['swds_gamma'].median():.4f}")

        t1 = surv[surv['tertile'] == 'T1 (low)']
        t3 = surv[surv['tertile'] == 'T3 (high)']
        lr = logrank_test(t1['time_years'], t3['time_years'],
                          t1['event'], t3['event'])
        p_used = lr.p_value
        print(f"    Log-rank p (T1 vs T3, InCHIANTI matched) = {p_used:.4g}")

        colors_km = ['forestgreen', 'orange', 'crimson']
        kmf = KaplanMeierFitter()
        for label, color in zip(['T1 (low)', 'T2 (mid)', 'T3 (high)'],
                                 colors_km):
            mask = surv['tertile'] == label
            if mask.sum() > 5:
                n_t, ev_t, _ = tertile_summary[label]
                kmf.fit(surv.loc[mask, 'time_years'],
                        surv.loc[mask, 'event'],
                        label=f'{label}  (n={n_t:,}, d={ev_t:,})')
                kmf.plot_survival_function(ax=ax, color=color,
                                           linewidth=1.8, ci_show=False)

        if p_used < 0.001:
            p_str = r'$p_{\mathrm{log\text{-}rank}} < 0.001$'
        else:
            p_str = fr'$p_{{\mathrm{{log\text{{-}}rank}}}} = {p_used:.4f}$'
        ax.text(0.02, 0.04, p_str, transform=ax.transAxes,
                fontsize=8, va='bottom')

        ax.set_xlabel('Years from baseline')
        ax.set_ylabel('Survival probability')
        ax.set_title(r'KM by SWDS-$\Gamma$ tertile — '
                     f'InCHIANTI age65+ (N={len(surv):,})')
        ax.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 0.08))
        ax.set_ylim(0, 1.02)
    except ImportError:
        ax.text(0.5, 0.5, 'lifelines not installed', ha='center',
                va='center', transform=ax.transAxes)
    except FileNotFoundError as e:
        ax.text(0.5, 0.5, str(e), ha='center', va='center',
                transform=ax.transAxes, fontsize=7, wrap=True)

    plt.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(fig, 'figure_mortality_prediction', OUTPUT_DIR)
    plt.close(fig)


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')
    warnings.filterwarnings('ignore', category=FutureWarning)

    merged, harm, files = load_data()
    results = compute_scores_and_cox(merged, harm)
    make_figure(results)

    print("\n" + "=" * 70)
    print("DONE: figure_mortality_prediction.pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
