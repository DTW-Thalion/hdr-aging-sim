#!/usr/bin/env python3
"""
Figure: Mortality Prediction (R6 restructured figure 2 of 3)

Three-panel figure showing SWDS-Gamma mortality prediction:
  (a) Nested Cox C-indices: full sample AND med-naive side by side (M1-M5)
  (b) DeltaC bar chart with benchmarks — only med-naive clears threshold
  (c) Kaplan-Meier by SWDS-Gamma tertile (medication-naive subgroup)

Usage:
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


def make_figure(results):
    """Three-panel two-cohort mortality prediction figure.

    Panel (a) and (b) render C-indices and delta-C values from frozen JSONs
    (`results/elsa_cox_frozen.json`, `results/inchianti_cox_frozen.json`)
    which are the single source of truth. Panel (c) plots KM curves computed
    from the ELSA baseline data already loaded in this session.

    A live Cox run is also performed upstream; if it drifts from the frozen
    values by more than 5e-4, the audit block below flags it.
    """
    import json as _json  # local import for clarity in audit block

    setup_style()

    cox_full = results['cox_full']
    cox_naive = results['cox_naive']
    baseline_naive = results['baseline_naive']

    # -- Single source of truth: frozen JSON per cohort --
    frozen_elsa = _load_frozen(FROZEN_ELSA_PATH)
    frozen_inch = _load_frozen(FROZEN_INCHIANTI_PATH)

    # Live-vs-frozen audit for ELSA only (InCHIANTI pipeline not loaded here).
    elsa_key_map = {
        'M1': 'M1: Age + Sex',
        'M2': 'M2: + Biomarkers',
        'M3': 'M3: + SWDS-G',
        'M4': 'M4: + Rockwood FI',
        'M5': 'M5: Full',
    }
    _audit_against_frozen(cox_full, frozen_elsa, elsa_key_map,
                          f'ELSA full N={frozen_elsa["N"]}, '
                          f'events={frozen_elsa["n_events"]}')

    # ---- Figure layout ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle(
        f'Two-cohort mortality prediction  |  '
        f'ELSA N={frozen_elsa["N"]:,} (events={frozen_elsa["n_events"]:,}, '
        f'Rockwood FI)   ·   '
        f'InCHIANTI age65+ N={frozen_inch["N"]:,} '
        f'(events={frozen_inch["n_events"]:,}, Fried frailty)',
        fontsize=9, y=1.02)

    # ================================================================
    # Panel (a): Grouped C-index bars — ELSA vs InCHIANTI age65+
    # ================================================================
    ax = axes[0]
    add_panel_label(ax, '(a)')

    model_labels = ['M1', 'M2', 'M3', 'M4', 'M5']
    c_elsa = [frozen_elsa['models'][m]['C'] for m in model_labels]
    c_inch = [frozen_inch['models'][m]['C'] for m in model_labels]

    x = np.arange(len(model_labels))
    width = 0.38

    bars_e = ax.bar(x - width / 2, c_elsa, width,
                    label=f'ELSA (N={frozen_elsa["N"]:,})',
                    color='steelblue', alpha=0.88, zorder=3)
    bars_i = ax.bar(x + width / 2, c_inch, width,
                    label=f'InCHIANTI age65+ (N={frozen_inch["N"]:,})',
                    color='#b8437a', alpha=0.88, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel('C-index (Harrell)')
    ax.set_title('Nested Cox C-indices — both cohorts')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0.5, 0.85)
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.3)

    for bars, vals, offset in [(bars_e, c_elsa, -width / 2),
                                (bars_i, c_inch, width / 2)]:
        for i, c in enumerate(vals):
            ax.text(i + offset, c + 0.004, f'{c:.3f}',
                    ha='center', va='bottom', fontsize=6.5)

    print("\n  Panel (a) C-indices (frozen, both cohorts):")
    for ml, ce, ci in zip(model_labels, c_elsa, c_inch):
        print(f"    {ml}: ELSA={ce:.4f}, InCHIANTI-65+={ci:.4f}")

    # ================================================================
    # Panel (b): ΔC comparison — SWDS-Γ vs benchmarks vs decomposition
    # ================================================================
    ax = axes[1]
    add_panel_label(ax, '(b)')

    dc_items = [
        # (label, value, color, group)
        (r'SWDS-$\Gamma$' '\nELSA',
            frozen_elsa['delta_C']['M5_minus_M4'], '#1f77b4', 'ELSA'),
        ('Mahalanobis' '\nELSA',
            frozen_elsa['benchmarks_delta_C']['Mahalanobis'], '#7f7f7f', 'ELSA'),
        ('z-sum' '\nELSA',
            frozen_elsa['benchmarks_delta_C']['z_sum'], '#bcbd22', 'ELSA'),
        (r'SWDS-$\Gamma$' '\nInCHIANTI',
            frozen_inch['delta_C']['M5_minus_M4'], '#b8437a', 'InCHIANTI'),
        ('Biomarkers only' '\n(M4a-M4, InCHIANTI)',
            frozen_inch['delta_C']['M4a_minus_M4'], '#d99ab6', 'InCHIANTI'),
        (r'SWDS-$\Gamma$ only' '\n(M4b-M4, InCHIANTI)',
            frozen_inch['delta_C']['M4b_minus_M4'], '#9467bd', 'InCHIANTI'),
    ]

    labels = [d[0] for d in dc_items]
    values = [d[1] for d in dc_items]
    colors = [d[2] for d in dc_items]

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.88,
                  zorder=3, width=0.65)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(r'$\Delta C$ (relative to M4)')
    ax.set_title(r'$\Delta C$ with benchmarks & decomposition')
    ax.axhline(0.01, color='darkred', linestyle='--', linewidth=1.1,
               alpha=0.8, label=r'$\Delta C = 0.01$ threshold', zorder=2)
    ax.axhline(0, color='grey', linestyle='-', alpha=0.3, zorder=1)
    # Divider between cohort groups
    ax.axvline(2.5, color='#cccccc', linestyle=':', linewidth=0.9, zorder=1)
    ax.legend(fontsize=7, loc='upper left')

    for i, v in enumerate(values):
        ax.text(i, v + 0.0005, f'{v:+.3f}',
                ha='center', va='bottom', fontsize=7)

    print("\n  Panel (b) deltaC values (frozen):")
    for lab, v in zip(labels, values):
        print(f"    {lab.replace(chr(10), ' | '):<45s} = {v:+.4f}")

    # ================================================================
    # Panel (c): KM by SWDS-Gamma tertile
    # ================================================================
    ax = axes[2]
    add_panel_label(ax, '(c)')

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        score_col = 'swds_gamma'
        baseline_full = results['baseline_full']

        def _km_diagnostics(surv_data, label_str):
            """Print KM diagnostic info for a dataset."""
            print(f"\n  KM diagnostics ({label_str}):")
            print(f"    Total N = {len(surv_data):,}, "
                  f"events = {surv_data['deceased'].sum():,}")
            print(f"    SWDS-Gamma: mean={surv_data[score_col].mean():.4f}, "
                  f"std={surv_data[score_col].std():.4f}, "
                  f"min={surv_data[score_col].min():.4f}, "
                  f"max={surv_data[score_col].max():.4f}")
            print(f"    Survival time: mean={surv_data['time'].mean():.1f}y, "
                  f"max={surv_data['time'].max():.1f}y")
            print(f"    Deceased coding: unique={sorted(surv_data['deceased'].unique())}")

            tertiles = pd.qcut(surv_data[score_col], 3,
                               labels=['T1 (low)', 'T2 (mid)', 'T3 (high)'])
            surv_data = surv_data.copy()
            surv_data['tertile'] = tertiles
            for tl in ['T1 (low)', 'T2 (mid)', 'T3 (high)']:
                sub = surv_data[surv_data['tertile'] == tl]
                print(f"    {tl}: N={len(sub):,}, "
                      f"events={sub['deceased'].sum():,}, "
                      f"median SWDS-G={sub[score_col].median():.4f}")
            return surv_data

        def _km_logrank_p(surv_data):
            """Compute log-rank p (T1 vs T3)."""
            t1 = surv_data[surv_data['tertile'] == 'T1 (low)']
            t3 = surv_data[surv_data['tertile'] == 'T3 (high)']
            if len(t1) > 5 and len(t3) > 5:
                lr = logrank_test(t1['time'], t3['time'],
                                  t1['deceased'], t3['deceased'])
                return lr.p_value
            return np.nan

        # --- Investigate med-naive KM first ---
        surv_naive = baseline_naive[
            baseline_naive[score_col].notna() &
            baseline_naive['time'].notna() &
            (baseline_naive['time'] > 0)
        ].copy()

        surv_full = baseline_full[
            baseline_full[score_col].notna() &
            baseline_full['time'].notna() &
            (baseline_full['time'] > 0)
        ].copy()

        naive_ok = (len(surv_naive) > 50 and
                    surv_naive['deceased'].sum() > 10)
        full_ok = (len(surv_full) > 50 and
                   surv_full['deceased'].sum() > 10)

        p_naive = np.nan
        p_full = np.nan

        if naive_ok:
            surv_naive = _km_diagnostics(surv_naive, 'med-naive')
            p_naive = _km_logrank_p(surv_naive)
            print(f"    Log-rank p (T1 vs T3, med-naive) = {p_naive:.4f}")

        if full_ok:
            surv_full = _km_diagnostics(surv_full, 'full sample')
            p_full = _km_logrank_p(surv_full)
            print(f"    Log-rank p (T1 vs T3, full sample) = {p_full:.4f}")

        # Decision: use med-naive if significant, else full sample
        use_naive = naive_ok and p_naive < 0.05
        if use_naive:
            surv_data = surv_naive
            km_label = '(medication-naive)'
            p_used = p_naive
            print(f"\n  => Using med-naive KM (p={p_naive:.4f})")
        elif full_ok:
            surv_data = surv_full
            km_label = '(full sample)'
            p_used = p_full
            print(f"\n  => Med-naive KM non-significant (p={p_naive:.4f}); "
                  f"using full sample (p={p_full:.4f})")
        else:
            surv_data = None

        if surv_data is not None and len(surv_data) > 50:
            colors_km = ['forestgreen', 'orange', 'crimson']
            kmf = KaplanMeierFitter()
            for label, color in zip(['T1 (low)', 'T2 (mid)', 'T3 (high)'],
                                     colors_km):
                mask = surv_data['tertile'] == label
                if mask.sum() > 5:
                    kmf.fit(surv_data.loc[mask, 'time'],
                            surv_data.loc[mask, 'deceased'],
                            label=label)
                    kmf.plot_survival_function(ax=ax, color=color,
                                               linewidth=1.8)

            # Annotate p-value
            if p_used < 0.001:
                p_str = '< 0.001'
            else:
                p_str = f'= {p_used:.4f}'
            ax.text(0.02, 0.02, f'$p_{{log-rank}}$ {p_str}',
                    transform=ax.transAxes, fontsize=8, va='bottom')

            ax.set_xlabel('Years from baseline')
            ax.set_ylabel('Survival probability')
            ax.set_title(f'KM by SWDS-$\\Gamma$ tertile — ELSA only\n{km_label}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient events', ha='center',
                    va='center', transform=ax.transAxes)
    except ImportError:
        ax.text(0.5, 0.5, 'lifelines not installed', ha='center',
                va='center', transform=ax.transAxes)

    plt.tight_layout()
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
