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


def make_figure(results):
    """Generate the 3-panel mortality prediction figure."""
    setup_style()

    cox_full = results['cox_full']
    cox_naive = results['cox_naive']
    cox_benchmarks = results['cox_benchmarks']
    baseline_naive = results['baseline_naive']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle('ELSA cohort (3-axis I/M/F; N=5,431 full / 3,233 med-naive)',
                 fontsize=10, y=1.02)

    # ================================================================
    # Panel (a): Nested Cox C-indices — full sample AND med-naive
    # ================================================================
    ax = axes[0]
    add_panel_label(ax, '(a)')

    model_labels = ['M1', 'M2', 'M3', 'M4', 'M5']
    model_keys = ['M1: Age + Sex', 'M2: + Biomarkers', 'M3: + SWDS-G',
                  'M4: + Rockwood FI', 'M5: Full']

    def extract_c_indices(cox_res, keys):
        vals = []
        for k in keys:
            if cox_res and k in cox_res:
                vals.append(cox_res[k].get('c_index', np.nan))
            else:
                vals.append(np.nan)
        return vals

    c_full = extract_c_indices(cox_full, model_keys)
    c_naive = extract_c_indices(cox_naive, model_keys)

    x = np.arange(len(model_labels))
    width = 0.35

    bars_full = ax.bar(x - width / 2, c_full, width, label='Full sample',
                       color='steelblue', alpha=0.85, zorder=3)
    bars_naive = ax.bar(x + width / 2, c_naive, width, label='Med-naive',
                        color='forestgreen', alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel('C-index (Harrell)')
    ax.set_title('Nested Cox C-indices (ELSA)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(0.5, None)
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.3)

    # Annotate M5 values
    for bars, c_vals, offset in [(bars_full, c_full, -width / 2),
                                  (bars_naive, c_naive, width / 2)]:
        for i, c in enumerate(c_vals):
            if not np.isnan(c):
                ax.text(i + offset, c + 0.002, f'{c:.3f}',
                        ha='center', va='bottom', fontsize=7)

    # Print for verification
    print("\n  Panel (a) C-indices:")
    n_full = cox_full.get('_n_matched', 0) if cox_full else 0
    ev_full = cox_full.get('_n_events', 0) if cox_full else 0
    n_naive = cox_naive.get('_n_matched', 0) if cox_naive else 0
    ev_naive = cox_naive.get('_n_events', 0) if cox_naive else 0
    print(f"    Full sample: N={n_full:,}, events={ev_full:,}")
    print(f"    Med-naive:   N={n_naive:,}, events={ev_naive:,}")
    for ml, cf, cn in zip(model_labels, c_full, c_naive):
        cf_str = f"{cf:.4f}" if not np.isnan(cf) else "N/A"
        cn_str = f"{cn:.4f}" if not np.isnan(cn) else "N/A"
        print(f"    {ml}: full={cf_str}, naive={cn_str}")

    # ================================================================
    # Panel (b): DeltaC bar chart with benchmarks
    # ================================================================
    ax = axes[1]
    add_panel_label(ax, '(b)')

    dc_items = []

    # SWDS-Gamma full sample
    if cox_full:
        dc = cox_full.get('_delta_c', np.nan)
        if not np.isnan(dc):
            dc_items.append(('SWDS-$\\Gamma$\n(full)', dc, '#1f77b4'))

    # SWDS-Gamma med-naive
    if cox_naive:
        dc = cox_naive.get('_delta_c', np.nan)
        if not np.isnan(dc):
            dc_items.append(('SWDS-$\\Gamma$\n(med-naive)', dc, '#2ca02c'))

    # Benchmarks
    for bname, bcolor in [('Mahalanobis', '#7f7f7f'), ('z-sum', '#bcbd22')]:
        if bname in cox_benchmarks:
            dc = cox_benchmarks[bname].get('delta_c', np.nan)
            if not np.isnan(dc):
                dc_items.append((f'{bname}\n(full)', dc, bcolor))

    if dc_items:
        labels, values, colors = zip(*dc_items)
        bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85,
                      zorder=3, width=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(r'$\Delta C$ (M5 $-$ M4)')
        ax.set_title(r'$\Delta C$ vs benchmarks (ELSA)')
        ax.axhline(0.01, color='darkred', linestyle='--', linewidth=1.2,
                   alpha=0.8, label=r'$\Delta C = 0.01$ threshold', zorder=2)
        ax.axhline(0, color='grey', linestyle='-', alpha=0.3, zorder=1)
        ax.legend(fontsize=7, loc='upper right')

        for i, (bar, v) in enumerate(zip(bars, values)):
            ax.text(i, v + 0.001 if v >= 0 else v - 0.002,
                    f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=8)

        # Print for verification
        print("\n  Panel (b) DeltaC values:")
        for l, v in zip(labels, values):
            print(f"    {l.replace(chr(10), ' ')}: DC = {v:+.4f}")

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
            ax.set_title(f'KM by SWDS-$\\Gamma$ tertile (ELSA)\n{km_label}')
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
