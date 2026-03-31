#!/usr/bin/env python3
"""
Figure: Coupling Tightening (R6 restructured figure 1 of 3)

Three-panel figure showing age-dependent tightening of multi-system coupling:
  (a) Visit-pair λ_max(Γ_change) age trend — the core within-person signal
  (b) Cross-sectional λ_max(Γ̂) — the artefact (variance compression)
  (c) SWDS-Γ distributions by age stratum — rightward shift with age

Usage:
    python scripts/run_figure_coupling_tightening.py

Outputs:
    outputs/figure_coupling_tightening.pdf
    outputs/figure_coupling_tightening.png
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

from hdr_sim.estimation import gamma_stability_proxy
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
cross_sectional_gamma = _val_mod.cross_sectional_gamma
within_person_gamma = _val_mod.within_person_gamma
visit_pair_gamma = _val_mod.visit_pair_gamma
compute_individual_swds = _val_mod.compute_individual_swds
filter_missing = _val_mod.filter_missing

AGE_STRATA = _val_mod.AGE_STRATA
AGE_STRATA_LABELS = _val_mod.AGE_STRATA_LABELS

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 2026
np.random.seed(SEED)

# Colours per age stratum
STRATUM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_data():
    """Load and build the merged ELSA analysis panel."""
    print("=" * 70)
    print("FIGURE: COUPLING TIGHTENING")
    print("=" * 70)

    files = load_all_files()
    panel, _ = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    merged = build_analysis_panel(panel, harm_long, mort, supp)
    return merged


def make_figure(merged):
    """Generate the 3-panel coupling tightening figure."""
    setup_style()

    # ---- Compute visit-pair analysis (3-axis) ----
    print("\n--- Computing visit-pair Gamma_change (3-axis) ---")
    vp_results = visit_pair_gamma(merged, model_key='3-axis', n_bootstrap=10000)

    # ---- Compute cross-sectional Gamma (3-axis) ----
    print("\n--- Computing cross-sectional Gamma (3-axis) ---")
    cs_results = cross_sectional_gamma(merged, model_key='3-axis')

    # ---- Compute within-person Gamma for SWDS-Gamma ----
    print("\n--- Computing within-person Gamma (3-axis) ---")
    within_results = within_person_gamma(merged, model_key='3-axis')

    # ---- Compute individual SWDS-Gamma scores ----
    print("\n--- Computing individual SWDS-Gamma scores ---")
    complete = compute_individual_swds(merged, within_results, model_key='3-axis')

    # ---- Build figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ================================================================
    # Panel (a): Visit-pair λ_max(Γ_change) age trend
    # ================================================================
    ax = axes[0]
    add_panel_label(ax, '(a)')

    if len(vp_results) > 0:
        vp = vp_results.sort_values('age_mid')
        x = vp['age_mid'].values
        y = vp['lambda_max'].values
        ci_lo = vp['lambda_max_ci_lo'].values
        ci_hi = vp['lambda_max_ci_hi'].values

        ax.bar(range(len(x)), y, color='steelblue', alpha=0.85, width=0.6,
               zorder=3)
        ax.errorbar(range(len(x)), y,
                    yerr=[y - ci_lo, ci_hi - y],
                    fmt='none', ecolor='black', capsize=5, capthick=1.2,
                    zorder=4)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(AGE_STRATA_LABELS)
        ax.set_xlabel('Age stratum')
        ax.set_ylabel(r'$\lambda_{\max}(\hat{\Gamma}_{\mathrm{change}})$')
        ax.set_title(r'Visit-pair $\lambda_{\max}(\hat{\Gamma}_{\mathrm{change}})$')

        # Jonckheere-Terpstra (approx via Kendall tau)
        tau_val, p_val = stats.kendalltau(x, y)
        ax.text(0.02, 0.98, f'$p_{{trend}}$ = {p_val:.4f}',
                transform=ax.transAxes, fontsize=9, va='top', ha='left')

        # Print values for verification
        print("\n  Panel (a) values:")
        for i, (label, val, lo, hi) in enumerate(zip(AGE_STRATA_LABELS, y, ci_lo, ci_hi)):
            print(f"    {label}: lambda_max = {val:.4f} [{lo:.4f}, {hi:.4f}]")
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # ================================================================
    # Panel (b): Cross-sectional λ_max(Γ̂) — the artefact
    # ================================================================
    ax = axes[1]
    add_panel_label(ax, '(b)')

    if len(cs_results) > 0:
        cs_agg = cs_results.groupby('age_group').agg(
            lambda_max_mean=('lambda_max', 'mean'),
            lambda_max_se=('lambda_max', 'sem'),
            age_mid=('age_mid', 'first'),
        ).reset_index()
        cs_agg = cs_agg.sort_values('age_mid')

        x = cs_agg['age_mid'].values
        y = cs_agg['lambda_max_mean'].values
        yerr = cs_agg['lambda_max_se'].values

        ax.bar(range(len(x)), y, color='coral', alpha=0.85, width=0.6,
               zorder=3)
        ax.errorbar(range(len(x)), y, yerr=yerr,
                    fmt='none', ecolor='black', capsize=5, capthick=1.2,
                    zorder=4)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(AGE_STRATA_LABELS)
        ax.set_xlabel('Age stratum')
        ax.set_ylabel(r'$\lambda_{\max}(\hat{\Gamma})$')
        ax.set_title(r'Cross-sectional $\lambda_{\max}(\hat{\Gamma})$')
        ax.text(0.5, -0.18, 'Cross-sectional artefact: variance compression',
                transform=ax.transAxes, fontsize=8, ha='center',
                style='italic', color='grey')

        # Print values for verification
        print("\n  Panel (b) values:")
        for label, val in zip(AGE_STRATA_LABELS, y):
            print(f"    {label}: lambda_max = {val:.4f}")
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # ================================================================
    # Panel (c): SWDS-Γ distributions by age stratum
    # ================================================================
    ax = axes[2]
    add_panel_label(ax, '(c)')

    score_col = 'swds_gamma'
    if complete is not None and score_col in complete.columns:
        from scipy.stats import gaussian_kde

        for i, ((lo, hi), label) in enumerate(zip(AGE_STRATA, AGE_STRATA_LABELS)):
            sub = complete[(complete['age'] >= lo) & (complete['age'] < hi)
                           & complete[score_col].notna()]
            if len(sub) > 30:
                vals = sub[score_col].values
                # Clip extreme outliers for KDE display
                p99 = np.percentile(vals, 99)
                vals_clipped = vals[vals <= p99]
                if len(vals_clipped) > 10:
                    kde = gaussian_kde(vals_clipped)
                    x_grid = np.linspace(0, p99, 200)
                    ax.plot(x_grid, kde(x_grid), color=STRATUM_COLORS[i],
                            linewidth=1.8, label=label)
                    ax.fill_between(x_grid, kde(x_grid), alpha=0.15,
                                    color=STRATUM_COLORS[i])

        ax.set_xlabel(r'SWDS-$\Gamma$')
        ax.set_ylabel('Density')
        ax.set_title(r'SWDS-$\Gamma$ by age stratum')
        ax.legend(fontsize=8, loc='upper right')

        # Print stratum means for verification
        print("\n  Panel (c) stratum means:")
        for (lo, hi), label in zip(AGE_STRATA, AGE_STRATA_LABELS):
            sub = complete[(complete['age'] >= lo) & (complete['age'] < hi)
                           & complete[score_col].notna()]
            if len(sub) > 0:
                print(f"    {label}: mean = {sub[score_col].mean():.4f}, "
                      f"N = {len(sub):,}")
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()
    save_figure(fig, 'figure_coupling_tightening', OUTPUT_DIR)
    plt.close(fig)


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

    merged = load_data()
    make_figure(merged)

    print("\n" + "=" * 70)
    print("DONE: figure_coupling_tightening.pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
