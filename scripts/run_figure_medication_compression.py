#!/usr/bin/env python3
"""
Figure: Medication Compression (R6 restructured figure 3 of 3)

Three-panel figure showing medication-induced variance compression:
  (a) Inter-axis correlation matrices: full sample vs med-naive (side by side)
  (b) lambda_max across model variants by age stratum
  (c) SWDS-Gamma vs age: full sample vs medication-naive overlay

Usage:
    python scripts/run_figure_medication_compression.py

Outputs:
    outputs/figure_medication_compression.pdf
    outputs/figure_medication_compression.png
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
from matplotlib.colors import TwoSlopeNorm
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
visit_pair_gamma = _val_mod.visit_pair_gamma
within_person_gamma = _val_mod.within_person_gamma
compute_individual_swds = _val_mod.compute_individual_swds
filter_missing = _val_mod.filter_missing

MODELS = _val_mod.MODELS
AGE_STRATA = _val_mod.AGE_STRATA
AGE_STRATA_LABELS = _val_mod.AGE_STRATA_LABELS
N_BOOTSTRAP = _val_mod.N_BOOTSTRAP
NURSE_WAVE_YEARS = _val_mod.NURSE_WAVE_YEARS

# Import medication sensitivity functions
_med_spec = importlib.util.spec_from_file_location(
    "run_elsa_medication_sensitivity",
    os.path.join(ROOT, 'scripts', 'run_elsa_medication_sensitivity.py'))
_med_mod = importlib.util.module_from_spec(_med_spec)
_med_spec.loader.exec_module(_med_mod)

build_pulse_only_variant = _med_mod.build_pulse_only_variant
build_medication_naive_subgroup = _med_mod.build_medication_naive_subgroup
build_medication_robust_variant = _med_mod.build_medication_robust_variant
register_model = _med_mod.register_model

# Also import cross-sectional SWDS from the corrected pipeline
_corr_spec = importlib.util.spec_from_file_location(
    "run_medication_sensitivity",
    os.path.join(ROOT, 'scripts', 'run_medication_sensitivity.py'))
_corr_mod = importlib.util.module_from_spec(_corr_spec)
_corr_spec.loader.exec_module(_corr_mod)

compute_swds_cross_sectional = _corr_mod.compute_swds_cross_sectional

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 2026
np.random.seed(SEED)

# Colours per age stratum
STRATUM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_data():
    """Load and build the merged ELSA panel."""
    print("=" * 70)
    print("FIGURE: MEDICATION COMPRESSION")
    print("=" * 70)

    files = load_all_files()
    panel, _ = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    merged = build_analysis_panel(panel, harm_long, mort, supp)
    return merged, harm, files


def compute_all_analyses(merged):
    """Run all analyses needed for the 3 panels."""
    axes_3 = ['dx_I', 'dx_M', 'dx_F']
    axes_4 = ['dx_I_4', 'dx_M_4', 'dx_N_4', 'dx_F_4']

    # ---- Panel (a): Correlation matrices ----
    print("\n--- Computing correlation matrices ---")
    four_axes = axes_4
    axis_labels_4 = ['I', 'M', 'N', 'F']

    full_complete = merged[merged['complete_4axis']]
    corr_full = None
    if len(full_complete) > 20:
        corr_full = full_complete[four_axes].corr()
        print("  Full sample correlation matrix computed")

    # Build medication-naive subgroup
    print("\n--- Building medication-naive subgroup ---")
    merged, naive_panel = build_medication_naive_subgroup(merged)

    naive_complete = naive_panel[naive_panel['complete_4axis']] \
        if 'complete_4axis' in naive_panel.columns else pd.DataFrame()
    corr_naive = None
    if len(naive_complete) > 20:
        corr_naive = naive_complete[four_axes].corr()
        print("  Med-naive correlation matrix computed")

    # ---- Panel (b): lambda_max across model variants ----
    print("\n--- Computing lambda_max across variants ---")
    variant_results = {}

    # 3-axis full
    print("\n  Variant: 3-axis (full)")
    vp_3full = visit_pair_gamma(merged, model_key='3-axis', n_bootstrap=300)
    variant_results['3-axis\nfull'] = vp_3full

    # 3-axis med-naive
    print("\n  Variant: 3-axis (med-naive)")
    # Rebuild longitudinal flags for naive panel
    if 'in_longitudinal' not in naive_panel.columns or naive_panel['in_longitudinal'].sum() == 0:
        vc = naive_panel.loc[naive_panel.get('complete_3axis', pd.Series(False))].groupby('idauniq').size()
        long_ids = vc[vc >= 2].index
        naive_panel['in_longitudinal'] = naive_panel['idauniq'].isin(long_ids)
    vp_3naive = visit_pair_gamma(naive_panel, model_key='3-axis', n_bootstrap=300)
    variant_results['3-axis\nmed-naive'] = vp_3naive

    # 4-axis full
    print("\n  Variant: 4-axis (full)")
    vp_4full = visit_pair_gamma(merged, model_key='4-axis', n_bootstrap=300)
    variant_results['4-axis\nfull'] = vp_4full

    # 4-axis med-naive
    print("\n  Variant: 4-axis (med-naive)")
    if 'in_longitudinal_4' not in naive_panel.columns or naive_panel['in_longitudinal_4'].sum() == 0:
        vc = naive_panel.loc[naive_panel.get('complete_4axis', pd.Series(False))].groupby('idauniq').size()
        long_ids = vc[vc >= 2].index
        naive_panel['in_longitudinal_4'] = naive_panel['idauniq'].isin(long_ids)
    vp_4naive = visit_pair_gamma(naive_panel, model_key='4-axis', n_bootstrap=300)
    variant_results['4-axis\nmed-naive'] = vp_4naive

    # 4-axis pulse-only
    print("\n  Variant: 4-axis (pulse-only N)")
    merged = build_pulse_only_variant(merged)
    # Register model in the validation module's MODELS dict so visit_pair_gamma sees it
    MODELS['pulse-only'] = {
        'axes': ['dx_I_4', 'dx_M_4', 'dx_N_pulse', 'dx_F_4'],
        'complete_col': 'complete_pulse',
        'label': 'I, M, N(pulse), F',
    }
    orig_long_4 = merged['in_longitudinal_4'].copy()
    merged['in_longitudinal_4'] = merged['in_longitudinal_pulse']
    vp_pulse = visit_pair_gamma(merged, model_key='pulse-only', n_bootstrap=300)
    variant_results['4-axis\npulse-N'] = vp_pulse
    merged['in_longitudinal_4'] = orig_long_4

    # 4-axis med-robust
    print("\n  Variant: 4-axis (med-robust)")
    merged = build_medication_robust_variant(merged)
    MODELS['med-robust'] = {
        'axes': ['dx_I_4', 'dx_M_robust', 'dx_N_pulse', 'dx_F_4'],
        'complete_col': 'complete_robust',
        'label': 'I, M(robust), N(pulse), F',
    }
    orig_long_4_2 = merged['in_longitudinal_4'].copy()
    merged['in_longitudinal_4'] = merged['in_longitudinal_robust']
    vp_robust = visit_pair_gamma(merged, model_key='med-robust', n_bootstrap=300)
    variant_results['4-axis\nmed-robust'] = vp_robust
    merged['in_longitudinal_4'] = orig_long_4_2

    # ---- Panel (c): SWDS-Gamma vs age overlay ----
    print("\n--- Computing SWDS-Gamma for full and med-naive ---")
    # Full sample SWDS-Gamma (3-axis, cross-sectional)
    complete_full = compute_swds_cross_sectional(merged, axes_3, 'complete_3axis')

    # Med-naive SWDS-Gamma
    complete_naive = compute_swds_cross_sectional(naive_panel, axes_3, 'complete_3axis')

    return {
        'corr_full': corr_full,
        'corr_naive': corr_naive,
        'variant_results': variant_results,
        'complete_full': complete_full,
        'complete_naive': complete_naive,
    }


def make_figure(analyses):
    """Generate the 3-panel medication compression figure."""
    setup_style()

    corr_full = analyses['corr_full']
    corr_naive = analyses['corr_naive']
    variant_results = analyses['variant_results']
    complete_full = analyses['complete_full']
    complete_naive = analyses['complete_naive']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # ================================================================
    # Panel (a): Inter-axis correlation matrices side by side
    # ================================================================
    ax = axes[0]
    add_panel_label(ax, '(a)')

    axis_labels = ['I', 'M', 'N', 'F']
    n_ax = len(axis_labels)

    if corr_full is not None and corr_naive is not None:
        # Combined heatmap: full on left, naive on right
        combined = np.full((n_ax, n_ax * 2 + 1), np.nan)
        combined[:, :n_ax] = corr_full.values
        combined[:, n_ax + 1:] = corr_naive.values

        # Use separate imshow calls for full and naive
        vmin, vmax = -1, 1
        cmap = 'RdBu_r'

        # Full sample (left)
        for i in range(n_ax):
            for j in range(n_ax):
                val = corr_full.values[i, j]
                color = plt.cm.RdBu_r((val + 1) / 2)
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=color, edgecolor='white',
                                     linewidth=0.5)
                ax.add_patch(rect)
                textcolor = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=textcolor)

        # Divider
        ax.axvline(n_ax - 0.5 + 0.5, color='black', linewidth=2)

        # Med-naive (right)
        offset = n_ax + 1
        for i in range(n_ax):
            for j in range(n_ax):
                val = corr_naive.values[i, j]
                color = plt.cm.RdBu_r((val + 1) / 2)
                rect = plt.Rectangle((offset + j - 0.5, i - 0.5), 1, 1,
                                     facecolor=color, edgecolor='white',
                                     linewidth=0.5)
                ax.add_patch(rect)
                textcolor = 'white' if abs(val) > 0.5 else 'black'
                ax.text(offset + j, i, f'{val:.2f}', ha='center',
                        va='center', fontsize=8, color=textcolor)

        ax.set_xlim(-0.5, 2 * n_ax + 0.5)
        ax.set_ylim(n_ax - 0.5, -0.5)
        ax.set_xticks(list(range(n_ax)) + [offset + k for k in range(n_ax)])
        ax.set_xticklabels(axis_labels * 2, fontsize=9)
        ax.set_yticks(range(n_ax))
        ax.set_yticklabels(axis_labels, fontsize=9)
        ax.set_title('Inter-axis correlations')

        # Sub-labels
        ax.text(n_ax / 2 - 0.5, n_ax + 0.3, 'Full sample', ha='center',
                fontsize=8, style='italic')
        ax.text(offset + n_ax / 2 - 0.5, n_ax + 0.3, 'Med-naive', ha='center',
                fontsize=8, style='italic', fontweight='bold')

        # Spines off for cleanliness
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        # Print for verification
        print("\n  Panel (a) off-diagonal correlations:")
        for i in range(n_ax):
            for j in range(i + 1, n_ax):
                cf = corr_full.values[i, j]
                cn = corr_naive.values[i, j]
                print(f"    {axis_labels[i]}-{axis_labels[j]}: "
                      f"full={cf:+.3f}, naive={cn:+.3f}, diff={cn-cf:+.3f}")
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes)

    # ================================================================
    # Panel (b): lambda_max across model variants by age stratum
    # ================================================================
    ax = axes[1]
    add_panel_label(ax, '(b)')

    variant_names = list(variant_results.keys())
    variant_colors = ['steelblue', '#1f77b4', 'darkred', '#d62728',
                      'darkorange', 'purple']
    markers = ['o', 's', 'o', 's', '^', 'D']

    any_plotted = False
    for idx, (vname, vp) in enumerate(variant_results.items()):
        if vp is None or len(vp) == 0:
            continue
        vp = vp.sort_values('age_mid')
        color = variant_colors[idx % len(variant_colors)]
        marker = markers[idx % len(markers)]
        ax.plot(vp['age_mid'], vp['lambda_max'], marker=marker,
                color=color, linewidth=1.5, markersize=5,
                label=vname, alpha=0.85)
        any_plotted = True

    if any_plotted:
        ax.set_xlabel('Age stratum midpoint')
        ax.set_ylabel(r'$\lambda_{\max}(\hat{\Gamma}_{\mathrm{change}})$')
        ax.set_title(r'$\lambda_{\max}$ across variants')
        ax.legend(fontsize=6.5, loc='upper left', ncol=2,
                  handlelength=1.5, columnspacing=0.8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)

    # Print for verification
    print("\n  Panel (b) lambda_max values:")
    for vname, vp in variant_results.items():
        if vp is None or len(vp) == 0:
            continue
        vp_sorted = vp.sort_values('age_mid')
        vals = ", ".join(f"{v:.3f}" for v in vp_sorted['lambda_max'])
        print(f"    {vname.replace(chr(10), ' ')}: [{vals}]")

    # ================================================================
    # Panel (c): SWDS-Gamma vs age — full vs med-naive overlay
    # ================================================================
    ax = axes[2]
    add_panel_label(ax, '(c)')

    score_col = 'swds_gamma'

    # Full sample (grey background)
    if complete_full is not None and score_col in complete_full.columns:
        valid_full = complete_full[complete_full[score_col].notna()]
        ax.scatter(valid_full['age'], valid_full[score_col],
                   alpha=0.05, s=3, color='lightgrey', zorder=1,
                   rasterized=True)

        # Full sample stratum means
        full_means_x = []
        full_means_y = []
        for (lo, hi) in AGE_STRATA:
            sub = valid_full[(valid_full['age'] >= lo) & (valid_full['age'] < hi)]
            if len(sub) > 0:
                full_means_x.append((lo + hi) / 2)
                full_means_y.append(sub[score_col].mean())
        if full_means_x:
            ax.plot(full_means_x, full_means_y, 'o--', color='grey',
                    markersize=8, linewidth=1.5, label='Full (mean)',
                    zorder=3)

    # Med-naive (coloured foreground)
    if complete_naive is not None and score_col in complete_naive.columns:
        valid_naive = complete_naive[complete_naive[score_col].notna()]

        for i, ((lo, hi), label) in enumerate(zip(AGE_STRATA, AGE_STRATA_LABELS)):
            sub = valid_naive[(valid_naive['age'] >= lo) & (valid_naive['age'] < hi)]
            if len(sub) > 0:
                ax.scatter(sub['age'], sub[score_col], alpha=0.15, s=5,
                           color=STRATUM_COLORS[i], zorder=2,
                           rasterized=True)

        # Med-naive stratum means
        naive_means_x = []
        naive_means_y = []
        for (lo, hi) in AGE_STRATA:
            sub = valid_naive[(valid_naive['age'] >= lo) & (valid_naive['age'] < hi)]
            if len(sub) > 0:
                naive_means_x.append((lo + hi) / 2)
                naive_means_y.append(sub[score_col].mean())
        if naive_means_x:
            ax.plot(naive_means_x, naive_means_y, 's-', color='darkgreen',
                    markersize=8, linewidth=2, label='Med-naive (mean)',
                    zorder=4, markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xlabel('Age')
    ax.set_ylabel(r'SWDS-$\Gamma$')
    ax.set_title(r'SWDS-$\Gamma$ vs age')
    ax.legend(fontsize=8, loc='upper left')

    # Print for verification
    print("\n  Panel (c) stratum means:")
    if full_means_x and naive_means_x:
        for label, fm, nm in zip(AGE_STRATA_LABELS, full_means_y, naive_means_y):
            print(f"    {label}: full={fm:.4f}, naive={nm:.4f}, "
                  f"diff={nm-fm:+.4f}")
        # Gradient comparison
        if len(full_means_y) >= 2 and len(naive_means_y) >= 2:
            full_grad = full_means_y[-1] - full_means_y[0]
            naive_grad = naive_means_y[-1] - naive_means_y[0]
            print(f"    Gradient (oldest-youngest): full={full_grad:+.4f}, "
                  f"naive={naive_grad:+.4f}")

    plt.tight_layout()
    save_figure(fig, 'figure_medication_compression', OUTPUT_DIR)
    plt.close(fig)


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')
    warnings.filterwarnings('ignore', category=FutureWarning)

    merged, harm, files = load_data()
    analyses = compute_all_analyses(merged)
    make_figure(analyses)

    print("\n" + "=" * 70)
    print("DONE: figure_medication_compression.pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
