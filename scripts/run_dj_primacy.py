#!/usr/bin/env python3
"""
D vs. J Primacy Decomposition of ELSA Age-Stratified Covariance
================================================================

Decomposes the age-dependent growth in Γ̂ into:
  - Within-axis variance growth (V_norm) — tracks D-degradation / τᵢ lengthening
  - Cross-axis correlation tightening (C_norm) — tracks J-degradation / coupling
  - Primacy ratio P = C_norm / V_norm — which mechanism dominates?

Theory:
  Under stationary OU with A = -D + J and diagonal Q:
    - diag(Γ̂) scales as qᵢ/(2dᵢ) — grows when D degrades (dᵢ shrinks)
    - Off-diagonal depends on both D and J
    - If J is fixed and only D degrades, correlation matrix R̂ stays ~constant
    - If J strengthens, off-diagonal correlations increase beyond D-decline alone
    - Rising C_norm relative to V_norm → J-degradation → hyperfunction theory

This is a Tier-1 analysis: uses only sample covariance Γ̂, no drift estimation,
no diffusion Q specification, no structural assumptions on A beyond the OU model.

Usage:
    python scripts/run_dj_primacy.py

Outputs:
    outputs/figure_dj_primacy.pdf / .png — 3-panel publication figure
    outputs/dj_primacy_results.txt       — numerical results and interpretation
    outputs/dj_primacy_results.json      — machine-readable results
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
from scipy import stats

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import data loading pipeline from existing validation script
from run_elsa_validation import (
    load_all_files,
    extract_nurse_biomarkers,
    prepare_harmonised,
    extract_mortality,
    extract_supplementary,
    harmonised_to_long,
    build_analysis_panel,
)

from hdr_sim.plotting import setup_style, add_panel_label, save_figure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# Finer 5-year bins (prompt specifies 50-55 as youthful reference)
AGE_STRATA_5Y = [
    (50, 55), (55, 60), (60, 65), (65, 70),
    (70, 75), (75, 80), (80, 85), (85, 90),
]
AGE_STRATA_5Y_LABELS = [
    '50–55', '55–60', '60–65', '65–70',
    '70–75', '75–80', '80–85', '85–90',
]
YOUNGEST_IDX = 0  # 50-55 is the reference

# 3-axis model
AXES = ['dx_I', 'dx_M', 'dx_F']
AXIS_LABELS = ['I (inflammaging)', 'M (metabolic)', 'F (functional)']
PAIR_LABELS = ['|r(I,M)|', '|r(I,F)|', '|r(M,F)|']

N_BOOTSTRAP = 2000
MIN_STRATUM_N = 30

# Colours consistent with repo style
COL_V = '#e74c3c'      # red — variance (D-degradation)
COL_C = '#3498db'       # blue — correlation (J-degradation)
COL_P = '#8e44ad'       # purple — primacy ratio
COL_P_NAIVE = '#27ae60' # green — primacy ratio (med-naive)


# ---------------------------------------------------------------------------
# Core decomposition functions
# ---------------------------------------------------------------------------
def compute_decomposition(X):
    """
    Given an N×3 data matrix, compute V, C, and per-axis/pair details.

    Returns dict with:
      Gamma_hat, V (mean diagonal), C (mean |off-diag correlation|),
      variances (per-axis), correlations (per-pair), R_hat (correlation matrix)
    """
    if len(X) < MIN_STRATUM_N:
        return None

    Gamma_hat = np.cov(X.T)
    n_axes = Gamma_hat.shape[0]

    # V: mean diagonal variance
    variances = np.diag(Gamma_hat)
    V = variances.mean()

    # R̂: correlation matrix = D^{-1/2} Γ̂ D^{-1/2}
    D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(variances, 1e-12)))
    R_hat = D_sqrt_inv @ Gamma_hat @ D_sqrt_inv

    # C: mean absolute off-diagonal correlation
    off_diag_abs = []
    pair_corrs = []
    for i in range(n_axes):
        for j in range(i + 1, n_axes):
            off_diag_abs.append(abs(R_hat[i, j]))
            pair_corrs.append(R_hat[i, j])  # signed for reporting
    C = np.mean(off_diag_abs)

    return {
        'Gamma_hat': Gamma_hat,
        'V': V,
        'C': C,
        'variances': variances,          # [Var(I), Var(M), Var(F)]
        'pair_correlations': pair_corrs,  # [r(I,M), r(I,F), r(M,F)]
        'R_hat': R_hat,
        'n': len(X),
    }


def bootstrap_decomposition(X, n_boot=N_BOOTSTRAP):
    """Bootstrap the decomposition to get CIs for V and C."""
    n = len(X)
    Vs, Cs = [], []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        result = compute_decomposition(X[idx])
        if result is not None:
            Vs.append(result['V'])
            Cs.append(result['C'])
    return np.array(Vs), np.array(Cs)


def run_decomposition_by_strata(merged, wave=2, label_prefix=''):
    """
    Run the D vs J decomposition across 5-year age strata for a given wave.

    Returns list of dicts with per-stratum results.
    """
    complete_mask = merged['complete_3axis']
    wave_data = merged[(merged['wave'] == wave) & complete_mask]

    print(f"\n  {label_prefix}Wave {wave}: {len(wave_data):,} complete observations")

    results = []
    for (lo, hi), label in zip(AGE_STRATA_5Y, AGE_STRATA_5Y_LABELS):
        stratum = wave_data[(wave_data['age'] >= lo) & (wave_data['age'] < hi)]
        X = stratum[AXES].dropna().values

        if len(X) < MIN_STRATUM_N:
            print(f"    {label}: N={len(X)} < {MIN_STRATUM_N}, skipping")
            continue

        decomp = compute_decomposition(X)
        V_boots, C_boots = bootstrap_decomposition(X)

        results.append({
            'age_group': label,
            'age_lo': lo,
            'age_hi': hi,
            'age_mid': (lo + hi) / 2,
            'n': decomp['n'],
            'V': decomp['V'],
            'C': decomp['C'],
            'V_ci': (np.percentile(V_boots, 2.5), np.percentile(V_boots, 97.5)),
            'C_ci': (np.percentile(C_boots, 2.5), np.percentile(C_boots, 97.5)),
            'variances': decomp['variances'].tolist(),
            'pair_correlations': decomp['pair_correlations'],
            'Gamma_hat': decomp['Gamma_hat'],
        })

        print(f"    {label}: N={decomp['n']:,}, V={decomp['V']:.4f}, "
              f"C={decomp['C']:.4f}")

    return results


def normalize_results(results):
    """
    Normalize V and C by the youngest stratum (index 0).
    Compute primacy ratio P = C_norm / V_norm.
    Also compute bootstrap CIs for normalized quantities.
    """
    if not results:
        return results

    V0 = results[0]['V']
    C0 = results[0]['C']

    for r in results:
        r['V_norm'] = r['V'] / V0
        r['C_norm'] = r['C'] / C0
        r['P'] = r['C_norm'] / r['V_norm'] if r['V_norm'] > 0 else np.nan

        # Propagate CIs (delta method approximation via bootstrap ratios)
        r['V_norm_ci'] = (r['V_ci'][0] / V0, r['V_ci'][1] / V0)
        r['C_norm_ci'] = (r['C_ci'][0] / C0, r['C_ci'][1] / C0)

        # P CI via bootstrap ratio (approximate)
        # P = C_norm / V_norm, so CI bounds combine
        if r['V_norm'] > 0:
            # Conservative: use worst-case combinations
            r['P_ci'] = (
                r['C_norm_ci'][0] / max(r['V_norm_ci'][1], 1e-12),
                r['C_norm_ci'][1] / max(r['V_norm_ci'][0], 1e-12),
            )
        else:
            r['P_ci'] = (np.nan, np.nan)

    return results


def fit_linear_trend(results, y_key='P'):
    """Fit OLS regression of y_key on age_mid. Return slope, p, CI."""
    ages = np.array([r['age_mid'] for r in results])
    ys = np.array([r[y_key] for r in results])

    mask = np.isfinite(ys)
    ages, ys = ages[mask], ys[mask]

    if len(ages) < 3:
        return {'slope': np.nan, 'p': np.nan, 'ci': (np.nan, np.nan),
                'intercept': np.nan, 'r2': np.nan}

    slope, intercept, r_value, p_value, std_err = stats.linregress(ages, ys)

    # 95% CI for slope
    t_crit = stats.t.ppf(0.975, df=len(ages) - 2)
    ci = (slope - t_crit * std_err, slope + t_crit * std_err)

    return {
        'slope': slope,
        'intercept': intercept,
        'p': p_value,
        'ci': ci,
        'r2': r_value**2,
        'std_err': std_err,
    }


def fit_quadratic(results, y_key='P'):
    """Test for nonlinearity with quadratic term."""
    ages = np.array([r['age_mid'] for r in results])
    ys = np.array([r[y_key] for r in results])

    mask = np.isfinite(ys)
    ages, ys = ages[mask], ys[mask]

    if len(ages) < 4:
        return {'quad_coeff': np.nan, 'quad_p': np.nan}

    # Fit: y = a + b*age + c*age^2
    X = np.column_stack([np.ones_like(ages), ages, ages**2])
    try:
        from numpy.linalg import lstsq
        coeffs, residuals, rank, sv = lstsq(X, ys, rcond=None)
        # Compare to linear model for F-test
        X_lin = np.column_stack([np.ones_like(ages), ages])
        coeffs_lin, res_lin, _, _ = lstsq(X_lin, ys, rcond=None)

        ss_res_full = np.sum((ys - X @ coeffs)**2)
        ss_res_red = np.sum((ys - X_lin @ coeffs_lin)**2)

        n = len(ages)
        if ss_res_full > 0 and n > 3:
            f_stat = ((ss_res_red - ss_res_full) / 1) / (ss_res_full / (n - 3))
            quad_p = 1 - stats.f.cdf(f_stat, 1, n - 3)
        else:
            quad_p = np.nan

        return {
            'quad_coeff': coeffs[2],
            'quad_p': quad_p,
            'coeffs': coeffs.tolist(),
        }
    except Exception:
        return {'quad_coeff': np.nan, 'quad_p': np.nan}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_dj_primacy(results_full, results_naive, fits_full, fits_naive,
                    output_name='figure_dj_primacy'):
    """Generate the 3-panel publication figure."""
    setup_style()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Panel (a): V_norm and C_norm vs age (full sample) ---
    ax = axes[0]
    ages = [r['age_mid'] for r in results_full]
    V_norms = [r['V_norm'] for r in results_full]
    C_norms = [r['C_norm'] for r in results_full]
    V_lo = [r['V_norm_ci'][0] for r in results_full]
    V_hi = [r['V_norm_ci'][1] for r in results_full]
    C_lo = [r['C_norm_ci'][0] for r in results_full]
    C_hi = [r['C_norm_ci'][1] for r in results_full]

    ax.plot(ages, V_norms, 'o-', color=COL_V, label=r'$V_{\rm norm}$ (variance growth)',
            linewidth=1.8, markersize=5)
    ax.fill_between(ages, V_lo, V_hi, color=COL_V, alpha=0.15)

    ax.plot(ages, C_norms, 's--', color=COL_C, label=r'$C_{\rm norm}$ (correlation tightening)',
            linewidth=1.8, markersize=5)
    ax.fill_between(ages, C_lo, C_hi, color=COL_C, alpha=0.15)

    ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Normalised index (ref = youngest)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Full sample')
    add_panel_label(ax, '(a)')

    # --- Panel (b): Primacy ratio P(s) — full sample ---
    ax = axes[1]
    Ps = [r['P'] for r in results_full]
    P_lo = [r['P_ci'][0] for r in results_full]
    P_hi = [r['P_ci'][1] for r in results_full]

    ax.plot(ages, Ps, 'D-', color=COL_P, linewidth=1.8, markersize=5,
            label='Full sample')
    ax.fill_between(ages, P_lo, P_hi, color=COL_P, alpha=0.15)

    # Linear fit
    fit = fits_full['P']
    if np.isfinite(fit['slope']):
        age_range = np.linspace(min(ages), max(ages), 100)
        y_fit = fit['intercept'] + fit['slope'] * age_range
        ax.plot(age_range, y_fit, '-', color=COL_P, alpha=0.5, linewidth=1.0)
        # Annotate
        p_str = f"p = {fit['p']:.3f}" if fit['p'] >= 0.001 else f"p < 0.001"
        ax.text(0.97, 0.03,
                f"slope = {fit['slope']:.4f}/yr\n{p_str}",
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='grey', alpha=0.8))

    ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel(r'Primacy ratio $P = C_{\rm norm} / V_{\rm norm}$')

    # Interpretation labels
    ax.text(0.03, 0.97, r'$P > 1$: coupling outpaces' + '\n' + 'capacity loss',
            transform=ax.transAxes, fontsize=7, ha='left', va='top', color=COL_C,
            fontstyle='italic')
    ax.text(0.03, 0.03, r'$P < 1$: capacity loss' + '\n' + 'outpaces coupling',
            transform=ax.transAxes, fontsize=7, ha='left', va='bottom', color=COL_V,
            fontstyle='italic')

    ax.set_title('Primacy ratio — full sample')
    add_panel_label(ax, '(b)')

    # --- Panel (c): Primacy ratio P(s) — medication-naive ---
    ax = axes[2]
    if results_naive:
        ages_n = [r['age_mid'] for r in results_naive]
        Ps_n = [r['P'] for r in results_naive]
        P_lo_n = [r['P_ci'][0] for r in results_naive]
        P_hi_n = [r['P_ci'][1] for r in results_naive]

        ax.plot(ages_n, Ps_n, 'D-', color=COL_P_NAIVE, linewidth=1.8, markersize=5,
                label='Med-naive')
        ax.fill_between(ages_n, P_lo_n, P_hi_n, color=COL_P_NAIVE, alpha=0.15)

        # Linear fit
        fit_n = fits_naive['P']
        if np.isfinite(fit_n['slope']):
            age_range = np.linspace(min(ages_n), max(ages_n), 100)
            y_fit = fit_n['intercept'] + fit_n['slope'] * age_range
            ax.plot(age_range, y_fit, '-', color=COL_P_NAIVE, alpha=0.5,
                    linewidth=1.0)
            p_str = f"p = {fit_n['p']:.3f}" if fit_n['p'] >= 0.001 else f"p < 0.001"
            ax.text(0.97, 0.03,
                    f"slope = {fit_n['slope']:.4f}/yr\n{p_str}",
                    transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='grey', alpha=0.8))

        ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.8, alpha=0.5)

        # Interpretation labels
        ax.text(0.03, 0.97, r'$P > 1$: coupling outpaces' + '\n' + 'capacity loss',
                transform=ax.transAxes, fontsize=7, ha='left', va='top', color=COL_C,
                fontstyle='italic')
        ax.text(0.03, 0.03, r'$P < 1$: capacity loss' + '\n' + 'outpaces coupling',
                transform=ax.transAxes, fontsize=7, ha='left', va='bottom', color=COL_V,
                fontstyle='italic')
    else:
        ax.text(0.5, 0.5, 'Insufficient med-naive data',
                transform=ax.transAxes, ha='center', va='center', fontsize=10)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel(r'Primacy ratio $P = C_{\rm norm} / V_{\rm norm}$')
    ax.set_title('Primacy ratio — medication-naive')
    add_panel_label(ax, '(c)')

    plt.tight_layout()
    save_figure(fig, output_name, OUTPUT_DIR)
    plt.close(fig)


def plot_pairwise_details(results_full, results_naive,
                          output_name='figure_dj_pairwise'):
    """Supplementary figure: per-axis variances and per-pair correlations."""
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axis_colors = ['#e74c3c', '#e67e22', '#27ae60']  # I=red, M=orange, F=green
    pair_colors = ['#8e44ad', '#2980b9', '#16a085']   # I-M=purple, I-F=blue, M-F=teal

    for col_idx, (results, sample_label) in enumerate(
            [(results_full, 'Full sample'), (results_naive, 'Medication-naive')]):
        if not results:
            continue

        ages = [r['age_mid'] for r in results]

        # --- Per-axis variances ---
        ax = axes[0, col_idx]
        for i, (alabel, color) in enumerate(zip(AXIS_LABELS, axis_colors)):
            vals = [r['variances'][i] for r in results]
            ax.plot(ages, vals, 'o-', color=color, label=alabel, markersize=4)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(r'Axis variance $\hat{\Gamma}_{ii}$')
        ax.set_title(f'Per-axis variance — {sample_label}')
        ax.legend(fontsize=8)
        add_panel_label(ax, f'({"a" if col_idx == 0 else "b"})')

        # --- Per-pair absolute correlations ---
        ax = axes[1, col_idx]
        for i, (plabel, color) in enumerate(zip(PAIR_LABELS, pair_colors)):
            vals = [abs(r['pair_correlations'][i]) for r in results]
            ax.plot(ages, vals, 's-', color=color, label=plabel, markersize=4)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Absolute pairwise correlation')
        ax.set_title(f'Pairwise correlations — {sample_label}')
        ax.legend(fontsize=8)
        ax.set_ylim(0, None)
        add_panel_label(ax, f'({"c" if col_idx == 0 else "d"})')

    plt.tight_layout()
    save_figure(fig, output_name, OUTPUT_DIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results summary writer
# ---------------------------------------------------------------------------
def write_results_summary(results_full, results_naive, fits_full, fits_naive,
                          quad_full, quad_naive):
    """Write the plain-text results summary."""
    lines = []
    lines.append("=" * 72)
    lines.append("D vs. J PRIMACY DECOMPOSITION — RESULTS SUMMARY")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Analysis: Decomposition of age-dependent growth in Γ̂ into")
    lines.append("  V_norm (within-axis variance growth → D-degradation)")
    lines.append("  C_norm (cross-axis correlation tightening → J-degradation)")
    lines.append("  P = C_norm / V_norm (primacy ratio)")
    lines.append("")
    lines.append("Reference stratum: age 50–55 (youngest)")
    lines.append("")

    for label, results, fits, quad in [
        ('FULL SAMPLE', results_full, fits_full, quad_full),
        ('MEDICATION-NAIVE SUBGROUP', results_naive, fits_naive, quad_naive),
    ]:
        lines.append("-" * 72)
        lines.append(f"  {label}")
        lines.append("-" * 72)

        if not results:
            lines.append("  Insufficient data for this subgroup.")
            lines.append("")
            continue

        # Table header
        lines.append(f"  {'Stratum':<10s} {'N':>6s} {'V_norm':>8s} {'C_norm':>8s} "
                     f"{'P':>8s} {'Var(I)':>8s} {'Var(M)':>8s} {'Var(F)':>8s} "
                     f"{'|r(I,M)|':>8s} {'|r(I,F)|':>8s} {'|r(M,F)|':>8s}")
        lines.append("  " + "-" * 100)

        for r in results:
            lines.append(
                f"  {r['age_group']:<10s} {r['n']:6d} "
                f"{r['V_norm']:8.4f} {r['C_norm']:8.4f} {r['P']:8.4f} "
                f"{r['variances'][0]:8.4f} {r['variances'][1]:8.4f} "
                f"{r['variances'][2]:8.4f} "
                f"{abs(r['pair_correlations'][0]):8.4f} "
                f"{abs(r['pair_correlations'][1]):8.4f} "
                f"{abs(r['pair_correlations'][2]):8.4f}"
            )

        lines.append("")

        # Linear regression results
        for key, y_label in [('V_norm', 'V_norm'), ('C_norm', 'C_norm'), ('P', 'P')]:
            fit = fits[key]
            lines.append(f"  Linear fit: {y_label} ~ age")
            if np.isfinite(fit['slope']):
                lines.append(f"    slope = {fit['slope']:.6f} per year")
                lines.append(f"    p-value = {fit['p']:.6f}")
                lines.append(f"    95% CI = [{fit['ci'][0]:.6f}, {fit['ci'][1]:.6f}]")
                lines.append(f"    R² = {fit['r2']:.4f}")
            else:
                lines.append("    Insufficient data for regression")
            lines.append("")

        # Quadratic test
        if np.isfinite(quad.get('quad_p', np.nan)):
            lines.append(f"  Quadratic test for P ~ age + age²:")
            lines.append(f"    quadratic coefficient = {quad['quad_coeff']:.6f}")
            lines.append(f"    p-value (F-test vs linear) = {quad['quad_p']:.4f}")
            if quad['quad_p'] < 0.05:
                lines.append("    → Significant nonlinearity detected")
            else:
                lines.append("    → No significant nonlinearity")
        lines.append("")

    # Interpretation
    lines.append("=" * 72)
    lines.append("INTERPRETATION")
    lines.append("=" * 72)
    lines.append("")

    # Determine what the data shows
    if results_full:
        P_trend = fits_full['P']['slope']
        P_p = fits_full['P']['p']

        if np.isfinite(P_trend) and np.isfinite(P_p):
            if P_trend > 0 and P_p < 0.05:
                lines.append("The primacy ratio P increases significantly with age (full sample).")
                lines.append("This indicates that cross-axis correlation tightening (J-degradation)")
                lines.append("outpaces within-axis variance growth (D-degradation) as aging proceeds.")
                lines.append("→ SUPPORTS hyperfunction/antagonistic pleiotropy class of theories:")
                lines.append("  aging is driven by strengthening pathological inter-axis coupling,")
                lines.append("  not merely by loss of individual regulatory capacity.")
            elif P_trend < 0 and P_p < 0.05:
                lines.append("The primacy ratio P decreases significantly with age (full sample).")
                lines.append("This indicates that within-axis variance growth (D-degradation)")
                lines.append("outpaces cross-axis correlation tightening (J-degradation).")
                lines.append("→ SUPPORTS damage/stochastic class of theories:")
                lines.append("  aging is driven by loss of individual regulatory capacity,")
                lines.append("  not by strengthening of pathological inter-axis coupling.")
            else:
                lines.append("The primacy ratio P does not show a statistically significant")
                lines.append(f"trend with age (slope = {P_trend:.4f}, p = {P_p:.3f}).")
                lines.append("Both D-degradation and J-degradation contribute proportionally.")
                lines.append("→ Neither damage/stochastic nor hyperfunction theories alone")
                lines.append("  dominate — both mechanisms contribute to aging in the OU framework.")
        else:
            lines.append("Insufficient data for primacy interpretation.")

    if results_naive:
        lines.append("")
        P_trend_n = fits_naive['P']['slope']
        P_p_n = fits_naive['P']['p']

        lines.append("Medication-naive subgroup (more biologically interpretable):")
        if np.isfinite(P_trend_n) and np.isfinite(P_p_n):
            lines.append(f"  P slope = {P_trend_n:.4f}/yr, p = {P_p_n:.4f}")
            if P_trend_n > 0 and P_p_n < 0.05:
                lines.append("  → Confirms J-primacy in unmedicated individuals")
            elif P_trend_n < 0 and P_p_n < 0.05:
                lines.append("  → Confirms D-primacy in unmedicated individuals")
            else:
                lines.append("  → No significant primacy shift in unmedicated individuals")

    lines.append("")
    lines.append("CAVEATS:")
    lines.append("  1. Small sample sizes at extreme ages (80–85, 85–90) reduce power")
    lines.append("  2. Cross-sectional design cannot distinguish cohort from age effects")
    lines.append("  3. The 3-axis model (I, M, F) captures only a subset of the full")
    lines.append("     regulatory network; results may differ with more axes")
    lines.append("  4. Medication use compresses covariance structure; the medication-naive")
    lines.append("     result is the more biologically interpretable one")
    lines.append("  5. Selection bias: healthier individuals survive to older strata")
    lines.append("  6. Wave 2 only — single cross-section, not longitudinal trend")
    lines.append("")

    text = '\n'.join(lines)
    out_path = os.path.join(OUTPUT_DIR, 'dj_primacy_results.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nResults summary written to {out_path}")

    return text


def write_results_json(results_full, results_naive, fits_full, fits_naive,
                       quad_full, quad_naive):
    """Write machine-readable JSON results."""

    def serialise_results(results):
        out = []
        for r in results:
            out.append({
                'age_group': r['age_group'],
                'age_mid': r['age_mid'],
                'n': r['n'],
                'V': round(r['V'], 6),
                'C': round(r['C'], 6),
                'V_norm': round(r['V_norm'], 6),
                'C_norm': round(r['C_norm'], 6),
                'P': round(r['P'], 6),
                'variances': {
                    'I': round(r['variances'][0], 6),
                    'M': round(r['variances'][1], 6),
                    'F': round(r['variances'][2], 6),
                },
                'pair_correlations': {
                    'I_M': round(r['pair_correlations'][0], 6),
                    'I_F': round(r['pair_correlations'][1], 6),
                    'M_F': round(r['pair_correlations'][2], 6),
                },
            })
        return out

    def serialise_fit(fit):
        return {k: round(v, 8) if isinstance(v, float) else
                [round(x, 8) for x in v] if isinstance(v, tuple) else v
                for k, v in fit.items() if k != 'std_err'}

    output = {
        'analysis': 'D vs J Primacy Decomposition',
        'model': '3-axis (I, M, F)',
        'reference_stratum': '50-55',
        'n_bootstrap': N_BOOTSTRAP,
        'full_sample': {
            'strata': serialise_results(results_full),
            'linear_fits': {k: serialise_fit(v) for k, v in fits_full.items()},
            'quadratic_test': {k: round(v, 8) if isinstance(v, float) else v
                              for k, v in quad_full.items()},
        },
        'medication_naive': {
            'strata': serialise_results(results_naive) if results_naive else [],
            'linear_fits': {k: serialise_fit(v) for k, v in fits_naive.items()}
            if fits_naive else {},
            'quadratic_test': {k: round(v, 8) if isinstance(v, float) else v
                              for k, v in quad_naive.items()}
            if quad_naive else {},
        },
    }

    out_path = os.path.join(OUTPUT_DIR, 'dj_primacy_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"JSON results written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("D vs. J PRIMACY DECOMPOSITION")
    print("Decomposing age-stratified Γ̂ into variance growth vs. correlation tightening")
    print("=" * 72)

    # --- Load data using existing pipeline ---
    files = load_all_files()
    panel, hba1c_units = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm)
    merged = build_analysis_panel(panel, harm_long, mort, supp)

    print(f"\n  Total panel: {len(merged):,} person-visits, "
          f"{merged['idauniq'].nunique():,} unique people")

    # --- Identify medication-naive subgroup ---
    print("\n--- Identifying medication-naive subgroup ---")
    hibpe_col = 'r2hibpe'
    diabe_col = 'r2diabe'

    if hibpe_col in harm.columns and diabe_col in harm.columns:
        med_naive_ids = harm[
            (harm[hibpe_col] == 0) & (harm[diabe_col] == 0)
        ]['idauniq'].values
        print(f"  Med-naive (no hibpe, no diabe at baseline): "
              f"{len(med_naive_ids):,} people")
    else:
        # Fallback: use hemda/hemdb from nurse data at wave 2
        print("  WARNING: r2hibpe/r2diabe not found, using hemda/hemdb fallback")
        w2_data = merged[merged['wave'] == 2]
        med_naive_ids = w2_data[
            (w2_data.get('hemda', pd.Series(dtype=float)) != 1) &
            (w2_data.get('hemdb', pd.Series(dtype=float)) != 1)
        ]['idauniq'].values
        print(f"  Med-naive (no hemda, no hemdb at wave 2): "
              f"{len(med_naive_ids):,} people")

    merged_naive = merged[merged['idauniq'].isin(med_naive_ids)]
    print(f"  Med-naive panel: {len(merged_naive):,} person-visits, "
          f"{merged_naive['idauniq'].nunique():,} people")

    # --- Run decomposition: try wave 2 first (best coverage), pool all waves ---
    # Use cross-sectional at wave 2 (baseline nurse visit, matches youthful reference)
    # If wave 2 has insufficient data in fine bins, pool across waves
    print("\n" + "=" * 72)
    print("FULL SAMPLE — Wave 2 cross-sectional")
    print("=" * 72)

    results_full_w2 = run_decomposition_by_strata(merged, wave=2,
                                                   label_prefix='Full — ')

    # If fine bins produce too few strata, also try pooling across nurse waves
    if len(results_full_w2) < 5:
        print("\n  Wave 2 alone has few 5-year strata with N >= 30.")
        print("  Pooling across all nurse waves for better coverage...")

        # Pool all waves: use each person-visit as an independent observation
        # (cross-sectional pooling)
        results_full_pooled = []
        complete_mask = merged['complete_3axis']
        pooled_data = merged[complete_mask]
        print(f"\n  Pooled data: {len(pooled_data):,} complete observations")

        for (lo, hi), label in zip(AGE_STRATA_5Y, AGE_STRATA_5Y_LABELS):
            stratum = pooled_data[(pooled_data['age'] >= lo) &
                                  (pooled_data['age'] < hi)]
            X = stratum[AXES].dropna().values

            if len(X) < MIN_STRATUM_N:
                print(f"    {label}: N={len(X)} < {MIN_STRATUM_N}, skipping")
                continue

            decomp = compute_decomposition(X)
            V_boots, C_boots = bootstrap_decomposition(X)

            results_full_pooled.append({
                'age_group': label,
                'age_lo': lo,
                'age_hi': hi,
                'age_mid': (lo + hi) / 2,
                'n': decomp['n'],
                'V': decomp['V'],
                'C': decomp['C'],
                'V_ci': (np.percentile(V_boots, 2.5),
                         np.percentile(V_boots, 97.5)),
                'C_ci': (np.percentile(C_boots, 2.5),
                         np.percentile(C_boots, 97.5)),
                'variances': decomp['variances'].tolist(),
                'pair_correlations': decomp['pair_correlations'],
                'Gamma_hat': decomp['Gamma_hat'],
            })
            print(f"    {label}: N={decomp['n']:,}, V={decomp['V']:.4f}, "
                  f"C={decomp['C']:.4f}")

        # Use pooled if it gives more strata
        if len(results_full_pooled) > len(results_full_w2):
            print("\n  Using pooled-wave results (more strata).")
            results_full = results_full_pooled
        else:
            results_full = results_full_w2
    else:
        results_full = results_full_w2

    # Normalize
    results_full = normalize_results(results_full)

    # --- Medication-naive ---
    print("\n" + "=" * 72)
    print("MEDICATION-NAIVE SUBGROUP — Wave 2 cross-sectional")
    print("=" * 72)

    results_naive_w2 = run_decomposition_by_strata(merged_naive, wave=2,
                                                    label_prefix='Med-naive — ')

    if len(results_naive_w2) < 5:
        print("\n  Pooling across all nurse waves for med-naive...")
        results_naive_pooled = []
        complete_mask = merged_naive['complete_3axis']
        pooled_naive = merged_naive[complete_mask]
        print(f"\n  Pooled med-naive: {len(pooled_naive):,} complete observations")

        for (lo, hi), label in zip(AGE_STRATA_5Y, AGE_STRATA_5Y_LABELS):
            stratum = pooled_naive[(pooled_naive['age'] >= lo) &
                                   (pooled_naive['age'] < hi)]
            X = stratum[AXES].dropna().values

            if len(X) < MIN_STRATUM_N:
                print(f"    {label}: N={len(X)} < {MIN_STRATUM_N}, skipping")
                continue

            decomp = compute_decomposition(X)
            V_boots, C_boots = bootstrap_decomposition(X)

            results_naive_pooled.append({
                'age_group': label,
                'age_lo': lo,
                'age_hi': hi,
                'age_mid': (lo + hi) / 2,
                'n': decomp['n'],
                'V': decomp['V'],
                'C': decomp['C'],
                'V_ci': (np.percentile(V_boots, 2.5),
                         np.percentile(V_boots, 97.5)),
                'C_ci': (np.percentile(C_boots, 2.5),
                         np.percentile(C_boots, 97.5)),
                'variances': decomp['variances'].tolist(),
                'pair_correlations': decomp['pair_correlations'],
                'Gamma_hat': decomp['Gamma_hat'],
            })
            print(f"    {label}: N={decomp['n']:,}, V={decomp['V']:.4f}, "
                  f"C={decomp['C']:.4f}")

        if len(results_naive_pooled) > len(results_naive_w2):
            print("\n  Using pooled-wave results for med-naive.")
            results_naive = results_naive_pooled
        else:
            results_naive = results_naive_w2
    else:
        results_naive = results_naive_w2

    results_naive = normalize_results(results_naive)

    # --- Statistical testing ---
    print("\n" + "=" * 72)
    print("STATISTICAL TESTING")
    print("=" * 72)

    fits_full = {}
    for key in ['V_norm', 'C_norm', 'P']:
        fit = fit_linear_trend(results_full, y_key=key)
        fits_full[key] = fit
        print(f"\n  Full sample — {key} ~ age:")
        print(f"    slope = {fit['slope']:.6f}, p = {fit['p']:.6f}, "
              f"R² = {fit['r2']:.4f}")
        print(f"    95% CI = [{fit['ci'][0]:.6f}, {fit['ci'][1]:.6f}]")

    quad_full = fit_quadratic(results_full, y_key='P')
    if np.isfinite(quad_full.get('quad_p', np.nan)):
        print(f"\n  Quadratic test (full): coeff = {quad_full['quad_coeff']:.6f}, "
              f"p = {quad_full['quad_p']:.4f}")

    fits_naive = {}
    quad_naive = {}
    if results_naive:
        for key in ['V_norm', 'C_norm', 'P']:
            fit = fit_linear_trend(results_naive, y_key=key)
            fits_naive[key] = fit
            print(f"\n  Med-naive — {key} ~ age:")
            print(f"    slope = {fit['slope']:.6f}, p = {fit['p']:.6f}, "
                  f"R² = {fit['r2']:.4f}")
            print(f"    95% CI = [{fit['ci'][0]:.6f}, {fit['ci'][1]:.6f}]")

        quad_naive = fit_quadratic(results_naive, y_key='P')
        if np.isfinite(quad_naive.get('quad_p', np.nan)):
            print(f"\n  Quadratic test (med-naive): "
                  f"coeff = {quad_naive['quad_coeff']:.6f}, "
                  f"p = {quad_naive['quad_p']:.4f}")

    # --- Generate figures ---
    print("\n" + "=" * 72)
    print("GENERATING FIGURES")
    print("=" * 72)

    plot_dj_primacy(results_full, results_naive, fits_full, fits_naive)
    plot_pairwise_details(results_full, results_naive)

    # --- Write results ---
    text = write_results_summary(results_full, results_naive,
                                 fits_full, fits_naive,
                                 quad_full, quad_naive)
    write_results_json(results_full, results_naive,
                       fits_full, fits_naive,
                       quad_full, quad_naive)

    # Print summary to console
    print("\n" + text)
    print("\nDone.")


if __name__ == '__main__':
    main()
