#!/usr/bin/env python3
"""
Visit-Pair Change-Covariance Estimator Bias Study
===================================================

Validates the approximation Gamma_change ≈ 2*Gamma used in the R6
change-covariance estimator.  The estimator tracks lambda_max(Gamma_change)
with age as evidence for coupling tightening.  The key assumption is that
between consecutive ELSA visits (~4 years apart) the OU process fully
equilibrates, so the change covariance Cov(Delta x) = 2*Gamma.

This script characterises estimator bias as a function of visit interval,
age, confounds, and tau-scaling.

Outputs:
  outputs/figure_estimator_bias.pdf / .png — 4-panel supplementary figure
  outputs/estimator_bias_results.json     — machine-readable results

Usage:
    python scripts/run_estimator_bias.py

Reference: HDR Ontology Manuscript R6, SI — Change-Covariance Estimator
"""

import json
import os
import sys
import time
import warnings

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy.linalg import solve_continuous_lyapunov, expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from hdr_sim.aging_params import configure, tau_of_age, J_of_age
from hdr_sim.dynamics import build_A, spectral_abscissa
from hdr_sim.estimation import stationary_covariance
from hdr_sim.plotting import setup_style, add_panel_label, save_figure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SAMPLES = 5000
N_MC = 200
Q_SIGMA2 = 0.01
AGES = [30, 40, 50, 60, 70, 80]
VISIT_INTERVALS_DAYS = [7, 14, 30, 90, 180, 365, 730, 1460, 2920]
BASE_SEED = 2024
AXES_3 = ('I', 'M', 'F')
AXES_7 = ('I', 'M', 'mito', 'P', 'C', 'N', 'F')

# Panel (a) age colours — match DJ validation palette
AGE_COLORS = {
    30: '#27ae60',  # green
    40: '#2ecc71',
    50: '#3498db',  # blue
    60: '#8e44ad',  # purple
    70: '#e67e22',  # orange
    80: '#e74c3c',  # red
}

# Panel (b)/(c) interval colours
INTERVAL_COLORS = {
    30:   '#e74c3c',   # 1 month — red
    180:  '#e67e22',   # 6 months — orange
    365:  '#8e44ad',   # 1 year — purple
    730:  '#3498db',   # 2 years — blue
    1460: '#27ae60',   # 4 years — green
}
INTERVAL_LABELS = {
    30:   '1 month',
    180:  '6 months',
    365:  '1 year',
    730:  '2 years',
    1460: '4 years',
}

# Panel (d) tau-scaling
TAU_SCALE_FACTORS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 10.0, 20.0]

# Confound parameters
SURVIVORSHIP_CHI2_PCTILE = 90   # percentile threshold for 3 df
MED_FRACTION = 0.40             # top 40% by ||x||
MED_COMPRESSION = 0.7           # variance compression factor


# ---------------------------------------------------------------------------
# Core simulation helpers
# ---------------------------------------------------------------------------

def build_system(age, axes, tau_scale=1.0, recalibrate=False):
    """Build A matrix and true stationary covariance at a given age.

    Parameters
    ----------
    age : float
    axes : tuple[str]
    tau_scale : float
        Multiplicative scaling factor for all tau values.
    recalibrate : bool
        If True, recalibrate c so alpha(30) matches target after tau scaling.

    Returns
    -------
    A : ndarray (n, n)
    Gamma_true : ndarray (n, n)
    """
    if tau_scale != 1.0 and recalibrate:
        # Re-configure with scaled tau
        from hdr_sim.csv_loader import _tau_for_axes, get_J_anchors
        tau_30_base, tau_80_base = _tau_for_axes(axes)
        configure(axes=axes, tau_30=tau_30_base * tau_scale,
                  tau_80=tau_80_base * tau_scale)

    tau = tau_of_age(age)
    J = J_of_age(age)

    if tau_scale != 1.0 and not recalibrate:
        # Just scale tau without recalibration (used for intermediate queries)
        tau = tau * tau_scale

    A = build_A(tau, J)
    n = len(axes)
    Q = Q_SIGMA2 * np.eye(n)

    alpha = spectral_abscissa(A)
    if alpha >= 0:
        Gamma_true = np.full((n, n), np.nan)
    else:
        Gamma_true = stationary_covariance(A, Q)
    return A, Gamma_true


def simulate_visit_pair(A, Gamma_true, dt, n_samples, rng):
    """Generate synthetic visit-pair data and compute change covariance.

    Returns
    -------
    Gamma_change : ndarray (n, n)
        Sample covariance of Delta x = x(t2) - x(t1).
    lam_max_change : float
        Largest eigenvalue of Gamma_change.
    """
    n = A.shape[0]

    # Check stability — if unstable, return NaN
    alpha = spectral_abscissa(A)
    if alpha >= 0:
        nan_arr = np.full((n, n), np.nan)
        nan_x = np.full((n_samples, n), np.nan)
        return nan_arr, np.nan, nan_x, nan_x, nan_x

    Phi = expm(A * dt)

    # Check for overflow in Phi
    if not np.all(np.isfinite(Phi)):
        nan_arr = np.full((n, n), np.nan)
        nan_x = np.full((n_samples, n), np.nan)
        return nan_arr, np.nan, nan_x, nan_x, nan_x

    # Innovation covariance: Gamma - Phi @ Gamma @ Phi.T
    PhiGPhiT = Phi @ Gamma_true @ Phi.T
    if not np.all(np.isfinite(PhiGPhiT)):
        nan_arr = np.full((n, n), np.nan)
        nan_x = np.full((n_samples, n), np.nan)
        return nan_arr, np.nan, nan_x, nan_x, nan_x

    innov_cov = Gamma_true - PhiGPhiT

    # Ensure positive semi-definiteness (numerical tolerance)
    innov_cov = (innov_cov + innov_cov.T) / 2
    eigvals_innov = np.linalg.eigvalsh(innov_cov)
    innov_cov += max(0, -np.min(eigvals_innov) + 1e-14) * np.eye(n)

    # Cholesky for sampling
    L_gamma = np.linalg.cholesky(Gamma_true)
    L_innov = np.linalg.cholesky(innov_cov)

    # Draw x1 from stationary, propagate to x2
    z1 = rng.standard_normal((n_samples, n))
    x1 = z1 @ L_gamma.T

    z2 = rng.standard_normal((n_samples, n))
    x2 = (x1 @ Phi.T) + (z2 @ L_innov.T)

    dx = x2 - x1
    Gamma_change = np.cov(dx, rowvar=False)
    lam_max = np.max(np.linalg.eigvalsh(Gamma_change))
    return Gamma_change, lam_max, x1, x2, dx


def apply_confounds(x1, x2, dx, n_axes):
    """Apply survivorship bias and medication compression.

    Returns
    -------
    dx_confounded : ndarray
    """
    n = x1.shape[0]

    # --- Survivorship: drop high-norm individuals ---
    norms_sq = np.sum(x1**2, axis=1)
    from scipy.stats import chi2
    threshold = chi2.ppf(SURVIVORSHIP_CHI2_PCTILE / 100.0, df=n_axes) * Q_SIGMA2
    keep_mask = norms_sq <= threshold

    x1_s = x1[keep_mask]
    x2_s = x2[keep_mask]

    # --- Medication compression on observed values ---
    norms = np.sqrt(np.sum(x1_s**2, axis=1))
    med_threshold = np.percentile(norms, 100 * (1 - MED_FRACTION))
    medicated = norms >= med_threshold

    # Compress observed biomarker values by sqrt(compression)
    compress = np.sqrt(MED_COMPRESSION)
    x1_obs = x1_s.copy()
    x2_obs = x2_s.copy()
    x1_obs[medicated] *= compress
    x2_obs[medicated] *= compress

    dx_conf = x2_obs - x1_obs
    return dx_conf


# ---------------------------------------------------------------------------
# Panel (a): Bias ratio vs visit interval, by age
# ---------------------------------------------------------------------------

def run_panel_a(axes):
    """Compute bias ratio lambda_max(Gamma_change) / (2 * lambda_max(Gamma_true))
    as a function of visit interval, for selected ages."""
    print("Panel (a): Bias ratio vs visit interval...")
    configure(axes=axes)

    ages_panel_a = [30, 50, 65, 80]
    results = {}

    for age in ages_panel_a:
        A, Gamma_true = build_system(age, axes)
        lam_true = np.max(np.linalg.eigvalsh(Gamma_true))
        ratios_by_dt = {}

        for dt in VISIT_INTERVALS_DAYS:
            mc_ratios = []
            for mc in range(N_MC):
                rng = np.random.default_rng(BASE_SEED + mc * 1000 + age)
                _, lam_change, _, _, _ = simulate_visit_pair(
                    A, Gamma_true, dt, N_SAMPLES, rng)
                mc_ratios.append(lam_change / (2 * lam_true))
            ratios_by_dt[dt] = {
                'mean': float(np.mean(mc_ratios)),
                'ci_lo': float(np.percentile(mc_ratios, 2.5)),
                'ci_hi': float(np.percentile(mc_ratios, 97.5)),
            }
        results[age] = ratios_by_dt
        print(f"  Age {age}: done")

    return results


# ---------------------------------------------------------------------------
# Panel (b): Recovered lambda_max age trend at selected intervals
# ---------------------------------------------------------------------------

def run_panel_b(axes):
    """Compute recovered lambda_max/2 age trend at selected intervals."""
    print("Panel (b): Age trend recovery (clean)...")
    configure(axes=axes)

    intervals = [30, 180, 365, 730, 1460]
    results = {'true': {}, 'intervals': {}}

    # True lambda_max(Gamma) at each age
    for age in AGES:
        A, Gamma_true = build_system(age, axes)
        results['true'][age] = float(np.max(np.linalg.eigvalsh(Gamma_true)))

    for dt in intervals:
        age_trends = {age: [] for age in AGES}
        for mc in range(N_MC):
            for age in AGES:
                rng = np.random.default_rng(BASE_SEED + mc * 1000 + age + dt)
                A, Gamma_true = build_system(age, axes)
                _, lam_change, _, _, _ = simulate_visit_pair(
                    A, Gamma_true, dt, N_SAMPLES, rng)
                age_trends[age].append(lam_change / 2)

        results['intervals'][dt] = {}
        for age in AGES:
            vals = age_trends[age]
            results['intervals'][dt][age] = {
                'mean': float(np.mean(vals)),
                'ci_lo': float(np.percentile(vals, 2.5)),
                'ci_hi': float(np.percentile(vals, 97.5)),
            }
        print(f"  dt={dt} days: done")

    return results


# ---------------------------------------------------------------------------
# Panel (c): Same with confounds
# ---------------------------------------------------------------------------

def run_panel_c(axes):
    """Recovered lambda_max age trend with survivorship + medication."""
    print("Panel (c): Age trend recovery (confounded)...")
    configure(axes=axes)
    n_axes = len(axes)

    intervals = [30, 180, 365, 730, 1460]
    results = {'true': {}, 'intervals': {}}

    for age in AGES:
        A, Gamma_true = build_system(age, axes)
        results['true'][age] = float(np.max(np.linalg.eigvalsh(Gamma_true)))

    for dt in intervals:
        age_trends = {age: [] for age in AGES}
        for mc in range(N_MC):
            for age in AGES:
                rng = np.random.default_rng(BASE_SEED + mc * 1000 + age + dt + 99)
                A, Gamma_true = build_system(age, axes)
                _, _, x1, x2, dx = simulate_visit_pair(
                    A, Gamma_true, dt, N_SAMPLES, rng)

                dx_conf = apply_confounds(x1, x2, dx, n_axes)
                if len(dx_conf) < 10:
                    age_trends[age].append(np.nan)
                    continue
                Gamma_change_conf = np.cov(dx_conf, rowvar=False)
                lam = np.max(np.linalg.eigvalsh(Gamma_change_conf))
                age_trends[age].append(lam / 2)

        results['intervals'][dt] = {}
        for age in AGES:
            vals = [v for v in age_trends[age] if not np.isnan(v)]
            if vals:
                results['intervals'][dt][age] = {
                    'mean': float(np.mean(vals)),
                    'ci_lo': float(np.percentile(vals, 2.5)),
                    'ci_hi': float(np.percentile(vals, 97.5)),
                }
            else:
                results['intervals'][dt][age] = {
                    'mean': float('nan'),
                    'ci_lo': float('nan'),
                    'ci_hi': float('nan'),
                }
        print(f"  dt={dt} days (confounded): done")

    return results


# ---------------------------------------------------------------------------
# Panel (d): Sensitivity to tau scaling
# ---------------------------------------------------------------------------

def run_panel_d(axes):
    """Bias ratio at age 80, dt=4yr, as a function of tau-scaling factor.

    Scales tau by k without recalibrating c, so the coupling matrix J stays
    fixed and only the equilibration timescale changes.  This directly tests
    whether 4 years is long enough for equilibration at each k.
    """
    print("Panel (d): Tau-scaling sensitivity...")

    # Get baseline calibrated J at the default tau
    configure(axes=axes)
    J_80_base = J_of_age(80)
    from hdr_sim.csv_loader import _tau_for_axes
    _, tau_80_base = _tau_for_axes(axes)

    results = {}
    n = len(axes)
    Q = Q_SIGMA2 * np.eye(n)

    for k in TAU_SCALE_FACTORS:
        tau_80_scaled = tau_80_base * k
        A = build_A(tau_80_scaled, J_80_base)
        dt = 1460  # 4 years

        alpha_80 = spectral_abscissa(A)
        if alpha_80 >= 0:
            results[k] = {
                'mean': float('nan'),
                'ci_lo': float('nan'),
                'ci_hi': float('nan'),
                'recovery_time_days': float('inf'),
                'alpha': float(alpha_80),
                'unstable': True,
            }
            print(f"  tau x{k}: UNSTABLE at age 80 (alpha={alpha_80:.4f})")
            continue

        Gamma_true = stationary_covariance(A, Q)
        lam_true = np.max(np.linalg.eigvalsh(Gamma_true))

        mc_ratios = []
        for mc in range(N_MC):
            rng = np.random.default_rng(BASE_SEED + mc * 1000 + int(k * 100))
            _, lam_change, _, _, _ = simulate_visit_pair(
                A, Gamma_true, dt, N_SAMPLES, rng)
            if np.isfinite(lam_change):
                mc_ratios.append(lam_change / (2 * lam_true))

        rec_time = 1.0 / abs(alpha_80)
        if mc_ratios:
            results[k] = {
                'mean': float(np.mean(mc_ratios)),
                'ci_lo': float(np.percentile(mc_ratios, 2.5)),
                'ci_hi': float(np.percentile(mc_ratios, 97.5)),
                'recovery_time_days': float(rec_time),
                'alpha': float(alpha_80),
            }
            print(f"  tau x{k}: bias={results[k]['mean']:.4f}, "
                  f"recovery={rec_time:.1f}d, alpha={alpha_80:.4f}")
        else:
            results[k] = {
                'mean': float('nan'),
                'ci_lo': float('nan'),
                'ci_hi': float('nan'),
                'recovery_time_days': float(rec_time),
                'alpha': float(alpha_80),
            }
            print(f"  tau x{k}: all MC runs produced NaN")

    # Restore default configuration
    configure(axes=axes)
    return results


# ---------------------------------------------------------------------------
# 7-axis sensitivity check
# ---------------------------------------------------------------------------

def run_seven_axis_check():
    """Check whether monotone lambda_max trend is preserved in 7-axis model."""
    print("7-axis check...")
    configure(axes=AXES_7)

    dt = 1460
    lam_true_trend = []
    lam_est_trend = []

    for age in AGES:
        A, Gamma_true = build_system(age, AXES_7)
        lam_true = np.max(np.linalg.eigvalsh(Gamma_true))
        lam_true_trend.append(lam_true)

        mc_vals = []
        for mc in range(min(N_MC, 50)):  # fewer MC for 7-axis
            rng = np.random.default_rng(BASE_SEED + mc * 1000 + age + 777)
            _, lam_change, _, _, _ = simulate_visit_pair(
                A, Gamma_true, dt, N_SAMPLES, rng)
            mc_vals.append(lam_change / 2)
        lam_est_trend.append(float(np.mean(mc_vals)))

    # Check monotonicity
    true_monotone = all(lam_true_trend[i] <= lam_true_trend[i+1]
                        for i in range(len(lam_true_trend)-1))
    est_monotone = all(lam_est_trend[i] <= lam_est_trend[i+1]
                       for i in range(len(lam_est_trend)-1))

    # Restore 3-axis config
    configure(axes=AXES_3)

    result = {
        'ages': AGES,
        'lam_true': [float(v) for v in lam_true_trend],
        'lam_est': [float(v) for v in lam_est_trend],
        'true_monotone': true_monotone,
        'est_monotone': est_monotone,
    }
    print(f"  True monotone: {true_monotone}, Estimated monotone: {est_monotone}")
    return result


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def make_figure(panel_a, panel_b, panel_c, panel_d):
    """Create the 4-panel supplementary figure."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Panel (a): Bias ratio vs visit interval ---
    ax = axes[0, 0]
    for age in [30, 50, 65, 80]:
        data = panel_a[age]
        dts = sorted(data.keys())
        means = [data[dt]['mean'] for dt in dts]
        ci_lo = [data[dt]['ci_lo'] for dt in dts]
        ci_hi = [data[dt]['ci_hi'] for dt in dts]
        color = AGE_COLORS.get(age, '#333333')
        ax.plot(dts, means, 'o-', color=color, label=f'Age {age}', markersize=4)
        ax.fill_between(dts, ci_lo, ci_hi, alpha=0.12, color=color)

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, zorder=0)
    ax.axvline(1460, color='grey', linestyle='--', linewidth=0.8, zorder=0,
               label='ELSA cadence (4 yr)')
    ax.set_xscale('log')
    ax.set_xlabel('Visit interval (days)')
    ax.set_ylabel(r'$\lambda_{\max}(\hat{\Gamma}_{\Delta}) \;/\; 2\lambda_{\max}(\Gamma)$')
    ax.set_title('Estimator bias ratio')
    ax.legend(fontsize=8, loc='lower right')
    add_panel_label(ax, '(a)')

    # --- Panel (b): Age trend, clean ---
    ax = axes[0, 1]
    true_ages = sorted(panel_b['true'].keys())
    true_vals = [panel_b['true'][a] for a in true_ages]
    ax.plot(true_ages, true_vals, 'k-', linewidth=2, label='True $\\lambda_{\\max}(\\Gamma)$')

    for dt in [30, 180, 365, 730, 1460]:
        data = panel_b['intervals'][dt]
        ages_sorted = sorted(data.keys())
        means = [data[a]['mean'] for a in ages_sorted]
        ci_lo = [data[a]['ci_lo'] for a in ages_sorted]
        ci_hi = [data[a]['ci_hi'] for a in ages_sorted]
        color = INTERVAL_COLORS[dt]
        label = INTERVAL_LABELS[dt]
        ax.plot(ages_sorted, means, 'o-', color=color, label=label, markersize=4)
        ax.fill_between(ages_sorted, ci_lo, ci_hi, alpha=0.12, color=color)

    ax.set_xlabel('Age')
    ax.set_ylabel(r'$\lambda_{\max}(\hat{\Gamma}_{\Delta})/2$')
    ax.set_title('Recovered age trend (clean)')
    ax.legend(fontsize=7, loc='upper left')
    add_panel_label(ax, '(b)')

    # --- Panel (c): Age trend, confounded ---
    ax = axes[1, 0]
    true_ages = sorted(panel_c['true'].keys())
    true_vals = [panel_c['true'][a] for a in true_ages]
    ax.plot(true_ages, true_vals, 'k-', linewidth=2, label='True $\\lambda_{\\max}(\\Gamma)$')

    for dt in [30, 180, 365, 730, 1460]:
        data = panel_c['intervals'][dt]
        ages_sorted = sorted(data.keys())
        means = [data[a]['mean'] for a in ages_sorted]
        ci_lo = [data[a]['ci_lo'] for a in ages_sorted]
        ci_hi = [data[a]['ci_hi'] for a in ages_sorted]
        color = INTERVAL_COLORS[dt]
        label = INTERVAL_LABELS[dt]
        ax.plot(ages_sorted, means, 'o-', color=color, label=label, markersize=4)
        ax.fill_between(ages_sorted, ci_lo, ci_hi, alpha=0.12, color=color)

    ax.set_xlabel('Age')
    ax.set_ylabel(r'$\lambda_{\max}(\hat{\Gamma}_{\Delta})/2$')
    ax.set_title('Recovered age trend (survivorship + medication)')
    ax.legend(fontsize=7, loc='upper left')
    add_panel_label(ax, '(c)')

    # --- Panel (d): Tau-scaling sensitivity ---
    ax = axes[1, 1]
    ks = sorted(panel_d.keys())
    means = [panel_d[k]['mean'] for k in ks]
    ci_lo = [panel_d[k]['ci_lo'] for k in ks]
    ci_hi = [panel_d[k]['ci_hi'] for k in ks]

    # Filter out NaN for plotting
    valid = [(k, m, lo, hi) for k, m, lo, hi in zip(ks, means, ci_lo, ci_hi)
             if np.isfinite(m)]
    if valid:
        vk, vm, vlo, vhi = zip(*valid)
        ax.plot(vk, vm, 'o-', color='#2c3e50', markersize=6, linewidth=1.5)
        ax.fill_between(vk, vlo, vhi, alpha=0.15, color='#2c3e50')
    # Mark unstable points
    unstable_ks = [k for k, m in zip(ks, means) if not np.isfinite(m)]
    for uk in unstable_ks:
        ax.axvline(uk, color='#e74c3c', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, zorder=0)
    ax.axhline(0.95, color='#e74c3c', linestyle=':', linewidth=0.8, zorder=0,
               label='5% bias threshold')
    ax.set_xlabel(r'$\tau$-scaling factor $k$')
    ax.set_ylabel('Bias ratio at age 80, $\\Delta t = 4$ yr')
    ax.set_title(r'Sensitivity to $\tau$ scaling')
    ax.legend(fontsize=8)
    add_panel_label(ax, '(d)')

    fig.tight_layout()
    save_figure(fig, 'figure_estimator_bias')
    plt.close(fig)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def build_json(panel_a, panel_b, panel_c, panel_d, seven_axis):
    """Assemble the full JSON output."""
    # Compute summary statistics
    bias_4yr_80 = panel_a[80][1460]['mean'] if 80 in panel_a else np.nan

    # Smallest dt where bias < 1%
    min_dt_1pct = None
    if 80 in panel_a:
        for dt in sorted(panel_a[80].keys()):
            if abs(panel_a[80][dt]['mean'] - 1.0) < 0.01:
                min_dt_1pct = dt
                break

    # Tau scaling threshold: smallest k where system becomes unstable
    # or bias exceeds 5%
    tau_threshold = None
    for k in sorted(panel_d.keys()):
        info = panel_d[k]
        if info.get('unstable', False):
            tau_threshold = k
            break
        if np.isfinite(info['mean']) and abs(info['mean'] - 1.0) > 0.05:
            tau_threshold = k
            break

    # Convert keys to strings for JSON
    def stringify_keys(d):
        if isinstance(d, dict):
            return {str(k): stringify_keys(v) for k, v in d.items()}
        return d

    output = {
        'config': {
            'axes': list(AXES_3),
            'N': N_SAMPLES,
            'n_mc': N_MC,
            'Q_sigma2': Q_SIGMA2,
            'visit_intervals_days': VISIT_INTERVALS_DAYS,
            'ages': AGES,
            'tau_scale_factors': TAU_SCALE_FACTORS,
        },
        'bias_ratios': stringify_keys(panel_a),
        'trend_recovery': stringify_keys(panel_b),
        'trend_recovery_confounded': stringify_keys(panel_c),
        'tau_sensitivity': stringify_keys(panel_d),
        'seven_axis_check': seven_axis,
        'summary': {
            'bias_at_4yr_age80': float(bias_4yr_80),
            'min_dt_for_1pct_bias': float(min_dt_1pct) if min_dt_1pct else None,
            'tau_scaling_threshold': float(tau_threshold) if tau_threshold else None,
            'seven_axis_monotone': seven_axis['est_monotone'],
        }
    }
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("Visit-Pair Change-Covariance Estimator Bias Study")
    print("=" * 60)

    # Configure for 3-axis model
    configure(axes=AXES_3)

    # Run all panels
    panel_a = run_panel_a(AXES_3)
    panel_b = run_panel_b(AXES_3)
    panel_c = run_panel_c(AXES_3)
    panel_d = run_panel_d(AXES_3)
    seven_axis = run_seven_axis_check()

    # Generate figure
    print("\nGenerating figure...")
    make_figure(panel_a, panel_b, panel_c, panel_d)

    # Build and save JSON
    results = build_json(panel_a, panel_b, panel_c, panel_d, seven_axis)
    json_path = os.path.join(OUTPUT_DIR, 'estimator_bias_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {json_path}")

    # Print summary
    elapsed = time.time() - t0
    s = results['summary']
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Bias at 4-yr / age 80:        {s['bias_at_4yr_age80']:.4f}")
    print(f"  Min dt for <1% bias:          {s['min_dt_for_1pct_bias']} days")
    print(f"  Tau-scaling threshold:         {s['tau_scaling_threshold']}x (instability/5% bias)")
    print(f"  7-axis monotone trend:         {s['seven_axis_monotone']}")
    print(f"  Elapsed time:                  {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
