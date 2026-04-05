#!/usr/bin/env python3
"""
D vs. J Primacy: Bayesian Model Comparison & Misspecification Robustness
=========================================================================

Extends the WP1 validation study (run_dj_validation.py) with two analyses:

  Analysis A — Bayesian model comparison and TOST equivalence test for the
               observed ELSA P-slope under 5 degradation regimes.
  Analysis B — Misspecification robustness: correlated noise (M1), mild
               nonlinearity (M2), latent omitted axis (M3).

Outputs:
  outputs/figure_dj_bayes_robust.pdf / .png — composite figure (4 panels)
  outputs/dj_bayes_robust_results.json      — machine-readable results
  outputs/dj_bayes_robust_summary.md        — SI-ready markdown

Usage:
    python scripts/run_dj_bayes_robust.py

Reference: HDR Ontology Manuscript R6, SI Section S9
"""

import json
import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats
from scipy.linalg import solve_continuous_lyapunov, sqrtm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import from existing validation infrastructure
from run_dj_validation import (
    build_A_dj_ratio,
    compute_VCP,
    compute_P_across_strata,
    apply_survivorship,
    apply_medication_compression,
    get_reference_alpha_trajectory,
    AGE_MIDS,
    AGE_LABELS,
    DJ_RATIOS,
    DJ_RATIO_NAMES,
    DJ_RATIO_VALUES,
    REGIME_COLORS,
    IDX_3_IN_4,
    MED_FRACTION_BY_AGE,
    MED_COMPRESSION,
)

from hdr_sim.aging_params import _TAU_30, _TAU_80, _J_30, _J_80
from hdr_sim.dynamics import build_A, spectral_abscissa
from hdr_sim.plotting import setup_style, add_panel_label, save_figure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_SEED = 42
N_SAMPLES = 5000
N_RUNS = 200
N_RUNS_M2 = 100         # reduced for Euler-Maruyama (slower)
N_SAMPLES_M2 = 2000     # reduced for M2 per stratum

# Observed ELSA P-slope (medication-naive subgroup)
ELSA_P_SLOPE = 0.00139456
ELSA_P_SLOPE_CI = (-0.00289077, 0.00567988)
# Derive SE from CI: half-width / t_crit(df=5, 0.975)
# 7 strata => df = 5 for linear regression
_ELSA_CI_HALF = (ELSA_P_SLOPE_CI[1] - ELSA_P_SLOPE_CI[0]) / 2.0
_ELSA_T_CRIT = stats.t.ppf(0.975, df=5)
ELSA_P_SLOPE_SE = _ELSA_CI_HALF / _ELSA_T_CRIT

# Jeffreys scale for BF interpretation
def jeffreys_label(bf):
    if bf > 100:
        return "decisive"
    elif bf > 30:
        return "very strong"
    elif bf > 10:
        return "strong"
    elif bf > 3:
        return "substantial"
    else:
        return "anecdotal"


# ---------------------------------------------------------------------------
# Analysis A: Bayesian model comparison
# ---------------------------------------------------------------------------

def load_wp1_slopes():
    """Load P-slope distributions from WP1 JSON or re-run if needed."""
    json_path = os.path.join(OUTPUT_DIR, 'dj_validation_results.json')

    # The JSON stores summary stats only (mean, sd, ci), not the full
    # distributions. We need to re-run the 5 regimes under "Both" confounds,
    # 3-axis only, to get the 200-value distributions.
    print("  Re-running 5 regimes (Both confounds, 3-axis) for distributions ...")

    ref_alphas = get_reference_alpha_trajectory()
    slopes = {}

    for regime_name, d_fraction in DJ_RATIOS.items():
        regime_slopes = []
        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx * 1000 + list(DJ_RATIOS.keys()).index(regime_name) * 7 + 20 * 7
            # Match the seed offset from WP1: done = 4 (Both is 4th confound for each regime)
            # Actually, reconstruct the exact WP1 seed:
            # done was computed as (regime_idx * 4 + confound_idx + 1)
            regime_idx = DJ_RATIO_NAMES.index(regime_name)
            done = regime_idx * 4 + 4  # 'Both' is 4th (index 3 + 1)
            seed = BASE_SEED + run_idx * 1000 + done * 7

            rng = np.random.default_rng(seed)
            np.random.seed(seed)

            Q = np.eye(3)
            Vs, Cs = [], []

            for age_mid in AGE_MIDS:
                alpha_target = ref_alphas[age_mid]
                A, alpha_actual = build_A_dj_ratio(age_mid, d_fraction, alpha_target, n_axes=3)

                if alpha_actual >= 0:
                    Vs.append(np.nan); Cs.append(np.nan)
                    continue

                Gamma = solve_continuous_lyapunov(A, -Q)
                eigvals = np.linalg.eigvalsh(Gamma)
                if np.min(eigvals) <= 0:
                    Vs.append(np.nan); Cs.append(np.nan)
                    continue

                X = rng.multivariate_normal(np.zeros(3), Gamma, size=N_SAMPLES)
                X = apply_survivorship(X, age_mid)
                X = apply_medication_compression(X, age_mid, 3)

                V, C = compute_VCP(X)
                Vs.append(V if V is not None else np.nan)
                Cs.append(C if C is not None else np.nan)

            Vs = np.array(Vs)
            Cs = np.array(Cs)
            _, _, Ps = compute_P_across_strata(Vs, Cs)

            valid = np.isfinite(Ps)
            if np.sum(valid) >= 3:
                slope, _, _, _, _ = stats.linregress(AGE_MIDS[valid], Ps[valid])
                regime_slopes.append(slope)
            else:
                regime_slopes.append(np.nan)

        slopes[regime_name] = np.array(regime_slopes)
        valid_count = np.sum(np.isfinite(slopes[regime_name]))
        print(f"    {regime_name}: mean={np.nanmean(slopes[regime_name]):.6f}, "
              f"sd={np.nanstd(slopes[regime_name]):.6f}, N_valid={valid_count}")

    return slopes


def run_bayesian_comparison(slopes):
    """Compute Bayes factors and posterior probabilities."""
    observed = ELSA_P_SLOPE

    # Fit Gaussian to each regime's distribution
    regime_stats = {}
    likelihoods = {}
    for name in DJ_RATIO_NAMES:
        valid = slopes[name][np.isfinite(slopes[name])]
        mu = np.mean(valid)
        sigma = np.std(valid, ddof=1)
        regime_stats[name] = {'mean': mu, 'std': sigma, 'n': len(valid)}
        likelihoods[name] = stats.norm.pdf(observed, loc=mu, scale=sigma)

    # Bayes factors (proportional vs each alternative)
    L_prop = likelihoods['50D/50J']
    bayes_factors = {}
    for name in DJ_RATIO_NAMES:
        if name != '50D/50J':
            bf = L_prop / max(likelihoods[name], 1e-300)
            bayes_factors[f'50D/50J vs {name}'] = bf

    # Posterior model probabilities (uniform prior)
    total_L = sum(likelihoods.values())
    posteriors = {name: L / total_L for name, L in likelihoods.items()}

    return {
        'regime_stats': regime_stats,
        'likelihoods': {k: float(v) for k, v in likelihoods.items()},
        'bayes_factors': {k: float(v) for k, v in bayes_factors.items()},
        'posteriors': {k: float(v) for k, v in posteriors.items()},
        'observed': observed,
    }


def run_tost(slopes):
    """TOST equivalence test for the observed ELSA P-slope."""
    # Proportional regime distribution
    prop_valid = slopes['50D/50J'][np.isfinite(slopes['50D/50J'])]
    prop_mean = np.mean(prop_valid)

    # Find nearest adjacent regime means for equivalence bound
    adj_names = ['75D/25J', '25D/75J']
    adj_means = []
    for name in adj_names:
        valid = slopes[name][np.isfinite(slopes[name])]
        adj_means.append(np.mean(valid))

    # Equivalence bound: half the distance to nearest adjacent regime
    min_dist = min(abs(prop_mean - m) for m in adj_means)
    delta = min_dist / 2.0

    # Test using the ELSA SE
    se = ELSA_P_SLOPE_SE
    obs = ELSA_P_SLOPE

    # TOST: test that obs is within [prop_mean - delta, prop_mean + delta]
    # H1: obs > prop_mean - delta  (lower bound)
    t_lower = (obs - (prop_mean - delta)) / se
    p_lower = 1.0 - stats.t.cdf(t_lower, df=5)  # one-sided, want small p

    # H2: obs < prop_mean + delta  (upper bound)
    t_upper = ((prop_mean + delta) - obs) / se
    p_upper = 1.0 - stats.t.cdf(t_upper, df=5)  # one-sided, want small p

    # Equivalence declared if both p < 0.05
    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < 0.05

    return {
        'prop_mean': float(prop_mean),
        'delta': float(delta),
        'equivalence_bounds': (float(prop_mean - delta), float(prop_mean + delta)),
        'observed': float(obs),
        'se': float(se),
        't_lower': float(t_lower),
        'p_lower': float(p_lower),
        't_upper': float(t_upper),
        'p_upper': float(p_upper),
        'p_tost': float(p_tost),
        'equivalent': equivalent,
    }


# ---------------------------------------------------------------------------
# Analysis B: Misspecification robustness
# ---------------------------------------------------------------------------

def run_misspecification_scenario(scenario_name, ref_alphas):
    """
    Run 5 regimes under a misspecification scenario.
    Returns dict of regime_name -> {'P_slopes': array, 'P_slope_mean': ..., ...}
    """
    n_axes = 3
    results = {}

    n_runs = N_RUNS_M2 if scenario_name == 'M2' else N_RUNS
    n_samples = N_SAMPLES_M2 if scenario_name == 'M2' else N_SAMPLES

    for regime_name, d_fraction in DJ_RATIOS.items():
        all_slopes = []

        for run_idx in range(n_runs):
            seed = BASE_SEED + run_idx * 1000 + DJ_RATIO_NAMES.index(regime_name) * 13 + 999
            rng = np.random.default_rng(seed)
            np.random.seed(seed)

            Vs, Cs = [], []

            for age_mid in AGE_MIDS:
                alpha_target = ref_alphas[age_mid]
                A, alpha_actual = build_A_dj_ratio(age_mid, d_fraction, alpha_target, n_axes=n_axes)

                if alpha_actual >= 0:
                    Vs.append(np.nan); Cs.append(np.nan)
                    continue

                if scenario_name == 'M1':
                    # Correlated noise: Q with off-diagonal rho=0.3
                    Q = np.eye(n_axes)
                    for i in range(n_axes):
                        for j in range(n_axes):
                            if i != j:
                                Q[i, j] = 0.3
                    Gamma = solve_continuous_lyapunov(A, -Q)
                    eigvals = np.linalg.eigvalsh(Gamma)
                    if np.min(eigvals) <= 0:
                        Vs.append(np.nan); Cs.append(np.nan)
                        continue
                    X = rng.multivariate_normal(np.zeros(n_axes), Gamma, size=n_samples)

                elif scenario_name == 'M2':
                    # Mild nonlinearity via vectorized Euler-Maruyama
                    # Extract J from A: J = A + D
                    tau_regime = build_tau_at_regime(age_mid, d_fraction, n_axes)
                    J_for_nl = A + np.diag(1.0 / tau_regime)
                    eps_nl = 0.1
                    dt = 0.05   # coarser step (stable enough with clamping)
                    T_total = 50.0
                    n_steps = int(T_total / dt)
                    sqrt_dt = np.sqrt(dt)
                    clamp = 10.0

                    # Vectorized: X is (n_samples, n_axes)
                    X_em = rng.standard_normal((n_samples, n_axes)) * 0.1
                    for _ in range(n_steps):
                        Ax = X_em @ A.T                          # (N, n)
                        Jx = X_em @ J_for_nl.T                   # (N, n)
                        nl = eps_nl * X_em * Jx                   # (N, n)
                        dW = rng.standard_normal(X_em.shape) * sqrt_dt
                        X_em = X_em + (Ax + nl) * dt + dW
                        np.clip(X_em, -clamp, clamp, out=X_em)

                    X = X_em
                    valid_mask = np.all(np.isfinite(X), axis=1)
                    X = X[valid_mask]
                    if len(X) < 30:
                        Vs.append(np.nan); Cs.append(np.nan)
                        continue

                elif scenario_name == 'M3':
                    # Latent omitted axis: generate from 4-axis, observe 3-axis
                    A_4, _ = build_A_dj_ratio(age_mid, d_fraction, alpha_target, n_axes=4)
                    Q_4 = np.eye(4)
                    # Check stability of 4-axis system
                    if spectral_abscissa(A_4) >= 0:
                        Vs.append(np.nan); Cs.append(np.nan)
                        continue
                    Gamma_4 = solve_continuous_lyapunov(A_4, -Q_4)
                    eigvals = np.linalg.eigvalsh(Gamma_4)
                    if np.min(eigvals) <= 0:
                        Vs.append(np.nan); Cs.append(np.nan)
                        continue
                    X_4 = rng.multivariate_normal(np.zeros(4), Gamma_4, size=n_samples)
                    # Project to I, M, F (drop N=index 2)
                    X = X_4[:, [0, 1, 3]]

                # Apply confounds (Both)
                X = apply_survivorship(X, age_mid)
                X = apply_medication_compression(X, age_mid, n_axes)

                V, C = compute_VCP(X)
                Vs.append(V if V is not None else np.nan)
                Cs.append(C if C is not None else np.nan)

            Vs = np.array(Vs)
            Cs = np.array(Cs)
            _, _, Ps = compute_P_across_strata(Vs, Cs)

            valid = np.isfinite(Ps)
            if np.sum(valid) >= 3:
                slope, _, _, _, _ = stats.linregress(AGE_MIDS[valid], Ps[valid])
                all_slopes.append(slope)
            else:
                all_slopes.append(np.nan)

        all_slopes = np.array(all_slopes)
        results[regime_name] = {
            'P_slopes': all_slopes,
            'P_slope_mean': float(np.nanmean(all_slopes)),
            'P_slope_std': float(np.nanstd(all_slopes)),
            'P_slope_ci': (float(np.nanpercentile(all_slopes, 2.5)),
                           float(np.nanpercentile(all_slopes, 97.5))),
        }

    return results


def build_tau_at_regime(age_mid, d_fraction, n_axes):
    """Get tau values at a specific regime parameterization."""
    tau_30 = _TAU_30.copy()
    tau_80 = _TAU_80.copy()
    if n_axes == 3:
        tau_30 = tau_30[IDX_3_IN_4]
        tau_80 = tau_80[IDX_3_IN_4]
    f = np.clip((age_mid - 30.0) / 50.0, 0.0, 1.0)
    delta_tau = tau_80 - tau_30
    # Use scale=1 for approximate tau (the exact scaling is computed by build_A_dj_ratio)
    return tau_30 + f * d_fraction * delta_tau


def compute_misspec_power(results_misspec, results_baseline=None):
    """Compute discrimination power for misspecification scenarios."""
    regime_pairs = list(zip(DJ_RATIO_NAMES[:-1], DJ_RATIO_NAMES[1:]))
    power_results = {}

    for r1, r2 in regime_pairs:
        s1 = results_misspec[r1]['P_slopes']
        s2 = results_misspec[r2]['P_slopes']
        v1 = s1[np.isfinite(s1)]
        v2 = s2[np.isfinite(s2)]

        if len(v1) >= 10 and len(v2) >= 10:
            t_stat, p_val = stats.ttest_ind(v1, v2, equal_var=False)
            pooled_std = np.sqrt((np.var(v1) + np.var(v2)) / 2)
            cohen_d = abs(np.mean(v1) - np.mean(v2)) / max(pooled_std, 1e-12)

            # Bootstrap power
            rng = np.random.default_rng(42)
            n_sig = 0
            n_boot = 1000
            for _ in range(n_boot):
                idx1 = rng.choice(len(v1), size=min(50, len(v1)), replace=True)
                idx2 = rng.choice(len(v2), size=min(50, len(v2)), replace=True)
                _, p_b = stats.ttest_ind(v1[idx1], v2[idx2], equal_var=False)
                if p_b < 0.05:
                    n_sig += 1
            power = n_sig / n_boot
        else:
            p_val, cohen_d, power = 1.0, 0.0, 0.0

        power_results[f'{r1} vs {r2}'] = {
            'power': power,
            'cohen_d': cohen_d,
            'p_overall': p_val,
        }

    # Check monotone ordering
    means = [results_misspec[name]['P_slope_mean'] for name in DJ_RATIO_NAMES]
    # Expect Pure D < 75D/25J < 50D/50J < 25D/75J < Pure J (or reverse)
    monotone_inc = all(means[i] <= means[i+1] + 1e-6 for i in range(len(means)-1))
    monotone_dec = all(means[i] >= means[i+1] - 1e-6 for i in range(len(means)-1))
    monotone = monotone_inc or monotone_dec

    return power_results, monotone


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure(bayes_results, tost_results, slopes,
                misspec_results, baseline_results):
    """4-panel composite figure."""
    setup_style()
    fig = plt.figure(figsize=(14, 10))

    # Use GridSpec for flexible layout
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # --- Panel (a): Posterior model probabilities ---
    ax_a = fig.add_subplot(gs[0, 0])
    names = DJ_RATIO_NAMES
    posteriors = [bayes_results['posteriors'][n] for n in names]
    colors = [REGIME_COLORS[n] for n in names]
    bars = ax_a.bar(range(len(names)), posteriors, color=colors, edgecolor='white', linewidth=0.8)

    # Annotate BFs
    for i, name in enumerate(names):
        if name != '50D/50J':
            bf_key = f'50D/50J vs {name}'
            bf = bayes_results['bayes_factors'][bf_key]
            label = jeffreys_label(bf)
            ax_a.text(i, posteriors[i] + 0.01, f'BF={bf:.1f}\n({label})',
                     ha='center', va='bottom', fontsize=6)

    ax_a.set_xticks(range(len(names)))
    ax_a.set_xticklabels(names, fontsize=8, rotation=15)
    ax_a.set_ylabel('Posterior probability')
    ax_a.set_title('Bayesian model comparison')
    ax_a.set_ylim(0, max(posteriors) * 1.35)
    add_panel_label(ax_a, '(a)')

    # --- Panel (b): TOST equivalence test ---
    ax_b = fig.add_subplot(gs[0, 1])
    obs = tost_results['observed']
    se = tost_results['se']
    prop_mean = tost_results['prop_mean']
    delta = tost_results['delta']
    eq_lo, eq_hi = tost_results['equivalence_bounds']

    # Shade equivalence region
    ax_b.axvspan(eq_lo, eq_hi, color='#27ae60', alpha=0.15, label='Equivalence region')

    # Regime means
    for name in DJ_RATIO_NAMES:
        rm = bayes_results['regime_stats'][name]['mean']
        ax_b.axvline(rm, color=REGIME_COLORS[name], linewidth=1.0, alpha=0.5, linestyle='--')
        ax_b.text(rm, 0.95, name, transform=ax_b.get_xaxis_transform(),
                 ha='center', va='top', fontsize=6, color=REGIME_COLORS[name], rotation=90)

    # Observed with CI
    ci_lo = obs - _ELSA_T_CRIT * se
    ci_hi = obs + _ELSA_T_CRIT * se
    ax_b.errorbar(obs, 0.5, xerr=[[obs - ci_lo], [ci_hi - obs]],
                 fmt='D', color='black', markersize=8, capsize=5, linewidth=2,
                 label=f'ELSA observed')

    # Equivalence bounds
    ax_b.axvline(eq_lo, color='#27ae60', linewidth=1.5, linestyle=':')
    ax_b.axvline(eq_hi, color='#27ae60', linewidth=1.5, linestyle=':')

    equiv_str = "EQUIVALENT" if tost_results['equivalent'] else "NOT equivalent"
    ax_b.text(0.5, 0.1, f'TOST p = {tost_results["p_tost"]:.4f}\n{equiv_str}',
             transform=ax_b.transAxes, ha='center', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='grey', alpha=0.9))

    ax_b.set_xlabel('P-slope (per year)')
    ax_b.set_yticks([])
    ax_b.set_title('TOST equivalence test')
    ax_b.legend(fontsize=7, loc='upper right')
    add_panel_label(ax_b, '(b)')

    # --- Panel (c): Misspecification robustness (3 sub-panels) ---
    scenario_names = ['M1', 'M2', 'M3']
    scenario_titles = [
        'M1: Correlated noise\n(Q off-diag = 0.3)',
        'M2: Mild nonlinearity\n(10% quadratic)',
        'M3: Latent omitted axis\n(4-axis → 3-axis)',
    ]

    for si, (sname, stitle) in enumerate(zip(scenario_names, scenario_titles)):
        ax = fig.add_subplot(gs[1, si])

        # Baseline (gray)
        if baseline_results is not None:
            base_means = [baseline_results[n]['P_slope_mean'] for n in DJ_RATIO_NAMES]
            base_ci_lo = [baseline_results[n]['P_slope_ci'][0] for n in DJ_RATIO_NAMES]
            base_ci_hi = [baseline_results[n]['P_slope_ci'][1] for n in DJ_RATIO_NAMES]
            base_errs = [np.array(base_means) - np.array(base_ci_lo),
                         np.array(base_ci_hi) - np.array(base_means)]
            ax.errorbar(DJ_RATIO_VALUES, base_means, yerr=base_errs,
                       fmt='s--', color='grey', alpha=0.4, linewidth=1.0,
                       markersize=3, capsize=2, label='Baseline')

        # Misspecified
        mr = misspec_results[sname]
        means = [mr[n]['P_slope_mean'] for n in DJ_RATIO_NAMES]
        ci_lo = [mr[n]['P_slope_ci'][0] for n in DJ_RATIO_NAMES]
        ci_hi = [mr[n]['P_slope_ci'][1] for n in DJ_RATIO_NAMES]
        errs = [np.array(means) - np.array(ci_lo),
                np.array(ci_hi) - np.array(means)]

        for i, name in enumerate(DJ_RATIO_NAMES):
            ax.errorbar(DJ_RATIO_VALUES[i], means[i],
                       yerr=[[means[i] - ci_lo[i]], [ci_hi[i] - means[i]]],
                       fmt='o', color=REGIME_COLORS[name], markersize=5,
                       capsize=3, linewidth=1.5)

        # Connect with line
        ax.plot(DJ_RATIO_VALUES, means, '-', color='#2c3e50', linewidth=1.0, alpha=0.5)

        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_xlabel('D-fraction')
        if si == 0:
            ax.set_ylabel('P-slope (per year)')
        ax.set_title(stitle, fontsize=9)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.invert_xaxis()
        if si == 0:
            ax.legend(fontsize=7)
        add_panel_label(ax, f'({"cde"[si]})')

    fig.tight_layout()
    save_figure(fig, 'figure_dj_bayes_robust', OUTPUT_DIR)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(bayes_results, tost_results, misspec_results,
                  misspec_power, baseline_results, elapsed):
    """Write JSON and markdown outputs."""

    # --- JSON ---
    out = {
        'analysis_A': {
            'bayes': {
                'regime_stats': bayes_results['regime_stats'],
                'likelihoods': bayes_results['likelihoods'],
                'bayes_factors': bayes_results['bayes_factors'],
                'bayes_factor_labels': {k: jeffreys_label(v)
                                        for k, v in bayes_results['bayes_factors'].items()},
                'posteriors': bayes_results['posteriors'],
                'observed': bayes_results['observed'],
            },
            'tost': tost_results,
            'elsa_se': float(ELSA_P_SLOPE_SE),
        },
        'analysis_B': {},
    }

    for sname in ['M1', 'M2', 'M3']:
        mr = misspec_results[sname]
        power, monotone = misspec_power[sname]
        out['analysis_B'][sname] = {
            'regimes': {name: {
                'P_slope_mean': mr[name]['P_slope_mean'],
                'P_slope_std': mr[name]['P_slope_std'],
                'P_slope_ci': list(mr[name]['P_slope_ci']),
            } for name in DJ_RATIO_NAMES},
            'power': {k: {kk: vv for kk, vv in v.items()}
                     for k, v in power.items()},
            'monotone': monotone,
        }

    # Add baseline for comparison
    out['baseline'] = {name: {
        'P_slope_mean': baseline_results[name]['P_slope_mean'],
        'P_slope_std': baseline_results[name]['P_slope_std'],
        'P_slope_ci': list(baseline_results[name]['P_slope_ci']),
    } for name in DJ_RATIO_NAMES}

    out['runtime_seconds'] = elapsed

    json_path = os.path.join(OUTPUT_DIR, 'dj_bayes_robust_results.json')
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved {json_path}")

    # --- Markdown ---
    lines = []
    lines.append("## Supplementary: Bayesian Model Comparison and Misspecification Robustness")
    lines.append("")

    # Section A
    lines.append("### S9.1 Bayesian Model Comparison")
    lines.append("")
    lines.append("We computed Bayes factors comparing the proportional co-degradation "
                 "regime (50D/50J) against each alternative, using simulation-calibrated "
                 "P-slope distributions as empirically derived priors and the ELSA "
                 f"medication-naive P-slope (+{ELSA_P_SLOPE:.4f}/yr) as the datum.")
    lines.append("")

    lines.append("| Comparison | BF | Evidence (Jeffreys) |")
    lines.append("|------------|-----|---------------------|")
    for k, v in bayes_results['bayes_factors'].items():
        lines.append(f"| {k} | {v:.1f} | {jeffreys_label(v)} |")
    lines.append("")

    lines.append("**Posterior model probabilities** (uniform prior across 5 regimes):")
    lines.append("")
    lines.append("| Regime | P(regime \\| data) |")
    lines.append("|--------|-------------------|")
    for name in DJ_RATIO_NAMES:
        p = bayes_results['posteriors'][name]
        lines.append(f"| {name} | {p:.4f} |")
    lines.append("")

    # Interpret
    best_regime = max(bayes_results['posteriors'], key=bayes_results['posteriors'].get)
    best_p = bayes_results['posteriors'][best_regime]
    lines.append(f"The {best_regime} regime receives the highest posterior probability "
                 f"({best_p:.3f}). ", )

    # Specific BF statements
    for k, v in bayes_results['bayes_factors'].items():
        alt = k.split(' vs ')[1]
        lines.append(f"The Bayes factor comparing proportional co-degradation to {alt} "
                     f"is {v:.1f}, providing {jeffreys_label(v)} evidence in favor of "
                     f"proportional co-degradation (Jeffreys scale).")
    lines.append("")

    # TOST
    lines.append("### S9.2 TOST Equivalence Test")
    lines.append("")
    lines.append(f"The equivalence bound was set to delta = {tost_results['delta']:.4f}/yr, "
                 f"half the distance between the proportional regime mean "
                 f"({tost_results['prop_mean']:.4f}/yr) and the nearest adjacent regime. "
                 f"The equivalence region is "
                 f"[{tost_results['equivalence_bounds'][0]:.4f}, "
                 f"{tost_results['equivalence_bounds'][1]:.4f}]/yr.")
    lines.append("")
    lines.append(f"- Lower bound test: t = {tost_results['t_lower']:.3f}, "
                 f"p = {tost_results['p_lower']:.4f}")
    lines.append(f"- Upper bound test: t = {tost_results['t_upper']:.3f}, "
                 f"p = {tost_results['p_upper']:.4f}")
    lines.append(f"- TOST p = {tost_results['p_tost']:.4f}")
    lines.append("")
    if tost_results['equivalent']:
        lines.append("The TOST equivalence test rejects at alpha = 0.05, confirming "
                     "that the observed P-slope lies within the equivalence region "
                     "around the proportional regime. This provides positive statistical "
                     "evidence for proportional co-degradation, beyond the mere absence "
                     "of a significant departure.")
    else:
        lines.append("The TOST equivalence test does not reject at alpha = 0.05. "
                     "The observed P-slope cannot be positively declared equivalent "
                     "to the proportional regime at this sample size, though the "
                     "Bayesian analysis provides complementary evidence.")
    lines.append("")

    # Section B
    lines.append("### S9.3 Misspecification Robustness")
    lines.append("")
    lines.append("Three misspecification scenarios were tested:")
    lines.append("")
    lines.append("- **M1 (Correlated noise)**: Q with off-diagonal rho = 0.3 between all axes")
    lines.append("- **M2 (Mild nonlinearity)**: 10% quadratic correction to the linear drift "
                 "(Euler-Maruyama simulation, reduced to 100 MC runs and 2,000 samples)")
    lines.append("- **M3 (Latent omitted axis)**: Data generated from 4-axis model, "
                 "P computed from 3-axis (I, M, F) projection")
    lines.append("")

    lines.append("| Scenario | Monotone | Min power | Max power |")
    lines.append("|----------|----------|-----------|-----------|")
    for sname in ['M1', 'M2', 'M3']:
        power, monotone = misspec_power[sname]
        powers = [v['power'] for v in power.values()]
        lines.append(f"| {sname} | {'Yes' if monotone else 'No'} | "
                     f"{min(powers):.3f} | {max(powers):.3f} |")
    lines.append("")

    lines.append("**Per-scenario P-slopes:**")
    lines.append("")
    for sname in ['M1', 'M2', 'M3']:
        mr = misspec_results[sname]
        lines.append(f"*{sname}*:")
        lines.append("")
        lines.append("| Regime | P-slope mean | 95% CI |")
        lines.append("|--------|-------------|--------|")
        for name in DJ_RATIO_NAMES:
            r = mr[name]
            lines.append(f"| {name} | {r['P_slope_mean']:.4f} | "
                         f"[{r['P_slope_ci'][0]:.4f}, {r['P_slope_ci'][1]:.4f}] |")
        lines.append("")

    # Misspec interpretation
    all_monotone = all(misspec_power[s][1] for s in ['M1', 'M2', 'M3'])
    all_min_powers = [min(v['power'] for v in misspec_power[s][0].values())
                      for s in ['M1', 'M2', 'M3']]
    robust_scenarios = [s for s in ['M1', 'M2', 'M3'] if misspec_power[s][1]]
    fragile_scenarios = [s for s in ['M1', 'M2', 'M3'] if not misspec_power[s][1]]

    for sname in ['M1', 'M2', 'M3']:
        power, monotone = misspec_power[sname]
        min_p = min(v['power'] for v in power.values())
        lines.append(f"Under misspecification scenario {sname}, the P statistic "
                     f"{'preserves' if monotone else 'does not preserve'} monotone "
                     f"ordering and {'maintains' if min_p >= 0.80 else 'does not maintain'} "
                     f"discrimination power >= 0.80 between adjacent regimes "
                     f"(minimum power = {min_p:.3f}).")
    lines.append("")

    # Bottom line
    lines.append("### Bottom Line")
    lines.append("")
    bayes_strength = jeffreys_label(min(bayes_results['bayes_factors'].values()))
    tost_word = "confirmed" if tost_results['equivalent'] else "not confirmed"
    robust_word = "robust" if all_monotone else ("partially robust" if len(robust_scenarios) >= 2 else "not robust")

    violated = []
    if 'M1' in robust_scenarios:
        pass
    else:
        violated.append("correlated noise")
    if 'M2' in robust_scenarios:
        pass
    else:
        violated.append("nonlinear drift")
    if 'M3' in robust_scenarios:
        pass
    else:
        violated.append("latent axis omission")

    lines.append(f"The proportional co-degradation interpretation is supported by "
                 f"Bayesian model comparison with at least {bayes_strength} evidence "
                 f"(minimum BF = {min(bayes_results['bayes_factors'].values()):.1f}) "
                 f"and TOST equivalence is {tost_word} (p = {tost_results['p_tost']:.4f}). "
                 f"The P statistic is {robust_word} to violations of the OU model assumptions "
                 f"tested here (correlated noise, mild nonlinearity, latent axis omission).")
    if violated:
        lines.append(f" Discrimination is weakened under: {', '.join(violated)}.")

    md_path = os.path.join(OUTPUT_DIR, 'dj_bayes_robust_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 72)
    print("D vs. J: BAYESIAN MODEL COMPARISON & MISSPECIFICATION ROBUSTNESS")
    print("=" * 72)

    # Step 1: Get P-slope distributions for Analysis A
    print("\nStep 1: Loading / re-running WP1 P-slope distributions ...")
    slopes = load_wp1_slopes()

    # Step 2: Bayesian comparison
    print("\nStep 2: Bayesian model comparison ...")
    bayes_results = run_bayesian_comparison(slopes)
    for k, v in bayes_results['bayes_factors'].items():
        print(f"  {k}: BF = {v:.1f} ({jeffreys_label(v)})")

    posteriors = bayes_results['posteriors']
    print(f"  Posteriors: {', '.join(f'{k}={v:.3f}' for k, v in posteriors.items())}")

    # Step 3: TOST
    print("\nStep 3: TOST equivalence test ...")
    tost_results = run_tost(slopes)
    print(f"  delta = {tost_results['delta']:.4f}/yr")
    print(f"  p_lower = {tost_results['p_lower']:.4f}, p_upper = {tost_results['p_upper']:.4f}")
    print(f"  TOST p = {tost_results['p_tost']:.4f}, equivalent = {tost_results['equivalent']}")

    # Step 4: Baseline (Both confounds, 3-axis, correctly specified)
    print("\nStep 4: Baseline (correctly specified) ...")
    # We already have the slopes from Step 1; construct baseline_results
    ref_alphas = get_reference_alpha_trajectory()
    baseline_results = {}
    for name in DJ_RATIO_NAMES:
        valid = slopes[name][np.isfinite(slopes[name])]
        baseline_results[name] = {
            'P_slope_mean': float(np.mean(valid)),
            'P_slope_std': float(np.std(valid)),
            'P_slope_ci': (float(np.percentile(valid, 2.5)),
                           float(np.percentile(valid, 97.5))),
        }

    # Step 5: Misspecification scenarios
    misspec_results = {}
    misspec_power = {}

    print("\nStep 5a: M1 — Correlated noise ...")
    misspec_results['M1'] = run_misspecification_scenario('M1', ref_alphas)
    misspec_power['M1'] = compute_misspec_power(misspec_results['M1'])
    print(f"  Monotone: {misspec_power['M1'][1]}")

    print("\nStep 5b: M2 — Mild nonlinearity (Euler-Maruyama, slower) ...")
    misspec_results['M2'] = run_misspecification_scenario('M2', ref_alphas)
    misspec_power['M2'] = compute_misspec_power(misspec_results['M2'])
    print(f"  Monotone: {misspec_power['M2'][1]}")

    print("\nStep 5c: M3 — Latent omitted axis ...")
    misspec_results['M3'] = run_misspecification_scenario('M3', ref_alphas)
    misspec_power['M3'] = compute_misspec_power(misspec_results['M3'])
    print(f"  Monotone: {misspec_power['M3'][1]}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f} seconds")

    # Step 6: Figures
    print("\nStep 6: Generating figure ...")
    plot_figure(bayes_results, tost_results, slopes,
                misspec_results, baseline_results)

    # Step 7: Write results
    print("\nStep 7: Writing results ...")
    write_results(bayes_results, tost_results, misspec_results,
                  misspec_power, baseline_results, elapsed)

    print("\n" + "=" * 72)
    print("DONE. All outputs saved to outputs/")
    print("=" * 72)


if __name__ == '__main__':
    main()
