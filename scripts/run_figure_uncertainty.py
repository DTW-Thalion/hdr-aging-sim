"""
J Matrix Uncertainty Propagation (Task 7)
==========================================
Monte Carlo over plausible J matrices:
  - Perturb each entry within its confidence-grade uncertainty band
  - Propagate to α(A) and ρ(Δt=1d)
  - Test: is the sign structure robust to entry-level uncertainty?
  - Test: which entries most affect α?

Supports arbitrary axis subsets via --axes (default: I M N F).
"""

import argparse
import json as json_module
import os
import sys
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hdr_sim.csv_loader import load_J_csv, build_J_basin, TAU_REGISTRY, _default_csv_path
from hdr_sim.j_matrix_spec import JMatrixSpec


def parse_args():
    parser = argparse.ArgumentParser(description='J Matrix Uncertainty Propagation')
    parser.add_argument('--j-matrix', type=str, default=None,
                        help='Path to J matrix CSV. Default: data/J_matrix_compiled_9x9.csv')
    parser.add_argument('--axes', type=str, nargs='+', default=None,
                        help='Axis subset (e.g., I M F). Default: I M N F.')
    return parser.parse_args()


def build_A(tau, J):
    return -np.diag(1.0 / tau) + J


def main():
    args = parse_args()
    np.random.seed(2026)
    os.makedirs('outputs', exist_ok=True)

    csv_path = args.j_matrix or _default_csv_path()
    j_spec = JMatrixSpec.from_csv(csv_path)
    AXES = tuple(args.axes) if args.axes else ('I', 'M', 'N', 'F')
    n_axes = len(AXES)

    # Load J matrices from CSV
    rows = load_J_csv(csv_path)
    J_healthy = build_J_basin(rows, basin='healthy', axes=AXES)
    J_disease = build_J_basin(rows, basin='disease', axes=AXES)

    # Build confidence grade matrix from CSV
    grade_to_cv = {'A': 0.20, 'B': 0.40, 'C': 0.70, '.': 0.0, '': 0.40}
    axis_idx = {a: i for i, a in enumerate(AXES)}
    grades = np.full((n_axes, n_axes), '.', dtype=object)
    for row in rows:
        src = row['axis_from'].strip()
        tgt = row['axis_to'].strip()
        if src in axis_idx and tgt in axis_idx:
            j = axis_idx[src]
            i = axis_idx[tgt]
            grade = row.get('confidence_grade', '').strip()
            grades[i, j] = grade if grade else 'B'

    # Interpolate J to age 55 (midpoint)
    t = np.clip((55 - 30) / 50, 0, 1)
    J_mid = (1 - t) * J_healthy + t * J_disease
    np.fill_diagonal(J_mid, 0.0)

    # Build tau at age 55
    tau_30_list, tau_80_list = [], []
    for ax in AXES:
        v30, v80 = TAU_REGISTRY[ax]
        tau_30_list.append(v30)
        tau_80_list.append(v80)
    tau_30_arr = np.array(tau_30_list)
    tau_80_arr = np.array(tau_80_list)
    tau_55 = (1 - t) * tau_30_arr + t * tau_80_arr

    n_off_diag = n_axes * (n_axes - 1)

    A_mid = build_A(tau_55, J_mid)
    alpha_mid = np.max(np.real(np.linalg.eigvals(A_mid)))
    rho_1d_mid = np.exp(alpha_mid)

    n_pos_mid = np.sum(J_mid[~np.eye(n_axes, dtype=bool)] > 0)
    n_neg_mid = np.sum(J_mid[~np.eye(n_axes, dtype=bool)] < 0)

    print("=" * 70)
    print(f"GROUND TRUTH (age 55, axes={list(AXES)})")
    print("=" * 70)
    print(f"  α(A) = {alpha_mid:.4f}")
    print(f"  ρ(Δt=1d) = {rho_1d_mid:.4f}")
    print(f"  Positive entries: {n_pos_mid}/{n_off_diag}")
    print(f"  Negative entries: {n_neg_mid}/{n_off_diag}")

    # ========================================================================
    # MONTE CARLO: PERTURB J WITHIN CONFIDENCE BANDS
    # ========================================================================
    n_mc = 10000

    alpha_samples = np.zeros(n_mc)
    rho_samples = np.zeros(n_mc)
    n_positive_samples = np.zeros(n_mc)
    n_sign_flips = np.zeros(n_mc)
    J_samples = np.zeros((n_mc, n_axes, n_axes))

    off_diag = ~np.eye(n_axes, dtype=bool)

    for k in range(n_mc):
        J_k = J_mid.copy()
        flips = 0
        for i in range(n_axes):
            for j in range(n_axes):
                if i != j:
                    g = grades[i, j]
                    cv = grade_to_cv.get(g, 0.40)
                    noise = np.exp(cv * np.random.randn())
                    J_k[i, j] = J_mid[i, j] * noise
                    if g == 'C' and np.random.rand() < 0.05:
                        J_k[i, j] = -J_k[i, j]
                    if np.sign(J_k[i, j]) != np.sign(J_mid[i, j]):
                        flips += 1

        A_k = build_A(tau_55, J_k)
        eigs = np.linalg.eigvals(A_k)
        alpha_samples[k] = np.max(np.real(eigs))
        rho_samples[k] = np.exp(alpha_samples[k])
        n_positive_samples[k] = np.sum(J_k[off_diag] > 0)
        n_sign_flips[k] = flips
        J_samples[k] = J_k

    print("\n" + "=" * 70)
    print(f"MONTE CARLO RESULTS (N = {n_mc})")
    print("=" * 70)

    print(f"\n  α(A) distribution:")
    print(f"    Mean:   {np.mean(alpha_samples):.4f}")
    print(f"    Median: {np.median(alpha_samples):.4f}")
    print(f"    95% CI: [{np.percentile(alpha_samples, 2.5):.4f}, "
          f"{np.percentile(alpha_samples, 97.5):.4f}]")
    print(f"    Point:  {alpha_mid:.4f}")

    print(f"\n  ρ(Δt=1d) distribution:")
    print(f"    Mean:   {np.mean(rho_samples):.4f}")
    print(f"    Median: {np.median(rho_samples):.4f}")
    print(f"    95% CI: [{np.percentile(rho_samples, 2.5):.4f}, "
          f"{np.percentile(rho_samples, 97.5):.4f}]")

    print(f"\n  Fraction α < 0 (stable): {np.mean(alpha_samples < 0):.1%}")
    print(f"  Fraction α ≥ 0 (unstable): {np.mean(alpha_samples >= 0):.1%}")

    print(f"\n  Positive entries (out of {n_off_diag} off-diagonal):")
    print(f"    Mean:   {np.mean(n_positive_samples):.1f}")
    print(f"    Median: {np.median(n_positive_samples):.0f}")
    print(f"    Range:  [{np.min(n_positive_samples):.0f}, {np.max(n_positive_samples):.0f}]")

    # ========================================================================
    # SENSITIVITY: WHICH ENTRIES MOST AFFECT α?
    # ========================================================================
    print("\n" + "=" * 70)
    print("SENSITIVITY: ENTRY-LEVEL IMPACT ON α(A)")
    print("=" * 70)

    sensitivities = np.zeros((n_axes, n_axes))
    for i in range(n_axes):
        for j in range(n_axes):
            if i != j:
                j_ij_samples = J_samples[:, i, j]
                corr = np.corrcoef(j_ij_samples, alpha_samples)[0, 1]
                sensitivities[i, j] = corr

    print(f"\n  Correlation of J_ij perturbation with α(A):")
    print(f"  {'':6s}", end='')
    for j in range(n_axes):
        print(f"  {AXES[j]:>6s}", end='')
    print()
    for i in range(n_axes):
        print(f"  {AXES[i]:6s}", end='')
        for j in range(n_axes):
            if i == j:
                print(f"  {'---':>6s}", end='')
            else:
                print(f"  {sensitivities[i,j]:+6.3f}", end='')
        print()

    # Top entries
    flat_sens = []
    for i in range(n_axes):
        for j in range(n_axes):
            if i != j:
                flat_sens.append((abs(sensitivities[i, j]), i, j, sensitivities[i, j]))
    flat_sens.sort(reverse=True)
    top_n = min(3, len(flat_sens))
    print(f"\n  Top {top_n} most influential entries:")
    for rank, (absval, i, j, val) in enumerate(flat_sens[:top_n]):
        print(f"    {rank+1}. J[{AXES[i]}←{AXES[j]}] = {J_mid[i,j]:+.4f} "
              f"(grade {grades[i,j]}, corr with α: {val:+.3f})")

    # ========================================================================
    # ACROSS AGE STRATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("UNCERTAINTY PROPAGATION ACROSS AGE STRATA")
    print("=" * 70)

    age_mids = [44.5, 54.5, 64.5, 74.5]
    n_mc_age = 5000

    alpha_by_age = {age: [] for age in age_mids}
    alpha_true_by_age = {}

    for age in age_mids:
        t_a = np.clip((age - 30) / 50, 0, 1)
        tau_a = (1 - t_a) * tau_30_arr + t_a * tau_80_arr
        J_a = (1 - t_a) * J_healthy + t_a * J_disease
        np.fill_diagonal(J_a, 0.0)

        A_a = build_A(tau_a, J_a)
        alpha_true = np.max(np.real(np.linalg.eigvals(A_a)))
        alpha_true_by_age[age] = alpha_true

        for k in range(n_mc_age):
            J_k = J_a.copy()
            for i in range(n_axes):
                for j in range(n_axes):
                    if i != j:
                        g = grades[i, j]
                        cv = grade_to_cv.get(g, 0.40)
                        J_k[i, j] = J_a[i, j] * np.exp(cv * np.random.randn())
            A_k = build_A(tau_a, J_k)
            ak = np.max(np.real(np.linalg.eigvals(A_k)))
            alpha_by_age[age].append(ak)

        alphas_age = alpha_by_age[age]
        print(f"  Age {age:.0f}: α_true={alpha_true:.4f}, "
              f"α_MC=[{np.percentile(alphas_age, 2.5):.4f}, {np.percentile(alphas_age, 97.5):.4f}], "
              f"P(unstable)={np.mean(np.array(alphas_age)>=0):.1%}")

    n_order_preserved = 0
    for k in range(n_mc_age):
        ordered = all(alpha_by_age[age_mids[i]][k] < alpha_by_age[age_mids[i+1]][k]
                      for i in range(len(age_mids)-1))
        if ordered:
            n_order_preserved += 1
    print(f"\n  P(monotone α ordering preserved): {n_order_preserved/n_mc_age:.1%}")

    # ========================================================================
    # FIGURE
    # ========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axes_str = ', '.join(AXES)
    fig.suptitle(f'Uncertainty Propagation: J → Stability Metrics ({axes_str})',
                 fontsize=13, fontweight='bold', y=0.98)

    # (a) α distribution at age 55
    ax = axs[0, 0]
    ax.hist(alpha_samples, bins=80, density=True, color='steelblue', alpha=0.7,
            edgecolor='white', linewidth=0.5)
    ax.axvline(x=alpha_mid, color='red', linewidth=2, linestyle='-',
               label=f'Point estimate ({alpha_mid:.4f})')
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='--', label='Instability (α=0)')
    ax.axvline(x=np.percentile(alpha_samples, 2.5), color='orange', linewidth=1,
               linestyle=':', label='95% CI')
    ax.axvline(x=np.percentile(alpha_samples, 97.5), color='orange', linewidth=1,
               linestyle=':')
    ax.set_xlabel(r'Spectral abscissa $\alpha(A)$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(r'(a) Distribution of $\alpha(A)$ under J uncertainty (age 55)', fontsize=11)
    ax.legend(fontsize=8)

    # (b) Positive entry count distribution
    ax = axs[0, 1]
    bin_lo = max(0, int(np.min(n_positive_samples)) - 1) + 0.5
    bin_hi = int(np.max(n_positive_samples)) + 1.5
    counts, bins, patches = ax.hist(n_positive_samples,
                                     bins=np.arange(bin_lo, bin_hi, 1),
                                     density=True, color='coral', alpha=0.7,
                                     edgecolor='white', linewidth=0.5)
    ax.axvline(x=n_pos_mid, color='red', linewidth=2, linestyle='-',
               label=f'Point estimate ({n_pos_mid}/{n_off_diag})')
    ax.set_xlabel(f'Number of positive off-diagonal entries (out of {n_off_diag})', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('(b) "Reinforcing" entry count under uncertainty', fontsize=11)
    ax.legend(fontsize=9)

    # (c) α uncertainty band across age strata
    ax = axs[1, 0]
    ages_plot = age_mids
    alpha_point = [alpha_true_by_age[a] for a in ages_plot]
    alpha_lo = [np.percentile(alpha_by_age[a], 2.5) for a in ages_plot]
    alpha_hi = [np.percentile(alpha_by_age[a], 97.5) for a in ages_plot]
    alpha_med = [np.median(alpha_by_age[a]) for a in ages_plot]

    ax.fill_between(ages_plot, alpha_lo, alpha_hi, alpha=0.2, color='steelblue',
                    label='95% credible interval')
    ax.plot(ages_plot, alpha_point, 'ko-', linewidth=2, markersize=8,
            label=r'Point estimate $\alpha$', zorder=5)
    ax.plot(ages_plot, alpha_med, 's--', color='steelblue', linewidth=1.5, markersize=6,
            label='MC median')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Age stratum midpoint', fontsize=11)
    ax.set_ylabel(r'$\alpha(A)$', fontsize=11)
    ax.set_title(r'(c) $\alpha$ with J-uncertainty band across ages', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')

    # (d) Sensitivity heatmap
    ax = axs[1, 1]
    mask = np.eye(n_axes, dtype=bool)
    sens_display = sensitivities.copy()
    sens_display[mask] = 0
    im = ax.imshow(sens_display, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='equal')
    for i in range(n_axes):
        for j in range(n_axes):
            if i != j:
                ax.text(j, i, f'{sensitivities[i,j]:+.2f}', ha='center', va='center',
                        fontsize=max(7, 11 - n_axes), fontweight='bold',
                        color='white' if abs(sensitivities[i,j]) > 0.3 else 'black')
            else:
                ax.text(j, i, '\u2014', ha='center', va='center',
                        fontsize=max(7, 11 - n_axes), color='gray')
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels([f'\u2190{a}' for a in AXES], fontsize=max(7, 10 - n_axes + 4))
    ax.set_yticks(range(n_axes))
    ax.set_yticklabels(list(AXES), fontsize=max(7, 10 - n_axes + 4))
    ax.set_xlabel('Source axis (column of J)', fontsize=11)
    ax.set_ylabel('Target axis (row of J)', fontsize=11)
    ax.set_title(r'(d) Sensitivity: corr($J_{ij}$, $\alpha$)', fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('outputs/figure_uncertainty.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('outputs/figure_uncertainty.png', dpi=150, bbox_inches='tight')
    print("\n  Saved figure_uncertainty.pdf/png")

    print("\nDone.")

    # Save provenance sidecar
    meta = {
        'j_matrix': j_spec.to_dict(),
        'axes': list(AXES),
        'script': 'run_figure_uncertainty.py',
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    with open('outputs/figure_uncertainty_meta.json', 'w') as f:
        json_module.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
