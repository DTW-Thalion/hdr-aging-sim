"""
J Matrix Uncertainty Propagation (Task 7)
==========================================
Monte Carlo over plausible J matrices:
  - Perturb each entry within its confidence-grade uncertainty band
  - Propagate to α(A) and ρ(Δt=1d)
  - Test: is "46/49 reinforcing" robust to entry-level uncertainty?
  - Test: which entries most affect α?
"""

import os
import sys
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import argparse
import json as json_module
from datetime import datetime, timezone

np.random.seed(2026)
os.makedirs('outputs', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='J Matrix Uncertainty Propagation')
    parser.add_argument('--j-matrix', type=str, default=None,
                        help='Path to J matrix CSV for provenance. Default: data/J_matrix_compiled_9x9.csv')
    parser.add_argument('--axes', type=str, nargs='+', default=None,
                        help='Axis subset (e.g., I M F). Default: script-specific.')
    return parser.parse_args()

_args = parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hdr_sim.j_matrix_spec import JMatrixSpec, load_default_spec

n_axes = 4
AXES = ['I', 'M', 'N', 'F']

# ============================================================================
# 1. GROUND TRUTH J AT AGE 55 (midpoint) WITH CONFIDENCE GRADES
# ============================================================================

# J matrix (from → to): J[i,j] = effect of axis j on axis i
# Sign convention: + = worsening j worsens i; - = activity in j improves i
J_mid = np.array([
    [ 0.0,    0.0475, 0.0150, -0.0278],  # I row
    [ 0.0600, 0.0,    0.0150, -0.0400],  # M row
    [ 0.0300, 0.0240, 0.0,    -0.0200],  # N row
    [ 0.0350, 0.0350, 0.0175,  0.0   ],  # F row (others → F)
])

# Confidence grades: A = high confidence, B = moderate, C = low
# Maps to relative uncertainty: A=20%, B=40%, C=70%
grades = np.array([
    ['.',  'A', 'B', 'A'],  # I row
    ['A',  '.', 'B', 'A'],  # M row
    ['B',  'B', '.', 'B'],  # N row
    ['A',  'A', 'B', '.'],  # F row
])

grade_to_cv = {'A': 0.20, 'B': 0.40, 'C': 0.70, '.': 0.0}

# Magnitude tiers: S=strong, M=moderate, W=weak
mag_tiers = np.array([
    ['.', 'S', 'M', 'S'],
    ['S', '.', 'M', 'S'],
    ['M', 'M', '.', 'M'],
    ['M', 'M', 'W', '.'],
])

tau_55 = np.array([14.0, 0.20, 0.025, 25.0])  # interpolated age 55

def build_A(tau, J):
    return -np.diag(1.0 / tau) + J

A_mid = build_A(tau_55, J_mid)
alpha_mid = np.max(np.real(np.linalg.eigvals(A_mid)))
rho_1d_mid = np.exp(alpha_mid)

print("=" * 70)
print("GROUND TRUTH (age 55)")
print("=" * 70)
print(f"  α(A) = {alpha_mid:.4f}")
print(f"  ρ(Δt=1d) = {rho_1d_mid:.4f}")
print(f"  Positive entries: {np.sum(J_mid[np.eye(4)==0] > 0)}/12")
print(f"  Negative entries: {np.sum(J_mid[np.eye(4)==0] < 0)}/12")


# ============================================================================
# 2. MONTE CARLO: PERTURB J WITHIN CONFIDENCE BANDS
# ============================================================================

n_mc = 10000

alpha_samples = np.zeros(n_mc)
rho_samples = np.zeros(n_mc)
n_positive_samples = np.zeros(n_mc)
n_sign_flips = np.zeros(n_mc)
J_samples = np.zeros((n_mc, n_axes, n_axes))

for k in range(n_mc):
    J_k = J_mid.copy()
    flips = 0
    for i in range(n_axes):
        for j in range(n_axes):
            if i != j:
                cv = grade_to_cv[grades[i, j]]
                # Perturb: multiplicative log-normal noise
                # This preserves sign in most draws but allows sign flips
                # for uncertain entries
                noise = np.exp(cv * np.random.randn())
                J_k[i, j] = J_mid[i, j] * noise
                # For grade C entries, also allow sign flips with small prob
                if grades[i, j] == 'C' and np.random.rand() < 0.05:
                    J_k[i, j] = -J_k[i, j]
                if np.sign(J_k[i, j]) != np.sign(J_mid[i, j]):
                    flips += 1

    A_k = build_A(tau_55, J_k)
    eigs = np.linalg.eigvals(A_k)
    alpha_samples[k] = np.max(np.real(eigs))
    rho_samples[k] = np.exp(alpha_samples[k])
    n_positive_samples[k] = np.sum(J_k[np.eye(4) == 0] > 0)
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

print(f"\n  Positive entries (out of 12 off-diagonal):")
print(f"    Mean:   {np.mean(n_positive_samples):.1f}")
print(f"    Median: {np.median(n_positive_samples):.0f}")
print(f"    Range:  [{np.min(n_positive_samples):.0f}, {np.max(n_positive_samples):.0f}]")
print(f"    P(≥9 positive): {np.mean(n_positive_samples >= 9):.1%}")
print(f"    P(all 9 same-sign positive): — checking below")

# Check: are the 9 same-sign entries (non-F) always positive?
always_positive_9 = 0
for k in range(n_mc):
    # Same-sign pairs: I↔M, I↔N, M↔N (6 entries) + F→all (3 entries, always +)
    # Actually, in 4-axis: non-F entries are (0,1),(0,2),(1,0),(1,2),(2,0),(2,1) = 6 entries
    # F-row: (3,0),(3,1),(3,2) = 3 entries, always positive (others→F)
    # F-col: (0,3),(1,3),(2,3) = 3 entries, negative (F→others, protective)
    non_F_entries = [J_samples[k, i, j] for i in range(3) for j in range(3) if i != j]
    if all(e > 0 for e in non_F_entries):
        always_positive_9 += 1

print(f"    P(all 6 non-F cross-couplings positive): {always_positive_9/n_mc:.1%}")


# ============================================================================
# 3. SENSITIVITY: WHICH ENTRIES MOST AFFECT α?
# ============================================================================

print("\n" + "=" * 70)
print("SENSITIVITY: ENTRY-LEVEL IMPACT ON α(A)")
print("=" * 70)

sensitivities = np.zeros((n_axes, n_axes))
for i in range(n_axes):
    for j in range(n_axes):
        if i != j:
            # Correlation between J_ij samples and α samples
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

# Top 3 most influential entries
flat_sens = []
for i in range(n_axes):
    for j in range(n_axes):
        if i != j:
            flat_sens.append((abs(sensitivities[i, j]), i, j, sensitivities[i, j]))
flat_sens.sort(reverse=True)
print(f"\n  Top 3 most influential entries:")
for rank, (absval, i, j, val) in enumerate(flat_sens[:3]):
    print(f"    {rank+1}. J[{AXES[i]}←{AXES[j]}] = {J_mid[i,j]:+.4f} "
          f"(grade {grades[i,j]}, corr with α: {val:+.3f})")


# ============================================================================
# 4. ACROSS AGE STRATA
# ============================================================================

print("\n" + "=" * 70)
print("UNCERTAINTY PROPAGATION ACROSS AGE STRATA")
print("=" * 70)

def get_params_age(age):
    t = np.clip((age - 30) / 50, 0, 1)
    tau_30 = np.array([7.0, 0.1, 0.01, 8.0])
    tau_80 = np.array([21.0, 0.4, 0.04, 42.0])
    tau = tau_30 * (1 - t) + tau_80 * t
    J_30 = np.array([
        [ 0.0,   0.015, 0.005, -0.04 ],
        [ 0.02,  0.0,   0.005, -0.06 ],
        [ 0.01,  0.008, 0.0,   -0.03 ],
        [ 0.01,  0.01,  0.005,  0.0  ],
    ])
    J_80 = np.array([
        [ 0.0,   0.08,  0.025, -0.015],
        [ 0.10,  0.0,   0.025, -0.02 ],
        [ 0.05,  0.04,  0.0,   -0.01 ],
        [ 0.06,  0.06,  0.03,   0.0  ],
    ])
    J = J_30 * (1 - t) + J_80 * t
    return tau, J

age_mids = [44.5, 54.5, 64.5, 74.5]
n_mc_age = 5000

alpha_by_age = {age: [] for age in age_mids}
rho_by_age = {age: [] for age in age_mids}
alpha_true_by_age = {}

for age in age_mids:
    tau_a, J_a = get_params_age(age)
    A_a = build_A(tau_a, J_a)
    alpha_true = np.max(np.real(np.linalg.eigvals(A_a)))
    alpha_true_by_age[age] = alpha_true

    for k in range(n_mc_age):
        J_k = J_a.copy()
        for i in range(n_axes):
            for j in range(n_axes):
                if i != j:
                    cv = grade_to_cv[grades[i, j]]
                    J_k[i, j] = J_a[i, j] * np.exp(cv * np.random.randn())
        A_k = build_A(tau_a, J_k)
        ak = np.max(np.real(np.linalg.eigvals(A_k)))
        alpha_by_age[age].append(ak)
        rho_by_age[age].append(np.exp(ak))

    alphas = alpha_by_age[age]
    print(f"  Age {age:.0f}: α_true={alpha_true:.4f}, "
          f"α_MC=[{np.percentile(alphas, 2.5):.4f}, {np.percentile(alphas, 97.5):.4f}], "
          f"P(unstable)={np.mean(np.array(alphas)>=0):.1%}")

# Check: is the ORDERING robust?
n_order_preserved = 0
for k in range(n_mc_age):
    ordered = all(alpha_by_age[age_mids[i]][k] < alpha_by_age[age_mids[i+1]][k]
                  for i in range(len(age_mids)-1))
    if ordered:
        n_order_preserved += 1

print(f"\n  P(monotone α ordering preserved): {n_order_preserved/n_mc_age:.1%}")


# ============================================================================
# 5. FIGURE
# ============================================================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Uncertainty Propagation: J → Stability Metrics',
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
counts, bins, patches = ax.hist(n_positive_samples, bins=np.arange(5.5, 13.5, 1),
                                 density=True, color='coral', alpha=0.7,
                                 edgecolor='white', linewidth=0.5)
ax.axvline(x=9, color='red', linewidth=2, linestyle='-',
           label='Point estimate (9/12)')
ax.set_xlabel('Number of positive off-diagonal entries (out of 12)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(b) "Reinforcing" entry count under uncertainty', fontsize=11)
ax.legend(fontsize=9)

# (c) α uncertainty band across age strata
ax = axs[1, 0]
ages = age_mids
alpha_point = [alpha_true_by_age[a] for a in ages]
alpha_lo = [np.percentile(alpha_by_age[a], 2.5) for a in ages]
alpha_hi = [np.percentile(alpha_by_age[a], 97.5) for a in ages]
alpha_med = [np.median(alpha_by_age[a]) for a in ages]

ax.fill_between(ages, alpha_lo, alpha_hi, alpha=0.2, color='steelblue',
                label='95% credible interval')
ax.plot(ages, alpha_point, 'ko-', linewidth=2, markersize=8,
        label=r'Point estimate $\alpha$', zorder=5)
ax.plot(ages, alpha_med, 's--', color='steelblue', linewidth=1.5, markersize=6,
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
                    fontsize=11, fontweight='bold',
                    color='white' if abs(sensitivities[i,j]) > 0.3 else 'black')
        else:
            ax.text(j, i, '—', ha='center', va='center', fontsize=11, color='gray')
ax.set_xticks(range(n_axes))
ax.set_xticklabels([f'←{a}' for a in AXES], fontsize=10)
ax.set_yticks(range(n_axes))
ax.set_yticklabels(AXES, fontsize=10)
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
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_csv_path = _args.j_matrix or os.path.join(_root, 'data', 'J_matrix_compiled_9x9.csv')
_j_spec = JMatrixSpec.from_csv(_csv_path)
_meta = {
    'j_matrix': _j_spec.to_dict(),
    'script': 'run_figure_uncertainty.py',
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
}
with open('outputs/figure_uncertainty_meta.json', 'w') as f:
    json_module.dump(_meta, f, indent=2)
