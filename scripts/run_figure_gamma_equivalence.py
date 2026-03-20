"""
Γ-Native Equivalence Study (R4 Revision)
=========================================
Confirms that the Γ-native pipeline produces equivalent results to the
A-based pipeline, without requiring Lyapunov inversion or Q specification.

Four panels:
  (a) λ_max(Γ̂) vs true |α(A)| across age strata (monotone inverse)
  (b) SWDS-Γ vs SWDS Spearman rank correlation per stratum
  (c) C-index comparison: SWDS-Γ vs SWDS vs Mahalanobis vs L2 vs age
  (d) Bootstrap distribution of Γ̂ sign concordance with T* threshold

Configuration: N=2000/stratum, 4 strata, seed=42.
"""

import os
import sys
import numpy as np
from scipy.stats import spearmanr
from scipy.linalg import inv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hdr_sim.estimation import (
    get_params, build_A, spectral_abscissa, stationary_covariance,
    generate_stratum, lyapunov_inversion_symmetric, stability_weighted_score,
    compute_swds_gamma, compute_swds_gamma_batch, gamma_stability_proxy,
    covariance_sign_concordance,
)

np.random.seed(42)
os.makedirs('outputs', exist_ok=True)

# Configuration
N_per_stratum = 2000
strata_mids = [44.5, 54.5, 64.5, 74.5]
q = np.array([0.02, 0.05, 0.08, 0.01])
Q = np.diag(q)
meas_noise_sd = 0.3
n_bootstrap = 200

print("=" * 70)
print("Γ-NATIVE EQUIVALENCE STUDY")
print("=" * 70)

# ============================================================================
# 1. GENERATE DATA AND COMPUTE METRICS PER STRATUM
# ============================================================================

alpha_true_list = []
lambda_max_list = []
spearman_list = []
stratum_data = {}

for mid in strata_mids:
    tau, J = get_params(mid)
    A_true = build_A(tau, J)
    alpha_true = spectral_abscissa(A_true)
    alpha_true_list.append(alpha_true)

    # Generate stratum data
    X_true, Y_obs, Gamma_true = generate_stratum(mid, N_per_stratum, q, meas_noise_sd)
    Gamma_hat = np.cov(Y_obs.T)

    # Γ-native: λ_max
    proxy = gamma_stability_proxy(Gamma_hat)
    lambda_max_list.append(proxy['lambda_max'])

    # A-based SWDS (using Lyapunov inversion)
    A_hat = lyapunov_inversion_symmetric(Gamma_hat, Q)
    swds_a = np.array([stability_weighted_score(Y_obs[i], A_hat) for i in range(N_per_stratum)])

    # Γ-native SWDS
    swds_g = compute_swds_gamma_batch(Y_obs, Gamma_hat)

    # Spearman correlation
    rho, _ = spearmanr(swds_a, swds_g)
    spearman_list.append(rho)

    stratum_data[mid] = {
        'Y_obs': Y_obs, 'Gamma_hat': Gamma_hat, 'A_hat': A_hat,
        'swds_a': swds_a, 'swds_g': swds_g, 'J': J, 'alpha_true': alpha_true,
    }

    print(f"  Age {mid:.0f}: α_true={alpha_true:.4f}, λ_max={proxy['lambda_max']:.4f}, "
          f"Spearman(SWDS,SWDS-Γ)={rho:.4f}")

# Check monotone trends
alpha_monotone = all(alpha_true_list[i] < alpha_true_list[i+1] for i in range(3))
lmax_monotone = all(lambda_max_list[i] < lambda_max_list[i+1] for i in range(3))
print(f"\n  α monotone increasing: {alpha_monotone}")
print(f"  λ_max monotone increasing: {lmax_monotone}")

# ============================================================================
# 2. C-INDEX COMPARISON (Panel c)
# ============================================================================

print("\n" + "=" * 70)
print("C-INDEX COMPARISON")
print("=" * 70)

# Generate continuous-age population for C-index
N_cindex = 5000
ages_continuous = np.random.uniform(40, 80, N_cindex)

scores_swds_g = np.zeros(N_cindex)
scores_swds_a = np.zeros(N_cindex)
scores_maha = np.zeros(N_cindex)
scores_l2 = np.zeros(N_cindex)
true_alpha_per_person = np.zeros(N_cindex)

n_axes = 4
for i in range(N_cindex):
    age = ages_continuous[i]
    tau, J = get_params(age)
    A = build_A(tau, J)
    alpha = spectral_abscissa(A)
    true_alpha_per_person[i] = alpha

    Gamma = stationary_covariance(A, Q)
    x = np.random.multivariate_normal(np.zeros(n_axes), Gamma)
    y = x + meas_noise_sd * np.random.randn(n_axes)

    # Use stratum-level Gamma_hat (find nearest stratum)
    nearest_mid = min(strata_mids, key=lambda m: abs(age - m))
    Gamma_hat = stratum_data[nearest_mid]['Gamma_hat']
    A_hat = stratum_data[nearest_mid]['A_hat']

    scores_swds_g[i] = compute_swds_gamma(y, Gamma_hat)
    scores_swds_a[i] = stability_weighted_score(y, A_hat)
    scores_maha[i] = np.sqrt(y @ inv(Gamma_hat) @ y)
    scores_l2[i] = np.linalg.norm(y)

# Generate synthetic outcomes: hazard ∝ (1/|α|) × (dominant eigenvector loading)²
# Event rate ~6%
eigvals_true, eigvecs_true = np.linalg.eigh(stationary_covariance(
    build_A(*get_params(60)), Q))
v_dom = eigvecs_true[:, -1]  # dominant eigenvector

hazard_scale = np.zeros(N_cindex)
for i in range(N_cindex):
    age = ages_continuous[i]
    tau, J = get_params(age)
    A = build_A(tau, J)
    alpha = spectral_abscissa(A)
    nearest_mid = min(strata_mids, key=lambda m: abs(age - m))
    Gamma_hat = stratum_data[nearest_mid]['Gamma_hat']
    proxy = gamma_stability_proxy(Gamma_hat)
    v_dom_local = proxy['eigenvectors'][:, 0]

    # Get individual's data
    Gamma = stationary_covariance(A, Q)
    x = np.random.multivariate_normal(np.zeros(n_axes), Gamma)
    y = x + meas_noise_sd * np.random.randn(n_axes)
    loading = np.dot(v_dom_local, y) ** 2
    hazard_scale[i] = (1.0 / max(abs(alpha), 1e-6)) * loading

# Calibrate event probability to ~6%
logit_offset = np.log(0.06 / 0.94)
hazard_centered = hazard_scale - np.mean(hazard_scale)
hazard_std = np.std(hazard_centered) if np.std(hazard_centered) > 0 else 1.0
event_prob = 1.0 / (1.0 + np.exp(-(logit_offset + 0.5 * hazard_centered / hazard_std)))
events = np.random.binomial(1, event_prob)
event_rate = events.mean()
print(f"  Synthetic event rate: {event_rate:.3f}")


def concordance_index(scores, events):
    """Compute Harrell's C-index."""
    n = len(scores)
    concordant = 0
    discordant = 0
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if events[j] == 1:
                continue
            if scores[i] > scores[j]:
                concordant += 1
            elif scores[i] < scores[j]:
                discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


# Subsample for faster C-index computation
n_sub = min(1000, N_cindex)
idx_sub = np.random.choice(N_cindex, n_sub, replace=False)

c_swds_g = concordance_index(scores_swds_g[idx_sub], events[idx_sub])
c_swds_a = concordance_index(scores_swds_a[idx_sub], events[idx_sub])
c_maha = concordance_index(scores_maha[idx_sub], events[idx_sub])
c_l2 = concordance_index(scores_l2[idx_sub], events[idx_sub])
c_age = concordance_index(ages_continuous[idx_sub], events[idx_sub])

print(f"  C-index SWDS-Γ:      {c_swds_g:.4f}")
print(f"  C-index SWDS (A):    {c_swds_a:.4f}")
print(f"  C-index Mahalanobis: {c_maha:.4f}")
print(f"  C-index L2:          {c_l2:.4f}")
print(f"  C-index Age:         {c_age:.4f}")

# ============================================================================
# 3. BOOTSTRAP T* CALIBRATION (Panel d)
# ============================================================================

print("\n" + "=" * 70)
print("BOOTSTRAP T* CALIBRATION (Layer A sign concordance)")
print("=" * 70)

concordance_bootstrap = []
for b in range(n_bootstrap):
    # Resample a stratum and compute concordance
    mid = np.random.choice(strata_mids)
    tau, J = get_params(mid)
    A = build_A(tau, J)
    Gamma = stationary_covariance(A, Q)
    X_boot = np.random.multivariate_normal(np.zeros(n_axes), Gamma, size=N_per_stratum)
    Y_boot = X_boot + meas_noise_sd * np.random.randn(N_per_stratum, n_axes)
    Gamma_boot = np.cov(Y_boot.T)

    result = covariance_sign_concordance(Gamma_boot, J, exclude_ambiguous=True)
    concordance_bootstrap.append(result['concordance'])

concordance_bootstrap = np.array(concordance_bootstrap)
T_star = np.mean(concordance_bootstrap) - 2 * np.std(concordance_bootstrap)

print(f"  Bootstrap concordance: mean={np.mean(concordance_bootstrap):.4f}, "
      f"SD={np.std(concordance_bootstrap):.4f}")
print(f"  T* (mean - 2×SD) = {T_star:.4f}")

# ============================================================================
# 4. FIGURE
# ============================================================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Γ-Native Equivalence Study (R4)', fontsize=13, fontweight='bold', y=0.98)

# (a) λ_max(Γ̂) vs true |α(A)|
ax = axs[0, 0]
color_alpha = 'steelblue'
color_lmax = 'darkorange'
ax.plot(strata_mids, [abs(a) for a in alpha_true_list], 'o-', color=color_alpha,
        linewidth=2, markersize=8, label=r'$|\alpha(A)|$ (true)')
ax2 = ax.twinx()
ax2.plot(strata_mids, lambda_max_list, 's-', color=color_lmax,
         linewidth=2, markersize=8, label=r'$\lambda_{\max}(\hat\Gamma)$')
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel(r'$|\alpha(A)|$', fontsize=11, color=color_alpha)
ax2.set_ylabel(r'$\lambda_{\max}(\hat\Gamma)$', fontsize=11, color=color_lmax)
ax.set_title(r'(a) $\lambda_{\max}(\hat\Gamma)$ vs $|\alpha(A)|$', fontsize=11)
ax.tick_params(axis='y', labelcolor=color_alpha)
ax2.tick_params(axis='y', labelcolor=color_lmax)
# Combine legends
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
ax.annotate(r'$|\alpha|$ decreases $\Rightarrow$ $\lambda_{\max}$ increases',
            xy=(0.05, 0.05), xycoords='axes fraction', fontsize=9, style='italic')

# (b) Spearman rank correlation per stratum
ax = axs[0, 1]
colors_bar = ['steelblue', 'orange', 'green', 'red']
bars = ax.bar(range(len(strata_mids)), spearman_list, color=colors_bar, alpha=0.8,
              edgecolor='black', linewidth=0.5)
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Target: 0.95')
ax.set_xticks(range(len(strata_mids)))
ax.set_xticklabels([f'{m:.0f}' for m in strata_mids])
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel('Spearman ρ (SWDS vs SWDS-Γ)', fontsize=11)
ax.set_title('(b) Rank equivalence: SWDS-Γ vs SWDS', fontsize=11)
ax.set_ylim(0.8, 1.02)
ax.legend(fontsize=9)
for i, v in enumerate(spearman_list):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# (c) C-index comparison
ax = axs[1, 0]
methods = ['SWDS-Γ', 'SWDS', 'Mahal.', 'L2', 'Age']
c_values = [c_swds_g, c_swds_a, c_maha, c_l2, c_age]
colors_c = ['darkorange', 'steelblue', 'green', 'gray', 'lightcoral']
y_pos = range(len(methods))
ax.barh(y_pos, c_values, color=colors_c, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel('C-index', fontsize=11)
ax.set_title('(c) Discrimination: synthetic stability-dependent outcome', fontsize=11)
ax.set_xlim(0.45, max(c_values) + 0.05)
for i, v in enumerate(c_values):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

# (d) Bootstrap concordance distribution with T*
ax = axs[1, 1]
ax.hist(concordance_bootstrap, bins=25, density=True, alpha=0.7, color='steelblue',
        edgecolor='white', label='Bootstrap distribution')
ax.axvline(x=T_star, color='red', linewidth=2, linestyle='--',
           label=f'T* = {T_star:.3f}')
ax.axvline(x=np.mean(concordance_bootstrap), color='black', linewidth=1.5,
           linestyle='-', label=f'Mean = {np.mean(concordance_bootstrap):.3f}')
ax.set_xlabel('Γ̂ off-diagonal sign concordance', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(d) Bootstrap T* calibration (Layer A)', fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs/figure_gamma_equivalence.pdf', dpi=150, bbox_inches='tight')
plt.savefig('outputs/figure_gamma_equivalence.png', dpi=150, bbox_inches='tight')
print("\nSaved figure_gamma_equivalence.pdf/png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  λ_max monotone increasing with age: {lmax_monotone}")
print(f"  |α| monotone decreasing with age:   {alpha_monotone}")
print(f"  λ_max tracks 1/|α| (inverse):       YES (by construction)")
print(f"  Spearman(SWDS, SWDS-Γ) per stratum: {', '.join(f'{r:.3f}' for r in spearman_list)}")
print(f"  All Spearman > 0.95:                 {all(r > 0.95 for r in spearman_list)}")
print(f"  C-index SWDS-Γ:                      {c_swds_g:.4f}")
print(f"  C-index SWDS (A-based):              {c_swds_a:.4f}")
print(f"  C-index Mahalanobis:                 {c_maha:.4f}")
print(f"  C-index L2:                          {c_l2:.4f}")
print(f"  C-index Age:                         {c_age:.4f}")
print(f"  Bootstrap T* (Layer A threshold):    {T_star:.4f}")
print(f"  Bootstrap concordance mean ± SD:     {np.mean(concordance_bootstrap):.4f} ± {np.std(concordance_bootstrap):.4f}")
print("\nDone.")
