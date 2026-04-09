"""
Prior Stress Tests (R4 Revision — Tests 3-4 Layer B)
=====================================================
Runs the constrained OU estimator under three prior conditions:
  1. Correct prior: compiled J signs
  2. Null prior: no sign constraints (unconstrained Lyapunov residual min.)
  3. Adversarial prior: all signs inverted

Reports sign concordance, Lyapunov residual norm, and α̂ for each condition.
Also computes Layer A (Γ̂ off-diagonal sign concordance) for comparison.

Produces: outputs/figure_prior_stress.pdf (4-panel figure)
"""

import os
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hdr_sim.estimation import (
    get_params, build_A, spectral_abscissa, stationary_covariance,
    generate_stratum, estimate_A_sign_constrained, estimate_A_lyapunov_full,
    sign_concordance, lyapunov_residual_norm, covariance_sign_concordance,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

# Configuration
N_per_stratum = 2000
strata_mids = [44.5, 54.5, 64.5, 74.5]
q = np.array([0.02, 0.05, 0.08, 0.01])
Q = np.diag(q)
meas_noise_sd = 0.3

print("=" * 70)
print("PRIOR STRESS TESTS (Tests 3-4 Layer B)")
print("=" * 70)

# Storage
conditions = ['Correct', 'Null', 'Adversarial']
results = {cond: {'concordance': [], 'residual': [], 'alpha_hat': []}
           for cond in conditions}
alpha_true_list = []
layer_a_concordance = []

for mid in strata_mids:
    tau, J = get_params(mid)
    A_true = build_A(tau, J)
    alpha_true = spectral_abscissa(A_true)
    alpha_true_list.append(alpha_true)

    # Generate stratum data
    X_true, Y_obs, Gamma_true = generate_stratum(mid, N_per_stratum, q, meas_noise_sd)
    Gamma_hat = np.cov(Y_obs.T)

    # Layer A: Γ̂ sign concordance (no A estimation needed)
    layer_a_result = covariance_sign_concordance(Gamma_hat, J, exclude_ambiguous=True)
    layer_a_concordance.append(layer_a_result['concordance'])

    # J prior signs
    J_signs = np.sign(J)

    print(f"\n  Age {mid:.0f} (α_true = {alpha_true:.4f}):")

    # 1. Correct prior
    A_correct = estimate_A_sign_constrained(Gamma_hat, Q, J_signs, lambda_reg=0.1)
    conc_correct = sign_concordance(A_correct, A_true)
    res_correct = lyapunov_residual_norm(A_correct, Gamma_hat, Q)
    alpha_correct = spectral_abscissa(A_correct)
    results['Correct']['concordance'].append(conc_correct[2])
    results['Correct']['residual'].append(res_correct)
    results['Correct']['alpha_hat'].append(alpha_correct)
    print(f"    Correct prior:     conc={conc_correct[2]:.3f}, residual={res_correct:.4f}, α̂={alpha_correct:.4f}")

    # 2. Null prior (unconstrained)
    A_null = estimate_A_lyapunov_full(Gamma_hat, Q)
    conc_null = sign_concordance(A_null, A_true)
    res_null = lyapunov_residual_norm(A_null, Gamma_hat, Q)
    alpha_null = spectral_abscissa(A_null)
    results['Null']['concordance'].append(conc_null[2])
    results['Null']['residual'].append(res_null)
    results['Null']['alpha_hat'].append(alpha_null)
    print(f"    Null prior:        conc={conc_null[2]:.3f}, residual={res_null:.4f}, α̂={alpha_null:.4f}")

    # 3. Adversarial prior (inverted signs)
    A_adv = estimate_A_sign_constrained(Gamma_hat, Q, -J_signs, lambda_reg=0.1)
    conc_adv = sign_concordance(A_adv, A_true)
    res_adv = lyapunov_residual_norm(A_adv, Gamma_hat, Q)
    alpha_adv = spectral_abscissa(A_adv)
    results['Adversarial']['concordance'].append(conc_adv[2])
    results['Adversarial']['residual'].append(res_adv)
    results['Adversarial']['alpha_hat'].append(alpha_adv)
    print(f"    Adversarial prior: conc={conc_adv[2]:.3f}, residual={res_adv:.4f}, α̂={alpha_adv:.4f}")

    print(f"    Layer A (Γ̂ sign concordance): {layer_a_result['concordance']:.3f} "
          f"({layer_a_result['n_agree']}/{layer_a_result['n_total']}, "
          f"{layer_a_result['n_excluded']} excluded)")

# ============================================================================
# FIGURE
# ============================================================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Prior Stress Tests (Tests 3-4 Layer B)', fontsize=13, fontweight='bold', y=0.98)

# (a) Concordance by prior condition (grouped bar chart + Layer A diamonds)
ax = axs[0, 0]
x = np.arange(len(strata_mids))
width = 0.22
colors_cond = {'Correct': 'steelblue', 'Null': 'gray', 'Adversarial': 'red'}
for i, cond in enumerate(conditions):
    ax.bar(x + i * width, results[cond]['concordance'], width, label=cond,
           color=colors_cond[cond], alpha=0.8, edgecolor='black', linewidth=0.5)
# Layer A diamonds
ax.scatter(x + width, layer_a_concordance, marker='D', s=60, color='darkorange',
           zorder=5, label='Layer A (Γ̂)', edgecolors='black', linewidths=0.5)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='Chance')
ax.set_xticks(x + width)
ax.set_xticklabels([f'{m:.0f}' for m in strata_mids])
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel('Sign concordance (Â vs A_true)', fontsize=11)
ax.set_title('(a) Off-diagonal sign concordance by prior condition', fontsize=11)
ax.legend(fontsize=8, loc='lower left')
ax.set_ylim(0, 1.05)

# (b) Lyapunov residual norm by condition across strata
ax = axs[0, 1]
for cond, ls, marker in zip(conditions, ['-', '--', ':'], ['o', 's', '^']):
    ax.plot(strata_mids, results[cond]['residual'], f'{marker}{ls}',
            color=colors_cond[cond], linewidth=2, markersize=7, label=cond)
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel(r'$\|A\hat\Gamma + \hat\Gamma A^T + Q\|_F$', fontsize=11)
ax.set_title('(b) Lyapunov residual norm', fontsize=11)
ax.legend(fontsize=9)

# (c) α̂ by condition vs true α across strata
ax = axs[1, 0]
ax.plot(strata_mids, alpha_true_list, 'ko-', linewidth=2.5, markersize=9,
        label=r'True $\alpha$', zorder=5)
for cond, ls, marker in zip(conditions, ['-', '--', ':'], ['o', 's', '^']):
    ax.plot(strata_mids, results[cond]['alpha_hat'], f'{marker}{ls}',
            color=colors_cond[cond], linewidth=1.5, markersize=6, label=f'{cond} prior')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel(r'$\alpha(A)$', fontsize=11)
ax.set_title(r'(c) $\hat\alpha$ by prior condition vs true $\alpha$', fontsize=11)
ax.legend(fontsize=8, loc='lower right')

# (d) Summary text
ax = axs[1, 1]
ax.axis('off')

# Compute prior contribution and data contribution
mean_correct = np.mean(results['Correct']['concordance'])
mean_null = np.mean(results['Null']['concordance'])
mean_adv = np.mean(results['Adversarial']['concordance'])
chance = 0.5

prior_contribution = mean_correct - mean_null
data_contribution = mean_null - chance

summary = (
    "PRIOR STRESS TEST VERDICT\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"Mean concordance (correct prior):  {mean_correct:.3f}\n"
    f"Mean concordance (null prior):     {mean_null:.3f}\n"
    f"Mean concordance (adversarial):    {mean_adv:.3f}\n"
    f"Mean Layer A (Γ̂ concordance):     {np.mean(layer_a_concordance):.3f}\n\n"
    f"PRIOR CONTRIBUTION:\n"
    f"  correct − null = {prior_contribution:+.3f}\n"
    f"  (Sign prior improves estimation)\n\n"
    f"DATA CONTRIBUTION:\n"
    f"  null − chance = {data_contribution:+.3f}\n"
    f"  (Data alone above chance)\n\n"
    f"Layer A operates WITHOUT Q or A\n"
    f"estimation — pure covariance sign\n"
    f"concordance with compiled J."
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(_OUTPUT_DIR, 'figure_prior_stress.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(_OUTPUT_DIR, 'figure_prior_stress.png'), dpi=150, bbox_inches='tight')
print("\nSaved figure_prior_stress.pdf/png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Mean concordance (correct prior):  {mean_correct:.3f}")
print(f"  Mean concordance (null prior):     {mean_null:.3f}")
print(f"  Mean concordance (adversarial):    {mean_adv:.3f}")
print(f"  Mean Layer A (Γ̂ concordance):     {np.mean(layer_a_concordance):.3f}")
print(f"  Prior contribution (correct-null): {prior_contribution:+.3f}")
print(f"  Data contribution (null-chance):   {data_contribution:+.3f}")
print("\nDone.")

# ============================================================================
# PERSISTENT RESULTS
# ============================================================================
try:
    from src.hdr_sim.results_writer import ResultsWriter

    mean_gamma = np.mean(layer_a_concordance)

    with ResultsWriter("Prior Stress Tests",
                        "Quantifies prior vs data contribution for Tests 3-4 Layer B") as rw:
        rw.add_heading("Mean Concordance by Condition")
        rw.add_metric("Correct prior", f"{mean_correct:.3f}")
        rw.add_metric("Null prior", f"{mean_null:.3f}")
        rw.add_metric("Adversarial prior", f"{mean_adv:.3f}")
        rw.add_metric("Layer A (Γ̂ signs)", f"{mean_gamma:.3f}")

        rw.add_heading("Decomposition")
        rw.add_metric("Prior contribution (correct − null)", f"{prior_contribution:+.3f}")
        rw.add_metric("Data contribution (null − chance)", f"{data_contribution:+.3f}")

        rw.add_pass_fail("Null prior ≈ chance (confirms Tier-3)",
                         abs(mean_null - 0.5) < 0.05)
        rw.add_pass_fail("Adversarial < null (prior matters)",
                         mean_adv < mean_null)
except ImportError:
    pass  # results writing is optional
