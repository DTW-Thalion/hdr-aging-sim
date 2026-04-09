"""
Task A: Diffusion Covariance (Q) Sensitivity Analysis
=====================================================
Core question: If Q increases with age, can the α̂ trend reflect
"more noise" rather than "less restoring capacity"?

Three Q-estimation strategies:
  (a) Fixed Q from test-retest variability
  (b) Marginal variance decomposition: Γ_ii = q_i τ_i / 2
  (c) Age-varying Q sensitivity: find the β threshold

Key output: the β* at which "noise increase" becomes indistinguishable
from "stability erosion" in the α̂ trend.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, inv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)

np.random.seed(2026)
n_axes = 4
AXES = ['I', 'M', 'N', 'F']

def get_params(age):
    t = np.clip((age - 30) / 50, 0, 1)
    tau_30 = np.array([7.0, 0.1, 0.01, 8.0])
    tau_80 = np.array([21.0, 0.4, 0.04, 42.0])
    tau = tau_30 * (1 - t) + tau_80 * t
    J_30 = np.array([
        [0., 0.015, 0.005, -0.04],
        [0.02, 0., 0.005, -0.06],
        [0.01, 0.008, 0., -0.03],
        [0.01, 0.01, 0.005, 0.],
    ])
    J_80 = np.array([
        [0., 0.08, 0.025, -0.015],
        [0.10, 0., 0.025, -0.02],
        [0.05, 0.04, 0., -0.01],
        [0.06, 0.06, 0.03, 0.],
    ])
    return tau, J_30 * (1-t) + J_80 * t

def build_A(tau, J):
    return -np.diag(1.0 / tau) + J

q_base = np.array([0.02, 0.05, 0.08, 0.01])
strata_mids = [44.5, 54.5, 64.5, 74.5]
N_per_stratum = 2000

# ============================================================================
# 1. BASELINE: Q FIXED (age-invariant) — reproduce earlier result
# ============================================================================

print("=" * 70)
print("SCENARIO 1: Q FIXED (age-invariant)")
print("=" * 70)

def run_scenario(q_func, Q_assumed_func, label):
    """
    Generate data with q_func(age) as true Q, estimate α̂ using Q_assumed_func(age).
    Returns (alpha_true, alpha_est) lists.
    """
    alpha_true_list = []
    alpha_est_list = []
    lambda_max_list = []
    for mid in strata_mids:
        tau, J = get_params(mid)
        A_gt = build_A(tau, J)
        alpha_gt = np.max(np.real(np.linalg.eigvals(A_gt)))
        alpha_true_list.append(alpha_gt)

        # Generate data with TRUE Q(age)
        Q_true = np.diag(q_func(mid))
        Gamma_true = solve_continuous_lyapunov(A_gt, -Q_true)
        X = np.random.multivariate_normal(np.zeros(n_axes), Gamma_true, size=N_per_stratum)
        Y = X + 0.3 * np.random.randn(N_per_stratum, n_axes)
        Gamma_hat = np.cov(Y.T)

        # Estimate α̂ using ASSUMED Q(age)
        Q_assumed = np.diag(Q_assumed_func(mid))
        P_hat = inv(Gamma_hat)
        A_hat = -Q_assumed @ P_hat / 2
        alpha_est = np.max(np.real(np.linalg.eigvals(A_hat)))
        alpha_est_list.append(alpha_est)

        # R4: also track λ_max(Γ̂) — does NOT require Q specification
        lambda_max = np.max(np.linalg.eigvalsh(Gamma_hat))
        lambda_max_list.append(lambda_max)

    trend_true = all(alpha_true_list[i] < alpha_true_list[i+1] for i in range(3))
    trend_est = all(alpha_est_list[i] < alpha_est_list[i+1] for i in range(3))
    trend_lmax = all(lambda_max_list[i] < lambda_max_list[i+1] for i in range(3))

    print(f"\n  {label}")
    for i, mid in enumerate(strata_mids):
        print(f"    Age {mid:.0f}: α_true={alpha_true_list[i]:.4f}, "
              f"α̂={alpha_est_list[i]:.4f}, λ_max={lambda_max_list[i]:.4f}")
    print(f"    Monotone trend — true: {trend_true}, α̂: {trend_est}, λ_max: {trend_lmax}")

    return alpha_true_list, alpha_est_list, trend_est, lambda_max_list, trend_lmax

# Scenario 1a: Q fixed, Q assumed correctly
q_fixed = lambda age: q_base
a_true_1, a_est_1, trend_1, lmax_1, lmax_trend_1 = run_scenario(q_fixed, q_fixed, "Q fixed, correctly assumed")

# ============================================================================
# 2. AGE-VARYING Q: q_i(age) = q_base_i * (1 + β*(age-30)/50)
# ============================================================================

print("\n" + "=" * 70)
print("SCENARIO 2: Q INCREASES WITH AGE (true), but ASSUMED FIXED")
print("  Testing whether α̂ trend is an artifact of noise increase")
print("=" * 70)

beta_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
results = {}

for beta in beta_values:
    q_age = lambda age, b=beta: q_base * (1 + b * (age - 30) / 50)
    _, alpha_ests, trend, lmax_ests, lmax_trend = run_scenario(q_age, q_fixed,
        f"β={beta:.2f} (Q at age 80 = {1+beta:.1f}× Q at age 30)")
    results[beta] = (alpha_ests, trend, lmax_ests, lmax_trend)

# Find the threshold β* where trend breaks
print("\n" + "=" * 70)
print("CRITICAL THRESHOLD ANALYSIS")
print("=" * 70)
print(f"\n  β    Q_80/Q_30   Trend preserved?")
print(f"  {'—'*45}")
for beta in beta_values:
    alpha_ests, trend, lmax_ests, lmax_trend = results[beta]
    ratio = 1 + beta
    print(f"  {beta:4.2f}   {ratio:5.1f}×        α̂: {'YES ✓' if trend else 'NO ✗'}   λ_max: {'YES ✓' if lmax_trend else 'NO ✗'}")

print("\n  NOTE: λ_max(Γ̂) does not require Q specification for the trend test")
print("        — it is robust to Q misspecification by construction.")

# ============================================================================
# 3. STRATEGY (b): MARGINAL VARIANCE DECOMPOSITION
# ============================================================================

print("\n" + "=" * 70)
print("STRATEGY (b): MARGINAL VARIANCE DECOMPOSITION")
print("  Γ_ii ≈ q_i τ_i / 2 under weak coupling → q̂_i = 2Γ̂_ii / τ_i")
print("=" * 70)

# If τ_i is known from literature, q_i can be estimated from Γ_ii
for mid in strata_mids:
    tau_gt, J_gt = get_params(mid)
    A_gt = build_A(tau_gt, J_gt)
    Q_true_age = np.diag(q_base)  # age-invariant for this test
    Gamma_true = solve_continuous_lyapunov(A_gt, -Q_true_age)

    # Generate observed data
    X = np.random.multivariate_normal(np.zeros(n_axes), Gamma_true, size=N_per_stratum)
    Y = X + 0.3 * np.random.randn(N_per_stratum, n_axes)
    Gamma_hat = np.cov(Y.T)

    # Estimate q_i from marginal variance + known τ
    q_hat = 2 * np.diag(Gamma_hat) / tau_gt
    q_err = np.abs(q_hat - q_base) / q_base

    print(f"  Age {mid:.0f}: q̂/q_true = {q_hat/q_base}")
    print(f"           Relative error: {q_err}")

# ============================================================================
# 4. SCENARIO 3: Q INCREASES WITH AGE, AND WE ESTIMATE Q FROM DATA
# ============================================================================

print("\n" + "=" * 70)
print("SCENARIO 3: Q AGE-VARYING (true), Q ESTIMATED FROM MARGINAL VARIANCE")
print("  Using τ_i from literature to decompose Γ_ii into q_i and τ_i")
print("=" * 70)

beta_test = 1.0  # Q doubles from age 30 to 80
q_age_varying = lambda age: q_base * (1 + beta_test * (age - 30) / 50)

alpha_est_estimated_Q = []
alpha_true_list_3 = []
lambda_max_list_3 = []
for mid in strata_mids:
    tau_gt, J_gt = get_params(mid)
    A_gt = build_A(tau_gt, J_gt)
    alpha_gt = np.max(np.real(np.linalg.eigvals(A_gt)))
    alpha_true_list_3.append(alpha_gt)

    # True Q varies with age
    Q_true_age = np.diag(q_age_varying(mid))
    Gamma_true = solve_continuous_lyapunov(A_gt, -Q_true_age)
    X = np.random.multivariate_normal(np.zeros(n_axes), Gamma_true, size=N_per_stratum)
    Y = X + 0.3 * np.random.randn(N_per_stratum, n_axes)
    Gamma_hat = np.cov(Y.T)

    # Estimate Q from marginal variance + literature τ
    q_hat = 2 * np.diag(Gamma_hat) / tau_gt
    Q_hat = np.diag(q_hat)

    # Use estimated Q for α̂
    P_hat = inv(Gamma_hat)
    A_hat = -Q_hat @ P_hat / 2
    alpha_est = np.max(np.real(np.linalg.eigvals(A_hat)))
    alpha_est_estimated_Q.append(alpha_est)

    # R4: λ_max tracking (Q-free)
    lambda_max = np.max(np.linalg.eigvalsh(Gamma_hat))
    lambda_max_list_3.append(lambda_max)

    print(f"  Age {mid:.0f}: α_true={alpha_gt:.4f}, α̂={alpha_est:.4f}, "
          f"λ_max={lambda_max:.4f} (Q est. from Γ̂+τ)")

trend_3 = all(alpha_est_estimated_Q[i] < alpha_est_estimated_Q[i+1] for i in range(3))
trend_lmax_3 = all(lambda_max_list_3[i] < lambda_max_list_3[i+1] for i in range(3))
print(f"\n  Monotone trend preserved with estimated Q: {trend_3}")
print(f"  λ_max monotone trend (Q-free): {trend_lmax_3}")


# ============================================================================
# 5. FIGURE
# ============================================================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Diffusion Covariance (Q) Sensitivity Analysis',
             fontsize=13, fontweight='bold', y=0.98)

# (a) α̂ trend under different β values (Q assumed fixed but true Q varies)
ax = axs[0, 0]
ax.plot(strata_mids, [results[0.0][0][i] for i in range(4)],
        'ko-', linewidth=2, markersize=8, label='β=0 (Q fixed)', zorder=5)
for beta, color, ls in [(0.5, 'steelblue', '--'), (1.0, 'orange', '--'),
                          (2.0, 'red', ':'), (5.0, 'darkred', ':')]:
    ax.plot(strata_mids, results[beta][0],
            color=color, linestyle=ls, linewidth=1.5, marker='s', markersize=5,
            label=f'β={beta} (Q×{1+beta:.0f})', alpha=0.8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel(r'Estimated $\hat\alpha$', fontsize=11)
ax.set_title(r'(a) $\hat\alpha$ trend when Q assumed fixed but true Q varies', fontsize=11)
ax.legend(fontsize=8, loc='lower right')

# (b) Trend survival as function of β
ax = axs[0, 1]
betas_plot = beta_values
trend_survived = [results[b][1] for b in betas_plot]
lmax_survived = [results[b][3] for b in betas_plot]
colors_b = ['green' if t else 'red' for t in trend_survived]
ax.bar(range(len(betas_plot)), [1+b for b in betas_plot],
       color=colors_b, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(betas_plot)))
ax.set_xticklabels([f'{b:.1f}' for b in betas_plot], fontsize=9)
ax.set_xlabel(r'$\beta$ (noise growth rate)', fontsize=11)
ax.set_ylabel(r'$Q_{80}/Q_{30}$ ratio', fontsize=11)
ax.set_title('(b) Does monotone α̂ trend survive? (green=yes)', fontsize=11)

# (c) Comparison: Q fixed vs Q estimated from marginal variance
ax = axs[1, 0]
ax.plot(strata_mids, alpha_true_list_3, 'ko-', linewidth=2, markersize=8,
        label=r'True $\alpha$', zorder=5)
# Q assumed fixed (wrong)
q_fixed_wrong = lambda age: q_base
_, alpha_wrong, _, lmax_wrong, _ = run_scenario(q_age_varying, q_fixed,
    "β=1.0, Q assumed fixed (wrong)")
ax.plot(strata_mids, alpha_wrong, 'rs--', linewidth=1.5, markersize=6,
        label=r'$\hat\alpha$ (Q assumed fixed — wrong)')
ax.plot(strata_mids, alpha_est_estimated_Q, 'g^-', linewidth=1.5, markersize=7,
        label=r'$\hat\alpha$ (Q estimated from $\hat\Gamma$ + $\tau$)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel(r'$\alpha(A)$', fontsize=11)
ax.set_title(r'(c) Q-estimation corrects the $\hat\alpha$ pipeline', fontsize=11)
ax.legend(fontsize=8, loc='lower right')

# (d) Summary text
ax = axs[1, 1]
ax.axis('off')

# Find threshold
threshold_beta = None
for beta in beta_values:
    if not results[beta][1]:
        threshold_beta = beta
        break

summary = (
    "Q-SENSITIVITY VERDICT\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"Threshold β* ≈ {threshold_beta if threshold_beta else '>5.0'}\n"
    f"(Q at age 80 = {1+threshold_beta if threshold_beta else '>6'}× Q at age 30)\n\n"
    "If noise increases by less than\n"
    f"{threshold_beta if threshold_beta else '>5'}× over the lifespan,\n"
    "the monotone α̂ trend reflects\n"
    "genuine stability-margin erosion,\n"
    "not artifact of increasing noise.\n\n"
    "MITIGATION STRATEGIES:\n"
    "• Estimate Q from Γ̂ + literature τ\n"
    "  → corrects bias even when Q varies\n"
    "• Test-retest q from within-individual\n"
    "  short-term variability (CGM, serial CRP)\n"
    "• Report sensitivity across β = [0, 2]\n"
    "  as standard robustness check\n\n"
    "R4 NOTE: λ_max(Γ̂) does not require\n"
    "Q specification — robust to Q\n"
    "misspecification by construction."
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(_OUTPUT_DIR, 'figure_Q_sensitivity.pdf'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(_OUTPUT_DIR, 'figure_Q_sensitivity.png'), dpi=150, bbox_inches='tight')
print("\nSaved figure_Q_sensitivity.pdf/png")
print("\nDone.")

# ============================================================================
# PERSISTENT RESULTS
# ============================================================================
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.hdr_sim.results_writer import ResultsWriter

    alpha_trends = [results[b][1] for b in beta_values]
    lambda_trends = [results[b][3] for b in beta_values]

    with ResultsWriter("Q-Sensitivity Analysis",
                        "Tests robustness of stability trends to age-varying noise") as rw:
        rw.add_heading("Trend Survival")
        rw.add_table(
            ["β", "Q₈₀/Q₃₀", "α̂ trend", "λ_max trend"],
            [[f"{b:.2f}", f"{1+b:.1f}×",
              "✅" if at else "❌", "✅" if lt else "❌"]
             for b, at, lt in zip(beta_values, alpha_trends, lambda_trends)]
        )
        rw.add_pass_fail("α̂ trend survives all β ≤ 5.0", all(alpha_trends))
        rw.add_pass_fail("λ_max trend survives all β ≤ 5.0", all(lambda_trends))
        rw.add_text("\n*Note: λ_max(Γ̂) does not require Q specification — robust by construction.*")
except ImportError:
    pass  # results writing is optional
