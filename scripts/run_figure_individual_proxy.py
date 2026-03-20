"""
Task B: Individual-Level Stability Proxy
========================================
Define an individual-level resilience metric from the population-level Â,
computable from a single cross-sectional biomarker panel.

Concept: project each individual's Δx onto the eigenvectors of Â,
weighted by proximity to instability (1/|Re(λ_k)|).

    s_p = Σ_k (v_k^T Δx_p)² / |Re(λ_k)|

This is a "recovery-time-weighted dysregulation score": it penalizes
deviation along slow-recovering (near-unstable) modes more heavily.

Compare against:
  (a) Mahalanobis distance (Cohen's dysregulation index equivalent)
  (b) Simple L2 norm of Δx
  (c) Stratum-level α̂ (ecological predictor)
  (d) First principal component of Δx

The proxy is individual-level and can be legitimately compared to
Rockwood/Fried in Cox models.
"""

import os
import numpy as np
from scipy.linalg import solve_continuous_lyapunov, inv
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)
os.makedirs('outputs', exist_ok=True)
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

q_true = np.array([0.02, 0.05, 0.08, 0.01])
Q_true = np.diag(q_true)

# ============================================================================
# 1. DEFINE THE STABILITY-WEIGHTED DYSREGULATION SCORE
# ============================================================================

def stability_weighted_score(delta_x, A_hat):
    """
    Compute the stability-weighted dysregulation score for an individual.
    
    s = Σ_k (v_k^T Δx)² / |Re(λ_k)|
    
    where v_k are the right eigenvectors and λ_k the eigenvalues of A_hat.
    Higher score = more vulnerable (more load on near-unstable modes).
    """
    eigvals, eigvecs = np.linalg.eig(A_hat)
    # Use real parts of eigenvectors for real-valued interpretation
    eigvecs_real = np.real(eigvecs)
    eigvals_real = np.real(eigvals)
    
    score = 0.0
    for k in range(len(eigvals)):
        projection = np.dot(eigvecs_real[:, k], delta_x)
        weight = 1.0 / max(abs(eigvals_real[k]), 1e-10)
        score += projection**2 * weight
    return score


def mahalanobis_score(delta_x, Gamma_hat):
    """Cohen-style dysregulation: Mahalanobis distance from youthful centroid."""
    P = inv(Gamma_hat)
    return np.sqrt(delta_x @ P @ delta_x)


def l2_score(delta_x):
    """Simple L2 norm."""
    return np.linalg.norm(delta_x)


def dominant_mode_projection(delta_x, A_hat):
    """Projection onto the single least-stable eigenvector."""
    eigvals, eigvecs = np.linalg.eig(A_hat)
    idx = np.argmax(np.real(eigvals))
    v1 = np.real(eigvecs[:, idx])
    v1 = v1 / np.linalg.norm(v1)
    return abs(np.dot(v1, delta_x))


# ============================================================================
# 2. SIMULATE A SYNTHETIC COHORT WITH OUTCOMES
# ============================================================================

def simulate_outcome(delta_x, A_true, base_hazard=0.01):
    """
    Simulate a binary health outcome (5-year incident event).
    Hazard proportional to true recovery time: h = h0 * (1/|α_individual|)
    where α_individual approximates the individual's local stability.
    """
    # Individual's "effective α" based on their state: 
    # if they are far along the unstable direction, their local dynamics
    # are effectively worse
    eigvals, eigvecs = np.linalg.eig(A_true)
    alpha_true = np.max(np.real(eigvals))
    
    # State-dependent hazard: individuals with larger Δx have worse outcomes
    # Modulated by distance along least-stable direction
    idx = np.argmax(np.real(eigvals))
    v1 = np.real(eigvecs[:, idx])
    v1 = v1 / np.linalg.norm(v1)
    
    # Hazard increases with both magnitude and alignment with instability
    load = abs(np.dot(v1, delta_x))
    hazard = base_hazard * np.exp(0.5 * load + 0.1 * np.linalg.norm(delta_x))
    
    # Bernoulli outcome (5-year event probability)
    prob_event = 1 - np.exp(-hazard * 5)
    return int(np.random.rand() < prob_event)


print("=" * 70)
print("SYNTHETIC COHORT WITH OUTCOMES")
print("=" * 70)

N_total = 5000
individuals = []

for i in range(N_total):
    age = np.random.uniform(40, 79)
    tau, J = get_params(age)
    A = build_A(tau, J)
    Gamma = solve_continuous_lyapunov(A, -Q_true)
    
    # Draw individual state from stationary distribution + measurement noise
    delta_x = np.random.multivariate_normal(np.zeros(n_axes), Gamma)
    y_obs = delta_x + 0.3 * np.random.randn(n_axes)
    
    # Simulate outcome
    event = simulate_outcome(delta_x, A)
    
    individuals.append({
        'age': age,
        'delta_x_true': delta_x,
        'y_obs': y_obs,
        'event': event,
        'A_true': A,
    })

event_rate = np.mean([ind['event'] for ind in individuals])
print(f"  N = {N_total}, event rate = {event_rate:.1%}")


# ============================================================================
# 3. ESTIMATE Â PER AGE STRATUM AND COMPUTE INDIVIDUAL SCORES
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING INDIVIDUAL-LEVEL SCORES")
print("=" * 70)

strata = [(40, 49), (50, 59), (60, 69), (70, 79)]
strata_A_hat = {}

for lo, hi in strata:
    stratum_inds = [ind for ind in individuals if lo <= ind['age'] <= hi]
    Y = np.array([ind['y_obs'] for ind in stratum_inds])
    Gamma_hat = np.cov(Y.T)
    
    # Estimate Â (symmetric approximation)
    P_hat = inv(Gamma_hat)
    A_hat = -Q_true @ P_hat / 2
    strata_A_hat[(lo, hi)] = (A_hat, Gamma_hat)
    
    alpha_hat = np.max(np.real(np.linalg.eigvals(A_hat)))
    print(f"  Stratum {lo}-{hi}: N={len(stratum_inds)}, α̂={alpha_hat:.4f}")

# Compute scores for each individual
scores = {'stability_weighted': [], 'mahalanobis': [], 'l2': [],
          'dominant_mode': [], 'age': [], 'event': [], 'alpha_stratum': []}

for ind in individuals:
    age = ind['age']
    y = ind['y_obs']
    
    # Find stratum
    for lo, hi in strata:
        if lo <= age <= hi:
            A_hat, Gamma_hat = strata_A_hat[(lo, hi)]
            break
    
    scores['stability_weighted'].append(stability_weighted_score(y, A_hat))
    scores['mahalanobis'].append(mahalanobis_score(y, Gamma_hat))
    scores['l2'].append(l2_score(y))
    scores['dominant_mode'].append(dominant_mode_projection(y, A_hat))
    scores['age'].append(age)
    scores['event'].append(ind['event'])
    scores['alpha_stratum'].append(np.max(np.real(np.linalg.eigvals(A_hat))))

for key in scores:
    scores[key] = np.array(scores[key])


# ============================================================================
# 4. EVALUATE PREDICTIVE PERFORMANCE
# ============================================================================

print("\n" + "=" * 70)
print("PREDICTIVE PERFORMANCE (AUC-like analysis)")
print("=" * 70)

def concordance_index(scores_arr, events):
    """Simple concordance: fraction of event/non-event pairs correctly ordered."""
    events_idx = np.where(events == 1)[0]
    nonevents_idx = np.where(events == 0)[0]
    
    if len(events_idx) == 0 or len(nonevents_idx) == 0:
        return 0.5
    
    # Sample pairs for efficiency
    n_pairs = min(50000, len(events_idx) * len(nonevents_idx))
    concordant = 0
    for _ in range(n_pairs):
        i = np.random.choice(events_idx)
        j = np.random.choice(nonevents_idx)
        if scores_arr[i] > scores_arr[j]:
            concordant += 1
        elif scores_arr[i] == scores_arr[j]:
            concordant += 0.5
    
    return concordant / n_pairs


# Compute C-index for each predictor
predictors = {
    'Stability-weighted (proposed)': scores['stability_weighted'],
    'Dominant-mode projection': scores['dominant_mode'],
    'Mahalanobis (Cohen-style)': scores['mahalanobis'],
    'L2 norm (simple)': scores['l2'],
    'Age alone': scores['age'],
    'Stratum α̂ (ecological)': -scores['alpha_stratum'],  # negate: higher α = worse
}

print(f"\n  {'Predictor':40s} {'C-index':>8s}")
print(f"  {'-'*52}")

c_indices = {}
for name, pred in predictors.items():
    c = concordance_index(pred, scores['event'])
    c_indices[name] = c
    marker = ' ★' if 'Stability' in name else ''
    print(f"  {name:40s} {c:8.3f}{marker}")

# Incremental value: stability-weighted beyond age
print(f"\n  Incremental C-index:")
print(f"    Stability-weighted vs Age alone:      "
      f"+{c_indices['Stability-weighted (proposed)'] - c_indices['Age alone']:.3f}")
print(f"    Stability-weighted vs Mahalanobis:     "
      f"+{c_indices['Stability-weighted (proposed)'] - c_indices['Mahalanobis (Cohen-style)']:.3f}")
print(f"    Stability-weighted vs ecological α̂:   "
      f"+{c_indices['Stability-weighted (proposed)'] - c_indices['Stratum α̂ (ecological)']:.3f}")


# ============================================================================
# 5. FIGURE
# ============================================================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Individual-Level Stability Proxy: Derivation and Validation',
             fontsize=13, fontweight='bold', y=0.98)

# (a) Score distributions by outcome
ax = axs[0, 0]
events_mask = scores['event'] == 1
nonevents_mask = scores['event'] == 0
ax.hist(scores['stability_weighted'][nonevents_mask], bins=50, density=True,
        alpha=0.6, color='steelblue', label='No event')
ax.hist(scores['stability_weighted'][events_mask], bins=50, density=True,
        alpha=0.6, color='red', label='Event')
ax.set_xlabel('Stability-weighted dysregulation score', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(a) Score distribution by outcome', fontsize=11)
ax.legend(fontsize=9)

# (b) C-index comparison
ax = axs[0, 1]
names = list(c_indices.keys())
values = list(c_indices.values())
colors_bar = ['#d62728' if 'Stability' in n else '#1f77b4' for n in names]
y_pos = np.arange(len(names))
ax.barh(y_pos, values, color=colors_bar, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels([n.replace(' (proposed)', '\n(proposed)').replace(' (Cohen-style)', '\n(Cohen)') 
                     for n in names], fontsize=8)
ax.set_xlabel('C-index', fontsize=11)
ax.set_title('(b) Predictive discrimination', fontsize=11)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, label='Random')
ax.set_xlim(0.45, max(values) + 0.03)
for i, v in enumerate(values):
    ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=9)
ax.invert_yaxis()

# (c) Stability score vs age, colored by outcome
ax = axs[1, 0]
ax.scatter(scores['age'][nonevents_mask],
           scores['stability_weighted'][nonevents_mask],
           alpha=0.1, s=5, color='steelblue', label='No event')
ax.scatter(scores['age'][events_mask],
           scores['stability_weighted'][events_mask],
           alpha=0.3, s=10, color='red', label='Event')
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('Stability-weighted score', fontsize=11)
ax.set_title('(c) Individual scores vs age (events in red)', fontsize=11)
ax.legend(fontsize=9, markerscale=3)

# (d) Eigenvector loading of the dominant mode across age strata
ax = axs[1, 1]
strata_mids = [44.5, 54.5, 64.5, 74.5]
for axis_idx, (axis_name, color) in enumerate(zip(AXES, ['red', 'orange', 'blue', 'green'])):
    loadings = []
    for mid in strata_mids:
        tau, J = get_params(mid)
        A = build_A(tau, J)
        eigvals, eigvecs = np.linalg.eig(A)
        idx = np.argmax(np.real(eigvals))
        v1 = np.abs(np.real(eigvecs[:, idx]))
        v1 = v1 / np.max(v1)
        loadings.append(v1[axis_idx])
    ax.plot(strata_mids, loadings, 'o-', color=color, linewidth=2,
            markersize=7, label=f'{axis_name} axis')
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel('Eigenvector loading (normalized)', fontsize=11)
ax.set_title('(d) Dominant-mode composition across age', fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs/figure_individual_proxy.pdf', dpi=150, bbox_inches='tight')
plt.savefig('outputs/figure_individual_proxy.png', dpi=150, bbox_inches='tight')
print("\nSaved figure_individual_proxy.pdf/png")
print("\nDone.")
