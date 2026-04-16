#!/usr/bin/env python3
"""Generate Figure 2b: 5-panel aging dynamics demo (3 top + 2 bottom)."""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg as _sla

from hdr_sim.dynamics import (build_A, spectral_abscissa, recovery_timescale,
                               damping_ratio, simulate)
from hdr_sim.aging_params import (tau_of_age, J_of_age, AXIS_NAMES, AXIS_COLORS,
                                   configure, get_fast_system)
from hdr_sim.csv_loader import _FAST_7_AXES
from hdr_sim.plotting import setup_style, add_panel_label, save_figure
from hdr_sim.j_matrix_spec import JMatrixSpec, load_default_spec


def simulate_expm(A, x0, dt, T, noise_std=0.0, perturbations=None, seed=42):
    """Matrix-exponential integrator: x[i+1] = expm(A*dt) @ x[i] + noise.

    Unconditionally stable regardless of eigenvalue magnitudes — avoids the
    Euler-Maruyama stiffness blowup that occurs when |lambda_max * dt| > 2.
    The fast subsystem has eigenvalues up to |lambda| ~ 333 at young ages,
    which exceeds the Euler stability limit of |lambda| < 200 at dt=0.01.
    """
    n_steps = int(T / dt)
    n = len(x0)
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros((n_steps + 1, n))
    x[0] = x0.copy()

    Phi = _sla.expm(A * dt)

    pert_dict = {}
    if perturbations:
        for p_time, p_axis, p_mag in perturbations:
            step_idx = max(0, min(int(round(p_time / dt)), n_steps))
            pert_dict.setdefault(step_idx, []).append((p_axis, p_mag))

    rng = np.random.default_rng(seed)
    for i in range(n_steps):
        if i in pert_dict:
            for axis, mag in pert_dict[i]:
                x[i, axis] += mag
        x[i + 1] = Phi @ x[i]
        if noise_std > 0:
            x[i + 1] += noise_std * np.sqrt(dt) * rng.standard_normal(n)
    return t, x


def parse_args():
    parser = argparse.ArgumentParser(description='Figure 2b: 5-panel aging dynamics demo')
    parser.add_argument('--j-matrix', type=str, default=None,
                        help='Path to J matrix CSV. Default: data/J_matrix_compiled_9x9.csv')
    parser.add_argument('--axes', type=str, nargs='+', default=None,
                        help='Axis subset (e.g., I M F). Default: script-specific.')
    return parser.parse_args()

_args = parse_args()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)
setup_style()
configure()

# Use 7-axis fast subsystem for stable dynamics across full lifespan
FAST_AXIS_NAMES = [f'{a}' for a in _FAST_7_AXES]
FAST_AXIS_COLORS = ['#e74c3c', '#e67e22', '#f39c12', '#1abc9c', '#2980b9', '#3498db', '#27ae60']

ages = np.arange(25, 121, 1)

# Compute spectral abscissa, recovery timescale, and damping ratio across ages
alphas = []
rec_times = []
zetas = []
for age in ages:
    _, A_fast, alpha_fast, _ = get_fast_system(age)
    alphas.append(alpha_fast)
    rec_times.append(recovery_timescale(A_fast))
    zetas.append(damping_ratio(A_fast))

alphas = np.array(alphas)
rec_times = np.array(rec_times)
zetas = np.array(zetas)

# --- 5-panel layout: 3 top + 2 bottom ---
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 6, hspace=0.35, wspace=0.8)

ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 0:3])
ax_e = fig.add_subplot(gs[1, 3:6])

# --- Panel A: Spectral abscissa vs. age ---
add_panel_label(ax_a, 'a')
ax_a.plot(ages, alphas, color='#2c3e50', linewidth=2)
ax_a.axhline(0, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
ax_a.fill_between(ages, 0, max(0.02, np.max(alphas) + 0.01),
                  color='#e74c3c', alpha=0.08, label='unstable')
ax_a.set_xlabel('Age (years)')
ax_a.set_ylabel(r'Spectral abscissa $\alpha(A)$')
ax_a.annotate('healthy margin', xy=(32, alphas[7]), fontsize=8,
              color='#27ae60', ha='left')
ax_a.annotate('frailty zone', xy=(78, alphas[-12] + 0.003), fontsize=8,
              color='#e74c3c', ha='right')

# --- Panel B: Recovery timescale vs. age ---
add_panel_label(ax_b, 'b')
ax_b.plot(ages, rec_times, color='#8e44ad', linewidth=2)
ax_b.set_xlabel('Age (years)')
ax_b.set_ylabel(r'Recovery timescale $1/|\alpha|$ (days)')
ax_b.set_ylim(0, min(200, np.max(rec_times) * 1.1))

# --- Panel C: Damping ratio ζ vs. age ---
add_panel_label(ax_c, 'c')
ax_c.plot(ages, zetas, color='#16a085', linewidth=2)
ax_c.set_xlabel('Age (years)')
ax_c.set_ylabel(r'Damping ratio $\zeta$')
ax_c.axhline(0.707, color='#95a5a6', linestyle=':', linewidth=0.8, alpha=0.7)
ax_c.annotate(r'$\zeta = 1/\sqrt{2}$', xy=(27, 0.715), fontsize=8,
              color='#95a5a6')
ax_c.annotate('overdamped', xy=(32, zetas[7] + 0.005), fontsize=8,
              color='#27ae60', ha='left')
ax_c.annotate('underdamped', xy=(78, zetas[-12] - 0.02), fontsize=8,
              color='#e74c3c', ha='right')
ax_c.set_ylim(0.5, 1.05)

# --- Panel D: Perturbation-response at 3 ages ---
add_panel_label(ax_d, 'd')

demo_ages = [30, 60, 80]
line_styles = ['-', '--', ':']
dt = 0.01
T = 100.0

for age, ls in zip(demo_ages, line_styles):
    _, A, _, _ = get_fast_system(age)
    n_ax = A.shape[0]
    x0 = np.zeros(n_ax); x0[0] = 2.0  # perturbation in I axis
    t, x = simulate_expm(A, x0, dt, T)
    norm = np.linalg.norm(x, axis=1)
    ax_d.plot(t, norm, ls, color='#2c3e50', linewidth=1.5,
              label=f'Age {age}')

ax_d.set_xlabel('Time (days)')
ax_d.set_ylabel(r'$\|\Delta \mathbf{x}(t)\|$')
ax_d.legend(frameon=False, loc='upper right')

# --- Panel E: Per-axis trajectories at age 80 ---
add_panel_label(ax_e, 'e')

_, A_80, _, _ = get_fast_system(80)
n_ax = A_80.shape[0]
x0 = np.zeros(n_ax)
perturbations = [(5.0, 0, 1.5)]  # perturbation in I at t=5
t, x = simulate_expm(A_80, x0, dt, T, noise_std=0.05, perturbations=perturbations)

for i in range(min(n_ax, len(FAST_AXIS_COLORS))):
    ax_e.plot(t, x[:, i], color=FAST_AXIS_COLORS[i], linewidth=1.5,
              label=FAST_AXIS_NAMES[i])

ax_e.set_xlabel('Time (days)')
ax_e.set_ylabel(r'$\Delta x_i(t)$')
ax_e.legend(frameon=False, loc='upper right', fontsize=8)

# Annotate perturbation
ax_e.annotate('perturbation\nin I', xy=(5, 1.5), xytext=(15, 1.6),
              fontsize=7, color=AXIS_COLORS[0],
              arrowprops=dict(arrowstyle='->', color=AXIS_COLORS[0], lw=0.8))

# Find approximate peak of M response for annotation
m_trace = x[:, 1]
peak_idx = np.argmax(np.abs(m_trace[int(5/dt):])) + int(5/dt)
if abs(m_trace[peak_idx]) > 0.05:
    ax_e.annotate(r'propagation to M via $J_{M \leftarrow I}$',
                  xy=(t[peak_idx], m_trace[peak_idx]),
                  xytext=(t[peak_idx] + 10, m_trace[peak_idx] + 0.3),
                  fontsize=7, color=AXIS_COLORS[1],
                  arrowprops=dict(arrowstyle='->', color=AXIS_COLORS[1], lw=0.8))

save_figure(fig, 'figure_2b', output_dir=_OUTPUT_DIR)
plt.close()

# Print calibration info
_, A25, alpha_25, _ = get_fast_system(25)
_, A80, alpha_80, _ = get_fast_system(80)
_, A120, alpha_120, _ = get_fast_system(120)
print(f"\nCalibration check (7-axis fast subsystem):")
print(f"  alpha(25) = {alpha_25:.4f}")
print(f"  alpha(80) = {alpha_80:.4f}")
print(f"  alpha(120) = {alpha_120:.4f}")
print(f"  Recovery timescale ratio (80/25) = {recovery_timescale(A80) / recovery_timescale(A25):.1f}x")
print(f"  Recovery timescale ratio (120/25) = {recovery_timescale(A120) / recovery_timescale(A25):.1f}x")

# Save provenance sidecar
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_csv_path = _args.j_matrix or os.path.join(_root, 'data', 'J_matrix_compiled_9x9.csv')
_j_spec = JMatrixSpec.from_csv(_csv_path)
_meta = {
    'j_matrix': _j_spec.to_dict(),
    'script': 'run_figure2b.py',
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
}
with open(os.path.join(_OUTPUT_DIR, 'figure_2b_meta.json'), 'w') as f:
    json.dump(_meta, f, indent=2)
