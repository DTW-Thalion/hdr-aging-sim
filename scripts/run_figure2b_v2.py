#!/usr/bin/env python3
"""Generate Figure 2b v2: spectral abscissa drift across 25–120 with V2 τ registry.

Extends the original figure2b to the full 25–120 age range using
literature-calibrated τ values and imputed J matrix.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

from hdr_sim.dynamics import build_A, spectral_abscissa, recovery_timescale, damping_ratio, simulate, simulate_expm
from hdr_sim.aging_params import configure_v2, tau_of_age, J_of_age, get_axis_names, get_axis_colors
from hdr_sim.plotting import setup_style, add_panel_label, save_figure
from hdr_sim.j_matrix_spec import JMatrixSpec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)

setup_style()

# Configure with V2 registry — use 4-axis for visualization
# (9-axis system has quasi-static E/B that constrain alpha near zero)
configure_v2(axes=('I', 'M', 'N', 'F'))

ages = np.arange(25, 121, 1)

# Compute spectral quantities
alphas = []
rec_times = []
zetas = []
for age in ages:
    tau = tau_of_age(age)
    J = J_of_age(age)
    A = build_A(tau, J)
    alpha = spectral_abscissa(A)
    alphas.append(alpha)
    rec_times.append(recovery_timescale(A))
    zetas.append(damping_ratio(A))

alphas = np.array(alphas)
rec_times = np.array(rec_times)
zetas = np.array(zetas)

# Find stability boundary
stable_mask = alphas < 0
if np.all(stable_mask):
    max_stable = 120
elif np.any(stable_mask):
    max_stable = ages[stable_mask][-1]
else:
    max_stable = 25

print(f"Max stable age: {max_stable}")
print(f"alpha(25) = {alphas[0]:.4f}")
print(f"alpha(80) = {alphas[55]:.4f}" if len(alphas) > 55 else "")

# Pyrkov target
pyrkov_ages = np.arange(25, 121, 1)
pyrkov_alpha = -0.134 * np.exp(-0.038 * (pyrkov_ages - 25))

# --- 5-panel layout ---
fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 6, hspace=0.35, wspace=0.8)

ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 0:3])
ax_e = fig.add_subplot(gs[1, 3:6])

# Panel A: Spectral abscissa vs age
add_panel_label(ax_a, 'a')
ax_a.plot(ages, alphas, color='#2c3e50', linewidth=2, label=r'$\alpha(A)$ V2')
ax_a.plot(pyrkov_ages, pyrkov_alpha, '--', color='#95a5a6', linewidth=1,
          label='Pyrkov target')
ax_a.axhline(0, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
ax_a.fill_between(ages, 0, max(0.02, np.max(alphas) + 0.01),
                  color='#e74c3c', alpha=0.08)
ax_a.set_xlabel('Age (years)')
ax_a.set_ylabel(r'Spectral abscissa $\alpha(A)$')
ax_a.set_title('V2 lit-calibrated (4-axis)')
ax_a.legend(frameon=False, fontsize=7)

# Panel B: Recovery timescale
add_panel_label(ax_b, 'b')
stable_ages = ages[stable_mask]
stable_rec = rec_times[stable_mask]
ax_b.plot(stable_ages, stable_rec, color='#8e44ad', linewidth=2)
ax_b.set_xlabel('Age (years)')
ax_b.set_ylabel(r'Recovery timescale $1/|\alpha|$ (days)')
if len(stable_rec) > 0:
    ax_b.set_ylim(0, min(500, np.max(stable_rec) * 1.1))

# Panel C: Damping ratio
add_panel_label(ax_c, 'c')
ax_c.plot(stable_ages, np.array(zetas)[stable_mask], color='#16a085', linewidth=2)
ax_c.set_xlabel('Age (years)')
ax_c.set_ylabel(r'Damping ratio $\zeta$')
ax_c.axhline(0.707, color='#95a5a6', linestyle=':', linewidth=0.8)

# Panel D: Perturbation response at 3 ages
add_panel_label(ax_d, 'd')
n_axes = len(tau_of_age(25))
demo_ages = [25, 50, min(max_stable, 80)]
line_styles = ['-', '--', ':']
dt = 0.01
T = 100.0

for age, ls in zip(demo_ages, line_styles):
    A = build_A(tau_of_age(age), J_of_age(age))
    x0 = np.zeros(n_axes)
    x0[0] = 2.0
    t, x = simulate_expm(A, x0, dt, T)
    norm = np.linalg.norm(x, axis=1)
    ax_d.plot(t, norm, ls, color='#2c3e50', linewidth=1.5, label=f'Age {age}')

ax_d.set_xlabel('Time (days)')
ax_d.set_ylabel(r'$\|\Delta \mathbf{x}(t)\|$')
ax_d.legend(frameon=False, loc='upper right')

# Panel E: Per-axis trajectories at boundary age
add_panel_label(ax_e, 'e')
demo_age = min(max_stable, 80)
A_demo = build_A(tau_of_age(demo_age), J_of_age(demo_age))
x0 = np.zeros(n_axes)
perturbations = [(5.0, 0, 1.5)]
t, x = simulate_expm(A_demo, x0, dt, T, noise_std=0.05, perturbations=perturbations)

axis_names = get_axis_names()
axis_colors = get_axis_colors()
for i in range(n_axes):
    ax_e.plot(t, x[:, i], color=axis_colors[i], linewidth=1.5, label=axis_names[i])

ax_e.set_xlabel('Time (days)')
ax_e.set_ylabel(r'$\Delta x_i(t)$')
ax_e.legend(frameon=False, loc='upper right', fontsize=8)
ax_e.set_title(f'Age {demo_age}')

save_figure(fig, 'figure_2b_v2', output_dir=_OUTPUT_DIR)
plt.close()

# Save provenance sidecar
_csv_path = os.path.join(_REPO_ROOT, 'data', 'J_matrix_compiled_9x9.csv')
_j_spec = JMatrixSpec.from_csv(_csv_path)
_meta = {
    'j_matrix': _j_spec.to_dict(),
    'script': 'run_figure2b_v2.py',
    'tau_registry': 'V2 (literature-calibrated)',
    'age_range': [25, 120],
    'max_stable_age': int(max_stable),
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
}
with open(os.path.join(_OUTPUT_DIR, 'figure_2b_v2_meta.json'), 'w') as f:
    json.dump(_meta, f, indent=2)

print(f"\nFigure saved to {os.path.join(_OUTPUT_DIR, 'figure_2b_v2.pdf')}")
