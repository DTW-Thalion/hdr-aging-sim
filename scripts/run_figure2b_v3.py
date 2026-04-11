#!/usr/bin/env python3
"""Generate Figure 2b v3: spectral-abscissa drift with fast-subsystem calibration.

Uses the two-timescale architecture: 6-axis fast subsystem (I, M, P, C, N, F)
calibrated for stability from age 25 to 120.

Panels:
  a) alpha_fast(age) with Pyrkov target and alpha_full
  b) Recovery timescale 1/|alpha_fast| (days)
  c) Damping ratio zeta(age)
  d) Perturbation response at 3 ages
  e) Per-axis trajectories at age 80

Saves to outputs/figure_2b_v3.pdf.
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
from scipy import linalg

from hdr_sim.csv_loader import (
    load_J_csv, build_J_basin_imputed,
    calibrate_stable_system, build_system_at_age,
    _ALL_9_AXES, _FAST_7_AXES,
)
from hdr_sim.dynamics import simulate, spectral_abscissa, damping_ratio
from hdr_sim.plotting import setup_style, add_panel_label, save_figure
from hdr_sim.aging_params import _AXIS_FULL_NAMES, _AXIS_COLORS_MAP
from hdr_sim.j_matrix_spec import JMatrixSpec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)

setup_style()

# Load and calibrate
rows = load_J_csv()
J_h = build_J_basin_imputed(rows, 'healthy', _ALL_9_AXES)
J_d = build_J_basin_imputed(rows, 'disease', _ALL_9_AXES)

cal = calibrate_stable_system(J_h, J_d, axes_fast=_FAST_7_AXES)
c = cal['c']
amp = cal['amplitude']

axes_fast = _FAST_7_AXES
axis_names = [_AXIS_FULL_NAMES.get(a, a) for a in axes_fast]
axis_colors = [_AXIS_COLORS_MAP.get(a, '#7f8c8d') for a in axes_fast]
n_fast = len(axes_fast)

# Sweep
ages = np.arange(25, 121, 1)
alphas_fast = np.zeros(len(ages))
alphas_full = np.zeros(len(ages))
rec_times = np.zeros(len(ages))
zetas = np.zeros(len(ages))

for i, age in enumerate(ages):
    A_full, A_fast, af, afull = build_system_at_age(
        age, J_h, J_d, c, amp, axes_fast=axes_fast)
    alphas_fast[i] = af
    alphas_full[i] = afull
    rec_times[i] = 1.0 / abs(af) if af != 0 else float('inf')
    zetas[i] = damping_ratio(A_fast)

# Pyrkov target
pyrkov = -0.134 * np.exp(-0.038 * (ages - 25))

print(f"Calibration: c={c:.4f}, amplitude={amp:.6f}")
print(f"alpha_fast(25) = {alphas_fast[0]:.4f}")
print(f"alpha_fast(80) = {alphas_fast[55]:.4f}")
print(f"alpha_fast(120) = {alphas_fast[-1]:.4f}")
print(f"alpha_full(25) = {alphas_full[0]:.4f}")

# --- Figure ---
fig = plt.figure(figsize=(14, 9))
gs = gridspec.GridSpec(2, 6, hspace=0.35, wspace=0.8)

ax_a = fig.add_subplot(gs[0, 0:2])
ax_b = fig.add_subplot(gs[0, 2:4])
ax_c = fig.add_subplot(gs[0, 4:6])
ax_d = fig.add_subplot(gs[1, 0:3])
ax_e = fig.add_subplot(gs[1, 3:6])

# Panel A: alpha_fast and alpha_full
add_panel_label(ax_a, 'a')
ax_a.plot(ages, alphas_fast, color='#2c3e50', linewidth=2,
          label=r'$\alpha_{fast}$')
ax_a.plot(ages, alphas_full, color='#95a5a6', linewidth=1, linestyle=':',
          label=r'$\alpha_{full}$ (E-dominated)')
ax_a.plot(ages, pyrkov, '--', color='#e67e22', linewidth=1,
          label='Pyrkov target')
ax_a.axhline(0, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
ax_a.fill_between(ages, 0, max(0.02, np.max(alphas_full) + 0.02),
                  color='#e74c3c', alpha=0.05)
ax_a.set_xlabel('Age (years)')
ax_a.set_ylabel(r'Spectral abscissa $\alpha(A)$')
ax_a.set_title('7-axis fast subsystem')
ax_a.legend(frameon=False, fontsize=7, loc='upper left')

# Panel B: Recovery timescale
add_panel_label(ax_b, 'b')
ax_b.plot(ages, rec_times, color='#8e44ad', linewidth=2)
ax_b.set_xlabel('Age (years)')
ax_b.set_ylabel(r'Recovery timescale $1/|\alpha_{fast}|$ (days)')
ax_b.set_ylim(0, min(300, np.max(rec_times) * 1.1))

# Panel C: Damping ratio
add_panel_label(ax_c, 'c')
ax_c.plot(ages, zetas, color='#16a085', linewidth=2)
ax_c.set_xlabel('Age (years)')
ax_c.set_ylabel(r'Damping ratio $\zeta$')
ax_c.axhline(0.707, color='#95a5a6', linestyle=':', linewidth=0.8)
ax_c.annotate(r'$\zeta = 1/\sqrt{2}$', xy=(27, 0.715), fontsize=8,
              color='#95a5a6')

# Panel D: Perturbation response at 3 ages
add_panel_label(ax_d, 'd')
demo_ages = [25, 60, 100]
line_styles = ['-', '--', ':']
dt = 0.01
T = 150.0

for age, ls in zip(demo_ages, line_styles):
    _, A_fast, _, _ = build_system_at_age(
        age, J_h, J_d, c, amp, axes_fast=axes_fast)
    x0 = np.zeros(n_fast)
    x0[0] = 2.0  # perturb I axis
    t, x = simulate(A_fast, x0, dt, T)
    norm = np.linalg.norm(x, axis=1)
    ax_d.plot(t, norm, ls, color='#2c3e50', linewidth=1.5,
              label=f'Age {age}')

ax_d.set_xlabel('Time (days)')
ax_d.set_ylabel(r'$\|\Delta \mathbf{x}(t)\|$')
ax_d.legend(frameon=False, loc='upper right')

# Panel E: Per-axis trajectories at age 80
add_panel_label(ax_e, 'e')
_, A_80, _, _ = build_system_at_age(80, J_h, J_d, c, amp, axes_fast=axes_fast)
x0 = np.zeros(n_fast)
perturbations = [(5.0, 0, 1.5)]
t, x = simulate(A_80, x0, dt, T, noise_std=0.05, perturbations=perturbations)

for i in range(n_fast):
    ax_e.plot(t, x[:, i], color=axis_colors[i], linewidth=1.5,
              label=axis_names[i])

ax_e.set_xlabel('Time (days)')
ax_e.set_ylabel(r'$\Delta x_i(t)$')
ax_e.legend(frameon=False, loc='upper right', fontsize=7)
ax_e.set_title('Age 80')

save_figure(fig, 'figure_2b_v4', output_dir=_OUTPUT_DIR)
plt.close()

# Provenance sidecar
_csv_path = os.path.join(_REPO_ROOT, 'data', 'J_matrix_compiled_9x9.csv')
_j_spec = JMatrixSpec.from_csv(_csv_path)
_meta = {
    'j_matrix': _j_spec.to_dict(),
    'script': 'run_figure2b_v3.py',
    'tau_registry': 'V2 (literature-calibrated)',
    'architecture': 'Two-timescale: 7-axis fast + 2-axis slow',
    'calibration': cal,
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
}
with open(os.path.join(_OUTPUT_DIR, 'figure_2b_v4_meta.json'), 'w') as f:
    json.dump(_meta, f, indent=2, default=float)

print(f"\nFigure saved to {os.path.join(_OUTPUT_DIR, 'figure_2b_v4.pdf')}")
