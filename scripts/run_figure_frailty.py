#!/usr/bin/env python3
"""Generate Figure 5: Frailty perturbation-response."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt

from hdr_sim.dynamics import build_A, spectral_radius_discrete, simulate
from hdr_sim.aging_params import configure, tau_of_age, J_of_age, AXIS_COLORS
from hdr_sim.plotting import setup_style, add_panel_label, save_figure
from hdr_sim.j_matrix_spec import JMatrixSpec, load_default_spec


def parse_args():
    parser = argparse.ArgumentParser(description='Figure 5: Frailty perturbation-response')
    parser.add_argument('--j-matrix', type=str, default=None,
                        help='Path to J matrix CSV. Default: data/J_matrix_compiled_9x9.csv')
    parser.add_argument('--axes', type=str, nargs='+', default=None,
                        help='Axis subset (e.g., I M F). Default: script-specific.')
    return parser.parse_args()

_args = parse_args()

os.makedirs('outputs', exist_ok=True)
setup_style()
configure()
from hdr_sim.aging_params import get_fast_system

ages = np.arange(25, 121, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Panel A: Discrete spectral radius rho vs. age ---
ax = axes[0]
add_panel_label(ax, 'a')

rhos = []
for age in ages:
    _, A_fast, _, _ = get_fast_system(age)
    rhos.append(spectral_radius_discrete(A_fast, dt=1.0))
rhos = np.array(rhos)

ax.plot(ages, rhos, color='#2c3e50', linewidth=2)
ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Age (years)')
ax.set_ylabel(r'Spectral radius $\rho(\Phi)$')

# Annotate frailty zone
frailty_mask = rhos > 0.95
if np.any(frailty_mask):
    frailty_start = ages[frailty_mask][0]
    ax.axvspan(frailty_start, ages[-1], color='#e74c3c', alpha=0.05)
    ax.annotate('frailty transition zone', xy=(frailty_start + 2, 0.97),
                fontsize=8, color='#e74c3c')

# --- Panel B: Per-axis I response at 3 ages ---
ax = axes[1]
add_panel_label(ax, 'b')

demo_ages = [30, 60, 80]
line_styles = ['-', '--', ':']
dt = 0.01
T = 100.0

for age, ls in zip(demo_ages, line_styles):
    _, A_fast, _, _ = get_fast_system(age)
    n_ax = A_fast.shape[0]
    x0 = np.zeros(n_ax); x0[0] = 2.0  # I impulse
    t, x = simulate(A_fast, x0, dt, T)
    ax.plot(t, x[:, 0], ls, color=AXIS_COLORS[0], linewidth=1.5,
            label=f'Age {age}')

ax.set_xlabel('Time (days)')
ax.set_ylabel(r'$\Delta x_I(t)$')
ax.legend(frameon=False, loc='upper right')

plt.tight_layout()
save_figure(fig, 'figure_frailty')
plt.close()

# Save provenance sidecar
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_csv_path = _args.j_matrix or os.path.join(_root, 'data', 'J_matrix_compiled_9x9.csv')
_j_spec = JMatrixSpec.from_csv(_csv_path)
_meta = {
    'j_matrix': _j_spec.to_dict(),
    'script': 'run_figure_frailty.py',
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
}
with open(os.path.join('outputs', 'figure_frailty_meta.json'), 'w') as f:
    json.dump(_meta, f, indent=2)
