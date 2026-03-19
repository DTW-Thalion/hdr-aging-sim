#!/usr/bin/env python3
"""Generate Figure 2b: 4-panel aging dynamics demo."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from hdr_sim.dynamics import build_A, spectral_abscissa, recovery_timescale, simulate
from hdr_sim.aging_params import tau_of_age, J_of_age, AXIS_NAMES, AXIS_COLORS
from hdr_sim.plotting import setup_style, add_panel_label, save_figure

os.makedirs('outputs', exist_ok=True)
setup_style()

ages = np.arange(25, 91, 1)

# Compute spectral abscissa and recovery timescale across ages
alphas = []
rec_times = []
for age in ages:
    A = build_A(tau_of_age(age), J_of_age(age))
    alphas.append(spectral_abscissa(A))
    rec_times.append(recovery_timescale(A))

alphas = np.array(alphas)
rec_times = np.array(rec_times)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# --- Panel A: Spectral abscissa vs. age ---
ax = axes[0, 0]
add_panel_label(ax, 'a')
ax.plot(ages, alphas, color='#2c3e50', linewidth=2)
ax.axhline(0, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
ax.fill_between(ages, 0, max(0.02, np.max(alphas) + 0.01),
                color='#e74c3c', alpha=0.08, label='unstable')
ax.set_xlabel('Age (years)')
ax.set_ylabel(r'Spectral abscissa $\alpha(A)$')
ax.annotate('healthy margin', xy=(32, alphas[7]), fontsize=8,
            color='#27ae60', ha='left')
ax.annotate('frailty zone', xy=(78, alphas[-12] + 0.003), fontsize=8,
            color='#e74c3c', ha='right')

# --- Panel B: Recovery timescale vs. age ---
ax = axes[0, 1]
add_panel_label(ax, 'b')
ax.plot(ages, rec_times, color='#8e44ad', linewidth=2)
ax.set_xlabel('Age (years)')
ax.set_ylabel(r'Recovery timescale $1/|\alpha|$ (time units)')
ax.set_ylim(0, min(200, np.max(rec_times) * 1.1))

# --- Panel C: Perturbation-response at 3 ages ---
ax = axes[1, 0]
add_panel_label(ax, 'c')

demo_ages = [30, 60, 80]
line_styles = ['-', '--', ':']
dt = 0.01
T = 100.0

for age, ls in zip(demo_ages, line_styles):
    A = build_A(tau_of_age(age), J_of_age(age))
    x0 = np.array([2.0, 0.0, 0.0, 0.0])
    t, x = simulate(A, x0, dt, T)
    norm = np.linalg.norm(x, axis=1)
    ax.plot(t, norm, ls, color='#2c3e50', linewidth=1.5,
            label=f'Age {age}')

ax.set_xlabel('Time (days)')
ax.set_ylabel(r'$\|\Delta \mathbf{x}(t)\|$')
ax.legend(frameon=False, loc='upper right')

# --- Panel D: Per-axis trajectories at age 80 ---
ax = axes[1, 1]
add_panel_label(ax, 'd')

A_80 = build_A(tau_of_age(80), J_of_age(80))
x0 = np.zeros(4)
perturbations = [(5.0, 0, 1.5)]  # perturbation in I at t=5
t, x = simulate(A_80, x0, dt, T, noise_std=0.05, perturbations=perturbations)

for i in range(4):
    ax.plot(t, x[:, i], color=AXIS_COLORS[i], linewidth=1.5,
            label=AXIS_NAMES[i])

ax.set_xlabel('Time (days)')
ax.set_ylabel(r'$\Delta x_i(t)$')
ax.legend(frameon=False, loc='upper right', fontsize=8)

# Annotate perturbation
ax.annotate('perturbation\nin I', xy=(5, 1.5), xytext=(15, 1.6),
            fontsize=7, color=AXIS_COLORS[0],
            arrowprops=dict(arrowstyle='->', color=AXIS_COLORS[0], lw=0.8))

# Find approximate peak of M response for annotation
m_trace = x[:, 1]
peak_idx = np.argmax(np.abs(m_trace[int(5/dt):])) + int(5/dt)
if abs(m_trace[peak_idx]) > 0.05:
    ax.annotate(r'propagation to M via $J_{M \leftarrow I}$',
                xy=(t[peak_idx], m_trace[peak_idx]),
                xytext=(t[peak_idx] + 10, m_trace[peak_idx] + 0.3),
                fontsize=7, color=AXIS_COLORS[1],
                arrowprops=dict(arrowstyle='->', color=AXIS_COLORS[1], lw=0.8))

plt.tight_layout()
save_figure(fig, 'figure_2b')
plt.close()

# Print calibration info
print(f"\nCalibration check:")
print(f"  α(30) = {spectral_abscissa(build_A(tau_of_age(30), J_of_age(30))):.4f}")
print(f"  α(80) = {spectral_abscissa(build_A(tau_of_age(80), J_of_age(80))):.4f}")
print(f"  Recovery timescale ratio (80/30) = {recovery_timescale(build_A(tau_of_age(80), J_of_age(80))) / recovery_timescale(build_A(tau_of_age(30), J_of_age(30))):.1f}×")
