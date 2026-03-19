#!/usr/bin/env python3
"""Generate Figure 4: T2D phase portrait (I-M projection)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt

from hdr_sim.plotting import setup_style, add_panel_label, save_figure

os.makedirs('outputs', exist_ok=True)
setup_style()

# Parameters for double-well potential along each axis
x_star = 2.0   # T2D attractor coordinate
s = 1.0        # approximate separatrix location (saddle point of 1D potential)
k = 0.08       # potential well depth scaling
c = 0.04       # I↔M cross-coupling strength


def t2d_field(xI, xM):
    """Nonlinear vector field with two stable attractors.

    Each axis has a cubic restoring force: f(x) = -k * x * (x - s) * (x - x_star)
    producing stable fixed points at 0 and x_star, with an unstable fixed point at s.
    Cross-coupling c links the two axes (I↔M pathological feedback).
    """
    fI = -k * xI * (xI - s) * (xI - x_star) + c * xM
    fM = -k * xM * (xM - s) * (xM - x_star) + c * xI
    return fI, fM


def simulate_nonlinear(x0, dt, n_steps):
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = x0
    for i in range(n_steps):
        dI, dM = t2d_field(traj[i, 0], traj[i, 1])
        traj[i+1, 0] = traj[i, 0] + dI * dt
        traj[i+1, 1] = traj[i, 1] + dM * dt
    return traj


dt = 0.02
n_steps = 30000

# Find attractors
traj_a0 = simulate_nonlinear(np.array([0.1, 0.1]), dt, n_steps)
traj_a1 = simulate_nonlinear(np.array([2.4, 2.4]), dt, n_steps)
attr0 = traj_a0[-1]
attr1 = traj_a1[-1]
print(f"Attractor 0: ({attr0[0]:.3f}, {attr0[1]:.3f})")
print(f"Attractor 1: ({attr1[0]:.3f}, {attr1[1]:.3f})")

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Vector field
grid_pts = 22
xI_range = np.linspace(-0.3, 2.8, grid_pts)
xM_range = np.linspace(-0.3, 2.8, grid_pts)
XI, XM = np.meshgrid(xI_range, xM_range)
UI, UM = t2d_field(XI, XM)
speed = np.sqrt(UI**2 + UM**2)

ax.quiver(XI, XM, UI / (speed + 0.01), UM / (speed + 0.01),
          speed / (speed.max() + 1e-10), cmap='Greys', alpha=0.3,
          scale=28, width=0.004)

# Compute separatrix by classifying endpoints
print("Computing separatrix...")
n_scan = 80
basin_map = np.zeros((n_scan, n_scan), dtype=int)
xI_scan = np.linspace(-0.1, 2.6, n_scan)
xM_scan = np.linspace(-0.1, 2.6, n_scan)

for ii, xi in enumerate(xI_scan):
    for jj, xm in enumerate(xM_scan):
        traj = simulate_nonlinear(np.array([xi, xm]), dt, 15000)
        final = traj[-1]
        d0 = np.linalg.norm(final - attr0)
        d1 = np.linalg.norm(final - attr1)
        basin_map[jj, ii] = 0 if d0 < d1 else 1

# Draw separatrix as contour at 0.5
ax.contour(xI_scan, xM_scan, basin_map, levels=[0.5],
           colors='black', linestyles='--', linewidths=1.5)

# Shade basins lightly
ax.contourf(xI_scan, xM_scan, basin_map, levels=[-0.5, 0.5, 1.5],
            colors=['#27ae60', '#e74c3c'], alpha=0.06)

# Trajectory 1: Green — near origin → healthy basin
traj1 = simulate_nonlinear(np.array([0.4, 0.5]), dt, n_steps)
ax.plot(traj1[:, 0], traj1[:, 1], color='#27ae60', linewidth=2, label='healthy recovery')
ax.plot(traj1[0, 0], traj1[0, 1], 'o', color='#27ae60', markersize=7, zorder=5)

# Trajectory 2: Yellow — starts near separatrix, returns to healthy
traj2 = simulate_nonlinear(np.array([0.6, 0.6]), dt, n_steps)
ax.plot(traj2[:, 0], traj2[:, 1], color='#f39c12', linewidth=2, label='near separatrix')
ax.plot(traj2[0, 0], traj2[0, 1], 'o', color='#f39c12', markersize=7, zorder=5)

# Trajectory 3: Red — T2D capture
traj3 = simulate_nonlinear(np.array([1.2, 1.1]), dt, n_steps)
ax.plot(traj3[:, 0], traj3[:, 1], color='#e74c3c', linewidth=2, label='T2D capture')
ax.plot(traj3[0, 0], traj3[0, 1], 'o', color='#e74c3c', markersize=7, zorder=5)

# Mark attractors
ax.plot(attr0[0], attr0[1], 's', color='#27ae60', markersize=10, zorder=6,
        markeredgecolor='white', markeredgewidth=1)
ax.plot(attr1[0], attr1[1], 's', color='#e74c3c', markersize=10, zorder=6,
        markeredgecolor='white', markeredgewidth=1)

# Print final positions for debugging
print(f"Traj1 final: ({traj1[-1,0]:.3f}, {traj1[-1,1]:.3f})")
print(f"Traj2 final: ({traj2[-1,0]:.3f}, {traj2[-1,1]:.3f})")
print(f"Traj3 final: ({traj3[-1,0]:.3f}, {traj3[-1,1]:.3f})")

# Annotations
ax.annotate('Basin 0\n(healthy)', xy=(attr0[0] + 0.15, attr0[1] - 0.25), fontsize=9,
            color='#27ae60', fontweight='bold', ha='center')
ax.annotate('Basin 1\n(T2D)', xy=(attr1[0], attr1[1] + 0.3), fontsize=9,
            color='#e74c3c', fontweight='bold', ha='center')
ax.annotate('separatrix', xy=(0.3, 1.6), fontsize=8, fontstyle='italic')

ax.set_xlabel(r'$\Delta x_I$ (inflammaging)')
ax.set_ylabel(r'$\Delta x_M$ (metabolic)')
ax.set_xlim(-0.3, 2.8)
ax.set_ylim(-0.3, 2.8)
ax.legend(frameon=False, loc='center left', fontsize=8,
          bbox_to_anchor=(0.0, 0.55))
ax.set_aspect('equal')

plt.tight_layout()
save_figure(fig, 'figure_t2d')
plt.close()
