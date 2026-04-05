#!/usr/bin/env python3
"""
Figure: Disease Demonstration Panels (Extended Data Figure 2)

Four-panel composite showing disease-specific submatrix dynamics:
  (a) T2D — phase portrait in dxI-dxM plane with bifurcation (reused)
  (b) Frailty — spectral radius approaching unity (reused)
  (c) Alzheimer's disease — threshold coupling with irreversible partition
  (d) Osteoporosis — B axis coupling submatrix with sarcopenia compounding

Usage:
    python scripts/run_figure_disease_demos.py

Outputs:
    outputs/figure_disease_demos.pdf
    outputs/figure_disease_demos.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from hdr_sim.dynamics import build_A, spectral_radius_discrete, simulate
from hdr_sim.aging_params import tau_of_age, J_of_age, AXIS_COLORS
from hdr_sim.plotting import setup_style, add_panel_label, save_figure

os.makedirs('outputs', exist_ok=True)
setup_style()

# ============================================================================
# Coupling values from J_matrix_compiled_9x9.csv
# ============================================================================
# AD-relevant couplings (healthy -> disease basin)
J_I_P = (0.08, 0.20)    # I → P (inflammatory proteasome damage)
J_P_I = (0.15, 0.35)    # P → I (misfolded protein DAMPs → NLRP3)
J_mito_P = (0.20, 0.40)  # mito → P (ROS-driven carbonylation)
J_P_mito = (0.08, 0.18)  # P → mito (OMM import impairment)

# Osteoporosis-relevant couplings (healthy -> disease basin)
J_I_B = (0.10, 0.25)    # I → B (inflammatory osteoclastogenesis)
J_M_B = (0.08, 0.20)    # M → B (AGE-impaired bone quality)
J_F_B = (-0.30, -0.18)  # F → B (mechanical loading, protective)
J_N_B = (0.15, 0.45)    # N → B (GIO: glucocorticoid-induced)


def _interp(anchors, f):
    """Interpolate between healthy (f=0) and disease (f=1) basin."""
    return anchors[0] * (1 - f) + anchors[1] * f


# ============================================================================
# Panel (a): T2D phase portrait — reuse logic from run_figure_t2d.py
# ============================================================================
def panel_t2d(ax):
    """T2D phase portrait in dxI-dxM plane."""
    x_star = 2.0
    s = 1.0
    k = 0.08
    c = 0.04

    def t2d_field(xI, xM):
        fI = -k * xI * (xI - s) * (xI - x_star) + c * xM
        fM = -k * xM * (xM - s) * (xM - x_star) + c * xI
        return fI, fM

    def sim(x0, dt=0.02, n_steps=30000):
        traj = np.zeros((n_steps + 1, 2))
        traj[0] = x0
        for i in range(n_steps):
            dI, dM = t2d_field(traj[i, 0], traj[i, 1])
            traj[i+1, 0] = traj[i, 0] + dI * dt
            traj[i+1, 1] = traj[i, 1] + dM * dt
        return traj

    # Attractors
    attr0 = sim(np.array([0.1, 0.1]))[-1]
    attr1 = sim(np.array([2.4, 2.4]))[-1]

    # Vector field
    grid_pts = 20
    xI_range = np.linspace(-0.3, 2.8, grid_pts)
    xM_range = np.linspace(-0.3, 2.8, grid_pts)
    XI, XM = np.meshgrid(xI_range, xM_range)
    UI, UM = t2d_field(XI, XM)
    speed = np.sqrt(UI**2 + UM**2)

    ax.quiver(XI, XM, UI / (speed + 0.01), UM / (speed + 0.01),
              speed / (speed.max() + 1e-10), cmap='Greys', alpha=0.25,
              scale=30, width=0.004)

    # Basin map
    n_scan = 60
    basin_map = np.zeros((n_scan, n_scan))
    xI_scan = np.linspace(-0.1, 2.6, n_scan)
    xM_scan = np.linspace(-0.1, 2.6, n_scan)
    for ii, xi in enumerate(xI_scan):
        for jj, xm in enumerate(xM_scan):
            final = sim(np.array([xi, xm]), n_steps=12000)[-1]
            d0 = np.linalg.norm(final - attr0)
            d1 = np.linalg.norm(final - attr1)
            basin_map[jj, ii] = 0 if d0 < d1 else 1

    ax.contour(xI_scan, xM_scan, basin_map, levels=[0.5],
               colors='black', linestyles='--', linewidths=1.2)
    ax.contourf(xI_scan, xM_scan, basin_map, levels=[-0.5, 0.5, 1.5],
                colors=['#27ae60', '#e74c3c'], alpha=0.06)

    # Trajectories
    for x0, color, label in [
        ([0.4, 0.5], '#27ae60', 'healthy recovery'),
        ([0.6, 0.6], '#f39c12', 'near separatrix'),
        ([1.2, 1.1], '#e74c3c', 'T2D capture'),
    ]:
        traj = sim(np.array(x0))
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.8, label=label)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=color, markersize=6, zorder=5)

    ax.plot(attr0[0], attr0[1], 's', color='#27ae60', markersize=8, zorder=6,
            markeredgecolor='white', markeredgewidth=0.8)
    ax.plot(attr1[0], attr1[1], 's', color='#e74c3c', markersize=8, zorder=6,
            markeredgecolor='white', markeredgewidth=0.8)

    ax.annotate('healthy', xy=(attr0[0] + 0.15, attr0[1] - 0.2), fontsize=8,
                color='#27ae60', fontweight='bold')
    ax.annotate('T2D', xy=(attr1[0], attr1[1] + 0.2), fontsize=8,
                color='#e74c3c', fontweight='bold', ha='center')
    ax.annotate('separatrix', xy=(0.3, 1.5), fontsize=7, fontstyle='italic')

    ax.set_xlabel(r'$\Delta x_I$ (inflammaging)')
    ax.set_ylabel(r'$\Delta x_M$ (metabolic)')
    ax.set_xlim(-0.3, 2.8)
    ax.set_ylim(-0.3, 2.8)
    ax.legend(frameon=False, loc='center left', fontsize=7,
              bbox_to_anchor=(0.0, 0.55))
    ax.set_aspect('equal')


# ============================================================================
# Panel (b): Frailty — spectral radius vs age
# ============================================================================
def panel_frailty(ax):
    """Spectral radius rho(Phi) vs age with impulse response inset."""
    ages = np.arange(25, 91, 1)
    rhos = []
    for age in ages:
        A = build_A(tau_of_age(age), J_of_age(age))
        rhos.append(spectral_radius_discrete(A, dt=1.0))
    rhos = np.array(rhos)

    ax.plot(ages, rhos, color='#2c3e50', linewidth=2)
    ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Age (years)')
    ax.set_ylabel(r'Spectral radius $\rho(\Phi)$')

    frailty_mask = rhos > 0.95
    if np.any(frailty_mask):
        frailty_start = ages[frailty_mask][0]
        ax.axvspan(frailty_start, ages[-1], color='#e74c3c', alpha=0.05)
        ax.annotate('frailty zone', xy=(frailty_start + 2, 0.97),
                    fontsize=7, color='#e74c3c')

    # Inset: I-axis impulse response at 3 ages
    ax_in = ax.inset_axes([0.12, 0.55, 0.38, 0.38])
    for age, ls in [(30, '-'), (60, '--'), (80, ':')]:
        A = build_A(tau_of_age(age), J_of_age(age))
        x0 = np.array([2.0, 0.0, 0.0, 0.0])
        t, x = simulate(A, x0, 0.01, 80.0)
        ax_in.plot(t, x[:, 0], ls, color=AXIS_COLORS[0], linewidth=1.2,
                   label=f'{age}')
    ax_in.set_xlabel('Days', fontsize=7)
    ax_in.set_ylabel(r'$\Delta x_I$', fontsize=7)
    ax_in.tick_params(labelsize=6)
    ax_in.legend(fontsize=6, frameon=False, title='Age', title_fontsize=6)


# ============================================================================
# Panel (c): Alzheimer's disease — threshold coupling submatrix
# ============================================================================
def panel_alzheimers(ax):
    """
    AD submatrix: {I, P_neural, mito} with threshold A-beta -> tau coupling.

    Shows the piecewise coupling J_{Ab->tau} that is 0 below threshold and
    strongly positive above, creating an irreversible neuronal loss partition.
    """
    # Amyloid burden parameter (proxy for disease progression)
    ab_load = np.linspace(0, 2.5, 300)

    # Threshold coupling: piecewise — zero below, ramps above
    ab_threshold = 1.0
    J_ab_tau_max = 0.45  # strong coupling in disease state

    J_coupling = np.where(
        ab_load < ab_threshold,
        0.0,
        J_ab_tau_max * (1 - np.exp(-2.5 * (ab_load - ab_threshold)))
    )

    # Background: total system instability proxy
    # As Ab load increases, the {I, P, mito} submatrix tightens
    def submatrix_spectral_radius(ab):
        """3x3 submatrix spectral radius for {I, P_neural, mito}."""
        f = np.clip(ab / 2.5, 0, 1)  # maps to disease fraction
        J_sub = np.array([
            [0, _interp(J_P_I, f), _interp(J_mito_P, f) * 0.3],
            [_interp(J_I_P, f), 0, _interp(J_mito_P, f)],
            [0.1 + 0.15 * f, _interp(J_P_mito, f), 0],
        ])
        # Simple instability proxy: largest eigenvalue magnitude
        eigs = np.linalg.eigvals(J_sub)
        return np.max(np.abs(eigs))

    rho_vals = [submatrix_spectral_radius(ab) for ab in ab_load]

    # Plot threshold coupling
    ax.plot(ab_load, J_coupling, color='#8e44ad', linewidth=2.2,
            label=r'$J_{A\beta \to \tau}$')

    # Plot submatrix spectral radius (secondary y-axis)
    ax2 = ax.twinx()
    ax2.plot(ab_load, rho_vals, color='#2c3e50', linewidth=1.5, linestyle='--',
             alpha=0.7, label=r'$\rho(J_{sub})$')
    ax2.set_ylabel(r'$\rho(J_{\mathrm{sub}})$', fontsize=9, color='#2c3e50')
    ax2.tick_params(axis='y', colors='#2c3e50')
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color('#2c3e50')
    ax2.spines['right'].set_linewidth(0.8)

    # Threshold line
    ax.axvline(ab_threshold, color='grey', linestyle=':', linewidth=1, alpha=0.6)
    ax.annotate('threshold', xy=(ab_threshold + 0.05, J_ab_tau_max * 0.85),
                fontsize=7, fontstyle='italic', color='grey')

    # Irreversible neuronal loss shading
    ax.axvspan(ab_threshold, 2.5, color='#e74c3c', alpha=0.06)
    ax.annotate('irreversible\nneuronal loss', xy=(1.8, J_ab_tau_max * 0.5),
                fontsize=7, color='#c0392b', ha='center',
                fontweight='bold', fontstyle='italic')

    # Pre-threshold zone
    ax.annotate('sub-threshold\n(reversible)', xy=(0.4, J_ab_tau_max * 0.3),
                fontsize=7, color='#27ae60', ha='center', fontstyle='italic')

    # Submatrix annotation
    ax.text(0.02, 0.98,
            r'Submatrix: $\{I, P_{\mathrm{neural}}, \mathrm{mito}\}$',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    ax.set_xlabel(r'Amyloid-$\beta$ burden (a.u.)')
    ax.set_ylabel(r'$J_{A\beta \to \tau}$ coupling strength', color='#8e44ad')
    ax.tick_params(axis='y', colors='#8e44ad')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-0.02, J_ab_tau_max * 1.15)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
              loc='center right', frameon=True, facecolor='white',
              edgecolor='grey')


# ============================================================================
# Panel (d): Osteoporosis — B axis coupling submatrix
# ============================================================================
def panel_osteoporosis(ax):
    """
    Osteoporosis coupling submatrix: I->B (+), M->B (+), F->B (-), N->B (+).

    Shows how sarcopenia (increasing dxF) weakens the protective F->B coupling.
    """
    # Disease progression parameter (0 = healthy, 1 = disease)
    f_range = np.linspace(0, 1, 100)

    # Coupling strengths as function of disease progression
    j_ib = [_interp(J_I_B, f) for f in f_range]
    j_mb = [_interp(J_M_B, f) for f in f_range]
    j_fb = [_interp(J_F_B, f) for f in f_range]
    j_nb = [_interp(J_N_B, f) for f in f_range]

    age_range = 30 + f_range * 50  # map f to age 30-80

    # Plot couplings
    ax.plot(age_range, j_ib, color='#e74c3c', linewidth=2, label=r'$I \to B$ (inflammatory)')
    ax.plot(age_range, j_mb, color='#e67e22', linewidth=2, label=r'$M \to B$ (glycaemic)')
    ax.plot(age_range, j_nb, color='#9b59b6', linewidth=2, label=r'$N \to B$ (glucocorticoid)')
    ax.plot(age_range, j_fb, color='#3498db', linewidth=2, label=r'$F \to B$ (mechanical)')

    ax.axhline(0, color='grey', linestyle='-', linewidth=0.5, alpha=0.4)

    # Annotate protective coupling weakening
    # F->B becomes less protective (more positive = less negative)
    sarco_age = 65
    sarco_f = (sarco_age - 30) / 50
    sarco_j = _interp(J_F_B, sarco_f)
    ax.annotate('sarcopenia\ncompounding',
                xy=(sarco_age, sarco_j),
                xytext=(sarco_age - 12, sarco_j + 0.15),
                fontsize=7, color='#2980b9', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2980b9',
                                connectionstyle='arc3,rad=0.2'))

    # Shade the net-positive (pathological) region
    j_net = np.array(j_ib) + np.array(j_mb) + np.array(j_nb) + np.array(j_fb)
    positive_mask = j_net > 0
    if np.any(positive_mask):
        first_pos = age_range[positive_mask][0]
        ax.axvspan(first_pos, 80, color='#e74c3c', alpha=0.04)
        ax.annotate('net pathological\nbone loss drive',
                    xy=(first_pos + 5, 0.35), fontsize=7,
                    color='#c0392b', fontstyle='italic')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel(r'Coupling strength $J_{\cdot \to B}$')
    ax.legend(fontsize=7, loc='upper left', frameon=True,
              facecolor='white', edgecolor='grey')
    ax.set_xlim(30, 80)


# ============================================================================
# Composite figure
# ============================================================================
def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    print("=" * 70)
    print("FIGURE: DISEASE DEMONSTRATION PANELS (ED Fig 2)")
    print("=" * 70)

    print("\n--- Panel (a): T2D phase portrait ---")
    panel_t2d(axes[0, 0])
    add_panel_label(axes[0, 0], '(a)')
    axes[0, 0].set_title('Type 2 Diabetes', fontsize=10)

    print("--- Panel (b): Frailty spectral radius ---")
    panel_frailty(axes[0, 1])
    add_panel_label(axes[0, 1], '(b)')
    axes[0, 1].set_title('Frailty transition', fontsize=10)

    print("--- Panel (c): Alzheimer\'s disease ---")
    panel_alzheimers(axes[1, 0])
    add_panel_label(axes[1, 0], '(c)')
    axes[1, 0].set_title("Alzheimer's disease", fontsize=10)

    print("--- Panel (d): Osteoporosis ---")
    panel_osteoporosis(axes[1, 1])
    add_panel_label(axes[1, 1], '(d)')
    axes[1, 1].set_title('Osteoporosis', fontsize=10)

    plt.tight_layout()
    save_figure(fig, 'figure_disease_demos')
    plt.close(fig)

    print("\n" + "=" * 70)
    print("DONE: figure_disease_demos.pdf")
    print("=" * 70)


if __name__ == '__main__':
    main()
