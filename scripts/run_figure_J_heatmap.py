#!/usr/bin/env python3
"""
Figure 2: 9x9 J Coupling Matrix Heatmap
=========================================

Annotated heatmap of the compiled mechanistic coupling matrix J.
Rows = target axis, columns = source axis.  Diverging RdBu_r colour map.
Diagonal cells greyed out (self-restoration, not coupling).
Unknown entries marked with '?' and hatching.

Outputs:
    outputs/figure_J_heatmap.pdf / .png

Usage:
    python scripts/run_figure_J_heatmap.py
"""

import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from hdr_sim.plotting import setup_style, save_figure

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AXES_9 = ['I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B']


def load_full_matrix(csv_path):
    """Load disease-basin J values, confidence grades, and unknown flags."""
    axis_idx = {a: i for i, a in enumerate(AXES_9)}
    n = len(AXES_9)
    J = np.full((n, n), np.nan)
    grades = [['' for _ in range(n)] for _ in range(n)]
    unknown_mask = np.zeros((n, n), dtype=bool)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['axis_from'].strip()
            tgt = row['axis_to'].strip()
            if src not in axis_idx or tgt not in axis_idx:
                continue
            j = axis_idx[src]
            i = axis_idx[tgt]
            grade = row.get('confidence_grade', '').strip()
            grades[i][j] = grade

            val_str = row['J_disease'].strip()
            sign = row['sign'].strip()

            if val_str in ('', 'NA', 'qual_only'):
                # Qualitative-only: use healthy basin value if available
                val_h = row['J_healthy'].strip()
                if val_h in ('', 'NA', 'qual_only', 'unknown'):
                    J[i, j] = 0.0
                    if sign == '?':
                        unknown_mask[i, j] = True
                else:
                    J[i, j] = float(val_h)
            elif val_str == 'unknown':
                J[i, j] = 0.0
                unknown_mask[i, j] = True
            else:
                J[i, j] = float(val_str)

    # Diagonal = 0 (self-restoration, not coupling)
    np.fill_diagonal(J, 0.0)

    return J, grades, unknown_mask


def main():
    print("Generating Figure 2: 9x9 J heatmap ...")

    csv_path = os.path.join(ROOT, 'data', 'J_matrix_compiled_9x9.csv')
    J, grades, unknown_mask = load_full_matrix(csv_path)
    n = len(AXES_9)

    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # Replace NaN with 0 for display
    J_display = np.where(np.isnan(J), 0.0, J)

    # Diverging colour map centred at 0
    abs_max = max(np.nanmax(np.abs(J_display)), 0.01)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    im = ax.imshow(J_display, cmap='RdBu_r', norm=norm, aspect='equal')

    # Grey out diagonal
    for i in range(n):
        rect = Rectangle((i - 0.5, i - 0.5), 1, 1,
                          linewidth=0, facecolor='#d5d8dc', zorder=2)
        ax.add_patch(rect)

    # Hatching for unknown entries
    for i in range(n):
        for j in range(n):
            if unknown_mask[i, j]:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  linewidth=0, facecolor='#d5d8dc',
                                  hatch='///', edgecolor='#7f8c8d',
                                  alpha=0.7, zorder=2)
                ax.add_patch(rect)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '\u2014', ha='center', va='center',
                        fontsize=8, color='#7f8c8d', zorder=3)
                continue
            if unknown_mask[i, j]:
                ax.text(j, i, '?', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='#2c3e50', zorder=3)
                continue

            val = J_display[i, j]
            grade = grades[i][j]

            # Value text
            if abs(val) < 0.005:
                val_str = 'q'  # qualitative only, non-zero sign
            else:
                val_str = f'{val:+.2f}'

            # Colour: white on dark cells, black on light
            text_color = 'white' if abs(val) > abs_max * 0.55 else '#2c3e50'

            # Two-line annotation: value and grade
            if grade:
                label = f'{val_str}\n({grade})'
            else:
                label = val_str
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=7, color=text_color, zorder=3)

    # Axis labels
    ax.set_xticks(range(n))
    ax.set_xticklabels(AXES_9, fontsize=11, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels(AXES_9, fontsize=11, fontweight='bold')
    ax.set_xlabel('Source axis (j)', fontsize=12)
    ax.set_ylabel('Target axis (i)', fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Coupling strength (SD per SD, disease basin)', fontsize=10)

    # Grid lines
    for i in range(n + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
        ax.axvline(i - 0.5, color='white', linewidth=0.5)

    fig.tight_layout()
    save_figure(fig, 'figure_J_heatmap', OUTPUT_DIR)
    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    main()
