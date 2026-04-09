#!/usr/bin/env python3
"""
Figure 1: 9-Axis Network Schematic
====================================

Publication-quality network diagram of the 9-axis HDR coupling matrix.
Nodes arranged in a circle, edges coloured by sign (red=pathological,
blue=protective, grey dashed=unknown/ambiguous).  Edge width proportional
to |J_ij|.

Outputs:
    outputs/figure_network_schematic.pdf / .png

Usage:
    python scripts/run_figure_network_schematic.py
"""

import csv
import os
import sys
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from hdr_sim.plotting import setup_style, save_figure
from hdr_sim.j_matrix_spec import JMatrixSpec, load_default_spec


def parse_args():
    parser = argparse.ArgumentParser(description='Figure 1: 9-Axis Network Schematic')
    parser.add_argument('--j-matrix', type=str, default=None,
                        help='Path to J matrix CSV. Default: data/J_matrix_compiled_9x9.csv')
    parser.add_argument('--axes', type=str, nargs='+', default=None,
                        help='Axis subset (e.g., I M F). Default: all 9 axes.')
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AXES_9 = ['I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B']
AXES_FULL_NAMES = [
    'Inflammaging', 'Metabolic', 'Epigenetic',
    'Mitochondrial', 'Proteostatic', 'Circadian',
    'Neuroendocrine', 'Functional', 'Bone',
]

# Node colours — one per axis
NODE_COLORS = [
    '#e74c3c',  # I — red
    '#e67e22',  # M — orange
    '#9b59b6',  # E — purple
    '#f39c12',  # mito — gold
    '#1abc9c',  # P — teal
    '#3498db',  # C — blue
    '#2980b9',  # N — dark blue
    '#27ae60',  # F — green
    '#95a5a6',  # B — grey
]


def load_9x9_J(csv_path):
    """Load the compiled 9x9 J matrix (disease basin) and sign info."""
    axis_idx = {a: i for i, a in enumerate(AXES_9)}
    J = np.zeros((9, 9))
    signs = {}  # (i, j) -> '+', '-', '?'

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['axis_from'].strip()
            tgt = row['axis_to'].strip()
            if src not in axis_idx or tgt not in axis_idx:
                continue
            j = axis_idx[src]
            i = axis_idx[tgt]
            sign = row['sign'].strip()
            signs[(i, j)] = sign

            val_str = row['J_disease'].strip()
            if val_str in ('', 'NA', 'qual_only', 'unknown'):
                # Use J_healthy as fallback for magnitude
                val_h = row['J_healthy'].strip()
                if val_h not in ('', 'NA', 'qual_only', 'unknown'):
                    J[i, j] = float(val_h)
                else:
                    J[i, j] = 0.0
            else:
                J[i, j] = float(val_str)

    return J, signs


def draw_curved_arrow(ax, start, end, color, linewidth, linestyle='-',
                      connectionstyle='arc3,rad=0.15', alpha=0.7):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(
        posA=start, posB=end,
        arrowstyle='->,head_length=6,head_width=4',
        connectionstyle=connectionstyle,
        color=color, linewidth=linewidth, linestyle=linestyle,
        alpha=alpha, zorder=1,
    )
    ax.add_patch(arrow)


def main():
    args = parse_args()
    print("Generating Figure 1: 9-axis network schematic ...")

    csv_path = args.j_matrix or os.path.join(ROOT, 'data', 'J_matrix_compiled_9x9.csv')
    j_spec = JMatrixSpec.from_csv(csv_path)
    J, signs = load_9x9_J(csv_path)

    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    n = len(AXES_9)
    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Start from top, go clockwise
    angles = np.pi / 2 - angles
    radius = 3.5
    positions = np.column_stack([radius * np.cos(angles),
                                  radius * np.sin(angles)])

    node_radius = 0.45

    # Draw edges first (behind nodes)
    max_J = max(np.max(np.abs(J)), 0.01)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sign = signs.get((i, j), None)
            if sign is None:
                continue

            mag = abs(J[i, j])
            # Determine colour and style
            if sign == '+':
                color = '#e74c3c'  # red — pathological
                ls = '-'
            elif sign == '-':
                color = '#3498db'  # blue — protective
                ls = '-'
            else:
                color = '#95a5a6'  # grey — unknown
                ls = '--'

            # Width proportional to magnitude (min 0.5, max 3.0)
            if mag > 0:
                lw = 0.5 + 2.5 * (mag / max_J)
            else:
                lw = 0.5

            # Shorten arrow to stop at node boundary
            start = positions[j]
            end = positions[i]
            direction = end - start
            dist = np.linalg.norm(direction)
            if dist < 0.01:
                continue
            unit = direction / dist
            start_adj = start + unit * (node_radius + 0.05)
            end_adj = end - unit * (node_radius + 0.05)

            # Determine curve direction to avoid overlap with reverse edge
            rad = 0.18
            if j > i:
                rad = -rad

            draw_curved_arrow(ax, tuple(start_adj), tuple(end_adj),
                              color=color, linewidth=lw, linestyle=ls,
                              connectionstyle=f'arc3,rad={rad}',
                              alpha=0.65)

    # Draw nodes
    for i in range(n):
        circle = plt.Circle(positions[i], node_radius, color=NODE_COLORS[i],
                            ec='white', linewidth=2, zorder=5, alpha=0.9)
        ax.add_patch(circle)
        # Short label centred
        ax.text(positions[i][0], positions[i][1] + 0.02,
                AXES_9[i], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', zorder=6)
        # Full name outside the circle
        offset = 1.18
        outer = positions[i] * offset / radius * (radius + node_radius + 0.35)
        ax.text(outer[0], outer[1], AXES_FULL_NAMES[i],
                ha='center', va='center', fontsize=8, color='#2c3e50',
                zorder=6)

    # Legend
    legend_elements = [
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color='#e74c3c'),
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color='#3498db'),
        mpatches.FancyArrow(0, 0, 0.1, 0, width=0.02, color='#95a5a6'),
    ]
    ax.legend(
        [plt.Line2D([0], [0], color='#e74c3c', lw=2),
         plt.Line2D([0], [0], color='#3498db', lw=2),
         plt.Line2D([0], [0], color='#95a5a6', lw=2, linestyle='--')],
        ['Pathological (+)', 'Protective (\u2212)', 'Unknown / qualitative only'],
        loc='lower center', fontsize=9, framealpha=0.9,
        ncol=3, bbox_to_anchor=(0.5, -0.02),
    )

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.tight_layout()
    save_figure(fig, 'figure_network_schematic', OUTPUT_DIR)
    meta = {
        'j_matrix': j_spec.to_dict(),
        'script': 'run_figure_network_schematic.py',
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    with open(os.path.join(OUTPUT_DIR, 'figure_network_schematic_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    plt.close(fig)
    print("Done.")


if __name__ == '__main__':
    main()
