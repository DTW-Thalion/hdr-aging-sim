"""Publication-quality figure generation utilities."""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 1.5,
    })


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add a bold panel label (a, b, c, d) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')


def save_figure(fig, name, output_dir='outputs'):
    """Save figure as both PDF and PNG."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'{name}.pdf')
    png_path = os.path.join(output_dir, f'{name}.png')
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, format='png', dpi=300)
    print(f'Saved {pdf_path} and {png_path}')
