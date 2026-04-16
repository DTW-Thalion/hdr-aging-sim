#!/usr/bin/env python3
"""
Generate 6-axis expansion figures for InCHIANTI.
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})

C_5AX = "#2166ac"
C_4COR = "#d6604d"
C_4HR = "#999999"
C_CONC = "#4daf4a"
C_DISC = "#e41a1c"

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def figure_lambda_max_comparison():
    """Lambda_max: 5-axis vs 4-axis-cortdh vs 4-axis-hr."""
    print("  Fig: lambda_max comparison...")
    data = load_json("results/inchianti_6axis_results.json")

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    configs = [
        ("5axis_IMNFB", "5-axis (I,M,N,F,B)", C_5AX, "o-"),
        ("4axis_cortdh", "4-axis cortisol/DHEAS", C_4COR, "s--"),
        ("4axis_hr", "4-axis resting HR", C_4HR, "D:"),
    ]

    for cfg_name, label, color, fmt in configs:
        res = data.get(cfg_name, {})
        if res.get("skipped"):
            continue
        lmax = res.get("lambda_max", [])
        if not lmax:
            continue
        strata = [r["stratum"] for r in lmax]
        vals = [r["lambda_max"] for r in lmax]
        ci_lo = [r["lambda_max"] - r["ci_lower"] for r in lmax]
        ci_hi = [r["ci_upper"] - r["lambda_max"] for r in lmax]
        x = np.arange(len(strata))
        ax.errorbar(x, vals, yerr=[ci_lo, ci_hi], fmt=fmt, color=color,
                    capsize=3, markersize=5, linewidth=1.5, label=label)

    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(strata, rotation=30, ha="right")
    ax.set_xlabel("Age stratum")
    ax.set_ylabel("lambda_max (Gamma_change)")
    ax.set_title("Coupling tightening: axis configuration comparison")
    ax.legend(frameon=False)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_lambda_max_6axis.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def figure_lead_lag_5axis():
    """5x5 lead-lag heatmap for the 5-axis model."""
    print("  Fig: 5-axis lead-lag heatmap...")
    data = load_json("results/inchianti_6axis_results.json")
    res = data.get("5axis_IMNFB", {})
    ll = res.get("lead_lag", [])
    if not ll:
        print("    No 5-axis lead-lag data")
        return

    axes = ["I", "M", "N", "F", "B"]
    n = len(axes)
    beta_mat = np.full((n, n), np.nan)
    p_mat = np.full((n, n), np.nan)
    conc_mat = np.full((n, n), False)

    for r in ll:
        i = axes.index(r["to"])
        j = axes.index(r["from"])
        beta_mat[i, j] = r["beta"]
        p_mat[i, j] = r["p_value"] if r["p_value"] is not None else np.nan
        conc_mat[i, j] = r["concordant"]

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    vmax = np.nanmax(np.abs(beta_mat)) * 1.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(beta_mat, cmap="RdBu_r", norm=norm, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color="#f0f0f0", zorder=2))
                continue
            if np.isnan(beta_mat[i, j]):
                continue
            p = p_mat[i, j]
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            ax.text(j, i, f"{beta_mat[i,j]:.3f}{stars}", ha="center", va="center",
                    fontsize=7, fontweight="bold" if stars else "normal")
            color = C_CONC if conc_mat[i, j] else C_DISC
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=1.5,
                                       edgecolor=color, facecolor="none", zorder=3))

    ax.set_xticks(range(n))
    ax.set_xticklabels(axes)
    ax.set_yticks(range(n))
    ax.set_yticklabels(axes)
    ax.set_xlabel("Source axis (t)")
    ax.set_ylabel("Target axis (t+1)")

    conc_info = res.get("concordance", {})
    ax.set_title(f"5-axis cross-lagged beta ({conc_info.get('n_concordant',0)}/{conc_info.get('n_tested',0)} concordant)")

    plt.colorbar(im, ax=ax, shrink=0.8, label="beta")
    conc_p = mpatches.Patch(edgecolor=C_CONC, facecolor="none", linewidth=1.5, label="Concordant")
    disc_p = mpatches.Patch(edgecolor=C_DISC, facecolor="none", linewidth=1.5, label="Discordant")
    ax.legend(handles=[conc_p, disc_p], loc="upper right", bbox_to_anchor=(1.0, -0.08),
              ncol=2, frameon=False, fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_lead_lag_6axis.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def figure_n_axis_sensitivity():
    """Compare cortisol/DHEAS vs resting HR N-axis."""
    print("  Fig: N-axis sensitivity...")
    data = load_json("results/inchianti_6axis_results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Panel a: lambda_max comparison
    for cfg, label, color, fmt in [
        ("4axis_cortdh", "Cortisol/DHEAS", C_4COR, "o-"),
        ("4axis_hr", "Resting HR", C_4HR, "s--"),
    ]:
        res = data.get(cfg, {})
        lmax = res.get("lambda_max", [])
        strata = [r["stratum"] for r in lmax]
        vals = [r["lambda_max"] for r in lmax]
        x = np.arange(len(strata))
        ax1.plot(x, vals, fmt, color=color, markersize=5, linewidth=1.5, label=label)

    ax1.set_xticks(range(len(strata)))
    ax1.set_xticklabels(strata, rotation=30, ha="right")
    ax1.set_ylabel("lambda_max")
    ax1.set_title("(a) lambda_max trajectory")
    ax1.legend(frameon=False)

    # Panel b: concordance comparison
    configs = ["4axis_cortdh", "4axis_hr"]
    labels = ["Cortisol/DHEAS", "Resting HR"]
    concs = []
    for cfg in configs:
        c = data.get(cfg, {}).get("concordance", {})
        concs.append(c.get("n_concordant", 0) / max(c.get("n_tested", 1), 1))

    ax2.bar(range(2), [c*100 for c in concs], color=[C_4COR, C_4HR], alpha=0.7,
            edgecolor="black", linewidth=0.5)
    ax2.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Concordance (%)")
    ax2.set_title("(b) Lead-lag concordance")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_n_axis_sensitivity.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def figure_survival():
    """Cox model comparison."""
    print("  Fig: Survival analysis...")
    try:
        data = load_json("results/inchianti_survival_analysis.json")
    except FileNotFoundError:
        print("    Survival results not found")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    models = ["M1_age_sex", "M2_biomarkers", "M3_swds", "M4_frailty", "M5_full"]
    model_labels = ["M1: Age+Sex", "M2: +Biomarkers", "M3: SWDS-G", "M4: +Frailty", "M5: Full"]

    subgroups = [("age65+", "Age 65+", "#2166ac"), ("med_naive", "Med-naive", "#d6604d")]
    x = np.arange(len(models))
    w = 0.35

    for i, (sg, label, color) in enumerate(subgroups):
        sg_data = data.get(sg, {})
        vals = []
        for m in models:
            v = sg_data.get(m)
            if isinstance(v, dict):
                vals.append(v["C"])
            else:
                vals.append(np.nan)
        ax.bar(x + i * w - w/2, vals, w, label=label, color=color, alpha=0.7,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=30, ha="right")
    ax.set_ylabel("Harrell's C-index")
    ax.set_title("Cox mortality prediction models")
    ax.legend(frameon=False)
    ax.set_ylim(0.6, 0.9)

    # Add delta_C annotation
    for sg, _, _ in subgroups:
        dc = data.get(sg, {}).get("delta_C_M5_M4")
        if dc is not None:
            ax.text(len(models)-1, 0.62, f"dC(M5-M4)={dc:+.3f}", fontsize=7, ha="center")
            break

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_survival.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def main():
    print("=" * 60)
    print("InCHIANTI: 6-Axis Figure Generation")
    print("=" * 60)

    figure_lambda_max_comparison()
    figure_lead_lag_5axis()
    figure_n_axis_sensitivity()
    figure_survival()

    print("\nAll 6-axis figures generated.")


if __name__ == "__main__":
    main()
