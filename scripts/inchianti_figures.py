#!/usr/bin/env python3
"""
Generate all InCHIANTI publication-quality figures.

Produces 5 PDF figures matching manuscript style (10pt, ~6.5 inch width).
Reads pre-computed results from results/inchianti_*.json and *.csv.
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

# Manuscript style
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})

# Nature-style muted palette
C_FULL = "#2166ac"       # blue
C_NAIVE = "#d6604d"      # muted red
C_ELSA = "#999999"       # grey
C_MED = "#4393c3"        # medicated blue
C_UNMED = "#d6604d"      # unmedicated red
C_POS = "#b2182b"        # positive coupling (red)
C_NEG = "#2166ac"        # negative coupling (blue)
C_CONCORDANT = "#4daf4a" # green border
C_DISCORDANT = "#e41a1c" # red border

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ELSA reference lambda_max values (from manuscript Table, 3-axis I,M,F)
ELSA_STRATA = ["50-59", "60-69", "70-79", "80+"]
ELSA_LMAX = [0.72, 0.70, 0.78, 0.87]  # ELSA 3-axis change-covariance lambda_max


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ==================================================================
# Figure A: Lambda_max trajectory
# ==================================================================
def figure_a_lambda_max():
    print("  Generating Figure A: lambda_max trajectory...")
    df = pd.read_csv("results/inchianti_lambda_max_by_age.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))

    # Panel (a): Change-covariance
    change_full = df[df["type"] == "change_covariance"]
    change_naive = df[df["type"] == "change_naive"]

    strata_labels = change_full["age_stratum"].values
    x = np.arange(len(strata_labels))

    # Full sample
    ax1.errorbar(x, change_full["lambda_max"],
                 yerr=[change_full["lambda_max"] - change_full["ci_lower"],
                       change_full["ci_upper"] - change_full["lambda_max"]],
                 fmt="o-", color=C_FULL, capsize=3, markersize=5, linewidth=1.5,
                 label="InCHIANTI full (4-axis)")

    # Medication-naive
    if len(change_naive) > 0:
        x_naive = np.arange(len(change_naive))
        ax1.errorbar(x_naive + 0.15, change_naive["lambda_max"],
                     yerr=[change_naive["lambda_max"] - change_naive["ci_lower"],
                           change_naive["ci_upper"] - change_naive["lambda_max"]],
                     fmt="s--", color=C_NAIVE, capsize=3, markersize=4, linewidth=1,
                     label="InCHIANTI med-naive")

    # ELSA reference (grey markers) -- align with 50-59 through 80+
    elsa_x = [i for i, s in enumerate(strata_labels) if s in ELSA_STRATA]
    if elsa_x:
        ax1.scatter(elsa_x, ELSA_LMAX[:len(elsa_x)], marker="D", color=C_ELSA,
                    s=30, zorder=5, label="ELSA (3-axis)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(strata_labels, rotation=30, ha="right")
    ax1.set_xlabel("Age stratum")
    ax1.set_ylabel("lambda_max (Gamma_change)")
    ax1.set_title("(a) Change-covariance")
    ax1.legend(loc="upper left", frameon=False)

    # Panel (b): Cross-sectional
    cross = df[df["type"] == "cross_sectional"]
    ax2.errorbar(np.arange(len(cross)), cross["lambda_max"],
                 yerr=[cross["lambda_max"] - cross["ci_lower"],
                       cross["ci_upper"] - cross["lambda_max"]],
                 fmt="o-", color=C_FULL, capsize=3, markersize=5, linewidth=1.5)
    ax2.set_xticks(np.arange(len(cross)))
    ax2.set_xticklabels(cross["age_stratum"].values, rotation=30, ha="right")
    ax2.set_xlabel("Age stratum")
    ax2.set_ylabel("lambda_max (Gamma_cross)")
    ax2.set_title("(b) Cross-sectional")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_lambda_max.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


# ==================================================================
# Figure B: Lead-lag matrix
# ==================================================================
def figure_b_lead_lag():
    print("  Generating Figure B: lead-lag heatmap...")
    df = pd.read_csv("results/inchianti_lead_lag_matrix.csv")
    summary = load_json("results/inchianti_lead_lag_summary.json")

    axis_names = ["I", "M", "N", "F"]
    n = len(axis_names)
    beta_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)
    concordance_matrix = np.full((n, n), False)

    for _, row in df.iterrows():
        from_ax = row["from_axis"]
        to_ax = row["to_axis"]
        i = axis_names.index(to_ax)
        j = axis_names.index(from_ax)
        beta_matrix[i, j] = row["beta"]
        p_matrix[i, j] = row["p_value"]
        concordance_matrix[i, j] = row["concordant"]

    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    # Heatmap
    vmax = np.nanmax(np.abs(beta_matrix)) * 1.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(beta_matrix, cmap="RdBu_r", norm=norm, aspect="equal")

    # Annotations
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=True, color="#f0f0f0", zorder=2))
                continue
            if np.isnan(beta_matrix[i, j]):
                continue

            # Significance stars
            p = p_matrix[i, j]
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"

            txt = f"{beta_matrix[i, j]:.3f}{stars}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if stars else "normal")

            # Border: green if concordant, red if discordant
            color = C_CONCORDANT if concordance_matrix[i, j] else C_DISCORDANT
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=1.5,
                                 edgecolor=color, facecolor="none", zorder=3)
            ax.add_patch(rect)

    ax.set_xticks(range(n))
    ax.set_xticklabels(axis_names)
    ax.set_yticks(range(n))
    ax.set_yticklabels(axis_names)
    ax.set_xlabel("Source axis (t)")
    ax.set_ylabel("Target axis (t+1)")
    ax.set_title("Cross-lagged beta coefficients")

    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("beta")

    # Legend
    conc_patch = mpatches.Patch(edgecolor=C_CONCORDANT, facecolor="none",
                                linewidth=1.5, label="Concordant with J")
    disc_patch = mpatches.Patch(edgecolor=C_DISCORDANT, facecolor="none",
                                linewidth=1.5, label="Discordant")
    ax.legend(handles=[conc_patch, disc_patch], loc="upper right",
              bbox_to_anchor=(1.0, -0.1), ncol=2, frameon=False, fontsize=7.5)

    n_conc = summary.get("n_concordant", 0)
    n_test = summary.get("n_tested", 0)
    p_binom = summary.get("binomial_p", "N/A")
    fig.text(0.5, -0.02, f"{n_conc}/{n_test} concordant (p = {p_binom:.3f})",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_lead_lag.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


# ==================================================================
# Figure C: Pi trajectory
# ==================================================================
def figure_c_pi():
    print("  Generating Figure C: Pi trajectory...")
    df = pd.read_csv("results/inchianti_pi_trajectory.csv")
    summary = load_json("results/inchianti_pi_trajectory.json")

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    x = np.arange(len(df))
    ax.errorbar(x, df["Pi_norm"],
                yerr=[df["Pi_norm"] - df["Pi_ci_lower"],
                      df["Pi_ci_upper"] - df["Pi_norm"]],
                fmt="o-", color=C_FULL, capsize=4, markersize=6, linewidth=1.5)

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(df) - 0.5, 1.05, "Pi = 1 (proportional)", fontsize=7.5,
            ha="right", color="#888888")

    slope = summary.get("pi_slope_per_year")
    if slope is not None:
        ax.text(0.05, 0.95, f"Slope = {slope:.4f}/yr",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.set_xticks(x)
    ax.set_xticklabels(df["age_stratum"].values, rotation=30, ha="right")
    ax.set_xlabel("Age stratum")
    ax.set_ylabel("Pi = C_norm / V_norm")
    ax.set_title("Proportional co-degradation index")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_pi.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


# ==================================================================
# Figure D: Medication dose-response (4-panel)
# ==================================================================
def figure_d_medication():
    print("  Generating Figure D: medication dose-response...")

    # Load raw medication results
    raw = pd.read_csv("results/inchianti_med_dose_response.csv")

    # Load refined results
    try:
        refined = load_json("results/inchianti_med_refined_results.json")
    except FileNotFoundError:
        print("    WARNING: refined results not found, using partial figure")
        refined = {}

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.5))

    # Panel (a): Raw lambda_max by medication class count
    ax = axes[0, 0]
    med_groups = raw[raw["group"].str.startswith("med_")]
    labels = [r["group"].replace("med_", "") for _, r in med_groups.iterrows()]
    vals = med_groups["lambda_max"].values
    ci_lo = med_groups["ci_lower"].values
    ci_hi = med_groups["ci_upper"].values
    x = np.arange(len(labels))
    ax.bar(x, vals, color=[C_UNMED if l == "0" else C_MED for l in labels],
           alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, vals, yerr=[vals - ci_lo, ci_hi - vals],
                fmt="none", color="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Medication classes")
    ax.set_ylabel("lambda_max (Gamma_change)")
    ax.set_title("(a) Raw (confounded by age)")

    # Panel (b): Age-stratified comparison
    ax = axes[0, 1]
    strat = refined.get("1a_age_stratified", [])
    if strat:
        decades = sorted(set(r["decade"] for r in strat))
        x = np.arange(len(decades))
        w = 0.35
        unmed = [next((r for r in strat if r["decade"] == d and r["group"] == "unmedicated"), None)
                 for d in decades]
        med = [next((r for r in strat if r["decade"] == d and r["group"] == "medicated"), None)
               for d in decades]

        for i, (u, m) in enumerate(zip(unmed, med)):
            if u and np.isfinite(u["lambda_max"]):
                ax.bar(i - w/2, u["lambda_max"], w, color=C_UNMED, alpha=0.7,
                       edgecolor="black", linewidth=0.5)
                ax.errorbar(i - w/2, u["lambda_max"],
                            yerr=[[u["lambda_max"] - u["ci_lower"]],
                                  [u["ci_upper"] - u["lambda_max"]]],
                            fmt="none", color="black", capsize=3)
            if m and np.isfinite(m["lambda_max"]):
                ax.bar(i + w/2, m["lambda_max"], w, color=C_MED, alpha=0.7,
                       edgecolor="black", linewidth=0.5)
                ax.errorbar(i + w/2, m["lambda_max"],
                            yerr=[[m["lambda_max"] - m["ci_lower"]],
                                  [m["ci_upper"] - m["lambda_max"]]],
                            fmt="none", color="black", capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(decades, rotation=30, ha="right")
        ax.legend([mpatches.Patch(color=C_UNMED, alpha=0.7),
                   mpatches.Patch(color=C_MED, alpha=0.7)],
                  ["Unmedicated", "Medicated"], frameon=False, fontsize=8)
    ax.set_xlabel("Age decade")
    ax.set_ylabel("lambda_max (Gamma_change)")
    ax.set_title("(b) Age-stratified")

    # Panel (c): SWDS-Gamma regression coefficients
    ax = axes[1, 0]
    reg = refined.get("1b_swds_regression", {})
    pred_names = ["n_med_classes", "n_comorbidities", "age", "female"]
    pred_labels = ["N med classes", "N comorbidities", "Age", "Female"]
    if reg:
        betas = [reg[n]["beta"] for n in pred_names]
        ses = [reg[n]["se"] for n in pred_names]
        ps = [reg[n]["p"] for n in pred_names]
        y_pos = np.arange(len(pred_names))

        colors = [C_POS if b > 0 else C_NEG for b in betas]
        ax.barh(y_pos, betas, xerr=[1.96 * s for s in ses], color=colors,
                alpha=0.7, edgecolor="black", linewidth=0.5, capsize=3)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pred_labels)

        # Add significance stars
        for i, (b, p) in enumerate(zip(betas, ps)):
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if star:
                ax.text(b + 0.01 * np.sign(b), i, star, va="center", fontsize=8)
    ax.set_xlabel("beta (SWDS-Gamma regression)")
    ax.set_title("(c) SWDS-Gamma predictors")

    # Panel (d): Within-hypertension matched comparison
    ax = axes[1, 1]
    htn = refined.get("1c_htn_matched", {})
    if "treated_lambda_max" in htn:
        bars = [htn["untreated_lambda_max"], htn["treated_lambda_max"]]
        cis = [htn["untreated_ci"], htn["treated_ci"]]
        colors_htn = [C_UNMED, C_MED]
        labels_htn = [f"Untreated\n(N~{htn['n_matched']})",
                      f"Treated\n(N~{htn['n_matched']})"]
        x = [0, 1]
        ax.bar(x, bars, color=colors_htn, alpha=0.7, edgecolor="black", linewidth=0.5)
        for i in range(2):
            ax.errorbar(x[i], bars[i],
                        yerr=[[bars[i] - cis[i][0]], [cis[i][1] - bars[i]]],
                        fmt="none", color="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_htn)
        ax.set_title(f"(d) HTN: age-matched (N={htn['n_matched']})")
    else:
        ax.text(0.5, 0.5, "Insufficient data\nfor age-matching",
                transform=ax.transAxes, ha="center", va="center", fontsize=9)
        ax.set_title("(d) HTN: age-matched")
    ax.set_ylabel("lambda_max (Gamma_change)")

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_medication.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


# ==================================================================
# Figure E: InCHIANTI vs ELSA comparison
# ==================================================================
def figure_e_comparison():
    print("  Generating Figure E: InCHIANTI vs ELSA comparison...")
    df = pd.read_csv("results/inchianti_lambda_max_by_age.csv")
    change = df[df["type"] == "change_covariance"]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    # InCHIANTI
    inchi_strata = change["age_stratum"].values
    inchi_lmax = change["lambda_max"].values
    x_inchi = np.arange(len(inchi_strata))

    ax.errorbar(x_inchi, inchi_lmax,
                yerr=[inchi_lmax - change["ci_lower"].values,
                      change["ci_upper"].values - inchi_lmax],
                fmt="o-", color=C_FULL, capsize=4, markersize=6, linewidth=1.5,
                label="InCHIANTI (4-axis: IL-6, HOMA-IR, HR, SPPB)")

    # ELSA -- map to matching InCHIANTI strata positions
    elsa_positions = []
    for es in ELSA_STRATA:
        for i, ins in enumerate(inchi_strata):
            if ins == es:
                elsa_positions.append(i)
                break

    if elsa_positions:
        ax.plot(elsa_positions, ELSA_LMAX[:len(elsa_positions)], "D--",
                color=C_ELSA, markersize=6, linewidth=1.5,
                label="ELSA (3-axis: CRP, HbA1c, grip)")

    ax.set_xticks(x_inchi)
    ax.set_xticklabels(inchi_strata, rotation=30, ha="right")
    ax.set_xlabel("Age stratum")
    ax.set_ylabel("lambda_max (Gamma_change)")
    ax.set_title("Coupling tightening across cohorts")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Add note about scale difference
    ax.text(0.98, 0.15,
            "Note: absolute values not\ndirectly comparable (different\naxes, units, standardisation)",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            style="italic", color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))

    plt.tight_layout()
    path = os.path.join(OUTDIR, "figure_inchianti_vs_elsa.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"    Saved: {path}")


def main():
    print("=" * 60)
    print("InCHIANTI Figure Generation")
    print("=" * 60)

    figure_a_lambda_max()
    figure_b_lead_lag()
    figure_c_pi()
    figure_d_medication()
    figure_e_comparison()

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
