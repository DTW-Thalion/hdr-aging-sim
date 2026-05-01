#!/usr/bin/env python3
"""
Step 7 -- Generate publication-quality figures.

Figures
-------
  figure1_recovery_examples.pdf   -- representative trajectories per age decade
  figure2_tau_vs_age.pdf          -- primary result: tau vs age, per axis
  figure3_cross_axis_coupling.pdf -- peak->tau heatmap (J-matrix concordance)
  figure4_outcome_prediction.pdf  -- AUC bars + delta C
  figure5_recovery_ordering.pdf   -- t_50 by axis, by age stratum
  figure6_model_selection.pdf     -- exp vs stretched vs linear, by stratum
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import permutations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from hdr_core import (
    AXES,
    J_SIGNS,
    age_stratum,
    banner,
    exp_recovery,
    load_config,
)

plt.rcParams["pdf.fonttype"] = 42  # editable text in Illustrator
plt.rcParams["font.family"] = "DejaVu Sans"


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure 1 -- recovery trajectory examples

def figure1_examples(eps: pd.DataFrame, fits: pd.DataFrame, cfg: dict, out_path: str):
    strata = cfg["analysis"]["age_strata"]
    primary_per_axis = {ax: cfg["axes"][ax]["primary"] for ax in AXES}

    eps_p = eps[eps["is_primary"]].copy()
    eps_p["stratum"] = eps_p["age"].apply(lambda a: age_stratum(a, strata))
    fig, axarr = plt.subplots(len(AXES), len(strata),
                               figsize=(2.5 * len(strata), 2.2 * len(AXES)),
                               sharex=False)
    if len(AXES) == 1:
        axarr = np.array([axarr])
    rng = np.random.default_rng(0)

    for ai, axis in enumerate(AXES):
        bm = primary_per_axis[axis]
        for si, (lo, hi) in enumerate(strata):
            stratum_key = f"{lo}-{hi}"
            ax_plot = axarr[ai, si]
            cohort = eps_p[(eps_p["axis"] == axis) &
                           (eps_p["biomarker"] == bm) &
                           (eps_p["stratum"] == stratum_key)]
            if len(cohort) == 0:
                ax_plot.set_axis_off()
                continue
            picks = cohort.sample(n=min(3, len(cohort)), random_state=int(rng.integers(0, 1e9)))
            for _, ep in picks.iterrows():
                t = np.array(ep["recovery_t_hours"])
                y = np.array(ep["recovery_y"])
                t_full = np.concatenate(([0], t))
                y_full = np.concatenate(([ep["peak_value"]], y))
                ax_plot.plot(t_full / 24, y_full, "o-", alpha=0.6, ms=3)
                fit_row = fits[(fits["hadm_id"] == ep["hadm_id"]) & (fits["axis"] == axis)]
                if len(fit_row) and np.isfinite(fit_row.iloc[0].get("tau_hours", np.nan)):
                    fr = fit_row.iloc[0]
                    tt = np.linspace(0, t_full.max(), 100)
                    yy = exp_recovery(tt, fr["exp_y_baseline"], fr["exp_y_peak"], fr["tau_hours"])
                    ax_plot.plot(tt / 24, yy, "-", alpha=0.4, lw=1)
            if ai == 0:
                ax_plot.set_title(f"age {stratum_key}", fontsize=9)
            if si == 0:
                ax_plot.set_ylabel(f"{axis} ({bm})", fontsize=9)
            if ai == len(AXES) - 1:
                ax_plot.set_xlabel("days from peak", fontsize=8)
            ax_plot.tick_params(labelsize=7)
    fig.suptitle("Figure 1 -- Recovery trajectories (representative)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2 -- tau vs age

def figure2_tau_vs_age(fits: pd.DataFrame, summary: dict, cfg: dict, out_path: str):
    strata = cfg["analysis"]["age_strata"]
    qc = fits[fits["tau_passes_qc"] & fits["is_primary"]]
    fig, axarr = plt.subplots(1, len(AXES), figsize=(3.0 * len(AXES), 3.0), sharey=False)

    for i, axis in enumerate(AXES):
        ax_plot = axarr[i]
        a = qc[qc["axis"] == axis]
        if len(a) < 30:
            ax_plot.text(0.5, 0.5, f"n={len(a)} (insufficient)",
                         transform=ax_plot.transAxes, ha="center")
            ax_plot.set_title(axis); continue

        ax_plot.scatter(a["age"], np.log(a["tau_hours"]), s=6, alpha=0.2, color="steelblue")

        # Per-stratum medians
        med_x, med_lo, med_hi = [], [], []
        for lo, hi in strata:
            sub = a[a["age"].between(lo, hi)]
            if len(sub) >= 5:
                med_x.append((lo + hi) / 2)
                vals = np.log(sub["tau_hours"].values)
                med_lo.append(np.percentile(vals, 25))
                med_hi.append(np.percentile(vals, 75))
        if med_x:
            ax_plot.plot(med_x, [np.log(np.median(a[a["age"].between(*s)]["tau_hours"]))
                                  for s in strata if len(a[a["age"].between(*s)]) >= 5],
                          "ko-", label="stratum median")

        # Regression line
        beta = summary.get("by_axis", {}).get(axis, {}).get("regression", {}).get("beta_log_tau_per_year", np.nan)
        intercept = np.log(np.median(a["tau_hours"].values)) - beta * np.median(a["age"].values)
        xs = np.array([a["age"].min(), a["age"].max()])
        ax_plot.plot(xs, intercept + beta * xs, "r--", lw=1.2, label=f"beta={beta:.4f}")

        pval = summary.get("by_axis", {}).get(axis, {}).get("regression", {}).get("pvalue_age", np.nan)
        fold = summary.get("by_axis", {}).get(axis, {}).get("regression", {}).get("tau_fold_change_per_decade", np.nan)
        ax_plot.set_title(f"{axis}\nfold/decade={fold:.2f}x, p={pval:.2g}", fontsize=9)
        ax_plot.set_xlabel("age (years)")
        if i == 0:
            ax_plot.set_ylabel("log(tau, hours)")
        ax_plot.tick_params(labelsize=8)
    fig.suptitle("Figure 2 -- Recovery timescale tau vs age", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3 -- cross-axis coupling heatmap

def figure3_coupling(coupling: dict, out_path: str):
    pairs = coupling.get("peak_to_tau", {}).get("pairs", [])
    if not pairs:
        print("  (no coupling pairs; skipping figure 3)")
        return
    mat = np.full((len(AXES), len(AXES)), np.nan)
    sig = np.zeros_like(mat, dtype=bool)
    for r in pairs:
        i = AXES.index(r["from"])
        j = AXES.index(r["to"])
        mat[j, i] = r["beta_peak_from"]
        sig[j, i] = (r.get("pvalue", 1.0) < 0.05)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 1e-4)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for j in range(len(AXES)):
        for i in range(len(AXES)):
            if np.isnan(mat[j, i]):
                txt = "-"
            else:
                txt = f"{mat[j, i]:+.3f}{'*' if sig[j, i] else ''}"
            ax.text(i, j, txt, ha="center", va="center", fontsize=7,
                    color="black" if abs(mat[j, i]) < vmax * 0.5 else "white")
    ax.set_xticks(range(len(AXES)))
    ax.set_yticks(range(len(AXES)))
    ax.set_xticklabels([f"peak {a}" for a in AXES])
    ax.set_yticklabels([f"log tau {a}" for a in AXES])
    ax.set_title(f"Figure 3 -- peak_i -> tau_j coupling\n"
                 f"concordance: {coupling.get('peak_to_tau', {}).get('n_concordant')}/"
                 f"{coupling.get('peak_to_tau', {}).get('n_pairs')}, "
                 f"p={coupling.get('peak_to_tau', {}).get('concordance_p_one_sided', float('nan')):.3g}",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="beta")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 4 -- outcome prediction

def figure4_outcomes(outcomes: dict, out_path: str):
    models = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    aucs = [outcomes["mortality_models"][m].get("auc") for m in models]
    cis = [outcomes["mortality_models"][m].get("auc_ci95", [None, None]) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    xs = np.arange(len(models))
    valid = [(i, m, auc, ci) for i, (m, auc, ci) in enumerate(zip(models, aucs, cis)) if auc is not None]
    if valid:
        idxs, ms, vs, cis_v = zip(*valid)
        ax1.bar(idxs, vs, color="steelblue", alpha=0.8)
        for i, ci in zip(idxs, cis_v):
            if ci and ci[0] is not None:
                ax1.errorbar(i, aucs[i], yerr=[[aucs[i] - ci[0]], [ci[1] - aucs[i]]],
                              color="black", capsize=3)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_ylabel("AUC (in-hospital mortality)")
    ax1.set_title("Mortality prediction by model")

    # Delta C panel
    deltas = outcomes.get("delta_c", {})
    keys = list(deltas.keys())
    deltas_v = [deltas[k].get("delta") for k in keys]
    cis_d = [deltas[k].get("ci95", [None, None]) for k in keys]
    ax2.axhline(0, color="grey", lw=0.5)
    for i, (k, v, ci) in enumerate(zip(keys, deltas_v, cis_d)):
        if v is None:
            continue
        ax2.plot(i, v, "ko")
        if ci and ci[0] is not None:
            ax2.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color="black", capsize=3)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, rotation=20, ha="right")
    ax2.set_ylabel("delta AUC")
    ax2.set_title("Delta C with bootstrap 95% CI")

    fig.suptitle("Figure 4 -- Outcome prediction", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 5 -- recovery ordering

def figure5_ordering(fits: pd.DataFrame, cfg: dict, out_path: str):
    strata = cfg["analysis"]["age_strata"]
    qc = fits[fits["tau_passes_qc"] & fits["is_primary"]].copy()
    qc["t50_h"] = qc["tau_hours"] * np.log(2)
    qc["stratum"] = qc["age"].apply(lambda a: age_stratum(a, strata))

    fig, ax = plt.subplots(figsize=(7, 4))
    positions = []
    labels = []
    data = []
    for si, (lo, hi) in enumerate(strata):
        stratum = f"{lo}-{hi}"
        for ai, axis in enumerate(AXES):
            sub = qc[(qc["stratum"] == stratum) & (qc["axis"] == axis)]
            if len(sub) >= 5:
                data.append(np.log(sub["t50_h"].values))
                positions.append(si * (len(AXES) + 1) + ai)
                labels.append(axis)
    if not data:
        print("  (insufficient data; skipping figure 5)")
        return
    parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.8)
    for p in parts["bodies"]:
        p.set_alpha(0.5)
    ax.set_xticks([si * (len(AXES) + 1) + (len(AXES) - 1) / 2 for si in range(len(strata))])
    ax.set_xticklabels([f"{lo}-{hi}" for lo, hi in strata])
    ax.set_ylabel("log(t_50, hours)")
    ax.set_xlabel("age stratum")
    ax.set_title("Figure 5 -- Time to 50% recovery, by axis & age stratum")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 6 -- model selection

def figure6_model_selection(fits: pd.DataFrame, cfg: dict, out_path: str):
    strata = cfg["analysis"]["age_strata"]
    qc = fits.copy()
    qc["stratum"] = qc["age"].apply(lambda a: age_stratum(a, strata))
    fractions = []
    labels = []
    for lo, hi in strata:
        stratum = f"{lo}-{hi}"
        sub = qc[qc["stratum"] == stratum]
        if len(sub) < 10:
            continue
        labels.append(stratum)
        counts = sub["best_model"].value_counts(normalize=True)
        fractions.append({m: float(counts.get(m, 0)) for m in ("exponential", "stretched", "linear")})
    if not fractions:
        print("  (insufficient data; skipping figure 6)")
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    xs = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    colors = {"exponential": "#5b9bd5", "stretched": "#ed7d31", "linear": "#a5a5a5"}
    for m in ("exponential", "stretched", "linear"):
        vals = np.array([f[m] for f in fractions])
        ax.bar(xs, vals, bottom=bottom, label=m, color=colors[m], alpha=0.85)
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("fraction of episodes")
    ax.set_title("Figure 6 -- Best recovery model by age stratum")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    banner("Step 7: figures")

    eps_path = os.path.join(out_dir, "recovery_episodes.parquet")
    fits_path = os.path.join(out_dir, "recovery_fits.parquet")
    summary_path = os.path.join(out_dir, "tau_vs_age.json")
    coupling_path = os.path.join(out_dir, "cross_axis_coupling.json")
    outcomes_path = os.path.join(out_dir, "outcome_prediction.json")

    if not os.path.exists(eps_path) or not os.path.exists(fits_path):
        print("ERROR: need recovery_episodes.parquet and recovery_fits.parquet (steps 2 & 3).")
        return 1

    eps = pd.read_parquet(eps_path)
    fits = pd.read_parquet(fits_path)
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    coupling = json.load(open(coupling_path)) if os.path.exists(coupling_path) else {}
    outcomes = json.load(open(outcomes_path)) if os.path.exists(outcomes_path) else {}

    figure1_examples(eps, fits, cfg, os.path.join(out_dir, "figure1_recovery_examples.pdf"))
    figure2_tau_vs_age(fits, summary, cfg, os.path.join(out_dir, "figure2_tau_vs_age.pdf"))
    if coupling:
        figure3_coupling(coupling, os.path.join(out_dir, "figure3_cross_axis_coupling.pdf"))
    if outcomes:
        figure4_outcomes(outcomes, os.path.join(out_dir, "figure4_outcome_prediction.pdf"))
    figure5_ordering(fits, cfg, os.path.join(out_dir, "figure5_recovery_ordering.pdf"))
    figure6_model_selection(fits, cfg, os.path.join(out_dir, "figure6_model_selection.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
