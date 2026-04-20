"""
Test E: Bone-axis coupling (5-axis extension with PTH).

HDR prediction:
  - F→B: higher SPPB → lower subsequent PTH (mechanical loading protects bone)
  - I→B: higher IL-6 → higher subsequent PTH (inflammatory bone resorption)
  - Bisphosphonate users (ATC M05 / FX1_M5) should show attenuated bone-axis
    coupling (if N is large enough).

Uses waves 0-3 only (PTH measured there).
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    load_panel_with_deltas, cross_lagged_regression,
    _build_lag_pairs, _fit_ols, assign_tertiles, format_ci, bonferroni,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def run():
    panel, ref = load_panel_with_deltas(waves=[0, 1, 2, 3], with_pth=True)
    results = {
        "prediction": ("F→B protective (β<0), I→B inflammatory (β>0); "
                       "bisphosphonates should attenuate if N large enough"),
        "pth_ref": ref.get("B"),
    }

    # Overall F→B and I→B
    for src in ["F", "I"]:
        r = cross_lagged_regression(panel, src, "B")
        results[f"{src}_to_B_overall"] = r.get("overall", {})

    # F→B by SPPB tertile
    pan = panel.copy()
    pan["sppb_tert"] = np.nan
    for w, sub in pan.groupby("wave"):
        pan.loc[sub.index, "sppb_tert"] = assign_tertiles(sub["sppb"]).values
    r_fb_tert = cross_lagged_regression(pan, "F", "B",
                                         group_col="sppb_tert")
    results["F_to_B_by_sppb_tertile"] = r_fb_tert

    # Bisphosphonate comparison (if N adequate)
    bisphos_res = {}
    for src in ["F", "I"]:
        pairs = _build_lag_pairs(panel, src, "B", group_col="med_bisphos")
        n_users = int((pairs["group"] == 1).sum())
        n_non = int((pairs["group"] == 0).sum())
        bisphos_res[f"{src}_to_B"] = {
            "n_bisphos_pairs": n_users,
            "n_nonuser_pairs": n_non,
        }
        if n_users >= 30 and n_non >= 30:
            r = cross_lagged_regression(panel, src, "B",
                                        group_col="med_bisphos")
            bisphos_res[f"{src}_to_B"].update(r)
        else:
            bisphos_res[f"{src}_to_B"]["skipped"] = (
                f"N_bisphos_pairs={n_users} < 30")
    results["bisphosphonate"] = bisphos_res

    # Bonferroni across two primary tests (F→B overall, I→B overall)
    ps = [results[f"{s}_to_B_overall"].get("p", np.nan) for s in ["F", "I"]]
    results["bonferroni_primary"] = bonferroni([p for p in ps if not np.isnan(p)])

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "test_e_bone.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _figure(results)
    _print_summary(results)
    return results


def _figure(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: overall F→B and I→B with 95% CI
    ax = axes[0]
    labels = ["F→B\n(protective?)", "I→B\n(inflammatory?)"]
    betas, ses = [], []
    for src in ["F", "I"]:
        o = results[f"{src}_to_B_overall"]
        betas.append(o.get("beta", np.nan))
        ses.append(1.96 * o.get("se", 0))
    colors = ["#2E7D32", "#C62828"]
    ax.bar(labels, betas, yerr=ses, capsize=5, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(r"$\beta$ (95% CI)")
    ax.set_title("(a) Overall bone-axis coupling")
    ax.grid(axis="y", alpha=0.3)
    for i, src in enumerate(["F", "I"]):
        o = results[f"{src}_to_B_overall"]
        p = o.get("p", np.nan)
        n = o.get("n", 0)
        ax.text(i, ax.get_ylim()[1] * 0.92,
                f"p={p:.3f}\nN={n}", ha="center", fontsize=8,
                fontstyle="italic")

    # Panel B: F→B by SPPB tertile
    ax = axes[1]
    r = results["F_to_B_by_sppb_tertile"]
    tlabels = ["Low", "Mid", "High"]
    betas, ses, ns = [], [], []
    for t in [0.0, 1.0, 2.0]:
        g = r.get("groups", {}).get(str(t), {})
        if "beta" in g:
            betas.append(g["beta"])
            ses.append(1.96 * g["se"])
            ns.append(g["n"])
        else:
            betas.append(np.nan); ses.append(0); ns.append(0)
    ax.bar(tlabels, betas, yerr=ses, capsize=5,
           color="#2E7D32", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("SPPB tertile")
    ax.set_ylabel(r"$\beta_{F \rightarrow B}$ (95% CI)")
    ax.set_title("(b) F→B by SPPB tertile")
    ax.grid(axis="y", alpha=0.3)
    for i, n in enumerate(ns):
        ax.text(i, ax.get_ylim()[1] * 0.92, f"N={n}", ha="center",
                fontsize=8, fontstyle="italic")

    fig.suptitle("Test E — Bone-axis coupling (InCHIANTI, waves 0-3)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_e_bone.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _print_summary(r):
    print("=" * 66)
    print("TEST E — BONE-AXIS COUPLING (5-axis, waves 0-3)")
    print("=" * 66)
    for src in ["F", "I"]:
        o = r[f"{src}_to_B_overall"]
        if "beta" in o:
            print(f"{src}→B overall: β={format_ci(o['beta'], o['se'])} "
                  f"(N={o['n']}, p={o['p']:.3f})")
        else:
            print(f"{src}→B overall: insufficient data")
    print(f"Bonferroni primary: {r['bonferroni_primary']}")
    print("\n[F→B by SPPB tertile]")
    for t in [0.0, 1.0, 2.0]:
        g = r["F_to_B_by_sppb_tertile"].get("groups", {}).get(str(t), {})
        if "beta" in g:
            label = ["Low", "Mid", "High"][int(t)]
            print(f"  {label:<6}: β={format_ci(g['beta'], g['se'])} "
                  f"(N={g['n']}, p={g['p']:.3f})")
    trend = r["F_to_B_by_sppb_tertile"].get("interaction", {})
    if "joint_wald_p" in trend:
        print(f"  Joint interaction p = {trend['joint_wald_p']}")
    print("\n[Bisphosphonate comparison]")
    for src in ["F", "I"]:
        b = r["bisphosphonate"][f"{src}_to_B"]
        print(f"  {src}→B: N_bisphos={b['n_bisphos_pairs']}, "
              f"N_non={b['n_nonuser_pairs']}", end="")
        if b.get("skipped"):
            print(f"  [SKIPPED: {b['skipped']}]")
        else:
            i = b.get("interaction", {})
            print(f"  interaction p = {i.get('p', float('nan')):.3f}")
    print("=" * 66)


if __name__ == "__main__":
    run()
