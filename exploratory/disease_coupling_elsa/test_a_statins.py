"""
Test A (ELSA): Statins and I→M coupling.

HDR prediction: statin users show weaker β_{I→M} than non-users.

Uses `med_statin` = `hechmd` (prescribed cholesterol-lowering meds)
from waves 4, 6, 8 (≈2000-2400 users per wave). Lag-pairs are
w4→w6 and w6→w8.

Three comparisons:
  1. Unadjusted stratification
  2. Baseline-biomarker adjusted (log CRP + HbA1c at wave w)
  3. Age-matched (±5 years)
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
    _build_lag_pairs, _fit_ols, age_match, format_ci, bonferroni,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def run():
    panel, _ = load_panel_with_deltas()

    unadj = cross_lagged_regression(panel, "I", "M", group_col="med_statin")

    panel_w = panel.copy()
    panel_w["crp_w"] = panel_w["log_crp"]
    panel_w["hba1c_w"] = panel_w["hba1c"]
    adj = cross_lagged_regression(
        panel_w, "I", "M", group_col="med_statin",
        extra_controls=["crp_w", "hba1c_w"],
    )

    pairs = _build_lag_pairs(panel, "I", "M", group_col="med_statin")
    matched = age_match(pairs, "group", age_col="age_w", caliper=5.0)
    matched_res = {"n_total": int(len(matched))}
    if len(matched) >= 60:
        rhs = "src_w + tgt_w + age_w"
        for gk, gn in [(0, "nostatin"), (1, "statin")]:
            sub = matched[matched["group"] == gk]
            if len(sub) >= 30:
                f = _fit_ols(sub, f"tgt_wn ~ {rhs}")
                matched_res[gn] = {
                    "beta": float(f.params["src_w"]),
                    "se": float(f.bse["src_w"]),
                    "p": float(f.pvalues["src_w"]),
                    "n": int(f.nobs),
                }
        matched["group_b"] = matched["group"].astype(float)
        fi = _fit_ols(matched, "tgt_wn ~ src_w * group_b + tgt_w + age_w")
        matched_res["interaction"] = {
            "beta": float(fi.params["src_w:group_b"]),
            "se": float(fi.bse["src_w:group_b"]),
            "p": float(fi.pvalues["src_w:group_b"]),
            "n": int(fi.nobs),
        }

    int_ps = [
        unadj.get("interaction", {}).get("p", np.nan),
        adj.get("interaction", {}).get("p", np.nan),
        matched_res.get("interaction", {}).get("p", np.nan),
    ]
    int_ps_bonf = bonferroni([p for p in int_ps if not np.isnan(p)])

    result = {
        "prediction": "Statin users show weaker β_{I→M}",
        "cohort": "ELSA",
        "statin_flag": "hechmd (prescribed cholesterol-lowering, waves 4/6/8)",
        "unadjusted": unadj,
        "baseline_biomarker_adjusted": adj,
        "age_matched": matched_res,
        "bonferroni_interaction_p": int_ps_bonf,
    }

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "test_a_statins.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    _figure(panel, unadj, adj, matched_res)
    _print_summary(result)
    return result


def _figure(panel, unadj, adj, matched_res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    models = ["Unadjusted", "Baseline-adjusted", "Age-matched"]
    colors = {"Non-user": "#5A8BB0", "Statin user": "#C27A4F"}

    ax = axes[0]
    xpos = np.arange(len(models))
    width = 0.35

    for label in ["Non-user", "Statin user"]:
        betas, ses = [np.nan] * len(models), [0] * len(models)
        gk = "0.0" if label == "Non-user" else "1.0"
        mk_unadj = unadj.get("groups", {}).get(gk, {})
        mk_adj = adj.get("groups", {}).get(gk, {})
        mk_match = matched_res.get(
            "nostatin" if label == "Non-user" else "statin", {})
        for i, m in enumerate([mk_unadj, mk_adj, mk_match]):
            if "beta" in m:
                betas[i] = m["beta"]
                ses[i] = 1.96 * m["se"]
        offset = (-width / 2) if label == "Non-user" else (width / 2)
        ax.bar(xpos + offset, betas, width, yerr=ses, capsize=3,
               color=colors[label], label=label, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel(r"$\beta_{I \rightarrow M}$ (95% CI)")
    ax.set_title("(a) β_{I→M}: statin vs non-statin (ELSA)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    pairs = _build_lag_pairs(panel, "I", "M", group_col="med_statin")
    for gval, lab in [(0.0, "Non-user"), (1.0, "Statin user")]:
        sub = pairs[pairs["group"] == gval]
        if len(sub) == 0:
            continue
        ax.scatter(sub["src_w"], sub["tgt_wn"], s=4, alpha=0.2,
                   color=colors[lab], label=f"{lab} (N={len(sub)})")
    ax.set_xlabel(r"$\Delta_I$ at wave $w$")
    ax.set_ylabel(r"$\Delta_M$ at wave $w+1$")
    ax.set_title("(b) I(w) → M(w+1) by statin status")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Test A — Statins and I→M coupling (ELSA)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_a_statins.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _print_summary(r):
    print("=" * 66)
    print("TEST A (ELSA) — STATINS AND I→M COUPLING")
    print("=" * 66)
    for mname, mres in [
        ("Unadjusted", r["unadjusted"]),
        ("Baseline-adjusted", r["baseline_biomarker_adjusted"]),
    ]:
        print(f"\n[{mname}]")
        for gk, gl in [("0.0", "Non-user"), ("1.0", "Statin user")]:
            g = mres.get("groups", {}).get(gk, {})
            if "beta" in g:
                print(f"  {gl:<12}: β={format_ci(g['beta'], g['se'])} "
                      f"(N={g['n']}, p={g['p']:.3f})")
        i = mres.get("interaction", {})
        if "p" in i:
            print(f"  Interaction p = {i['p']:.3f}  "
                  f"(β_interact = {i.get('beta', float('nan')):+.4f})")
    m = r["age_matched"]
    print(f"\n[Age-matched (±5y)]  N_total_pairs={m.get('n_total', 0)}")
    for key, lab in [("nostatin", "Non-user"), ("statin", "Statin user")]:
        g = m.get(key, {})
        if "beta" in g:
            print(f"  {lab:<12}: β={format_ci(g['beta'], g['se'])} "
                  f"(N={g['n']}, p={g['p']:.3f})")
    i = m.get("interaction", {})
    if "p" in i:
        print(f"  Interaction p = {i['p']:.3f}")
    print(f"\nBonferroni-corrected interaction p-values: "
          f"{r['bonferroni_interaction_p']}")
    print("=" * 66)


if __name__ == "__main__":
    run()
