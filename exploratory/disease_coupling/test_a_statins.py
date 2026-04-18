"""
Test A: Statins and I→M coupling.

HDR prediction: statins dampen the inflammatory drive on metabolism.
Statin users should show weaker β_{I→M} than non-users.

Three comparisons:
  1. All pairs, statin vs non-statin
  2. Age-matched (±5 y) statin vs non-statin
  3. Baseline-biomarker-adjusted (baseline Δ_I and Δ_M as additional controls)
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
    panel, ref = load_panel_with_deltas()

    # Main: I→M stratified by med_statin
    unadj = cross_lagged_regression(panel, "I", "M", group_col="med_statin")

    # Baseline-biomarker adjustment (Δ_I and Δ_M are already in the base model,
    # so we add the raw biomarker levels at wave w as additional controls).
    panel_w = panel.copy()
    panel_w["il6_w"] = panel_w["log_il6"]
    panel_w["homa_w"] = np.log(panel_w["homa_ir"] + 0.1)
    adj = cross_lagged_regression(
        panel_w, "I", "M", group_col="med_statin",
        extra_controls=["il6_w", "homa_w"],
    )

    # Age-matched: build lag-pair table, add group, match on age_w
    pairs = _build_lag_pairs(panel, "I", "M", group_col="med_statin")
    matched = age_match(pairs, "group", age_col="age_w", caliper=5.0)
    matched_res = {"n_total": int(len(matched))}
    if len(matched) >= 60:
        m0 = matched[matched["group"] == 0]
        m1 = matched[matched["group"] == 1]
        rhs = "src_w + tgt_w + age_w"
        if len(m0) >= 30:
            f0 = _fit_ols(m0, f"tgt_wn ~ {rhs}")
            matched_res["nostatin"] = {
                "beta": float(f0.params["src_w"]),
                "se": float(f0.bse["src_w"]),
                "p": float(f0.pvalues["src_w"]),
                "n": int(f0.nobs),
            }
        if len(m1) >= 30:
            f1 = _fit_ols(m1, f"tgt_wn ~ {rhs}")
            matched_res["statin"] = {
                "beta": float(f1.params["src_w"]),
                "se": float(f1.bse["src_w"]),
                "p": float(f1.pvalues["src_w"]),
                "n": int(f1.nobs),
            }
        # Interaction on matched
        matched["group_b"] = matched["group"].astype(float)
        f_int = _fit_ols(matched, f"tgt_wn ~ src_w * group_b + tgt_w + age_w")
        matched_res["interaction"] = {
            "beta": float(f_int.params["src_w:group_b"]),
            "se": float(f_int.bse["src_w:group_b"]),
            "p": float(f_int.pvalues["src_w:group_b"]),
            "n": int(f_int.nobs),
        }

    # Bonferroni across three interaction p-values (3 comparisons)
    int_ps = [
        unadj.get("interaction", {}).get("p", np.nan),
        adj.get("interaction", {}).get("p", np.nan),
        matched_res.get("interaction", {}).get("p", np.nan),
    ]
    int_ps_bonf = bonferroni([p for p in int_ps if not np.isnan(p)])

    result = {
        "prediction": "Statin users show weaker β_{I→M}",
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

    # Panel A: β_{I→M} with 95% CI across three models × (nostatin, statin)
    models = ["Unadjusted", "Baseline-adjusted", "Age-matched"]
    data = []
    for mname, mres in [
        ("Unadjusted", unadj.get("groups", {})),
        ("Baseline-adjusted", adj.get("groups", {})),
        ("Age-matched", {
            "0.0": matched_res.get("nostatin", {}),
            "1.0": matched_res.get("statin", {}),
        }),
    ]:
        for grp_key, label in [("0.0", "Non-user"), ("1.0", "Statin user")]:
            g = mres.get(grp_key, {})
            if "beta" in g:
                data.append((mname, label, g["beta"], g["se"], g["n"]))

    ax = axes[0]
    xpos = np.arange(len(models))
    width = 0.35
    colors = {"Non-user": "#5A8BB0", "Statin user": "#C27A4F"}
    for label in ["Non-user", "Statin user"]:
        betas = [None] * len(models)
        ses = [None] * len(models)
        for mname, lab, beta, se, n in data:
            if lab == label:
                betas[models.index(mname)] = beta
                ses[models.index(mname)] = se
        offset = -width / 2 if label == "Non-user" else width / 2
        x = xpos + offset
        betas_plot = [b if b is not None else np.nan for b in betas]
        ses_plot = [1.96 * s if s is not None else 0 for s in ses]
        ax.bar(x, betas_plot, width, yerr=ses_plot,
               capsize=3, color=colors[label], label=label, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel(r"$\beta_{I \rightarrow M}$ (95% CI)")
    ax.set_title("(a) β_{I→M}: statin vs non-statin")
    ax.legend(fontsize=8, loc="best")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: scatter of ΔIL-6(w) vs ΔHOMA-IR(w+1), partial residuals
    pairs = _build_lag_pairs(panel, "I", "M", group_col="med_statin")
    ax = axes[1]
    for gval, lab in [(0.0, "Non-user"), (1.0, "Statin user")]:
        sub = pairs[pairs["group"] == gval]
        if len(sub) == 0:
            continue
        ax.scatter(sub["src_w"], sub["tgt_wn"], s=6, alpha=0.25,
                   color=colors[lab], label=f"{lab} (N={len(sub)})")
    ax.set_xlabel(r"$\Delta_I$ at wave $w$")
    ax.set_ylabel(r"$\Delta_M$ at wave $w+1$")
    ax.set_title("(b) I(w) → M(w+1) by statin status")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Test A — Statins and I→M coupling (InCHIANTI)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_a_statins.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _print_summary(r):
    print("=" * 66)
    print("TEST A — STATINS AND I→M COUPLING")
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
                  f"(β_interact = {i['beta']:+.4f})")
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
    print(f"\nBonferroni-corrected interaction p-values (3 tests): "
          f"{r['bonferroni_interaction_p']}")
    print("=" * 66)


if __name__ == "__main__":
    run()
