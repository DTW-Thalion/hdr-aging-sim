"""
Test B (ELSA): Antidiabetics and D-vs-J discrimination.

HDR prediction: antidiabetic drugs primarily target τ_M (D-matrix).
  - I→M should NOT change (J-matrix intact)
  - M→I should WEAKEN (improved metabolism → less inflammatory drive)

Uses `med_antidm` = `hemdb` across waves 2/4/6/8.
"""

import os
import json
import sys
import numpy as np
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

    result = {
        "prediction": "I→M unchanged (J-matrix), M→I weakened (D-matrix)",
        "cohort": "ELSA",
        "antidm_flag": "hemdb",
    }

    for direction in ["I_to_M", "M_to_I"]:
        src, tgt = direction.split("_to_")
        unadj = cross_lagged_regression(
            panel, src, tgt, group_col="med_antidm")
        pairs = _build_lag_pairs(panel, src, tgt, group_col="med_antidm")
        matched = age_match(pairs, "group", caliper=5.0)
        rhs = "src_w + tgt_w + age_w"
        matched_res = {"n_total": int(len(matched))}
        for gk, gn in [(0, "nonuser"), (1, "antidm")]:
            sub = matched[matched["group"] == gk]
            if len(sub) >= 30:
                f = _fit_ols(sub, f"tgt_wn ~ {rhs}")
                matched_res[gn] = {
                    "beta": float(f.params["src_w"]),
                    "se": float(f.bse["src_w"]),
                    "p": float(f.pvalues["src_w"]),
                    "n": int(f.nobs),
                }
        if len(matched) >= 60:
            matched["group_b"] = matched["group"].astype(float)
            fi = _fit_ols(matched, "tgt_wn ~ src_w * group_b + tgt_w + age_w")
            matched_res["interaction"] = {
                "beta": float(fi.params["src_w:group_b"]),
                "se": float(fi.bse["src_w:group_b"]),
                "p": float(fi.pvalues["src_w:group_b"]),
                "n": int(fi.nobs),
            }
        result[direction] = {
            "unadjusted": unadj,
            "age_matched": matched_res,
        }

    p_IM = result["I_to_M"]["unadjusted"]["interaction"]["p"]
    p_MI = result["M_to_I"]["unadjusted"]["interaction"]["p"]
    bonf = bonferroni([p_IM, p_MI])
    result["d_vs_j_discrimination"] = {
        "I_to_M_interaction_p": p_IM,
        "M_to_I_interaction_p": p_MI,
        "bonferroni_p": bonf,
        "IM_unchanged": bool(p_IM > 0.05),
        "MI_changed": bool(bonf[1] < 0.05),
        "discrimination_supported": bool((p_IM > 0.05) and (bonf[1] < 0.05)),
    }

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "test_b_metformin.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    _figure(result)
    _print_summary(result)
    return result


def _figure(result):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = ["I→M\n(J-matrix)", "M→I\n(D-matrix)"]
    groups = [("Non-user", "0.0", "#5A8BB0"),
              ("Antidiabetic", "1.0", "#C27A4F")]
    x = np.arange(len(labels))
    width = 0.35

    for i, (label, gkey, color) in enumerate(groups):
        betas, ses = [], []
        for direction in ["I_to_M", "M_to_I"]:
            g = result[direction]["unadjusted"]["groups"].get(gkey, {})
            betas.append(g.get("beta", np.nan))
            ses.append(1.96 * g.get("se", 0))
        offset = (-width / 2) if i == 0 else (width / 2)
        ax.bar(x + offset, betas, width, yerr=ses, capsize=4,
               color=color, label=label, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$\beta$ (95% CI)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for i, direction in enumerate(["I_to_M", "M_to_I"]):
        p = result[direction]["unadjusted"]["interaction"]["p"]
        ax.text(x[i], ax.get_ylim()[1] * 0.92,
                f"interaction p={p:.3f}", ha="center", fontsize=9,
                fontstyle="italic")

    d = result["d_vs_j_discrimination"]
    status = "SUPPORTED" if d["discrimination_supported"] else "NOT SUPPORTED"
    ax.set_title(f"Test B (ELSA) — Antidiabetics D-vs-J: {status}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_b_metformin.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _print_summary(r):
    print("=" * 66)
    print("TEST B (ELSA) — ANTIDIABETICS AND D-vs-J DISCRIMINATION")
    print("=" * 66)
    for direction in ["I_to_M", "M_to_I"]:
        u = r[direction]["unadjusted"]
        print(f"\n[{direction.replace('_',' ')}]")
        for gk, gl in [("0.0", "Non-user"), ("1.0", "Antidm")]:
            g = u["groups"].get(gk, {})
            if "beta" in g:
                print(f"  {gl:<12}: β={format_ci(g['beta'], g['se'])} "
                      f"(N={g['n']}, p={g['p']:.3f})")
        print(f"  Interaction p = {u['interaction']['p']:.3f}")
    d = r["d_vs_j_discrimination"]
    print("\n[D-vs-J Discrimination]")
    print(f"  I→M interaction p = {d['I_to_M_interaction_p']:.3f} "
          f"(want: NS → unchanged J-matrix)")
    print(f"  M→I interaction p = {d['M_to_I_interaction_p']:.3f} "
          f"(want: sig → weakened D-matrix)")
    print(f"  Bonferroni p     = {d['bonferroni_p']}")
    print(f"  Supported: {d['discrimination_supported']}")
    print("=" * 66)


if __name__ == "__main__":
    run()
