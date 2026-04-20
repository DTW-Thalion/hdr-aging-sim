"""
Test D: NSAIDs and I→* coupling.

HDR prediction: NSAIDs suppress inflammatory signalling → entire I→ row
should weaken. Tests: I→M, I→N, I→F.

Caveat: NSAIDs are OTC; med_nsaid may undercount. N reported carefully.
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

TARGETS = ["M", "N", "F"]


def run():
    panel, _ = load_panel_with_deltas()

    results = {"prediction": "NSAIDs weaken I→M, I→N, I→F",
               "by_target": {}}

    interaction_ps = []
    for tgt in TARGETS:
        unadj = cross_lagged_regression(panel, "I", tgt,
                                        group_col="med_nsaid")
        pairs = _build_lag_pairs(panel, "I", tgt, group_col="med_nsaid")
        matched = age_match(pairs, "group", caliper=5.0)
        rhs = "src_w + tgt_w + age_w"
        matched_res = {"n_total": int(len(matched))}
        for gk, gn in [(0, "nonuser"), (1, "nsaid")]:
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
            fi = _fit_ols(matched, f"tgt_wn ~ src_w * group_b + tgt_w + age_w")
            matched_res["interaction"] = {
                "beta": float(fi.params["src_w:group_b"]),
                "se": float(fi.bse["src_w:group_b"]),
                "p": float(fi.pvalues["src_w:group_b"]),
                "n": int(fi.nobs),
            }
        results["by_target"][tgt] = {
            "unadjusted": unadj,
            "age_matched": matched_res,
        }
        interaction_ps.append(unadj["interaction"]["p"])

    results["interaction_p_raw"] = interaction_ps
    results["interaction_p_bonf"] = bonferroni(interaction_ps)
    n_attenuated = sum(
        1 for tgt in TARGETS
        if (results["by_target"][tgt]["unadjusted"]["groups"].get("1.0", {}).get("beta", 0) is not None and
            abs(results["by_target"][tgt]["unadjusted"]["groups"].get("1.0", {}).get("beta", 0)) <
            abs(results["by_target"][tgt]["unadjusted"]["groups"].get("0.0", {}).get("beta", 0)))
    )
    results["summary"] = {
        "n_attenuated_of_3": n_attenuated,
        "all_interactions_sig_after_bonf": bool(all(p < 0.05 for p in results["interaction_p_bonf"])),
    }

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "test_d_nsaid.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _figure(results)
    _print_summary(results)
    return results


def _figure(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"I→{t}" for t in TARGETS]
    x = np.arange(len(labels))
    width = 0.35
    for i, (gkey, gname, color) in enumerate(
        [("0.0", "Non-user", "#5A8BB0"),
         ("1.0", "NSAID user", "#C27A4F")]
    ):
        betas, ses = [], []
        for tgt in TARGETS:
            g = results["by_target"][tgt]["unadjusted"]["groups"].get(gkey, {})
            betas.append(g.get("beta", np.nan))
            ses.append(1.96 * g.get("se", 0))
        offset = (-width / 2) if i == 0 else (width / 2)
        ax.bar(x + offset, betas, width, yerr=ses, capsize=4,
               color=color, label=gname, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"$\beta$ (95% CI)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, tgt in enumerate(TARGETS):
        p = results["by_target"][tgt]["unadjusted"]["interaction"]["p"]
        ax.text(x[i], ax.get_ylim()[1] * 0.92, f"p={p:.3f}",
                ha="center", fontsize=9, fontstyle="italic")
    ax.set_title("Test D — NSAIDs and I→ row coupling", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_d_nsaid.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _print_summary(r):
    print("=" * 66)
    print("TEST D — NSAIDs AND I→* COUPLING")
    print("=" * 66)
    for tgt in TARGETS:
        u = r["by_target"][tgt]["unadjusted"]
        print(f"\n[I→{tgt}]")
        for gk, gl in [("0.0", "Non-user"), ("1.0", "NSAID")]:
            g = u["groups"].get(gk, {})
            if "beta" in g:
                print(f"  {gl:<10}: β={format_ci(g['beta'], g['se'])} "
                      f"(N={g['n']}, p={g['p']:.3f})")
        print(f"  Interaction p = {u['interaction']['p']:.3f}")
    print(f"\nBonferroni-corrected interaction p (3 tests): "
          f"{[f'{p:.3f}' for p in r['interaction_p_bonf']]}")
    s = r["summary"]
    print(f"N attenuated in NSAID users: {s['n_attenuated_of_3']}/3")
    print("=" * 66)


if __name__ == "__main__":
    run()
