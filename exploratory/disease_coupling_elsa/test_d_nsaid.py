"""
Test D (ELSA): NSAIDs (OA pain-med proxy) and I→* coupling.

HDR prediction: NSAIDs suppress inflammatory signalling; I→M, I→N, I→F
should all weaken in users.

ELSA has no dedicated NSAID flag. Two proxies are used:
  - `hepmed`  "knee or hip pain: whether taking medication (has
              osteoarthritis)" — available at waves 4 and 6 only. This
              is OA-pain-med use, heavily but not exclusively NSAIDs.
  - `hehrtmd` "blood-thinning medication for CVD" — available at wave 4
              only. Aspirin proxy (partial NSAID).

Because each flag is only observed at 1–2 waves, lag-pair availability
is limited. We run both proxies and report separately.
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
PROXIES = [("med_nsaid", "OA pain-med (hepmed)"),
           ("med_aspirin", "Blood-thinner (hehrtmd)")]


def _run_proxy(panel, proxy_col):
    out = {"proxy": proxy_col, "by_target": {}}
    interaction_ps = []
    for tgt in TARGETS:
        unadj = cross_lagged_regression(panel, "I", tgt, group_col=proxy_col)
        pairs = _build_lag_pairs(panel, "I", tgt, group_col=proxy_col)
        matched = age_match(pairs, "group", caliper=5.0)
        matched_res = {"n_total": int(len(matched))}
        rhs = "src_w + tgt_w + age_w"
        for gk, gn in [(0, "nonuser"), (1, "user")]:
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
        out["by_target"][tgt] = {
            "unadjusted": unadj,
            "age_matched": matched_res,
        }
        ip = unadj.get("interaction", {}).get("p", np.nan)
        if isinstance(ip, float) and not np.isnan(ip):
            interaction_ps.append(ip)
    out["interaction_p_raw"] = interaction_ps
    out["interaction_p_bonf"] = bonferroni(interaction_ps) if interaction_ps else []
    return out


def run():
    panel, _ = load_panel_with_deltas()
    results = {"prediction": "NSAIDs weaken I→M, I→N, I→F", "cohort": "ELSA",
               "proxies": {}}

    for proxy_col, proxy_label in PROXIES:
        results["proxies"][proxy_col] = _run_proxy(panel, proxy_col)
        results["proxies"][proxy_col]["label"] = proxy_label

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "test_d_nsaid.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _figure(results)
    _print_summary(results)
    return results


def _figure(results):
    fig, axes = plt.subplots(1, len(PROXIES), figsize=(12, 4.5),
                             sharey=True)
    if len(PROXIES) == 1:
        axes = [axes]
    for idx, (proxy_col, proxy_label) in enumerate(PROXIES):
        ax = axes[idx]
        pr = results["proxies"][proxy_col]
        x = np.arange(len(TARGETS))
        width = 0.35
        for i, (gkey, gname, color) in enumerate(
            [("0.0", "Non-user", "#5A8BB0"),
             ("1.0", "User", "#C27A4F")]
        ):
            betas, ses = [], []
            for tgt in TARGETS:
                g = pr["by_target"][tgt]["unadjusted"]["groups"].get(gkey, {})
                betas.append(g.get("beta", np.nan))
                ses.append(1.96 * g.get("se", 0))
            offset = (-width / 2) if i == 0 else (width / 2)
            ax.bar(x + offset, betas, width, yerr=ses, capsize=3,
                   color=color, label=gname, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"I→{t}" for t in TARGETS], fontsize=10)
        ax.set_title(proxy_label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        for i, tgt in enumerate(TARGETS):
            p = pr["by_target"][tgt]["unadjusted"]["interaction"].get(
                "p", np.nan)
            if isinstance(p, float) and not np.isnan(p):
                ax.text(x[i], ax.get_ylim()[1] * 0.92, f"p={p:.3f}",
                        ha="center", fontsize=8, fontstyle="italic")
    axes[0].set_ylabel(r"$\beta$ (95% CI)")
    fig.suptitle("Test D (ELSA) — NSAIDs/aspirin proxies and I→ row",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_d_nsaid.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _print_summary(r):
    print("=" * 66)
    print("TEST D (ELSA) — NSAIDs/ASPIRIN PROXIES AND I→ ROW")
    print("=" * 66)
    for proxy_col, pr in r["proxies"].items():
        print(f"\n[Proxy: {pr['label']} ({proxy_col})]")
        for tgt in TARGETS:
            u = pr["by_target"][tgt]["unadjusted"]
            print(f"  I→{tgt}:")
            for gk, gl in [("0.0", "Non-user"), ("1.0", "User")]:
                g = u["groups"].get(gk, {})
                if "beta" in g:
                    print(f"    {gl:<10}: β={format_ci(g['beta'], g['se'])} "
                          f"(N={g['n']}, p={g['p']:.3f})")
            ip = u["interaction"].get("p", float("nan"))
            print(f"    Interaction p = {ip}")
        bonf = pr.get("interaction_p_bonf", [])
        print(f"  Bonferroni p: {[f'{x:.3f}' for x in bonf]}")
    print("=" * 66)


if __name__ == "__main__":
    run()
