"""
Test C: Exercise capacity (SPPB) as coupling modifier.

HDR prediction: the F→ row is predominantly protective. Higher SPPB → weaker
pathological coupling on I↔M↔N pairs; stronger protective F→I coupling.

Method: SPPB tertile at wave w as group; compare β for all 12 available
cross-axis directions. Trend p via ordinal interaction with tertile index.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    load_panel_with_deltas, _build_lag_pairs, _fit_ols,
    assign_tertiles, format_ci, bonferroni,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

AXIS_PAIRS = [
    ("I", "M"), ("M", "I"),
    ("I", "N"), ("N", "I"),
    ("I", "F"), ("F", "I"),
    ("M", "F"), ("F", "M"),
    ("M", "N"), ("N", "M"),
    ("F", "N"), ("N", "F"),
]


def _tertile_fit(panel, source, target):
    """
    Fit the source→target cross-lagged by SPPB tertile at wave w.
    Returns {'tertile_0_low': {...}, 'tertile_1_mid': {...}, 'tertile_2_high': {...},
            'interaction': {...}, 'trend_p': float}.
    """
    # Assign SPPB tertile at each wave
    p = panel.copy()
    p["sppb_tert"] = np.nan
    for w, sub in p.groupby("wave"):
        tert = assign_tertiles(sub["sppb"])
        p.loc[sub.index, "sppb_tert"] = tert.values

    pairs = _build_lag_pairs(p, source, target,
                             group_col="sppb_tert")
    out = {"source": source, "target": target,
           "n_pairs_total": int(len(pairs))}

    rhs = "src_w + tgt_w + age_w"
    for t in [0.0, 1.0, 2.0]:
        sub = pairs[pairs["group"] == t]
        if len(sub) < 30:
            out[f"tertile_{int(t)}"] = {"n": int(len(sub)), "skipped": True}
            continue
        f = _fit_ols(sub, f"tgt_wn ~ {rhs}")
        out[f"tertile_{int(t)}"] = {
            "beta": float(f.params["src_w"]),
            "se": float(f.bse["src_w"]),
            "p": float(f.pvalues["src_w"]),
            "n": int(f.nobs),
        }

    if len(pairs) >= 90:
        # Treat tertile as continuous for linear trend interaction
        pairs["tert_lin"] = pairs["group"].astype(float)
        try:
            f = _fit_ols(
                pairs, f"tgt_wn ~ src_w * tert_lin + tgt_w + age_w")
            out["trend"] = {
                "beta_interaction": float(f.params["src_w:tert_lin"]),
                "se": float(f.bse["src_w:tert_lin"]),
                "p": float(f.pvalues["src_w:tert_lin"]),
                "n": int(f.nobs),
            }
        except Exception as e:
            out["trend"] = {"error": str(e)}
    return out


def run():
    panel, _ = load_panel_with_deltas()

    results = {"prediction": "High-SPPB shows weaker pathological coupling "
                             "and stronger protective F→I",
               "pairs": {}}

    trend_ps = []
    for src, tgt in AXIS_PAIRS:
        key = f"{src}_{tgt}"
        r = _tertile_fit(panel, src, tgt)
        results["pairs"][key] = r
        trend_p = r.get("trend", {}).get("p", np.nan)
        if not np.isnan(trend_p):
            trend_ps.append((key, trend_p))

    # Bonferroni across 12 trend tests
    raw = [p for _, p in trend_ps]
    bonf = bonferroni(raw) if raw else []
    results["trend_bonferroni"] = {
        key: float(bp) for (key, _), bp in zip(trend_ps, bonf)
    }

    # Count pairs where high-SPPB β magnitude is SMALLER than low-SPPB β
    # (i.e. pathological coupling weaker). Exception: for protective F→I
    # we expect MORE NEGATIVE β (stronger), so we count those separately.
    weaker_count = 0
    stronger_protective_F = 0
    for key, r in results["pairs"].items():
        lo = r.get("tertile_0", {}).get("beta", np.nan)
        hi = r.get("tertile_2", {}).get("beta", np.nan)
        if np.isnan(lo) or np.isnan(hi):
            continue
        if abs(hi) < abs(lo):
            weaker_count += 1
        if key == "F_I":
            # F→I: more-negative (or less-positive) β in high-SPPB = stronger
            # protective. Sign convention: delta_F positive = decline, so a
            # protective F would LOWER delta_I. Here delta_F is flipped so
            # positive delta_F = decline; for protective effect we expect
            # src_w having negative association with tgt_wn.
            # We simply record the direction.
            stronger_protective_F = (hi < lo)
    results["summary"] = {
        "n_pairs_weaker_in_high_sppb": int(weaker_count),
        "n_pairs_total": len(AXIS_PAIRS),
        "F_to_I_stronger_protective_in_high_sppb": bool(stronger_protective_F),
    }

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    with open(os.path.join(RESULTS, "test_c_exercise.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _figure(results)
    _print_summary(results)
    return results


def _figure(results):
    """3×12 heatmap: rows = SPPB tertile (low/mid/high), cols = axis pair."""
    pair_labels = [f"{s}→{t}" for s, t in AXIS_PAIRS]
    M = np.full((3, len(AXIS_PAIRS)), np.nan)
    for j, (s, t) in enumerate(AXIS_PAIRS):
        r = results["pairs"][f"{s}_{t}"]
        for k in range(3):
            g = r.get(f"tertile_{k}", {})
            if "beta" in g:
                M[k, j] = g["beta"]

    fig, ax = plt.subplots(figsize=(12, 3.4))
    vmax = np.nanmax(np.abs(M)) if np.any(~np.isnan(M)) else 0.1
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(pair_labels)))
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low SPPB", "Mid SPPB", "High SPPB"], fontsize=9)
    # Annotate β values
    for i in range(3):
        for j in range(len(pair_labels)):
            if not np.isnan(M[i, j]):
                txt = f"{M[i,j]:+.2f}"
                color = "white" if abs(M[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=8)
    # Star significant trend p-values
    for j, (s, t) in enumerate(AXIS_PAIRS):
        r = results["pairs"][f"{s}_{t}"]
        p = r.get("trend", {}).get("p", np.nan)
        if not np.isnan(p) and p < 0.05:
            ax.text(j, -0.6, "*", ha="center", fontsize=14, color="black")
    fig.colorbar(im, ax=ax, label=r"$\beta$ (cross-lagged)")
    ax.set_title("Test C — β by axis pair and SPPB tertile "
                 "(* = linear trend p<0.05)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "test_c_exercise.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _print_summary(r):
    print("=" * 78)
    print("TEST C — SPPB TERTILES AS COUPLING MODIFIER")
    print("=" * 78)
    print(f"{'Pair':<8} {'β_low':>12}  {'β_mid':>12}  {'β_high':>12}  "
          f"{'trend_p':>8}  {'trend_bonf':>10}")
    trend_bonf = r["trend_bonferroni"]
    for s, t in AXIS_PAIRS:
        key = f"{s}_{t}"
        d = r["pairs"][key]
        def _b(k):
            g = d.get(f"tertile_{k}", {})
            return f"{g.get('beta', float('nan')):+.4f}" if "beta" in g else "  skip"
        tp = d.get("trend", {}).get("p", float("nan"))
        tb = trend_bonf.get(key, float("nan"))
        print(f"{s}→{t:<5} {_b(0):>12}  {_b(1):>12}  {_b(2):>12}  "
              f"{tp:>8.3f}  {tb:>10.3f}")
    s = r["summary"]
    print(f"\nN pairs weaker in high-SPPB: "
          f"{s['n_pairs_weaker_in_high_sppb']}/{s['n_pairs_total']}")
    print(f"F→I stronger-protective in high-SPPB: "
          f"{s['F_to_I_stronger_protective_in_high_sppb']}")
    print("=" * 78)


if __name__ == "__main__":
    run()
