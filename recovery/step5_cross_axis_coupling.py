#!/usr/bin/env python3
"""
Step 5 -- Cross-axis coupling during recovery.

Three analyses:

(1) Peak-to-tau cross-lagged regression
    For each ordered pair (i -> j):
      log(tau_j) ~ peak_i + peak_j + age + sex + perturbation_type
    The sign of beta on peak_i should match the J-matrix prediction
    (J_signs in hdr_core).

(2) Daily co-recovery correlation
    For patients with simultaneous daily measurements on two axes during the
    recovery window, compute the cross-correlation of daily-change values
    Delta_y_i(t) and Delta_y_j(t).

(3) Recovery-sequence ordering
    Per patient, rank axes by t_50 = tau * ln(2). Compare to a
    literature-calibrated "expected" ordering using Kendall tau.

Output
------
  results/cross_axis_coupling.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import permutations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau, pearsonr, binomtest

from hdr_core import (
    AXES,
    J_SIGNS,
    banner,
    load_config,
    write_json,
)

# Literature-calibrated "expected" recovery order: fast -> slow.
# Heart rate (autonomic) recovers fastest; renal/inflammation slowest.
EXPECTED_ORDER_FAST_TO_SLOW = ["N", "M", "I", "renal"]


# ---------------------------------------------------------------------------
# (1) Peak-to-tau cross-lagged regression

def peak_tau_regression(fits_long: pd.DataFrame, episodes: pd.DataFrame,
                         from_axis: str, to_axis: str) -> dict | None:
    """
    Predict log(tau_j) of the to-axis from peak_i of the from-axis,
    controlling for own peak, age, sex, perturbation_type.

    fits_long has columns: hadm_id, axis, tau_hours, ...
    episodes has columns: hadm_id, axis, peak_value, ...
    """
    tau_to = fits_long[(fits_long["axis"] == to_axis) & (fits_long["tau_passes_qc"])][
        ["hadm_id", "tau_hours", "age", "sex", "perturbation_type"]
    ].rename(columns={"tau_hours": "tau_to"})

    peak_from = (episodes[episodes["axis"] == from_axis]
                 .sort_values("is_primary", ascending=False)
                 .drop_duplicates("hadm_id"))[["hadm_id", "peak_value"]] \
                 .rename(columns={"peak_value": "peak_from"})

    peak_to = (episodes[episodes["axis"] == to_axis]
               .sort_values("is_primary", ascending=False)
               .drop_duplicates("hadm_id"))[["hadm_id", "peak_value"]] \
               .rename(columns={"peak_value": "peak_to"})

    df = (tau_to.merge(peak_from, on="hadm_id", how="inner")
                 .merge(peak_to, on="hadm_id", how="inner"))
    df = df.dropna(subset=["tau_to", "peak_from", "peak_to", "age"])
    if len(df) < 50:
        return None

    df["log_tau"] = np.log(df["tau_to"])
    X = pd.DataFrame({
        "peak_from": df["peak_from"].values,
        "peak_to": df["peak_to"].values,
        "age": df["age"].values,
    })
    if df["sex"].notna().any():
        X["sex_M"] = (df["sex"] == "M").astype(float).values
    if "perturbation_type" in df.columns:
        for ptype in df["perturbation_type"].dropna().unique():
            if ptype != "general":
                X[f"ptype_{ptype}"] = (df["perturbation_type"] == ptype).astype(float).values
    X = sm.add_constant(X)
    y = df["log_tau"].values

    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception as e:
        return {"error": str(e), "n": len(df)}

    beta = float(model.params.get("peak_from", np.nan))
    sign = "+1" if beta > 0 else ("-1" if beta < 0 else "0")
    expected = J_SIGNS.get((from_axis, to_axis))
    expected_sign = "+1" if expected == 1 else "-1"

    return {
        "from": from_axis,
        "to": to_axis,
        "n": int(len(df)),
        "beta_peak_from": beta,
        "se": float(model.bse.get("peak_from", np.nan)),
        "ci95": [
            float(model.conf_int().loc["peak_from", 0]),
            float(model.conf_int().loc["peak_from", 1]),
        ],
        "pvalue": float(model.pvalues.get("peak_from", np.nan)),
        "observed_sign": sign,
        "expected_sign": expected_sign,
        "concordant": (sign == expected_sign),
    }


def all_pairs_peak_tau(fits: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    res = []
    for f, t in permutations(AXES, 2):
        r = peak_tau_regression(fits, episodes, f, t)
        if r is not None:
            res.append(r)
    if not res:
        return {"pairs": [], "concordance": None}
    n_concord = sum(1 for r in res if r.get("concordant"))
    n_total = len(res)
    bt = binomtest(n_concord, n_total, p=0.5, alternative="greater")
    return {
        "pairs": res,
        "n_pairs": n_total,
        "n_concordant": n_concord,
        "concordance_p_one_sided": float(bt.pvalue),
    }


# ---------------------------------------------------------------------------
# (2) Daily co-recovery correlation

def daily_co_recovery(panel: pd.DataFrame, episodes: pd.DataFrame,
                      cfg: dict) -> dict:
    """
    For each axis pair, compute mean Pearson r between daily diffs of primary
    biomarkers across patients with >= 5 paired daily observations during
    recovery.
    """
    primary_per_axis = {ax: cfg["axes"][ax]["primary"] for ax in AXES}
    rec_window_h = cfg["episodes"]["recovery_window_days"] * 24.0

    # Build a wide daily panel per (hadm_id, day) of primary biomarker values
    panel = panel.copy()
    panel["day"] = (panel["hours_from_admit"] // 24).astype(int)
    keep = panel[(panel["axis"].isin(AXES)) &
                 (panel.apply(lambda r: r["biomarker"] == primary_per_axis.get(r["axis"]), axis=1)) &
                 (panel["hours_from_admit"] >= 0) &
                 (panel["hours_from_admit"] <= rec_window_h)]
    daily = (keep.groupby(["hadm_id", "axis", "day"], as_index=False)
                  ["value"].median()
                  .pivot_table(index=["hadm_id", "day"], columns="axis", values="value")
                  .reset_index())

    out = []
    for ax_i, ax_j in permutations(AXES, 2):
        if ax_i not in daily.columns or ax_j not in daily.columns:
            continue
        rs = []
        for hadm, g in daily.groupby("hadm_id"):
            g = g.sort_values("day")
            di = g[ax_i].diff().values
            dj = g[ax_j].diff().values
            mask = np.isfinite(di) & np.isfinite(dj)
            if mask.sum() >= 5:
                if np.std(di[mask]) == 0 or np.std(dj[mask]) == 0:
                    continue
                r, _p = pearsonr(di[mask], dj[mask])
                if np.isfinite(r):
                    rs.append(r)
        if len(rs) >= 30:
            mean_r = float(np.mean(rs))
            t = mean_r * np.sqrt(len(rs)) / np.sqrt(max(1 - mean_r ** 2, 1e-9))
            from scipy.stats import t as t_dist
            pval = float(2 * (1 - t_dist.cdf(abs(t), df=len(rs) - 1)))
            sign_obs = "+1" if mean_r > 0 else "-1"
            expected = J_SIGNS.get((ax_i, ax_j))
            expected_sign = "+1" if expected == 1 else ("-1" if expected == -1 else None)
            out.append({
                "from": ax_i,
                "to": ax_j,
                "n_patients": int(len(rs)),
                "mean_r": mean_r,
                "p_value": pval,
                "observed_sign": sign_obs,
                "expected_sign": expected_sign,
                "concordant": (sign_obs == expected_sign) if expected_sign else None,
            })
    return {"pairs": out}


# ---------------------------------------------------------------------------
# (3) Recovery-sequence ordering

def recovery_ordering(fits: pd.DataFrame) -> dict:
    """Per (hadm_id), rank axes by t_50; compute Kendall tau vs expected."""
    qc = fits[(fits["tau_passes_qc"]) & (fits["is_primary"])].copy()
    qc["t50_h"] = qc["tau_hours"] * np.log(2)

    expected_rank = {a: i for i, a in enumerate(EXPECTED_ORDER_FAST_TO_SLOW)}
    taus_per_hadm = []
    for hadm, g in qc.groupby("hadm_id"):
        if g["axis"].nunique() < 3:
            continue
        observed = g.sort_values("t50_h")["axis"].tolist()
        observed_rank = {a: i for i, a in enumerate(observed)}
        common = list(set(observed_rank) & set(expected_rank))
        if len(common) < 3:
            continue
        x = [expected_rank[a] for a in common]
        y = [observed_rank[a] for a in common]
        tau_k, _ = kendalltau(x, y)
        if np.isfinite(tau_k):
            taus_per_hadm.append(tau_k)

    if not taus_per_hadm:
        return {"n_patients": 0, "kendall_tau_mean": None}
    return {
        "n_patients": int(len(taus_per_hadm)),
        "kendall_tau_mean": float(np.mean(taus_per_hadm)),
        "kendall_tau_median": float(np.median(taus_per_hadm)),
        "expected_order": EXPECTED_ORDER_FAST_TO_SLOW,
        "fraction_positive": float(np.mean(np.array(taus_per_hadm) > 0)),
    }


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    fits_path = os.path.join(out_dir, "recovery_fits.parquet")
    eps_path = os.path.join(out_dir, "recovery_episodes.parquet")
    if not os.path.exists(fits_path) or not os.path.exists(eps_path):
        print("ERROR: missing recovery_fits.parquet or recovery_episodes.parquet.")
        return 1

    banner("Step 5: cross-axis coupling")
    fits = pd.read_parquet(fits_path)
    eps = pd.read_parquet(eps_path)

    # Pick MIMIC or ISARIC panel for daily-coreco analysis
    panel_path = os.path.join(out_dir, "mimic_biomarker_panel.parquet")
    if not os.path.exists(panel_path):
        panel_path = os.path.join(out_dir, "isaric_biomarker_panel.parquet")
    panel = pd.read_parquet(panel_path) if os.path.exists(panel_path) else pd.DataFrame()

    print("\n(1) peak-to-tau cross-lagged regression")
    peak_tau = all_pairs_peak_tau(fits, eps)
    print(f"  n pairs computed: {peak_tau.get('n_pairs', 0)}  "
          f"concordant: {peak_tau.get('n_concordant', 0)}/{peak_tau.get('n_pairs', 0)}  "
          f"p={peak_tau.get('concordance_p_one_sided', float('nan')):.4g}")

    print("\n(2) daily co-recovery correlation")
    if panel.empty:
        print("  (no panel; skipping)")
        co_rec = {"pairs": []}
    else:
        co_rec = daily_co_recovery(panel, eps, cfg)
        print(f"  n axis pairs: {len(co_rec['pairs'])}")

    print("\n(3) recovery-sequence ordering")
    seq = recovery_ordering(fits)
    print(f"  patients with >=3 axis fits: {seq.get('n_patients')}  "
          f"mean Kendall tau: {seq.get('kendall_tau_mean')}")

    out = {
        "peak_to_tau": peak_tau,
        "daily_co_recovery": co_rec,
        "recovery_ordering": seq,
    }
    out_path = os.path.join(out_dir, "cross_axis_coupling.json")
    write_json(out, out_path)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
