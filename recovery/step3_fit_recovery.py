#!/usr/bin/env python3
"""
Step 3 -- Fit exponential recovery curves per (episode, axis).

For each episode (one row per primary biomarker per admission), fit:
  - Exponential:   y(t) = y_b + (y_p - y_b) * exp(-t / tau)        (3 params)
  - Stretched:     y(t) = y_b + (y_p - y_b) * exp(-(t/tau)^beta)   (4 params)
  - Linear:        y(t) = y_p - k * t                              (2 params)

Compare by AIC. Quality filters:
  - R^2 >= 0.30
  - tau within bounds (1h, 720h)
  - >= 4 points in recovery phase

Output
------
  results/recovery_fits.parquet
  results/recovery_fit_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from hdr_core import (
    aic,
    banner,
    exp_recovery,
    linear_recovery,
    load_config,
    primary_biomarker,
    r_squared,
    stretched_exp_recovery,
)

TAU_MIN_H = 1.0
TAU_MAX_H = 720.0
R2_THRESHOLD = 0.30
MIN_POINTS = 4


def _fit_exp(t, y, y_baseline_init, y_peak_init):
    """Fit exponential, returning (params, y_hat, rss, aic_val) or None."""
    try:
        popt, _ = curve_fit(
            exp_recovery, t, y,
            p0=[y_baseline_init, y_peak_init, 24.0],
            bounds=([min(y) - 5 * abs(min(y)) - 1, min(y) - 5 * abs(min(y)) - 1, TAU_MIN_H],
                    [max(y) + 5 * abs(max(y)) + 1, max(y) + 5 * abs(max(y)) + 1, TAU_MAX_H]),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return None
    y_hat = exp_recovery(t, *popt)
    rss = float(np.sum((y - y_hat) ** 2))
    return popt, y_hat, rss, aic(rss, len(t), 3)


def _fit_stretched(t, y, y_baseline_init, y_peak_init):
    try:
        popt, _ = curve_fit(
            stretched_exp_recovery, t, y,
            p0=[y_baseline_init, y_peak_init, 24.0, 1.0],
            bounds=([min(y) - 5 * abs(min(y)) - 1, min(y) - 5 * abs(min(y)) - 1, TAU_MIN_H, 0.2],
                    [max(y) + 5 * abs(max(y)) + 1, max(y) + 5 * abs(max(y)) + 1, TAU_MAX_H, 3.0]),
            maxfev=3000,
        )
    except (RuntimeError, ValueError):
        return None
    y_hat = stretched_exp_recovery(t, *popt)
    rss = float(np.sum((y - y_hat) ** 2))
    return popt, y_hat, rss, aic(rss, len(t), 4)


def _fit_linear(t, y, y_peak_init):
    try:
        popt, _ = curve_fit(linear_recovery, t, y, p0=[y_peak_init, 0.1], maxfev=2000)
    except (RuntimeError, ValueError):
        return None
    y_hat = linear_recovery(t, *popt)
    rss = float(np.sum((y - y_hat) ** 2))
    return popt, y_hat, rss, aic(rss, len(t), 2)


def fit_one_episode(row: pd.Series) -> Optional[dict]:
    """Fit exp / stretched / linear; return dict with results or None on failure."""
    rec_t = np.asarray(row["recovery_t_hours"], dtype=float)
    rec_y = np.asarray(row["recovery_y"], dtype=float)
    valid = np.isfinite(rec_t) & np.isfinite(rec_y)
    rec_t, rec_y = rec_t[valid], rec_y[valid]
    if len(rec_t) < MIN_POINTS:
        return None

    # Prepend the peak (t=0, y=peak_value) -- it's the perturbation start
    t_full = np.concatenate(([0.0], rec_t))
    y_full = np.concatenate(([row["peak_value"]], rec_y))

    y_peak_init = float(row["peak_value"])
    y_baseline_init = (float(row["estimated_baseline"])
                       if np.isfinite(row.get("estimated_baseline", np.nan))
                       else float(np.median(rec_y)))

    out = {
        "subject_id": int(row["subject_id"]),
        "hadm_id": int(row["hadm_id"]),
        "axis": row["axis"],
        "biomarker": row["biomarker"],
        "n_points": int(len(t_full)),
        "y_peak": y_peak_init,
        "y_baseline_init": y_baseline_init,
    }

    fit_e = _fit_exp(t_full, y_full, y_baseline_init, y_peak_init)
    fit_s = _fit_stretched(t_full, y_full, y_baseline_init, y_peak_init)
    fit_l = _fit_linear(t_full, y_full, y_peak_init)

    if fit_e is not None:
        popt, y_hat, rss, a = fit_e
        out.update({
            "exp_y_baseline": float(popt[0]),
            "exp_y_peak": float(popt[1]),
            "tau_hours": float(popt[2]),
            "tau_days": float(popt[2]) / 24.0,
            "exp_R2": float(r_squared(y_full, y_hat)),
            "exp_AIC": float(a),
        })
    else:
        out.update({"exp_y_baseline": np.nan, "exp_y_peak": np.nan,
                    "tau_hours": np.nan, "tau_days": np.nan,
                    "exp_R2": np.nan, "exp_AIC": np.inf})

    if fit_s is not None:
        popt, y_hat, rss, a = fit_s
        out.update({
            "stretched_tau": float(popt[2]),
            "stretched_beta": float(popt[3]),
            "stretched_R2": float(r_squared(y_full, y_hat)),
            "stretched_AIC": float(a),
        })
    else:
        out.update({"stretched_tau": np.nan, "stretched_beta": np.nan,
                    "stretched_R2": np.nan, "stretched_AIC": np.inf})

    if fit_l is not None:
        popt, y_hat, rss, a = fit_l
        out.update({
            "linear_k": float(popt[1]),
            "linear_R2": float(r_squared(y_full, y_hat)),
            "linear_AIC": float(a),
        })
    else:
        out.update({"linear_k": np.nan, "linear_R2": np.nan, "linear_AIC": np.inf})

    aics = {"exponential": out["exp_AIC"], "stretched": out["stretched_AIC"], "linear": out["linear_AIC"]}
    out["best_model"] = min(aics, key=aics.get)

    # Monotone recovery flag
    if len(rec_y) >= 3:
        diffs = np.diff(rec_y)
        if row["biomarker"] in ("MAP",):  # negative-sign biomarkers: should rise during recovery
            non_monotone = (diffs < -0.1 * np.std(rec_y)).any()
        else:
            non_monotone = (diffs > 0.1 * np.std(rec_y)).any()
        out["non_monotone"] = bool(non_monotone)
    else:
        out["non_monotone"] = False

    # Quality flag for tau
    tau_h = out.get("tau_hours")
    out["tau_at_bound"] = bool(np.isclose(tau_h, TAU_MIN_H) or np.isclose(tau_h, TAU_MAX_H))
    out["tau_passes_qc"] = bool(
        np.isfinite(out.get("exp_R2", np.nan)) and
        out["exp_R2"] >= R2_THRESHOLD and
        not out["tau_at_bound"]
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    eps_path = os.path.join(out_dir, "recovery_episodes.parquet")
    if not os.path.exists(eps_path):
        print(f"ERROR: {eps_path} not found. Run step2 first.")
        return 1

    banner("Step 3: recovery curve fitting")
    eps = pd.read_parquet(eps_path)
    print(f"  episodes: {len(eps):,}")

    if cfg["episodes"].get("exclude_deaths_from_recovery", True):
        before = len(eps)
        eps = eps[eps["survived"]]
        print(f"  dropping non-survivors: {before - len(eps):,} (left {len(eps):,})")

    fits = []
    bad = 0
    for i, row in eps.reset_index(drop=True).iterrows():
        fit = fit_one_episode(row)
        if fit is None:
            bad += 1
            continue
        fits.append({**fit, **{
            "age": float(row["age"]),
            "sex": row.get("sex"),
            "perturbation_type": row.get("perturbation_type"),
            "sofa_score": float(row.get("sofa_score") or np.nan),
            "los_days": float(row["los_days"]),
            "survived": bool(row["survived"]),
            "is_primary": bool(row["is_primary"]),
        }})
        if (i + 1) % 5000 == 0:
            print(f"    fit {i + 1:,}/{len(eps):,} (skipped {bad})")

    if not fits:
        print("ERROR: no fits succeeded.")
        return 1
    fit_df = pd.DataFrame(fits)
    out_path = os.path.join(out_dir, "recovery_fits.parquet")
    fit_df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(fit_df):,} fits, {bad:,} skipped)")

    # Summary
    summary = {
        "n_fits_total": int(len(fit_df)),
        "n_fits_pass_qc": int(fit_df["tau_passes_qc"].sum()),
        "fraction_qc_pass": float(fit_df["tau_passes_qc"].mean()),
        "median_tau_days_by_axis": {},
        "best_model_counts": {k: int(v) for k, v in fit_df["best_model"].value_counts().items()},
        "best_model_fraction": {k: float(v) for k, v in fit_df["best_model"].value_counts(normalize=True).items()},
    }
    qc = fit_df[fit_df["tau_passes_qc"]]
    for axis in cfg["axes"].keys():
        a = qc[(qc["axis"] == axis) & (qc["is_primary"])]
        if len(a) >= 3:
            summary["median_tau_days_by_axis"][axis] = {
                "median": float(a["tau_days"].median()),
                "iqr_lo": float(a["tau_days"].quantile(0.25)),
                "iqr_hi": float(a["tau_days"].quantile(0.75)),
                "n": int(len(a)),
            }
        else:
            summary["median_tau_days_by_axis"][axis] = {"median": None, "n": int(len(a))}

    with open(os.path.join(out_dir, "recovery_fit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {os.path.join(out_dir, 'recovery_fit_summary.json')}")

    print("\nMedian tau (days) by axis (primary biomarker, QC-passing fits):")
    for axis, s in summary["median_tau_days_by_axis"].items():
        if s.get("median") is not None:
            print(f"  {axis:<6}: median={s['median']:6.2f}d  IQR=[{s['iqr_lo']:.2f}, {s['iqr_hi']:.2f}]  n={s['n']}")
        else:
            print(f"  {axis:<6}: insufficient QC-passing fits (n={s['n']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
