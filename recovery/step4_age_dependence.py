#!/usr/bin/env python3
"""
Step 4 -- Primary HDR test: do recovery timescales lengthen with age?

For each axis, regress log(tau) on age, controlling for sex, perturbation_type,
and (if available) SOFA. Report:
  - beta (hours / year of age) for log(tau) and back-transformed tau
  - 95% bootstrap CIs
  - p-values
  - tau ratio (oldest stratum / youngest stratum)
  - per-stratum medians + bootstrap CIs

Also computes a multivariate recovery score: geometric mean of axis-tau per
patient (over patients with QC-passing fits in >= 2 axes).

Output
------
  results/tau_vs_age.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau, spearmanr

from hdr_core import (
    AXES,
    age_stratum,
    banner,
    bootstrap_ci,
    load_config,
    write_json,
)


def regress_log_tau_on_age(df: pd.DataFrame) -> dict:
    """OLS: log(tau_hours) ~ age + sex + perturbation_type [+ sofa_score]."""
    df = df.dropna(subset=["tau_hours", "age"]).copy()
    if len(df) < 30:
        return {"n": len(df), "note": "insufficient n"}
    df["log_tau"] = np.log(df["tau_hours"])
    X = pd.DataFrame({
        "age": df["age"].values,
    })
    if "sex" in df.columns:
        X["sex_M"] = (df["sex"] == "M").astype(float).values
    if "perturbation_type" in df.columns:
        for ptype in df["perturbation_type"].dropna().unique():
            if ptype != "general":
                X[f"ptype_{ptype}"] = (df["perturbation_type"] == ptype).astype(float).values
    if "sofa_score" in df.columns and df["sofa_score"].notna().any():
        X["sofa"] = df["sofa_score"].fillna(df["sofa_score"].median()).values

    X = sm.add_constant(X)
    y = df["log_tau"].values

    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception as e:
        return {"n": len(df), "error": str(e)}

    return {
        "n": int(len(df)),
        "beta_log_tau_per_year": float(model.params.get("age", np.nan)),
        "se_log_tau_per_year": float(model.bse.get("age", np.nan)),
        "ci95_log_tau": [
            float(model.conf_int().loc["age", 0]) if "age" in model.params.index else np.nan,
            float(model.conf_int().loc["age", 1]) if "age" in model.params.index else np.nan,
        ],
        "pvalue_age": float(model.pvalues.get("age", np.nan)),
        "r_squared": float(model.rsquared),
        # Convert to "tau increase per decade":
        "tau_fold_change_per_decade": float(np.exp(10 * model.params.get("age", 0.0))),
        "covariates": list(X.columns),
    }


def stratum_summary(df: pd.DataFrame, strata: list, n_boot: int, rng) -> list[dict]:
    out = []
    df = df.copy()
    df["stratum"] = df["age"].apply(lambda a: age_stratum(a, strata))
    for lo, hi in strata:
        key = f"{lo}-{hi}"
        sub = df[df["stratum"] == key]["tau_hours"].dropna()
        n = len(sub)
        if n >= 5:
            point, lo_ci, hi_ci = bootstrap_ci(sub.values, np.median, n_boot=n_boot, rng=rng)
            out.append({"stratum": key, "n": n,
                        "median_tau_h": point, "ci95_lo": lo_ci, "ci95_hi": hi_ci,
                        "median_tau_days": point / 24.0,
                        "ci95_lo_days": lo_ci / 24.0,
                        "ci95_hi_days": hi_ci / 24.0})
        else:
            out.append({"stratum": key, "n": n, "median_tau_h": None})
    return out


def compute_multivariate_tau(df: pd.DataFrame) -> pd.DataFrame:
    """
    Geometric mean of tau across axes per (subject_id, hadm_id).
    Restricted to QC-passing primary-biomarker fits in >= 2 axes.
    """
    qc = df[(df["tau_passes_qc"]) & (df["is_primary"])]
    grouped = qc.groupby(["subject_id", "hadm_id"]).agg(
        tau_geo_h=("tau_hours", lambda s: float(np.exp(np.mean(np.log(s.values))))),
        n_axes=("axis", "nunique"),
        age=("age", "first"),
        sex=("sex", "first"),
        perturbation_type=("perturbation_type", "first"),
        sofa_score=("sofa_score", "first"),
    ).reset_index()
    return grouped[grouped["n_axes"] >= 2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    fits_path = os.path.join(out_dir, "recovery_fits.parquet")
    if not os.path.exists(fits_path):
        print(f"ERROR: {fits_path} not found. Run step3 first.")
        return 1

    banner("Step 4: tau vs age")
    fits = pd.read_parquet(fits_path)
    qc = fits[fits["tau_passes_qc"] & fits["is_primary"]].copy()
    print(f"  total fits: {len(fits):,}  QC + primary: {len(qc):,}")

    rng = np.random.default_rng(cfg["analysis"]["seed"])
    strata = cfg["analysis"]["age_strata"]
    n_boot = cfg["analysis"]["n_bootstrap"]

    result = {"by_axis": {}, "multivariate": {}}

    for axis in cfg["axes"].keys():
        a = qc[qc["axis"] == axis]
        print(f"\n  axis={axis:<6}  n={len(a):,}")
        if len(a) < 30:
            print("    (insufficient n; skipping regression)")
            result["by_axis"][axis] = {"n": int(len(a)), "skipped": True}
            continue

        reg = regress_log_tau_on_age(a)
        ss = stratum_summary(a, strata, n_boot, rng)

        # Tau ratio old/young
        old_med = next((s["median_tau_h"] for s in ss if s["stratum"].startswith("80")), None)
        young_med = next((s["median_tau_h"] for s in ss
                          if s["stratum"].startswith("18")), None)
        ratio = (old_med / young_med) if old_med and young_med else None

        # Subgroups
        subgroups = {}
        for ptype in a["perturbation_type"].dropna().unique():
            sub = a[a["perturbation_type"] == ptype]
            if len(sub) >= 30:
                subgroups[ptype] = regress_log_tau_on_age(sub)
        for sex in ("M", "F"):
            sub = a[a["sex"] == sex]
            if len(sub) >= 30:
                subgroups[f"sex_{sex}"] = regress_log_tau_on_age(sub)

        result["by_axis"][axis] = {
            "regression": reg,
            "strata": ss,
            "tau_ratio_oldest_to_youngest": ratio,
            "subgroups": subgroups,
        }

        beta = reg.get("beta_log_tau_per_year", np.nan)
        ratio_str = f"{ratio:.2f}x" if ratio else "n/a"
        print(f"    beta={beta:.5f} log(h)/yr  p={reg.get('pvalue_age', float('nan')):.3g}  "
              f"fold/decade={reg.get('tau_fold_change_per_decade', float('nan')):.2f}x  "
              f"old/young={ratio_str}")

    # Multivariate
    print("\n  multivariate (geometric-mean tau across axes)")
    multi = compute_multivariate_tau(fits)
    print(f"    patients with >=2 QC axes: {len(multi):,}")
    if len(multi) >= 30:
        multi_renamed = multi.rename(columns={"tau_geo_h": "tau_hours"})
        reg = regress_log_tau_on_age(multi_renamed)
        ss = stratum_summary(multi_renamed, strata, n_boot, rng)
        old_med = next((s["median_tau_h"] for s in ss if s["stratum"].startswith("80")), None)
        young_med = next((s["median_tau_h"] for s in ss if s["stratum"].startswith("18")), None)
        ratio = (old_med / young_med) if old_med and young_med else None
        result["multivariate"] = {
            "n": int(len(multi)),
            "regression": reg,
            "strata": ss,
            "tau_ratio_oldest_to_youngest": ratio,
        }
        print(f"    beta={reg.get('beta_log_tau_per_year', float('nan')):.5f}  "
              f"old/young={ratio:.2f}x" if ratio else f"    beta={reg.get('beta_log_tau_per_year', float('nan')):.5f}")
    else:
        result["multivariate"] = {"n": int(len(multi)), "skipped": True}

    out_path = os.path.join(out_dir, "tau_vs_age.json")
    write_json(result, out_path)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
