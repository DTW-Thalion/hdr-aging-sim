#!/usr/bin/env python3
"""
InCHIANTI medication dose-response stratification.

Tests whether medication compression is genuine or confounding by indication:
1. Lambda_max(Gamma_change) by medication class count (0, 1, 2, 3+)
2. Regression: lambda_max ~ n_meds + n_comorbidities + age + sex
3. Within-hypertension comparison: treated vs untreated
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdr_sim.inchianti import (
    load_inchianti_panel, compute_youthful_reference, standardize_axes
)

N_BOOT = 10_000
RNG = np.random.default_rng(42)
AXES = ["delta_I", "delta_M", "delta_N", "delta_F"]


def lambda_max_of_cov(X):
    if len(X) < 3:
        return np.nan
    C = np.cov(X, rowvar=False)
    return float(np.max(eigvalsh(C)))


def bootstrap_lambda_max(X, n_boot=N_BOOT):
    n = len(X)
    if n < 5:
        return np.nan, np.nan
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boots[i] = lambda_max_of_cov(X[idx])
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def compute_change_vectors(panel_std):
    rows = []
    for subj, grp in panel_std.groupby("code98"):
        grp = grp.sort_values("wave")
        for i in range(len(grp) - 1):
            r0, r1 = grp.iloc[i], grp.iloc[i + 1]
            if any(pd.isna(r0[ax]) or pd.isna(r1[ax]) for ax in AXES):
                continue
            change = {ax: r1[ax] - r0[ax] for ax in AXES}
            change["code98"] = subj
            change["age_t"] = r0["age"]
            # Carry baseline medication count and comorbidity count
            change["n_med_classes"] = r0.get("n_med_classes", np.nan)
            change["n_comorbidities"] = r0.get("n_comorbidities", np.nan)
            change["sex"] = r0.get("sex", np.nan)
            change["dx_htn"] = r0.get("dx_htn", np.nan)
            change["med_antihtn"] = r0.get("med_antihtn", np.nan)
            rows.append(change)
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("InCHIANTI: Medication Dose-Response Stratification")
    print("=" * 60)

    panel = load_inchianti_panel()
    ref = compute_youthful_reference(panel)
    panel_std = standardize_axes(panel, ref)
    changes = compute_change_vectors(panel_std)
    print(f"Total change-pairs: {len(changes)}")

    results = []

    # ── 1. Lambda_max by medication class count ──
    print("\n--- Lambda_max by medication class count ---")
    for n_med in [0, 1, 2, "3+"]:
        if n_med == "3+":
            mask = changes["n_med_classes"] >= 3
        else:
            mask = changes["n_med_classes"] == n_med
        sub = changes.loc[mask, AXES].values
        n = len(sub)
        lmax = lambda_max_of_cov(sub)
        ci_lo, ci_hi = bootstrap_lambda_max(sub)
        print(f"  {n_med} classes: N={n:5d}, lambda_max={lmax:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        results.append({
            "group": f"med_{n_med}", "n": n,
            "lambda_max": lmax, "ci_lower": ci_lo, "ci_upper": ci_hi,
        })

    # ── 2. Regression: per-subject SWDS-like score ──
    # For each subject, compute their squared Mahalanobis distance contribution
    # Then regress on n_meds + n_comorbidities + age + sex
    print("\n--- Regression: individual coupling contribution ---")
    # Use squared change magnitude as individual-level proxy
    changes["change_magnitude"] = np.sqrt(sum(changes[ax]**2 for ax in AXES))

    reg_mask = changes[["change_magnitude", "n_med_classes", "n_comorbidities", "age_t", "sex"]].notna().all(axis=1)
    reg = changes[reg_mask].copy()
    n_reg = len(reg)
    print(f"Regression N = {n_reg}")

    if n_reg > 30:
        y = reg["change_magnitude"].values
        X = np.column_stack([
            np.ones(n_reg),
            reg["n_med_classes"].values,
            reg["n_comorbidities"].values,
            reg["age_t"].values,
            (reg["sex"] == 2).astype(float).values,
        ])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat
        sigma2 = np.sum(resid**2) / (n_reg - 5)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(XtX_inv))

        names = ["intercept", "n_med_classes", "n_comorbidities", "age", "female"]
        reg_results = {}
        for i, name in enumerate(names):
            t_stat = beta[i] / se[i]
            from scipy.stats import t as t_dist
            p_val = 2 * t_dist.sf(abs(t_stat), df=n_reg - 5)
            print(f"  {name:20s}: beta={beta[i]:>8.4f}, SE={se[i]:.4f}, t={t_stat:.2f}, p={p_val:.4f}")
            reg_results[name] = {
                "beta": float(beta[i]), "se": float(se[i]),
                "t": float(t_stat), "p": float(p_val),
            }

    # ── 3. Within-hypertension comparison ──
    print("\n--- Within-hypertension: treated vs untreated ---")
    htn_mask = changes["dx_htn"] == 1
    htn_changes = changes[htn_mask]
    print(f"Hypertensive subjects with change data: {len(htn_changes)}")

    for label, med_val in [("treated", 1), ("untreated", 0)]:
        mask = htn_changes["med_antihtn"] == med_val
        sub = htn_changes.loc[mask, AXES].values
        n = len(sub)
        lmax = lambda_max_of_cov(sub)
        ci_lo, ci_hi = bootstrap_lambda_max(sub)
        print(f"  {label:12s}: N={n:5d}, lambda_max={lmax:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        results.append({
            "group": f"htn_{label}", "n": n,
            "lambda_max": lmax, "ci_lower": ci_lo, "ci_upper": ci_hi,
        })

    # Save
    os.makedirs("results", exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv("results/inchianti_med_dose_response.csv", index=False)

    summary = {
        "description": "InCHIANTI medication dose-response analysis",
        "n_change_pairs": len(changes),
        "stratified_results": results,
        "regression": reg_results if n_reg > 30 else None,
    }
    with open("results/inchianti_med_dose_response.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to results/inchianti_med_dose_response.csv")


if __name__ == "__main__":
    main()
