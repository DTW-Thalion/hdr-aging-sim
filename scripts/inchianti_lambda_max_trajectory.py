#!/usr/bin/env python3
"""
InCHIANTI 4-axis lambda_max trajectory by age stratum.

Computes the largest eigenvalue of the change-covariance matrix Gamma_change
for within-person consecutive-wave changes in (delta_I, delta_M, delta_N, delta_F).

Bootstrap 95% CIs on lambda_max per age stratum.
Also computes cross-sectional Gamma for comparison.
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

AGE_STRATA = {
    "20-49": (20, 49),
    "50-59": (50, 59),
    "60-69": (60, 69),
    "70-79": (70, 79),
    "80+":   (80, 120),
}


def lambda_max_of_cov(X):
    """Compute largest eigenvalue of the sample covariance of X (n x p)."""
    if len(X) < 3:
        return np.nan
    C = np.cov(X, rowvar=False)
    return float(np.max(eigvalsh(C)))


def bootstrap_lambda_max(X, n_boot=N_BOOT):
    """Bootstrap CI for lambda_max of covariance."""
    n = len(X)
    if n < 5:
        return np.nan, np.nan
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boots[i] = lambda_max_of_cov(X[idx])
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def compute_change_vectors(panel_std):
    """
    Compute within-person visit-pair changes for 4 axes.
    Returns DataFrame with (code98, wave_t, age_t, change vector).
    """
    rows = []
    for subj, grp in panel_std.groupby("code98"):
        grp = grp.sort_values("wave")
        for i in range(len(grp) - 1):
            r0 = grp.iloc[i]
            r1 = grp.iloc[i + 1]
            # Both rows must have all 4 axes
            if any(pd.isna(r0[ax]) or pd.isna(r1[ax]) for ax in AXES):
                continue
            change = {ax: r1[ax] - r0[ax] for ax in AXES}
            change["code98"] = subj
            change["wave_t"] = r0["wave"]
            change["age_t"] = r0["age"]
            change["age_mid"] = (r0["age"] + r1["age"]) / 2
            rows.append(change)
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("InCHIANTI: 4-Axis Lambda_max Trajectory")
    print("=" * 60)

    panel = load_inchianti_panel()
    ref = compute_youthful_reference(panel)
    panel_std = standardize_axes(panel, ref)

    # ── Change-covariance lambda_max by age stratum ──
    print("\n--- Change-covariance analysis ---")
    changes = compute_change_vectors(panel_std)
    print(f"Total change-pairs: {len(changes)}")

    results = []
    for name, (lo, hi) in AGE_STRATA.items():
        mask = (changes["age_mid"] >= lo) & (changes["age_mid"] <= hi)
        sub = changes.loc[mask, AXES].values
        n = len(sub)
        lmax = lambda_max_of_cov(sub)
        ci_lo, ci_hi = bootstrap_lambda_max(sub)
        print(f"  {name:8s}: N={n:5d}, lambda_max={lmax:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        results.append({
            "age_stratum": name, "type": "change_covariance",
            "n_pairs": n, "lambda_max": lmax,
            "ci_lower": ci_lo, "ci_upper": ci_hi,
        })

    # ── Cross-sectional Gamma lambda_max by age stratum ──
    print("\n--- Cross-sectional covariance (expected: decrease with age due to survivorship) ---")
    for name, (lo, hi) in AGE_STRATA.items():
        mask = (panel_std["age"] >= lo) & (panel_std["age"] <= hi)
        sub = panel_std.loc[mask, AXES].dropna().values
        n = len(sub)
        lmax = lambda_max_of_cov(sub)
        ci_lo, ci_hi = bootstrap_lambda_max(sub)
        print(f"  {name:8s}: N={n:5d}, lambda_max={lmax:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        results.append({
            "age_stratum": name, "type": "cross_sectional",
            "n_pairs": n, "lambda_max": lmax,
            "ci_lower": ci_lo, "ci_upper": ci_hi,
        })

    # ── Medication-naive subgroup ──
    print("\n--- Change-covariance: medication-naive subgroup ---")
    naive_mask = panel_std["n_med_classes"] == 0
    panel_naive = panel_std[naive_mask]
    changes_naive = compute_change_vectors(panel_naive)
    print(f"Naive change-pairs: {len(changes_naive)}")
    for name, (lo, hi) in AGE_STRATA.items():
        mask = (changes_naive["age_mid"] >= lo) & (changes_naive["age_mid"] <= hi)
        sub = changes_naive.loc[mask, AXES].values
        n = len(sub)
        if n < 5:
            print(f"  {name:8s}: N={n} (too few)")
            continue
        lmax = lambda_max_of_cov(sub)
        ci_lo, ci_hi = bootstrap_lambda_max(sub)
        print(f"  {name:8s}: N={n:5d}, lambda_max={lmax:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        results.append({
            "age_stratum": name, "type": "change_naive",
            "n_pairs": n, "lambda_max": lmax,
            "ci_lower": ci_lo, "ci_upper": ci_hi,
        })

    # Save
    os.makedirs("results", exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv("results/inchianti_lambda_max_by_age.csv", index=False)
    print(f"\nSaved to results/inchianti_lambda_max_by_age.csv")

    # JSON summary
    summary = {
        "description": "InCHIANTI 4-axis lambda_max trajectory",
        "axes": ["I (log IL-6)", "M (log HOMA-IR)", "N (resting HR)", "F (SPPB, sign-flipped)"],
        "n_change_pairs_total": len(changes),
        "n_change_pairs_naive": len(changes_naive),
        "results": results,
    }
    with open("results/inchianti_lambda_max_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Check monotonic increase in change-covariance lambda_max
    change_results = [r for r in results if r["type"] == "change_covariance" and not np.isnan(r["lambda_max"])]
    lmax_vals = [r["lambda_max"] for r in change_results]
    monotonic = all(lmax_vals[i] <= lmax_vals[i+1] for i in range(len(lmax_vals)-1))
    print(f"\nMonotonic increase in change-cov lambda_max: {monotonic}")
    print("Change-cov lambda_max values:", [f"{v:.4f}" for v in lmax_vals])


if __name__ == "__main__":
    main()
