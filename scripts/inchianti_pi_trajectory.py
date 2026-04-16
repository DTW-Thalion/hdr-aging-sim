#!/usr/bin/env python3
"""
InCHIANTI independent Pi trajectory analysis.

Replicates SI Note 6: Pi = C_norm / V_norm test for proportional
co-degradation vs D-dominated vs J-dominated aging.

Pi > 1 increasing with age → J-dominated (coupling grows faster than variance)
Pi ~ 1 constant → proportional co-degradation
Pi < 1 decreasing → D-dominated
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdr_sim.inchianti import (
    load_inchianti_panel, compute_youthful_reference, standardize_axes
)

N_BOOT = 5000
RNG = np.random.default_rng(42)
AXES = ["delta_I", "delta_M", "delta_N", "delta_F"]


def compute_pi(X):
    """Compute Pi = C_norm / V_norm for a group of standardized axis values."""
    if len(X) < 10:
        return np.nan, np.nan, np.nan
    C = np.cov(X, rowvar=False)
    n = C.shape[0]
    V = np.mean(np.diag(C))  # mean variance
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(abs(C[i, j]))
    C_mean = np.mean(off_diag)  # mean absolute off-diagonal
    return V, C_mean, C_mean / V if V > 0 else np.nan


def main():
    print("=" * 60)
    print("InCHIANTI: Pi Trajectory (C_norm / V_norm)")
    print("=" * 60)

    panel = load_inchianti_panel()
    ref = compute_youthful_reference(panel)
    panel_std = standardize_axes(panel, ref)

    # Use cross-sectional data (more complete than change vectors)
    complete = panel_std.dropna(subset=AXES)
    print(f"4-axis complete observations: {len(complete)}")

    # Age strata for Pi computation
    strata = {
        "20-49": (20, 49),
        "50-59": (50, 59),
        "60-69": (60, 69),
        "70-79": (70, 79),
        "80+":   (80, 120),
    }

    # Reference = youngest stratum
    ref_name = "20-49"
    ref_lo, ref_hi = strata[ref_name]
    ref_data = complete[(complete["age"] >= ref_lo) & (complete["age"] <= ref_hi)][AXES].values
    V_ref, C_ref, Pi_ref = compute_pi(ref_data)

    results = []
    print(f"\n{'Stratum':<10s} {'N':>6s} {'V_norm':>8s} {'C_norm':>8s} {'Pi':>8s}")
    print("-" * 50)

    for name, (lo, hi) in strata.items():
        mask = (complete["age"] >= lo) & (complete["age"] <= hi)
        X = complete.loc[mask, AXES].values
        n = len(X)
        V, C_mean, Pi = compute_pi(X)

        # Normalize to reference
        V_norm = V / V_ref if V_ref > 0 else np.nan
        C_norm = C_mean / C_ref if C_ref > 0 else np.nan
        Pi_norm = C_norm / V_norm if V_norm > 0 else np.nan

        # Bootstrap CI on Pi_norm
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = RNG.integers(0, n, size=n)
            _, _, pi_b = compute_pi(X[idx])
            # Normalize to reference
            idx_ref = RNG.integers(0, len(ref_data), size=len(ref_data))
            _, C_ref_b, _ = compute_pi(ref_data[idx_ref])
            V_ref_b, _, _ = compute_pi(ref_data[idx_ref])
            V_b, C_b, _ = compute_pi(X[idx])
            V_n = V_b / V_ref_b if V_ref_b > 0 else np.nan
            C_n = C_b / C_ref_b if C_ref_b > 0 else np.nan
            boots[b] = C_n / V_n if V_n > 0 else np.nan
        ci_lo = float(np.nanpercentile(boots, 2.5))
        ci_hi = float(np.nanpercentile(boots, 97.5))

        print(f"  {name:<8s} {n:>6d} {V_norm:>8.3f} {C_norm:>8.3f} {Pi_norm:>8.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

        results.append({
            "age_stratum": name, "n": n,
            "V_raw": float(V), "C_raw": float(C_mean),
            "V_norm": float(V_norm), "C_norm": float(C_norm),
            "Pi_norm": float(Pi_norm),
            "Pi_ci_lower": ci_lo, "Pi_ci_upper": ci_hi,
        })

    # Fit linear trend: Pi_norm vs age midpoint
    ages_mid = [35, 55, 65, 75, 85]
    pi_vals = [r["Pi_norm"] for r in results]
    valid = [(a, p) for a, p in zip(ages_mid, pi_vals) if np.isfinite(p)]
    if len(valid) >= 3:
        a_arr = np.array([v[0] for v in valid])
        p_arr = np.array([v[1] for v in valid])
        slope, intercept = np.polyfit(a_arr, p_arr, 1)
        print(f"\nPi-slope per year: {slope:.6f}")
        print(f"ELSA medication-naive Pi-slope: +0.0014/yr")
        print(f"InCHIANTI Pi-slope: {slope:.4f}/yr")
        trend = "J-dominated" if slope > 0.0005 else ("D-dominated" if slope < -0.0005 else "proportional")
        print(f"Interpretation: {trend}")
    else:
        slope = np.nan
        trend = "insufficient data"

    # Medication-naive subgroup
    print("\n--- Medication-naive subgroup ---")
    naive = complete[complete["n_med_classes"] == 0]
    print(f"Naive 4-axis complete: {len(naive)}")
    naive_results = []
    for name, (lo, hi) in strata.items():
        mask = (naive["age"] >= lo) & (naive["age"] <= hi)
        X = naive.loc[mask, AXES].values
        n = len(X)
        if n < 10:
            print(f"  {name}: N={n} (too few)")
            continue
        V, C_mean, Pi = compute_pi(X)
        V_norm = V / V_ref if V_ref > 0 else np.nan
        C_norm = C_mean / C_ref if C_ref > 0 else np.nan
        Pi_norm = C_norm / V_norm if V_norm > 0 else np.nan
        print(f"  {name:<8s}: N={n:5d}, Pi_norm={Pi_norm:.3f}")
        naive_results.append({"age_stratum": name, "n": n, "Pi_norm": float(Pi_norm)})

    # Save
    os.makedirs("results", exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv("results/inchianti_pi_trajectory.csv", index=False)

    summary = {
        "description": "InCHIANTI Pi trajectory (C_norm/V_norm)",
        "reference_stratum": ref_name,
        "pi_slope_per_year": float(slope) if np.isfinite(slope) else None,
        "interpretation": trend,
        "elsa_naive_slope": 0.0014,
        "full_sample": results,
        "naive_subgroup": naive_results,
    }
    with open("results/inchianti_pi_trajectory.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to results/inchianti_pi_trajectory.csv")


if __name__ == "__main__":
    main()
