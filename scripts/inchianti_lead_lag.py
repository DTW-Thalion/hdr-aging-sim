#!/usr/bin/env python3
"""
InCHIANTI 6-pair lead-lag analysis.

For each ordered pair (i, j) in {I, M, N, F}, regress Delta_x_j(t+1) on
Delta_x_i(t), controlling for Delta_x_j(t) and age. Tests directional
coupling against compiled J-matrix sign predictions.
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdr_sim.inchianti import (
    load_inchianti_panel, compute_youthful_reference, standardize_axes
)

N_BOOT = 5000
RNG = np.random.default_rng(42)
AXES = ["delta_I", "delta_M", "delta_N", "delta_F"]
AXIS_NAMES = {"delta_I": "I", "delta_M": "M", "delta_N": "N", "delta_F": "F"}

# Compiled J-matrix sign predictions (from manuscript Table 2)
# J[j,i] = sign of coupling from axis i to axis j
# Positive = axis i increasing drives axis j to increase
J_SIGNS = {
    ("I", "M"): +1,  # Inflammation drives insulin resistance
    ("M", "I"): +1,  # Insulin resistance drives inflammation
    ("I", "N"): +1,  # Inflammation impairs autonomic function
    ("N", "I"): -1,  # Poor autonomic function increases inflammation (via vagal withdrawal)
    ("I", "F"): +1,  # Inflammation reduces physical function
    ("F", "I"): +1,  # Poor function increases inflammation (via immobility)
    ("M", "N"): +1,  # Metabolic dysfunction impairs autonomic
    ("N", "M"): -1,  # Poor autonomic worsens metabolism (sign uncertain, using - for vagal)
    ("M", "F"): +1,  # Metabolic dysfunction reduces function
    ("F", "M"): +1,  # Poor function worsens metabolism (deconditioning)
    ("N", "F"): +1,  # Poor autonomic reduces exercise capacity
    ("F", "N"): +1,  # Poor function worsens autonomic (deconditioning)
}


def build_triplets(panel_std):
    """
    Build triplets of consecutive waves for lead-lag analysis.
    Returns DataFrame with (code98, wave_t, age_t, and standardized axes at t and t+1).
    """
    rows = []
    for subj, grp in panel_std.groupby("code98"):
        grp = grp.sort_values("wave")
        for i in range(len(grp) - 1):
            r0 = grp.iloc[i]
            r1 = grp.iloc[i + 1]
            # Both must have all 4 axes
            if any(pd.isna(r0[ax]) or pd.isna(r1[ax]) for ax in AXES):
                continue
            row = {"code98": subj, "wave_t": r0["wave"], "age_t": r0["age"]}
            for ax in AXES:
                row[f"{ax}_t0"] = r0[ax]
                row[f"{ax}_t1"] = r1[ax]
                row[f"d_{ax}"] = r1[ax] - r0[ax]
            rows.append(row)
    return pd.DataFrame(rows)


def cross_lagged_beta(triplets, from_ax, to_ax, n_boot=N_BOOT):
    """
    Compute cross-lagged regression coefficient:
    d_to_ax ~ beta * from_ax_t0 + gamma * to_ax_t0 + delta * age_t

    Returns beta, ci_lower, ci_upper, p_value, n.
    """
    y = triplets[f"d_{to_ax}"].values
    X_from = triplets[f"{from_ax}_t0"].values
    X_auto = triplets[f"{to_ax}_t0"].values
    X_age = triplets["age_t"].values

    # Remove NaN rows
    valid = np.isfinite(y) & np.isfinite(X_from) & np.isfinite(X_auto) & np.isfinite(X_age)
    y, X_from, X_auto, X_age = y[valid], X_from[valid], X_auto[valid], X_age[valid]
    n = len(y)

    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, n

    # OLS regression
    X = np.column_stack([np.ones(n), X_from, X_auto, X_age])
    try:
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, n

    beta_cross = beta_hat[1]  # coefficient of from_ax

    # Residual-based p-value
    y_hat = X @ beta_hat
    resid = y - y_hat
    sigma2 = np.sum(resid**2) / (n - 4)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * XtX_inv[1, 1])
        t_stat = beta_cross / se
        from scipy.stats import t as t_dist
        p_val = 2 * t_dist.sf(abs(t_stat), df=n - 4)
    except:
        se = np.nan
        p_val = np.nan

    # Bootstrap CI
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        try:
            b_hat = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            boots[b] = b_hat[1]
        except:
            boots[b] = np.nan
    ci_lo = float(np.nanpercentile(boots, 2.5))
    ci_hi = float(np.nanpercentile(boots, 97.5))

    return float(beta_cross), ci_lo, ci_hi, float(p_val), n


def main():
    print("=" * 60)
    print("InCHIANTI: 6-Pair Lead-Lag Analysis")
    print("=" * 60)

    panel = load_inchianti_panel()
    ref = compute_youthful_reference(panel)
    panel_std = standardize_axes(panel, ref)

    triplets = build_triplets(panel_std)
    print(f"Total consecutive-wave pairs with 4-axis data: {len(triplets)}")

    results = []
    n_concordant = 0
    n_total = 0

    print(f"\n{'Pair':<10s} {'beta':>8s} {'95% CI':>20s} {'p':>10s} {'N':>6s} {'Pred':>6s} {'Obs':>6s} {'Match':>6s}")
    print("-" * 80)

    for (to_name, from_name), predicted_sign in J_SIGNS.items():
        from_ax = f"delta_{from_name}"
        to_ax = f"delta_{to_name}"

        beta, ci_lo, ci_hi, p_val, n = cross_lagged_beta(triplets, from_ax, to_ax)

        if np.isnan(beta):
            obs_sign = 0
            match = "N/A"
        else:
            obs_sign = 1 if beta > 0 else -1
            match = "YES" if obs_sign == predicted_sign else "NO"
            n_total += 1
            if obs_sign == predicted_sign:
                n_concordant += 1

        pair_str = f"{from_name}->{to_name}"
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if not np.isnan(ci_lo) else "N/A"
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        pred_str = "+" if predicted_sign > 0 else "-"
        obs_str = "+" if obs_sign > 0 else ("-" if obs_sign < 0 else "?")

        print(f"  {pair_str:<8s} {beta:>8.4f} {ci_str:>20s} {p_str:>10s} {n:>6d} {pred_str:>6s} {obs_str:>6s} {match:>6s}")

        results.append({
            "from_axis": from_name, "to_axis": to_name,
            "pair": pair_str,
            "beta": beta, "ci_lower": ci_lo, "ci_upper": ci_hi,
            "p_value": p_val, "n": n,
            "predicted_sign": predicted_sign,
            "observed_sign": int(obs_sign),
            "concordant": match == "YES",
        })

    # Sign concordance test
    if n_total > 0:
        p_binom = binomtest(n_concordant, n_total, 0.5, alternative="greater").pvalue
        print(f"\nSign concordance: {n_concordant}/{n_total} pairs ({n_concordant/n_total*100:.0f}%)")
        print(f"Binomial test (vs chance 50%): p = {p_binom:.4f}")
    else:
        p_binom = np.nan

    # Save
    os.makedirs("results", exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv("results/inchianti_lead_lag_matrix.csv", index=False)

    summary = {
        "description": "InCHIANTI 6-pair lead-lag cross-lagged regression",
        "n_pairs_total": len(triplets),
        "n_concordant": n_concordant,
        "n_tested": n_total,
        "concordance_rate": n_concordant / n_total if n_total > 0 else None,
        "binomial_p": float(p_binom) if not np.isnan(p_binom) else None,
        "results": results,
    }
    with open("results/inchianti_lead_lag_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to results/inchianti_lead_lag_matrix.csv")


if __name__ == "__main__":
    main()
