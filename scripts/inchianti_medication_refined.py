#!/usr/bin/env python3
"""
Refined InCHIANTI medication dose-response analysis.

Four sub-analyses testing whether medication compression is genuine:
  1a. Age-stratified lambda_max: medicated vs unmedicated within each decade
  1b. SWDS-Gamma individual-level regression
  1c. Age-matched within-hypertension comparison
  1d. Off-diagonal correlation test (direct compression hypothesis)
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.stats import t as t_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdr_sim.inchianti import (
    load_inchianti_panel, compute_youthful_reference, standardize_axes
)

N_BOOT = 10_000
RNG = np.random.default_rng(42)
AXES = ["delta_I", "delta_M", "delta_N", "delta_F"]
AGE_DECADES = {"50-59": (50, 59), "60-69": (60, 69), "70-79": (70, 79), "80+": (80, 120)}


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


def mean_abs_offdiag_corr(X):
    """Compute mean absolute off-diagonal correlation from data matrix X (n x p)."""
    if len(X) < 5:
        return np.nan
    C = np.corrcoef(X, rowvar=False)
    p = C.shape[0]
    vals = []
    for i in range(p):
        for j in range(i + 1, p):
            if np.isfinite(C[i, j]):
                vals.append(abs(C[i, j]))
    return float(np.mean(vals)) if vals else np.nan


def compute_swds_gamma(delta_x, Gamma):
    """
    SWDS-Gamma for a single change vector.
    SWDS_i = sum_k lambda_k * (v_k^T delta_x_i)^2 / sum_j lambda_j
    """
    eigvals, eigvecs = np.linalg.eigh(Gamma)
    eigvals = np.maximum(eigvals, 0)  # clip tiny negatives
    denom = eigvals.sum()
    if denom == 0:
        return np.nan
    projections = eigvecs.T @ delta_x  # (p,) projections onto eigenvectors
    return float(np.sum(eigvals * projections**2) / denom)


def compute_change_vectors(panel_std):
    """Build change vectors with all metadata for medication analysis."""
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
            change["age_mid"] = (r0["age"] + r1["age"]) / 2
            change["n_med_classes"] = r0.get("n_med_classes", np.nan)
            change["n_comorbidities"] = r0.get("n_comorbidities", np.nan)
            change["sex"] = r0.get("sex", np.nan)
            change["dx_htn"] = r0.get("dx_htn", np.nan)
            change["med_antihtn"] = r0.get("med_antihtn", np.nan)
            change["medicated"] = 1 if r0.get("n_med_classes", 0) >= 1 else 0
            rows.append(change)
    return pd.DataFrame(rows)


def ols_regression(y, X, var_names):
    """Run OLS, return dict of {name: {beta, se, t, p}}."""
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    sigma2 = np.sum(resid**2) / (n - k)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return {name: {"beta": float(beta[i]), "se": np.nan, "t": np.nan, "p": np.nan}
                for i, name in enumerate(var_names)}
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    results = {}
    for i, name in enumerate(var_names):
        t_stat = beta[i] / se[i] if se[i] > 0 else np.nan
        p_val = 2 * t_dist.sf(abs(t_stat), df=n - k) if np.isfinite(t_stat) else np.nan
        results[name] = {
            "beta": float(beta[i]), "se": float(se[i]),
            "t": float(t_stat), "p": float(p_val),
        }
    return results


def main():
    print("=" * 60)
    print("InCHIANTI: Refined Medication Dose-Response Analysis")
    print("=" * 60)

    panel = load_inchianti_panel()
    ref = compute_youthful_reference(panel)
    panel_std = standardize_axes(panel, ref)
    changes = compute_change_vectors(panel_std)
    print(f"Total change-pairs: {len(changes)}")

    all_results = {}

    # ============================================================══
    # 1a. Age-stratified lambda_max: medicated vs unmedicated
    # ============================================================══
    print("\n=== 1a. Age-stratified lambda_max ===")
    strat_results = []
    for decade, (lo, hi) in AGE_DECADES.items():
        mask_age = (changes["age_mid"] >= lo) & (changes["age_mid"] <= hi)
        for label, med_mask in [("unmedicated", changes["medicated"] == 0),
                                ("medicated", changes["medicated"] == 1)]:
            sub = changes.loc[mask_age & med_mask, AXES].values
            n = len(sub)
            lmax = lambda_max_of_cov(sub)
            ci_lo, ci_hi = bootstrap_lambda_max(sub)
            print(f"  {decade} {label:12s}: N={n:4d}  lmax={lmax:8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
            strat_results.append({
                "decade": decade, "group": label, "n": n,
                "lambda_max": lmax, "ci_lower": ci_lo, "ci_upper": ci_hi,
            })
    all_results["1a_age_stratified"] = strat_results

    # ============================================================══
    # 1b. SWDS-Gamma individual-level regression
    # ============================================================══
    print("\n=== 1b. SWDS-Gamma individual regression ===")
    # Compute population Gamma from all change vectors
    all_change_vals = changes[AXES].dropna().values
    Gamma_pop = np.cov(all_change_vals, rowvar=False)

    # Compute per-individual SWDS-Gamma
    swds_vals = []
    for _, row in changes.iterrows():
        dx = row[AXES].values.astype(float)
        if np.any(np.isnan(dx)):
            swds_vals.append(np.nan)
        else:
            swds_vals.append(compute_swds_gamma(dx, Gamma_pop))
    changes["swds_gamma"] = swds_vals

    # Regression: SWDS-Gamma ~ n_meds + n_comorbidities + age + sex
    reg_cols = ["swds_gamma", "n_med_classes", "n_comorbidities", "age_t", "sex"]
    reg = changes.dropna(subset=reg_cols).copy()
    n_reg = len(reg)
    print(f"  Regression N = {n_reg}")

    if n_reg > 30:
        y = reg["swds_gamma"].values
        X = np.column_stack([
            np.ones(n_reg),
            reg["n_med_classes"].values,
            reg["n_comorbidities"].values,
            reg["age_t"].values,
            (reg["sex"] == 2).astype(float).values,
        ])
        var_names = ["intercept", "n_med_classes", "n_comorbidities", "age", "female"]
        reg_results = ols_regression(y, X, var_names)
        for name, r in reg_results.items():
            print(f"    {name:20s}: beta={r['beta']:>8.4f}, SE={r['se']:.4f}, "
                  f"t={r['t']:.2f}, p={r['p']:.4f}")
        all_results["1b_swds_regression"] = reg_results
        all_results["1b_swds_regression"]["_n"] = n_reg

    # ============================================================══
    # 1c. Age-matched within-hypertension comparison
    # ============================================================══
    print("\n=== 1c. Within-hypertension age-matched comparison ===")
    htn = changes[changes["dx_htn"] == 1].copy()
    treated = htn[htn["med_antihtn"] == 1].copy()
    untreated = htn[htn["med_antihtn"] == 0].copy()
    print(f"  Hypertensive: {len(htn)} total, {len(treated)} treated, {len(untreated)} untreated")

    # Nearest-neighbour age matching without replacement
    matched_t = []
    matched_u = []
    used_u = set()
    for _, t_row in treated.iterrows():
        best_dist = 999
        best_idx = None
        for u_idx, u_row in untreated.iterrows():
            if u_idx in used_u:
                continue
            dist = abs(t_row["age_t"] - u_row["age_t"])
            if dist < best_dist and dist <= 5:
                best_dist = dist
                best_idx = u_idx
        if best_idx is not None:
            matched_t.append(t_row)
            matched_u.append(untreated.loc[best_idx])
            used_u.add(best_idx)

    n_matched = len(matched_t)
    print(f"  Matched pairs (±5 yr): {n_matched}")

    htn_match_results = {"n_matched": n_matched}
    if n_matched >= 20:
        mt = pd.DataFrame(matched_t)
        mu = pd.DataFrame(matched_u)
        print(f"  Treated age: {mt['age_t'].mean():.1f} ± {mt['age_t'].std():.1f}")
        print(f"  Untreated age: {mu['age_t'].mean():.1f} ± {mu['age_t'].std():.1f}")

        lmax_t = lambda_max_of_cov(mt[AXES].values)
        ci_t = bootstrap_lambda_max(mt[AXES].values)
        lmax_u = lambda_max_of_cov(mu[AXES].values)
        ci_u = bootstrap_lambda_max(mu[AXES].values)

        print(f"  Treated   lmax = {lmax_t:.4f} [{ci_t[0]:.4f}, {ci_t[1]:.4f}]")
        print(f"  Untreated lmax = {lmax_u:.4f} [{ci_u[0]:.4f}, {ci_u[1]:.4f}]")

        htn_match_results.update({
            "treated_age_mean": float(mt["age_t"].mean()),
            "untreated_age_mean": float(mu["age_t"].mean()),
            "treated_lambda_max": lmax_t, "treated_ci": list(ci_t),
            "untreated_lambda_max": lmax_u, "untreated_ci": list(ci_u),
            "delta_lambda_max": lmax_t - lmax_u,
        })
    else:
        # Fall back to age-adjusted regression within HTN subgroup
        print("  Too few matched pairs; using age-adjusted regression instead")
        if len(htn) > 30:
            y = np.sqrt(sum(htn[ax]**2 for ax in AXES)).values
            X = np.column_stack([
                np.ones(len(htn)),
                htn["med_antihtn"].values,
                htn["age_t"].values,
            ])
            reg_htn = ols_regression(y, X, ["intercept", "med_antihtn", "age"])
            for name, r in reg_htn.items():
                print(f"    {name}: beta={r['beta']:.4f}, p={r['p']:.4f}")
            htn_match_results["regression_fallback"] = reg_htn

    all_results["1c_htn_matched"] = htn_match_results

    # ============================================================══
    # 1d. Off-diagonal correlation test
    # ============================================================══
    print("\n=== 1d. Off-diagonal correlation test ===")
    offdiag_results = []
    for decade, (lo, hi) in AGE_DECADES.items():
        mask_age = (changes["age_mid"] >= lo) & (changes["age_mid"] <= hi)
        for label, med_mask in [("unmedicated", changes["medicated"] == 0),
                                ("medicated", changes["medicated"] == 1)]:
            sub = changes.loc[mask_age & med_mask, AXES].dropna().values
            n = len(sub)
            r_mean = mean_abs_offdiag_corr(sub)

            # Bootstrap CI
            if n >= 10:
                boots = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idx = RNG.integers(0, n, size=n)
                    boots[b] = mean_abs_offdiag_corr(sub[idx])
                ci_lo = float(np.nanpercentile(boots, 2.5))
                ci_hi = float(np.nanpercentile(boots, 97.5))
            else:
                ci_lo = ci_hi = np.nan

            print(f"  {decade} {label:12s}: N={n:4d}  mean|r|={r_mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
            offdiag_results.append({
                "decade": decade, "group": label, "n": n,
                "mean_abs_corr": r_mean, "ci_lower": ci_lo, "ci_upper": ci_hi,
            })
    all_results["1d_offdiag_corr"] = offdiag_results

    # Save all results
    os.makedirs("results", exist_ok=True)
    with open("results/inchianti_med_refined_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to results/inchianti_med_refined_results.json")

    # Also save SWDS-Gamma values for figure generation
    changes[["code98", "age_t", "age_mid", "n_med_classes", "medicated",
             "swds_gamma", "dx_htn", "med_antihtn"]].to_csv(
        "results/inchianti_swds_gamma_individual.csv", index=False)

    return all_results


if __name__ == "__main__":
    main()
