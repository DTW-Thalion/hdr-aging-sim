#!/usr/bin/env python3
"""
InCHIANTI nonlinearity robustness test for cross-lagged regressions.

For each of the 12 ordered pairs (i -> j) in {I, M, N, F}, fit four OLS
models predicting Delta_x_j(t+1):

  M0 (linear):      d_j ~ x_i + x_j + age + const
  M1 (quadratic):   d_j ~ x_i + x_i^2 + x_j + age + const
  M2 (interaction): d_j ~ x_i + x_i*x_j + x_j + age + const
  M3 (both):        d_j ~ x_i + x_i^2 + x_i*x_j + x_j + age + const

All models use HC3 heteroscedasticity-robust standard errors.  For each
pair we report the quadratic coefficient (beta_1) and the interaction
coefficient (eta) with two-sided p-values, a joint Wald F-test of
M3 vs M0 (both nonlinear terms = 0), and residual diagnostics on M0
(regression of residuals on x_i^2, and a normality test).

Bonferroni threshold: 12 pairs x 2 nonlinear terms = 24 tests,
significance threshold p < 0.05/24 ~= 0.002083.

Output:
  results/nonlinearity_test.json  -- machine-readable results
  stdout                          -- summary table + LaTeX paragraph
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro, normaltest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdr_sim.inchianti import (
    load_inchianti_panel, compute_youthful_reference, standardize_axes
)

AXES = ["delta_I", "delta_M", "delta_N", "delta_F"]
AXIS_NAMES = {"delta_I": "I", "delta_M": "M", "delta_N": "N", "delta_F": "F"}

# All 12 ordered (from, to) pairs from the lead-lag analysis.
ORDERED_PAIRS = [
    ("I", "M"), ("M", "I"),
    ("I", "N"), ("N", "I"),
    ("I", "F"), ("F", "I"),
    ("M", "N"), ("N", "M"),
    ("M", "F"), ("F", "M"),
    ("N", "F"), ("F", "N"),
]

BONFERRONI_THRESHOLD = 0.05 / 24  # 24 nonlinear-term tests


def build_triplets(panel_std):
    """Build consecutive-wave triplets exactly as in inchianti_lead_lag.py."""
    rows = []
    for subj, grp in panel_std.groupby("code98"):
        grp = grp.sort_values("wave")
        for i in range(len(grp) - 1):
            r0 = grp.iloc[i]
            r1 = grp.iloc[i + 1]
            if any(pd.isna(r0[ax]) or pd.isna(r1[ax]) for ax in AXES):
                continue
            row = {"code98": subj, "wave_t": r0["wave"], "age_t": r0["age"]}
            for ax in AXES:
                row[f"{ax}_t0"] = r0[ax]
                row[f"{ax}_t1"] = r1[ax]
                row[f"d_{ax}"] = r1[ax] - r0[ax]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_pair(triplets, from_name, to_name):
    """Fit M0..M3 for a single (from -> to) pair and collect diagnostics."""
    from_ax = f"delta_{from_name}"
    to_ax = f"delta_{to_name}"

    y = triplets[f"d_{to_ax}"].values
    xi = triplets[f"{from_ax}_t0"].values
    xj = triplets[f"{to_ax}_t0"].values
    age = triplets["age_t"].values

    valid = np.isfinite(y) & np.isfinite(xi) & np.isfinite(xj) & np.isfinite(age)
    y, xi, xj, age = y[valid], xi[valid], xj[valid], age[valid]
    n = len(y)

    xi_sq = xi * xi
    xi_xj = xi * xj

    # Design matrices for each model.  Column names matter for f_test.
    def _fit(cols, names):
        X = np.column_stack(cols)
        X = sm.add_constant(X, has_constant="add")
        exog_names = ["const"] + names
        m = sm.OLS(y, X).fit(cov_type="HC3")
        m.model.exog_names[:] = exog_names  # name columns for f_test
        return m

    m0 = _fit([xi, xj, age], ["x_i", "x_j", "age"])
    m1 = _fit([xi, xi_sq, xj, age], ["x_i", "x_i_sq", "x_j", "age"])
    m2 = _fit([xi, xi_xj, xj, age], ["x_i", "x_i_xj", "x_j", "age"])
    m3 = _fit([xi, xi_sq, xi_xj, xj, age], ["x_i", "x_i_sq", "x_i_xj", "x_j", "age"])

    linear_beta = float(m0.params[m0.model.exog_names.index("x_i")])
    linear_p = float(m0.pvalues[m0.model.exog_names.index("x_i")])

    quad_beta = float(m1.params[m1.model.exog_names.index("x_i_sq")])
    quad_p = float(m1.pvalues[m1.model.exog_names.index("x_i_sq")])

    inter_beta = float(m2.params[m2.model.exog_names.index("x_i_xj")])
    inter_p = float(m2.pvalues[m2.model.exog_names.index("x_i_xj")])

    # Joint Wald test: both nonlinear terms = 0 in M3 (M3 vs M0).
    ftest = m3.f_test("x_i_sq = 0, x_i_xj = 0")
    ftest_F = float(np.squeeze(ftest.fvalue))
    ftest_p = float(np.squeeze(ftest.pvalue))

    # Residual diagnostics on M0: regress M0 residuals on x_i^2.
    resid = m0.resid
    Xr = sm.add_constant(xi_sq)
    mr = sm.OLS(resid, Xr).fit(cov_type="HC3")
    resid_quad_R2 = float(mr.rsquared)
    resid_quad_p = float(mr.pvalues[1])

    # Normality test on residuals.
    if n < 5000:
        stat, pnorm = shapiro(resid)
        norm_test = "shapiro"
    else:
        stat, pnorm = normaltest(resid)
        norm_test = "dagostino_pearson"

    return {
        "from": from_name,
        "to": to_name,
        "pair": f"{from_name}->{to_name}",
        "n": int(n),
        "linear_beta": linear_beta,
        "linear_p": linear_p,
        "quad_beta": quad_beta,
        "quad_p": quad_p,
        "interaction_beta": inter_beta,
        "interaction_p": inter_p,
        "ftest_F": ftest_F,
        "ftest_p": ftest_p,
        "residual_quad_R2": resid_quad_R2,
        "residual_quad_p": resid_quad_p,
        "residual_normality_stat": float(stat),
        "residual_normality_p": float(pnorm),
        "residual_normality_test": norm_test,
    }


def main():
    print("=" * 72)
    print("InCHIANTI: Nonlinearity Robustness Test for Cross-Lagged Regressions")
    print("=" * 72)

    panel = load_inchianti_panel()
    ref = compute_youthful_reference(panel)
    panel_std = standardize_axes(panel, ref)
    triplets = build_triplets(panel_std)
    print(f"Total consecutive-wave triplets with 4-axis data: {len(triplets)}\n")

    results = [fit_pair(triplets, f, t) for (f, t) in ORDERED_PAIRS]

    # Summary table.
    header = (
        f"{'Pair':<6s} {'N':>5s} {'beta_quad':>11s} {'p_quad':>10s} "
        f"{'beta_inter':>11s} {'p_inter':>10s} {'F(M3vM0)':>10s} "
        f"{'p_F':>10s} {'Resid R^2':>10s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['pair']:<6s} {r['n']:>5d} "
            f"{r['quad_beta']:>11.4f} {r['quad_p']:>10.4f} "
            f"{r['interaction_beta']:>11.4f} {r['interaction_p']:>10.4f} "
            f"{r['ftest_F']:>10.3f} {r['ftest_p']:>10.4f} "
            f"{r['residual_quad_R2']:>10.4f}"
        )

    # Bonferroni survival (individual terms).
    any_sig = any(
        (r["quad_p"] < BONFERRONI_THRESHOLD) or
        (r["interaction_p"] < BONFERRONI_THRESHOLD)
        for r in results
    )
    n_quad_sig = sum(1 for r in results if r["quad_p"] < BONFERRONI_THRESHOLD)
    n_inter_sig = sum(1 for r in results if r["interaction_p"] < BONFERRONI_THRESHOLD)
    n_ftest_sig_uncorrected = sum(1 for r in results if r["ftest_p"] < 0.05)
    n_ftest_sig_bonf = sum(1 for r in results if r["ftest_p"] < 0.05 / 12)
    min_p = min(
        min(r["quad_p"], r["interaction_p"]) for r in results
    )

    print()
    print(f"Bonferroni threshold (24 tests): p < {BONFERRONI_THRESHOLD:.6f}")
    print(f"Quadratic terms surviving Bonferroni:   {n_quad_sig}/12")
    print(f"Interaction terms surviving Bonferroni: {n_inter_sig}/12")
    print(f"Joint F-tests (M3 vs M0) p < 0.05 uncorrected: "
          f"{n_ftest_sig_uncorrected}/12")
    print(f"Joint F-tests surviving Bonferroni (12 tests, p<0.00417): "
          f"{n_ftest_sig_bonf}/12")
    print(f"Minimum nonlinear-term p-value across all tests: {min_p:.4g}")

    summary_text = (
        f"Tested 12 ordered axis pairs x 2 nonlinear terms (quadratic in "
        f"the predictor and predictor x autoregressor interaction) under "
        f"a Bonferroni-corrected threshold of p<{BONFERRONI_THRESHOLD:.4f}. "
        f"Quadratic terms surviving correction: {n_quad_sig}/12; "
        f"interaction terms surviving correction: {n_inter_sig}/12; "
        f"joint F-tests (M3 vs M0) surviving Bonferroni (p<{0.05/12:.4f}): "
        f"{n_ftest_sig_bonf}/12; minimum nonlinear-term p-value = "
        f"{min_p:.3g}."
    )

    out = {
        "description": (
            "InCHIANTI nonlinearity robustness test for linear OU "
            "cross-lagged regressions across 12 ordered axis pairs."
        ),
        "n_triplets": int(len(triplets)),
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
        "n_quad_significant_after_correction": int(n_quad_sig),
        "n_interaction_significant_after_correction": int(n_inter_sig),
        "n_ftest_significant_uncorrected": int(n_ftest_sig_uncorrected),
        "n_ftest_significant_after_correction": int(n_ftest_sig_bonf),
        "any_significant_after_correction": bool(any_sig),
        "min_nonlinear_pvalue": float(min_p),
        "pairs": results,
        "summary": summary_text,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/nonlinearity_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # LaTeX paragraph.
    if any_sig:
        verdict = (
            "A subset of nonlinear terms reached Bonferroni-corrected "
            "significance, indicating residual nonlinear structure within "
            "the observed biomarker range; see Table~\\ref{tab:nonlin}."
        )
    else:
        verdict = (
            "No nonlinear term reached Bonferroni-corrected significance, "
            "and all joint F-tests comparing the full nonlinear specification "
            "(M3) to the linear baseline (M0) were consistent with the null "
            "of linearity within the observed biomarker range."
        )

    latex = (
        "\\paragraph*{Supplementary Note 10: Linearity of cross-lagged "
        "regressions.} "
        "To probe the adequacy of the linear OU approximation used in the "
        "main lead-lag analysis, we re-fit each of the 12 ordered axis-pair "
        "cross-lagged regressions with three nonlinear augmentations: a "
        "quadratic term in the predictor ($x_i^2$), a predictor by "
        "autoregressor interaction ($x_i \\cdot x_j$), and their "
        "combination. "
        "All models use heteroscedasticity-consistent (HC3) standard errors "
        f"and the same {int(len(triplets))} consecutive-wave triplets as "
        "the linear analysis. "
        "Across 12 pairs $\\times$ 2 nonlinear terms (24 tests), the "
        "Bonferroni-corrected significance threshold is "
        f"$p<{BONFERRONI_THRESHOLD:.4f}$. "
        f"Quadratic terms clearing this threshold: {n_quad_sig}/12; "
        f"interaction terms: {n_inter_sig}/12; joint F-tests of the full "
        "nonlinear specification against the linear baseline surviving "
        f"a 12-test Bonferroni correction: {n_ftest_sig_bonf}/12 "
        f"(minimum nonlinear-term $p = {min_p:.3g}$). "
        f"{verdict} "
        "Behaviour far from equilibrium is not directly probed by these "
        "in-sample tests and remains a separate open question."
    )

    print("\n" + "=" * 72)
    print("LaTeX paragraph for Supplementary Note 10:")
    print("=" * 72)
    print(latex)


if __name__ == "__main__":
    main()
