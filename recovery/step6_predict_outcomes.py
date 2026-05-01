#!/usr/bin/env python3
"""
Step 6 -- Outcome prediction: do recovery dynamics predict outcomes beyond
admission severity?

Models compared:
  M1: age + sex
  M2: M1 + admission peaks (peak_I, peak_M, peak_N, peak_renal)
  M3: M1 + recovery timescales  (tau_I, tau_M, tau_N, tau_renal)
  M4: M1 + geometric-mean tau
  M5: M1 + peaks + taus
  M6: M1 + SOFA
  M7: M1 + SOFA + taus

Outcomes:
  - In-hospital mortality (logistic regression, AUC + bootstrap CI)
  - Length of stay (linear regression, R^2 + RMSE)
  - 30-day mortality (Cox PH on dod, if available)

For mortality the comparison metric is C-index (=AUC for binary).
Reports:
  - AUC per model
  - Delta C(M3 - M2), Delta C(M7 - M6) with bootstrap 95% CI
  - Hazard ratio per SD for each tau in M3

Note on survivorship bias: tau is only defined for survivors; for patients
who died, we impute tau as the cohort 95th percentile (slowest recovery) so
they remain in the prediction set rather than being silently excluded.

Output
------
  results/outcome_prediction.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from hdr_core import (
    AXES,
    banner,
    bootstrap_ci,
    load_config,
    write_json,
)

try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def assemble_design(fits: pd.DataFrame, episodes: pd.DataFrame,
                    admissions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge admissions with per-axis tau and per-axis peak. Imputes:
      - missing tau (non-survivors or QC failures) -> per-axis 95th pct
      - missing peak  -> per-axis median (rare; should usually be present)
    Returns one row per hadm_id.
    """
    base = admissions[[
        "subject_id", "hadm_id", "admittime", "dischtime",
        "age_at_admit", "sex", "hospital_expire_flag", "los_days",
        "sofa_score", "dod", "perturbation_type",
    ]].copy() if "perturbation_type" in admissions.columns else admissions.copy()
    if "perturbation_type" not in base.columns:
        base["perturbation_type"] = "general"

    # Wide tau per axis
    tau_wide = (fits[fits["is_primary"]]
                .pivot_table(index="hadm_id", columns="axis", values="tau_hours", aggfunc="first"))
    tau_wide.columns = [f"tau_{c}" for c in tau_wide.columns]

    # Wide peak per axis (primary biomarker)
    peak_wide = (episodes[episodes["is_primary"]]
                 .pivot_table(index="hadm_id", columns="axis", values="peak_value", aggfunc="first"))
    peak_wide.columns = [f"peak_{c}" for c in peak_wide.columns]

    out = (base.merge(tau_wide, left_on="hadm_id", right_index=True, how="left")
                .merge(peak_wide, left_on="hadm_id", right_index=True, how="left"))

    # Impute missing tau per axis with 95th percentile (slowest recovery proxy)
    for ax in AXES:
        col = f"tau_{ax}"
        if col not in out.columns:
            out[col] = np.nan
        if out[col].notna().any():
            slow = out[col].quantile(0.95)
            out[col] = out[col].fillna(slow)
        else:
            out[col] = 720.0
    # Impute missing peak with median
    for ax in AXES:
        col = f"peak_{ax}"
        if col not in out.columns:
            out[col] = np.nan
        if out[col].notna().any():
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = 0.0
    # Geometric-mean tau
    out["tau_geo"] = np.exp(np.mean(np.log(out[[f"tau_{ax}" for ax in AXES]]), axis=1))

    return out


def build_X(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Build design matrix for the named model (M1..M7)."""
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age_at_admit"].values
    X["sex_M"] = (df["sex"] == "M").astype(float).values
    if model_id in ("M2", "M5"):
        for ax in AXES:
            X[f"peak_{ax}"] = df[f"peak_{ax}"].values
    if model_id in ("M3", "M5", "M7"):
        for ax in AXES:
            X[f"log_tau_{ax}"] = np.log(df[f"tau_{ax}"].values)
    if model_id == "M4":
        X["log_tau_geo"] = np.log(df["tau_geo"].values)
    if model_id in ("M6", "M7"):
        X["sofa"] = df["sofa_score"].fillna(df["sofa_score"].median()).values
    return X


def fit_logit_auc(X: pd.DataFrame, y: np.ndarray, n_boot: int, rng) -> dict:
    """Fit logistic regression; return AUC + bootstrap 95% CI."""
    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    Xv, yv = X.values[valid], y[valid]
    if len(np.unique(yv)) < 2 or len(yv) < 50:
        return {"n": int(len(yv)), "auc": None}
    sc = StandardScaler().fit(Xv)
    Xs = sc.transform(Xv)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs").fit(Xs, yv)
    p = clf.predict_proba(Xs)[:, 1]
    auc = float(roc_auc_score(yv, p))

    boots = []
    n = len(yv)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(roc_auc_score(yv[idx], p[idx]))
        except ValueError:
            continue
    return {
        "n": int(len(yv)),
        "auc": auc,
        "auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None],
        "scores": p.tolist(),
        "labels": yv.tolist(),
        "valid_idx": valid.tolist(),
    }


def delta_c_with_ci(scores_a: np.ndarray, scores_b: np.ndarray, y: np.ndarray,
                    n_boot: int, rng) -> dict:
    """Bootstrap delta C-index between two scoring functions on the same labels."""
    valid = np.isfinite(scores_a) & np.isfinite(scores_b) & np.isfinite(y)
    sa, sb, y = scores_a[valid], scores_b[valid], y[valid]
    if len(np.unique(y)) < 2 or len(y) < 50:
        return {"delta": None}
    point = float(roc_auc_score(y, sa) - roc_auc_score(y, sb))
    boots = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boots.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
        except ValueError:
            continue
    return {
        "delta": point,
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None],
        "p_one_sided_gt0": float(np.mean(np.asarray(boots) <= 0)) if boots else None,
    }


def hazard_ratios_per_sd(df: pd.DataFrame, n_boot: int, rng) -> dict:
    """Logistic-regression coefficients per SD of log-tau, exp'd to OR (proxy HR)."""
    cols = [f"log_tau_{ax}" for ax in AXES]
    df_in = df[["hospital_expire_flag", "age_at_admit", "sex", "tau_I", "tau_M", "tau_N", "tau_renal"]].copy()
    for ax in AXES:
        df_in[f"log_tau_{ax}"] = np.log(df_in[f"tau_{ax}"])
    df_in["sex_M"] = (df_in["sex"] == "M").astype(float)
    X = df_in[cols + ["age_at_admit", "sex_M"]]
    valid = np.isfinite(X.values).all(axis=1) & np.isfinite(df_in["hospital_expire_flag"].values)
    if valid.sum() < 100:
        return {}
    Xv = X.values[valid]
    yv = df_in["hospital_expire_flag"].values[valid]
    sc = StandardScaler().fit(Xv)
    Xs = sc.transform(Xv)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs").fit(Xs, yv)
    out = {}
    for i, name in enumerate(cols):
        coef = float(clf.coef_[0, i])
        out[name] = {
            "odds_ratio_per_sd": float(np.exp(coef)),
        }
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    fits_path = os.path.join(out_dir, "recovery_fits.parquet")
    eps_path = os.path.join(out_dir, "recovery_episodes.parquet")
    adm_mimic = os.path.join(out_dir, "mimic_admissions.parquet")
    adm_isaric = os.path.join(out_dir, "isaric_admissions.parquet")
    adm_path = adm_mimic if os.path.exists(adm_mimic) else adm_isaric
    if not all(os.path.exists(p) for p in [fits_path, eps_path, adm_path]):
        print("ERROR: missing one of recovery_fits.parquet / recovery_episodes.parquet / admissions.parquet")
        return 1

    banner("Step 6: outcome prediction")
    fits = pd.read_parquet(fits_path)
    eps = pd.read_parquet(eps_path)
    adm = pd.read_parquet(adm_path)
    print(f"  fits: {len(fits):,}  episodes: {len(eps):,}  admissions: {len(adm):,}")

    # Ensure per-admission perturbation_type is present
    if "perturbation_type" not in adm.columns and "perturbation_type" in eps.columns:
        ptype = eps[["hadm_id", "perturbation_type"]].drop_duplicates("hadm_id")
        adm = adm.merge(ptype, on="hadm_id", how="left")

    df = assemble_design(fits, eps, adm)
    df = df[df["los_days"].between(0.5, 60)]
    print(f"  design rows: {len(df):,}")

    rng = np.random.default_rng(cfg["analysis"]["seed"])
    n_boot = cfg["analysis"]["n_bootstrap"]

    # In-hospital mortality
    y_mort = df["hospital_expire_flag"].astype(int).values
    print(f"  in-hospital deaths: {int(y_mort.sum()):,} ({y_mort.mean()*100:.1f}%)")

    results_mort = {}
    for mid in ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
        X = build_X(df, mid)
        r = fit_logit_auc(X, y_mort, n_boot, rng)
        results_mort[mid] = {k: v for k, v in r.items() if k not in ("scores", "labels", "valid_idx")}
        results_mort[mid]["_scores"] = r.get("scores")
        results_mort[mid]["_valid"] = r.get("valid_idx")
        auc = r.get("auc")
        print(f"    {mid}: n={r.get('n'):,}  AUC={auc:.4f}" if auc else f"    {mid}: insufficient data")

    # Delta C calculations
    deltas = {}
    def _scores(mid):
        s = np.full(len(df), np.nan)
        sub = results_mort[mid]
        if sub.get("_scores") is None:
            return s
        idx = np.where(np.asarray(sub["_valid"]))[0]
        s[idx] = sub["_scores"]
        return s

    if results_mort["M3"].get("auc") and results_mort["M2"].get("auc"):
        deltas["M3_minus_M2"] = delta_c_with_ci(_scores("M3"), _scores("M2"), y_mort, n_boot, rng)
    if results_mort["M5"].get("auc") and results_mort["M2"].get("auc"):
        deltas["M5_minus_M2"] = delta_c_with_ci(_scores("M5"), _scores("M2"), y_mort, n_boot, rng)
    if results_mort["M7"].get("auc") and results_mort["M6"].get("auc"):
        deltas["M7_minus_M6"] = delta_c_with_ci(_scores("M7"), _scores("M6"), y_mort, n_boot, rng)

    # Strip private fields before serialising
    for mid in results_mort:
        results_mort[mid].pop("_scores", None)
        results_mort[mid].pop("_valid", None)

    # Hazard ratios per SD log-tau
    hr = hazard_ratios_per_sd(df, n_boot, rng)

    # Length of stay regression
    print("\n  length-of-stay regression")
    los_results = {}
    for mid in ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
        X = build_X(df, mid)
        valid = np.isfinite(X.values).all(axis=1) & np.isfinite(df["los_days"].values)
        Xv, yv = X.values[valid], df["los_days"].values[valid]
        if len(yv) < 100:
            los_results[mid] = {"n": int(len(yv))}
            continue
        sc = StandardScaler().fit(Xv)
        Xs = sc.transform(Xv)
        ln = LinearRegression().fit(Xs, yv)
        pred = ln.predict(Xs)
        rmse = float(np.sqrt(mean_squared_error(yv, pred)))
        r2 = float(ln.score(Xs, yv))
        los_results[mid] = {"n": int(len(yv)), "r2": r2, "rmse": rmse}
        print(f"    {mid}: n={len(yv):,}  R^2={r2:.4f}  RMSE={rmse:.2f}d")

    # Cox PH on 30-day mortality (if dod available)
    cox_results = None
    if HAS_LIFELINES and "dod" in df.columns:
        cph_df = df.copy()
        cph_df["dod"] = pd.to_datetime(cph_df["dod"], errors="coerce")
        cph_df["admittime"] = pd.to_datetime(cph_df["admittime"], errors="coerce")
        cph_df["days_to_dod"] = (cph_df["dod"] - cph_df["admittime"]).dt.days
        cph_df["event_30d"] = (cph_df["days_to_dod"] >= 0) & (cph_df["days_to_dod"] <= 30)
        cph_df["t_30d"] = cph_df["days_to_dod"].clip(lower=0).fillna(30).clip(upper=30)
        cox_input = pd.DataFrame({
            "T": cph_df["t_30d"].values,
            "E": cph_df["event_30d"].astype(int).values,
            "age": cph_df["age_at_admit"].values,
            "sex_M": (cph_df["sex"] == "M").astype(float).values,
            "log_tau_geo": np.log(cph_df["tau_geo"].values),
        }).dropna()
        if len(cox_input) >= 200 and cox_input["E"].sum() >= 10:
            try:
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(cox_input, duration_col="T", event_col="E")
                cox_results = {
                    "n": int(len(cox_input)),
                    "n_events": int(cox_input["E"].sum()),
                    "concordance": float(cph.concordance_index_),
                    "hr_log_tau_geo": float(np.exp(cph.params_["log_tau_geo"])),
                    "hr_log_tau_geo_ci95": [
                        float(np.exp(cph.confidence_intervals_.loc["log_tau_geo", "95% lower-bound"])),
                        float(np.exp(cph.confidence_intervals_.loc["log_tau_geo", "95% upper-bound"])),
                    ],
                    "hr_log_tau_geo_p": float(cph.summary.loc["log_tau_geo", "p"]),
                }
                print(f"\n  Cox PH 30d: n={cox_results['n']:,}, events={cox_results['n_events']:,}")
                print(f"    HR(log tau_geo) = {cox_results['hr_log_tau_geo']:.3f}  C-index={cox_results['concordance']:.4f}")
            except Exception as e:
                cox_results = {"error": str(e)}

    out = {
        "n_admissions": int(len(df)),
        "n_deaths_inhosp": int(y_mort.sum()),
        "mortality_models": results_mort,
        "delta_c": deltas,
        "hazard_ratios_per_sd": hr,
        "los_models": los_results,
        "cox_30d": cox_results,
    }
    out_path = os.path.join(out_dir, "outcome_prediction.json")
    write_json(out, out_path)
    print(f"\nWrote {out_path}")

    if "M3_minus_M2" in deltas and deltas["M3_minus_M2"].get("delta") is not None:
        d = deltas["M3_minus_M2"]
        print(f"\nKey result -- delta C(M3 vs M2): {d['delta']:+.4f}  CI95={d.get('ci95')}")
    if "M7_minus_M6" in deltas and deltas["M7_minus_M6"].get("delta") is not None:
        d = deltas["M7_minus_M6"]
        print(f"Key result -- delta C(M7 vs M6): {d['delta']:+.4f}  CI95={d.get('ci95')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
