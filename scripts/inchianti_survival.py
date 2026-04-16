#!/usr/bin/env python3
"""
InCHIANTI survival analysis: SWDS-Gamma vs mortality.

Cox proportional hazards models comparing SWDS-Gamma to biomarkers
and Fried frailty for mortality prediction.
"""

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdr_sim.inchianti import load_inchianti_panel, compute_youthful_reference, standardize_axes

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("WARNING: lifelines not installed. Install with: pip install lifelines")


def compute_swds_gamma(delta_x, Gamma):
    """SWDS-Gamma = sum_k lambda_k (v_k^T dx)^2 / sum_j lambda_j"""
    eigvals, eigvecs = np.linalg.eigh(Gamma)
    eigvals = np.maximum(eigvals, 0)
    denom = eigvals.sum()
    if denom == 0:
        return np.nan
    proj = eigvecs.T @ delta_x
    return float(np.sum(eigvals * proj**2) / denom)


def main():
    print("=" * 60)
    print("InCHIANTI: Survival Analysis")
    print("=" * 60)

    if not HAS_LIFELINES:
        print("Cannot run without lifelines. Exiting.")
        return

    panel = load_inchianti_panel()
    ref = compute_youthful_reference(panel)
    panel_std = standardize_axes(panel, ref)

    # ── Build survival dataset from baseline ──
    import pyreadstat, datetime
    DATA_ROOT = os.path.join(os.path.expanduser("~"), "Downloads", "inCHIANTI", "InCHIANTI_CD_Share")
    vs_path = os.path.join(DATA_ROOT, "Vital_Status", "1.Data", "SAS_Datasets",
                           "Master_thru_Follow-up5", "ana_raw.sas7bdat")
    vs, _ = pyreadstat.read_sas7bdat(vs_path)

    bl = panel_std[panel_std["wave"] == 0].copy()
    # Merge vital status
    vs_sub = vs[["CODE98", "DECEASED", "DATA_MOR", "DATA_ULT", "IXDATE"]].copy()
    vs_sub = vs_sub.rename(columns={"CODE98": "code98"})
    bl = bl.merge(vs_sub, on="code98", how="left")

    # Compute survival time (years from baseline interview to death or last contact)
    bl["event"] = (bl["DECEASED"] == 1).astype(float)
    bl["time_days"] = np.where(
        bl["event"] == 1,
        bl["DATA_MOR"] - bl["IXDATE"],
        bl["DATA_ULT"] - bl["IXDATE"]
    )
    bl["time_years"] = bl["time_days"] / 365.25
    # Exclude negative or zero times
    bl = bl[bl["time_years"] > 0].copy()

    print(f"Baseline N = {len(bl)}, deaths = {int(bl['event'].sum())}")
    print(f"Median follow-up: {bl['time_years'].median():.1f} years")

    # ── Compute SWDS-Gamma at baseline ──
    axes = ["delta_I", "delta_M", "delta_N", "delta_F"]
    complete_mask = bl[axes].notna().all(axis=1)
    bl_complete = bl[complete_mask].copy()
    print(f"4-axis complete at baseline: {len(bl_complete)}")

    # Population Gamma from baseline cross-sectional data
    Gamma = np.cov(bl_complete[axes].values, rowvar=False)

    swds_vals = []
    for _, row in bl_complete.iterrows():
        dx = row[axes].values.astype(float)
        swds_vals.append(compute_swds_gamma(dx, Gamma))
    bl_complete["swds_gamma"] = swds_vals
    bl_complete["log_swds"] = np.log(bl_complete["swds_gamma"].clip(lower=0.001))

    # ── Prepare covariates ──
    bl_complete["female"] = (bl_complete["sex"] == 2).astype(float)
    bl_complete["log_il6"] = np.log(bl_complete["il6"].clip(lower=0.01))
    bl_complete["log_homa"] = np.log(bl_complete["homa_ir"].clip(lower=0.01))

    # ── Fit Cox models ──
    results = {}

    # Restrict to 65+ where frailty is defined
    for subgroup_name, mask in [("age65+", bl_complete["age"] >= 65),
                                 ("full", pd.Series(True, index=bl_complete.index)),
                                 ("med_naive", bl_complete["n_med_classes"] == 0)]:
        sub = bl_complete[mask].dropna(subset=["time_years", "event", "age", "female"]).copy()
        print(f"\n--- Subgroup: {subgroup_name} (N={len(sub)}, deaths={int(sub['event'].sum())}) ---")

        models = {}

        # M1: age + sex
        try:
            cph = CoxPHFitter()
            cph.fit(sub[["time_years", "event", "age", "female"]], "time_years", "event")
            c1 = cph.concordance_index_
            models["M1_age_sex"] = {"C": float(c1), "N": len(sub)}
            print(f"  M1 (age+sex): C = {c1:.4f}")
        except Exception as e:
            print(f"  M1 failed: {e}")

        # M2: + biomarkers
        bio_cols = ["log_il6", "log_homa", "resting_hr", "sppb"]
        m2_cols = ["time_years", "event", "age", "female"] + bio_cols
        sub_m2 = sub.dropna(subset=m2_cols)
        try:
            cph = CoxPHFitter()
            cph.fit(sub_m2[m2_cols], "time_years", "event")
            c2 = cph.concordance_index_
            models["M2_biomarkers"] = {"C": float(c2), "N": len(sub_m2)}
            print(f"  M2 (+biomarkers): C = {c2:.4f}")
        except Exception as e:
            print(f"  M2 failed: {e}")

        # M3: SWDS-Gamma alone
        sub_m3 = sub.dropna(subset=["log_swds"])
        try:
            cph = CoxPHFitter()
            cph.fit(sub_m3[["time_years", "event", "log_swds"]], "time_years", "event")
            c3 = cph.concordance_index_
            models["M3_swds"] = {"C": float(c3), "N": len(sub_m3)}
            print(f"  M3 (SWDS-Gamma): C = {c3:.4f}")
        except Exception as e:
            print(f"  M3 failed: {e}")

        # M4: + Fried frailty (65+ only)
        if "dx_frailty" in sub.columns and subgroup_name != "full":
            sub_m4 = sub.dropna(subset=["dx_frailty"])
            try:
                cph = CoxPHFitter()
                cph.fit(sub_m4[["time_years", "event", "age", "female", "dx_frailty"]],
                        "time_years", "event")
                c4 = cph.concordance_index_
                models["M4_frailty"] = {"C": float(c4), "N": len(sub_m4)}
                print(f"  M4 (+frailty): C = {c4:.4f}")
            except Exception as e:
                print(f"  M4 failed: {e}")

            # M5: Full model
            m5_cols = ["time_years", "event", "age", "female", "dx_frailty"] + bio_cols + ["log_swds"]
            sub_m5 = sub.dropna(subset=m5_cols)
            try:
                cph = CoxPHFitter()
                cph.fit(sub_m5[m5_cols], "time_years", "event")
                c5 = cph.concordance_index_
                models["M5_full"] = {"C": float(c5), "N": len(sub_m5)}
                delta_c = c5 - c4 if "M4_frailty" in models else np.nan
                print(f"  M5 (full): C = {c5:.4f}, delta_C(M5-M4) = {delta_c:+.4f}")
                models["delta_C_M5_M4"] = float(delta_c) if np.isfinite(delta_c) else None
            except Exception as e:
                print(f"  M5 failed: {e}")

        results[subgroup_name] = models

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/inchianti_survival_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to results/inchianti_survival_analysis.json")

    # Summary table
    print("\n--- Cox Model C-index Summary ---")
    print(f"{'Model':<20s} {'age65+':<15s} {'full':<15s} {'med_naive':<15s}")
    for model in ["M1_age_sex", "M2_biomarkers", "M3_swds", "M4_frailty", "M5_full", "delta_C_M5_M4"]:
        vals = []
        for sg in ["age65+", "full", "med_naive"]:
            v = results.get(sg, {}).get(model)
            if isinstance(v, dict):
                vals.append(f"{v['C']:.4f}")
            elif isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(f"{v:+.4f}")
            else:
                vals.append("--")
        print(f"  {model:<20s} {vals[0]:<15s} {vals[1]:<15s} {vals[2]:<15s}")


if __name__ == "__main__":
    main()
