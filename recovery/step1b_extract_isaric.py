#!/usr/bin/env python3
"""
Step 1b -- Extract ISARIC daily CRF data into the same long-panel schema.

ISARIC-WHO COVID-19 dataset (945K hospitalised patients across 76 countries).
The CRF has admission and daily fields; this script harmonises the daily
fields into the same schema as step1_extract_mimic.py:

  results/isaric_biomarker_panel.parquet
  results/isaric_admissions.parquet

ISARIC field mapping
--------------------
  daily_crp_lborres     -> I / CRP        (mg/L)
  daily_lbor_lborres    -> I / lymphocytes (10^9/L)        [for NLR if neutrophils available]
  daily_neut_lborres    -> I / neutrophils                  [if column present]
  daily_puls_vsorres    -> N / heart_rate (bpm)
  daily_systolic_vsorres + daily_diastolic_vsorres -> N / MAP (derived)
  daily_temp_vsorres    -> (not mapped; could feed I)
  daily_creatinine      -> renal / creatinine (mg/dL or umol/L)
  daily_bun_lborres     -> renal / BUN
  daily_glucose         -> M / glucose

Site-completion filter:
  Drop sites whose daily lab completion rate < cfg episodes.min_isaric_completion
  (default 0.5).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from hdr_core import banner, load_config

ISARIC_FIELD_MAP = {
    # axis, biomarker, transform_unit
    "daily_crp_lborres":        ("I", "CRP", None),
    "daily_neut_lborres":       ("I", "neutrophils", None),
    "daily_lbor_lborres":       ("I", "lymphocytes", None),
    "daily_puls_vsorres":       ("N", "heart_rate", None),
    "daily_creatinine_lborres": ("renal", "creatinine", None),
    "daily_bun_lborres":        ("renal", "BUN", None),
    "daily_glucose_lborres":    ("M", "glucose", None),
}


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def derive_map(df: pd.DataFrame) -> pd.DataFrame:
    """MAP = (SBP + 2*DBP) / 3 from daily systolic / diastolic columns."""
    sbp_col, dbp_col = "daily_systolic_vsorres", "daily_diastolic_vsorres"
    if sbp_col not in df.columns or dbp_col not in df.columns:
        return pd.DataFrame()
    sbp = pd.to_numeric(df[sbp_col], errors="coerce")
    dbp = pd.to_numeric(df[dbp_col], errors="coerce")
    map_v = (sbp + 2 * dbp) / 3.0
    out = df[["subject_id", "hadm_id", "daily_dsstdat"]].copy()
    out["value"] = map_v
    out["axis"] = "N"
    out["biomarker"] = "MAP"
    return out.dropna(subset=["value"])


def derive_nlr_isaric(df_long: pd.DataFrame) -> pd.DataFrame:
    """NLR = neutrophils / lymphocytes paired by (hadm_id, day)."""
    n = df_long[df_long["biomarker"] == "neutrophils"][["hadm_id", "charttime", "value"]] \
        .rename(columns={"value": "neutrophils"})
    l = df_long[df_long["biomarker"] == "lymphocytes"][["hadm_id", "charttime", "value"]] \
        .rename(columns={"value": "lymphocytes"})
    if n.empty or l.empty:
        return pd.DataFrame()
    merged = n.merge(l, on=["hadm_id", "charttime"], how="inner")
    merged = merged[merged["lymphocytes"] > 0]
    merged["value"] = merged["neutrophils"] / merged["lymphocytes"]
    merged["axis"] = "I"
    merged["biomarker"] = "NLR"
    sub_map = df_long[["hadm_id", "subject_id"]].drop_duplicates()
    return merged.merge(sub_map, on="hadm_id", how="left")[
        ["subject_id", "hadm_id", "charttime", "axis", "biomarker", "value"]
    ]


def filter_low_completion_sites(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop sites where daily-CRP completion < threshold (per site, per admission-day)."""
    if "siteid" not in df.columns:
        print("  (no siteid column -- skipping site-completion filter)")
        return df

    daily = df[df["biomarker"] == "CRP"]
    if daily.empty:
        return df
    site_completion = daily.groupby("siteid").apply(
        lambda g: g["value"].notna().mean()
    )
    keep = set(site_completion[site_completion >= threshold].index)
    print(f"  sites kept: {len(keep)} of {len(site_completion)} (threshold {threshold:.0%})")
    return df[df["siteid"].isin(keep)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    isaric_path = cfg.get("isaric_file")
    if not isaric_path:
        print("ERROR: config.isaric_file is null. Set it to your ISARIC CSV/parquet and rerun.")
        return 2

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    banner("Step 1b: ISARIC extraction")
    print(f"Loading {isaric_path} ...")
    if isaric_path.endswith(".parquet"):
        df = pd.read_parquet(isaric_path)
    else:
        df = pd.read_csv(isaric_path, low_memory=False)

    # ID columns
    if "subjid" in df.columns and "subject_id" not in df.columns:
        df["subject_id"] = df["subjid"]
    if "usubjid" in df.columns and "hadm_id" not in df.columns:
        df["hadm_id"] = df["usubjid"]
    if "hadm_id" not in df.columns:
        df["hadm_id"] = df["subject_id"]

    # Demographics
    age_col = "calc_age" if "calc_age" in df.columns else "age_estimateyears"
    df["age_at_admit"] = pd.to_numeric(df.get(age_col), errors="coerce")
    df["sex"] = df.get("sex", pd.Series(index=df.index, dtype="object")).map(
        lambda x: "M" if str(x).upper().startswith("M") else ("F" if str(x).upper().startswith("F") else None)
    )

    # Times
    df["admittime"] = _to_dt(df.get("hostdat"))
    df["dischtime"] = _to_dt(df.get("dsstdat"))
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    if "dsterm" in df.columns:
        df["hospital_expire_flag"] = (df["dsterm"].astype(str).str.lower().str.contains("death|died|fatal", regex=True)).astype(int)
    else:
        df["hospital_expire_flag"] = 0

    df["has_infection_dx"] = True  # all ISARIC records are COVID
    df["has_elective_admission"] = False
    df["sofa_score"] = np.nan       # not available in standard ISARIC CRF

    # Pivot daily fields to long format
    daily_date_col = "daily_dsstdat" if "daily_dsstdat" in df.columns else "daily_dsstdtc"
    df["charttime"] = _to_dt(df.get(daily_date_col, df["admittime"]))
    df = df[df["charttime"].notna()]

    long_pieces = []
    for col, (axis, bm, _u) in ISARIC_FIELD_MAP.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        sub = pd.DataFrame({
            "subject_id": df["subject_id"],
            "hadm_id": df["hadm_id"],
            "charttime": df["charttime"],
            "axis": axis,
            "biomarker": bm,
            "value": s,
        })
        if "siteid" in df.columns:
            sub["siteid"] = df["siteid"]
        long_pieces.append(sub.dropna(subset=["value"]))

    map_df = derive_map(df)
    if not map_df.empty:
        map_long = map_df.rename(columns={"daily_dsstdat": "charttime"})
        long_pieces.append(map_long)

    if not long_pieces:
        print("ERROR: no recognised ISARIC daily fields present in input.")
        return 2
    long = pd.concat(long_pieces, ignore_index=True)
    long = long.dropna(subset=["value"])

    # NLR
    nlr = derive_nlr_isaric(long)
    if not nlr.empty:
        long = pd.concat([long, nlr], ignore_index=True)

    # Site completion filter
    threshold = cfg["episodes"].get("min_isaric_completion", 0.5)
    long = filter_low_completion_sites(long, threshold=threshold)

    # hours_from_admit
    admit_lookup = (df.dropna(subset=["admittime"])
                      .drop_duplicates("hadm_id")
                      .set_index("hadm_id")["admittime"])
    long = long.merge(admit_lookup.rename("admittime"), left_on="hadm_id", right_index=True, how="left")
    long["hours_from_admit"] = (long["charttime"] - long["admittime"]).dt.total_seconds() / 3600.0
    long = long.dropna(subset=["hours_from_admit", "value"])

    long_out = long[["subject_id", "hadm_id", "charttime", "hours_from_admit",
                     "axis", "biomarker", "value"]]
    adm_out = (df.drop_duplicates("hadm_id")
                 [["subject_id", "hadm_id", "admittime", "dischtime",
                   "age_at_admit", "sex", "hospital_expire_flag", "los_days",
                   "has_infection_dx", "has_elective_admission", "sofa_score"]])
    adm_out["admission_type"] = "ISARIC_COVID"
    adm_out["race"] = "UNKNOWN"
    adm_out["dod"] = pd.NaT

    long_path = os.path.join(out_dir, "isaric_biomarker_panel.parquet")
    adm_path = os.path.join(out_dir, "isaric_admissions.parquet")
    long_out.to_parquet(long_path, index=False)
    adm_out.to_parquet(adm_path, index=False)
    print(f"\nWrote {long_path}  ({len(long_out):,} rows)")
    print(f"Wrote {adm_path}    ({len(adm_out):,} admissions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
