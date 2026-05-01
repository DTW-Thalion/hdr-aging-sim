#!/usr/bin/env python3
"""
Step 2 -- Identify recovery episodes.

For each (admission, axis), find the peak perturbation value within the first
`peak_window_days` and the recovery trajectory from peak to discharge (capped
at `recovery_window_days`). Tag with cohort labels (infection / surgical /
sepsis / general).

Input
-----
  results/mimic_biomarker_panel.parquet
  results/mimic_admissions.parquet

(or the ISARIC equivalents, auto-detected)

Output
------
  results/recovery_episodes.parquet
    One row per (hadm_id, axis, biomarker). Each row carries:
      - peak_value, peak_time_hours
      - recovery_t (list[float], hours from peak)
      - recovery_y (list[float], biomarker values)
      - n_recovery_points
      - estimated_baseline (from prior admissions or age/sex reference)
      - perturbation_type (general/infection/surgical/sepsis)
      - age, sex, los_days, survived
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from hdr_core import (
    AXES,
    banner,
    load_config,
    primary_biomarker,
    sepsis3_flag,
)


def _detect_dataset(out_dir: str) -> tuple[str, str, str]:
    """Return (panel_path, adm_path, label) -- prefer MIMIC if present."""
    mimic_panel = os.path.join(out_dir, "mimic_biomarker_panel.parquet")
    mimic_adm = os.path.join(out_dir, "mimic_admissions.parquet")
    isaric_panel = os.path.join(out_dir, "isaric_biomarker_panel.parquet")
    isaric_adm = os.path.join(out_dir, "isaric_admissions.parquet")
    if os.path.exists(mimic_panel) and os.path.exists(mimic_adm):
        return mimic_panel, mimic_adm, "mimic"
    if os.path.exists(isaric_panel) and os.path.exists(isaric_adm):
        return isaric_panel, isaric_adm, "isaric"
    raise FileNotFoundError(
        "No biomarker panel found. Run step1_extract_mimic.py or step1b_extract_isaric.py first."
    )


def baseline_from_prior_admissions(panel: pd.DataFrame, adm: pd.DataFrame,
                                   biomarker: str) -> pd.Series:
    """
    For each admission, estimate baseline as the median of values measured
    >= 14 days BEFORE this admission for the same subject (i.e. last lab
    value from prior outpatient visit or prior admission).
    """
    sub_panel = panel[panel["biomarker"] == biomarker][
        ["subject_id", "hadm_id", "charttime", "value"]
    ].copy()
    if sub_panel.empty:
        return pd.Series(dtype=float)
    sub_panel = sub_panel.merge(
        adm[["hadm_id", "admittime"]], on="hadm_id", how="left",
    )
    out = {}
    for subj, g in sub_panel.groupby("subject_id"):
        for hadm, this_admit in g[["hadm_id", "admittime"]].drop_duplicates().itertuples(index=False):
            prior_mask = (g["hadm_id"] != hadm) & (g["charttime"] < this_admit - pd.Timedelta(days=14))
            prior_vals = g.loc[prior_mask, "value"]
            if len(prior_vals) >= 1:
                out[hadm] = float(prior_vals.median())
    return pd.Series(out, name="prior_baseline")


def reference_baseline(cfg: dict, axis: str, biomarker: str) -> float:
    """Population-level reference: midpoint of [ref_low, ref_high] from config."""
    for bm in cfg["axes"][axis]["biomarkers"]:
        if bm["name"] == biomarker:
            lo = bm.get("ref_low")
            hi = bm.get("ref_high")
            if lo is not None and hi is not None:
                return float((lo + hi) / 2)
    return np.nan


def find_peak(values: pd.DataFrame, sign: str, peak_window_h: float) -> tuple[float, float] | None:
    """
    values: columns [hours_from_admit, value], one biomarker, one admission.
    Returns (peak_value, peak_time_h) within the peak window. The peak is
    the maximum (sign=positive) or minimum (sign=negative) value.
    """
    in_window = values[(values["hours_from_admit"] >= 0) &
                       (values["hours_from_admit"] <= peak_window_h)]
    if in_window.empty:
        return None
    if sign == "negative":
        idx = in_window["value"].idxmin()
    else:
        idx = in_window["value"].idxmax()
    return float(in_window.at[idx, "value"]), float(in_window.at[idx, "hours_from_admit"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    panel_path, adm_path, label = _detect_dataset(out_dir)
    banner(f"Step 2: episode definition ({label})")
    print(f"  panel: {panel_path}")
    print(f"  admissions: {adm_path}")

    panel = pd.read_parquet(panel_path)
    adm = pd.read_parquet(adm_path)
    print(f"  N admissions: {len(adm):,}  panel rows: {len(panel):,}")

    # Eligibility filters on admissions
    e = cfg["episodes"]
    elig = adm[
        adm["los_days"].between(e["min_los_days"], e["max_los_days"]) &
        adm["age_at_admit"].notna() &
        (adm["age_at_admit"] >= 18)
    ].copy()
    print(f"  eligible by LOS/age: {len(elig):,}")

    # Cohort flags
    elig["is_sepsis"] = False
    if "sofa_score" in elig.columns and "has_infection_dx" in elig.columns:
        try:
            elig["is_sepsis"] = sepsis3_flag(elig[["sofa_score", "has_infection_dx"]])
        except ValueError:
            pass

    def cohort_label(row) -> str:
        if row.get("is_sepsis"):
            return "sepsis"
        if row.get("has_infection_dx"):
            return "infection"
        if row.get("has_elective_admission"):
            return "surgical"
        return "general"

    elig["perturbation_type"] = elig.apply(cohort_label, axis=1)
    print("  perturbation_type counts:")
    for c, n in elig["perturbation_type"].value_counts().items():
        print(f"    {c:<10}: {n:,}")

    peak_window_h = e["peak_window_days"] * 24.0
    recovery_window_h = e["recovery_window_days"] * 24.0
    min_pts = e["min_measurements_per_axis"]

    # Iterate per axis x primary biomarker
    rows = []
    for axis in cfg["axes"].keys():
        for bm in cfg["axes"][axis]["biomarkers"]:
            biomarker = bm["name"]
            sign = bm.get("sign", "positive")
            print(f"\n  axis={axis:<6} biomarker={biomarker}")

            sub = panel[(panel["axis"] == axis) & (panel["biomarker"] == biomarker)]
            sub = sub[sub["hadm_id"].isin(elig["hadm_id"])]
            if sub.empty:
                print(f"    (no measurements after eligibility filter)")
                continue

            # Baseline from prior admissions (where available)
            prior_bl = baseline_from_prior_admissions(sub, elig, biomarker)
            ref_bl = reference_baseline(cfg, axis, biomarker)

            n_eligible = 0
            n_with_peak = 0
            n_with_recovery = 0
            for hadm, g in sub.groupby("hadm_id"):
                g = g.sort_values("hours_from_admit")
                if len(g) < min_pts:
                    continue
                n_eligible += 1
                peak = find_peak(g[["hours_from_admit", "value"]], sign, peak_window_h)
                if peak is None:
                    continue
                peak_value, peak_time_h = peak
                n_with_peak += 1
                rec_mask = (g["hours_from_admit"] > peak_time_h) & \
                           (g["hours_from_admit"] <= peak_time_h + recovery_window_h)
                rec = g.loc[rec_mask, ["hours_from_admit", "value"]]
                if len(rec) < min_pts - 1:  # need at least min_pts-1 points after peak (peak counts as 1)
                    continue
                n_with_recovery += 1

                # Recovery time relative to peak
                rec_t = (rec["hours_from_admit"] - peak_time_h).tolist()
                rec_y = rec["value"].tolist()

                bl = float(prior_bl.get(hadm, np.nan))
                if np.isnan(bl):
                    bl = ref_bl

                arow = elig.loc[elig["hadm_id"] == hadm].iloc[0]
                survived = int(arow.get("hospital_expire_flag", 0)) == 0
                rows.append({
                    "subject_id": int(arow["subject_id"]),
                    "hadm_id": int(hadm),
                    "axis": axis,
                    "biomarker": biomarker,
                    "is_primary": (biomarker == primary_biomarker(cfg, axis)),
                    "age": float(arow["age_at_admit"]),
                    "sex": arow.get("sex"),
                    "los_days": float(arow["los_days"]),
                    "survived": survived,
                    "perturbation_type": arow["perturbation_type"],
                    "sofa_score": float(arow.get("sofa_score") or np.nan),
                    "peak_value": peak_value,
                    "peak_time_hours": peak_time_h,
                    "estimated_baseline": bl,
                    "recovery_t_hours": rec_t,
                    "recovery_y": rec_y,
                    "n_recovery_points": len(rec_t),
                })
            print(f"    eligible: {n_eligible:,}  with peak: {n_with_peak:,}  "
                  f"with recovery: {n_with_recovery:,}")

    if not rows:
        print("\nERROR: no episodes meet criteria.")
        return 1
    eps = pd.DataFrame(rows)
    eps_path = os.path.join(out_dir, "recovery_episodes.parquet")
    eps.to_parquet(eps_path, index=False)
    print(f"\nWrote {eps_path}  ({len(eps):,} episodes)")

    # Optionally drop deaths from the recovery-fit set
    if e.get("exclude_deaths_from_recovery", True):
        n_dead = (~eps["survived"]).sum()
        print(f"  ({n_dead:,} episodes are non-survivors -- step 3 will drop them from fitting)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
