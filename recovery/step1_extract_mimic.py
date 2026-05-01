#!/usr/bin/env python3
"""
Step 1 -- Extract & harmonise MIMIC-IV data.

Streams labevents and chartevents in chunks, filtering by the itemids we care
about (defined in config.yaml), aggregates ICU vitals to daily medians where
configured, derives NLR, and writes a long-format biomarker panel:

  results/mimic_biomarker_panel.parquet
    columns: subject_id, hadm_id, charttime, hours_from_admit,
             axis, biomarker, value

Plus an admissions summary:
  results/mimic_admissions.parquet
    columns: subject_id, hadm_id, admittime, dischtime, admission_type, race,
             age_at_admit, sex, hospital_expire_flag, los_days, dod,
             has_infection_dx, has_elective_admission, sofa_score

Usage:
    python step1_extract_mimic.py [--config config.yaml]

Memory: chunk size set to 5M rows. Plan for ~32 GB RAM peak when reading
chartevents (which has hundreds of millions of rows pre-filter).
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from glob import glob
from typing import Iterable

import numpy as np
import pandas as pd

from hdr_core import (
    AXES,
    all_itemids,
    axis_biomarkers,
    banner,
    load_config,
)

CHUNK = 5_000_000


# ---------------------------------------------------------------------------
# Path resolution

def _maybe(path: str) -> str | None:
    return path if os.path.exists(path) else None


def resolve_mimic_paths(mimic_dir: str) -> dict:
    """Find the standard MIMIC-IV CSV/parquet files under mimic_dir."""
    candidates = {
        "patients":      ["hosp/patients.csv.gz", "hosp/patients.csv", "hosp/patients.parquet"],
        "admissions":    ["hosp/admissions.csv.gz", "hosp/admissions.csv", "hosp/admissions.parquet"],
        "labevents":     ["hosp/labevents.csv.gz", "hosp/labevents.csv", "hosp/labevents.parquet"],
        "d_labitems":    ["hosp/d_labitems.csv.gz", "hosp/d_labitems.csv", "hosp/d_labitems.parquet"],
        "chartevents":   ["icu/chartevents.csv.gz", "icu/chartevents.csv", "icu/chartevents.parquet"],
        "d_items":       ["icu/d_items.csv.gz", "icu/d_items.csv", "icu/d_items.parquet"],
        "icustays":      ["icu/icustays.csv.gz", "icu/icustays.csv", "icu/icustays.parquet"],
        "diagnoses_icd": ["hosp/diagnoses_icd.csv.gz", "hosp/diagnoses_icd.csv", "hosp/diagnoses_icd.parquet"],
    }
    out = {}
    for key, opts in candidates.items():
        for rel in opts:
            full = os.path.join(mimic_dir, rel)
            if os.path.exists(full):
                out[key] = full
                break
    missing = [k for k in candidates if k not in out]
    if missing:
        raise FileNotFoundError(
            f"Missing MIMIC-IV tables under {mimic_dir}: {missing}\n"
            f"Expected layout:\n"
            f"  {mimic_dir}/hosp/patients.csv.gz\n"
            f"  {mimic_dir}/hosp/admissions.csv.gz\n"
            f"  ...\n"
            f"  {mimic_dir}/icu/chartevents.csv.gz"
        )
    return out


def _read_table(path: str, **kwargs) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def _iter_table_chunks(path: str, chunksize=CHUNK, **kwargs):
    """Yield DataFrame chunks regardless of csv/parquet/gz."""
    if path.endswith(".parquet"):
        # parquet: read full (memory-permitting) -- caller already filters columns
        yield pd.read_parquet(path, **kwargs)
    else:
        for chunk in pd.read_csv(path, chunksize=chunksize, **kwargs):
            yield chunk


# ---------------------------------------------------------------------------
# Patients / admissions

def load_admissions(paths: dict) -> pd.DataFrame:
    """
    Build admissions table with:
      subject_id, hadm_id, admittime, dischtime, admission_type, race,
      age_at_admit, sex, hospital_expire_flag, dod, los_days
    """
    pat = _read_table(paths["patients"],
                      usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
    pat["dod"] = pd.to_datetime(pat["dod"], errors="coerce")

    adm_cols = ["subject_id", "hadm_id", "admittime", "dischtime",
                "admission_type", "race", "hospital_expire_flag"]
    adm = _read_table(paths["admissions"], usecols=adm_cols)
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")

    df = adm.merge(pat, on="subject_id", how="left")
    df["age_at_admit"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
    df["sex"] = df["gender"].map({"M": "M", "F": "F"})
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0

    # Anchor: MIMIC top-codes ages 90+ as exactly 91. Drop biologically implausible.
    df = df[df["age_at_admit"].between(18, 110)].copy()

    return df[[
        "subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
        "race", "age_at_admit", "sex", "hospital_expire_flag", "dod", "los_days",
    ]]


def add_diagnoses(adm: pd.DataFrame, paths: dict, infection_prefixes: list[str]) -> pd.DataFrame:
    dx = _read_table(paths["diagnoses_icd"], usecols=["hadm_id", "icd_code", "icd_version"])
    dx["icd_code"] = dx["icd_code"].astype(str).str.strip().str.upper()

    # has_infection_dx: any ICD-10 code starting with one of the configured prefixes
    pref = tuple(p.upper() for p in infection_prefixes)
    inf = (dx[dx["icd_version"] == 10]
           .assign(_inf=lambda d: d["icd_code"].str.startswith(pref))
           .groupby("hadm_id")["_inf"].any().rename("has_infection_dx"))
    adm = adm.merge(inf, left_on="hadm_id", right_index=True, how="left")
    adm["has_infection_dx"] = adm["has_infection_dx"].fillna(False)
    return adm


# ---------------------------------------------------------------------------
# Labevents / chartevents

def stream_labevents(paths: dict, target_itemids: set[int]) -> pd.DataFrame:
    """Filter labevents to target itemids, stream in chunks."""
    print(f"  streaming labevents (filter to {len(target_itemids)} itemids)...")
    cols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]
    pieces = []
    n_chunks = 0
    n_kept = 0
    for chunk in _iter_table_chunks(paths["labevents"], usecols=cols):
        n_chunks += 1
        kept = chunk[chunk["itemid"].isin(target_itemids)].copy()
        if len(kept):
            kept["valuenum"] = pd.to_numeric(kept["valuenum"], errors="coerce")
            kept = kept.dropna(subset=["valuenum", "hadm_id"])
            n_kept += len(kept)
            pieces.append(kept)
        if n_chunks % 10 == 0:
            print(f"    [chunk {n_chunks:>4}] cumulative kept rows: {n_kept:,}")
    if not pieces:
        return pd.DataFrame(columns=cols)
    df = pd.concat(pieces, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["hadm_id"] = df["hadm_id"].astype("Int64")
    return df


def stream_chartevents(paths: dict, target_itemids: set[int]) -> pd.DataFrame:
    """Filter chartevents (ICU vitals) to target itemids."""
    print(f"  streaming chartevents (filter to {len(target_itemids)} itemids)...")
    cols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]
    pieces = []
    n_chunks = 0
    n_kept = 0
    for chunk in _iter_table_chunks(paths["chartevents"], usecols=cols):
        n_chunks += 1
        kept = chunk[chunk["itemid"].isin(target_itemids)].copy()
        if len(kept):
            kept["valuenum"] = pd.to_numeric(kept["valuenum"], errors="coerce")
            kept = kept.dropna(subset=["valuenum", "hadm_id"])
            n_kept += len(kept)
            pieces.append(kept)
        if n_chunks % 10 == 0:
            print(f"    [chunk {n_chunks:>4}] cumulative kept rows: {n_kept:,}")
    if not pieces:
        return pd.DataFrame(columns=cols)
    df = pd.concat(pieces, ignore_index=True)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["hadm_id"] = df["hadm_id"].astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Daily aggregation for chartevents

def aggregate_daily_median(df: pd.DataFrame, admittime_lookup: pd.Series) -> pd.DataFrame:
    """Collapse to one row per (hadm_id, itemid, calendar day) with median value."""
    df = df.copy()
    df["day"] = df["charttime"].dt.floor("1D")
    grp = (df.groupby(["subject_id", "hadm_id", "itemid", "day"])
             .agg(valuenum=("valuenum", "median"))
             .reset_index())
    grp = grp.rename(columns={"day": "charttime"})
    return grp


# ---------------------------------------------------------------------------
# NLR derivation

def derive_nlr(lab_long: pd.DataFrame, neut_id: int, lymph_id: int,
               nlr_axis: str = "I") -> pd.DataFrame:
    """Compute NLR = neutrophils / lymphocytes per (hadm_id, charttime).

    Pairs the closest neutrophil and lymphocyte measurement within a 6-hour window.
    """
    n = lab_long[lab_long["itemid"] == neut_id][["hadm_id", "charttime", "valuenum"]] \
        .rename(columns={"valuenum": "neutrophils"})
    l = lab_long[lab_long["itemid"] == lymph_id][["hadm_id", "charttime", "valuenum"]] \
        .rename(columns={"valuenum": "lymphocytes"})
    if n.empty or l.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "axis", "biomarker", "value"])

    out_pieces = []
    for hadm, n_g in n.groupby("hadm_id"):
        l_g = l[l["hadm_id"] == hadm]
        if l_g.empty:
            continue
        merged = pd.merge_asof(
            n_g.sort_values("charttime"),
            l_g.sort_values("charttime")[["charttime", "lymphocytes"]],
            on="charttime",
            direction="nearest",
            tolerance=pd.Timedelta("6h"),
        )
        merged = merged.dropna(subset=["lymphocytes"])
        merged = merged[merged["lymphocytes"] > 0]
        merged["nlr"] = merged["neutrophils"] / merged["lymphocytes"]
        out_pieces.append(merged[["hadm_id", "charttime", "nlr"]])

    if not out_pieces:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "charttime", "axis", "biomarker", "value"])
    out = pd.concat(out_pieces, ignore_index=True)
    out["axis"] = nlr_axis
    out["biomarker"] = "NLR"
    out = out.rename(columns={"nlr": "value"})
    # subject_id mapped back from any neutrophil row
    sub_map = lab_long[["hadm_id", "subject_id"]].drop_duplicates()
    out = out.merge(sub_map, on="hadm_id", how="left")
    return out[["subject_id", "hadm_id", "charttime", "axis", "biomarker", "value"]]


# ---------------------------------------------------------------------------
# Build long panel

def build_panel(lab_long: pd.DataFrame, chart_long: pd.DataFrame, cfg: dict,
                itemid_meta: dict[int, dict]) -> pd.DataFrame:
    """Combine lab + chart events into a unified long panel keyed by axis & biomarker."""

    def _apply_meta(df):
        df = df.copy()
        df["axis"] = df["itemid"].map(lambda i: itemid_meta.get(int(i), {}).get("axis"))
        df["biomarker"] = df["itemid"].map(lambda i: itemid_meta.get(int(i), {}).get("name"))
        return df.dropna(subset=["axis", "biomarker"])

    lab = _apply_meta(lab_long)
    chart = _apply_meta(chart_long)

    # Restrict lab to non-derived components
    derived_components = {"neutrophils", "lymphocytes"}
    lab_main = lab[~lab["biomarker"].isin(derived_components)]
    chart_main = chart  # chart never holds derived components

    long = pd.concat([
        lab_main[["subject_id", "hadm_id", "charttime", "axis", "biomarker", "valuenum"]]
            .rename(columns={"valuenum": "value"}),
        chart_main[["subject_id", "hadm_id", "charttime", "axis", "biomarker", "valuenum"]]
            .rename(columns={"valuenum": "value"}),
    ], ignore_index=True)

    # NLR
    nlr_cfg = next((bm for ax in cfg["axes"].values()
                    for bm in ax["biomarkers"] if bm.get("source") == "derived" and bm["name"] == "NLR"),
                   None)
    if nlr_cfg is not None:
        nlr = derive_nlr(lab, int(nlr_cfg["neutrophil_itemid"]), int(nlr_cfg["lymphocyte_itemid"]))
        if not nlr.empty:
            long = pd.concat([long, nlr], ignore_index=True)

    return long


def attach_admit_time(long: pd.DataFrame, adm: pd.DataFrame) -> pd.DataFrame:
    long = long.merge(
        adm[["hadm_id", "admittime"]],
        on="hadm_id", how="left",
    )
    long["hours_from_admit"] = (long["charttime"] - long["admittime"]).dt.total_seconds() / 3600.0
    long = long.dropna(subset=["hours_from_admit", "value"])
    long = long[long["hours_from_admit"] >= -24]  # allow 1 day pre-admit
    return long.drop(columns=["admittime"])


# ---------------------------------------------------------------------------
# SOFA (simplified)

def compute_simple_sofa(adm: pd.DataFrame, lab_long: pd.DataFrame,
                        chart_long: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Compute a partial SOFA score per admission using the lab/chart data already
    extracted: creatinine (renal subscore) + MAP (cardio subscore). This is a
    LOWER BOUND on true SOFA -- platelets, bilirubin, GCS, PaO2/FiO2 require
    additional itemids that aren't in the recovery axis set. Step 6 will
    optionally upgrade this when the user supplies the extra itemids.
    """
    sofa = pd.Series(0, index=adm["hadm_id"], dtype=int)

    # Renal sub-score from creatinine peak in first 24h
    creat = lab_long[lab_long["itemid"] == 50912][["hadm_id", "charttime", "valuenum"]]
    if not creat.empty:
        creat = creat.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
        creat["h"] = (pd.to_datetime(creat["charttime"]) - pd.to_datetime(creat["admittime"])).dt.total_seconds() / 3600
        creat_24 = creat[creat["h"].between(0, 24)]
        peak = creat_24.groupby("hadm_id")["valuenum"].max()
        renal_score = peak.apply(_creat_to_sofa).rename("renal_sofa")
        sofa = sofa.add(renal_score, fill_value=0)

    # Cardio sub-score from MAP nadir in first 24h
    if 220052 in chart_long["itemid"].unique():
        map_df = chart_long[chart_long["itemid"] == 220052][["hadm_id", "charttime", "valuenum"]]
        map_df = map_df.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
        map_df["h"] = (pd.to_datetime(map_df["charttime"]) - pd.to_datetime(map_df["admittime"])).dt.total_seconds() / 3600
        map_24 = map_df[map_df["h"].between(0, 24)]
        nadir = map_24.groupby("hadm_id")["valuenum"].min()
        cardio_score = nadir.apply(_map_to_sofa).rename("cardio_sofa")
        sofa = sofa.add(cardio_score, fill_value=0)

    return sofa.fillna(0).astype(int).rename("sofa_score")


def _creat_to_sofa(creat_mgdl: float) -> int:
    if creat_mgdl >= 5.0: return 4
    if creat_mgdl >= 3.5: return 3
    if creat_mgdl >= 2.0: return 2
    if creat_mgdl >= 1.2: return 1
    return 0


def _map_to_sofa(map_mmhg: float) -> int:
    if map_mmhg < 70: return 1
    return 0


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    if not cfg.get("mimic_dir"):
        print("ERROR: config.mimic_dir is null. Set it to your MIMIC-IV root and rerun.")
        print("       (For ISARIC instead, run step1b_extract_isaric.py.)")
        return 2

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    banner("Step 1: MIMIC-IV extraction")
    paths = resolve_mimic_paths(cfg["mimic_dir"])
    for k, v in paths.items():
        print(f"  {k:<14} -> {v}")

    print("\nLoading admissions / patients ...")
    adm = load_admissions(paths)
    print(f"  N admissions: {len(adm):,}  (subjects: {adm['subject_id'].nunique():,})")

    print("\nAttaching infection diagnoses ...")
    adm = add_diagnoses(adm, paths, cfg["episodes"]["cohorts"]["infection"]["icd_prefixes"])
    print(f"  with infection dx: {adm['has_infection_dx'].sum():,}")

    elective_types = set(cfg["episodes"]["cohorts"]["surgical"]["admission_type"])
    adm["has_elective_admission"] = adm["admission_type"].isin(elective_types)

    # Build itemid sets per source
    meta = all_itemids(cfg)
    lab_ids = {iid for iid, m in meta.items() if m["source"] == "labevents"}
    chart_ids = {iid for iid, m in meta.items() if m["source"] == "chartevents"}
    print(f"\nTarget itemids: lab={len(lab_ids)} chart={len(chart_ids)}")

    print("\nExtracting labevents (large -- streamed) ...")
    lab_long = stream_labevents(paths, lab_ids)
    print(f"  rows kept: {len(lab_long):,}")

    print("\nExtracting chartevents (very large -- streamed) ...")
    chart_long_raw = stream_chartevents(paths, chart_ids)
    print(f"  rows kept (raw): {len(chart_long_raw):,}")

    # Daily aggregation per config
    daily_ids = {iid for iid, m in meta.items()
                 if m.get("aggregation") == "daily_median"}
    if daily_ids:
        admit_lookup = adm.set_index("hadm_id")["admittime"]
        chart_daily = chart_long_raw[chart_long_raw["itemid"].isin(daily_ids)]
        chart_daily = aggregate_daily_median(chart_daily, admit_lookup)
        chart_other = chart_long_raw[~chart_long_raw["itemid"].isin(daily_ids)]
        chart_long = pd.concat([chart_daily, chart_other], ignore_index=True)
    else:
        chart_long = chart_long_raw
    print(f"  rows after daily aggregation: {len(chart_long):,}")

    print("\nComputing simplified SOFA (renal+cardio sub-scores from first 24h) ...")
    sofa = compute_simple_sofa(adm, lab_long, chart_long, cfg)
    adm = adm.merge(sofa.rename("sofa_score"), left_on="hadm_id", right_index=True, how="left")
    adm["sofa_score"] = adm["sofa_score"].fillna(0).astype(int)

    print("\nBuilding long-format biomarker panel ...")
    long = build_panel(lab_long, chart_long, cfg, meta)
    long = attach_admit_time(long, adm)
    print(f"  panel rows: {len(long):,}  unique hadm: {long['hadm_id'].nunique():,}")

    panel_path = os.path.join(out_dir, "mimic_biomarker_panel.parquet")
    adm_path = os.path.join(out_dir, "mimic_admissions.parquet")
    long.to_parquet(panel_path, index=False)
    adm.to_parquet(adm_path, index=False)
    print(f"\nWrote {panel_path}")
    print(f"Wrote {adm_path}")

    # Quick coverage report
    print("\nCoverage by axis (% admissions with >= 4 measurements of primary biomarker):")
    for axis, ax_cfg in cfg["axes"].items():
        prim = ax_cfg["primary"]
        sub = long[(long["axis"] == axis) & (long["biomarker"] == prim)]
        per_adm = sub.groupby("hadm_id").size()
        pct = (per_adm >= 4).mean() * 100 if len(per_adm) else 0
        print(f"  {axis:<6} ({prim:<12}): {pct:5.1f}%  ({(per_adm >= 4).sum():,} admissions)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
