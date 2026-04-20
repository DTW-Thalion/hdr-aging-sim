"""
Step 1 — Extract and harmonise UK Biobank data.

Reads the UKB participant data file in either wide (field-encoded columns)
or melted/long format, resolves per-instance column names, applies HDR
axis transformations (log-CRP, log-cystatin, grip max, HbA1c unit
detection), and writes a long-format panel (eid × instance) to Parquet.

Usage:
    python step1_extract.py [--test] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from hdr_core import config_hash, save_json

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# Regex variants the field resolver accepts.
_FIELD_PATTERNS = [
    r"^{fid}-(?P<i>\d+)\.(?P<a>\d+)$",    # canonical: 30710-0.0
    r"^f\.{fid}\.(?P<i>\d+)\.(?P<a>\d+)$",  # R convention: f.30710.0.0
    r"^p{fid}_i(?P<i>\d+)(?:_a(?P<a>\d+))?$",  # Spark: p30710_i0_a0
    r"^x{fid}_(?P<i>\d+)_(?P<a>\d+)$",    # some custom exports
]


def _resolve_field_column(
    columns: List[str], fid: str, instance: int, array_index: int = 0,
) -> Optional[str]:
    """Find the column name matching `fid` at the requested instance/array."""
    for pat in _FIELD_PATTERNS:
        regex = re.compile(pat.format(fid=fid))
        for c in columns:
            m = regex.match(c)
            if not m:
                continue
            i = int(m.group("i"))
            a = int(m.group("a")) if m.group("a") else 0
            if i == instance and a == array_index:
                return c
    return None


def _collect_all_field_columns(columns: List[str], fid: str) -> List[Tuple[str, int, int]]:
    """Return list of (colname, instance, array_index) for every match of fid."""
    hits: List[Tuple[str, int, int]] = []
    for pat in _FIELD_PATTERNS:
        regex = re.compile(pat.format(fid=fid))
        for c in columns:
            m = regex.match(c)
            if m:
                i = int(m.group("i"))
                a = int(m.group("a")) if m.group("a") else 0
                hits.append((c, i, a))
    # de-duplicate on colname
    seen = set()
    out = []
    for t in hits:
        if t[0] not in seen:
            seen.add(t[0])
            out.append(t)
    return out


def _detect_format(df: pd.DataFrame) -> str:
    """Return 'wide' or 'long' based on column heuristics."""
    if set(df.columns) >= {"eid", "field_id", "instance", "value"}:
        return "long"
    return "wide"


def _long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert melted long format to wide UKB-style columns."""
    print("[step1] Converting long → wide format…")
    df = df.copy()
    df["array_index"] = df.get("array_index", 0)
    df["colname"] = (
        df["field_id"].astype(str)
        + "-"
        + df["instance"].astype(int).astype(str)
        + "."
        + df["array_index"].astype(int).astype(str)
    )
    pivoted = df.pivot_table(
        index="eid", columns="colname", values="value", aggfunc="first"
    ).reset_index()
    pivoted.columns.name = None
    return pivoted


def _load_raw(cfg: dict, test_mode: bool = False) -> pd.DataFrame:
    """Load participant data, handling both wide and long formats."""
    p = Path(cfg["data_path"])
    if not p.exists():
        raise FileNotFoundError(
            f"Data file not found: {p}\nEdit config.yaml → data_path."
        )
    fmt = cfg.get("data_format", "csv")
    print(f"[step1] Loading {p} (format={fmt})…")
    if fmt == "parquet":
        df = pd.read_parquet(p)
    elif fmt == "feather":
        df = pd.read_feather(p)
    else:
        read_kwargs = {"low_memory": False}
        if test_mode:
            read_kwargs["nrows"] = 1000
        df = pd.read_csv(p, **read_kwargs)
    if _detect_format(df) == "long":
        df = _long_to_wide(df)
        if test_mode:
            df = df.head(1000)
    print(f"[step1] Loaded {len(df):,} rows, {len(df.columns):,} columns.")
    return df


def _apply_withdrawals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    path = cfg.get("withdrawal_path")
    if not path:
        return df
    p = Path(path)
    if not p.exists():
        print(f"[step1] WARNING: withdrawal file {p} not found; skipping.")
        return df
    with open(p) as f:
        withdrawn = {int(line.strip()) for line in f if line.strip()}
    n0 = len(df)
    df = df[~df["eid"].astype(int).isin(withdrawn)]
    print(f"[step1] Withdrawals applied: {n0 - len(df):,} eids excluded.")
    return df


def _get_scalar(df: pd.DataFrame, fid: Optional[str], instance: int) -> pd.Series:
    """Return column series for field id / instance, or NaN if unavailable."""
    if fid is None:
        return pd.Series(np.nan, index=df.index)
    col = _resolve_field_column(list(df.columns), fid, instance)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _convert_hba1c(series: pd.Series) -> pd.Series:
    """Auto-detect DCCT (%) vs IFCC (mmol/mol) and convert to IFCC."""
    finite = series.dropna()
    if finite.empty:
        return series
    med = finite.median()
    if med < 15:  # plausibly DCCT
        print(f"[step1] HbA1c median {med:.2f} < 15 → treating as DCCT %, converting to IFCC.")
        return 10.929 * (series - 2.15)
    return series


def _build_medication_flags(df: pd.DataFrame, cfg: dict, instance: int) -> pd.DataFrame:
    """Flag statin, antihypertensive, insulin use from fields 6153/6177/2986."""
    meds = cfg["medications"]
    out = pd.DataFrame(index=df.index)
    # 6153/6177 have multiple array indices (up to 3 or 4 selections)
    statin_cols = []
    bp_cols = []
    hrt_cols = []
    for fid in [meds["medication_women"], meds["medication_men"]]:
        for col, inst, arr in _collect_all_field_columns(list(df.columns), fid):
            if inst != instance:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            statin_cols.append(vals == 1)
            bp_cols.append(vals == 2)
            hrt_cols.append(vals == 4)
    out["med_statin"] = (
        np.any(np.vstack([c.values for c in statin_cols]), axis=0) if statin_cols else False
    )
    out["med_antihtn"] = (
        np.any(np.vstack([c.values for c in bp_cols]), axis=0) if bp_cols else False
    )
    out["med_hrt"] = (
        np.any(np.vstack([c.values for c in hrt_cols]), axis=0) if hrt_cols else False
    )
    insulin_col = _resolve_field_column(list(df.columns), meds["insulin_use"], instance)
    if insulin_col is not None:
        out["med_insulin"] = pd.to_numeric(df[insulin_col], errors="coerce") == 1
    else:
        out["med_insulin"] = False
    out["n_med_classes"] = (
        out[["med_statin", "med_antihtn", "med_insulin"]].astype(int).sum(axis=1)
    )
    return out.astype({"med_statin": bool, "med_antihtn": bool, "med_insulin": bool})


def _build_comorbidity_flags(df: pd.DataFrame, cfg: dict, instance: int) -> pd.DataFrame:
    co = cfg["comorbidities"]
    out = pd.DataFrame(index=df.index)
    out["co_diabetes"] = _get_scalar(df, co["diabetes_diagnosed"], instance) == 1
    out["co_cancer"] = _get_scalar(df, co["cancer_diagnosed"], instance) == 1
    # 6150: vascular/heart — hypertension is code 4; any of heart attack(1), angina(2),
    # stroke(3), hypertension(4) count as a deficit.
    vh_cols = []
    for col, inst, arr in _collect_all_field_columns(list(df.columns), co["vascular_heart"]):
        if inst != instance:
            continue
        vh_cols.append(pd.to_numeric(df[col], errors="coerce"))
    if vh_cols:
        m = pd.concat(vh_cols, axis=1)
        out["co_hypertension"] = (m == 4).any(axis=1)
        out["co_heart"] = m.isin([1, 2, 3]).any(axis=1)
    else:
        out["co_hypertension"] = False
        out["co_heart"] = False
    out["comorbidity_count"] = out[
        ["co_diabetes", "co_cancer", "co_hypertension", "co_heart"]
    ].astype(int).sum(axis=1)
    return out


def _build_frailty_index(df: pd.DataFrame, cfg: dict, instance: int) -> pd.Series:
    """Rockwood-style Frailty Index: proportion of deficits present (0–1)."""
    fi_fields = cfg["fi_fields"]
    available = 0
    deficit = pd.Series(0, index=df.index, dtype=float)
    # Binary deficits (presence of any positive code)
    binary_map = {
        "2443": {1},          # diabetes
        "6150": {1, 2, 3, 4}, # vascular/heart problems
        "6152": {5, 7},       # DVT / pulmonary embolism
        "2453": {1},          # cancer
        "1220": {1, 2, 3, 4}, # long-standing illness
        "2178": {3, 4},       # overall health poor/fair
        "924":  {1, 2},       # walking pace slow
        "2188": {1},
    }
    for fid in fi_fields:
        fid_s = str(fid)
        if fid_s in binary_map:
            vals = _get_scalar(df, fid_s, instance)
            if vals.notna().any():
                deficit = deficit + vals.isin(binary_map[fid_s]).astype(float)
                available += 1
        elif fid_s == "46":  # grip: sex-specific weakness
            gmax = pd.concat(
                [
                    _get_scalar(df, "46", instance),
                    _get_scalar(df, "47", instance),
                ],
                axis=1,
            ).max(axis=1, skipna=True)
            sex = _get_scalar(df, cfg["sex_field"], instance=0)  # sex is time-invariant
            weak = (((sex == 1) & (gmax < 30)) | ((sex == 0) & (gmax < 20))).astype(float)
            deficit = deficit + weak
            available += 1
        elif fid_s in {"864", "884"}:  # low physical activity
            vals = _get_scalar(df, fid_s, instance)
            if vals.notna().any():
                deficit = deficit + (vals == 0).astype(float)
                available += 1
    if available == 0:
        return pd.Series(np.nan, index=df.index)
    return deficit / available


def _age_at_instance(df: pd.DataFrame, cfg: dict, instance: int) -> pd.Series:
    """Age (years) at the given assessment instance."""
    # Prefer explicit age field at instance 0; derive from birth year/month otherwise.
    if instance == 0:
        age0 = _get_scalar(df, cfg["age_field"], 0)
        if age0.notna().any():
            return age0
    birth_y = _get_scalar(df, cfg["birth_year_field"], 0)
    birth_m = _get_scalar(df, cfg["birth_month_field"], 0).fillna(6)
    ass_col = _resolve_field_column(list(df.columns), cfg["assessment_date_field"], instance)
    if ass_col is None:
        return pd.Series(np.nan, index=df.index)
    ass_date = pd.to_datetime(df[ass_col], errors="coerce")
    birth_dates = pd.to_datetime(
        dict(year=birth_y, month=birth_m, day=15), errors="coerce"
    )
    days = (ass_date - birth_dates).dt.days
    return days / 365.25


def _extract_instance(df: pd.DataFrame, cfg: dict, instance: int) -> pd.DataFrame:
    """Build per-instance rows with all configured axis biomarkers."""
    out = pd.DataFrame({"eid": df["eid"].values, "instance": instance})
    out["age"] = _age_at_instance(df, cfg, instance).values
    out["sex"] = _get_scalar(df, cfg["sex_field"], 0).values  # time-invariant
    out["ethnicity"] = _get_scalar(df, cfg["ethnicity_field"], 0).values
    out["smoking"] = _get_scalar(df, cfg["smoking_field"], instance).values
    out["townsend"] = _get_scalar(df, cfg["townsend_field"], 0).values

    ax_I = cfg["axis_I"]
    crp = _get_scalar(df, ax_I["primary"], instance)
    out["crp"] = crp.values
    out["delta_I"] = np.log(np.clip(crp, 0.01, None))

    ax_M = cfg["axis_M"]
    hba1c = _convert_hba1c(_get_scalar(df, ax_M["primary"], instance))
    out["hba1c"] = hba1c.values
    glucose = _get_scalar(df, ax_M.get("glucose_field"), instance)
    insulin = _get_scalar(df, ax_M.get("insulin_field"), instance)
    bmi = _get_scalar(df, ax_M.get("bmi_field"), instance)
    if insulin.notna().any():
        out["homa_ir"] = (glucose * insulin / 22.5).values
    else:
        out["homa_ir"] = np.nan
    out["bmi"] = bmi.values
    # Use HbA1c if available, else glucose
    m_series = hba1c.copy()
    m_series = m_series.where(m_series.notna(), glucose)
    out["delta_M"] = m_series.values

    ax_F = cfg["axis_F"]
    grip_l = _get_scalar(df, ax_F.get("grip_left", "46"), instance)
    grip_r = _get_scalar(df, ax_F.get("grip_right", "47"), instance)
    grip_max = pd.concat([grip_l, grip_r], axis=1).max(axis=1, skipna=True)
    out["grip_max"] = grip_max.values
    out["delta_F"] = -grip_max.values  # sign-flip: higher = worse

    ax_N = cfg["axis_N"]
    pulse = _get_scalar(df, ax_N["primary"], instance)
    sbp = _get_scalar(df, ax_N.get("sbp_field"), instance)
    dbp = _get_scalar(df, ax_N.get("dbp_field"), instance)
    out["pulse_rate"] = pulse.values
    out["sbp"] = sbp.values
    out["dbp"] = dbp.values
    out["delta_N"] = pulse.values

    ax_P = cfg["axis_P"]
    cystc = _get_scalar(df, ax_P["primary"], instance)
    out["cystatin_c"] = cystc.values
    out["delta_P"] = np.log(np.clip(cystc, 0.01, None))

    # Axis B: use best-available BMD field
    ax_B = cfg["axis_B"]
    bmd = None
    for key in ("femoral_neck_bmd", "total_hip_bmd", "heel_bmd"):
        fid = ax_B.get(key)
        if fid is None:
            continue
        vals = _get_scalar(df, fid, instance)
        if vals.notna().any():
            bmd = vals
            break
    out["bmd"] = bmd.values if bmd is not None else np.nan
    out["delta_B"] = -out["bmd"].values  # sign-flip: lower BMD = worse

    # Axis C — initially NaN; populated by step8 if accelerometry is available.
    out["accel_overall"] = _get_scalar(
        df, cfg["axis_C"].get("accel_overall"), instance
    ).values
    out["sleep_duration"] = _get_scalar(
        df, cfg["axis_C"].get("sleep_duration"), instance
    ).values
    out["delta_C"] = np.nan

    # Frailty & medications
    meds = _build_medication_flags(df, cfg, instance)
    out = out.join(meds.reset_index(drop=True))
    comorb = _build_comorbidity_flags(df, cfg, instance)
    out = out.join(comorb.reset_index(drop=True))
    out["frailty_index"] = _build_frailty_index(df, cfg, instance).values

    return out


def _exclude_age(panel: pd.DataFrame) -> pd.DataFrame:
    n0 = len(panel)
    keep = panel["age"].between(37, 80) & panel["sex"].notna() & panel["age"].notna()
    panel = panel[keep].copy()
    print(f"[step1] Age/sex exclusions: {n0 - len(panel):,} rows removed.")
    return panel


def _attach_mortality(panel: pd.DataFrame, cfg: dict, source_df: pd.DataFrame) -> pd.DataFrame:
    """Attach date of death and compute survival time from instance-0 assessment."""
    death_col = _resolve_field_column(list(source_df.columns), cfg["death_date_field"].split("-")[0], 0)
    if death_col is None:
        # death date may be stored with the specific instance suffix already
        death_col = cfg["death_date_field"] if cfg["death_date_field"] in source_df.columns else None
    ass_col = _resolve_field_column(list(source_df.columns), cfg["assessment_date_field"], 0)
    if ass_col is None:
        print("[step1] WARNING: baseline assessment date not found; survival unavailable.")
        panel["death_date"] = pd.NaT
        panel["baseline_date"] = pd.NaT
        panel["time_years"] = np.nan
        panel["event"] = np.nan
        return panel
    mort = pd.DataFrame({
        "eid": source_df["eid"].values,
        "baseline_date": pd.to_datetime(source_df[ass_col], errors="coerce"),
    })
    if death_col is not None:
        mort["death_date"] = pd.to_datetime(source_df[death_col], errors="coerce")
    else:
        mort["death_date"] = pd.NaT
    censor = pd.to_datetime(cfg.get("censor_date", "2023-12-31"))
    end_date = mort["death_date"].where(mort["death_date"].notna(), censor)
    mort["time_years"] = (end_date - mort["baseline_date"]).dt.days / 365.25
    mort["event"] = mort["death_date"].notna().astype(float)
    panel = panel.merge(mort, on="eid", how="left")
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract and harmonise UK Biobank data.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--test", action="store_true",
                    help="Load only 1000 rows for format verification.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_raw(cfg, test_mode=args.test)
    df = _apply_withdrawals(df, cfg)

    if "eid" not in df.columns:
        raise KeyError(
            "'eid' column missing. Check that your export preserves the participant ID."
        )

    instances = cfg["instances"]
    panels: List[pd.DataFrame] = []
    for inst in instances:
        print(f"[step1] Extracting instance {inst}…")
        panels.append(_extract_instance(df, cfg, inst))
    panel = pd.concat(panels, ignore_index=True)
    panel = _exclude_age(panel)

    # Mortality attached at baseline only, carried to all instances per eid.
    mort = _attach_mortality(
        panel[panel["instance"] == min(instances)][["eid"]].drop_duplicates(),
        cfg,
        df,
    )
    panel = panel.merge(
        mort[["eid", "baseline_date", "death_date", "time_years", "event"]],
        on="eid",
        how="left",
    )

    # Save
    out_parq = out_dir / "ukb_panel_long.parquet"
    panel.to_parquet(out_parq, index=False)
    print(f"[step1] Wrote {out_parq} ({len(panel):,} rows, {panel['eid'].nunique():,} eids).")

    # Summary metadata
    summary = {
        "script": "step1_extract.py",
        "config_hash": config_hash(cfg),
        "test_mode": bool(args.test),
        "n_rows": int(len(panel)),
        "n_eids": int(panel["eid"].nunique()),
        "instances": instances,
        "columns": list(panel.columns),
        "n_by_instance": panel.groupby("instance").size().to_dict(),
        "axis_completeness": {
            ax: int(panel[ax].notna().sum())
            for ax in ["delta_I", "delta_M", "delta_F", "delta_N", "delta_C", "delta_B", "delta_P"]
            if ax in panel.columns
        },
        "n_deaths": int(panel.drop_duplicates("eid")["event"].fillna(0).sum()),
    }
    save_json(out_dir / "ukb_panel_summary.json", summary)
    print(f"[step1] Summary → {out_dir / 'ukb_panel_summary.json'}")


if __name__ == "__main__":
    main()
