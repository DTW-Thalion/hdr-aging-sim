"""
Step 8 — Circadian axis from accelerometry (optional).

Two tiers of processing:

    Tier 1 — Pre-derived UKB summary fields (90001–90060). Compute a
    circadian proxy from fragmentation / amplitude fields. Fast.

    Tier 2 — Raw CWA files. Compute interdaily stability (IS), intradaily
    variability (IV), relative amplitude (RA) from hourly activity
    profiles. Compute-intensive (hours to days).

Writes results/ukb_circadian_proxy.parquet (eid, circadian_score, components)
and merges the score into ukb_panel_long.parquet (column: delta_C).

Skipped automatically if no accelerometry data is configured.

Usage:
    python step8_circadian.py [--config config.yaml] [--tier 1|2]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from hdr_core import config_hash, save_json


def _load_accel_summary(cfg: dict) -> Optional[pd.DataFrame]:
    path = cfg.get("accel_summary_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[step8] accel_summary_path {p} not found.")
        return None
    if str(p).lower().endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def _tier1_proxy(accel_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Tier 1: UKB summary fields. The HDR circadian score is a z-scored
    fragmentation index: higher = worse circadian fidelity.
    """
    print("[step8] Tier 1: computing circadian proxy from summary fields…")
    # Search for standard UKB accelerometry summary fields.
    # Field 90087 = fraction of day in moderate activity
    # Field 90012 = overall mean acceleration
    # Field 90015–18 = acceleration by time-of-day quadrants (6h windows)
    overall_col = cfg["axis_C"].get("accel_overall", "90012")
    moderate_col = cfg["axis_C"].get("accel_moderate", "90087")
    # Try both naming conventions
    from step1_extract import _resolve_field_column
    cols = list(accel_df.columns)
    overall_c = _resolve_field_column(cols, str(overall_col), 0)
    moderate_c = _resolve_field_column(cols, str(moderate_col), 0)

    out = pd.DataFrame({"eid": accel_df["eid"].astype(int).values})

    # Day-night amplitude from quadrants (fields 90015–90018)
    quad_cols: List[str] = []
    for qfid in ("90015", "90016", "90017", "90018"):
        c = _resolve_field_column(cols, qfid, 0)
        if c is not None:
            quad_cols.append(c)
    if len(quad_cols) == 4:
        q = accel_df[quad_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        # Day window (midday) divided by night window (midnight)
        day_idx, night_idx = 1, 3  # heuristic; UKB quadrants are 0-6, 6-12, 12-18, 18-24
        ratio = q[:, day_idx] / np.clip(q[:, night_idx], 1e-3, None)
        out["day_night_ratio"] = ratio
    else:
        out["day_night_ratio"] = np.nan

    if overall_c:
        out["overall_accel"] = pd.to_numeric(accel_df[overall_c], errors="coerce").values
    if moderate_c:
        out["moderate_frac"] = pd.to_numeric(accel_df[moderate_c], errors="coerce").values

    # Composite circadian score: low amplitude + low moderate activity = worse.
    # z-score each component, flip sign so that higher = worse.
    components = []
    for col in ("day_night_ratio", "overall_accel", "moderate_frac"):
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            if s.notna().sum() < 100:
                continue
            z = (s - s.mean()) / (s.std(ddof=1) or 1.0)
            # Flip: higher ratio / higher activity = BETTER → negate so higher score = worse.
            components.append(-z.values)
    if components:
        out["circadian_score"] = np.nanmean(np.vstack(components), axis=0)
    else:
        out["circadian_score"] = np.nan
    return out


def _tier2_raw(cfg: dict) -> Optional[pd.DataFrame]:
    """
    Tier 2: iterate raw CWA files and compute IS / IV / RA.
    This is a placeholder showing the intended interface — in production,
    use actipy (https://github.com/OxWearables/actipy) or the Oxford
    biobankAccelerometerAnalysis toolkit to produce hourly activity
    series, then apply the formulae below.
    """
    raw_dir = cfg.get("accel_raw_dir")
    if not raw_dir or not Path(raw_dir).exists():
        print("[step8] Tier 2: raw CWA directory not found; skipping.")
        return None

    print(f"[step8] Tier 2: scanning {raw_dir}…")
    cwas = sorted(Path(raw_dir).glob("*.cwa*"))
    if not cwas:
        print("[step8] Tier 2: no CWA files found.")
        return None

    try:
        import actipy
    except ImportError:
        print("[step8] Tier 2: `pip install actipy` to enable raw CWA processing.")
        return None

    rows: List[dict] = []
    for cwa in tqdm(cwas, desc="raw CWA"):
        eid = _eid_from_filename(cwa.stem)
        if eid is None:
            continue
        try:
            df, info = actipy.read_device(str(cwa), lowpass_hz=20, calibrate_gravity=True)
            # Resample to hourly activity magnitude
            df["enmo"] = np.sqrt(df[["x", "y", "z"]].pow(2).sum(axis=1)).clip(lower=0) - 1
            df["enmo"] = df["enmo"].clip(lower=0)
            hourly = df["enmo"].resample("1H").mean()
            if len(hourly) < 72:  # at least 3 days
                continue
            features = _compute_circadian_features(hourly)
            features["eid"] = eid
            rows.append(features)
        except Exception as e:
            print(f"[step8] CWA {cwa.name} failed: {e}")
    if not rows:
        return None
    return pd.DataFrame(rows)


def _eid_from_filename(stem: str) -> Optional[int]:
    """Extract eid from filename; UKB convention is '<eid>_90001_0_0' etc."""
    tokens = stem.split("_")
    for t in tokens:
        try:
            return int(t)
        except ValueError:
            continue
    return None


def _compute_circadian_features(hourly: pd.Series) -> Dict[str, float]:
    """
    Given an hourly activity series (datetime index), compute:
        IS (interdaily stability)
        IV (intradaily variability)
        RA (relative amplitude = (M10 - L5) / (M10 + L5))
        L5 onset hour
    """
    h = hourly.dropna()
    if len(h) < 24:
        return {"IS": np.nan, "IV": np.nan, "RA": np.nan, "L5_onset": np.nan}
    # Hour of day series
    hod = h.index.hour
    # Hourly means across all days (24 values)
    hourly_means = np.array([h[hod == i].mean() for i in range(24)])
    grand_mean = float(h.mean())
    # IS: ratio of variance of hourly means (scaled) to total variance
    if np.var(h, ddof=0) == 0:
        IS = np.nan
    else:
        n = len(h)
        IS = float(
            (24 * np.sum((hourly_means - grand_mean) ** 2))
            / (len(hourly_means) * n * np.var(h, ddof=0))
        )
        # Alternative clean formulation: use sum of squared deviations
        IS = float(
            np.sum((hourly_means - grand_mean) ** 2)
            / max(np.var(h, ddof=0), 1e-9) / 24
        )
    # IV: fragmentation
    diffs = np.diff(h.values)
    IV = float(np.sum(diffs**2) / (len(diffs) * max(np.var(h, ddof=0), 1e-9)))
    # M10 / L5 using rolling 10/5-hour windows
    win10 = h.rolling(10, min_periods=8).mean()
    win5 = h.rolling(5, min_periods=4).mean()
    M10 = float(win10.max()) if win10.notna().any() else np.nan
    L5_val = float(win5.min()) if win5.notna().any() else np.nan
    L5_idx = win5.idxmin() if win5.notna().any() else None
    RA = (M10 - L5_val) / (M10 + L5_val) if (M10 + L5_val) > 0 else np.nan
    L5_onset = L5_idx.hour if L5_idx is not None else np.nan
    return {"IS": IS, "IV": IV, "RA": RA, "L5_onset": L5_onset}


def _merge_into_panel(circ: pd.DataFrame, panel_path: Path) -> None:
    """Merge circadian score into ukb_panel_long.parquet as delta_C."""
    panel = pd.read_parquet(panel_path)
    score_col = "circadian_score" if "circadian_score" in circ.columns else None
    if score_col is None and "IV" in circ.columns:
        # Tier 2 — use IV (higher = worse) as the axis value
        circ["circadian_score"] = circ["IV"]
        score_col = "circadian_score"
    if score_col is None:
        print("[step8] No circadian score column; skipping merge.")
        return
    keep = circ[["eid", score_col]].rename(columns={score_col: "circadian_score_new"})
    panel = panel.merge(keep, on="eid", how="left")
    # Only update baseline-instance rows with the score (one accel measurement per eid)
    base_mask = panel["instance"] == panel["instance"].min()
    panel.loc[base_mask, "delta_C"] = panel.loc[base_mask, "circadian_score_new"]
    panel = panel.drop(columns=["circadian_score_new"])
    panel.to_parquet(panel_path, index=False)
    print(f"[step8] delta_C populated for {panel['delta_C'].notna().sum():,} rows.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tier", choices=["1", "2"], default="1")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["output_dir"])

    if args.tier == "1":
        accel = _load_accel_summary(cfg)
        if accel is None or accel.empty:
            print("[step8] No accelerometry summary data; skipping.")
            return
        circ = _tier1_proxy(accel, cfg)
    else:
        circ = _tier2_raw(cfg)
        if circ is None or circ.empty:
            return

    out_path = out_dir / "ukb_circadian_proxy.parquet"
    circ.to_parquet(out_path, index=False)
    print(f"[step8] Wrote {out_path}.")

    qc = {
        "script": "step8_circadian.py",
        "config_hash": config_hash(cfg),
        "tier": args.tier,
        "n_participants": int(circ["eid"].nunique()),
        "score_mean": (
            float(circ["circadian_score"].mean())
            if "circadian_score" in circ.columns else None
        ),
        "score_sd": (
            float(circ["circadian_score"].std(ddof=1))
            if "circadian_score" in circ.columns else None
        ),
        "missing_pct": float(
            circ.get("circadian_score", pd.Series(np.nan, index=circ.index))
            .isna().mean()
        ),
    }
    save_json(out_dir / "ukb_circadian_qc.json", qc)

    _merge_into_panel(circ, out_dir / "ukb_panel_long.parquet")


if __name__ == "__main__":
    main()
