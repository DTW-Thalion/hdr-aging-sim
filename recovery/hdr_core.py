"""
Shared utilities for HDR recovery-dynamics analysis.

Bundled here (rather than imported from src.hdr_sim) so the recovery package
is self-contained and can be moved/copied without resolving the sibling
package layout. Functions here are deliberately small and stateless.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

AXES = ["I", "M", "N", "renal"]

# J-matrix sign predictions for axes used in this analysis.
# Adapted from data/J_matrix_compiled.csv. We use the renal axis as a proxy
# for the P-axis (proteostasis / clearance capacity); the I->P, M->P, N->P
# signs from the compiled matrix are used for I->renal, etc.
J_SIGNS = {
    ("I", "M"): +1,
    ("M", "I"): +1,
    ("I", "N"): +1,
    ("N", "I"): -1,
    ("I", "renal"): +1,
    ("renal", "I"): +1,
    ("M", "N"): +1,
    ("N", "M"): -1,
    ("M", "renal"): +1,
    ("renal", "M"): +1,
    ("N", "renal"): +1,
    ("renal", "N"): +1,
}


# ---------------------------------------------------------------------------
# Config

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def axis_biomarkers(cfg: dict, axis: str) -> list[dict]:
    return cfg["axes"][axis]["biomarkers"]


def primary_biomarker(cfg: dict, axis: str) -> str:
    return cfg["axes"][axis]["primary"]


def all_itemids(cfg: dict) -> dict[int, dict]:
    """
    Map MIMIC-IV itemid -> {axis, name, source, transform, sign, ...}.
    Includes derived-component itemids (neutrophils, lymphocytes for NLR).
    """
    out: dict[int, dict] = {}
    for axis, ax_cfg in cfg["axes"].items():
        for bm in ax_cfg["biomarkers"]:
            if bm.get("source") == "derived":
                for sub_key in ("neutrophil_itemid", "lymphocyte_itemid"):
                    if sub_key in bm:
                        out[int(bm[sub_key])] = {
                            "axis": axis,
                            "name": sub_key.replace("_itemid", ""),
                            "source": "labevents",
                            "for_derived": bm["name"],
                        }
            else:
                out[int(bm["itemid"])] = {
                    "axis": axis,
                    "name": bm["name"],
                    "source": bm["source"],
                    "transform": bm.get("transform", "none"),
                    "sign": bm.get("sign", "positive"),
                    "aggregation": bm.get("aggregation"),
                }
    return out


# ---------------------------------------------------------------------------
# Recovery model

def exp_recovery(t, y_baseline, y_peak, tau):
    """y(t) = y_baseline + (y_peak - y_baseline) * exp(-t / tau)."""
    return y_baseline + (y_peak - y_baseline) * np.exp(-t / tau)


def stretched_exp_recovery(t, y_baseline, y_peak, tau, beta):
    """y(t) = y_baseline + (y_peak - y_baseline) * exp(-(t/tau)^beta)."""
    return y_baseline + (y_peak - y_baseline) * np.exp(-((t / tau) ** beta))


def linear_recovery(t, y_peak, k):
    """y(t) = y_peak - k * t (null model: constant-rate recovery)."""
    return y_peak - k * t


def aic(rss: float, n: int, k: int) -> float:
    """AIC for least-squares fit. k = number of free parameters."""
    if n <= k or rss <= 0:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def r_squared(y, y_hat) -> float:
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Stats

def bootstrap_ci(values: np.ndarray, statistic=np.median, n_boot=2000,
                 alpha=0.05, rng=None) -> tuple[float, float, float]:
    """Return (point estimate, lo, hi) for a percentile bootstrap CI."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan, np.nan)
    rng = rng if rng is not None else np.random.default_rng(0)
    point = float(statistic(values))
    boots = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = statistic(values[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (point, lo, hi)


def age_stratum(age: float, strata: list) -> str | None:
    for lo, hi in strata:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return None


# ---------------------------------------------------------------------------
# Transforms

def apply_transform(x: np.ndarray, transform: str) -> np.ndarray:
    """Apply log/identity transform; clip log inputs to a small positive value."""
    x = np.asarray(x, dtype=float)
    if transform == "log":
        return np.log(np.clip(x, 1e-3, None))
    return x


# ---------------------------------------------------------------------------
# IO

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(f"Not JSON-serialisable: {type(o)}")


# ---------------------------------------------------------------------------
# Sepsis-3 (lightweight implementation -- used in step2)

def sepsis3_flag(admission_features: pd.DataFrame) -> pd.Series:
    """
    Apply a simplified Sepsis-3 rule: SOFA >= 2 AND suspected infection.
    Caller must supply per-admission columns:
      - sofa_score
      - has_infection_dx (bool, from ICD prefixes)
    """
    needed = {"sofa_score", "has_infection_dx"}
    missing = needed - set(admission_features.columns)
    if missing:
        raise ValueError(f"sepsis3_flag missing columns: {missing}")
    return (admission_features["sofa_score"] >= 2) & admission_features["has_infection_dx"].astype(bool)


# ---------------------------------------------------------------------------
# Logging helper

def banner(msg: str, char: str = "=", width: int = 70) -> None:
    line = char * width
    print(line)
    print(msg)
    print(line)
