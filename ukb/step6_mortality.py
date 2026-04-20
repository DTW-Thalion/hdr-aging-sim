"""
Step 6 — Cox mortality models.

Matched complete-case sample across M1–M5 (and comparators M3a, M3b, M4a,
M4b). Event-stratified subject-level bootstrap for ΔC 95% CI. Subgroup
analyses by medication, sex, age stratum, ethnicity.

Usage:
    python step6_mortality.py [--config config.yaml] [--tier tier2]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

from hdr_core import (
    compute_swds_gamma_batch,
    config_hash,
    header_comment,
    save_json,
)
from step3_cross_sectional import AXIS_COLS, _compute_reference, _standardize, _feasible_axes

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def _fit_c(df: pd.DataFrame, covs: List[str]) -> float:
    """Fit Cox with given covariates; return C-index. NaN-safe."""
    if not HAS_LIFELINES:
        return float("nan")
    cols = ["time_years", "event"] + covs
    sub = df[cols].dropna()
    if len(sub) < 50 or sub["event"].sum() < 10:
        return float("nan")
    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(sub, duration_col="time_years", event_col="event",
                show_progress=False)
        return float(cph.concordance_index_)
    except Exception as e:
        print(f"[step6] Cox failed ({covs}): {e}")
        return float("nan")


def _build_baseline_sample(panel: pd.DataFrame, cfg: dict,
                            axis_cols: List[str]) -> pd.DataFrame:
    """Baseline row + mortality + SWDS-Γ. Matched complete-case across models."""
    base = panel[panel["instance"] == panel["instance"].min()].copy()
    ref_lo, ref_hi = cfg["analysis"]["youthful_reference_age"]
    ref = _compute_reference(base, ref_lo, ref_hi, axis_cols)
    base = _standardize(base, ref, axis_cols)

    # SWDS-Γ from baseline covariance
    X = base[axis_cols].dropna().to_numpy()
    if len(X) < 100:
        return pd.DataFrame()
    Gamma = np.cov(X, rowvar=False)
    complete = base[axis_cols].notna().all(axis=1)
    swds = np.full(len(base), np.nan)
    swds[complete.values] = compute_swds_gamma_batch(
        base.loc[complete, axis_cols].to_numpy(), Gamma
    )
    base["swds_gamma"] = swds
    base["log_swds"] = np.log(np.clip(base["swds_gamma"], 1e-4, None))

    # Mahalanobis (uses Γ inverse)
    inv_G = np.linalg.pinv(Gamma)
    maha = np.full(len(base), np.nan)
    maha[complete.values] = np.einsum(
        "ij,jk,ik->i",
        base.loc[complete, axis_cols].to_numpy(),
        inv_G,
        base.loc[complete, axis_cols].to_numpy(),
    )
    base["mahalanobis"] = maha
    base["log_maha"] = np.log(np.clip(maha, 1e-4, None))

    # Z-sum: raw sum of z-scores
    base["z_sum"] = base[axis_cols].sum(axis=1)

    # Covariates
    base["female"] = (base["sex"] == 0).astype(float)
    base["smoking_current"] = (base["smoking"] == 2).astype(float)

    return base


def _event_stratified_bootstrap(df: pd.DataFrame, covs_base: List[str],
                                 covs_full: List[str], n_boot: int,
                                 rng: np.random.Generator) -> Dict[str, float]:
    """Bootstrap ΔC = C(full) − C(base)."""
    if not HAS_LIFELINES:
        return {"mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n_boot": 0}
    events_idx = df.index[df["event"] == 1].to_numpy()
    non_events_idx = df.index[df["event"] == 0].to_numpy()
    if len(events_idx) < 10 or len(non_events_idx) < 10:
        return {"mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n_boot": 0}
    dcs = np.full(n_boot, np.nan)
    for b in tqdm(range(n_boot), desc="bootstrap ΔC", leave=False):
        e_sample = rng.choice(events_idx, size=len(events_idx), replace=True)
        ne_sample = rng.choice(non_events_idx, size=len(non_events_idx), replace=True)
        idx = np.concatenate([e_sample, ne_sample])
        sub = df.loc[idx].reset_index(drop=True)
        c_base = _fit_c(sub, covs_base)
        c_full = _fit_c(sub, covs_full)
        if np.isfinite(c_base) and np.isfinite(c_full):
            dcs[b] = c_full - c_base
    valid = dcs[np.isfinite(dcs)]
    if len(valid) < 50:
        return {"mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "n_boot": int(len(valid))}
    return {
        "mean": float(np.mean(valid)),
        "ci_lower": float(np.percentile(valid, 2.5)),
        "ci_upper": float(np.percentile(valid, 97.5)),
        "n_boot": int(len(valid)),
    }


def run_tier(panel: pd.DataFrame, cfg: dict, tier_name: str,
             rng: np.random.Generator) -> dict:
    axis_cols = _feasible_axes(panel, cfg["analysis"][f"{tier_name}_axes"])
    if len(axis_cols) < 3:
        return {"tier": tier_name, "status": "skipped"}
    axis_labels = [k for k, v in AXIS_COLS.items() if v in axis_cols]

    base = _build_baseline_sample(panel, cfg, axis_cols)
    if base.empty:
        return {"tier": tier_name, "status": "no_baseline"}

    # Biomarker columns (raw) for M2/M4a
    biomarker_cols = []
    for col in ["crp", "hba1c", "grip_max", "pulse_rate", "cystatin_c",
                "bmd", "bmi"]:
        if col in base.columns and base[col].notna().any():
            # log-transform skewed
            if col in ("crp", "cystatin_c"):
                base[f"log_{col}"] = np.log(np.clip(base[col], 1e-3, None))
                biomarker_cols.append(f"log_{col}")
            else:
                biomarker_cols.append(col)

    # All-model matched complete-case
    required = (
        ["time_years", "event", "age", "female", "smoking_current",
         "co_diabetes", "co_hypertension",
         "frailty_index", "log_swds", "log_maha", "z_sum"]
        + biomarker_cols
    )
    required = [c for c in required if c in base.columns]
    matched = base.dropna(subset=required).copy()
    matched = matched[matched["time_years"] > 0]
    n_matched = len(matched)
    n_events = int(matched["event"].sum())
    print(f"[step6] {tier_name}: matched N={n_matched:,}, deaths={n_events:,}")
    if n_matched < 500 or n_events < 50:
        return {"tier": tier_name, "status": "insufficient_sample",
                "n": n_matched, "events": n_events}

    # Fit models
    M1 = ["age", "female"]
    M_adj = ["age", "female", "smoking_current", "co_diabetes", "co_hypertension"]
    M2 = M_adj + biomarker_cols
    M3 = ["log_swds"]
    M3a = ["log_maha"]
    M3b = ["z_sum"]
    M4 = M_adj + ["frailty_index"]
    M4a = M4 + biomarker_cols
    M4b = M4 + ["log_swds"]
    M5 = M4 + biomarker_cols + ["log_swds"]

    models = {}
    for name, covs in [
        ("M1_age_sex", M1),
        ("M2_biomarkers", M2),
        ("M3_swds", M3),
        ("M3a_mahalanobis", M3a),
        ("M3b_z_sum", M3b),
        ("M4_frailty", M4),
        ("M4a_frailty_bio", M4a),
        ("M4b_frailty_swds", M4b),
        ("M5_full", M5),
    ]:
        c = _fit_c(matched, covs)
        models[name] = {"C": c, "N": int(len(matched)), "covs": covs}
        print(f"[step6]   {name}: C = {c:.4f}")

    # ΔC bootstrap (key comparisons)
    n_boot_cox = cfg["analysis"]["n_bootstrap_cox"]
    delta_bootstraps = {}
    for label, base_covs, full_covs in [
        ("delta_C_M5_M4",  M4,  M5),
        ("delta_C_M5_M4a", M4a, M5),
        ("delta_C_M4b_M4", M4,  M4b),
    ]:
        print(f"[step6]   bootstrapping {label}…")
        delta_bootstraps[label] = _event_stratified_bootstrap(
            matched, base_covs, full_covs, n_boot_cox, rng
        )

    # Subgroups
    subgroup_results = {}
    subgroup_masks = {
        "medication_naive": matched["n_med_classes"].fillna(0) == 0,
        "male": matched["female"] == 0,
        "female": matched["female"] == 1,
    }
    for lo, hi in cfg["analysis"]["age_strata"]:
        subgroup_masks[f"age_{lo}_{hi}"] = matched["age"].between(lo, hi)
    for key, mask in subgroup_masks.items():
        sub = matched[mask.values if isinstance(mask, pd.Series) else mask]
        if len(sub) < 500 or sub["event"].sum() < 20:
            continue
        sub_models = {}
        for name, covs in [("M4", M4), ("M4a", M4a), ("M5", M5)]:
            sub_models[f"C_{name}"] = _fit_c(sub, covs)
        subgroup_results[key] = {
            "n": int(len(sub)),
            "events": int(sub["event"].sum()),
            **sub_models,
            "delta_C_M5_M4": float(sub_models["C_M5"] - sub_models["C_M4"])
            if np.isfinite(sub_models["C_M5"]) and np.isfinite(sub_models["C_M4"])
            else None,
            "delta_C_M5_M4a": float(sub_models["C_M5"] - sub_models["C_M4a"])
            if np.isfinite(sub_models["C_M5"]) and np.isfinite(sub_models["C_M4a"])
            else None,
        }

    return {
        "tier": tier_name,
        "axes": axis_labels,
        "n_matched": n_matched,
        "n_events": n_events,
        "biomarker_cols": biomarker_cols,
        "models": models,
        "delta_bootstraps": delta_bootstraps,
        "subgroups": subgroup_results,
    }


def _plot(result: dict, out_path: Path) -> None:
    if result.get("status"):
        return
    models = result["models"]
    order = ["M1_age_sex", "M2_biomarkers", "M3_swds", "M3a_mahalanobis",
             "M3b_z_sum", "M4_frailty", "M4a_frailty_bio", "M4b_frailty_swds",
             "M5_full"]
    names = [m for m in order if m in models]
    c_vals = [models[m]["C"] for m in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, c_vals)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("C-index")
    ax.set_xlim(0.5, max(c_vals) * 1.02 if c_vals else 1.0)
    for i, v in enumerate(c_vals):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center")
    db = result["delta_bootstraps"].get("delta_C_M5_M4a", {})
    ax.set_title(
        f"Mortality C-index — {result['tier']} "
        f"(N={result['n_matched']:,}, events={result['n_events']:,}; "
        f"ΔC(M5-M4a) = {db.get('mean')} "
        f"[{db.get('ci_lower')}, {db.get('ci_upper')}])"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tier", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["output_dir"])
    panel = pd.read_parquet(out_dir / "ukb_panel_long.parquet")
    rng = np.random.default_rng(cfg["analysis"]["seed"])

    if not HAS_LIFELINES:
        print("[step6] WARNING: lifelines not installed; skipping.")
        return

    tiers = [args.tier] if args.tier else ["tier1", "tier2", "tier3", "tier4"]
    for t in tiers:
        print(f"[step6] Running {t}…")
        res = run_tier(panel, cfg, t, rng)
        res["_header"] = header_comment("step6_mortality.py", config_hash(cfg))
        save_json(out_dir / f"ukb_mortality_{t}.json", res)
        _plot(res, out_dir / f"figure_ukb_mortality_{t}.pdf")
        print(f"[step6] Wrote ukb_mortality_{t}.json + figure.")


if __name__ == "__main__":
    main()
