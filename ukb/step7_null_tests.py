"""
Step 7 — Null tests for the λ_max age trend.

    1. Age-permutation null: shuffle age labels, recompute max/min λ_max ratio.
    2. Random-panel null: sample K biomarkers from UKB candidates, compute
       their age-stratified λ_max ratio; compare observed HDR ratio to the
       distribution.
    3. Univariate control: λ_max / max(σ²_i) at each stratum.
    4. Axis-substitution null: substitute a different biomarker for each HDR
       axis from the same physiological domain.

Replicates Supplementary Note 11 of the HDR manuscript on larger data.

Usage:
    python step7_null_tests.py [--config config.yaml] [--tier tier2]
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
    config_hash,
    header_comment,
    lambda_max_of_cov,
    max_axis_variance,
    permutation_trend_test,
    random_panel_null,
    save_json,
)
from step3_cross_sectional import AXIS_COLS, _compute_reference, _standardize, _feasible_axes


# Candidate UKB biomarkers for the random-panel null.
# These are continuous fields with N > 100,000 non-missing at baseline.
RANDOM_PANEL_CANDIDATES = [
    ("albumin",       "30600"),
    ("alk_phos",      "30610"),
    ("ALT",           "30620"),
    ("AST",           "30630"),
    ("bilirubin_tot", "30840"),
    ("calcium",       "30680"),
    ("cholesterol",   "30690"),
    ("creatinine",    "30700"),
    ("GGT",           "30730"),
    ("HDL",           "30760"),
    ("IGF1",          "30770"),
    ("LDL",           "30780"),
    ("LipoA",         "30790"),
    ("phosphate",     "30810"),
    ("SHBG",          "30830"),
    ("testosterone",  "30850"),
    ("total_protein", "30860"),
    ("triglycerides", "30870"),
    ("urate",         "30880"),
    ("urea",          "30670"),
    ("vitaminD",      "30890"),
    ("haemoglobin",   "30020"),
    ("RBC",           "30010"),
    ("WBC",           "30000"),
    ("platelet",      "30080"),
]


AXIS_SUBSTITUTES = {
    # Map HDR axis → alternative biomarker columns in the extracted panel.
    # If the alternative is not available, substitution is skipped.
    "I": ["log_alb_proxy"],       # placeholder — albumin z (inverse)
    "M": ["bmi"],                 # instead of HbA1c
    "F": ["grip_max"],            # same axis, no effective substitute here
    "N": ["sbp"],                 # instead of pulse
    "C": ["sleep_duration"],
    "B": ["bmd"],
    "P": ["log_cystatin_c_proxy"],
}


def _resolve_random_panel(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return columns for random-panel candidates that are present in the panel."""
    # The extracted panel might not include all these fields — we look for the
    # column pattern at baseline instance only. If the user wants the random
    # panel to be fully exercised, they can add the candidate fields to the
    # data export and re-run step1.
    from step1_extract import _resolve_field_column
    cols_present = {}
    base = panel[panel["instance"] == panel["instance"].min()]
    for name, fid in RANDOM_PANEL_CANDIDATES:
        # In the extracted panel we used explicit names (crp, hba1c, etc.), but
        # if the source file provided these fields they may appear as raw
        # column names (e.g. '30600-0.0'). Search for a matching column.
        candidate_names = [
            name, f"field_{fid}",
            _resolve_field_column(list(base.columns), fid, 0) or "",
        ]
        for cname in candidate_names:
            if cname and cname in base.columns and base[cname].notna().sum() > 1000:
                cols_present[name] = cname
                break
    print(f"[step7] Random-panel candidates available: {len(cols_present)} / "
          f"{len(RANDOM_PANEL_CANDIDATES)}")
    return cols_present


def run_tier(panel: pd.DataFrame, cfg: dict, tier_name: str,
             rng: np.random.Generator) -> dict:
    axis_cols = _feasible_axes(panel, cfg["analysis"][f"{tier_name}_axes"])
    if len(axis_cols) < 3:
        return {"tier": tier_name, "status": "skipped"}
    axis_labels = [k for k, v in AXIS_COLS.items() if v in axis_cols]
    print(f"[step7] {tier_name}: axes = {axis_labels}")

    base = panel[panel["instance"] == panel["instance"].min()].copy()
    ref_lo, ref_hi = cfg["analysis"]["youthful_reference_age"]
    ref = _compute_reference(base, ref_lo, ref_hi, axis_cols)
    base_std = _standardize(base, ref, axis_cols)

    strata = cfg["analysis"]["age_strata"]

    # Observed λ_max ratio
    observed_ratio = _observed_ratio(base_std, axis_cols, strata)

    # 1. Age-permutation null
    perm_res = permutation_trend_test(
        base_std[axis_cols + ["age"]].dropna(),
        axis_cols, "age", strata,
        n_perm=cfg["analysis"]["n_permutations"], rng=rng,
    )

    # 2. Random-panel null
    rp_cols = _resolve_random_panel(panel, cfg)
    rp_result = {}
    if rp_cols:
        base_rp = base.copy()
        rp_result = random_panel_null(
            base_rp, list(rp_cols.values()),
            "age", strata, k=len(axis_cols),
            n_panels=cfg["analysis"]["n_random_panels"],
            observed_ratio=observed_ratio, rng=rng,
        )

    # 3. Univariate control by stratum
    univar_per_stratum: List[dict] = []
    for lo, hi in strata:
        mask = base_std["age"].between(lo, hi)
        X = base_std.loc[mask, axis_cols].dropna().to_numpy()
        if len(X) < 50:
            continue
        lm = lambda_max_of_cov(X)
        max_var = max_axis_variance(X)
        univar_per_stratum.append({
            "stratum": f"{lo}-{hi}",
            "n": int(len(X)),
            "lambda_max": lm,
            "max_axis_variance": max_var,
            "ratio": float(lm / max_var) if max_var > 0 else None,
        })

    # 4. Axis-substitution null
    substitution_results: Dict[str, float] = {}
    for ax_label in axis_labels:
        subs = AXIS_SUBSTITUTES.get(ax_label, [])
        for sub_col in subs:
            if sub_col not in base.columns or base[sub_col].notna().sum() < 1000:
                continue
            new_cols = [AXIS_COLS[a] for a in axis_labels if a != ax_label] + [sub_col]
            new_cols = [c for c in new_cols if c in base.columns]
            ratio = _observed_ratio(base, new_cols, strata)
            substitution_results[f"{ax_label}→{sub_col}"] = ratio
            break

    return {
        "tier": tier_name,
        "axes": axis_labels,
        "observed_ratio": observed_ratio,
        "age_permutation_null": perm_res,
        "random_panel_null": rp_result,
        "univariate_control": univar_per_stratum,
        "axis_substitution": substitution_results,
    }


def _observed_ratio(base: pd.DataFrame, cols: Sequence[str],
                    strata: Sequence) -> float:
    lmax_vals: List[float] = []
    for lo, hi in strata:
        mask = base["age"].between(lo, hi)
        X = base.loc[mask, cols].dropna().to_numpy()
        if len(X) < 50:
            continue
        v = lambda_max_of_cov(X)
        if np.isfinite(v) and v > 0:
            lmax_vals.append(v)
    if len(lmax_vals) < 2:
        return float("nan")
    return float(max(lmax_vals) / min(lmax_vals))


def _plot(result: dict, out_path: Path) -> None:
    if result.get("status"):
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ag = result["age_permutation_null"]
    axes[0, 0].axvline(ag.get("observed_ratio", np.nan), color="red",
                       label=f"observed={ag.get('observed_ratio')}")
    axes[0, 0].axvline(ag.get("null_p95", np.nan), color="gray", linestyle="--",
                       label=f"null p95={ag.get('null_p95')}")
    axes[0, 0].set_title(f"Age-permutation null (p = {ag.get('p_value')})")
    axes[0, 0].set_xlabel("max/min λ_max ratio")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    rp = result["random_panel_null"]
    if rp.get("null_ratios"):
        axes[0, 1].hist(rp["null_ratios"], bins=30, alpha=0.7)
        axes[0, 1].axvline(result["observed_ratio"], color="red",
                           label=f"observed={result['observed_ratio']:.2f}")
        axes[0, 1].set_title(f"Random-panel null (p = {rp.get('p_value')})")
        axes[0, 1].set_xlabel("max/min λ_max ratio")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    if result["univariate_control"]:
        xs = [r["stratum"] for r in result["univariate_control"]]
        ys = [r["ratio"] for r in result["univariate_control"]]
        axes[1, 0].plot(xs, ys, "o-")
        axes[1, 0].axhline(1.0, color="gray", linestyle="--")
        axes[1, 0].set_title("Univariate control by stratum")
        axes[1, 0].set_ylabel("λ_max / max σ²")
        axes[1, 0].grid(alpha=0.3)

    if result["axis_substitution"]:
        labels = list(result["axis_substitution"].keys())
        vals = list(result["axis_substitution"].values())
        axes[1, 1].barh(labels, vals)
        axes[1, 1].axvline(result["observed_ratio"], color="red", linestyle="--",
                           label=f"HDR ratio={result['observed_ratio']:.2f}")
        axes[1, 1].set_title("Axis substitution")
        axes[1, 1].set_xlabel("max/min λ_max ratio")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Null tests — {result['tier']}")
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

    tiers = [args.tier] if args.tier else ["tier1", "tier2", "tier3", "tier4"]
    for t in tiers:
        print(f"[step7] Running {t}…")
        res = run_tier(panel, cfg, t, rng)
        res["_header"] = header_comment("step7_null_tests.py", config_hash(cfg))
        save_json(out_dir / f"ukb_null_tests_{t}.json", res)
        _plot(res, out_dir / f"figure_ukb_null_tests_{t}.pdf")
        print(f"[step7] Wrote ukb_null_tests_{t}.json + figure.")


if __name__ == "__main__":
    main()
