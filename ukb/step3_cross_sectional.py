"""
Step 3 — Cross-sectional covariance analysis.

For each feasible tier, computes age-stratified covariance on the baseline
assessment:
    - λ_max(Γ̂) with bootstrap 95% CI
    - per-axis variance by age
    - pairwise |correlation| by age
    - univariate control ratio (λ_max / max axis variance)
    - J-sign concordance
    - Π decomposition (V_norm, C_norm, Π)

Outputs results/ukb_cross_sectional_{tier}.json and figure.

Usage:
    python step3_cross_sectional.py [--config config.yaml] [--tier tier2]
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
    bootstrap_lambda_max,
    compute_pi_decomposition,
    config_hash,
    covariance_sign_concordance,
    gamma_stability_proxy,
    header_comment,
    lambda_max_of_cov,
    load_j_signs,
    max_axis_variance,
    save_json,
)


AXIS_COLS = {
    "I": "delta_I", "M": "delta_M", "F": "delta_F", "N": "delta_N",
    "C": "delta_C", "B": "delta_B", "P": "delta_P",
}


def _compute_reference(base: pd.DataFrame, ref_lo: int, ref_hi: int,
                       axis_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Youthful reference mean + SD per axis."""
    mask = base["age"].between(ref_lo, ref_hi)
    ref = base[mask]
    out: Dict[str, Dict[str, float]] = {}
    for col in axis_cols:
        vals = pd.to_numeric(ref[col], errors="coerce").dropna()
        if vals.empty:
            out[col] = {"mean": 0.0, "sd": 1.0}
        else:
            mu = float(vals.mean())
            sd = float(vals.std(ddof=1)) or 1.0
            out[col] = {"mean": mu, "sd": sd}
    return out


def _standardize(df: pd.DataFrame, ref: Dict[str, Dict[str, float]],
                 axis_cols: Sequence[str]) -> pd.DataFrame:
    """(x - mu_ref) / sd_ref per axis."""
    for col in axis_cols:
        mu = ref[col]["mean"]
        sd = ref[col]["sd"]
        df[col] = (pd.to_numeric(df[col], errors="coerce") - mu) / sd
    return df


def _feasible_axes(panel: pd.DataFrame, tier_axes: Sequence[str],
                   min_nonmissing: int = 1000) -> List[str]:
    cols = [AXIS_COLS[a] for a in tier_axes if AXIS_COLS.get(a)]
    kept = [c for c in cols if c in panel.columns and panel[c].notna().sum() >= min_nonmissing]
    dropped = [c for c in cols if c not in kept]
    if dropped:
        print(f"[step3]   Skipping axes with insufficient data: {dropped}")
    return kept


def run_tier(panel: pd.DataFrame, cfg: dict, tier_name: str,
             j_signs: dict, rng: np.random.Generator) -> dict:
    tier_axes_raw = cfg["analysis"][f"{tier_name}_axes"]
    axis_cols = _feasible_axes(panel, tier_axes_raw)
    if len(axis_cols) < 3:
        print(f"[step3] {tier_name}: < 3 axes with data; skipping.")
        return {"tier": tier_name, "status": "skipped"}

    axis_labels = [k for k, v in AXIS_COLS.items() if v in axis_cols]
    print(f"[step3] {tier_name}: axes = {axis_labels}")

    base = panel[panel["instance"] == panel["instance"].min()].copy()
    ref_lo, ref_hi = cfg["analysis"]["youthful_reference_age"]
    ref = _compute_reference(base, ref_lo, ref_hi, axis_cols)
    base_std = _standardize(base.copy(), ref, axis_cols)

    strata = cfg["analysis"]["age_strata"]
    n_boot = cfg["analysis"]["n_bootstrap"]

    per_stratum: List[dict] = []
    Gamma_by_stratum: Dict[str, np.ndarray] = {}
    for lo, hi in tqdm(strata, desc=f"{tier_name} strata"):
        mask = base_std["age"].between(lo, hi)
        X = base_std.loc[mask, axis_cols].dropna().to_numpy()
        name = f"{lo}-{hi}"
        if len(X) < 50:
            per_stratum.append({"stratum": name, "n": int(len(X)), "status": "insufficient"})
            continue
        Gamma = np.cov(X, rowvar=False)
        Gamma_by_stratum[name] = Gamma
        proxy = gamma_stability_proxy(Gamma)
        ci_lo, ci_hi = bootstrap_lambda_max(X, n_boot=n_boot, rng=rng)
        max_var = max_axis_variance(X)
        univar_ratio = proxy["lambda_max"] / max_var if max_var > 0 else float("nan")
        pi_d = compute_pi_decomposition(Gamma)
        sign_c = covariance_sign_concordance(Gamma, axis_labels, j_signs)
        corr = np.corrcoef(X, rowvar=False)
        n_off = len(axis_cols) * (len(axis_cols) - 1)
        mean_abs_corr = float(
            np.abs(corr - np.eye(len(axis_cols))).sum() / max(n_off, 1)
        )
        per_stratum.append({
            "stratum": name,
            "n": int(len(X)),
            "lambda_max": proxy["lambda_max"],
            "lambda_max_ci": [ci_lo, ci_hi],
            "trace": proxy["trace"],
            "kappa": proxy["kappa"],
            "per_axis_variance": {
                a: float(np.var(X[:, i], ddof=1)) for i, a in enumerate(axis_labels)
            },
            "mean_abs_corr": mean_abs_corr,
            "univar_ratio": float(univar_ratio),
            "Pi": pi_d,
            "sign_concordance": sign_c,
        })

    # Trend test: Kendall tau on λ_max vs age midpoint
    from scipy.stats import kendalltau
    trend = {"tau": None, "p_value": None}
    xs, ys = [], []
    for s, row in zip(strata, per_stratum):
        if row.get("status") == "insufficient":
            continue
        mid = (s[0] + s[1]) / 2
        xs.append(mid)
        ys.append(row["lambda_max"])
    if len(xs) >= 3:
        tau, p = kendalltau(xs, ys)
        trend = {"tau": float(tau), "p_value": float(p)}

    # λ_max max/min ratio
    lmax_vals = [r["lambda_max"] for r in per_stratum if "lambda_max" in r]
    lmax_ratio = (
        float(max(lmax_vals) / min(lmax_vals))
        if lmax_vals and min(lmax_vals) > 0 else float("nan")
    )

    payload = {
        "tier": tier_name,
        "axes": axis_labels,
        "axis_cols": axis_cols,
        "youthful_reference_age": [ref_lo, ref_hi],
        "reference_mu_sd": ref,
        "per_stratum": per_stratum,
        "trend_kendall": trend,
        "lambda_max_ratio": lmax_ratio,
    }
    return payload


def _plot(result: dict, out_path: Path) -> None:
    if result.get("status") == "skipped":
        return
    strata = [r for r in result["per_stratum"] if r.get("lambda_max") is not None]
    if not strata:
        return
    ages = [r["stratum"] for r in strata]
    lmax = [r["lambda_max"] for r in strata]
    ci_lo = [r["lambda_max_ci"][0] for r in strata]
    ci_hi = [r["lambda_max_ci"][1] for r in strata]
    univar = [r["univar_ratio"] for r in strata]
    mac = [r["mean_abs_corr"] for r in strata]
    pi_vals = [r["Pi"]["Pi"] for r in strata]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].errorbar(
        ages, lmax,
        yerr=[np.array(lmax) - np.array(ci_lo), np.array(ci_hi) - np.array(lmax)],
        fmt="o-", capsize=4,
    )
    axes[0, 0].set_title(f"λ_max(Γ̂) — {result['tier']}")
    axes[0, 0].set_ylabel("λ_max")
    axes[0, 0].set_xlabel("Age stratum")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ages, univar, "s-")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--")
    axes[0, 1].set_title("Univariate control: λ_max / max σ²")
    axes[0, 1].set_ylabel("ratio")
    axes[0, 1].set_xlabel("Age stratum")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(ages, mac, "^-")
    axes[1, 0].set_title("Mean |off-diagonal correlation|")
    axes[1, 0].set_ylabel("mean |r|")
    axes[1, 0].set_xlabel("Age stratum")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ages, pi_vals, "D-")
    axes[1, 1].set_title("Π = C_norm / V_norm")
    axes[1, 1].set_ylabel("Π")
    axes[1, 1].set_xlabel("Age stratum")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(
        f"Cross-sectional HDR — {result['tier']} "
        f"(λ_max ratio = {result['lambda_max_ratio']:.2f}x, "
        f"Kendall τ = {result['trend_kendall']['tau']}, "
        f"p = {result['trend_kendall']['p_value']})"
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tier", default=None, help="Run a single tier (e.g. tier2).")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])
    panel = pd.read_parquet(out_dir / "ukb_panel_long.parquet")
    j_signs = load_j_signs(cfg["j_matrix_path"])
    rng = np.random.default_rng(cfg["analysis"]["seed"])

    tiers = [args.tier] if args.tier else ["tier1", "tier2", "tier3", "tier4"]
    for t in tiers:
        print(f"[step3] Running {t}…")
        res = run_tier(panel, cfg, t, j_signs, rng)
        res["_header"] = header_comment("step3_cross_sectional.py", config_hash(cfg))
        save_json(out_dir / f"ukb_cross_sectional_{t}.json", res)
        _plot(res, out_dir / f"figure_ukb_cross_sectional_{t}.pdf")
        print(f"[step3] Wrote ukb_cross_sectional_{t}.json + figure.")


if __name__ == "__main__":
    main()
