"""
Step 4 — Within-person visit-pair change-covariance.

Core HDR test. For participants with complete axis data at two instances
(typically baseline + imaging), computes Δ(Δx) = Δx(t2) − Δx(t0) and the
age-stratified change-covariance matrix. Outputs λ_max trajectory, Kendall τ
trend, permutation-trend p-value, univariate control, Π decomposition, and
medication-stratified trajectories.

Usage:
    python step4_longitudinal.py [--config config.yaml] [--tier tier2]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
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
    permutation_trend_test,
    save_json,
)
from step3_cross_sectional import AXIS_COLS, _compute_reference, _standardize, _feasible_axes


def _build_change_panel(panel: pd.DataFrame, axis_cols: Sequence[str]) -> pd.DataFrame:
    """Build within-person visit-pair change vectors."""
    panel = panel.sort_values(["eid", "instance"]).copy()
    rows: List[dict] = []
    for eid, grp in panel.groupby("eid", sort=False):
        grp = grp.sort_values("instance")
        for i in range(len(grp) - 1):
            r0 = grp.iloc[i]
            r1 = grp.iloc[i + 1]
            vals_0 = r0[axis_cols].to_numpy(dtype=float)
            vals_1 = r1[axis_cols].to_numpy(dtype=float)
            if not (np.all(np.isfinite(vals_0)) and np.all(np.isfinite(vals_1))):
                continue
            row = {
                "eid": eid,
                "age_t0": r0["age"],
                "age_t1": r1["age"],
                "age_mid": (r0["age"] + r1["age"]) / 2,
                "instance_t0": int(r0["instance"]),
                "instance_t1": int(r1["instance"]),
            }
            for col in axis_cols:
                row[f"d_{col}"] = r1[col] - r0[col]
            # Preserve medication flags from baseline row (for stratification)
            for flag in ("med_statin", "med_antihtn", "med_insulin", "n_med_classes"):
                if flag in r0:
                    row[flag] = r0[flag]
            rows.append(row)
    return pd.DataFrame(rows)


def _lambda_trajectory(changes: pd.DataFrame, delta_cols: List[str],
                       strata: Sequence, n_boot: int,
                       rng: np.random.Generator,
                       j_signs: dict, axis_labels: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    for lo, hi in strata:
        mask = changes["age_mid"].between(lo, hi)
        X = changes.loc[mask, delta_cols].to_numpy(dtype=float)
        X = X[np.all(np.isfinite(X), axis=1)]
        name = f"{lo}-{hi}"
        if len(X) < 50:
            out.append({"stratum": name, "n": int(len(X)), "status": "insufficient"})
            continue
        Gamma = np.cov(X, rowvar=False)
        proxy = gamma_stability_proxy(Gamma)
        ci_lo, ci_hi = bootstrap_lambda_max(X, n_boot=n_boot, rng=rng)
        univar = proxy["lambda_max"] / max_axis_variance(X)
        pi_d = compute_pi_decomposition(Gamma)
        sign_c = covariance_sign_concordance(Gamma, axis_labels, j_signs)
        out.append({
            "stratum": name,
            "n": int(len(X)),
            "lambda_max": proxy["lambda_max"],
            "lambda_max_ci": [ci_lo, ci_hi],
            "trace": proxy["trace"],
            "kappa": proxy["kappa"],
            "univar_ratio": float(univar) if np.isfinite(univar) else None,
            "Pi": pi_d,
            "sign_concordance": sign_c,
        })
    return out


def run_tier(panel: pd.DataFrame, cfg: dict, tier_name: str,
             j_signs: dict, rng: np.random.Generator) -> dict:
    tier_axes_raw = cfg["analysis"][f"{tier_name}_axes"]
    axis_cols = _feasible_axes(panel, tier_axes_raw)
    if len(axis_cols) < 3:
        return {"tier": tier_name, "status": "skipped"}
    axis_labels = [k for k, v in AXIS_COLS.items() if v in axis_cols]
    print(f"[step4] {tier_name}: axes = {axis_labels}")

    base = panel[panel["instance"] == panel["instance"].min()].copy()
    ref_lo, ref_hi = cfg["analysis"]["youthful_reference_age"]
    ref = _compute_reference(base, ref_lo, ref_hi, axis_cols)
    panel_std = _standardize(panel.copy(), ref, axis_cols)

    print("[step4] Building change panel…")
    changes = _build_change_panel(panel_std, axis_cols)
    print(f"[step4] {tier_name}: {len(changes):,} change-pairs.")
    if len(changes) < 200:
        return {"tier": tier_name, "status": "insufficient_pairs",
                "n_pairs": int(len(changes))}

    delta_cols = [f"d_{c}" for c in axis_cols]
    strata = cfg["analysis"]["age_strata"]
    n_boot = cfg["analysis"]["n_bootstrap"]

    # Full-sample trajectory
    traj_all = _lambda_trajectory(
        changes, delta_cols, strata, n_boot, rng, j_signs, axis_labels
    )

    # Kendall tau trend test
    xs, ys = [], []
    for s, r in zip(strata, traj_all):
        if r.get("lambda_max") is None:
            continue
        xs.append((s[0] + s[1]) / 2)
        ys.append(r["lambda_max"])
    tau_res = {"tau": None, "p_value": None}
    if len(xs) >= 3:
        tau, p = kendalltau(xs, ys)
        tau_res = {"tau": float(tau), "p_value": float(p)}

    # Permutation trend test
    perm_panel = changes[delta_cols + ["age_mid"]].copy()
    perm = permutation_trend_test(
        perm_panel, delta_cols, "age_mid", strata,
        n_perm=cfg["analysis"]["n_permutations"], rng=rng,
    )

    # Medication-stratified
    med_results: Dict[str, list] = {}
    if "n_med_classes" in changes.columns:
        naive = changes[changes["n_med_classes"].fillna(0) == 0]
        med_on = changes[changes["n_med_classes"].fillna(0) >= 1]
        if len(naive) > 200:
            med_results["medication_naive"] = _lambda_trajectory(
                naive, delta_cols, strata, n_boot, rng, j_signs, axis_labels
            )
        if len(med_on) > 200:
            med_results["on_medication"] = _lambda_trajectory(
                med_on, delta_cols, strata, n_boot, rng, j_signs, axis_labels
            )

    lmax_vals = [r["lambda_max"] for r in traj_all if r.get("lambda_max") is not None]
    lmax_ratio = (
        max(lmax_vals) / min(lmax_vals) if lmax_vals and min(lmax_vals) > 0
        else float("nan")
    )

    return {
        "tier": tier_name,
        "axes": axis_labels,
        "n_pairs": int(len(changes)),
        "per_stratum": traj_all,
        "trend_kendall": tau_res,
        "permutation_trend": perm,
        "lambda_max_ratio": lmax_ratio,
        "medication_stratified": med_results,
    }


def _plot(result: dict, out_path: Path) -> None:
    if result.get("status"):
        return
    strata = [r for r in result["per_stratum"] if r.get("lambda_max") is not None]
    if not strata:
        return
    ages = [r["stratum"] for r in strata]
    lmax = np.array([r["lambda_max"] for r in strata])
    ci_lo = np.array([r["lambda_max_ci"][0] for r in strata])
    ci_hi = np.array([r["lambda_max_ci"][1] for r in strata])
    pi_vals = [r["Pi"]["Pi"] for r in strata]
    univar = [r["univar_ratio"] for r in strata]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].errorbar(ages, lmax, yerr=[lmax - ci_lo, ci_hi - lmax],
                        fmt="o-", capsize=4, label="all")
    for key, traj in result.get("medication_stratified", {}).items():
        sub = [r for r in traj if r.get("lambda_max") is not None]
        if not sub:
            continue
        xs = [r["stratum"] for r in sub]
        ys = [r["lambda_max"] for r in sub]
        axes[0, 0].plot(xs, ys, "--", alpha=0.7, label=key)
    axes[0, 0].set_title(f"λ_max(Γ̂_change) — {result['tier']}")
    axes[0, 0].set_ylabel("λ_max")
    axes[0, 0].set_xlabel("Age (mid)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ages, univar, "s-")
    axes[0, 1].axhline(1.0, color="gray", linestyle="--")
    axes[0, 1].set_title("Univariate control")
    axes[0, 1].set_ylabel("λ_max / max σ²")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(ages, pi_vals, "D-")
    axes[1, 0].set_title("Π = C_norm / V_norm")
    axes[1, 0].set_ylabel("Π")
    axes[1, 0].grid(alpha=0.3)

    ns = [r["n"] for r in strata]
    axes[1, 1].bar(ages, ns)
    axes[1, 1].set_title("N per stratum")
    axes[1, 1].set_ylabel("N pairs")
    axes[1, 1].grid(alpha=0.3)

    perm = result.get("permutation_trend", {})
    fig.suptitle(
        f"Longitudinal HDR — {result['tier']} "
        f"(λ_max ratio = {result['lambda_max_ratio']:.2f}x, "
        f"Kendall τ = {result['trend_kendall']['tau']}, "
        f"perm-p = {perm.get('p_value')})"
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
    j_signs = load_j_signs(cfg["j_matrix_path"])
    rng = np.random.default_rng(cfg["analysis"]["seed"])

    tiers = [args.tier] if args.tier else ["tier1", "tier2", "tier3", "tier4"]
    for t in tiers:
        print(f"[step4] Running {t}…")
        res = run_tier(panel, cfg, t, j_signs, rng)
        res["_header"] = header_comment("step4_longitudinal.py", config_hash(cfg))
        save_json(out_dir / f"ukb_longitudinal_{t}.json", res)
        _plot(res, out_dir / f"figure_ukb_longitudinal_{t}.pdf")
        print(f"[step4] Wrote ukb_longitudinal_{t}.json + figure.")


if __name__ == "__main__":
    main()
