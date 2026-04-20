"""
Step 5 — Cross-lagged regression (lead-lag analysis).

For each ordered axis pair (i → j), fit:
    Δx_j(t2) − Δx_j(t0) ~ β · Δx_i(t0) + γ · Δx_j(t0) + δ · age_t0 + c

β is the cross-lagged effect. Its sign is compared to the compiled J-matrix
prediction. Subject-clustered bootstrap 95% CI, BH-FDR within tier, sign
concordance vs J, subgroup analyses.

Usage:
    python step5_lead_lag.py [--config config.yaml] [--tier tier2]
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
    lead_lag_all_pairs,
    load_j_signs,
    save_json,
)
from step3_cross_sectional import AXIS_COLS, _compute_reference, _standardize, _feasible_axes


def _build_triplets(panel_std: pd.DataFrame, axis_labels: Sequence[str],
                    axis_cols: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame with columns <label>_t0, <label>_t1, d_<label>, age_t, subject_id."""
    panel_std = panel_std.sort_values(["eid", "instance"])
    rows: List[dict] = []
    for eid, grp in panel_std.groupby("eid", sort=False):
        grp = grp.sort_values("instance")
        for i in range(len(grp) - 1):
            r0, r1 = grp.iloc[i], grp.iloc[i + 1]
            vals_0 = r0[axis_cols].to_numpy(dtype=float)
            vals_1 = r1[axis_cols].to_numpy(dtype=float)
            if not (np.all(np.isfinite(vals_0)) and np.all(np.isfinite(vals_1))):
                continue
            row: Dict[str, float] = {
                "subject_id": eid,
                "age_t": float(r0["age"]),
            }
            for label, col in zip(axis_labels, axis_cols):
                row[f"{label}_t0"] = float(r0[col])
                row[f"{label}_t1"] = float(r1[col])
                row[f"d_{label}"] = float(r1[col] - r0[col])
            for flag in ("sex", "med_statin", "med_antihtn", "med_insulin",
                         "n_med_classes", "ethnicity"):
                if flag in r0:
                    row[flag] = r0[flag]
            rows.append(row)
    return pd.DataFrame(rows)


def run_tier(panel: pd.DataFrame, cfg: dict, tier_name: str,
             j_signs: dict, rng: np.random.Generator) -> dict:
    axis_cols = _feasible_axes(panel, cfg["analysis"][f"{tier_name}_axes"])
    if len(axis_cols) < 2:
        return {"tier": tier_name, "status": "skipped"}
    axis_labels = [k for k, v in AXIS_COLS.items() if v in axis_cols]
    print(f"[step5] {tier_name}: axes = {axis_labels}")

    base = panel[panel["instance"] == panel["instance"].min()].copy()
    ref_lo, ref_hi = cfg["analysis"]["youthful_reference_age"]
    ref = _compute_reference(base, ref_lo, ref_hi, axis_cols)
    panel_std = _standardize(panel.copy(), ref, axis_cols)

    triplets = _build_triplets(panel_std, axis_labels, axis_cols)
    print(f"[step5] {tier_name}: {len(triplets):,} triplets.")
    if len(triplets) < 200:
        return {"tier": tier_name, "status": "insufficient_triplets",
                "n": int(len(triplets))}

    n_boot = cfg["analysis"]["n_bootstrap"] // 2  # halve for speed
    main_res = lead_lag_all_pairs(
        triplets, axis_labels, j_signs, n_boot=n_boot, rng=rng
    )

    # Subgroups
    subgroups: Dict[str, dict] = {}
    for key, mask in {
        "medication_naive": triplets.get("n_med_classes", pd.Series(0)).fillna(0) == 0,
        "male": triplets.get("sex", pd.Series(np.nan)) == 1,
        "female": triplets.get("sex", pd.Series(np.nan)) == 0,
    }.items():
        sub = triplets[mask.values if isinstance(mask, pd.Series) else mask]
        if len(sub) < 500:
            continue
        subgroups[key] = lead_lag_all_pairs(
            sub, axis_labels, j_signs, n_boot=n_boot, rng=rng
        )

    return {
        "tier": tier_name,
        "axes": axis_labels,
        "n_triplets": int(len(triplets)),
        "main": main_res,
        "subgroups": subgroups,
    }


def _heatmap(result: dict, out_path: Path) -> None:
    if result.get("status"):
        return
    axes = result["axes"]
    pairs = result["main"]["pairs"]
    n = len(axes)
    beta_mat = np.full((n, n), np.nan)
    star_mat = np.full((n, n), "", dtype=object)
    for p in pairs:
        i = axes.index(p["from_axis"])
        j = axes.index(p["to_axis"])
        beta_mat[j, i] = p["beta"]
        if p.get("q_fdr") is not None and p["q_fdr"] < 0.05:
            star_mat[j, i] = "*"
    fig, ax = plt.subplots(figsize=(6 + 0.5 * n, 5 + 0.5 * n))
    vmax = np.nanmax(np.abs(beta_mat)) or 1.0
    im = ax.imshow(beta_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(axes)
    ax.set_yticklabels(axes)
    ax.set_xlabel("from axis (lagged)")
    ax.set_ylabel("to axis (ΔT)")
    for i in range(n):
        for j in range(n):
            if np.isfinite(beta_mat[i, j]):
                ax.text(j, i, f"{beta_mat[i, j]:.3f}{star_mat[i, j]}",
                        ha="center", va="center", fontsize=8,
                        color="black" if abs(beta_mat[i, j]) < 0.5 * vmax else "white")
    plt.colorbar(im, ax=ax, label="β (cross-lagged)")
    concordance = result["main"]["concordance_rate"]
    p_bin = result["main"]["binomial_p"]
    ax.set_title(
        f"Lead-lag — {result['tier']} "
        f"(concordance = {concordance:.2f}, binomial p = {p_bin:.3g}; "
        f"* = FDR<0.05)"
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
        print(f"[step5] Running {t}…")
        res = run_tier(panel, cfg, t, j_signs, rng)
        res["_header"] = header_comment("step5_lead_lag.py", config_hash(cfg))
        save_json(out_dir / f"ukb_lead_lag_{t}.json", res)
        _heatmap(res, out_dir / f"figure_ukb_lead_lag_{t}.pdf")
        print(f"[step5] Wrote ukb_lead_lag_{t}.json + figure.")


if __name__ == "__main__":
    main()
