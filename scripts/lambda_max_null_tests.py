#!/usr/bin/env python3
"""Permutation / random-panel null tests for the age-stratified λ_max trajectory.

Contextualises the reported 19-fold InCHIANTI increase in λ_max(Γ̂_change)
from ages 20–49 to 80+ and the ~1.2-fold ELSA increase.

Null Test 1 — InCHIANTI age permutation (1000 reps):
    shuffle age labels across visit-pair change vectors; recompute
    age-stratified λ_max and its max/min ratio.

Null Test 2 — InCHIANTI random biomarker panels (500 reps):
    replace the 4 HDR axes with 4 randomly drawn continuous biomarkers from
    the InCHIANTI panel (CRP, fibrinogen, HDL, LDL, triglycerides, cystatin C,
    albumin, grip, gait_speed) and measure the max/min ratio.

Null Test 3 — InCHIANTI univariate variance control:
    compare the multivariate λ_max to the max individual biomarker variance
    in each stratum. Excess above 1.0 is attributable to cross-axis
    covariance (coupling).

Null Test 4 — ELSA age permutation (1000 reps):
    same permutation procedure on ELSA 3-axis (dx_I, dx_M, dx_F) visit-pair
    change vectors, with the ELSA age strata (50-59, 60-69, 70-79, 80+).

Outputs
    results/lambda_max_null_tests.json
    outputs/figure_lambda_max_null.pdf (and .png)

Usage
    python scripts/lambda_max_null_tests.py
    python scripts/lambda_max_null_tests.py --n-perm 200 --skip-elsa  # quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hdr_sim.inchianti import (
    load_inchianti_panel, compute_youthful_reference, standardize_axes,
)
from hdr_sim.plotting import setup_style, add_panel_label, save_figure

RNG = np.random.default_rng(42)

INCHIANTI_AXES = ["delta_I", "delta_M", "delta_N", "delta_F"]
INCHIANTI_STRATA = [
    ("20-49", 20, 49),
    ("50-59", 50, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80+",   80, 120),
]

ELSA_AXES = ["dx_I", "dx_M", "dx_F"]
ELSA_STRATA = [
    ("50-59", 50, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80+",   80, 120),
]

# Candidate pool for Null Test 2 (InCHIANTI columns set in
# src/hdr_sim/inchianti.py:333-359). `grip` and `gait_speed` are retained —
# they are separate columns from the SPPB composite used as the F axis.
INCHIANTI_RANDOM_CANDIDATES = [
    "crp_hs", "fibrinogen", "hdl", "ldl", "triglycerides",
    "cystatin_c", "albumin_pct", "grip", "gait_speed",
]

OUTPUT_JSON = os.path.join(ROOT, "results", "lambda_max_null_tests.json")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def lambda_max_of_cov(X: np.ndarray) -> float:
    """Largest eigenvalue of the sample covariance of X (n × p)."""
    if X.shape[0] < 3:
        return np.nan
    C = np.cov(X, rowvar=False)
    if np.any(~np.isfinite(C)):
        return np.nan
    return float(np.max(eigvalsh(C)))


def stratum_lambda_max(X: np.ndarray, age_mid: np.ndarray,
                       strata) -> dict[str, float]:
    """λ_max of the sample covariance of X within each (label, lo, hi) stratum."""
    out = {}
    for label, lo, hi in strata:
        mask = (age_mid >= lo) & (age_mid <= hi)
        if mask.sum() < 3:
            out[label] = np.nan
            continue
        out[label] = lambda_max_of_cov(X[mask])
    return out


def max_min_ratio(traj: dict[str, float]) -> float:
    """Max / min of a stratum trajectory, ignoring NaN."""
    vals = np.array([v for v in traj.values() if np.isfinite(v) and v > 0])
    if len(vals) < 2:
        return np.nan
    return float(np.max(vals) / np.min(vals))


def _build_change_vectors(sub: pd.DataFrame, axes: list[str],
                          id_col: str) -> pd.DataFrame:
    """Shared visit-pair builder for both InCHIANTI (code98) and ELSA
    (idauniq). `sub` must already be sorted by [id_col, wave].

    pandas' `groupby.shift(1)` produces NaN across group boundaries, which is
    exactly the within-subject constraint we need — no need to compare
    id_col to its shifted version (the grouping column is not in the shifted
    frame).
    """
    cols_to_shift = ["wave", "age"] + list(axes)
    prev = sub.groupby(id_col)[cols_to_shift].shift(1)

    mask = sub["age"].notna() & prev["age"].notna()
    for ax in axes:
        mask &= sub[ax].notna() & prev[ax].notna()

    out = pd.DataFrame({
        id_col: sub.loc[mask, id_col].values,
        "wave_t": prev.loc[mask, "wave"].values,
        "age_mid": (sub.loc[mask, "age"].values +
                    prev.loc[mask, "age"].values) / 2.0,
    })
    for ax in axes:
        out[ax] = (sub.loc[mask, ax].values - prev.loc[mask, ax].values)
    return out


def build_inchianti_change_vectors(panel: pd.DataFrame,
                                   axes: list[str]) -> pd.DataFrame:
    """Within-person consecutive-wave change vectors, InCHIANTI (code98)."""
    need = list(axes) + ["code98", "wave", "age"]
    sub = panel[need].sort_values(["code98", "wave"]).reset_index(drop=True)
    return _build_change_vectors(sub, axes, "code98")


def build_elsa_change_vectors(merged: pd.DataFrame,
                              axes: list[str]) -> pd.DataFrame:
    """Within-person consecutive-wave change vectors, ELSA (idauniq)."""
    need = list(axes) + ["idauniq", "wave", "age"]
    sub = merged[need].sort_values(["idauniq", "wave"]).reset_index(drop=True)
    return _build_change_vectors(sub, axes, "idauniq")


# ---------------------------------------------------------------------------
# Null Test 1 / 4: age-permutation null
# ---------------------------------------------------------------------------
def age_permutation_null(change_vectors: pd.DataFrame, axes: list[str],
                         strata, n_perm: int, label: str = "",
                         progress_every: int = 200) -> dict:
    """Shuffle age_mid across change vectors; recompute stratum λ_max trajectory
    and record the max/min ratio under each permutation.
    """
    X = change_vectors[axes].values.astype(float)
    ages = change_vectors["age_mid"].values.astype(float)

    observed_traj = stratum_lambda_max(X, ages, strata)
    observed_ratio = max_min_ratio(observed_traj)

    null_ratios = np.empty(n_perm)
    null_ratios.fill(np.nan)
    t0 = time.time()
    for i in range(n_perm):
        shuffled = RNG.permutation(ages)
        traj = stratum_lambda_max(X, shuffled, strata)
        null_ratios[i] = max_min_ratio(traj)

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            valid = null_ratios[:i + 1]
            valid = valid[np.isfinite(valid)]
            pct = (valid >= observed_ratio).mean() if len(valid) else np.nan
            print(f"    [{label}] {i+1:>5d}/{n_perm} perms done "
                  f"(elapsed {elapsed:.1f}s, null median="
                  f"{np.nanmedian(valid):.2f}, "
                  f"p(null ≥ obs)={pct:.4f})")

    valid = null_ratios[np.isfinite(null_ratios)]
    pct_rank = float((valid <= observed_ratio).mean()) if len(valid) else np.nan
    # One-sided p: H0: no age structure → ratio ~ null distribution; we ask
    # how often the null exceeds the observed ratio.
    p_value = float((valid >= observed_ratio).mean()) if len(valid) else np.nan

    return {
        "observed_ratio": float(observed_ratio),
        "observed_trajectory": {k: (float(v) if np.isfinite(v) else None)
                                for k, v in observed_traj.items()},
        "null_mean": float(np.nanmean(valid)) if len(valid) else None,
        "null_sd": float(np.nanstd(valid, ddof=1)) if len(valid) > 1 else None,
        "null_median": float(np.nanmedian(valid)) if len(valid) else None,
        "null_p025": float(np.nanpercentile(valid, 2.5)) if len(valid) else None,
        "null_p975": float(np.nanpercentile(valid, 97.5)) if len(valid) else None,
        "null_percentile_rank": pct_rank,
        "p_value": p_value,
        "n_permutations": int(n_perm),
        "n_valid": int(len(valid)),
        "n_change_vectors": int(len(change_vectors)),
        "null_ratios": valid.tolist(),
    }


# ---------------------------------------------------------------------------
# Null Test 2: random biomarker panels (InCHIANTI)
# ---------------------------------------------------------------------------
def random_panel_null(panel: pd.DataFrame, hdr_ratio: float,
                      candidate_pool: list[str], strata,
                      n_panels: int = 500, panel_size: int = 4,
                      progress_every: int = 50) -> dict:
    """Draw random 4-biomarker panels from non-HDR candidates and compute the
    age-stratum λ_max ratio for each.

    Biomarkers are z-scored panel-wide (panel mean, panel SD) before the
    visit-pair change is taken, so every candidate lives on a comparable
    scale — this matches the HDR axes' standardisation to a youthful
    reference up to a scalar.
    """
    avail = [c for c in candidate_pool if c in panel.columns]
    print(f"  Candidates in panel: {avail}")

    # Require ≥2 waves with ≥100 non-missing per wave (per task spec)
    usable = []
    for c in avail:
        waves_ok = 0
        for w, grp in panel.groupby("wave"):
            if grp[c].notna().sum() >= 100:
                waves_ok += 1
        if waves_ok >= 2:
            usable.append(c)
        else:
            print(f"    {c}: only {waves_ok} wave(s) with ≥100 non-missing — dropped")

    # Build a z-scored working copy so each candidate is dimensionless and
    # comparable in magnitude to the HDR deltas (which are z-scored to a
    # youthful reference).
    working = panel[["code98", "wave", "age"] + usable].copy()
    for c in usable:
        mu = working[c].mean()
        sd = working[c].std()
        if sd is None or not np.isfinite(sd) or sd < 1e-9:
            working[c] = np.nan
        else:
            working[c] = (working[c] - mu) / sd

    effective_size = panel_size if len(usable) >= panel_size else len(usable)
    if effective_size < panel_size:
        print(f"  WARNING: only {len(usable)} usable candidates — falling back "
              f"to {effective_size}-biomarker panels.")

    # Enumerate all combinations if the space is smaller than n_panels — no
    # point resampling with replacement from a tiny set.
    from itertools import combinations
    all_combos = list(combinations(usable, effective_size))
    if len(all_combos) == 0:
        raise RuntimeError(f"No candidate biomarkers available for null panel "
                           f"(usable={usable}, effective_size={effective_size})")

    if len(all_combos) <= n_panels:
        combos = all_combos
        print(f"  Enumerating all {len(all_combos)} combinations (≤ n_panels).")
    else:
        idx = RNG.choice(len(all_combos), size=n_panels, replace=False)
        combos = [all_combos[i] for i in idx]

    ratios = []
    trajs = []
    selected = []
    t0 = time.time()
    for i, combo in enumerate(combos):
        axes = list(combo)
        cv = build_inchianti_change_vectors(working, axes)
        if len(cv) < 100:
            ratios.append(np.nan)
            trajs.append(None)
            selected.append(axes)
            continue
        X = cv[axes].values
        ages = cv["age_mid"].values
        traj = stratum_lambda_max(X, ages, strata)
        ratios.append(max_min_ratio(traj))
        trajs.append({k: (float(v) if np.isfinite(v) else None)
                      for k, v in traj.items()})
        selected.append(axes)

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            finite = [r for r in ratios if np.isfinite(r)]
            print(f"    [random-panel] {i+1:>4d}/{len(combos)} panels done "
                  f"(elapsed {elapsed:.1f}s, null median="
                  f"{np.median(finite) if finite else float('nan'):.2f})")

    arr = np.array(ratios, dtype=float)
    valid = arr[np.isfinite(arr)]
    pct_rank = float((valid <= hdr_ratio).mean()) if len(valid) else np.nan
    p_value = float((valid >= hdr_ratio).mean()) if len(valid) else np.nan

    return {
        "hdr_observed_ratio": float(hdr_ratio),
        "null_mean": float(valid.mean()) if len(valid) else None,
        "null_sd": float(valid.std(ddof=1)) if len(valid) > 1 else None,
        "null_median": float(np.median(valid)) if len(valid) else None,
        "null_p025": float(np.percentile(valid, 2.5)) if len(valid) else None,
        "null_p975": float(np.percentile(valid, 97.5)) if len(valid) else None,
        "null_percentile_rank": pct_rank,
        "p_value": p_value,
        "n_panels_run": int(len(combos)),
        "n_valid": int(len(valid)),
        "n_candidates": int(len(usable)),
        "effective_panel_size": int(effective_size),
        "fallback_used": bool(effective_size < panel_size),
        "usable_candidates": usable,
        "null_ratios": valid.tolist(),
        "panels": [{"axes": axes, "ratio": (float(r) if np.isfinite(r) else None)}
                   for axes, r in zip(selected, ratios)],
    }


# ---------------------------------------------------------------------------
# Null Test 3: univariate variance control (InCHIANTI)
# ---------------------------------------------------------------------------
def univariate_variance_control(change_vectors: pd.DataFrame,
                                axes: list[str], strata) -> dict:
    """Per-stratum multivariate λ_max vs max individual-axis variance.

    Ratio ≥ 1 by the Rayleigh bound; any excess above 1 is cross-axis
    covariance (coupling structure).
    """
    by_stratum = {}
    for label, lo, hi in strata:
        mask = ((change_vectors["age_mid"] >= lo) &
                (change_vectors["age_mid"] <= hi))
        X = change_vectors.loc[mask, axes].values
        n = len(X)
        if n < 3:
            by_stratum[label] = {
                "lambda_max": None,
                "max_univariate_var": None,
                "ratio": None,
                "per_axis_var": {ax: None for ax in axes},
                "n": n,
            }
            continue
        lmax = lambda_max_of_cov(X)
        var_per = np.var(X, axis=0, ddof=1)
        max_uv = float(np.max(var_per))
        by_stratum[label] = {
            "lambda_max": float(lmax),
            "max_univariate_var": max_uv,
            "ratio": float(lmax / max_uv) if max_uv > 0 else None,
            "per_axis_var": {ax: float(v) for ax, v in zip(axes, var_per)},
            "n": int(n),
        }
    return {"by_stratum": by_stratum}


# ---------------------------------------------------------------------------
# ELSA loader (via importlib, per repo convention)
# ---------------------------------------------------------------------------
def load_elsa_change_vectors() -> pd.DataFrame:
    print("\n--- ELSA: loading 3-axis change vectors ---")
    spec = importlib.util.spec_from_file_location(
        "run_elsa_validation",
        os.path.join(ROOT, "scripts", "run_elsa_validation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    files = mod.load_all_files()
    panel, _hba1c_units = mod.extract_nurse_biomarkers(files)
    harm = mod.prepare_harmonised(files)
    mort = mod.extract_mortality(files, harm)
    supp = mod.extract_supplementary(files)
    harm_long = mod.harmonised_to_long(harm)
    merged = mod.build_analysis_panel(panel, harm_long, mort, supp)

    in_long = merged["in_longitudinal"] & merged["complete_3axis"]
    cv = build_elsa_change_vectors(merged.loc[in_long], ELSA_AXES)
    print(f"  ELSA 3-axis visit-pairs: {len(cv):,}")
    return cv


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(results: dict, out_name: str = "figure_lambda_max_null"):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) InCHIANTI age permutation
    ax = axes[0, 0]
    r1 = results["inchianti_age_permutation"]
    nulls = np.array(r1["null_ratios"])
    obs = r1["observed_ratio"]
    ax.hist(nulls, bins=40, color="#bfbfbf", edgecolor="white")
    ax.axvline(obs, color="#d62728", linewidth=2,
               label=f"observed = {obs:.1f}×")
    ax.axvline(float(np.median(nulls)), color="#2b8cbe", linewidth=1.2,
               linestyle="--", label=f"null median = {np.median(nulls):.2f}×")
    ax.set_xlabel("max/min λ_max ratio")
    ax.set_ylabel("permutations")
    ax.set_title(f"InCHIANTI age-permutation null\n"
                 f"(n={r1['n_permutations']}, p={r1['p_value']:.4f})")
    ax.legend(frameon=False, loc="best")
    add_panel_label(ax, "a")

    # (b) InCHIANTI random panel
    ax = axes[0, 1]
    r2 = results["inchianti_random_panel"]
    nulls = np.array(r2["null_ratios"])
    obs = r2["hdr_observed_ratio"]
    ax.hist(nulls, bins=min(30, max(6, len(nulls) // 2)),
            color="#bfbfbf", edgecolor="white")
    ax.axvline(obs, color="#d62728", linewidth=2,
               label=f"HDR = {obs:.1f}×")
    ax.axvline(float(np.median(nulls)), color="#2b8cbe", linewidth=1.2,
               linestyle="--", label=f"null median = {np.median(nulls):.2f}×")
    ax.set_xlabel("max/min λ_max ratio")
    ax.set_ylabel("random panels")
    fb = " (3-biomarker fallback)" if r2["fallback_used"] else ""
    ax.set_title(f"InCHIANTI random biomarker panels{fb}\n"
                 f"(n_panels={r2['n_panels_run']}, "
                 f"p={r2['p_value']:.4f})")
    ax.legend(frameon=False, loc="best")
    add_panel_label(ax, "b")

    # (c) Univariate variance control
    ax = axes[1, 0]
    r3 = results["inchianti_univariate_control"]["by_stratum"]
    strata_labels = [s for s, _, _ in INCHIANTI_STRATA]
    lmax_vals = [r3[s]["lambda_max"] for s in strata_labels]
    univ_vals = [r3[s]["max_univariate_var"] for s in strata_labels]
    xs = range(len(strata_labels))
    ax.plot(xs, lmax_vals, "o-", color="#d62728", linewidth=2,
            label="multivariate λ_max")
    ax.plot(xs, univ_vals, "s--", color="#2b8cbe", linewidth=1.5,
            label="max individual variance")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(strata_labels)
    ax.set_xlabel("age stratum")
    ax.set_ylabel("variance / eigenvalue")
    ax.set_title("InCHIANTI univariate variance control\n"
                 "(gap = coupling contribution to λ_max)")
    ax.legend(frameon=False, loc="best")
    add_panel_label(ax, "c")

    # (d) ELSA age permutation
    ax = axes[1, 1]
    r4 = results.get("elsa_age_permutation")
    if r4 is not None and r4.get("null_ratios"):
        nulls = np.array(r4["null_ratios"])
        obs = r4["observed_ratio"]
        ax.hist(nulls, bins=40, color="#bfbfbf", edgecolor="white")
        ax.axvline(obs, color="#d62728", linewidth=2,
                   label=f"observed = {obs:.2f}×")
        ax.axvline(float(np.median(nulls)), color="#2b8cbe",
                   linewidth=1.2, linestyle="--",
                   label=f"null median = {np.median(nulls):.2f}×")
        ax.set_xlabel("max/min λ_max ratio")
        ax.set_ylabel("permutations")
        ax.set_title(f"ELSA age-permutation null\n"
                     f"(n={r4['n_permutations']}, p={r4['p_value']:.4f})")
        ax.legend(frameon=False, loc="best")
    else:
        ax.text(0.5, 0.5, "ELSA panel skipped\n(--skip-elsa)", ha="center",
                va="center", transform=ax.transAxes, fontsize=11,
                color="#666666")
        ax.set_xticks([]); ax.set_yticks([])
    add_panel_label(ax, "d")

    fig.tight_layout()
    save_figure(fig, out_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Permutations for age-permutation tests (1 & 4)")
    parser.add_argument("--n-panels", type=int, default=500,
                        help="Random-panel draws for test 2")
    parser.add_argument("--skip-inchianti", action="store_true")
    parser.add_argument("--skip-elsa", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"λ_max null tests  (n_perm={args.n_perm}, n_panels={args.n_panels})")
    print("=" * 70)

    results = {
        "_description": (
            "Permutation and random-panel null tests for the age-stratified "
            "λ_max(Γ̂_change) trajectory. Contextualises the manuscript's "
            "reported ~19-fold InCHIANTI increase and ~1.2-fold ELSA increase "
            "against nulls that control for marginal biomarker distributions, "
            "inter-biomarker correlation, and biomarker-panel selection."
        ),
        "method": {
            "seed": 42,
            "n_permutations": args.n_perm,
            "n_random_panels": args.n_panels,
            "inchianti_axes": INCHIANTI_AXES,
            "inchianti_strata": [{"label": s, "lo": lo, "hi": hi}
                                 for s, lo, hi in INCHIANTI_STRATA],
            "elsa_axes": ELSA_AXES,
            "elsa_strata": [{"label": s, "lo": lo, "hi": hi}
                            for s, lo, hi in ELSA_STRATA],
        },
    }

    t0 = time.time()

    # ---------------- InCHIANTI ----------------
    if not args.skip_inchianti:
        print("\n--- InCHIANTI: loading panel + standardising axes ---")
        panel = load_inchianti_panel()
        ref = compute_youthful_reference(panel)
        panel_std = standardize_axes(panel, ref)
        cv = build_inchianti_change_vectors(panel_std, INCHIANTI_AXES)
        print(f"  InCHIANTI 4-axis visit-pairs: {len(cv):,}")

        # Null Test 1
        print("\n--- Null Test 1: InCHIANTI age permutation ---")
        results["inchianti_age_permutation"] = age_permutation_null(
            cv, INCHIANTI_AXES, INCHIANTI_STRATA,
            n_perm=args.n_perm, label="inchianti",
        )
        r1 = results["inchianti_age_permutation"]
        print(f"  observed ratio = {r1['observed_ratio']:.2f}")
        print(f"  null median    = {r1['null_median']:.2f}")
        print(f"  null 95% CI    = [{r1['null_p025']:.2f}, "
              f"{r1['null_p975']:.2f}]")
        print(f"  p(null ≥ obs)  = {r1['p_value']:.4f}")

        hdr_ratio = r1["observed_ratio"]

        # Null Test 3 (cheap — deterministic)
        print("\n--- Null Test 3: univariate variance control ---")
        results["inchianti_univariate_control"] = univariate_variance_control(
            cv, INCHIANTI_AXES, INCHIANTI_STRATA,
        )
        for s, _, _ in INCHIANTI_STRATA:
            row = results["inchianti_univariate_control"]["by_stratum"][s]
            if row["ratio"] is not None:
                print(f"  {s}: λ_max={row['lambda_max']:.4f}, "
                      f"max univ. var={row['max_univariate_var']:.4f}, "
                      f"ratio={row['ratio']:.3f} (N={row['n']})")

        # Null Test 2 (random panels — rebuilds change vectors per draw)
        print("\n--- Null Test 2: InCHIANTI random biomarker panels ---")
        results["inchianti_random_panel"] = random_panel_null(
            panel, hdr_ratio=hdr_ratio,
            candidate_pool=INCHIANTI_RANDOM_CANDIDATES,
            strata=INCHIANTI_STRATA,
            n_panels=args.n_panels,
        )
        r2 = results["inchianti_random_panel"]
        print(f"  HDR ratio     = {r2['hdr_observed_ratio']:.2f}")
        print(f"  random median = "
              f"{r2['null_median']:.2f} "
              f"[{r2['null_p025']:.2f}, {r2['null_p975']:.2f}]")
        print(f"  p(null ≥ HDR) = {r2['p_value']:.4f}")
        print(f"  candidates    = {r2['usable_candidates']}")

    # ---------------- ELSA ----------------
    if not args.skip_elsa:
        try:
            elsa_cv = load_elsa_change_vectors()
        except Exception as e:
            print(f"  ELSA load failed: {e!r} — skipping Null Test 4.")
            elsa_cv = None

        if elsa_cv is not None and len(elsa_cv) > 0:
            print("\n--- Null Test 4: ELSA age permutation ---")
            results["elsa_age_permutation"] = age_permutation_null(
                elsa_cv, ELSA_AXES, ELSA_STRATA,
                n_perm=args.n_perm, label="elsa",
            )
            r4 = results["elsa_age_permutation"]
            print(f"  observed ratio = {r4['observed_ratio']:.2f}")
            print(f"  null median    = {r4['null_median']:.2f}")
            print(f"  null 95% CI    = [{r4['null_p025']:.2f}, "
                  f"{r4['null_p975']:.2f}]")
            print(f"  p(null ≥ obs)  = {r4['p_value']:.4f}")

    elapsed = time.time() - t0

    # ---------------- Persist ----------------
    # For the JSON, drop the bulky null_ratios lists — the figure has already
    # been generated and reviewers need summary stats, not the full samples.
    # But we keep them temporarily so make_figure can use them, then strip
    # before writing.
    make_figure(results)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    compact = _compact_results(results)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    print(f"\nSaved -> {os.path.relpath(OUTPUT_JSON, ROOT)}  "
          f"(total {elapsed/60:.1f} min)")

    # ---------------- Summary ----------------
    print("\n" + "=" * 70)
    print("Null-test summary")
    print("=" * 70)
    if "inchianti_age_permutation" in results:
        r = results["inchianti_age_permutation"]
        print(f"  InCHIANTI age permutation    obs {r['observed_ratio']:6.2f}× "
              f"vs null {r['null_median']:5.2f} "
              f"[{r['null_p025']:5.2f}, {r['null_p975']:5.2f}]  "
              f"p={r['p_value']:.4f}")
    if "inchianti_random_panel" in results:
        r = results["inchianti_random_panel"]
        print(f"  InCHIANTI random panel       HDR {r['hdr_observed_ratio']:6.2f}× "
              f"vs null {r['null_median']:5.2f} "
              f"[{r['null_p025']:5.2f}, {r['null_p975']:5.2f}]  "
              f"p={r['p_value']:.4f}  "
              f"(n_candidates={r['n_candidates']})")
    if "inchianti_univariate_control" in results:
        bys = results["inchianti_univariate_control"]["by_stratum"]
        first = INCHIANTI_STRATA[0][0]
        last = INCHIANTI_STRATA[-1][0]
        if (bys[first]["ratio"] is not None and bys[last]["ratio"] is not None):
            print(f"  InCHIANTI univariate control {first}: "
                  f"λmax/maxUV = {bys[first]['ratio']:.3f};  "
                  f"{last}: {bys[last]['ratio']:.3f}")
    if "elsa_age_permutation" in results:
        r = results["elsa_age_permutation"]
        print(f"  ELSA age permutation         obs {r['observed_ratio']:6.2f}× "
              f"vs null {r['null_median']:5.2f} "
              f"[{r['null_p025']:5.2f}, {r['null_p975']:5.2f}]  "
              f"p={r['p_value']:.4f}")


def _compact_results(results: dict) -> dict:
    """Strip the bulky null_ratios arrays from a copy of `results` for JSON."""
    out = {}
    for k, v in results.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if kk != "null_ratios"}
            # Also trim the per-panel detail to axes + ratio (drop nothing
            # — panels list is small, <=500 items — but format is already
            # minimal).
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
