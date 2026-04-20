"""
HDR Core — self-contained bundle of analysis functions.

Drop-in module for the UK Biobank pipeline. No imports from hdr_sim, so it
can be used standalone. Functions are copied / adapted from:
    src/hdr_sim/estimation.py
    scripts/inchianti_lambda_max_trajectory.py
    scripts/inchianti_lead_lag.py
    scripts/inchianti_survival.py
    scripts/lambda_max_null_tests.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.stats import binomtest, kendalltau
from statsmodels.stats.multitest import multipletests


AXIS_ORDER_9 = ["I", "M", "E", "mito", "P", "C", "N", "F", "B"]

# Hardcoded fallback signs — load the actual CSV for authoritative values.
J_SIGNS_FALLBACK: Dict[Tuple[str, str], Optional[int]] = {
    ("I", "M"): +1, ("M", "I"): +1,
    ("I", "N"): +1, ("N", "I"): -1,
    ("I", "F"): +1, ("F", "I"): -1,
    ("I", "C"): +1, ("C", "I"): +1,
    ("I", "B"): +1, ("B", "I"): -1,
    ("I", "P"): +1, ("P", "I"): +1,
    ("M", "N"): +1, ("N", "M"): +1,
    ("M", "F"): +1, ("F", "M"): -1,
    ("M", "C"): +1, ("C", "M"): +1,
    ("M", "B"): +1, ("B", "M"): -1,
    ("M", "P"): +1, ("P", "M"): +1,
    ("N", "F"): -1, ("F", "N"): -1,
    ("N", "C"): +1, ("C", "N"): +1,
    ("N", "B"): +1, ("B", "N"): None,
    ("F", "C"): -1, ("C", "F"): -1,
    ("F", "B"): -1, ("B", "F"): +1,
    ("F", "P"): -1, ("P", "F"): -1,
    ("C", "B"): None, ("B", "C"): None,
    ("C", "P"): +1, ("P", "C"): +1,
    ("B", "P"): +1, ("P", "B"): None,
}


# ─────────────────────────────────────────────────────────────────────────────
# J-matrix loading
# ─────────────────────────────────────────────────────────────────────────────

def load_j_signs(csv_path: str | os.PathLike) -> Dict[Tuple[str, str], Optional[int]]:
    """
    Load compiled J-matrix sign predictions from CSV.

    The CSV has columns: axis_from, axis_to, sign, magnitude_tier,
    confidence_grade, ... Returns a dict mapping (from_axis, to_axis) -> sign
    (+1, -1, or None for unknown/qual_only).
    """
    p = Path(csv_path)
    if not p.exists():
        print(f"[hdr_core] J matrix not found at {p}; using hardcoded fallback.")
        return dict(J_SIGNS_FALLBACK)

    df = pd.read_csv(p)
    signs: Dict[Tuple[str, str], Optional[int]] = {}
    for _, row in df.iterrows():
        f = str(row["axis_from"]).strip()
        t = str(row["axis_to"]).strip()
        s = str(row["sign"]).strip()
        if s == "+":
            signs[(f, t)] = +1
        elif s == "-":
            signs[(f, t)] = -1
        else:
            signs[(f, t)] = None
    return signs


# ─────────────────────────────────────────────────────────────────────────────
# SWDS-Γ and λ_max utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_swds_gamma(delta_x: np.ndarray, Gamma_hat: np.ndarray) -> float:
    """
    Γ-native stability-weighted dysregulation score.
    SWDS-Γ(Δx) = Δxᵀ Γ̂ Δx / tr(Γ̂)
    """
    quad = float(delta_x @ Gamma_hat @ delta_x)
    tr = float(np.trace(Gamma_hat))
    if tr <= 0 or not np.isfinite(tr):
        return float("nan")
    return quad / tr


def compute_swds_gamma_batch(X: np.ndarray, Gamma_hat: np.ndarray) -> np.ndarray:
    """Vectorised SWDS-Γ for an (N × n) batch."""
    tr = float(np.trace(Gamma_hat))
    if tr <= 0 or not np.isfinite(tr):
        return np.full(X.shape[0], np.nan)
    return np.sum((X @ Gamma_hat) * X, axis=1) / tr


def gamma_stability_proxy(Gamma_hat: np.ndarray) -> Dict[str, Any]:
    """Eigenvalue decomposition of Γ̂ with stability-relevant summaries."""
    eigenvalues, eigenvectors = np.linalg.eigh(Gamma_hat)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    lam_max = float(eigenvalues[0])
    lam_min = float(eigenvalues[-1])
    return {
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "kappa": float(lam_max / max(lam_min, 1e-15)),
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "trace": float(np.sum(eigenvalues)),
    }


def lambda_max_of_cov(X: np.ndarray) -> float:
    """Largest eigenvalue of the sample covariance of X (N × p)."""
    if X is None or len(X) < 3:
        return float("nan")
    X = np.asarray(X, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 3:
        return float("nan")
    C = np.cov(X, rowvar=False)
    if C.ndim == 0:
        return float(C)
    return float(np.max(eigvalsh(C)))


def bootstrap_lambda_max(
    X: np.ndarray,
    n_boot: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Tuple[float, float]:
    """Bootstrap CI for λ_max of the sample covariance."""
    if rng is None:
        rng = np.random.default_rng(42)
    X = np.asarray(X, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    n = len(X)
    if n < 5:
        return float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = lambda_max_of_cov(X[idx])
    return float(np.nanpercentile(boots, ci[0])), float(np.nanpercentile(boots, ci[1]))


def max_axis_variance(X: np.ndarray) -> float:
    """Largest per-axis variance — used for the univariate control ratio."""
    X = np.asarray(X, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 3:
        return float("nan")
    return float(np.max(np.var(X, axis=0, ddof=1)))


# ─────────────────────────────────────────────────────────────────────────────
# Π decomposition
# ─────────────────────────────────────────────────────────────────────────────

def compute_pi_decomposition(Gamma_hat: np.ndarray) -> Dict[str, float]:
    """
    Π = C_norm / V_norm decomposition.

    V_norm = tr(Γ̂) / n (average axis variance)
    C_norm = mean |off-diagonal Γ̂| (average absolute coupling magnitude)
    Π      = C_norm / V_norm

    Π < ~0.5 → D-dominated regime (variance drives Γ)
    Π > ~1.0 → J-dominated regime (coupling drives Γ)
    """
    n = Gamma_hat.shape[0]
    diag_var = np.diag(Gamma_hat)
    V_norm = float(np.mean(diag_var))
    mask = ~np.eye(n, dtype=bool)
    C_norm = float(np.mean(np.abs(Gamma_hat[mask])))
    pi_val = C_norm / V_norm if V_norm > 0 else float("nan")
    return {"V_norm": V_norm, "C_norm": C_norm, "Pi": pi_val}


def compute_pi_by_stratum(
    Gamma_by_stratum: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Apply Π decomposition across a set of per-stratum covariance matrices."""
    rows = []
    for name, Gamma in Gamma_by_stratum.items():
        d = compute_pi_decomposition(Gamma)
        d["stratum"] = name
        rows.append(d)
    return pd.DataFrame(rows)[["stratum", "V_norm", "C_norm", "Pi"]]


# ─────────────────────────────────────────────────────────────────────────────
# Sign concordance
# ─────────────────────────────────────────────────────────────────────────────

def covariance_sign_concordance(
    Gamma_hat: np.ndarray,
    axes: Sequence[str],
    j_signs: Dict[Tuple[str, str], Optional[int]],
) -> Dict[str, Any]:
    """
    Sign concordance between off-diagonal Γ̂ entries and compiled J-matrix
    predictions. For same-sign bidirectional pairs the prediction is
    unambiguous; mixed-sign and unknown pairs are flagged as excluded.
    """
    n = len(axes)
    n_agree = 0
    n_total = 0
    n_excluded = 0
    pair_details: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ai, aj = axes[i], axes[j]
            s_ij = j_signs.get((ai, aj))
            s_ji = j_signs.get((aj, ai))
            if s_ij is None and s_ji is None:
                continue
            if s_ij is None or s_ji is None or s_ij != s_ji:
                n_excluded += 1
                pair_details.append({"pair": f"{ai}-{aj}", "status": "ambiguous"})
                continue
            predicted = s_ij
            observed = int(np.sign(Gamma_hat[i, j]))
            match = (observed == predicted)
            n_agree += int(match)
            n_total += 1
            pair_details.append({
                "pair": f"{ai}-{aj}",
                "predicted": int(predicted),
                "observed": observed,
                "match": bool(match),
            })
    concordance = n_agree / n_total if n_total > 0 else float("nan")
    p_binom = (
        float(binomtest(n_agree, n_total, 0.5, alternative="greater").pvalue)
        if n_total > 0
        else float("nan")
    )
    return {
        "concordance": concordance,
        "n_agree": n_agree,
        "n_total": n_total,
        "n_excluded": n_excluded,
        "binomial_p": p_binom,
        "pair_details": pair_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-lagged regression
# ─────────────────────────────────────────────────────────────────────────────

def cross_lagged_regression(
    triplets: pd.DataFrame,
    from_ax: str,
    to_ax: str,
    age_col: str = "age_t",
    n_boot: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    OLS cross-lagged regression with subject-clustered bootstrap CI.

    Model: d_<to_ax> ~ β·<from_ax>_t0 + γ·<to_ax>_t0 + δ·age_t + c

    Expects `triplets` to have columns:
        <from_ax>_t0, <to_ax>_t0, d_<to_ax>, <age_col>, subject_id
    """
    if rng is None:
        rng = np.random.default_rng(42)

    y = triplets[f"d_{to_ax}"].to_numpy(dtype=float)
    X_from = triplets[f"{from_ax}_t0"].to_numpy(dtype=float)
    X_auto = triplets[f"{to_ax}_t0"].to_numpy(dtype=float)
    X_age = triplets[age_col].to_numpy(dtype=float)

    valid = (
        np.isfinite(y) & np.isfinite(X_from) & np.isfinite(X_auto) & np.isfinite(X_age)
    )
    y, X_from, X_auto, X_age = y[valid], X_from[valid], X_auto[valid], X_age[valid]
    subjects = (
        triplets.loc[valid, "subject_id"].to_numpy()
        if "subject_id" in triplets.columns
        else np.arange(len(y))
    )
    n = len(y)
    if n < 10:
        return {
            "beta": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "n": int(n),
        }

    X = np.column_stack([np.ones(n), X_from, X_auto, X_age])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta_cross = float(beta_hat[1])

    # OLS p-value (not clustered — bootstrap CI is the conservative inference).
    y_hat = X @ beta_hat
    resid = y - y_hat
    sigma2 = float(np.sum(resid**2) / max(n - 4, 1))
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
        from scipy.stats import t as t_dist

        t_stat = beta_cross / se
        p_val = float(2 * t_dist.sf(abs(t_stat), df=n - 4))
    except np.linalg.LinAlgError:
        p_val = float("nan")

    # Subject-clustered bootstrap
    unique_subjects = np.unique(subjects)
    n_subj = len(unique_subjects)
    boots = np.empty(n_boot)
    subj_to_rows: Dict[Any, np.ndarray] = {}
    for s in unique_subjects:
        subj_to_rows[s] = np.flatnonzero(subjects == s)
    for b in range(n_boot):
        sidx = rng.integers(0, n_subj, size=n_subj)
        rows = np.concatenate([subj_to_rows[unique_subjects[i]] for i in sidx])
        try:
            b_hat, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
            boots[b] = b_hat[1]
        except np.linalg.LinAlgError:
            boots[b] = np.nan
    ci_lo = float(np.nanpercentile(boots, 2.5))
    ci_hi = float(np.nanpercentile(boots, 97.5))

    return {
        "beta": beta_cross,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "p_value": p_val,
        "n": int(n),
    }


def lead_lag_all_pairs(
    triplets: pd.DataFrame,
    axes: Sequence[str],
    j_signs: Dict[Tuple[str, str], Optional[int]],
    n_boot: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Run cross-lagged regression for every ordered pair; return concordance + FDR."""
    if rng is None:
        rng = np.random.default_rng(42)

    results: List[Dict[str, Any]] = []
    for from_ax in axes:
        for to_ax in axes:
            if from_ax == to_ax:
                continue
            pred = j_signs.get((from_ax, to_ax))
            fit = cross_lagged_regression(
                triplets, from_ax, to_ax, n_boot=n_boot, rng=rng
            )
            beta = fit["beta"]
            obs_sign = (
                int(np.sign(beta)) if np.isfinite(beta) and beta != 0 else 0
            )
            concordant = (pred is not None and obs_sign == pred)
            results.append({
                "from_axis": from_ax,
                "to_axis": to_ax,
                "pair": f"{from_ax}->{to_ax}",
                "predicted_sign": pred,
                "observed_sign": obs_sign,
                "concordant": bool(concordant),
                **fit,
            })

    # BH-FDR
    p_vals = np.array([r["p_value"] for r in results], dtype=float)
    p_mask = np.isfinite(p_vals)
    q_vals = np.full_like(p_vals, np.nan)
    if p_mask.sum() > 0:
        _, q_adj, _, _ = multipletests(p_vals[p_mask], method="fdr_bh")
        q_vals[p_mask] = q_adj
    for r, q in zip(results, q_vals):
        r["q_fdr"] = float(q) if np.isfinite(q) else None

    tested = [r for r in results if r["predicted_sign"] is not None]
    n_concordant = sum(1 for r in tested if r["concordant"])
    n_total = len(tested)
    binom_p = (
        float(binomtest(n_concordant, n_total, 0.5, alternative="greater").pvalue)
        if n_total > 0
        else float("nan")
    )
    return {
        "n_pairs": len(results),
        "n_tested_vs_J": n_total,
        "n_concordant": n_concordant,
        "concordance_rate": n_concordant / n_total if n_total > 0 else float("nan"),
        "binomial_p": binom_p,
        "pairs": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Null tests
# ─────────────────────────────────────────────────────────────────────────────

def permutation_trend_test(
    panel: pd.DataFrame,
    axis_cols: Sequence[str],
    age_col: str,
    strata: Sequence[Tuple[float, float]],
    n_perm: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Age-permutation null for λ_max age trend.

    H0: age label carries no information about λ_max of the axis-covariance.
    Statistic: max/min ratio of λ_max across age strata.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    mat = panel[list(axis_cols)].to_numpy(dtype=float)
    ages = panel[age_col].to_numpy(dtype=float)
    ok = np.all(np.isfinite(mat), axis=1) & np.isfinite(ages)
    mat = mat[ok]
    ages = ages[ok]

    def _ratio(a: np.ndarray) -> float:
        vals = []
        for lo, hi in strata:
            m = (a >= lo) & (a <= hi)
            if m.sum() < 5:
                continue
            vals.append(lambda_max_of_cov(mat[m]))
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        if len(vals) < 2:
            return float("nan")
        return max(vals) / min(vals)

    observed = _ratio(ages)
    if not np.isfinite(observed):
        return {"observed_ratio": float("nan"), "p_value": float("nan"), "n_perm": 0}

    null_vals = np.empty(n_perm)
    ages_shuf = ages.copy()
    for i in range(n_perm):
        rng.shuffle(ages_shuf)
        null_vals[i] = _ratio(ages_shuf)
    exceed = int(np.sum(null_vals >= observed))
    p = (exceed + 1) / (n_perm + 1)
    return {
        "observed_ratio": float(observed),
        "p_value": float(p),
        "n_perm": int(n_perm),
        "null_mean": float(np.nanmean(null_vals)),
        "null_p95": float(np.nanpercentile(null_vals, 95)),
    }


def random_panel_null(
    panel: pd.DataFrame,
    candidate_cols: Sequence[str],
    age_col: str,
    strata: Sequence[Tuple[float, float]],
    k: int,
    n_panels: int = 500,
    observed_ratio: float = float("nan"),
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Random-panel null — sample k biomarkers from the candidate pool and compute
    the max/min λ_max ratio across age strata. Compare the observed HDR panel
    ratio against this null distribution.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    candidates = [c for c in candidate_cols if c in panel.columns]
    from itertools import combinations

    all_combos = list(combinations(candidates, k))
    if len(all_combos) > n_panels:
        sampled_idx = rng.choice(len(all_combos), size=n_panels, replace=False)
        combos = [all_combos[i] for i in sampled_idx]
    else:
        combos = all_combos

    ratios = []
    ages = panel[age_col].to_numpy(dtype=float)
    for cols in combos:
        mat = panel[list(cols)].to_numpy(dtype=float)
        ok = np.all(np.isfinite(mat), axis=1) & np.isfinite(ages)
        if ok.sum() < 50:
            continue
        m2, a2 = mat[ok], ages[ok]
        vals = []
        for lo, hi in strata:
            sel = (a2 >= lo) & (a2 <= hi)
            if sel.sum() < 5:
                continue
            vals.append(lambda_max_of_cov(m2[sel]))
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        if len(vals) < 2:
            continue
        ratios.append(max(vals) / min(vals))

    if not ratios:
        return {
            "n_panels": 0,
            "observed_ratio": float(observed_ratio),
            "p_value": float("nan"),
        }
    ratios = np.array(ratios)
    p = (
        float(np.mean(ratios >= observed_ratio))
        if np.isfinite(observed_ratio)
        else float("nan")
    )
    return {
        "n_panels": int(len(ratios)),
        "k": int(k),
        "observed_ratio": float(observed_ratio),
        "p_value": p,
        "null_mean": float(np.mean(ratios)),
        "null_median": float(np.median(ratios)),
        "null_p95": float(np.percentile(ratios, 95)),
        "null_ratios": ratios.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def config_hash(cfg: Dict[str, Any]) -> str:
    """SHA-1 of the effective config — used in result headers for provenance."""
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def save_json(path: str | os.PathLike, payload: Dict[str, Any]) -> None:
    """Write JSON with NaN → None coercion and deterministic formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _clean(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return [_clean(x) for x in o.tolist()]
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    with open(p, "w", encoding="utf-8") as f:
        json.dump(_clean(payload), f, indent=2, default=str)


def header_comment(script: str, cfg_hash: str) -> str:
    """One-line comment suitable for CSV / JSON headers."""
    from datetime import datetime

    return (
        f"# generated by {script} on {datetime.utcnow().isoformat(timespec='seconds')}Z"
        f"; config_hash={cfg_hash}"
    )
