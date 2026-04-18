"""
Shared utilities for exploratory disease-specific coupling analyses.

Self-contained: does not import from src/hdr_sim. The existing parquet panel
(data/inchianti_panel.parquet) is used as the data source. PTH (5th axis)
and bisphosphonate flags are loaded on demand from raw SAS files.

Functions
---------
load_panel_with_deltas(waves=None, with_pth=False)
    Returns panel with delta_I, delta_M, delta_N, delta_F (and delta_B if
    with_pth=True) z-scored against a healthy young reference group.

cross_lagged_regression(panel, source, target, group_col=None, extra_controls=None)
    Fits Δx_target(w+1) = β·Δx_source(w) + γ·Δx_target(w) + δ·age_w (+ group
    interaction if group_col given). Returns β, SE, p, N per group plus the
    interaction p-value.

age_match(df, group_col, age_col='age_w', caliper=5.0)
    Nearest-neighbor 1:1 matching within a ±caliper-year age window
    (without replacement).

interaction_pvalue(df, source, target, group_col)
    Single model: Δtarget(w+1) ~ Δsource(w) * group + Δtarget(w) + age_w.

HC3 robust standard errors are used throughout. OLS via statsmodels.
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Ensure arrow characters print on Windows (cp1252 stdout)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PANEL_PATH = os.path.join(_REPO, "data", "inchianti_panel.parquet")
DATA_ROOT = os.path.expanduser(
    os.path.join("~", "Downloads", "inCHIANTI", "InCHIANTI_CD_Share")
)

# Raw-SAS paths for PTH / bisphosphonate loading (waves 0-3)
_RAW_WAVE_CONFIG = {
    0: {
        "lab": "Baseline_V8/English/4.Data/SAS_Datasets/Assays/labo_raw.sas7bdat",
        "drug": "Baseline_V8/English/4.Data/SAS_Datasets/Drugs/fmc_ana.sas7bdat",
        "pth_var": "X_PTH",
        "drug_prefix": "FX1_",
    },
    1: {
        "lab": "Follow-up1_V5/4.Data/SAS_Datasets/Assays/labf1raw.sas7bdat",
        "drug": "Follow-up1_V5/4.Data/SAS_Datasets/Drugs/fmcf1ana.sas7bdat",
        "pth_var": "Y_PTH_I",
        "drug_prefix": "FY1_",
    },
    2: {
        "lab": "Follow-up2_V4/4.Data/SAS_Datasets/Assays/labf2raw.sas7bdat",
        "drug": "Follow-up2_V4/4.Data/SAS_Datasets/Drugs/fmcf2ana.sas7bdat",
        "pth_var": "Z_PTH_I",
        "drug_prefix": "FZ1_",
    },
    3: {
        "lab": "Follow-up3_V3/4.Data/SAS_Datasets/Assays/labf3raw.sas7bdat",
        "drug": "Follow-up3_V3/4.Data/SAS_Datasets/Drugs/fmcf3ana.sas7bdat",
        "pth_var": "Q_PTH_I",
        "drug_prefix": "FQ1_",
    },
}

DX_EXCLUDE_HEALTHY = [
    "dx_htn", "dx_dm", "dx_metsyn", "dx_frailty",
    "dx_mi", "dx_chf", "dx_stroke", "dx_dement",
]


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def _compute_youthful_reference(panel, axes_log, age_min=20, age_max=30,
                                expand_to=35):
    """Mean/SD of each axis in healthy 20-30yo (expand to 35 if N<50)."""
    bl = panel[panel["wave"] == 0].copy()
    mask = pd.Series(True, index=bl.index)
    for dx in DX_EXCLUDE_HEALTHY:
        if dx in bl.columns:
            mask &= (bl[dx] != 1) | bl[dx].isna()
    mask_age = (bl["age"] >= age_min) & (bl["age"] <= age_max)
    ref = bl[mask & mask_age]
    if len(ref) < 50:
        mask_age = (bl["age"] >= age_min) & (bl["age"] <= expand_to)
        ref = bl[mask & mask_age]

    out = {}
    for axis, (var, log_it) in axes_log.items():
        vals = ref[var].dropna()
        if log_it and var != "log_il6":
            vals = np.log(vals + 0.1)
        out[axis] = {
            "var": var,
            "mean_ref": float(vals.mean()) if len(vals) > 0 else np.nan,
            "sd_ref": float(vals.std()) if len(vals) > 1 else np.nan,
            "n_ref": int(len(vals)),
            "log": bool(log_it),
        }
    # SPPB ceiling fix: if SD=0 in reference, widen to healthy <60
    if out["F"]["sd_ref"] == 0 or np.isnan(out["F"]["sd_ref"]):
        wider = bl[mask & (bl["age"] < 60)]
        sppb_wider = wider["sppb"].dropna()
        out["F"]["sd_ref"] = float(sppb_wider.std())
        out["F"]["sd_source"] = f"healthy <60 (N={len(sppb_wider)})"
    return out


def _standardize(panel, ref):
    """Add delta_I, delta_M, delta_N, delta_F columns (positive = decline)."""
    p = panel.copy()
    p["delta_I"] = (p["log_il6"] - ref["I"]["mean_ref"]) / ref["I"]["sd_ref"]
    log_homa = np.log(p["homa_ir"] + 0.1)
    p["delta_M"] = (log_homa - ref["M"]["mean_ref"]) / ref["M"]["sd_ref"]
    p["delta_N"] = (p["resting_hr"] - ref["N"]["mean_ref"]) / ref["N"]["sd_ref"]
    # SPPB: higher = better, flip sign so positive = decline
    p["delta_F"] = -(p["sppb"] - ref["F"]["mean_ref"]) / ref["F"]["sd_ref"]
    if "delta_B" in ref or "B" in ref:
        # PTH: higher = worse (secondary hyperparathyroidism) → positive = decline
        log_pth = np.log(p["pth"] + 0.1)
        p["delta_B"] = (log_pth - ref["B"]["mean_ref"]) / ref["B"]["sd_ref"]
    return p


def load_panel_with_deltas(waves=None, with_pth=False):
    """
    Load panel with z-scored axis deltas.

    Returns
    -------
    (panel, ref) tuple. `panel` has columns:
        code98, wave, age, sex, site, il6, homa_ir, resting_hr, sppb,
        med_statin, med_antihtn, med_antidm, med_nsaid, med_glucocort,
        delta_I, delta_M, delta_N, delta_F, (delta_B if with_pth)
    `ref` is dict of {axis: {mean_ref, sd_ref, n_ref, ...}}.
    """
    panel = pd.read_parquet(PANEL_PATH)
    if with_pth:
        panel = _merge_pth(panel)
        panel = _merge_bisphosphonate(panel)
    if waves is not None:
        panel = panel[panel["wave"].isin(waves)].copy()

    axes_log = {
        "I": ("log_il6", True),
        "M": ("homa_ir", True),
        "N": ("resting_hr", False),
        "F": ("sppb", False),
    }
    if with_pth and "pth" in panel.columns:
        axes_log["B"] = ("pth", True)

    ref = _compute_youthful_reference(panel, axes_log)
    panel = _standardize(panel, ref)
    return panel, ref


def _merge_pth(panel):
    """Merge PTH values (waves 0-3) from raw SAS files."""
    import pyreadstat

    panel = panel.copy()
    panel["pth"] = np.nan
    for wave, cfg in _RAW_WAVE_CONFIG.items():
        lab_path = os.path.join(DATA_ROOT, cfg["lab"])
        if not os.path.exists(lab_path):
            continue
        try:
            df, _ = pyreadstat.read_sas7bdat(lab_path)
        except Exception as e:
            warnings.warn(f"Failed to read {lab_path}: {e}")
            continue
        pth_var = cfg["pth_var"]
        if pth_var not in df.columns:
            continue
        sub = df[["CODE98", pth_var]].dropna().rename(
            columns={"CODE98": "code98", pth_var: "pth"}
        )
        sub = sub.drop_duplicates(subset=["code98"])
        mask = panel["wave"] == wave
        merged = panel.loc[mask, ["code98"]].merge(sub, on="code98", how="left")
        panel.loc[mask, "pth"] = merged["pth"].values
    return panel


def _merge_bisphosphonate(panel):
    """Merge bisphosphonate flag (FX1_M5 etc. = ATC M05) for waves 0-3."""
    import pyreadstat

    panel = panel.copy()
    panel["med_bisphos"] = np.nan
    for wave, cfg in _RAW_WAVE_CONFIG.items():
        drug_path = os.path.join(DATA_ROOT, cfg["drug"])
        if not os.path.exists(drug_path):
            continue
        try:
            df, _ = pyreadstat.read_sas7bdat(drug_path)
        except Exception:
            continue
        col = cfg["drug_prefix"] + "M5"
        if col not in df.columns:
            continue
        sub = df[["CODE98", col]].drop_duplicates(subset=["CODE98"]).rename(
            columns={"CODE98": "code98", col: "med_bisphos"}
        )
        mask = panel["wave"] == wave
        merged = panel.loc[mask, ["code98"]].merge(sub, on="code98", how="left")
        panel.loc[mask, "med_bisphos"] = merged["med_bisphos"].values
    return panel


# ─────────────────────────────────────────────────────────────────
# Cross-lagged regression
# ─────────────────────────────────────────────────────────────────

def _build_lag_pairs(panel, source, target, group_col=None,
                     extra_cols=None):
    """
    Build one row per consecutive-wave pair (w, w+1) with columns:
    code98, wave, src_w, tgt_w, tgt_wn, age_w [, group, ...extras].
    """
    src_col = f"delta_{source}"
    tgt_col = f"delta_{target}"
    need = ["code98", "wave", "age", src_col, tgt_col]
    if group_col is not None:
        need.append(group_col)
    if extra_cols:
        for c in extra_cols:
            if c not in need:
                need.append(c)
    sub = panel[need].sort_values(["code98", "wave"]).reset_index(drop=True)
    # Next-wave target
    sub["next_wave"] = sub.groupby("code98")["wave"].shift(-1)
    sub["tgt_wn"] = sub.groupby("code98")[tgt_col].shift(-1)
    sub = sub[(sub["next_wave"] - sub["wave"]) == 1].copy()
    out = pd.DataFrame({
        "code98": sub["code98"],
        "wave": sub["wave"],
        "src_w": sub[src_col],
        "tgt_w": sub[tgt_col],
        "tgt_wn": sub["tgt_wn"],
        "age_w": sub["age"],
    })
    if group_col is not None:
        out["group"] = sub[group_col]
    if extra_cols:
        for c in extra_cols:
            out[c] = sub[c]
    out = out.dropna(subset=["src_w", "tgt_w", "tgt_wn", "age_w"])
    if group_col is not None:
        out = out.dropna(subset=["group"])
    return out


def _fit_ols(df, formula):
    """OLS fit with HC3 robust SEs via statsmodels."""
    import statsmodels.formula.api as smf
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    return model


def cross_lagged_regression(panel, source, target, group_col=None,
                            extra_controls=None, min_n=30):
    """
    Run cross-lagged regression. If group_col is given, fits separately in
    each group AND the interaction model.

    Returns
    -------
    dict with keys:
      - overall: {beta, se, p, n}  (if group_col is None OR always available)
      - groups: {group_value: {beta, se, p, n}}  (if group_col given)
      - interaction: {beta_interaction, se, p, n}  (if group_col given)
    """
    extra_controls = extra_controls or []
    pairs = _build_lag_pairs(panel, source, target,
                             group_col=group_col,
                             extra_cols=extra_controls)
    out = {"source": source, "target": target,
           "n_pairs_total": int(len(pairs))}

    base_rhs = "src_w + tgt_w + age_w"
    for c in extra_controls:
        base_rhs += f" + {c}"

    # Overall fit (ignoring group)
    if len(pairs) >= min_n:
        m = _fit_ols(pairs, f"tgt_wn ~ {base_rhs}")
        out["overall"] = {
            "beta": float(m.params["src_w"]),
            "se": float(m.bse["src_w"]),
            "p": float(m.pvalues["src_w"]),
            "n": int(m.nobs),
        }

    if group_col is None:
        return out

    # Per-group fits
    groups = {}
    for g_val, sub in pairs.groupby("group"):
        if len(sub) < min_n:
            groups[str(g_val)] = {"n": int(len(sub)), "skipped": True}
            continue
        m = _fit_ols(sub, f"tgt_wn ~ {base_rhs}")
        groups[str(g_val)] = {
            "beta": float(m.params["src_w"]),
            "se": float(m.bse["src_w"]),
            "p": float(m.pvalues["src_w"]),
            "n": int(m.nobs),
        }
    out["groups"] = groups

    # Interaction model (group as 0/1 for binary, or C(group) for categorical)
    # Coerce group to numeric if it only has two distinct non-null values {0,1} etc.
    g_unique = sorted(pairs["group"].dropna().unique())
    if len(g_unique) == 2 and set(g_unique).issubset({0, 1, 0.0, 1.0, True, False}):
        pairs["group_b"] = pairs["group"].astype(float)
        formula = f"tgt_wn ~ src_w * group_b + tgt_w + age_w"
        for c in extra_controls:
            formula += f" + {c}"
        m = _fit_ols(pairs, formula)
        interaction_key = "src_w:group_b"
        out["interaction"] = {
            "beta": float(m.params.get(interaction_key, np.nan)),
            "se": float(m.bse.get(interaction_key, np.nan)),
            "p": float(m.pvalues.get(interaction_key, np.nan)),
            "n": int(m.nobs),
            "contrast": f"{g_unique[1]} - {g_unique[0]}",
        }
    else:
        # Categorical interaction (e.g. tertiles)
        formula = f"tgt_wn ~ src_w * C(group) + tgt_w + age_w"
        for c in extra_controls:
            formula += f" + {c}"
        m = _fit_ols(pairs, formula)
        int_terms = {k: {"beta": float(v), "se": float(m.bse[k]),
                         "p": float(m.pvalues[k])}
                     for k, v in m.params.items() if "src_w:" in k}
        out["interaction"] = {
            "terms": int_terms,
            "n": int(m.nobs),
            "joint_wald_p": _joint_wald_interaction_p(m, "src_w:"),
        }
    return out


def _joint_wald_interaction_p(model, term_prefix):
    """Joint Wald test for all coefficients starting with term_prefix."""
    import numpy as np
    terms = [n for n in model.params.index if n.startswith(term_prefix)]
    if not terms:
        return np.nan
    try:
        test = model.wald_test(terms, scalar=True)
        return float(test.pvalue)
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────
# Age matching
# ─────────────────────────────────────────────────────────────────

def age_match(df, group_col, age_col="age_w", caliper=5.0, seed=0):
    """
    Nearest-neighbor 1:1 matching without replacement: for each row with
    group==1, find the closest group==0 row within |age|<=caliper years.
    Returns the subset of df with matched rows only.
    """
    rng = np.random.default_rng(seed)
    d = df.dropna(subset=[group_col, age_col]).copy()
    grp = d[group_col].astype(float)
    treat = d[grp == 1].sample(frac=1.0, random_state=seed)
    ctrl = d[grp == 0].copy()
    ctrl_used = np.zeros(len(ctrl), dtype=bool)
    ctrl_ages = ctrl[age_col].values
    ctrl_idx = ctrl.index.values

    matched_idx = []
    for _, row in treat.iterrows():
        age_t = row[age_col]
        diffs = np.abs(ctrl_ages - age_t)
        # mask out used controls
        diffs_masked = np.where(ctrl_used, np.inf, diffs)
        best = np.argmin(diffs_masked)
        if diffs_masked[best] <= caliper:
            ctrl_used[best] = True
            matched_idx.append(row.name)
            matched_idx.append(ctrl_idx[best])
    return d.loc[matched_idx].copy()


# ─────────────────────────────────────────────────────────────────
# Tertile helper
# ─────────────────────────────────────────────────────────────────

def assign_tertiles(series, labels=(0, 1, 2)):
    """Return integer tertile labels for a series (0=low, 1=mid, 2=high)."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 9:
        return pd.Series(np.nan, index=series.index)
    try:
        q = pd.qcut(valid, 3, labels=labels, duplicates="drop")
    except ValueError:
        # Fallback for heavily tied distributions (e.g. SPPB ceiling)
        # Use rank-based split
        ranks = valid.rank(method="first")
        q = pd.qcut(ranks, 3, labels=labels, duplicates="drop")
    out = pd.Series(np.nan, index=series.index)
    out.loc[valid.index] = q.astype(float).values
    return out


# ─────────────────────────────────────────────────────────────────
# Summary formatting
# ─────────────────────────────────────────────────────────────────

def format_ci(beta, se, z=1.96):
    lo = beta - z * se
    hi = beta + z * se
    return f"{beta:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def bonferroni(pvals):
    """Bonferroni-adjusted p-values (clipped to [0,1])."""
    return [min(1.0, p * len(pvals)) for p in pvals]
