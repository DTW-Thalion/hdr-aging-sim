"""
Shared utilities for disease-specific coupling analyses on ELSA.

Mirrors the InCHIANTI `_utils.py` API but loads ELSA's consolidated nurse
biomarkers + supplementary variables + harmonised HDR subset. No imports
from `src/hdr_sim`.

Panel structure:
  code98 (= idauniq) | wave | age | sex | log_il6 (log CRP) | homa_ir (NaN —
  no insulin in ELSA; we use hba1c+bmi composite for M-axis instead) |
  hscrp | hba1c | chol_hdl_ratio | log_trig | sysval | diaval | pulval |
  grip_max | walk_speed |
  med_statin (OTC only) | med_antidm (hemdb) | med_antihtn (hemda) |
  delta_I | delta_M | delta_N | delta_F

Cross-lagged regression and age-matching helpers are identical to the
InCHIANTI versions; we copy them here to keep the folder self-contained.

ELSA has 4 nurse waves: 2, 4, 6, 8. That gives 3 consecutive pairs.
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

# UTF-8 stdout for Windows arrow characters
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
ELSA_DIR = os.path.join(_REPO, "data", "elsa")
NURSE_PATH = os.path.join(ELSA_DIR, "elsa_nurse_biomarkers_consolidated.tab")
SUPP_PATH = os.path.join(ELSA_DIR, "elsa_supplementary_variables.tab")
HARMON_PATH = os.path.join(ELSA_DIR, "gh_elsa_h_hdr_subset.tab")
MED_FLAGS_PATH = os.path.join(ELSA_DIR, "elsa_med_flags_waves_2_4_6_8.parquet")

NURSE_WAVES = [2, 4, 6, 8]
GRIP_COLS = ["mmgsd1", "mmgsd2", "mmgsd3"]


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def _melt_harmonised(h):
    """
    Extract a long-format panel of age, sex, diabetes, highbp, walk_speed
    at each of waves 2, 4, 6, 8 from the harmonised HRS-style file.
    """
    rows = []
    for w in NURSE_WAVES:
        sub_cols = {"idauniq": "idauniq"}
        for src, dst in [
            (f"r{w}agey", "age"),
            (f"r{w}diabe", "diabetes"),
            (f"r{w}hibpe", "highbp"),
            (f"r{w}walkra", "walking_difficulty"),
        ]:
            if src in h.columns:
                sub_cols[src] = dst
        if "ragender" in h.columns:
            sub_cols["ragender"] = "sex"
        s = h[list(sub_cols.keys())].rename(columns=sub_cols).copy()
        s["wave"] = w
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def _load_raw_panel():
    nurse = pd.read_csv(NURSE_PATH, sep="\t")
    supp = pd.read_csv(SUPP_PATH, sep="\t")
    harmon = pd.read_csv(HARMON_PATH, sep="\t")

    # Force lowercase column names for robustness
    nurse.columns = [c.lower() for c in nurse.columns]
    supp.columns = [c.lower() for c in supp.columns]
    harmon.columns = [c.lower() for c in harmon.columns]

    # Merge nurse + supplementary on (idauniq, wave) to get hemda/hemdb
    panel = nurse.merge(
        supp[["idauniq", "wave", "hemda", "hemdb"]],
        on=["idauniq", "wave"],
        how="left",
        suffixes=("", "_supp"),
    )
    for c in ["hemda", "hemdb"]:
        alt = c + "_supp"
        if alt in panel.columns:
            panel[c] = panel[c].fillna(panel[alt])
            panel.drop(columns=[alt], inplace=True)

    # Add age, sex, diabetes/highbp, walking difficulty from harmonised
    hlong = _melt_harmonised(harmon)
    # Harmonised file has some string-coded missing values — coerce to numeric
    for c in ["age", "diabetes", "highbp", "walking_difficulty", "sex"]:
        if c in hlong.columns:
            hlong[c] = pd.to_numeric(hlong[c], errors="coerce")
    panel = panel.merge(hlong, on=["idauniq", "wave"], how="left")

    # Merge medication flags extracted from raw core waves (hechmd, hepmed,
    # hehrtmd, heostec, heins, heacea, hehno, hehelf). Overrides hemda/hemdb
    # from the supplementary file where non-null.
    if os.path.exists(MED_FLAGS_PATH):
        med = pd.read_parquet(MED_FLAGS_PATH)
        med.columns = [c.lower() for c in med.columns]
        # Avoid duplicate-column conflicts: suffix and coalesce
        panel = panel.merge(med, on=["idauniq", "wave"], how="left",
                            suffixes=("", "_raw"))
        for c in ["hemda", "hemdb"]:
            alt = c + "_raw"
            if alt in panel.columns:
                panel[c] = panel[alt].fillna(panel[c])
                panel.drop(columns=[alt], inplace=True)

    # Max grip strength across trials
    grip_dom = [c for c in GRIP_COLS if c in panel.columns]
    panel["grip_max"] = panel[grip_dom].max(axis=1) if grip_dom else np.nan

    # Walk speed placeholder (not in consolidated; use walking difficulty
    # as inverse proxy if walk_speed missing)
    panel["walk_speed"] = np.nan  # Fill if we recover it from raw in future

    # Derived composites
    # Chol/HDL ratio
    if "chol" in panel.columns and "hdl" in panel.columns:
        panel["chol_hdl_ratio"] = panel["chol"] / panel["hdl"].replace(0, np.nan)
    else:
        panel["chol_hdl_ratio"] = np.nan
    # log(trig)
    panel["log_trig"] = np.log(panel["trig"].replace(0, np.nan)) if "trig" in panel.columns else np.nan
    # log CRP
    panel["log_crp"] = np.log(panel["hscrp"].clip(lower=0.01)) if "hscrp" in panel.columns else np.nan

    # Rename idauniq → code98 for API parity with InCHIANTI utils
    panel = panel.rename(columns={"idauniq": "code98"})
    return panel


def _binarize_elsa_yn(series):
    """ELSA encodes 1=Yes, 2=No. Map to 1=Yes, 0=No; other → NaN."""
    if series is None:
        return pd.Series(np.nan)
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index)
    out[s == 1] = 1.0
    out[s == 2] = 0.0
    return out


def _compute_ref_elsa(panel):
    """Reference group: age 50-55 at wave 2."""
    ref = panel[(panel["wave"] == 2) & (panel["age"] >= 50) & (panel["age"] <= 55)]
    return ref


def load_panel_with_deltas():
    """
    Load ELSA panel and add delta_I, delta_M, delta_N, delta_F.

    Axes (ELSA 4-axis, matching scripts/run_elsa_validation.py):
      I: log CRP (single biomarker)
      M: z-mean of hba1c, chol/hdl ratio, log(trig)
      N: z-mean of sysval, diaval, pulval
      F: grip_max (sign-flipped)  [walk_speed not in consolidated file]

    Medication flags (1 = user, 0 = non-user, NaN = unknown):
      med_statin  → `statins` (OTC-only; see README caveat)
      med_antidm  → `hemdb`   (antidiabetic meds)
      med_antihtn → `hemda`   (antihypertensive meds)
    """
    panel = _load_raw_panel()
    ref = _compute_ref_elsa(panel)

    def _z(col, logit=False):
        x = np.log(panel[col].replace(0, np.nan)) if logit else panel[col]
        r = np.log(ref[col].replace(0, np.nan).dropna()) if logit else ref[col].dropna()
        if len(r) < 10:
            mu, sd = np.nanmean(x), np.nanstd(x)
        else:
            mu, sd = r.mean(), r.std()
        sd = max(sd, 1e-6)
        return (x - mu) / sd

    # I axis
    panel["delta_I"] = _z("hscrp", logit=True)

    # M axis: composite of (hba1c, chol/hdl, log_trig)
    m_zs = []
    for c in ["hba1c", "chol_hdl_ratio"]:
        if c in panel.columns:
            m_zs.append(_z(c))
    if "trig" in panel.columns:
        m_zs.append(_z("trig", logit=True))
    if m_zs:
        panel["delta_M"] = pd.concat(m_zs, axis=1).mean(axis=1, skipna=True)
    else:
        panel["delta_M"] = np.nan

    # N axis: composite of (sysval, diaval, pulval)
    n_zs = []
    for c in ["sysval", "diaval", "pulval"]:
        if c in panel.columns:
            n_zs.append(_z(c))
    panel["delta_N"] = pd.concat(n_zs, axis=1).mean(axis=1, skipna=True) if n_zs else np.nan

    # F axis: grip_max (sign-flipped: higher grip = healthier)
    panel["delta_F"] = -_z("grip_max")

    # Medication flags (all encoded 1=Yes, 2=No in ELSA; mapped to 0/1)
    #   med_statin      → hechmd (prescribed cholesterol-lowering meds).
    #                     Available w4, w6, w8. Much larger N than the
    #                     OTC-only `statins` flag (~2000 vs 98 per wave).
    #   med_statin_otc  → statins (OTC statin purchases). Kept for comparison.
    #   med_antidm      → hemdb (antidiabetic, 1=Yes/2=No, conditional on dx DM).
    #   med_antihtn     → hemda (antihypertensive, conditional on dx HTN).
    #   med_nsaid       → hepmed (knee/hip pain meds, OA patients only, w4+w6).
    #                     This is an NSAID proxy — not exact.
    #   med_aspirin     → hehrtmd (blood-thinning meds for CVD, w4 only).
    #   med_bisphos     → heostec (osteoporosis meds, w2/w6/w8).
    #   med_insulin     → heins (diabetic insulin injections).
    panel["med_statin"] = _binarize_elsa_yn(panel.get("hechmd"))
    panel["med_statin_otc"] = _binarize_elsa_yn(panel.get("statins"))
    panel["med_antidm"] = _binarize_elsa_yn(panel.get("hemdb"))
    panel["med_antihtn"] = _binarize_elsa_yn(panel.get("hemda"))
    panel["med_nsaid"] = _binarize_elsa_yn(panel.get("hepmed"))
    panel["med_aspirin"] = _binarize_elsa_yn(panel.get("hehrtmd"))
    panel["med_bisphos"] = _binarize_elsa_yn(panel.get("heostec"))
    panel["med_insulin"] = _binarize_elsa_yn(panel.get("heins"))

    ref_info = {
        "age_range": "50-55 at wave 2",
        "n_ref": int(len(ref)),
    }
    return panel, ref_info


# ─────────────────────────────────────────────────────────────────
# Cross-lagged regression (identical API to InCHIANTI utils)
# ─────────────────────────────────────────────────────────────────

def _build_lag_pairs(panel, source, target, group_col=None,
                     extra_cols=None):
    """One row per consecutive-wave pair (w, w+next) with src/tgt deltas."""
    src_col = f"delta_{source}"
    tgt_col = f"delta_{target}"
    need = ["code98", "wave", "age", src_col, tgt_col]
    if group_col is not None and group_col not in need:
        need.append(group_col)
    if extra_cols:
        for c in extra_cols:
            if c not in need:
                need.append(c)
    sub = panel[[c for c in need if c in panel.columns]].sort_values(
        ["code98", "wave"]).reset_index(drop=True)
    sub["next_wave"] = sub.groupby("code98")["wave"].shift(-1)
    sub["tgt_wn"] = sub.groupby("code98")[tgt_col].shift(-1)
    # ELSA waves are spaced by 2 (2→4→6→8), so consecutive diff == 2
    sub = sub[(sub["next_wave"] - sub["wave"]) == 2].copy()
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
            if c in sub.columns:
                out[c] = sub[c]
    out = out.dropna(subset=["src_w", "tgt_w", "tgt_wn", "age_w"])
    if group_col is not None:
        out = out.dropna(subset=["group"])
    return out


def _fit_ols(df, formula):
    import statsmodels.formula.api as smf
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def cross_lagged_regression(panel, source, target, group_col=None,
                            extra_controls=None, min_n=30):
    """
    Identical API to the InCHIANTI version: returns dict with 'overall',
    'groups', and 'interaction' keys.
    """
    extra_controls = extra_controls or []
    pairs = _build_lag_pairs(panel, source, target,
                             group_col=group_col,
                             extra_cols=extra_controls)
    out = {"source": source, "target": target,
           "n_pairs_total": int(len(pairs))}

    rhs = "src_w + tgt_w + age_w"
    for c in extra_controls:
        rhs += f" + {c}"

    if len(pairs) >= min_n:
        m = _fit_ols(pairs, f"tgt_wn ~ {rhs}")
        out["overall"] = {
            "beta": float(m.params["src_w"]),
            "se": float(m.bse["src_w"]),
            "p": float(m.pvalues["src_w"]),
            "n": int(m.nobs),
        }

    if group_col is None:
        return out

    groups = {}
    for g_val, sub in pairs.groupby("group"):
        if len(sub) < min_n:
            groups[str(g_val)] = {"n": int(len(sub)), "skipped": True}
            continue
        m = _fit_ols(sub, f"tgt_wn ~ {rhs}")
        groups[str(g_val)] = {
            "beta": float(m.params["src_w"]),
            "se": float(m.bse["src_w"]),
            "p": float(m.pvalues["src_w"]),
            "n": int(m.nobs),
        }
    out["groups"] = groups

    g_unique = sorted(pairs["group"].dropna().unique())
    if len(g_unique) == 2 and set(g_unique).issubset({0, 1, 0.0, 1.0}):
        pairs["group_b"] = pairs["group"].astype(float)
        formula = f"tgt_wn ~ src_w * group_b + tgt_w + age_w"
        for c in extra_controls:
            formula += f" + {c}"
        m = _fit_ols(pairs, formula)
        out["interaction"] = {
            "beta": float(m.params.get("src_w:group_b", np.nan)),
            "se": float(m.bse.get("src_w:group_b", np.nan)),
            "p": float(m.pvalues.get("src_w:group_b", np.nan)),
            "n": int(m.nobs),
            "contrast": f"{g_unique[1]} - {g_unique[0]}",
        }
    else:
        pairs["tert_lin"] = pairs["group"].astype(float)
        try:
            m = _fit_ols(pairs,
                         f"tgt_wn ~ src_w * tert_lin + tgt_w + age_w")
            out["interaction"] = {
                "beta": float(m.params["src_w:tert_lin"]),
                "se": float(m.bse["src_w:tert_lin"]),
                "p": float(m.pvalues["src_w:tert_lin"]),
                "n": int(m.nobs),
            }
        except Exception as e:
            out["interaction"] = {"error": str(e)}
    return out


# ─────────────────────────────────────────────────────────────────
# Age matching and tertile helpers
# ─────────────────────────────────────────────────────────────────

def age_match(df, group_col, age_col="age_w", caliper=5.0, seed=0):
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
        diffs_masked = np.where(ctrl_used, np.inf, diffs)
        best = np.argmin(diffs_masked)
        if diffs_masked[best] <= caliper:
            ctrl_used[best] = True
            matched_idx.append(row.name)
            matched_idx.append(ctrl_idx[best])
    return d.loc[matched_idx].copy()


def assign_tertiles(series, labels=(0, 1, 2)):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 9:
        return pd.Series(np.nan, index=series.index)
    try:
        q = pd.qcut(valid, 3, labels=labels, duplicates="drop")
    except ValueError:
        ranks = valid.rank(method="first")
        q = pd.qcut(ranks, 3, labels=labels, duplicates="drop")
    out = pd.Series(np.nan, index=series.index)
    out.loc[valid.index] = q.astype(float).values
    return out


# ─────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────

def format_ci(beta, se, z=1.96):
    lo = beta - z * se
    hi = beta + z * se
    return f"{beta:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def bonferroni(pvals):
    return [min(1.0, p * len(pvals)) for p in pvals]
