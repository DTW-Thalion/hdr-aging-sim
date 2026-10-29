#!/usr/bin/env python3
"""
Extract autoregressive coefficients (phi) from the cross-lagged lead-lag fits.

Task 1.1 deliverable D (see docs/units_convention.md). The lead-lag scripts
(`scripts/elsa_lead_lag.py`, `scripts/inchianti_lead_lag.py`,
`scripts/inchianti_6axis_analysis.py`) all fit the same specification

    d_target = beta * source_t0 + gamma * target_t0 + delta * age_t + const

where d_target = x_target(t+1) - x_target(t).  Rearranging,

    x_target(t+1) = (1 + gamma) * x_target(t) + beta * x_source(t) + ...

so the VAR(1) autoregressive coefficient of the target is

    phi_jj = 1 + gamma

and the long-run (steady-state) gain implied by the pair is

    g = beta / (1 - phi_jj) = -beta / gamma      (requires gamma < 0)

The published scripts report beta but not gamma, so the gain-calibration table
in HDR-mechanistic (`results/gain_calibration_table.md`) could only bracket the
long-run gain over assumed phi.  This script exports beta, its SE and p, gamma,
phi_jj and its SE, N, N_subjects and the realised visit interval for every
ordered pair, so the bracket can be replaced by a point estimate.

Cohorts
-------
  elsa          3-axis (I, M, F); data in data/elsa/ (in-repo)
  inchianti     4-axis (I, M, N=resting HR, F)  -- the manuscript's 12-pair set
                5-axis (I, M, N=cortisol/DHEAS, F, B=PTH) -- the B-axis pairs
                requires the InCHIANTI CD Share (not in-repo; --data-root)

Usage
-----
    python scripts/extract_lead_lag_phi.py --cohort elsa
    python scripts/extract_lead_lag_phi.py --cohort inchianti --data-root <path>
    python scripts/extract_lead_lag_phi.py --merge a.json b.json -o results/lead_lag_phi_table.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Core regression: returns beta AND the autoregressive coefficient
# ---------------------------------------------------------------------------

def cross_lagged_with_phi(y, x_from, x_auto, x_age, subj=None):
    """Fit d_target ~ beta*source_t0 + gamma*target_t0 + delta*age + const.

    Returns a dict with beta, se_beta, p_beta, gamma, se_gamma, p_gamma,
    phi_jj (= 1 + gamma), se_phi (= se_gamma), n, n_subjects, r2.
    """
    valid = (np.isfinite(y) & np.isfinite(x_from) &
             np.isfinite(x_auto) & np.isfinite(x_age))
    y, x_from, x_auto, x_age = y[valid], x_from[valid], x_auto[valid], x_age[valid]
    n = int(len(y))
    n_subj = int(len(np.unique(subj[valid]))) if subj is not None else None

    if n < 10:
        return {"n": n, "n_subjects": n_subj, "beta": None, "phi_jj": None,
                "note": "fewer than 10 usable observations"}

    X = np.column_stack([np.ones(n), x_from, x_auto, x_age])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(n - X.shape[1], 1)
    sigma2 = float(np.sum(resid ** 2) / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))

    def _p(i):
        return float(2 * t_dist.sf(abs(coef[i] / se[i]), df=dof)) if se[i] > 0 else None

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    gamma = float(coef[2])
    phi = 1.0 + gamma

    out = {
        "n": n,
        "n_subjects": n_subj,
        "beta": float(coef[1]),
        "se_beta": float(se[1]),
        "p_beta": _p(1),
        "gamma": gamma,
        "se_gamma": float(se[2]),
        "p_gamma": _p(2),
        "phi_jj": float(phi),
        "se_phi_jj": float(se[2]),
        "age_coef": float(coef[3]),
        "r2": float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else None,
    }
    # Long-run gain beta / (1 - phi) = -beta / gamma
    if gamma < 0:
        out["long_run_gain"] = float(-coef[1] / gamma)
        # delta-method SE, treating cov(beta, gamma) from the OLS covariance
        cov = sigma2 * XtX_inv
        b, g = float(coef[1]), gamma
        d_db, d_dg = -1.0 / g, b / (g ** 2)
        var = (d_db ** 2 * cov[1, 1] + d_dg ** 2 * cov[2, 2]
               + 2 * d_db * d_dg * cov[1, 2])
        out["se_long_run_gain"] = float(np.sqrt(var)) if var > 0 else None
    else:
        out["long_run_gain"] = None
        out["se_long_run_gain"] = None
        out["note"] = ("gamma >= 0: the target is not mean-reverting over this "
                       "interval, so beta/(1-phi) is not defined")
    return out


# ---------------------------------------------------------------------------
# Compiled J lookup (for sign-concordance reporting)
# ---------------------------------------------------------------------------

def compiled_entry(rows, source, target):
    for r in rows:
        if r["axis_from"].strip() == source and r["axis_to"].strip() == target:
            def _v(k):
                s = r[k].strip()
                try:
                    return float(s)
                except ValueError:
                    return s
            return {
                "sign": r["sign"].strip(),
                "magnitude_tier": r["magnitude_tier"].strip(),
                "confidence_grade": r["confidence_grade"].strip(),
                "J_healthy": _v("J_healthy"),
                "J_pre_disease": _v("J_pre_disease"),
                "J_disease": _v("J_disease"),
            }
    return None


# ---------------------------------------------------------------------------
# ELSA
# ---------------------------------------------------------------------------

def run_elsa():
    from scripts.elsa_lead_lag import (
        load_all_files, extract_nurse_biomarkers, prepare_harmonised,
        extract_mortality, extract_supplementary, harmonised_to_long,
        build_analysis_panel, build_triplets, AXES as ELSA_AXES,
    )

    files = load_all_files()
    panel, _ = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    merged = build_analysis_panel(panel, harm_long, mort, supp)
    trip = build_triplets(merged)

    # ELSA nurse-visit waves are 2 calendar years apart per wave index, so this
    # is a NOMINAL interval derived from the wave index, not a realised age gap
    # (the InCHIANTI path uses the real age difference). Recorded as such in
    # interval_basis so the two cohorts are not silently compared.
    gap = (trip["wave_t1"].astype(float) - trip["wave_t"].astype(float)) * 2.0
    trip = trip.assign(_interval_years=gap)

    axes = list(ELSA_AXES.keys())  # ['I','M','F']
    return _pairs_from_triplets(
        trip, axes, id_col="idauniq", age_col="age_t",
        cohort="ELSA", model="elsa_3axis_IMF",
        proxies={"I": "log CRP (+fibrinogen)",
                 "M": "HbA1c + BMI composite",
                 "F": "grip strength (+gait, chair rises)"},
        nominal_interval=4.0)


# ---------------------------------------------------------------------------
# InCHIANTI
# ---------------------------------------------------------------------------

def run_inchianti(data_root):
    from hdr_sim.inchianti import (
        load_inchianti_panel, compute_youthful_reference, standardize_axes)
    from scripts.inchianti_6axis_analysis import (
        augment_panel, get_axis_configs, standardize_config)

    panel = load_inchianti_panel(data_root=data_root)

    results = []

    # ---- 4-axis (I, M, N = resting HR, F): the manuscript's 12-pair set ----
    ref = compute_youthful_reference(panel)
    std4 = standardize_axes(panel, ref)
    trip4 = _build_triplets_generic(std4, ["delta_I", "delta_M", "delta_N", "delta_F"],
                                    id_col="code98")
    results += _pairs_from_triplets(
        trip4, ["I", "M", "N", "F"], id_col="code98", age_col="age_t",
        cohort="InCHIANTI", model="inchianti_4axis_hr",
        proxies={"I": "IL-6", "M": "HOMA-IR",
                 "N": "resting heart rate (ECG)", "F": "SPPB"},
        nominal_interval=3.0)

    # ---- 5-axis (I, M, N = cortisol/DHEAS, F, B = PTH) ----
    panel5 = panel.copy()
    # the 5-axis configs standardise on log_homa_ir, which
    # scripts/inchianti_6axis_analysis.py builds in its own main()
    panel5["log_homa_ir"] = np.log(panel5["homa_ir"].clip(lower=0.01))
    panel5 = augment_panel(panel5)
    cfg = get_axis_configs()["5axis_IMNFB"]
    std5, _delta_cols = standardize_config(panel5, cfg)
    trip5 = _build_triplets_generic(
        std5, ["delta_I", "delta_M", "delta_N", "delta_F", "delta_B"],
        id_col="code98")
    results += _pairs_from_triplets(
        trip5, ["I", "M", "N", "F", "B"], id_col="code98", age_col="age_t",
        cohort="InCHIANTI_5axis", model="inchianti_5axis_IMNFB",
        proxies={"I": "IL-6", "M": "HOMA-IR", "N": "cortisol/DHEAS ratio",
                 "F": "SPPB", "B": "PTH (waves 0-3)"},
        nominal_interval=3.0)
    return results


def _build_triplets_generic(panel_std, axis_cols, id_col="code98"):
    """Consecutive-wave triplets; records the realised age gap per pair."""
    rows = []
    for subj, grp in panel_std.groupby(id_col):
        grp = grp.sort_values("wave")
        for i in range(len(grp) - 1):
            r0, r1 = grp.iloc[i], grp.iloc[i + 1]
            if any(pd.isna(r0[a]) or pd.isna(r1[a]) for a in axis_cols):
                continue
            row = {id_col: subj, "wave_t": r0["wave"], "wave_t1": r1["wave"],
                   "age_t": r0["age"],
                   "_interval_years": float(r1["age"]) - float(r0["age"])}
            for a in axis_cols:
                short = a.replace("delta_", "")
                row[f"{short}_t0"] = r0[a]
                row[f"d_{short}"] = r1[a] - r0[a]
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pair loop
# ---------------------------------------------------------------------------

def _pairs_from_triplets(trip, axes, id_col, age_col, cohort, model,
                         proxies, nominal_interval):
    from hdr_sim.csv_loader import load_J_csv
    rows_csv = load_J_csv()

    out = []
    for source in axes:
        for target in axes:
            if source == target:
                continue
            fit = cross_lagged_with_phi(
                trip[f"d_{target}"].values.astype(float),
                trip[f"{source}_t0"].values.astype(float),
                trip[f"{target}_t0"].values.astype(float),
                trip[age_col].values.astype(float),
                subj=trip[id_col].values)

            iv = trip["_interval_years"].astype(float)
            # 95% normal-approximation CIs; the prompt asks for an interval on
            # every pair and SEs alone are not one.
            if fit.get("beta") is not None:
                for k in ("beta", "phi_jj", "long_run_gain"):
                    v, se = fit.get(k), fit.get("se_" + k)
                    fit[k + "_ci95"] = (
                        [float(v - 1.959964 * se), float(v + 1.959964 * se)]
                        if v is not None and se is not None else None)
            fit.update({
                "interval_basis": ("nominal, from wave index"
                                   if cohort.startswith("ELSA")
                                   else "realised age difference"),
                "cohort": cohort,
                "model": model,
                "source": source,
                "target": target,
                "pair": f"{source}->{target}",
                "source_proxy": proxies.get(source),
                "target_proxy": proxies.get(target),
                "interval_years_mean": float(iv.mean()),
                "interval_years_median": float(iv.median()),
                "interval_years_sd": float(iv.std()),
                "interval_years_nominal": nominal_interval,
                "compiled_J": compiled_entry(rows_csv, source, target),
            })
            out.append(fit)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def wrap(results, cohorts_note=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/extract_lead_lag_phi.py",
        "specification": ("d_target = beta*source_t0 + gamma*target_t0 "
                          "+ delta*age_t + const;  phi_jj = 1 + gamma;  "
                          "long_run_gain = beta/(1-phi_jj) = -beta/gamma"),
        "units": {"beta": "SD target per SD source per visit interval",
                  "phi_jj": "dimensionless AR(1) coefficient per visit interval",
                  "long_run_gain": "SD target per SD source at steady state"},
        "note": cohorts_note,
        "pairs": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["elsa", "inchianti"])
    ap.add_argument("--data-root", default=None,
                    help="InCHIANTI_CD_Share path (InCHIANTI only)")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="merge previously written per-cohort JSON files")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    if args.merge:
        merged, notes = [], []
        for p in args.merge:
            with open(p) as f:
                d = json.load(f)
            merged += d["pairs"]
            if d.get("note"):
                notes.append(d["note"])
        payload = wrap(merged, "; ".join(notes) if notes else None)
    elif args.cohort == "elsa":
        payload = wrap(run_elsa(), "ELSA nurse-visit waves 2/4/6/8, in-repo data/elsa/")
    elif args.cohort == "inchianti":
        payload = wrap(run_inchianti(args.data_root),
                       "InCHIANTI CD Share, run against a local copy; "
                       "participant-level data not exported")
    else:
        ap.error("pass --cohort or --merge")

    out = args.output or os.path.join(ROOT, "results", "lead_lag_phi_table.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out}  ({len(payload['pairs'])} ordered pairs)")

    for r in payload["pairs"]:
        if r.get("beta") is None:
            continue
        g = r.get("long_run_gain")
        print(f"  {r['cohort']:<18} {r['pair']:<8} beta={r['beta']:+.4f} "
              f"(SE {r['se_beta']:.4f}, p={r['p_beta']:.4g})  "
              f"phi={r['phi_jj']:.4f} (SE {r['se_phi_jj']:.4f})  "
              f"N={r['n']}  g={'%.4f' % g if g is not None else 'n/d'}")


if __name__ == "__main__":
    main()
