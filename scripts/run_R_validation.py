#!/usr/bin/env python3
"""
Predicted vs empirical correlation structure — the first out-of-sample test of
the compiled coupling matrix.

Pre-registered in docs/prereg_R_test.md. Read that first; the failure criteria
were fixed before this script produced any output.

Predicted:  A = -D(I - G),  A Gamma + Gamma A' = -Q,  R_pred = corr(Gamma)
Observed:   R_emp = corr of within-person changes dx = x(t+dt) - x(t)

With G = 0 and Q = sigma^2 I the solution is Gamma = (sigma^2/2) diag(tau), so
R_pred = I exactly. Every predicted off-diagonal is produced by G alone.

H4 is the decisive test: G is all-positive, so any non-negative matrix predicts
positive correlations everywhere. The real G must beat a permutation null that
reshuffles the same numbers among the same positions.

Usage
-----
    python scripts/run_R_validation.py --cohort elsa
    python scripts/run_R_validation.py --cohort inchianti --data-root <path>
    python scripts/run_R_validation.py --merge a.json b.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from hdr_sim.csv_loader import load_J_csv, build_J_basin  # noqa: E402

ALL9 = ("I", "M", "E", "mito", "P", "C", "N", "F", "B")
KAPPA_SOURCES = ("F", "B")
REGIME_DEPENDENT = (("N", "I"), ("N", "B"), ("C", "I"))
N_PERM = 10000
N_BOOT = 5000
SEED = 2026
SCALE_CI_FROM_PHI = [0.188, 1.342]     # H3 pre-registered interval


# ---------------------------------------------------------------------------
# model side
# ---------------------------------------------------------------------------

def load_tau():
    with open(os.path.join(ROOT, "data", "tau_registry.json")) as f:
        return json.load(f)


def tau_at(reg, axis, age):
    e = reg["axes"][axis]
    a = [25.0, 80.0, 120.0]
    t = [float(e["tau25"]), float(e["tau80"]), float(e["tau120"])]
    age = float(np.clip(age, a[0], a[-1]))
    for k in range(2):
        if a[k] <= age <= a[k + 1]:
            f = (age - a[k]) / (a[k + 1] - a[k])
            return float(np.exp((1 - f) * np.log(t[k]) + f * np.log(t[k + 1])))
    return t[-1]


def adopted_G(axes):
    """Healthy-basin G on the requested axis subset, protective entries removed
    to kappa and sign-reversing regime-dependent entries zeroed."""
    rows = load_J_csv(os.path.join(ROOT, "data", "J_matrix_compiled_9x9.csv"))
    M9 = build_J_basin(rows, "healthy", ALL9)
    idx = [ALL9.index(a) for a in axes]
    G = M9[np.ix_(idx, idx)].copy()
    np.fill_diagonal(G, 0.0)
    for s in KAPPA_SOURCES:
        if s in axes:
            G[:, axes.index(s)] = 0.0
    for src, tgt in REGIME_DEPENDENT:
        if src in axes and tgt in axes:
            G[axes.index(tgt), axes.index(src)] = 0.0
    assert (G >= 0).all()
    return G


def predicted_R(G, tau, q_mode="I"):
    n = len(tau)
    D = np.diag(1.0 / tau)
    A = -D @ (np.eye(n) - G)
    if q_mode == "I":
        Q = np.eye(n)
    elif q_mode == "D":
        Q = D
    elif q_mode == "D2":
        Q = D @ D
    else:
        raise ValueError(q_mode)
    Gam = solve_continuous_lyapunov(A, -Q)
    d = np.sqrt(np.clip(np.diag(Gam), 1e-300, None))
    return Gam / np.outer(d, d)


def offdiag(M):
    n = M.shape[0]
    return np.array([M[i, j] for i in range(n) for j in range(n) if i != j])


# ---------------------------------------------------------------------------
# data side
# ---------------------------------------------------------------------------

def changes_from_panel(panel_std, axis_cols, id_col, age_col="age"):
    """Within-person consecutive-wave changes, with the age at the first visit."""
    rows = []
    for subj, grp in panel_std.groupby(id_col):
        grp = grp.sort_values("wave")
        for i in range(len(grp) - 1):
            r0, r1 = grp.iloc[i], grp.iloc[i + 1]
            if any(pd.isna(r0[a]) or pd.isna(r1[a]) for a in axis_cols):
                continue
            rec = {"subj": subj, "age": float(r0[age_col])}
            for a in axis_cols:
                rec["d_" + a] = float(r1[a]) - float(r0[a])
            rows.append(rec)
    return pd.DataFrame(rows)


def empirical_R(changes, axis_cols):
    X = changes[["d_" + a for a in axis_cols]].values
    X = X[np.isfinite(X).all(axis=1)]
    return np.corrcoef(X, rowvar=False), int(len(X))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def run_tests(strata, rng):
    """strata: list of dicts with r_pred, r_emp (off-diagonal vectors), label."""
    pred = np.concatenate([s["r_pred"] for s in strata])
    emp = np.concatenate([s["r_emp"] for s in strata])

    # H1 sign concordance
    sign_ok = int(((pred > 0) == (emp > 0)).sum())
    h1 = {"concordant": sign_ok, "n": int(len(pred)),
          "rate": float(sign_ok / len(pred)),
          "fail": bool(sign_ok / len(pred) <= 0.5)}

    # H2 pattern correlation, bootstrap CI
    r = float(np.corrcoef(pred, emp)[0, 1])
    boots = np.empty(N_BOOT)
    n = len(pred)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        boots[b] = np.corrcoef(pred[i], emp[i])[0, 1]
    lo, hi = np.percentile(boots[np.isfinite(boots)], [2.5, 97.5])
    h2 = {"r": r, "ci95": [float(lo), float(hi)],
          "fail": bool(lo <= 0 <= hi)}

    # H3 level: slope through the origin
    b_hat = float(np.sum(pred * emp) / np.sum(pred * pred))
    sb = np.empty(N_BOOT)
    for k in range(N_BOOT):
        i = rng.integers(0, n, n)
        sb[k] = np.sum(pred[i] * emp[i]) / np.sum(pred[i] * pred[i])
    blo, bhi = np.percentile(sb, [2.5, 97.5])
    h3 = {"slope": b_hat, "ci95": [float(blo), float(bhi)],
          "prereg_interval": SCALE_CI_FROM_PHI,
          "fail": bool(bhi < SCALE_CI_FROM_PHI[0] or blo > SCALE_CI_FROM_PHI[1])}

    return {"H1_sign": h1, "H2_pattern": h2, "H3_level": h3,
            "n_offdiag_total": int(len(pred))}


def permutation_null(strata_spec, G, rng):
    """H4. Reshuffle the non-zero off-diagonal entries of G among the non-zero
    off-diagonal POSITIONS, recompute R_pred, recompute the H2 correlation."""
    n = G.shape[0]
    pos = [(i, j) for i in range(n) for j in range(n)
           if i != j and G[i, j] != 0]
    vals = np.array([G[i, j] for i, j in pos])

    emp = np.concatenate([s["r_emp"] for s in strata_spec])
    pred_real = np.concatenate([s["r_pred"] for s in strata_spec])
    r_real = float(np.corrcoef(pred_real, emp)[0, 1])

    null = np.empty(N_PERM)
    for k in range(N_PERM):
        Gp = np.zeros_like(G)
        for (i, j), v in zip(pos, rng.permutation(vals)):
            Gp[i, j] = v
        try:
            pv = np.concatenate([
                offdiag(predicted_R(Gp, s["tau"], s["q_mode"]))
                for s in strata_spec])
            null[k] = np.corrcoef(pv, emp)[0, 1]
        except Exception:
            null[k] = np.nan
    null = null[np.isfinite(null)]
    p = float((null >= r_real).sum() + 1) / (len(null) + 1)
    return {"r_real": r_real, "n_perm": int(len(null)),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_q95": float(np.percentile(null, 95)),
            "p_value": p, "fail": bool(p >= 0.05),
            "note": ("one-sided: the real G must predict the observed "
                     "correlation structure better than a reshuffling of the "
                     "same numbers among the same positions")}


# ---------------------------------------------------------------------------
# cohort drivers
# ---------------------------------------------------------------------------

def build_strata(changes, axes, reg, q_mode, G, n_strata=4):
    qs = np.quantile(changes["age"], np.linspace(0, 1, n_strata + 1))
    qs[0] -= 1e-9
    out = []
    for k in range(n_strata):
        sub = changes[(changes["age"] > qs[k]) & (changes["age"] <= qs[k + 1])]
        if len(sub) < 50:
            continue
        R_emp, n = empirical_R(sub, axes)
        age_mid = float(sub["age"].median())
        tau = np.array([tau_at(reg, a, age_mid) for a in axes])
        R_pred = predicted_R(G, tau, q_mode)
        out.append({"label": f"age {qs[k]:.0f}-{qs[k+1]:.0f}",
                    "age_median": age_mid, "n_changes": n,
                    "tau": tau, "q_mode": q_mode,
                    "r_pred": offdiag(R_pred), "r_emp": offdiag(R_emp),
                    "R_pred": R_pred.tolist(), "R_emp": R_emp.tolist()})
    return out


def run_elsa(q_mode):
    from scripts.elsa_lead_lag import (
        load_all_files, extract_nurse_biomarkers, prepare_harmonised,
        extract_mortality, extract_supplementary, harmonised_to_long,
        build_analysis_panel, AXES as EA)
    files = load_all_files()
    panel, _ = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    merged = build_analysis_panel(panel, harmonised_to_long(harm, waves=[2, 4, 6, 8]),
                                  extract_mortality(files, harm),
                                  extract_supplementary(files))
    axes = list(EA.keys())
    cols = [EA[a] for a in axes]
    std = merged.rename(columns={EA[a]: a for a in axes})
    ch = changes_from_panel(std, axes, "idauniq")
    return axes, ch, "ELSA", "elsa_3axis_IMF"


def run_inchianti(data_root, q_mode):
    from hdr_sim.inchianti import (load_inchianti_panel, compute_youthful_reference,
                                   standardize_axes)
    panel = load_inchianti_panel(data_root=data_root)
    ref = compute_youthful_reference(panel)
    std = standardize_axes(panel, ref)
    axes = ["I", "M", "N", "F"]
    std = std.rename(columns={f"delta_{a}": a for a in axes})
    ch = changes_from_panel(std, axes, "code98")
    return axes, ch, "InCHIANTI", "inchianti_4axis_hr"


# ---------------------------------------------------------------------------

def analyse(axes, changes, cohort, model, q_mode):
    reg = load_tau()
    G = adopted_G(tuple(axes))
    rng = np.random.default_rng(SEED)
    strata = build_strata(changes, axes, reg, q_mode, G)
    res = run_tests(strata, rng)
    res["H4_specificity"] = permutation_null(strata, G, rng)

    # H5 level by stratum
    per = []
    for s in strata:
        p, e = s["r_pred"], s["r_emp"]
        per.append({"stratum": s["label"], "age_median": s["age_median"],
                    "n_changes": s["n_changes"],
                    "slope": float(np.sum(p * e) / np.sum(p * p)),
                    "r": float(np.corrcoef(p, e)[0, 1])})
    sl = [x["slope"] for x in per]
    res["H5_age_invariance"] = {
        "by_stratum": per,
        "slope_min": float(min(sl)), "slope_max": float(max(sl)),
        "ratio": float(max(sl) / min(sl)) if min(sl) > 0 else None,
        "fail": bool(min(sl) <= 0 or max(sl) / min(sl) > 2.0)}

    return {"cohort": cohort, "model": model, "axes": axes, "q_mode": q_mode,
            "n_strata": len(strata),
            "strata": [{k: v for k, v in s.items()
                        if k not in ("r_pred", "r_emp", "tau")} for s in strata],
            "tests": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["elsa", "inchianti"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--merge", nargs="+", default=None)
    ap.add_argument("-o", "--output", default=None)
    a = ap.parse_args()

    if a.merge:
        blocks = []
        for p in a.merge:
            with open(p) as f:
                blocks.append(json.load(f))
        # H6 cross-cohort
        prim = [b["results"]["I"] for b in blocks]
        cis = [r["tests"]["H3_level"]["ci95"] for r in prim]
        overlap = max(cis[0][0], cis[1][0]) <= min(cis[0][1], cis[1][1]) \
            if len(cis) == 2 else None
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prereg": "docs/prereg_R_test.md",
            "cohorts": blocks,
            "H6_cross_cohort_invariance": {
                "slopes": [{"cohort": r["cohort"],
                            "slope": r["tests"]["H3_level"]["slope"],
                            "ci95": r["tests"]["H3_level"]["ci95"]} for r in prim],
                "cis_overlap": overlap,
                "fail": (None if overlap is None else bool(not overlap))},
        }
        out = a.output or os.path.join(ROOT, "results", "R_validation.json")
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out}")
        return

    if a.cohort == "elsa":
        axes, ch, cohort, model = run_elsa("I")
    else:
        axes, ch, cohort, model = run_inchianti(a.data_root, "I")

    print(f"{cohort}: {len(ch)} within-person changes, axes {axes}")
    results = {}
    for q in ("I", "D", "D2"):
        results[q] = analyse(axes, ch, cohort, model, q)

    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "prereg": "docs/prereg_R_test.md",
               "cohort": cohort, "model": model, "axes": axes,
               "n_changes_total": int(len(ch)),
               "results": results,
               "primary_q_mode": "I"}
    out = a.output or os.path.join(ROOT, "results", f"R_validation_{a.cohort}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    for q in ("I", "D", "D2"):
        t = results[q]["tests"]
        print(f"\n--- Q = {q} ---")
        print(f"  H1 sign      {t['H1_sign']['concordant']}/{t['H1_sign']['n']} "
              f"({t['H1_sign']['rate']:.2f})  fail={t['H1_sign']['fail']}")
        print(f"  H2 pattern   r = {t['H2_pattern']['r']:+.3f} "
              f"CI [{t['H2_pattern']['ci95'][0]:+.3f}, {t['H2_pattern']['ci95'][1]:+.3f}]"
              f"  fail={t['H2_pattern']['fail']}")
        print(f"  H3 level     b = {t['H3_level']['slope']:.3f} "
              f"CI [{t['H3_level']['ci95'][0]:.3f}, {t['H3_level']['ci95'][1]:.3f}]"
              f"  fail={t['H3_level']['fail']}")
        h4 = t["H4_specificity"]
        print(f"  H4 SPECIFIC  r_real = {h4['r_real']:+.3f}  null {h4['null_mean']:+.3f}"
              f" +/- {h4['null_sd']:.3f} (q95 {h4['null_q95']:+.3f})  "
              f"p = {h4['p_value']:.4f}  fail={h4['fail']}")
        h5 = t["H5_age_invariance"]
        print(f"  H5 age inv.  slope {h5['slope_min']:.3f}-{h5['slope_max']:.3f}"
              f"  fail={h5['fail']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
