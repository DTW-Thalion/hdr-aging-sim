#!/usr/bin/env python3
"""
Regenerate every derived number quoted in docs/units_convention.md and
docs/protective_coupling_formulation.md.

Nothing in those documents is typed by hand: run this and diff.

Inputs
------
  data/J_matrix_compiled_9x9.csv   compiled coupling entries (unchanged)
  data/tau_registry.json           unified recovery timescales (deliverable B)
  results/lead_lag_phi_table.json  beta and phi per ordered pair (deliverable D)

Output
------
  results/units_convention_numerics.json
  a readable table on stdout

Conventions
-----------
  G[i][j] = g_{i<-j}, the dimensionless steady-state gain of source j on target i.
  J[i][j] = J_{i<-j} in d^-1 under the rate reading.
  Rate:  A = -D + J,      D = diag(1/tau)
  Gain:  A = -D(I - G),   equivalently J = D G, so J_ij = g_ij / tau_i
  For non-negative G, A is Hurwitz iff rho(G) < 1.

Usage
-----
    python scripts/units_convention_numerics.py
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hdr_sim.csv_loader import load_J_csv, build_J_basin  # noqa: E402

ALL9 = ("I", "M", "E", "mito", "P", "C", "N", "F", "B")
FAST7 = ("I", "M", "mito", "P", "C", "N", "F")
BASINS = ("healthy", "pre_disease", "disease")

# Monte Carlo spec, identical to scripts/run_figure_uncertainty.py
# (Supplementary Note 1): lognormal multiplicative noise with a CV set by
# the confidence grade, plus a 5% sign-flip probability on grade-C entries.
GRADE_CV = {"A": 0.20, "B": 0.40, "C": 0.70, "": 0.40}
N_MC = 20000
MC_SEED = 2026

# Entries that move from J to the damping modulation kappa
# (docs/protective_coupling_formulation.md). Keyed by source axis.
KAPPA_SOURCES = ("F", "B")
# Regime-dependent entries: zero in the healthy basin, positive elsewhere.
REGIME_DEPENDENT = (("N", "I"), ("N", "B"), ("C", "I"))  # (source, target)

# The recalibration anchor (Task 1.3).
ANCHOR_AGE = 120
ANCHOR_ALPHA = -0.004
PUBLISHED_C = 0.89


# ---------------------------------------------------------------------------
# tau
# ---------------------------------------------------------------------------

def load_tau_registry():
    with open(os.path.join(ROOT, "data", "tau_registry.json")) as f:
        return json.load(f)


def tau_at(reg, axis, age):
    """Geometric (log-linear) interpolation between the declared anchors."""
    e = reg["axes"][axis]
    a = [25.0, 80.0, 120.0]
    t = [float(e["tau25"]), float(e["tau80"]), float(e["tau120"])]
    age = float(np.clip(age, a[0], a[-1]))
    for k in range(2):
        if a[k] <= age <= a[k + 1]:
            f = (age - a[k]) / (a[k + 1] - a[k])
            return float(np.exp((1 - f) * np.log(t[k]) + f * np.log(t[k + 1])))
    return t[-1]


def tau_vec(reg, age, axes=FAST7):
    return np.array([tau_at(reg, a, age) for a in axes])


# ---------------------------------------------------------------------------
# matrices
# ---------------------------------------------------------------------------

def basin_matrix(rows, basin, axes=FAST7):
    """Compiled numeric entries only (no tier imputation), signed."""
    M9 = build_J_basin(rows, basin, ALL9)
    idx = [ALL9.index(a) for a in axes]
    M = M9[np.ix_(idx, idx)].copy()
    np.fill_diagonal(M, 0.0)
    return M


def grade_matrix(rows, axes=FAST7):
    ai = {a: i for i, a in enumerate(axes)}
    g = np.full((len(axes), len(axes)), "", dtype=object)
    for r in rows:
        s, t = r["axis_from"].strip(), r["axis_to"].strip()
        if s in ai and t in ai:
            g[ai[t], ai[s]] = r.get("confidence_grade", "").strip() or "B"
    return g


def drop_sources(M, sources, axes=FAST7):
    out = M.copy()
    for s in sources:
        if s in axes:
            out[:, axes.index(s)] = 0.0
    return out


def zero_regime_dependent(M, axes=FAST7):
    out = M.copy()
    for src, tgt in REGIME_DEPENDENT:
        if src in axes and tgt in axes:
            out[axes.index(tgt), axes.index(src)] = 0.0
    return out


def rho(G):
    return float(np.max(np.abs(np.linalg.eigvals(G))))


def alpha_rate(tau, J, c=1.0):
    return float(np.max(np.real(np.linalg.eigvals(-np.diag(1.0 / tau) + c * J))))


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def section_rho(rows):
    """rho(G_fast) per basin, with the Monte Carlo band."""
    grades = grade_matrix(rows)
    rng = np.random.default_rng(MC_SEED)
    cv = np.array([[GRADE_CV.get(grades[i, j], 0.40) for j in range(7)]
                   for i in range(7)])
    out = {}
    for basin in BASINS:
        G = np.abs(basin_matrix(rows, basin))
        point = rho(G)
        mask = G > 0
        samples = np.empty(N_MC)
        for k in range(N_MC):
            Gk = G * np.exp(cv * rng.standard_normal((7, 7)))
            Gk[~mask] = 0.0
            np.fill_diagonal(Gk, 0.0)
            samples[k] = rho(Gk)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        out[basin] = {
            "rho_G_fast": point,
            "rho_G_fast_F_row_removed": rho(np.abs(drop_sources(G, ("F",)))),
            "loop_gain_I_M": float(G[FAST7.index("I"), FAST7.index("M")]
                                   * G[FAST7.index("M"), FAST7.index("I")]),
            "n_nonzero_fast_entries": int(mask.sum()),
            "mc_mean": float(samples.mean()),
            "mc_ci95": [float(lo), float(hi)],
            "mc_P_rho_ge_1": float((samples >= 1.0).mean()),
            "mc_n": N_MC,
        }
    return out


def section_alpha(rows, reg):
    """alpha_fast under the rate reading, unified tau, published c."""
    Jh = basin_matrix(rows, "healthy")
    Jd = basin_matrix(rows, "disease")
    out = {}
    for age in (25, 55, 80, 120):
        tau = tau_vec(reg, age)
        out[str(age)] = {
            "tau": {a: float(t) for a, t in zip(FAST7, tau)},
            "slowest_fast_axis": FAST7[int(np.argmax(tau))],
            "alpha_uncoupled": float(-1.0 / tau.max()),
            "alpha_healthy_c089": alpha_rate(tau, Jh, PUBLISHED_C),
            "alpha_disease_c089": alpha_rate(tau, Jd, PUBLISHED_C),
        }
    return out


def section_recalibration(rows, reg):
    """alpha_fast with the protective rows moved to kappa, and the
    calibration scalar that puts alpha_fast(120, healthy) on the anchor."""
    Jh = basin_matrix(rows, "healthy")
    variants = {
        "F_row_removed": drop_sources(Jh, ("F",)),
        "F_and_B_rows_removed": drop_sources(Jh, KAPPA_SOURCES),
        "F_and_B_rows_removed_plus_regime_dependent_zeroed":
            zero_regime_dependent(drop_sources(Jh, KAPPA_SOURCES)),
    }
    out = {}
    for name, J in variants.items():
        assert (J < 0).sum() == 0, f"{name} still carries negative entries"
        f = lambda c: alpha_rate(tau_vec(reg, ANCHOR_AGE), J, c) - ANCHOR_ALPHA
        c = brentq(f, 0.0, 50.0, xtol=1e-12)
        out[name] = {
            "c_star": float(c),
            "alpha_fast_120_healthy": alpha_rate(tau_vec(reg, 120), J, c),
            "alpha_fast_80_healthy": alpha_rate(tau_vec(reg, 80), J, c),
            "alpha_fast_25_healthy": alpha_rate(tau_vec(reg, 25), J, c),
            "at_published_c": {
                "alpha_fast_25_healthy": alpha_rate(tau_vec(reg, 25), J, PUBLISHED_C),
                "alpha_fast_120_healthy": alpha_rate(tau_vec(reg, 120), J, PUBLISHED_C),
            },
        }
    return out


def section_tau_sensitivity(rows, reg):
    """What the two contested tau decisions are worth, in alpha_fast."""
    Jh = basin_matrix(rows, "healthy")
    J_noF = drop_sources(Jh, ("F",))
    c_star = brentq(
        lambda c: alpha_rate(tau_vec(reg, ANCHOR_AGE), J_noF, c) - ANCHOR_ALPHA,
        0.0, 50.0, xtol=1e-12)
    alts = {
        "N_hpa_sub_channel_1d": ("N", 1.0),
        "C_mechanistic_2d": ("C", 2.0),
        "F_mechanistic_25d": ("F", 25.0),
    }
    out = {}
    for age in (25, 80, 120):
        tau = tau_vec(reg, age)
        row = {"adopted": {
            "alpha_healthy_c089": alpha_rate(tau, Jh, PUBLISHED_C),
            "alpha_healthy_F_removed_c_star": alpha_rate(tau, J_noF, c_star)}}
        for name, (axis, val) in alts.items():
            t2 = tau.copy()
            t2[FAST7.index(axis)] = val
            row[name] = {
                "alpha_healthy_c089": alpha_rate(t2, Jh, PUBLISHED_C),
                "alpha_healthy_F_removed_c_star": alpha_rate(t2, J_noF, c_star)}
        out[str(age)] = row
    out["c_star_used"] = float(c_star)
    return out


def section_empirical_gains(rows, reg):
    """Each significant cross-lagged pair against the two readings."""
    path = os.path.join(ROOT, "results", "lead_lag_phi_table.json")
    if not os.path.exists(path):
        return {"note": "results/lead_lag_phi_table.json not present"}
    with open(path) as f:
        tbl = json.load(f)

    out = []
    for p in tbl["pairs"]:
        if p.get("beta") is None or p["p_beta"] > 0.05:
            continue
        cj = p.get("compiled_J") or {}
        jh, jd = cj.get("J_healthy"), cj.get("J_disease")
        rec = {
            "model": p["model"], "pair": p["pair"],
            "beta": p["beta"], "p_beta": p["p_beta"], "phi_jj": p["phi_jj"],
            "empirical_long_run_gain": p.get("long_run_gain"),
            "se_empirical_long_run_gain": p.get("se_long_run_gain"),
            "compiled_as_gain_healthy": jh if isinstance(jh, float) else None,
            "compiled_as_gain_disease": jd if isinstance(jd, float) else None,
        }
        # rate reading: implied steady-state gain is tau_target * J
        if isinstance(jh, float):
            rec["rate_implied_gain_25_healthy"] = jh * tau_at(reg, p["target"], 25)
        if isinstance(jd, float):
            rec["rate_implied_gain_80_disease"] = jd * tau_at(reg, p["target"], 80)
        out.append(rec)
    return out


def section_entries_moving(rows):
    """Enumerate the entries that move from J to kappa, from the CSV."""
    moving, regime = [], []
    for r in rows:
        s, t = r["axis_from"].strip(), r["axis_to"].strip()
        sign = r["sign"].strip()
        rec = {"source": s, "target": t, "sign": sign,
               "tier": r["magnitude_tier"].strip(),
               "grade": r["confidence_grade"].strip(),
               "J_healthy": r["J_healthy"].strip(),
               "J_pre_disease": r["J_pre_disease"].strip(),
               "J_disease": r["J_disease"].strip()}
        if s in KAPPA_SOURCES and sign == "-":
            moving.append(rec)
        if (s, t) in REGIME_DEPENDENT:
            regime.append(rec)
    return {"move_to_kappa": moving, "regime_dependent": regime,
            "n_move_to_kappa": len(moving)}


# ---------------------------------------------------------------------------

def main():
    rows = load_J_csv(os.path.join(ROOT, "data", "J_matrix_compiled_9x9.csv"))
    reg = load_tau_registry()

    result = {
        "generated_by": "scripts/units_convention_numerics.py",
        "inputs": {
            "J": "data/J_matrix_compiled_9x9.csv",
            "tau": "data/tau_registry.json",
            "phi": "results/lead_lag_phi_table.json",
        },
        "monte_carlo_spec": {
            "grade_cv": GRADE_CV, "n": N_MC, "seed": MC_SEED,
            "source": "scripts/run_figure_uncertainty.py (Supplementary Note 1)"},
        "rho_G": section_rho(rows),
        "alpha_fast_rate_reading": section_alpha(rows, reg),
        "recalibration": section_recalibration(rows, reg),
        "tau_sensitivity": section_tau_sensitivity(rows, reg),
        "empirical_gains": section_empirical_gains(rows, reg),
        "entries": section_entries_moving(rows),
    }

    out = os.path.join(ROOT, "results", "units_convention_numerics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    r = result["rho_G"]
    print("rho(G_fast), compiled numeric entries, |values|")
    print(f"  {'basin':<13}{'rho':>7}{'F-row out':>11}{'MC 95%':>20}{'P(rho>=1)':>11}")
    for b in BASINS:
        d = r[b]
        print(f"  {b:<13}{d['rho_G_fast']:>7.3f}{d['rho_G_fast_F_row_removed']:>11.3f}"
              f"   [{d['mc_ci95'][0]:.3f}, {d['mc_ci95'][1]:.3f}]{d['mc_P_rho_ge_1']:>11.3f}")

    print("\nalpha_fast, rate reading, unified tau, c = 0.89")
    print(f"  {'age':>5}{'slowest':>9}{'uncoupled':>11}{'healthy':>10}{'disease':>10}")
    for age in ("25", "55", "80", "120"):
        d = result["alpha_fast_rate_reading"][age]
        print(f"  {age:>5}{d['slowest_fast_axis']:>9}{d['alpha_uncoupled']:>11.4f}"
              f"{d['alpha_healthy_c089']:>10.4f}{d['alpha_disease_c089']:>10.4f}")

    print(f"\nrecalibration on alpha_fast({ANCHOR_AGE}, healthy) = {ANCHOR_ALPHA}")
    for k, d in result["recalibration"].items():
        print(f"  {k:<52} c* = {d['c_star']:.3f}   "
              f"a(25) = {d['alpha_fast_25_healthy']:+.4f}   "
              f"a(80) = {d['alpha_fast_80_healthy']:+.4f}")

    print(f"\nentries moving to kappa: {result['entries']['n_move_to_kappa']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
