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

# Monte Carlo spec, following scripts/run_figure_uncertainty.py
# (Supplementary Note 1): lognormal multiplicative noise with a CV set by the
# confidence grade.
# NOT carried over: that script also flips the sign of a grade-C entry with
# probability 0.05. Under the adopted convention G is non-negative by
# construction (the protective entries move to kappa), so a sign flip would
# produce a matrix the convention does not admit and rho(G) would stop being
# a stability condition. The band below therefore propagates magnitude
# uncertainty only. Any claim of identity with that script is wrong.
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

def _mc_band(G, cv, seed):
    """Monte Carlo band on rho for a fixed non-negative G."""
    rng = np.random.default_rng(seed)
    mask = G > 0
    samples = np.empty(N_MC)
    for k in range(N_MC):
        Gk = G * np.exp(cv * rng.standard_normal(G.shape))
        Gk[~mask] = 0.0
        np.fill_diagonal(Gk, 0.0)
        samples[k] = rho(Gk)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"rho": rho(G),
            "n_nonzero_fast_entries": int(mask.sum()),
            "mc_mean": float(samples.mean()),
            "mc_ci95": [float(lo), float(hi)],
            "mc_P_rho_ge_1": float((samples >= 1.0).mean()),
            "mc_n": N_MC}


def section_rho(rows):
    """rho(G_fast) per basin, under two different matrices.

    adopted/  -- THE convention-correct object. The protective entries move to
                 the damping modulation (docs/protective_coupling_formulation.md),
                 so they are REMOVED from G, leaving G >= 0 and rho(G) < 1 an
                 exact stability condition. This is what may be quoted.

    abs_diagnostic/ -- the pre-decision diagnostic reproduced from
                 HDR-mechanistic/results/units_evidence_package.md Section 4,
                 which takes |J| over every compiled entry including the
                 protective ones. Retained ONLY so that document's numbers
                 remain traceable. Taking |-0.35| as a gain of +0.35 asserts
                 the opposite sign to the compiled entry, so this is not a
                 quantity the adopted convention defines. Do not quote it.
    """
    grades = grade_matrix(rows)
    cv = np.array([[GRADE_CV.get(grades[i, j], 0.40) for j in range(7)]
                   for i in range(7)])
    out = {}
    for n, basin in enumerate(BASINS):
        M = basin_matrix(rows, basin)
        G_adopted = drop_sources(M, KAPPA_SOURCES)     # protective rows -> kappa
        assert (G_adopted >= 0).all(), f"{basin}: G still signed after kappa removal"
        G_abs = np.abs(M)
        rec = {"adopted": _mc_band(G_adopted, cv, MC_SEED + n),
               "abs_diagnostic": _mc_band(G_abs, cv, MC_SEED + n)}
        rec["loop_gain_I_M"] = float(
            G_adopted[FAST7.index("I"), FAST7.index("M")]
            * G_adopted[FAST7.index("M"), FAST7.index("I")])
        out[basin] = rec
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
    # Each alternative substitutes the rejected value for one axis. The
    # rejected mechanistic values for C and F carry their own age formula
    # (HDR-mechanistic docs/tau_reconciliation.md:
    #  tau = lo + (hi-lo)*(age-30)/50, clipped), so they are substituted as
    # trajectories, not as single numbers -- substituting a constant would
    # flatten the age trajectory by construction and overstate the case.
    def _mech_traj(lo, hi):
        return lambda age: lo + (hi - lo) * float(np.clip((age - 30) / 50, 0, 1))

    alts = {
        "N_hpa_sub_channel": ("N", _mech_traj(0.5, 2.0)),
        "C_mechanistic_1_to_3d": ("C", _mech_traj(1.0, 3.0)),
        "F_mechanistic_8_to_42d": ("F", _mech_traj(8.0, 42.0)),
    }
    ages = (25, 40, 55, 70, 80, 100, 120)
    out = {}
    for age in ages:
        tau = tau_vec(reg, age)
        row = {"adopted": {
            "slowest_fast_axis": FAST7[int(np.argmax(tau))],
            "alpha_healthy_c089": alpha_rate(tau, Jh, PUBLISHED_C),
            "alpha_healthy_F_removed_c_star": alpha_rate(tau, J_noF, c_star)}}
        for name, (axis, fn) in alts.items():
            t2 = tau.copy()
            t2[FAST7.index(axis)] = fn(age)
            row[name] = {
                "slowest_fast_axis": FAST7[int(np.argmax(t2))],
                "tau_substituted": float(fn(age)),
                "alpha_healthy_c089": alpha_rate(t2, Jh, PUBLISHED_C),
                "alpha_healthy_F_removed_c_star": alpha_rate(t2, J_noF, c_star)}
        out[str(age)] = row

    # Fold change in the recalibrated margin across the lifespan, per variant:
    # this is the quantity that says whether the age trajectory survives the
    # alternative tau, as distinct from whether the level moves.
    folds = {}
    for name in ["adopted"] + list(alts):
        a25 = out["25"][name]["alpha_healthy_F_removed_c_star"]
        a120 = out["120"][name]["alpha_healthy_F_removed_c_star"]
        folds[name] = float(a25 / a120) if a120 != 0 else None
    out["fold_change_25_to_120_F_removed_c_star"] = folds
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


def section_j_signs_audit():
    """Audit the hard-coded sign predictions in scripts/inchianti_lead_lag.py.

    That module defines J_SIGNS with comments written (source, target) but
    consumes the keys as (target, source):

        for (to_name, from_name), predicted_sign in J_SIGNS.items():

    so an entry's prediction lands on the reverse of the pair its comment
    describes. This recomputes the 12-pair concordance count three ways:
    as shipped, with the key order corrected, and reading the predicted sign
    from the compiled CSV (which under the adopted convention is +1 for every
    channel = J entry).
    """
    import csv as _csv
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "_ill", os.path.join(ROOT, "scripts", "inchianti_lead_lag.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    J_SIGNS = mod.J_SIGNS

    rows = {(r["axis_from"].strip(), r["axis_to"].strip()): r
            for r in load_J_csv(os.path.join(ROOT, "data",
                                             "J_matrix_compiled_9x9.csv"))}
    path = os.path.join(ROOT, "results", "inchianti_lead_lag_matrix.csv")
    if not os.path.exists(path):
        return {"note": "results/inchianti_lead_lag_matrix.csv not present"}
    obs = {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            obs[(r["from_axis"], r["to_axis"])] = (
                float(r["beta"]), int(r["predicted_sign"]))

    pairs, n_ship = [], 0
    n_keyfix = n_csv = 0
    for (k0, k1), sgn in J_SIGNS.items():
        to_, from_ = k0, k1                       # as consumed
        beta, pred = obs[(from_, to_)]
        intended = J_SIGNS.get((from_, to_))      # what the comment order means
        compiled = rows[(from_, to_)]["sign"].strip()
        comp_s = 1 if compiled == "+" else -1
        obs_s = 1 if beta > 0 else -1
        shipped_ok = obs_s == pred
        keyfix_ok = (obs_s == intended) if intended is not None else None
        csv_ok = obs_s == comp_s
        n_ship += bool(shipped_ok)
        n_keyfix += bool(keyfix_ok)
        n_csv += bool(csv_ok)
        pairs.append({
            "pair": f"{from_}->{to_}", "beta": beta,
            "sign_applied_as_shipped": pred,
            "sign_the_comment_order_intends": intended,
            "compiled_sign": compiled,
            "concordant_as_shipped": bool(shipped_ok),
            "concordant_key_order_corrected": keyfix_ok,
            "concordant_compiled_sign": bool(csv_ok),
            "verdict_flips_on_key_fix": intended is not None and intended != pred,
        })
    flipped = [p["pair"] for p in pairs if p["verdict_flips_on_key_fix"]]
    return {
        "pairs": pairs,
        "n_pairs_affected_by_key_order": len(flipped),
        "pairs_affected": flipped,
        "concordance_as_shipped": f"{n_ship}/12",
        "concordance_key_order_corrected": f"{n_keyfix}/12",
        "concordance_compiled_sign": f"{n_csv}/12",
        "net_effect_on_count": (
            "none: the same number of verdicts flip each way, so the published "
            "9/12 is numerically unaffected by the bug. What changes is WHICH "
            "pairs are counted discordant."),
    }


def section_scale_diagnostic(rows, reg):
    """Is the compiled level consistent with the measured gains?

    Compares each significant, positively-signed cross-lagged pair's empirical
    steady-state gain against the midpoint of its compiled healthy/disease
    values read as gains.  Protective (F- and B-row) sources are EXCLUDED:
    their compiled values move to kappa and are no longer gains, so including
    them compares two different quantities and inflates the ratio.

    Reports the geometric-mean ratio, its dispersion, and a one-sample t-test
    on the log ratio against 1.  A ratio significantly above 1 would mean the
    compiled tiers are systematically too large.
    """
    from scipy import stats

    kept, excluded = [], []
    for e in section_empirical_gains(rows, reg):
        g = e.get("empirical_long_run_gain")
        jh, jd = e.get("compiled_as_gain_healthy"), e.get("compiled_as_gain_disease")
        if g is None or jh is None or jd is None or g <= 0:
            continue
        rec = {"model": e["model"], "pair": e["pair"],
               "empirical_gain": g,
               "compiled_gain_midpoint": (abs(jh) + abs(jd)) / 2.0}
        rec["ratio"] = rec["compiled_gain_midpoint"] / g
        (excluded if jh < 0 else kept).append(rec)

    lr = np.log([r["ratio"] for r in kept])
    t = stats.ttest_1samp(lr, 0.0)
    gm = float(np.exp(lr.mean()))
    # 95% CI on the geometric mean
    se = float(lr.std(ddof=1) / np.sqrt(len(lr)))
    crit = float(stats.t.ppf(0.975, len(lr) - 1))
    ci = [float(np.exp(lr.mean() - crit * se)), float(np.exp(lr.mean() + crit * se))]

    return {
        "pairs_used": kept,
        "pairs_excluded_protective": excluded,
        "n": int(len(lr)),
        "geometric_mean_ratio": gm,
        "geometric_mean_ratio_ci95": ci,
        "log_ratio_sd": float(lr.std(ddof=1)),
        "multiplicative_spread": float(np.exp(lr.std(ddof=1))),
        "ratio_range": [float(min(r["ratio"] for r in kept)),
                        float(max(r["ratio"] for r in kept))],
        "t_test_log_ratio_vs_zero": {"t": float(t.statistic), "p": float(t.pvalue)},
        "interpretation": (
            "A geometric-mean ratio of 1 means the compiled level agrees with "
            "the measured gains. The CI is what matters: it bounds how far the "
            "level of G is identified by the data we have."),
    }


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
        "scale_diagnostic": section_scale_diagnostic(rows, reg),
        "j_signs_audit": section_j_signs_audit(),
        "entries": section_entries_moving(rows),
    }

    out = os.path.join(ROOT, "results", "units_convention_numerics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    r = result["rho_G"]
    print("rho(G_fast)  [adopted = protective entries moved to kappa, G >= 0]")
    print(f"  {'basin':<13}{'rho':>7}{'MC 95%':>20}{'P(rho>=1)':>11}   "
          f"{'(abs diag.)':>12}")
    for b in BASINS:
        a, x = r[b]["adopted"], r[b]["abs_diagnostic"]
        print(f"  {b:<13}{a['rho']:>7.3f}   [{a['mc_ci95'][0]:.3f}, "
              f"{a['mc_ci95'][1]:.3f}]{a['mc_P_rho_ge_1']:>11.4f}   {x['rho']:>12.3f}")

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

    sd = result["scale_diagnostic"]
    print(f"\ncompiled level vs measured gains (protective sources excluded)")
    print(f"  n = {sd['n']}   geometric-mean ratio = {sd['geometric_mean_ratio']:.2f}  "
          f"95% CI [{sd['geometric_mean_ratio_ci95'][0]:.2f}, "
          f"{sd['geometric_mean_ratio_ci95'][1]:.2f}]  "
          f"p = {sd['t_test_log_ratio_vs_zero']['p']:.3f}")

    ja = result["j_signs_audit"]
    if "pairs" in ja:
        print(f"\nJ_SIGNS key-order audit (scripts/inchianti_lead_lag.py)")
        print(f"  pairs affected: {ja['n_pairs_affected_by_key_order']} "
              f"({', '.join(ja['pairs_affected'])})")
        print(f"  as shipped {ja['concordance_as_shipped']}   "
              f"key-order corrected {ja['concordance_key_order_corrected']}   "
              f"compiled-sign {ja['concordance_compiled_sign']}")

    print(f"\nentries moving to kappa: {result['entries']['n_move_to_kappa']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
