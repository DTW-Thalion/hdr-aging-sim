#!/usr/bin/env python3
"""
Task 1.3 -- single-anchor recalibration under Convention G.

STATUS: the parameterisation is an OPEN DECISION. This script computes all
three candidates and does not pick one. Run it to reproduce the fork; pass
--parameterisation to emit a single choice once one is made.

The problem. Under Convention G there is no free scalar: J = D G exactly, so
alpha_fast is a PREDICTION, not a fit. The published coupling scalar c existed
only because the rate reading needed a units bridge, and gains are already
SD per SD. Taking the adopted G at the compiled level predicts
alpha_fast(120) = -0.0195 (51-day recovery) against the anchor's -0.004
(250 days) -- a 4.9x miss. Separately, the published alpha_fast(25) = -0.19
lies BELOW the uncoupled abscissa of -0.1667 and is unreachable with G >= 0 at
any scale; it was attainable only via the negative F row in J.

Three ways to close it, computed here as `scalar`, `compiled` and `tau_anchor`:

  scalar      Reinstate c and fit it on the anchor. Gives c = 2.00, which under
              gains asserts the compiled gains are 2x too SMALL -- against
              cross-lagged evidence that they are ~2x too large (a 4x
              disagreement). Puts rho(cG) = 0.940, so the HEALTHY basin is
              near-critical at every age (rho is tau-free), and the Monte Carlo
              gives P(alpha >= 0) = 0.28 at every age including 25. Coupling
              then supplies 82% of the late-life margin, which contradicts the
              convention's own claim that aging is D-only within a fixed G.

  compiled    Keep G as compiled, drop the anchor. alpha_fast(120) = -0.0195.
              Pyrkov's 250-day figure is not reproduced. Coupling supplies 12%
              of the late-life margin.

  tau_anchor  Keep the anchor, let it constrain tau_I(120) rather than the
              coupling. Requires tau_I(120) = 222 d against the declared 45 d,
              which is flagged tau120_extrapolated in data/tau_registry.json
              and has no measurement behind it. Implied Gompertz gamma = 0.042
              against 0.026 from the 25/80 anchors. Coupling supplies 11% of
              the margin. COST: Online Methods can no longer cite Pyrkov as
              independent agreement -- tau_I(120) becomes calibrated ON it.

`compiled` and `tau_anchor` agree that coupling supplies ~11-12% of the
late-life margin, i.e. D-dominated aging, consistent with Pi ~ 1.0. Only
`scalar` disagrees, and it is the outlier on the physics as well as on the
magnitudes.

Fixed inputs, settled upstream:
  - adopted G: protective entries to kappa (protective_coupling_formulation Sec 3),
    sign-reversing regime-dependent entries zeroed in the healthy basin (Sec 4)
  - tau from data/tau_registry.json
  - scale: compiled level, unscaled (decision of 2026-10-29). alpha is invariant
    to a global rescaling of G in any case -- c absorbs it exactly, which the
    scale_invariance_check verifies -- so that decision affects rho(G) and the
    loop gain only, not this calibration.

Outputs
-------
  results/task_1_3_fork.json          (default: all three, no choice made)
  results/task_1_3_recalibration.json (only with --parameterisation)

Usage
-----
    python scripts/run_task_1_3_recalibration.py
    python scripts/run_task_1_3_recalibration.py --parameterisation tau_anchor
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hdr_sim.csv_loader import load_J_csv, build_J_basin  # noqa: E402

ALL9 = ("I", "M", "E", "mito", "P", "C", "N", "F", "B")
FAST7 = ("I", "M", "mito", "P", "C", "N", "F")

ANCHOR_AGE = 120.0
ANCHOR_ALPHA = -0.004
PUBLISHED_C = 0.89
PUBLISHED_ALPHA_25 = -0.19

KAPPA_SOURCES = ("F", "B")
REGIME_DEPENDENT = (("N", "I"), ("N", "B"), ("C", "I"))  # (source, target)

GRADE_CV = {"A": 0.20, "B": 0.40, "C": 0.70, "": 0.40}
N_MC = 20000
MC_SEED = 2026
N_AGE_GRID = 7001          # matches the published "0/7,000 age points" claim


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


def tau_vec(reg, age):
    return np.array([tau_at(reg, a, age) for a in FAST7])


def adopted_G(rows, basin="healthy"):
    """Healthy-basin G under the adopted convention: protective entries removed
    to kappa, sign-reversing regime-dependent entries zeroed."""
    M9 = build_J_basin(rows, basin, ALL9)
    idx = [ALL9.index(a) for a in FAST7]
    G = M9[np.ix_(idx, idx)].copy()
    np.fill_diagonal(G, 0.0)
    for s in KAPPA_SOURCES:
        if s in FAST7:
            G[:, FAST7.index(s)] = 0.0
    if basin == "healthy":
        for src, tgt in REGIME_DEPENDENT:
            if src in FAST7 and tgt in FAST7:
                G[FAST7.index(tgt), FAST7.index(src)] = 0.0
    assert (G >= 0).all(), "adopted G must be non-negative"
    return G


def grade_cv_matrix(rows):
    ai = {a: i for i, a in enumerate(FAST7)}
    g = np.full((7, 7), "", dtype=object)
    for r in rows:
        s, t = r["axis_from"].strip(), r["axis_to"].strip()
        if s in ai and t in ai:
            g[ai[t], ai[s]] = r.get("confidence_grade", "").strip() or "B"
    return np.array([[GRADE_CV.get(g[i, j], 0.40) for j in range(7)]
                     for i in range(7)])


def alpha(tau, G, c):
    """alpha(A) with A = -D + c*G/... -- note G enters as a RATE via J = D G,
    so the calibrated system is A = -D(I - c G). Equivalent to -D + c*D*G."""
    D = np.diag(1.0 / tau)
    return float(np.max(np.real(np.linalg.eigvals(-D @ (np.eye(len(tau)) - c * G)))))


def rho(G):
    return float(np.max(np.abs(np.linalg.eigvals(G))))


# ---------------------------------------------------------------------------

VARIANTS = ("scalar", "compiled", "tau_anchor")


def tau_vec_variant(reg, age, variant, tau_I_120=None):
    """tau at an age. `tau_anchor` replaces the I-axis 120 anchor and
    re-interpolates the 80->120 leg geometrically through it."""
    t = tau_vec(reg, age)
    if variant == "tau_anchor" and tau_I_120 is not None and age > 80:
        iI = FAST7.index("I")
        f = (age - 80.0) / 40.0
        t[iI] = float(np.exp((1 - f) * np.log(tau_at(reg, "I", 80.0))
                             + f * np.log(tau_I_120)))
    return t


def solve_variant(reg, G, variant):
    """Return (c, tau_I_120) for a variant."""
    if variant == "scalar":
        c = brentq(lambda c: alpha(tau_vec(reg, ANCHOR_AGE), G, c) - ANCHOR_ALPHA,
                   0.0, 1.0 / rho(G) - 1e-9, xtol=1e-13)
        return float(c), None
    if variant == "compiled":
        return 1.0, None
    t120 = tau_vec(reg, ANCHOR_AGE)
    iI = FAST7.index("I")

    def f(x):
        t = t120.copy()
        t[iI] = x
        return alpha(t, G, 1.0) - ANCHOR_ALPHA
    return 1.0, float(brentq(f, 45.0, 5000.0, xtol=1e-9))


def evaluate(reg, G, cv, variant, full=False):
    c, tI = solve_variant(reg, G, variant)
    ages = [25, 40, 55, 70, 80, 100, 120]
    traj = {}
    for age in ages:
        t = tau_vec_variant(reg, age, variant, tI)
        a = alpha(t, G, c)
        traj[str(age)] = {
            "alpha": a,
            "recovery_timescale_days": float(1.0 / abs(a)) if a < 0 else None,
            "alpha_uncoupled": float(-1.0 / t.max()),
            "tau_I": float(t[FAST7.index("I")]),
            "slowest_fast_axis": FAST7[int(np.argmax(t))],
        }
    a120 = traj["120"]["alpha"]
    au120 = traj["120"]["alpha_uncoupled"]
    rec = {
        "variant": variant,
        "c": c,
        "tau_I_120": tI if tI is not None else tau_vec(reg, 120)[FAST7.index("I")],
        "tau_I_120_is_calibrated": variant == "tau_anchor",
        "rho_cG_healthy": float(c * rho(G)),
        "alpha_fast": traj,
        "alpha_fast_25_OUTPUT": traj["25"]["alpha"],
        "alpha_fast_120": a120,
        "anchor_met": bool(abs(a120 - ANCHOR_ALPHA) < 1e-6),
        "coupling_share_of_margin_at_120": float(1.0 - a120 / au120),
    }
    if variant == "tau_anchor":
        rec["implied_gompertz_gamma_25_to_120"] = float(
            np.log(tI / tau_vec(reg, 25)[FAST7.index("I")]) / 95.0)
        rec["declared_gompertz_gamma_25_to_80"] = float(
            np.log(tau_at(reg, "I", 80.0) / tau_at(reg, "I", 25.0)) / 55.0)

    if not full:
        return rec

    grid = np.linspace(25.0, 120.0, N_AGE_GRID)
    al = np.array([alpha(tau_vec_variant(reg, a, variant, tI), G, c) for a in grid])
    d = np.diff(al)
    rec["monotonicity"] = {
        "n_age_points": int(N_AGE_GRID),
        "criterion": "alpha_fast non-decreasing in age",
        "violations": int((d < 0).sum()),
        "stable_at_all_ages": bool((al < 0).all()),
        "n_ages_unstable": int((al >= 0).sum()),
        "fold_change_25_to_120": float(al[0] / al[-1]),
    }
    rng = np.random.default_rng(MC_SEED)
    mask = G > 0
    mc_ages = [25, 55, 80, 120]
    taus = {a: tau_vec_variant(reg, a, variant, tI) for a in mc_ages}
    samp = {a: np.empty(N_MC) for a in mc_ages}
    for k in range(N_MC):
        Gk = G * np.exp(cv * rng.standard_normal((7, 7)))
        Gk[~mask] = 0.0
        np.fill_diagonal(Gk, 0.0)
        for a in mc_ages:
            samp[a][k] = alpha(taus[a], Gk, c)
    rec["monte_carlo_alpha"] = {
        "spec": ("lognormal multiplicative noise per non-zero entry, CV by "
                 "confidence grade (A 0.20, B 0.40, C 0.70); no grade-C sign "
                 "flip, G must stay non-negative"),
        "n": N_MC, "seed": MC_SEED,
        "by_age": {str(a): {
            "mean": float(samp[a].mean()),
            "ci95": [float(np.percentile(samp[a], 2.5)),
                     float(np.percentile(samp[a], 97.5))],
            "P_alpha_ge_0": float((samp[a] >= 0).mean())} for a in mc_ages},
    }
    return rec


def scale_invariance(reg, G):
    out = []
    for s in (0.25, 0.5, 1.0, 2.0):
        Gs = s * G
        c = brentq(lambda c: alpha(tau_vec(reg, ANCHOR_AGE), Gs, c) - ANCHOR_ALPHA,
                   0.0, 1.0 / rho(Gs) - 1e-9, xtol=1e-13)
        out.append({"s": s, "c_star": float(c), "c_star_times_s": float(c * s),
                    "alpha_fast_25": alpha(tau_vec(reg, 25), Gs, c),
                    "rho_sG": rho(Gs)})
    return {"rows": out,
            "conclusion": ("c*s is constant and alpha_fast is unchanged: a global "
                           "rescaling of G is not identified by the alpha "
                           "calibration, so the scale decision affects rho(G) and "
                           "the loop gain only.")}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parameterisation", choices=VARIANTS, default=None,
                    help="emit a single chosen parameterisation; omit to emit "
                         "the undecided fork")
    args = ap.parse_args()

    rows = load_J_csv(os.path.join(ROOT, "data", "J_matrix_compiled_9x9.csv"))
    reg = load_tau()
    G = adopted_G(rows, "healthy")
    cv = grade_cv_matrix(rows)

    common = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/run_task_1_3_recalibration.py",
        "task": "Task 1.3 single-anchor recalibration under Convention G",
        "decision_provenance": {
            "convention": "G (gains) -- docs/units_convention.md, confirmed 2026-10-29",
            "tau": "data/tau_registry.json",
            "protective_entries": "moved to kappa -- protective_coupling_formulation Sec 3",
            "regime_dependent": "N->I, N->B, C->I zeroed in the healthy basin -- Sec 4",
            "scale": "compiled level, unscaled (decision of 2026-10-29)",
        },
        "anchor": {"age": ANCHOR_AGE, "alpha_fast": ANCHOR_ALPHA,
                   "source": "Pyrkov et al. 2021 DOSI autocorrelation"},
        "published_for_comparison": {
            "c": PUBLISHED_C, "alpha_fast_25": PUBLISHED_ALPHA_25,
            "free_parameters": 2,
            "note": ("the published alpha_fast(25) was FITTED (the objective "
                     "maximised |alpha(25)|) and lies below the uncoupled "
                     "abscissa of -0.1667, so it is unreachable with G >= 0"),
        },
        "rho_G_healthy_adopted": rho(G),
        "scale_invariance_check": scale_invariance(reg, G),
    }

    if args.parameterisation:
        payload = dict(common)
        payload["chosen_parameterisation"] = args.parameterisation
        payload["result"] = evaluate(reg, G, cv, args.parameterisation, full=True)
        path = os.path.join(ROOT, "results", "task_1_3_recalibration.json")
    else:
        payload = dict(common)
        payload["status"] = ("PARAMETERISATION UNDECIDED -- all three computed, "
                             "none chosen. See the module docstring.")
        payload["variants"] = {v: evaluate(reg, G, cv, v, full=True)
                               for v in VARIANTS}
        path = os.path.join(ROOT, "results", "task_1_3_fork.json")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"rho(G_healthy, adopted) = {rho(G):.4f}   "
          f"(published c = {PUBLISHED_C}, published alpha_25 = {PUBLISHED_ALPHA_25})")
    vs = ([args.parameterisation] if args.parameterisation else list(VARIANTS))
    for v in vs:
        r = (payload["result"] if args.parameterisation else payload["variants"][v])
        print(f"\n--- {v} ---")
        print(f"  c = {r['c']:.4f}   tau_I(120) = {r['tau_I_120']:.1f} d"
              f"{' (CALIBRATED)' if r['tau_I_120_is_calibrated'] else ''}"
              f"   rho(cG) = {r['rho_cG_healthy']:.3f}")
        print(f"  alpha_fast(25) = {r['alpha_fast_25_OUTPUT']:+.4f} [output]   "
              f"alpha_fast(120) = {r['alpha_fast_120']:+.5f}   "
              f"anchor met: {r['anchor_met']}")
        print(f"  coupling share of the margin at 120: "
              f"{100*r['coupling_share_of_margin_at_120']:.0f}%")
        m = r["monotonicity"]
        print(f"  monotonicity: {m['violations']}/{m['n_age_points']} violations; "
              f"stable at all ages: {m['stable_at_all_ages']}; "
              f"fold 25->120: {m['fold_change_25_to_120']:.1f}x")
        mc = r["monte_carlo_alpha"]["by_age"]
        print("  MC P(alpha>=0): " + "  ".join(
            f"age {a}: {mc[a]['P_alpha_ge_0']:.3f}" for a in ("25", "55", "80", "120")))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
