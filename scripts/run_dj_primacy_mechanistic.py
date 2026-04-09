#!/usr/bin/env python
"""D/J Primacy Decomposition using the mechanistic-evidence model.

Replicates R6 Supplementary Note 6's D/J primacy analysis using
mechanistic-evidence priors instead of the original parameterisation.

Five degradation regimes:
  - pure-D:    only tau degrades with age (J fixed at age-30 level)
  - 75D/25J:   75% D-degradation, 25% J-strengthening
  - 50D/50J:   equal mix (R6 baseline)
  - 25D/75J:   25% D, 75% J
  - pure-J:    only J strengthens with age (tau fixed at age-30 level)

Under 4 confound conditions (clean, survivorship, medication, both).

Produces: results/dj_primacy_mechanistic.json
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import numpy as np
from scipy import linalg

from hdr_sim.mechanistic_model import HDRMechanisticModel
from hdr_sim.observation_model import ObservationModel
from hdr_sim.synthetic_cohort import SyntheticCohort, CohortData
from hdr_sim.tier1_pipeline import Tier1Pipeline


# ======================================================================
# Configuration
# ======================================================================

REGIMES = [
    ("pure_D",   1.0, 0.0),
    ("75D_25J",  0.75, 0.25),
    ("50D_50J",  0.50, 0.50),
    ("25D_75J",  0.25, 0.75),
    ("pure_J",   0.0, 1.0),
]

AGE_STRATA = [(50, 60), (60, 70), (70, 80), (80, 90)]
N_PERSONS = 3000
N_VISITS = 4
VISIT_INTERVAL = 4  # years
SEED_BASE = 1000


# ======================================================================
# Regime-specific cohort generator
# ======================================================================

def generate_regime_cohort(
    model, obs, D_weight, J_weight,
    n_persons, seed, survivorship=False, medication=False,
):
    """Generate cohort under a specific D/J degradation regime.

    D_weight: fraction of tau degradation applied (0=fixed tau, 1=full age-dependent tau)
    J_weight: fraction of J degradation applied (0=fixed J_30, 1=full age-dependent J)
    """
    rng = np.random.default_rng(seed)
    N = n_persons
    K = N_VISITS
    n_latent = 9
    n_obs = obs.n_obs
    Q = 0.01 * np.eye(n_latent)

    baseline_ages = rng.uniform(50, 90, size=N)
    visit_ages = np.zeros((N, K))
    for k in range(K):
        visit_ages[:, k] = baseline_ages + k * VISIT_INTERVAL

    latent = np.zeros((N, K, n_latent))
    observed = np.zeros((N, K, n_obs))
    alive = np.ones((N, K), dtype=bool)
    medicated = np.zeros((N, n_latent), dtype=bool)

    if medication:
        for axis_name, frac in SyntheticCohort.DEFAULT_MED_PREVALENCE.items():
            if axis_name not in model._axis_idx:
                continue
            idx = model._axis_idx[axis_name]
            medicated[:, idx] = rng.random(N) < frac

    # Pre-compute matrices for the reference ages
    tau_ref = model._tau_of_age_full(30).copy()
    c = model.calibration_scalar

    def build_regime_A(age):
        """Build A matrix under the specified D/J regime."""
        f = model._interp_fraction(age)

        # tau: interpolate between tau_ref and full age-dependent tau
        tau_full = model._tau_of_age_full(age)
        tau_regime = tau_ref + D_weight * (tau_full - tau_ref)
        D = np.diag(1.0 / tau_regime)

        # J: interpolate between J_30 and full age-dependent J
        J_30 = c * model._build_J_raw(30)
        J_full = c * model._build_J_raw(age)
        J_regime = J_30 + J_weight * (J_full - J_30)

        return -D + J_regime

    for i in range(N):
        age_0 = visit_ages[i, 0]
        A_0 = build_regime_A(age_0)
        Gamma_0 = _safe_lyapunov(A_0, Q)
        x = rng.multivariate_normal(np.zeros(n_latent), Gamma_0)

        if medication:
            x = _compress(x, medicated[i])

        latent[i, 0] = x
        observed[i, 0] = obs.observe(x, seed=rng)

        for k in range(1, K):
            if not alive[i, k - 1]:
                alive[i, k] = False
                continue

            age_k = visit_ages[i, k]
            A_k = build_regime_A(age_k)
            dt_days = VISIT_INTERVAL * 365.25
            Phi = linalg.expm(A_k * dt_days)
            Gamma_k = _safe_lyapunov(A_k, Q)
            Sigma_eta = Gamma_k - Phi @ Gamma_k @ Phi.T
            Sigma_eta = _ensure_psd(Sigma_eta, n_latent)
            L = linalg.cholesky(Sigma_eta + 1e-14 * np.eye(n_latent), lower=True)
            eta = L @ rng.standard_normal(n_latent)
            x = Phi @ latent[i, k - 1] + eta

            if medication:
                x = _compress(x, medicated[i])

            latent[i, k] = x
            observed[i, k] = obs.observe(x, seed=rng)

        # Survivorship
        if survivorship:
            for k in range(1, K):
                if not alive[i, k - 1]:
                    alive[i, k:] = False
                    break
                norm_x = np.linalg.norm(latent[i, k])
                p_drop = 1.0 - np.exp(-0.1 * norm_x)
                if rng.random() < p_drop:
                    alive[i, k:] = False
                    break

    return CohortData(
        person_ids=np.arange(N),
        baseline_ages=baseline_ages,
        visit_ages=visit_ages,
        latent_states=latent,
        observed=observed,
        alive=alive,
        medicated=medicated,
        n_persons=N,
        n_visits=K,
        n_obs=n_obs,
        biomarker_names=obs.biomarker_names,
        cohort_name=obs._cohort,
        metadata={"D_weight": D_weight, "J_weight": J_weight},
    )


def _safe_lyapunov(A, Q):
    Gamma = linalg.solve_continuous_lyapunov(A, -Q)
    Gamma = (Gamma + Gamma.T) / 2
    ev, V = np.linalg.eigh(Gamma)
    ev = np.maximum(ev, 1e-14)
    return V @ np.diag(ev) @ V.T


def _ensure_psd(M, n):
    M = (M + M.T) / 2
    ev, V = np.linalg.eigh(M)
    ev = np.maximum(ev, 0)
    return V @ np.diag(ev) @ V.T


def _compress(x, med_flags, compression=0.7):
    x = x.copy()
    for idx in range(len(med_flags)):
        if med_flags[idx]:
            x[idx] *= compression
    return x


# ======================================================================
# Main
# ======================================================================

def main():
    results_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "dj_primacy_mechanistic.json")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_total = time.time()

    model = HDRMechanisticModel(age=65)
    obs = ObservationModel("ELSA_3axis", axes=model.AXES)

    CONDITIONS = [
        ("clean",        False, False),
        ("survivorship", True,  False),
        ("medication",   False, True),
        ("both",         True,  True),
    ]

    output = {"timestamp": timestamp, "regimes": {}}

    for regime_name, D_w, J_w in REGIMES:
        print(f"\n=== Regime: {regime_name} (D={D_w}, J={J_w}) ===")
        regime_results = {}

        for cond_name, surv, med in CONDITIONS:
            seed = SEED_BASE + hash((regime_name, cond_name)) % 10000
            print(f"  Condition: {cond_name} ...", end=" ", flush=True)

            t0 = time.time()
            data = generate_regime_cohort(
                model, obs, D_w, J_w,
                n_persons=N_PERSONS, seed=seed,
                survivorship=surv, medication=med,
            )
            pipe = Tier1Pipeline(data)
            primacy = pipe.compute_primacy_ratio()
            gamma_c = pipe.compute_gamma_change()
            elapsed = time.time() - t0

            # Compute Pi slope across age strata
            if len(primacy) >= 2:
                ages = [r["age_mid"] for r in primacy]
                pis = [r["Pi"] for r in primacy]
                pi_slope = (pis[-1] - pis[0]) / (ages[-1] - ages[0])
            else:
                pi_slope = float("nan")

            regime_results[cond_name] = {
                "primacy_ratio": primacy,
                "gamma_change": gamma_c,
                "Pi_slope": float(pi_slope),
                "elapsed_s": round(elapsed, 1),
            }

            pi_vals = [r["Pi"] for r in primacy]
            print(f"Pi slope={pi_slope:.5f}, Pi={[f'{p:.3f}' for p in pi_vals]}, {elapsed:.1f}s")

        output["regimes"][regime_name] = regime_results

    # Regime separation check: Pi_slope should vary systematically
    # pure-D should have lowest Pi slope; pure-J should have highest
    clean_slopes = {}
    for rname, rdata in output["regimes"].items():
        if "clean" in rdata:
            clean_slopes[rname] = rdata["clean"]["Pi_slope"]

    output["regime_separation"] = clean_slopes
    if len(clean_slopes) >= 2:
        slope_order = sorted(clean_slopes.items(), key=lambda x: x[1])
        output["regime_separation_ordered"] = [
            {"regime": k, "Pi_slope": v} for k, v in slope_order
        ]
        # Check: pure_J should have higher slope than pure_D
        output["regime_separation_clear"] = (
            clean_slopes.get("pure_J", 0) > clean_slopes.get("pure_D", 0)
        )
    else:
        output["regime_separation_clear"] = False

    total_elapsed = time.time() - t_total
    output["total_elapsed_s"] = round(total_elapsed, 1)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nJSON: {json_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s")

    # Print regime separation
    print("\n=== REGIME SEPARATION (clean condition, Pi slope) ===")
    for k, v in sorted(clean_slopes.items(), key=lambda x: x[1]):
        print(f"  {k:12s}: Pi_slope = {v:+.6f}")
    print(f"  Clear separation (pure_J > pure_D): {output['regime_separation_clear']}")


if __name__ == "__main__":
    main()
