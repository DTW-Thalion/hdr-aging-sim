"""In silico clinical trial simulator.

Generates parallel control and treatment cohorts, applies
interventions to the treatment arm, and measures outcome differences.

Supports simple parallel-group RCTs and 2^k factorial designs,
including the R6 proposed 2x2x2 trial (Colchicine x Exercise x
Circadian hygiene).
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy import linalg

from .intervention import InterventionModel
from .mechanistic_model import HDRMechanisticModel, _spectral_abscissa
from .observation_model import ObservationModel


@dataclass
class ArmResult:
    """Results for a single trial arm."""
    label: str = ""
    active_interventions: list = field(default_factory=list)
    n_persons: int = 0
    alpha: float = 0.0
    recovery_time: float = 0.0
    lambda_max_gamma: float = 0.0
    swds_mean: float = 0.0
    swds_std: float = 0.0
    trajectory_norm_mean: float = 0.0


@dataclass
class TrialResults:
    """Results from a parallel-group RCT."""
    arms: list = field(default_factory=list)
    delta_alpha: float = 0.0
    delta_swds: float = 0.0
    delta_lambda_max_gamma: float = 0.0


@dataclass
class FactorialResults:
    """Results from a factorial trial."""
    arms: list = field(default_factory=list)
    main_effects: dict = field(default_factory=dict)
    interactions_2way: dict = field(default_factory=dict)
    interactions_3way: dict = field(default_factory=dict)


class TrialSimulator:
    """In silico clinical trial simulator.

    Generates parallel control and treatment cohorts, applies
    interventions to the treatment arm, and measures outcome differences.
    """

    def __init__(
        self,
        model,
        intervention_model,
        observation_model,
        n_per_arm=500,
        age_range=(65, 80),
    ):
        self._model = model
        self._intv = intervention_model
        self._obs = observation_model
        self._n_per_arm = n_per_arm
        self._age_lo, self._age_hi = age_range

    # ------------------------------------------------------------------
    # Simple RCT
    # ------------------------------------------------------------------

    def run_rct(
        self,
        intervention_id,
        duration_years=2,
        visit_interval_months=3,
        seed=42,
    ):
        """Simulate a simple parallel-group RCT.

        1. Generate matched control and treatment cohorts
        2. Treatment arm: apply intervention -> modified A
        3. Simulate both arms for duration
        4. Compute: delta-alpha, delta-SWDS, delta-lambda_max(Gamma)

        Returns: TrialResults
        """
        A_ctrl = self._model.A
        A_treat, _, _ = self._intv.apply(intervention_id)

        ctrl = self._simulate_arm(
            "control", [], A_ctrl, duration_years,
            visit_interval_months, seed,
        )
        treat = self._simulate_arm(
            intervention_id, [intervention_id], A_treat,
            duration_years, visit_interval_months, seed + 1,
        )

        return TrialResults(
            arms=[ctrl, treat],
            delta_alpha=treat.alpha - ctrl.alpha,
            delta_swds=treat.swds_mean - ctrl.swds_mean,
            delta_lambda_max_gamma=(
                treat.lambda_max_gamma - ctrl.lambda_max_gamma
            ),
        )

    # ------------------------------------------------------------------
    # Factorial design
    # ------------------------------------------------------------------

    def run_factorial(
        self,
        intervention_ids,
        duration_years=2,
        visit_interval_months=3,
        seed=42,
    ):
        """Run a 2^k factorial trial.

        Generate 2^k arms (all combinations of on/off for each
        intervention).  Compute main effects and interaction terms.

        A positive interaction = synergy (combination > sum of parts).

        Returns: FactorialResults
        """
        k = len(intervention_ids)
        arms = []

        # Generate all 2^k combinations
        for bits in product([0, 1], repeat=k):
            active = [
                intervention_ids[i]
                for i in range(k)
                if bits[i] == 1
            ]
            label = "+".join(active) if active else "control"

            if active:
                A_arm, _, _ = self._intv.apply_combination(active)
            else:
                A_arm = self._model.A

            arm_seed = seed + sum(b * (2 ** i) for i, b in enumerate(bits))
            arm_result = self._simulate_arm(
                label, list(active), A_arm,
                duration_years, visit_interval_months, arm_seed,
            )
            arms.append((bits, arm_result))

        # Build lookup: bits -> swds_mean (primary outcome)
        outcome = {bits: arm.swds_mean for bits, arm in arms}
        alpha_out = {bits: arm.alpha for bits, arm in arms}

        # Control arm
        ctrl_bits = tuple([0] * k)
        ctrl_swds = outcome[ctrl_bits]
        ctrl_alpha = alpha_out[ctrl_bits]

        # Main effects: average effect of turning on factor i
        main_effects = {}
        for i in range(k):
            on_vals, off_vals = [], []
            on_alpha, off_alpha = [], []
            for bits, arm in arms:
                if bits[i] == 1:
                    on_vals.append(arm.swds_mean)
                    on_alpha.append(arm.alpha)
                else:
                    off_vals.append(arm.swds_mean)
                    off_alpha.append(arm.alpha)
            main_effects[intervention_ids[i]] = {
                "delta_swds": float(np.mean(on_vals) - np.mean(off_vals)),
                "delta_alpha": float(np.mean(on_alpha) - np.mean(off_alpha)),
            }

        # 2-way interactions
        interactions_2way = {}
        for i in range(k):
            for j in range(i + 1, k):
                # Interaction = effect of both - effect of i alone - effect of j alone
                # In factorial notation: AB - A - B + control
                iid_i = intervention_ids[i]
                iid_j = intervention_ids[j]

                # Find arms
                both_on = [
                    arm.swds_mean for bits, arm in arms
                    if bits[i] == 1 and bits[j] == 1
                ]
                i_only = [
                    arm.swds_mean for bits, arm in arms
                    if bits[i] == 1 and bits[j] == 0
                ]
                j_only = [
                    arm.swds_mean for bits, arm in arms
                    if bits[i] == 0 and bits[j] == 1
                ]
                neither = [
                    arm.swds_mean for bits, arm in arms
                    if bits[i] == 0 and bits[j] == 0
                ]

                interaction = (
                    np.mean(both_on) - np.mean(i_only)
                    - np.mean(j_only) + np.mean(neither)
                )
                interactions_2way[f"{iid_i}*{iid_j}"] = {
                    "interaction_swds": float(interaction),
                    "synergistic": interaction < 0,  # lower SWDS = better
                }

        # 3-way interaction (if k >= 3)
        interactions_3way = {}
        if k >= 3:
            for i in range(k):
                for j in range(i + 1, k):
                    for m in range(j + 1, k):
                        # 3-way = ABC - AB - AC - BC + A + B + C - control
                        vals = {}
                        for bits, arm in arms:
                            key = (bits[i], bits[j], bits[m])
                            vals.setdefault(key, []).append(arm.swds_mean)

                        def _m(key):
                            return np.mean(vals.get(key, [0]))

                        int3 = (
                            _m((1, 1, 1)) - _m((1, 1, 0))
                            - _m((1, 0, 1)) - _m((0, 1, 1))
                            + _m((1, 0, 0)) + _m((0, 1, 0))
                            + _m((0, 0, 1)) - _m((0, 0, 0))
                        )
                        key3 = (
                            f"{intervention_ids[i]}*"
                            f"{intervention_ids[j]}*"
                            f"{intervention_ids[m]}"
                        )
                        interactions_3way[key3] = {
                            "interaction_swds": float(int3),
                            "synergistic": int3 < 0,
                        }

        return FactorialResults(
            arms=[arm for _, arm in arms],
            main_effects=main_effects,
            interactions_2way=interactions_2way,
            interactions_3way=interactions_3way,
        )

    # ------------------------------------------------------------------
    # R6 proposed design
    # ------------------------------------------------------------------

    def replicate_r6_design(self, seed=42):
        """Replicate the R6 proposed 2x2x2 factorial.

        Colchicine x Exercise x Circadian hygiene.
        Population: ages 65-80, Fried pre-frail criteria simulated.
        Duration: 2 years, quarterly visits.

        Returns: FactorialResults
        """
        return self.run_factorial(
            intervention_ids=["colchicine", "exercise_resistance",
                              "circadian_hygiene"],
            duration_years=2,
            visit_interval_months=3,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Arm simulation
    # ------------------------------------------------------------------

    def _simulate_arm(
        self, label, active_ids, A, duration_years,
        visit_interval_months, seed,
    ):
        """Simulate one arm of a trial and compute summary outcomes."""
        rng = np.random.default_rng(seed)
        N = self._n_per_arm
        n_latent = A.shape[0]
        n_obs = self._obs.n_obs
        Q = 0.01 * np.eye(n_latent)

        # Structural metrics from A
        alpha = _spectral_abscissa(A)
        recovery = 1.0 / abs(alpha) if alpha != 0 else float("inf")

        if alpha < 0:
            Gamma = linalg.solve_continuous_lyapunov(A, -Q)
            Gamma = (Gamma + Gamma.T) / 2
            ev, V = np.linalg.eigh(Gamma)
            ev = np.maximum(ev, 1e-14)
            Gamma = V @ np.diag(ev) @ V.T
            lmg = float(np.max(ev))
        else:
            Gamma = Q.copy()
            lmg = float("inf")

        # Simulate N persons from the stationary distribution
        # (cross-sectional snapshot at end of trial)
        dt_days = visit_interval_months * 30.44
        n_visits = int(duration_years * 12 / visit_interval_months)
        Phi = linalg.expm(A * dt_days) if alpha < 0 else np.eye(n_latent)

        # Discrete noise
        if alpha < 0:
            Sigma_eta = Gamma - Phi @ Gamma @ Phi.T
            Sigma_eta = (Sigma_eta + Sigma_eta.T) / 2
            se_ev, se_V = np.linalg.eigh(Sigma_eta)
            se_ev = np.maximum(se_ev, 0)
            Sigma_eta = se_V @ np.diag(se_ev) @ se_V.T
            L_eta = linalg.cholesky(
                Sigma_eta + 1e-14 * np.eye(n_latent), lower=True
            )
        else:
            L_eta = 0.1 * np.eye(n_latent)

        # Sample baseline ages and run per-person trajectories
        ages = rng.uniform(self._age_lo, self._age_hi, size=N)
        final_states = np.zeros((N, n_latent))
        traj_norms = np.zeros(N)

        for i in range(N):
            x = rng.multivariate_normal(np.zeros(n_latent), Gamma)
            for _ in range(n_visits):
                eta = L_eta @ rng.standard_normal(n_latent)
                x = Phi @ x + eta
            final_states[i] = x
            traj_norms[i] = np.linalg.norm(x)

        # Project to observation space and compute SWDS
        Y = self._obs.project(final_states)
        Gamma_obs = np.cov(Y.T) if N > 1 else np.eye(n_obs)
        trace_ref = max(np.trace(Gamma_obs), 1e-10)
        swds = np.sum((Y @ Gamma_obs) * Y, axis=1) / trace_ref

        return ArmResult(
            label=label,
            active_interventions=list(active_ids),
            n_persons=N,
            alpha=float(alpha),
            recovery_time=float(recovery),
            lambda_max_gamma=float(lmg),
            swds_mean=float(np.mean(swds)),
            swds_std=float(np.std(swds)),
            trajectory_norm_mean=float(np.mean(traj_norms)),
        )
