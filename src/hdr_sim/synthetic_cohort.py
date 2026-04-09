"""Generate synthetic longitudinal cohort data from the HDR model.

Produces data matching the structure of ELSA (or other cohorts):
N persons, each with baseline age, followed through n_visits at
regular intervals, with optional survivorship bias and medication
effects.

Operates on the fast subsystem (7 axes by default).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from .mechanistic_model import HDRMechanisticModel
from .observation_model import ObservationModel


@dataclass
class CohortData:
    """Container for synthetic cohort data."""

    person_ids: np.ndarray        # (N,)
    baseline_ages: np.ndarray     # (N,)
    visit_ages: np.ndarray        # (N, n_visits)
    latent_states: np.ndarray     # (N, n_visits, n_fast)
    observed: np.ndarray          # (N, n_visits, n_obs)
    alive: np.ndarray             # (N, n_visits) bool
    medicated: np.ndarray         # (N, n_fast) bool — per-axis medication flag
    n_persons: int = 0
    n_visits: int = 0
    n_obs: int = 0
    biomarker_names: list = field(default_factory=list)
    cohort_name: str = ""
    metadata: dict = field(default_factory=dict)


class SyntheticCohort:
    """Generate synthetic longitudinal cohort data from the HDR model.

    Operates on the fast subsystem.
    """

    # Default medication prevalence (matching ELSA demographics)
    DEFAULT_MED_PREVALENCE = {
        "I": 0.10,
        "M": 0.30,
        "N": 0.40,
    }

    def __init__(
        self,
        model,
        observation_model,
        n_persons=5000,
        age_range=(50, 90),
        n_visits=4,
        visit_interval_years=4,
        seed=42,
    ):
        """Configure cohort parameters."""
        self._model = model
        self._obs = observation_model
        self._n_persons = n_persons
        self._age_lo, self._age_hi = age_range
        self._n_visits = n_visits
        self._visit_interval = visit_interval_years
        self._seed = seed
        self._n_fast = model.n
        self._fast_axis_idx = model._fast_axis_idx

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, survivorship=False, medication=False):
        """Generate the full synthetic cohort on the fast subsystem.

        Returns: CohortData
        """
        rng = np.random.default_rng(self._seed)
        N = self._n_persons
        K = self._n_visits
        n_obs = self._obs.n_obs
        n_fast = self._n_fast

        baseline_ages = rng.uniform(self._age_lo, self._age_hi, size=N)

        visit_ages = np.zeros((N, K))
        for k in range(K):
            visit_ages[:, k] = baseline_ages + k * self._visit_interval

        latent = np.zeros((N, K, n_fast))
        observed = np.zeros((N, K, n_obs))
        alive = np.ones((N, K), dtype=bool)
        medicated = np.zeros((N, n_fast), dtype=bool)

        if medication:
            medicated = self._assign_medication(N, rng)

        Q = 0.01 * np.eye(n_fast)

        for i in range(N):
            age_0 = visit_ages[i, 0]
            self._model.set_age(age_0)
            A_0 = self._model.A
            Gamma_0 = self._safe_lyapunov(A_0, Q)

            # Draw from stationary around equilibrium shift
            x_eq = self._model.compute_equilibrium_shift()
            x = rng.multivariate_normal(x_eq, Gamma_0)

            if medication:
                x = self._apply_med_compression(x, medicated[i])

            latent[i, 0] = x
            observed[i, 0] = self._obs.observe(x, seed=rng)

            if survivorship:
                age_frac = max(0, (baseline_ages[i] - self._age_lo)
                               / max(self._age_hi - self._age_lo, 1))
                y_proj = self._obs.project(x)
                norm_y = np.linalg.norm(y_proj)
                cumulative_hazard = 0.3 * norm_y * (1 + 4 * age_frac ** 2)
                p_survive = np.exp(-cumulative_hazard)
                if rng.random() > p_survive:
                    alive[i, :] = False
                    continue

            for k in range(1, K):
                if not alive[i, k - 1]:
                    alive[i, k] = False
                    continue

                age_k = visit_ages[i, k]
                self._model.set_age(age_k)
                A_k = self._model.A
                x_eq_k = self._model.compute_equilibrium_shift()

                dt_days = self._visit_interval * 365.25
                Phi = linalg.expm(A_k * dt_days)
                Gamma_k = self._safe_lyapunov(A_k, Q)
                Sigma_eta = Gamma_k - Phi @ Gamma_k @ Phi.T
                Sigma_eta = self._ensure_psd(Sigma_eta, n_fast)

                L_eta = linalg.cholesky(
                    Sigma_eta + 1e-14 * np.eye(n_fast), lower=True
                )
                eta = L_eta @ rng.standard_normal(n_fast)
                # Transition centred on equilibrium shift
                x = Phi @ (latent[i, k - 1] - x_eq_k) + x_eq_k + eta

                if medication:
                    x = self._apply_med_compression(x, medicated[i])

                latent[i, k] = x
                observed[i, k] = self._obs.observe(x, seed=rng)

            if survivorship:
                alive[i] = self._survivorship_mask(latent[i], alive[i], rng)

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
            biomarker_names=self._obs.biomarker_names,
            cohort_name=self._obs._cohort,
            metadata={
                "age_range": (self._age_lo, self._age_hi),
                "visit_interval_years": self._visit_interval,
                "survivorship": survivorship,
                "medication": medication,
                "seed": self._seed,
                "n_fast_axes": n_fast,
                "fast_axes": self._model.FAST_AXES,
            },
        )

    # ------------------------------------------------------------------
    # Survivorship
    # ------------------------------------------------------------------

    def apply_survivorship(self, data, hazard_scale=0.1):
        """Remove individuals proportional to their state magnitude."""
        rng = np.random.default_rng(self._seed + 999)
        for i in range(data.n_persons):
            for k in range(1, data.n_visits):
                if not data.alive[i, k - 1]:
                    data.alive[i, k] = False
                    continue
                norm_x = np.linalg.norm(data.latent_states[i, k])
                p_drop = 1.0 - np.exp(-hazard_scale * norm_x)
                if rng.random() < p_drop:
                    data.alive[i, k:] = False
                    break
        return data

    def apply_medication(self, data, prevalence=None, compression=0.7):
        """Compress variance of medicated individuals' biomarkers."""
        if prevalence is None:
            prevalence = self.DEFAULT_MED_PREVALENCE

        rng = np.random.default_rng(self._seed + 888)
        for axis_name, frac in prevalence.items():
            if axis_name not in self._fast_axis_idx:
                continue
            idx = self._fast_axis_idx[axis_name]
            mask = rng.random(data.n_persons) < frac
            data.medicated[mask, idx] = True

        for i in range(data.n_persons):
            for k in range(data.n_visits):
                if not data.alive[i, k]:
                    continue
                x = data.latent_states[i, k].copy()
                x = self._apply_med_compression(x, data.medicated[i], compression)
                data.observed[i, k] = self._obs.observe(x, seed=rng)

        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _survivorship_mask(self, latent_traj, alive, rng, hazard_scale=0.1):
        """Per-visit dropout based on projected state magnitude."""
        mask = alive.copy()
        for k in range(1, len(mask)):
            if not mask[k - 1]:
                mask[k] = False
                continue
            y = self._obs.project(latent_traj[k])
            norm_y = np.linalg.norm(y)
            p_drop = 1.0 - np.exp(-hazard_scale * norm_y)
            if rng.random() < p_drop:
                mask[k:] = False
                break
        return mask

    def _assign_medication(self, N, rng):
        """Assign per-axis medication flags for fast axes."""
        medicated = np.zeros((N, self._n_fast), dtype=bool)
        for axis_name, frac in self.DEFAULT_MED_PREVALENCE.items():
            if axis_name not in self._fast_axis_idx:
                continue
            idx = self._fast_axis_idx[axis_name]
            medicated[:, idx] = rng.random(N) < frac
        return medicated

    @staticmethod
    def _apply_med_compression(x, med_flags, compression=0.7):
        """Compress state components on medicated axes."""
        x = x.copy()
        for idx in range(len(med_flags)):
            if med_flags[idx]:
                x[idx] *= compression
        return x

    @staticmethod
    def _safe_lyapunov(A, Q):
        """Solve Lyapunov and ensure PSD."""
        Gamma = linalg.solve_continuous_lyapunov(A, -Q)
        Gamma = (Gamma + Gamma.T) / 2
        eigvals, V = np.linalg.eigh(Gamma)
        eigvals = np.maximum(eigvals, 1e-14)
        return V @ np.diag(eigvals) @ V.T

    @staticmethod
    def _ensure_psd(M, n):
        """Symmetrise and clip negative eigenvalues."""
        M = (M + M.T) / 2
        eigvals, V = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 0)
        return V @ np.diag(eigvals) @ V.T
