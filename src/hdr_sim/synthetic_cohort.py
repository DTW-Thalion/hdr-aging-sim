"""Generate synthetic longitudinal cohort data from the HDR model.

Produces data matching the structure of ELSA (or other cohorts):
N persons, each with baseline age, followed through n_visits at
regular intervals, with optional survivorship bias and medication
effects.
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
    latent_states: np.ndarray     # (N, n_visits, 9)
    observed: np.ndarray          # (N, n_visits, n_obs)
    alive: np.ndarray             # (N, n_visits) bool — True if still in cohort
    medicated: np.ndarray         # (N, 9) bool — per-axis medication flag
    n_persons: int = 0
    n_visits: int = 0
    n_obs: int = 0
    biomarker_names: list = field(default_factory=list)
    cohort_name: str = ""
    metadata: dict = field(default_factory=dict)


class SyntheticCohort:
    """Generate synthetic longitudinal cohort data from the HDR model.

    Produces data matching the structure of ELSA (or other cohorts):
    N persons, each with baseline age, followed through n_visits at
    regular intervals, with optional survivorship bias and medication
    effects.
    """

    # Default medication prevalence (matching ELSA demographics)
    DEFAULT_MED_PREVALENCE = {
        "I": 0.10,   # ~10% anti-inflammatory
        "M": 0.30,   # ~30% statins/metformin
        "N": 0.40,   # ~40% antihypertensives
    }

    AXIS_NAMES = ["I", "M", "E", "mito", "P", "C", "N", "F", "B"]

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
        """Configure cohort parameters.

        Defaults match ELSA design: 5000 persons, ages 50-90,
        4 visits, 4-year intervals.
        """
        self._model = model
        self._obs = observation_model
        self._n_persons = n_persons
        self._age_lo, self._age_hi = age_range
        self._n_visits = n_visits
        self._visit_interval = visit_interval_years
        self._seed = seed

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, survivorship=False, medication=False):
        """Generate the full synthetic cohort.

        For each person:
        1. Sample baseline age uniformly from age_range
        2. At each visit, compute age-specific model A(age)
        3. Draw latent state from the discrete-time transition
        4. Apply observation model: y = C @ x + v
        5. If survivorship: apply attrition based on state magnitude
        6. If medication: compress variance of selected axes

        Returns: CohortData
        """
        rng = np.random.default_rng(self._seed)
        N = self._n_persons
        K = self._n_visits
        n_obs = self._obs.n_obs
        n_latent = 9

        # Sample baseline ages
        baseline_ages = rng.uniform(self._age_lo, self._age_hi, size=N)

        # Build visit-age matrix
        visit_ages = np.zeros((N, K))
        for k in range(K):
            visit_ages[:, k] = baseline_ages + k * self._visit_interval

        # Storage
        latent = np.zeros((N, K, n_latent))
        observed = np.zeros((N, K, n_obs))
        alive = np.ones((N, K), dtype=bool)
        medicated = np.zeros((N, n_latent), dtype=bool)

        # Medication assignment (before simulation)
        if medication:
            medicated = self._assign_medication(N, rng)

        # Process noise
        Q = 0.01 * np.eye(n_latent)

        # Generate per person
        for i in range(N):
            # Visit 0: draw from stationary distribution
            age_0 = visit_ages[i, 0]
            self._model.set_age(age_0)
            A_0 = self._model.A
            Gamma_0 = self._safe_lyapunov(A_0, Q)
            x = rng.multivariate_normal(np.zeros(n_latent), Gamma_0)

            if medication:
                x = self._apply_med_compression(x, medicated[i])

            latent[i, 0] = x
            observed[i, 0] = self._obs.observe(x, seed=rng)

            # Baseline survivorship: older individuals with high
            # dysregulation are less likely to have survived to their
            # current age (pre-study selection).  Uses projected state
            # (biomarker-relevant axes) so the selection artefact
            # manifests in the observed covariance.
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

            # Subsequent visits: discrete-time propagation
            for k in range(1, K):
                if not alive[i, k - 1]:
                    alive[i, k] = False
                    continue

                age_k = visit_ages[i, k]
                self._model.set_age(age_k)
                A_k = self._model.A

                dt_days = self._visit_interval * 365.25
                Phi = linalg.expm(A_k * dt_days)
                Gamma_k = self._safe_lyapunov(A_k, Q)
                Sigma_eta = Gamma_k - Phi @ Gamma_k @ Phi.T
                Sigma_eta = self._ensure_psd(Sigma_eta, n_latent)

                L_eta = linalg.cholesky(
                    Sigma_eta + 1e-14 * np.eye(n_latent), lower=True
                )
                eta = L_eta @ rng.standard_normal(n_latent)
                x = Phi @ latent[i, k - 1] + eta

                if medication:
                    x = self._apply_med_compression(x, medicated[i])

                latent[i, k] = x
                observed[i, k] = self._obs.observe(x, seed=rng)

            # Survivorship: mark dropout based on state magnitude
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
            },
        )

    # ------------------------------------------------------------------
    # Survivorship
    # ------------------------------------------------------------------

    def apply_survivorship(self, data, hazard_scale=0.1):
        """Remove individuals proportional to their state magnitude.

        Higher ||x|| -> higher dropout probability per visit.
        Modifies data.alive in place.
        """
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
        """Compress variance of medicated individuals' biomarkers.

        Default prevalence: ~30% statins/metformin, ~40% antihypertensives.
        """
        if prevalence is None:
            prevalence = self.DEFAULT_MED_PREVALENCE

        rng = np.random.default_rng(self._seed + 888)
        for axis_name, frac in prevalence.items():
            idx = self.AXIS_NAMES.index(axis_name)
            mask = rng.random(data.n_persons) < frac
            data.medicated[mask, idx] = True

        # Compress observed biomarkers for medicated axes
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
        """Assign per-axis medication flags."""
        medicated = np.zeros((N, 9), dtype=bool)
        for axis_name, frac in self.DEFAULT_MED_PREVALENCE.items():
            idx = self.AXIS_NAMES.index(axis_name)
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
