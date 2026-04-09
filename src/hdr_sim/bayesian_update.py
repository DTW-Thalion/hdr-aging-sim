"""Scaffold for Bayesian updating of J priors from cohort data.

Uses ABC (Approximate Bayesian Computation) for tractability.
The mechanistic priors are the starting point; Tier-1 observables
from cohort data serve as summary statistics for acceptance/rejection.

References:
    R6 SI Note 3 — Identifiability: full A recovery from Gamma alone is
    Tier-3 non-identifiable when the drift graph has directed 2-cycles.
    J is a literature-derived prior, not an empirical estimate.  Fitting
    to cohort data should UPDATE the prior within the identifiable
    subspace, not try to recover J from scratch.

What IS identifiable from Tier-1 data:
    - Off-diagonal signs of Gamma (partial correlations) -> constrain J signs
    - lambda_max(Gamma) trend with age -> constrain stability erosion rate
    - SWDS distribution -> constrain individual-level risk
    - Pi primacy ratio -> constrain the D/J degradation balance
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, stats

from .estimation import partial_correlations
from .mechanistic_model import HDRMechanisticModel, _spectral_abscissa
from .observation_model import ObservationModel
from .tier1_pipeline import Tier1Pipeline


@dataclass
class ABCResults:
    """Container for ABC posterior results."""

    n_proposals: int = 0
    n_accepted: int = 0
    acceptance_rate: float = 0.0
    tolerance: float = 0.0
    # Per-entry posterior: entry_id -> array of accepted values
    posterior_samples: dict = field(default_factory=dict)
    distances: np.ndarray = field(default_factory=lambda: np.array([]))
    # Summary
    posterior_means: dict = field(default_factory=dict)
    posterior_stds: dict = field(default_factory=dict)
    prior_means: dict = field(default_factory=dict)
    prior_stds: dict = field(default_factory=dict)


class BayesianPriorUpdate:
    """Scaffold for Bayesian updating of J priors from cohort data.

    Uses ABC (Approximate Bayesian Computation) for tractability.
    The mechanistic priors are the starting point; Tier-1 observables
    from cohort data serve as summary statistics for acceptance/rejection.
    """

    # Weights for the distance metric (normalised internally)
    STAT_WEIGHTS = {
        "lambda_max_trend_slope": 2.0,
        "off_diag_sign_concordance": 1.5,
        "pi_slope": 1.0,
        "swds_p75": 1.0,
    }

    def __init__(self, model, prior_spec_path=None, observation_model=None):
        self._model = model
        self._n = model._n

        if observation_model is None:
            observation_model = ObservationModel("ELSA_3axis")
        self._obs = observation_model

        if prior_spec_path is None:
            prior_spec_path = os.path.join(
                model._evidence_dir, "prior_specification.json"
            )
        elif not os.path.isabs(prior_spec_path):
            prior_spec_path = os.path.join(
                model._evidence_dir, os.path.basename(prior_spec_path)
            )

        with open(prior_spec_path, encoding="utf-8") as f:
            self._priors = json.load(f)

        # Active entries (truncated-normal priors)
        self._active = {
            eid: spec
            for eid, spec in self._priors.items()
            if spec.get("distribution") == "truncated_normal"
        }

        # Entry mapping (same structure as PriorSensitivityAnalysis)
        self._entry_ids = []
        self._entry_rows = []
        self._entry_cols = []
        self._entry_j30 = []
        self._entry_j80 = []

        for eid in self._active:
            if eid not in model._entries:
                continue
            entry = model._entries[eid]
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            if src not in model._axis_idx or tgt not in model._axis_idx:
                continue
            self._entry_ids.append(eid)
            self._entry_rows.append(model._axis_idx[tgt])
            self._entry_cols.append(model._axis_idx[src])
            self._entry_j30.append(float(entry.get("J_value_age30", 0.0)))
            self._entry_j80.append(float(entry.get("J_value_age80", 0.0)))

        self._entry_rows = np.array(self._entry_rows, dtype=int)
        self._entry_cols = np.array(self._entry_cols, dtype=int)
        self._entry_j30 = np.array(self._entry_j30)
        self._entry_j80 = np.array(self._entry_j80)
        self._n_entries = len(self._entry_ids)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def compute_summary_statistics(self, cohort_data):
        """Extract Tier-1 summary statistics from cohort data.

        Returns: dict with lambda_max_trend, off_diagonal_signs,
                 pi_slope, swds_percentiles.
        """
        pipe = Tier1Pipeline(cohort_data)

        # 1. lambda_max(Gamma_change) trend
        gc = pipe.compute_gamma_change()
        if len(gc) >= 2:
            ages = [r["age_mid"] for r in gc]
            lm = [r["lambda_max"] for r in gc]
            slope = (lm[-1] - lm[0]) / (ages[-1] - ages[0])
        else:
            slope = 0.0

        # 2. Off-diagonal sign concordance with compiled J
        gx = pipe.compute_gamma_cross_sectional()
        if gx:
            # Use the youngest stratum's observed covariance
            youngest = min(gx, key=lambda r: r["age_mid"])
            idx_young = self._strata_indices(cohort_data, youngest)
            if len(idx_young) >= 30:
                Y = cohort_data.observed[idx_young, 0]
                Gamma_obs = np.cov(Y.T)
                J_obs = self._model.J
                # Project J to observation space
                C = self._obs.C
                J_proj = C @ J_obs @ C.T
                conc = self._sign_concordance_matrix(Gamma_obs, J_proj)
            else:
                conc = 0.5
        else:
            conc = 0.5

        # 3. Pi primacy slope
        pr = pipe.compute_primacy_ratio()
        if len(pr) >= 2:
            pi_ages = [r["age_mid"] for r in pr]
            pi_vals = [r["Pi"] for r in pr]
            pi_slope = (pi_vals[-1] - pi_vals[0]) / (pi_ages[-1] - pi_ages[0])
        else:
            pi_slope = 0.0

        # 4. SWDS percentiles
        swds = pipe.compute_swds()
        swds_p75 = 0.0
        if swds:
            all_means = [v["swds_mean"] for v in swds.values()]
            swds_p75 = float(np.percentile(all_means, 75)) if all_means else 0.0

        return {
            "lambda_max_trend_slope": float(slope),
            "off_diag_sign_concordance": float(conc),
            "pi_slope": float(pi_slope),
            "swds_p75": float(swds_p75),
        }

    # ------------------------------------------------------------------
    # Simulate-and-summarise
    # ------------------------------------------------------------------

    def simulate_and_summarise(self, j80_vec, age=65, n_subjects=2000, seed=None):
        """Given a J_80 sample vector, simulate a cohort and summarise.

        1. Build model with sampled J
        2. Generate synthetic cohort (cross-sectional at 4 ages)
        3. Compute summary statistics

        Returns: summary statistics dict
        """
        from .synthetic_cohort import SyntheticCohort, CohortData

        # Build A from sampled J_80 at several ages
        rng = np.random.default_rng(seed)
        Q = 0.01 * np.eye(self._n)
        c = self._model.calibration_scalar

        # Generate a lightweight synthetic cohort
        # (use the model with a temporary J override)
        ages_strata = [(50, 60), (60, 70), (70, 80), (80, 90)]
        n_per = n_subjects // 4
        n_obs = self._obs.n_obs

        all_obs = []
        all_ages = []
        for lo, hi in ages_strata:
            age_mid = (lo + hi) / 2
            A = self._build_A_from_vec(j80_vec, age_mid)
            alpha = _spectral_abscissa(A)
            if alpha >= 0:
                # Unstable — skip
                continue
            Gamma = linalg.solve_continuous_lyapunov(A, -Q)
            Gamma = (Gamma + Gamma.T) / 2
            ev, V = np.linalg.eigh(Gamma)
            ev = np.maximum(ev, 1e-14)
            Gamma = V @ np.diag(ev) @ V.T
            X = rng.multivariate_normal(np.zeros(self._n), Gamma, size=n_per)
            Y = self._obs.observe(X, seed=rng)
            all_obs.append(Y)
            all_ages.extend(
                rng.uniform(lo, hi, size=n_per).tolist()
            )

        if not all_obs:
            return {
                "lambda_max_trend_slope": 0.0,
                "off_diag_sign_concordance": 0.5,
                "pi_slope": 0.0,
                "swds_p75": 0.0,
            }

        Y_all = np.vstack(all_obs)
        ages_arr = np.array(all_ages)
        N = len(ages_arr)

        # Build a minimal CohortData for the Tier-1 pipeline
        data = CohortData(
            person_ids=np.arange(N),
            baseline_ages=ages_arr,
            visit_ages=ages_arr.reshape(N, 1),
            latent_states=np.zeros((N, 1, self._n)),
            observed=Y_all.reshape(N, 1, n_obs),
            alive=np.ones((N, 1), dtype=bool),
            medicated=np.zeros((N, self._n), dtype=bool),
            n_persons=N,
            n_visits=1,
            n_obs=n_obs,
            biomarker_names=self._obs.biomarker_names,
        )

        # For single-visit data, compute cross-sectional metrics
        pipe = Tier1Pipeline(data)
        gx = pipe.compute_gamma_cross_sectional()

        if len(gx) >= 2:
            ages_gx = [r["age_mid"] for r in gx]
            lm_gx = [r["lambda_max"] for r in gx]
            slope = (lm_gx[-1] - lm_gx[0]) / (ages_gx[-1] - ages_gx[0])
        else:
            slope = 0.0

        # Sign concordance from youngest stratum
        conc = 0.5
        if gx:
            youngest = min(gx, key=lambda r: r["age_mid"])
            mask = (ages_arr >= youngest["age_lo"]) & (ages_arr < youngest["age_hi"])
            Y_young = Y_all[mask]
            if len(Y_young) >= 30:
                Gamma_obs = np.cov(Y_young.T)
                C = self._obs.C
                J_proj = C @ self._build_J_from_vec(j80_vec, 55) @ C.T
                conc = self._sign_concordance_matrix(Gamma_obs, J_proj)

        # Primacy from cross-sectional
        pr = pipe.compute_primacy_ratio()
        if len(pr) >= 2:
            pi_slope = (pr[-1]["Pi"] - pr[0]["Pi"]) / (
                pr[-1]["age_mid"] - pr[0]["age_mid"]
            )
        else:
            pi_slope = 0.0

        # SWDS
        swds = pipe.compute_swds()
        swds_p75 = 0.0
        if swds:
            means = [v["swds_mean"] for v in swds.values()]
            swds_p75 = float(np.percentile(means, 75)) if means else 0.0

        return {
            "lambda_max_trend_slope": float(slope),
            "off_diag_sign_concordance": float(conc),
            "pi_slope": float(pi_slope),
            "swds_p75": float(swds_p75),
        }

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def distance(self, summary_observed, summary_simulated):
        """Weighted Euclidean distance between observed and simulated
        summary statistics.  Each statistic is normalised by the
        observed value (or 1.0 if zero) before weighting.
        """
        d2 = 0.0
        for key, weight in self.STAT_WEIGHTS.items():
            obs = summary_observed.get(key, 0.0)
            sim = summary_simulated.get(key, 0.0)
            scale = max(abs(obs), 1e-6)
            d2 += weight * ((obs - sim) / scale) ** 2
        return float(np.sqrt(d2))

    # ------------------------------------------------------------------
    # ABC rejection sampling
    # ------------------------------------------------------------------

    def run_abc(
        self,
        observed_summary,
        n_proposals=100000,
        tolerance_quantile=0.01,
        seed=42,
    ):
        """ABC rejection sampling.

        1. For each proposal, sample J_80 from prior
        2. Simulate and compute summary statistics
        3. Compute distance to observed summary
        4. Accept top tolerance_quantile fraction

        Returns: ABCResults with posterior samples.
        """
        rng = np.random.default_rng(seed)

        # Pre-sample all proposals
        proposals = np.zeros((n_proposals, self._n_entries))
        for k, eid in enumerate(self._entry_ids):
            spec = self._active[eid]
            mu = spec["mean"]
            sigma = spec["std"]
            lb = spec.get("lower_bound")
            ub = spec.get("upper_bound")
            a = (lb - mu) / sigma if lb is not None else -100.0
            b = (ub - mu) / sigma if ub is not None else 100.0
            proposals[:, k] = stats.truncnorm.rvs(
                a, b, loc=mu, scale=sigma, size=n_proposals,
                random_state=rng,
            )

        # Evaluate each proposal
        distances = np.full(n_proposals, np.inf)
        for i in range(n_proposals):
            try:
                summary_sim = self.simulate_and_summarise(
                    proposals[i], n_subjects=500, seed=rng,
                )
                distances[i] = self.distance(observed_summary, summary_sim)
            except Exception:
                pass

        # Accept top quantile
        valid = np.isfinite(distances)
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            return ABCResults(n_proposals=n_proposals)

        threshold = np.percentile(distances[valid], tolerance_quantile * 100)
        accepted = valid & (distances <= threshold)
        n_accepted = int(np.sum(accepted))

        # Build posterior samples per entry
        posterior_samples = {}
        posterior_means = {}
        posterior_stds = {}
        prior_means = {}
        prior_stds = {}

        for k, eid in enumerate(self._entry_ids):
            vals = proposals[accepted, k]
            posterior_samples[eid] = vals
            posterior_means[eid] = float(np.mean(vals)) if len(vals) > 0 else 0.0
            posterior_stds[eid] = float(np.std(vals)) if len(vals) > 0 else 0.0
            prior_means[eid] = self._active[eid]["mean"]
            prior_stds[eid] = self._active[eid]["std"]

        return ABCResults(
            n_proposals=n_proposals,
            n_accepted=n_accepted,
            acceptance_rate=n_accepted / n_proposals,
            tolerance=float(threshold),
            posterior_samples=posterior_samples,
            distances=distances[accepted],
            posterior_means=posterior_means,
            posterior_stds=posterior_stds,
            prior_means=prior_means,
            prior_stds=prior_stds,
        )

    # ------------------------------------------------------------------
    # Sign concordance test (R6 Tests 3-4)
    # ------------------------------------------------------------------

    def sign_concordance_test(self, observed_gamma, n_null=1000, seed=42):
        """Test R6 Tests 3-4: off-diagonal signs of observed Gamma vs
        compiled J.

        Returns: concordance fraction, p-value vs null (random signs).
        """
        C = self._obs.C
        J_proj = C @ self._model.J @ C.T

        conc_obs = self._sign_concordance_matrix(observed_gamma, J_proj)

        # Null distribution: random sign assignments
        rng = np.random.default_rng(seed)
        null_concs = np.zeros(n_null)
        n_obs = observed_gamma.shape[0]
        for i in range(n_null):
            J_null = J_proj.copy()
            for a in range(n_obs):
                for b in range(n_obs):
                    if a != b and J_null[a, b] != 0:
                        J_null[a, b] = abs(J_null[a, b]) * rng.choice([-1, 1])
            null_concs[i] = self._sign_concordance_matrix(
                observed_gamma, J_null
            )

        p_value = float(np.mean(null_concs >= conc_obs))

        return {
            "concordance": float(conc_obs),
            "p_value": float(p_value),
            "null_mean": float(np.mean(null_concs)),
            "null_std": float(np.std(null_concs)),
        }

    # ------------------------------------------------------------------
    # Posterior summary
    # ------------------------------------------------------------------

    def posterior_summary(self, abc_results):
        """Summarise the posterior: per-entry mean, std, credible intervals.
        Compare to prior: which entries tightened? Which shifted?
        """
        summary = []
        for eid in self._entry_ids:
            post_mean = abc_results.posterior_means.get(eid, 0.0)
            post_std = abc_results.posterior_stds.get(eid, 0.0)
            prior_mean = abc_results.prior_means.get(eid, 0.0)
            prior_std = abc_results.prior_stds.get(eid, 0.0)

            vals = abc_results.posterior_samples.get(eid, np.array([]))
            if len(vals) > 0:
                ci_lo = float(np.percentile(vals, 2.5))
                ci_hi = float(np.percentile(vals, 97.5))
            else:
                ci_lo = ci_hi = post_mean

            tightened = post_std < prior_std * 0.9
            shifted = abs(post_mean - prior_mean) > 0.5 * prior_std

            summary.append({
                "entry_id": eid,
                "prior_mean": float(prior_mean),
                "prior_std": float(prior_std),
                "posterior_mean": float(post_mean),
                "posterior_std": float(post_std),
                "ci_95": (ci_lo, ci_hi),
                "tightened": tightened,
                "shifted": shifted,
                "std_ratio": float(post_std / prior_std) if prior_std > 0 else 1.0,
            })

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_A_from_vec(self, j80_vec, age):
        """Build A matrix from a J_80 sample vector at a given age."""
        f = HDRMechanisticModel._interp_fraction(age)
        tau = self._model._tau_of_age(age)
        c = self._model.calibration_scalar
        n = self._n

        D = np.diag(1.0 / tau)
        J = np.zeros((n, n))

        mask = np.abs(self._entry_j80) > 1e-15
        ratio = np.where(
            mask, j80_vec / np.where(mask, self._entry_j80, 1.0), 1.0
        )
        j30_s = self._entry_j30 * ratio
        j_vals = c * ((1.0 - f) * j30_s + f * j80_vec)
        J[self._entry_rows, self._entry_cols] = j_vals

        return -D + J

    def _build_J_from_vec(self, j80_vec, age):
        """Build the J matrix from a J_80 sample vector."""
        f = HDRMechanisticModel._interp_fraction(age)
        c = self._model.calibration_scalar
        n = self._n
        J = np.zeros((n, n))

        mask = np.abs(self._entry_j80) > 1e-15
        ratio = np.where(
            mask, j80_vec / np.where(mask, self._entry_j80, 1.0), 1.0
        )
        j30_s = self._entry_j30 * ratio
        j_vals = c * ((1.0 - f) * j30_s + f * j80_vec)
        J[self._entry_rows, self._entry_cols] = j_vals
        return J

    @staticmethod
    def _sign_concordance_matrix(Gamma, J_ref):
        """Concordance between off-diagonal signs of Gamma and J_ref."""
        n = Gamma.shape[0]
        agree = total = 0
        for i in range(n):
            for j in range(n):
                if i != j and J_ref[i, j] != 0:
                    if np.sign(Gamma[i, j]) == np.sign(J_ref[i, j]):
                        agree += 1
                    total += 1
        return agree / total if total > 0 else 0.5

    def _strata_indices(self, data, stratum_info):
        """Return indices of persons in the given age stratum at visit 0."""
        lo = stratum_info["age_lo"]
        hi = stratum_info["age_hi"]
        mask = (
            (data.visit_ages[:, 0] >= lo)
            & (data.visit_ages[:, 0] < hi)
            & data.alive[:, 0]
        )
        return np.where(mask)[0]
