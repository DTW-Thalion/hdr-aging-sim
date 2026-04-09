"""Monte Carlo sensitivity analysis of system properties to J uncertainty.

Uses the Bayesian prior specification from the mechanistic export
to sample J entries and measure the distribution of key observables.

References:
    R6 Supplementary Note 6 (10,000-draw MC with compiled confidence grades)
    HDR-mechanistic pipeline v3.5 (per-entry prior widths)
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, stats

from .mechanistic_model import HDRMechanisticModel, _spectral_abscissa


@dataclass
class MCResults:
    """Container for Monte Carlo sensitivity analysis results."""

    n_draws: int = 0
    ages: list = field(default_factory=list)
    alpha: dict = field(default_factory=dict)
    recovery_time: dict = field(default_factory=dict)
    damping_ratio: dict = field(default_factory=dict)
    lambda_max_gamma: dict = field(default_factory=dict)
    n_positive_offdiag: dict = field(default_factory=dict)
    beta_IM: dict = field(default_factory=dict)
    stable_fraction: dict = field(default_factory=dict)
    monotone_fraction: float = 0.0

    def summary(self):
        """Return summary statistics as a serialisable dict."""
        out = {
            "n_draws": self.n_draws,
            "ages": self.ages,
            "monotone_fraction": self.monotone_fraction,
        }
        for age in self.ages:
            a = self.alpha[age]
            rt = self.recovery_time[age]
            lmg = self.lambda_max_gamma[age]
            beta = self.beta_IM[age]
            out[f"age_{age}"] = {
                "alpha_mean": float(np.mean(a)),
                "alpha_std": float(np.std(a)),
                "alpha_q05": float(np.percentile(a, 5)),
                "alpha_q50": float(np.median(a)),
                "alpha_q95": float(np.percentile(a, 95)),
                "recovery_mean": float(np.mean(rt)),
                "recovery_std": float(np.std(rt)),
                "damping_mean": float(np.mean(self.damping_ratio[age])),
                "lambda_max_gamma_mean": float(np.nanmean(lmg)),
                "lambda_max_gamma_std": float(np.nanstd(lmg)),
                "n_positive_offdiag_mean": float(
                    np.mean(self.n_positive_offdiag[age])
                ),
                "beta_IM_mean": float(np.mean(beta)),
                "beta_IM_std": float(np.std(beta)),
                "beta_IM_q05": float(np.percentile(beta, 5)),
                "stable_fraction": float(self.stable_fraction[age]),
            }
        return out


class PriorSensitivityAnalysis:
    """Monte Carlo sensitivity analysis of system properties to J uncertainty.

    Uses the Bayesian prior specification from the mechanistic export
    to sample J entries and measure the distribution of key observables.
    """

    def __init__(
        self,
        model,
        prior_spec_path=None,
        prior_dict=None,
        n_draws=10000,
        seed=42,
        prior_scale=1.0,
    ):
        """Load prior distributions for each J entry.

        Args:
            model: HDRMechanisticModel instance.
            prior_spec_path: path to prior_specification.json (resolved
                relative to model._evidence_dir if not absolute).
            prior_dict: dict of priors, used instead of file if provided.
            n_draws: number of Monte Carlo draws.
            seed: random seed.
            prior_scale: multiplier on prior std.  Default 1.0 uses
                the full prior width.  Use <1.0 to narrow the MC
                sampling distribution (e.g. 0.5 halves the width),
                matching R6's confidence-grade perturbation approach.
        """
        self._model = model
        self._n_draws = n_draws
        self._seed = seed
        self._n = model._n
        self._prior_scale = prior_scale

        # Load priors
        if prior_dict is not None:
            self._priors = prior_dict
        else:
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

        # Separate active (truncated_normal) from excluded (point_mass)
        self._active_priors = {
            eid: spec
            for eid, spec in self._priors.items()
            if spec.get("distribution") == "truncated_normal"
        }

        # Pre-compute entry mapping for fast matrix construction
        self._entry_ids = []
        self._entry_rows = []
        self._entry_cols = []
        self._entry_j30 = []
        self._entry_j80 = []

        for eid in self._active_priors:
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

        # Fixed entries: in model but without priors (use nominal values)
        sampled_set = set(self._entry_ids)
        fixed_r, fixed_c, fixed_30, fixed_80 = [], [], [], []
        for eid, entry in model._entries.items():
            if eid in sampled_set:
                continue
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            if src not in model._axis_idx or tgt not in model._axis_idx:
                continue
            fixed_r.append(model._axis_idx[tgt])
            fixed_c.append(model._axis_idx[src])
            fixed_30.append(float(entry.get("J_value_age30", 0.0)))
            fixed_80.append(float(entry.get("J_value_age80", 0.0)))
        self._fixed_rows = np.array(fixed_r, dtype=int)
        self._fixed_cols = np.array(fixed_c, dtype=int)
        self._fixed_j30 = np.array(fixed_30)
        self._fixed_j80 = np.array(fixed_80)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _pre_sample(self, rng, n_draws=None):
        """Sample all entries for all draws.

        Returns: (n_draws, n_entries) array of sampled J_80 values.

        When prior_scale < 1.0, the prior std is multiplied by
        prior_scale before sampling, narrowing the distribution while
        keeping the same mean and sign constraints.
        """
        if n_draws is None:
            n_draws = self._n_draws
        samples = np.zeros((n_draws, self._n_entries))
        for k, eid in enumerate(self._entry_ids):
            spec = self._active_priors[eid]
            mu = spec["mean"]
            sigma = spec["std"] * self._prior_scale
            if sigma < 1e-15:
                samples[:, k] = mu
                continue
            lb = spec.get("lower_bound")
            ub = spec.get("upper_bound")
            a = (lb - mu) / sigma if lb is not None else -100.0
            b = (ub - mu) / sigma if ub is not None else 100.0
            samples[:, k] = stats.truncnorm.rvs(
                a, b, loc=mu, scale=sigma, size=n_draws, random_state=rng
            )
        return samples

    def _build_A(self, j80_vec, age):
        """Build A matrix from a sampled J_80 vector at a given age.

        For each sampled entry, J_30 is scaled proportionally to J_80
        to preserve the age trajectory shape.  The calibration scalar
        is applied uniformly.

        Returns: (A, J) — the dynamics matrix and the coupling matrix.
        """
        f = HDRMechanisticModel._interp_fraction(age)
        tau = self._model._tau_of_age(age)
        c = self._model.calibration_scalar
        n = self._n

        D = np.diag(1.0 / tau)
        J = np.zeros((n, n))

        # Sampled entries
        mask = np.abs(self._entry_j80) > 1e-15
        ratio = np.where(
            mask, j80_vec / np.where(mask, self._entry_j80, 1.0), 1.0
        )
        j30_s = self._entry_j30 * ratio
        j_vals = c * ((1.0 - f) * j30_s + f * j80_vec)
        J[self._entry_rows, self._entry_cols] = j_vals

        # Fixed entries (in model but without priors)
        if len(self._fixed_rows) > 0:
            j_fixed = c * (
                (1.0 - f) * self._fixed_j30 + f * self._fixed_j80
            )
            J[self._fixed_rows, self._fixed_cols] = j_fixed

        return -D + J, J

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def run_mc(self, ages=None):
        """Monte Carlo analysis across age strata.

        For each draw:
        1. Sample each active J entry from its truncated-normal prior
        2. Excluded entries stay at 0
        3. Build A = -D + J(sampled)
        4. Compute observables: alpha, recovery, damping, lambda_max(Gamma)
        5. Compute I-M bifurcation margin
        6. Record stability

        Returns: MCResults object with per-age distributions.
        """
        if ages is None:
            ages = [30, 40, 50, 60, 70, 80]

        rng = np.random.default_rng(self._seed)
        n = self._n_draws
        all_samples = self._pre_sample(rng, n)

        results = MCResults(n_draws=n, ages=list(ages))
        for age in ages:
            results.alpha[age] = np.zeros(n)
            results.recovery_time[age] = np.zeros(n)
            results.damping_ratio[age] = np.zeros(n)
            results.lambda_max_gamma[age] = np.zeros(n)
            results.n_positive_offdiag[age] = np.zeros(n, dtype=int)
            results.beta_IM[age] = np.zeros(n)

        i_I = self._model._axis_idx["I"]
        i_M = self._model._axis_idx["M"]
        Q = 0.01 * np.eye(self._n)
        offdiag = ~np.eye(self._n, dtype=bool)
        monotone_count = 0

        for k in range(n):
            j80_vec = all_samples[k]
            alpha_list = []

            for age in ages:
                A, J_mat = self._build_A(j80_vec, age)
                eigenvalues = linalg.eig(A, right=False)
                re_eigs = np.real(eigenvalues)
                alpha = float(np.max(re_eigs))
                results.alpha[age][k] = alpha
                alpha_list.append(alpha)

                results.recovery_time[age][k] = (
                    1.0 / abs(alpha) if alpha != 0 else np.inf
                )

                idx_dom = int(np.argmax(re_eigs))
                lam1 = eigenvalues[idx_dom]
                results.damping_ratio[age][k] = float(
                    np.abs(np.real(lam1)) / max(np.abs(lam1), 1e-15)
                )

                if alpha < 0:
                    try:
                        Gamma = linalg.solve_continuous_lyapunov(A, -Q)
                        results.lambda_max_gamma[age][k] = float(
                            np.max(np.linalg.eigvalsh(Gamma))
                        )
                    except Exception:
                        results.lambda_max_gamma[age][k] = np.nan
                else:
                    results.lambda_max_gamma[age][k] = np.nan

                results.n_positive_offdiag[age][k] = int(
                    np.sum(J_mat[offdiag] > 0)
                )

                tau = self._model._tau_of_age(age)
                results.beta_IM[age][k] = float(
                    1.0 / (tau[i_I] * tau[i_M])
                    - J_mat[i_I, i_M] * J_mat[i_M, i_I]
                )

            # Monotone α ordering (allow tiny numerical noise)
            if all(
                alpha_list[i + 1] >= alpha_list[i] - 1e-14
                for i in range(len(alpha_list) - 1)
            ):
                monotone_count += 1

        for age in ages:
            results.stable_fraction[age] = float(
                np.mean(results.alpha[age] < 0)
            )
        results.monotone_fraction = monotone_count / n
        return results

    # ------------------------------------------------------------------
    # One-at-a-time sensitivity
    # ------------------------------------------------------------------

    def entry_sensitivity(self, target="spectral_abscissa", age=65):
        """One-at-a-time sensitivity analysis.

        For each J entry: perturb by +/- 1 prior_width, hold all others
        at nominal.  Measure change in target quantity.

        Targets: 'spectral_abscissa', 'lambda_max_gamma', 'bifurcation_margin'.

        Returns: ranked list of dicts (coupling_id, importance, ...).
        """
        nominal_vec = np.array(
            [self._active_priors[eid]["mean"] for eid in self._entry_ids]
        )
        A_nom, J_nom = self._build_A(nominal_vec, age)
        Q = 0.01 * np.eye(self._n)

        def _evaluate(A, J_mat):
            if target == "spectral_abscissa":
                return _spectral_abscissa(A)
            elif target == "lambda_max_gamma":
                if _spectral_abscissa(A) >= 0:
                    return np.nan
                G = linalg.solve_continuous_lyapunov(A, -Q)
                return float(np.max(np.linalg.eigvalsh(G)))
            elif target == "bifurcation_margin":
                i_I = self._model._axis_idx["I"]
                i_M = self._model._axis_idx["M"]
                tau = self._model._tau_of_age(age)
                return float(
                    1.0 / (tau[i_I] * tau[i_M])
                    - J_mat[i_I, i_M] * J_mat[i_M, i_I]
                )
            raise ValueError(f"Unknown target: {target}")

        val_nom = _evaluate(A_nom, J_nom)
        sensitivities = []

        for k, eid in enumerate(self._entry_ids):
            spec = self._active_priors[eid]
            sigma = spec["std"]
            lb = spec.get("lower_bound")
            ub = spec.get("upper_bound")

            vec_plus = nominal_vec.copy()
            val_p = spec["mean"] + sigma
            if lb is not None:
                val_p = max(val_p, lb)
            if ub is not None:
                val_p = min(val_p, ub)
            vec_plus[k] = val_p

            vec_minus = nominal_vec.copy()
            val_m = spec["mean"] - sigma
            if lb is not None:
                val_m = max(val_m, lb)
            if ub is not None:
                val_m = min(val_m, ub)
            vec_minus[k] = val_m

            A_p, J_p = self._build_A(vec_plus, age)
            A_m, J_m = self._build_A(vec_minus, age)
            y_p = _evaluate(A_p, J_p)
            y_m = _evaluate(A_m, J_m)

            delta = y_p - y_m
            actual_step = val_p - val_m
            deriv = delta / actual_step if abs(actual_step) > 1e-15 else 0.0
            importance = abs(deriv) * sigma

            sensitivities.append(
                {
                    "coupling_id": eid,
                    "importance": float(importance),
                    "sensitivity": float(abs(delta)),
                    "derivative": float(deriv),
                    "delta_plus": float(y_p - val_nom),
                    "delta_minus": float(y_m - val_nom),
                    "prior_std": float(sigma),
                }
            )

        sensitivities.sort(key=lambda x: x["importance"], reverse=True)
        return sensitivities

    # ------------------------------------------------------------------
    # Sobol-like variance decomposition
    # ------------------------------------------------------------------

    def sobol_indices(self, age=65, n_samples=4096):
        """First-order and total Sobol sensitivity indices for alpha and
        lambda_max(Gamma).

        Uses a simple variance-based decomposition: for each entry, fix it
        at nominal while sampling all others, to estimate its contribution
        to output variance.

        Falls back to SALib (Saltelli sampling) if available.

        Returns: dict of entry_id -> {S1_alpha, ST_alpha, ...}
        """
        try:
            from SALib.sample import saltelli as _  # noqa: F401

            return self._sobol_salib(age, n_samples)
        except ImportError:
            pass
        return self._sobol_variance(age, n_samples)

    def _sobol_variance(self, age, n_samples):
        """Variance-based decomposition (fallback when SALib is absent)."""
        rng = np.random.default_rng(self._seed + 7777)
        Q = 0.01 * np.eye(self._n)

        # Full MC for baseline variance
        all_s = self._pre_sample(rng, n_samples)
        alpha_full = np.zeros(n_samples)
        lmg_full = np.zeros(n_samples)

        for k in range(n_samples):
            A, _ = self._build_A(all_s[k], age)
            alpha_full[k] = _spectral_abscissa(A)
            if alpha_full[k] < 0:
                try:
                    G = linalg.solve_continuous_lyapunov(A, -Q)
                    lmg_full[k] = float(np.max(np.linalg.eigvalsh(G)))
                except Exception:
                    lmg_full[k] = np.nan
            else:
                lmg_full[k] = np.nan

        var_alpha = float(np.var(alpha_full))
        valid_lmg = lmg_full[~np.isnan(lmg_full)]
        var_lmg = float(np.var(valid_lmg)) if len(valid_lmg) > 10 else 0.0

        n_per = min(n_samples, 512)
        nominal_vec = np.array(
            [self._active_priors[eid]["mean"] for eid in self._entry_ids]
        )

        results = {}
        for idx, eid in enumerate(self._entry_ids):
            rng_e = np.random.default_rng(
                self._seed + abs(hash(eid)) % (2**31)
            )
            samples_e = self._pre_sample(rng_e, n_per)
            samples_e[:, idx] = nominal_vec[idx]

            alpha_e = np.zeros(n_per)
            lmg_e = np.zeros(n_per)

            for k in range(n_per):
                A, _ = self._build_A(samples_e[k], age)
                alpha_e[k] = _spectral_abscissa(A)
                if alpha_e[k] < 0:
                    try:
                        G = linalg.solve_continuous_lyapunov(A, -Q)
                        lmg_e[k] = float(np.max(np.linalg.eigvalsh(G)))
                    except Exception:
                        lmg_e[k] = np.nan
                else:
                    lmg_e[k] = np.nan

            var_a_f = float(np.var(alpha_e))
            S1_a = max(0.0, 1.0 - var_a_f / var_alpha) if var_alpha > 1e-20 else 0.0

            valid_e = lmg_e[~np.isnan(lmg_e)]
            S1_l = (
                max(0.0, 1.0 - float(np.var(valid_e)) / var_lmg)
                if var_lmg > 1e-20 and len(valid_e) > 5
                else 0.0
            )

            results[eid] = {
                "S1_alpha": float(S1_a),
                "ST_alpha": float(S1_a),
                "S1_lambda_max_gamma": float(S1_l),
                "ST_lambda_max_gamma": float(S1_l),
            }

        return results

    def _sobol_salib(self, age, n_samples):
        """SALib-based Sobol analysis."""
        from SALib.sample import saltelli
        from SALib.analyze import sobol as sobol_analyze

        entry_ids = list(self._entry_ids)
        bounds = []
        for eid in entry_ids:
            spec = self._active_priors[eid]
            lb = (
                spec["lower_bound"]
                if spec.get("lower_bound") is not None
                else spec["mean"] - 4 * spec["std"]
            )
            ub = (
                spec["upper_bound"]
                if spec.get("upper_bound") is not None
                else spec["mean"] + 4 * spec["std"]
            )
            bounds.append([lb, ub])

        problem = {
            "num_vars": len(entry_ids),
            "names": entry_ids,
            "bounds": bounds,
        }
        param_values = saltelli.sample(
            problem, n_samples, calc_second_order=False
        )

        Q = 0.01 * np.eye(self._n)
        alpha_vals = np.zeros(len(param_values))
        lmg_vals = np.zeros(len(param_values))

        for k, params in enumerate(param_values):
            A, _ = self._build_A(params, age)
            alpha_vals[k] = _spectral_abscissa(A)
            if alpha_vals[k] < 0:
                try:
                    G = linalg.solve_continuous_lyapunov(A, -Q)
                    lmg_vals[k] = float(np.max(np.linalg.eigvalsh(G)))
                except Exception:
                    lmg_vals[k] = np.nan
            else:
                lmg_vals[k] = np.nan

        Si_alpha = sobol_analyze.analyze(
            problem, alpha_vals, calc_second_order=False
        )

        valid = ~np.isnan(lmg_vals)
        Si_lmg = None
        if np.sum(valid) > 0.8 * len(lmg_vals):
            lmg_vals[~valid] = float(np.nanmean(lmg_vals))
            Si_lmg = sobol_analyze.analyze(
                problem, lmg_vals, calc_second_order=False
            )

        results = {}
        for i, eid in enumerate(entry_ids):
            results[eid] = {
                "S1_alpha": float(Si_alpha["S1"][i]),
                "ST_alpha": float(Si_alpha["ST"][i]),
                "S1_lambda_max_gamma": (
                    float(Si_lmg["S1"][i]) if Si_lmg else 0.0
                ),
                "ST_lambda_max_gamma": (
                    float(Si_lmg["ST"][i]) if Si_lmg else 0.0
                ),
            }

        return results
