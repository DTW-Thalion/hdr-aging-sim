"""Stress-test the prior architecture against specific scenarios.

Replicates and extends R6 Supplementary Note 6 analysis for the
9-axis mechanistic model parameterised from the enriched evidence base.
"""

import json
import os

import numpy as np
from scipy import linalg

from .estimation import partial_correlations
from .mechanistic_model import HDRMechanisticModel, _spectral_abscissa
from .sensitivity import PriorSensitivityAnalysis


class PriorStressTest:
    """Stress-test the prior architecture against specific scenarios.

    Replicates and extends R6 Supplementary Note 6 analysis.
    """

    def __init__(self, model, prior_spec_path=None):
        self._model = model
        self._n = model._n
        self._axis_idx = model._axis_idx

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

        self._prior_spec_path = prior_spec_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_cohort(self, A, Q, n_subjects, seed=None):
        """Generate cross-sectional samples from stationary distribution."""
        Gamma = linalg.solve_continuous_lyapunov(A, -Q)
        Gamma = (Gamma + Gamma.T) / 2
        eigvals, V = np.linalg.eigh(Gamma)
        eigvals = np.maximum(eigvals, 1e-14)
        Gamma = V @ np.diag(eigvals) @ V.T
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(np.zeros(self._n), Gamma, size=n_subjects)

    @staticmethod
    def _concordance_nonzero(pcor, J_ref):
        """Sign concordance for non-zero J_ref entries only."""
        n = J_ref.shape[0]
        agree = total = 0
        for i in range(n):
            for j in range(n):
                if i != j and J_ref[i, j] != 0:
                    if np.sign(pcor[i, j]) == np.sign(J_ref[i, j]):
                        agree += 1
                    total += 1
        return (agree / total if total > 0 else 0.0), agree, total

    # ------------------------------------------------------------------
    # Concordance tests
    # ------------------------------------------------------------------

    def test_correct_prior(self, n_subjects=2000, age=65, seed=100):
        """Generate synthetic data from compiled J.  Compute partial
        correlations.  Measure sign concordance with compiled J signs.

        Expected: >50% concordance (true signs partially recovered).
        """
        self._model.set_age(age)
        A = self._model.A
        J = self._model.J
        Q = 0.01 * np.eye(self._n)

        X = self._generate_cohort(A, Q, n_subjects, seed=seed)
        pcor = partial_correlations(X)
        conc, agree, total = self._concordance_nonzero(pcor, J)

        return {
            "concordance": float(conc),
            "n_agree": agree,
            "n_total": total,
            "n_subjects": n_subjects,
            "age": age,
            "label": "correct_prior",
        }

    def test_null_prior(self, n_subjects=2000, age=65, seed=200):
        """Random J signs.  Expected: ~50% (chance)."""
        self._model.set_age(age)
        A = self._model.A
        J = self._model.J
        Q = 0.01 * np.eye(self._n)

        X = self._generate_cohort(A, Q, n_subjects, seed=seed)
        pcor = partial_correlations(X)

        # Randomise J signs (keep magnitudes)
        rng = np.random.default_rng(seed + 1)
        J_null = J.copy()
        for i in range(self._n):
            for j in range(self._n):
                if i != j and J_null[i, j] != 0:
                    J_null[i, j] = abs(J_null[i, j]) * rng.choice([-1, 1])

        conc, agree, total = self._concordance_nonzero(pcor, J_null)
        return {
            "concordance": float(conc),
            "n_agree": agree,
            "n_total": total,
            "n_subjects": n_subjects,
            "age": age,
            "label": "null_prior",
        }

    def test_adversarial_prior(self, n_subjects=2000, age=65, seed=300):
        """Flip all compiled J signs.  Expected: <50%."""
        self._model.set_age(age)
        A = self._model.A
        J = self._model.J
        Q = 0.01 * np.eye(self._n)

        X = self._generate_cohort(A, Q, n_subjects, seed=seed)
        pcor = partial_correlations(X)

        conc, agree, total = self._concordance_nonzero(pcor, -J)
        return {
            "concordance": float(conc),
            "n_agree": agree,
            "n_total": total,
            "n_subjects": n_subjects,
            "age": age,
            "label": "adversarial_prior",
        }

    # ------------------------------------------------------------------
    # Structural tests
    # ------------------------------------------------------------------

    def test_grade_ablation(self, age=65):
        """Set all low-confidence entries to zero.  Measure delta-alpha
        and delta-Gamma.

        Since the current export has no Grade C entries, this ablates
        non-informative (R6-compilation-only) entries, testing whether
        the mechanistically-validated entries alone produce stable
        dynamics.
        """
        self._model.set_age(age)
        A_full = self._model.A
        alpha_full = _spectral_abscissa(A_full)

        J_ab = self._model.J.copy()
        D = self._model.D
        ablated = []

        for eid, entry in self._model._entries.items():
            grade = entry.get("confidence_grade", "")
            informative = True
            if eid in self._priors:
                informative = self._priors[eid].get("informative", True)

            # Ablate: Grade C (if any) OR non-informative R6-only entries
            if grade == "C" or (not informative):
                src = entry["source_axis"]
                tgt = entry["target_axis"]
                if src in self._axis_idx and tgt in self._axis_idx:
                    i = self._axis_idx[tgt]
                    j = self._axis_idx[src]
                    J_ab[i, j] = 0.0
                    ablated.append(eid)

        A_ab = -D + J_ab
        alpha_ab = _spectral_abscissa(A_ab)
        delta = alpha_ab - alpha_full
        relative = abs(delta) / abs(alpha_full) if alpha_full != 0 else 0.0

        Q = 0.01 * np.eye(self._n)
        Gamma_full = linalg.solve_continuous_lyapunov(A_full, -Q)
        lmg_full = float(np.max(np.linalg.eigvalsh(Gamma_full)))

        if alpha_ab < 0:
            Gamma_ab = linalg.solve_continuous_lyapunov(A_ab, -Q)
            lmg_ab = float(np.max(np.linalg.eigvalsh(Gamma_ab)))
        else:
            lmg_ab = float("inf")

        return {
            "alpha_full": float(alpha_full),
            "alpha_ablated": float(alpha_ab),
            "delta_alpha": float(delta),
            "delta_alpha_relative": float(relative),
            "stable_after_ablation": alpha_ab < 0,
            "n_ablated": len(ablated),
            "ablated_entries": ablated,
            "lambda_max_gamma_full": lmg_full,
            "lambda_max_gamma_ablated": lmg_ab,
        }

    def test_exclusion_impact(self, age=65, excluded_magnitude=0.03):
        """Compare system with vs without the 3 excluded entries.

        Quantify: delta-alpha, delta-lambda_max(Gamma), delta-SWDS.
        Expected: <5% change (confirming exclusions are safe).
        """
        self._model.set_age(age)
        A_wo = self._model.A
        alpha_wo = _spectral_abscissa(A_wo)

        J_w = self._model.J.copy()
        D = self._model.D
        c = self._model.calibration_scalar
        added = []

        for eid, spec in self._priors.items():
            if spec.get("distribution") != "point_mass":
                continue
            if spec.get("value", 0) != 0:
                continue

            # Look up axes from model's excluded entries
            if eid in self._model._excluded:
                entry = self._model._excluded[eid]
                src = entry.get("source_axis", "")
                tgt = entry.get("target_axis", "")
            else:
                parts = eid.split("_")
                if len(parts) < 3:
                    continue
                src, tgt = parts[1], "_".join(parts[2:])

            if src not in self._axis_idx or tgt not in self._axis_idx:
                continue

            i = self._axis_idx[tgt]
            j = self._axis_idx[src]
            J_w[i, j] = c * excluded_magnitude
            added.append(eid)

        A_w = -D + J_w
        alpha_w = _spectral_abscissa(A_w)
        delta = alpha_w - alpha_wo
        relative = abs(delta) / abs(alpha_wo) if alpha_wo != 0 else 0.0

        Q = 0.01 * np.eye(self._n)
        Gamma_wo = linalg.solve_continuous_lyapunov(A_wo, -Q)
        lmg_wo = float(np.max(np.linalg.eigvalsh(Gamma_wo)))

        if alpha_w < 0:
            Gamma_w = linalg.solve_continuous_lyapunov(A_w, -Q)
            lmg_w = float(np.max(np.linalg.eigvalsh(Gamma_w)))
        else:
            lmg_w = float("inf")

        # SWDS comparison on synthetic cohort
        rng = np.random.default_rng(42)
        X = rng.multivariate_normal(np.zeros(self._n), Gamma_wo, size=500)
        swds_wo = np.sum((X @ Gamma_wo) * X, axis=1) / np.trace(Gamma_wo)

        if alpha_w < 0:
            swds_w = np.sum((X @ Gamma_w) * X, axis=1) / np.trace(Gamma_w)
            swds_delta = float(np.mean(np.abs(swds_w - swds_wo)))
        else:
            swds_delta = float("inf")

        return {
            "alpha_without": float(alpha_wo),
            "alpha_with": float(alpha_w),
            "delta_alpha": float(delta),
            "delta_alpha_relative": float(relative),
            "stable_with_exclusions": alpha_w < 0,
            "n_added": len(added),
            "added_entries": added,
            "lambda_max_gamma_without": lmg_wo,
            "lambda_max_gamma_with": lmg_w,
            "mean_abs_delta_swds": swds_delta,
            "exclusion_safe": relative < 0.05,
        }

    def test_decomposition_vs_uniform(self, n_draws=2000, age=65):
        """Compare narrowed priors (from decomposition confidence
        assessments) vs widened priors over the same range.

        Measure: reduction in alpha uncertainty band.
        """
        model = self._model

        # Narrow (actual) priors
        sa_narrow = PriorSensitivityAnalysis(model, n_draws=n_draws, seed=555)
        mc_narrow = sa_narrow.run_mc(ages=[age])
        alpha_narrow = mc_narrow.alpha[age]

        # Widened priors: double the std of informative entries
        widened = {}
        for eid, spec in self._priors.items():
            widened[eid] = dict(spec)
            if (
                spec.get("distribution") == "truncated_normal"
                and spec.get("informative", False)
            ):
                widened[eid]["std"] = spec["std"] * 2.0

        sa_wide = PriorSensitivityAnalysis(
            model, prior_dict=widened, n_draws=n_draws, seed=555
        )
        mc_wide = sa_wide.run_mc(ages=[age])
        alpha_wide = mc_wide.alpha[age]

        std_n = float(np.std(alpha_narrow))
        std_w = float(np.std(alpha_wide))
        reduction = 1.0 - std_n / std_w if std_w > 0 else 0.0

        ci_n = float(
            np.percentile(alpha_narrow, 95) - np.percentile(alpha_narrow, 5)
        )
        ci_w = float(
            np.percentile(alpha_wide, 95) - np.percentile(alpha_wide, 5)
        )
        ci_red = 1.0 - ci_n / ci_w if ci_w > 0 else 0.0

        n_informative = sum(
            1
            for s in self._priors.values()
            if s.get("distribution") == "truncated_normal"
            and s.get("informative", False)
        )

        return {
            "std_narrow": std_n,
            "std_wide": std_w,
            "std_reduction_pct": float(reduction * 100),
            "ci90_narrow": ci_n,
            "ci90_wide": ci_w,
            "ci90_reduction_pct": float(ci_red * 100),
            "n_informative_entries": n_informative,
            "n_draws": n_draws,
        }
