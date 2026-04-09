"""Apply evidence-based interventions to the HDR dynamical system.

Each intervention modifies specific J entries and/or tau values,
producing a new A matrix that represents the treated dynamics.

The intervention_library.json from the mechanistic export provides
molecular-level detail for each intervention: which J entries are
modified, by how much (delta_J_fraction), with what evidence level,
and verified against ChEMBL.
"""

import copy
import json
import os

import numpy as np
from scipy import linalg

from .mechanistic_model import HDRMechanisticModel, _spectral_abscissa


class InterventionModel:
    """Apply evidence-based interventions to the HDR dynamical system.

    Each intervention modifies specific J entries and/or tau values,
    producing a new A matrix that represents the treated dynamics.
    """

    def __init__(
        self,
        model,
        library_path=None,
    ):
        """Load the intervention library.

        Args:
            model: HDRMechanisticModel instance (used as the baseline).
            library_path: path to intervention_library.json.
        """
        self._model = model

        if library_path is None:
            library_path = os.path.join(
                model._evidence_dir, "intervention_library.json"
            )
        elif not os.path.isabs(library_path):
            library_path = os.path.join(
                model._evidence_dir, os.path.basename(library_path)
            )

        with open(library_path, encoding="utf-8") as f:
            self._library = json.load(f)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_interventions(self):
        """Return all available interventions with metadata.

        Returns: list of dicts with id, name, evidence_level,
                 n_couplings, n_tau, coupling_ids.
        """
        result = []
        for iid, spec in self._library.items():
            result.append({
                "id": iid,
                "name": spec["name"],
                "evidence_level": spec.get("evidence_level", "unknown"),
                "n_couplings": len(spec.get("affected_couplings", {})),
                "n_tau": len(spec.get("affected_tau", {})),
                "coupling_ids": list(spec.get("affected_couplings", {}).keys()),
                "tau_ids": list(spec.get("affected_tau", {}).keys()),
                "chembl_ids": spec.get("chembl_ids"),
            })
        return result

    # ------------------------------------------------------------------
    # Applying interventions
    # ------------------------------------------------------------------

    def apply(self, intervention_id):
        """Apply a single intervention.

        The intervention modifies:
          J[i,j] -> J[i,j] * (1 + delta_fraction)
          tau_k  -> tau_k  * (1 + delta_fraction)

        Returns: (A_treated, J_treated, D_treated) — modified matrices.
        """
        if intervention_id not in self._library:
            raise ValueError(
                f"Unknown intervention: {intervention_id!r}. "
                f"Available: {list(self._library)}"
            )

        spec = self._library[intervention_id]
        J_new = self._model.J.copy()
        tau_new = self._model.tau.copy()
        axis_idx = self._model._axis_idx

        # Apply J modifications.
        # Convention: delta_J_fraction < 0 always means "reduce the
        # pathological contribution."  For pathological entries (J > 0)
        # this shrinks the coupling.  For protective entries (J < 0)
        # this STRENGTHENS the protective effect (increases |J|).
        for coupling_id, effect in spec.get("affected_couplings", {}).items():
            if coupling_id not in self._model._entries:
                continue
            entry = self._model._entries[coupling_id]
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            if src not in axis_idx or tgt not in axis_idx:
                continue
            j_col = axis_idx[src]
            i_row = axis_idx[tgt]
            delta = effect["delta_J_fraction"]
            if J_new[i_row, j_col] >= 0:
                J_new[i_row, j_col] *= (1.0 + delta)
            else:
                J_new[i_row, j_col] *= (1.0 - delta)

        # Apply tau modifications
        for axis_name, effect in spec.get("affected_tau", {}).items():
            if axis_name not in axis_idx:
                continue
            idx = axis_idx[axis_name]
            delta = effect["delta_tau_fraction"]
            tau_new[idx] *= (1.0 + delta)

        D_new = np.diag(1.0 / tau_new)
        A_new = -D_new + J_new

        return A_new, J_new, D_new

    def apply_combination(self, intervention_ids):
        """Apply multiple interventions simultaneously.

        Modifications are multiplicative: each intervention's delta
        applies to the already-modified matrix.

        Returns: (A_treated, J_treated, D_treated)
        """
        J_new = self._model.J.copy()
        tau_new = self._model.tau.copy()
        axis_idx = self._model._axis_idx

        for iid in intervention_ids:
            if iid not in self._library:
                raise ValueError(f"Unknown intervention: {iid!r}")
            spec = self._library[iid]

            for coupling_id, effect in spec.get("affected_couplings", {}).items():
                if coupling_id not in self._model._entries:
                    continue
                entry = self._model._entries[coupling_id]
                src = entry["source_axis"]
                tgt = entry["target_axis"]
                if src not in axis_idx or tgt not in axis_idx:
                    continue
                j_col = axis_idx[src]
                i_row = axis_idx[tgt]
                delta = effect["delta_J_fraction"]
                if J_new[i_row, j_col] >= 0:
                    J_new[i_row, j_col] *= (1.0 + delta)
                else:
                    J_new[i_row, j_col] *= (1.0 - delta)

            for axis_name, effect in spec.get("affected_tau", {}).items():
                if axis_name not in axis_idx:
                    continue
                idx = axis_idx[axis_name]
                tau_new[idx] *= (1.0 + effect["delta_tau_fraction"])

        D_new = np.diag(1.0 / tau_new)
        A_new = -D_new + J_new

        return A_new, J_new, D_new

    # ------------------------------------------------------------------
    # Effect computation
    # ------------------------------------------------------------------

    def compute_effect(self, intervention_id, metric="spectral_abscissa"):
        """Compute the effect of an intervention on a stability metric.

        Metrics: 'spectral_abscissa', 'dominant_recovery_time',
                 'lambda_max_gamma', 'bifurcation_margin_IM'.

        Returns: dict with baseline, treated, delta, pct_change.
        """
        baseline = self._evaluate_metric(self._model.A, metric)
        A_treated, J_treated, D_treated = self.apply(intervention_id)
        treated = self._evaluate_metric(A_treated, metric, J=J_treated)

        delta = treated - baseline
        pct = delta / abs(baseline) * 100 if baseline != 0 else 0.0

        return {
            "intervention_id": intervention_id,
            "metric": metric,
            "baseline": float(baseline),
            "treated": float(treated),
            "delta": float(delta),
            "pct_change": float(pct),
        }

    def rank_interventions(self, metric="spectral_abscissa", age=70):
        """Rank all interventions by their effect on the chosen metric.

        Returns: sorted list of dicts (most beneficial first).
        """
        original_age = self._model.age
        self._model.set_age(age)

        results = []
        for iid in self._library:
            eff = self.compute_effect(iid, metric=metric)
            eff["name"] = self._library[iid]["name"]
            eff["n_couplings"] = len(
                self._library[iid].get("affected_couplings", {})
            )
            eff["evidence_level"] = self._library[iid].get(
                "evidence_level", "unknown"
            )
            results.append(eff)

        # For spectral_abscissa: more negative delta = better
        # For recovery_time: more negative delta = better (faster recovery)
        # For lambda_max_gamma: more negative delta = better (less variance)
        results.sort(key=lambda x: x["delta"])

        self._model.set_age(original_age)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_metric(self, A, metric, J=None):
        """Evaluate a stability metric from an A matrix."""
        if metric == "spectral_abscissa":
            return _spectral_abscissa(A)
        elif metric == "dominant_recovery_time":
            alpha = _spectral_abscissa(A)
            return 1.0 / abs(alpha) if alpha != 0 else float("inf")
        elif metric == "lambda_max_gamma":
            Q = 0.01 * np.eye(A.shape[0])
            if _spectral_abscissa(A) >= 0:
                return float("inf")
            Gamma = linalg.solve_continuous_lyapunov(A, -Q)
            return float(np.max(np.linalg.eigvalsh(Gamma)))
        elif metric == "bifurcation_margin_IM":
            if J is None:
                J = self._model.J
            tau = self._model.tau
            i_I = self._model._axis_idx["I"]
            i_M = self._model._axis_idx["M"]
            return float(
                1.0 / (tau[i_I] * tau[i_M])
                - J[i_I, i_M] * J[i_M, i_I]
            )
        raise ValueError(f"Unknown metric: {metric}")
