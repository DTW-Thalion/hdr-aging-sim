"""Apply evidence-based interventions to the HDR dynamical system.

Each intervention modifies specific J entries and/or tau values,
producing a new A matrix that represents the treated dynamics.

The intervention_library.json from the mechanistic export provides
molecular-level detail for each intervention: which J entries are
modified, by how much (delta_J_fraction), with what evidence level,
and verified against ChEMBL.

Operates on the fast subsystem (7 axes: I, M, mito, P, C, N, F).
Interventions affecting quasi-static axes (E, B) modify the forcing
vector, not the eigenvalue structure.
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
        """Return all available interventions with metadata."""
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
        """Apply a single intervention to the fast subsystem.

        Returns: (A_treated, J_treated, D_treated) — fast-subsystem matrices.
        """
        if intervention_id not in self._library:
            raise ValueError(
                f"Unknown intervention: {intervention_id!r}. "
                f"Available: {list(self._library)}"
            )

        spec = self._library[intervention_id]
        J_new = self._model.J.copy()  # fast subsystem
        tau_new = self._model.tau.copy()  # fast subsystem
        fast_axis_idx = self._model._fast_axis_idx

        for coupling_id, effect in spec.get("affected_couplings", {}).items():
            if coupling_id not in self._model._entries:
                continue
            entry = self._model._entries[coupling_id]
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            # Only modify if both axes are in the fast subsystem
            if src not in fast_axis_idx or tgt not in fast_axis_idx:
                continue
            j_col = fast_axis_idx[src]
            i_row = fast_axis_idx[tgt]
            delta = effect["delta_J_fraction"]
            if J_new[i_row, j_col] >= 0:
                J_new[i_row, j_col] *= (1.0 + delta)
            else:
                J_new[i_row, j_col] *= (1.0 - delta)

        for axis_name, effect in spec.get("affected_tau", {}).items():
            if axis_name not in fast_axis_idx:
                continue
            idx = fast_axis_idx[axis_name]
            delta = effect["delta_tau_fraction"]
            tau_new[idx] *= (1.0 + delta)

        D_new = np.diag(1.0 / tau_new)
        A_new = -D_new + J_new

        return A_new, J_new, D_new

    def apply_combination(self, intervention_ids):
        """Apply multiple interventions simultaneously.

        Returns: (A_treated, J_treated, D_treated) — fast-subsystem matrices.
        """
        J_new = self._model.J.copy()
        tau_new = self._model.tau.copy()
        fast_axis_idx = self._model._fast_axis_idx

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
                if src not in fast_axis_idx or tgt not in fast_axis_idx:
                    continue
                j_col = fast_axis_idx[src]
                i_row = fast_axis_idx[tgt]
                delta = effect["delta_J_fraction"]
                if J_new[i_row, j_col] >= 0:
                    J_new[i_row, j_col] *= (1.0 + delta)
                else:
                    J_new[i_row, j_col] *= (1.0 - delta)

            for axis_name, effect in spec.get("affected_tau", {}).items():
                if axis_name not in fast_axis_idx:
                    continue
                idx = fast_axis_idx[axis_name]
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
        """Rank all interventions by their effect on the chosen metric."""
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

        results.sort(key=lambda x: x["delta"])

        self._model.set_age(original_age)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_metric(self, A, metric, J=None):
        """Evaluate a stability metric from a fast-subsystem A matrix."""
        n = A.shape[0]
        if metric == "spectral_abscissa":
            return _spectral_abscissa(A)
        elif metric == "dominant_recovery_time":
            alpha = _spectral_abscissa(A)
            return 1.0 / abs(alpha) if alpha != 0 else float("inf")
        elif metric == "lambda_max_gamma":
            Q = 0.01 * np.eye(n)
            if _spectral_abscissa(A) >= 0:
                return float("inf")
            Gamma = linalg.solve_continuous_lyapunov(A, -Q)
            return float(np.max(np.linalg.eigvalsh(Gamma)))
        elif metric == "bifurcation_margin_IM":
            if J is None:
                J = self._model.J
            tau = self._model.tau
            fai = self._model._fast_axis_idx
            i_I = fai["I"]
            i_M = fai["M"]
            return float(
                1.0 / (tau[i_I] * tau[i_M])
                - J[i_I, i_M] * J[i_M, i_I]
            )
        raise ValueError(f"Unknown metric: {metric}")
