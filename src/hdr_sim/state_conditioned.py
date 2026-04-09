"""Switched linear dynamical system (SLDS) with healthy/disease basins.

7 J entries have different values in healthy vs disease regimes.
At each simulation step, the current state determines which basin
the system occupies, selecting the appropriate A matrix.

This extends the HDRMechanisticModel with state-dependent switching
for entries that exhibit qualitatively different behaviour in healthy
vs disease regimes (e.g., pulsatile vs continuous PTH on bone).
"""

import json
import os
import warnings

import numpy as np
from scipy import linalg

from .mechanistic_model import HDRMechanisticModel


class StateSwitchedModel:
    """Switched linear dynamical system (SLDS) with healthy/disease basins.

    7 J entries have different values in healthy vs disease regimes.
    At each simulation step, the current state determines which basin
    the system occupies, selecting the appropriate A matrix.
    """

    def __init__(self, evidence_dir="data/mechanistic_evidence", age=65):
        """Load state-conditioning specs from state_conditioning_export.json.

        Build A_healthy and A_disease matrices.
        """
        self._base = HDRMechanisticModel(evidence_dir=evidence_dir, age=age)
        self._evidence_dir = self._base._evidence_dir
        self._n = self._base._n
        self._axis_idx = self._base._axis_idx

        # Load state conditioning specs
        sc_path = os.path.join(self._evidence_dir, "state_conditioning_export.json")
        with open(sc_path, encoding="utf-8") as f:
            self._sc_specs = json.load(f)

        # Parse thresholds into a convenient format
        self._thresholds = []
        for entry_id, spec in self._sc_specs.items():
            threshold = spec.get("threshold", {})
            self._thresholds.append({
                "entry_id": entry_id,
                "axis_source": self._get_source_axis(entry_id),
                "axis_target": self._get_target_axis(entry_id),
                "healthy_fraction": spec["healthy_regime"]["J_fraction"],
                "disease_fraction": spec["disease_regime"]["J_fraction"],
                "threshold_value": threshold.get("value"),
                "threshold_direction": threshold.get("direction", "above_is_disease"),
                "threshold_biomarker": threshold.get("biomarker", ""),
            })

        # Build both matrices
        self._age = age
        self._rebuild_matrices()

    def _get_source_axis(self, entry_id):
        """Extract source axis from entry ID like J_I_M -> I."""
        parts = entry_id.split("_")
        return parts[1] if len(parts) >= 3 else ""

    def _get_target_axis(self, entry_id):
        """Extract target axis from entry ID like J_I_M -> M."""
        parts = entry_id.split("_")
        return parts[2] if len(parts) >= 3 else ""

    def _rebuild_matrices(self):
        """Build A_healthy and A_disease from the base model + conditioning specs."""
        J_base = self._base.J  # base J matrix (uses full J values)
        D = self._base.D

        J_healthy = J_base.copy()
        J_disease = J_base.copy()

        for spec in self._thresholds:
            src = spec["axis_source"]
            tgt = spec["axis_target"]
            if src not in self._axis_idx or tgt not in self._axis_idx:
                continue
            j = self._axis_idx[src]
            i = self._axis_idx[tgt]

            base_val = J_base[i, j]
            J_healthy[i, j] = base_val * spec["healthy_fraction"]
            J_disease[i, j] = base_val * spec["disease_fraction"]

        self._J_healthy = J_healthy
        self._J_disease = J_disease
        self._A_healthy = -D + J_healthy
        self._A_disease = -D + J_disease

    @property
    def A_healthy(self):
        """Return the A matrix for the healthy basin."""
        return self._A_healthy.copy()

    @property
    def A_disease(self):
        """Return the A matrix for the disease basin."""
        return self._A_disease.copy()

    def set_age(self, age):
        """Update age and rebuild both A matrices."""
        self._age = age
        self._base.set_age(age)
        self._rebuild_matrices()

    def classify_basin(self, x):
        """Determine which basin the current state occupies.

        Uses a simple heuristic: if the majority of state-conditioned
        source axes have positive (above-mean) dysregulation, classify
        as disease. The state vector x represents deviations from the
        healthy equilibrium, so positive values indicate dysregulation.

        Returns: "healthy" or "disease"
        """
        # Collect the source axes involved in state conditioning
        disease_votes = 0
        total_votes = 0

        for spec in self._thresholds:
            src = spec["axis_source"]
            if src not in self._axis_idx:
                continue
            idx = self._axis_idx[src]
            total_votes += 1

            # In state-space, x > 0 means dysregulated (above equilibrium)
            # A threshold_direction of "above_is_disease" means high values = disease
            # "below_is_disease" means low values = disease
            if spec["threshold_direction"] == "below_is_disease":
                if x[idx] < 0:  # below equilibrium
                    disease_votes += 1
            else:  # above_is_disease (default)
                if x[idx] > 0:  # above equilibrium
                    disease_votes += 1

        if total_votes == 0:
            return "healthy"
        return "disease" if disease_votes > total_votes / 2 else "healthy"

    def simulate_switched(self, x0, T, dt=None, Q=None, seed=None):
        """Simulate the SLDS: at each step, classify basin, select A_k, propagate.

        Records basin switches for analysis.

        Returns: times, states, basins (per-step basin assignment)
        """
        if Q is None:
            Q = 0.01 * np.eye(self._n)
        L = linalg.cholesky(Q, lower=True)

        # Auto-select dt if not provided
        if dt is None:
            max_eig = max(
                np.max(np.abs(linalg.eig(self._A_healthy, right=False))),
                np.max(np.abs(linalg.eig(self._A_disease, right=False))),
            )
            dt = min(0.5 / max(max_eig, 1e-6), 0.1)

        rng = np.random.default_rng(seed)
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        states = np.zeros((n_steps + 1, self._n))
        basins = []

        states[0] = np.asarray(x0, dtype=float)
        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            basin = self.classify_basin(states[i])
            basins.append(basin)
            A = self._A_disease if basin == "disease" else self._A_healthy

            dW = rng.standard_normal(self._n)
            states[i + 1] = states[i] + A @ states[i] * dt + L @ dW * sqrt_dt

        # Final step basin
        basins.append(self.classify_basin(states[-1]))

        return times, states, basins
