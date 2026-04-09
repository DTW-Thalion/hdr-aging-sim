"""9-axis HDR dynamical system parameterised from mechanistic evidence.

Loads the enriched J matrix from data/mechanistic_evidence/ and builds
the A = -D + J dynamics matrix with age-dependent parameterisation.

This is a parallel model class that uses the richer mechanistic evidence
export from HDR-mechanistic. The original R6 code (aging_params.py,
csv_loader.py) is unmodified and continues to work independently.
"""

import csv
import json
import os
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import brentq


# Age-scaling factor for τ: at age 80, τ is this multiple of the
# reference value. Linear interpolation between ages 30 and 80.
_TAU_AGE_SCALE_80 = 3.0

# Fraction of the critical coupling scalar to use. At the critical
# value, α(A_80) = 0 (bifurcation). Using 50% provides a stability
# margin while retaining meaningful coupling dynamics.
_COUPLING_FRACTION = 0.50


def _spectral_abscissa(A):
    """Max real part of eigenvalues of A."""
    return float(np.max(np.real(linalg.eig(A, right=False))))


def _find_critical_scalar(J_raw, tau):
    """Find the critical scalar c_crit where α(-D + c*J) = 0.

    At c < c_crit the system is stable; at c > c_crit it is unstable.
    Uses Brent's method.
    """
    D = np.diag(1.0 / tau)

    def alpha_at(c):
        return _spectral_abscissa(-D + c * J_raw)

    # At c=0, α = -min(1/τ) < 0. Search for c where α first reaches 0.
    c_hi = 0.1
    for _ in range(60):
        if alpha_at(c_hi) > 0:
            return brentq(lambda c: alpha_at(c), 0.0, c_hi, xtol=1e-12)
        c_hi *= 2.0

    # Coupling doesn't destabilise — return large value
    return c_hi


class HDRMechanisticModel:
    """9-axis HDR dynamical system parameterised from mechanistic evidence.

    Loads the enriched J matrix from data/mechanistic_evidence/ and builds
    the A = -D + J dynamics matrix with age-dependent parameterisation.

    Calibration: a single scalar c is applied to all J entries so that
    α(A) at age 30 matches the target spectral abscissa (≈ -0.13).
    The same scalar is applied at all ages, preserving the literature-derived
    age trajectory of coupling strengths.
    """

    AXES = ["I", "M", "E", "mito", "P", "C", "N", "F", "B"]

    def __init__(self, evidence_dir="data/mechanistic_evidence", age=65,
                 coupling_fraction=_COUPLING_FRACTION):
        """Load J matrix and τ values from the mechanistic export.

        Steps:
        1. Read J_matrix_mechanistic_9x9.json
        2. Read tau_reference → build D = diag(1/τ_i)
        3. Calibrate: find critical c where α(A_80) = 0, then use
           coupling_fraction * c_crit as the operational scalar
        4. Interpolate J values for the given age, apply c
        5. Build A = -D + J
        6. Verify stability: α(A) < 0; warn if not
        """
        self._evidence_dir = self._resolve_dir(evidence_dir)
        self._n = len(self.AXES)
        self._axis_idx = {a: i for i, a in enumerate(self.AXES)}

        # Load the JSON export
        json_path = os.path.join(self._evidence_dir, "J_matrix_mechanistic_9x9.json")
        with open(json_path, encoding="utf-8") as f:
            self._export = json.load(f)

        # Extract τ reference values (days)
        tau_ref = self._export["tau_reference"]
        self._tau_ref = np.array([tau_ref[a]["value_days"] for a in self.AXES])

        # Extract active entries from JSON (filter to 9-axis model only)
        self._entries = {}
        for entry_id, entry in self._export["entries"].items():
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            if src in self._axis_idx and tgt in self._axis_idx:
                self._entries[entry_id] = entry

        # Supplement with CSV entries not already in JSON (r6_only, etc.)
        self._load_csv_supplement()

        # Extract excluded entries
        self._excluded = {}
        for entry_id, entry in self._export.get("excluded_entries", {}).items():
            self._excluded[entry_id] = entry

        # Calibrate: find critical c at age 80, then scale down
        J_raw_80 = self._build_J_raw(80)
        tau_80 = self._tau_of_age(80)
        c_crit = _find_critical_scalar(J_raw_80, tau_80)
        self._calibration_scalar = coupling_fraction * c_crit

        # Set age and build matrices
        self._age = None
        self._A_matrix = None
        self._D_matrix = None
        self._J_matrix = None
        self.set_age(age)

    def _resolve_dir(self, evidence_dir):
        """Resolve evidence directory relative to repo root."""
        if os.path.isabs(evidence_dir):
            return evidence_dir
        if os.path.isdir(evidence_dir):
            return os.path.abspath(evidence_dir)
        # Try relative to package location (src/hdr_sim/ -> repo root)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(pkg_dir))
        candidate = os.path.join(repo_root, evidence_dir)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(
            f"Cannot find evidence directory: {evidence_dir}"
        )

    def _load_csv_supplement(self):
        """Load entries from the mechanistic CSV that are not in the JSON.

        The CSV may contain r6_only or r6_bridged_protective entries that
        were not included in the JSON export. These are added with minimal
        metadata so they contribute to the J matrix.
        """
        csv_path = os.path.join(self._evidence_dir, "J_matrix_mechanistic_9x9.csv")
        if not os.path.isfile(csv_path):
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                coupling_id = row["coupling_id"].strip()
                if coupling_id in self._entries:
                    continue  # JSON takes priority

                status = row.get("status", "").strip()
                if status in ("excluded", "unpopulated", ""):
                    continue

                src = row["source"].strip()
                tgt = row["target"].strip()
                if src not in self._axis_idx or tgt not in self._axis_idx:
                    continue

                j30_str = row.get("J_30", "0").strip()
                j80_str = row.get("J_80", "0").strip()
                try:
                    j30 = float(j30_str)
                    j80 = float(j80_str)
                except ValueError:
                    continue

                if j30 == 0 and j80 == 0:
                    continue

                sign = int(row.get("sign", "0").strip() or "0")

                self._entries[coupling_id] = {
                    "source_axis": src,
                    "target_axis": tgt,
                    "sign": sign,
                    "J_value_age30": j30,
                    "J_value_age80": j80,
                    "reconciliation_status": status,
                    "magnitude_tier": row.get("magnitude", "").strip(),
                    "confidence_grade": row.get("grade", "").strip(),
                    "causal_level": row.get("causal", "").strip(),
                    "decomposition_available": row.get("decomposed", "").strip().lower() == "true",
                    "primary_mediator": row.get("key_mediator", "").strip() or None,
                }

    @staticmethod
    def _interp_fraction(age):
        """Return interpolation fraction: 0 at age 30, 1 at age 80, clamped."""
        return float(np.clip((age - 30.0) / 50.0, 0.0, 1.0))

    def _tau_of_age(self, age):
        """Return 9-vector of τ_i(age) in days.

        τ(age) = τ_ref * (1 + (scale-1) * f), where f is the age fraction.
        At age 30: τ = τ_ref. At age 80: τ = scale * τ_ref.
        """
        f = self._interp_fraction(age)
        return self._tau_ref * (1.0 + (_TAU_AGE_SCALE_80 - 1.0) * f)

    def _build_J_raw(self, age):
        """Build the uncalibrated 9×9 coupling matrix at a given age.

        J(age) = J_30 + (J_80 - J_30) * (age - 30) / 50
        """
        f = self._interp_fraction(age)
        J = np.zeros((self._n, self._n))

        for entry in self._entries.values():
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            j = self._axis_idx[src]  # source = column
            i = self._axis_idx[tgt]  # target = row
            j30 = entry["J_value_age30"]
            j80 = entry["J_value_age80"]
            J[i, j] = (1.0 - f) * j30 + f * j80

        return J

    @property
    def calibration_scalar(self):
        """The scalar c applied: J_sim = c * J_raw."""
        return self._calibration_scalar

    @property
    def A(self):
        """Return the 9×9 dynamics matrix A = -D + J."""
        return self._A_matrix.copy()

    @property
    def D(self):
        """Return the 9×9 diagonal self-restoration matrix."""
        return self._D_matrix.copy()

    @property
    def J(self):
        """Return the 9×9 coupling matrix (calibrated)."""
        return self._J_matrix.copy()

    @property
    def age(self):
        """Current age parameterisation."""
        return self._age

    @property
    def tau(self):
        """Current τ vector (days)."""
        return self._tau_of_age(self._age)

    @property
    def spectral_abscissa(self):
        """α(A) = max_k Re(λ_k(A))"""
        return _spectral_abscissa(self._A_matrix)

    @property
    def dominant_recovery_time(self):
        """1/|α| — time for the dominant mode to decay by 1/e"""
        alpha = self.spectral_abscissa
        if alpha == 0:
            return float("inf")
        return 1.0 / abs(alpha)

    @property
    def damping_ratio(self):
        """ζ = |Re(λ_1)|/|λ_1| of the least-stable eigenvalue.

        ζ ≈ 1: overdamped (healthy). ζ declining: underdamped, oscillatory (frailty).
        """
        eigenvalues = linalg.eig(self._A_matrix, right=False)
        idx = np.argmax(np.real(eigenvalues))
        lam1 = eigenvalues[idx]
        return float(np.abs(np.real(lam1)) / max(np.abs(lam1), 1e-15))

    def set_age(self, age):
        """Update all age-dependent parameters and rebuild A."""
        self._age = age
        tau = self._tau_of_age(age)
        self._D_matrix = np.diag(1.0 / tau)
        self._J_matrix = self._calibration_scalar * self._build_J_raw(age)
        self._A_matrix = -self._D_matrix + self._J_matrix

        # Check stability
        alpha = self.spectral_abscissa
        if alpha >= 0:
            warnings.warn(
                f"System is unstable at age {age}: α(A) = {alpha:.6f}. "
                f"Expected α < 0 for a stable system.",
                stacklevel=2,
            )

    def simulate_ou(self, x0, T, dt=None, Q=None, seed=None):
        """Simulate the OU process using Euler-Maruyama.

        dx = A*x*dt + sqrt(Q)*dW

        Args:
            x0: initial state (9,) — use np.zeros(9) for equilibrium start
            T: total simulation time (in days)
            dt: time step (days). If None, auto-selects for stability
                based on the fastest eigenvalue (dt = 0.5/max|λ|).
            Q: process noise covariance (9×9), default = identity scaled by 0.01
            seed: random seed for reproducibility

        Returns:
            times: (N,) array of time points
            states: (N, 9) array of state trajectories
        """
        if Q is None:
            Q = 0.01 * np.eye(self._n)

        A = self._A_matrix
        has_noise = np.any(Q != 0)

        if has_noise:
            L = linalg.cholesky(Q, lower=True)

        if dt is None:
            # Auto-select dt for Euler-Maruyama stability
            max_eig = np.max(np.abs(linalg.eig(A, right=False)))
            dt = min(0.5 / max(max_eig, 1e-6), 0.1)

        rng = np.random.default_rng(seed)
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        states = np.zeros((n_steps + 1, self._n))
        states[0] = np.asarray(x0, dtype=float)

        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            drift = A @ states[i] * dt
            if has_noise:
                dW = rng.standard_normal(self._n)
                states[i + 1] = states[i] + drift + L @ dW * sqrt_dt
            else:
                states[i + 1] = states[i] + drift

        return times, states

    def simulate_discrete(self, x0, n_visits, visit_interval_days, Q=None, seed=None):
        """Simulate at discrete visit times only (cohort-style).

        Computes the matrix exponential Φ = exp(A * Δt) and uses
        the exact discrete-time transition: x(t+Δt) ~ N(Φ*x(t), Σ_η)
        where Σ_η is the discrete-time noise covariance computed from
        the Lyapunov equation.

        Returns: (n_visits, 9) array of states at visit times.
        """
        if Q is None:
            Q = 0.01 * np.eye(self._n)

        dt = visit_interval_days
        A = self._A_matrix
        Phi = linalg.expm(A * dt)

        # Discrete-time noise covariance:
        # Σ_η = Γ - Φ Γ Φᵀ  where Γ solves AΓ + ΓAᵀ = -Q
        Gamma = linalg.solve_continuous_lyapunov(A, -Q)
        Sigma_eta = Gamma - Phi @ Gamma @ Phi.T

        # Ensure symmetry and PSD via eigenvalue clipping
        Sigma_eta = (Sigma_eta + Sigma_eta.T) / 2
        eigvals_s, V = np.linalg.eigh(Sigma_eta)
        eigvals_s = np.maximum(eigvals_s, 0)
        Sigma_eta = V @ np.diag(eigvals_s) @ V.T

        L_eta = linalg.cholesky(Sigma_eta + 1e-14 * np.eye(self._n), lower=True)

        rng = np.random.default_rng(seed)
        states = np.zeros((n_visits, self._n))
        states[0] = np.asarray(x0, dtype=float)

        for k in range(1, n_visits):
            eta = L_eta @ rng.standard_normal(self._n)
            states[k] = Phi @ states[k - 1] + eta

        return states

    def perturb(self, axis, magnitude=2.0):
        """Create an initial condition with an impulse perturbation on one axis.

        Returns: (9,) array with magnitude at the specified axis index, 0 elsewhere.
        """
        if isinstance(axis, str):
            axis = self._axis_idx[axis]
        x0 = np.zeros(self._n)
        x0[axis] = magnitude
        return x0

    def compute_stationary_covariance(self, Q=None):
        """Solve the Lyapunov equation AΓ + ΓAᵀ = -Q for Γ.

        Uses scipy.linalg.solve_continuous_lyapunov.

        Returns: (9, 9) positive-definite stationary covariance.
        """
        if Q is None:
            Q = 0.01 * np.eye(self._n)
        return linalg.solve_continuous_lyapunov(self._A_matrix, -Q)

    def compute_swds(self, x_individual, Gamma=None):
        """Compute SWDS-Γ for an individual state vector.

        SWDS(x) = xᵀ Γ x / tr(Γ)
        If Gamma not provided, computes from current A.
        """
        if Gamma is None:
            Gamma = self.compute_stationary_covariance()
        quadratic = x_individual @ Gamma @ x_individual
        return float(quadratic / np.trace(Gamma))

    def age_trajectory(self, ages=None):
        """Compute stability metrics across ages.

        Default ages: [30, 40, 50, 60, 70, 80]

        Returns: list of dicts with keys:
          age, alpha, recovery_time, damping_ratio, stable,
          lambda_max_gamma, bifurcation_margin_IM
        """
        if ages is None:
            ages = [30, 40, 50, 60, 70, 80]

        original_age = self._age
        results = []

        for age in ages:
            self.set_age(age)
            alpha = self.spectral_abscissa
            Q = 0.01 * np.eye(self._n)

            lam_max_gamma = None
            if alpha < 0:
                Gamma = self.compute_stationary_covariance(Q)
                lam_max_gamma = float(np.max(np.linalg.eigvalsh(Gamma)))

            results.append({
                "age": age,
                "alpha": alpha,
                "recovery_time": self.dominant_recovery_time,
                "damping_ratio": self.damping_ratio,
                "stable": alpha < 0,
                "lambda_max_gamma": lam_max_gamma,
                "bifurcation_margin_IM": self.bifurcation_margin("I", "M"),
            })

        self.set_age(original_age)
        return results

    def bifurcation_margin(self, axis_i="I", axis_j="M"):
        """Compute β = 1/(τ_i*τ_j) - J_ij*J_ji for a bidirectional pair.

        β > 0: healthy basin stable. β → 0: approaching bifurcation.
        """
        i = self._axis_idx[axis_i]
        j = self._axis_idx[axis_j]
        tau = self._tau_of_age(self._age)
        J = self._J_matrix
        return float(1.0 / (tau[i] * tau[j]) - J[i, j] * J[j, i])

    def get_entry_info(self, coupling_id):
        """Return the full metadata for a J entry from the mechanistic export."""
        if coupling_id in self._entries:
            return dict(self._entries[coupling_id])
        if coupling_id in self._excluded:
            return dict(self._excluded[coupling_id])
        return None
