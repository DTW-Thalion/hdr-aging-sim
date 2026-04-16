"""9-axis HDR dynamical system with two-timescale decomposition.

Loads the enriched J matrix from data/mechanistic_evidence/ and builds
the A = -D + J dynamics matrix.  E and B are treated as quasi-static:
they drift secularly with age and enter the fast subsystem as constant
forcing, but do not participate in perturbation-recovery eigenvalue
computation.

This class uses a **7-axis fast subsystem** (I, M, mito, P, C, N, F)
with PMID-cited tau values.  The tau registry uses bioenergetic
functional recovery timescales (e.g., mito tau = 1d via PGC-1a
signaling cycle, not 36d protein half-life).  The csv_loader module
provides ``calibrate_stable_system()`` for 7+2 calibration with c=0.89
and full 25-120 stability.

The full 9x9 system is available via A_full for reference.

This is a parallel model class.  The original R6 code (aging_params.py,
csv_loader.py) is unmodified and continues to work independently.
"""

import csv
import json
import os
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import brentq


# Calibration target: spectral abscissa of the fast subsystem at age 30.
# The uncoupled α = -1/τ_F = -0.125 at age 30.  With coupling, this shifts
# slightly.  -0.12 is achievable and within 11% of the R6 value (-0.134).
_ALPHA_TARGET_AGE30 = -0.12


def _spectral_abscissa(A):
    """Max real part of eigenvalues of A."""
    return float(np.max(np.real(linalg.eig(A, right=False))))


def _find_calibrated_scalar(J_raw, tau, target_alpha):
    """Find scalar c where α(-D + c*J) = target_alpha using Brent's method.

    At c=0, α = max(-1/τ_i) which is very negative.
    As c increases (with net-destabilising coupling), α increases.
    We seek the c that produces the target α.
    """
    D = np.diag(1.0 / tau)

    def objective(c):
        return _spectral_abscissa(-D + c * J_raw) - target_alpha

    obj_0 = objective(0.0)

    # Search for a bracket where the objective changes sign
    c_hi = 1.0
    for _ in range(60):
        obj_hi = objective(c_hi)
        if obj_0 * obj_hi < 0:
            return brentq(objective, 0.0, c_hi, xtol=1e-10)
        c_hi *= 2.0

    # Fallback: minimise distance to target
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(
        lambda c: abs(objective(c)), bounds=(0, c_hi), method="bounded"
    )
    return res.x


def _find_critical_scalar(J_raw, tau):
    """Find c_crit where α(-D + c*J) = 0 (kept for sensitivity.py)."""
    D = np.diag(1.0 / tau)

    def alpha_at(c):
        return _spectral_abscissa(-D + c * J_raw)

    c_hi = 0.1
    for _ in range(60):
        if alpha_at(c_hi) > 0:
            return brentq(lambda c: alpha_at(c), 0.0, c_hi, xtol=1e-12)
        c_hi *= 2.0
    return c_hi


class HDRMechanisticModel:
    """9-axis HDR dynamical system with two-timescale decomposition.

    The fast subsystem (7 axes: I, M, mito, P, C, N, F) determines
    perturbation-recovery dynamics (alpha, zeta, recovery time).  The
    quasi-static axes E and B drift with age and enter as constant forcing
    on the fast system.

    Calibration: a single scalar c is applied to all J entries so that
    alpha(A_fast) at age 30 matches the R6 target (approx -0.134).

    With V2.2 corrected tau values (mito tau=1-5d, bioenergetic recovery),
    the 7-axis fast subsystem achieves c=0.89 with full 25-120 stability
    via ``calibrate_stable_system()`` in csv_loader.
    """

    AXES = ["I", "M", "E", "mito", "P", "C", "N", "F", "B"]

    def __init__(self, evidence_dir="data/mechanistic_evidence", age=65,
                 target_alpha=_ALPHA_TARGET_AGE30, e_max_drift=1.0):
        """Load evidence and build the two-timescale model.

        Args:
            evidence_dir: path to mechanistic evidence directory.
            age: initial age for parameterisation.
            target_alpha: calibration target for α(A_fast, age=30).
            e_max_drift: Δx_E at age 80 in SD units (default 1.0).
        """
        self._evidence_dir = self._resolve_dir(evidence_dir)
        self._n_full = len(self.AXES)
        self._axis_idx = {a: i for i, a in enumerate(self.AXES)}
        self._e_max_drift = e_max_drift

        # Load the JSON export
        json_path = os.path.join(
            self._evidence_dir, "J_matrix_mechanistic_9x9.json"
        )
        with open(json_path, encoding="utf-8") as f:
            self._export = json.load(f)

        # --- Timescale architecture ---
        # Use the short-horizon subsystem (7 axes) as the dynamical system.
        # Both E (τ≈1000d) and B (τ≈90-120d) are quasi-static: they drift
        # secularly with age and enter as constant forcing, but do not
        # participate in perturbation-recovery eigenvalue computation.
        ts = self._export.get("timescale_architecture", {})
        fast_list = ts.get(
            "dynamical_subsystem_short_horizon",
            [a for a in self.AXES if a not in ("E", "B")],
        )
        self.FAST_AXES = list(fast_list)
        qs_list = list(ts.get("quasi_static_axes", ["E"]))
        # B is also quasi-static for perturbation-recovery dynamics
        if "B" not in qs_list:
            qs_list.append("B")
        self.QUASI_STATIC_AXES = qs_list
        self._n_fast = len(self.FAST_AXES)
        self._fast_axis_idx = {a: i for i, a in enumerate(self.FAST_AXES)}
        # Indices of fast axes in the full 9-dim system
        self._fast_idx = [self._axis_idx[a] for a in self.FAST_AXES]
        self._fast_idx_arr = np.array(self._fast_idx)
        # Quasi-static indices in full system
        self._qs_idx = [self._axis_idx[a] for a in self.QUASI_STATIC_AXES]
        self._qs_idx_arr = np.array(self._qs_idx)

        # --- Per-axis τ ranges from JSON ---
        tau_ref = self._export["tau_reference"]
        self._tau_ranges = {}
        for a in self.AXES:
            ref = tau_ref[a]
            rng = ref.get("range_days")
            if rng and len(rng) == 2:
                self._tau_ranges[a] = (float(rng[0]), float(rng[1]))
            else:
                val = float(ref["value_days"])
                self._tau_ranges[a] = (val, val)

        # --- Load entries ---
        self._entries = {}
        for entry_id, entry in self._export["entries"].items():
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            if src in self._axis_idx and tgt in self._axis_idx:
                self._entries[entry_id] = entry

        self._load_csv_supplement()

        self._excluded = {}
        for entry_id, entry in self._export.get("excluded_entries", {}).items():
            self._excluded[entry_id] = entry

        # --- Calibrate on the fast subsystem at age 30 ---
        tau_full_30 = self._tau_of_age_full(30)
        J_raw_full_30 = self._build_J_raw(30)
        # Extract fast subsystem
        ix = np.ix_(self._fast_idx, self._fast_idx)
        tau_fast_30 = tau_full_30[self._fast_idx_arr]
        J_raw_fast_30 = J_raw_full_30[ix]
        self._calibration_scalar = _find_calibrated_scalar(
            J_raw_fast_30, tau_fast_30, target_alpha
        )

        # --- Build matrices at initial age ---
        self._age = None
        self._A_matrix = None      # 7×7 fast
        self._A_full = None         # 9×9 full
        self._D_matrix = None       # 7×7 fast
        self._D_full = None
        self._J_matrix = None       # 7×7 fast
        self._J_full = None
        self._tau_fast_vec = None
        self._tau_full_vec = None
        self._forcing_E = None      # n_fast-vector
        self._delta_xE = 0.0
        self.set_age(age)

    # ------------------------------------------------------------------
    # Path resolution and data loading
    # ------------------------------------------------------------------

    def _resolve_dir(self, evidence_dir):
        """Resolve evidence directory relative to repo root."""
        if os.path.isabs(evidence_dir):
            return evidence_dir
        if os.path.isdir(evidence_dir):
            return os.path.abspath(evidence_dir)
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(pkg_dir))
        candidate = os.path.join(repo_root, evidence_dir)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(
            f"Cannot find evidence directory: {evidence_dir}"
        )

    def _load_csv_supplement(self):
        """Load entries from the CSV that are not in the JSON."""
        csv_path = os.path.join(
            self._evidence_dir, "J_matrix_mechanistic_9x9.csv"
        )
        if not os.path.isfile(csv_path):
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                coupling_id = row["coupling_id"].strip()
                if coupling_id in self._entries:
                    continue

                status = row.get("status", "").strip()
                if status in ("excluded", "unpopulated", ""):
                    continue

                src = row["source"].strip()
                tgt = row["target"].strip()
                if src not in self._axis_idx or tgt not in self._axis_idx:
                    continue

                try:
                    j30 = float(row.get("J_30", "0").strip())
                    j80 = float(row.get("J_80", "0").strip())
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
                    "decomposition_available": (
                        row.get("decomposed", "").strip().lower() == "true"
                    ),
                    "primary_mediator": (
                        row.get("key_mediator", "").strip() or None
                    ),
                }

    # ------------------------------------------------------------------
    # Tau and J construction
    # ------------------------------------------------------------------

    @staticmethod
    def _interp_fraction(age):
        """Return interpolation fraction: 0 at age 30, 1 at age 80, clamped."""
        return float(np.clip((age - 30.0) / 50.0, 0.0, 1.0))

    def _tau_of_age_full(self, age):
        """Return 9-vector of τ_i(age) using per-axis ranges from JSON.

        τ_i(age) = lo + (hi - lo) * (age - 30) / 50, clamped to [30, 80].
        """
        f = self._interp_fraction(age)
        tau = np.zeros(self._n_full)
        for i, a in enumerate(self.AXES):
            lo, hi = self._tau_ranges[a]
            tau[i] = lo + (hi - lo) * f
        return tau

    def _build_J_raw(self, age):
        """Build the uncalibrated 9×9 coupling matrix at a given age."""
        f = self._interp_fraction(age)
        J = np.zeros((self._n_full, self._n_full))

        for entry in self._entries.values():
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            j = self._axis_idx[src]
            i = self._axis_idx[tgt]
            j30 = entry["J_value_age30"]
            j80 = entry["J_value_age80"]
            J[i, j] = (1.0 - f) * j30 + f * j80

        return J

    # ------------------------------------------------------------------
    # Age setting
    # ------------------------------------------------------------------

    def set_age(self, age):
        """Update all age-dependent parameters and rebuild matrices.

        Builds the full 9×9 system, extracts the 7×7 fast subsystem,
        and computes the E forcing vector.
        """
        self._age = age

        # Full 9×9
        tau_full = self._tau_of_age_full(age)
        D_full = np.diag(1.0 / tau_full)
        J_full = self._calibration_scalar * self._build_J_raw(age)
        A_full = -D_full + J_full

        self._tau_full_vec = tau_full
        self._D_full = D_full
        self._J_full = J_full
        self._A_full = A_full

        # Extract fast 7×7 subsystem
        ix = np.ix_(self._fast_idx, self._fast_idx)
        self._tau_fast_vec = tau_full[self._fast_idx_arr]
        self._D_matrix = D_full[ix]
        self._J_matrix = J_full[ix]
        self._A_matrix = A_full[ix]

        # Quasi-static forcing from E and B
        # Δx_qs(age) = (age-30)/50 * max_drift for each quasi-static axis
        f = self._interp_fraction(age)
        self._qs_drift = {}
        forcing = np.zeros(self._n_fast)
        for qs_ax in self.QUASI_STATIC_AXES:
            qs_full_idx = self._axis_idx[qs_ax]
            drift = f * self._e_max_drift
            self._qs_drift[qs_ax] = drift
            # Forcing on each fast axis from this quasi-static axis
            forcing += J_full[self._fast_idx_arr, qs_full_idx] * drift
        self._forcing_E = forcing
        self._delta_xE = self._qs_drift.get("E", 0.0)

        # Check fast-subsystem stability
        alpha = self.spectral_abscissa
        if alpha >= 0:
            warnings.warn(
                f"Fast subsystem unstable at age {age}: α = {alpha:.6f}",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Properties — fast subsystem (primary)
    # ------------------------------------------------------------------

    @property
    def calibration_scalar(self):
        """The scalar c applied: J_sim = c * J_raw."""
        return self._calibration_scalar

    @property
    def A(self):
        """Return the 7×7 fast-subsystem dynamics matrix."""
        return self._A_matrix.copy()

    @property
    def D(self):
        """Return the 7×7 fast-subsystem self-restoration matrix."""
        return self._D_matrix.copy()

    @property
    def J(self):
        """Return the 7×7 fast-subsystem coupling matrix (calibrated)."""
        return self._J_matrix.copy()

    @property
    def tau(self):
        """Current τ vector (days) for fast axes."""
        return self._tau_fast_vec.copy()

    @property
    def n(self):
        """Dimension of the fast subsystem."""
        return self._n_fast

    @property
    def age(self):
        """Current age parameterisation."""
        return self._age

    @property
    def fast_axes(self):
        """List of dynamical axis names."""
        return list(self.FAST_AXES)

    @property
    def spectral_abscissa(self):
        """α(A_fast) = max_k Re(λ_k(A_fast))"""
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
        """ζ = |Re(λ_1)|/|λ_1| of the least-stable eigenvalue of A_fast."""
        eigenvalues = linalg.eig(self._A_matrix, right=False)
        idx = np.argmax(np.real(eigenvalues))
        lam1 = eigenvalues[idx]
        return float(np.abs(np.real(lam1)) / max(np.abs(lam1), 1e-15))

    # ------------------------------------------------------------------
    # Properties — full system (reference)
    # ------------------------------------------------------------------

    @property
    def A_full(self):
        """Return the full 9×9 dynamics matrix (for reference/validation)."""
        return self._A_full.copy()

    @property
    def D_full(self):
        """Return the full 9×9 self-restoration matrix."""
        return self._D_full.copy()

    @property
    def J_full(self):
        """Return the full 9×9 coupling matrix."""
        return self._J_full.copy()

    @property
    def tau_full(self):
        """Full 9-vector of τ (days)."""
        return self._tau_full_vec.copy()

    @property
    def n_full(self):
        """Dimension of the full system."""
        return self._n_full

    # ------------------------------------------------------------------
    # Properties — quasi-static
    # ------------------------------------------------------------------

    @property
    def quasi_static_forcing(self):
        """Return the n_fast-vector of E-mediated forcing on the fast subsystem.

        f_E[i] = J_full[fast_i, E_idx] * Δx_E(age).
        This shifts equilibrium but does not change α.
        """
        return self._forcing_E.copy()

    @property
    def quasi_static_state(self):
        """Dict of quasi-static axis drift values: {'E': Δx_E, 'B': Δx_B}."""
        return dict(self._qs_drift)

    # ------------------------------------------------------------------
    # Simulation — fast subsystem
    # ------------------------------------------------------------------

    def simulate_ou(self, x0, T, dt=None, Q=None, seed=None):
        """Simulate the fast subsystem with quasi-static E forcing.

        dx = (A_fast * x + f_E) * dt + sqrt(Q) * dW

        The system recovers toward x_eq = -A_fast^{-1} * f_E.

        Args:
            x0: initial state (n_fast,).
            T: total simulation time (days).
            dt: time step. If None, auto-selects for stability.
            Q: process noise (n_fast × n_fast). Default: 0.01 * I.
            seed: random seed.

        Returns: (times, states) where states is (N, n_fast).
        """
        n = self._n_fast
        if Q is None:
            Q = 0.01 * np.eye(n)

        A = self._A_matrix
        f_E = self._forcing_E
        has_noise = np.any(Q != 0)

        if has_noise:
            L = linalg.cholesky(Q, lower=True)

        if dt is None:
            max_eig = np.max(np.abs(linalg.eig(A, right=False)))
            dt = min(0.5 / max(max_eig, 1e-6), 0.1)

        rng = np.random.default_rng(seed)
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        states = np.zeros((n_steps + 1, n))
        states[0] = np.asarray(x0, dtype=float)
        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            drift = (A @ states[i] + f_E) * dt
            if has_noise:
                dW = rng.standard_normal(n)
                states[i + 1] = states[i] + drift + L @ dW * sqrt_dt
            else:
                states[i + 1] = states[i] + drift

        return times, states

    def simulate_ou_full(self, x0, T, dt=None, Q=None, seed=None):
        """Simulate the full 9×9 system (for comparison/validation).

        Uses A_full. α will be ≈ -0.001 (E-dominated).

        Args:
            x0: initial state (9,).
            T: total simulation time (days).
            dt: time step. If None, auto-selects.
            Q: process noise (9×9). Default: 0.01 * I.
            seed: random seed.

        Returns: (times, states) where states is (N, 9).
        """
        n = self._n_full
        if Q is None:
            Q = 0.01 * np.eye(n)

        A = self._A_full
        has_noise = np.any(Q != 0)

        if has_noise:
            L = linalg.cholesky(Q, lower=True)

        if dt is None:
            max_eig = np.max(np.abs(linalg.eig(A, right=False)))
            dt = min(0.5 / max(max_eig, 1e-6), 0.1)

        rng = np.random.default_rng(seed)
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        states = np.zeros((n_steps + 1, n))
        states[0] = np.asarray(x0, dtype=float)
        sqrt_dt = np.sqrt(dt)

        for i in range(n_steps):
            drift = A @ states[i] * dt
            if has_noise:
                dW = rng.standard_normal(n)
                states[i + 1] = states[i] + drift + L @ dW * sqrt_dt
            else:
                states[i + 1] = states[i] + drift

        return times, states

    def simulate_discrete(self, x0, n_visits, visit_interval_days,
                          Q=None, seed=None):
        """Simulate at discrete visit times (fast subsystem with E forcing).

        Uses exact discrete-time transition with equilibrium shift.

        Returns: (n_visits, n_fast) array of states at visit times.
        """
        n = self._n_fast
        if Q is None:
            Q = 0.01 * np.eye(n)

        dt = visit_interval_days
        A = self._A_matrix
        f_E = self._forcing_E
        Phi = linalg.expm(A * dt)

        # Equilibrium shift from E forcing
        x_eq = np.linalg.solve(A, -f_E)

        # Discrete noise covariance
        Gamma = linalg.solve_continuous_lyapunov(A, -Q)
        Sigma_eta = Gamma - Phi @ Gamma @ Phi.T
        Sigma_eta = (Sigma_eta + Sigma_eta.T) / 2
        eigvals_s, V = np.linalg.eigh(Sigma_eta)
        eigvals_s = np.maximum(eigvals_s, 0)
        Sigma_eta = V @ np.diag(eigvals_s) @ V.T
        L_eta = linalg.cholesky(Sigma_eta + 1e-14 * np.eye(n), lower=True)

        rng = np.random.default_rng(seed)
        states = np.zeros((n_visits, n))
        states[0] = np.asarray(x0, dtype=float)

        # Transition: x_{k+1} = Phi @ (x_k - x_eq) + x_eq + eta
        for k in range(1, n_visits):
            eta = L_eta @ rng.standard_normal(n)
            states[k] = Phi @ (states[k - 1] - x_eq) + x_eq + eta

        return states

    # ------------------------------------------------------------------
    # Perturbation and covariance
    # ------------------------------------------------------------------

    def perturb(self, axis, magnitude=2.0):
        """Create an impulse perturbation on one fast axis.

        Raises ValueError if axis is quasi-static (E, B).
        """
        if isinstance(axis, str):
            if axis in self.QUASI_STATIC_AXES:
                raise ValueError(
                    f"Cannot perturb quasi-static axis {axis!r}. "
                    f"Fast axes: {self.FAST_AXES}"
                )
            axis = self._fast_axis_idx[axis]
        x0 = np.zeros(self._n_fast)
        x0[axis] = magnitude
        return x0

    def compute_stationary_covariance(self, Q=None):
        """Solve AΓ + ΓAᵀ = -Q for the fast subsystem.

        Returns: (n_fast, n_fast) covariance around the shifted equilibrium.
        """
        if Q is None:
            Q = 0.01 * np.eye(self._n_fast)
        return linalg.solve_continuous_lyapunov(self._A_matrix, -Q)

    def compute_swds(self, x_individual, Gamma=None):
        """Compute SWDS-Γ for a fast-subsystem state vector.

        SWDS(x) = xᵀ Γ x / tr(Γ)
        """
        if Gamma is None:
            Gamma = self.compute_stationary_covariance()
        quadratic = x_individual @ Gamma @ x_individual
        return float(quadratic / np.trace(Gamma))

    def compute_equilibrium_shift(self):
        """Compute x_eq = -A_fast⁻¹ * f_E.

        The steady-state displacement of the fast subsystem caused by
        epigenetic drift. Grows with age as Δx_E increases.
        """
        if np.allclose(self._forcing_E, 0):
            return np.zeros(self._n_fast)
        return np.linalg.solve(self._A_matrix, -self._forcing_E)

    # ------------------------------------------------------------------
    # Age trajectory and bifurcation
    # ------------------------------------------------------------------

    def age_trajectory(self, ages=None):
        """Compute stability metrics across ages (fast subsystem).

        Default ages: [30, 40, 50, 60, 70, 80]

        Returns: list of dicts with keys:
          age, alpha, recovery_time, damping_ratio, stable,
          lambda_max_gamma, bifurcation_margin_IM,
          equilibrium_shift_norm, delta_xE
        """
        if ages is None:
            ages = [30, 40, 50, 60, 70, 80]

        original_age = self._age
        results = []

        for age in ages:
            self.set_age(age)
            alpha = self.spectral_abscissa
            Q = 0.01 * np.eye(self._n_fast)

            lam_max_gamma = None
            if alpha < 0:
                Gamma = self.compute_stationary_covariance(Q)
                lam_max_gamma = float(np.max(np.linalg.eigvalsh(Gamma)))

            x_eq = self.compute_equilibrium_shift()

            results.append({
                "age": age,
                "alpha": alpha,
                "recovery_time": self.dominant_recovery_time,
                "damping_ratio": self.damping_ratio,
                "stable": alpha < 0,
                "lambda_max_gamma": lam_max_gamma,
                "bifurcation_margin_IM": self.bifurcation_margin("I", "M"),
                "equilibrium_shift_norm": float(np.linalg.norm(x_eq)),
                "delta_xE": self._delta_xE,
            })

        self.set_age(original_age)
        return results

    def bifurcation_margin(self, axis_i="I", axis_j="M"):
        """Compute β = 1/(τ_i*τ_j) - J_ij*J_ji for a fast-axis pair.

        β > 0: healthy basin stable. β → 0: approaching bifurcation.
        """
        i = self._fast_axis_idx[axis_i]
        j = self._fast_axis_idx[axis_j]
        tau = self._tau_fast_vec
        J = self._J_matrix
        return float(1.0 / (tau[i] * tau[j]) - J[i, j] * J[j, i])

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_entry_info(self, coupling_id):
        """Return the full metadata for a J entry from the mechanistic export."""
        if coupling_id in self._entries:
            return dict(self._entries[coupling_id])
        if coupling_id in self._excluded:
            return dict(self._excluded[coupling_id])
        return None
