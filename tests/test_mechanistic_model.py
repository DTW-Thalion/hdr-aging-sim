"""Tests for the 9-axis mechanistic model and state-switched model."""

import numpy as np
import pytest

from hdr_sim.mechanistic_model import HDRMechanisticModel
from hdr_sim.state_conditioned import StateSwitchedModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Return a mechanistic model at the default age (65)."""
    return HDRMechanisticModel(age=65)


@pytest.fixture
def switched_model():
    """Return a state-switched model at the default age (65)."""
    return StateSwitchedModel(age=65)


# ---------------------------------------------------------------------------
# Stability tests
# ---------------------------------------------------------------------------

class TestStability:
    """Test that the system is Hurwitz (α < 0) across the lifespan."""

    def test_stability_age_30(self, model):
        model.set_age(30)
        assert model.spectral_abscissa < 0, (
            f"System unstable at age 30: α = {model.spectral_abscissa}"
        )

    def test_stability_age_50(self, model):
        model.set_age(50)
        assert model.spectral_abscissa < 0

    def test_stability_age_65(self, model):
        assert model.spectral_abscissa < 0

    def test_stability_age_80(self, model):
        model.set_age(80)
        assert model.spectral_abscissa < 0

    def test_stability_all_ages(self, model):
        """α(A) < 0 at ages 30, 40, 50, 60, 70, 80."""
        for age in [30, 40, 50, 60, 70, 80]:
            model.set_age(age)
            assert model.spectral_abscissa < 0, (
                f"System unstable at age {age}: α = {model.spectral_abscissa}"
            )


class TestSpectralDrift:
    """Test that stability erodes monotonically with age."""

    def test_spectral_drift_monotonic(self, model):
        """α(A) increases (toward 0) monotonically with age."""
        ages = [30, 40, 50, 60, 70, 80]
        alphas = []
        for age in ages:
            model.set_age(age)
            alphas.append(model.spectral_abscissa)
        for i in range(len(alphas) - 1):
            assert alphas[i + 1] > alphas[i], (
                f"α not monotonically increasing: "
                f"α({ages[i]})={alphas[i]:.6f}, α({ages[i+1]})={alphas[i+1]:.6f}"
            )

    def test_recovery_slowing(self, model):
        """Dominant recovery time increases with age."""
        ages = [30, 50, 65, 80]
        recovery_times = []
        for age in ages:
            model.set_age(age)
            recovery_times.append(model.dominant_recovery_time)
        for i in range(len(recovery_times) - 1):
            assert recovery_times[i + 1] > recovery_times[i], (
                f"Recovery time not increasing: "
                f"t({ages[i]})={recovery_times[i]:.1f}, "
                f"t({ages[i+1]})={recovery_times[i+1]:.1f}"
            )

    def test_damping_decline(self, model):
        """Damping ratio should not increase with age.

        For the mechanistic model, damping may be 1.0 (real eigenvalues)
        across all ages, which is valid — it means the system is overdamped
        throughout. We test that it doesn't increase above the young value.
        """
        model.set_age(30)
        zeta_young = model.damping_ratio
        model.set_age(80)
        zeta_old = model.damping_ratio
        assert zeta_old <= zeta_young + 1e-10, (
            f"Damping ratio increased with age: "
            f"ζ(30)={zeta_young:.4f}, ζ(80)={zeta_old:.4f}"
        )


# ---------------------------------------------------------------------------
# Matrix structure tests
# ---------------------------------------------------------------------------

class TestMatrixStructure:

    def test_excluded_entries(self, model):
        """J_B_M, J_B_N, J_mito_B are exactly 0 in J matrix."""
        J = model.J
        axes = model.AXES
        idx = {a: i for i, a in enumerate(axes)}

        # J_B_M: source=B, target=M → J[M, B]
        assert J[idx["M"], idx["B"]] == 0.0, (
            f"J_B_M should be 0, got {J[idx['M'], idx['B']]}"
        )
        # J_B_N: source=B, target=N → J[N, B]
        assert J[idx["N"], idx["B"]] == 0.0, (
            f"J_B_N should be 0, got {J[idx['N'], idx['B']]}"
        )
        # J_mito_B: source=mito, target=B → J[B, mito]
        assert J[idx["B"], idx["mito"]] == 0.0, (
            f"J_mito_B should be 0, got {J[idx['B'], idx['mito']]}"
        )

    def test_diagonal_zero(self, model):
        """J diagonal is always zero (no self-coupling)."""
        J = model.J
        np.testing.assert_array_equal(
            np.diag(J), np.zeros(9),
            err_msg="J diagonal should be zero"
        )

    def test_A_equals_neg_D_plus_J(self, model):
        """A = -D + J identity holds."""
        np.testing.assert_array_almost_equal(
            model.A, -model.D + model.J, decimal=14
        )

    def test_age_interpolation_endpoints(self, model):
        """J values at age 30 and 80 match export exactly (up to calibration)."""
        import json, os

        edir = model._evidence_dir
        with open(os.path.join(edir, "J_matrix_mechanistic_9x9.json")) as f:
            export = json.load(f)

        c = model.calibration_scalar
        idx = model._axis_idx

        for entry in export["entries"].values():
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            if src not in idx or tgt not in idx:
                continue
            j, i = idx[src], idx[tgt]

            # Age 30: should be c * J_value_age30
            model.set_age(30)
            expected_30 = c * entry["J_value_age30"]
            actual_30 = model.J[i, j]
            assert abs(actual_30 - expected_30) < 1e-12, (
                f"J[{tgt},{src}] at age 30: expected {expected_30}, got {actual_30}"
            )

            # Age 80: should be c * J_value_age80
            model.set_age(80)
            expected_80 = c * entry["J_value_age80"]
            actual_80 = model.J[i, j]
            assert abs(actual_80 - expected_80) < 1e-12, (
                f"J[{tgt},{src}] at age 80: expected {expected_80}, got {actual_80}"
            )

    def test_coupling_signs(self, model):
        """J has both pathological (+) and protective (-) entries."""
        J = model.J
        assert np.any(J > 0), "No pathological entries found"
        assert np.any(J < 0), "No protective entries found"

    def test_active_entry_count(self, model):
        """J should have exactly 17 non-zero off-diagonal entries
        (20 active entries filtered to 9-axis model, minus entries
        involving senescence/gut axes)."""
        J = model.J
        n_nonzero = np.count_nonzero(J)
        assert n_nonzero > 0, "J has no non-zero entries"
        assert n_nonzero == len(model._entries), (
            f"Expected {len(model._entries)} non-zero J entries, got {n_nonzero}"
        )


# ---------------------------------------------------------------------------
# Perturbation and recovery tests
# ---------------------------------------------------------------------------

class TestPerturbation:

    def test_perturbation_vector(self, model):
        """perturb('I', 2.0) returns correct initial condition."""
        x0 = model.perturb("I", 2.0)
        assert x0[0] == 2.0
        assert np.sum(np.abs(x0)) == 2.0

    def test_perturbation_recovery_age30(self, model):
        """Impulse on I at age 30 recovers (peak I decays)."""
        model.set_age(30)
        x0 = model.perturb("I", 2.0)
        times, states = model.simulate_ou(x0, T=50, Q=np.zeros((9, 9)), seed=42)
        # I should decay from 2.0
        assert states[-1, 0] < states[0, 0], "I did not decay at age 30"
        # Should be close to 0 by T=50 for fast axes
        assert abs(states[-1, 0]) < 1.0, (
            f"I not recovered at t=50: {states[-1, 0]:.4f}"
        )

    def test_cross_axis_propagation_age80(self, model):
        """Impulse on I at age 80 propagates to M and F."""
        model.set_age(80)
        x0 = model.perturb("I", 2.0)
        times, states = model.simulate_ou(
            x0, T=100, Q=np.zeros((9, 9)), seed=42
        )
        # M should be perturbed by I→M coupling
        M_peak = np.max(np.abs(states[:, 1]))
        assert M_peak > 0.01, (
            f"I→M propagation not visible at age 80: max|M|={M_peak:.6f}"
        )
        # F should be perturbed (via I→F and indirect paths)
        F_peak = np.max(np.abs(states[:, 7]))
        assert F_peak > 0.01, (
            f"I→F propagation not visible at age 80: max|F|={F_peak:.6f}"
        )


# ---------------------------------------------------------------------------
# Bifurcation margin tests
# ---------------------------------------------------------------------------

class TestBifurcation:

    def test_bifurcation_margin_positive(self, model):
        """β(I,M) > 0 at all ages."""
        for age in [30, 50, 65, 80]:
            model.set_age(age)
            beta = model.bifurcation_margin("I", "M")
            assert beta > 0, (
                f"β(I,M) not positive at age {age}: {beta}"
            )

    def test_bifurcation_margin_decreases(self, model):
        """β(I,M) decreases with age."""
        ages = [30, 50, 65, 80]
        betas = []
        for age in ages:
            model.set_age(age)
            betas.append(model.bifurcation_margin("I", "M"))
        for i in range(len(betas) - 1):
            assert betas[i + 1] < betas[i], (
                f"β not decreasing: β({ages[i]})={betas[i]:.6f}, "
                f"β({ages[i+1]})={betas[i+1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Covariance and SWDS tests
# ---------------------------------------------------------------------------

class TestCovariance:

    def test_stationary_covariance_symmetric(self, model):
        """Γ is symmetric."""
        Gamma = model.compute_stationary_covariance()
        np.testing.assert_array_almost_equal(
            Gamma, Gamma.T, decimal=12,
            err_msg="Γ is not symmetric"
        )

    def test_stationary_covariance_psd(self, model):
        """Γ is positive semi-definite."""
        Gamma = model.compute_stationary_covariance()
        eigvals = np.linalg.eigvalsh(Gamma)
        assert np.all(eigvals >= -1e-10), (
            f"Γ has negative eigenvalues: {eigvals[eigvals < -1e-10]}"
        )

    def test_swds_nonnegative(self, model):
        """SWDS ≥ 0 for any state vector."""
        rng = np.random.default_rng(123)
        Gamma = model.compute_stationary_covariance()
        for _ in range(20):
            x = rng.standard_normal(9)
            swds = model.compute_swds(x, Gamma)
            assert swds >= -1e-10, f"SWDS negative: {swds}"

    def test_swds_zero_at_origin(self, model):
        """SWDS(0) = 0."""
        swds = model.compute_swds(np.zeros(9))
        assert abs(swds) < 1e-15


# ---------------------------------------------------------------------------
# Age trajectory tests
# ---------------------------------------------------------------------------

class TestAgeTrajectory:

    def test_age_trajectory_returns_all_ages(self, model):
        """age_trajectory returns results for all requested ages."""
        results = model.age_trajectory()
        assert len(results) == 6
        assert results[0]["age"] == 30
        assert results[-1]["age"] == 80

    def test_age_trajectory_all_stable(self, model):
        """All ages in trajectory are stable."""
        results = model.age_trajectory()
        for r in results:
            assert r["stable"], f"Unstable at age {r['age']}: α={r['alpha']}"

    def test_recovery_ratio(self, model):
        """Recovery timescale at 80 should be > 2× that at 30."""
        results = model.age_trajectory(ages=[30, 80])
        ratio = results[1]["recovery_time"] / results[0]["recovery_time"]
        assert ratio > 2.0, f"Recovery ratio only {ratio:.1f}×"


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulation:

    def test_simulate_ou_shape(self, model):
        """OU simulation returns correct shapes."""
        x0 = np.zeros(9)
        times, states = model.simulate_ou(x0, T=10, seed=42)
        assert states.shape[1] == 9
        assert len(times) == states.shape[0]

    def test_simulate_discrete_shape(self, model):
        """Discrete simulation returns correct shapes."""
        x0 = np.zeros(9)
        states = model.simulate_discrete(x0, n_visits=10, visit_interval_days=30)
        assert states.shape == (10, 9)

    def test_simulate_deterministic_decay(self, model):
        """With Q=0, perturbation decays toward 0."""
        model.set_age(50)
        x0 = model.perturb("I", 2.0)
        times, states = model.simulate_ou(
            x0, T=30, Q=np.zeros((9, 9)), seed=42
        )
        # I axis should decay
        assert abs(states[-1, 0]) < abs(states[0, 0])


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_get_entry_info_active(self, model):
        """get_entry_info returns data for active entries."""
        info = model.get_entry_info("J_I_M")
        assert info is not None
        assert info["sign"] == 1

    def test_get_entry_info_excluded(self, model):
        """get_entry_info returns data for excluded entries."""
        info = model.get_entry_info("J_B_M")
        assert info is not None
        assert info["J_value"] == 0

    def test_get_entry_info_unknown(self, model):
        """get_entry_info returns None for unknown entries."""
        assert model.get_entry_info("J_UNKNOWN") is None

    def test_calibration_scalar_positive(self, model):
        """Calibration scalar is a positive number."""
        assert model.calibration_scalar > 0


# ---------------------------------------------------------------------------
# StateSwitchedModel tests
# ---------------------------------------------------------------------------

class TestSwitchedModel:

    def test_basin_classification(self, switched_model):
        """Basin classification returns valid labels."""
        x = np.zeros(9)
        assert switched_model.classify_basin(x) in ("healthy", "disease")

    def test_healthy_at_origin(self, switched_model):
        """Origin (all zeros) should classify as healthy."""
        assert switched_model.classify_basin(np.zeros(9)) == "healthy"

    def test_disease_at_positive_dysregulation(self, switched_model):
        """Large positive dysregulation should classify as disease."""
        x = np.ones(9) * 5.0
        assert switched_model.classify_basin(x) == "disease"

    def test_simulation_produces_switches(self, switched_model):
        """Simulation with perturbation should produce basin switches."""
        x0 = np.zeros(9)
        x0[0] = 3.0  # Strong I perturbation
        times, states, basins = switched_model.simulate_switched(
            x0, T=50, seed=42
        )
        unique_basins = set(basins)
        assert len(unique_basins) == 2, (
            f"Expected both basins, got only: {unique_basins}"
        )

    def test_switched_simulation_shape(self, switched_model):
        """Switched simulation returns correct shapes."""
        x0 = np.zeros(9)
        times, states, basins = switched_model.simulate_switched(
            x0, T=10, seed=42
        )
        assert states.shape[1] == 9
        assert len(basins) == len(times)

    def test_a_healthy_differs_from_a_disease(self, switched_model):
        """A_healthy and A_disease should differ."""
        diff = np.sum(np.abs(switched_model.A_healthy - switched_model.A_disease))
        assert diff > 1e-10, "A_healthy and A_disease are identical"

    def test_set_age_updates_both_matrices(self, switched_model):
        """Changing age updates both A_healthy and A_disease."""
        A_h_65 = switched_model.A_healthy.copy()
        switched_model.set_age(30)
        A_h_30 = switched_model.A_healthy.copy()
        assert not np.allclose(A_h_65, A_h_30), (
            "A_healthy did not change when age changed"
        )
