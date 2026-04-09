"""Tests for the two-timescale mechanistic model and state-switched model."""

import numpy as np
import pytest

from hdr_sim.mechanistic_model import HDRMechanisticModel, _spectral_abscissa
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
# Two-timescale structure tests
# ---------------------------------------------------------------------------

class TestTimescaleArchitecture:
    """Test the two-timescale decomposition."""

    def test_fast_subsystem_dimension(self, model):
        """A is 7×7 (fast subsystem without E or B)."""
        assert model.A.shape == (7, 7)
        assert model.n == 7

    def test_full_system_dimension(self, model):
        """A_full is 9×9."""
        assert model.A_full.shape == (9, 9)
        assert model.n_full == 9

    def test_fast_axes_list(self, model):
        """Fast axes are I, M, mito, P, C, N, F."""
        assert model.FAST_AXES == ["I", "M", "mito", "P", "C", "N", "F"]

    def test_quasi_static_axes(self, model):
        """E and B are quasi-static."""
        assert "E" in model.QUASI_STATIC_AXES
        assert "B" in model.QUASI_STATIC_AXES

    def test_fast_vs_full_alpha(self, model):
        """Fast α ≈ -0.12 to -0.01; full α ≈ -0.001 (E-dominated)."""
        model.set_age(30)
        alpha_fast = model.spectral_abscissa
        alpha_full = _spectral_abscissa(model.A_full)
        assert alpha_fast < -0.05, f"Fast α too close to 0: {alpha_fast}"
        assert abs(alpha_full) < 0.01, f"Full α not E-dominated: {alpha_full}"

    def test_tau_per_axis_ranges(self, model):
        """τ values match JSON range endpoints at ages 30 and 80."""
        model.set_age(30)
        fai = model._fast_axis_idx
        assert abs(model.tau[fai["I"]] - 7.0) < 0.01
        assert abs(model.tau[fai["F"]] - 8.0) < 0.01
        assert abs(model.tau[fai["M"]] - 0.1) < 0.01

        model.set_age(80)
        assert abs(model.tau[fai["I"]] - 25.0) < 0.01
        assert abs(model.tau[fai["F"]] - 42.0) < 0.01


# ---------------------------------------------------------------------------
# Stability tests
# ---------------------------------------------------------------------------

class TestStability:

    def test_stability_all_ages(self, model):
        """α(A_fast) < 0 at ages 30–80."""
        for age in [30, 40, 50, 60, 70, 80]:
            model.set_age(age)
            assert model.spectral_abscissa < 0, (
                f"Unstable at age {age}: α = {model.spectral_abscissa}"
            )

    def test_alpha_age30_near_target(self, model):
        """α(30) ≈ -0.12 (within 20% of R6 target -0.134)."""
        model.set_age(30)
        alpha = model.spectral_abscissa
        assert -0.20 < alpha < -0.05, f"α(30) = {alpha} out of expected range"

    def test_alpha_age80_near_target(self, model):
        """α(80) ≈ -0.01 (still negative, much closer to 0)."""
        model.set_age(80)
        alpha = model.spectral_abscissa
        assert -0.05 < alpha < 0, f"α(80) = {alpha} out of expected range"


class TestSpectralDrift:

    def test_spectral_drift_monotonic(self, model):
        """α increases monotonically with age."""
        ages = [30, 40, 50, 60, 70, 80]
        alphas = []
        for age in ages:
            model.set_age(age)
            alphas.append(model.spectral_abscissa)
        for i in range(len(alphas) - 1):
            assert alphas[i + 1] > alphas[i], (
                f"α not monotonic: α({ages[i]})={alphas[i]:.6f}, "
                f"α({ages[i+1]})={alphas[i+1]:.6f}"
            )

    def test_recovery_slowing(self, model):
        """Recovery time increases with age."""
        ages = [30, 50, 65, 80]
        rts = []
        for age in ages:
            model.set_age(age)
            rts.append(model.dominant_recovery_time)
        for i in range(len(rts) - 1):
            assert rts[i + 1] > rts[i]

    def test_recovery_ratio(self, model):
        """Recovery ratio 80/30 ≈ 4-15× (R6 range)."""
        model.set_age(30)
        rt30 = model.dominant_recovery_time
        model.set_age(80)
        rt80 = model.dominant_recovery_time
        ratio = rt80 / rt30
        assert 3.0 <= ratio <= 20.0, f"Recovery ratio = {ratio:.1f}×"

    def test_damping_decline(self, model):
        """Damping ratio should not increase with age."""
        model.set_age(30)
        zeta_young = model.damping_ratio
        model.set_age(80)
        zeta_old = model.damping_ratio
        assert zeta_old <= zeta_young + 1e-10


# ---------------------------------------------------------------------------
# Matrix structure tests
# ---------------------------------------------------------------------------

class TestMatrixStructure:

    def test_excluded_entries_full(self, model):
        """J_B_M, J_B_N, J_mito_B are exactly 0 in the full J matrix."""
        J = model.J_full
        idx = model._axis_idx
        assert J[idx["M"], idx["B"]] == 0.0  # J_B_M
        assert J[idx["N"], idx["B"]] == 0.0  # J_B_N
        assert J[idx["B"], idx["mito"]] == 0.0  # J_mito_B

    def test_diagonal_zero(self, model):
        """J diagonal is zero (fast subsystem)."""
        np.testing.assert_array_equal(np.diag(model.J), np.zeros(model.n))

    def test_A_equals_neg_D_plus_J(self, model):
        """A = -D + J for the fast subsystem."""
        np.testing.assert_array_almost_equal(
            model.A, -model.D + model.J, decimal=14
        )

    def test_coupling_signs(self, model):
        """J has both pathological (+) and protective (-) entries."""
        J = model.J
        assert np.any(J > 0), "No pathological entries"
        assert np.any(J < 0), "No protective entries"

    def test_age_interpolation_endpoints(self, model):
        """J values at ages 30 and 80 match export (fast subsystem)."""
        import json, os
        with open(os.path.join(
            model._evidence_dir, "J_matrix_mechanistic_9x9.json"
        )) as f:
            export = json.load(f)

        c = model.calibration_scalar
        fai = model._fast_axis_idx

        for entry in export["entries"].values():
            src = entry["source_axis"]
            tgt = entry["target_axis"]
            if src not in fai or tgt not in fai:
                continue
            j, i = fai[src], fai[tgt]

            model.set_age(30)
            expected = c * entry["J_value_age30"]
            assert abs(model.J[i, j] - expected) < 1e-12

            model.set_age(80)
            expected = c * entry["J_value_age80"]
            assert abs(model.J[i, j] - expected) < 1e-12


# ---------------------------------------------------------------------------
# Quasi-static forcing tests
# ---------------------------------------------------------------------------

class TestQuasiStaticForcing:

    def test_forcing_zero_at_age30(self, model):
        """Quasi-static forcing is zero at age 30 (no drift yet)."""
        model.set_age(30)
        assert np.allclose(model.quasi_static_forcing, 0)

    def test_forcing_nonzero_at_age80(self, model):
        """Quasi-static forcing is nonzero at age 80."""
        model.set_age(80)
        assert np.linalg.norm(model.quasi_static_forcing) > 0

    def test_equilibrium_shift_increases(self, model):
        """Equilibrium shift grows with age."""
        shifts = []
        for age in [30, 50, 65, 80]:
            model.set_age(age)
            shifts.append(np.linalg.norm(model.compute_equilibrium_shift()))
        for i in range(len(shifts) - 1):
            assert shifts[i + 1] >= shifts[i] - 1e-10

    def test_quasi_static_state_has_E_and_B(self, model):
        """quasi_static_state includes E and B drift values."""
        model.set_age(65)
        qs = model.quasi_static_state
        assert "E" in qs
        assert "B" in qs


# ---------------------------------------------------------------------------
# Perturbation and recovery tests
# ---------------------------------------------------------------------------

class TestPerturbation:

    def test_perturbation_vector(self, model):
        x0 = model.perturb("I", 2.0)
        assert x0[model._fast_axis_idx["I"]] == 2.0
        assert np.sum(np.abs(x0)) == 2.0

    def test_perturb_quasi_static_raises(self, model):
        with pytest.raises(ValueError):
            model.perturb("E")
        with pytest.raises(ValueError):
            model.perturb("B")

    def test_perturbation_recovery_age30(self, model):
        model.set_age(30)
        x0 = model.perturb("I", 2.0)
        n = model.n
        times, states = model.simulate_ou(
            x0, T=50, Q=np.zeros((n, n)), seed=42
        )
        assert abs(states[-1, 0]) < 1.0

    def test_cross_axis_propagation_age80(self, model):
        model.set_age(80)
        x0 = model.perturb("I", 2.0)
        n = model.n
        times, states = model.simulate_ou(
            x0, T=100, Q=np.zeros((n, n)), seed=42
        )
        fai = model._fast_axis_idx
        M_peak = np.max(np.abs(states[:, fai["M"]]))
        F_peak = np.max(np.abs(states[:, fai["F"]]))
        assert M_peak > 0.001, f"I→M propagation not visible: {M_peak}"
        assert F_peak > 0.001, f"I→F propagation not visible: {F_peak}"


# ---------------------------------------------------------------------------
# Bifurcation margin tests
# ---------------------------------------------------------------------------

class TestBifurcation:

    def test_bifurcation_margin_positive(self, model):
        for age in [30, 50, 65, 80]:
            model.set_age(age)
            assert model.bifurcation_margin("I", "M") > 0

    def test_bifurcation_margin_decreases(self, model):
        ages = [30, 50, 65, 80]
        betas = []
        for age in ages:
            model.set_age(age)
            betas.append(model.bifurcation_margin("I", "M"))
        for i in range(len(betas) - 1):
            assert betas[i + 1] < betas[i]


# ---------------------------------------------------------------------------
# Covariance and SWDS tests
# ---------------------------------------------------------------------------

class TestCovariance:

    def test_stationary_covariance_symmetric(self, model):
        Gamma = model.compute_stationary_covariance()
        np.testing.assert_array_almost_equal(Gamma, Gamma.T, decimal=12)

    def test_stationary_covariance_psd(self, model):
        Gamma = model.compute_stationary_covariance()
        eigvals = np.linalg.eigvalsh(Gamma)
        assert np.all(eigvals >= -1e-10)

    def test_swds_nonnegative(self, model):
        rng = np.random.default_rng(123)
        Gamma = model.compute_stationary_covariance()
        for _ in range(20):
            x = rng.standard_normal(model.n)
            assert model.compute_swds(x, Gamma) >= -1e-10

    def test_swds_zero_at_origin(self, model):
        assert abs(model.compute_swds(np.zeros(model.n))) < 1e-15


# ---------------------------------------------------------------------------
# Age trajectory tests
# ---------------------------------------------------------------------------

class TestAgeTrajectory:

    def test_age_trajectory_returns_all_ages(self, model):
        results = model.age_trajectory()
        assert len(results) == 6
        assert results[0]["age"] == 30
        assert results[-1]["age"] == 80

    def test_age_trajectory_all_stable(self, model):
        for r in model.age_trajectory():
            assert r["stable"], f"Unstable at age {r['age']}"

    def test_age_trajectory_has_equilibrium_shift(self, model):
        results = model.age_trajectory()
        assert results[0]["equilibrium_shift_norm"] < 1e-10  # age 30
        assert results[-1]["equilibrium_shift_norm"] > 0  # age 80


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulation:

    def test_simulate_ou_shape(self, model):
        x0 = np.zeros(model.n)
        times, states = model.simulate_ou(x0, T=10, seed=42)
        assert states.shape[1] == model.n
        assert len(times) == states.shape[0]

    def test_simulate_ou_full_shape(self, model):
        x0 = np.zeros(model.n_full)
        times, states = model.simulate_ou_full(x0, T=10, seed=42)
        assert states.shape[1] == 9
        assert len(times) == states.shape[0]

    def test_simulate_discrete_shape(self, model):
        x0 = np.zeros(model.n)
        states = model.simulate_discrete(x0, n_visits=10, visit_interval_days=30)
        assert states.shape == (10, model.n)

    def test_deterministic_decay(self, model):
        model.set_age(50)
        x0 = model.perturb("I", 2.0)
        n = model.n
        times, states = model.simulate_ou(
            x0, T=30, Q=np.zeros((n, n)), seed=42
        )
        assert abs(states[-1, 0]) < abs(states[0, 0])


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_get_entry_info_active(self, model):
        info = model.get_entry_info("J_I_M")
        assert info is not None
        assert info["sign"] == 1

    def test_get_entry_info_excluded(self, model):
        info = model.get_entry_info("J_B_M")
        assert info is not None
        assert info["J_value"] == 0

    def test_calibration_scalar_positive(self, model):
        assert model.calibration_scalar > 0

    def test_calibration_scalar_range(self, model):
        """c should be in the expected range (0.1-1.0)."""
        assert 0.1 < model.calibration_scalar < 1.0


# ---------------------------------------------------------------------------
# StateSwitchedModel tests
# ---------------------------------------------------------------------------

class TestSwitchedModel:

    def test_healthy_at_origin(self, switched_model):
        assert switched_model.classify_basin(np.zeros(switched_model._n)) == "healthy"

    def test_disease_at_positive_dysregulation(self, switched_model):
        x = np.ones(switched_model._n) * 5.0
        assert switched_model.classify_basin(x) == "disease"

    def test_simulation_produces_switches(self, switched_model):
        x0 = np.zeros(switched_model._n)
        x0[0] = 3.0
        times, states, basins = switched_model.simulate_switched(
            x0, T=50, seed=42
        )
        assert len(set(basins)) == 2

    def test_switched_simulation_shape(self, switched_model):
        x0 = np.zeros(switched_model._n)
        times, states, basins = switched_model.simulate_switched(
            x0, T=10, seed=42
        )
        assert states.shape[1] == switched_model._n
        assert len(basins) == len(times)

    def test_a_healthy_differs_from_a_disease(self, switched_model):
        diff = np.sum(np.abs(switched_model.A_healthy - switched_model.A_disease))
        assert diff > 1e-10

    def test_set_age_updates_both_matrices(self, switched_model):
        A_h_65 = switched_model.A_healthy.copy()
        switched_model.set_age(30)
        assert not np.allclose(A_h_65, switched_model.A_healthy)
