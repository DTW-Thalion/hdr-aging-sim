"""Tests for arbitrary axis subset handling (2×2 through 9×9)."""

import numpy as np
import pytest

from hdr_sim.csv_loader import (
    load_J_csv,
    build_J_basin,
    get_calibration_scalar,
    get_J_anchors,
    TAU_REGISTRY,
    _tau_for_axes,
    _default_csv_path,
)
from hdr_sim.aging_params import configure, reset, tau_of_age, J_of_age, get_axis_names


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rows_9x9():
    return load_J_csv()


@pytest.fixture(autouse=True)
def reset_config():
    """Reset aging_params config between tests."""
    yield
    reset()


# ---------------------------------------------------------------------------
# build_J_basin: various subsets
# ---------------------------------------------------------------------------

class TestBuildJBasin:
    def test_2x2_shape(self, rows_9x9):
        J = build_J_basin(rows_9x9, 'healthy', axes=('I', 'M'))
        assert J.shape == (2, 2)

    def test_2x2_diagonal_zero(self, rows_9x9):
        J = build_J_basin(rows_9x9, 'healthy', axes=('I', 'M'))
        np.testing.assert_array_equal(np.diag(J), [0.0, 0.0])

    def test_3x3_shape(self, rows_9x9):
        J = build_J_basin(rows_9x9, 'healthy', axes=('I', 'M', 'F'))
        assert J.shape == (3, 3)

    def test_3x3_sign_counts(self, rows_9x9):
        """I, M, F subset: verify signs match the full 9x9 for these axes."""
        J = build_J_basin(rows_9x9, 'healthy', axes=('I', 'M', 'F'))
        # I->M and M->I should be positive (pathological)
        assert J[1, 0] > 0  # M row, I col: I->M
        assert J[0, 1] > 0  # I row, M col: M->I
        # F->I and F->M should be negative (protective)
        assert J[0, 2] < 0  # I row, F col: F->I
        assert J[1, 2] < 0  # M row, F col: F->M

    def test_5x5_shape(self, rows_9x9):
        J = build_J_basin(rows_9x9, 'healthy', axes=('I', 'M', 'E', 'mito', 'F'))
        assert J.shape == (5, 5)

    def test_5x5_signs(self, rows_9x9):
        """5-axis subset: verify F column is negative (protective)."""
        axes = ('I', 'M', 'E', 'mito', 'F')
        J = build_J_basin(rows_9x9, 'healthy', axes=axes)
        f_col = axes.index('F')
        for i in range(5):
            if i != f_col and J[i, f_col] != 0:
                assert J[i, f_col] < 0, f"Expected F->{axes[i]} < 0, got {J[i, f_col]}"

    def test_full_9x9(self, rows_9x9):
        axes = ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B')
        J = build_J_basin(rows_9x9, 'healthy', axes=axes)
        assert J.shape == (9, 9)
        np.testing.assert_array_equal(np.diag(J), np.zeros(9))


# ---------------------------------------------------------------------------
# Calibration scalar: various dimensions
# ---------------------------------------------------------------------------

class TestCalibrationScalar:
    def test_2x2_calibration(self, rows_9x9):
        axes = ('I', 'M')
        J = build_J_basin(rows_9x9, 'healthy', axes=axes)
        tau, _ = _tau_for_axes(axes)
        c = get_calibration_scalar(J, tau, target_alpha=-0.134)
        assert c > 0
        # Verify calibrated alpha
        A = -np.diag(1.0 / tau) + c * J
        alpha = np.max(np.real(np.linalg.eigvals(A)))
        assert abs(alpha - (-0.134)) < 0.01

    def test_3x3_calibration(self, rows_9x9):
        axes = ('I', 'M', 'F')
        J = build_J_basin(rows_9x9, 'healthy', axes=axes)
        tau, _ = _tau_for_axes(axes)
        c = get_calibration_scalar(J, tau, target_alpha=-0.134)
        assert c > 0

    def test_5x5_calibration(self, rows_9x9):
        axes = ('I', 'M', 'E', 'mito', 'F')
        J = build_J_basin(rows_9x9, 'healthy', axes=axes)
        tau, _ = _tau_for_axes(axes)
        c = get_calibration_scalar(J, tau, target_alpha=-0.134)
        assert c > 0


# ---------------------------------------------------------------------------
# get_J_anchors with dynamic tau
# ---------------------------------------------------------------------------

class TestGetJAnchors:
    def test_2x2_anchors(self):
        J_30, J_80, c = get_J_anchors(axes=('I', 'M'))
        assert J_30.shape == (2, 2)
        assert J_80.shape == (2, 2)
        assert c > 0

    def test_3x3_anchors(self):
        J_30, J_80, c = get_J_anchors(axes=('I', 'M', 'F'))
        assert J_30.shape == (3, 3)
        assert c > 0

    def test_4x4_anchors_default(self):
        """Default 4-axis should still work."""
        J_30, J_80, c = get_J_anchors()
        assert J_30.shape == (4, 4)
        assert c > 0


# ---------------------------------------------------------------------------
# tau_for_axes
# ---------------------------------------------------------------------------

class TestTauForAxes:
    def test_2_axes(self):
        t30, t80 = _tau_for_axes(('I', 'M'))
        assert len(t30) == 2
        assert len(t80) == 2
        assert t30[0] == TAU_REGISTRY['I'][0]
        assert t80[1] == TAU_REGISTRY['M'][1]

    def test_unknown_axis_raises(self):
        with pytest.raises(ValueError, match="No τ entry"):
            _tau_for_axes(('I', 'UNKNOWN'))


# ---------------------------------------------------------------------------
# configure() with variable axes
# ---------------------------------------------------------------------------

class TestConfigureVariableAxes:
    def test_2axis_configure(self):
        configure(axes=('I', 'M'))
        tau = tau_of_age(50)
        assert tau.shape == (2,)
        J = J_of_age(50)
        assert J.shape == (2, 2)

    def test_3axis_configure(self):
        configure(axes=('I', 'M', 'F'))
        tau = tau_of_age(30)
        assert tau.shape == (3,)
        J = J_of_age(30)
        assert J.shape == (3, 3)

    def test_get_axis_names_2axis(self):
        configure(axes=('I', 'M'))
        names = get_axis_names()
        assert len(names) == 2
        assert 'inflammaging' in names[0].lower()

    def test_stability_2axis(self):
        """2-axis system should be stable at age 30."""
        configure(axes=('I', 'M'))
        tau = tau_of_age(30)
        J = J_of_age(30)
        A = -np.diag(1.0 / tau) + J
        alpha = np.max(np.real(np.linalg.eigvals(A)))
        assert alpha < 0, f"Expected stable system, got α={alpha}"
