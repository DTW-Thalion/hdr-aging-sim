"""Unit tests for stability conditions and dynamics."""

import numpy as np
import pytest
from scipy import linalg

from hdr_sim.dynamics import build_A, spectral_abscissa, recovery_timescale, spectral_radius_discrete
from hdr_sim.aging_params import configure, tau_of_age, J_of_age
from hdr_sim.csv_loader import load_J_csv, build_J_basin

# Explicit configuration for the 4-axis model
configure()


def _A_at_age(age):
    return build_A(tau_of_age(age), J_of_age(age))


def test_stable_system_has_negative_abscissa():
    """Young healthy system should have α < 0."""
    A = _A_at_age(30)
    alpha = spectral_abscissa(A)
    assert alpha < 0, f"Expected α < 0 at age 30, got {alpha}"


def test_aged_system_closer_to_zero():
    """α(age=80) > α(age=30), both negative."""
    alpha_30 = spectral_abscissa(_A_at_age(30))
    alpha_80 = spectral_abscissa(_A_at_age(80))
    assert alpha_30 < 0
    assert alpha_80 < 0
    assert alpha_80 > alpha_30, (
        f"Expected α(80) > α(30), got α(80)={alpha_80}, α(30)={alpha_30}"
    )


def test_diagonal_J_is_zero():
    """J matrix should have zero diagonal."""
    for age in [30, 50, 65, 80]:
        J = J_of_age(age)
        np.testing.assert_array_equal(np.diag(J), np.zeros(4),
                                      err_msg=f"J diagonal nonzero at age {age}")


def test_recovery_timescale_increases_with_age():
    """1/|α| at age 80 > 1/|α| at age 30."""
    rt_30 = recovery_timescale(_A_at_age(30))
    rt_80 = recovery_timescale(_A_at_age(80))
    assert rt_80 > rt_30, (
        f"Expected recovery timescale at 80 > 30, got {rt_80} vs {rt_30}"
    )


def test_f_column_is_negative():
    """F→I, F→M, F→N entries should all be negative (protective)."""
    for age in [30, 50, 80]:
        J = J_of_age(age)
        # F is column 3; rows 0,1,2 are I,M,N
        for i in range(3):
            assert J[i, 3] < 0, (
                f"Expected J[{i},3] < 0 at age {age}, got {J[i, 3]}"
            )


def test_spectral_radius_discrete():
    """ρ(e^{AΔt}) < 1 when α(A) < 0."""
    for age in [30, 50, 80]:
        A = _A_at_age(age)
        alpha = spectral_abscissa(A)
        assert alpha < 0, f"System not stable at age {age}"
        rho = spectral_radius_discrete(A, dt=1.0)
        assert rho < 1.0, (
            f"Expected ρ < 1 at age {age} (α={alpha}), got ρ={rho}"
        )


def test_calibration_alpha_range():
    """α should be in the target calibration range."""
    alpha_30 = spectral_abscissa(_A_at_age(30))
    alpha_80 = spectral_abscissa(_A_at_age(80))
    assert -0.20 <= alpha_30 <= -0.05, f"α(30) = {alpha_30} out of range [-0.20, -0.05]"
    assert -0.05 <= alpha_80 <= -0.005, f"α(80) = {alpha_80} out of range [-0.05, -0.005]"


def test_recovery_ratio():
    """Recovery timescale at 80 should be 3-15× that at 30."""
    rt_30 = recovery_timescale(_A_at_age(30))
    rt_80 = recovery_timescale(_A_at_age(80))
    ratio = rt_80 / rt_30
    assert 3.0 <= ratio <= 15.0, f"Recovery ratio = {ratio}, expected 3-15×"


def test_csv_loaded():
    """J matrices should be loaded from CSV, not hardcoded."""
    rows = load_J_csv()
    # Default CSV is the 9×9 matrix (72 off-diagonal entries)
    assert len(rows) == 72, f"Expected 72 CSV rows, got {len(rows)}"
    # Verify that J_of_age(30) signs match CSV healthy-basin signs
    # for the 4-axis subset used by the simulation model
    J_csv = build_J_basin(rows, 'healthy', ('I', 'M', 'N', 'F'))
    J_sim = J_of_age(30)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            if J_csv[i, j] != 0:
                assert np.sign(J_sim[i, j]) == np.sign(J_csv[i, j]), (
                    f"Sign mismatch at [{i},{j}]: sim={J_sim[i,j]}, csv={J_csv[i,j]}"
                )


def test_csv_basin_structure():
    """Disease-basin couplings should be stronger than healthy-basin."""
    rows = load_J_csv()
    J_h = build_J_basin(rows, 'healthy', ('I', 'M', 'N', 'F'))
    J_d = build_J_basin(rows, 'disease', ('I', 'M', 'N', 'F'))
    # Positive entries should be larger in disease basin
    mask = J_h > 0
    assert np.all(J_d[mask] >= J_h[mask]), (
        "Expected disease-basin positive couplings >= healthy-basin"
    )
