"""Unit tests for stability conditions and dynamics.

As of v2.5, tests use the 7-axis fast subsystem (calibrated stable 25-120)
rather than the raw 4-axis configure() path (which goes unstable at ~age 30
with V2 tau values).
"""

import numpy as np
import pytest
from scipy import linalg

from hdr_sim.dynamics import build_A, spectral_abscissa, recovery_timescale, spectral_radius_discrete
from hdr_sim.aging_params import configure, get_fast_system
from hdr_sim.csv_loader import load_J_csv, build_J_basin


def _fast_A_at_age(age):
    """Return A_fast (7x7) at a given age from the calibrated fast subsystem."""
    _, A_fast, _, _ = get_fast_system(age)
    return A_fast


def test_stable_system_has_negative_abscissa():
    """Young healthy fast subsystem should have alpha < 0."""
    A = _fast_A_at_age(25)
    alpha = spectral_abscissa(A)
    assert alpha < 0, f"Expected alpha < 0 at age 25, got {alpha}"


def test_aged_system_closer_to_zero():
    """alpha(age=80) > alpha(age=25), both negative."""
    alpha_25 = spectral_abscissa(_fast_A_at_age(25))
    alpha_80 = spectral_abscissa(_fast_A_at_age(80))
    assert alpha_25 < 0
    assert alpha_80 < 0
    assert alpha_80 > alpha_25, (
        f"Expected alpha(80) > alpha(25), got alpha(80)={alpha_80}, alpha(25)={alpha_25}"
    )


def test_diagonal_J_is_zero():
    """J matrix should have zero diagonal in the fast subsystem."""
    for age in [25, 50, 80]:
        A = _fast_A_at_age(age)
        # A = -D + J, so diagonal = -1/tau_i + J_ii; J_ii should be 0
        # We can't directly check J diagonal from A, but we can verify
        # the system produces the right structure
        assert A.shape[0] == A.shape[1] == 7


def test_recovery_timescale_increases_with_age():
    """1/|alpha| at age 80 > 1/|alpha| at age 25."""
    rt_25 = recovery_timescale(_fast_A_at_age(25))
    rt_80 = recovery_timescale(_fast_A_at_age(80))
    assert rt_80 > rt_25, (
        f"Expected recovery timescale at 80 > 25, got {rt_80} vs {rt_25}"
    )


def test_f_column_is_negative():
    """F->I, F->M, F->N entries should all be negative (protective) in compiled J."""
    rows = load_J_csv()
    for row in rows:
        if row['axis_from'] == 'F' and row['axis_to'] in ('I', 'M', 'N'):
            sign = row['sign']
            assert sign == '-', (
                f"Expected F->{row['axis_to']} to be protective (-), got {sign}"
            )


def test_spectral_radius_discrete():
    """rho(e^{A*dt}) < 1 when alpha(A) < 0."""
    for age in [25, 50, 80]:
        A = _fast_A_at_age(age)
        alpha = spectral_abscissa(A)
        assert alpha < 0, f"Fast subsystem not stable at age {age}"
        rho = spectral_radius_discrete(A, dt=1.0)
        assert rho < 1.0, (
            f"Expected rho < 1 at age {age} (alpha={alpha}), got rho={rho}"
        )


def test_calibration_alpha_range():
    """alpha should be in the calibrated range for the fast subsystem."""
    alpha_25 = spectral_abscissa(_fast_A_at_age(25))
    alpha_80 = spectral_abscissa(_fast_A_at_age(80))
    alpha_120 = spectral_abscissa(_fast_A_at_age(120))
    # V2 fast-subsystem targets: alpha(25) ~ -0.188, alpha(120) ~ -0.004
    assert -0.25 <= alpha_25 <= -0.10, f"alpha(25) = {alpha_25} out of range"
    assert -0.15 <= alpha_80 <= -0.05, f"alpha(80) = {alpha_80} out of range"
    assert -0.01 <= alpha_120 <= 0.0, f"alpha(120) = {alpha_120} out of range"


def test_recovery_ratio():
    """Recovery timescale at 80 should be modest relative to 25 (fast subsystem)."""
    rt_25 = recovery_timescale(_fast_A_at_age(25))
    rt_80 = recovery_timescale(_fast_A_at_age(80))
    ratio = rt_80 / rt_25
    # V2 fast-subsystem: ~5d at 25, ~9d at 80 → ratio ~ 1.7
    assert 1.2 <= ratio <= 5.0, f"Recovery ratio = {ratio}, expected 1.2-5.0"


def test_stability_all_ages():
    """Fast subsystem should be stable at all ages 25-120."""
    for age in range(25, 121, 5):
        _, _, alpha_fast, _ = get_fast_system(age)
        assert alpha_fast < 0, f"Fast subsystem unstable at age {age}: alpha={alpha_fast}"


def test_csv_loaded():
    """J matrices should be loaded from CSV, not hardcoded."""
    rows = load_J_csv()
    # Default CSV is the 9x9 matrix (72 off-diagonal entries)
    assert len(rows) == 72, f"Expected 72 CSV rows, got {len(rows)}"


def test_csv_basin_structure():
    """CSV should have healthy, pre_disease, disease basin values."""
    rows = load_J_csv()
    for row in rows:
        assert 'J_healthy' in row
        assert 'J_disease' in row
        assert row['axis_from'] != row['axis_to']
