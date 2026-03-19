"""Age-parameterised τ_i(age) and J(age) functions."""

import numpy as np

AXIS_NAMES = ['I (inflammaging)', 'M (metabolic)', 'N (neuroendocrine)', 'F (functional)']
AXIS_COLORS = ['#e74c3c', '#e67e22', '#3498db', '#27ae60']  # red, orange, blue, green

# Anchor values for τ at ages 30 and 80
# Biologically motivated (normalised time units where 1 unit ≈ 1 day)
_TAU_30 = np.array([7.0, 0.1, 0.01, 8.0])     # CRP ~1wk, glucose ~2-3h, HRR ~1-2min, muscle ~8d
_TAU_80 = np.array([25.0, 0.30, 0.04, 42.0])   # CRP ~3.5wk, glucose ~7h, HRR ~58min, muscle ~6wk

# Anchor values for J at ages 30 and 80
#          I      M      N      F
_J_30 = np.array([
    [0.00, 0.02, 0.01, -0.05],  # I row: F→I protective (exercise anti-inflammatory)
    [0.02, 0.00, 0.01, -0.06],  # M row: F→M protective (exercise insulin sensitivity)
    [0.01, 0.01, 0.00, -0.04],  # N row: F→N protective (exercise vagal tone)
    [0.01, 0.02, 0.01,  0.00],  # F row: weak pathological
])

_J_80 = np.array([
    [0.00, 0.14, 0.07, -0.02],  # I row: I↔M 7× stronger; N→I 7× (Kortebein 2007)
    [0.14, 0.00, 0.05, -0.02],  # M row: I↔M 7× stronger; F→M weakened
    [0.06, 0.05, 0.00, -0.01],  # N row: stronger; F→N weakened
    [0.06, 0.08, 0.04,  0.00],  # F row: all → F stronger (sarcopenia drivers)
])


def _interp_fraction(age: float) -> float:
    """Return interpolation fraction: 0 at age 30, 1 at age 80, clamped."""
    return np.clip((age - 30.0) / 50.0, 0.0, 1.0)


def tau_of_age(age: float) -> np.ndarray:
    """Return 4-vector of τ_i at a given chronological age.

    Uses linear interpolation between age 30 and age 80 anchors.
    """
    f = _interp_fraction(age)
    return (1.0 - f) * _TAU_30 + f * _TAU_80


def J_of_age(age: float) -> np.ndarray:
    """Return 4×4 coupling matrix at a given age.

    Convention: J[i,j] = effect of axis j on axis i.
    Positive = pathological (dysfunction in j worsens i).
    Negative = protective (activity in j improves i).

    Diagonal is always 0. Uses linear interpolation between
    age 30 and age 80 anchors.
    """
    f = _interp_fraction(age)
    J = (1.0 - f) * _J_30 + f * _J_80
    np.fill_diagonal(J, 0.0)
    return J
