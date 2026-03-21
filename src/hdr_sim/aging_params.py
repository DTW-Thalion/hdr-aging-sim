"""Age-parameterised τ_i(age) and J(age) functions.

J coupling matrices are loaded from data/J_matrix_compiled.csv (the systematic
literature-derived mechanistic coupling matrix) and scaled by a calibration
scalar to map SD-per-SD literature values to simulation coupling rates.

The CSV provides basin-stratified values (healthy / pre-disease / disease).
The healthy basin maps to the age 30 anchor, and the disease basin maps to
the age 80 anchor.  Linear interpolation is used for intermediate ages.
"""

import numpy as np
from .csv_loader import get_J_anchors

AXIS_NAMES = ['I (inflammaging)', 'M (metabolic)', 'N (neuroendocrine)', 'F (functional)']
AXIS_COLORS = ['#e74c3c', '#e67e22', '#3498db', '#27ae60']  # red, orange, blue, green

# Anchor values for τ at ages 30 and 80
# Biologically motivated (normalised time units where 1 unit ≈ 1 day)
_TAU_30 = np.array([7.0, 0.1, 0.01, 8.0])     # CRP ~1wk, glucose ~2-3h, HRR ~1-2min, muscle ~8d
_TAU_80 = np.array([25.0, 0.30, 0.04, 42.0])   # CRP ~3.5wk, glucose ~7h, HRR ~58min, muscle ~6wk

# Anchor values for J at ages 30 and 80
# Loaded from data/J_matrix_compiled.csv with calibration scalar applied.
# Convention: J[i,j] = effect of axis j on axis i.
#   Columns (j) = source axis: I, M, N, F
#   Rows (i) = target axis:    I, M, N, F
_J_30, _J_80, _CALIBRATION_SCALAR = get_J_anchors()


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
    age 30 (healthy basin) and age 80 (disease basin) anchors
    loaded from J_matrix_compiled.csv.
    """
    f = _interp_fraction(age)
    J = (1.0 - f) * _J_30 + f * _J_80
    np.fill_diagonal(J, 0.0)
    return J
