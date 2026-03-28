"""Load J coupling matrix from CSV.

The default CSV (data/J_matrix_compiled_9x9.csv) contains all 72
off-diagonal entries of the 9-axis mechanistic coupling matrix J_mech,
with basin-stratified values (healthy, pre-disease, disease) in SD-per-SD
units derived from systematic literature review.  The legacy 8-axis
version (data/J_matrix_compiled.csv, 56 entries) is retained for
reproducibility of prior analyses.

This module extracts any axis subset and applies a calibration scalar
to map SD-per-SD literature values to simulation coupling rates (day^-1).
"""

import os
import csv
import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _default_csv_path():
    """Return path to J_matrix_compiled_9x9.csv relative to the package root."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))          # src/hdr_sim/
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))         # repo root
    return os.path.join(repo_root, 'data', 'J_matrix_compiled_9x9.csv')


def load_J_csv(csv_path=None):
    """Load J_matrix_compiled.csv and return list of row dicts.

    Parameters
    ----------
    csv_path : str or None
        Path to CSV file.  If None, uses the default repo location.

    Returns
    -------
    rows : list[dict]
        Each dict has keys matching the CSV header columns.
    """
    if csv_path is None:
        csv_path = _default_csv_path()
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# J matrix construction
# ---------------------------------------------------------------------------

_BASIN_COL = {
    'healthy': 'J_healthy',
    'pre_disease': 'J_pre_disease',
    'disease': 'J_disease',
}


def build_J_basin(rows, basin='healthy', axes=('I', 'M', 'N', 'F')):
    """Build an n x n coupling matrix for a given basin and axis subset.

    Parameters
    ----------
    rows : list[dict]
        Output of :func:`load_J_csv`.
    basin : str
        One of 'healthy', 'pre_disease', 'disease'.
    axes : tuple[str]
        Axis labels in desired order.  Default is the 4-axis model
        ('I', 'M', 'N', 'F').  Works with any subset of the 9-axis
        model ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B').

    Returns
    -------
    J : np.ndarray, shape (n, n)
        Coupling matrix in SD-per-SD units.
        Convention: J[i,j] = effect of axes[j] on axes[i].
        Diagonal is zero.  Missing / unknown / qualitative-only entries
        are 0.0.
    """
    col = _BASIN_COL[basin]
    n = len(axes)
    axis_idx = {a: i for i, a in enumerate(axes)}
    J = np.zeros((n, n))

    for row in rows:
        src = row['axis_from'].strip()
        tgt = row['axis_to'].strip()
        if src not in axis_idx or tgt not in axis_idx:
            continue
        val_str = row[col].strip()
        if val_str in ('', 'NA', 'qual_only', 'unknown'):
            continue
        j = axis_idx[src]   # source = column
        i = axis_idx[tgt]   # target = row
        J[i, j] = float(val_str)

    np.fill_diagonal(J, 0.0)
    return J


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _spectral_abscissa(A):
    """Max real part of eigenvalues of A."""
    return float(np.max(np.real(np.linalg.eigvals(A))))


def _build_A(tau, J):
    """A = -diag(1/tau) + J."""
    return -np.diag(1.0 / tau) + J


def get_calibration_scalar(J_raw, tau, target_alpha):
    """Find scalar c such that alpha(build_A(tau, c*J_raw)) == target_alpha.

    Uses Brent's method.  Handles both net-destabilising coupling (α increases
    with c, e.g. pathological loops dominate) and net-stabilising coupling
    (α decreases with c, e.g. protective F column dominates).

    Parameters
    ----------
    J_raw : np.ndarray
        Unscaled coupling matrix (SD-per-SD from CSV).
    tau : np.ndarray
        Recovery time constants.
    target_alpha : float
        Desired spectral abscissa (must be negative).

    Returns
    -------
    c : float
        Calibration scalar.
    """
    def alpha_at(c):
        return _spectral_abscissa(_build_A(tau, c * J_raw))

    alpha_0 = alpha_at(0.0)

    # Find a bracket [c_lo, c_hi] where objective crosses zero
    def objective(c):
        return alpha_at(c) - target_alpha

    obj_0 = objective(0.0)

    # Search outward for a sign change
    c_hi = 1.0
    for _ in range(50):
        obj_hi = objective(c_hi)
        if obj_0 * obj_hi < 0:
            # Sign change found — bracket is [0, c_hi]
            return brentq(objective, 0.0, c_hi, xtol=1e-10)
        c_hi *= 2.0

    # No sign change found — alpha is monotone and doesn't reach target.
    # Return the c that gets closest.
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda c: abs(objective(c)), bounds=(0, c_hi), method='bounded')
    return res.x


# ---------------------------------------------------------------------------
# Convenience: calibrated J anchors for the 4-axis model
# ---------------------------------------------------------------------------

# Target spectral abscissa at age 30 — matches prior calibration (α ≈ −0.134)
_TARGET_ALPHA_30 = -0.134

# Recovery time constants (not in CSV — biologically motivated)
_TAU_30 = np.array([7.0, 0.1, 0.01, 8.0])
_TAU_80 = np.array([25.0, 0.30, 0.04, 42.0])

# Default 4-axis subset
_DEFAULT_AXES = ('I', 'M', 'N', 'F')


def _legacy_csv_path():
    """Return path to legacy 8-axis J_matrix_compiled.csv."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))
    return os.path.join(repo_root, 'data', 'J_matrix_compiled.csv')


def get_J_anchors(axes=_DEFAULT_AXES, target_alpha=_TARGET_ALPHA_30,
                  csv_path=None):
    """Return calibrated (J_30, J_80) anchor matrices loaded from CSV.

    The CSV's basin-stratified SD-per-SD values are scaled by a single
    calibration scalar ``c`` chosen so that the spectral abscissa at
    age 30 matches ``target_alpha``.  The same scalar is applied to
    J_disease (age 80 anchor), preserving the literature-derived ratio
    between healthy and disease coupling strengths.

    Parameters
    ----------
    axes : tuple[str]
        Axis labels.  Default ('I', 'M', 'N', 'F').
    target_alpha : float
        Target spectral abscissa at age 30.
    csv_path : str or None
        Path to CSV.  If None, uses the legacy 8-axis CSV
        (``data/J_matrix_compiled.csv``) to preserve existing
        simulation calibration.

    Returns
    -------
    J_30 : np.ndarray
        Calibrated coupling matrix for age 30 (healthy basin).
    J_80 : np.ndarray
        Calibrated coupling matrix for age 80 (disease basin).
    calibration_scalar : float
        The scalar applied: J_sim = c * J_csv.
    """
    rows = load_J_csv(csv_path or _legacy_csv_path())
    J_healthy = build_J_basin(rows, basin='healthy', axes=axes)
    J_disease = build_J_basin(rows, basin='disease', axes=axes)

    tau_young = _TAU_30 if len(axes) == 4 else _TAU_30[:len(axes)]
    c = get_calibration_scalar(J_healthy, tau_young, target_alpha)

    return c * J_healthy, c * J_disease, c
