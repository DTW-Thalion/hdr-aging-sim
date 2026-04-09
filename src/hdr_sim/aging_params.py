"""Age-parameterised τ_i(age) and J(age) functions.

J coupling matrices are loaded from CSV and scaled by a calibration scalar
to map SD-per-SD literature values to simulation coupling rates.

The CSV provides basin-stratified values (healthy / pre-disease / disease).
The healthy basin maps to the age 30 anchor, and the disease basin maps to
the age 80 anchor.  Linear interpolation is used for intermediate ages.

Call ``configure()`` before using ``tau_of_age()`` or ``J_of_age()``.
If called without prior configuration, auto-configures with defaults and
emits a DeprecationWarning.
"""

import warnings
import numpy as np
from .csv_loader import load_J_csv, build_J_basin, get_calibration_scalar

AXIS_NAMES = ['I (inflammaging)', 'M (metabolic)', 'N (neuroendocrine)', 'F (functional)']
AXIS_COLORS = ['#e74c3c', '#e67e22', '#3498db', '#27ae60']  # red, orange, blue, green

# ---------------------------------------------------------------------------
# τ registry: biologically motivated recovery time constants (day units)
# keyed by axis label → (tau_30, tau_80)
# ---------------------------------------------------------------------------
_TAU_REGISTRY = {
    'I':    (7.0,    25.0),    # CRP ~1wk / ~3.5wk
    'M':    (0.1,    0.3),     # glucose ~2-3h / ~7h
    'N':    (0.01,   0.04),    # HRR ~1-2min / ~58min
    'F':    (8.0,    42.0),    # muscle ~8d / ~6wk
    'E':    (1000.0, 1500.0),  # epigenetic ~years
    'mito': (1.0,    3.0),     # mitochondrial ~days
    'P':    (0.5,    2.0),     # proteostasis ~days
    'C':    (1.0,    3.0),     # cellular senescence ~days
    'B':    (90.0,   120.0),   # bone/body composition ~months
}

# ---------------------------------------------------------------------------
# Lazy configuration state
# ---------------------------------------------------------------------------
_config = None


def configure(j_matrix_path=None, axes=None, target_alpha=-0.134,
              tau_30=None, tau_80=None):
    """Initialise the aging parameters module with a specific J-matrix.

    Must be called before any calls to tau_of_age() or J_of_age().
    Can be called multiple times to reconfigure (e.g., switching J versions).

    Parameters
    ----------
    j_matrix_path : str or None
        Path to J matrix CSV. If None, uses the legacy 8-axis CSV
        (data/J_matrix_compiled.csv) for backward compatibility with
        the original 4-axis calibration.
    axes : tuple[str] or None
        Axis subset. If None, uses ('I', 'M', 'N', 'F').
    target_alpha : float
        Target spectral abscissa at age 30 for calibration.
    tau_30, tau_80 : np.ndarray or None
        Recovery time constants. If None, looks up from the τ registry.
    """
    global _config
    from .j_matrix_spec import JMatrixSpec

    if axes is None:
        axes = ('I', 'M', 'N', 'F')

    # Resolve CSV path
    if j_matrix_path is None:
        import os
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(pkg_dir))
        j_matrix_path = os.path.join(repo_root, 'data', 'J_matrix_compiled.csv')

    # Build J-matrix spec for provenance
    j_spec = JMatrixSpec.from_csv(j_matrix_path)

    # Load CSV and build basin matrices
    rows = load_J_csv(j_matrix_path)
    J_healthy = build_J_basin(rows, basin='healthy', axes=axes)
    J_disease = build_J_basin(rows, basin='disease', axes=axes)

    # Resolve tau vectors
    if tau_30 is None or tau_80 is None:
        t30_list, t80_list = [], []
        for ax in axes:
            if ax not in _TAU_REGISTRY:
                raise ValueError(
                    f"No τ entry for axis {ax!r}. "
                    f"Known axes: {sorted(_TAU_REGISTRY.keys())}"
                )
            t30_val, t80_val = _TAU_REGISTRY[ax]
            t30_list.append(t30_val)
            t80_list.append(t80_val)
        if tau_30 is None:
            tau_30 = np.array(t30_list)
        if tau_80 is None:
            tau_80 = np.array(t80_list)

    # Calibrate
    calibration_scalar = get_calibration_scalar(J_healthy, tau_30, target_alpha)

    _config = {
        'J_30': calibration_scalar * J_healthy,
        'J_80': calibration_scalar * J_disease,
        'calibration_scalar': calibration_scalar,
        'axes': axes,
        'tau_30': tau_30,
        'tau_80': tau_80,
        'j_spec': j_spec,
    }


def get_config():
    """Return the current configuration dict, or None if not configured."""
    return _config


def reset():
    """Clear configuration (for testing)."""
    global _config
    _config = None


# ---------------------------------------------------------------------------
# Backward-compatible module-level constants
# These are properties that auto-configure on first access.
# ---------------------------------------------------------------------------
_TAU_30 = None  # will be populated by configure()
_TAU_80 = None


def _ensure_configured():
    """Auto-configure with defaults if not yet configured, with deprecation warning."""
    if _config is None:
        warnings.warn(
            "Implicit configuration is deprecated. "
            "Call hdr_sim.configure() explicitly.",
            DeprecationWarning,
            stacklevel=3,
        )
        configure()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _interp_fraction(age: float) -> float:
    """Return interpolation fraction: 0 at age 30, 1 at age 80, clamped."""
    return np.clip((age - 30.0) / 50.0, 0.0, 1.0)


def tau_of_age(age: float) -> np.ndarray:
    """Return n-vector of τ_i at a given chronological age.

    Uses linear interpolation between age 30 and age 80 anchors.
    """
    _ensure_configured()
    f = _interp_fraction(age)
    return (1.0 - f) * _config['tau_30'] + f * _config['tau_80']


def J_of_age(age: float) -> np.ndarray:
    """Return n×n coupling matrix at a given age.

    Convention: J[i,j] = effect of axis j on axis i.
    Positive = pathological (dysfunction in j worsens i).
    Negative = protective (activity in j improves i).

    Diagonal is always 0. Uses linear interpolation between
    age 30 (healthy basin) and age 80 (disease basin) anchors.
    """
    _ensure_configured()
    f = _interp_fraction(age)
    J = (1.0 - f) * _config['J_30'] + f * _config['J_80']
    np.fill_diagonal(J, 0.0)
    return J
