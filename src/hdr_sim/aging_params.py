"""Age-parameterised tau_i(age) and J(age) functions.

J coupling matrices are loaded from CSV and scaled by a calibration scalar
to map SD-per-SD literature values to simulation coupling rates.

Two configuration modes:

**Legacy** (``configure()``): The CSV provides basin-stratified values
(healthy / pre-disease / disease).  The healthy basin maps to the age 30
anchor, and the disease basin maps to the age 80 anchor.  Linear
interpolation is used for intermediate ages.

**V2** (``configure_v2()``): Uses the literature-calibrated TAU_REGISTRY_V2
with three anchors (ages 25, 80, 120) and axis-specific trajectory shapes
(Gompertz, saturating-exp, piecewise-linear).  J interpolation uses a
Gompertz-like trajectory.  Supports qual_only-imputed J matrices (68/72
nonzero).

Call ``configure()`` or ``configure_v2()`` before using ``tau_of_age()``
or ``J_of_age()``.  If called without prior configuration, auto-configures
with legacy defaults and emits a DeprecationWarning.
"""

import warnings
import numpy as np
from .csv_loader import (
    load_J_csv, build_J_basin, build_J_basin_imputed,
    get_calibration_scalar, TAU_REGISTRY, TAU_REGISTRY_V2,
    tau_vector as _tau_vector_v2, J_at_age as _J_at_age_v2,
)

# ---------------------------------------------------------------------------
# Axis display metadata
# ---------------------------------------------------------------------------
_AXIS_FULL_NAMES = {
    'I':    'I (inflammaging)',
    'M':    'M (metabolic)',
    'N':    'N (neuroendocrine)',
    'F':    'F (functional)',
    'E':    'E (epigenetic)',
    'mito': 'mito (mitochondrial)',
    'P':    'P (proteostatic)',
    'C':    'C (circadian)',
    'B':    'B (bone/body composition)',
}

_AXIS_COLORS_MAP = {
    'I':    '#e74c3c',  # red
    'M':    '#e67e22',  # orange
    'N':    '#3498db',  # blue
    'F':    '#27ae60',  # green
    'E':    '#9b59b6',  # purple
    'mito': '#f39c12',  # gold
    'P':    '#1abc9c',  # teal
    'C':    '#2980b9',  # dark blue
    'B':    '#95a5a6',  # grey
}

# Backward-compatible module-level lists for the default 4-axis model
AXIS_NAMES = [_AXIS_FULL_NAMES[a] for a in ('I', 'M', 'N', 'F')]
AXIS_COLORS = [_AXIS_COLORS_MAP[a] for a in ('I', 'M', 'N', 'F')]

# Re-export for backward compatibility
_TAU_REGISTRY = TAU_REGISTRY

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
            if ax not in TAU_REGISTRY:
                raise ValueError(
                    f"No τ entry for axis {ax!r}. "
                    f"Known axes: {sorted(TAU_REGISTRY.keys())}"
                )
            t30_val, t80_val = TAU_REGISTRY[ax]
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


def configure_v2(j_matrix_path=None, axes=None, target_alpha=-0.134,
                  impute_qual=True):
    """Initialise with the V2 literature-calibrated τ registry.

    Uses 3-anchor τ trajectories (ages 25–120) and Gompertz J interpolation.

    Parameters
    ----------
    j_matrix_path : str or None
        Path to J matrix CSV. If None, uses the 9x9 compiled CSV.
    axes : tuple[str] or None
        Axis subset. If None, uses all 9 axes.
    target_alpha : float
        Target spectral abscissa at age 25 for calibration.
    impute_qual : bool
        If True, impute qual_only entries from tier defaults.
    """
    global _config
    from .j_matrix_spec import JMatrixSpec

    if axes is None:
        axes = ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B')

    # Resolve CSV path
    if j_matrix_path is None:
        import os
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(pkg_dir))
        j_matrix_path = os.path.join(repo_root, 'data', 'J_matrix_compiled_9x9.csv')

    j_spec = JMatrixSpec.from_csv(j_matrix_path)

    rows = load_J_csv(j_matrix_path)
    build_fn = build_J_basin_imputed if impute_qual else build_J_basin
    J_healthy = build_fn(rows, basin='healthy', axes=axes)
    J_disease = build_fn(rows, basin='disease', axes=axes)

    tau_25 = _tau_vector_v2(axes, 25)
    calibration_scalar = get_calibration_scalar(J_healthy, tau_25, target_alpha)

    _config = {
        'J_25': calibration_scalar * J_healthy,
        'J_80': calibration_scalar * J_disease,
        # Legacy aliases for backward compatibility
        'J_30': calibration_scalar * J_healthy,
        'calibration_scalar': calibration_scalar,
        'axes': axes,
        'tau_30': tau_25,  # legacy alias
        'tau_80': _tau_vector_v2(axes, 80),
        'j_spec': j_spec,
        'v2': True,
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


def get_axis_names():
    """Return human-readable names for the currently configured axes."""
    _ensure_configured()
    return [_AXIS_FULL_NAMES.get(a, a) for a in _config['axes']]


def get_axis_colors():
    """Return colour hex codes for the currently configured axes."""
    _ensure_configured()
    return [_AXIS_COLORS_MAP.get(a, '#7f8c8d') for a in _config['axes']]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _interp_fraction(age: float) -> float:
    """Return interpolation fraction: 0 at age 30, 1 at age 80, clamped."""
    return np.clip((age - 30.0) / 50.0, 0.0, 1.0)


def tau_of_age(age: float) -> np.ndarray:
    """Return n-vector of τ_i at a given chronological age.

    If configured with V2 (configure_v2), uses literature-calibrated
    trajectory functions (Gompertz, saturating-exp, piecewise-linear).
    Otherwise uses linear interpolation between age 30 and 80 anchors.
    """
    _ensure_configured()
    if _config.get('v2'):
        return _tau_vector_v2(_config['axes'], age)
    f = _interp_fraction(age)
    return (1.0 - f) * _config['tau_30'] + f * _config['tau_80']


def J_of_age(age: float) -> np.ndarray:
    """Return n×n coupling matrix at a given age.

    Convention: J[i,j] = effect of axis j on axis i.
    Positive = pathological (dysfunction in j worsens i).
    Negative = protective (activity in j improves i).

    If configured with V2, uses Gompertz-like interpolation anchored
    at ages 25 and 80, with extrapolation beyond 80.
    Otherwise uses linear interpolation between age 30 and 80 anchors.
    """
    _ensure_configured()
    if _config.get('v2'):
        J = _J_at_age_v2(_config['J_25'], _config['J_80'], age)
        np.fill_diagonal(J, 0.0)
        return J
    f = _interp_fraction(age)
    J = (1.0 - f) * _config['J_30'] + f * _config['J_80']
    np.fill_diagonal(J, 0.0)
    return J
