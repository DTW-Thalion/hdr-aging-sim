"""Age-parameterised tau_i(age) and J(age) functions.

The 9x9 compiled J matrix (``data/J_matrix_compiled_9x9.csv``) is loaded
and scaled by a calibration scalar to map SD-per-SD literature values to
simulation coupling rates.  PMID-cited tau values from the 3-anchor
registry (ages 25, 80, 120) with axis-specific trajectory shapes
(Gompertz, saturating-exp, piecewise-linear) provide recovery time
constants.  Gompertz-like J interpolation with a tunable blend amplitude
allows the coupling matrix to evolve with age.

``configure()`` calibrates the 7-axis fast subsystem (I, M, mito, P, C,
N, F) via ``calibrate_stable_system()``, guaranteeing stability at all ages
25-120.  Any requested axis subset is extracted from this calibrated system.

Call ``configure()`` before using ``tau_of_age()`` or ``J_of_age()``.
If called without prior configuration, auto-configures with defaults and
emits a DeprecationWarning.
"""

import warnings
import numpy as np
from .csv_loader import (
    load_J_csv, build_J_basin, build_J_basin_imputed,
    get_calibration_scalar, TAU_REGISTRY, TAU_REGISTRY_V2,
    tau_vector as _tau_vector_v2, J_at_age as _J_at_age_v2,
    calibrate_stable_system, build_system_at_age,
    _ALL_9_AXES, _FAST_7_AXES,
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
    """Initialise the aging parameters module.

    As of v2.5, uses the fast-subsystem calibration (7-axis, stable 25-120)
    and extracts the requested axis subset. This guarantees stable dynamics
    at all ages for tau_of_age() / J_of_age() / build_A() workflows.

    Must be called before any calls to tau_of_age() or J_of_age().
    Can be called multiple times to reconfigure.
    """
    global _config, _fast_cal
    from .j_matrix_spec import JMatrixSpec
    from .csv_loader import (j_at_age_blended, _default_csv_path)

    if axes is None:
        axes = ('I', 'M', 'N', 'F')

    csv_path = j_matrix_path or _default_csv_path()
    j_spec = JMatrixSpec.from_csv(csv_path)

    # Ensure fast-subsystem calibration is cached
    if _fast_cal is None:
        rows = load_J_csv(csv_path)
        J_h = build_J_basin_imputed(rows, 'healthy', _ALL_9_AXES)
        J_d = build_J_basin_imputed(rows, 'disease', _ALL_9_AXES)
        _fast_cal = calibrate_stable_system(J_h, J_d,
                                            axes_all=_ALL_9_AXES,
                                            axes_fast=_FAST_7_AXES)
        _fast_cal['J_h'] = J_h
        _fast_cal['J_d'] = J_d

    # Build subset-extraction indices
    all_axes = list(_ALL_9_AXES)
    sub_idx = [all_axes.index(a) for a in axes if a in all_axes]

    # Compute J anchors for the subset at ages 25 and 80
    c = _fast_cal['c']
    amp = _fast_cal['amplitude']
    gamma = _fast_cal.get('gamma', 0.038)
    J_25_full = j_at_age_blended(_fast_cal['J_h'], _fast_cal['J_d'], 25, c, amp, gamma)
    J_80_full = j_at_age_blended(_fast_cal['J_h'], _fast_cal['J_d'], 80, c, amp, gamma)

    _config = {
        'J_25': J_25_full[np.ix_(sub_idx, sub_idx)],
        'J_80': J_80_full[np.ix_(sub_idx, sub_idx)],
        'J_30': J_25_full[np.ix_(sub_idx, sub_idx)],  # legacy alias
        'calibration_scalar': c,
        'axes': axes,
        'tau_30': _tau_vector_v2(axes, 25),  # legacy alias
        'tau_80': _tau_vector_v2(axes, 80),
        'j_spec': j_spec,
        'v2': True,
        '_fast_cal': _fast_cal,
        '_sub_idx': sub_idx,
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


# ---------------------------------------------------------------------------
# Fast-subsystem convenience (V2)
# ---------------------------------------------------------------------------

_fast_cal = None  # cached calibration result


def get_fast_system(age, axes_fast=None):
    """Return (A_full, A_fast, alpha_fast, alpha_full) at a given age.

    Uses the 7-axis fast subsystem (I,M,mito,P,C,N,F) calibrated via
    calibrate_stable_system() for guaranteed stability ages 25-120.
    The calibration result is cached after the first call.

    Parameters
    ----------
    age : float
        Chronological age (25-120).
    axes_fast : tuple[str] or None
        Fast-subsystem axes. Default: _FAST_7_AXES.

    Returns
    -------
    A_full : ndarray (9x9)
    A_fast : ndarray (7x7 or len(axes_fast) x len(axes_fast))
    alpha_fast : float
    alpha_full : float
    """
    global _fast_cal
    if axes_fast is None:
        axes_fast = _FAST_7_AXES

    if _fast_cal is None:
        rows = load_J_csv()
        J_h = build_J_basin_imputed(rows, 'healthy', _ALL_9_AXES)
        J_d = build_J_basin_imputed(rows, 'disease', _ALL_9_AXES)
        _fast_cal = calibrate_stable_system(J_h, J_d,
                                            axes_all=_ALL_9_AXES,
                                            axes_fast=axes_fast)
        _fast_cal['J_h'] = J_h
        _fast_cal['J_d'] = J_d

    return build_system_at_age(
        age, _fast_cal['J_h'], _fast_cal['J_d'],
        _fast_cal['c'], _fast_cal['amplitude'],
        axes_all=_ALL_9_AXES, axes_fast=axes_fast,
    )


def get_fast_tau_J(age, axes_fast=None):
    """Return (tau, J, A) for the fast subsystem at a given age.

    Convenience function for scripts that need the raw components
    for D/J decomposition analyses.
    """
    from .csv_loader import tau_vector as _tv, j_at_age_blended, j_blend_fraction
    global _fast_cal
    if axes_fast is None:
        axes_fast = _FAST_7_AXES

    # Ensure calibration is cached
    if _fast_cal is None:
        get_fast_system(age, axes_fast)

    # Tau from V2 registry
    tau = _tv(axes_fast, age)

    # Calibrated J at age (fast-subsystem submatrix)
    all_axes = list(_ALL_9_AXES)
    fast_idx = [all_axes.index(a) for a in axes_fast]
    c = _fast_cal['c']
    amp = _fast_cal['amplitude']
    gamma = _fast_cal.get('gamma', 0.038)
    J_full = j_at_age_blended(_fast_cal['J_h'], _fast_cal['J_d'],
                               age, c, amp, gamma)
    J = J_full[np.ix_(fast_idx, fast_idx)]

    A = -np.diag(1.0 / tau) + J
    return tau, J, A
