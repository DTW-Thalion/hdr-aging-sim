"""Load J coupling matrix from CSV and provide age-dependent tau/J functions.

The default CSV (data/J_matrix_compiled_9x9.csv) contains all 72
off-diagonal entries of the 9-axis mechanistic coupling matrix J_mech,
with basin-stratified values (healthy, pre-disease, disease) in SD-per-SD
units derived from systematic literature review.  The legacy 8-axis
version (data/J_matrix_compiled.csv, 56 entries) is retained for
reproducibility of prior analyses.

This module extracts any axis subset and applies a calibration scalar
to map SD-per-SD literature values to simulation coupling rates (day^-1).

Two tau registries are available:
  - TAU_REGISTRY_LEGACY (aliased as TAU_REGISTRY): 2-anchor (ages 30/80),
    linear interpolation.  Original ad-hoc values.
  - TAU_REGISTRY_V2: 3-anchor (ages 25/80/120) with PMID-cited values and
    axis-specific trajectory shapes (Gompertz, saturating-exp, piecewise-linear).

Additional V2 features:
  - build_J_basin_imputed(): fills qual_only entries from tier defaults (68/72 fill)
  - J_at_age(): Gompertz-like J interpolation for ages 25-120
  - calibrate_three_point(): three-point calibration with Pyrkov targets

Fast-subsystem calibration (two-timescale architecture):
  - calibrate_stable_system(): joint (c, amplitude) optimization for 25-120
    stability on the 6-axis fast subsystem (I, M, P, C, N, F)
  - calibrate_fast_subsystem(): single-point c calibration at age 25
  - j_blend_fraction(), find_j_blend_amplitude(): Gompertz J trajectory
  - j_at_age_blended(), build_system_at_age(): two-timescale system builder
  - Axis constants: _ALL_9_AXES, _FAST_7_AXES, _FAST_6_AXES, _SLOW_3_AXES
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
# τ registry (LEGACY): ad-hoc recovery time constants (day units)
# keyed by axis label → (tau_30, tau_80).  Retained for reproducibility.
# ---------------------------------------------------------------------------
TAU_REGISTRY_LEGACY = {
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

# FROZEN: legacy alias retained for reproducibility of pre-v2.4 analyses.
# As of v2.5, the default TAU_REGISTRY points to TAU_REGISTRY_V2.
TAU_REGISTRY_LEGACY_FROZEN = TAU_REGISTRY_LEGACY


# ---------------------------------------------------------------------------
# Trajectory functions for age-dependent τ
# ---------------------------------------------------------------------------

def _gompertz_tau(tau0, gamma):
    """τ(a) = tau0 * exp(gamma * a). Fit tau0, gamma to match anchors."""
    return lambda a: tau0 * np.exp(gamma * a)


def _saturating_exp_tau(tau_young, tau_max, k):
    """τ(a) = tau_max - (tau_max - tau_young) * exp(-k * (a - 25))"""
    return lambda a: tau_max - (tau_max - tau_young) * np.exp(-k * (a - 25))


def _piecewise_linear_tau(t25, t80, t120):
    """Linear interpolation through 3 anchors at ages 25, 80, 120."""
    def f(a):
        a = np.clip(a, 25, 120)
        if np.isscalar(a):
            if a <= 80:
                return t25 + (t80 - t25) * (a - 25) / 55.0
            else:
                return t80 + (t120 - t80) * (a - 80) / 40.0
        else:
            result = np.where(
                a <= 80,
                t25 + (t80 - t25) * (a - 25) / 55.0,
                t80 + (t120 - t80) * (a - 80) / 40.0,
            )
            return result
    return f


# ---------------------------------------------------------------------------
# τ registry V2: literature-calibrated, 3-anchor (ages 25, 80, 120)
# ---------------------------------------------------------------------------

TAU_REGISTRY_V2 = {
    # axis: {tau_25, tau_80, tau_120, trajectory, pmid, trajectory_fn}
    'I': {
        'tau_25': 4.0, 'tau_80': 17.0, 'tau_120': 45.0,
        'trajectory': 'gompertz',
        'pmid': '27467771',
        'trajectory_fn': _gompertz_tau(tau0=2.07, gamma=0.0263),
    },
    'M': {
        'tau_25': 0.08, 'tau_80': 0.21, 'tau_120': 0.35,
        'trajectory': 'piecewise-linear',
        'pmid': '18268070',
        'trajectory_fn': _piecewise_linear_tau(0.08, 0.21, 0.35),
    },
    'E': {
        'tau_25': 500, 'tau_80': 2000, 'tau_120': 5000,
        'trajectory': 'piecewise-exp',
        'pmid': '15509558',
        'trajectory_fn': _piecewise_linear_tau(500, 2000, 5000),
    },
    # Structural turnover constants (NOT used in D matrix):
    # MITO_PROTEIN_HALFLIFE_YOUNG = 36  # days, Rooyackers 1996 PMID 8986817
    # MITO_PROTEIN_HALFLIFE_OLD = 57    # days, same study, age 73
    # These measure hardware replacement time, not functional recovery time.
    'mito': {
        'tau_25': 1.0, 'tau_80': 2.0, 'tau_120': 5.0,
        'trajectory': 'piecewise-linear',
        'pmid': '12563009',  # Pilegaard 2003 (PGC-1a signaling cycle)
        'pmid_supporting': ['18653753', '23187721', '10878112', '10481231'],
        'trajectory_fn': _piecewise_linear_tau(1.0, 2.0, 5.0),
        'note': ('Bioenergetic functional recovery (PGC-1a mRNA->protein->baseline), '
                 'NOT mitochondrial protein pool half-life (36d, Rooyackers 1996). '
                 'tauPCr = 22-33s (Conley 2000) confirms immediate bioenergetic '
                 'recovery; full signaling cycle ~1 day young, ~2 days old.'),
    },
    'P': {
        'tau_25': 1.5, 'tau_80': 3.0, 'tau_120': 4.0,
        'trajectory': 'piecewise-linear',
        'pmid': '24437518',
        'trajectory_fn': _piecewise_linear_tau(1.5, 3.0, 4.0),
    },
    'C': {
        'tau_25': 6.0, 'tau_80': 10.0, 'tau_120': 18.0,
        'trajectory': 'piecewise-linear',
        'pmid': '1557592',
        'trajectory_fn': _piecewise_linear_tau(6.0, 10.0, 18.0),
    },
    'N': {
        'tau_25': 0.003, 'tau_80': 0.005, 'tau_120': 0.008,
        'trajectory': 'piecewise-linear',
        'pmid': '29581219',
        'trajectory_fn': _piecewise_linear_tau(0.003, 0.005, 0.008),
    },
    'F': {
        'tau_25': 2.0, 'tau_80': 3.5, 'tau_120': 6.0,
        'trajectory': 'piecewise-gompertz',
        'pmid': '9252485',
        'trajectory_fn': _piecewise_linear_tau(2.0, 3.5, 6.0),
    },
    'B': {
        'tau_25': 135, 'tau_80': 250, 'tau_120': 500,
        'trajectory': 'piecewise-linear',
        'pmid': '3213608',
        'trajectory_fn': _piecewise_linear_tau(135, 250, 500),
    },
}

# Default TAU_REGISTRY now points to V2 (literature-calibrated, PMID-cited).
# As of v2.5, all new code should use TAU_REGISTRY (= V2).
TAU_REGISTRY = TAU_REGISTRY_V2


def tau_at_age(axis, age):
    """Return τ for a single axis at a given age using the V2 registry.

    Parameters
    ----------
    axis : str
        Axis label (e.g. 'I', 'M', 'mito').
    age : float
        Chronological age in years (25–120).

    Returns
    -------
    float
        Recovery time constant in days.
    """
    if axis not in TAU_REGISTRY_V2:
        raise ValueError(
            f"No V2 τ entry for axis {axis!r}. "
            f"Known axes: {sorted(TAU_REGISTRY_V2.keys())}"
        )
    return float(TAU_REGISTRY_V2[axis]['trajectory_fn'](age))


def tau_vector(axes, age):
    """Return τ ndarray for multiple axes at a given age (V2 registry).

    Parameters
    ----------
    axes : tuple[str]
        Axis labels.
    age : float
        Chronological age (25–120).

    Returns
    -------
    np.ndarray
        Shape (n,) vector of τ values in days.
    """
    return np.array([tau_at_age(ax, age) for ax in axes])


def _tau_for_axes(axes):
    """Return (tau_25, tau_80) arrays for the given axis labels.

    Uses the V2 registry (3-anchor dict format). Returns tau at ages
    25 and 80 for backward compatibility with code expecting 2-anchor tuples.
    New code should prefer tau_vector(axes, age) for arbitrary ages.
    """
    t25, t80 = [], []
    for ax in axes:
        if ax in TAU_REGISTRY_V2:
            entry = TAU_REGISTRY_V2[ax]
            t25.append(entry['tau_25'])
            t80.append(entry['tau_80'])
        elif ax in TAU_REGISTRY_LEGACY:
            # Fallback for any axis not yet in V2
            v30, v80 = TAU_REGISTRY_LEGACY[ax]
            t25.append(v30)
            t80.append(v80)
        else:
            raise ValueError(
                f"No tau entry for axis {ax!r}. "
                f"Known V2 axes: {sorted(TAU_REGISTRY_V2.keys())}"
            )
    return np.array(t25), np.array(t80)


# ---------------------------------------------------------------------------
# Convenience: calibrated J anchors for the 4-axis model
# ---------------------------------------------------------------------------

# Target spectral abscissa at age 30 — matches prior calibration (α ≈ −0.134)
_TARGET_ALPHA_30 = -0.134

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
        Axis labels.  Default ('I', 'M', 'N', 'F').  Works for any
        subset from 2 to 9 axes.
    target_alpha : float
        Target spectral abscissa at age 30.
    csv_path : str or None
        Path to CSV.  If None, uses the 9-axis compiled CSV
        (``data/J_matrix_compiled_9x9.csv``).

    Returns
    -------
    J_30 : np.ndarray
        Calibrated coupling matrix for age 30 (healthy basin).
    J_80 : np.ndarray
        Calibrated coupling matrix for age 80 (disease basin).
    calibration_scalar : float
        The scalar applied: J_sim = c * J_csv.
    """
    rows = load_J_csv(csv_path or _default_csv_path())
    J_healthy = build_J_basin(rows, basin='healthy', axes=axes)
    J_disease = build_J_basin(rows, basin='disease', axes=axes)

    tau_young, _ = _tau_for_axes(axes)
    c = get_calibration_scalar(J_healthy, tau_young, target_alpha)

    return c * J_healthy, c * J_disease, c


# ---------------------------------------------------------------------------
# Qual-only imputation (Step 3)
# ---------------------------------------------------------------------------

TIER_DEFAULTS = {
    'S': {'healthy': 0.20, 'disease': 0.40},
    'M': {'healthy': 0.10, 'disease': 0.20},
    'W': {'healthy': 0.05, 'disease': 0.10},
}


def build_J_basin_imputed(rows, basin='healthy', axes=('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B')):
    """Build J matrix with qual_only entries imputed from tier defaults.

    Entries with sign '+' or '-' and a magnitude tier (S/M/W) but
    qual_only numeric value get the tier default with appropriate sign.
    Entries with sign '?' remain 0.0.

    Parameters
    ----------
    rows : list[dict]
        Output of :func:`load_J_csv`.
    basin : str
        One of 'healthy', 'pre_disease', 'disease'.
    axes : tuple[str]
        Axis labels in desired order.

    Returns
    -------
    J : np.ndarray, shape (n, n)
        Coupling matrix with imputed qual_only entries.
    """
    col = _BASIN_COL[basin]
    # Map basin to tier default key
    tier_basin_key = 'disease' if basin == 'disease' else 'healthy'

    n = len(axes)
    axis_idx = {a: i for i, a in enumerate(axes)}
    J = np.zeros((n, n))

    for row in rows:
        src = row['axis_from'].strip()
        tgt = row['axis_to'].strip()
        if src not in axis_idx or tgt not in axis_idx:
            continue

        j = axis_idx[src]   # source = column
        i = axis_idx[tgt]   # target = row

        val_str = row[col].strip()
        sign = row.get('sign', '?').strip()
        tier = row.get('magnitude_tier', 'unknown').strip()

        if val_str not in ('', 'NA', 'qual_only', 'unknown'):
            J[i, j] = float(val_str)
        elif val_str == 'qual_only' and sign in ('+', '-') and tier in TIER_DEFAULTS:
            default_mag = TIER_DEFAULTS[tier][tier_basin_key]
            J[i, j] = default_mag if sign == '+' else -default_mag
        # else: leave as 0.0 (unknown sign or missing tier)

    np.fill_diagonal(J, 0.0)
    return J


# ---------------------------------------------------------------------------
# Three-point calibration (Step 2)
# ---------------------------------------------------------------------------

# Target α trajectory: α(a) = -0.134 * exp(-0.038 * (a - 25))
_TARGET_ALPHA_25 = -0.134
_TARGET_ALPHA_80 = -0.030
_TARGET_ALPHA_120 = -0.004


def calibrate_three_point(J_h, J_d, axes, targets=None):
    """Calibrate coupling scalar c at age 25 and verify at 80/120.

    Parameters
    ----------
    J_h : np.ndarray
        Unscaled healthy-basin J (anchored at age 25).
    J_d : np.ndarray
        Unscaled disease-basin J (anchored at age 80).
    axes : tuple[str]
        Axis labels.
    targets : dict or None
        {25: alpha_25, 80: alpha_80, 120: alpha_120}.  Defaults to Pyrkov.

    Returns
    -------
    c : float
        Calibration scalar.
    alpha_25 : float
        Achieved α at age 25.
    alpha_80 : float
        Achieved α at age 80.
    alpha_120 : float
        Achieved α at age 120.
    """
    import logging
    log = logging.getLogger(__name__)

    if targets is None:
        targets = {25: _TARGET_ALPHA_25, 80: _TARGET_ALPHA_80, 120: _TARGET_ALPHA_120}

    tau_25 = tau_vector(axes, 25)
    c = get_calibration_scalar(J_h, tau_25, targets[25])

    results = {}
    for age in (25, 80, 120):
        tau_a = tau_vector(axes, age)
        J_a = J_at_age(c * J_h, c * J_d, age)
        A_a = _build_A(tau_a, J_a)
        alpha_a = _spectral_abscissa(A_a)
        results[age] = alpha_a

        target_a = targets[age]
        ratio = abs(alpha_a / target_a) if target_a != 0 else float('inf')
        if ratio > 2.0 or ratio < 0.5:
            log.warning(
                f"α({age}) = {alpha_a:.4f}, target = {target_a:.4f} "
                f"(ratio = {ratio:.2f}× — outside 2× tolerance)"
            )

    return c, results[25], results[80], results[120]


# ---------------------------------------------------------------------------
# Gompertz-like J interpolation (Step 4)
# ---------------------------------------------------------------------------

def J_at_age(J_25, J_80, age, gompertz_gamma=0.038):
    """Interpolate J matrix at a given age using Gompertz-like trajectory.

    Pathological entries (J > 0): increase on Gompertz curve
    Protective entries (J < 0): weaken (toward zero) on inverse Gompertz
    Zero entries: remain zero

    Parameters
    ----------
    J_25 : np.ndarray
        Calibrated coupling matrix at age 25 (healthy basin).
    J_80 : np.ndarray
        Calibrated coupling matrix at age 80 (disease basin).
    age : float
        Chronological age (25–120).  Extrapolates beyond 80.
    gompertz_gamma : float
        Gompertz acceleration parameter.

    Returns
    -------
    J : np.ndarray
        Interpolated coupling matrix at the given age.
    """
    frac_gompertz = (np.exp(gompertz_gamma * (age - 25)) - 1) / \
                    (np.exp(gompertz_gamma * 55) - 1)
    frac = np.clip(frac_gompertz, 0, None)  # allow > 1 for extrapolation

    J = J_25 + (J_80 - J_25) * frac
    return J


# ---------------------------------------------------------------------------
# V2 convenience: get calibrated J anchors with new τ registry
# ---------------------------------------------------------------------------

def get_J_anchors_v2(axes=('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B'),
                     target_alpha=-0.134, csv_path=None, impute_qual=True):
    """Return calibrated (J_25, J_80, c) using the V2 τ registry.

    Parameters
    ----------
    axes : tuple[str]
        Axis labels.
    target_alpha : float
        Target α at age 25.
    csv_path : str or None
        Path to CSV.  If None, uses 9x9 compiled CSV.
    impute_qual : bool
        If True, use tier-default imputation for qual_only entries.

    Returns
    -------
    J_25 : np.ndarray
        Calibrated coupling matrix at age 25.
    J_80 : np.ndarray
        Calibrated coupling matrix at age 80.
    c : float
        Calibration scalar.
    """
    rows = load_J_csv(csv_path or _default_csv_path())
    build_fn = build_J_basin_imputed if impute_qual else build_J_basin
    J_healthy = build_fn(rows, basin='healthy', axes=axes)
    J_disease = build_fn(rows, basin='disease', axes=axes)

    tau_25 = tau_vector(axes, 25)
    c = get_calibration_scalar(J_healthy, tau_25, target_alpha)

    return c * J_healthy, c * J_disease, c


# ---------------------------------------------------------------------------
# Fast-subsystem calibration (two-timescale architecture)
# ---------------------------------------------------------------------------

_ALL_9_AXES = ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B')
_FAST_7_AXES = ('I', 'M', 'mito', 'P', 'C', 'N', 'F')
_FAST_6_AXES = ('I', 'M', 'P', 'C', 'N', 'F')
_SLOW_2_AXES = ('E', 'B')
_SLOW_3_AXES = ('E', 'mito', 'B')


def _extract_submatrix(J_full, axes_all, axes_sub):
    """Extract a submatrix for a subset of axes."""
    idx = [axes_all.index(a) for a in axes_sub]
    return J_full[np.ix_(idx, idx)]


def calibrate_fast_subsystem(
    J_full_h, J_full_d,
    axes_all=_ALL_9_AXES,
    axes_fast=_FAST_7_AXES,
    target_alpha_fast=-0.134,
):
    """Calibrate c on the fast subsystem at age 25.

    Finds c such that alpha(A_fast(25)) = target.  Does NOT guarantee
    stability at older ages — use calibrate_stable_system() for that.

    Parameters
    ----------
    J_full_h, J_full_d : np.ndarray
        Unscaled full healthy/disease-basin J.
    axes_all : tuple[str]
        All axis labels in J_full order.
    axes_fast : tuple[str]
        Fast-subsystem axis labels.
    target_alpha_fast : float
        Target spectral abscissa for the fast subsystem at age 25.

    Returns
    -------
    c : float
        Calibration scalar.
    alpha_fast_25 : float
        Achieved fast-subsystem alpha at age 25.
    alpha_full_25 : float
        Full-system alpha at age 25.
    """
    J_fast_h = _extract_submatrix(J_full_h, axes_all, axes_fast)
    tau_fast_25 = tau_vector(axes_fast, 25)

    c = get_calibration_scalar(J_fast_h, tau_fast_25, target_alpha_fast)

    # Verify fast subsystem
    A_fast = _build_A(tau_fast_25, c * J_fast_h)
    alpha_fast = _spectral_abscissa(A_fast)

    # Verify full system
    tau_full_25 = tau_vector(axes_all, 25)
    A_full = _build_A(tau_full_25, c * J_full_h)
    alpha_full = _spectral_abscissa(A_full)

    return c, float(alpha_fast), float(alpha_full)


def calibrate_stable_system(
    J_full_h, J_full_d,
    axes_all=_ALL_9_AXES,
    axes_fast=_FAST_6_AXES,
    target_alpha_120=-0.004,
    gamma=0.038,
):
    """Find (c, amplitude) ensuring fast-subsystem stability 25-120.

    Jointly calibrates the coupling scalar c and blend amplitude A so
    that alpha_fast(120) = target_alpha_120 and the fast subsystem
    remains stable at all ages 25-120.  Maximises |alpha_fast(25)| to
    get the strongest young-adult stability margin.

    The 6-axis fast subsystem (I, M, P, C, N, F) excludes the
    intermediate-timescale mito axis (tau=36-65d) which constrains the
    7-axis system.  Mito joins E and B in the slow/quasi-static cluster.

    Parameters
    ----------
    J_full_h, J_full_d : np.ndarray
        Unscaled full healthy/disease-basin J.
    axes_all : tuple[str]
        All axis labels.
    axes_fast : tuple[str]
        Fast-subsystem axis labels.
    target_alpha_120 : float
        Target alpha_fast at age 120.
    gamma : float
        Gompertz blending parameter.

    Returns
    -------
    dict with keys:
        c : float - calibration scalar
        amplitude : float - J blend amplitude
        gamma : float - Gompertz parameter
        axes_fast : tuple - fast axis labels
        alpha_fast_25 : float - achieved alpha at 25
        alpha_fast_120 : float - achieved alpha at 120
    """
    J_fh = _extract_submatrix(J_full_h, axes_all, axes_fast)
    J_fd = _extract_submatrix(J_full_d, axes_all, axes_fast)
    delta_J = J_fd - J_fh
    blend_120 = j_blend_fraction(120, gamma)
    tau_25 = tau_vector(axes_fast, 25)
    tau_120 = tau_vector(axes_fast, 120)

    # Find c_max: maximum c where alpha_fast(120)=0 with amp=0
    def alpha_120_amp0(c):
        return _spectral_abscissa(_build_A(tau_120, c * J_fh))

    # Search for c_max
    c_lo, c_hi = 0.0, 0.1
    while alpha_120_amp0(c_hi) < 0 and c_hi < 100:
        c_hi *= 2
    if alpha_120_amp0(c_hi) < 0:
        # Even c=100 doesn't destabilise — use that
        c_max = c_hi
    else:
        c_max = brentq(alpha_120_amp0, c_lo, c_hi)

    # Search over c values: for each c, find amplitude for target alpha(120)
    best = None
    n_search = 80
    for c_try in np.linspace(0.01, c_max * 0.99, n_search):
        a120_amp0 = _spectral_abscissa(_build_A(tau_120, c_try * J_fh))
        if a120_amp0 > target_alpha_120:
            continue  # already above target with amp=0

        a120_amp1 = _spectral_abscissa(
            _build_A(tau_120, c_try * (J_fh + blend_120 * delta_J)))
        if a120_amp1 <= target_alpha_120:
            amp = 1.0
        else:
            try:
                amp = brentq(
                    lambda a: _spectral_abscissa(
                        _build_A(tau_120, c_try * (J_fh + a * blend_120 * delta_J))
                    ) - target_alpha_120,
                    0.0, 1.0, xtol=1e-10)
            except ValueError:
                continue

        # Verify stability at all ages
        all_stable = True
        for age in range(25, 121):
            tau_a = tau_vector(axes_fast, age)
            bl = j_blend_fraction(age, gamma)
            J_a = J_fh + amp * bl * delta_J
            if _spectral_abscissa(_build_A(tau_a, c_try * J_a)) >= 0:
                all_stable = False
                break

        if not all_stable:
            continue

        a25 = _spectral_abscissa(_build_A(tau_25, c_try * J_fh))
        if best is None or a25 < best['alpha_fast_25']:
            best = {
                'c': float(c_try),
                'amplitude': float(amp),
                'gamma': gamma,
                'axes_fast': axes_fast,
                'alpha_fast_25': float(a25),
                'alpha_fast_120': float(target_alpha_120),
            }

    if best is None:
        raise RuntimeError(
            "Could not find stable (c, amplitude) pair for "
            f"axes_fast={axes_fast}, target_alpha_120={target_alpha_120}"
        )

    return best


def j_blend_fraction(age, gamma=0.038):
    """Gompertz-shaped blending from J_healthy (age 25) toward J_disease.

    f(25) = 0, increases slowly before 60, accelerates after.
    Normalised so f(120) = raw Gompertz value (not clamped to 1).
    """
    raw = (np.exp(gamma * (age - 25)) - 1) / (np.exp(gamma * 95) - 1)
    return max(0.0, raw)


def find_j_blend_amplitude(J_h, J_d, axes_fast, c, gamma=0.038,
                           target_alpha_120=-0.004):
    """Find the J blend amplitude A such that alpha_fast(120) = target.

    J(age) = c * [J_h + A * blend(age) * (J_d - J_h)]
    A = 1.0 means full J_disease reached at blend(120).
    A < 1.0 means the system never reaches full disease coupling.

    Parameters
    ----------
    J_h : np.ndarray
        Unscaled fast-subsystem healthy J.
    J_d : np.ndarray
        Unscaled fast-subsystem disease J.
    axes_fast : tuple[str]
        Fast-subsystem axis labels.
    c : float
        Calibration scalar.
    gamma : float
        Gompertz acceleration parameter.
    target_alpha_120 : float
        Target alpha_fast at age 120.

    Returns
    -------
    amplitude : float
        J blend amplitude.
    """
    delta_J = J_d - J_h
    blend_120 = j_blend_fraction(120, gamma)
    tau_fast_120 = tau_vector(axes_fast, 120)

    def alpha_at_120(amplitude):
        J_120 = J_h + amplitude * blend_120 * delta_J
        A = _build_A(tau_fast_120, c * J_120)
        return _spectral_abscissa(A)

    obj = lambda amp: alpha_at_120(amp) - target_alpha_120

    # Check bracket
    obj_0 = obj(0.0)
    obj_1 = obj(1.0)

    if obj_0 * obj_1 < 0:
        return float(brentq(obj, 0.0, 1.0, xtol=1e-10))
    elif obj_1 < 0:
        # Full amplitude still stable — try higher
        for hi in [2.0, 5.0, 10.0, 20.0]:
            if obj(hi) > 0:
                return float(brentq(obj, 1.0, hi, xtol=1e-10))
        return 10.0  # fallback
    else:
        # Even zero amplitude overshoots — return minimal
        return 0.0


def j_at_age_blended(J_h, J_d, age, c, amplitude, gamma=0.038):
    """Compute calibrated J matrix at a given age with Gompertz blending.

    Parameters
    ----------
    J_h : np.ndarray
        Unscaled healthy-basin J.
    J_d : np.ndarray
        Unscaled disease-basin J.
    age : float
        Chronological age (25-120).
    c : float
        Calibration scalar.
    amplitude : float
        J blend amplitude (from find_j_blend_amplitude).
    gamma : float
        Gompertz acceleration parameter.

    Returns
    -------
    J : np.ndarray
        Calibrated J matrix at the given age.
    """
    blend = j_blend_fraction(age, gamma)
    J = J_h + amplitude * blend * (J_d - J_h)
    return c * J


def build_system_at_age(age, J_h_full, J_d_full, c, amplitude,
                        axes_all=_ALL_9_AXES, axes_fast=_FAST_7_AXES,
                        gamma=0.038):
    """Build the full A matrix at a given age with two-timescale structure.

    Parameters
    ----------
    age : float
        Chronological age (25-120).
    J_h_full : np.ndarray
        Unscaled full healthy-basin J (9x9).
    J_d_full : np.ndarray
        Unscaled full disease-basin J (9x9).
    c : float
        Calibration scalar (from calibrate_fast_subsystem).
    amplitude : float
        J blend amplitude (from find_j_blend_amplitude).
    axes_all : tuple[str]
        All axis labels.
    axes_fast : tuple[str]
        Fast-subsystem axis labels.
    gamma : float
        Gompertz acceleration parameter.

    Returns
    -------
    A_full : np.ndarray
        Full 9x9 dynamics matrix.
    A_fast : np.ndarray
        7x7 fast-subsystem dynamics matrix.
    alpha_fast : float
        Fast-subsystem spectral abscissa.
    alpha_full : float
        Full-system spectral abscissa.
    """
    tau_full = tau_vector(axes_all, age)
    J_full = j_at_age_blended(J_h_full, J_d_full, age, c, amplitude, gamma)

    A_full = _build_A(tau_full, J_full)
    alpha_full = float(_spectral_abscissa(A_full))

    fast_idx = [axes_all.index(a) for a in axes_fast]
    A_fast = A_full[np.ix_(fast_idx, fast_idx)]
    alpha_fast = float(_spectral_abscissa(A_fast))

    return A_full, A_fast, alpha_fast, alpha_full
