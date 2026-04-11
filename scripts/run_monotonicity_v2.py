#!/usr/bin/env python3
"""Monotonicity verification v2: fast-subsystem with stable calibration.

Checks that alpha_fast, recovery timescale, and correlation structure
change monotonically with age under the two-timescale calibration.

Tests:
  1. 6-axis fast subsystem (I, M, P, C, N, F) — primary
  2. 6-axis Metzler projection (zero out negative off-diagonal)
  3. Spectral abscissa monotonicity
  4. Recovery timescale monotonicity

Saves to outputs/monotonicity_v2.json.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np
from datetime import datetime, timezone

from hdr_sim.csv_loader import (
    load_J_csv, build_J_basin_imputed,
    calibrate_stable_system, _extract_submatrix,
    _spectral_abscissa, _build_A, tau_vector,
    j_blend_fraction,
    _ALL_9_AXES, _FAST_7_AXES,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def metzler_projection(J):
    """Zero out negative off-diagonal entries."""
    J_m = J.copy()
    n = J_m.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and J_m[i, j] < 0:
                J_m[i, j] = 0.0
    return J_m


def check_monotonicity(values, label):
    """Check if values are monotonically non-decreasing."""
    violations = 0
    for i in range(1, len(values)):
        if values[i] < values[i-1] - 1e-12:
            violations += 1
    return {
        'quantity': label,
        'monotone': violations == 0,
        'n_violations': violations,
        'n_points': len(values),
    }


def run_monotonicity(cal, J_h_full, J_d_full, label, use_metzler=False,
                     n_ages=7000):
    """Run monotonicity check on the fast subsystem."""
    axes_fast = tuple(cal['axes_fast'])
    c = cal['c']
    amp = cal['amplitude']
    gamma = cal['gamma']

    J_fh = _extract_submatrix(J_h_full, _ALL_9_AXES, axes_fast)
    J_fd = _extract_submatrix(J_d_full, _ALL_9_AXES, axes_fast)

    if use_metzler:
        J_fh = metzler_projection(J_fh)
        J_fd = metzler_projection(J_fd)

    delta_J = J_fd - J_fh

    ages = np.linspace(25, 120, n_ages)
    alphas = np.zeros(len(ages))
    rec_times = np.zeros(len(ages))

    for i, age in enumerate(ages):
        tau_a = tau_vector(axes_fast, float(age))
        bl = j_blend_fraction(float(age), gamma)
        J_a = J_fh + amp * bl * delta_J
        A_a = _build_A(tau_a, c * J_a)
        alpha = _spectral_abscissa(A_a)
        alphas[i] = alpha
        rec_times[i] = 1.0 / abs(alpha) if alpha != 0 else float('inf')

    checks = [
        check_monotonicity(alphas, 'spectral_abscissa_increasing'),
        check_monotonicity(rec_times, 'recovery_timescale_increasing'),
    ]

    return {
        'label': label,
        'axes_fast': list(axes_fast),
        'use_metzler': use_metzler,
        'n_age_points': len(ages),
        'alpha_range': [float(alphas[0]), float(alphas[-1])],
        'rec_time_range': [float(rec_times[0]), float(rec_times[-1])],
        'all_stable': bool(np.all(alphas < 0)),
        'monotonicity_checks': checks,
    }


def main():
    rows = load_J_csv()
    J_h = build_J_basin_imputed(rows, 'healthy', _ALL_9_AXES)
    J_d = build_J_basin_imputed(rows, 'disease', _ALL_9_AXES)

    cal = calibrate_stable_system(J_h, J_d, axes_fast=_FAST_7_AXES)

    all_results = []

    configs = [
        ("7-axis fast (standard)", False),
        ("7-axis fast (Metzler)", True),
    ]

    for label, metzler in configs:
        print(f"\n--- {label} ---")
        result = run_monotonicity(cal, J_h, J_d, label,
                                  use_metzler=metzler, n_ages=7000)
        all_results.append(result)
        print(f"  All stable: {result['all_stable']}")
        print(f"  alpha range: [{result['alpha_range'][0]:.4f}, "
              f"{result['alpha_range'][1]:.4f}]")
        print(f"  Recovery time range: [{result['rec_time_range'][0]:.1f}d, "
              f"{result['rec_time_range'][1]:.1f}d]")
        for chk in result['monotonicity_checks']:
            status = "PASS" if chk['monotone'] else f"FAIL ({chk['n_violations']})"
            print(f"  {chk['quantity']}: {status}")

    output = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'script': 'run_monotonicity_v2.py',
        'calibration': cal,
        'results': all_results,
    }

    out_path = os.path.join(_OUTPUT_DIR, 'monotonicity_v2.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
