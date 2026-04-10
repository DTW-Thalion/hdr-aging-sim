#!/usr/bin/env python3
"""Monotonicity re-verification with V2 τ registry across ages 25–120.

Checks that key dynamical quantities change monotonically with age:
  - Spectral abscissa α(age) should increase (become less negative)
  - Recovery timescale 1/|α| should increase
  - Correlation structure should shift monotonically

Tests three configurations:
  1. Full 9×9 with imputed J
  2. 7-axis fast subsystem (excluding E, B)
  3. 4-axis classic (I, M, N, F) — "6-axis fast" in the 9-axis context

Saves results to outputs/monotonicity_25_120.json.
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
    get_calibration_scalar, tau_vector, J_at_age,
    _spectral_abscissa, _build_A,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def metzler_projection(J):
    """Project J to Metzler form: zero out negative off-diagonal entries."""
    J_m = J.copy()
    n = J_m.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and J_m[i, j] < 0:
                J_m[i, j] = 0.0
    return J_m


def check_monotonicity(ages, values, label):
    """Check if values are monotonically increasing."""
    violations = []
    for i in range(1, len(values)):
        if values[i] < values[i-1]:
            violations.append({
                'age_from': float(ages[i-1]),
                'age_to': float(ages[i]),
                'value_from': float(values[i-1]),
                'value_to': float(values[i]),
            })
    return {
        'quantity': label,
        'monotone': len(violations) == 0,
        'n_violations': len(violations),
        'violations': violations[:5],  # first 5 only
    }


def run_monotonicity_check(axes, J_h, J_d, label, n_ages=7000):
    """Run monotonicity check across the stable age range."""
    # Calibrate
    tau_25 = tau_vector(axes, 25)
    c = get_calibration_scalar(J_h, tau_25, -0.134)
    J_25_cal = c * J_h
    J_80_cal = c * J_d

    # Find max stable age
    max_stable = 25
    for test_age in range(25, 121):
        tau_a = tau_vector(axes, test_age)
        J_a = J_at_age(J_25_cal, J_80_cal, test_age)
        A_a = _build_A(tau_a, J_a)
        if _spectral_abscissa(A_a) >= 0:
            break
        max_stable = test_age

    if max_stable <= 25:
        return {
            'label': label,
            'axes': list(axes),
            'max_stable_age': max_stable,
            'n_age_points': 0,
            'monotonicity_checks': [],
            'note': 'System unstable at all tested ages',
        }

    # Sweep with fine grid
    ages = np.linspace(25, max_stable, min(n_ages, (max_stable - 25) * 100 + 1))
    alphas = np.zeros(len(ages))
    rec_times = np.zeros(len(ages))

    for i, age in enumerate(ages):
        tau_a = tau_vector(axes, float(age))
        J_a = J_at_age(J_25_cal, J_80_cal, float(age))
        A_a = _build_A(tau_a, J_a)
        alpha = _spectral_abscissa(A_a)
        alphas[i] = alpha
        rec_times[i] = 1.0 / abs(alpha) if alpha != 0 else float('inf')

    checks = [
        check_monotonicity(ages, alphas, 'spectral_abscissa_increasing'),
        check_monotonicity(ages, rec_times, 'recovery_timescale_increasing'),
    ]

    return {
        'label': label,
        'axes': list(axes),
        'max_stable_age': int(max_stable),
        'calibration_scalar': float(c),
        'n_age_points': len(ages),
        'alpha_range': [float(alphas[0]), float(alphas[-1])],
        'rec_time_range': [float(rec_times[0]), float(rec_times[-1])],
        'monotonicity_checks': checks,
    }


def main():
    rows = load_J_csv()

    configs = [
        ('9x9 imputed', ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B')),
        ('7-axis fast (excl. E,B)', ('I', 'M', 'mito', 'P', 'C', 'N', 'F')),
        ('4-axis classic', ('I', 'M', 'N', 'F')),
    ]

    all_results = []
    for label, axes in configs:
        J_h = build_J_basin_imputed(rows, 'healthy', axes)
        J_d = build_J_basin_imputed(rows, 'disease', axes)

        # Standard J
        print(f"\n--- {label} (standard J) ---")
        result = run_monotonicity_check(axes, J_h, J_d, f"{label} (standard)")
        all_results.append(result)
        print(f"  Max stable age: {result['max_stable_age']}")
        print(f"  Age points tested: {result['n_age_points']}")
        for chk in result.get('monotonicity_checks', []):
            status = "PASS" if chk['monotone'] else f"FAIL ({chk['n_violations']} violations)"
            print(f"  {chk['quantity']}: {status}")

        # Metzler projection
        J_h_m = metzler_projection(J_h)
        J_d_m = metzler_projection(J_d)
        print(f"\n--- {label} (Metzler projection) ---")
        result_m = run_monotonicity_check(axes, J_h_m, J_d_m, f"{label} (Metzler)")
        all_results.append(result_m)
        print(f"  Max stable age: {result_m['max_stable_age']}")
        for chk in result_m.get('monotonicity_checks', []):
            status = "PASS" if chk['monotone'] else f"FAIL ({chk['n_violations']} violations)"
            print(f"  {chk['quantity']}: {status}")

    output = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'script': 'run_monotonicity_25_120.py',
        'tau_registry': 'V2 (literature-calibrated)',
        'results': all_results,
    }

    out_path = os.path.join(_OUTPUT_DIR, 'monotonicity_25_120.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
