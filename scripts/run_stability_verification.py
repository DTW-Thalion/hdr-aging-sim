#!/usr/bin/env python3
"""Stability verification across ages 25–120 with the V2 τ registry.

Tests multiple axis subsets:
  1. Full 9×9 with imputed J (68/72 nonzero)
  2. Full 9×9 with original J (42/72 nonzero)
  3. 7-axis fast subsystem (excluding E, B)
  4. 4-axis classic (I, M, N, F)

Reports achieved α vs Pyrkov target trajectory and dominant recovery timescale.
Saves results to outputs/stability_verification_25_120.json.
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
    load_J_csv, build_J_basin, build_J_basin_imputed,
    get_calibration_scalar, tau_vector, J_at_age,
    TAU_REGISTRY_V2, _spectral_abscissa, _build_A,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def pyrkov_target(age):
    """Pyrkov 2021 target trajectory: alpha(a) = -0.134 * exp(-0.038 * (a - 25))."""
    return -0.134 * np.exp(-0.038 * (age - 25))


def run_stability_sweep(axes, J_h, J_d, label):
    """Calibrate at age 25 and sweep stability across ages."""
    ages = [25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    tau_25 = tau_vector(axes, 25)

    # Calibrate at age 25
    target_alpha_25 = -0.134
    c = get_calibration_scalar(J_h, tau_25, target_alpha_25)

    J_25_cal = c * J_h
    J_80_cal = c * J_d

    results = []
    for age in ages:
        tau_a = tau_vector(axes, age)
        J_a = J_at_age(J_25_cal, J_80_cal, age)
        A_a = _build_A(tau_a, J_a)
        alpha_a = float(_spectral_abscissa(A_a))
        target = float(pyrkov_target(age))
        recovery = float(1.0 / abs(alpha_a)) if alpha_a != 0 else float('inf')

        results.append({
            'age': age,
            'alpha': alpha_a,
            'alpha_target': target,
            'stable': alpha_a < 0,
            'recovery_timescale_days': recovery,
        })

    # Find max stable age
    max_stable = 25
    for r in results:
        if r['stable']:
            max_stable = r['age']
        else:
            break

    return {
        'label': label,
        'axes': list(axes),
        'n_axes': len(axes),
        'calibration_scalar': float(c),
        'alpha_at_25': results[0]['alpha'],
        'max_stable_age': max_stable,
        'all_stable': all(r['stable'] for r in results),
        'age_sweep': results,
    }


def main():
    rows = load_J_csv()

    configs = [
        ('9x9 imputed (68/72)', ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B'),
         build_J_basin_imputed, build_J_basin_imputed),
        ('9x9 original (42/72)', ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B'),
         build_J_basin, build_J_basin),
        ('7-axis fast (excl. E,B)', ('I', 'M', 'mito', 'P', 'C', 'N', 'F'),
         build_J_basin_imputed, build_J_basin_imputed),
        ('4-axis classic (I,M,N,F)', ('I', 'M', 'N', 'F'),
         build_J_basin_imputed, build_J_basin_imputed),
    ]

    all_results = []
    for label, axes, build_h, build_d in configs:
        J_h = build_h(rows, 'healthy', axes)
        J_d = build_d(rows, 'disease', axes)
        nz = np.count_nonzero(J_h)
        n_off = len(axes) * (len(axes) - 1)

        print(f"\n{'='*60}")
        print(f"  {label}  ({nz}/{n_off} nonzero)")
        print(f"{'='*60}")

        result = run_stability_sweep(axes, J_h, J_d, label)
        result['nonzero_entries'] = int(nz)
        result['total_off_diag'] = int(n_off)
        all_results.append(result)

        print(f"  Calibration scalar c = {result['calibration_scalar']:.6f}")
        print(f"  Max stable age = {result['max_stable_age']}")
        print(f"  All ages stable? {result['all_stable']}")
        print(f"  {'Age':>5} | {'alpha':>10} | {'target':>10} | {'stable':>6} | {'t_rec (d)':>10}")
        print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}-+-{'-'*10}")
        for r in result['age_sweep']:
            t_rec = f"{r['recovery_timescale_days']:.1f}" if r['recovery_timescale_days'] < 1e6 else "inf"
            print(f"  {r['age']:5d} | {r['alpha']:10.4f} | {r['alpha_target']:10.4f} | "
                  f"{'YES' if r['stable'] else 'NO':>6} | {t_rec:>10}")

    # Save results
    output = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'script': 'run_stability_verification.py',
        'tau_registry': 'V2 (literature-calibrated)',
        'pyrkov_target': 'alpha(a) = -0.134 * exp(-0.038 * (a - 25))',
        'results': all_results,
    }

    out_path = os.path.join(_OUTPUT_DIR, 'stability_verification_25_120.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
