#!/usr/bin/env python3
"""Generate comparison table: old vs new τ values and α trajectory.

Outputs to outputs/tau_comparison_old_vs_new.md.
"""

import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from hdr_sim.csv_loader import (
    TAU_REGISTRY_LEGACY, TAU_REGISTRY_V2,
    tau_vector, load_J_csv, build_J_basin, build_J_basin_imputed,
    get_calibration_scalar, J_at_age, _spectral_abscissa, _build_A,
    _tau_for_axes,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_DIR = os.path.join(_REPO_ROOT, 'outputs')
os.makedirs(_OUTPUT_DIR, exist_ok=True)


def main():
    lines = []
    lines.append("# Tau Registry Comparison: Legacy vs V2 (Literature-Calibrated)")
    lines.append("")
    lines.append("## Recovery Time Constants (days)")
    lines.append("")
    lines.append("| Axis | Legacy tau(30) | Legacy tau(80) | V2 tau(25) | V2 tau(80) | V2 tau(120) | Trajectory | PMID |")
    lines.append("|------|---------------|---------------|-----------|-----------|------------|------------|------|")

    all_axes = ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B')
    for ax in all_axes:
        t30, t80 = TAU_REGISTRY_LEGACY[ax]
        v2 = TAU_REGISTRY_V2[ax]
        lines.append(
            f"| {ax:4s} | {t30:11.3f} | {t80:11.3f} | {v2['tau_25']:7.3f} | "
            f"{v2['tau_80']:7.3f} | {v2['tau_120']:8.3f} | {v2['trajectory']:10s} | {v2['pmid']} |"
        )

    lines.append("")
    lines.append("## Key Changes")
    lines.append("")
    lines.append("| Axis | Change Factor (tau_25/tau_30) | Notes |")
    lines.append("|------|------------------------------|-------|")
    for ax in all_axes:
        t30_old = TAU_REGISTRY_LEGACY[ax][0]
        t25_new = TAU_REGISTRY_V2[ax]['tau_25']
        ratio = t25_new / t30_old
        note = ""
        if ratio > 5:
            note = f"**{ratio:.0f}× increase** — order-of-magnitude correction"
        elif ratio < 0.3:
            note = f"**{ratio:.2f}× decrease** — order-of-magnitude correction"
        elif abs(ratio - 1.0) > 0.3:
            note = f"{ratio:.2f}× change"
        else:
            note = f"~{ratio:.2f}× (minor adjustment)"
        lines.append(f"| {ax:4s} | {ratio:24.2f} | {note} |")

    # Spectral abscissa comparison with 4-axis model
    lines.append("")
    lines.append("## Spectral Abscissa Trajectory (4-axis: I, M, N, F)")
    lines.append("")
    # Note: using ASCII-only 'alpha' to avoid encoding issues on Windows

    axes4 = ('I', 'M', 'N', 'F')
    rows = load_J_csv()

    # Legacy
    J_h_legacy = build_J_basin(rows, 'healthy', axes4)
    J_d_legacy = build_J_basin(rows, 'disease', axes4)
    tau_30_legacy, tau_80_legacy = _tau_for_axes(axes4)
    c_legacy = get_calibration_scalar(J_h_legacy, tau_30_legacy, -0.134)

    # V2
    J_h_v2 = build_J_basin_imputed(rows, 'healthy', axes4)
    J_d_v2 = build_J_basin_imputed(rows, 'disease', axes4)
    tau_25_v2 = tau_vector(axes4, 25)
    c_v2 = get_calibration_scalar(J_h_v2, tau_25_v2, -0.134)

    lines.append(f"Legacy calibration scalar: c = {c_legacy:.6f}")
    lines.append(f"V2 calibration scalar: c = {c_v2:.6f}")
    lines.append("")
    lines.append("| Age | alpha (Legacy) | alpha (V2) | Pyrkov Target | Legacy Stable? | V2 Stable? |")
    lines.append("|-----|---------------|-----------|--------------|----------------|------------|")

    for age in [25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]:
        # Pyrkov target
        pyrkov = -0.134 * np.exp(-0.038 * (age - 25))

        # Legacy: linear interp between age 30 and 80
        f_legacy = np.clip((age - 30.0) / 50.0, 0.0, 1.0)
        tau_leg = (1.0 - f_legacy) * tau_30_legacy + f_legacy * tau_80_legacy
        J_leg = (1.0 - f_legacy) * (c_legacy * J_h_legacy) + f_legacy * (c_legacy * J_d_legacy)
        A_leg = _build_A(tau_leg, J_leg)
        alpha_leg = _spectral_abscissa(A_leg)

        # V2: Gompertz interp
        tau_v2a = tau_vector(axes4, age)
        J_v2a = J_at_age(c_v2 * J_h_v2, c_v2 * J_d_v2, age)
        A_v2a = _build_A(tau_v2a, J_v2a)
        alpha_v2 = _spectral_abscissa(A_v2a)

        lines.append(
            f"| {age:3d} | {alpha_leg:9.4f} | {alpha_v2:6.4f} | {pyrkov:12.4f} | "
            f"{'Yes' if alpha_leg < 0 else 'NO':14s} | {'Yes' if alpha_v2 < 0 else 'NO':10s} |"
        )

    lines.append("")
    lines.append("## J Matrix Fill Comparison")
    lines.append("")
    axes9 = ('I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B')
    J_orig = build_J_basin(rows, 'healthy', axes9)
    J_imp = build_J_basin_imputed(rows, 'healthy', axes9)
    lines.append(f"- Original (no imputation): {np.count_nonzero(J_orig)}/72 nonzero (58% fill)")
    lines.append(f"- Imputed (tier defaults): {np.count_nonzero(J_imp)}/72 nonzero (94% fill)")
    lines.append(f"- Unknown sign entries (remain 0): 4/72")

    out_path = os.path.join(_OUTPUT_DIR, 'tau_comparison_old_vs_new.md')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Comparison table saved to {out_path}")
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
