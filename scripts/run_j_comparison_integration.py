#!/usr/bin/env python3
"""Integration test for J-matrix parameterisation.

Runs the core validation pipeline twice — once with the provenance snapshot
and once with the current default CSV — then compares outputs.

Since the provenance file is a frozen copy of the compiled CSV, the two runs
should produce identical results.  Any diff indicates a bug in the J-matrix
plumbing.

Usage:
    python scripts/run_j_comparison_integration.py
"""

import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hdr_sim.j_matrix_spec import JMatrixSpec

PROVENANCE_CSV = os.path.join(ROOT, 'data', 'provenance', 'J_R6_ontology_v1.6.csv')
DEFAULT_CSV = os.path.join(ROOT, 'data', 'J_matrix_compiled_9x9.csv')

PROVENANCE_OUT = os.path.join(ROOT, 'outputs', 'R6_provenance')
LATEST_OUT = os.path.join(ROOT, 'outputs', 'latest')


def run_pipeline(j_matrix_path, output_dir, label):
    """Run the full pipeline with the given J-matrix, writing to output_dir."""
    print(f"\n{'=' * 60}")
    print(f"Running pipeline: {label}")
    print(f"  J-matrix: {j_matrix_path}")
    print(f"  Output:   {output_dir}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(ROOT, 'scripts', 'run_full_pipeline.py'),
        '--j-matrix', j_matrix_path,
        '--output-dir', output_dir,
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        return False

    print(f"  Completed in {elapsed:.1f}s")
    return True


def run_comparison(baseline_json, candidate_json, output_dir):
    """Run compare_j_runs.py on two pipeline outputs."""
    cmd = [
        sys.executable,
        os.path.join(ROOT, 'scripts', 'compare_j_runs.py'),
        '--baseline', baseline_json,
        '--candidate', candidate_json,
        '--output-dir', output_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    if result.returncode != 0:
        print(f"Comparison failed: {result.stderr[-500:]}")
        return None

    report_path = os.path.join(output_dir, 'j_comparison_report.json')
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    t_total = time.time()

    # Verify input files exist
    for path, label in [(PROVENANCE_CSV, 'Provenance CSV'),
                        (DEFAULT_CSV, 'Default CSV')]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            return 1

    # Build specs
    prov_spec = JMatrixSpec.from_csv(PROVENANCE_CSV)
    default_spec = JMatrixSpec.from_csv(DEFAULT_CSV)
    sha_match = prov_spec.sha256 == default_spec.sha256

    # Run pipeline twice
    ok1 = run_pipeline(PROVENANCE_CSV, PROVENANCE_OUT, 'Provenance (R6 ontology v1.6)')
    if not ok1:
        print("\nFATAL: Provenance pipeline run failed.")
        return 1

    ok2 = run_pipeline(DEFAULT_CSV, LATEST_OUT, 'Default (latest compiled)')
    if not ok2:
        print("\nFATAL: Default pipeline run failed.")
        return 1

    # Compare outputs
    print(f"\n{'=' * 60}")
    print("Comparing pipeline outputs...")
    print(f"{'=' * 60}")

    baseline_json = os.path.join(PROVENANCE_OUT, 'full_pipeline.json')
    candidate_json = os.path.join(LATEST_OUT, 'full_pipeline.json')

    comparison_out = os.path.join(ROOT, 'outputs')
    report = run_comparison(baseline_json, candidate_json, comparison_out)

    if report is None:
        print("\nFATAL: Comparison failed.")
        return 1

    # Extract results
    numerical_diffs = report.get('numerical_diffs', [])
    test_diffs = report.get('test_outcome_diffs', [])
    n_sig = sum(1 for d in numerical_diffs if d.get('significant'))

    elapsed = time.time() - t_total

    # Print summary
    print(f"\n{'=' * 60}")
    print("=== J-MATRIX COMPARISON INTEGRATION TEST ===")
    print(f"{'=' * 60}")
    print(f"Provenance: {PROVENANCE_CSV}")
    print(f"  SHA: {prov_spec.sha256[:16]}...")
    print(f"Default:    {DEFAULT_CSV}")
    print(f"  SHA: {default_spec.sha256[:16]}...")
    print(f"SHA match:  {'YES' if sha_match else 'NO'}")
    print(f"Numerical diffs: {len(numerical_diffs)} ({n_sig} significant)")
    print(f"Test outcome diffs: {len(test_diffs)}")
    print(f"Elapsed: {elapsed:.1f}s")

    passed = (n_sig == 0 and len(test_diffs) == 0)
    if passed:
        print(f"\nSTATUS: PASS — pipeline is correctly parameterised on J-matrix input")
    else:
        print(f"\nSTATUS: FAIL — unexpected differences detected")
        if numerical_diffs:
            print("\n  Significant numerical diffs:")
            for d in numerical_diffs:
                if d.get('significant'):
                    print(f"    {d['metric']}: {d['baseline']} -> {d['candidate']}")
        if test_diffs:
            print("\n  Test outcome diffs:")
            for d in test_diffs:
                print(f"    {d['test']}: {d['baseline']} -> {d['candidate']}")

    print(f"{'=' * 60}")
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
