#!/usr/bin/env python3
"""Compare pipeline outputs produced by different J-matrix versions.

Usage:
    python scripts/compare_j_runs.py \\
        --baseline outputs/R6_provenance/elsa_results_ledger.json \\
        --candidate outputs/latest/elsa_results_ledger.json

    python scripts/compare_j_runs.py \\
        --baseline results/full_pipeline_v1.json \\
        --candidate results/full_pipeline_v2.json

Produces:
    outputs/j_comparison_report.json  -- structured diff
    outputs/j_comparison_report.md    -- human-readable report
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare pipeline outputs from different J-matrix versions')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to baseline pipeline JSON output')
    parser.add_argument('--candidate', type=str, required=True,
                        help='Path to candidate pipeline JSON output')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory. Default: outputs/')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Absolute change threshold for "significant". Default: 0.01')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_j_metadata(data):
    """Extract J-matrix metadata from pipeline output."""
    j = data.get('j_matrix', {})
    return {
        'sha256': j.get('sha256', 'unknown'),
        'n_axes': j.get('n_axes', None),
        'sign_counts': j.get('sign_counts', {}),
        'axes': j.get('axes', []),
        'version': j.get('version', 'unknown'),
    }


def deep_get(d, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def extract_numerical_metrics(data):
    """Extract key numerical metrics from various pipeline JSON formats.

    Handles:
    - ELSA validation results (elsa_4axis_results.json)
    - Full pipeline output (full_pipeline.json)
    - Intervention analysis (intervention_analysis.json)
    - D/J validation results (dj_validation_results.json)
    """
    metrics = {}

    # ELSA-style: critical_result with delta_C
    cr = data.get('critical_result', {})
    if cr:
        for key in ['delta_c_3axis', 'delta_c_4axis']:
            if cr.get(key) is not None:
                metrics[key] = cr[key]
        for key in ['3axis_exceeds_threshold', '4axis_exceeds_threshold']:
            if key in cr:
                metrics[key] = cr[key]

    # ELSA-style: model-level results
    for model_key in ['3-axis', '4-axis']:
        model = data.get(model_key, {})

        # Cross-sectional lambda_max per stratum
        for item in model.get('cross_sectional', []):
            stratum = item.get('stratum', item.get('age_group', ''))
            lmax = item.get('lambda_max')
            if stratum and lmax is not None:
                metrics[f'lambda_max_{model_key}_{stratum}'] = lmax

        # Cox model results
        cox = model.get('cox_models', {})
        for k, v in cox.items():
            if isinstance(v, (int, float)) and k.startswith('_'):
                metrics[f'cox_{model_key}_{k}'] = v

    # Full pipeline style: steps with nested metrics
    steps = data.get('steps', {})
    for step_key, step_data in steps.items():
        if isinstance(step_data, dict):
            for k, v in step_data.items():
                if isinstance(v, (int, float)):
                    metrics[f'{step_key}.{k}'] = v

    # Intervention analysis: single_ranking top entry
    ranking = data.get('single_ranking', [])
    if ranking:
        top = ranking[0]
        if isinstance(top, dict):
            metrics['top_intervention_delta'] = top.get('delta', None)
            metrics['top_intervention_baseline'] = top.get('baseline', None)

    # D/J validation: phase results
    for phase_key in ['phase_1', 'phase_2', 'phase_3']:
        phase = data.get(phase_key, {})
        if isinstance(phase, dict):
            for k, v in phase.items():
                if isinstance(v, (int, float)):
                    metrics[f'{phase_key}.{k}'] = v

    # Sign concordance
    sc = data.get('sign_concordance', {})
    if isinstance(sc, dict):
        for k, v in sc.items():
            if isinstance(v, (int, float)):
                metrics[f'sign_concordance.{k}'] = v

    return metrics


def extract_test_outcomes(data):
    """Extract pass/fail test outcomes from pipeline JSON."""
    outcomes = {}

    # ELSA critical_result thresholds
    cr = data.get('critical_result', {})
    for key in ['3axis_exceeds_threshold', '4axis_exceeds_threshold']:
        if key in cr:
            outcomes[key] = 'PASS' if cr[key] else 'FAIL'

    # Full pipeline acceptance checks
    checks = data.get('acceptance_checks', {})
    if isinstance(checks, dict):
        for k, v in checks.items():
            if isinstance(v, bool):
                outcomes[k] = 'PASS' if v else 'FAIL'

    # Steps with pass/fail
    steps = data.get('steps', {})
    for step_key, step_data in steps.items():
        if isinstance(step_data, dict):
            for k, v in step_data.items():
                if isinstance(v, bool) and ('pass' in k.lower() or
                                             'check' in k.lower() or
                                             'accept' in k.lower()):
                    outcomes[f'{step_key}.{k}'] = 'PASS' if v else 'FAIL'

    return outcomes


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_metrics(baseline_metrics, candidate_metrics, threshold):
    """Compare numerical metrics and flag significant changes."""
    all_keys = sorted(set(baseline_metrics) | set(candidate_metrics))
    diffs = []

    for key in all_keys:
        b_val = baseline_metrics.get(key)
        c_val = candidate_metrics.get(key)

        if b_val is None and c_val is None:
            continue

        entry = {
            'metric': key,
            'baseline': b_val,
            'candidate': c_val,
        }

        if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
            abs_change = abs(c_val - b_val)
            entry['abs_change'] = round(abs_change, 8)
            entry['significant'] = abs_change >= threshold
        elif b_val != c_val:
            entry['abs_change'] = None
            entry['significant'] = True
        else:
            entry['abs_change'] = 0
            entry['significant'] = False

        # Only include if there's actually a difference
        if b_val != c_val:
            diffs.append(entry)

    return diffs


def compare_test_outcomes(baseline_outcomes, candidate_outcomes):
    """Compare test pass/fail outcomes."""
    all_keys = sorted(set(baseline_outcomes) | set(candidate_outcomes))
    diffs = []

    for key in all_keys:
        b = baseline_outcomes.get(key, 'N/A')
        c = candidate_outcomes.get(key, 'N/A')
        if b != c:
            diffs.append({
                'test': key,
                'baseline': b,
                'candidate': c,
            })

    return diffs


def generate_summary(j_diffs, numerical_diffs, test_diffs):
    """Generate one-paragraph summary."""
    parts = []

    if j_diffs:
        parts.append(f"J-matrix metadata has {len(j_diffs)} difference(s)")
    else:
        parts.append("J-matrix metadata is identical")

    n_sig = sum(1 for d in numerical_diffs if d.get('significant'))
    n_total = len(numerical_diffs)
    if n_total == 0:
        parts.append("no numerical metrics changed")
    elif n_sig == 0:
        parts.append(f"{n_total} metric(s) shifted but none significantly")
    else:
        parts.append(f"{n_sig} of {n_total} changed metric(s) are significant")

    if test_diffs:
        parts.append(f"{len(test_diffs)} test outcome(s) changed")
    else:
        parts.append("no test outcomes changed")

    return ". ".join(parts) + "."


def generate_markdown(report):
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# J-Matrix Run Comparison Report")
    lines.append("")
    lines.append(f"Generated: {report['timestamp']}")
    lines.append("")

    # J-matrix metadata
    lines.append("## J-Matrix Metadata")
    lines.append("")
    bj = report['baseline_j']
    cj = report['candidate_j']
    lines.append(f"| | Baseline | Candidate |")
    lines.append(f"|---|---|---|")
    lines.append(f"| SHA-256 | `{bj['sha256'][:16]}...` | `{cj['sha256'][:16]}...` |")
    lines.append(f"| Axes | {bj['n_axes']} | {cj['n_axes']} |")
    lines.append(f"| Version | {bj['version']} | {cj['version']} |")
    b_sc = bj.get('sign_counts', {})
    c_sc = cj.get('sign_counts', {})
    lines.append(f"| Signs (+/-/?) | "
                 f"{b_sc.get('positive', '?')}/{b_sc.get('negative', '?')}/{b_sc.get('unknown', '?')} | "
                 f"{c_sc.get('positive', '?')}/{c_sc.get('negative', '?')}/{c_sc.get('unknown', '?')} |")
    lines.append("")

    # Numerical diffs
    diffs = report.get('numerical_diffs', [])
    if diffs:
        lines.append("## Numerical Differences")
        lines.append("")
        lines.append("| Metric | Baseline | Candidate | Change | Significant |")
        lines.append("|---|---|---|---|---|")
        for d in diffs:
            b = d['baseline']
            c = d['candidate']
            b_str = f"{b:.6f}" if isinstance(b, float) else str(b)
            c_str = f"{c:.6f}" if isinstance(c, float) else str(c)
            ch = d.get('abs_change')
            ch_str = f"{ch:.6f}" if isinstance(ch, float) else str(ch)
            sig = "YES" if d.get('significant') else "no"
            lines.append(f"| {d['metric']} | {b_str} | {c_str} | {ch_str} | {sig} |")
        lines.append("")
    else:
        lines.append("## Numerical Differences")
        lines.append("")
        lines.append("No numerical differences found.")
        lines.append("")

    # Test outcome diffs
    test_diffs = report.get('test_outcome_diffs', [])
    if test_diffs:
        lines.append("## Test Outcome Changes")
        lines.append("")
        lines.append("| Test | Baseline | Candidate |")
        lines.append("|---|---|---|")
        for d in test_diffs:
            lines.append(f"| {d['test']} | {d['baseline']} | {d['candidate']} |")
        lines.append("")
    else:
        lines.append("## Test Outcome Changes")
        lines.append("")
        lines.append("No test outcomes changed.")
        lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(report.get('summary', 'No summary available.'))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = args.output_dir or OUTPUT_DIR

    # Load inputs
    with open(args.baseline, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    with open(args.candidate, 'r', encoding='utf-8') as f:
        candidate_data = json.load(f)

    # Extract components
    baseline_j = extract_j_metadata(baseline_data)
    candidate_j = extract_j_metadata(candidate_data)

    baseline_metrics = extract_numerical_metrics(baseline_data)
    candidate_metrics = extract_numerical_metrics(candidate_data)

    baseline_tests = extract_test_outcomes(baseline_data)
    candidate_tests = extract_test_outcomes(candidate_data)

    # Compare J metadata
    j_diffs = []
    for key in ['sha256', 'n_axes', 'sign_counts', 'axes', 'version']:
        if baseline_j.get(key) != candidate_j.get(key):
            j_diffs.append(key)

    # Compare metrics
    numerical_diffs = compare_metrics(baseline_metrics, candidate_metrics,
                                       args.threshold)

    # Compare test outcomes
    test_diffs = compare_test_outcomes(baseline_tests, candidate_tests)

    # Summary
    summary = generate_summary(j_diffs, numerical_diffs, test_diffs)

    # Build report
    report = {
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'baseline_file': args.baseline,
        'candidate_file': args.candidate,
        'baseline_j': baseline_j,
        'candidate_j': candidate_j,
        'j_metadata_diffs': j_diffs,
        'numerical_diffs': numerical_diffs,
        'test_outcome_diffs': test_diffs,
        'summary': summary,
    }

    # Write outputs
    json_path = os.path.join(out_dir, 'j_comparison_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON report: {json_path}")

    md_path = os.path.join(out_dir, 'j_comparison_report.md')
    md = generate_markdown(report)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Markdown report: {md_path}")

    # Print summary to console
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  J metadata diffs: {len(j_diffs)}")
    print(f"  Numerical diffs:  {len(numerical_diffs)} "
          f"({sum(1 for d in numerical_diffs if d.get('significant'))} significant)")
    print(f"  Test outcome diffs: {len(test_diffs)}")
    print(f"\n  {summary}")
    print(f"{'=' * 60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
