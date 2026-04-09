#!/usr/bin/env python3
"""Full J matrix audit: CSV vs manuscript table consistency.

Supports both the 9x9 (default) and legacy 8x8 J matrix CSVs.
Usage:
    python scripts/verify_J_counts.py                              # 9x9 default
    python scripts/verify_J_counts.py --csv data/J_matrix_compiled.csv  # 8x8 legacy
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from hdr_sim.j_matrix_spec import JMatrixSpec


# ---------------------------------------------------------------------------
# Expected counts per CSV version
# ---------------------------------------------------------------------------

EXPECTED = {
    8: {
        'rows': 56,
        'positive': 44,
        'negative': 7,
        'unknown': 5,
        'axes': {'I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F'},
        'expected_unknown': {'E->P', 'E->C', 'E->N', 'E->F', 'P->E'},
        'expected_negative': {'F->I', 'F->M', 'F->E', 'F->mito', 'F->P', 'F->C', 'F->N'},
    },
    9: {
        'rows': 72,
        'positive': 57,
        'negative': 11,
        'unknown': 4,
        'axes': {'I', 'M', 'E', 'mito', 'P', 'C', 'N', 'F', 'B'},
        'expected_unknown': {'B->E', 'B->mito', 'B->P', 'B->C'},
        'expected_negative': {
            'F->I', 'F->M', 'F->E', 'F->mito', 'F->P', 'F->C', 'F->N', 'F->B',
            'B->I', 'B->M', 'B->N',
        },
    },
}


def _normalise_sign(raw):
    """Normalise sign string to +, -, ?, or the original."""
    raw = raw.strip()
    if raw in ('+', 'positive', '+1', '1'):
        return '+'
    if raw in ('-', 'negative', '-1'):
        return '-'
    if raw in ('?', 'unknown', ''):
        return '?'
    if raw in ('0', 'zero'):
        return '0'
    return raw


def main():
    parser = argparse.ArgumentParser(description='Verify J matrix CSV sign counts.')
    parser.add_argument('--csv', default='data/J_matrix_compiled_9x9.csv',
                        help='Path to J matrix CSV (default: 9x9)')
    args = parser.parse_args()

    j_spec = JMatrixSpec.from_csv(args.csv)
    print(f"J-matrix SHA-256: {j_spec.sha256}")
    print(f"J-matrix spec: n_axes={j_spec.n_axes}, signs={j_spec.sign_counts}")

    with open(args.csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"CSV file: {args.csv}")
    print(f"Total CSV rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys())}")

    # -----------------------------------------------------------------------
    # Detect dimension from axes present
    # -----------------------------------------------------------------------
    axes_found = sorted(set(r['axis_from'].strip() for r in rows) |
                        set(r['axis_to'].strip() for r in rows))
    n_axes = len(axes_found)
    print(f"Axes ({n_axes}): {axes_found}")

    if n_axes not in EXPECTED:
        print(f"\nWARNING: No expected counts for {n_axes}-axis matrix.")
        print("Proceeding with count-only mode.\n")
        exp = None
    else:
        exp = EXPECTED[n_axes]

    # -----------------------------------------------------------------------
    # Count signs
    # -----------------------------------------------------------------------
    counts = {'positive': 0, 'negative': 0, 'unknown': 0, 'zero': 0, 'other': 0}
    detail = defaultdict(list)

    for row in rows:
        src = row['axis_from'].strip()
        tgt = row['axis_to'].strip()
        sign = _normalise_sign(row.get('sign', ''))
        label = f"{src}->{tgt}"

        if sign == '+':
            counts['positive'] += 1
            detail['positive'].append(label)
        elif sign == '-':
            counts['negative'] += 1
            detail['negative'].append(label)
        elif sign == '?':
            counts['unknown'] += 1
            detail['unknown'].append(label)
        elif sign == '0':
            counts['zero'] += 1
            detail['zero'].append(label)
        else:
            counts['other'] += 1
            detail['other'].append(f"{label} (sign={sign!r})")

    print(f"\n=== CSV SIGN COUNTS ===")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
    total = sum(counts.values())
    print(f"  TOTAL: {total}")

    print(f"\n=== UNKNOWN ENTRIES ===")
    for item in detail['unknown']:
        print(f"  {item}")

    print(f"\n=== NEGATIVE ENTRIES ===")
    for item in detail['negative']:
        print(f"  {item}")

    if detail['other']:
        print(f"\n=== OTHER (UNEXPECTED) ===")
        for item in detail['other']:
            print(f"  {item}")

    # -----------------------------------------------------------------------
    # Per-axis incoming / outgoing summary table
    # -----------------------------------------------------------------------
    incoming = defaultdict(int)   # axis_to counts
    outgoing = defaultdict(int)   # axis_from counts
    for row in rows:
        src = row['axis_from'].strip()
        tgt = row['axis_to'].strip()
        outgoing[src] += 1
        incoming[tgt] += 1

    print(f"\n=== PER-AXIS ENTRY COUNTS ===")
    print(f"  {'Axis':<6} {'Incoming':>8} {'Outgoing':>8} {'Total':>6}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*6}")
    for ax in axes_found:
        inc = incoming.get(ax, 0)
        out = outgoing.get(ax, 0)
        print(f"  {ax:<6} {inc:>8} {out:>8} {inc+out:>6}")

    # -----------------------------------------------------------------------
    # B-axis specific checks (9-axis only)
    # -----------------------------------------------------------------------
    if 'B' in set(axes_found):
        b_incoming = [r for r in rows if r['axis_to'].strip() == 'B']
        b_outgoing = [r for r in rows if r['axis_from'].strip() == 'B']
        print(f"\n=== B-AXIS ENTRIES ===")
        print(f"  Incoming (X->B): {len(b_incoming)}")
        for r in b_incoming:
            print(f"    {r['axis_from'].strip()}->B: sign={r['sign'].strip()}")
        print(f"  Outgoing (B->X): {len(b_outgoing)}")
        for r in b_outgoing:
            print(f"    B->{r['axis_to'].strip()}: sign={r['sign'].strip()}")

        # Verify all expected B entries present
        other_axes = [a for a in axes_found if a != 'B']
        expected_b_in = {f"{a}->B" for a in other_axes}
        expected_b_out = {f"B->{a}" for a in other_axes}
        actual_b_in = {f"{r['axis_from'].strip()}->B" for r in b_incoming}
        actual_b_out = {f"B->{r['axis_to'].strip()}" for r in b_outgoing}
        b_complete = (actual_b_in == expected_b_in and actual_b_out == expected_b_out)
        print(f"  All B entries present: [{'PASS' if b_complete else 'FAIL'}]")
        if not b_complete:
            missing_in = expected_b_in - actual_b_in
            missing_out = expected_b_out - actual_b_out
            if missing_in:
                print(f"    Missing incoming: {sorted(missing_in)}")
            if missing_out:
                print(f"    Missing outgoing: {sorted(missing_out)}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("J MATRIX AUDIT SUMMARY")
    print(f"{'='*60}")

    checks = []

    if exp:
        row_pass = len(rows) == exp['rows']
        pos_pass = counts['positive'] == exp['positive']
        neg_pass = counts['negative'] == exp['negative']
        unk_pass = counts['unknown'] == exp['unknown']
        zero_pass = counts['zero'] == 0
        axes_pass = set(axes_found) == exp['axes']
        assigned = len(rows) - counts['unknown']
        assigned_exp = exp['rows'] - exp['unknown']
        assigned_pass = assigned == assigned_exp

        actual_unk = set(detail['unknown'])
        unk_entries_pass = actual_unk == exp['expected_unknown']

        actual_neg = set(detail['negative'])
        neg_entries_pass = actual_neg == exp['expected_negative']

        print(f"  Mode:               {n_axes}x{n_axes}")
        print(f"  CSV rows:           {len(rows):>3}  (exp {exp['rows']})  [{'PASS' if row_pass else 'FAIL'}]")
        print(f"  Axes:               {n_axes:>3}  (exp {len(exp['axes'])})  [{'PASS' if axes_pass else 'FAIL'}]")
        print(f"  Positive:           {counts['positive']:>3}  (exp {exp['positive']})  [{'PASS' if pos_pass else 'FAIL'}]")
        print(f"  Negative:           {counts['negative']:>3}  (exp {exp['negative']})  [{'PASS' if neg_pass else 'FAIL'}]")
        print(f"  Unknown:            {counts['unknown']:>3}  (exp {exp['unknown']})  [{'PASS' if unk_pass else 'FAIL'}]")
        print(f"  Zero:               {counts['zero']:>3}  (exp 0)  [{'PASS' if zero_pass else 'FAIL'}]")
        print(f"  Assigned-sign:      {assigned:>3}  (exp {assigned_exp})  [{'PASS' if assigned_pass else 'FAIL'}]")
        print(f"  Unknown entries correct: [{'PASS' if unk_entries_pass else 'FAIL'}]")
        if not unk_entries_pass:
            print(f"    Expected: {sorted(exp['expected_unknown'])}")
            print(f"    Actual:   {sorted(actual_unk)}")
        print(f"  Negative entries correct: [{'PASS' if neg_entries_pass else 'FAIL'}]")
        if not neg_entries_pass:
            print(f"    Expected: {sorted(exp['expected_negative'])}")
            print(f"    Actual:   {sorted(actual_neg)}")

        checks = [row_pass, pos_pass, neg_pass, unk_pass, zero_pass,
                   axes_pass, assigned_pass, unk_entries_pass, neg_entries_pass]

        if 'B' in set(axes_found):
            checks.append(b_complete)
    else:
        print(f"  Rows: {len(rows)}, Axes: {n_axes}")
        print(f"  Positive: {counts['positive']}, Negative: {counts['negative']}, Unknown: {counts['unknown']}")

    all_pass = all(checks) if checks else True
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
