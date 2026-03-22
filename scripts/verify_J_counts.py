#!/usr/bin/env python3
"""Full J matrix audit: CSV vs manuscript table consistency."""

import csv
import sys

def main():
    with open('data/J_matrix_compiled.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total CSV rows: {len(rows)}")
    print(f"Columns: {list(rows[0].keys())}")

    # Step 2: Count signs
    counts = {'positive': 0, 'negative': 0, 'unknown': 0, 'zero': 0, 'other': 0}
    detail = {'positive': [], 'negative': [], 'unknown': [], 'zero': [], 'other': []}

    for row in rows:
        src = row.get('axis_from', '').strip()
        tgt = row.get('axis_to', '').strip()
        sign = row.get('sign', '').strip()
        label = f"{src}->{tgt}"

        if sign in ['+', 'positive', '+1']:
            counts['positive'] += 1
            detail['positive'].append(label)
        elif sign in ['-', 'negative', '-1']:
            counts['negative'] += 1
            detail['negative'].append(label)
        elif sign in ['?', 'unknown', '']:
            counts['unknown'] += 1
            detail['unknown'].append(label)
        elif sign in ['0', 'zero']:
            counts['zero'] += 1
            detail['zero'].append(label)
        else:
            counts['other'] += 1
            detail['other'].append(f"{label} (sign={sign!r})")

    print("\n=== CSV SIGN COUNTS ===")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
    total = sum(counts.values())
    print(f"  TOTAL: {total}")

    print("\n=== UNKNOWN ENTRIES ===")
    for item in detail['unknown']:
        print(f"  {item}")

    print("\n=== NEGATIVE ENTRIES ===")
    for item in detail['negative']:
        print(f"  {item}")

    if detail['other']:
        print("\n=== OTHER (UNEXPECTED) ===")
        for item in detail['other']:
            print(f"  {item}")

    # Step 5: P->C check
    print("\n=== P->C ENTRY ===")
    for row in rows:
        src = row.get('axis_from', '').strip()
        tgt = row.get('axis_to', '').strip()
        if src == 'P' and tgt == 'C':
            for k, v in row.items():
                print(f"  {k}: {v}")

    # Step 6: Full magnitude/grade audit against manuscript table
    manuscript_table = {
        ('I','M'): ('+','S','A'), ('I','E'): ('+','M','B'), ('I','mito'): ('+','M','B'),
        ('I','P'): ('+','M','B'), ('I','C'): ('+','W','C'), ('I','N'): ('+','M','B'),
        ('I','F'): ('+','S','A'),
        ('M','I'): ('+','M','A'), ('M','E'): ('+','W','C'), ('M','mito'): ('+','S','A'),
        ('M','P'): ('+','M','B'), ('M','C'): ('+','M','B'), ('M','N'): ('+','M','B'),
        ('M','F'): ('+','M','B'),
        ('E','I'): ('+','W','C'), ('E','M'): ('+','M','B'), ('E','mito'): ('+','M','B'),
        ('E','P'): ('?',None,None), ('E','C'): ('?',None,None),
        ('E','N'): ('?',None,None), ('E','F'): ('?',None,None),
        ('mito','I'): ('+','S','A'), ('mito','M'): ('+','S','A'), ('mito','E'): ('+','M','B'),
        ('mito','P'): ('+','S','A'), ('mito','C'): ('+','S','A'), ('mito','N'): ('+','M','B'),
        ('mito','F'): ('+','S','A'),
        ('P','I'): ('+','S','A'), ('P','M'): ('+','W','C'), ('P','E'): ('?',None,None),
        ('P','mito'): ('+','S','A'), ('P','C'): ('+','M','B'), ('P','N'): ('+','W','C'),
        ('P','F'): ('+','M','B'),
        ('C','I'): ('+','M','B'), ('C','M'): ('+','S','A'), ('C','E'): ('+','W','C'),
        ('C','mito'): ('+','S','A'), ('C','P'): ('+','S','A'), ('C','N'): ('+','S','A'),
        ('C','F'): ('+','M','B'),
        ('N','I'): ('+','S','A'), ('N','M'): ('+','S','A'), ('N','E'): ('+','W','C'),
        ('N','mito'): ('+','M','B'), ('N','P'): ('+','W','C'), ('N','C'): ('+','M','B'),
        ('N','F'): ('+','M','B'),
        ('F','I'): ('-','S','A'), ('F','M'): ('-','S','A'), ('F','E'): ('-','M','B'),
        ('F','mito'): ('-','S','A'), ('F','P'): ('-','M','B'), ('F','C'): ('-','M','B'),
        ('F','N'): ('-','S','A'),
    }

    mismatches = []
    for row in rows:
        src = row.get('axis_from', '').strip()
        tgt = row.get('axis_to', '').strip()
        key = (src, tgt)
        if key not in manuscript_table:
            mismatches.append(f"CSV has {src}->{tgt} which is not in manuscript table")
            continue

        expected_sign, expected_mag, expected_grade = manuscript_table[key]
        csv_sign = row.get('sign', '').strip()
        csv_mag = row.get('magnitude_tier', '').strip()
        csv_grade = row.get('confidence_grade', '').strip()

        # Normalize sign
        if csv_sign in ['+', 'positive', '+1', '1']:
            csv_sign_norm = '+'
        elif csv_sign in ['-', 'negative', '-1']:
            csv_sign_norm = '-'
        elif csv_sign in ['?', 'unknown', '']:
            csv_sign_norm = '?'
        elif csv_sign in ['0', 'zero']:
            csv_sign_norm = '0'
        else:
            csv_sign_norm = csv_sign

        if csv_sign_norm != expected_sign:
            mismatches.append(f"{src}->{tgt}: sign CSV='{csv_sign}' vs manuscript='{expected_sign}'")

        if expected_mag is not None:
            if csv_mag.upper() != expected_mag.upper():
                mismatches.append(f"{src}->{tgt}: magnitude CSV='{csv_mag}' vs manuscript='{expected_mag}'")

        if expected_grade is not None:
            if csv_grade.upper() != expected_grade.upper():
                mismatches.append(f"{src}->{tgt}: grade CSV='{csv_grade}' vs manuscript='{expected_grade}'")

    print(f"\n{'='*60}")
    if mismatches:
        print(f"MISMATCHES FOUND: {len(mismatches)}")
        print(f"{'='*60}")
        for m in mismatches:
            print(f"  X {m}")
    else:
        print("ALL 56 ENTRIES MATCH between CSV and manuscript table")

    # Final summary
    print(f"\n{'='*60}")
    print("J MATRIX AUDIT SUMMARY")
    print(f"{'='*60}")
    row_pass = len(rows) == 56
    pos_pass = counts['positive'] == 44
    neg_pass = counts['negative'] == 7
    unk_pass = counts['unknown'] == 5
    zero_pass = counts['zero'] == 0
    all_match = len(mismatches) == 0
    assigned = 56 - counts['unknown']
    assigned_pass = assigned == 51

    print(f"  CSV rows:           {len(rows):>3}  [{'PASS' if row_pass else 'FAIL'}]")
    print(f"  Positive:           {counts['positive']:>3}  [{'PASS' if pos_pass else 'FAIL'}]")
    print(f"  Negative:           {counts['negative']:>3}  [{'PASS' if neg_pass else 'FAIL'}]")
    print(f"  Unknown:            {counts['unknown']:>3}  [{'PASS' if unk_pass else 'FAIL'}]")
    print(f"  Zero:               {counts['zero']:>3}  [{'PASS' if zero_pass else 'FAIL'}]")
    print(f"  Assigned (51):      {assigned:>3}  [{'PASS' if assigned_pass else 'FAIL'}]")
    print(f"  All entries match:       [{'PASS' if all_match else 'FAIL'}]")
    print(f"  Discrepancies:      {len(mismatches):>3}")

    # Check unknown entries are the expected 5
    expected_unknown = {'E->P', 'E->C', 'E->N', 'E->F', 'P->E'}
    actual_unknown = set(detail['unknown'])
    unk_entries_pass = actual_unknown == expected_unknown
    print(f"  Unknown entries correct: [{'PASS' if unk_entries_pass else 'FAIL'}]")
    if not unk_entries_pass:
        print(f"    Expected: {sorted(expected_unknown)}")
        print(f"    Actual:   {sorted(actual_unknown)}")

    # Check negative entries are all F->
    expected_neg = {'F->I', 'F->M', 'F->E', 'F->mito', 'F->P', 'F->C', 'F->N'}
    actual_neg = set(detail['negative'])
    neg_entries_pass = actual_neg == expected_neg
    print(f"  Negative entries correct: [{'PASS' if neg_entries_pass else 'FAIL'}]")
    if not neg_entries_pass:
        print(f"    Expected: {sorted(expected_neg)}")
        print(f"    Actual:   {sorted(actual_neg)}")

    all_pass = all([row_pass, pos_pass, neg_pass, unk_pass, zero_pass,
                    all_match, assigned_pass, unk_entries_pass, neg_entries_pass])
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())
