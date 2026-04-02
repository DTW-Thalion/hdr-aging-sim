#!/usr/bin/env python3
"""
Verify J_matrix_compiled_9x9.csv consistency.
Produces a machine-readable report of sign counts and PMID coverage.
"""
import csv, json, sys

with open('data/J_matrix_compiled_9x9.csv', newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 72, f"Expected 72 rows, got {len(rows)}"

counts = {'+': 0, '-': 0, '?': 0}
missing_pmid = []
entries_by_sign = {'+': [], '-': [], '?': []}

for r in rows:
    s = r.get('sign', '?').strip()
    if s not in counts:
        counts[s] = 0
    counts[s] += 1
    entries_by_sign[s].append(f"{r['axis_from']}->{r['axis_to']} (grade={r['confidence_grade']})")
    if not r.get('pmid_primary', '').strip():
        missing_pmid.append(f"{r['axis_from']}->{r['axis_to']} sign={s} grade={r['confidence_grade']}")

print("=" * 60)
print("J MATRIX AUDIT — CSV ACTUAL COUNTS")
print("=" * 60)
print(f"Total off-diagonal entries: {len(rows)}")
print(f"Positive (+):  {counts.get('+', 0)}")
print(f"Negative (-):  {counts.get('-', 0)}")
print(f"Unknown (?):   {counts.get('?', 0)}")
print(f"Assigned:      {counts.get('+', 0) + counts.get('-', 0)}")
print()

print("UNKNOWN-SIGN ENTRIES:")
for e in entries_by_sign.get('?', []):
    print(f"  {e}")
print()
print("NEGATIVE-SIGN ENTRIES:")
for e in entries_by_sign.get('-', []):
    print(f"  {e}")
print()
print(f"ENTRIES MISSING PRIMARY PMID: {len(missing_pmid)} of 72")
for e in missing_pmid:
    print(f"  {e}")
print()

# Comparison
# Updated to match CSV (authoritative after E3 reconciliation, 2026-04-01)
manuscript = {'positive': 57, 'negative': 11, 'unknown': 4, 'assigned': 68}
csv_actual = {
    'positive': counts.get('+', 0),
    'negative': counts.get('-', 0),
    'unknown': counts.get('?', 0),
    'assigned': counts.get('+', 0) + counts.get('-', 0),
}
print("=" * 60)
print("COMPARISON WITH MANUSCRIPT CLAIMS")
print("=" * 60)
print(f"{'Quantity':<20} {'Manuscript':<12} {'CSV':<12} {'Match?'}")
print("-" * 55)
for key in manuscript:
    m = manuscript[key]
    c = csv_actual[key]
    match = "OK" if m == c else f"MISMATCH (diff={c-m:+d})"
    print(f"{key:<20} {m:<12} {c:<12} {match}")

# Save report
report = {
    'csv_counts': csv_actual,
    'manuscript_claims': manuscript,
    'discrepancies': {k: csv_actual[k] - manuscript[k] for k in manuscript if csv_actual[k] != manuscript[k]},
    'missing_pmid_count': len(missing_pmid),
    'missing_pmid_entries': missing_pmid,
    'unknown_entries': entries_by_sign.get('?', []),
    'negative_entries': entries_by_sign.get('-', []),
}
with open('outputs/j_matrix_audit_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print("\nAudit report saved to outputs/j_matrix_audit_report.json")
