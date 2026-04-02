#!/usr/bin/env python3
"""
One-time script to populate missing pmid_primary entries in J_matrix_compiled_9x9.csv.
Run once, then verify with verify_J_matrix_counts.py.
"""
import csv

# Map of (axis_from, axis_to) -> pmid_primary to populate (only where currently empty)
pmid_map = {
    # Grade A
    ('mito', 'P'): 'kastle2011proteasome',
    ('N', 'C'): 'knezevic2023cortisol',
    ('N', 'M'): 'sharma2020stressdiabetes',
    # Grade B
    ('P', 'M'): 'montane2012iappstress',
    ('P', 'mito'): 'pickles2018mitophagy',
    ('F', 'P'): 'dokladny2015hspautophagy',
    ('F', 'C'): 'hower2018exercisecircadian',
    ('F', 'N'): 'besnier2017exerciseautonomic',
    ('P', 'F'): 'askanas2015ibmproteostasis',
    ('C', 'F'): 'su2025chronoexercise',
    ('N', 'F'): 'schakman2008gcmyopathy',
    # Grade C negative
    ('B', 'I'): 'komori2020osteocalcin',
    ('B', 'N'): 'obri2018osteocalcinbrain',
    # Grade C theoretical
    ('E', 'M'): 'dayeh2015betacell',
    ('P', 'E'): 'torresarciga2022histone',
    ('C', 'E'): 'masri2015circadian',
    ('N', 'E'): 'watkeys2018glucocorticoid',
    ('E', 'mito'): 'ruizandres2016inflammation',
    ('C', 'mito'): 'imai2014nad',
    ('N', 'mito'): 'martin2023cachexia',
    ('E', 'P'): 'garciaprat2016autophagy',
    ('C', 'P'): 'juste2021chronophagy',
    ('N', 'P'): 'silva2018stress',
    ('E', 'C'): 'cronin2016bmal1',
    ('mito', 'C'): 'levine2020nad',
    ('P', 'C'): 'vriend2014proteasome',
    ('E', 'N'): 'watkeys2018glucocorticoid',
    # mito->N: pending (no good review found)
    ('P', 'N'): 'borghammer2019braingut',
    ('E', 'B'): 'zhu2024osteoblast',
    ('mito', 'B'): 'sabini2023oxphos',
    ('P', 'B'): 'vriend2015ubiquitin',
    ('C', 'B'): 'luo2021circadian',
    # Unknown sign entries (B->X) — no PMID needed (sign=?, evidence_type=unknown)
    # B->E, B->mito, B->P, B->C already have no PMID and that's correct
}

infile = 'data/J_matrix_compiled_9x9.csv'
outfile = 'data/J_matrix_compiled_9x9.csv'

with open(infile, newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

populated = 0
skipped = 0
for row in rows:
    key = (row['axis_from'], row['axis_to'])
    if key in pmid_map and not row.get('pmid_primary', '').strip():
        row['pmid_primary'] = pmid_map[key]
        populated += 1
        print(f"  Populated: {key[0]}->{key[1]} = {pmid_map[key]}")
    elif key in pmid_map and row.get('pmid_primary', '').strip():
        skipped += 1
        print(f"  Skipped (already has PMID): {key[0]}->{key[1]}")

with open(outfile, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nDone: {populated} PMIDs populated, {skipped} skipped (already had PMID)")
print(f"Remaining missing: check with verify_J_matrix_counts.py")
