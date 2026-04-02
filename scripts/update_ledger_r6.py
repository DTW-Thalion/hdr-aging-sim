#!/usr/bin/env python3
"""Update results ledger to v1.5-R6-figures."""
import json, hashlib, os

with open('outputs/elsa_results_ledger.json') as f:
    ledger = json.load(f)

ledger['pipeline_version'] = 'v1.5-R6-figures'

if 'r6_figures' not in ledger:
    ledger['r6_figures'] = {}

for fp in [
    'scripts/run_figure_coupling_tightening.py',
    'scripts/run_figure_mortality_prediction.py',
    'scripts/run_figure_medication_compression.py',
    'outputs/figure_coupling_tightening.pdf',
    'outputs/figure_mortality_prediction.pdf',
    'outputs/figure_medication_compression.pdf',
]:
    if os.path.exists(fp):
        sha = hashlib.sha256(open(fp, 'rb').read()).hexdigest()
        ledger['r6_figures'][os.path.basename(fp)] = {
            'path': fp,
            'sha256': sha,
            'size_bytes': os.path.getsize(fp),
        }
    else:
        print(f"WARNING: {fp} not found")

ledger['r6_notes'] = (
    'R6 restructures ELSA validation figures into 3 separate 3-panel '
    'figures (coupling tightening, mortality prediction, medication '
    'compression) to support the revised Results narrative where '
    'medication compression is the central finding.'
)

with open('outputs/elsa_results_ledger.json', 'w') as f:
    json.dump(ledger, f, indent=2)
print("Ledger updated to v1.5-R6-figures")
