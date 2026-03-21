# ELSA Data

This directory should contain 4 data files from the English Longitudinal Study of Ageing.

**These files are NOT included in the repository** due to UK Data Service redistribution terms.

## How to obtain the data

1. Register at https://beta.ukdataservice.ac.uk/
2. Search for Study Number **SN 5050**
3. Download in TAB format
4. Run the consolidation script (see `scripts/consolidate_elsa.py`) to produce the required files

## Required files

| File | Description | Rows | Cols |
|------|-------------|------|------|
| `gh_elsa_h_hdr_subset.tab` | Harmonised demographics, mortality, conditions | ~21,679 | ~212 |
| `elsa_supplementary_variables.tab` | ADL/IADL, mobility, CES-D, medications | ~39,528 | ~43 |
| `h_elsa_eol_a2.tab` | End-of-life: death age, death year | ~983 | ~215 |
| `elsa_nurse_biomarkers_consolidated.tab` | All nurse biomarkers (CRP, HbA1c, grip, BP, lipids, etc.) | ~38,558 | ~41 |

## Verification

After obtaining data, run:

```bash
python scripts/run_medication_sensitivity.py
```

This should reproduce the R5 baseline result (3-axis matched ΔC = +0.0087).

### Results ledger

`outputs/elsa_results_ledger.json` contains SHA-256 hashes of the expected data files and all numerical results reported in the manuscript. Use this to verify your extract matches ours.
