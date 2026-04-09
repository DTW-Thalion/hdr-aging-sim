This directory contains frozen J-matrix snapshots used for reproducibility of published results. Files in this directory must never be modified after creation. Each file is identified by its SHA-256 hash recorded in the output metadata of any results produced against it.

## Files

- `J_R6_ontology_v1.6.csv` — Frozen copy of the 9×9 compiled J-matrix used for R6 manuscript results.

## Usage

The provenance snapshot serves as a baseline for comparison when the J-matrix is updated:

```bash
# Integration test: runs pipeline against both provenance and default CSV, verifies zero diffs
python scripts/run_j_comparison_integration.py

# Manual comparison after updating the J-matrix
python scripts/compare_j_runs.py \
    --baseline outputs/R6_provenance/full_pipeline.json \
    --candidate outputs/latest/full_pipeline.json
```

## Adding new snapshots

When publishing results against a new J-matrix version, copy it here with a descriptive name and never modify it afterwards. The `JMatrixSpec.from_csv()` function will compute its SHA-256 hash for embedding in output metadata.
