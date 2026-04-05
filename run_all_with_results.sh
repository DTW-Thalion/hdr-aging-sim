#!/bin/bash
# Run full HDR test suite and generate results/RESULTS.md
set -e

echo "=========================================="
echo "HDR Aging Simulation — Full Test Suite"
echo "=========================================="

# Clear previous results and write fresh header
python3 -c "
import sys; sys.path.insert(0, '.')
from src.hdr_sim.results_writer import clear_results
clear_results()
print('results/RESULTS.md cleared')
"

# Unit tests
echo ""
echo "--- Unit Tests ---"
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20

# Simulation scripts
echo ""
echo "--- Gamma Equivalence Study ---"
python3 scripts/run_figure_gamma_equivalence.py

echo ""
echo "--- Prior Stress Tests ---"
python3 scripts/run_figure_prior_stress.py

echo ""
echo "--- Q-Sensitivity Analysis ---"
python3 scripts/run_figure_Q_sensitivity.py

echo ""
echo "--- NHANES Feasibility ---"
python3 scripts/run_nhanes_feasibility.py

echo ""
echo "--- R6 D vs. J Primacy Validation ---"
python3 scripts/run_dj_validation.py

# ELSA validation (only if data is present)
if [ -f "data/elsa/gh_elsa_h_hdr_subset.tab" ]; then
    echo ""
    echo "--- ELSA Cohort Validation ---"
    python3 scripts/run_elsa_validation.py

    echo ""
    echo "--- R6 Figure: Coupling Tightening ---"
    python3 scripts/run_figure_coupling_tightening.py

    echo ""
    echo "--- R6 Figure: Mortality Prediction ---"
    python3 scripts/run_figure_mortality_prediction.py

    echo ""
    echo "--- R6 Figure: Medication Compression ---"
    python3 scripts/run_figure_medication_compression.py

    echo ""
    echo "--- ELSA ICI Deployment Assessment ---"
    python3 scripts/run_elsa_ici_deployment.py
else
    echo ""
    echo "--- ELSA Cohort Validation: SKIPPED (data not present) ---"
    python3 -c "
import sys; sys.path.insert(0, '.')
from src.hdr_sim.results_writer import ResultsWriter
with ResultsWriter('ELSA Cohort Validation', 'SKIPPED — ELSA data files not found in data/elsa/') as rw:
    rw.add_text('Place ELSA data files in data/elsa/ and re-run to execute Phase 3.')
"
fi

echo ""
echo "=========================================="
echo "Results written to results/RESULTS.md"
echo "=========================================="
cat results/RESULTS.md
