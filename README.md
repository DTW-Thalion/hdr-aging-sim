# HDR Aging Dynamics Toy Simulation

Numerical simulation of the Homeodynamic Remediation (HDR) framework's aging dynamics,
demonstrating how age-related parameter degradation produces critical slowing-down,
increased vulnerability to perturbation, and basin transitions.

Companion code for: White (2026), "Aging as Controller Failure: A Formal Ontology
for Multi-Axis Physiological Decline."

## Key result

The simulation shows that a 4-axis linear dynamical system with biologically parameterised
recovery times (τ_i) and inter-axis coupling (J), when degraded according to known
age-related trajectories, produces:

- Progressive drift of the spectral abscissa toward instability (α → 0⁻)
- Critical slowing-down: recovery timescale diverges as α → 0
- Identical perturbations produce age-dependent responses (fast recovery at 30,
  near-catastrophic excursion at 80)
- Cross-axis propagation of perturbations through the coupling matrix

These are the dynamical signatures of aging predicted by the HDR formalism and
independently observed by Scheffer et al. (2018, PNAS) and Pyrkov et al. (2021,
Nat Commun).

## Setup

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python scripts/run_figure2b.py        # Main figure: 4-panel aging demo
python scripts/run_figure_frailty.py   # Frailty perturbation-response
python scripts/run_figure_t2d.py       # T2D phase portrait
```

Figures are saved to `outputs/`.

## R2 Revision: Recoverability and Uncertainty Simulations

Two new simulation studies added for the revised manuscript (Appendix F):

### Synthetic Data Recoverability Study

Tests which quantities (J signs, α, τ_i, ρ) are recoverable from realistic cohort sampling frequencies (2–6 visits over 15 years, annual spacing).

```bash
python scripts/run_figure_recoverability.py
```

Key findings:
* α(A) trend: recoverable via Lyapunov inversion (Spearman r = 1.0)
* J sign concordance via partial correlations: ~58% (weak-coupling assumption violated at ε ≈ 0.9–3.6)
* J sign concordance via constrained OU estimation: ~73% (with J prior as Bayesian regulariser)
* τ_i: recoverable at ≤1-day sampling; fails at ≥1-week
* Φ from yearly visits: uninformative (ρ(Δt=1yr) ≈ 10⁻⁶ to 10⁻²²)
* Critical insight: sparse cohort visits sample the stationary distribution, not dynamical transitions

### J Matrix Uncertainty Propagation

Monte Carlo (N = 10,000) over plausible J matrices within confidence-grade uncertainty bands (grade A: 20% CV, grade B: 40% CV, grade C: 70% CV).

```bash
python scripts/run_figure_uncertainty.py
```

Key findings:
* "9/12 positive" structure preserved in 100% of draws (4-axis model)
* Monotone α ordering preserved in 100% of draws across all age strata
* I↔M feedback loop dominates α sensitivity (r ≈ 0.63)
* N-axis entries have negligible influence on α (<0.05 correlation)
* System stable (α < 0) in 100% of draws at all ages

### Estimation Module

Reusable functions for stability analysis and parameter recovery:

```python
from hdr_sim import (
    lyapunov_inversion_symmetric,  # A_hat from cross-sectional covariance
    estimate_A_sign_constrained,    # With J prior regularisation
    sign_concordance,               # Compare estimated vs compiled J signs
    generate_stratum,               # Synthetic cross-sectional data
)
```

### Machine-Readable Coupling Matrix

`data/J_matrix_compiled.csv` contains all 56 off-diagonal entries of the 8×8 J matrix with sign, magnitude tier (S/M/W), confidence grade (A/B/C), interpolated values at ages 30 and 80, primary evidence PMID, evidence type, and brief rationale.

### Figures

| Script | Output | Manuscript location |
|--------|--------|---------------------|
| `run_figure2b.py` | `outputs/figure_2b.pdf` | Figure 1 (spectral-abscissa drift) |
| `run_figure_frailty.py` | `outputs/figure_frailty.pdf` | Figure 2 (frailty dynamics) |
| `run_figure_t2d.py` | `outputs/figure_t2d.pdf` | Figure 3 (T2D phase portrait) |
| `run_figure_recoverability.py` | `outputs/figure_recoverability.pdf` | Figure 4 (recoverability) |
| `run_figure_uncertainty.py` | `outputs/figure_uncertainty.pdf` | Figure 5 (uncertainty) |
