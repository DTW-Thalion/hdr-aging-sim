# HDR Aging Dynamics Toy Simulation

Numerical simulation of the Homeodynamic Remediation (HDR) framework's aging dynamics,
demonstrating how age-related parameter degradation produces critical slowing-down,
increased vulnerability to perturbation, and basin transitions.

Companion code for: White (2026), "Aging as Controller Failure: A Formal Ontology
for Multi-Axis Physiological Decline."

## Key result

The simulation shows that a 4-axis linear dynamical system with biologically parameterised
recovery times (τ_i) and inter-axis coupling (J) loaded from the literature-derived
`data/J_matrix_compiled.csv`, when degraded according to known age-related trajectories,
produces:

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

For the NHANES feasibility script, additional dependencies are needed:

```bash
pip install pandas pyreadstat
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

### Machine-Readable Coupling Matrix (CSV-Driven Architecture)

`data/J_matrix_compiled.csv` is the **single source of truth** for J coupling values. It contains all 56 off-diagonal entries of the 8×8 mechanistic coupling matrix with:

- **Basin-stratified values**: `J_healthy`, `J_pre_disease`, `J_disease` (SD-per-SD units) replacing the former age-interpolated `J_value_age30`/`J_value_age80`
- **Age trajectory**: `increasing`, `decreasing`, `stable`, or `unknown`
- **Evidence metadata**: sign, magnitude tier (S/M/W/unknown), confidence grade (A/B/C), primary PMID, evidence type, and rationale with citations

**How it works**: At import time, `hdr_sim.csv_loader` loads the CSV, extracts the 4-axis subset (I, M, N, F) for the healthy and disease basins, and applies a calibration scalar (c ≈ 0.30) to map SD-per-SD literature values to simulation coupling rates (day⁻¹). The calibration scalar is computed via Brent's method to match the target spectral abscissa α(30) ≈ −0.134. This preserves the literature-derived relative coupling structure while maintaining dynamical stability.

```python
from hdr_sim import load_J_csv, build_J_basin, get_J_anchors

# Load full 8-axis matrix for any basin
rows = load_J_csv()
J_8x8 = build_J_basin(rows, basin='disease',
                       axes=('I','M','E','mito','P','C','N','F'))

# Get calibrated 4-axis anchors used by the simulation
J_30, J_80, c = get_J_anchors()  # c ≈ 0.30
```

The full 8×8 CSV contains entries derived from systematic review of 58 references using evidence triangulation (statistical, molecular, causal). See White (2026), "The Mechanistic Coupling Matrix J_mech" for methodology.

### Figures

| Script | Output | Manuscript location |
|--------|--------|---------------------|
| `run_figure2b.py` | `outputs/figure_2b.pdf` | Figure 1 (spectral-abscissa drift; script named after original Fig 2b in early drafts) |
| `run_figure_frailty.py` | `outputs/figure_frailty.pdf` | Figure 2 (frailty dynamics) |
| `run_figure_t2d.py` | `outputs/figure_t2d.pdf` | Figure 3 (T2D phase portrait) |
| `run_figure_recoverability.py` | `outputs/figure_recoverability.pdf` | Figure 4 (recoverability) |
| `run_figure_uncertainty.py` | `outputs/figure_uncertainty.pdf` | Figure 5 (uncertainty) |

## R3 Revision: Q Sensitivity, Individual Proxy, and NHANES Feasibility

Three additional simulation studies for the R3 manuscript revision:

### Diffusion Covariance (Q) Sensitivity Analysis

Tests whether the α̂ stability trend is robust to age-varying noise intensity.

```bash
python scripts/run_figure_Q_sensitivity.py
```

Key finding: The monotone α̂ trend survives even when noise increases 6× from age 30 to 80 (β=5.0). The trend is driven by τ_i degradation, not by Q — confirming that "resilience declines" is not an artifact of "noise increases."

### Stability-Weighted Dysregulation Score (SWDS)

Individual-level stability proxy that resolves the ecological fallacy in Tests 5–6.

```bash
python scripts/run_figure_individual_proxy.py
```

Key finding: SWDS (C-index ≈ 0.57) outperforms Mahalanobis distance (ΔC ≈ +0.03), L2 norm (ΔC ≈ +0.01), and age alone (ΔC ≈ +0.05) for predicting stability-dependent outcomes in synthetic data. The eigenvector weighting extracts individual-level information that isotropic measures miss.

### NHANES Real-Data Feasibility Demonstration

End-to-end pipeline on publicly available NHANES 2011-2012 data.

```bash
pip install pyreadstat pandas  # additional dependencies
python scripts/run_nhanes_feasibility.py
```

Key findings (mixed — reported honestly):
* Pipeline feasibility: ✓ — all steps execute on standard public data (N=2,589)
* Mean trends: ✓ — metabolic dysfunction and functional decline increase with age
* Variance-based α̂: ✗ — cross-sectional variance compresses with age due to survivorship bias and medication effects, producing a reversed α̂ trend
* Diagnosis: identifies a specific data-design requirement (longitudinal within-person covariance, not cross-sectional) rather than a framework failure

Note: NHANES XPT files are downloaded from CDC servers at runtime (~7 MB total). If download fails (e.g., government shutdown), see `data/nhanes/README.md` for manual download instructions.

### Updated Figures Table

| Script | Output | Manuscript location |
|--------|--------|---------------------|
| `run_figure2b.py` | `outputs/figure_2b.pdf` | Figure 1 |
| `run_figure_frailty.py` | `outputs/figure_frailty.pdf` | Figure 2 |
| `run_figure_t2d.py` | `outputs/figure_t2d.pdf` | Figure 3 |
| `run_figure_recoverability.py` | `outputs/figure_recoverability.pdf` | Figure 4 |
| `run_figure_uncertainty.py` | `outputs/figure_uncertainty.pdf` | Figure 5 |
| `run_figure_Q_sensitivity.py` | `outputs/figure_Q_sensitivity.pdf` | Figure 6 |
| `run_figure_individual_proxy.py` | `outputs/figure_individual_proxy.pdf` | Figure 7 |
| `run_nhanes_feasibility.py` | `outputs/figure_nhanes.pdf` | Figure 8 |
| `run_figure_gamma_equivalence.py` | `outputs/figure_gamma_equivalence.pdf` | Figure X (Γ-native equivalence) |
| `run_figure_prior_stress.py` | `outputs/figure_prior_stress.pdf` | Figure Y (prior stress tests) |
| `run_elsa_validation.py` | `outputs/figure_elsa_validation.pdf` | Figure Z (ELSA cohort validation) |

## R4 Revision: Γ-Native Pivot

### Γ-Native Stability Proxy (SWDS-Γ)

The R4 revision replaces the Lyapunov-inversion pipeline with a Γ-native approach that reads stability information directly from the eigenstructure of the observed covariance matrix Γ̂, without estimating the drift matrix A.

SWDS-Γ(Δx) = Δxᵀ Γ̂ Δx / tr(Γ̂)

Key result: SWDS-Γ produces near-identical individual rankings to the A-based SWDS (Spearman > 0.95) while requiring no Q specification, no Lyapunov inversion, and no commutation assumptions.

### Phase 3: ELSA Cohort Validation

The ELSA validation (`scripts/run_elsa_validation.py`) executes the full Γ-native pipeline on longitudinal data from the English Longitudinal Study of Ageing. Requires ELSA data files in `data/elsa/` (see `data/elsa/README.md` for access).

```bash
pip install lifelines  # additional dependency for Cox models
python scripts/run_elsa_validation.py
```

Key analyses:
* Cross-sectional λ_max(Γ̂) by age stratum (replicates NHANES approach for comparison)
* Within-person λ_max(Γ̂_within) by age stratum (key result — tests stability erosion using within-person covariance, avoiding survivorship/medication confounds)
* Individual SWDS-Γ scores and distribution
* 5 nested Cox mortality models (age+sex, +biomarkers, +SWDS-Γ, +Rockwood FI, full)
* Kaplan-Meier survival by SWDS-Γ tertile

### New Estimation Functions

```python
from hdr_sim import (
    # ...existing...
    compute_swds_gamma,          # Γ-native individual stability score
    compute_swds_gamma_batch,    # Batch computation
    gamma_stability_proxy,       # λ_max, κ, eigenstructure from Γ̂
    covariance_sign_concordance, # Tests 3-4 Layer A
    lyapunov_residual_norm,      # Tests 3-4 Layer B residual
)
```
