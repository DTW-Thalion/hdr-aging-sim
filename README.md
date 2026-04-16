# HDR Aging Dynamics Toy Simulation

Numerical simulation of the Homeodynamic Remediation (HDR) framework's aging dynamics,
demonstrating how age-related parameter degradation produces critical slowing-down,
increased vulnerability to perturbation, and basin transitions.

Companion code for: White (2026), "Aging as Controller Failure: A Formal Ontology
for Multi-Axis Physiological Decline."

## Key result

The simulation shows that a multi-axis linear dynamical system with PMID-cited
recovery times (τ_i) and inter-axis coupling (J) loaded from the compiled
`data/J_matrix_compiled_9x9.csv`, when degraded according to known age-related trajectories,
produces:

- Progressive drift of the spectral abscissa toward instability (α → 0⁻)
- Critical slowing-down: recovery timescale diverges as α → 0
- Identical perturbations produce age-dependent responses (fast recovery at 30,
  near-catastrophic excursion at 80)
- Cross-axis propagation of perturbations through the coupling matrix

These are the dynamical signatures of aging predicted by the HDR formalism and
independently observed by Scheffer et al. (2018, PNAS) and Pyrkov et al. (2021,
Nat Commun).

## Two-timescale architecture (mechanistic model)

The 9-axis mechanistic model (`HDRMechanisticModel`) uses a two-timescale
decomposition based on singular perturbation theory:

- **Fast subsystem** (7 axes: I, M, mito, P, C, N, F): perturbation-recovery
  dynamics with tau ranging from 0.003d (N) to 6d (C at age 25) in the V2
  registry. These determine the spectral abscissa, recovery time, and damping
  ratio. Calibrated to alpha_fast(25) = -0.188, stable through age 120.
- **Quasi-static axes** (E, B): tau_E = 500-5000d (epigenetic drift),
  tau_B = 135-500d (bone remodelling). These drift secularly with age and
  enter the fast subsystem as constant forcing, shifting its equilibrium
  without changing eigenvalues.

Note: the V2 mito tau (1-5d) measures bioenergetic functional recovery
(PGC-1a signaling cycle), not mitochondrial protein pool half-life (36d,
Rooyackers 1996). This correction places mito squarely in the fast
cluster with a clean 22x timescale gap to the slow axes (max fast: 6d C,
min slow: 135d B).

All stability metrics (alpha, zeta, recovery time) refer to the fast subsystem.
The full 9x9 system is available via `model.A_full` for validation.

## Setup

```bash
pip install numpy scipy matplotlib
```

For the NHANES feasibility script, additional dependencies are needed:

```bash
pip install pandas pyreadstat
```

For the InCHIANTI replication analysis:

```bash
pip install pandas pyreadstat pyarrow openpyxl
```

## Usage

```bash
python scripts/run_figure_network_schematic.py  # Main Fig 1: 9-axis network diagram
python scripts/run_figure_J_heatmap.py          # Main Fig 2: 9x9 coupling heatmap
python scripts/run_figure2b.py                  # Spectral-abscissa drift (7-axis fast subsystem, 25-120)
python scripts/run_figure2b_v3.py               # V2 fast-subsystem calibration (6-axis, 25-120)
python scripts/run_figure_frailty.py            # Frailty perturbation-response
python scripts/run_figure_t2d.py                # T2D phase portrait
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

Monte Carlo (N = 10,000) over plausible J matrices within confidence-grade uncertainty bands (grade A: 20% CV, grade B: 40% CV, grade C: 70% CV). Supports arbitrary axis subsets via `--axes`.

```bash
python scripts/run_figure_uncertainty.py                  # default 4-axis (I, M, N, F)
python scripts/run_figure_uncertainty.py --axes I M       # 2-axis subset
python scripts/run_figure_uncertainty.py --axes I M E F   # custom 4-axis
```

Key findings (default 4-axis):
* "9/12 positive" structure preserved in 100% of draws
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

`data/J_matrix_compiled_9x9.csv` is the **single source of truth** for the 9×9 coupling matrix J_mech. It contains all 72 off-diagonal entries of the 9 regulatory capacity axes. A frozen copy of the former 8×8 version is archived in `data/legacy/` for reproducibility of intermediate analyses.

### PMID-Cited Recovery Time Constants (tau registry)

| Axis | Name | tau(25) | tau(80) | tau(120) | Trajectory | PMID |
|------|------|---------|---------|----------|------------|------|
| I | Inflammatory resolution capacity | 4d | 17d | 45d | Gompertz | 27467771 |
| M | Metabolic regulatory gain | 0.08d | 0.21d | 0.35d | Piecewise-linear | 18268070 |
| E | Epigenetic maintenance fidelity | 500d | 2000d | 5000d | Piecewise-exp | 15509558 |
| mito | Mitochondrial bioenergetic capacity | 1.0d | 2.0d | 5.0d | Piecewise-linear | 12563009 |
| P | Proteostatic clearance capacity | 1.5d | 3.0d | 4.0d | Piecewise-linear | 24437518 |
| C | Circadian oscillator amplitude | 6.0d | 10d | 18d | Piecewise-linear | 1557592 |
| N | Neuroendocrine feedback gain | 0.003d | 0.005d | 0.008d | Piecewise-linear | 29581219 |
| F | Functional reserve (musculoskeletal) | 2.0d | 3.5d | 6.0d | Piecewise-Gompertz | 9252485 |
| B | Bone remodelling regulatory balance | 135d | 250d | 500d | Piecewise-linear | 3213608 |

All tau values cite primary literature for the mechanistic recovery process (e.g., mito measures PGC-1a bioenergetic signaling cycle, not protein pool half-life; F measures muscle protein synthesis return to baseline, not remodelling).

CSV columns include:

- **Basin-stratified values**: `J_healthy`, `J_pre_disease`, `J_disease` (SD-per-SD units)
- **Age trajectory**: `increasing`, `decreasing`, `stable`, or `unknown`
- **Evidence metadata**: sign, magnitude tier (S/M/W/unknown), confidence grade (A/B/C), primary PMID, evidence type, mechanism brief, and biomarker role notes

Sign distribution (9×9): 57 positive (pathological), 11 negative (protective), 4 unknown.

**How it works**: `hdr_sim.aging_params.configure()` calibrates the 7-axis fast subsystem (I, M, mito, P, C, N, F) via `calibrate_stable_system()`, guaranteeing stability at all ages 25-120. It then extracts the requested axis subset. The calibration scalar (c=0.890) and Gompertz blend amplitude (A=0.00236) are computed once and cached. Axis subsets from 2×2 through 9×9 are supported; all subsets inherit the fast-subsystem stability guarantee.

```python
from hdr_sim import configure, tau_of_age, J_of_age
from hdr_sim import load_J_csv, build_J_basin_imputed
from hdr_sim import tau_at_age, tau_vector
from hdr_sim.aging_params import get_fast_system

# Standard configuration (any axis subset, stable 25-120)
configure(axes=('I', 'M', 'F'))       # 3-axis subset
tau = tau_of_age(65)                   # 3-element array
J = J_of_age(65)                      # 3×3 matrix (stable at all ages)

configure(axes=('I', 'M', 'N', 'F'))  # 4-axis subset
tau = tau_of_age(100)                  # works for ages 25-120
J = J_of_age(100)                     # Gompertz-interpolated

# Direct tau access (no configure needed)
tau_I_50 = tau_at_age('I', 50)            # single axis, single age
tau_vec = tau_vector(('I', 'M', 'F'), 65) # multi-axis vector

# Fast-subsystem dynamics (7-axis, guaranteed stable)
A_full, A_fast, alpha_fast, alpha_full = get_fast_system(80)
# alpha_fast ~ -0.108, recovery ~ 9.2 days

# Raw J matrix access
rows = load_J_csv()
J_imp = build_J_basin_imputed(rows, basin='healthy',
                              axes=('I','M','E','mito','P','C','N','F','B'))

# Provenance tracking via JMatrixSpec
from hdr_sim.j_matrix_spec import JMatrixSpec
spec = JMatrixSpec.from_csv('data/J_matrix_compiled_9x9.csv')
print(spec.sha256, spec.sign_counts)  # deterministic identity
```

The full 9×9 CSV contains entries derived from systematic literature review using evidence triangulation (statistical, molecular, causal). See White (2026), "The Mechanistic Coupling Matrix J_mech" for methodology.

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
| `run_medication_sensitivity.py` | `outputs/elsa_medication_sensitivity.json` | R5 corrected Cox models |
| `run_figure_coupling_tightening.py` | `outputs/figure_coupling_tightening.pdf` | R6 coupling tightening |
| `run_figure_mortality_prediction.py` | `outputs/figure_mortality_prediction.pdf` | R6 mortality prediction |
| `run_figure_medication_compression.py` | `outputs/figure_medication_compression.pdf` | R6 medication compression |
| `run_figure_network_schematic.py` | `outputs/figure_network_schematic.pdf` | Main Fig 1: 9-axis network diagram |
| `run_figure_J_heatmap.py` | `outputs/figure_J_heatmap.pdf` | Main Fig 2: 9×9 annotated J coupling heatmap |
| `run_dj_primacy.py` | `outputs/figure_dj_primacy.pdf` | R6 D vs J primacy (3-panel) |
| `run_figure_dj_pairwise.py` | `outputs/figure_dj_pairwise.pdf` | Supp Fig 4: pairwise variance/correlation vs age |
| `run_dj_validation.py` | `outputs/figure_dj_validation.pdf` | Supp Fig 5: D vs J validation (2×2 panel) |
| `run_dj_power.py` | `outputs/figure_dj_power.pdf` | Supp Fig 6: D vs J power analysis (3-panel) |
| `run_dj_bayes_robust.py` | `outputs/figure_dj_bayes_robust.pdf` | Supp Fig 7: Bayesian + misspecification robustness |
| `run_pi_regime_analysis.py` | `outputs/pi_regime_analysis.json` | SI Note 6: Π strong-coupling regime analysis |
| `run_counterintuitive_predictions.py` | `outputs/counterintuitive_predictions.json` | Coupling-matrix counterintuitive predictions (5 tests) |
| `run_estimator_bias.py` | `outputs/figure_estimator_bias.pdf` | Supp Fig 8: Change-covariance estimator bias (2×2 panel) |
| `run_elsa_ici_deployment.py` | `outputs/elsa_ici_deployment.json` | ELSA ICI deployment assessment |
| `run_elsa_ici_deployment.py` | `outputs/elsa_ici_deployment_table.txt` | ICI deployment manuscript table |
| `run_elsa_basin_recovery.py` | `outputs/elsa_basin_recovery.json` | Data-driven basin recovery (GMM/HMM) |
| `run_elsa_basin_recovery.py` | `outputs/elsa_basin_recovery_table.txt` | Basin recovery comparison table |
| `run_figure_disease_demos.py` | `outputs/figure_disease_demos.pdf` | ED Fig 2: disease demo panels (T2D, frailty, AD, osteoporosis) |
| `run_figure2b_v2.py` | `outputs/figure_2b_v2.pdf` | V2: Spectral-abscissa drift with lit-calibrated tau (4-axis) |
| `run_figure2b_v3.py` | `outputs/figure_2b_v4.pdf` | V2.2: Spectral drift with 7+2 fast-subsystem calibration (25-120) |
| `run_stability_verification.py` | `outputs/stability_verification_25_120.json` | V2: Stability sweep across 25-120, 4 axis configs |
| `run_stability_verification_v2.py` | `outputs/stability_verification_v2.json` | V2: Fast-subsystem calibration (6-axis + 7-axis) |
| `run_monotonicity_25_120.py` | `outputs/monotonicity_25_120.json` | V2: Monotonicity re-verification with V2 tau |
| `run_monotonicity_v2.py` | `outputs/monotonicity_v2.json` | V2: Fast-subsystem monotonicity (6-axis, Metzler) |
| `run_tau_comparison.py` | `outputs/tau_comparison_old_vs_new.md` | V2: Legacy vs lit-calibrated tau comparison table |

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
* Kaplan-Meier survival by SWDS-Γ tertile (full sample; medication-naive subgroup shows no significant separation)

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

## R5 Revision: Medication Sensitivity and Corrected Cox Models

### Corrected Medication Sensitivity Analysis

The R5 revision adds corrected medication sensitivity analyses with matched-sample Cox models.

```bash
python scripts/run_medication_sensitivity.py
```

Key corrections in R5:
- **1a**: All 5 nested Cox models run on the same matched sample (same N)
- **1b**: 3-axis M2 uses exactly R4 covariates (no sysval)
- **1c**: hemda/hemdb excluded from Cox adjustment covariates
- **1d**: SWDS-Γ uses cross-sectional stratum covariance
- **1e**: Survival time construction matches R4 exactly

## R6 Main Figures: Network Schematic and Coupling Heatmap

### 9-Axis Network Schematic (Main Figure 1)

Publication-quality network diagram of the full 9-axis HDR coupling matrix, loaded from `data/J_matrix_compiled_9x9.csv`.

```bash
python scripts/run_figure_network_schematic.py
```

9 nodes in circular layout with edges coloured by sign: red = pathological (+), blue = protective (−), grey dashed = unknown/qualitative only. Edge width proportional to |J_ij|.

### 9×9 J Coupling Heatmap (Main Figure 2)

Annotated heatmap of the compiled mechanistic coupling matrix J (disease basin).

```bash
python scripts/run_figure_J_heatmap.py
```

Diverging RdBu_r colour map. Cells annotated with numeric values and confidence grades (A/B/C). Diagonal cells greyed out (self-restoration). Unknown entries marked with '?' and hatching.

## R6 Analysis: D vs. J Primacy Decomposition

Decomposes the age-dependent growth in Γ̂ into within-axis variance growth (D-degradation) vs. cross-axis correlation tightening (J-degradation), testing whether aging is primarily driven by loss of individual regulatory capacity (damage/stochastic theories) or by strengthening of pathological inter-axis coupling (hyperfunction theory).

```bash
python scripts/run_dj_primacy.py
```

This is a **Tier-1 analysis**: uses only the sample covariance Γ̂, no drift estimation, no diffusion Q specification, no structural assumptions on A beyond the OU model.

Key findings:
* **Both V_norm and C_norm decline** with age (p < 0.01 for both), reflecting medication-compressed covariance in older strata
* **Primacy ratio P ≈ 1.0** across all strata in the medication-naive subgroup (slope = +0.0014/yr, p = 0.456)
* **Proportional co-degradation**: D and J degrade in lock-step — neither damage/stochastic nor hyperfunction theories alone dominate
* The I↔M coupling is the dominant pair (~0.30–0.40), while I↔F and M↔F are weak (~0.05–0.10)
* F-axis (functional) variance declines steeply with age — survivorship bias (weakest die first)

| Script | Output | Description |
|--------|--------|-------------|
| `run_dj_primacy.py` | `outputs/figure_dj_primacy.pdf` | 3-panel: V/C trends, primacy ratio (full), primacy ratio (med-naive) |
| `run_figure_dj_pairwise.py` | `outputs/figure_dj_pairwise.pdf` | 4-panel: per-axis variances and pairwise correlations (ELSA with synthetic fallback) |
| `run_dj_primacy.py` | `outputs/dj_primacy_results.txt` | Full numerical results and interpretation |
| `run_dj_primacy.py` | `outputs/dj_primacy_results.json` | Machine-readable results |

### Pairwise Variance and Correlation Growth (Supplementary Figure 4)

Standalone figure showing individual axis variances (Γ_II, Γ_MM, Γ_FF) and pairwise absolute correlations |r(I,M)|, |r(I,F)|, |r(M,F)| across age strata, comparing full sample vs medication-naive subgroup.

```bash
python scripts/run_figure_dj_pairwise.py
```

If ELSA data is available in `data/elsa/`, uses real data. Otherwise generates synthetic illustration data from the HDR model (clearly labelled). Bootstrap 95% CIs with 1000 resamples.

### D vs. J Primacy Simulation Validation

Simulation-based validation that the primacy ratio P = C_norm / V_norm can discriminate D-only from J-only from proportional degradation under realistic strong-coupling conditions (ε ~ 0.9–3.6) with confounds. Responds to the reviewer objection that weak-coupling intuition may not hold at the HDR parameterization.

```bash
python scripts/run_dj_validation.py
```

Design:
* **Phase 1** — Ground truth discrimination: 5 degradation regimes (Pure D, 75D/25J, 50D/50J, 25D/75J, Pure J) with total spectral-abscissa drift α(A) held constant via binary-search calibration. 7 age strata, N=5,000 samples/stratum, 200 MC runs.
* **Phase 2** — Confounds: survivorship bias (top 5%/decade removed by ||Δx||²) and medication compression (I/M variance × 0.6 for age-increasing fraction of individuals).
* **Phase 3** — Discrimination power and minimum detectable effect (MDE) analysis.

Key findings:
* **P discriminates D-only from J-only** even under strong coupling: Pure D yields negative P-slopes (~−0.010/yr), Pure J yields positive slopes (~+0.022/yr), with massive effect sizes (Cohen's d > 3, all p < 10⁻⁸⁰)
* **Adjacent-regime discrimination power ≥ 0.85** for all pairs under clean conditions; all pairs achieve p < 0.05
* **Monotone ordering survives realistic confounds** (survivorship + medication): P-slopes remain ordered across D/J regimes under the "Both" condition
* **MDE at 80% power**: 0.0004/yr (3-axis), 0.0007/yr (4-axis) — the ELSA observed P-slope of +0.0014/yr exceeds the MDE, confirming adequate study power
* **ELSA result is consistent with proportional co-degradation** (50D/50J mean slope = +0.005/yr; observed +0.0014/yr falls within the simulated distribution)

Both 3-axis (I, M, F) and 4-axis (I, M, N, F) models produce qualitatively identical conclusions.

| Script | Output | Description |
|--------|--------|-------------|
| `run_dj_validation.py` | `outputs/figure_dj_validation.pdf` | 2×2 panel: P(s) curves and P-slope vs D-fraction |
| `run_dj_power.py` | `outputs/figure_dj_power.pdf` | 3-panel: discrimination power, MDE vs sample size |
| `run_dj_validation.py` | `outputs/dj_validation_results.txt` | Full numerical results, power stats, and interpretation |
| `run_dj_validation.py` | `outputs/dj_validation_results.json` | Machine-readable results |
| `run_dj_validation.py` | `outputs/dj_validation_summary.md` | SI-ready markdown for supplementary materials |

### Standalone Power Analysis (Supplementary Figure 6)

Standalone script for the discrimination power figure, including a MDE-vs-sample-size curve not present in the `run_dj_validation.py` output.

```bash
python scripts/run_dj_power.py
```

Three panels: (a) discrimination power between adjacent regime pairs (3-axis), (b) same for 4-axis, (c) MDE curve vs sample size per stratum (N = 500–10,000) with observed ELSA slope (+0.0014/yr) and 3-axis MDE (0.0004/yr) marked.

### Bayesian Model Comparison & Misspecification Robustness

Extends the simulation validation with (A) Bayesian model comparison and TOST equivalence test for the observed ELSA P-slope, and (B) misspecification robustness under three violations of OU model assumptions.

```bash
python scripts/run_dj_bayes_robust.py
```

**Analysis A — Bayesian model comparison:**
* Bayes factors computed using simulation-calibrated P-slope distributions as empirical priors and the ELSA medication-naive P-slope (+0.0014/yr) as the datum
* **50D/50J receives the highest posterior probability (0.456)** under uniform prior across 5 regimes
* BF vs Pure D = 4.7 (substantial), vs Pure J = 386 (decisive), vs 25D/75J = 3.6 (substantial), vs 75D/25J = 1.4 (anecdotal)
* TOST equivalence test does not reject (p = 0.61) — the observed slope falls slightly below the proportional regime's equivalence region, though the Bayesian analysis provides complementary support

**Analysis B — Misspecification robustness:**
* **M1 (Correlated noise, Q off-diag ρ=0.3)**: monotone ordering broken between adjacent D-dominated regimes; endpoint discrimination (Pure D vs Pure J) preserved with power ≥ 0.84
* **M2 (Mild nonlinearity, 10% quadratic)**: monotone ordering preserved; wider CIs from Euler-Maruyama simulation reduce power for adjacent pairs
* **M3 (Latent omitted axis, 4→3-axis projection)**: monotone ordering preserved; discrimination results nearly identical to baseline

| Script | Output | Description |
|--------|--------|-------------|
| `run_dj_bayes_robust.py` | `outputs/figure_dj_bayes_robust.pdf` | 5-panel: posteriors, TOST, M1/M2/M3 robustness |
| `run_dj_bayes_robust.py` | `outputs/dj_bayes_robust_results.json` | Machine-readable Bayes factors, TOST, power |
| `run_dj_bayes_robust.py` | `outputs/dj_bayes_robust_summary.md` | SI-ready markdown (Section S9) |

### Π Statistic — Strong-Coupling Regime Analysis

Analyses whether the primacy ratio Π = C_norm / V_norm retains its discrimination property at the strong-coupling strengths (ε = ρ(D⁻¹J) ~ 0.45–3.17) present in the HDR parameterisation, where the perturbative expansion that motivates Π is quantitatively inaccurate.

```bash
python scripts/run_pi_regime_analysis.py
```

Key findings:
* **Coupling strength**: ε_spectral ranges from 0.45 (age 30) to 3.17 (age 80) in the 3-axis model, exceeding unity for ages ≥ 50; the 7-axis model remains perturbative (ε < 0.5)
* **Perturbative error**: First-order correlation approximation incurs 18%–131% relative error (3-axis), confirming the expansion is quantitatively uninformative at these coupling strengths
* **Monotonicity**: Simple pure-D/pure-J monotonicity of Π does not hold in the strong-coupling regime
* **Conclusion**: Π is justified by the simulation validation (Supp Figs 5–7), not by the perturbative expansion — the simulation demonstrates empirical discrimination with Cohen's d > 3 under realistic confounds

| Script | Output | Description |
|--------|--------|-------------|
| `run_pi_regime_analysis.py` | `outputs/pi_regime_analysis.json` | Coupling strengths, weak-coupling errors, monotonicity tests |

### Change-Covariance Estimator Bias Study

Validates the approximation Γ̂_change ≈ 2Γ used by the visit-pair change-covariance estimator. The R6 claim that λ_max(Γ̂_change) increases with age relies on the assumption that between consecutive ELSA visits (~4 years apart) the OU process fully equilibrates.

```bash
python scripts/run_estimator_bias.py
```

Design:
* **Panel (a)** — Bias ratio λ_max(Γ̂_change) / (2 λ_max(Γ_true)) as a function of visit interval Δt, for ages 30, 50, 65, 80. N=5,000 samples, 200 MC runs per condition.
* **Panel (b)** — Recovered λ_max age trend at five visit intervals (1 month to 4 years) vs true λ_max(Γ), with 95% CI bands.
* **Panel (c)** — Same as (b) with survivorship bias (χ² 90th percentile threshold) and medication compression (40% medicated, 0.7× variance).
* **Panel (d)** — Sensitivity to τ-scaling: bias ratio at age 80 with τ multiplied by 1× to 20×, holding J fixed.

Key findings:
* **Bias at 4 yr / age 80: 1.001** (0.1% — essentially unbiased)
* **Min Δt for <1% bias: 180 days** (6 months) — ELSA's 4-year cadence provides a >7× safety margin
* **τ-scaling threshold: 4.0×** — at k=4 the recovery time (1,850 days) exceeds the visit interval, causing 37.5% bias; at k=4.5 the system becomes unstable
* **7-axis monotone trend: preserved** — the λ_max increase with age holds for the full 7-axis (I, M, mito, P, C, N, F) model at 4-year cadence

| Script | Output | Description |
|--------|--------|-------------|
| `run_estimator_bias.py` | `outputs/figure_estimator_bias.pdf` | 2×2 panel: bias ratio, age trends (clean/confounded), τ-scaling |
| `run_estimator_bias.py` | `outputs/estimator_bias_results.json` | Machine-readable bias ratios, trend recovery, τ sensitivity |

### Counterintuitive Predictions from the Coupling Matrix

Systematic search through the calibrated J matrix and ELSA data for predictions that are derivable from the coupling structure, non-obvious, and testable with existing data.

```bash
python scripts/run_counterintuitive_predictions.py
```

Five tests, ranked by strength of evidence:

1. **Asymmetric coupling lead-lag** (p=7×10⁻⁵, Bonferroni-corrected): The J matrix is asymmetric — J[M→I] > J[I→M] at all ages. This predicts that metabolic deterioration temporally precedes inflammatory deterioration. Cross-lagged analysis on 1,963 visit triples confirms: r(ΔM₁₂,ΔI₂₃)=−0.090 vs r(ΔI₁₂,ΔM₂₃)=−0.002. This is a directional prediction that symmetric or undirected frameworks cannot make.
2. **Partial correlation structure** (83% sign concordance): The coupling matrix correctly predicts which axis pairs show positive vs negative partial correlations across 4 age strata — a stronger structural test than overall covariance growth.
3. **Coupling-direction dependent mortality** (p_adj=0.008): Displacement along the dominant covariance eigenvector predicts mortality (r=0.043) better than orthogonal displacement (r=0.020), confirming that the DIRECTION of dysregulation matters, not just magnitude.
4. **Age-dependent eigenvector rotation**: Model predicts 38° rotation toward F-axis; observed rotation direction does not match, likely due to medication compression distorting the observed covariance at older ages.
5. **Non-monotone axis-specific variance**: Model predicts all axes monotone-increasing, but ELSA data shows F-axis and M-axis variance decreasing at older ages (survivorship bias + medication compression dominate the raw signal).

| Script | Output | Description |
|--------|--------|-------------|
| `run_counterintuitive_predictions.py` | `outputs/counterintuitive_predictions.json` | 5 structural prediction tests against ELSA data |

### ELSA ICI Deployment Assessment

Retrospective evaluation of the ICI deployment threshold (ω_min) against real ELSA longitudinal biomarker data, determining whether existing cohort data density is sufficient for safe closed-loop deployment.

```bash
python scripts/run_elsa_ici_deployment.py
```

Key findings:
* **N = 6,245** participants with ≥2 complete 3-axis waves (I, M, F) across waves 2/4/6/8 (2004–2016)
* **d_min = 0.021** — low basin separability between healthy (age 50–64) and disease (65+) covariance structures
* **ω_min = 144** at μ̄ = 0.05 — far exceeds the max T_k^eff = 1.4 available from ELSA's 4-yearly cadence
* **0% of participants** meet the deployment threshold under ELSA actual sensing
* **Daily wearable sensing** would meet the threshold after ~18 months — a concrete pilot study design target
* Basin assignment sensitivity: age-based and score-based methods produce qualitatively consistent conclusions (both confirm insufficient ELSA density), though score-based yields higher d_min (0.42 vs 0.02)

| Script | Output | Description |
|--------|--------|-------------|
| `run_elsa_ici_deployment.py` | `outputs/elsa_ici_deployment.json` | Full machine-readable results |
| `run_elsa_ici_deployment.py` | `outputs/elsa_ici_deployment_table.txt` | Manuscript summary table |

### R6 Pipeline Audit

Adversarial review identified three items requiring resolution before manuscript submission.

#### C-index Reconciliation

The pipeline's `run_matched_cox()` now prints both `CoxPHFitter.concordance_index_` (the fitted model's own Harrell's C) and `lifelines.utils.concordance_index()` for comparison. Result: the two methods produce **identical values to 6 decimal places** — there is no sign convention or partial hazard computation issue.

Pipeline C-indices (authoritative):

| Model | Full (N=5,431) | Med-naive (N=3,233) |
|-------|----------------|---------------------|
| M1: Age + Sex | 0.6019 | 0.5860 |
| M2: + Biomarkers | 0.6145 | 0.6050 |
| M3: + SWDS-Γ | 0.6030 | 0.5861 |
| M4: + Rockwood FI | 0.6093 | 0.6000 |
| M5: Full | 0.6180 | 0.6131 |
| **ΔC (M5−M4)** | **+0.009** | **+0.013** |

The ΔC values are consistent with manuscript Table 1. The absolute C-index offset (~0.10–0.14) reflects a different analysis specification in the manuscript; the pipeline values are the authoritative source.

#### Biomarker Specification

Definitive biomarker definitions are documented in `outputs/biomarker_specification.txt`:

- **I axis (3-axis)**: `dx_I = z(log(CRP))` — CRP only
- **M axis (3-axis)**: `dx_M = (z(HbA1c) + z(BMI)) / √2` — HbA1c and BMI
- **F axis (3-axis)**: `dx_F = −z(grip_max)` — grip strength only (reversed)
- **Cox M2 covariates**: `log_crp, hba1c, grip_max, bmival` (raw biomarkers, not composite axes)

The 4-axis model extends with fibrinogen (I_4), chol/HDL + triglycerides (M_4), BP + pulse (N_4), and gait speed (F_4).

#### Extended Data Figure 2: Disease Demonstration Panels

```bash
python scripts/run_figure_disease_demos.py
```

Four-panel composite (`outputs/figure_disease_demos.pdf`):
- **(a) T2D**: Phase portrait in ΔxI–ΔxM plane with bifurcation separatrix
- **(b) Frailty**: Spectral radius ρ(Φ) approaching unity with age, impulse response inset
- **(c) Alzheimer's disease**: Threshold coupling J_{Aβ→τ} in {I, P_neural, mito} submatrix; irreversible neuronal loss partition above amyloid threshold
- **(d) Osteoporosis**: B-axis coupling submatrix (I→B, M→B, N→B pathological; F→B protective) with sarcopenia compounding effect

Coupling values sourced from `data/J_matrix_compiled_9x9.csv`.

| Script | Output | Description |
|--------|--------|-------------|
| `run_figure_disease_demos.py` | `outputs/figure_disease_demos.pdf` | ED Fig 2: 4-panel disease demos |
| — | `outputs/biomarker_specification.txt` | Definitive biomarker specification table |

### R6 Utility Scripts

| Script | Purpose |
|--------|---------|
| `verify_J_counts.py` | Audits J matrix CSV sign counts (57+, 11−, 4? for 9×9) with `--csv` for any CSV. Prints SHA-256 and `JMatrixSpec` summary. |
| `sync_j_from_companion.py` | Copies latest J-matrix CSV from companion repo, compares against provenance snapshot, appends to `data/sync_log.json`. Use `--dry-run` to preview. |
| `compare_j_runs.py` | Compares two pipeline JSON outputs (from different J versions), producing `j_comparison_report.json` and `.md`. |
| `run_j_comparison_integration.py` | Integration test: runs pipeline twice (provenance vs default CSV), verifies zero diffs. |
| `populate_pmids.py` | Populates missing `pmid_primary` entries in `J_matrix_compiled_9x9.csv` |
| `update_ledger_r6.py` | Updates `outputs/elsa_results_ledger.json` with SHA-256 hashes for R6 figure files |

### Reproducibility

`outputs/elsa_results_ledger.json` contains SHA-256 hashes of all data files and every numerical result reported in Appendix H. Use this to verify that your ELSA data extract and pipeline output match ours.

## Tau Registry and Calibration Architecture

The simulation uses PMID-cited recovery time constants at three age anchors (25, 80, 120) with axis-specific trajectory shapes, covering ages 25-120.

### Key features

1. **PMID-cited tau values** (`TAU_REGISTRY`): Three-anchor recovery time constants with Gompertz, saturating-exponential, and piecewise-linear trajectory functions. Each axis cites the primary literature for the specific recovery process measured (e.g., mito = PGC-1a bioenergetic signaling cycle, not protein pool half-life; F = muscle protein synthesis return to baseline).

2. **Qual-only imputation** (`build_J_basin_imputed()`): Fills qual_only entries from tier defaults (S=0.20/0.40, M=0.10/0.20, W=0.05/0.10), giving 68/72 (94%) nonzero J entries.

3. **Gompertz J interpolation** (`j_at_age_blended()`): Non-linear aging trajectory for the coupling matrix with Gompertz-shaped blending function and tunable blend amplitude.

4. **Fast-subsystem calibration** (`calibrate_stable_system()`): Jointly finds the coupling scalar c and J blend amplitude A ensuring the 7-axis fast subsystem remains stable from age 25 to 120:
   - c = 0.890, alpha_fast(25) = -0.188, alpha_fast(120) = -0.004
   - Monotonic critical slowing-down (recovery: 5.3d at 25 -> 250d at 120)
   - J blend amplitude A = 0.00236 (tau degradation dominates J evolution)

### Two-timescale calibration architecture

The single-scalar calibration fails for the full 9x9 system because the E axis (tau_E=500d) constrains alpha to -0.002. The correct approach calibrates on the **7-axis fast subsystem** (I, M, mito, P, C, N, F) explicitly:

| Cluster | Axes | tau range (age 25) | Role |
|---------|------|-------------------|------|
| Fast | I, M, mito, P, C, N, F | 0.003-6d | Perturbation-recovery dynamics; alpha, SWDS-Gamma, Lambda_max |
| Quasi-static | E, B | 135-500d | Secular drift; constant forcing on fast subsystem |

The timescale separation is 22x (max fast: 6d C, min slow: 135d B), providing clean singular perturbation decomposition.

The calibration procedure:
1. Find (c, amplitude) such that alpha_fast(120) = -0.004 with stability at all ages
2. Apply the same c to the full 9x9 J matrix
3. Report alpha_fast (empirically testable) and alpha_full (E-dominated)

Note on mito tau: the V2.2 registry uses bioenergetic functional recovery (PGC-1a signaling cycle, 1-5d) rather than the mitochondrial protein pool half-life (36d, Rooyackers 1996). This is the same conceptual distinction applied to the I axis (CRP half-life vs inflammatory resolution programme duration).

### Stability findings

| Configuration | Max stable age | alpha(25) | Recovery at 120 | Notes |
|--------------|---------------|-----------|-----------------|-------|
| **7-axis fast (I,M,mito,P,C,N,F)** | **120** | **-0.188** | **250d** | Primary calibration; all axis subsets inherit stability |
| Any axis subset (e.g., I,M,F) | 120 | -0.375 | N/A | Extracted from fast subsystem; always stable |
| Full 9×9 system | ~48 | -0.018 | -- | alpha_full goes positive (expected: E-axis dominated) |

### V2 scripts

```bash
# Fast-subsystem calibration (primary, corrected mito tau)
python scripts/run_stability_verification_v2.py # 7-axis + 6-axis stability sweep
python scripts/run_monotonicity_v2.py           # Monotonicity verification
python scripts/run_figure2b_v3.py               # Spectral drift figure (v4, 25-120)

# Initial V2 exploration (full 9x9 and subsystem comparison)
python scripts/run_stability_verification.py    # All configurations comparison
python scripts/run_monotonicity_25_120.py       # Monotonicity 9x9/7-axis/4-axis
python scripts/run_figure2b_v2.py               # V2 figure (4-axis)
python scripts/run_tau_comparison.py            # Old vs new tau comparison table
```

### Configuration

As of v2.5, there is a single configuration path. `configure()` uses the 9×9 compiled CSV, PMID-cited tau registry, and fast-subsystem calibration by default. The former `configure_v2()` function is still available but `configure()` produces identical results. The legacy 2-anchor tau values and 8×8 CSV are archived in `data/legacy/` and `TAU_REGISTRY_LEGACY_FROZEN` for reproducibility of intermediate analyses only.

## Part 2: Mechanistic-Evidence-Informed Model

Part 2 extends the 4-axis toy simulation with a 9-axis dynamical system parameterised from the enriched mechanistic evidence base produced by the HDR-mechanistic repository. All Part 2 code is **purely additive** — the original R1–R6 scripts and modules are unmodified.

**Note:** The files in `data/mechanistic_evidence/` are automatically synced from `DTW-Thalion/HDR-mechanistic` via GitHub Actions whenever the evidence exports are updated. No manual copy step is needed.

### Architecture

```
data/mechanistic_evidence/
  J_matrix_mechanistic_9x9.json   ← 39 active entries with age-interpolated values
  prior_specification.json         ← per-entry Bayesian priors (truncated normal)
  intervention_library.json        ← 18 interventions with delta_J_fraction
  state_conditioning_export.json   ← 7 healthy/disease switching specs

src/hdr_sim/
  mechanistic_model.py     ← HDRMechanisticModel (9-axis A = -D + J)
  state_conditioned.py     ← StateSwitchedModel (healthy/disease SLDS)
  sensitivity.py           ← PriorSensitivityAnalysis (MC, OAT, Sobol)
  prior_stress.py          ← PriorStressTest (concordance, ablation)
  observation_model.py     ← ObservationModel (C matrix, ELSA/InCHIANTI configs)
  synthetic_cohort.py      ← SyntheticCohort + CohortData
  tier1_pipeline.py        ← Tier1Pipeline (Gamma_change, SWDS, primacy)
  intervention.py          ← InterventionModel (18 evidence-based interventions)
  trial_simulator.py       ← TrialSimulator (RCT, factorial, R6 design)
  bayesian_update.py       ← BayesianPriorUpdate (ABC scaffold)
```

### Quick start

```bash
pip install -r requirements_part2.txt

# Run the full end-to-end pipeline (~5s)
python scripts/run_full_pipeline.py

# Run with a specific J-matrix CSV and output directory
python scripts/run_full_pipeline.py --j-matrix data/provenance/J_R6_ontology_v1.6.csv --output-dir results/provenance/

# Run individual analyses (all accept --j-matrix)
python scripts/run_sensitivity.py          # MC sensitivity (10K draws)
python scripts/run_synthetic_validation.py  # Synthetic cohort validation
python scripts/run_intervention_analysis.py # Intervention ranking + factorial
python scripts/run_dj_primacy_mechanistic.py # D/J primacy with mechanistic priors
```

Reports are written to `results/` (or `--output-dir` if specified).

### Module descriptions

| Module | Class | Purpose |
|--------|-------|---------|
| `mechanistic_model` | `HDRMechanisticModel` | 9-axis dynamical system with age-dependent A = -D + J, calibrated via Brent's method. Loads enriched J matrix from JSON export. |
| `state_conditioned` | `StateSwitchedModel` | Switched linear system with 7 state-conditioned entries that change between healthy and disease regimes. |
| `sensitivity` | `PriorSensitivityAnalysis` | Monte Carlo uncertainty propagation through J priors. 10,000-draw MC, one-at-a-time sensitivity, variance-based Sobol indices. |
| `prior_stress` | `PriorStressTest` | Stress tests: correct/null/adversarial concordance, grade ablation, exclusion impact, decomposition vs uniform priors. |
| `observation_model` | `ObservationModel` | Observation matrix C projecting 9-dim latent state to biomarker space (ELSA 3-axis, InCHIANTI 4-axis, full 9-axis configs). |
| `synthetic_cohort` | `SyntheticCohort` | Longitudinal cohort generation matching ELSA design: N persons, discrete visits, survivorship bias, medication effects. |
| `tier1_pipeline` | `Tier1Pipeline` | Computes Tier-1 observables: lambda_max(Gamma_change), lambda_max(Gamma_cross), SWDS-Gamma, primacy ratio Pi. |
| `intervention` | `InterventionModel` | Applies 18 evidence-based interventions to J/tau with sign-aware convention. Ranks by stability improvement. |
| `trial_simulator` | `TrialSimulator` | In silico RCT and 2^k factorial designs. Replicates R6 proposed colchicine x exercise x circadian hygiene trial. |
| `bayesian_update` | `BayesianPriorUpdate` | ABC scaffold for updating J priors from Tier-1 cohort observables within the identifiable subspace. |

### Scripts and outputs

All pipeline scripts accept `--j-matrix <path>` for J-matrix selection and embed `JMatrixSpec` provenance in their JSON output.

| Script | Output | Description |
|--------|--------|-------------|
| `run_full_pipeline.py` | `results/full_pipeline_report.md` | End-to-end 10-step pipeline. Accepts `--output-dir`. |
| `run_sensitivity.py` | `results/sensitivity_report.md` | MC sensitivity analysis (10K draws), entry ranking, stress tests |
| `run_synthetic_validation.py` | `results/synthetic_validation_report.md` | ELSA-like synthetic cohort under 4 confound conditions |
| `run_intervention_analysis.py` | `results/intervention_report.md` | Intervention ranking, R6 factorial. Accepts `--output-dir`. |
| `run_dj_primacy_mechanistic.py` | `results/dj_primacy_mechanistic.json` | D/J primacy with 5 degradation regimes x 4 confound conditions |

### J-matrix provenance and comparison workflow

The J-matrix CSV is the primary input to all simulation and validation pipelines.
Every script accepts `--j-matrix <path>` to override the default CSV, and every
JSON output includes a `j_matrix` key with the SHA-256 hash, sign counts, and
axis set of the CSV used. A frozen provenance snapshot is stored in
`data/provenance/J_R6_ontology_v1.6.csv`.

**Syncing a new J-matrix from the companion repo:**

```bash
# 1. Sync the new CSV
python scripts/sync_j_from_companion.py --source ../hdr-jmatrix-mechanistic/data/J_matrix_mechanistic_9x9.csv

# 2. Run the full test suite with the new J
bash run_all_with_results.sh

# 3. Compare against provenance baseline
python scripts/compare_j_runs.py \
    --baseline outputs/R6_provenance/elsa_results_ledger.json \
    --candidate outputs/latest/elsa_results_ledger.json

# 4. Review the diff — if any test outcomes changed, investigate
cat outputs/j_comparison_report.md
```

**Testing a hypothetical J (e.g., what if the E-I coupling is strong?):**

```bash
# Edit a copy of the CSV
cp data/J_matrix_compiled_9x9.csv /tmp/J_hypothetical.csv
# ... edit /tmp/J_hypothetical.csv ...

# Run the pipeline against it
python scripts/run_elsa_validation.py --j-matrix /tmp/J_hypothetical.csv

# Compare
python scripts/compare_j_runs.py \
    --baseline outputs/R6_provenance/elsa_results_ledger.json \
    --candidate outputs/latest/elsa_results_ledger.json
```

**Running the parameterisation sanity check:**

```bash
# Runs pipeline twice (provenance vs default) and verifies zero diffs
python scripts/run_j_comparison_integration.py

# Or as part of the full suite
bash run_all_with_results.sh --compare-j
```

### Relationship to HDR-mechanistic

The `data/mechanistic_evidence/` directory contains exports from the [HDR-mechanistic](https://github.com/DTW-Thalion/HDR-mechanistic) repository (pipeline v3.5). That repository performs the mechanistic decomposition of each J entry into molecular pathway steps, validates against ChEMBL, and produces the enriched evidence base consumed here. This repository (hdr-aging-sim) is the simulation consumer — it does not modify the mechanistic evidence.

## InCHIANTI Replication: 4-Axis HDR Validation

Independent replication of the ELSA coupling-tightening finding using the InCHIANTI cohort (Chianti region, Tuscany, Italy). InCHIANTI provides three advantages over ELSA: (1) younger age range (20–102 vs 50–90+), (2) IL-6 + HOMA-IR biomarkers (vs CRP + HbA1c), and (3) ATC-coded medication records (vs diagnosis-only proxies).

### Data access

InCHIANTI data is access-restricted. Request access via the InCHIANTI study group. Place the data files at `~/Downloads/inCHIANTI/InCHIANTI_CD_Share/` (or pass `--data-root` to scripts). The `.gitignore` excludes all InCHIANTI data files from the repository.

### 4-Axis panel

| Axis | Biomarker | Variable | InCHIANTI advantage over ELSA |
|------|-----------|----------|-------------------------------|
| I (Inflammatory) | IL-6 (pg/mL) | `X_IL6` | Direct regulatory cytokine (vs CRP acute-phase proxy) |
| M (Metabolic) | HOMA-IR (computed) | `X_GLU × X_INSULN / 405` | Direct insulin resistance (vs HbA1c glycaemic control) |
| N (Neuroautonomic) | Resting HR (bpm) | `X_FC` | Not pharmacologically confounded like BP; RMSSD not in standard release |
| F (Functional) | SPPB (0–12) | `PXSPS` | Validated composite (vs grip strength alone) |

### Running the analysis

```bash
pip install pyreadstat pyarrow openpyxl

# Extract harmonised panel from raw SAS files (saves to data/inchianti_panel.parquet)
python scripts/inchianti_extract_panel.py

# Individual analyses
python scripts/inchianti_qc.py                         # Cohort description
python scripts/inchianti_lambda_max_trajectory.py       # Lambda_max by age stratum
python scripts/inchianti_lead_lag.py                    # 6-pair cross-lagged regression
python scripts/inchianti_medication_dose_response.py    # Medication compression test (raw)
python scripts/inchianti_medication_refined.py          # Refined: age-stratified, SWDS-Gamma, HTN matched
python scripts/inchianti_pi_trajectory.py               # Pi = C_norm / V_norm trajectory

# Generate publication figures
python scripts/inchianti_figures.py

# Consolidate results for manuscript
python scripts/inchianti_manuscript_summary.py
```

### Key results

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| **Lambda_max trajectory** | 1.17 (20–49) → 22.1 (80+) [4-axis]; 3.28 → 28.7 [5-axis], monotonically increasing | ✅ Replicates ELSA coupling-tightening |
| **Lead-lag concordance** | 4-axis HR: 9/12 (75%, p=0.073, Conv B); 5-axis: 12/20 (60%); see concordance audit | Convention B (biological direction) recommended |
| **Medication dose-response** | Within-decade CIs overlap; SWDS-Gamma n_meds p = 0.35; HTN matched: treated > untreated | Confounding by indication, not genuine compression |
| **Pi trajectory** | Slope = −0.019/yr (D-dominated) | ⚠️ Divergent from ELSA (+0.001/yr) |
| **Survival** | deltaC(M5-M4) = +0.014 (age 65+), +0.014 (med-naive) | ✅ Comparable to ELSA (+0.009/+0.013) |

**Lead-lag concordance audit:** A 3-convention comparison (`scripts/verify_lead_lag_concordance.py`) resolves the concordance discrepancy between the original (9/12) and expanded (6/12) analyses. Convention B (biological direction: all beta > 0 predicted) gives 9/12 (75%, p=0.073) for the 4-axis HR model. Convention A (naive J sign) gives 8/12 (67%). The discrepancy arose because the expanded script incorrectly applied a sign-flip adjustment for protective F-axis entries. Convention B is recommended because cross-lagged regressions in declining systems measure co-decline direction, not individual coupling signs. See `results/lead_lag_concordance_audit.md`.

### Axis configurations tested

| Config | Axes | N pairs | Concordance (Conv A / Conv B) | Pi slope |
|--------|------|---------|-------------------------------|----------|
| 5-axis | I (IL-6), M (HOMA-IR), N (cortisol/DHEAS), F (SPPB), B (PTH) | 1,629 | 7/20 (35%) / 12/20 (60%) | -0.010/yr |
| 4-axis cortisol/DHEAS | I, M, N (cortisol/DHEAS ratio), F | 1,660 | 6/12 (50%) / 7/12 (58%) | -0.017/yr |
| 4-axis resting HR | I, M, N (resting HR), F | 1,523 | 8/12 (67%) / 9/12 (75%) | -0.019/yr |
| 4-axis NLR | I (NLR), M, N (cortisol/DHEAS), F | 1,688 | 6/12 (50%) / 7/12 (58%) | -0.016/yr |

### Cohort summary

- **N = 1,453** at baseline (ages 20–102), 6 waves over 18 years
- **176 young adults (20–49)** — key advantage, ELSA lacks this age range
- **950 deaths** (65%) over median 14.7 years follow-up
- **1,629 five-axis complete change pairs** from 957+ subjects
- Cortisol/DHEAS ratio available waves 0–2; PTH available waves 0–3
- Cystatin C and CTX-1 available at baseline only (no longitudinal P or B_resorption)
- RMSSD/HRV not in standard data release

### Limitations and divergences

1. **Lead-lag concordance convention**: Three conventions were audited (`results/lead_lag_concordance_audit.md`). Convention B (biological direction: all pairs predict positive beta in delta-space) gives 9/12 (75%, p=0.073) for 4-axis HR, matching the original report. Convention A (naive J sign) gives 8/12 (67%). An intermediate "correction" that reported 6/12 was itself erroneous (double-counted the F-axis sign flip). The recommended Convention B concordance is the one to cite.

2. **RMSSD unavailable**: The N-axis uses cortisol/DHEAS ratio (primary) or resting HR (secondary). Neither captures beat-to-beat parasympathetic modulation. Cortisol/DHEAS shows lower concordance (7/12 Conv B) than resting HR (9/12), suggesting resting HR better captures the coupling structure despite being a cruder measure. Cystatin C and CTX-1 are baseline-only.

3. **Pi divergence**: All configurations show D-dominated Pi trajectories (slopes -0.010 to -0.019/yr), opposing the ELSA result (+0.001/yr). Consistent across N-axis choices, suggesting a systematic cohort or biomarker-panel difference.

4. **Temporal coverage**: HOMA-IR and cortisol/DHEAS available waves 0-2 only (~6 years). PTH extends to wave 3. Full 5-axis coverage is limited to waves 0-2.

5. **SPPB ceiling effect**: All healthy 20-30 year-olds scored 12/12 (SD = 0). Reference SD computed from healthy adults < 60.

### Scripts and outputs

| Script | Output | Description |
|--------|--------|-------------|
| `inchianti_extract_panel.py` | `data/inchianti_panel.parquet` | Harmonised 6-wave panel (`.gitignored`) |
| `inchianti_qc.py` | `results/inchianti_qc_report.md` | Cohort description and data availability |
| `inchianti_lambda_max_trajectory.py` | `results/inchianti_lambda_max_by_age.csv` | Lambda_max by age stratum with bootstrap CIs |
| `inchianti_lead_lag.py` | `results/inchianti_lead_lag_matrix.csv` | 12 cross-lagged regression coefficients |
| `inchianti_medication_dose_response.py` | `results/inchianti_med_dose_response.csv` | Medication stratification + regression (raw) |
| `inchianti_medication_refined.py` | `results/inchianti_med_refined_results.json` | Age-stratified, SWDS-Gamma, HTN matched, off-diag |
| `inchianti_figures.py` | `outputs/figure_inchianti_*.pdf` | 5 publication figures (A-E) |
| `inchianti_pi_trajectory.py` | `results/inchianti_pi_trajectory.csv` | Pi = C_norm / V_norm by age stratum |
| `inchianti_6axis_analysis.py` | `results/inchianti_6axis_results.json` | 5-axis, 4 config comparison (lambda, lead-lag, Pi) |
| `inchianti_survival.py` | `results/inchianti_survival_analysis.json` | Cox models: SWDS-Gamma vs mortality |
| `inchianti_figures_6axis.py` | `outputs/figure_inchianti_*_6axis.pdf` | 6-axis figures (comparison, 5x5 heatmap, N-axis, survival) |
| `inchianti_manuscript_summary.py` | `results/inchianti_summary_for_manuscript.md` | Consolidated manuscript-ready results |

### Dependencies

Core (in `pyproject.toml`): numpy, scipy, matplotlib.
Part 2 additions (in `requirements_part2.txt`): pytest, lifelines (for Cox models).
InCHIANTI analysis: pyreadstat, pyarrow, openpyxl (for SAS file reading and parquet output).
Optional: SALib (for Sobol indices).
