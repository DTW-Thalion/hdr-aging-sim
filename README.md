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
python scripts/run_figure_network_schematic.py  # Main Fig 1: 9-axis network diagram
python scripts/run_figure_J_heatmap.py          # Main Fig 2: 9x9 coupling heatmap
python scripts/run_figure2b.py                  # Spectral-abscissa drift demo
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

The machine-readable coupling matrix is provided in `data/J_matrix_compiled_9x9.csv` (9×9, 72 off-diagonal entries); the legacy 8×8 version is retained as `data/J_matrix_compiled.csv` for reproducibility of prior analyses.

`data/J_matrix_compiled_9x9.csv` is the **single source of truth** for the 9×9 coupling matrix J_mech. It contains all 72 off-diagonal entries of the 9 regulatory capacity axes:

| Axis | Name | τ (days) |
|------|------|----------|
| I | Inflammatory resolution capacity | 7–25 |
| M | Metabolic regulatory gain | 0.1–0.3 |
| E | Epigenetic maintenance fidelity | ~1000 |
| mito | Mitochondrial bioenergetic capacity | 1–3 |
| P | Proteostatic clearance capacity | 0.5–2 |
| C | Circadian oscillator amplitude | 1–3 |
| N | Neuroendocrine feedback gain | 0.5–2 |
| F | Functional reserve (musculoskeletal) | 8–42 |
| B | Bone remodelling regulatory balance | 90–120 |

CSV columns include:

- **Basin-stratified values**: `J_healthy`, `J_pre_disease`, `J_disease` (SD-per-SD units)
- **Age trajectory**: `increasing`, `decreasing`, `stable`, or `unknown`
- **Evidence metadata**: sign, magnitude tier (S/M/W/unknown), confidence grade (A/B/C), primary PMID, evidence type, mechanism brief, and biomarker role notes

Sign distribution (9×9): 57 positive (pathological), 11 negative (protective), 4 unknown.

**How it works**: At import time, `hdr_sim.csv_loader` loads the CSV, extracts the 4-axis subset (I, M, N, F) for the healthy and disease basins, and applies a calibration scalar (c ≈ 0.30) to map SD-per-SD literature values to simulation coupling rates (day⁻¹). The calibration scalar is computed via Brent's method to match the target spectral abscissa α(30) ≈ −0.134. This preserves the literature-derived relative coupling structure while maintaining dynamical stability.

The ELSA validation uses a 3-axis reduction (I, M, F) and is unaffected by the B-axis addition.

```python
from hdr_sim import load_J_csv, build_J_basin, get_J_anchors

# Load full 9-axis matrix for any basin (default: 9x9 CSV)
rows = load_J_csv()
J_9x9 = build_J_basin(rows, basin='disease',
                       axes=('I','M','E','mito','P','C','N','F','B'))

# Get calibrated 4-axis anchors used by the simulation
# (uses legacy 8x8 CSV to preserve calibration)
J_30, J_80, c = get_J_anchors()  # c ≈ 0.30
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
| `run_elsa_ici_deployment.py` | `outputs/elsa_ici_deployment.json` | ELSA ICI deployment assessment |
| `run_elsa_ici_deployment.py` | `outputs/elsa_ici_deployment_table.txt` | ICI deployment manuscript table |
| `run_elsa_basin_recovery.py` | `outputs/elsa_basin_recovery.json` | Data-driven basin recovery (GMM/HMM) |
| `run_elsa_basin_recovery.py` | `outputs/elsa_basin_recovery_table.txt` | Basin recovery comparison table |
| `run_figure_disease_demos.py` | `outputs/figure_disease_demos.pdf` | ED Fig 2: disease demo panels (T2D, frailty, AD, osteoporosis) |

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
| `verify_J_matrix_counts.py` | Audits 9×9 J matrix CSV sign counts (57+, 11−, 4?) and generates `outputs/j_matrix_audit_report.json` |
| `populate_pmids.py` | Populates missing `pmid_primary` entries in `J_matrix_compiled_9x9.csv` (67/72 now cited; 5 gaps in B-axis unknowns and one theoretical mito→N entry) |
| `update_ledger_r6.py` | Updates `outputs/elsa_results_ledger.json` with SHA-256 hashes for R6 figure files |

### Reproducibility

`outputs/elsa_results_ledger.json` contains SHA-256 hashes of all data files and every numerical result reported in Appendix H. Use this to verify that your ELSA data extract and pipeline output match ours.

## Part 2: Mechanistic-Evidence-Informed Model

Part 2 extends the 4-axis toy simulation with a 9-axis dynamical system parameterised from the enriched mechanistic evidence base produced by the HDR-mechanistic repository. All Part 2 code is **purely additive** — the original R1–R6 scripts and modules are unmodified.

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

# Run individual analyses
python scripts/run_sensitivity.py          # MC sensitivity (10K draws)
python scripts/run_synthetic_validation.py  # Synthetic cohort validation
python scripts/run_intervention_analysis.py # Intervention ranking + factorial
python scripts/run_dj_primacy_mechanistic.py # D/J primacy with mechanistic priors
```

Reports are written to `results/`.

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

| Script | Output | Description |
|--------|--------|-------------|
| `run_full_pipeline.py` | `results/full_pipeline_report.md` | End-to-end 10-step pipeline (model, sensitivity, cohort, Tier-1, interventions, ABC) |
| `run_sensitivity.py` | `results/sensitivity_report.md` | MC sensitivity analysis (10K draws), entry ranking, stress tests |
| `run_synthetic_validation.py` | `results/synthetic_validation_report.md` | ELSA-like synthetic cohort under 4 confound conditions |
| `run_intervention_analysis.py` | `results/intervention_report.md` | Intervention ranking, R6 factorial, pairwise interactions |
| `run_dj_primacy_mechanistic.py` | `results/dj_primacy_mechanistic.json` | D/J primacy with 5 degradation regimes x 4 confound conditions |

### Relationship to HDR-mechanistic

The `data/mechanistic_evidence/` directory contains exports from the [HDR-mechanistic](https://github.com/DTW-Thalion/HDR-mechanistic) repository (pipeline v3.5). That repository performs the mechanistic decomposition of each J entry into molecular pathway steps, validates against ChEMBL, and produces the enriched evidence base consumed here. This repository (hdr-aging-sim) is the simulation consumer — it does not modify the mechanistic evidence.

### Dependencies

Core (in `pyproject.toml`): numpy, scipy, matplotlib.
Part 2 additions (in `requirements_part2.txt`): pytest, lifelines (for Cox models).
Optional: SALib (for Sobol indices).
