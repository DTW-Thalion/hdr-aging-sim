# Methods

Formal statement of every statistical procedure in the pipeline. All
notation follows the HDR manuscript (Whitney et al., *Homeodynamic
Remediation: a 9-axis framework for aging*). Functions referenced here
live in [hdr_core.py](../hdr_core.py) unless otherwise noted.

---

## 1. Notation

- **K** — number of axes in the tier (3–6)
- **x_{p,i}** — raw biomarker value for participant *p* at instance *i*
- **Δx_{p,i}** — standardised axis vector: (x_{p,i} − μ_ref) / σ_ref,
  where reference statistics are computed on the youngest stratum
  (40–49 by default)
- **Γ̂** — K × K sample covariance of Δx at a given stratum or change
  window
- **J** — compiled K × K directional coupling matrix, signs only,
  loaded from `J_matrix_compiled_9x9.csv`

Sign conventions:

- All axes are sign-aligned so that **positive Δx = worse homeostasis**.
  Grip strength and BMD are therefore negated at extraction time
  ([step1_extract.py](../step1_extract.py)).
- Lead-lag uses **Convention B** throughout: sign of β is compared to
  the compiled J entry directly, so a correct concordance means β > 0.
  Convention A (negative signs for inverse-coupled pairs) is never
  reported — see the project memory for rationale.

---

## 2. Standardisation

For axis *k*, the youthful reference is:

```
μ_k   = mean {x_{p,0,k} : age_p,0 ∈ [40, 49]}
σ_k   = sd   {x_{p,0,k} : age_p,0 ∈ [40, 49]}
Δx_{p,i,k} = (x_{p,i,k} − μ_k) / σ_k
```

Reference is computed once on the baseline instance (instance 0) and
applied unchanged to every subsequent instance. Same-reference
standardisation is what makes the cross-covariance at instance 2
directly comparable to baseline.

---

## 3. Stability eigenvalue λ_max

### 3.1 Cross-sectional (step 3)

For each age stratum *s*:

```
Γ̂_s = (1 / (N_s − 1)) · Σ_p (Δx_{p,s} − Δx̄_s)(Δx_{p,s} − Δx̄_s)ᵀ
λ_max(s) = max eigenvalue of Γ̂_s
```

Bootstrap 95% CI: resample participants with replacement within stratum
*s*, recompute λ_max on each resample, take percentile (2.5, 97.5).
Default `n_bootstrap = 10,000`.

### 3.2 Change-covariance (step 4)

For each pair of consecutive instances (t₀, t₁) with both Δx complete:

```
δ_{p} = Δx_{p, t₁} − Δx_{p, t₀}
Γ̂_change(s) = cov(δ_p) for participants with (age_{t₀} + age_{t₁})/2 ∈ s
λ_max_change(s) = max eigenvalue of Γ̂_change(s)
```

Same bootstrap procedure, with resampling at the participant-pair level.

### 3.3 Trend tests

**Kendall τ:**

```
τ = kendalltau({(age_mid_s, λ_max(s))})
```

Reports τ and its asymptotic p-value (scipy).

**Age-permutation null:**

```
for b in 1..B:
    shuffle age labels (within the analysis sample)
    recompute λ_max(s) for each stratum under shuffled ages
    compute ratio(b) = max(λ_max) / min(λ_max)
p = (#{ratio(b) ≥ observed_ratio} + 1) / (B + 1)
```

Default B = 1,000 ([analysis.n_permutations](../config.yaml)).

### 3.4 Univariate control

```
univariate_ratio(s) = λ_max(s) / max_k Var(Δx_k)
```

- Ratio ≈ 1.0 → dominant eigenvalue is mostly a single-axis variance
  artefact (no genuine multi-axis coupling signature).
- Ratio > ~1.2 → multi-axis coupling contributes measurably to the
  eigenvalue.

Reported in both step 3 (cross-sectional) and step 4 (change).

### 3.5 Random-panel null

Let *C* be the pool of UKB continuous biomarkers not in any HDR axis
(see [FIELD_MAPPING.md](FIELD_MAPPING.md)). For each random draw of
K columns from *C*:

```
for panel in random_panels(C, size=K):
    compute λ_max_ratio across age strata for this panel
```

Compare the observed HDR λ_max ratio to the resulting distribution.
Tail probability:

```
p = #{random_ratio ≥ observed_ratio} / #panels
```

If the total number of K-combinations is smaller than the configured
`n_random_panels`, the code enumerates all of them; otherwise it
samples without replacement.

---

## 4. Π decomposition

```
V_norm(s) = mean_k Γ̂_{s, kk}            (average axis variance)
C_norm(s) = mean_{i≠j} |Γ̂_{s, ij}|       (average |off-diagonal|)
Π(s)      = C_norm(s) / V_norm(s)
```

Interpretation (manuscript Figure 5):

- Π ≪ 1 → D-dominated regime: diffusion drives variance
- Π ≈ 1 → J-dominated regime: coupling drives variance
- Monotonic increase of Π with age → coupling plays a larger role as
  stability erodes

Π is reported for every stratum in steps 3 and 4.

---

## 5. Lead-lag cross-lagged regression (step 5)

For each ordered pair (i → j), i ≠ j:

```
d Δx_{j} ≡ Δx_{j,t₁} − Δx_{j,t₀}

d Δx_{j} = β · Δx_{i,t₀}
        + γ · Δx_{j,t₀}
        + δ · age_{t₀}
        + c + ε
```

Fit by OLS. The cross-lagged coefficient of interest is β.

### 5.1 Inference

- **Nominal p-value**: standard OLS, two-sided.
- **Subject-clustered bootstrap 95% CI**: resample participants with
  replacement (clusters, not observations). Default `n_boot = 5,000`
  in step 5 (half the general bootstrap count for speed).
- **BH-FDR** within tier across all K(K−1) ordered pairs.

### 5.2 Sign concordance

A pair (i → j) is **concordant** iff sign(β̂) = J_{i,j} and J_{i,j} is
in {−1, +1} (i.e. not NULL / unknown).

Aggregate test:

```
n_concordant / n_tested_vs_J
p_binom = P(X ≥ n_concordant | X ~ Binomial(n_tested, 0.5))
```

(one-sided, greater than 0.5).

### 5.3 Subgroup analyses

Step 5 re-runs the whole pair matrix on:

- `medication_naive` (n_med_classes = 0)
- `male` (sex = 1)
- `female` (sex = 0)

Any subgroup with fewer than 500 triplets is skipped.

---

## 6. SWDS-Γ

The Γ-native stability-weighted dysregulation score:

```
SWDS-Γ(Δx) = Δxᵀ Γ̂ Δx / tr(Γ̂)
```

Computed at baseline using the baseline Γ̂ and each participant's
baseline Δx. In step 6 the score enters Cox models as `log_swds`.

### Comparators

- **Mahalanobis** distance: `Δxᵀ Γ̂⁻¹ Δx`. Enters Cox as `log_maha`.
- **Z-sum**: `Σ_k Δx_k`. Enters Cox as `z_sum`.

The three scores test different hypotheses:

- **SWDS-Γ** — high variance directions = near-unstable modes. Weights
  deviation along slow-recovering modes more.
- **Mahalanobis** — *inverse* weighting. High-variance directions count
  *less*. Classical "outlier" metric.
- **Z-sum** — no coupling information at all; equivalent to the
  Frailty Index concept of additive deficits.

---

## 7. Cox mortality models (step 6)

All models use lifelines' `CoxPHFitter(penalizer=0.01)` with
time_years, event.

### 7.1 Model menu

| Model | Covariates                                                        |
|-------|--------------------------------------------------------------------|
| M1    | age + sex                                                          |
| M2    | M1 + smoking + diabetes + hypertension + raw biomarkers            |
| M3    | log_swds only                                                      |
| M3a   | log_maha only                                                      |
| M3b   | z_sum only                                                         |
| M4    | M1 + smoking + diabetes + hypertension + frailty_index             |
| M4a   | M4 + raw biomarkers                                                |
| M4b   | M4 + log_swds                                                      |
| M5    | M4 + raw biomarkers + log_swds                                     |

Raw biomarkers = `log(CRP)`, HbA1c, grip, pulse, `log(cystatin)`,
BMD, BMI — whichever are available and non-missing.

### 7.2 Matched complete-case sample

All models are fit on the **same N** — the intersection of non-missing
required covariates across every model. This lets C-index comparisons
and ΔC interpretations avoid selection bias between models.

### 7.3 ΔC bootstrap

Event-stratified subject-level bootstrap (`n_bootstrap_cox = 2000`):

```
for b in 1..B:
    resample events with replacement (size = n_events)
    resample non-events with replacement (size = n_non_events)
    fit Cox_base on resample
    fit Cox_full on resample
    ΔC(b) = C_full − C_base
CI = percentile(ΔC, [2.5, 97.5])
```

Event-stratification preserves the event rate across resamples, which
stabilises C-index estimates.

Key ΔC contrasts reported:

- `delta_C_M5_M4`  — "does SWDS add beyond frailty?" (headline number)
- `delta_C_M5_M4a` — "does SWDS add beyond frailty + biomarkers?"
- `delta_C_M4b_M4` — "does SWDS alone add to frailty?" (no biomarkers)

### 7.4 Subgroups

Re-fits M4, M4a, M5 on each of:

- `medication_naive` (n_med_classes = 0)
- `male`, `female`
- Each age stratum separately

Subgroup ΔC differences are descriptive only — no formal interaction
test is performed (UKB power lets you examine trends without pooling).

---

## 8. Circadian axis (step 8)

### 8.1 Tier 1 — summary fields

Given UKB fields (90012 overall acceleration, 90087 moderate-activity
fraction, 90015–18 acceleration by 6-hour window):

```
day_night_ratio   = Q_1200-1800 / Q_0000-0600  (heuristic window split)
components        = {−z(day_night_ratio), −z(overall), −z(moderate)}
circadian_score   = nanmean(components)
```

Sign flip ensures higher score = worse circadian fidelity.

### 8.2 Tier 2 — raw CWA

For an hourly activity series *h* over the wear period:

```
hourly_means  = {mean h | hour_of_day = i}  for i = 0..23
grand_mean    = mean(h)

IS            = Σ_i (hourly_means[i] − grand_mean)² / (24 · Var(h))   (interdaily stability)
IV            = Σ_t (h[t+1] − h[t])² / ((T−1) · Var(h))                (intradaily variability)

M10           = max over 10-hour rolling windows of h
L5            = min over 5-hour rolling windows of h
RA            = (M10 − L5) / (M10 + L5)                                (relative amplitude)
L5_onset      = hour at which L5 window starts
```

Axis value (Tier 2) = `IV` (higher fragmentation = worse). Sign
convention matches Tier 1 so the two can be combined if both are
available.

---

## 9. Feasibility logic

[run_all.py](../run_all.py) determines which tiers to run based on the
extracted panel's axis completeness. A tier is *feasible* iff every
listed axis has ≥ 1,000 non-missing values at baseline. Tiers that
fail feasibility are reported in `ukb_hdr_summary.md` as skipped.

---

## 10. References

- Whitney et al., *Homeodynamic Remediation: a 9-axis framework for
  aging*, 2025 — parent manuscript.
- Appendix F: Γ-native SWDS proposition (identical rankings to A-based
  SWDS under isotropic noise + symmetric Hurwitz A).
- Supplementary Note 11: Null-test methodology mirrored in step 7.
- Supplementary Note 10: Nonlinearity robustness check (not ported to
  UKB — baseline linearity already validated in InCHIANTI).
- Witting et al. (1990): IS / IV / RA circadian rhythm measures.
- Rockwood & Mitnitski (2007): Frailty Index construction.

For InCHIANTI / ELSA comparator numbers see the parent repository
`scripts/inchianti_*.py` and `scripts/elsa_*.py`.
