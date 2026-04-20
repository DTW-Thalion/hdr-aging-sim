# Interpretation Guide

How to read every headline number the pipeline produces, and how to
translate a result into a manuscript claim (or a decision that the
claim is not supported).

Companion to [METHODS.md](METHODS.md), which defines each quantity.

---

## The one-page version

Open `results/ukb_hdr_summary.md`. The core question HDR asks is whether
physiological homeostasis erodes along a shared manifold with age.
Four quantities answer four distinct sub-questions:

| Quantity            | Question                                                           | "Yes" looks like                      |
|---------------------|--------------------------------------------------------------------|---------------------------------------|
| λ_max ratio (long.) | Does the dominant stability mode grow with age?                   | > 1.3×, monotonic, perm-p < 0.05      |
| Sign concordance    | Do cross-lagged effects match the coupling predictions?           | > 70%, binomial-p < 0.05              |
| ΔC(M5 − M4)         | Does SWDS-Γ predict mortality beyond age + frailty?               | > 0.01, lower 95% CI > 0              |
| ΔC(M5 − M4a)        | Does SWDS-Γ add *beyond raw biomarkers*?                          | > 0 with lower CI > 0                 |
| Π slope             | Is coupling doing more of the variance-shaping as we age?         | Monotonically increasing              |

A "hit" on all five is the strongest possible validation. A "hit" on
three of five, with well-powered null tests, is still a strong positive
result — the HDR manuscript accepts partial replication and reports
which specific predictions did not replicate.

---

## 1. λ_max trajectory (steps 3 + 4)

### 1.1 What to look at

- `ukb_longitudinal_{tier}.json → per_stratum[*].lambda_max`
- `ukb_longitudinal_{tier}.json → lambda_max_ratio`
- `ukb_longitudinal_{tier}.json → trend_kendall`
- `ukb_longitudinal_{tier}.json → permutation_trend.p_value`

### 1.2 Reading the numbers

A **monotonically increasing** λ_max from stratum 40–49 through 80+ is
the primary HDR signature. In published cohorts:

- InCHIANTI (N=1,453, 4-axis): **18.8× ratio**, perm p < 0.001
- ELSA (N=6,420, 3-axis): **1.24× ratio**, perm p = 0.032
- UKB (this pipeline): unknown until run, expected 1.5–3× given the
  larger axis set and weaker survivorship selection

**Why the InCHIANTI and ELSA ratios differ by an order of magnitude.**
InCHIANTI's 4-axis set includes the F-axis (grip strength), which
dominates the eigenvalue. ELSA drops F and N and is 3-axis; its
λ_max ratio is more modest. UKB should sit between the two — with a
4-axis tier ≈ InCHIANTI minus survivorship, plus the 5-6 axis tiers
showing whether bone and circadian contribute.

**If the ratio is < 1.1 and Kendall τ is non-significant**, stop and
check:

1. Is the youthful reference actually in the youngest age range
   present in your data? If UKB baseline starts at 40 and your
   config's `youthful_reference_age` is `[40, 49]`, you are fine.
   If you inadvertently set `[20, 30]` there is no reference pool.
2. Are any axes entirely missing at later ages? Check
   `ukb_qc_stats.json → distributions_by_instance`.
3. Is `delta_I` log-transformed? Non-log CRP is extremely heavy-tailed
   and the eigenvalue will be dominated by a handful of outliers.

### 1.3 Permutation null

`permutation_trend.p_value < 0.05` confirms that the observed λ_max
trajectory is specific to real age labels — it would not arise by
chance under age-label shuffling. This is the standard we report in
the manuscript.

---

## 2. Univariate control ratio

### 2.1 What to look at

`per_stratum[*].univar_ratio` (ratio of λ_max to the largest axis
variance in that stratum).

### 2.2 Reading the number

- **Ratio ≈ 1.0** → the dominant eigenvalue is essentially a single
  axis's variance, with no meaningful contribution from off-diagonal
  coupling. Reports as **"single-axis dominated"** in the manuscript.
  This is what InCHIANTI's 4-axis set shows (F-axis dominates).
- **Ratio > 1.2** → multi-axis coupling contributes substantively.
- **Ratio > 1.5** → strong multi-axis signature.

UKB's large axis set (5–6) should show ratios > 1 even in tier 2; if
it does not, this is evidence that the UKB sample retains the same
single-axis dominance pattern seen in InCHIANTI, likely driven by
grip strength.

See the project memory on λ_max null tests for the reasoning.

---

## 3. Sign concordance with J matrix

### 3.1 What to look at

- `ukb_lead_lag_{tier}.json → main.n_concordant / n_tested_vs_J`
- `ukb_lead_lag_{tier}.json → main.binomial_p`

### 3.2 Reading the number

Published reference:

- InCHIANTI: 6/12 concordant (50%) — the I→M and F→I pairs replicate
  strongly but F→M is anomalously negative in both InCHIANTI and ELSA
  (survival-selection hypothesis).
- ELSA: 5/6 (83%).

UKB has more axes and therefore more pairs. Expect:

- **Tier 2 (4 axes, 12 pairs)** → comparable to InCHIANTI
- **Tier 3 (5 axes, 20 pairs)** → adds I→C, C→N, F→C predictions
- **Tier 4 (6 axes, 30 pairs)** → adds F→B (grade-A), B→I, etc.

**Pair-level FDR** (`q_fdr < 0.05`) is a stronger claim than sign
concordance. ELSA has 4/6 pairs surviving FDR; InCHIANTI has 0/12.
The question is whether UKB, with N ≈ 100K longitudinal pairs, pushes
more pairs to FDR significance while retaining sign concordance.

### 3.3 When the binomial test fails

If `binomial_p > 0.05` the sign pattern is indistinguishable from
coin-flip. Two non-failure interpretations are possible:

1. The directional predictions in J are wrong (science update).
2. UKB's interval (4–14 years) is too long for cross-lagged regression
   to recover short-timescale coupling. In that case, use the
   first-repeat subset (instances 0→1, 6-year gap) by setting
   `instances: [0, 1]` in the config.

---

## 4. ΔC (M5 − M4) — the mortality test

### 4.1 What to look at

- `ukb_mortality_{tier}.json → delta_bootstraps.delta_C_M5_M4`

Structure: `{mean, ci_lower, ci_upper, n_boot}`.

### 4.2 Reading the numbers

Pre-specified thresholds from the manuscript:

- **"Clinically meaningful"**: ΔC ≥ 0.01 with lower CI bound > 0.
- **"Significant vs 0"**: lower CI bound > 0.

Published:

- InCHIANTI N=923, 698 deaths: ΔC = +0.014, 95% CI [+0.008, +0.022]
- ELSA full N=5,431, 1,122 deaths: ΔC = +0.009, 95% CI [+0.003, +0.015]
- ELSA med-naive N=2,489, 375 deaths: ΔC = +0.013, 95% CI [+0.003, +0.026]

None clears the 0.01 threshold with a clear margin in ELSA; InCHIANTI
does. UKB should give the definitive answer with N ≈ 500K.

### 4.3 ΔC (M5 − M4a) — the stronger claim

`delta_C_M5_M4a` asks: **does SWDS-Γ add anything beyond raw biomarkers
alone?** In InCHIANTI, M5 = M4a after rounding (both C = 0.760) — i.e.
the coupling-weighted score carries **no unique information** beyond
simply including the biomarkers separately.

If UKB produces ΔC(M5 − M4a) > 0 with a clear lower CI, that is the
strongest possible positive result: the coupling structure itself
(encoded in Γ̂) is informative, not just the marginal axis levels.

### 4.4 Subgroups

- **Medication-naive**: if ΔC is larger here than in the full sample,
  medication is compressing the variance-covariance structure (the
  "medication compression" hypothesis from ELSA).
- **By age**: HDR predicts stronger SWDS performance in older strata
  where coupling is stronger. If ΔC is uniform across ages, either
  the prediction is wrong or the age strata are too coarse to reveal
  it.
- **By sex / ethnicity**: primarily a generalisability check.

---

## 5. Π decomposition

### 5.1 What to look at

`per_stratum[*].Pi → {V_norm, C_norm, Pi}` in steps 3 and 4.

### 5.2 Reading the numbers

Π = C_norm / V_norm.

- Π ≪ 1 → diffusion drives variance (stable homeostasis)
- Π ≈ 1 → coupling equals diffusion (intermediate)
- Π > 1 → coupling drives variance (eroded homeostasis)

**Monotonic increase of Π with age** is the signature the manuscript
calls *coupling tightening* (see `scripts/run_figure_coupling_tightening.py`).

The published InCHIANTI result shows Π rising from ~0.3 (40–49) to
~0.7 (80+). UKB's wider axis set should smooth this curve; a plateau
or reversal at the oldest age would be evidence for the
survival-selection caveat.

---

## 6. Key cross-lagged pairs

### 6.1 The headline pairs

The strongest directional predictions that UKB should replicate:

| Pair    | J prediction | InCHIANTI p | ELSA p    | What it would tell us          |
|---------|--------------|-------------|-----------|--------------------------------|
| I → M   | +            | 0.031       | 3.2e-20   | Inflammation drives metabolic decline |
| F → I   | +            | FDR-sig     | FDR-sig   | Muscle protects against inflammation |
| M → I   | +            | sig         | sig       | Metabolic dysfunction drives inflammation |

### 6.2 Novel pairs unlocked by tier 3–4

| Pair    | J prediction | Grade | What it tests                    |
|---------|--------------|-------|----------------------------------|
| I → C   | +            | A     | Inflammation disrupts circadian  |
| F → B   | −            | A     | Muscle loading protects bone     |
| C → N   | +            | A     | Circadian → autonomic coupling   |
| P → I   | +            | A     | Proteostatic failure → inflammation |

These are the first directly testable in UKB because InCHIANTI/ELSA
lacked the C or B axes.

### 6.3 The F → M anomaly

In both InCHIANTI and ELSA, F → M comes out **negative** — opposite
the J prediction. This has been flagged in the manuscript as a
survival-selection artefact: participants with preserved muscle at
baseline who also have declining metabolism are selectively lost to
follow-up, producing a spurious negative cross-lagged β.

If UKB's F → M is also negative, survival selection becomes the
leading explanation. If it is *positive*, InCHIANTI/ELSA's pattern is
cohort-specific.

---

## 7. Null-test interpretation

### 7.1 Age-permutation null

`observed_ratio` well above `null_p95` with `p_value < 0.05` →
trajectory is real, not a chance age-labelling artefact.

### 7.2 Random-panel null

`p_value < 0.05` → HDR's axis selection produces a steeper λ_max
trajectory than a random set of K UKB biomarkers. This is the
strongest falsifier of "it's just UKB effects".

Caveat: some random panels happen to include grip strength or CRP
themselves (they are in the candidate pool only if not already an HDR
axis). The null's informational content depends on having enough
*non-HDR* candidates — confirm `n_panels > 100` in the output.

### 7.3 Univariate control by stratum

Should stay below the λ_max ratio itself (if you compute the two side
by side). A 2× λ_max ratio with univariate ratio of 1.9 means 95% of
the effect is single-axis variance widening; only 5% is genuine
coupling widening. That is a publishable caveat, not a negative
result.

### 7.4 Axis-substitution null

Substituting BMI for HbA1c, or albumin for CRP, should degrade the
λ_max ratio if HDR's axis selection is doing real work. If the
substitute produces an equal or better ratio, the choice of axis
biomarker is essentially arbitrary — that is a real and manuscript-
worthy finding.

---

## 8. When results disagree

### 8.1 ΔC positive but sign concordance chance-level

The biomarkers carry prognostic information but not via the predicted
coupling structure. Consistent with "this is just multi-biomarker
aggregation, not HDR-specific coupling". Report honestly.

### 8.2 Sign concordance high but ΔC ≈ 0

Coupling predictions replicate, but the coupling-weighted score adds
no prognostic information over raw biomarkers. Most likely
explanation: UKB baseline biomarkers already capture the health
signal, and the coupling structure is correlated with them so tightly
that no incremental value remains. The manuscript discusses this
possibility — not a refutation of HDR, but a limit on SWDS-Γ's
practical use in well-measured cohorts.

### 8.3 λ_max trend absent but Π rising

Worth investigating. Possible that the population-level eigenvalue is
stabilised by survival selection while the coupling structure itself
continues to rigidify. This is exactly the kind of substructure UKB's
N is large enough to resolve.

---

## 9. Writing up

For manuscript text, quote these numbers from `ukb_hdr_summary.md`:

- N (baseline, imaging, deaths, follow-up)
- λ_max ratio + Kendall τ + permutation p
- Sign concordance (n/n) + binomial p
- Key pair-level β + FDR q
- ΔC(M5 − M4) + 95% CI
- ΔC(M5 − M4a) + 95% CI
- Null-test p-values

For supplementary tables, export the per-stratum λ_max and per-pair
lead-lag CSVs — both are preserved in the JSON outputs.

For figures, the per-step PDFs are publication-ready at 300 dpi with
sensible axis labels. Export as individual panels by opening them in
Illustrator / Inkscape if you need to re-arrange a figure layout.

---

## 10. Honest reporting

The HDR manuscript commits in advance to reporting **every tier run and
every specified analysis**, whether the result replicates or not.
Replication failures in UKB should be reported alongside successes —
the framework's value depends on it falsifying honestly, not on a
hand-picked confirmation.

Specific pre-specified analyses:

- All tiers that feasibility allows
- All pair-level lead-lag tests in the tier (not a subset)
- All five published ΔC contrasts
- All four null tests (perm, random-panel, univariate, substitution)

If the pipeline skips an analysis due to sample-size feasibility, that
counts as "ran but underpowered" rather than "did not attempt".
