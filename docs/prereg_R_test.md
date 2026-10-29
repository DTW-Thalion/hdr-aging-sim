# Pre-registration — predicted vs empirical correlation structure

**Written before any result was computed.** Git history is the evidence: this
file is committed in its own commit, ahead of
`scripts/run_R_validation.py` producing any output.

Purpose: the first quantitative out-of-sample test of the compiled coupling
matrix. Online Methods states that all empirical results in R7 operate
exclusively within Tier-1 — functionals of the observed covariance — and the
Discussion makes a virtue of three of four results being independent of the
compiled prior. The consequence is that the compiled `G` itself has almost no
empirical exposure. The cross-lagged analysis is the sole exception and, until
`results/lead_lag_phi_table.json`, it tested signs rather than magnitudes.

---

## 1. What is predicted

Under Convention G the fast subsystem is

```
ẋ = A x + w ,   A = −D(I − G) ,   D = diag(1/τ) ,   Cov(w) = Q
```

with stationary covariance `Γ` solving the Lyapunov equation
`A Γ + Γ Aᵀ = −Q`, and predicted correlation matrix
`R_pred = diag(Γ)^(−1/2) Γ diag(Γ)^(−1/2)`.

**The test is on R, not Γ.** `Q` is not identified. A global noise scale
cancels in the correlation matrix but not in the covariance, so `R` is the
object that survives not knowing `Q`.

**Why this is a clean test of `G`.** With `G = 0` and `Q = σ²I`, the solution is
`Γ = (σ²/2)·diag(τ)`, so `R_pred = I` exactly. **Every off-diagonal element of
`R_pred` is produced by `G` and by nothing else.** There is no part of the
predicted off-diagonal structure that comes from τ, from the noise scale, or
from the age trajectory.

## 2. What is observed

`R_emp` = correlation matrix of **within-person changes** `Δx = x(t+Δt) − x(t)`
across consecutive waves, computed per age stratum.

Justification for using changes rather than the cross-sectional covariance:
for an OU process the increment covariance is
`Γ_Δ = 2Γ − ΦΓ − ΓΦᵀ` with `Φ = exp(AΔt)`. At the cohort intervals (3–4 y)
against fast-axis τ of 0.003–6 d, `Φ ≈ 0`, so `Γ_Δ → 2Γ` and
`R(Γ_Δ) = R(Γ)`. The factor of 2 cancels in the correlation. The
cross-sectional covariance is additionally contaminated by stable
between-person differences, which the project's own cohort-design rule
excludes from primary validation.

## 3. Hypotheses and failure criteria

Stated before computation. Each names the statistic and the value that
constitutes failure.

**H1 — sign concordance.** The sign of each predicted off-diagonal matches the
empirical sign.
*Fail if* concordance ≤ 50% (chance), per cohort.
*Note in advance:* with `G ≥ 0` every predicted off-diagonal is positive, so H1
is weak and is reported only for completeness. It is not evidence for `G`; see
H4.

**H2 — pattern.** Pearson correlation `r` between the vector of predicted and
the vector of empirical off-diagonal elements, pooled across age strata.
*Fail if* the 95% bootstrap CI for `r` includes 0.

**H3 — level.** Slope `b` of the regression `R_emp_offdiag ~ b · R_pred_offdiag`
through the origin. `b` is the empirical scale of `G`.
*Fail if* the 95% CI for `b` excludes the interval [0.188, 1.342] — the scale
range already implied by the cross-lagged gains
(`results/units_convention_numerics.json`, `scale_diagnostic`). That would mean
the two independent routes to the level disagree.
*This is not a test of whether `b` = 1.* The compiled level is a prior, not a
measurement, and `b` is the second independent estimate of it.

**H4 — specificity, the decisive test.** `G` is all-positive, so any
non-negative matrix predicts positive correlations everywhere and H1–H3 could
be passed by a matrix carrying no real structure. The test is therefore against
a permutation null: shuffle the non-zero off-diagonal entries of `G` among the
non-zero off-diagonal positions, 10 000 times, recompute `R_pred` and the H2
correlation each time.
*Fail if* the real `G` does not beat the permutation null at p < 0.05 — i.e. if
the specific pattern of compiled entries predicts the observed correlation
structure no better than a random rearrangement of the same numbers.
**H4 is the result. H1–H3 without H4 are uninformative.**

**H5 — age invariance of the level.** `b` estimated separately per age stratum.
*Fail if* `b` varies by more than 2× across strata, which would mean the
coupling level is not a fixed property of the basin and the "aging is D-only
within a fixed G" claim is wrong in the data.

**H6 — cross-cohort invariance of the level.** `b` estimated separately in
InCHIANTI and ELSA.
*Fail if* the 95% CIs do not overlap. `G` is claimed to be a property of human
physiology, not of a cohort.

## 4. Sensitivity, pre-specified

- `Q = σ²I` is primary. `Q = σ²D` (noise proportional to restoring rate) and
  `Q = σ²D²` are reported as sensitivity. A result that holds only for one `Q`
  is reported as `Q`-dependent and not claimed.
- τ from `data/tau_registry.json`.
- `G` is the adopted matrix: protective entries removed to κ, sign-reversing
  regime-dependent entries zeroed in the healthy basin. Basin assignment per
  age stratum is NOT used — the healthy-basin `G` is used throughout, because
  under Convention G a basin change is not an age effect.
- Age strata: quartiles of the age distribution within each cohort, so that
  stratum boundaries are not chosen after seeing the result.

## 5. Things this test cannot do

- It cannot identify `Q`, so it cannot test the diagonal of `Γ`, only `R`.
- It cannot separate `G` from any transformation that preserves `R`.
- It cannot test the protective couplings, which no longer appear in `G`.
- It cannot test E or B in ELSA, and B only in the InCHIANTI 5-axis model.
- A pass does not validate the compiled magnitudes. H3 estimates the level; it
  does not confirm it.

## 6. What a failure would mean

- **H4 fails** → the compiled coupling pattern carries no information about the
  observed correlation structure. That is the strongest negative result
  available from existing data and it would mean the `J` matrix cannot be used
  as a prior in its current form. It would not touch the three
  architecture-validating results, which do not use `G`.
- **H2 or H3 fail with H4 passing** → the pattern is real but the level or the
  linearity is wrong; recalibrate rather than discard.
- **H5 or H6 fail** → `G` is not basin-fixed or not cohort-invariant, and the
  framework's claim that coupling is a stable property of physiology needs
  qualifying.

No outcome here changes the units decision, which rests on dimensional
coherence and the φ bound, not on this test.
