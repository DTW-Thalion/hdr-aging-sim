# Protective couplings as damping modulation

**Task 1.2, formulation only.** The κ table itself is built in HDR-mechanistic.
This document fixes the equations, the entry list, the concordance definition
and the tier→κ mapping principle, so that it can be.

Depends on `docs/units_convention.md` (Convention G adopted) and
`data/tau_registry.json`. Derived figures from
`scripts/units_convention_numerics.py` → `results/units_convention_numerics.json`.

---

## 1. The problem

Eleven compiled entries are negative. Eight originate from the skeletal muscle
axis F and three from the bone axis B
(`data/J_matrix_compiled_9x9.csv`; enumerated in §3). In the additive form
`A = −D + J` a negative off-diagonal entry is a negative-feedback loop, so the
model currently asserts that **sarcopenia improves inflammation, bone and
mitochondrial function**. The same entries carry the published stability margin:
with the F row zeroed, `α_fast(25, healthy)` moves from −0.1889 to −0.0917 at the
published `c = 0.89`, and `α_fast(120, healthy)` from −0.0150 to **+0.1543** —
the model is unstable in late life once the protective row is removed
(`results/units_convention_numerics.json`, key `recalibration`).

Three things are wrong at once and they have one cause. The biology is inverted
relative to the manuscript's own osteosarcopenia prose. The margin depends on
the inversion. And the cross-lagged data needs a special reading ("Convention B")
to count the F pairs as concordant, because the measured β for `F→I` is positive
(+0.0251 ELSA, +0.0143 InCHIANTI; `results/lead_lag_phi_table.json`) while the
compiled entry is −0.35.

Protection is not negative coupling. Muscle mass and bone loading do not drive
inflammation down; they make the inflammatory axis **recover faster from a
displacement**. That is a statement about `D`, not about `J`.

---

## 2. Formulation

### 2.1 Damping modulation

Let `Δx_s` be the standardised departure of a protective source `s ∈ {F, B}`
from the youthful reference, in the compiled sign convention where a positive
departure is the pathological direction (sarcopenia, bone loss). For each target
`i` and protective source `s`:

```
  1          1       1 + κ_is · max(0, −Δx_s)
───────  =  ───  ·  ──────────────────────────                           (1)
τ_i^eff     τ_i      1 + κ_is · max(0,  Δx_s)
```

with `κ_is ≥ 0`. Several protective sources act multiplicatively:

```
  1          1                 1 + κ_is · max(0, −Δx_s)
───────  =  ───  ·    ∏      ──────────────────────────                  (2)
τ_i^eff     τ_i    s ∈ {F,B}   1 + κ_is · max(0,  Δx_s)
```

and the dynamics become

```
ẋ_i  =  −(1 / τ_i^eff(x)) · x_i  +  Σ_j J_ij x_j ,     J = D G,  G ≥ 0   (3)
```

where `G` is now **all-positive**: every entry means "worse source, worse
target".

### 2.2 Behaviour

- **Source displaced in the pathological direction** (`Δx_s > 0`, sarcopenia or
  bone loss): the numerator is 1, the denominator exceeds 1, so
  `τ_i^eff = τ_i · (1 + κ_is Δx_s) > τ_i`. Recovery of every target the source
  protects **slows**. This is the osteosarcopenia mechanism stated correctly:
  losing muscle does not push inflammation up, it leaves the system slower to
  put inflammation back down.
- **Source displaced in the protective direction** (`Δx_s < 0`, a trained
  individual): `τ_i^eff = τ_i / (1 + κ_is |Δx_s|) < τ_i`. Recovery speeds up.
  This is the exercise-intervention effect and it is now where it belongs — in
  the recovery rate, which is what exercise RCTs actually measure.
- `τ_i^eff > 0` for all `Δx_s` and all `κ_is ≥ 0`, so the form cannot flip the
  sign of self-restoration however far the source is displaced. The additive
  form has no such guarantee.

### 2.3 Linearisation at the reference

Write `f(Δ) = 1/τ_i^eff` as a function of a single `Δx_s = Δ`. The two branches
are chosen so that `f` is C¹ at the origin:

```
Δ > 0 :  f(Δ) = (1/τ_i) · (1 + κ_is Δ)^(−1)   ⇒  f′(0⁺) = −κ_is / τ_i
Δ < 0 :  f(Δ) = (1/τ_i) · (1 − κ_is Δ)        ⇒  f′(0⁻) = −κ_is / τ_i
```

so `f(0) = 1/τ_i` and `f′(0) = −κ_is/τ_i`, with no kink. Differentiating the
self-restoration term of (3):

```
 ∂                            ∂(1/τ_i^eff)          κ_is
───  [ −(1/τ_i^eff) x_i ]  =  − x_i · ───────────  =  x_i · ──── ,       (4)
∂x_s                              ∂x_s               τ_i
```

which **vanishes at the reference**, because `x_i = 0` there. Therefore:

> **The κ terms contribute nothing to the Jacobian at the youthful reference.**
> The linearised system at the set-point is exactly `A = −D(I − G)` with
> `G ≥ 0`, and stability is governed by `ρ(G) < 1` alone
> (`docs/units_convention.md` §1, condition G2).

The protective couplings are second-order by construction: they matter only when
the target is already displaced, which is precisely when recovery rate is the
thing that matters. They shape the recovery trajectory and the basin geometry;
they do not create or destroy the equilibrium.

### 2.4 Consequence for the stability margin

The fast-subsystem margin no longer receives any F-row contribution. The margin
is now set by `D` and by the all-positive `G`, and the compiled protective
entries cannot prop it up. This is the point of the exercise: the published
`α_fast(25) = −0.19` was reachable only because the uncoupled abscissa is bounded
at −0.1667 by `τ_C = 6 d` and the negative F row pushed past it. With the F row
removed, the coupling can only make `α` less negative, so `α` must be
recalibrated on a single anchor rather than fitted at both ends (§6).

---

## 3. Entries that move to κ

Eleven entries, all currently `sign = '-'` in `data/J_matrix_compiled_9x9.csv`.
`channel = kappa` per `docs/units_convention.md` §6.1.

### F row — 8 entries

| Entry | Tier / grade | g_healthy | g_pre_disease | g_disease |
|---|---|---|---|---|
| F→I | S / A | −0.35 | −0.30 | −0.25 |
| F→M | S / A | −0.30 | −0.25 | −0.20 |
| F→E | M / B | −0.12 | −0.10 | −0.08 |
| F→mito | M / A | −0.15 | −0.12 | −0.10 |
| F→P | W / B | qual_only | qual_only | qual_only |
| F→C | M / B | −0.12 | −0.10 | −0.08 |
| F→N | M / B | −0.18 | −0.15 | −0.12 |
| F→B | S / A | −0.30 | −0.25 | −0.18 |

### B row — 3 entries

| Entry | Tier / grade | g_healthy | g_pre_disease | g_disease |
|---|---|---|---|---|
| B→I | W / C | qual_only | qual_only | qual_only |
| B→M | M / B | −0.10 | −0.08 | −0.05 |
| B→N | W / C | qual_only | qual_only | qual_only |

After the move, `G` has **no negative entries**. The sign column becomes
redundant for `channel = J` rows and is retained only as provenance.

---

## 4. Regime-dependent entries

Three compiled entries are recorded in
`HDR-mechanistic/docs/sign_convention_reconciliation.md` as weakly negative in
the healthy regime and positive in disease. The compiled CSV currently carries
the **disease-regime value in all three basin columns**, which is why they read
as positive everywhere:

| Entry | Healthy biology | Disease biology | Compiled now (h / p / d) |
|---|---|---|---|
| N→I | acute cortisol is anti-inflammatory | GR resistance | 0.12 / 0.18 / 0.25 |
| N→B | pulsatile PTH is anabolic | continuous PTH is catabolic | 0.15 / 0.25 / 0.45 |
| C→I | temporal gating of the inflammatory response | circadian collapse | 0.10 / 0.15 / 0.22 |

**Chosen treatment: zero in the healthy basin; keep the compiled positive values
in pre-disease and disease.**

Rationale. These are not protective couplings in the F/B sense — the mechanism
is a sign reversal at a regime boundary, not a modulation of recovery rate — so
κ is the wrong instrument. Setting them to zero in the healthy basin states what
the evidence supports (the healthy-regime effect is weak and of contested sign)
without asserting a negative entry that would reintroduce the problem of §1. It
is also the conservative choice for the stability margin: it *removes* coupling
from the healthy basin, so it cannot flatter the healthy-basin result.

Cost, stated plainly: zeroing them removes stabilising-by-absence coupling from
the healthy basin and so raises the calibration scalar needed to hit the anchor,
from `c* = 0.209` to `c* = 0.281` (§6). The alternative — carrying a small
negative healthy-basin value — would restore exactly the pathology this document
exists to remove.

CSV: `regime_dependent = true` for these three rows;
`g_healthy = 0`, `g_pre_disease` and `g_disease` unchanged.

---

## 5. Tier → κ mapping principle

κ is a dimensionless sensitivity: `κ_is` is the fractional lengthening of
`τ_i` per 1 SD of pathological displacement in source `s` (from (1), at small
`Δ`, `τ_i^eff ≈ τ_i (1 + κ_is Δx_s)`).

**Principle: κ is calibrated against the gain-denominated exercise-RCT effect
sizes already in `HDR-mechanistic/data/inputs/effect_size_bounds.json`, not
against the compiled tier values.**

The mapping is identification, not rescaling. An exercise RCT reports a
*steady-state* effect: `J_F_I = 0.2–0.4 SD CRP reduction per exercise
intervention` (meta-analysis of exercise RCTs), `J_F_M = 0.3–0.5 SD HOMA-IR
improvement` (DPP), `J_B_F = 0.05–0.15` (MR). Under the damping form, holding
`Δx_F` at a sustained value `−δ` scales `τ_i` by `1/(1 + κ_iF δ)`, and the
steady-state displacement of target `i` under a fixed exogenous drive scales
with `τ_i`. So a measured steady-state fractional improvement `ε_is` in target
`i` for a sustained 1 SD protective displacement of `s` gives

```
κ_is  =  ε_is / (1 − ε_is)                                               (5)
```

Worked, for the two grade-A anchors (midpoints):

| Entry | ε from effect_size_bounds.json | κ by (5) |
|---|---|---|
| F→I | 0.30 | 0.43 |
| F→M | 0.40 | 0.67 |
| F→B | 0.10 | 0.11 |

Entries without an RCT-derived effect size take the tier band from
`effect_size_bounds.json` (strong ≥ 0.5, moderate 0.2–0.5, weak 0.05–0.2 SD per
SD) through (5), with the confidence grade carried into the Monte Carlo CV as in
`scripts/run_figure_uncertainty.py`. The `qual_only` entries (F→P, B→I, B→N)
stay `qual_only`; they are weak/grade-C and (5) would manufacture precision.

**Constraint the κ table must satisfy:** `κ_is` is identified by an
*interventional* effect size, so an entry whose only evidence is cross-sectional
or observational must not be promoted to a numeric κ. Of the eleven, only F→I,
F→M and F→B have interventional or MR support. The other eight are tier-derived
and should be marked as such in the κ table.

---

## 6. Numerical check

From `results/units_convention_numerics.json`, 7-axis fast subsystem, healthy
basin, compiled numeric entries only, unified τ from `data/tau_registry.json`:

| Configuration | c | α_fast(25) | α_fast(80) | α_fast(120) |
|---|---|---|---|---|
| Published (F row present, negative) | 0.89 | **−0.1889** | −0.1154 | −0.0150 |
| F row removed, published scale | 0.89 | −0.0917 | — | **+0.1543** |
| F row removed, recalibrated on α(120) = −0.004 | **0.209** | **−0.1615** | −0.0456 | **−0.0040** |
| F and B rows removed, regime-dependent zeroed | **0.281** | **−0.1666** | −0.0493 | **−0.0040** |

The prediction in the repair workplan was `c ≈ 0.21 → α_fast(120) = −0.004`,
`α_fast(25) ≈ −0.16`. Reproduced: `c* = 0.209`, `α_fast(25) = −0.1615`.

Two readings of this table matter.

**The published scale is not usable once the F row moves.** At `c = 0.89` with an
all-positive G the system is unstable from well before age 120
(`α_fast(120) = +0.1543`). The scale must be refitted, which is Task 1.3.

**`α_fast(25)` is now an output, not a target.** Recalibrating on the single
late-life anchor produces −0.1615 at age 25, against the published −0.19. The
two differ by 15%, and the recalibrated value sits just inside the uncoupled
bound of −0.1667 set by `τ_C = 6 d` — as it must, since an all-positive G can
only move `α` toward zero. The published −0.19 was never reachable with
non-negative coupling. Do not report the agreement as a validation; report
−0.1615 as what the single-anchor fit yields.

**τ sensitivity** (`tau_sensitivity`). Each rejected value is substituted as
its own *age trajectory* (HDR-mechanistic's formula
`τ = lo + (hi−lo)·(age−30)/50`), not as a single number — substituting a
constant flattens the age trajectory by construction and overstates the case.

At the published `c = 0.89` with the signed J, i.e. the published model:

| Age | adopted | N → HPA (0.5→2 d) | C → mech (1→3 d) | F → mech (8→42 d) |
|---|---|---|---|---|
| 25 | −0.1889 | −0.1722 | −0.3486 | −0.1336 |
| 55 | −0.1542 | −0.1159 | −0.2036 | −0.0222 |
| 80 | −0.1154 | −0.0705 | −0.1243 | **+0.0190** |
| 120 | −0.0150 | **+0.0023** | −0.0269 | **+0.0578** |

Fold change in the recalibrated margin from age 25 to 120, F row removed at
`c* = 0.209`: adopted 40.4×, C → mechanistic 22.3×, F → mechanistic 31.3×,
N → HPA 309×.

Read this carefully, because it does not say what a single-value substitution
would have suggested.

- **The age trajectory is robust to all three.** Under the mechanistic τ_F the
  margin still erodes 31× across the lifespan. The qualitative claim — recovery
  margin degrades with age — does not depend on the F decision.
- **The published model's stability does.** Under the mechanistic τ_F at
  `c = 0.89` the fast subsystem crosses zero at roughly age 70 and is unstable
  at 80 (+0.0190). So τ_F decides whether the *published calibration* is
  admissible, not whether the framework's central claim holds.
- **τ_F changes which axis sets the margin.** Under the adopted value the
  slowest fast axis is C up to ~45 and I thereafter; under the mechanistic
  value it is F at every age up to ~100. That is a substantive structural
  difference and a reviewer will see it.
- The N sub-channel choice is worth 0.002–0.05 in `α` and flips the sign at
  age 120 (−0.0150 → +0.0023), so it is not quite free either, though it does
  not touch the ages the cohorts cover.

The registry's rejection of the grip-strength value
(`data/tau_registry.json`, axes.F.reconciliation) rests on the
perturbation-recovery criterion, not on which answer is convenient. It is still
the first thing a reviewer should attack, and §6 of
`docs/units_convention.md` records what would settle it empirically.

---

## 7. Cross-lagged concordance, restated

With `G ≥ 0`, concordance has one definition and needs no convention remark:

> **A pair is concordant when `β > 0`: worsening in the source predicts
> subsequent worsening in the target.**

This is now the native prediction of the model for every entry, including the
F and B rows, because the compiled matrix no longer contains anything that
predicts a negative β. The three-way "Convention A / B / C" apparatus in
`results/lead_lag_concordance_audit.md` can be retired, and the manuscript's
"Convention B" remark deleted.

`results/lead_lag_phi_table.json` records both counts per model, under
`concordance_definitions`:

| Model | compiled-sign (old Convention A) | all-positive (this document) |
|---|---|---|
| InCHIANTI 4-axis | 8 / 12 | 9 / 12 |
| InCHIANTI 5-axis | 7 / 20 | 12 / 20 |
| ELSA 3-axis | 5 / 6 | 5 / 6 |

Note what this does and does not buy. The all-positive definition raises the
InCHIANTI 5-axis count from 7/20 to 12/20, but 12/20 is not significant
(binomial p = 0.25 against chance). The honest statement is that the
reformulation removes an artefact in the *definition*, not that it rescues the
concordance result. The 4-axis result (9/12, binomial p = 0.073) is unchanged
in substance.

### 7.1 The I→N sign question, resolved

`HDR-mechanistic/results/gain_calibration_table.md` flags `I→N` (InCHIANTI
β = −0.043, p = 0.026) as "negative yet listed as concordant" and asks whether
the resting-heart-rate proxy sign is being handled correctly.

**It is not a proxy-sign problem.** The N proxy is `resting_hr` with `sign: +1`
in `scripts/inchianti_6axis_analysis.py` (config `4axis_hr`), i.e. higher
resting heart rate = worse N, which is the correct orientation. The problem is
in the sign-prediction dictionary. `scripts/inchianti_lead_lag.py` defines
`J_SIGNS` with comments written as (source, target) but consumes the keys as
(target, source):

```python
for (to_name, from_name), predicted_sign in J_SIGNS.items():
```

so the entry `("N","I"): -1` — whose comment reads "Poor autonomic function
increases inflammation (via vagal withdrawal)", a statement about N→I — is
applied to the pair **I→N**. The compiled entry for I→N is
`+0.12 / +0.18 / +0.25` (row `I,N,+,M,B` in `data/J_matrix_compiled_9x9.csv`).
The script predicted −1, observed −1, and recorded "concordant"; against the
compiled matrix the pair is **discordant**. The same inversion affects `M→N`
(predicted −1, compiled +0.18).

`results/lead_lag_concordance_audit.md` already reflects the correct reading
from the other side: it lists I→N among the pairs discordant under the
all-positive definition.

**Action.** Delete `J_SIGNS` from `scripts/inchianti_lead_lag.py` and read the
predicted sign from the CSV, which under this document is `+1` for every
`channel = J` entry. Correct the two affected pairs wherever the 9/12 and
5/6 counts appear in the manuscript; the counts above already use the corrected
definition.

---

## 8. Acceptance

- [x] Damping-modulation equations (1)–(3), with the C¹ linearisation (4) and
      the no-Jacobian-contribution result.
- [x] Entry list: 8 F-row, 3 B-row (§3); regime-dependent treatment (§4).
- [x] Concordance definition under all-positive `G` (§7), with the I→N question
      answered (§7.1).
- [x] Tier→κ principle tied to `effect_size_bounds.json` (§5), with the
      interventional-evidence constraint.
- [x] Numerical check: `α_fast(25)` and `α_fast(120)` at the published and
      recalibrated scales (§6).

## 9. Handover to HDR-mechanistic

Build the κ table with: equation (5) as the identification rule; the eleven
entries of §3; interventional evidence required for a numeric κ; tier bands and
grade-CVs from `effect_size_bounds.json`. Return it as
`data/exports/kappa_table.json` with per-entry `kappa`, `evidence_type`,
`confidence_grade` and the effect size it was identified from.
