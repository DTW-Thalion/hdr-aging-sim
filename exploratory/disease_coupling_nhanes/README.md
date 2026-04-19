# Exploratory: Disease-Specific Coupling Modification — NHANES

## Tests A–E are not feasible in NHANES-Continuous.

All five coupling-modification tests (A statins, B antidiabetics,
C physical-function tertile, D NSAIDs, E bone) use **cross-lagged
regression on consecutive wave pairs of the same individual**:

$$\Delta_{target}(w+1) = \beta \cdot \Delta_{source}(w) + \gamma \cdot \Delta_{target}(w) + \delta \cdot \text{age}_w + \varepsilon$$

NHANES-Continuous (1999–present) is **cross-sectional** — each cycle
surveys a new independent sample. There are no repeat visits of the
same person across cycles, so `Δ_x(w+1) − Δ_x(w)` cannot be formed,
and the β coefficient above is undefined.

## What NHANES could support instead

- **Cross-sectional association** of Δ_source with Δ_target conditional
  on medication status at a single cycle. This is a fundamentally
  different hypothesis — it measures the *contemporaneous partial
  correlation* modified by treatment, not the *temporal cross-lagged
  coupling* the HDR framework predicts. Because it collapses the
  temporal structure, it cannot discriminate I→M from M→I and cannot
  test directional prediction — the D-vs-J discrimination (Test B)
  becomes impossible in principle.
- **NHANES III (1988–1994) + linked mortality follow-up** has repeat
  measures on ~4 years of follow-up for a subset, but the panel design
  differs enough from the coupling-modification framework that
  re-implementing from scratch is outside the scope of this
  exploratory folder.

## Existing NHANES work in this repo

See [`scripts/run_nhanes_feasibility.py`](../../scripts/run_nhanes_feasibility.py)
for the existing NHANES feasibility analysis, which tests **single
cross-sectional cycles** for consistency with the HDR framework — a
valid but distinct line of evidence.

## Recommendation

Do not attempt to force-fit Tests A–E into a NHANES framework. If
coupling-modification evidence beyond InCHIANTI + ELSA is wanted, the
next natural cohort is one of:

- **HRS** (Health and Retirement Study) — matched ELSA-style panel,
  US population, biomarker subsample with 4-year intervals.
- **SHARE** (Survey of Health, Ageing and Retirement in Europe) —
  European multi-country longitudinal, similar structure.
- **MIDUS** (Midlife in the United States) — biomarker project with
  two-visit panel ~10 years apart.
- **NHANES III longitudinal subset** — 1988–1994 baseline + ~4 year
  follow-up. A limited substitute for the full coupling battery.
