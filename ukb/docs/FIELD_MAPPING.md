# UK Biobank Field Mapping

Canonical reference for every UKB field the pipeline reads. Field IDs
are stable across UKB data releases; column-name formatting varies by
export tool (see [SETUP.md §2.3](SETUP.md#23-other-naming-conventions)).

All numeric fields are passed through `pd.to_numeric(errors="coerce")`
before use. Any value that fails to parse becomes `NaN` and is treated
as missing.

---

## Demographics and administrative

| Field | Name                  | Instance | Notes |
|-------|-----------------------|----------|-------|
| eid   | Participant ID        | —        | Integer, unique |
| 21003 | Age at recruitment    | 0        | Used directly at instance 0 |
| 34    | Year of birth         | 0        | Used to derive age at later instances |
| 52    | Month of birth        | 0        | Defaulted to June (day 15) if missing |
| 31    | Sex                   | 0        | 0=female, 1=male |
| 53    | Date of assessment    | 0,1,2,3  | Per-instance |
| 21000 | Ethnic background     | 0        | Used for subgroup analyses only |
| 189   | Townsend deprivation  | 0        | Covariate |
| 20116 | Smoking status        | 0,1,2,3  | 0=never, 1=previous, 2=current |

Age at instance > 0 is computed as:
`(assessment_date - YYYY-MM-15) / 365.25`
using the birth year/month. This is accurate to ±15 days.

---

## HDR axis biomarkers

### Axis I — Inflammatory resolution

| Field | Biomarker     | Units | Transform    |
|-------|---------------|-------|--------------|
| 30710 | CRP (serum)   | mg/L  | log(clip(x, 0.01)) |

Axis sign: **positive** (higher = worse).

### Axis M — Metabolic gain

| Field | Biomarker          | Units      | Transform |
|-------|--------------------|------------|-----------|
| 30750 | HbA1c              | mmol/mol   | auto unit detect |
| 30740 | Glucose (fallback) | mmol/L     | linear    |
| 30820 | Insulin            | pmol/L     | for HOMA-IR only |
| 21001 | BMI                | kg/m²      | composite option |

Axis sign: **positive**.

HbA1c auto-conversion: if median of non-missing values < 15, treated as
DCCT % and converted to IFCC via `10.929 · (x − 2.15)`. Otherwise
already IFCC.

### Axis F — Skeletal muscle balance

| Field | Biomarker              | Units |
|-------|------------------------|-------|
| 46    | Hand grip strength, L  | kg    |
| 47    | Hand grip strength, R  | kg    |

Axis value = `max(L, R)` per instance, then sign-flipped (higher grip →
*lower* Δx_F).

Axis sign: **negative** (lower = worse). Sign flip is applied so all
axes share the convention "positive Δx = worse homeostasis".

### Axis N — Neuroautonomic balance

| Field | Biomarker                | Units  |
|-------|--------------------------|--------|
| 102   | Pulse rate, automated    | bpm    |
| 4080  | SBP, automated (alt)     | mmHg   |
| 4079  | DBP, automated (alt)     | mmHg   |

Axis sign: **positive** (higher resting HR = worse). SBP/DBP are
available as alternatives but are heavily medicated at baseline — use
with caution.

### Axis C — Circadian fidelity

Requires accelerometry. Tier 1 (summary fields):

| Field         | Biomarker                             |
|---------------|---------------------------------------|
| 90012         | Overall average acceleration (mg)     |
| 90087         | Fraction in moderate activity         |
| 90015–90018   | Acceleration by 6-hour time-of-day window |
| 1160          | Sleep duration (self-report, hrs)     |
| 1180          | Chronotype / morning-evening          |

Tier 2 (raw CWA): interdaily stability (IS), intradaily variability
(IV), relative amplitude (RA) — computed by [step8_circadian.py](../step8_circadian.py).

Axis sign: **positive** (more fragmentation / lower amplitude = worse).

### Axis B — Bone remodeling balance

| Field | Biomarker                    | Notes |
|-------|------------------------------|-------|
| 78    | Heel bone mineral density    | Most participants have this |
| 23246 | Femoral neck BMD (DXA)       | Imaging subset only |
| 23247 | Total hip BMD (DXA)          | Imaging subset only |
| 23248 | Lumbar spine BMD (DXA)       | Imaging subset only |

The config picks the first available field. For the imaging subset
replace `heel_bmd: "78"` with `femoral_neck_bmd: "23246"` to use DXA.

Axis sign: **negative** (lower BMD = worse); sign-flipped as for grip.

### Axis P — Proteostatic clearance (proxy)

| Field | Biomarker   | Units | Transform |
|-------|-------------|-------|-----------|
| 30720 | Cystatin C  | mg/L  | log       |

Axis sign: **positive** (higher cystatin = worse renal clearance =
proxy for proteostatic burden).

---

## Medications

UKB splits "medication for cholesterol, BP, or diabetes" by sex:

| Field | Population | Codes |
|-------|------------|-------|
| 6153  | Women      | 1=cholesterol, 2=BP, 3=insulin, 4=HRT, 5=OCP, −7=none |
| 6177  | Men        | 1=cholesterol, 2=BP, 3=insulin, −7=none               |
| 2986  | Both       | Insulin use (0/1)                                     |

The extractor collapses both fields per eid. Each of fields 6153 / 6177
has up to 4 array entries (one per reported medication).

Derived flags in the panel:

- `med_statin` — any array value = 1
- `med_antihtn` — any = 2
- `med_insulin` — 6153/6177 = 3 OR field 2986 = 1
- `med_hrt` — 6153 = 4 (women only)
- `n_med_classes` — sum of statin/antihtn/insulin

---

## Comorbidities

| Field | Condition                                           |
|-------|-----------------------------------------------------|
| 2443  | Diabetes diagnosed by doctor (0/1)                  |
| 6150  | Vascular/heart problems: 1=heart attack, 2=angina, 3=stroke, 4=hypertension, −7=none |
| 6152  | Blood clot / DVT: 5=DVT, 7=pulmonary embolism, other |
| 2453  | Cancer diagnosed by doctor (0/1)                    |

Derived flags:

- `co_diabetes`, `co_cancer` — field value == 1
- `co_hypertension` — 6150 any array = 4
- `co_heart` — 6150 any array in {1, 2, 3}
- `comorbidity_count` — sum of the four

---

## Frailty Index deficits

The Rockwood-style FI is the proportion of deficits present among the
available items. Fields used:

| Field | Deficit criterion                                    |
|-------|------------------------------------------------------|
| 2443  | Diabetes                                             |
| 6150  | Any vascular/heart problem                           |
| 6152  | DVT / pulmonary embolism                             |
| 2453  | Cancer                                               |
| 1220  | Long-standing illness (codes 1–4)                    |
| 2178  | Overall health poor/fair (codes 3–4)                 |
| 924   | Walking pace slow (codes 1–2)                        |
| 2188  | Long-standing illness / disability / infirmity       |
| 46/47 | Weakness: sex-specific grip cutoff (M<30kg, F<20kg)  |
| 864   | Days/week moderate activity == 0                     |
| 884   | Days/week vigorous activity == 0                     |

`frailty_index = sum(deficits) / n_available`. If zero items are
available at an instance the FI is `NaN`.

---

## Mortality linkage

| Field | Name              | Notes |
|-------|-------------------|-------|
| 40000 | Date of death     | Primary |
| 40001 | Primary cause (ICD-10) | All-cause mortality does not use this — retained for future analyses |

Censoring: set `analysis.censor_date` in `config.yaml` to the
administrative cut-off date of your current data release. Participants
alive on that date are censored there.

`time_years = (death_date or censor_date − baseline_date) / 365.25`.
Rows with `time_years ≤ 0` are dropped by step 6.

---

## Random-panel null candidates

[step7_null_tests.py](../step7_null_tests.py) draws random K-biomarker
panels from this pool for the random-panel null. Each has N > 100,000
at baseline:

| Field | Biomarker        | Field | Biomarker       |
|-------|------------------|-------|-----------------|
| 30600 | Albumin          | 30790 | Lipoprotein A   |
| 30610 | Alk. phosphatase | 30810 | Phosphate       |
| 30620 | ALT              | 30830 | SHBG            |
| 30630 | AST              | 30850 | Testosterone    |
| 30840 | Total bilirubin  | 30860 | Total protein   |
| 30680 | Calcium          | 30870 | Triglycerides   |
| 30690 | Cholesterol      | 30880 | Urate           |
| 30700 | Creatinine       | 30670 | Urea            |
| 30730 | GGT              | 30890 | Vitamin D       |
| 30760 | HDL              | 30020 | Haemoglobin     |
| 30770 | IGF-1            | 30010 | Red cell count  |
| 30780 | LDL              | 30000 | White cell count|
|       |                  | 30080 | Platelet count  |

For the null to be maximally informative, include as many of these in
your UKB export as you have approval for. The null adjusts its
effective pool automatically based on what it finds.

---

## Extending the field list

To add a new biomarker:

1. Add the field ID to `config.yaml` in the appropriate axis block or
   as a new axis.
2. If it belongs to an existing axis, edit the per-axis extraction
   block in [step1_extract.py](../step1_extract.py) (`_extract_instance`).
3. If it's a *new axis*, add it to:
   - `AXIS_COLS` in [step3_cross_sectional.py](../step3_cross_sectional.py)
   - Tier definitions in `config.yaml`
   - `J_SIGNS_FALLBACK` in [hdr_core.py](../hdr_core.py) (or extend the
     J-matrix CSV)
4. Re-run step 1 and subsequent steps.

Adding an axis that is not in the compiled J matrix will still work —
it simply contributes to the covariance estimate without a sign
prediction to test against.
