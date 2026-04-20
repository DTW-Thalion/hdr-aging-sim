# Troubleshooting

Practical error-by-error guidance. If you hit something not covered here,
check the script's stderr/stdout first, then open an issue on the
[parent repository](https://github.com/DTW-Thalion/hdr-aging-sim/issues).

---

## 1. Step 1 — data extraction

### `FileNotFoundError: Data file not found: /path/to/ukb_participant_data.csv`

Your `config.yaml → data_path` still points at the placeholder. Edit it.

### `KeyError: 'eid' column missing`

- Check the first few columns of your CSV: `head -1 ukb_export.csv`.
- If eid is there but named differently (e.g. `f.eid` or `participant_id`),
  rename it before loading, or edit `_load_raw` / `_extract_instance`
  in [step1_extract.py](../step1_extract.py).

### "Axis completeness = 0 for axis X"

Printed to stdout when no column for the configured field ID resolves.
Common causes:

1. **Field not in your approved basket.** Check UKB Data Showcase —
   your application may not include this field. Request it or set the
   axis to null in the config.
2. **Column-name format the extractor doesn't know.**
   Run `python -c "import pandas as pd; print(list(pd.read_csv('export.csv', nrows=0).columns)[:20])"`
   and look for your field. Add a regex to `_FIELD_PATTERNS` in
   [step1_extract.py](../step1_extract.py) if needed.
3. **Instance number mismatch.** Some fields are only collected at
   instance 0 (baseline questionnaire) but you've asked for instance 2
   (imaging). Config `instances: [0, 2]` is the right default; check
   your specific field's availability in UKB Showcase.

### "HbA1c median = 45.2 → treating as IFCC mmol/mol"

Auto-detection worked. If your cohort actually uses DCCT (%), force
the conversion by temporarily editing `_convert_hba1c` in
[step1_extract.py](../step1_extract.py).

### Memory error / `MemoryError: Unable to allocate …`

Set `chunk_size: 50000` in the config. The loader will process the CSV
in chunks. Parquet / feather formats don't need this; they stream by
default.

### Extraction is slow (> 30 minutes for 500K participants)

- Convert CSV to Parquet once: `pd.read_csv(...).to_parquet(...)`. After
  that, step 1 runs in < 5 minutes.
- Ensure `data_format: "parquet"` in the config if you did the conversion.

### Withdrawal file not applying

- Confirm the file is plain text, one eid per line, no header.
- Whitespace / BOM: `sed -i '1s/^\xEF\xBB\xBF//' withdrawals.txt`.
- The extractor reports "X eids excluded" — if that number is 0 but
  your file has content, the eids may be strings rather than integers;
  strip quotes.

---

## 2. Step 2 — QC

### Completeness looks wrong (e.g. 0% for grip strength but raw data clearly has values)

Step 2 reads `delta_F`, not `grip_max`. If `delta_F` is all NaN even
though `grip_max` is present, the sign-flip / standardisation in
step 1 may have failed silently. Inspect the parquet directly:

```python
import pandas as pd
p = pd.read_parquet("results/ukb_panel_long.parquet")
print(p[["grip_max", "delta_F", "instance"]].dropna().head())
```

### N by instance << expected

- Instance 2 is the imaging assessment (~100K of the ~500K cohort).
  That is the correct number.
- Instance 3 is repeat imaging (~60K). If you configured `instances:
  [0, 3]` but only have instance 2 in your export, step 1 silently
  produces empty rows for instance 3.

### Frailty Index all NaN

Step 1 needs at least one deficit field to be non-empty per participant.
If the FI is uniformly NaN:

- Check that the self-report fields (2443, 6150, 2178, 924) are present
  in your export.
- The FI is built at each instance; for instances > 0 many of the
  questionnaire fields aren't re-administered. A NaN FI at instance 2
  with a valid FI at instance 0 is expected.

---

## 3. Step 3 — cross-sectional

### "< 3 axes with data; skipping"

The tier requires at least 3 axes with ≥ 1000 non-missing values at
baseline. If you're targeting tier 3 (adds C) but accelerometry is
unavailable, expect tier 1 and 2 to run while tier 3/4 skip.

### `np.linalg.LinAlgError: Eigenvalues did not converge`

Rare. Usually means Γ̂ is degenerate because one axis is a linear
combination of others. Check `np.linalg.cond(cov)` on the stratum
data; if > 1e12 one of your axes is redundant.

### Bootstrap CI wider than expected

If you see CIs spanning an order of magnitude, either:

- N in that stratum is very small (< 100). Coverage is mis-calibrated
  at small N.
- An outlier biomarker value is dominating. Check
  `ukb_qc_stats.json → distributions_by_instance` for 5-SD outliers.

---

## 4. Step 4 — longitudinal

### "Total change-pairs: 0"

Only counts participants with all tier axes complete at TWO instances.
For tier 3 (with C), that requires participants who had accelerometry
at both instance 0 and instance 2. Very few do. Expect tier 3
longitudinal to be sparse.

### Change-panel build takes forever

It's iterating groups in pure pandas. For N=500K this can take
~20 minutes. If you hit 2+ hours, something is wrong — check that
`panel["eid"]` is sorted and that no eid has > 4 rows.

### Permutation p = 1.0 / 0.0 (boundary)

- p = 1.0 → observed trajectory is indistinguishable from random label
  shuffling. Genuine null — trend is not real.
- p = 0.0 → the observed ratio exceeded every permutation. Strong
  positive. The code reports p = (exceed + 1) / (B + 1) so the minimum
  possible is 1/(B+1) ≈ 0.001 for B = 1000.

---

## 5. Step 5 — lead-lag

### "n_tested_vs_J = 0"

The J matrix has NULL entries for the axes in your tier. This happens
if you added a new axis not in the compiled J matrix. The code still
runs but sign-concordance is vacuous.

### Bootstrap CIs span zero even for large β

Subject-clustered bootstrap is conservative by design. If the point
estimate is clearly nonzero but the CI crosses zero, the effect is
present at the group level but not reliable at the individual-cluster
level — usually driven by a handful of influential subjects.

### FDR q values are all identical

All p-values in the tier are roughly the same. With K × (K − 1) pairs
and strong multi-testing correction, that can collapse to a single
effective q. Not an error.

---

## 6. Step 6 — mortality

### `lifelines` not installed

```bash
pip install lifelines
```

Already in `requirements.txt`. If install fails on Windows, you may
need Microsoft Visual C++ Build Tools first (scipy dependency).

### "matched N << expected"

The matched-complete-case sample is the intersection of non-missing
across all covariates used in any model. If you see N = 20K from a
500K cohort, one covariate is heavily missing. Check:

```python
import pandas as pd
p = pd.read_parquet("results/ukb_panel_long.parquet")
base = p[p["instance"] == 0]
for c in ["delta_I", "delta_M", "delta_F", "delta_N",
         "frailty_index", "smoking", "co_diabetes", "co_hypertension"]:
    print(c, base[c].notna().mean())
```

The single most-missing covariate gates the whole model.

### `ConvergenceError: Check for high collinearity`

Two biomarkers are near-identical. Most commonly:

- HOMA-IR and HbA1c both in the same model (drop one)
- BMI and raw metabolic biomarkers (consider removing BMI if the tier
  already has HbA1c)

Remove offending columns from `biomarker_cols` in step 6.

### ΔC bootstrap returns NaN mean

Either:

- < 50 valid resamples completed (events or non-events too small).
  Try `analysis.n_bootstrap_cox: 5000` to get more valid samples.
- Models are failing inside the bootstrap loop. Check stderr for the
  exception message.

### Cox fit takes forever

`lifelines` is slow on large N. Two mitigations:

- Set `CoxPHFitter(penalizer=0.01)` (already the default in step 6)
- Down-sample the bootstrap to `n_bootstrap_cox: 500` for a first
  pass, then scale up once the numbers look sensible.

---

## 7. Step 7 — null tests

### "Random-panel candidates available: 0 / 25"

Your UKB export doesn't include any of the non-HDR biomarkers listed
in [FIELD_MAPPING.md](FIELD_MAPPING.md#random-panel-null-candidates).
Request them or accept that the random-panel null will not run.

### `observed_ratio = NaN`

One of the strata has < 50 participants with complete axis data, so
λ_max couldn't be computed. Reduce the number of strata (collapse
40–49 and 50–59 into a single 40–59 stratum) or widen them.

### Permutation null p = exactly 1/(B+1)

No shuffled permutation exceeded the observed ratio. Strong positive
signal. Increase `n_permutations` to 10,000 if you need a tighter
lower bound.

---

## 8. Step 8 — circadian

### "No accelerometry summary data; skipping"

Expected when `accel_summary_path: null` in the config. The pipeline
proceeds with tier 1 and 2 (which don't need C).

### Raw CWA processing: `actipy` fails to read file

Some CWA files are corrupted (battery died mid-wear). `actipy` emits
an exception; step 8 catches it and skips that file. If too many are
skipped, try a different toolkit (biobankAccelerometerAnalysis).

### `delta_C` populated for baseline but not imaging

Each participant only has ONE accelerometry window in UKB (typically
2013–2015). The pipeline writes the score to the baseline instance
row only. Longitudinal C-axis analysis is therefore not possible with
UKB's single-wear accelerometry design.

---

## 9. Run-all orchestration

### One step fails but others "succeeded" on missing data

`run_all.py` isolates each step in `try/except`. If step 3 produces
garbage because step 1 silently produced an empty panel, step 4 will
also produce garbage and report "ok". Always check
`ukb_qc_report.md` first — if the QC looks wrong, don't trust the
downstream results.

### `--only step4` runs but needs step3's output

Most steps are independent (they re-read the panel). Step 4 builds
its own change panel. Step 6 builds its own baseline sample. Only
step 8 modifies the panel (by writing `delta_C` back). Re-running
steps out of order is safe except that changing axis completeness
after steps 3–7 have run leaves stale results on disk — delete the
old JSONs before re-running.

### Step times much longer than docs

Two common causes:

- `n_bootstrap: 10000` on a 500K dataset is slow. Drop to 2000 for
  a first pass.
- Windows file I/O is 2–3× slower than Linux for Parquet reads. Move
  the data to a local SSD.

---

## 10. Reporting issues

When opening an issue, include:

- Step that failed (`step1_extract.py`)
- Full stderr output
- UKB data format (wide / long, column-name convention)
- Python version and `pip list` output
- `config_hash` from the last successful output

The package is intentionally conservative — it errors loudly rather
than silently producing wrong numbers. If you see unexpected success,
that is more worrying than an error.
