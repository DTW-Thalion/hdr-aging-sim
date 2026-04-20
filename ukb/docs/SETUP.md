# Setup Guide — HDR UK Biobank Pipeline

Step-by-step preparation for running this package end-to-end. If you have
already set up an HDR analysis environment (InCHIANTI or ELSA scripts)
you can skim — the UKB pipeline uses the same Python stack.

---

## 1. Prerequisites

### 1.1 UK Biobank data access

You must have:

- An **approved UK Biobank application** with access to at least the fields
  listed in [FIELD_MAPPING.md](FIELD_MAPPING.md).
- A local export of the participant data basket, refreshed against the
  current data release.
- The **current withdrawal list** (UKB ships a text file of withdrawn
  participant IDs with each refresh). Keep it alongside the data file
  and point `config.yaml → withdrawal_path` at it.
- **Mortality-linkage** fields (40000, 40001) and their companion
  administrative censoring date.
- (Optional) **Accelerometry** summary fields (90001–90060) or raw `.cwa`
  files if you have the activity-monitoring subset.
- (Optional) **DXA** BMD fields from the imaging subset.

The package does *not* query UKB's Data Portal; you run it on an export
that lives inside your approved environment.

### 1.2 Compute environment

| Resource      | Minimum | Recommended |
|---------------|---------|-------------|
| CPU cores     | 4       | 16+         |
| RAM           | 16 GB   | 64 GB       |
| Disk          | 10 GB   | 50 GB       |
| Python        | 3.9     | 3.11        |

The default pipeline fits comfortably in 16 GB as long as `chunk_size: 0`
(the default) can hold your full export in memory. If N > ~1M rows and
50+ columns, either enable chunked reads (`chunk_size: 50000`) or convert
the CSV to Parquet once.

### 1.3 Python packages

From the `ukb/` directory:

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All packages are open source and BSD/MIT licensed.

---

## 2. Prepare the data export

### 2.1 Recommended format: wide CSV

The simplest export is a single CSV with:

- One row per participant (`eid`)
- One column per `(field_id, instance, array_index)` tuple, named
  `{field}-{instance}.{array}` — e.g. `30710-0.0` for CRP at instance 0

This is the default format produced by UKB's `ukbconv` tool
(`ukbconv <bulk> csv`) and by most R tidying pipelines (`ukbtools::ukb_df`).

### 2.2 Alternative: melted/long basket

If your export is in long form — columns `eid, field_id, instance,
array_index, value` — the extraction script auto-detects and pivots it.
This format is common from Python exports using `pyukbb`.

### 2.3 Other naming conventions

The extractor accepts:

- Canonical `30710-0.0`
- R-style `f.30710.0.0` (from `ukbtools`)
- Spark/Parquet `p30710_i0_a0`
- A few lesser-known variants

If your export uses a convention we don't recognise, add a regex to
`_FIELD_PATTERNS` in [step1_extract.py](../step1_extract.py).

### 2.4 File placement

You have two options:

**Option A — leave data where UKB dropped it:**

```yaml
# config.yaml
data_path: "/path/to/your/ukb_export.csv"
withdrawal_path: "/path/to/withdrawals_YYYYMMDD.txt"
```

**Option B — symlink into the package:**

```bash
ln -s /abs/path/to/ukb_export.csv ukb/data/ukb.csv
ln -s /abs/path/to/withdrawals.txt ukb/data/withdrawals.txt
```

then set relative paths in the config. Option B keeps the whole analysis
together and is friendlier to shared-storage environments.

---

## 3. Configure

Open [config.yaml](../config.yaml) and work through the sections in order:

1. **Data paths** — edit `data_path`, `withdrawal_path`, `mortality_path`.
2. **Field IDs** — check each axis block. Most users can leave defaults.
3. **Instances** — default `[0, 2]` pairs baseline with the imaging
   assessment (longest interval, largest N at 2nd visit). Change to
   `[0, 1]` for the first-repeat assessment (~20K, ~6 yr interval).
4. **Analysis parameters** — the defaults match the InCHIANTI/ELSA
   manuscript. Lower `n_bootstrap` to 1000 for a fast sanity run.
5. **Tier definitions** — leave as-is unless you have additional axes.

The `config_hash` computed from your final YAML appears in every output
header for provenance.

---

## 4. First-time test run

### 4.1 Dry-run on 1,000 participants

```bash
python step1_extract.py --test
```

This loads 1,000 rows, exercises the field resolver, and writes a small
panel to `results/ukb_panel_long.parquet`. Inspect it:

```python
import pandas as pd
df = pd.read_parquet("results/ukb_panel_long.parquet")
print(df.head())
print(df[["delta_I", "delta_M", "delta_F", "delta_N"]].notna().mean())
```

If any HDR axis has 0% non-missing, investigate the corresponding field ID
in `config.yaml` before proceeding. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### 4.2 QC sanity check

```bash
python step2_qc.py
cat results/ukb_qc_report.md
```

Expected baseline ballpark:

- N instance 0: ~500,000
- CRP completeness: ~90%
- HbA1c completeness: ~90%
- Grip strength completeness: ~95%
- Pulse rate completeness: ~98%

If any axis is < 50% complete, something is likely misconfigured.

---

## 5. Run the full pipeline

```bash
python run_all.py
```

Takes 2–4 hours on a 16-core workstation. The runner:

- Runs all eight steps in order
- Isolates each step in `try/except` so one failure doesn't block the rest
- Auto-detects which tiers are feasible from the extracted panel
- Writes `results/ukb_hdr_summary.md` with the headline numbers

You can skip steps:

```bash
python run_all.py --skip step6,step8          # skip slow Cox + accel
python run_all.py --only step3,step4          # only eigenvalue analyses
```

Results directory stays under the package root by default (`./results/`).
Override with `output_dir` in the config.

---

## 6. Before sharing results

The package writes **only aggregate statistics** (JSON, Markdown, PDF) to
`results/`. The extracted panel (`ukb_panel_long.parquet`) is
individual-level and must stay inside the UKB environment.

Pre-share checklist:

- [ ] `results/ukb_panel_long.parquet` is NOT copied out
- [ ] `results/ukb_circadian_proxy.parquet` is NOT copied out (if present)
- [ ] JSON outputs contain counts and aggregate stats only — verify no raw
      biomarker arrays have leaked in
- [ ] Summary report rounding preserves privacy thresholds required by
      your UKB application
- [ ] Figure captions do not reveal individual trajectories

For Cox ΔC bootstrap distributions, UKB permits sharing the `ci_lower` /
`ci_upper` / `mean` fields we write by default. If you want to export the
full 2,000-point bootstrap distribution, confirm with your UKB application
co-ordinator — we do NOT write this to JSON by default.

---

## 7. Reproducibility

- Seed is fixed at 42 (override in `analysis.seed`). All bootstraps and
  permutations use `np.random.default_rng(seed)` with deterministic
  stream splits.
- The config hash appears in every output file header. If two runs
  produce different config hashes, compare the YAML.
- `J_matrix_compiled_9x9.csv` is bundled with the package and frozen.
- A re-run on identical inputs with the same seed produces byte-identical
  JSON and identical figures (modulo matplotlib version).

---

## 8. Next steps

- [FIELD_MAPPING.md](FIELD_MAPPING.md) — field-by-field reference
- [METHODS.md](METHODS.md) — statistical methods and formulas
- [INTERPRETATION.md](INTERPRETATION.md) — how to read each output
- [OUTPUTS.md](OUTPUTS.md) — complete output-file catalogue
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — error-by-error fixes
