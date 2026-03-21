# ELSA Data

This directory should contain the English Longitudinal Study of Ageing (ELSA) data files (Waves 0–11).

**These files are NOT included in the repository** due to UK Data Service redistribution terms.

## How to obtain the data

1. Register at https://beta.ukdataservice.ac.uk/
2. Search for Study Number **SN 5050** ("English Longitudinal Study of Ageing: Waves 0-11, 1998-2024")
3. Download in TAB format
4. Place the required .tab files in this directory

## Required files (9 total)

| File | Purpose |
|------|---------|
| `gh_elsa_h_hdr_subset.tab` | Harmonised subset: demographics, conditions, ADL, CES-D, mortality status (212 cols × 21,679 people, ~10 MB) |
| `elsa_supplementary_variables.tab` | Individual ADL/mobility/medication/CES-D/physical activity (43 cols × 39,528 rows, ~3.5 MB) |
| `h_elsa_eol_a2.tab` | End-of-life: age at death, death year (215 vars) |
| `wave_2_nurse_data_v2.tab` | Blood biomarkers 2004-05 (210 vars × 7,666 people) |
| `wave_4_nurse_data.tab` | Blood biomarkers 2008-09 (249 vars) |
| `wave_6_elsa_nurse_data_v2.tab` | Blood biomarkers 2012-13 (419 vars) |
| `wave_8_elsa_nurse_data_eul_v1.tab` | Blood biomarkers 2016-17 (161 vars, half sample) |
| `elsa_nurse_w8w9_data_eul.tab` | Combined W8+W9 nurse data (169 vars) |
| `wave_11_elsa_nurse_data_eul.tab` | Blood biomarkers 2023-24 (261 vars) — optional |
