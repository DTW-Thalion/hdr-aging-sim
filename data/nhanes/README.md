# NHANES Data

This directory contains (or will download) NHANES 2011-2012 SAS transport files from the CDC website. Files are downloaded automatically by `scripts/run_nhanes_feasibility.py`.

## Files

| File | NHANES Table | Contents |
|------|-------------|----------|
| `demo.XPT` | DEMO_G | Demographics (age, sex, exam status) |
| `ghb.XPT` | GHB_G | Glycohemoglobin (HbA1c) |
| `mgx.XPT` | MGX_G | Muscle strength (grip) |
| `bmx.XPT` | BMX_G | Body measures (BMI) |

## Source

All data are publicly available from the CDC NHANES website:
https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&Cycle=2011-2012

## Variables Used

- `RIDAGEYR`: Age in years at screening
- `RIAGENDR`: Gender (1=Male, 2=Female)
- `LBXGH`: Glycohemoglobin (%)
- `MGDCGSZ`: Combined grip strength (kg)
- `BMXBMI`: Body mass index (kg/m²)

## Citation

Centers for Disease Control and Prevention (CDC). National Center for Health
Statistics (NCHS). National Health and Nutrition Examination Survey Data.
Hyattsville, MD: U.S. Department of Health and Human Services, Centers for
Disease Control and Prevention, 2011-2012.
