"""
InCHIANTI data download manifest.
Maps Box file IDs to local paths for systematic download.
"""

# Critical files needed for the 4-axis HDR analysis
# Format: (box_file_id, local_relative_path, description)

DOWNLOAD_MANIFEST = [
    # === BASELINE (V8) ===
    # Already downloaded: labo_raw.sas7bdat
    ("1256803467956", "Baseline_V8/English/4.Data/SAS_Datasets/EKG_ENG_Doppler/mar_raw.sas7bdat", "BL ECG/Doppler"),
    ("1256790461712", "Baseline_V8/English/4.Data/SAS_Datasets/Physical_Exam/per_ana.sas7bdat", "BL Physical Exam analytic"),
    ("1256805568105", "Baseline_V8/English/4.Data/SAS_Datasets/Drugs/fmc_ana.sas7bdat", "BL Drugs"),
    ("1256787606107", "Baseline_V8/English/4.Data/SAS_Datasets/Diseases/adju_ana.sas7bdat", "BL Diseases"),
    ("1256804798694", "Baseline_V8/English/4.Data/SAS_Datasets/Interview/int_rawe.sas7bdat", "BL Interview"),
    ("1256803045980", "Baseline_V8/English/4.Data/SAS_Datasets/Medical_Exam/cli_rawe.sas7bdat", "BL Clinical exam"),

    # === FOLLOW-UP 1 (V5) ===
    ("1256799980781", "Follow-up1_V5/English/4.Data/SAS_Datasets/Assays/labf1raw.sas7bdat", "FU1 Assays"),
    ("1256790993066", "Follow-up1_V5/English/4.Data/SAS_Datasets/EKG_ENG_Doppler/marf1raw.sas7bdat", "FU1 ECG"),
    ("1256793499195", "Follow-up1_V5/English/4.Data/SAS_Datasets/Physical_Exam/pef1_ana.sas7bdat", "FU1 Physical Exam"),
    ("1256808229614", "Follow-up1_V5/English/4.Data/SAS_Datasets/Drugs/fmcf1ana.sas7bdat", "FU1 Drugs"),
    ("1256808324529", "Follow-up1_V5/English/4.Data/SAS_Datasets/Diseases/adjf1ana.sas7bdat", "FU1 Diseases"),

    # === FOLLOW-UP 2 (V4) ===
    ("1256789847846", "Follow-up2_V4/English/4.Data/SAS_Datasets/Assays/labf2raw.sas7bdat", "FU2 Assays"),
    ("1256812016078", "Follow-up2_V4/English/4.Data/SAS_Datasets/EKG_ENG_Doppler/marf2raw.sas7bdat", "FU2 ECG"),
    ("1256802363806", "Follow-up2_V4/English/4.Data/SAS_Datasets/Physical_Exam/pef2_ana.sas7bdat", "FU2 Physical Exam"),
    ("1256804748139", "Follow-up2_V4/English/4.Data/SAS_Datasets/Drugs/fmcf2ana.sas7bdat", "FU2 Drugs"),
    ("1256808471978", "Follow-up2_V4/English/4.Data/SAS_Datasets/Diseases/adjf2ana.sas7bdat", "FU2 Diseases"),

    # === FOLLOW-UP 3 (V3) ===
    ("1256806081054", "Follow-up3_V3/English/4.Data/SAS_Datasets/Assays/labf3raw.sas7bdat", "FU3 Assays"),
    ("1256792526818", "Follow-up3_V3/English/4.Data/SAS_Datasets/EKG_ENG_Doppler/marf3raw.sas7bdat", "FU3 ECG"),
    ("1256807612649", "Follow-up3_V3/English/4.Data/SAS_Datasets/Physical_Exam/pef3_ana.sas7bdat", "FU3 Physical Exam"),
    ("1256802571114", "Follow-up3_V3/English/4.Data/SAS_Datasets/Drugs/fmcf3ana.sas7bdat", "FU3 Drugs"),
    ("1256804444113", "Follow-up3_V3/English/4.Data/SAS_Datasets/Diseases/adjf3ana.sas7bdat", "FU3 Diseases"),

    # === FOLLOW-UP 4 (V2) — no ECG ===
    ("1256798190118", "Follow-up4_v2/English/4.Data/SAS_Datasets/Assays/labf4raw.sas7bdat", "FU4 Assays"),
    ("1256814958296", "Follow-up4_v2/English/4.Data/SAS_Datasets/Physical_Exam/pef4_ana.sas7bdat", "FU4 Physical Exam"),
    ("1256803453328", "Follow-up4_v2/English/4.Data/SAS_Datasets/Drugs/fmcf4ana.sas7bdat", "FU4 Drugs"),
    ("1256805711811", "Follow-up4_v2/English/4.Data/SAS_Datasets/Diseases/adjf4ana.sas7bdat", "FU4 Diseases"),

    # === FOLLOW-UP 5 (V1) — no assays, no ECG, no drugs ===
    ("1256815845065", "Follow-up5_v1/English/4.Data/SAS_Datasets/Physical_Exam/pef5_ana.sas7bdat", "FU5 Physical Exam"),

    # === VITAL STATUS ===
    ("1256817580324", "Vital_Status/1.Data/SAS_Datasets/Master_thru_Follow-up5/ana_raw.sas7bdat", "Vital status master"),
    ("1256817439498", "Vital_Status/1.Data/SAS_Datasets/Diseases_thru_Follow-up4/mala_ana.sas7bdat", "Diseases thru FU4"),
]

if __name__ == "__main__":
    print(f"Total files to download: {len(DOWNLOAD_MANIFEST)}")
    for i, (box_id, path, desc) in enumerate(DOWNLOAD_MANIFEST, 1):
        print(f"  {i:2d}. {desc:30s} -> {path}")
    print(f"\nBox URLs:")
    for box_id, path, desc in DOWNLOAD_MANIFEST:
        print(f"  https://app.box.com/file/{box_id}")
