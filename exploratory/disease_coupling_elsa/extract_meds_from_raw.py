"""
One-shot extractor: pulls medication + self-rated health columns from the
raw ELSA core wave files (inside ELISA Study 5050_V1.zip) and writes a
small parquet the `_utils.py` can merge with the existing biomarker panel.

Run this once; it writes data/elsa/elsa_med_flags_waves_2_4_6_8.parquet.
Delete the intermediate .tab extracts after use to save disk.

The raw zip must be at:
  ~/Downloads/ELISA Study 5050_V1.zip
"""

import os
import sys
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_PATH = os.path.join(REPO, "data", "elsa", "elsa_med_flags_waves_2_4_6_8.parquet")
RAW_ZIP = os.path.expanduser(os.path.join("~", "Downloads", "ELISA Study 5050_V1.zip"))

# Medication + self-rated-health columns to extract per wave.
# Each wave's file is named differently inside the zip.
WAVE_FILES = {
    2: "UKDA-5050-tab/tab/wave_2_core_data_v4.tab",
    4: "UKDA-5050-tab/tab/wave_4_elsa_data_v3.tab",
    6: "UKDA-5050-tab/tab/wave_6_elsa_data_v2.tab",
    8: "UKDA-5050-tab/tab/wave_8_elsa_data_eul_v2.tab",
}

# Column names we want (case-insensitive match; missing ones are skipped).
WANTED = [
    "idauniq",
    "hechmd",    # Cholesterol: prescribed statin
    "hemdb",     # Diabetes: meds
    "hemda",     # BP: meds
    "heins",     # Diabetes: insulin injections
    "heacea",    # Diabetes: ACE-I
    "hehrtmd",   # Blood-thinning (aspirin proxy)
    "hepmed",    # Knee/hip pain meds (NSAID proxy)
    "heostec",   # Osteoporosis: meds (bisphosphonate proxy)
    "hehno",     # HRT: currently taking
    "hehelf",    # Self-rated health
]


def _read_tab_selective(tab_path, wanted_lower):
    """Read only selected columns from a big tab file, case-insensitive."""
    # Peek header
    header = pd.read_csv(tab_path, sep="\t", nrows=0)
    lookup = {c.lower(): c for c in header.columns}
    to_use = [lookup[w] for w in wanted_lower if w in lookup]
    if not to_use:
        return pd.DataFrame()
    df = pd.read_csv(tab_path, sep="\t", usecols=to_use)
    df.columns = [c.lower() for c in df.columns]
    return df


def main():
    if not os.path.exists(RAW_ZIP):
        print(f"Raw zip not found at {RAW_ZIP}", file=sys.stderr)
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="elsa_extract_")
    print(f"Staging directory: {tmpdir}")
    per_wave = []
    try:
        with zipfile.ZipFile(RAW_ZIP, "r") as zf:
            for wave, member in WAVE_FILES.items():
                print(f"  Wave {wave}: extracting {member}")
                extracted = zf.extract(member, tmpdir)
                df = _read_tab_selective(extracted, WANTED)
                df["wave"] = wave
                got_cols = [c for c in WANTED if c in df.columns]
                missing = [c for c in WANTED if c not in df.columns]
                print(f"    got {len(df)} rows, cols: {got_cols}")
                if missing:
                    print(f"    missing in this wave: {missing}")
                per_wave.append(df)
                # Delete the big extracted file immediately
                try:
                    os.remove(extracted)
                except OSError:
                    pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    combined = pd.concat(per_wave, ignore_index=True, sort=False)
    # Ensure all columns exist
    for c in WANTED:
        if c not in combined.columns:
            combined[c] = np.nan

    print(f"\nCombined shape: {combined.shape}")
    print("Coverage by wave:")
    for w in sorted(combined["wave"].unique()):
        sub = combined[combined["wave"] == w]
        row = [f"  w{w}: N={len(sub)}"]
        for c in ["hechmd", "hemdb", "hemda", "hehrtmd", "hepmed",
                  "heostec", "heins", "hehno"]:
            if c in sub.columns:
                ne = (sub[c] == 1).sum()
                row.append(f"{c}+={ne}")
        print(" ".join(row))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote: {OUT_PATH}  ({os.path.getsize(OUT_PATH)} bytes)")


if __name__ == "__main__":
    main()
