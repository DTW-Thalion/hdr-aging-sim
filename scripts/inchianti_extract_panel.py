#!/usr/bin/env python3
"""
Extract harmonized InCHIANTI panel data from raw SAS files.

Reads the InCHIANTI_CD_Share directory (288 MB of SAS7BDAT files across
6 waves), harmonizes variable names, computes derived variables, and
saves a single analysis-ready panel to data/inchianti_panel.parquet.

Usage:
    python scripts/inchianti_extract_panel.py [--data-root PATH]

The output panel has one row per subject-wave with standardized columns.
This file is .gitignored (InCHIANTI data is access-restricted).
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.hdr_sim.inchianti import load_inchianti_panel, compute_youthful_reference

def main():
    parser = argparse.ArgumentParser(description="Extract InCHIANTI panel data")
    parser.add_argument("--data-root", default=None,
                        help="Path to InCHIANTI_CD_Share directory")
    args = parser.parse_args()

    print("=" * 60)
    print("InCHIANTI Panel Extraction")
    print("=" * 60)

    # Load all waves
    print("\nLoading panel data...")
    panel = load_inchianti_panel(data_root=args.data_root)
    print(f"\nTotal panel: {len(panel)} rows, {panel['code98'].nunique()} unique subjects")
    print(f"Waves: {sorted(panel['wave'].unique())}")

    # Summary per wave
    print("\nPer-wave summary:")
    for w in sorted(panel["wave"].unique()):
        wdf = panel[panel["wave"] == w]
        n_il6 = wdf["il6"].notna().sum()
        n_homa = wdf["homa_ir"].notna().sum()
        n_hr = wdf["resting_hr"].notna().sum()
        n_sppb = wdf["sppb"].notna().sum()
        print(f"  Wave {w}: N={len(wdf):5d}  IL-6={n_il6:5d}  HOMA-IR={n_homa:5d}  "
              f"HR={n_hr:5d}  SPPB={n_sppb:5d}")

    # Compute youthful reference
    print("\nComputing youthful reference...")
    ref = compute_youthful_reference(panel)
    for axis in ["I", "M", "N", "F"]:
        r = ref[axis]
        print(f"  {axis}: mean={r['mean_ref']:.3f}, sd={r['sd_ref']:.3f}, "
              f"n={r['n_ref']}, var={r['var']}")
    print(f"  Reference group: {ref['_meta']}")

    # Save outputs
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Save panel as parquet
    parquet_path = os.path.join("data", "inchianti_panel.parquet")
    panel.to_parquet(parquet_path, index=False)
    print(f"\nSaved panel to {parquet_path} ({os.path.getsize(parquet_path) / 1e6:.1f} MB)")

    # Save youthful reference
    ref_path = os.path.join("results", "inchianti_youthful_reference.json")
    with open(ref_path, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"Saved youthful reference to {ref_path}")

    # 4-axis complete data summary
    mask_4axis = (
        panel["il6"].notna() &
        panel["homa_ir"].notna() &
        panel["resting_hr"].notna() &
        panel["sppb"].notna()
    )
    n_4ax = mask_4axis.sum()
    n_subj_4ax = panel.loc[mask_4axis, "code98"].nunique()
    print(f"\n4-axis complete observations: {n_4ax} ({n_subj_4ax} subjects)")

    # Subjects with >=2 waves of 4-axis complete
    if n_4ax > 0:
        waves_per_subj = panel.loc[mask_4axis].groupby("code98")["wave"].nunique()
        for k in [2, 3, 4]:
            print(f"  Subjects with >={k} waves of 4-axis complete: {(waves_per_subj >= k).sum()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
