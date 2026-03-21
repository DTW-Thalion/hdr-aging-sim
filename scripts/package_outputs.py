#!/usr/bin/env python3
"""
Package all figures (PDF/PNG) and result files (JSON) into a single ZIP.

Usage:
    python scripts/package_outputs.py

Output:
    outputs/hdr_aging_sim_outputs.zip
"""
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
ZIP_PATH = os.path.join(OUTPUT_DIR, 'hdr_aging_sim_outputs.zip')

EXTENSIONS = ('.pdf', '.png', '.json')


def main():
    files = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(EXTENSIONS) and f != os.path.basename(ZIP_PATH)
    )

    if not files:
        print("No output files found to package.")
        return

    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            path = os.path.join(OUTPUT_DIR, f)
            zf.write(path, f)
            print(f"  + {f}")

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"\nPackaged {len(files)} files -> {ZIP_PATH}")
    print(f"Archive size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
