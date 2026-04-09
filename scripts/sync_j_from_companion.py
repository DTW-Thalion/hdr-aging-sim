#!/usr/bin/env python3
"""Sync the J-matrix CSV from the hdr-jmatrix-mechanistic companion repo.

Usage:
    python scripts/sync_j_from_companion.py --source ../hdr-jmatrix-mechanistic/data/J_matrix_mechanistic_9x9.csv
    python scripts/sync_j_from_companion.py --source /path/to/companion/csv

What it does:
1. Copies the source CSV to data/J_matrix_compiled_9x9.csv
2. Computes SHA-256 hash
3. Compares sign counts and axis set against the provenance snapshot
4. Writes a sync log entry to data/sync_log.json
5. Prints a summary of what changed
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from hdr_sim.j_matrix_spec import JMatrixSpec, load_provenance_spec


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sync J-matrix CSV from companion repo')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source J-matrix CSV in the companion repo')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without copying')
    return parser.parse_args()


def main():
    args = parse_args()

    source = os.path.abspath(args.source)
    if not os.path.exists(source):
        print(f"ERROR: Source file not found: {source}")
        return 1

    dest = os.path.join(ROOT, 'data', 'J_matrix_compiled_9x9.csv')
    sync_log_path = os.path.join(ROOT, 'data', 'sync_log.json')

    # Build spec for the incoming file
    incoming = JMatrixSpec.from_csv(source)

    # Build spec for the current file (if it exists)
    if os.path.exists(dest):
        current = JMatrixSpec.from_csv(dest)
    else:
        current = None

    # Load provenance spec for comparison
    try:
        provenance = load_provenance_spec()
    except FileNotFoundError:
        provenance = None

    # Compute diffs vs provenance
    diff_vs_provenance = []
    if provenance is not None:
        diffs = incoming.validate_against(provenance)
        if not diffs:
            diff_vs_provenance.append("Identical to provenance snapshot")
        else:
            for d in diffs:
                diff_vs_provenance.append(d)
    else:
        diff_vs_provenance.append("No provenance snapshot available for comparison")

    # Compute diffs vs current
    diff_vs_current = []
    if current is not None:
        if current.sha256 == incoming.sha256:
            diff_vs_current.append("No changes (SHA-256 identical)")
        else:
            diffs = incoming.validate_against(current)
            for d in diffs:
                diff_vs_current.append(d)
            if not diffs:
                diff_vs_current.append(
                    "SHA-256 differs but sign counts and axes are identical "
                    "(magnitude or metadata changes only)")
    else:
        diff_vs_current.append("No existing file at destination")

    # Print summary
    print("=" * 60)
    print("J-MATRIX SYNC SUMMARY")
    print("=" * 60)
    print(f"  Source:      {source}")
    print(f"  Destination: {dest}")
    print(f"  SHA-256:     {incoming.sha256[:16]}...")
    print(f"  Axes ({incoming.n_axes}):   {list(incoming.axes)}")
    print(f"  Signs:       {incoming.sign_counts}")

    print(f"\n  Changes vs current file:")
    for d in diff_vs_current:
        print(f"    - {d}")

    print(f"\n  Changes vs provenance snapshot:")
    for d in diff_vs_provenance:
        print(f"    - {d}")

    if args.dry_run:
        print("\n  [DRY RUN] No files modified.")
        return 0

    # Copy the file
    if current is not None and current.sha256 == incoming.sha256:
        print("\n  File unchanged, skipping copy.")
    else:
        shutil.copy2(source, dest)
        print(f"\n  Copied {source} -> {dest}")

    # Update sync log
    if os.path.exists(sync_log_path):
        with open(sync_log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)
    else:
        log = []

    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_path": args.source,
        "sha256": incoming.sha256,
        "sign_counts": incoming.sign_counts,
        "n_axes": incoming.n_axes,
        "diff_vs_provenance": diff_vs_provenance,
        "diff_vs_current": diff_vs_current,
    }
    log.append(entry)

    with open(sync_log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    print(f"  Sync log updated: {sync_log_path}")

    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
