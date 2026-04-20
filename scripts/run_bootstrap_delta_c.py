#!/usr/bin/env python3
"""Bootstrap 95% CIs for delta-C (M5 - M4) in InCHIANTI and ELSA.

Produces results/bootstrap_delta_c.json with confidence intervals for:
  - InCHIANTI age65+ matched (N=923, 698 deaths):  M5 vs M4, M4a vs M4, M4b vs M4
  - ELSA 3-axis full matched   (N=5431, 1122 deaths):      M5 vs M4
  - ELSA 3-axis med-naive      (N≈3233, ≈618 deaths):      M5 vs M4
    (med-naive = r2hibpe == 0 AND r2diabe == 0 at baseline)

Point estimates match the values frozen in:
  - results/inchianti_cox_frozen.json  (dC = +0.014, +0.014, +0.007)
  - results/elsa_cox_frozen.json       (dC = +0.009 full, +0.013 med-naive)

Usage:
    python scripts/run_bootstrap_delta_c.py                       # default n_boot=2000
    python scripts/run_bootstrap_delta_c.py --n-boot 500          # quick check
    python scripts/run_bootstrap_delta_c.py --skip-elsa           # inchianti only
    python scripts/run_bootstrap_delta_c.py --elsa-cached         # reuse results/elsa_baseline_matched.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from hdr_sim.bootstrap import bootstrap_delta_c

INCHIANTI_MATCHED_CSV = os.path.join(ROOT, "results", "inchianti_baseline_matched.csv")
ELSA_MATCHED_CSV = os.path.join(ROOT, "results", "elsa_baseline_matched.csv")
OUTPUT_JSON = os.path.join(ROOT, "results", "bootstrap_delta_c.json")

INCHIANTI_BASE = ["age", "female", "dx_frailty"]
INCHIANTI_BIO = ["log_il6", "log_homa", "resting_hr", "sppb"]
INCHIANTI_M4 = ["age", "female", "dx_frailty"]
INCHIANTI_M4A = INCHIANTI_M4 + INCHIANTI_BIO
INCHIANTI_M4B = INCHIANTI_M4 + ["log_swds"]
INCHIANTI_M5 = INCHIANTI_M4 + INCHIANTI_BIO + ["log_swds"]

ELSA_ADJ = ["smoking", "diabetes", "highbp"]
ELSA_BIO = ["log_crp", "hba1c", "grip_max", "bmival"]
ELSA_M4 = ["age", "sex", "rockwood_fi"] + ELSA_ADJ
ELSA_M5 = ["age", "sex"] + ELSA_BIO + ["swds_gamma", "rockwood_fi"] + ELSA_ADJ


def load_inchianti_matched() -> pd.DataFrame:
    if not os.path.exists(INCHIANTI_MATCHED_CSV):
        raise FileNotFoundError(
            f"{INCHIANTI_MATCHED_CSV} not found. "
            f"Run scripts/inchianti_survival.py first to produce it."
        )
    df = pd.read_csv(INCHIANTI_MATCHED_CSV)
    n, d = len(df), int(df["event"].sum())
    print(f"  InCHIANTI matched baseline: N={n}, events={d} "
          f"(expected N=923, 698)")
    required = set(INCHIANTI_M5 + ["time_years", "event"])
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"InCHIANTI matched baseline missing columns: {missing}")
    return df


def build_elsa_matched(use_cache: bool = False) -> pd.DataFrame:
    """Reconstruct the ELSA 3-axis full matched baseline by invoking the
    data-loading pipeline from scripts.run_medication_sensitivity.

    Mirrors the matching logic in run_matched_cox (same N across all 5
    nested models). The returned frame also carries `idauniq` and a
    `med_naive` flag (1 = r2hibpe == 0 AND r2diabe == 0 at baseline) so
    the med-naive subsample can be filtered without reloading the raw
    ELSA data.
    """
    if use_cache and os.path.exists(ELSA_MATCHED_CSV):
        df = pd.read_csv(ELSA_MATCHED_CSV)
        if "med_naive" in df.columns and "idauniq" in df.columns:
            n, d = len(df), int(df["deceased"].sum())
            nn = int(df["med_naive"].sum())
            nd = int(df.loc[df["med_naive"] == 1, "deceased"].sum())
            print(f"  ELSA matched baseline (cached): N={n}, events={d}; "
                  f"med-naive subset N={nn}, events={nd}")
            return df
        print("  ELSA cache is missing idauniq/med_naive columns — "
              "rebuilding from raw data.")

    print("  Rebuilding ELSA 3-axis full matched baseline via "
          "scripts.run_medication_sensitivity...")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    # Import lazily — this triggers data loading imports
    import run_medication_sensitivity as rms

    merged, harm, files = rms.load_all_data()
    axes_3 = ["dx_I", "dx_M", "dx_F"]
    complete_3 = rms.compute_swds_cross_sectional(merged, axes_3, "complete_3axis")

    if any(col in complete_3.columns for col in rms.ADL_ITEMS):
        complete_3["rockwood_fi"] = complete_3.apply(rms.compute_rockwood_fi, axis=1)
    else:
        complete_3["rockwood_fi"] = np.nan

    baseline_3 = rms.build_survival_baseline(complete_3)

    required = [c for c in ELSA_M5 + ["time", "deceased", "idauniq"]
                if c in baseline_3.columns]
    missing = set(ELSA_M5 + ["time", "deceased", "idauniq"]) - set(required)
    if missing:
        raise RuntimeError(f"ELSA baseline missing expected columns: {missing}")

    matched = baseline_3[required].dropna().copy()
    matched["deceased"] = matched["deceased"].astype(int)

    # Attach med-naive flag using r2hibpe / r2diabe from the harmonised
    # file (matching the definition in run_medication_sensitivity.py:835).
    hibpe_col, diabe_col = "r2hibpe", "r2diabe"
    if hibpe_col in harm.columns and diabe_col in harm.columns:
        med_naive_ids = set(
            harm.loc[(harm[hibpe_col] == 0) & (harm[diabe_col] == 0),
                     "idauniq"].values
        )
        print(f"  Med-naive (no {hibpe_col}, no {diabe_col} at baseline): "
              f"{len(med_naive_ids):,} people")
    else:
        print(f"  WARNING: {hibpe_col}/{diabe_col} not in harmonised; "
              f"falling back to hemda/hemdb at wave 2.")
        wave2 = merged[merged["wave"] == 2]
        med_naive_ids = set(
            wave2.loc[(wave2.get("hemda", pd.Series(dtype=float)) != 1) &
                      (wave2.get("hemdb", pd.Series(dtype=float)) != 1),
                      "idauniq"].values
        )
    matched["med_naive"] = matched["idauniq"].isin(med_naive_ids).astype(int)

    n, d = len(matched), int(matched["deceased"].sum())
    nn = int(matched["med_naive"].sum())
    nd = int(matched.loc[matched["med_naive"] == 1, "deceased"].sum())
    print(f"  ELSA matched baseline: N={n}, events={d} "
          f"(expected N=5431, 1122)")
    print(f"  ELSA med-naive subset:  N={nn}, events={nd} "
          f"(expected N~3233, ~618)")

    matched.to_csv(ELSA_MATCHED_CSV, index=False)
    print(f"  Saved matched baseline -> {os.path.relpath(ELSA_MATCHED_CSV, ROOT)}")
    return matched


def fmt_row(label: str, r: dict) -> str:
    lo, hi = r["ci_95"]
    return (f"  {label:<24s} {r['delta_c_point']:+.4f}  "
            f"[{lo:+.4f}, {hi:+.4f}]   "
            f"p={r['p_value']:.4f}   "
            f"N={r['n']} events={r['events']}   "
            f"boot={r['n_successful']}/{r['n_boot']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="Bootstrap resamples (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-inchianti", action="store_true")
    parser.add_argument("--skip-elsa", action="store_true")
    parser.add_argument("--skip-elsa-full", action="store_true",
                        help="Skip the ELSA-full bootstrap (useful when only "
                             "the med-naive subset needs to be (re)computed)")
    parser.add_argument("--skip-elsa-mednative", action="store_true",
                        help="Skip the ELSA med-naive bootstrap")
    parser.add_argument("--elsa-cached", action="store_true",
                        help=f"Reuse {os.path.relpath(ELSA_MATCHED_CSV, ROOT)} "
                             "if present, skip ELSA data reload")
    parser.add_argument("--merge-existing", action="store_true",
                        help=f"If {os.path.relpath(OUTPUT_JSON, ROOT)} already "
                             "exists, preserve its contents for any section "
                             "this run skipped.")
    args = parser.parse_args()

    print("=" * 70)
    print("Bootstrap 95% CI for delta-C (M5 - M4)")
    print(f"  n_boot={args.n_boot}, seed={args.seed}, event-stratified")
    print("=" * 70)

    out = {
        "_description": (
            "Bootstrap 95% confidence intervals for ΔC = C(M5) - C(M4) in "
            "Cox models of mortality. Event-stratified resampling preserves "
            "the number of deaths across bootstrap samples."
        ),
        "method": {
            "resampling": "event-stratified (separate resample within "
                          "event==0 and event==1 strata)",
            "c_index": "Harrell's C via CoxPHFitter.concordance_index_",
            "penalizer_fallback": 0.01,
            "ci": "percentile (2.5, 97.5)",
            "p_value": "fraction of bootstrap delta_c <= 0 "
                       "(one-sided H0: M5 does not improve on M4)",
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "inchianti_age65": {},
        "elsa_full": {},
        "elsa_mednative": {},
    }

    t0 = time.time()

    # -------------------- InCHIANTI --------------------
    if not args.skip_inchianti:
        print("\n--- InCHIANTI (age65+, matched) ---")
        inch = load_inchianti_matched()

        tcol, ecol = "time_years", "event"

        print("\n[InCHIANTI M5 vs M4]")
        out["inchianti_age65"]["M5_vs_M4"] = bootstrap_delta_c(
            inch, INCHIANTI_M4, INCHIANTI_M5,
            time_col=tcol, event_col=ecol,
            n_boot=args.n_boot, seed=args.seed,
        )

        print("\n[InCHIANTI M4a vs M4]  (frailty + biomarkers vs frailty)")
        out["inchianti_age65"]["M4a_vs_M4"] = bootstrap_delta_c(
            inch, INCHIANTI_M4, INCHIANTI_M4A,
            time_col=tcol, event_col=ecol,
            n_boot=args.n_boot, seed=args.seed + 1,
        )

        print("\n[InCHIANTI M4b vs M4]  (frailty + SWDS-Gamma vs frailty)")
        out["inchianti_age65"]["M4b_vs_M4"] = bootstrap_delta_c(
            inch, INCHIANTI_M4, INCHIANTI_M4B,
            time_col=tcol, event_col=ecol,
            n_boot=args.n_boot, seed=args.seed + 2,
        )

    # -------------------- ELSA --------------------
    if not args.skip_elsa:
        print("\n--- ELSA (3-axis full, matched) ---")
        elsa = build_elsa_matched(use_cache=args.elsa_cached)

        tcol, ecol = "time", "deceased"

        if not args.skip_elsa_full:
            print("\n[ELSA M5 vs M4 — full matched]")
            out["elsa_full"]["M5_vs_M4"] = bootstrap_delta_c(
                elsa, ELSA_M4, ELSA_M5,
                time_col=tcol, event_col=ecol,
                n_boot=args.n_boot, seed=args.seed + 3,
            )

        if not args.skip_elsa_mednative:
            if "med_naive" in elsa.columns:
                elsa_mn = elsa[elsa["med_naive"] == 1].reset_index(drop=True)
                nn, nd = len(elsa_mn), int(elsa_mn[ecol].sum())

                # Drop near-zero-variance covariates — in the med-naive
                # subset diabetes/highbp are ~always 0 by construction and
                # cause Cox convergence failures. Matches the behaviour of
                # run_medication_sensitivity.run_matched_cox:565.
                drop_covs = [c for c in ELSA_ADJ
                             if c in elsa_mn.columns
                             and elsa_mn[c].std() < 0.01]
                m4_mn = [c for c in ELSA_M4 if c not in drop_covs]
                m5_mn = [c for c in ELSA_M5 if c not in drop_covs]
                print(f"\n[ELSA M5 vs M4 — med-naive]  N={nn}, events={nd}")
                if drop_covs:
                    print(f"  Dropping near-zero-variance covariates "
                          f"(std < 0.01): {drop_covs}")

                out["elsa_mednative"]["M5_vs_M4"] = bootstrap_delta_c(
                    elsa_mn, m4_mn, m5_mn,
                    time_col=tcol, event_col=ecol,
                    n_boot=args.n_boot, seed=args.seed + 4,
                )
            else:
                print("\n  Skipping ELSA med-naive bootstrap (no med_naive "
                      "column in matched baseline — rebuild the cache).")

    elapsed = time.time() - t0

    # -------------------- Merge with existing (optional) --------------------
    if args.merge_existing and os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            prior = json.load(f)
        for section in ("inchianti_age65", "elsa_full", "elsa_mednative"):
            prior_sec = prior.get(section, {}) or {}
            new_sec = out.get(section, {}) or {}
            for comp, val in prior_sec.items():
                if comp not in new_sec and isinstance(val, dict):
                    new_sec[comp] = val
            out[section] = new_sec
        print(f"\n  Merged prior results from {os.path.relpath(OUTPUT_JSON, ROOT)} "
              "for sections this run skipped.")

    # -------------------- Persist --------------------
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {os.path.relpath(OUTPUT_JSON, ROOT)}  "
          f"(total {elapsed/60:.1f} min)")

    # -------------------- Summary table --------------------
    print("\n" + "=" * 70)
    print(f"Bootstrap delta-C ({args.n_boot} resamples, 95% percentile CI):")
    print("=" * 70)
    print(f"  {'comparison':<24s} {'point':<8s}  "
          f"{'ci_95':<20s}   p        N / events / boot")

    if not args.skip_inchianti:
        for label, key in [("InCHIANTI M5 - M4", "M5_vs_M4"),
                           ("InCHIANTI M4a - M4", "M4a_vs_M4"),
                           ("InCHIANTI M4b - M4", "M4b_vs_M4")]:
            r = out["inchianti_age65"][key]
            print(fmt_row(label, r))

    if not args.skip_elsa:
        r = out["elsa_full"]["M5_vs_M4"]
        print(fmt_row("ELSA M5 - M4 (full)", r))
        rmn = out.get("elsa_mednative", {}).get("M5_vs_M4")
        if rmn is not None:
            print(fmt_row("ELSA M5 - M4 (med-naive)", rmn))

    # Threshold interpretation
    print("\n--- Reviewer-facing interpretation ---")
    for name, r in [(k, v) for section
                    in ("inchianti_age65", "elsa_full", "elsa_mednative")
                    for k, v in out[section].items() if isinstance(v, dict)
                    and "ci_95" in v]:
        lo, hi = r["ci_95"]
        bits = []
        bits.append("CI excludes 0" if r["ci_excludes_zero"] else "CI includes 0")
        bits.append("CI excludes 0.01" if r["ci_excludes_0_01"]
                    else "CI includes 0.01")
        print(f"  {name:<12s} dC={r['delta_c_point']:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  -- {'; '.join(bits)}")


if __name__ == "__main__":
    main()
