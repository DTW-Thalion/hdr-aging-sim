#!/usr/bin/env python
"""Synthetic cohort validation of the mechanistic model.

Generates an ELSA-like synthetic cohort (N=5000, 3-axis, 4 visits) and
runs the Tier-1 pipeline under four conditions:
  a. Clean (no confounds)
  b. Survivorship only
  c. Medication only
  d. Both (realistic)

Verifies that the mechanistic parameterisation reproduces the R6
empirical patterns.

Produces: results/synthetic_validation_report.md
          results/synthetic_validation.json
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from hdr_sim.mechanistic_model import HDRMechanisticModel
from hdr_sim.observation_model import ObservationModel
from hdr_sim.synthetic_cohort import SyntheticCohort
from hdr_sim.tier1_pipeline import Tier1Pipeline


def _run_condition(label, model, obs, surv, med, seed):
    """Generate cohort and run Tier-1 pipeline for one condition."""
    print(f"\n--- Condition: {label} ---")
    t0 = time.time()
    gen = SyntheticCohort(
        model, obs,
        n_persons=5000, age_range=(50, 90),
        n_visits=4, visit_interval_years=4, seed=seed,
    )
    data = gen.generate(survivorship=surv, medication=med)
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.1f}s  (N={data.n_persons}, visits={data.n_visits})")

    # Attrition summary
    alive_last = data.alive[:, -1].sum()
    print(f"  Alive at final visit: {alive_last}/{data.n_persons}")

    pipe = Tier1Pipeline(data)
    result = pipe.full_analysis()
    result["generation_time_s"] = round(elapsed, 1)
    result["condition"] = label

    # Print key findings
    gc = result["gamma_change"]
    if gc:
        lm_vals = [r["lambda_max"] for r in gc]
        ages = [r["age_mid"] for r in gc]
        print(f"  lambda_max(Gamma_change): {' -> '.join(f'{v:.3f}' for v in lm_vals)}")
        print(f"    ages: {ages}")
        print(f"    increases: {result['summary']['gamma_change_increases']}")

    gx = result["gamma_cross_sectional"]
    if gx:
        lm_x = [r["lambda_max"] for r in gx]
        print(f"  lambda_max(Gamma_cross): {' -> '.join(f'{v:.3f}' for v in lm_x)}")

    swds = result["swds"]
    if swds:
        means = [(k, v["swds_mean"]) for k, v in sorted(swds.items())]
        print(f"  SWDS means: {' -> '.join(f'{m:.3f}' for _, m in means)}")
        print(f"    increases: {result['summary']['swds_increases']}")

    return result, data


def _check_medication_compression(result_clean, result_med):
    """Check that medication compresses off-diagonal correlations."""
    pr_clean = result_clean.get("primacy_ratio", [])
    pr_med = result_med.get("primacy_ratio", [])
    if not pr_clean or not pr_med:
        return False, 0.0

    # Average C across strata
    c_clean = sum(r["C"] for r in pr_clean) / len(pr_clean)
    c_med = sum(r["C"] for r in pr_med) / len(pr_med)
    compressed = c_med < c_clean
    reduction = (c_clean - c_med) / c_clean if c_clean > 0 else 0.0
    return compressed, reduction


def _check_gamma_cross_decreases(result_surv):
    """Check that cross-sectional lambda_max decreases with age (survivorship)."""
    gx = result_surv.get("gamma_cross_sectional", [])
    if len(gx) < 2:
        return False
    vals = [r["lambda_max"] for r in sorted(gx, key=lambda r: r["age_mid"])]
    return vals[-1] < vals[0]


def _check_swds_full_vs_naive(result_full, result_clean):
    """Full-sample SWDS weaker than medication-naive (clean)."""
    swds_full = result_full.get("swds", {})
    swds_clean = result_clean.get("swds", {})
    if not swds_full or not swds_clean:
        return False

    # Compare mean SWDS in oldest stratum
    oldest_key = max(swds_full.keys())
    if oldest_key not in swds_clean:
        return False

    return swds_full[oldest_key]["swds_mean"] <= swds_clean[oldest_key]["swds_mean"]


def main():
    results_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_total = time.time()

    model = HDRMechanisticModel(age=65)
    obs = ObservationModel("ELSA_3axis")

    output = {"timestamp": timestamp, "conditions": {}}

    # Run four conditions
    res_a, _ = _run_condition("clean", model, obs, surv=False, med=False, seed=42)
    res_b, _ = _run_condition("survivorship", model, obs, surv=True, med=False, seed=43)
    res_c, _ = _run_condition("medication", model, obs, surv=False, med=True, seed=44)
    res_d, _ = _run_condition("both", model, obs, surv=True, med=True, seed=45)

    output["conditions"] = {
        "clean": res_a,
        "survivorship": res_b,
        "medication": res_c,
        "both": res_d,
    }

    # Acceptance checks
    gc_increases = res_a["summary"]["gamma_change_increases"]
    gc_cross_decreases = _check_gamma_cross_decreases(res_b)
    swds_increases = res_a["summary"]["swds_increases"]
    med_compressed, med_reduction = _check_medication_compression(res_a, res_c)
    swds_full_weaker = _check_swds_full_vs_naive(res_d, res_a)

    checks = {
        "gamma_change_increases_clean": gc_increases,
        "gamma_cross_decreases_survivorship": gc_cross_decreases,
        "swds_increases_with_age": swds_increases,
        "medication_compresses_correlations": med_compressed,
        "swds_full_weaker_than_naive": swds_full_weaker,
    }
    output["acceptance_checks"] = checks

    total_elapsed = time.time() - t_total
    output["total_elapsed_s"] = round(total_elapsed, 1)

    # Write JSON
    json_path = os.path.join(results_dir, "synthetic_validation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON: {json_path}")

    # Write markdown report
    md_path = os.path.join(results_dir, "synthetic_validation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_generate_report(output, checks, med_reduction))
    print(f"Report: {md_path}")

    # Print acceptance summary
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")
    print("\n=== ACCEPTANCE CRITERIA ===")
    for key, val in checks.items():
        status = "PASS" if val else "FAIL"
        print(f"  [{status}] {key}")


def _generate_report(output, checks, med_reduction):
    lines = []

    def w(s=""):
        lines.append(s)

    w("# Synthetic Cohort Validation Report")
    w(f"\nGenerated: {output['timestamp']}")
    w(f"Total elapsed: {output['total_elapsed_s']}s")
    w("\nDesign: N=5000, ages 50-90, 4 visits at 4-year intervals, ELSA 3-axis (I, M, F)")

    for label in ("clean", "survivorship", "medication", "both"):
        cond = output["conditions"][label]
        w(f"\n## Condition: {label}")
        w(f"Generation time: {cond['generation_time_s']}s")

        gc = cond.get("gamma_change", [])
        if gc:
            w("\n### lambda_max(Gamma_change) by age stratum\n")
            w("| Age | N | lambda_max | trace |")
            w("|-----|---|-----------|-------|")
            for r in gc:
                w(f"| {r['age_lo']}-{r['age_hi']} | {r['n']} | {r['lambda_max']:.4f} | {r['trace']:.4f} |")

        gx = cond.get("gamma_cross_sectional", [])
        if gx:
            w("\n### lambda_max(Gamma_cross) by age stratum\n")
            w("| Age | N | lambda_max |")
            w("|-----|---|-----------|")
            for r in gx:
                w(f"| {r['age_lo']}-{r['age_hi']} | {r['n']} | {r['lambda_max']:.4f} |")

        swds = cond.get("swds", {})
        if swds:
            w("\n### SWDS-Gamma by age stratum\n")
            w("| Age | N | mean | std | median |")
            w("|-----|---|------|-----|--------|")
            for k in sorted(swds.keys()):
                v = swds[k]
                w(f"| {v['age_lo']}-{v['age_hi']} | {v['n']} "
                  f"| {v['swds_mean']:.4f} | {v['swds_std']:.4f} "
                  f"| {v['swds_median']:.4f} |")

        pr = cond.get("primacy_ratio", [])
        if pr:
            w("\n### Primacy ratio\n")
            w("| Age | V_norm | C_norm | Pi |")
            w("|-----|--------|--------|-----|")
            for r in pr:
                w(f"| {r['age_lo']}-{r['age_hi']} | {r['V_norm']:.3f} | {r['C_norm']:.3f} | {r['Pi']:.3f} |")

    w("\n## Acceptance Criteria\n")
    for key, val in checks.items():
        status = "PASS" if val else "FAIL"
        w(f"- [{status}] **{key}**")

    if med_reduction > 0:
        w(f"\nMedication correlation reduction: {med_reduction*100:.1f}%")

    w("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
