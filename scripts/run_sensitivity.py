#!/usr/bin/env python
"""Run the full sensitivity analysis suite and produce outputs.

Produces:
    results/sensitivity_analysis.json   — all numerical results
    results/sensitivity_report.md       — human-readable report

Usage:
    python scripts/run_sensitivity.py [--n-draws 10000] [--seed 42]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

# Ensure repo root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from hdr_sim.mechanistic_model import HDRMechanisticModel  # noqa: E402
from hdr_sim.prior_stress import PriorStressTest  # noqa: E402
from hdr_sim.sensitivity import PriorSensitivityAnalysis  # noqa: E402


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _fmt(x, decimals=6):
    return f"{x:.{decimals}f}"


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prior-scale",
        type=float,
        default=0.12,
        help="Multiplier on prior std for MC sampling (default 0.12, "
        "matching R6 confidence-grade perturbation widths).",
    )
    args = parser.parse_args()

    results_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "sensitivity_analysis.json")
    md_path = os.path.join(results_dir, "sensitivity_report.md")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_total = time.time()

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    print("Setting up model ...")
    model = HDRMechanisticModel(age=65)

    output = {
        "timestamp": timestamp,
        "model_info": {
            "n_axes": model._n,
            "axes": model.AXES,
            "calibration_scalar": model.calibration_scalar,
            "n_active_entries": len(model._entries),
            "n_excluded_entries": len(model._excluded),
            "prior_scale": args.prior_scale,
        },
    }

    # ------------------------------------------------------------------
    # 2. Monte Carlo analysis
    # ------------------------------------------------------------------
    print(
        f"Running MC analysis (N={args.n_draws}, "
        f"prior_scale={args.prior_scale}) ..."
    )
    t0 = time.time()
    sa = PriorSensitivityAnalysis(
        model, n_draws=args.n_draws, seed=args.seed,
        prior_scale=args.prior_scale,
    )
    ages = [30, 40, 50, 60, 70, 80]
    mc = sa.run_mc(ages=ages)
    mc_elapsed = time.time() - t0
    print(f"  MC completed in {mc_elapsed:.1f}s")

    output["mc_results"] = mc.summary()
    output["mc_results"]["elapsed_seconds"] = round(mc_elapsed, 1)

    # ------------------------------------------------------------------
    # 3. Entry sensitivity ranking
    # ------------------------------------------------------------------
    print("Computing entry sensitivities ...")
    sens_alpha = sa.entry_sensitivity(target="spectral_abscissa", age=65)
    sens_beta = sa.entry_sensitivity(target="bifurcation_margin", age=65)

    output["entry_sensitivity_alpha"] = sens_alpha[:15]
    output["entry_sensitivity_bifurcation"] = sens_beta[:15]

    # ------------------------------------------------------------------
    # 4. Stress tests
    # ------------------------------------------------------------------
    print("Running stress tests ...")
    stress = PriorStressTest(model)

    print("  Concordance: correct prior ...")
    conc_correct = stress.test_correct_prior()
    print(f"    concordance = {conc_correct['concordance']:.3f}")

    print("  Concordance: null prior ...")
    conc_null = stress.test_null_prior()
    print(f"    concordance = {conc_null['concordance']:.3f}")

    print("  Concordance: adversarial prior ...")
    conc_adv = stress.test_adversarial_prior()
    print(f"    concordance = {conc_adv['concordance']:.3f}")

    print("  Grade / confidence ablation ...")
    grade_ab = stress.test_grade_ablation()
    print(
        f"    ablated {grade_ab['n_ablated']} entries, "
        f"delta_alpha_rel = {grade_ab['delta_alpha_relative']:.4f}"
    )

    print("  Exclusion impact ...")
    excl = stress.test_exclusion_impact(excluded_magnitude=0.01)
    print(
        f"    delta_alpha_rel = {excl['delta_alpha_relative']:.4f}, "
        f"safe = {excl['exclusion_safe']}"
    )

    print("  Decomposition vs uniform priors ...")
    decomp = stress.test_decomposition_vs_uniform(n_draws=2000, age=65)
    print(f"    CI90 reduction = {decomp['ci90_reduction_pct']:.1f}%")

    output["stress_tests"] = {
        "concordance_correct": conc_correct,
        "concordance_null": conc_null,
        "concordance_adversarial": conc_adv,
        "grade_ablation": grade_ab,
        "exclusion_impact": excl,
        "decomposition_vs_uniform": decomp,
    }

    # ------------------------------------------------------------------
    # 5. Acceptance criteria checks
    # ------------------------------------------------------------------
    im_ids = {"J_I_M", "J_M_I"}
    top5_alpha = {s["coupling_id"] for s in sens_alpha[:5]}
    top3_beta = {s["coupling_id"] for s in sens_beta[:3]}
    im_dominant = bool(
        im_ids & top5_alpha or im_ids & top3_beta
    )

    output["acceptance_checks"] = {
        "stability_gt99_all_ages": all(
            mc.stable_fraction[a] > 0.99 for a in ages
        ),
        "monotone_alpha_gt99": mc.monotone_fraction > 0.99,
        "im_loop_dominant": im_dominant,
        "exclusion_safe_lt5pct": excl["exclusion_safe"],
        "concordance_ordering": (
            conc_correct["concordance"]
            > conc_null["concordance"]
            > conc_adv["concordance"]
        ),
    }

    total_elapsed = time.time() - t_total
    output["total_elapsed_seconds"] = round(total_elapsed, 1)
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 6. Write JSON
    # ------------------------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"JSON written: {json_path}")

    # ------------------------------------------------------------------
    # 7. Write Markdown report
    # ------------------------------------------------------------------
    md = _generate_report(output, mc, sens_alpha, sens_beta)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Report written: {md_path}")

    # Print acceptance summary
    checks = output["acceptance_checks"]
    print("\n=== ACCEPTANCE CRITERIA ===")
    for key, val in checks.items():
        status = "PASS" if val else "FAIL"
        print(f"  [{status}] {key}: {val}")


# ======================================================================
# Report generation
# ======================================================================


def _generate_report(output, mc, sens_alpha, sens_beta):
    lines = []

    def _w(s=""):
        lines.append(s)

    _w("# Sensitivity Analysis Report")
    _w(f"\nGenerated: {output['timestamp']}")
    _w(f"MC draws: {mc.n_draws}")
    _w(f"Total elapsed: {output['total_elapsed_seconds']}s")

    # -- MC results --
    _w("\n## 1. Monte Carlo Analysis")
    _w("\n### 1.1 Stability by Age\n")
    _w(
        "| Age | Stable % | alpha mean | alpha [5%, 95%] "
        "| Recovery (d) | beta_IM mean | beta_IM [5%] |"
    )
    _w("|-----|----------|------------|-----------------|"
       "--------------|-------------|-------------|")

    mc_s = output["mc_results"]
    for age in mc.ages:
        d = mc_s[f"age_{age}"]
        _w(
            f"| {age} "
            f"| {d['stable_fraction']*100:.1f} "
            f"| {d['alpha_mean']:.6f} "
            f"| [{d['alpha_q05']:.6f}, {d['alpha_q95']:.6f}] "
            f"| {d['recovery_mean']:.1f} "
            f"| {d['beta_IM_mean']:.6f} "
            f"| {d['beta_IM_q05']:.6f} |"
        )

    _w(f"\n### 1.2 Monotone alpha Ordering")
    _w(
        f"\nFraction of draws with monotonically increasing alpha "
        f"across ages: **{mc.monotone_fraction*100:.1f}%**"
    )

    # -- Entry sensitivity --
    _w("\n## 2. Entry Sensitivity Ranking (alpha, OAT)")
    _w("\n| Rank | Entry | Importance | delta | Prior sigma |")
    _w("|------|-------|-----------|-------|-------------|")
    for rank, s in enumerate(sens_alpha[:15], 1):
        _w(
            f"| {rank} | {s['coupling_id']} "
            f"| {s['importance']:.6f} "
            f"| {s['sensitivity']:.6f} "
            f"| {s['prior_std']:.4f} |"
        )

    _w("\n### 2.1 Bifurcation Margin Sensitivity (top 10)")
    _w("\n| Rank | Entry | Importance | delta |")
    _w("|------|-------|-----------|-------|")
    for rank, s in enumerate(sens_beta[:10], 1):
        _w(
            f"| {rank} | {s['coupling_id']} "
            f"| {s['importance']:.6f} "
            f"| {s['sensitivity']:.6f} |"
        )

    # -- Stress tests --
    _w("\n## 3. Stress Tests")

    _w("\n### 3.1 Concordance Analysis\n")
    _w("| Scenario | Concordance | n_agree / n_total |")
    _w("|----------|-------------|-------------------|")
    st = output["stress_tests"]
    for key in ("concordance_correct", "concordance_null", "concordance_adversarial"):
        d = st[key]
        _w(f"| {d['label']} | {d['concordance']:.3f} | {d['n_agree']}/{d['n_total']} |")

    _w(
        f"\nOrdering: correct ({st['concordance_correct']['concordance']:.3f}) "
        f"> null ({st['concordance_null']['concordance']:.3f}) "
        f"> adversarial ({st['concordance_adversarial']['concordance']:.3f}): "
        f"**{'PASS' if output['acceptance_checks']['concordance_ordering'] else 'FAIL'}**"
    )

    _w("\n### 3.2 Confidence Grade Ablation")
    ga = st["grade_ablation"]
    _w(f"\n- Entries ablated: {ga['n_ablated']} (non-informative / R6-only)")
    _w(f"- alpha_full: {ga['alpha_full']:.6f}")
    _w(f"- alpha_ablated: {ga['alpha_ablated']:.6f}")
    _w(f"- |delta_alpha| / |alpha|: {ga['delta_alpha_relative']:.4f}")
    _w(f"- Stable after ablation: {ga['stable_after_ablation']}")

    _w("\n### 3.3 Exclusion Impact")
    ei = st["exclusion_impact"]
    _w(f"\n- Excluded entries added back: {ei['n_added']} ({ei['added_entries']})")
    _w(f"- alpha_without: {ei['alpha_without']:.6f}")
    _w(f"- alpha_with: {ei['alpha_with']:.6f}")
    _w(f"- |delta_alpha| / |alpha|: {ei['delta_alpha_relative']:.4f}")
    _w(f"- Exclusion safe (<5%): **{ei['exclusion_safe']}**")
    _w(f"- Mean |delta_SWDS|: {ei['mean_abs_delta_swds']:.6f}")

    _w("\n### 3.4 Decomposition-Informed vs Uniform Prior")
    dv = st["decomposition_vs_uniform"]
    _w(f"\n- Informative entries: {dv['n_informative_entries']}")
    _w(f"- std(alpha) narrow: {dv['std_narrow']:.6f}")
    _w(f"- std(alpha) wide (2x sigma): {dv['std_wide']:.6f}")
    _w(f"- std reduction: {dv['std_reduction_pct']:.1f}%")
    _w(f"- CI90 narrow: {dv['ci90_narrow']:.6f}")
    _w(f"- CI90 wide: {dv['ci90_wide']:.6f}")
    _w(f"- CI90 reduction: {dv['ci90_reduction_pct']:.1f}%")

    # -- Acceptance summary --
    _w("\n## 4. Acceptance Criteria Summary\n")
    checks = output["acceptance_checks"]
    for key, val in checks.items():
        status = "PASS" if val else "FAIL"
        _w(f"- [{status}] **{key}**: {val}")

    _w("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
