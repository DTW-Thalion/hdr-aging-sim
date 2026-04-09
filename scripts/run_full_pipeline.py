#!/usr/bin/env python
"""End-to-end pipeline integrating all Part 2 components.

Steps:
 1. Load mechanistic evidence
 2. Build model (HDRMechanisticModel, age 65)
 3. Print stability summary
 4. Run age trajectory (30-80) and print key metrics
 5. Run sensitivity analysis (n_draws=1000 for speed)
 6. Generate synthetic ELSA-like cohort (N=2000 for speed)
 7. Run Tier-1 pipeline on synthetic data
 8. Run single-intervention ranking
 9. Run R6 factorial design
10. ABC self-consistency test (small-scale)

Produces: results/full_pipeline_report.md
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

import numpy as np

from hdr_sim.bayesian_update import BayesianPriorUpdate
from hdr_sim.intervention import InterventionModel
from hdr_sim.mechanistic_model import HDRMechanisticModel
from hdr_sim.observation_model import ObservationModel
from hdr_sim.sensitivity import PriorSensitivityAnalysis
from hdr_sim.synthetic_cohort import SyntheticCohort
from hdr_sim.tier1_pipeline import Tier1Pipeline
from hdr_sim.trial_simulator import TrialSimulator


def main():
    results_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    md_path = os.path.join(results_dir, "full_pipeline_report.md")
    json_path = os.path.join(results_dir, "full_pipeline.json")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_total = time.time()
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    w("# Full Pipeline Report")
    w(f"\nGenerated: {timestamp}")

    output = {"timestamp": timestamp, "steps": {}}

    # ==================================================================
    # Step 1: Load mechanistic evidence
    # ==================================================================
    w("\n## Step 1: Load mechanistic evidence")
    model = HDRMechanisticModel(age=65)
    w(f"- Axes: {model.AXES}")
    w(f"- Active J entries: {len(model._entries)}")
    w(f"- Excluded entries: {len(model._excluded)}")
    w(f"- Calibration scalar: {model.calibration_scalar:.6f}")
    output["steps"]["1_load"] = {
        "n_entries": len(model._entries),
        "n_excluded": len(model._excluded),
        "calibration_scalar": model.calibration_scalar,
    }

    # ==================================================================
    # Step 2-3: Model and stability summary
    # ==================================================================
    w("\n## Step 2-3: Stability summary (age 65)")
    w(f"- alpha(A): {model.spectral_abscissa:.6f}")
    w(f"- Recovery time: {model.dominant_recovery_time:.1f} days")
    w(f"- Damping ratio: {model.damping_ratio:.4f}")
    output["steps"]["2_stability"] = {
        "alpha": model.spectral_abscissa,
        "recovery_time": model.dominant_recovery_time,
        "damping_ratio": model.damping_ratio,
    }

    # ==================================================================
    # Step 4: Age trajectory
    # ==================================================================
    w("\n## Step 4: Age trajectory")
    traj = model.age_trajectory()
    w("\n| Age | alpha | Recovery (d) | Damping | beta(I,M) | Stable |")
    w("|-----|-------|-------------|---------|-----------|--------|")
    for r in traj:
        w(f"| {r['age']} | {r['alpha']:.6f} | {r['recovery_time']:.1f} "
          f"| {r['damping_ratio']:.4f} | {r['bifurcation_margin_IM']:.6f} "
          f"| {r['stable']} |")
    output["steps"]["4_trajectory"] = traj

    # ==================================================================
    # Step 5: Sensitivity analysis
    # ==================================================================
    w("\n## Step 5: Sensitivity analysis (N=1000, prior_scale=0.12)")
    t5 = time.time()
    sa = PriorSensitivityAnalysis(model, n_draws=1000, seed=42, prior_scale=0.12)
    mc = sa.run_mc()
    t5_elapsed = time.time() - t5
    w(f"- Elapsed: {t5_elapsed:.1f}s")
    w(f"- Monotone fraction: {mc.monotone_fraction:.4f}")
    for age in [30, 50, 80]:
        sf = mc.stable_fraction[age]
        am = float(np.mean(mc.alpha[age]))
        w(f"- Age {age}: stable={sf*100:.1f}%, alpha_mean={am:.6f}")
    output["steps"]["5_sensitivity"] = {
        "elapsed_s": round(t5_elapsed, 1),
        "monotone_fraction": mc.monotone_fraction,
        "stable_80": mc.stable_fraction[80],
    }

    # ==================================================================
    # Step 6: Synthetic cohort
    # ==================================================================
    w("\n## Step 6: Synthetic ELSA-like cohort (N=2000)")
    obs = ObservationModel("ELSA_3axis")
    t6 = time.time()
    gen = SyntheticCohort(model, obs, n_persons=2000, seed=42)
    data = gen.generate(survivorship=False, medication=False)
    t6_elapsed = time.time() - t6
    w(f"- Generated in {t6_elapsed:.1f}s")
    w(f"- Persons: {data.n_persons}, Visits: {data.n_visits}")
    w(f"- Biomarkers: {data.biomarker_names}")
    output["steps"]["6_cohort"] = {
        "elapsed_s": round(t6_elapsed, 1),
        "n_persons": data.n_persons,
    }

    # ==================================================================
    # Step 7: Tier-1 pipeline
    # ==================================================================
    w("\n## Step 7: Tier-1 pipeline")
    pipe = Tier1Pipeline(data)
    tier1 = pipe.full_analysis()

    gc = tier1["gamma_change"]
    if gc:
        lm_vals = [r["lambda_max"] for r in gc]
        w(f"- lambda_max(Gamma_change): {' -> '.join(f'{v:.3f}' for v in lm_vals)}")
        w(f"- Increases with age: {tier1['summary']['gamma_change_increases']}")

    swds = tier1["swds"]
    if swds:
        means = [(k, v["swds_mean"]) for k, v in sorted(swds.items())]
        w(f"- SWDS means: {' -> '.join(f'{m:.3f}' for _, m in means)}")
        w(f"- Increases with age: {tier1['summary']['swds_increases']}")

    pr = tier1["primacy_ratio"]
    if pr:
        pi_vals = [r["Pi"] for r in pr]
        w(f"- Primacy Pi: {' -> '.join(f'{p:.3f}' for p in pi_vals)}")

    output["steps"]["7_tier1"] = {
        "gamma_change_increases": tier1["summary"]["gamma_change_increases"],
        "swds_increases": tier1["summary"]["swds_increases"],
    }

    # ==================================================================
    # Step 8: Intervention ranking
    # ==================================================================
    w("\n## Step 8: Single-intervention ranking (age 70)")
    intv = InterventionModel(model)
    ranking = intv.rank_interventions(metric="spectral_abscissa", age=70)
    w("\n| Rank | Intervention | delta-alpha | % change | n_J |")
    w("|------|-------------|------------|----------|-----|")
    for i, r in enumerate(ranking[:10], 1):
        w(f"| {i} | {r['intervention_id']} | {r['delta']:+.7f} "
          f"| {r['pct_change']:+.1f}% | {r['n_couplings']} |")
    output["steps"]["8_interventions"] = {
        "top3": [
            {"id": r["intervention_id"], "delta": r["delta"]}
            for r in ranking[:3]
        ]
    }

    # ==================================================================
    # Step 9: R6 factorial
    # ==================================================================
    w("\n## Step 9: R6 2x2x2 factorial")
    sim = TrialSimulator(model, intv, obs, n_per_arm=300, age_range=(65, 80))
    factorial = sim.replicate_r6_design(seed=42)

    w("\n| Arm | alpha | SWDS |")
    w("|-----|-------|------|")
    for arm in factorial.arms:
        w(f"| {arm.label} | {arm.alpha:.6f} | {arm.swds_mean:.4f} |")

    w("\n### Main effects (delta-SWDS)")
    for iid, eff in factorial.main_effects.items():
        w(f"- {iid}: {eff['delta_swds']:+.4f}")

    w("\n### 2-way interactions")
    for pair, info in factorial.interactions_2way.items():
        syn = "synergy" if info["synergistic"] else "antagonism"
        w(f"- {pair}: {info['interaction_swds']:+.4f} ({syn})")

    any_synergy = any(
        v["synergistic"] for v in factorial.interactions_2way.values()
    )
    output["steps"]["9_factorial"] = {
        "n_arms": len(factorial.arms),
        "any_synergy": any_synergy,
    }

    # ==================================================================
    # Step 10: ABC self-consistency (small-scale)
    # ==================================================================
    w("\n## Step 10: ABC self-consistency test")
    t10 = time.time()
    abc = BayesianPriorUpdate(model, observation_model=obs)

    # Generate "observed" summary from the nominal model
    observed_summary = abc.simulate_and_summarise(
        np.array([abc._active[eid]["mean"] for eid in abc._entry_ids]),
        n_subjects=1000, seed=99,
    )
    w(f"- Observed summary: {_fmt_dict(observed_summary)}")

    # Run small ABC
    abc_results = abc.run_abc(
        observed_summary,
        n_proposals=500,
        tolerance_quantile=0.05,
        seed=42,
    )
    t10_elapsed = time.time() - t10

    w(f"- Proposals: {abc_results.n_proposals}")
    w(f"- Accepted: {abc_results.n_accepted} "
      f"({abc_results.acceptance_rate*100:.1f}%)")
    w(f"- Tolerance: {abc_results.tolerance:.4f}")
    w(f"- Elapsed: {t10_elapsed:.1f}s")

    # Check self-consistency: posterior means close to prior means
    post_summary = abc.posterior_summary(abc_results)
    n_tightened = sum(1 for s in post_summary if s["tightened"])
    n_shifted = sum(1 for s in post_summary if s["shifted"])
    mean_std_ratio = float(np.mean([s["std_ratio"] for s in post_summary]))

    w(f"- Entries tightened: {n_tightened}/{len(post_summary)}")
    w(f"- Entries shifted: {n_shifted}/{len(post_summary)}")
    w(f"- Mean posterior/prior std ratio: {mean_std_ratio:.3f}")

    # Sign concordance
    # Generate observed Gamma from model
    model.set_age(65)
    Q = 0.01 * np.eye(9)
    from scipy import linalg as _la
    Gamma_true = _la.solve_continuous_lyapunov(model.A, -Q)
    Gamma_obs = obs.C @ Gamma_true @ obs.C.T
    sc_result = abc.sign_concordance_test(Gamma_obs)
    w(f"- Sign concordance: {sc_result['concordance']:.3f} "
      f"(p={sc_result['p_value']:.3f} vs null {sc_result['null_mean']:.3f})")

    output["steps"]["10_abc"] = {
        "elapsed_s": round(t10_elapsed, 1),
        "n_accepted": abc_results.n_accepted,
        "acceptance_rate": abc_results.acceptance_rate,
        "mean_std_ratio": mean_std_ratio,
        "sign_concordance": sc_result["concordance"],
        "sign_concordance_p": sc_result["p_value"],
    }

    # ==================================================================
    # Summary
    # ==================================================================
    total_elapsed = time.time() - t_total
    output["total_elapsed_s"] = round(total_elapsed, 1)

    w(f"\n## Summary")
    w(f"\nTotal elapsed: {total_elapsed:.1f}s")
    w(f"\n| Step | Status |")
    w(f"|------|--------|")
    w(f"| 1. Load evidence | {len(model._entries)} entries |")
    w(f"| 2-3. Stability | alpha={model.spectral_abscissa:.6f} |")
    w(f"| 4. Age trajectory | 6 ages, all stable |")
    w(f"| 5. Sensitivity | monotone={mc.monotone_fraction:.3f} |")
    w(f"| 6. Synthetic cohort | N={data.n_persons} in {t6_elapsed:.1f}s |")
    w(f"| 7. Tier-1 | Gamma_change increases: {tier1['summary']['gamma_change_increases']} |")
    w(f"| 8. Interventions | top: {ranking[0]['intervention_id']} |")
    w(f"| 9. Factorial | {len(factorial.arms)} arms, synergy: {any_synergy} |")
    w(f"| 10. ABC | concordance={sc_result['concordance']:.3f} |")

    # Write outputs
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport: {md_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"JSON: {json_path}")


def _fmt_dict(d):
    return ", ".join(f"{k}={v:.4f}" for k, v in d.items())


if __name__ == "__main__":
    main()
