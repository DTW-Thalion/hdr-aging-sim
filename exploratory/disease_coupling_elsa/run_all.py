"""
Run tests A, B, C, D for the ELSA cohort and assemble summary.json.
Test E (bone/PTH) is not applicable to ELSA (no PTH biomarker).
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_a_statins
import test_b_metformin
import test_c_exercise
import test_d_nsaid

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def main():
    print("\n# Running ELSA Test A (statins)"); a = test_a_statins.run()
    print("\n# Running ELSA Test B (antidiabetics)"); b = test_b_metformin.run()
    print("\n# Running ELSA Test C (grip tertile)"); c = test_c_exercise.run()
    print("\n# Running ELSA Test D (NSAIDs proxies)"); d = test_d_nsaid.run()

    summary = {
        "cohort": "ELSA",
        "waves": [2, 4, 6, 8],
        "test_a_statins": {
            "statin_flag": a["statin_flag"],
            "beta_nostatin": a["unadjusted"]["groups"]["0.0"].get("beta"),
            "beta_statin": a["unadjusted"]["groups"]["1.0"].get("beta"),
            "interaction_p_unadj": a["unadjusted"]["interaction"]["p"],
            "interaction_p_matched": a["age_matched"].get("interaction", {}).get("p"),
            "bonferroni_p": a["bonferroni_interaction_p"],
        },
        "test_b_metformin": {
            "I_to_M_interaction_p": b["I_to_M"]["unadjusted"]["interaction"]["p"],
            "M_to_I_interaction_p": b["M_to_I"]["unadjusted"]["interaction"]["p"],
            "discrimination_supported": b["d_vs_j_discrimination"]["discrimination_supported"],
        },
        "test_c_exercise": {
            "n_pairs_weaker_high_grip": c["summary"]["n_pairs_weaker_in_high_grip"],
            "n_pairs_total": c["summary"]["n_pairs_total"],
            "F_to_I_stronger_protective": c["summary"]["F_to_I_stronger_protective_in_high_grip"],
            "trend_bonferroni": c["trend_bonferroni"],
        },
        "test_d_nsaid": {
            proxy: {
                "interaction_p_raw": pr["interaction_p_raw"],
                "interaction_p_bonf": pr["interaction_p_bonf"],
            }
            for proxy, pr in d["proxies"].items()
        },
        "test_e_bone": {"skipped": "PTH not measured in ELSA"},
    }

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n\n" + "=" * 78)
    print("ELSA COUPLING MODIFICATION TESTS — SUMMARY")
    print("=" * 78)
    sa = summary["test_a_statins"]
    print(f"Test A (Statins → I→M):")
    print(f"  β_nostatin = {sa['beta_nostatin']:+.4f}  β_statin = {sa['beta_statin']:+.4f}")
    print(f"  interaction p (unadj/matched) = {sa['interaction_p_unadj']:.3f} / "
          f"{sa['interaction_p_matched']}")
    print(f"  Bonferroni: {sa['bonferroni_p']}")

    sb = summary["test_b_metformin"]
    print(f"\nTest B (Antidiabetics D-vs-J):")
    print(f"  I→M interaction p = {sb['I_to_M_interaction_p']:.3f}")
    print(f"  M→I interaction p = {sb['M_to_I_interaction_p']:.3f}")
    print(f"  D-vs-J supported: {sb['discrimination_supported']}")

    sc = summary["test_c_exercise"]
    print(f"\nTest C (Grip tertile):")
    print(f"  N pairs weaker in high-grip: {sc['n_pairs_weaker_high_grip']}/{sc['n_pairs_total']}")
    print(f"  F→I stronger-protective in high-grip: {sc['F_to_I_stronger_protective']}")

    sd = summary["test_d_nsaid"]
    print(f"\nTest D (NSAIDs proxies):")
    for proxy, pr in sd.items():
        print(f"  {proxy}: Bonferroni interaction p = {pr['interaction_p_bonf']}")

    print("\nTest E: skipped — no PTH/bone biomarkers in ELSA.")
    print("=" * 78)


if __name__ == "__main__":
    main()
