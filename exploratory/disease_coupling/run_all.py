"""
Run all five coupling-modification tests in sequence and assemble a
summary.json that the README can reference.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_a_statins
import test_b_metformin
import test_c_exercise
import test_d_nsaid
import test_e_bone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _line(label, value):
    return f"  {label:<42} {value}"


def main():
    print("\n\n" + "#" * 78)
    print("# Running Test A (Statins)")
    print("#" * 78)
    a = test_a_statins.run()

    print("\n\n" + "#" * 78)
    print("# Running Test B (Antidiabetics)")
    print("#" * 78)
    b = test_b_metformin.run()

    print("\n\n" + "#" * 78)
    print("# Running Test C (SPPB tertile)")
    print("#" * 78)
    c = test_c_exercise.run()

    print("\n\n" + "#" * 78)
    print("# Running Test D (NSAIDs)")
    print("#" * 78)
    d = test_d_nsaid.run()

    print("\n\n" + "#" * 78)
    print("# Running Test E (Bone axis)")
    print("#" * 78)
    e = test_e_bone.run()

    summary = {
        "test_a_statins": {
            "prediction": "Statin users show weaker β_{I→M}",
            "beta_nostatin": a["unadjusted"]["groups"]["0.0"]["beta"],
            "beta_statin": a["unadjusted"]["groups"]["1.0"]["beta"],
            "interaction_p_unadj": a["unadjusted"]["interaction"]["p"],
            "interaction_p_matched": a["age_matched"].get("interaction", {}).get("p"),
            "bonferroni_p": a["bonferroni_interaction_p"],
            "supported": False,  # filled below
        },
        "test_b_metformin": {
            "prediction": "M→I weakens, I→M unchanged (D-vs-J)",
            "I_to_M_interaction_p": b["I_to_M"]["unadjusted"]["interaction"]["p"],
            "M_to_I_interaction_p": b["M_to_I"]["unadjusted"]["interaction"]["p"],
            "discrimination_supported": b["d_vs_j_discrimination"]["discrimination_supported"],
        },
        "test_c_exercise": {
            "prediction": "High SPPB → weaker pathological coupling, stronger F→I protection",
            "n_pairs_weaker_high_sppb": c["summary"]["n_pairs_weaker_in_high_sppb"],
            "n_pairs_total": c["summary"]["n_pairs_total"],
            "F_to_I_stronger_protective": c["summary"]["F_to_I_stronger_protective_in_high_sppb"],
            "trend_bonferroni": c["trend_bonferroni"],
        },
        "test_d_nsaid": {
            "prediction": "NSAIDs attenuate I→M, I→N, I→F",
            "interaction_p_raw": d["interaction_p_raw"],
            "interaction_p_bonf": d["interaction_p_bonf"],
            "n_attenuated_of_3": d["summary"]["n_attenuated_of_3"],
        },
        "test_e_bone": {
            "prediction": "F→B protective (β>0 w/ sign-flipped ΔF), I→B inflammatory (β>0)",
            "F_to_B_beta": e["F_to_B_overall"].get("beta"),
            "F_to_B_p": e["F_to_B_overall"].get("p"),
            "I_to_B_beta": e["I_to_B_overall"].get("beta"),
            "I_to_B_p": e["I_to_B_overall"].get("p"),
            "bonferroni_p_primary": e["bonferroni_primary"],
            "bisphos_skipped_due_to_N": (
                e["bisphosphonate"]["F_to_B"].get("skipped") is not None
                or e["bisphosphonate"]["F_to_B"].get("n_bisphos_pairs", 0) < 30
            ),
        },
    }

    # Set `supported` on test A: interaction significant AND β_statin < β_nostatin
    sa = summary["test_a_statins"]
    sa["supported"] = bool(
        sa["interaction_p_unadj"] < 0.05
        and abs(sa["beta_statin"]) < abs(sa["beta_nostatin"])
    )

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Formatted summary to stdout
    print("\n\n" + "=" * 78)
    print("EXPLORATORY COUPLING MODIFICATION TESTS — SUMMARY")
    print("=" * 78)
    sa = summary["test_a_statins"]
    print(f"Test A (Statins → I→M):")
    print(_line("β_nostatin =", f"{sa['beta_nostatin']:+.4f}"))
    print(_line("β_statin =", f"{sa['beta_statin']:+.4f}"))
    print(_line("interaction p (unadj / matched) =",
                f"{sa['interaction_p_unadj']:.3f} / "
                f"{sa['interaction_p_matched']:.3f}"))
    print(_line("supported:", sa["supported"]))

    sb = summary["test_b_metformin"]
    print(f"\nTest B (Antidiabetics D-vs-J):")
    print(_line("I→M interaction p =", f"{sb['I_to_M_interaction_p']:.3f}"))
    print(_line("M→I interaction p =", f"{sb['M_to_I_interaction_p']:.3f}"))
    print(_line("D-vs-J discrimination supported:",
                sb["discrimination_supported"]))

    sc = summary["test_c_exercise"]
    print(f"\nTest C (SPPB tertile):")
    print(_line("N pairs weaker in high-SPPB:",
                f"{sc['n_pairs_weaker_high_sppb']}/{sc['n_pairs_total']}"))
    print(_line("F→I stronger protective in high-SPPB:",
                sc["F_to_I_stronger_protective"]))
    print(_line("N pairs with Bonferroni trend p<0.05:",
                sum(1 for p in sc["trend_bonferroni"].values() if p < 0.05)))

    sd = summary["test_d_nsaid"]
    print(f"\nTest D (NSAIDs → I→ row):")
    print(_line("N attenuated of 3:", f"{sd['n_attenuated_of_3']}/3"))
    print(_line("Bonferroni p (I→M, I→N, I→F):",
                [f"{p:.3f}" for p in sd["interaction_p_bonf"]]))

    se = summary["test_e_bone"]
    print(f"\nTest E (Bone coupling):")
    print(_line("F→B β, p:",
                f"{se['F_to_B_beta']:+.4f}, p={se['F_to_B_p']:.3f}"))
    print(_line("I→B β, p:",
                f"{se['I_to_B_beta']:+.4f}, p={se['I_to_B_p']:.3f}"))
    print(_line("Bisphosphonate skipped (N<30):",
                se["bisphos_skipped_due_to_N"]))
    print("=" * 78)


if __name__ == "__main__":
    main()
