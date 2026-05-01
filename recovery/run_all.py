#!/usr/bin/env python3
"""
Master runner for the HDR recovery-dynamics pipeline.

Detects which dataset is configured (MIMIC vs ISARIC), runs steps 1-7 in
order, and writes a markdown summary at results/recovery_analysis_summary.md.

Each step is a separate script invoked via subprocess so that failures are
isolated and rerun-friendly. Pass `--skip-extract` if you've already done
step 1 in a previous run.

Usage:
    python run_all.py [--config config.yaml] [--skip-extract] [--from STEP]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from hdr_core import banner, load_config


STEPS = [
    ("1",  "step1_extract_mimic.py"),
    ("1b", "step1b_extract_isaric.py"),
    ("2",  "step2_define_episodes.py"),
    ("3",  "step3_fit_recovery.py"),
    ("4",  "step4_age_dependence.py"),
    ("5",  "step5_cross_axis_coupling.py"),
    ("6",  "step6_predict_outcomes.py"),
    ("7",  "step7_figures.py"),
]


def _run(script: str, config: str) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"\n>>> python {script}")
    return subprocess.call(
        [sys.executable, os.path.join(here, script), "--config", config],
        cwd=here,
    )


def _read_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def write_summary(out_dir: str, dataset_label: str) -> None:
    fit_summary = _read_json(os.path.join(out_dir, "recovery_fit_summary.json"))
    tau_age = _read_json(os.path.join(out_dir, "tau_vs_age.json"))
    coupling = _read_json(os.path.join(out_dir, "cross_axis_coupling.json"))
    outcomes = _read_json(os.path.join(out_dir, "outcome_prediction.json"))

    md = []
    md.append(f"# HDR Recovery Dynamics -- Summary ({dataset_label})\n")

    md.append("## Sample\n")
    n_total = fit_summary.get("n_fits_total")
    n_qc = fit_summary.get("n_fits_pass_qc")
    md.append(f"- Total fits: **{n_total}**, QC-passing: **{n_qc}** "
              f"({fit_summary.get('fraction_qc_pass', 0)*100:.1f}%)\n")
    md.append(f"- Best-fit model counts: {fit_summary.get('best_model_counts')}\n")

    md.append("\n## Primary result -- tau vs age\n")
    md.append("| Axis | beta (log h / yr) | p | fold/decade | tau ratio (oldest/youngest) | n |\n")
    md.append("|------|-------------------|---|-------------|----------------------------|----|\n")
    for axis, info in tau_age.get("by_axis", {}).items():
        if info.get("skipped"):
            md.append(f"| {axis} | -- | -- | -- | -- | {info.get('n')} |\n")
            continue
        reg = info.get("regression", {})
        ratio = info.get("tau_ratio_oldest_to_youngest")
        ratio_str = f"{ratio:.2f}x" if ratio else "-"
        md.append(f"| {axis} | {reg.get('beta_log_tau_per_year', float('nan')):.5f} "
                  f"| {reg.get('pvalue_age', float('nan')):.3g} "
                  f"| {reg.get('tau_fold_change_per_decade', float('nan')):.2f}x "
                  f"| {ratio_str} "
                  f"| {reg.get('n')} |\n")
    multi = tau_age.get("multivariate", {})
    if not multi.get("skipped"):
        reg = multi.get("regression", {})
        md.append(f"\nMultivariate (geometric-mean tau): beta = "
                  f"{reg.get('beta_log_tau_per_year', float('nan')):.5f}, "
                  f"p = {reg.get('pvalue_age', float('nan')):.3g}, "
                  f"fold/decade = {reg.get('tau_fold_change_per_decade', float('nan')):.2f}x.\n")

    md.append("\n## Cross-axis coupling\n")
    pt = coupling.get("peak_to_tau", {})
    md.append(f"- Peak->tau pairs concordant with J-matrix: "
              f"**{pt.get('n_concordant')}/{pt.get('n_pairs')}** "
              f"(p = {pt.get('concordance_p_one_sided', float('nan')):.4g}, one-sided binomial).\n")
    seq = coupling.get("recovery_ordering", {})
    md.append(f"- Recovery-sequence Kendall tau (mean): {seq.get('kendall_tau_mean')}; "
              f"fraction positive: {seq.get('fraction_positive')}.\n")

    md.append("\n## Outcome prediction\n")
    mm = outcomes.get("mortality_models", {})
    md.append("| Model | n | AUC | 95% CI |\n|-------|---|-----|--------|\n")
    for m in ("M1", "M2", "M3", "M4", "M5", "M6", "M7"):
        sub = mm.get(m, {})
        if sub.get("auc") is None:
            md.append(f"| {m} | {sub.get('n', '-')} | -- | -- |\n")
        else:
            ci = sub.get("auc_ci95", [None, None])
            md.append(f"| {m} | {sub.get('n')} | {sub.get('auc'):.4f} | "
                      f"[{ci[0]:.4f}, {ci[1]:.4f}] |\n")
    deltas = outcomes.get("delta_c", {})
    if deltas:
        md.append("\nKey delta C values:\n")
        for k, v in deltas.items():
            if v.get("delta") is None:
                continue
            ci = v.get("ci95", [None, None])
            md.append(f"- **{k}** = {v['delta']:+.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}], "
                      f"P(<=0) ~ {v.get('p_one_sided_gt0', float('nan')):.3f}\n")

    cox = outcomes.get("cox_30d")
    if cox and not cox.get("error"):
        md.append(f"\n30-day Cox: HR per SD log-tau-geo = {cox['hr_log_tau_geo']:.3f} "
                  f"(95% CI {cox['hr_log_tau_geo_ci95'][0]:.3f}, {cox['hr_log_tau_geo_ci95'][1]:.3f}, "
                  f"p = {cox['hr_log_tau_geo_p']:.3g}). Concordance = {cox['concordance']:.4f}.\n")

    md.append("\n## Recovery model selection\n")
    bm_frac = fit_summary.get("best_model_fraction", {})
    for m, f in bm_frac.items():
        md.append(f"- {m}: {f*100:.1f}%\n")

    out_path = os.path.join(out_dir, "recovery_analysis_summary.md")
    with open(out_path, "w") as f:
        f.write("".join(md))
    print(f"\nWrote {out_path}")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--skip-extract", action="store_true",
                   help="Skip step 1/1b (use existing extracted panel)")
    p.add_argument("--from", dest="start", default="1",
                   choices=["1", "2", "3", "4", "5", "6", "7"],
                   help="Start from this step (skipping extraction).")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    banner("HDR Recovery Dynamics -- master runner")

    use_mimic = bool(cfg.get("mimic_dir"))
    use_isaric = bool(cfg.get("isaric_file"))
    if not (use_mimic or use_isaric):
        print("ERROR: neither mimic_dir nor isaric_file is set in config.yaml.")
        return 2
    dataset_label = "MIMIC-IV" if use_mimic else "ISARIC"
    print(f"Dataset: {dataset_label}")

    # Build step list to actually run
    start_idx = next(i for i, (s, _) in enumerate(STEPS) if s == args.start)
    todo = []
    for sid, script in STEPS[start_idx:]:
        if args.skip_extract and sid in ("1", "1b"):
            continue
        if sid == "1" and not use_mimic:
            continue
        if sid == "1b" and not use_isaric:
            continue
        todo.append((sid, script))

    print(f"Running steps: {[t[0] for t in todo]}\n")

    for sid, script in todo:
        rc = _run(script, args.config)
        if rc != 0:
            print(f"!! step {sid} ({script}) returned {rc}; aborting.")
            return rc

    write_summary(out_dir, dataset_label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
