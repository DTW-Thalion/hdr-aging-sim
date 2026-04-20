"""
Step 2 — Quality control and sample description.

Reads results/ukb_panel_long.parquet produced by step1 and writes
    results/ukb_qc_report.md
    results/ukb_qc_stats.json
Nothing individual-level leaves this step.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from hdr_core import config_hash, save_json


AXIS_COLS = {
    "I": "delta_I",
    "M": "delta_M",
    "F": "delta_F",
    "N": "delta_N",
    "C": "delta_C",
    "B": "delta_B",
    "P": "delta_P",
}


def _summary_stats(s: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": None, "sd": None, "median": None,
                "iqr_lo": None, "iqr_hi": None, "min": None, "max": None,
                "outliers_5sd": 0}
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    outliers = int(((s - mu).abs() > 5 * sd).sum()) if sd > 0 else 0
    return {
        "n": int(len(s)),
        "mean": mu,
        "sd": sd,
        "median": float(s.median()),
        "iqr_lo": float(s.quantile(0.25)),
        "iqr_hi": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
        "outliers_5sd": outliers,
    }


def _longitudinal_coverage(panel: pd.DataFrame, axes: List[str]) -> int:
    """Number of eids with complete K-axis data at ≥ 2 instances."""
    inst_complete = panel[axes].notna().all(axis=1)
    sub = panel[inst_complete][["eid", "instance"]]
    counts = sub.groupby("eid")["instance"].nunique()
    return int((counts >= 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])
    panel = pd.read_parquet(out_dir / "ukb_panel_long.parquet")

    # N by instance
    n_by_instance = panel.groupby("instance").size().to_dict()

    # Axis completeness by instance
    completeness: Dict[int, Dict[str, float]] = {}
    for inst, sub in panel.groupby("instance"):
        completeness[int(inst)] = {
            ax: round(float(sub[col].notna().mean()), 4)
            for ax, col in AXIS_COLS.items()
            if col in sub.columns
        }

    # Distributions per instance per axis
    distributions: Dict[int, Dict[str, Dict[str, float]]] = {}
    for inst, sub in panel.groupby("instance"):
        distributions[int(inst)] = {
            col: _summary_stats(sub[col])
            for col in sub.columns
            if col.startswith("delta_") or col in {
                "crp", "hba1c", "grip_max", "pulse_rate", "cystatin_c",
                "bmd", "bmi", "sbp", "dbp",
            }
        }

    # Age distribution
    age_summary = {
        int(inst): _summary_stats(sub["age"])
        for inst, sub in panel.groupby("instance")
    }
    # By decade
    panel["_age_decade"] = (panel["age"] // 10 * 10).astype("Int64")
    age_decade_counts = (
        panel.groupby(["instance", "_age_decade"])
        .size()
        .reset_index(name="n")
        .to_dict("records")
    )

    # Medication prevalence by age stratum (baseline instance)
    base_inst = min(panel["instance"].unique())
    base = panel[panel["instance"] == base_inst].copy()
    med_by_age: List[Dict[str, float]] = []
    for lo, hi in cfg["analysis"]["age_strata"]:
        mask = base["age"].between(lo, hi)
        if mask.sum() == 0:
            continue
        med_by_age.append({
            "stratum": f"{lo}-{hi}",
            "n": int(mask.sum()),
            "statin": float(base.loc[mask, "med_statin"].fillna(False).mean()),
            "antihtn": float(base.loc[mask, "med_antihtn"].fillna(False).mean()),
            "insulin": float(base.loc[mask, "med_insulin"].fillna(False).mean()),
        })

    # Longitudinal coverage per tier
    tier_axes = {
        "tier1": [AXIS_COLS[a] for a in cfg["analysis"]["tier1_axes"] if AXIS_COLS.get(a)],
        "tier2": [AXIS_COLS[a] for a in cfg["analysis"]["tier2_axes"] if AXIS_COLS.get(a)],
        "tier3": [AXIS_COLS[a] for a in cfg["analysis"]["tier3_axes"] if AXIS_COLS.get(a)],
        "tier4": [AXIS_COLS[a] for a in cfg["analysis"]["tier4_axes"] if AXIS_COLS.get(a)],
    }
    tier_coverage = {
        name: _longitudinal_coverage(panel, axes) for name, axes in tier_axes.items()
    }

    # Mortality summary
    unique_subj = panel.drop_duplicates("eid")
    n_deaths = int(unique_subj["event"].fillna(0).sum())
    median_follow = (
        float(unique_subj["time_years"].median())
        if unique_subj["time_years"].notna().any() else None
    )

    mortality_by_age: List[Dict[str, float]] = []
    for lo, hi in cfg["analysis"]["age_strata"]:
        sub = unique_subj[unique_subj["age"].between(lo, hi)]
        if len(sub) == 0:
            continue
        mortality_by_age.append({
            "stratum": f"{lo}-{hi}",
            "n": int(len(sub)),
            "deaths": int(sub["event"].fillna(0).sum()),
            "event_rate": float(sub["event"].fillna(0).mean()),
        })

    stats = {
        "script": "step2_qc.py",
        "config_hash": config_hash(cfg),
        "n_by_instance": n_by_instance,
        "completeness_by_instance": completeness,
        "distributions_by_instance": distributions,
        "age_summary": age_summary,
        "age_decade_counts": age_decade_counts,
        "medication_prevalence_baseline": med_by_age,
        "longitudinal_coverage_by_tier": tier_coverage,
        "n_deaths": n_deaths,
        "median_follow_up_years": median_follow,
        "mortality_by_age_stratum": mortality_by_age,
    }
    save_json(out_dir / "ukb_qc_stats.json", stats)

    # Markdown report
    lines: List[str] = []
    lines.append("# UK Biobank HDR — QC Report")
    lines.append("")
    lines.append(f"Config hash: `{stats['config_hash']}`")
    lines.append("")
    lines.append("## Sample size by instance")
    lines.append("")
    lines.append("| Instance | N |")
    lines.append("|---|---|")
    for inst, n in sorted(n_by_instance.items()):
        lines.append(f"| {inst} | {n:,} |")

    lines.append("")
    lines.append("## Axis completeness (fraction non-missing)")
    lines.append("")
    header = "| Axis | " + " | ".join(f"Instance {i}" for i in sorted(completeness)) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(completeness) + 1))
    for ax in AXIS_COLS:
        row = f"| {ax} | "
        row += " | ".join(
            f"{completeness[i].get(ax, 0.0):.3f}" for i in sorted(completeness)
        )
        row += " |"
        lines.append(row)

    lines.append("")
    lines.append("## Age distribution")
    lines.append("")
    lines.append("| Instance | mean | sd | median | min | max |")
    lines.append("|---|---|---|---|---|---|")
    for inst, s in age_summary.items():
        if s["n"] == 0:
            continue
        lines.append(
            f"| {inst} | {s['mean']:.1f} | {s['sd']:.1f} | {s['median']:.1f} | "
            f"{s['min']:.1f} | {s['max']:.1f} |"
        )

    lines.append("")
    lines.append("## Medication prevalence at baseline")
    lines.append("")
    lines.append("| Stratum | N | statin | antihtn | insulin |")
    lines.append("|---|---|---|---|---|")
    for row in med_by_age:
        lines.append(
            f"| {row['stratum']} | {row['n']:,} | {row['statin']:.3f} | "
            f"{row['antihtn']:.3f} | {row['insulin']:.3f} |"
        )

    lines.append("")
    lines.append("## Longitudinal coverage (N with ≥2 instances complete)")
    lines.append("")
    lines.append("| Tier | Axes | N longitudinal |")
    lines.append("|---|---|---|")
    for name in ("tier1", "tier2", "tier3", "tier4"):
        axes = ", ".join(cfg["analysis"][f"{name}_axes"])
        lines.append(f"| {name} | {axes} | {tier_coverage[name]:,} |")

    lines.append("")
    lines.append("## Mortality")
    lines.append("")
    lines.append(f"- Total deaths: **{n_deaths:,}**")
    if median_follow is not None:
        lines.append(f"- Median follow-up: **{median_follow:.1f} years**")
    lines.append("")
    lines.append("| Stratum | N | Deaths | Event rate |")
    lines.append("|---|---|---|---|")
    for row in mortality_by_age:
        lines.append(
            f"| {row['stratum']} | {row['n']:,} | {row['deaths']:,} | "
            f"{row['event_rate']:.3f} |"
        )
    lines.append("")

    (out_dir / "ukb_qc_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[step2] Wrote {out_dir / 'ukb_qc_report.md'}")
    print(f"[step2] Wrote {out_dir / 'ukb_qc_stats.json'}")

    print(
        f"[step2] Summary: N_baseline={n_by_instance.get(base_inst, 0):,}, "
        f"deaths={n_deaths:,}, tier2_long={tier_coverage['tier2']:,}"
    )


if __name__ == "__main__":
    main()
