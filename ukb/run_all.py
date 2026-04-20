"""
Master runner for the HDR UK Biobank pipeline.

Runs steps 1–8 in order, auto-detecting feasibility of each axis tier based
on available data. Each step is wrapped in try/except so that downstream
steps still run on whatever data is available. Produces a markdown summary
report at results/ukb_hdr_summary.md with the headline numbers a manuscript
author needs.

Usage:
    python run_all.py [--config config.yaml] [--skip step3,step6] [--only step3]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from hdr_core import config_hash


STEPS = [
    ("step1_extract", "Extract & harmonise data"),
    ("step2_qc", "Quality control + sample description"),
    ("step3_cross_sectional", "Cross-sectional Γ̂ trajectory"),
    ("step4_longitudinal", "Change-covariance λ_max trajectory"),
    ("step5_lead_lag", "Cross-lagged lead-lag tests"),
    ("step6_mortality", "Cox mortality + SWDS-Γ"),
    ("step7_null_tests", "Permutation and random-panel nulls"),
    ("step8_circadian", "Circadian axis from accelerometry"),
]


def _run_step(name: str, config: str) -> Dict[str, object]:
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f"[run_all] Running {name}…")
    print("=" * 70)
    try:
        cmd = [sys.executable, f"{name}.py", "--config", config]
        result = subprocess.run(cmd, check=False)
        status = "ok" if result.returncode == 0 else f"rc={result.returncode}"
    except Exception as e:
        traceback.print_exc()
        status = f"exception: {e}"
    dur = time.time() - t0
    print(f"[run_all] {name}: {status} ({dur:.1f}s)")
    return {"name": name, "status": status, "duration_sec": round(dur, 1)}


def _feasible_tiers(cfg: dict, panel_summary: Optional[dict]) -> List[str]:
    if panel_summary is None:
        return ["tier1", "tier2"]
    cov = panel_summary.get("axis_completeness", {})
    feasible = []
    for tier in ("tier1", "tier2", "tier3", "tier4"):
        axes = cfg["analysis"][f"{tier}_axes"]
        ok = True
        for a in axes:
            col = f"delta_{a}"
            if cov.get(col, 0) < 1000:
                ok = False
                break
        if ok:
            feasible.append(tier)
    return feasible


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(v: object, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v != v:  # NaN
            return "—"
        return f"{v:.{digits}f}"
    return str(v)


def _summary_report(out_dir: Path, cfg: dict, step_log: List[dict]) -> None:
    lines: List[str] = []
    lines.append("# UK Biobank HDR Analysis Summary")
    lines.append("")
    lines.append(f"Config hash: `{config_hash(cfg)}`")
    lines.append("")

    # Sample
    panel_summary = _load_json(out_dir / "ukb_panel_summary.json") or {}
    qc = _load_json(out_dir / "ukb_qc_stats.json") or {}
    lines.append("## Sample")
    lines.append("")
    n_by_inst = panel_summary.get("n_by_instance", {})
    if n_by_inst:
        for inst, n in sorted(n_by_inst.items(), key=lambda x: int(x[0])):
            lines.append(f"- N instance {inst}: **{int(n):,}**")
    lines.append(f"- Deaths: **{panel_summary.get('n_deaths', '—'):,}**")
    mf = qc.get("median_follow_up_years")
    if mf is not None:
        lines.append(f"- Median follow-up: **{mf:.1f} years**")
    lc = qc.get("longitudinal_coverage_by_tier", {})
    for tier, n in lc.items():
        lines.append(f"- {tier} longitudinal N: **{int(n):,}**")
    lines.append("")

    # Per-tier key results
    for tier in ("tier1", "tier2", "tier3", "tier4"):
        xs = _load_json(out_dir / f"ukb_cross_sectional_{tier}.json")
        lg = _load_json(out_dir / f"ukb_longitudinal_{tier}.json")
        ll = _load_json(out_dir / f"ukb_lead_lag_{tier}.json")
        mt = _load_json(out_dir / f"ukb_mortality_{tier}.json")
        nt = _load_json(out_dir / f"ukb_null_tests_{tier}.json")
        if not any([xs, lg, ll, mt, nt]):
            continue
        lines.append(f"## {tier}")
        if xs and xs.get("axes"):
            lines.append(f"- Axes: `{xs['axes']}`")
        lines.append("")
        lines.append("### Cross-sectional")
        if xs and not xs.get("status"):
            tk = xs.get("trend_kendall", {})
            lines.append(
                f"- λ_max ratio: **{_fmt(xs.get('lambda_max_ratio'), 2)}×**, "
                f"Kendall τ = {_fmt(tk.get('tau'))} "
                f"(p = {_fmt(tk.get('p_value'))})"
            )
        lines.append("")
        lines.append("### Longitudinal (change-covariance)")
        if lg and not lg.get("status"):
            tk = lg.get("trend_kendall", {})
            perm = lg.get("permutation_trend", {})
            lines.append(
                f"- λ_max ratio: **{_fmt(lg.get('lambda_max_ratio'), 2)}×**, "
                f"Kendall τ = {_fmt(tk.get('tau'))} "
                f"(p = {_fmt(tk.get('p_value'))}); "
                f"permutation p = **{_fmt(perm.get('p_value'))}**"
            )
            lines.append(f"- N pairs: {_fmt(lg.get('n_pairs'))}")
        lines.append("")
        lines.append("### Lead-lag")
        if ll and not ll.get("status"):
            main = ll.get("main", {})
            lines.append(
                f"- Sign concordance: **{main.get('n_concordant')}/{main.get('n_tested_vs_J')}** "
                f"(binomial p = {_fmt(main.get('binomial_p'), 3)})"
            )
            # Highlight I→M, F→I, F→B
            pairs = main.get("pairs", [])
            for key in ("I->M", "M->I", "F->I", "F->B", "I->C", "C->N", "P->I"):
                match = next(
                    (p for p in pairs if p.get("pair") == key),
                    None,
                )
                if match:
                    q = match.get("q_fdr")
                    lines.append(
                        f"  - {key}: β = {_fmt(match.get('beta'))} "
                        f"[{_fmt(match.get('ci_lower'))}, {_fmt(match.get('ci_upper'))}], "
                        f"p = {_fmt(match.get('p_value'), 3)}, "
                        f"FDR-q = {_fmt(q, 3) if q is not None else '—'}"
                    )
        lines.append("")
        lines.append("### Mortality")
        if mt and not mt.get("status"):
            models = mt.get("models", {})
            for mname in ("M4_frailty", "M4a_frailty_bio", "M5_full"):
                if mname in models:
                    c = models[mname].get("C")
                    lines.append(f"  - {mname}: C = {_fmt(c)}")
            for dname in ("delta_C_M5_M4", "delta_C_M5_M4a", "delta_C_M4b_M4"):
                d = mt.get("delta_bootstraps", {}).get(dname, {})
                if d:
                    lines.append(
                        f"  - {dname}: {_fmt(d.get('mean'))} "
                        f"[{_fmt(d.get('ci_lower'))}, {_fmt(d.get('ci_upper'))}]"
                    )
        lines.append("")
        lines.append("### Nulls")
        if nt and not nt.get("status"):
            ap_n = nt.get("age_permutation_null", {})
            rp = nt.get("random_panel_null", {})
            lines.append(
                f"- Age-permutation: observed = {_fmt(ap_n.get('observed_ratio'), 2)}, "
                f"p = {_fmt(ap_n.get('p_value'), 3)}"
            )
            if rp:
                lines.append(
                    f"- Random-panel (k={rp.get('k')}, n={rp.get('n_panels')}): "
                    f"p = {_fmt(rp.get('p_value'), 3)}"
                )
        lines.append("")

    # Comparison table
    lines.append("## Comparison to InCHIANTI / ELSA")
    lines.append("")
    lines.append("| Metric | InCHIANTI | ELSA | UK Biobank |")
    lines.append("|---|---|---|---|")
    t2 = _load_json(out_dir / "ukb_longitudinal_tier2.json") or {}
    lead2 = _load_json(out_dir / "ukb_lead_lag_tier2.json") or {}
    mort2 = _load_json(out_dir / "ukb_mortality_tier2.json") or {}
    ukb_n = panel_summary.get("n_eids", "—")
    ukb_lmax = t2.get("lambda_max_ratio") if not t2.get("status") else None
    ukb_dc = (
        mort2.get("delta_bootstraps", {}).get("delta_C_M5_M4", {}).get("mean")
        if not mort2.get("status") else None
    )
    ukb_im_p = None
    if lead2.get("main", {}).get("pairs"):
        im = next((p for p in lead2["main"]["pairs"] if p.get("pair") == "I->M"), None)
        if im:
            ukb_im_p = im.get("p_value")

    lines.append(f"| N | 1,453 | 6,420 | {ukb_n:,} |")
    lines.append(f"| Axes | 4 | 3 | (tier2: 4) |")
    lines.append(f"| λ_max ratio (longitudinal) | 18.8× | 1.24× | "
                 f"{_fmt(ukb_lmax, 2)}× |")
    lines.append(f"| ΔC (M5−M4) | +0.014 | +0.009 | {_fmt(ukb_dc, 4)} |")
    lines.append(f"| I→M p | 0.031 | 3.2e-20 | {_fmt(ukb_im_p, 3)} |")
    lines.append("")

    # Step log
    lines.append("## Step log")
    lines.append("")
    lines.append("| Step | Status | Duration (s) |")
    lines.append("|---|---|---|")
    for s in step_log:
        lines.append(f"| {s['name']} | {s['status']} | {s['duration_sec']} |")
    lines.append("")

    (out_dir / "ukb_hdr_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[run_all] Summary → {out_dir / 'ukb_hdr_summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--skip", default="", help="comma-separated step names to skip")
    ap.add_argument("--only", default="", help="comma-separated step names to run exclusively")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    print(f"[run_all] Config: {args.config}  (hash={config_hash(cfg)})")
    print(f"[run_all] Output: {out_dir.resolve()}")
    print(f"[run_all] Skipping: {skip or '(none)'}")
    if only:
        print(f"[run_all] Only: {only}")

    step_log: List[dict] = []
    for name, desc in STEPS:
        if only and name not in only:
            continue
        if name in skip:
            continue
        step_log.append(_run_step(name, args.config))

    # Feasibility-aware notes
    panel_summary = _load_json(out_dir / "ukb_panel_summary.json")
    feasible = _feasible_tiers(cfg, panel_summary)
    print(f"[run_all] Feasible tiers: {feasible}")

    _summary_report(out_dir, cfg, step_log)

    # Console summary
    print("\n" + "=" * 70)
    print("[run_all] Pipeline complete.")
    print(f"[run_all] Summary: {out_dir / 'ukb_hdr_summary.md'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
