#!/usr/bin/env python3
"""
Merge per-cohort phi extractions into results/lead_lag_phi_table.json.

This exists because the two cohorts cannot be run in the same place. ELSA runs
in-repo from data/elsa/. InCHIANTI needs the CD Share, which is not in the repo
and should not be copied into one; it is run wherever a local copy lives and
only the aggregate coefficients come back. This script is the join, and it is
the ONLY thing that writes results/lead_lag_phi_table.json -- an earlier version
of that file was assembled by hand, which is how fields appeared in it that no
script could produce.

Everything the merge adds beyond the per-cohort rows (proxy labels, the compiled
J entry, the two concordance flags, provenance) is computed here from
data/J_matrix_compiled_9x9.csv, so the file is reproducible end to end.

Usage
-----
    python scripts/extract_lead_lag_phi.py --cohort elsa -o /tmp/elsa.json
    # on the machine holding the CD Share:
    python scripts/extract_lead_lag_phi.py --cohort inchianti \
        --data-root <path> -o /tmp/inchianti.json
    python scripts/merge_lead_lag_phi.py /tmp/elsa.json /tmp/inchianti.json

Pass --check to verify the committed file against the inputs without writing.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from hdr_sim.csv_loader import load_J_csv  # noqa: E402

OUT = os.path.join(ROOT, "results", "lead_lag_phi_table.json")

PROXIES = {
    "inchianti_4axis_hr": {"I": "IL-6", "M": "HOMA-IR",
                           "N": "resting heart rate (ECG)", "F": "SPPB"},
    "inchianti_5axis_IMNFB": {"I": "IL-6", "M": "HOMA-IR",
                              "N": "cortisol/DHEAS ratio", "F": "SPPB",
                              "B": "PTH (waves 0-3)"},
    "elsa_3axis_IMF": {"I": "log CRP (+fibrinogen)",
                       "M": "HbA1c + BMI composite",
                       "F": "grip strength (+gait, chair rises)"},
}
NOMINAL = {"inchianti_4axis_hr": 3.0, "inchianti_5axis_IMNFB": 3.0,
           "elsa_3axis_IMF": 4.0}
INTERVAL_BASIS = {"inchianti_4axis_hr": "realised age difference",
                  "inchianti_5axis_IMNFB": "realised age difference",
                  "elsa_3axis_IMF": "nominal, from wave index"}

Z = 1.959964


def compiled_entry(rows, source, target):
    for r in rows:
        if r["axis_from"].strip() == source and r["axis_to"].strip() == target:
            def _v(k):
                s = r[k].strip()
                try:
                    return float(s)
                except ValueError:
                    return s
            return {"sign": r["sign"].strip(),
                    "magnitude_tier": r["magnitude_tier"].strip(),
                    "confidence_grade": r["confidence_grade"].strip(),
                    "J_healthy": _v("J_healthy"),
                    "J_pre_disease": _v("J_pre_disease"),
                    "J_disease": _v("J_disease")}
    return None


def finish(r, rows):
    m = r["model"]
    r["pair"] = f"{r['source']}->{r['target']}"
    r["source_proxy"] = PROXIES[m].get(r["source"])
    r["target_proxy"] = PROXIES[m].get(r["target"])
    r["interval_years_nominal"] = NOMINAL[m]
    r.setdefault("interval_basis", INTERVAL_BASIS[m])
    for k in ("beta", "phi_jj", "long_run_gain"):
        v, se = r.get(k), r.get("se_" + k)
        if r.get(k + "_ci95") is None:
            r[k + "_ci95"] = ([round(v - Z * se, 10), round(v + Z * se, 10)]
                              if v is not None and se is not None else None)
    cj = compiled_entry(rows, r["source"], r["target"])
    r["compiled_J"] = cj
    if cj and r.get("beta") is not None:
        cs = cj["sign"]
        r["concordant_with_compiled_sign"] = (
            ((r["beta"] > 0) if cs == "+" else (r["beta"] < 0))
            if cs in ("+", "-") else None)
        r["concordant_all_positive"] = bool(r["beta"] > 0)
    for k in ("age_coef", "r2"):
        r.pop(k, None)
    return r


def build(inputs):
    rows = load_J_csv(os.path.join(ROOT, "data", "J_matrix_compiled_9x9.csv"))
    pairs, notes = [], []
    for path in inputs:
        with open(path) as f:
            d = json.load(f)
        if d.get("note"):
            notes.append(d["note"])
        for r in d["pairs"]:
            pairs.append(finish(r, rows))
    pairs.sort(key=lambda r: (r["model"], r["source"], r["target"]))

    return {
        "schema_version": "1.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/merge_lead_lag_phi.py "
                        "(from scripts/extract_lead_lag_phi.py per cohort)",
        "task": "HDR formalism repair Phase 1, deliverable D "
                "(docs/units_convention.md)",
        "specification": (
            "Each pair is the cross-lagged fit used by scripts/elsa_lead_lag.py, "
            "scripts/inchianti_lead_lag.py and scripts/inchianti_6axis_analysis.py: "
            "d_target = beta*source_t0 + gamma*target_t0 + delta*age_t + const, "
            "where d_target = x_target(t+1) - x_target(t). Rearranged, "
            "x_target(t+1) = (1+gamma)*x_target(t) + beta*x_source(t) + ..., so "
            "phi_jj = 1 + gamma and the implied steady-state gain is "
            "beta/(1-phi_jj) = -beta/gamma."),
        "units": {
            "beta": "SD target per SD source per visit interval",
            "phi_jj": "dimensionless AR(1) coefficient per visit interval",
            "long_run_gain": "SD target per SD source at steady state",
            "*_ci95": "95% normal-approximation interval, estimate +/- 1.96 SE",
            "interval_years_*": "see interval_basis: realised age difference for "
                                "InCHIANTI, nominal wave-index interval for ELSA"},
        "provenance": {
            "ELSA": "run in-repo from data/elsa/ (UK Data Service SN 5050 extract); "
                    "nurse-visit waves 2/4/6/8",
            "InCHIANTI": "run against a local copy of the InCHIANTI CD Share; "
                         "participant-level data was not copied or exported, only "
                         "these aggregate coefficients",
            "compiled_J": "data/J_matrix_compiled_9x9.csv",
            "rounding": "InCHIANTI rows are stored to 6 significant figures, so "
                        "identities such as long_run_gain = beta/(1-phi_jj) hold "
                        "to ~1e-6 relative on those rows and exactly on ELSA rows.",
            "note": "; ".join(notes) if notes else None},
        "concordance_definitions": {
            "concordant_with_compiled_sign": (
                "beta has the sign of the compiled J entry as currently written. "
                "Protective F- and B-row entries are negative, so a positive beta "
                "counts as discordant under this definition."),
            "concordant_all_positive": (
                "beta > 0: worse source predicts worse target. The native "
                "definition once the protective couplings move to kappa and J is "
                "all-positive (docs/protective_coupling_formulation.md).")},
        "pairs": pairs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="per-cohort JSON from extract_lead_lag_phi.py")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed file instead of writing")
    a = ap.parse_args()

    payload = build(a.inputs)

    if a.check:
        with open(OUT) as f:
            cur = json.load(f)
        drift = []
        cur_by = {(p["model"], p["pair"]): p for p in cur["pairs"]}
        for p in payload["pairs"]:
            q = cur_by.get((p["model"], p["pair"]))
            if q is None:
                drift.append(f"missing {p['model']} {p['pair']}")
                continue
            for k in ("beta", "phi_jj", "long_run_gain", "n"):
                if p.get(k) is None or q.get(k) is None:
                    continue
                if abs(p[k] - q[k]) > 1e-6 * max(1.0, abs(p[k])):
                    drift.append(f"{p['model']} {p['pair']} {k}: {q[k]} -> {p[k]}")
        print("\n".join(drift) if drift
              else f"OK: {len(payload['pairs'])} pairs match the committed file")
        return 1 if drift else 0

    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {OUT}: {len(payload['pairs'])} ordered pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
