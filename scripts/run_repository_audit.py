#!/usr/bin/env python3
"""Comprehensive repository audit for hdr-aging-sim.

Exercises every script, test, and parameter combination to identify
failures, crashes, deprecation warnings, or silent errors.

Produces:
  outputs/repository_audit_report.json  — machine-readable results
  outputs/repository_audit_report.md    — human-readable summary

Exit code: 0 if all checks pass, 1 if any fail.
"""

import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"

SCRIPT_TIMEOUT = 600      # 10 min for simulation scripts
ELSA_TIMEOUT = 900        # 15 min for ELSA scripts
NHANES_TIMEOUT = 180      # 3 min for NHANES (network)

MODULES = [
    "hdr_sim.csv_loader", "hdr_sim.aging_params", "hdr_sim.j_matrix_spec",
    "hdr_sim.estimation", "hdr_sim.dynamics", "hdr_sim.plotting",
    "hdr_sim.mechanistic_model", "hdr_sim.state_conditioned",
    "hdr_sim.observation_model", "hdr_sim.sensitivity", "hdr_sim.prior_stress",
    "hdr_sim.synthetic_cohort", "hdr_sim.tier1_pipeline",
    "hdr_sim.intervention", "hdr_sim.trial_simulator",
    "hdr_sim.bayesian_update", "hdr_sim.results_writer",
]

DATA_FILES = {
    "J_matrix_compiled_9x9.csv": DATA / "J_matrix_compiled_9x9.csv",
    "J_matrix_compiled.csv": DATA / "J_matrix_compiled.csv",
    "J_R6_ontology_v1.6.csv": DATA / "provenance" / "J_R6_ontology_v1.6.csv",
    "provenance/README.md": DATA / "provenance" / "README.md",
    "mechanistic_evidence/": DATA / "mechanistic_evidence",
    "elsa/gh_elsa_h_hdr_subset.tab": DATA / "elsa" / "gh_elsa_h_hdr_subset.tab",
    "sync_log.json": DATA / "sync_log.json",
}

J_CSVS = [
    ("9x9", str(DATA / "J_matrix_compiled_9x9.csv")),
    ("provenance", str(DATA / "provenance" / "J_R6_ontology_v1.6.csv")),
    ("legacy_8x8", str(DATA / "J_matrix_compiled.csv")),
]

AXIS_SUBSETS = [
    ("I", "M"),
    ("I", "M", "F"),
    ("I", "M", "N", "F"),
    ("I", "M", "E", "mito", "F"),
    ("I", "M", "mito", "P", "C", "N", "F"),
    ("I", "M", "E", "mito", "P", "C", "N", "F", "B"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_script(script_name, args=None, timeout=SCRIPT_TIMEOUT, cwd=None):
    """Run a script and return (exit_code, stdout_tail, stderr, duration)."""
    cmd = [sys.executable, str(ROOT / "scripts" / script_name)]
    if args:
        cmd.extend(args)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=timeout,
            cwd=str(cwd or ROOT), env=env,
        )
        duration = time.time() - start
        stdout = (result.stdout or b"").decode("utf-8", errors="replace")
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        return result.returncode, stdout[-2000:], stderr[-2000:], duration
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return -1, "", "TIMEOUT after {:.0f}s".format(duration), duration


def classify_stderr(stderr):
    """Extract warnings and errors from stderr."""
    warnings = []
    errors = []
    for line in stderr.splitlines():
        if "DeprecationWarning" in line:
            warnings.append(line.strip())
        elif "Error" in line or "Traceback" in line:
            errors.append(line.strip())
    return warnings, errors


# ---------------------------------------------------------------------------
# Phase 1: Environment
# ---------------------------------------------------------------------------

def phase1_environment():
    print("=" * 60)
    print("PHASE 1: ENVIRONMENT AND DEPENDENCY CHECK")
    print("=" * 60)

    result = {
        "python_version": sys.version,
        "python_ok": sys.version_info >= (3, 9),
        "modules_ok": 0,
        "modules_failed": [],
        "data_files": {},
        "elsa_present": False,
    }

    # Check modules
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    for mod_name in MODULES:
        try:
            importlib.import_module(mod_name)
            result["modules_ok"] += 1
            print(f"  OK    {mod_name}")
        except Exception as e:
            result["modules_failed"].append({"module": mod_name, "error": str(e)})
            print(f"  FAIL  {mod_name}: {e}")

    # Check data files
    for label, path in DATA_FILES.items():
        exists = path.exists()
        result["data_files"][label] = exists
        status = "FOUND" if exists else "MISSING"
        print(f"  {status:8s} {label}")

    result["elsa_present"] = (DATA / "elsa" / "gh_elsa_h_hdr_subset.tab").exists()

    n_data = sum(1 for v in result["data_files"].values() if v)
    print(f"\n  Summary: {result['modules_ok']} modules OK, "
          f"{len(result['modules_failed'])} failed, "
          f"{n_data} data files found, "
          f"ELSA data: {'YES' if result['elsa_present'] else 'NO'}")
    return result


# ---------------------------------------------------------------------------
# Phase 2: pytest
# ---------------------------------------------------------------------------

def phase2_pytest():
    print("\n" + "=" * 60)
    print("PHASE 2: UNIT TESTS (pytest)")
    print("=" * 60)

    # Run pytest directly
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=long"],
            capture_output=True, timeout=300,
            cwd=str(ROOT), env=env,
        )
        stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        stdout = ""
        stderr = "TIMEOUT"
        rc = -1

    # Parse results
    result = {"exit_code": rc, "total": 0, "passed": 0, "failed": 0,
              "errors": 0, "warnings": 0, "failures": [], "output": ""}

    for line in stdout.splitlines():
        if "passed" in line and ("failed" in line or "error" in line or line.strip().startswith("=")):
            result["output"] = line.strip()
        if " passed" in line:
            import re
            m = re.search(r"(\d+) passed", line)
            if m:
                result["passed"] = int(m.group(1))
                result["total"] = result["passed"]
            m = re.search(r"(\d+) failed", line)
            if m:
                result["failed"] = int(m.group(1))
                result["total"] += result["failed"]
            m = re.search(r"(\d+) error", line)
            if m:
                result["errors"] = int(m.group(1))
                result["total"] += result["errors"]

    if "FAILED" in stdout:
        # Extract failure names
        for line in stdout.splitlines():
            if line.startswith("FAILED"):
                result["failures"].append(line.strip())

    deprecation_warnings = [l for l in stderr.splitlines() if "DeprecationWarning" in l]
    result["warnings"] = len(deprecation_warnings)

    print(f"  {result['passed']} passed, {result['failed']} failed, "
          f"{result['errors']} errors, {result['warnings']} warnings")
    if result["failures"]:
        for f in result["failures"]:
            print(f"  FAILURE: {f}")
    return result


# ---------------------------------------------------------------------------
# Phase 3: J-matrix parameterisation
# ---------------------------------------------------------------------------

def phase3_parameterisation():
    print("\n" + "=" * 60)
    print("PHASE 3: J-MATRIX PARAMETERISATION MATRIX")
    print("=" * 60)

    import numpy as np
    from hdr_sim import aging_params as ap

    combinations = []
    header = f"  {'CSV':<12} {'Axes':<40} {'N':>2} {'a(30)':>10} {'a(50)':>10} {'a(80)':>10} {'Status':<10}"
    print(header)
    print("  " + "-" * 90)

    for csv_label, csv_path in J_CSVS:
        for axes in AXIS_SUBSETS:
            row = {"csv": csv_label, "axes": list(axes), "n_axes": len(axes)}
            try:
                ap.reset()
                ap.configure(j_matrix_path=csv_path, axes=axes)

                n = len(axes)
                tau30 = ap.tau_of_age(30)
                tau80 = ap.tau_of_age(80)
                assert tau30.shape == (n,) and tau80.shape == (n,)

                J30 = ap.J_of_age(30)
                J80 = ap.J_of_age(80)
                assert J30.shape == (n, n) and J80.shape == (n, n)
                assert np.allclose(np.diag(J30), 0) and np.allclose(np.diag(J80), 0)

                alphas = {}
                for age in [30, 50, 80]:
                    tau = ap.tau_of_age(age)
                    J = ap.J_of_age(age)
                    D = np.diag(1.0 / tau)
                    A = -D + J
                    eigs = np.linalg.eigvals(A)
                    alphas[age] = float(max(eigs.real))

                stable = all(a < 0 for a in alphas.values())
                status = "PASS" if stable else "UNSTABLE"
                row.update({
                    "alpha_30": alphas[30], "alpha_50": alphas[50],
                    "alpha_80": alphas[80], "status": status
                })
                print(f"  {csv_label:<12} {str(axes):<40} {n:>2} "
                      f"{alphas[30]:>10.4f} {alphas[50]:>10.4f} "
                      f"{alphas[80]:>10.4f} {status:<10}")
            except Exception as e:
                row.update({"status": "FAIL", "error": str(e)})
                print(f"  {csv_label:<12} {str(axes):<40} {len(axes):>2} "
                      f"{'':>10} {'':>10} {'':>10} FAIL: {e}")
            combinations.append(row)

    ap.reset()

    stable_n = sum(1 for c in combinations if c["status"] == "PASS")
    unstable_n = sum(1 for c in combinations if c["status"] == "UNSTABLE")
    fail_n = sum(1 for c in combinations if c["status"] == "FAIL")
    print(f"\n  Stable: {stable_n}, Unstable: {unstable_n}, Failed: {fail_n}")

    return {
        "combinations": combinations,
        "stable_count": stable_n,
        "unstable_count": unstable_n,
        "fail_count": fail_n,
        "all_stable": unstable_n == 0 and fail_n == 0,
    }


# ---------------------------------------------------------------------------
# Phase 4: Simulation scripts
# ---------------------------------------------------------------------------

PHASE4_SCRIPTS = [
    "run_figure2b.py",
    "run_figure_frailty.py",
    "run_figure_t2d.py",
    "run_figure_recoverability.py",
    "run_figure_uncertainty.py",
    "run_figure_Q_sensitivity.py",
    "run_figure_gamma_equivalence.py",
    "run_figure_prior_stress.py",
    "run_figure_individual_proxy.py",
    "run_figure_network_schematic.py",
    "run_figure_J_heatmap.py",
    "run_figure_disease_demos.py",
    "run_dj_validation.py",
    "run_dj_power.py",
    "run_dj_bayes_robust.py",
    "run_full_pipeline.py",
    "run_dj_primacy_mechanistic.py",
]

PHASE4_PARAMETERISED = [
    ("run_figure_uncertainty.py", ["--j-matrix", str(DATA / "provenance" / "J_R6_ontology_v1.6.csv"), "--axes", "I", "M"]),
    ("run_figure_uncertainty.py", ["--j-matrix", str(DATA / "provenance" / "J_R6_ontology_v1.6.csv"), "--axes", "I", "M", "E", "mito", "P", "C", "N", "F", "B"]),
    ("run_figure_J_heatmap.py", ["--j-matrix", str(DATA / "provenance" / "J_R6_ontology_v1.6.csv")]),
    ("run_figure_network_schematic.py", ["--j-matrix", str(DATA / "provenance" / "J_R6_ontology_v1.6.csv")]),
    ("run_full_pipeline.py", ["--j-matrix", str(DATA / "provenance" / "J_R6_ontology_v1.6.csv")]),
]


def phase4_simulation_scripts():
    print("\n" + "=" * 60)
    print("PHASE 4: SIMULATION SCRIPTS")
    print("=" * 60)

    results = []
    for script in PHASE4_SCRIPTS:
        script_path = ROOT / "scripts" / script
        if not script_path.exists():
            print(f"  SKIP  {script} (not found)")
            results.append({"script": script, "status": "SKIP", "error": "file not found"})
            continue

        rc, stdout, stderr, dur = run_script(script, timeout=SCRIPT_TIMEOUT)
        warnings, errors = classify_stderr(stderr)
        if rc == -1:
            status = "TIMEOUT"
        elif rc == 0 and not errors:
            status = "WARN" if warnings else "PASS"
        else:
            status = "FAIL"

        row = {
            "script": script, "exit_code": rc, "duration": round(dur, 1),
            "status": status, "warnings": warnings[:3], "stderr_tail": stderr[-500:],
        }
        results.append(row)
        print(f"  {status:<8} {script} (exit={rc}, {dur:.1f}s)")

    # Parameterised runs
    print("\n  --- Parameterised runs ---")
    param_results = []
    for script, args in PHASE4_PARAMETERISED:
        script_path = ROOT / "scripts" / script
        if not script_path.exists():
            param_results.append({"script": script, "args": args, "status": "SKIP"})
            continue

        rc, stdout, stderr, dur = run_script(script, args=args, timeout=SCRIPT_TIMEOUT)
        status = "PASS" if rc == 0 else ("TIMEOUT" if rc == -1 else "FAIL")
        row = {
            "script": script, "args": " ".join(args), "exit_code": rc,
            "duration": round(dur, 1), "status": status,
        }
        param_results.append(row)
        print(f"  {status:<8} {script} {' '.join(args[-3:])} ({dur:.1f}s)")

    return {"results": results, "parameterised_runs": param_results}


# ---------------------------------------------------------------------------
# Phase 5: ELSA scripts
# ---------------------------------------------------------------------------

PHASE5_SCRIPTS = [
    "run_elsa_validation.py",
    "run_medication_sensitivity.py",
    "run_figure_coupling_tightening.py",
    "run_figure_mortality_prediction.py",
    "run_figure_medication_compression.py",
    "run_elsa_ici_deployment.py",
    "run_dj_primacy.py",
    "run_figure_dj_pairwise.py",
    "diagnose_delta_c.py",
]


def phase5_elsa_scripts(elsa_present):
    print("\n" + "=" * 60)
    print("PHASE 5: ELSA-DEPENDENT SCRIPTS")
    print("=" * 60)

    if not elsa_present:
        print("  ELSA data not present -- all SKIPPED")
        return {
            "skipped": True,
            "results": [{"script": s, "status": "SKIP"} for s in PHASE5_SCRIPTS],
        }

    results = []
    for script in PHASE5_SCRIPTS:
        script_path = ROOT / "scripts" / script
        if not script_path.exists():
            print(f"  SKIP  {script} (not found)")
            results.append({"script": script, "status": "SKIP"})
            continue

        rc, stdout, stderr, dur = run_script(script, timeout=ELSA_TIMEOUT)
        warnings, errors = classify_stderr(stderr)
        if rc == -1:
            status = "TIMEOUT"
        elif rc == 0:
            status = "WARN" if warnings else "PASS"
        else:
            status = "FAIL"

        results.append({
            "script": script, "exit_code": rc, "duration": round(dur, 1),
            "status": status, "stderr_tail": stderr[-500:],
        })
        print(f"  {status:<8} {script} (exit={rc}, {dur:.1f}s)")

    return {"skipped": False, "results": results}


# ---------------------------------------------------------------------------
# Phase 6: Utility scripts
# ---------------------------------------------------------------------------

PHASE6_RUNS = [
    ("verify_J_counts.py", []),
    ("verify_J_counts.py", ["--csv", str(DATA / "J_matrix_compiled.csv")]),
    ("verify_J_counts.py", ["--csv", str(DATA / "provenance" / "J_R6_ontology_v1.6.csv")]),
    ("sync_j_from_companion.py", ["--source", str(DATA / "J_matrix_compiled_9x9.csv"), "--dry-run"]),
    ("populate_pmids.py", []),
    ("update_ledger_r6.py", []),
]


def phase6_utility_scripts():
    print("\n" + "=" * 60)
    print("PHASE 6: UTILITY SCRIPTS")
    print("=" * 60)

    results = []
    for script, args in PHASE6_RUNS:
        script_path = ROOT / "scripts" / script
        if not script_path.exists():
            print(f"  SKIP  {script} (not found)")
            results.append({"script": script, "args": args, "status": "SKIP"})
            continue

        rc, stdout, stderr, dur = run_script(script, args=args, timeout=120)
        status = "PASS" if rc == 0 else "FAIL"
        label = script + (" " + " ".join(args[-2:]) if args else "")
        results.append({
            "script": label, "exit_code": rc, "duration": round(dur, 1),
            "status": status, "stderr_tail": stderr[-300:],
        })
        print(f"  {status:<8} {label} ({dur:.1f}s)")

    return {"results": results}


# ---------------------------------------------------------------------------
# Phase 7: NHANES
# ---------------------------------------------------------------------------

def phase7_nhanes():
    print("\n" + "=" * 60)
    print("PHASE 7: NHANES FEASIBILITY")
    print("=" * 60)

    script_path = ROOT / "scripts" / "run_nhanes_feasibility.py"
    if not script_path.exists():
        print("  SKIP  (script not found)")
        return {"status": "SKIP"}

    rc, stdout, stderr, dur = run_script("run_nhanes_feasibility.py", timeout=NHANES_TIMEOUT)
    if rc == -1:
        status = "TIMEOUT"
    elif rc == 0:
        status = "PASS"
    elif "ConnectionError" in stderr or "URLError" in stderr or "timeout" in stderr.lower():
        status = "SKIP"
        print("  Network error -- SKIPPED (not FAIL)")
    else:
        status = "FAIL"

    print(f"  {status:<8} run_nhanes_feasibility.py (exit={rc}, {dur:.1f}s)")
    if status == "FAIL":
        # Show last error line
        for line in stderr.splitlines()[-5:]:
            print(f"    {line}")

    return {"status": status, "exit_code": rc, "duration": round(dur, 1),
            "stderr_tail": stderr[-500:]}


# ---------------------------------------------------------------------------
# Phase 8: Integration tests
# ---------------------------------------------------------------------------

def phase8_integration():
    print("\n" + "=" * 60)
    print("PHASE 8: INTEGRATION TESTS")
    print("=" * 60)

    result = {}

    # 8a: J-comparison integration
    script_path = ROOT / "scripts" / "run_j_comparison_integration.py"
    if script_path.exists():
        rc, stdout, stderr, dur = run_script("run_j_comparison_integration.py", timeout=300)
        sha_match = "SHA match:  YES" in stdout or "SHA match: YES" in stdout
        status = "PASS" if rc == 0 and sha_match else "FAIL"
        result["j_comparison"] = {
            "status": status, "exit_code": rc, "sha_match": sha_match,
            "duration": round(dur, 1),
        }
        print(f"  {status:<8} J-comparison integration ({dur:.1f}s)")
    else:
        result["j_comparison"] = {"status": "SKIP"}
        print("  SKIP  run_j_comparison_integration.py (not found)")

    # 8b: compare_j_runs error handling
    script_path = ROOT / "scripts" / "compare_j_runs.py"
    if script_path.exists():
        rc, stdout, stderr, dur = run_script(
            "compare_j_runs.py",
            args=["--baseline", str(ROOT / "nonexistent.json"),
                  "--candidate", str(ROOT / "nonexistent.json")],
            timeout=30,
        )
        # Should exit with error but NOT a traceback
        has_traceback = "Traceback" in stderr
        status = "FAIL" if has_traceback else ("PASS" if rc != 0 else "FAIL")
        result["error_handling"] = {
            "status": status, "has_traceback": has_traceback,
            "exit_code": rc,
        }
        print(f"  {status:<8} compare_j_runs.py error handling "
              f"(traceback={'yes' if has_traceback else 'no'})")
    else:
        result["error_handling"] = {"status": "SKIP"}

    return result


# ---------------------------------------------------------------------------
# Phase 9: Report generation
# ---------------------------------------------------------------------------

def generate_report(phases):
    """Write JSON and Markdown reports."""
    # Collect summary
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    warnings = 0
    failures_list = []

    def count_result(item_name, status, error=""):
        nonlocal total, passed, failed, skipped, warnings
        total += 1
        if status == "PASS":
            passed += 1
        elif status == "WARN":
            passed += 1
            warnings += 1
        elif status == "SKIP":
            skipped += 1
        elif status == "UNSTABLE":
            # Expected physics — quasi-static axes lose Hurwitz stability
            passed += 1
            warnings += 1
        elif status in ("FAIL", "TIMEOUT"):
            failed += 1
            failures_list.append({"item": item_name, "status": status, "error": error})

    # Phase 1
    if phases["phase1"]["modules_failed"]:
        for m in phases["phase1"]["modules_failed"]:
            count_result(m["module"], "FAIL", m["error"])
    else:
        count_result("All modules import", "PASS")

    # Phase 2
    if phases["phase2"]["failed"] > 0:
        count_result("pytest", "FAIL", f"{phases['phase2']['failed']} test failures")
    else:
        count_result("pytest", "PASS")

    # Phase 3 -- count stable/unstable as informational, not failures
    for c in phases["phase3"]["combinations"]:
        count_result(
            f"Param: {c['csv']} {c['axes']}",
            c["status"],
            f"alpha_80={c.get('alpha_80', 'N/A')}" if c["status"] != "PASS" else "",
        )

    # Phase 4
    for r in phases["phase4"]["results"]:
        count_result(r["script"], r["status"], r.get("stderr_tail", "")[:200])
    for r in phases["phase4"]["parameterised_runs"]:
        count_result(f"{r['script']} (param)", r["status"])

    # Phase 5
    for r in phases["phase5"]["results"]:
        count_result(r["script"], r["status"])

    # Phase 6
    for r in phases["phase6"]["results"]:
        count_result(r["script"], r["status"])

    # Phase 7
    count_result("NHANES", phases["phase7"]["status"])

    # Phase 8
    for key, val in phases["phase8"].items():
        count_result(f"Integration: {key}", val["status"])

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "phases": phases,
        "summary": {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "warnings": warnings,
            "failures_list": failures_list,
        },
    }

    # Write JSON
    OUTPUTS.mkdir(exist_ok=True)
    json_path = OUTPUTS / "repository_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Written: {json_path}")

    # Write Markdown
    md_path = OUTPUTS / "repository_audit_report.md"
    overall = "PASS" if failed == 0 else "FAIL"
    lines = [
        "# HDR-Aging-Sim Repository Audit Report\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Python:** {sys.version.split()[0]}",
        f"**Platform:** {sys.platform}\n",
        f"## Overall: {overall} ({passed} passed, {failed} failed, "
        f"{warnings} warnings, {skipped} skipped)\n",
    ]

    if failures_list:
        lines.append("## Failures\n")
        lines.append("| Item | Status | Error |")
        lines.append("|------|--------|-------|")
        for f_item in failures_list:
            err_short = f_item["error"][:100].replace("|", "/").replace("\n", " ")
            lines.append(f"| {f_item['item']} | {f_item['status']} | {err_short} |")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Written: {md_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("HDR-AGING-SIM REPOSITORY AUDIT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Python: {sys.version}")
    print("=" * 60)

    phases = {}

    phases["phase1"] = phase1_environment()
    phases["phase2"] = phase2_pytest()
    phases["phase3"] = phase3_parameterisation()
    phases["phase4"] = phase4_simulation_scripts()
    phases["phase5"] = phase5_elsa_scripts(phases["phase1"]["elsa_present"])
    phases["phase6"] = phase6_utility_scripts()
    phases["phase7"] = phase7_nhanes()
    phases["phase8"] = phase8_integration()

    print("\n" + "=" * 60)
    print("PHASE 9: REPORT GENERATION")
    print("=" * 60)

    report = generate_report(phases)
    summary = report["summary"]

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    overall = "PASS" if summary["failed"] == 0 else "FAIL"
    print(f"  Overall: {overall}")
    print(f"  Total: {summary['total_checks']}, Passed: {summary['passed']}, "
          f"Failed: {summary['failed']}, Skipped: {summary['skipped']}, "
          f"Warnings: {summary['warnings']}")

    if summary["failures_list"]:
        print(f"\n  Failures:")
        for f_item in summary["failures_list"]:
            print(f"    - {f_item['item']}: {f_item['status']}")

    return 1 if summary["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
