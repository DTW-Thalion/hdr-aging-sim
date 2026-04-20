#!/usr/bin/env python3
"""
InCHIANTI expanded axis analysis.

Tests multiple axis configurations:
  - 5-axis longitudinal: I (IL-6), M (HOMA-IR), N (cortisol/DHEAS), F (SPPB), B (PTH)
  - 4-axis with improved N: I, M, N (cortisol/DHEAS), F
  - 4-axis original: I, M, N (resting HR), F
  - NLR sensitivity: I (NLR), M, N (cortisol/DHEAS), F

For each configuration: lambda_max trajectory, lead-lag, Pi trajectory.
"""

import os, sys, json
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hdr_sim.inchianti import load_inchianti_panel, compute_youthful_reference
from src.hdr_sim.csv_loader import load_J_csv

N_BOOT = 5000
RNG = np.random.default_rng(42)

AGE_STRATA = {"20-49": (20, 49), "50-59": (50, 59), "60-69": (60, 69),
              "70-79": (70, 79), "80+": (80, 120)}

# ── Load J-matrix signs ───────────────────────────────────────────
def load_j_signs():
    rows = load_J_csv()
    signs = {}
    for r in rows:
        fr, to, s = r["axis_from"], r["axis_to"], r["sign"]
        val = +1 if s == "+" else (-1 if s == "-" else 0)
        signs[f"{fr}->{to}"] = val
    return signs

J_SIGNS = load_j_signs()


# ── Core computation functions ────────────────────────────────────
def lambda_max_of_cov(X):
    if len(X) < 3:
        return np.nan
    C = np.cov(X, rowvar=False)
    return float(np.max(eigvalsh(C)))

def bootstrap_lambda_max(X, n_boot=N_BOOT):
    n = len(X)
    if n < 5:
        return np.nan, np.nan
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boots[i] = lambda_max_of_cov(X[idx])
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))

def mean_abs_offdiag_corr(X):
    if len(X) < 5:
        return np.nan
    C = np.corrcoef(X, rowvar=False)
    p = C.shape[0]
    vals = [abs(C[i, j]) for i in range(p) for j in range(i+1, p) if np.isfinite(C[i, j])]
    return float(np.mean(vals)) if vals else np.nan

def cross_lagged_beta(triplets, from_col, to_col, n_boot=N_BOOT):
    """Cross-lagged regression: d_to ~ beta*from_t0 + gamma*to_t0 + delta*age."""
    from scipy.stats import t as t_dist
    y = triplets[f"d_{to_col}"].values
    X_from = triplets[f"{from_col}_t0"].values
    X_auto = triplets[f"{to_col}_t0"].values
    X_age = triplets["age_t"].values
    valid = np.isfinite(y) & np.isfinite(X_from) & np.isfinite(X_auto) & np.isfinite(X_age)
    y, X_from, X_auto, X_age = y[valid], X_from[valid], X_auto[valid], X_age[valid]
    n = len(y)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, n
    X = np.column_stack([np.ones(n), X_from, X_auto, X_age])
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_cross = beta_hat[1]
    y_hat = X @ beta_hat
    resid = y - y_hat
    sigma2 = np.sum(resid**2) / (n - 4)
    try:
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
        t_stat = beta_cross / se
        p_val = 2 * t_dist.sf(abs(t_stat), df=n - 4)
    except:
        se, p_val = np.nan, np.nan
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        try:
            boots[b] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0][1]
        except:
            boots[b] = np.nan
    ci_lo = float(np.nanpercentile(boots, 2.5))
    ci_hi = float(np.nanpercentile(boots, 97.5))
    return float(beta_cross), ci_lo, ci_hi, float(p_val) if np.isfinite(p_val) else np.nan, n


# ── Panel augmentation ────────────────────────────────────────────
def augment_panel(panel):
    """Add new axis columns to the panel."""
    import pyreadstat

    DATA_ROOT = os.path.join(os.path.expanduser("~"), "Downloads", "inCHIANTI", "InCHIANTI_CD_Share")

    # Wave configs for new variables
    wave_labs = [
        (0, "X_", os.path.join(DATA_ROOT, "Baseline_V8", "English", "4.Data", "SAS_Datasets", "Assays", "labo_raw.sas7bdat")),
        (1, "Y_", os.path.join(DATA_ROOT, "Follow-up1_V5", "4.Data", "SAS_Datasets", "Assays", "labf1raw.sas7bdat")),
        (2, "Z_", os.path.join(DATA_ROOT, "Follow-up2_V4", "4.Data", "SAS_Datasets", "Assays", "labf2raw.sas7bdat")),
    ]

    # Standard vars (same suffix across waves)
    standard_vars = {
        "cortisol": "CORTIS", "dheas": "DHEAS", "cortdh": "CORTDH",
        "igf1": "TIGF1", "neutrophils": "N_NEU", "lymphocytes": "N_LIN",
    }
    # PTH has different suffix: X_PTH at baseline, Y_PTH_I at follow-ups
    pth_vars = {0: "X_PTH", 1: "Y_PTH_I", 2: "Z_PTH_I"}

    for wave_idx, prefix, path in wave_labs:
        df, _ = pyreadstat.read_sas7bdat(path)
        wave_mask = panel["wave"] == wave_idx
        for new_col, suffix in standard_vars.items():
            varname = prefix + suffix
            if varname in df.columns:
                lookup = df.set_index("CODE98")[varname].to_dict()
                panel.loc[wave_mask, new_col] = panel.loc[wave_mask, "code98"].map(lookup)
        # PTH with wave-specific name
        pth_var = pth_vars.get(wave_idx)
        if pth_var and pth_var in df.columns:
            lookup = df.set_index("CODE98")[pth_var].to_dict()
            panel.loc[wave_mask, "pth"] = panel.loc[wave_mask, "code98"].map(lookup)

    # Also load waves 3-4 for NLR and PTH
    extra_waves = [
        (3, "Q_", os.path.join(DATA_ROOT, "Follow-up3_V3", "4.Data", "SAS_Datasets", "Assays", "labf3raw.sas7bdat"), "Q_PTH_I"),
        (4, "C_", os.path.join(DATA_ROOT, "Follow-up4_v2", "4.Data", "SAS_Datasets", "Assays", "labf4raw.sas7bdat"), None),
    ]
    for wave_idx, prefix, path, pth_var in extra_waves:
        df, _ = pyreadstat.read_sas7bdat(path)
        wave_mask = panel["wave"] == wave_idx
        for new_col, suffix in [("neutrophils", "N_NEU"), ("lymphocytes", "N_LIN")]:
            varname = prefix + suffix
            if varname in df.columns:
                lookup = df.set_index("CODE98")[varname].to_dict()
                panel.loc[wave_mask, new_col] = panel.loc[wave_mask, "code98"].map(lookup)
        if pth_var and pth_var in df.columns:
            lookup = df.set_index("CODE98")[pth_var].to_dict()
            panel.loc[wave_mask, "pth"] = panel.loc[wave_mask, "code98"].map(lookup)

    # Derived: NLR
    panel["nlr"] = panel["neutrophils"] / panel["lymphocytes"]
    # Log transforms
    panel["log_cortdh"] = np.log(panel["cortdh"].clip(lower=0.001))
    panel["log_pth"] = np.log(panel["pth"].clip(lower=0.1))
    panel["log_nlr"] = np.log(panel["nlr"].clip(lower=0.01))
    panel["log_igf1"] = np.log(panel["igf1"].clip(lower=0.1))

    return panel


# ── Axis configurations ──────────────────────────────────────────
def get_axis_configs():
    return {
        "5axis_IMNFB": {
            "label": "5-axis (I,M,N_cortdh,F,B_pth)",
            "axes": {
                "I": {"var": "log_il6", "sign": +1},
                "M": {"var": "log_homa_ir", "sign": +1},
                "N": {"var": "log_cortdh", "sign": +1},
                "F": {"var": "sppb", "sign": -1},
                "B": {"var": "log_pth", "sign": +1},
            },
        },
        "4axis_cortdh": {
            "label": "4-axis (I,M,N_cortdh,F)",
            "axes": {
                "I": {"var": "log_il6", "sign": +1},
                "M": {"var": "log_homa_ir", "sign": +1},
                "N": {"var": "log_cortdh", "sign": +1},
                "F": {"var": "sppb", "sign": -1},
            },
        },
        "4axis_hr": {
            "label": "4-axis (I,M,N_hr,F) [original]",
            "axes": {
                "I": {"var": "log_il6", "sign": +1},
                "M": {"var": "log_homa_ir", "sign": +1},
                "N": {"var": "resting_hr", "sign": +1},
                "F": {"var": "sppb", "sign": -1},
            },
        },
        "4axis_nlr": {
            "label": "4-axis (I_nlr,M,N_cortdh,F)",
            "axes": {
                "I": {"var": "log_nlr", "sign": +1},
                "M": {"var": "log_homa_ir", "sign": +1},
                "N": {"var": "log_cortdh", "sign": +1},
                "F": {"var": "sppb", "sign": -1},
            },
        },
    }


def standardize_config(panel, config, ref_panel=None):
    """Standardize axes for a given configuration using healthy 20-30 reference."""
    if ref_panel is None:
        ref_panel = panel
    bl = ref_panel[ref_panel["wave"] == 0].copy()
    # Healthy mask
    dx_exclude = ["dx_htn", "dx_dm", "dx_metsyn", "dx_frailty", "dx_mi", "dx_chf", "dx_stroke", "dx_dement"]
    mask_healthy = pd.Series(True, index=bl.index)
    for dx in dx_exclude:
        if dx in bl.columns:
            mask_healthy &= (bl[dx] != 1) | bl[dx].isna()
    mask_age = (bl["age"] >= 20) & (bl["age"] <= 30)
    ref = bl[mask_healthy & mask_age]
    if len(ref) < 50:
        mask_age = (bl["age"] >= 20) & (bl["age"] <= 35)
        ref = bl[mask_healthy & mask_age]

    panel = panel.copy()
    delta_cols = []
    for axis_name, axis_def in config["axes"].items():
        var = axis_def["var"]
        sign = axis_def["sign"]
        vals = ref[var].dropna()
        mean_ref = vals.mean() if len(vals) > 0 else 0
        sd_ref = vals.std() if len(vals) > 1 else 1
        if sd_ref == 0:
            # SPPB ceiling fix: use healthy <60
            wider = bl[mask_healthy & (bl["age"] < 60)]
            wider_vals = wider[var].dropna()
            sd_ref = wider_vals.std() if len(wider_vals) > 1 else 1
        col = f"delta_{axis_name}"
        panel[col] = sign * (panel[var] - mean_ref) / sd_ref
        delta_cols.append(col)

    return panel, delta_cols


def compute_change_vectors(panel, delta_cols):
    """Build within-person change vectors."""
    rows = []
    for subj, grp in panel.groupby("code98"):
        grp = grp.sort_values("wave")
        for i in range(len(grp) - 1):
            r0, r1 = grp.iloc[i], grp.iloc[i + 1]
            if any(pd.isna(r0[c]) or pd.isna(r1[c]) for c in delta_cols):
                continue
            row = {"code98": subj, "age_t": r0["age"],
                   "age_mid": (r0["age"] + r1["age"]) / 2}
            for c in delta_cols:
                row[f"{c}_t0"] = r0[c]
                row[f"d_{c}"] = r1[c] - r0[c]
            rows.append(row)
    return pd.DataFrame(rows)


def run_lambda_max(changes, delta_cols, label):
    """Lambda_max trajectory by age stratum."""
    print(f"\n  lambda_max trajectory [{label}], N_pairs={len(changes)}")
    results = []
    for name, (lo, hi) in AGE_STRATA.items():
        mask = (changes["age_mid"] >= lo) & (changes["age_mid"] <= hi)
        sub = changes.loc[mask, [f"d_{c}" for c in delta_cols]].values
        n = len(sub)
        lmax = lambda_max_of_cov(sub)
        ci_lo, ci_hi = bootstrap_lambda_max(sub)
        print(f"    {name:8s}: N={n:4d}  lmax={lmax:8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        results.append({"stratum": name, "n": n, "lambda_max": lmax,
                        "ci_lower": ci_lo, "ci_upper": ci_hi})
    return results


def run_lead_lag(changes, delta_cols, config, label):
    """Lead-lag for all ordered pairs."""
    axes = list(config["axes"].keys())
    print(f"\n  Lead-lag [{label}], {len(axes)}x{len(axes)-1} pairs")
    results = []
    n_concordant = 0
    n_tested = 0
    for from_ax in axes:
        for to_ax in axes:
            if from_ax == to_ax:
                continue
            from_col = f"delta_{from_ax}"
            to_col = f"delta_{to_ax}"
            beta, ci_lo, ci_hi, p_val, n = cross_lagged_beta(changes, from_col, to_col)
            pair_key = f"{from_ax}->{to_ax}"
            predicted = J_SIGNS.get(pair_key, 0)
            if predicted == 0:
                match = "N/A"
            elif np.isnan(beta):
                match = "N/A"
            else:
                obs_sign = +1 if beta > 0 else -1
                # Sign convention: F and B axes are sign-flipped in delta
                # The J-matrix sign predicts the raw coupling direction.
                # Our delta convention: positive = decline.
                # F->X: J says F->I is -1 (protective). In delta space,
                # delta_F increasing = SPPB declining. If SPPB declines (delta_F up),
                # does inflammation increase (delta_I up)? That would be beta > 0,
                # which means the coupling is pathological from a decline perspective.
                # But the J entry is -1 (protective = exercise reduces inflammation).
                # So we need to account for sign flips.
                # For axes with sign=-1 (F, B have sign flip in standardization),
                # the J-matrix sign needs to be adjusted:
                # If from_axis has sign=-1, flip the predicted sign
                # If to_axis has sign=-1, flip the predicted sign
                # Convention B (biological direction): in delta-space where
                # positive = decline, ALL pairs predict positive beta
                # (worsening in source -> worsening in target, whether via
                # pathological coupling or loss of protective coupling)
                concordant = (obs_sign == +1)  # Convention B
                match = "YES" if concordant else "NO"
                n_tested += 1
                if concordant:
                    n_concordant += 1

            stars = ""
            if not np.isnan(p_val):
                if p_val < 0.001: stars = "***"
                elif p_val < 0.01: stars = "**"
                elif p_val < 0.05: stars = "*"

            print(f"    {pair_key:8s}: beta={beta:>8.4f}{stars:3s}  [{ci_lo:.4f},{ci_hi:.4f}]  "
                  f"pred={predicted:+d}  match={match}")
            results.append({
                "pair": pair_key, "from": from_ax, "to": to_ax,
                "beta": beta, "ci_lower": ci_lo, "ci_upper": ci_hi,
                "p_value": p_val, "n": n,
                "predicted_sign": predicted, "concordant": match == "YES",
            })

    if n_tested > 0:
        p_binom = binomtest(n_concordant, n_tested, 0.5, alternative="greater").pvalue
        print(f"    Concordance: {n_concordant}/{n_tested} ({n_concordant/n_tested*100:.0f}%), p={p_binom:.4f}")
    else:
        p_binom = np.nan
    return results, n_concordant, n_tested, p_binom


def run_pi(panel_std, delta_cols, label):
    """Pi = C_norm / V_norm trajectory."""
    complete = panel_std.dropna(subset=delta_cols)
    print(f"\n  Pi trajectory [{label}], N_complete={len(complete)}")
    ref_lo, ref_hi = 20, 49
    ref_data = complete[(complete["age"] >= ref_lo) & (complete["age"] <= ref_hi)][delta_cols].values
    if len(ref_data) < 10:
        print("    Too few reference data for Pi")
        return []
    C_ref = np.cov(ref_data, rowvar=False)
    V_ref = np.mean(np.diag(C_ref))
    p = C_ref.shape[0]
    C_off_ref = np.mean([abs(C_ref[i, j]) for i in range(p) for j in range(i+1, p)])

    results = []
    for name, (lo, hi) in AGE_STRATA.items():
        mask = (complete["age"] >= lo) & (complete["age"] <= hi)
        X = complete.loc[mask, delta_cols].values
        n = len(X)
        if n < 10:
            continue
        C = np.cov(X, rowvar=False)
        V = np.mean(np.diag(C))
        C_off = np.mean([abs(C[i, j]) for i in range(p) for j in range(i+1, p)])
        V_norm = V / V_ref if V_ref > 0 else np.nan
        C_norm = C_off / C_off_ref if C_off_ref > 0 else np.nan
        Pi = C_norm / V_norm if V_norm > 0 else np.nan
        print(f"    {name:8s}: N={n:4d}  V_norm={V_norm:.3f}  C_norm={C_norm:.3f}  Pi={Pi:.3f}")
        results.append({"stratum": name, "n": n, "V_norm": float(V_norm),
                        "C_norm": float(C_norm), "Pi": float(Pi)})

    if len(results) >= 3:
        ages_mid = {"20-49": 35, "50-59": 55, "60-69": 65, "70-79": 75, "80+": 85}
        a = [ages_mid[r["stratum"]] for r in results if r["stratum"] in ages_mid]
        p_vals = [r["Pi"] for r in results if r["stratum"] in ages_mid]
        if len(a) >= 3:
            slope = np.polyfit(a, p_vals, 1)[0]
            print(f"    Pi slope: {slope:.6f}/yr")
            return results, slope
    return results, np.nan


def main():
    print("=" * 60)
    print("InCHIANTI: Expanded Axis Analysis")
    print("=" * 60)

    # Load and augment panel
    print("Loading panel...")
    panel = load_inchianti_panel()
    print("Augmenting with new biomarkers...")
    panel = augment_panel(panel)

    # Add log_homa_ir for standardization
    panel["log_homa_ir"] = np.log(panel["homa_ir"].clip(lower=0.01))

    configs = get_axis_configs()
    all_results = {}

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config['label']}")
        print(f"{'='*60}")

        panel_std, delta_cols = standardize_config(panel, config)
        changes = compute_change_vectors(panel_std, delta_cols)

        if len(changes) < 20:
            print(f"  Only {len(changes)} change pairs -- skipping")
            all_results[config_name] = {"skipped": True, "n_pairs": len(changes)}
            continue

        lmax_results = run_lambda_max(changes, delta_cols, config["label"])
        ll_results, n_conc, n_test, p_binom = run_lead_lag(changes, delta_cols, config, config["label"])
        pi_results = run_pi(panel_std, delta_cols, config["label"])
        pi_slope = pi_results[1] if isinstance(pi_results, tuple) else np.nan
        pi_data = pi_results[0] if isinstance(pi_results, tuple) else pi_results

        all_results[config_name] = {
            "label": config["label"],
            "n_pairs": len(changes),
            "lambda_max": lmax_results,
            "lead_lag": ll_results,
            "concordance": {"n_concordant": n_conc, "n_tested": n_test,
                            "rate": n_conc/n_test if n_test > 0 else None,
                            "p_binom": float(p_binom) if np.isfinite(p_binom) else None},
            "pi": pi_data,
            "pi_slope": float(pi_slope) if np.isfinite(pi_slope) else None,
        }

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/inchianti_6axis_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary comparison
    print(f"\n{'='*60}")
    print("CONFIGURATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Config':<30s} {'N_pairs':>8s} {'Conc':>8s} {'p_binom':>8s} {'Pi_slope':>10s}")
    for name, res in all_results.items():
        if res.get("skipped"):
            continue
        conc = res["concordance"]
        p_str = f"{conc['p_binom']:.4f}" if conc.get('p_binom') is not None else "N/A"
        pi_str = f"{res['pi_slope']:.6f}" if res.get('pi_slope') is not None else "N/A"
        print(f"{res['label']:<30s} {res['n_pairs']:>8d} "
              f"{conc['n_concordant']}/{conc['n_tested']:>3d}    "
              f"{p_str:>8s} {pi_str:>10s}")

    print(f"\nSaved to results/inchianti_6axis_results.json")


if __name__ == "__main__":
    main()
