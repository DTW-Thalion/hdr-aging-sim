#!/usr/bin/env python3
"""
ELSA Retrospective ICI Deployment Assessment

Evaluates the ICI deployment threshold (ω_min) against real ELSA
longitudinal biomarker data (3-axis: I, M, F) to determine whether
existing cohort data density is sufficient for safe closed-loop
deployment, and how much additional sensing density would be needed.

Reuses the data loading and z-scoring infrastructure from
run_elsa_validation.py and the J-matrix loading from
hdr_sim.csv_loader.

Usage:
    python scripts/run_elsa_ici_deployment.py

Outputs:
    outputs/elsa_ici_deployment.json  — full machine-readable results
    outputs/elsa_ici_deployment_table.txt — manuscript table
"""

import argparse
import json
import os
import sys
import warnings

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import linalg

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

# Import from existing ELSA pipeline
from run_elsa_validation import (
    load_all_files,
    extract_nurse_biomarkers,
    prepare_harmonised,
    extract_mortality,
    extract_supplementary,
    harmonised_to_long,
    build_analysis_panel,
    NURSE_WAVE_YEARS,
)

# Import J-matrix loader
from hdr_sim.csv_loader import (
    load_J_csv,
    build_J_basin,
    get_calibration_scalar,
    _default_csv_path,
)
from hdr_sim.j_matrix_spec import JMatrixSpec, load_default_spec

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='ELSA ICI Deployment Assessment')
    parser.add_argument('--j-matrix', type=str, default=None,
                        help='Path to J matrix CSV. Default: data/J_matrix_compiled_9x9.csv')
    parser.add_argument('--axes', type=str, nargs='+', default=None,
                        help='Axis subset (e.g., I M F). Default: script-specific.')
    return parser.parse_args()


_args = parse_args()

# Nurse waves with blood biomarkers
BLOOD_WAVES = [2, 4, 6, 8]

# Measurement error estimates (z-score units)
# CRP: within-person CV ~40-50%
# HbA1c: within-person CV ~3-5%
# Grip: within-person CV ~6-8%
R_BASELINE = np.diag([0.45**2, 0.05**2, 0.07**2])

# μ̄ values for sensitivity analysis
MU_BAR_VALUES = [0.01, 0.05, 0.10, 0.20]

# Number of basins
K = 2

# Sensing scenarios
SENSING_SCENARIOS = {
    'ELSA actual': {'f_s': 0.25, 'p_miss': None, 'rho': 0.3,
                    'description': 'Current cohort data (4-yearly blood draws)'},
    'Monthly panels': {'f_s': 12, 'p_miss': 0.10, 'rho': 0.6,
                       'description': 'Clinical monitoring (monthly blood panels)'},
    'Weekly wearable': {'f_s': 52, 'p_miss': 0.05, 'rho': 0.5,
                        'description': 'Wearable + weekly blood draw'},
    'Daily wearable': {'f_s': 365, 'p_miss': 0.05, 'rho': 0.4,
                       'description': 'Full wearable stack (daily sensing)'},
}


# =========================================================================
# Step 1: Construct Longitudinal Panel
# =========================================================================
def construct_longitudinal_panel(merged):
    """
    Identify individuals with complete 3-axis biomarker data at ≥2
    nurse-visit waves (2, 4, 6, 8).
    """
    print("\n" + "=" * 70)
    print("ICI DEPLOYMENT ASSESSMENT — Step 1: Longitudinal Panel")
    print("=" * 70)

    # Filter to blood waves only and complete 3-axis observations
    blood = merged[
        merged['wave'].isin(BLOOD_WAVES) &
        merged['complete_3axis']
    ].copy()

    print(f"  Complete 3-axis observations at blood waves: {len(blood):,}")
    print(f"  Unique participants: {blood['idauniq'].nunique():,}")

    # Build panel: for each participant, collect all complete waves
    panel = []
    for idauniq, group in blood.groupby('idauniq'):
        obs = []
        for _, row in group.sort_values('wave').iterrows():
            obs.append({
                'wave': int(row['wave']),
                'year': NURSE_WAVE_YEARS.get(int(row['wave']), np.nan),
                'age': row['age'],
                'dx_I': row['dx_I'],
                'dx_M': row['dx_M'],
                'dx_F': row['dx_F'],
            })
        if len(obs) >= 2:
            panel.append({
                'id': idauniq,
                'n_waves': len(obs),
                'observations': obs,
            })

    n_2plus = len(panel)
    n_3plus = sum(1 for p in panel if p['n_waves'] >= 3)
    n_4 = sum(1 for p in panel if p['n_waves'] >= 4)

    print(f"\n  Longitudinal panel:")
    print(f"    ≥2 waves: {n_2plus:,}")
    print(f"    ≥3 waves: {n_3plus:,}")
    print(f"    All 4 waves: {n_4:,}")

    return panel, blood


def panel_to_dataframe(panel):
    """Flatten panel list to a DataFrame for covariance estimation."""
    rows = []
    for p in panel:
        for obs in p['observations']:
            rows.append({
                'idauniq': p['id'],
                'wave': obs['wave'],
                'year': obs['year'],
                'age': obs['age'],
                'dx_I': obs['dx_I'],
                'dx_M': obs['dx_M'],
                'dx_F': obs['dx_F'],
            })
    return pd.DataFrame(rows)


# =========================================================================
# Step 2: Estimate Basin Parameters
# =========================================================================
def estimate_basin_parameters(panel_df, method='age'):
    """
    Estimate SLDS parameters for K=2 basins.

    Parameters
    ----------
    panel_df : DataFrame with dx_I, dx_M, dx_F, age columns
    method : 'age' or 'score'
        'age': 50-64 = healthy, 65+ = disease
        'score': median split on composite dysregulation score

    Returns
    -------
    dict with basin parameters
    """
    print(f"\n  Basin assignment method: {method}")

    axes = ['dx_I', 'dx_M', 'dx_F']

    if method == 'age':
        panel_df['basin'] = np.where(panel_df['age'] < 65, 'healthy', 'disease')
    else:
        # Score-based: composite dysregulation = mean of all 3 axes
        panel_df['composite'] = panel_df[axes].mean(axis=1)
        median_score = panel_df['composite'].median()
        panel_df['basin'] = np.where(
            panel_df['composite'] < median_score, 'healthy', 'disease')
        print(f"    Median composite score: {median_score:.3f}")

    n_healthy = (panel_df['basin'] == 'healthy').sum()
    n_disease = (panel_df['basin'] == 'disease').sum()
    print(f"    Healthy basin: {n_healthy:,} observations")
    print(f"    Disease basin: {n_disease:,} observations")

    # Cross-sectional covariance per basin
    basins = {}
    for basin_name in ['healthy', 'disease']:
        subset = panel_df[panel_df['basin'] == basin_name][axes].dropna()
        Sigma_x = np.cov(subset.values, rowvar=False)
        mean_x = subset.mean().values
        basins[basin_name] = {
            'Sigma_x': Sigma_x,
            'mean_x': mean_x,
            'n_obs': len(subset),
        }
        print(f"    {basin_name} Σ_x diagonal: "
              f"[{Sigma_x[0,0]:.4f}, {Sigma_x[1,1]:.4f}, {Sigma_x[2,2]:.4f}]")

    # Basin prevalence
    total = n_healthy + n_disease
    basins['healthy']['pi'] = n_healthy / total
    basins['disease']['pi'] = n_disease / total
    print(f"    Basin prevalence: healthy={basins['healthy']['pi']:.3f}, "
          f"disease={basins['disease']['pi']:.3f}")

    return basins


def get_A_matrices():
    """
    Get calibrated A matrices for the 3-axis (I, M, F) model from
    the J-matrix CSV.
    """
    print("\n  Loading J-matrix and constructing A matrices...")

    # Load J-matrix for 3-axis subset
    rows = load_J_csv(_args.j_matrix or _default_csv_path())
    J_healthy = build_J_basin(rows, basin='healthy', axes=('I', 'M', 'F'))
    J_disease = build_J_basin(rows, basin='disease', axes=('I', 'M', 'F'))

    print(f"    J_healthy (3×3):\n{J_healthy}")
    print(f"    J_disease (3×3):\n{J_disease}")

    # Recovery time constants for 3-axis model (I, M, F)
    # Using the same biology-motivated values as the 4-axis model,
    # selecting the I, M, F entries
    tau_healthy = np.array([7.0, 0.1, 8.0])   # days
    tau_disease = np.array([25.0, 0.30, 42.0])  # days

    # Calibrate: find scalar c so that spectral abscissa at healthy = -0.134
    target_alpha = -0.134
    A_base_healthy = -np.diag(1.0 / tau_healthy)
    c = get_calibration_scalar(J_healthy, tau_healthy, target_alpha)

    A_healthy = -np.diag(1.0 / tau_healthy) + c * J_healthy
    A_disease = -np.diag(1.0 / tau_disease) + c * J_disease

    alpha_h = np.max(np.real(np.linalg.eigvals(A_healthy)))
    alpha_d = np.max(np.real(np.linalg.eigvals(A_disease)))

    print(f"    Calibration scalar c = {c:.6f}")
    print(f"    α(A_healthy) = {alpha_h:.4f}")
    print(f"    α(A_disease) = {alpha_d:.4f}")

    return A_healthy, A_disease, c


# =========================================================================
# Step 3: Compute d_min (Basin Separability)
# =========================================================================
def compute_d_min(basins, R, C=None):
    """
    Compute KL-divergence-based basin separability metric.

    d_kj = 0.5 * (trace(inv(Σ_y_j) @ Σ_y_k) - n + log(det(Σ_y_j)/det(Σ_y_k)))
    d_min = min(d_healthy→disease, d_disease→healthy)
    """
    n = 3  # dimensions
    if C is None:
        C = np.eye(n)

    Sigma_y = {}
    for basin_name in ['healthy', 'disease']:
        Sigma_x = basins[basin_name]['Sigma_x']
        Sigma_y[basin_name] = C @ Sigma_x @ C.T + R

    # KL divergence in both directions
    def kl_div(Sigma_k, Sigma_j):
        inv_j = linalg.inv(Sigma_j)
        sign_j, logdet_j = np.linalg.slogdet(Sigma_j)
        sign_k, logdet_k = np.linalg.slogdet(Sigma_k)
        return 0.5 * (
            np.trace(inv_j @ Sigma_k) - n +
            logdet_j - logdet_k
        )

    d_hd = kl_div(Sigma_y['healthy'], Sigma_y['disease'])
    d_dh = kl_div(Sigma_y['disease'], Sigma_y['healthy'])
    d_min = min(d_hd, d_dh)

    return d_min, d_hd, d_dh, Sigma_y


# =========================================================================
# Step 4: Compute μ̄ from ISS Parameters
# =========================================================================
def compute_iss_parameters(A_healthy, A_disease):
    """
    Compute ISS-based μ̄ from the calibrated A matrices.

    η = min_k(1 - ρ(A_cl,k))  — contraction margin
    c_ISS(k) = λ_max(P_k) / λ_min(P_k) — Lyapunov condition number

    Returns dict with ISS parameters and estimated μ̄ values.
    """
    print("\n  Computing ISS parameters...")

    results = {}
    for name, A in [('healthy', A_healthy), ('disease', A_disease)]:
        # Spectral radius
        eigs = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigs))
        alpha = np.max(np.real(eigs))

        # Contraction margin (for continuous-time: η based on spectral abscissa)
        eta = -alpha  # margin from stability boundary

        # Solve Lyapunov equation: A'P + PA = -I
        P = linalg.solve_continuous_lyapunov(A.T, -np.eye(A.shape[0]))
        eigs_P = np.linalg.eigvals(P)
        lam_max_P = np.max(np.real(eigs_P))
        lam_min_P = np.min(np.real(eigs_P))
        c_ISS = lam_max_P / max(lam_min_P, 1e-12)

        results[name] = {
            'spectral_abscissa': float(alpha),
            'spectral_radius': float(rho),
            'eta': float(eta),
            'c_ISS': float(c_ISS),
            'P_cond': float(c_ISS),
        }

        print(f"    {name}: α={alpha:.4f}, η={eta:.4f}, c_ISS={c_ISS:.2f}")

    # Mismatch bounds from between-basin parameter differences
    Delta_A = np.linalg.norm(A_healthy - A_disease, ord=2)
    print(f"    ΔA (operator norm) = {Delta_A:.4f}")

    # Compute μ̄ using ISS formula
    # μ̄ = min_k [ε_control · η / (c_ISS(k) · ΔA)]²
    # Use ε_control = 1.0 (unit control authority)
    eps_control = 1.0
    mu_bar_iss_values = []
    for name in ['healthy', 'disease']:
        eta = results[name]['eta']
        c_ISS = results[name]['c_ISS']
        if Delta_A > 0:
            mu_bar_k = (eps_control * eta / (c_ISS * Delta_A)) ** 2
        else:
            mu_bar_k = 1.0  # no mismatch
        mu_bar_iss_values.append(mu_bar_k)
        results[name]['mu_bar'] = float(mu_bar_k)

    mu_bar_iss = min(mu_bar_iss_values)
    print(f"    ISS-derived μ̄ = {mu_bar_iss:.6f}")

    results['Delta_A'] = float(Delta_A)
    results['mu_bar_iss'] = float(mu_bar_iss)

    return results


# =========================================================================
# Step 5: Compute ω_min and T_k^eff Per Participant
# =========================================================================
def compute_omega_min(d_min, mu_bar):
    """ω_min = (1/d_min) * log((K-1)/μ̄)"""
    if d_min <= 0 or mu_bar <= 0:
        return float('inf')
    return (1.0 / d_min) * np.log((K - 1) / mu_bar)


def compute_T_eff_per_participant(panel, basins, rho_k=0.3):
    """
    Compute effective observation count T_k^eff for each participant.

    T_k^eff = T_waves * π_k * (1 - p_miss) * (1 - ρ_k)
    """
    results = []
    for p in panel:
        T_waves = p['n_waves']
        p_miss = 1.0 - (T_waves / len(BLOOD_WAVES))

        # Per-participant basin prevalence from their trajectory
        ages = [obs['age'] for obs in p['observations']]
        n_healthy = sum(1 for a in ages if a < 65)
        n_disease = sum(1 for a in ages if a >= 65)
        total = n_healthy + n_disease

        pi_healthy = n_healthy / total if total > 0 else 0.5
        pi_disease = n_disease / total if total > 0 else 0.5

        # Effective obs per basin (use min across basins for conservatism)
        T_eff_healthy = T_waves * pi_healthy * (1 - p_miss) * (1 - rho_k)
        T_eff_disease = T_waves * pi_disease * (1 - p_miss) * (1 - rho_k)
        T_eff = min(T_eff_healthy, T_eff_disease)

        results.append({
            'id': p['id'],
            'n_waves': T_waves,
            'p_miss': p_miss,
            'pi_healthy': pi_healthy,
            'pi_disease': pi_disease,
            'T_eff_healthy': T_eff_healthy,
            'T_eff_disease': T_eff_disease,
            'T_eff_min': T_eff,
        })

    return results


# =========================================================================
# Step 6: Counterfactual Sensing Scenarios
# =========================================================================
def compute_sensing_scenarios(omega_min, basins):
    """
    For each sensing scenario, compute T_deploy (years to meet threshold)
    and the effective observation rate.
    """
    pi_min = min(basins['healthy']['pi'], basins['disease']['pi'])

    scenarios = {}
    for name, params in SENSING_SCENARIOS.items():
        f_s = params['f_s']
        p_miss = params['p_miss'] if params['p_miss'] is not None else 0.0
        rho = params['rho']

        # Effective observations per year
        eff_rate = f_s * pi_min * (1 - p_miss) * (1 - rho)

        if eff_rate > 0 and omega_min < float('inf'):
            T_deploy_years = omega_min / eff_rate
            T_deploy_months = T_deploy_years * 12
        else:
            T_deploy_years = float('inf')
            T_deploy_months = float('inf')

        scenarios[name] = {
            'f_s': f_s,
            'p_miss': p_miss,
            'rho': rho,
            'eff_rate_per_year': float(eff_rate),
            'T_deploy_years': float(T_deploy_years),
            'T_deploy_months': float(T_deploy_months),
            'description': params['description'],
        }

    return scenarios


# =========================================================================
# Step 7: Sensitivity Analysis
# =========================================================================
def sensitivity_R_matrix(basins, R_baseline, scales=[0.5, 1.0, 2.0]):
    """Compute d_min at different R-matrix scalings."""
    print("\n  R-matrix sensitivity analysis:")
    results = {}
    for scale in scales:
        R_scaled = R_baseline * scale
        d_min, d_hd, d_dh, _ = compute_d_min(basins, R_scaled)
        results[f'R_scale_{scale}'] = {
            'scale': scale,
            'd_min': float(d_min),
            'd_hd': float(d_hd),
            'd_dh': float(d_dh),
        }
        print(f"    R×{scale}: d_min={d_min:.4f} "
              f"(d_h→d={d_hd:.4f}, d_d→h={d_dh:.4f})")
    return results


def sensitivity_basin_method(merged):
    """Run full analysis with both age-based and score-based basin assignment."""
    results = {}
    for method in ['age', 'score']:
        panel_df = merged[
            merged['wave'].isin(BLOOD_WAVES) &
            merged['complete_3axis']
        ][['idauniq', 'wave', 'age', 'dx_I', 'dx_M', 'dx_F']].copy()

        basins = estimate_basin_parameters(panel_df, method=method)
        d_min, d_hd, d_dh, _ = compute_d_min(basins, R_BASELINE)

        results[method] = {
            'd_min': float(d_min),
            'd_hd': float(d_hd),
            'd_dh': float(d_dh),
            'pi_healthy': float(basins['healthy']['pi']),
            'pi_disease': float(basins['disease']['pi']),
        }
    return results


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 70)
    print("ELSA Retrospective ICI Deployment Assessment")
    print("=" * 70)

    # --- Load data using existing pipeline ---
    files = load_all_files()
    panel, hba1c_units = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm, waves=BLOOD_WAVES)
    merged = build_analysis_panel(panel, harm_long, mort, supp)

    # --- Step 1: Construct longitudinal panel ---
    longitudinal_panel, blood_df = construct_longitudinal_panel(merged)
    panel_df = panel_to_dataframe(longitudinal_panel)

    N_panel = len(longitudinal_panel)
    if N_panel == 0:
        print("ERROR: No participants with ≥2 complete waves. Aborting.")
        return

    # --- Step 2: Estimate basin parameters (age-based primary) ---
    print("\n" + "=" * 70)
    print("ICI DEPLOYMENT ASSESSMENT — Step 2: Basin Parameters")
    print("=" * 70)
    basins = estimate_basin_parameters(panel_df.copy(), method='age')

    # Get A matrices from J-matrix
    A_healthy, A_disease, cal_scalar = get_A_matrices()

    # --- Step 3: Compute d_min ---
    print("\n" + "=" * 70)
    print("ICI DEPLOYMENT ASSESSMENT — Step 3: Basin Separability (d_min)")
    print("=" * 70)

    C = np.eye(3)  # all three axes directly observed
    d_min, d_hd, d_dh, Sigma_y = compute_d_min(basins, R_BASELINE, C)

    print(f"\n  d_min = {d_min:.4f}")
    print(f"  d(healthy→disease) = {d_hd:.4f}")
    print(f"  d(disease→healthy) = {d_dh:.4f}")

    # Sensitivity on R
    R_sensitivity = sensitivity_R_matrix(basins, R_BASELINE)

    # --- Step 4: ISS Parameters ---
    print("\n" + "=" * 70)
    print("ICI DEPLOYMENT ASSESSMENT — Step 4: ISS Parameters")
    print("=" * 70)
    iss_results = compute_iss_parameters(A_healthy, A_disease)

    # --- Step 5: ω_min and T_k^eff ---
    print("\n" + "=" * 70)
    print("ICI DEPLOYMENT ASSESSMENT — Step 5: ω_min and Deployment Assessment")
    print("=" * 70)

    omega_results = {}
    for mu_bar in MU_BAR_VALUES:
        omega_min = compute_omega_min(d_min, mu_bar)
        T_eff_results = compute_T_eff_per_participant(longitudinal_panel, basins)

        n_meets = sum(1 for r in T_eff_results if r['T_eff_min'] >= omega_min)
        frac_meets = n_meets / N_panel if N_panel > 0 else 0.0

        T_eff_values = [r['T_eff_min'] for r in T_eff_results]
        max_T_eff = max(T_eff_values) if T_eff_values else 0

        print(f"\n  μ̄ = {mu_bar}:")
        print(f"    ω_min = {omega_min:.4f}")
        print(f"    Max T_k^eff in ELSA = {max_T_eff:.4f}")
        print(f"    Participants meeting threshold: {n_meets}/{N_panel} "
              f"({frac_meets*100:.1f}%)")

        omega_results[str(mu_bar)] = {
            'mu_bar': mu_bar,
            'omega_min': float(omega_min),
            'n_meets': n_meets,
            'frac_meets': float(frac_meets),
            'max_T_eff': float(max_T_eff),
        }

    # --- Step 6: Sensing Scenarios ---
    print("\n" + "=" * 70)
    print("ICI DEPLOYMENT ASSESSMENT — Step 6: Counterfactual Sensing")
    print("=" * 70)

    # Primary analysis at μ̄ = 0.05
    mu_bar_primary = 0.05
    omega_min_primary = compute_omega_min(d_min, mu_bar_primary)
    scenarios = compute_sensing_scenarios(omega_min_primary, basins)

    print(f"\n  ω_min at μ̄ = {mu_bar_primary}: {omega_min_primary:.4f}")
    print(f"\n  {'Scenario':<20s} {'f_s':>8s} {'ρ':>6s} {'T_deploy':>12s} {'T_months':>10s}")
    print("  " + "-" * 60)
    for name, sc in scenarios.items():
        T_str = f"{sc['T_deploy_years']:.2f} yr" if sc['T_deploy_years'] < 1000 else "∞"
        M_str = f"{sc['T_deploy_months']:.1f}" if sc['T_deploy_months'] < 12000 else "∞"
        print(f"  {name:<20s} {sc['f_s']:>8.2f} {sc['rho']:>6.2f} {T_str:>12s} {M_str:>10s}")

    # Fraction of ELSA participants who would meet threshold under each scenario
    # For ELSA actual, use per-participant T_eff already computed
    T_eff_results = compute_T_eff_per_participant(longitudinal_panel, basins)
    scenario_fractions = {}
    for name, sc in scenarios.items():
        if name == 'ELSA actual':
            # Use actual per-participant data
            n_meets = sum(1 for r in T_eff_results
                          if r['T_eff_min'] >= omega_min_primary)
        else:
            # For hypothetical scenarios, compute T_eff assuming
            # each participant has been observed for 12 years (ELSA span)
            observation_span = 12  # years (2004-2016)
            T_total = sc['f_s'] * observation_span
            pi_min = min(basins['healthy']['pi'], basins['disease']['pi'])
            T_eff_hyp = T_total * pi_min * (1 - sc['p_miss']) * (1 - sc['rho'])
            n_meets = N_panel if T_eff_hyp >= omega_min_primary else 0

        frac = n_meets / N_panel if N_panel > 0 else 0.0
        scenario_fractions[name] = {
            'n_meets': n_meets,
            'frac_meets': float(frac),
        }
        print(f"  {name}: {n_meets}/{N_panel} participants ({frac*100:.1f}%) "
              f"would meet threshold")

    # --- Sensitivity: basin assignment method ---
    print("\n" + "=" * 70)
    print("ICI DEPLOYMENT ASSESSMENT — Sensitivity: Basin Assignment")
    print("=" * 70)
    basin_sensitivity = sensitivity_basin_method(merged)
    for method, res in basin_sensitivity.items():
        omega = compute_omega_min(res['d_min'], mu_bar_primary)
        print(f"  {method}: d_min={res['d_min']:.4f}, ω_min={omega:.4f}")

    # =====================================================================
    # Compile results
    # =====================================================================
    headline = {
        'N_panel': N_panel,
        'N_3plus_waves': sum(1 for p in longitudinal_panel if p['n_waves'] >= 3),
        'N_4_waves': sum(1 for p in longitudinal_panel if p['n_waves'] >= 4),
        'd_min': float(d_min),
        'd_hd': float(d_hd),
        'd_dh': float(d_dh),
        'omega_min_mu005': float(omega_min_primary),
        'frac_meets_ELSA': float(omega_results['0.05']['frac_meets']),
        'T_deploy_daily_months': float(
            scenarios['Daily wearable']['T_deploy_months']),
        'T_deploy_weekly_months': float(
            scenarios['Weekly wearable']['T_deploy_months']),
    }

    full_results = {
        'headline': headline,
        'omega_by_mu_bar': omega_results,
        'sensing_scenarios': scenarios,
        'scenario_fractions': scenario_fractions,
        'iss_parameters': iss_results,
        'R_sensitivity': R_sensitivity,
        'basin_sensitivity': basin_sensitivity,
        'basin_parameters': {
            name: {
                'Sigma_x': basins[name]['Sigma_x'].tolist(),
                'mean_x': basins[name]['mean_x'].tolist(),
                'n_obs': basins[name]['n_obs'],
                'pi': basins[name]['pi'],
            }
            for name in ['healthy', 'disease']
        },
        'A_matrices': {
            'A_healthy': A_healthy.tolist(),
            'A_disease': A_disease.tolist(),
            'calibration_scalar': float(cal_scalar),
        },
        'Sigma_y': {
            name: Sigma_y[name].tolist() for name in ['healthy', 'disease']
        },
    }

    # Add J-matrix provenance
    _csv_path = _args.j_matrix or _default_csv_path()
    j_spec = JMatrixSpec.from_csv(_csv_path)
    full_results['j_matrix'] = j_spec.to_dict()

    # --- Save JSON ---
    json_path = os.path.join(OUTPUT_DIR, 'elsa_ici_deployment.json')
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    # --- Save manuscript table ---
    table_path = os.path.join(OUTPUT_DIR, 'elsa_ici_deployment_table.txt')
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("ELSA ICI Deployment Assessment — Summary Table\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Longitudinal Panel\n")
        f.write(f"   N (≥2 waves):  {headline['N_panel']:,}\n")
        f.write(f"   N (≥3 waves):  {headline['N_3plus_waves']:,}\n")
        f.write(f"   N (4 waves):   {headline['N_4_waves']:,}\n\n")

        f.write("2. Basin Separability\n")
        f.write(f"   d_min:         {headline['d_min']:.4f}\n")
        f.write(f"   d(H→D):        {headline['d_hd']:.4f}\n")
        f.write(f"   d(D→H):        {headline['d_dh']:.4f}\n\n")

        f.write("3. Deployment Threshold (μ̄ = 0.05)\n")
        f.write(f"   ω_min:         {headline['omega_min_mu005']:.4f}\n")
        f.write(f"   ELSA fraction: {headline['frac_meets_ELSA']*100:.1f}%\n\n")

        f.write("4. ω_min Across μ̄ Values\n")
        f.write(f"   {'μ̄':>8s} {'ω_min':>10s} {'ELSA %':>10s}\n")
        f.write(f"   {'-'*30}\n")
        for mu_str, ores in omega_results.items():
            f.write(f"   {float(mu_str):>8.2f} {ores['omega_min']:>10.4f} "
                    f"{ores['frac_meets']*100:>9.1f}%\n")
        f.write("\n")

        f.write("5. Sensing Scenario Comparison\n")
        f.write(f"   {'Scenario':<20s} {'f_s':>8s} {'p_miss':>8s} "
                f"{'ρ':>6s} {'T_deploy':>12s}\n")
        f.write(f"   {'-'*56}\n")
        for name, sc in scenarios.items():
            T_str = (f"{sc['T_deploy_months']:.1f} mo"
                     if sc['T_deploy_months'] < 12000 else ">>1000 yr")
            p_miss_str = (f"{sc['p_miss']:.2f}"
                          if sc['p_miss'] is not None else "varies")
            f.write(f"   {name:<20s} {sc['f_s']:>8.2f} {p_miss_str:>8s} "
                    f"{sc['rho']:>6.2f} {T_str:>12s}\n")
        f.write("\n")

        f.write("6. R-Matrix Sensitivity\n")
        for key, rs in R_sensitivity.items():
            f.write(f"   R×{rs['scale']}: d_min = {rs['d_min']:.4f}\n")
        f.write("\n")

        f.write("7. Basin Assignment Sensitivity\n")
        for method, bs in basin_sensitivity.items():
            omega = compute_omega_min(bs['d_min'], mu_bar_primary)
            f.write(f"   {method}: d_min = {bs['d_min']:.4f}, "
                    f"ω_min = {omega:.4f}\n")

    print(f"  Table saved to: {table_path}")

    # --- Print headline summary ---
    print("\n" + "=" * 70)
    print("HEADLINE SUMMARY")
    print("=" * 70)
    print(f"  N participants (≥2 waves):     {headline['N_panel']:,}")
    print(f"  d_min:                         {headline['d_min']:.4f}")
    print(f"  ω_min (μ̄=0.05):               {headline['omega_min_mu005']:.4f}")
    print(f"  ELSA fraction meeting thresh:  {headline['frac_meets_ELSA']*100:.1f}%")
    print(f"  Daily wearable T_deploy:       "
          f"{headline['T_deploy_daily_months']:.1f} months")
    print(f"  Weekly wearable T_deploy:      "
          f"{headline['T_deploy_weekly_months']:.1f} months")

    return full_results


if __name__ == '__main__':
    main()
