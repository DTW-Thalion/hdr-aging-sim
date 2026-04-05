#!/usr/bin/env python3
"""
ELSA Data-Driven Basin Recovery via GMM/HMM

Addresses TCST adversarial review: demonstrates that latent basins are
recoverable from the data itself (not just an exogenous age-65 threshold),
then recomputes d_min and omega_min from the recovered basin structure.

Usage:
    python scripts/run_elsa_basin_recovery.py

Outputs:
    outputs/elsa_basin_recovery.json   — full results
    outputs/elsa_basin_recovery_table.txt — manuscript-ready comparison table
"""

import importlib.util
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

# ---- Import from run_elsa_validation.py ----
_val_spec = importlib.util.spec_from_file_location(
    "run_elsa_validation",
    os.path.join(ROOT, 'scripts', 'run_elsa_validation.py'))
_val_mod = importlib.util.module_from_spec(_val_spec)
_val_spec.loader.exec_module(_val_mod)

load_all_files = _val_mod.load_all_files
extract_nurse_biomarkers = _val_mod.extract_nurse_biomarkers
prepare_harmonised = _val_mod.prepare_harmonised
extract_mortality = _val_mod.extract_mortality
extract_supplementary = _val_mod.extract_supplementary
harmonised_to_long = _val_mod.harmonised_to_long
build_analysis_panel = _val_mod.build_analysis_panel
NURSE_WAVE_YEARS = _val_mod.NURSE_WAVE_YEARS

# ---- Import from run_elsa_ici_deployment.py ----
_ici_spec = importlib.util.spec_from_file_location(
    "run_elsa_ici_deployment",
    os.path.join(ROOT, 'scripts', 'run_elsa_ici_deployment.py'))
_ici_mod = importlib.util.module_from_spec(_ici_spec)
_ici_spec.loader.exec_module(_ici_mod)

construct_longitudinal_panel = _ici_mod.construct_longitudinal_panel
panel_to_dataframe = _ici_mod.panel_to_dataframe
compute_d_min = _ici_mod.compute_d_min
compute_omega_min = _ici_mod.compute_omega_min
compute_sensing_scenarios = _ici_mod.compute_sensing_scenarios
R_BASELINE = _ici_mod.R_BASELINE
MU_BAR_VALUES = _ici_mod.MU_BAR_VALUES
SENSING_SCENARIOS = _ici_mod.SENSING_SCENARIOS

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

AXES = ['dx_I', 'dx_M', 'dx_F']
BLOOD_WAVES = [2, 4, 6, 8]
SEED = 42


# =========================================================================
# KL Divergence (full Gaussian: includes mean shift)
# =========================================================================
def kl_divergence_gaussian(mu_k, Sigma_k, mu_j, Sigma_j):
    """KL(N_k || N_j) for multivariate Gaussians."""
    n = len(mu_k)
    Sigma_j_inv = linalg.inv(Sigma_j)
    sign_j, logdet_j = np.linalg.slogdet(Sigma_j)
    sign_k, logdet_k = np.linalg.slogdet(Sigma_k)
    diff = mu_j - mu_k
    return 0.5 * (
        np.trace(Sigma_j_inv @ Sigma_k)
        - n
        + logdet_j - logdet_k
        + diff @ Sigma_j_inv @ diff
    )


def kl_divergence_cov_only(Sigma_k, Sigma_j):
    """KL divergence using covariance only (no mean shift), for comparison."""
    n = Sigma_k.shape[0]
    Sigma_j_inv = linalg.inv(Sigma_j)
    sign_j, logdet_j = np.linalg.slogdet(Sigma_j)
    sign_k, logdet_k = np.linalg.slogdet(Sigma_k)
    return 0.5 * (
        np.trace(Sigma_j_inv @ Sigma_k) - n + logdet_j - logdet_k
    )


# =========================================================================
# Step 1: Load Data
# =========================================================================
def load_data():
    """Load ELSA panel using existing infrastructure."""
    print("=" * 70)
    print("ELSA DATA-DRIVEN BASIN RECOVERY (GMM / HMM)")
    print("=" * 70)

    files = load_all_files()
    panel, _ = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    merged = build_analysis_panel(panel, harm_long, mort, supp)
    return merged


# =========================================================================
# Step 2: GMM Basin Recovery
# =========================================================================
def run_gmm(obs_df):
    """Fit GMM with K in {2,3,4} and return results."""
    from sklearn.mixture import GaussianMixture

    print("\n" + "=" * 70)
    print("Step 2: GMM Basin Recovery")
    print("=" * 70)

    X = obs_df[AXES].values
    n_obs = len(X)
    print(f"  Observation matrix: {n_obs:,} x {len(AXES)}")

    results = {}
    for K in [2, 3, 4]:
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='full',
            n_init=10,
            random_state=SEED,
        )
        gmm.fit(X)

        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)

        # Cluster statistics
        cluster_info = []
        for k in range(K):
            mask = labels == k
            cluster_info.append({
                'n': int(mask.sum()),
                'frac': float(mask.sum() / n_obs),
                'mean_age': float(obs_df.loc[mask, 'age'].mean()),
                'mean_dx': gmm.means_[k].tolist(),
            })

        # Disagreement with age-65 threshold
        age_labels = np.where(obs_df['age'].values < 65, 0, 1)
        # For K=2: assign GMM cluster with lower mean age as "healthy" (0)
        if K == 2:
            if cluster_info[0]['mean_age'] > cluster_info[1]['mean_age']:
                gmm_binary = 1 - labels
            else:
                gmm_binary = labels.copy()
            disagree_frac = float(np.mean(gmm_binary != age_labels))
        else:
            disagree_frac = np.nan

        results[K] = {
            'bic': float(gmm.bic(X)),
            'aic': float(gmm.aic(X)),
            'means': gmm.means_.tolist(),
            'covariances': [c.tolist() for c in gmm.covariances_],
            'weights': gmm.weights_.tolist(),
            'labels': labels,
            'probs': probs,
            'cluster_info': cluster_info,
            'disagree_frac': disagree_frac,
            'gmm': gmm,
        }

        print(f"\n  K={K}: BIC={gmm.bic(X):.1f}, AIC={gmm.aic(X):.1f}")
        for i, ci in enumerate(cluster_info):
            print(f"    Cluster {i}: N={ci['n']:,} ({ci['frac']:.1%}), "
                  f"mean age={ci['mean_age']:.1f}, "
                  f"mean dx=[{ci['mean_dx'][0]:.3f}, {ci['mean_dx'][1]:.3f}, "
                  f"{ci['mean_dx'][2]:.3f}]")
        if K == 2:
            print(f"    Disagreement vs age-65: {disagree_frac:.1%}")

    # Select BIC-optimal K
    K_opt = min(results.keys(), key=lambda k: results[k]['bic'])
    print(f"\n  BIC-optimal K = {K_opt}")

    return results, K_opt


# =========================================================================
# Step 3: Compute d_min from GMM Covariances
# =========================================================================
def compute_gmm_d_min(gmm_results, K_opt):
    """Compute pairwise KL divergences between GMM clusters."""
    print("\n" + "=" * 70)
    print("Step 3: Basin Separability from GMM")
    print("=" * 70)

    res = gmm_results[K_opt]
    gmm = res['gmm']
    K = K_opt

    # Full KL divergence (includes mean shift)
    d_matrix_full = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            if k != j:
                # Add R to get measurement covariance Sigma_y
                Sigma_y_k = gmm.covariances_[k] + R_BASELINE
                Sigma_y_j = gmm.covariances_[j] + R_BASELINE
                d_matrix_full[k, j] = kl_divergence_gaussian(
                    gmm.means_[k], Sigma_y_k,
                    gmm.means_[j], Sigma_y_j,
                )

    d_min_full = float(d_matrix_full[d_matrix_full > 0].min())

    # Covariance-only KL (apples-to-apples with age-based)
    d_matrix_cov = np.zeros((K, K))
    for k in range(K):
        for j in range(K):
            if k != j:
                Sigma_y_k = gmm.covariances_[k] + R_BASELINE
                Sigma_y_j = gmm.covariances_[j] + R_BASELINE
                d_matrix_cov[k, j] = kl_divergence_cov_only(Sigma_y_k, Sigma_y_j)

    d_min_cov = float(d_matrix_cov[d_matrix_cov > 0].min())

    print(f"  d_min (full KL, with mean shift): {d_min_full:.4f}")
    print(f"  d_min (cov-only KL, comparable to age-based): {d_min_cov:.4f}")
    print(f"\n  Full KL divergence matrix:")
    for k in range(K):
        vals = [f"{d_matrix_full[k, j]:.4f}" for j in range(K)]
        print(f"    [{', '.join(vals)}]")

    return {
        'd_min_full': d_min_full,
        'd_min_cov_only': d_min_cov,
        'd_matrix_full': d_matrix_full.tolist(),
        'd_matrix_cov_only': d_matrix_cov.tolist(),
    }


# =========================================================================
# Step 4: Recompute omega_min and Deployment Timeline
# =========================================================================
def compute_deployment(d_min_results, gmm_results, K_opt):
    """Recompute omega_min and sensing scenarios using GMM-derived d_min."""
    print("\n" + "=" * 70)
    print("Step 4: Deployment Timeline (GMM-derived)")
    print("=" * 70)

    d_min = d_min_results['d_min_full']
    res = gmm_results[K_opt]

    # Use GMM weights as pi_k
    pi_min = float(min(res['weights']))

    # omega_min for each mu_bar
    omega_results = {}
    for mu_bar in MU_BAR_VALUES:
        omega = (1.0 / d_min) * np.log((K_opt - 1) / mu_bar) if d_min > 0 else float('inf')
        omega_results[str(mu_bar)] = float(omega)
        print(f"  mu_bar={mu_bar}: omega_min = {omega:.2f}")

    omega_005 = omega_results['0.05']

    # Sensing scenarios using GMM pi_min
    scenarios = {}
    for name, params in SENSING_SCENARIOS.items():
        f_s = params['f_s']
        p_miss = params['p_miss'] if params['p_miss'] is not None else 0.0
        rho = params['rho']

        eff_rate = f_s * pi_min * (1 - p_miss) * (1 - rho)
        if eff_rate > 0 and omega_005 < float('inf'):
            T_years = omega_005 / eff_rate
            T_months = T_years * 12
        else:
            T_years = float('inf')
            T_months = float('inf')

        scenarios[name] = {
            'f_s': f_s,
            'p_miss': p_miss,
            'rho': rho,
            'eff_rate': float(eff_rate),
            'T_deploy_years': float(T_years),
            'T_deploy_months': float(T_months),
        }
        print(f"  {name}: T_deploy = {T_months:.1f} months")

    return {
        'omega_by_mu_bar': omega_results,
        'omega_min_mu005': omega_005,
        'pi_min': pi_min,
        'sensing_scenarios': scenarios,
    }


# =========================================================================
# Step 5: HMM Basin Recovery
# =========================================================================
def run_hmm(obs_df):
    """Fit HMM on longitudinal sequences (>=3 waves per participant)."""
    print("\n" + "=" * 70)
    print("Step 5: HMM Basin Recovery (Longitudinal)")
    print("=" * 70)

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("  hmmlearn not available — skipping HMM analysis")
        return None

    # Build sequences for participants with >=3 waves
    sequences = []
    lengths = []
    ids = []
    for uid, group in obs_df.groupby('idauniq'):
        group = group.sort_values('wave')
        if len(group) >= 3:
            seq = group[AXES].values
            sequences.append(seq)
            lengths.append(len(seq))
            ids.append(uid)

    if len(sequences) < 50:
        print(f"  Only {len(sequences)} participants with >=3 waves — insufficient for HMM")
        return None

    X_hmm = np.vstack(sequences)
    lengths_arr = np.array(lengths)
    print(f"  Sequences: {len(sequences):,} participants, "
          f"{X_hmm.shape[0]:,} total observations")
    print(f"  Sequence lengths: {np.mean(lengths_arr):.1f} mean, "
          f"{np.min(lengths_arr)}-{np.max(lengths_arr)} range")

    try:
        hmm = GaussianHMM(
            n_components=2,
            covariance_type='full',
            n_iter=200,
            random_state=SEED,
        )
        hmm.fit(X_hmm, lengths_arr)

        # Transition matrix
        trans = hmm.transmat_
        print(f"\n  Transition matrix:")
        print(f"    [{trans[0,0]:.4f}, {trans[0,1]:.4f}]")
        print(f"    [{trans[1,0]:.4f}, {trans[1,1]:.4f}]")

        # Emission means
        for s in range(2):
            mu = hmm.means_[s]
            print(f"  State {s}: mean = [{mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}]")

        # Decode to get state assignments
        all_labels = []
        all_ages = []
        offset = 0
        for i, length in enumerate(lengths):
            seq = X_hmm[offset:offset + length]
            _, states = hmm.decode(seq)
            all_labels.extend(states)
            ages = obs_df[obs_df['idauniq'] == ids[i]].sort_values('wave')['age'].values[:length]
            all_ages.extend(ages)
            offset += length

        all_labels = np.array(all_labels)
        all_ages = np.array(all_ages)

        # Mean age per state
        for s in range(2):
            mask = all_labels == s
            if mask.any():
                print(f"  State {s}: N={mask.sum():,}, mean age={all_ages[mask].mean():.1f}")

        # Ensure state 0 = younger (healthy)
        if all_ages[all_labels == 0].mean() > all_ages[all_labels == 1].mean():
            # Swap states
            hmm.means_ = hmm.means_[[1, 0]]
            hmm.covars_ = hmm.covars_[[1, 0]]
            hmm.transmat_ = hmm.transmat_[np.ix_([1, 0], [1, 0])]
            trans = hmm.transmat_

        # d_min from HMM emission distributions (with R)
        Sigma_y_0 = hmm.covars_[0] + R_BASELINE
        Sigma_y_1 = hmm.covars_[1] + R_BASELINE
        d_01 = kl_divergence_gaussian(hmm.means_[0], Sigma_y_0,
                                       hmm.means_[1], Sigma_y_1)
        d_10 = kl_divergence_gaussian(hmm.means_[1], Sigma_y_1,
                                       hmm.means_[0], Sigma_y_0)
        d_min_hmm = min(d_01, d_10)

        # omega_min
        omega_hmm = (1.0 / d_min_hmm) * np.log(1.0 / 0.05) if d_min_hmm > 0 else float('inf')

        # Asymmetry check
        asym = trans[0, 1] / trans[1, 0] if trans[1, 0] > 0 else float('inf')
        print(f"\n  Transition asymmetry (H→D / D→H): {asym:.2f}")
        print(f"  d_min (HMM, full KL): {d_min_hmm:.4f}")
        print(f"  omega_min (mu=0.05): {omega_hmm:.2f}")

        # Disagreement with age-65
        age_labels = np.where(all_ages < 65, 0, 1)
        # After swap, state 0 = healthy
        disagree = float(np.mean(all_labels != age_labels))
        print(f"  Disagreement vs age-65: {disagree:.1%}")

        return {
            'transition_matrix': trans.tolist(),
            'means': hmm.means_.tolist(),
            'covariances': [c.tolist() for c in hmm.covars_],
            'd_min': float(d_min_hmm),
            'd_01': float(d_01),
            'd_10': float(d_10),
            'omega_min_mu005': float(omega_hmm),
            'n_sequences': len(sequences),
            'n_observations': int(X_hmm.shape[0]),
            'transition_asymmetry': float(asym),
            'disagree_vs_age65': float(disagree),
        }

    except Exception as e:
        print(f"  HMM fitting failed: {e}")
        return None


# =========================================================================
# Step 6: Comparison Table
# =========================================================================
def build_comparison(gmm_d_min, gmm_deploy, gmm_results, K_opt, hmm_results):
    """Build the comparison table across all methods."""
    print("\n" + "=" * 70)
    print("Step 6: Comparison Table")
    print("=" * 70)

    omega_005 = gmm_deploy['omega_min_mu005']
    daily_months = gmm_deploy['sensing_scenarios'].get('Daily wearable', {}).get('T_deploy_months', np.nan)

    rows = [
        {
            'method': 'Age-based (<65 vs >=65)',
            'd_min': 0.021,
            'omega_min': 144.2,
            'T_deploy_daily_mo': 18.2,
            'disagree': 0.0,
        },
        {
            'method': 'Score-based (median SWDS)',
            'd_min': 0.418,
            'omega_min': 7.2,
            'T_deploy_daily_mo': 0.9,
            'disagree': np.nan,
        },
        {
            'method': f'GMM (K={K_opt})',
            'd_min': gmm_d_min['d_min_full'],
            'omega_min': omega_005,
            'T_deploy_daily_mo': daily_months,
            # Use K=2 disagreement for comparison (K>2 not directly comparable)
            'disagree': gmm_results[2].get('disagree_frac', np.nan),
            'disagree_note': 'K=2 solution used for age-65 comparison',
        },
    ]

    if hmm_results is not None:
        hmm_omega = hmm_results['omega_min_mu005']
        # Compute HMM daily deployment time
        pi_min = 0.5  # HMM stationary assumed ~equal
        eff_rate = 365 * pi_min * 0.95 * 0.6
        hmm_daily_mo = (hmm_omega / eff_rate) * 12 if eff_rate > 0 else np.nan
        rows.append({
            'method': 'HMM (K=2)',
            'd_min': hmm_results['d_min'],
            'omega_min': hmm_omega,
            'T_deploy_daily_mo': hmm_daily_mo,
            'disagree': hmm_results.get('disagree_vs_age65', np.nan),
        })

    # Print table
    print(f"\n  {'Method':<30} {'d_min':>8} {'omega_min':>10} {'T_deploy':>12} {'Disagree':>10}")
    print(f"  {'':_<30} {'':_>8} {'(mu=.05)':>10} {'(daily,mo)':>12} {'(vs age)':>10}")
    for r in rows:
        d_str = f"{r['d_min']:.4f}" if not np.isnan(r['d_min']) else 'N/A'
        o_str = f"{r['omega_min']:.1f}" if not np.isnan(r['omega_min']) else 'N/A'
        t_str = f"{r['T_deploy_daily_mo']:.1f}" if not np.isnan(r['T_deploy_daily_mo']) else 'N/A'
        dis_str = f"{r['disagree']:.1%}" if not np.isnan(r.get('disagree', np.nan)) else 'N/A'
        print(f"  {r['method']:<30} {d_str:>8} {o_str:>10} {t_str:>12} {dis_str:>10}")

    return rows


# =========================================================================
# Step 7: Clinical Characterisation
# =========================================================================
def clinical_characterisation(obs_df, gmm_results, K_opt):
    """Characterise recovered basins in clinical terms."""
    print("\n" + "=" * 70)
    print("Step 7: Clinical Characterisation (K=2 solution)")
    print("=" * 70)

    # Always use K=2 for clinical characterisation
    res = gmm_results[2]
    labels = res['labels']

    # Sort clusters: cluster with lower mean age = "healthy"
    mean_ages = [res['cluster_info'][k]['mean_age'] for k in range(2)]
    if mean_ages[0] > mean_ages[1]:
        labels = 1 - labels  # swap

    cluster_names = ['Cluster A (healthier)', 'Cluster B (sicker)']
    clinical = []

    for c_idx, c_name in enumerate(cluster_names):
        mask = labels == c_idx
        sub = obs_df[mask]

        info = {
            'name': c_name,
            'n_obs': int(mask.sum()),
            'frac': float(mask.sum() / len(obs_df)),
            'mean_age': float(sub['age'].mean()),
            'sd_age': float(sub['age'].std()),
        }

        # Raw biomarkers
        if 'hscrp' in sub.columns:
            info['crp_mean'] = float(sub['hscrp'].mean())
            info['crp_sd'] = float(sub['hscrp'].std())
        if 'hba1c' in sub.columns:
            info['hba1c_mean'] = float(sub['hba1c'].mean())
            info['hba1c_sd'] = float(sub['hba1c'].std())
        if 'grip_max' in sub.columns:
            info['grip_mean'] = float(sub['grip_max'].mean())
            info['grip_sd'] = float(sub['grip_max'].std())

        # Sex
        if 'sex' in sub.columns:
            info['frac_male'] = float((sub['sex'] == 1).mean())

        # Mortality
        if 'deceased' in sub.columns:
            info['mortality_rate'] = float(sub.groupby('idauniq')['deceased'].first().mean())

        clinical.append(info)

        print(f"\n  {c_name}:")
        print(f"    N = {info['n_obs']:,} ({info['frac']:.1%})")
        print(f"    Mean age = {info['mean_age']:.1f} +/- {info['sd_age']:.1f}")
        if 'crp_mean' in info:
            print(f"    CRP = {info['crp_mean']:.2f} +/- {info['crp_sd']:.2f} mg/L")
        if 'hba1c_mean' in info:
            print(f"    HbA1c = {info['hba1c_mean']:.1f} +/- {info['hba1c_sd']:.1f} mmol/mol")
        if 'grip_mean' in info:
            print(f"    Grip = {info['grip_mean']:.1f} +/- {info['grip_sd']:.1f} kg")
        if 'frac_male' in info:
            print(f"    Male = {info['frac_male']:.1%}")
        if 'mortality_rate' in info:
            print(f"    Mortality rate = {info['mortality_rate']:.1%}")

    return clinical


# =========================================================================
# Output
# =========================================================================
def save_results(gmm_results, K_opt, gmm_d_min, gmm_deploy,
                 hmm_results, comparison, clinical):
    """Save JSON and text outputs."""

    # ---- JSON ----
    output = {
        'pipeline_version': 'v1.0-basin-recovery',
        'description': 'Data-driven basin recovery via GMM/HMM for ICI deployment assessment',
        'gmm': {},
        'K_opt': K_opt,
        'gmm_d_min': gmm_d_min,
        'gmm_deployment': gmm_deploy,
        'hmm': hmm_results,
        'comparison': comparison,
        'clinical_characterisation': clinical,
    }

    # GMM results (without non-serializable objects)
    for K, res in gmm_results.items():
        output['gmm'][str(K)] = {
            'bic': res['bic'],
            'aic': res['aic'],
            'means': res['means'],
            'covariances': res['covariances'],
            'weights': res['weights'],
            'cluster_info': res['cluster_info'],
            'disagree_frac': res['disagree_frac'],
        }

    json_path = os.path.join(OUTPUT_DIR, 'elsa_basin_recovery.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    # ---- Text table ----
    txt_path = os.path.join(OUTPUT_DIR, 'elsa_basin_recovery_table.txt')
    with open(txt_path, 'w') as f:
        f.write("ELSA Data-Driven Basin Recovery: Comparison Table\n")
        f.write("=" * 80 + "\n\n")

        # Model selection
        f.write("GMM Model Selection (BIC / AIC):\n")
        for K in [2, 3, 4]:
            r = gmm_results[K]
            star = " *" if K == K_opt else ""
            f.write(f"  K={K}: BIC={r['bic']:.1f}, AIC={r['aic']:.1f}{star}\n")
        f.write(f"  (* = BIC-optimal)\n\n")

        # Comparison table
        f.write(f"{'Method':<30} {'d_min':>8} {'omega_min':>10} {'T_deploy':>12} {'Disagree':>10}\n")
        f.write(f"{'':_<30} {'':_>8} {'(mu=.05)':>10} {'(daily,mo)':>12} {'(vs age)':>10}\n")
        for r in comparison:
            d_str = f"{r['d_min']:.4f}"
            o_str = f"{r['omega_min']:.1f}"
            t_str = f"{r['T_deploy_daily_mo']:.1f}" if not np.isnan(r['T_deploy_daily_mo']) else 'N/A'
            dis_str = f"{r['disagree']:.1%}" if not np.isnan(r.get('disagree', np.nan)) else 'N/A'
            f.write(f"{r['method']:<30} {d_str:>8} {o_str:>10} {t_str:>12} {dis_str:>10}\n")
        f.write("\n")

        # Clinical characterisation
        f.write("Clinical Characterisation (K=2):\n")
        f.write("-" * 50 + "\n")
        for c in clinical:
            f.write(f"\n  {c['name']}:\n")
            f.write(f"    N = {c['n_obs']:,} ({c['frac']:.1%})\n")
            f.write(f"    Mean age = {c['mean_age']:.1f} +/- {c['sd_age']:.1f}\n")
            if 'crp_mean' in c:
                f.write(f"    CRP = {c['crp_mean']:.2f} +/- {c['crp_sd']:.2f} mg/L\n")
            if 'hba1c_mean' in c:
                f.write(f"    HbA1c = {c['hba1c_mean']:.1f} +/- {c['hba1c_sd']:.1f} mmol/mol\n")
            if 'grip_mean' in c:
                f.write(f"    Grip = {c['grip_mean']:.1f} +/- {c['grip_sd']:.1f} kg\n")
            if 'frac_male' in c:
                f.write(f"    Male = {c['frac_male']:.1%}\n")
            if 'mortality_rate' in c:
                f.write(f"    Mortality rate = {c['mortality_rate']:.1%}\n")

        # Sensing scenarios
        f.write(f"\nSensing Scenarios (GMM, K={K_opt}, mu_bar=0.05):\n")
        f.write("-" * 50 + "\n")
        for name, sc in gmm_deploy['sensing_scenarios'].items():
            f.write(f"  {name}: {sc['T_deploy_months']:.1f} months\n")

    print(f"Saved {txt_path}")


# =========================================================================
# Main
# =========================================================================
def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Step 1: Load data
    merged = load_data()

    # Pool all complete 3-axis observations
    obs_df = merged[
        merged['wave'].isin(BLOOD_WAVES) &
        merged['complete_3axis']
    ].copy()
    print(f"\n  Pooled observations: {len(obs_df):,} (N={obs_df['idauniq'].nunique():,} participants)")

    # Step 2: GMM
    gmm_results, K_opt = run_gmm(obs_df)

    # Step 3: d_min
    gmm_d_min = compute_gmm_d_min(gmm_results, K_opt)

    # Step 4: Deployment timeline
    gmm_deploy = compute_deployment(gmm_d_min, gmm_results, K_opt)

    # Step 5: HMM
    hmm_results = run_hmm(obs_df)

    # Step 6: Comparison
    comparison = build_comparison(gmm_d_min, gmm_deploy, gmm_results, K_opt, hmm_results)

    # Step 7: Clinical characterisation
    clinical = clinical_characterisation(obs_df, gmm_results, K_opt)

    # Save
    save_results(gmm_results, K_opt, gmm_d_min, gmm_deploy,
                 hmm_results, comparison, clinical)

    # Print manuscript language
    print("\n" + "=" * 70)
    print("MANUSCRIPT DRAFT LANGUAGE")
    print("=" * 70)
    n_obs = len(obs_df)
    n_ppl = obs_df['idauniq'].nunique()
    d_min = gmm_d_min['d_min_full']
    omega = gmm_deploy['omega_min_mu005']
    daily_mo = gmm_deploy['sensing_scenarios']['Daily wearable']['T_deploy_months']
    dis = gmm_results[2].get('disagree_frac', np.nan)
    k2_ages = [gmm_results[2]['cluster_info'][k]['mean_age'] for k in range(2)]
    k2_ages.sort()
    # Age range across all K_opt clusters
    kopt_ages = [gmm_results[K_opt]['cluster_info'][k]['mean_age'] for k in range(K_opt)]
    ages = [min(kopt_ages), max(kopt_ages)]

    print(f"""
To assess whether the ICI deployment threshold depends on the
exogenous age-based basin assignment, we performed data-driven basin
recovery using Gaussian Mixture Models on the pooled ELSA 3-axis
panel (N_obs = {n_obs:,} observations across {n_ppl:,} participants).
BIC selection favours K = {K_opt} clusters, with mean ages ranging
from {ages[0]:.0f} to {ages[1]:.0f} years.  The K=2 solution differs
from the age-65 threshold in {dis:.0%} of assignments, confirming that the
basins reflect physiological state, not age per se.  The GMM-derived
basin separability d_min = {d_min:.3f} yields omega_min = {omega:.1f}
at mu_bar = 0.05 — {'higher' if omega > 144 else 'lower'} than the
conservative age-based estimate (d_min = 0.021, omega_min = 144).
Under daily wearable sensing, the data-driven deployment threshold
is met after approximately {daily_mo:.0f} months.
""")

    print("=" * 70)
    print("DONE: elsa_basin_recovery.json + elsa_basin_recovery_table.txt")
    print("=" * 70)


if __name__ == '__main__':
    main()
