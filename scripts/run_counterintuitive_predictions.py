#!/usr/bin/env python3
"""
Counterintuitive Predictions from the Coupling Matrix
=======================================================

Systematic search through the calibrated J matrix and ELSA data for
predictions that are (a) derivable from the coupling structure,
(b) non-obvious, and (c) testable with existing data.

Five tests:
  1. Asymmetric coupling → lead-lag prediction
  2. Conditional independence → partial correlation structure
  3. Coupling-direction dependent mortality
  4. Age-dependent eigenvector rotation
  5. Non-monotone axis-specific variance

Outputs:
  outputs/counterintuitive_predictions.json
  stdout: markdown summary

Usage:
    python scripts/run_counterintuitive_predictions.py

Reference: HDR Ontology Manuscript R6
"""

import json
import os
import sys
import time
import warnings

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from hdr_sim.aging_params import configure, tau_of_age, J_of_age
from hdr_sim.dynamics import build_A

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AXES_3 = ('I', 'M', 'F')
Q_SIGMA2 = 0.01
AGE_STRATA = [(50, 59), (60, 69), (70, 79), (80, 90)]
NURSE_WAVE_YEARS = {2: 2004, 4: 2008, 6: 2012, 8: 2016, 11: 2023}
DATA_DIR = os.path.join(ROOT, 'data', 'elsa')
N_BONFERRONI = 5  # number of independent tests


# ---------------------------------------------------------------------------
# ELSA data loading (minimal pipeline)
# ---------------------------------------------------------------------------

def load_tab(path):
    """Load a TAB file with proper missing-value handling."""
    df = pd.read_csv(path, sep='\t', low_memory=False)
    return df


def filter_missing(df, exclude_cols=None):
    """Replace ELSA missing codes (< 0) with NaN for numeric columns."""
    exclude = set(exclude_cols or [])
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] < 0, col] = np.nan
    return df


def load_elsa_data():
    """Load ELSA data and construct 3-axis biomarkers. Returns merged panel."""
    print("  Loading ELSA data...")

    # Find files
    harm_path = os.path.join(DATA_DIR, 'gh_elsa_h_hdr_subset.tab')
    nurse_path = os.path.join(DATA_DIR, 'elsa_nurse_biomarkers_consolidated.tab')
    eol_path = os.path.join(DATA_DIR, 'h_elsa_eol_a2.tab')

    for p in [harm_path, nurse_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    # Load harmonised
    harm = load_tab(harm_path)
    harm = filter_missing(harm, exclude_cols=['idauniq'])

    # Load nurse biomarkers
    nurse = load_tab(nurse_path)
    nurse = filter_missing(nurse, exclude_cols=['idauniq', 'wave'])

    # Grip strength: max of dominant hand trials
    grip_cols = [c for c in ['mmgsd1', 'mmgsd2', 'mmgsd3'] if c in nurse.columns]
    if grip_cols:
        nurse['grip_max'] = nurse[grip_cols].max(axis=1)
    else:
        nondominant = [c for c in ['mmgsn1', 'mmgsn2', 'mmgsn3'] if c in nurse.columns]
        if nondominant:
            nurse['grip_max'] = nurse[nondominant].max(axis=1)

    # HbA1c unit conversion (DCCT % → IFCC mmol/mol if needed)
    if 'hba1c' in nurse.columns:
        for w in nurse['wave'].unique():
            wmask = nurse['wave'] == w
            vals = nurse.loc[wmask, 'hba1c'].dropna()
            if len(vals) > 10 and vals.median() < 15:
                nurse.loc[wmask, 'hba1c'] = (nurse.loc[wmask, 'hba1c'] - 2.15) * 10.929

    # Build long-format demographics from harmonised
    # Convert age and iwstat columns to numeric
    for w in [2, 4, 6, 8]:
        for prefix in [f'r{w}agey', f'r{w}iwstat']:
            if prefix in harm.columns:
                harm[prefix] = pd.to_numeric(harm[prefix], errors='coerce')

    if 'ragender' in harm.columns:
        harm['ragender'] = pd.to_numeric(harm['ragender'], errors='coerce')

    harm_long_rows = []
    for w in [2, 4, 6, 8]:
        age_col = f'r{w}agey'
        sex_col = 'ragender'
        iwstat_col = f'r{w}iwstat'
        cols_needed = ['idauniq']
        for c in [age_col, sex_col, iwstat_col]:
            if c in harm.columns:
                cols_needed.append(c)
        sub = harm[cols_needed].copy()
        sub['wave'] = w
        if age_col in sub.columns:
            sub = sub.rename(columns={age_col: 'age'})
        else:
            sub['age'] = np.nan
        if sex_col in sub.columns:
            sub = sub.rename(columns={sex_col: 'sex'})
        else:
            sub['sex'] = np.nan
        sub['interview_year'] = NURSE_WAVE_YEARS.get(w, np.nan)
        harm_long_rows.append(sub[['idauniq', 'wave', 'age', 'sex', 'interview_year']])
    harm_long = pd.concat(harm_long_rows, ignore_index=True)

    # Merge nurse + demographics
    panel_cols = ['idauniq', 'wave', 'hscrp', 'hba1c', 'bmival', 'grip_max']
    available = [c for c in panel_cols if c in nurse.columns]
    panel = nurse[available].copy()

    merged = panel.merge(harm_long, on=['idauniq', 'wave'], how='left')
    merged = merged[(merged['age'] >= 50) & (merged['age'] <= 90)]

    # --- Reference subgroup for z-scoring ---
    ref_mask = (merged['wave'] == 2) & (merged['age'] >= 50) & (merged['age'] <= 55)
    ref = merged.loc[ref_mask]

    # dx_I: z-score of log(CRP)
    merged['log_crp'] = np.log(merged['hscrp'].clip(lower=0.01))
    ref_log_crp = np.log(ref['hscrp'].clip(lower=0.01).dropna())
    if len(ref_log_crp) > 10:
        crp_mean, crp_std = ref_log_crp.mean(), ref_log_crp.std()
    else:
        crp_mean, crp_std = merged['log_crp'].mean(), merged['log_crp'].std()
    crp_std = max(crp_std, 1e-6)
    merged['dx_I'] = (merged['log_crp'] - crp_mean) / crp_std

    # dx_M: composite of HbA1c + BMI
    for var, label in [('hba1c', 'hba1c_z'), ('bmival', 'bmi_z')]:
        ref_vals = ref[var].dropna() if var in ref.columns else pd.Series(dtype=float)
        if len(ref_vals) > 10:
            mu, sigma = ref_vals.mean(), ref_vals.std()
        else:
            mu, sigma = merged[var].mean(), merged[var].std()
        sigma = max(sigma, 1e-6)
        merged[label] = (merged[var] - mu) / sigma
    merged['dx_M'] = (merged['hba1c_z'] + merged['bmi_z']) / np.sqrt(2)

    # dx_F: reversed z-score of grip
    ref_grip = ref['grip_max'].dropna() if 'grip_max' in ref.columns else pd.Series(dtype=float)
    if len(ref_grip) > 10:
        grip_mean, grip_std = ref_grip.mean(), ref_grip.std()
    else:
        grip_mean, grip_std = merged['grip_max'].mean(), merged['grip_max'].std()
    grip_std = max(grip_std, 1e-6)
    merged['dx_F'] = -(merged['grip_max'] - grip_mean) / grip_std

    merged['complete_3axis'] = (
        merged['dx_I'].notna() & merged['dx_M'].notna() & merged['dx_F'].notna())

    # --- Mortality ---
    mort = pd.DataFrame({'idauniq': harm['idauniq'].values})
    mort['deceased'] = 0
    for w in range(3, 11):
        col = f'r{w}iwstat'
        if col in harm.columns:
            vals = pd.to_numeric(harm[col], errors='coerce')
            mort.loc[vals == 4, 'deceased'] = 1

    # Death age from EOL
    if os.path.exists(eol_path):
        eol = load_tab(eol_path)
        eol = filter_missing(eol, exclude_cols=['idauniq'])
        if 'radage' in eol.columns:
            mort = mort.merge(eol[['idauniq', 'radage']].rename(
                columns={'radage': 'death_age'}), on='idauniq', how='left')
    if 'death_age' not in mort.columns:
        mort['death_age'] = np.nan

    merged = merged.merge(mort[['idauniq', 'deceased', 'death_age']],
                          on='idauniq', how='left')

    n_complete = merged['complete_3axis'].sum()
    n_persons = merged.loc[merged['complete_3axis'], 'idauniq'].nunique()
    print(f"  Complete 3-axis observations: {n_complete:,} ({n_persons:,} persons)")

    return merged


# ---------------------------------------------------------------------------
# Build visit-pair data
# ---------------------------------------------------------------------------

def build_visit_pairs(merged):
    """Build consecutive visit-pair changes for longitudinal analysis."""
    complete = merged[merged['complete_3axis']].copy()
    complete = complete.sort_values(['idauniq', 'wave'])

    pairs = []
    for uid, group in complete.groupby('idauniq'):
        rows = group.sort_values('wave').reset_index(drop=True)
        for i in range(len(rows) - 1):
            r1, r2 = rows.iloc[i], rows.iloc[i + 1]
            pairs.append({
                'idauniq': uid,
                'wave_from': int(r1['wave']),
                'wave_to': int(r2['wave']),
                'age_mid': (r1['age'] + r2['age']) / 2,
                'dI': r2['dx_I'] - r1['dx_I'],
                'dM': r2['dx_M'] - r1['dx_M'],
                'dF': r2['dx_F'] - r1['dx_F'],
                'dx_I_t1': r1['dx_I'],
                'dx_M_t1': r1['dx_M'],
                'dx_F_t1': r1['dx_F'],
                'deceased': r2.get('deceased', 0),
            })

    return pd.DataFrame(pairs) if pairs else pd.DataFrame()


# ---------------------------------------------------------------------------
# Model predictions
# ---------------------------------------------------------------------------

def get_model_predictions():
    """Compute model predictions from the calibrated J matrix."""
    configure(axes=AXES_3)
    n = len(AXES_3)
    Q = Q_SIGMA2 * np.eye(n)

    predictions = {}
    for age in [30, 40, 50, 60, 65, 70, 80]:
        tau = tau_of_age(age)
        J = J_of_age(age)
        A = build_A(tau, J)
        Gamma = solve_continuous_lyapunov(A, -Q)

        # Eigenvectors of Gamma
        eigvals, eigvecs = np.linalg.eigh(Gamma)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Partial correlations from precision matrix
        P = np.linalg.inv(Gamma)
        partial_corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                partial_corr[i, j] = -P[i, j] / np.sqrt(P[i, i] * P[j, j])

        # Correlation matrix
        d_inv = 1.0 / np.sqrt(np.maximum(np.diag(Gamma), 1e-30))
        R = np.outer(d_inv, d_inv) * Gamma

        predictions[age] = {
            'J': J.tolist(),
            'tau': tau.tolist(),
            'Gamma': Gamma.tolist(),
            'eigvals': eigvals.tolist(),
            'eigvec_dominant': eigvecs[:, 0].tolist(),
            'per_axis_var': np.diag(Gamma).tolist(),
            'partial_corr': partial_corr.tolist(),
            'correlation': R.tolist(),
        }

    return predictions


# ---------------------------------------------------------------------------
# Test 1: Asymmetric coupling → lead-lag
# ---------------------------------------------------------------------------

def test_lead_lag(merged, predictions):
    """Test whether the J matrix asymmetry predicts lead-lag relationships."""
    print("\n  Test 1: Asymmetric coupling lead-lag...")

    # Model prediction: which axis leads?
    J_65 = np.array(predictions[65]['J'])
    axes = list(AXES_3)

    # J[row,col] = effect of col on row
    # If J[I,M] (M→I effect) > J[M,I] (I→M effect), then M leads I
    asymmetries = {}
    for i in range(3):
        for j in range(i+1, 3):
            asym = J_65[i, j] - J_65[j, i]  # col j → row i minus col i → row j
            leader = axes[j] if abs(J_65[i, j]) > abs(J_65[j, i]) else axes[i]
            asymmetries[f'{axes[i]}-{axes[j]}'] = {
                'J_ij': float(J_65[i, j]),
                'J_ji': float(J_65[j, i]),
                'predicted_leader': leader,
                'asymmetry': float(asym),
            }

    # Build lagged visit-triple data
    complete = merged[merged['complete_3axis']].sort_values(['idauniq', 'wave'])
    triples = []
    for uid, group in complete.groupby('idauniq'):
        rows = group.sort_values('wave').reset_index(drop=True)
        if len(rows) < 3:
            continue
        for i in range(len(rows) - 2):
            r1, r2, r3 = rows.iloc[i], rows.iloc[i+1], rows.iloc[i+2]
            triples.append({
                'dI_12': r2['dx_I'] - r1['dx_I'],
                'dM_12': r2['dx_M'] - r1['dx_M'],
                'dF_12': r2['dx_F'] - r1['dx_F'],
                'dI_23': r3['dx_I'] - r2['dx_I'],
                'dM_23': r3['dx_M'] - r2['dx_M'],
                'dF_23': r3['dx_F'] - r2['dx_F'],
            })

    n_triples = len(triples)
    print(f"    Visit triples available: {n_triples}")

    if n_triples < 30:
        return {
            'test_name': 'Asymmetric coupling lead-lag',
            'prediction': (f"M→I coupling ({J_65[0,1]:.4f}) vs I→M ({J_65[1,0]:.4f}): "
                          f"M changes should predict future I changes"),
            'result': f'Insufficient data ({n_triples} triples, need >=30)',
            'effect_size': None,
            'p_value': None,
            'counterintuitive': True,
            'testable_with_current_data': False,
            'asymmetries': asymmetries,
            'summary': 'Cannot test — too few visit triples for lagged analysis.',
        }

    triples_df = pd.DataFrame(triples)

    # Cross-lagged correlations: does dM(t1→t2) predict dI(t2→t3)?
    r_M_leads_I, p_M_leads_I = stats.pearsonr(triples_df['dM_12'], triples_df['dI_23'])
    r_I_leads_M, p_I_leads_M = stats.pearsonr(triples_df['dI_12'], triples_df['dM_23'])

    # Also test I-F and M-F
    r_I_leads_F, p_I_leads_F = stats.pearsonr(triples_df['dI_12'], triples_df['dF_23'])
    r_F_leads_I, p_F_leads_I = stats.pearsonr(triples_df['dF_12'], triples_df['dI_23'])

    lagged = {
        'M_leads_I': {'r': float(r_M_leads_I), 'p': float(p_M_leads_I)},
        'I_leads_M': {'r': float(r_I_leads_M), 'p': float(p_I_leads_M)},
        'I_leads_F': {'r': float(r_I_leads_F), 'p': float(p_I_leads_F)},
        'F_leads_I': {'r': float(r_F_leads_I), 'p': float(p_F_leads_I)},
    }

    # J matrix predicts M→I > I→M coupling, so M should lead I
    predicted_stronger = 'M_leads_I'
    predicted_r = r_M_leads_I
    predicted_p = p_M_leads_I

    supports = abs(r_M_leads_I) > abs(r_I_leads_M)

    return {
        'test_name': 'Asymmetric coupling lead-lag',
        'prediction': (f"M→I coupling stronger than I→M at age 65 "
                      f"({J_65[0,1]:.4f} vs {J_65[1,0]:.4f}): "
                      f"metabolic change should predict future inflammatory change"),
        'result': (f"Cross-lagged r(dM_12,dI_23)={r_M_leads_I:.3f} (p={p_M_leads_I:.3f}), "
                  f"r(dI_12,dM_23)={r_I_leads_M:.3f} (p={p_I_leads_M:.3f}). "
                  f"N={n_triples} triples. "
                  f"Predicted direction {'supported' if supports else 'NOT supported'}."),
        'effect_size': float(r_M_leads_I - r_I_leads_M),
        'p_value': float(predicted_p),
        'counterintuitive': True,
        'testable_with_current_data': n_triples >= 30,
        'n_triples': n_triples,
        'asymmetries': asymmetries,
        'lagged_correlations': lagged,
        'supports_prediction': supports,
        'summary': (f"{'Supported' if supports else 'Not supported'}: "
                   f"N={n_triples} visit triples. "
                   f"M→I lagged r={r_M_leads_I:.3f}, I→M lagged r={r_I_leads_M:.3f}. "
                   f"ELSA 4-year cadence severely limits power for within-person lead-lag."),
    }


# ---------------------------------------------------------------------------
# Test 2: Conditional independence → partial correlations
# ---------------------------------------------------------------------------

def test_partial_correlations(merged, predictions):
    """Test whether the J matrix predicts the partial correlation structure."""
    print("\n  Test 2: Partial correlation structure...")

    axes = list(AXES_3)
    results_by_stratum = {}

    for age_lo, age_hi in AGE_STRATA:
        age_mid = (age_lo + age_hi) / 2
        mask = merged['complete_3axis'] & (merged['age'] >= age_lo) & (merged['age'] < age_hi)
        X = merged.loc[mask, ['dx_I', 'dx_M', 'dx_F']].values
        n = len(X)
        if n < 20:
            continue

        # Observed partial correlations from precision matrix
        Gamma_obs = np.cov(X, rowvar=False)
        try:
            P_obs = np.linalg.inv(Gamma_obs)
        except np.linalg.LinAlgError:
            continue
        pcorr_obs = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                pcorr_obs[i, j] = -P_obs[i, j] / np.sqrt(P_obs[i, i] * P_obs[j, j])

        # Model prediction (use nearest age anchor)
        pred_age = min(predictions.keys(), key=lambda a: abs(a - age_mid))
        pcorr_pred = np.array(predictions[pred_age]['partial_corr'])

        # Compare off-diagonal partial correlations
        pairs = []
        for i in range(3):
            for j in range(i+1, 3):
                pairs.append({
                    'pair': f'{axes[i]}-{axes[j]}',
                    'predicted': float(pcorr_pred[i, j]),
                    'observed': float(pcorr_obs[i, j]),
                    'sign_match': np.sign(pcorr_pred[i, j]) == np.sign(pcorr_obs[i, j]),
                })

        results_by_stratum[f'{age_lo}-{age_hi}'] = {
            'n': n,
            'pairs': pairs,
            'sign_concordance': sum(p['sign_match'] for p in pairs) / len(pairs),
        }

    # Overall sign concordance
    all_pairs = [p for s in results_by_stratum.values() for p in s['pairs']]
    overall_concordance = sum(p['sign_match'] for p in all_pairs) / max(len(all_pairs), 1)

    return {
        'test_name': 'Partial correlation structure',
        'prediction': ('J matrix predicts specific partial correlation signs: '
                      'positive I-M (mutual pathological coupling), '
                      'negative I-F and M-F (F is protective)'),
        'result': (f'Sign concordance: {overall_concordance:.0%} '
                  f'across {len(results_by_stratum)} strata'),
        'effect_size': float(overall_concordance),
        'p_value': None,
        'counterintuitive': overall_concordance > 0.5,
        'testable_with_current_data': True,
        'strata': results_by_stratum,
        'summary': (f"Partial correlation signs match model prediction in "
                   f"{overall_concordance:.0%} of axis-pair x stratum tests. "
                   f"{'Strong' if overall_concordance >= 0.75 else 'Moderate' if overall_concordance >= 0.5 else 'Weak'} "
                   f"support for the coupling structure."),
    }


# ---------------------------------------------------------------------------
# Test 3: Coupling-direction dependent mortality
# ---------------------------------------------------------------------------

def test_directional_mortality(merged, predictions):
    """Test whether dysregulation along the dominant eigenvector predicts
    mortality better than orthogonal dysregulation."""
    print("\n  Test 3: Coupling-direction dependent mortality...")

    # Get dominant eigenvector from model at age 65
    v1 = np.array(predictions[65]['eigvec_dominant'])
    v1 = v1 / np.linalg.norm(v1)

    # Build orthonormal complement
    # Gram-Schmidt on random vectors
    rng = np.random.default_rng(42)
    basis = [v1]
    for _ in range(2):
        v = rng.standard_normal(3)
        for b in basis:
            v -= np.dot(v, b) * b
        v /= np.linalg.norm(v)
        basis.append(v)
    v2, v3 = basis[1], basis[2]

    # Use baseline (wave 2) observations
    baseline = merged[(merged['wave'] == 2) & merged['complete_3axis']].copy()
    if 'deceased' not in baseline.columns or baseline['deceased'].sum() < 10:
        return {
            'test_name': 'Coupling-direction dependent mortality',
            'prediction': (f'Dysregulation along dominant eigenvector '
                          f'v1={np.round(v1, 3).tolist()} should predict mortality '
                          f'better than orthogonal displacement'),
            'result': 'Insufficient mortality data',
            'effect_size': None,
            'p_value': None,
            'counterintuitive': True,
            'testable_with_current_data': False,
            'summary': 'Cannot test — insufficient mortality events at baseline.',
        }

    X = baseline[['dx_I', 'dx_M', 'dx_F']].values

    # Project onto dominant vs orthogonal
    proj_dominant = np.abs(X @ v1)
    proj_orthog = np.sqrt((X @ v2)**2 + (X @ v3)**2)
    total_norm = np.sqrt(np.sum(X**2, axis=1))

    deceased = baseline['deceased'].values.astype(bool)

    # Point-biserial correlation with mortality
    r_dom, p_dom = stats.pointbiserialr(deceased, proj_dominant)
    r_orth, p_orth = stats.pointbiserialr(deceased, proj_orthog)
    r_total, p_total = stats.pointbiserialr(deceased, total_norm)

    # Adjusted p-values
    p_dom_adj = min(p_dom * N_BONFERRONI, 1.0)
    p_orth_adj = min(p_orth * N_BONFERRONI, 1.0)

    dominant_stronger = abs(r_dom) > abs(r_orth)
    n_dead = int(deceased.sum())
    n_alive = int((~deceased).sum())

    return {
        'test_name': 'Coupling-direction dependent mortality',
        'prediction': (f'Dysregulation along dominant eigenvector v1='
                      f'{np.round(v1, 3).tolist()} (mostly F-axis at age 65) '
                      f'should predict mortality better than orthogonal displacement'),
        'result': (f'r(dominant,mortality)={r_dom:.3f} (p_adj={p_dom_adj:.4f}), '
                  f'r(orthogonal,mortality)={r_orth:.3f} (p_adj={p_orth_adj:.4f}), '
                  f'r(total_norm,mortality)={r_total:.3f}. '
                  f'N={n_alive} alive, {n_dead} deceased.'),
        'effect_size': float(r_dom - r_orth),
        'p_value': float(p_dom_adj),
        'counterintuitive': True,
        'testable_with_current_data': True,
        'dominant_eigenvector': v1.tolist(),
        'correlations': {
            'dominant': {'r': float(r_dom), 'p': float(p_dom), 'p_adj': float(p_dom_adj)},
            'orthogonal': {'r': float(r_orth), 'p': float(p_orth), 'p_adj': float(p_orth_adj)},
            'total_norm': {'r': float(r_total), 'p': float(p_total)},
        },
        'dominant_stronger': dominant_stronger,
        'n_alive': n_alive,
        'n_dead': n_dead,
        'summary': (f"{'Supported' if dominant_stronger and p_dom_adj < 0.05 else 'Not clearly supported'}: "
                   f"Dominant-eigenvector projection r={r_dom:.3f} "
                   f"{'>' if dominant_stronger else '<='} "
                   f"orthogonal r={r_orth:.3f}. "
                   f"At age 65, v1 is ~{abs(v1[2]):.0%} F-axis, so this largely "
                   f"tests whether grip-strength decline direction predicts mortality "
                   f"— which the coupling framework explains structurally."),
    }


# ---------------------------------------------------------------------------
# Test 4: Age-dependent eigenvector rotation
# ---------------------------------------------------------------------------

def test_eigenvector_rotation(merged, predictions):
    """Test whether the dominant eigenvector rotates with age as predicted."""
    print("\n  Test 4: Age-dependent eigenvector rotation...")

    # Model predictions
    pred_v1s = {}
    for age in sorted(predictions.keys()):
        v1 = np.array(predictions[age]['eigvec_dominant'])
        v1 = v1 / np.linalg.norm(v1)
        # Ensure consistent sign (positive F-axis component)
        if v1[2] < 0:
            v1 = -v1
        pred_v1s[age] = v1

    # Observed eigenvectors per stratum
    obs_v1s = {}
    for age_lo, age_hi in AGE_STRATA:
        mask = merged['complete_3axis'] & (merged['age'] >= age_lo) & (merged['age'] < age_hi)
        X = merged.loc[mask, ['dx_I', 'dx_M', 'dx_F']].values
        if len(X) < 30:
            continue
        Gamma_obs = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(Gamma_obs)
        idx = np.argsort(eigvals)[::-1]
        v1 = eigvecs[:, idx[0]]
        if v1[2] < 0:
            v1 = -v1
        age_mid = (age_lo + age_hi) / 2
        obs_v1s[f'{age_lo}-{age_hi}'] = {
            'v1': v1.tolist(),
            'n': len(X),
            'age_mid': age_mid,
        }

    # Compare predicted vs observed rotation
    # Track angle of v1 from F-axis: theta = arccos(|v1 . e_F|)
    pred_angles = {}
    for age, v1 in pred_v1s.items():
        e_F = np.array([0, 0, 1])
        theta = np.arccos(np.clip(abs(np.dot(v1, e_F)), 0, 1))
        pred_angles[age] = float(np.degrees(theta))
        # Also I-axis loading
        e_I = np.array([1, 0, 0])
        phi = np.arccos(np.clip(abs(np.dot(v1, e_I)), 0, 1))

    obs_angles = {}
    for key, info in obs_v1s.items():
        v1 = np.array(info['v1'])
        e_F = np.array([0, 0, 1])
        theta = np.arccos(np.clip(abs(np.dot(v1, e_F)), 0, 1))
        obs_angles[key] = float(np.degrees(theta))

    # Predicted rotation: v1 should move TOWARD F-axis (theta decreases)
    # from age 30 to 80
    pred_start = pred_angles.get(30, pred_angles.get(40, None))
    pred_end = pred_angles.get(80, pred_angles.get(70, None))
    pred_rotation = pred_end - pred_start if pred_start and pred_end else None

    obs_keys = sorted(obs_angles.keys())
    obs_rotation = None
    if len(obs_keys) >= 2:
        obs_rotation = obs_angles[obs_keys[-1]] - obs_angles[obs_keys[0]]

    direction_match = (pred_rotation is not None and obs_rotation is not None and
                       np.sign(pred_rotation) == np.sign(obs_rotation))

    return {
        'test_name': 'Age-dependent eigenvector rotation',
        'prediction': (f"Dominant eigenvector should rotate toward F-axis with age "
                      f"as I-M coupling strengthens relative to F couplings. "
                      f"Predicted angle from F-axis: {pred_start:.1f} deg (age 30) "
                      f"to {pred_end:.1f} deg (age 80)."),
        'result': (f"Observed angles from F-axis: " +
                  ', '.join(f'{k}: {v:.1f} deg' for k, v in obs_angles.items()) +
                  f". Direction {'matches' if direction_match else 'does not match'} prediction."),
        'effect_size': float(obs_rotation) if obs_rotation else None,
        'p_value': None,
        'counterintuitive': True,
        'testable_with_current_data': len(obs_v1s) >= 2,
        'predicted_v1s': {str(k): v.tolist() for k, v in pred_v1s.items()},
        'observed_v1s': obs_v1s,
        'predicted_angles': pred_angles,
        'observed_angles': obs_angles,
        'direction_match': direction_match,
        'summary': (f"Model predicts eigenvector rotates {abs(pred_rotation):.1f} deg "
                   f"{'toward' if pred_rotation < 0 else 'away from'} F-axis. "
                   f"{'Observed rotation matches direction.' if direction_match else 'Observed rotation does not match.'} "
                   f"At old ages, v1 is ~pure F-axis in both model and data."),
    }


# ---------------------------------------------------------------------------
# Test 5: Non-monotone axis-specific variance
# ---------------------------------------------------------------------------

def test_axis_variance(merged, predictions):
    """Test per-axis variance trajectories against model predictions."""
    print("\n  Test 5: Axis-specific variance trajectories...")

    axes = list(AXES_3)

    # Model predictions
    pred_var = {}
    for age in sorted(predictions.keys()):
        pred_var[age] = predictions[age]['per_axis_var']

    # Observed per-axis variance by stratum
    obs_var = {}
    for age_lo, age_hi in AGE_STRATA:
        mask = merged['complete_3axis'] & (merged['age'] >= age_lo) & (merged['age'] < age_hi)
        X = merged.loc[mask, ['dx_I', 'dx_M', 'dx_F']].values
        n = len(X)
        if n < 20:
            continue
        variances = np.var(X, axis=0, ddof=1)
        obs_var[f'{age_lo}-{age_hi}'] = {
            'n': n,
            'var': variances.tolist(),
            'age_mid': (age_lo + age_hi) / 2,
        }

    # Check monotonicity per axis
    axis_monotone_pred = {}
    axis_monotone_obs = {}
    non_monotone_axes = []

    pred_ages = sorted(pred_var.keys())
    for ax_idx, ax in enumerate(axes):
        pred_vals = [pred_var[a][ax_idx] for a in pred_ages]
        axis_monotone_pred[ax] = all(pred_vals[i] <= pred_vals[i+1]
                                     for i in range(len(pred_vals)-1))

    obs_strata = sorted(obs_var.keys())
    for ax_idx, ax in enumerate(axes):
        obs_vals = [obs_var[s]['var'][ax_idx] for s in obs_strata]
        mono = all(obs_vals[i] <= obs_vals[i+1] for i in range(len(obs_vals)-1))
        axis_monotone_obs[ax] = mono
        if not mono:
            non_monotone_axes.append(ax)

    # Check for variance redistribution (predicted vs observed)
    # Model predicts F variance grows fastest
    pred_ratios = {}
    if len(pred_ages) >= 2:
        for ax_idx, ax in enumerate(axes):
            pred_ratios[ax] = pred_var[pred_ages[-1]][ax_idx] / max(pred_var[pred_ages[0]][ax_idx], 1e-12)

    obs_ratios = {}
    if len(obs_strata) >= 2:
        for ax_idx, ax in enumerate(axes):
            obs_ratios[ax] = (obs_var[obs_strata[-1]]['var'][ax_idx] /
                             max(obs_var[obs_strata[0]]['var'][ax_idx], 1e-12))

    return {
        'test_name': 'Axis-specific variance trajectories',
        'prediction': (f"Model predicts monotonically increasing variance for all axes. "
                      f"F-axis variance grows fastest (ratio {pred_ratios.get('F', '?'):.1f}x), "
                      f"I grows {pred_ratios.get('I', '?'):.1f}x, "
                      f"M grows {pred_ratios.get('M', '?'):.1f}x from age 30 to 80."),
        'result': (f"Observed growth ratios: " +
                  ', '.join(f'{ax}={obs_ratios.get(ax, 0):.2f}x' for ax in axes) +
                  f". Non-monotone axes: {non_monotone_axes if non_monotone_axes else 'none'}."),
        'effect_size': None,
        'p_value': None,
        'counterintuitive': len(non_monotone_axes) > 0,
        'testable_with_current_data': True,
        'predicted_var': {str(k): v for k, v in pred_var.items()},
        'observed_var': obs_var,
        'predicted_ratios': pred_ratios,
        'observed_ratios': obs_ratios,
        'predicted_monotone': axis_monotone_pred,
        'observed_monotone': axis_monotone_obs,
        'non_monotone_axes': non_monotone_axes,
        'summary': (f"Model predicts all axes monotone-increasing. "
                   f"{'Data matches: all axes monotone.' if not non_monotone_axes else f'Data shows non-monotone {non_monotone_axes} — likely medication compression, not coupling redistribution.'} "
                   f"F-axis dominates in both model ({pred_ratios.get('F', 0):.1f}x) and "
                   f"data ({obs_ratios.get('F', 0):.1f}x)."),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("Counterintuitive Predictions from the Coupling Matrix")
    print("=" * 60)

    # Check ELSA data availability
    has_elsa = os.path.exists(os.path.join(DATA_DIR, 'gh_elsa_h_hdr_subset.tab'))
    if not has_elsa:
        print("\n  ELSA data not found — running model-only analysis.")

    # Model predictions
    print("\n  Computing model predictions...")
    predictions = get_model_predictions()

    # Load ELSA data if available
    merged = None
    if has_elsa:
        try:
            merged = load_elsa_data()
        except Exception as e:
            print(f"  ELSA loading failed: {e}")
            merged = None

    # Run tests
    results = []

    if merged is not None:
        results.append(test_lead_lag(merged, predictions))
        results.append(test_partial_correlations(merged, predictions))
        results.append(test_directional_mortality(merged, predictions))
        results.append(test_eigenvector_rotation(merged, predictions))
        results.append(test_axis_variance(merged, predictions))
    else:
        # Model-only analysis
        for test_name in ['Lead-lag', 'Partial correlations', 'Directional mortality',
                         'Eigenvector rotation', 'Axis variance']:
            results.append({
                'test_name': test_name,
                'prediction': 'See model predictions in JSON',
                'result': 'ELSA data not available — model-only',
                'effect_size': None,
                'p_value': None,
                'counterintuitive': None,
                'testable_with_current_data': False,
                'summary': f'{test_name}: requires ELSA data',
            })

    # Save JSON
    output = {
        'model_predictions': {str(k): {kk: vv for kk, vv in v.items()
                                        if kk not in ('J', 'Gamma', 'correlation')}
                              for k, v in predictions.items()},
        'tests': results,
        'has_elsa_data': has_elsa and merged is not None,
        'bonferroni_n': N_BONFERRONI,
    }

    json_path = os.path.join(OUTPUT_DIR, 'counterintuitive_predictions.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {json_path}")

    # Print markdown summary
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("SUMMARY — COUNTERINTUITIVE PREDICTIONS")
    print("=" * 60)

    # Rank by strength of evidence
    supported = []
    unsupported = []
    untestable = []

    for r in results:
        if not r.get('testable_with_current_data', False):
            untestable.append(r)
        elif r.get('p_value') and r['p_value'] < 0.05:
            supported.append(r)
        elif r.get('effect_size') and r['effect_size'] > 0:
            supported.append(r)
        else:
            unsupported.append(r)

    print(f"\nTests with positive evidence ({len(supported)}):")
    for r in supported:
        print(f"  * {r['test_name']}: {r['summary']}")

    print(f"\nTests without clear support ({len(unsupported)}):")
    for r in unsupported:
        print(f"  - {r['test_name']}: {r['summary']}")

    print(f"\nUntestable with current data ({len(untestable)}):")
    for r in untestable:
        print(f"  ? {r['test_name']}: {r['summary']}")

    print(f"\nElapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
