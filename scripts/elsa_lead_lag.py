#!/usr/bin/env python3
"""
ELSA 3-axis lead-lag cross-lagged regression.

For each ordered pair (i, j) in {I, M, F}, regress Delta_x_j(t+1) on
x_i(t), controlling for x_j(t) and age. Concordance uses Convention B
(predicted beta > 0 for all pairs in delta-space; see feedback memory).
Three subgroups: (1) full sample, (2) medication-naive (no diag. HTN,
no diag. DM), (3) age 70+.

Mirrors scripts/inchianti_lead_lag.py for direct two-cohort comparison.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest, t as t_dist
from statsmodels.stats.multitest import multipletests

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

# Reuse the ELSA loading pipeline
from scripts.run_elsa_validation import (
    load_all_files, extract_nurse_biomarkers, prepare_harmonised,
    extract_mortality, extract_supplementary, harmonised_to_long,
    build_analysis_panel,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='DataFrame is highly fragmented')

N_BOOT = 10000
SEED = 42
RNG = np.random.default_rng(SEED)

# ELSA 3-axis columns (positive = decline, standardised)
AXES = {'I': 'dx_I', 'M': 'dx_M', 'F': 'dx_F'}

# Compiled J-matrix signs from data/J_matrix_compiled_9x9.csv.
# Retained for traceability only — concordance uses Convention B
# (predicted beta > 0 for ALL pairs; see feedback memory).
J_SIGNS = {
    ('M', 'I'): +1,
    ('I', 'M'): +1,
    ('F', 'I'): -1,
    ('I', 'F'): +1,
    ('F', 'M'): -1,
    ('M', 'F'): +1,
}

# InCHIANTI reference results for direct comparison
INCHIANTI_REF = {
    'I->M': {'beta': 0.033, 'p': 0.031},
    'F->I': {'beta': 0.014, 'p': 0.0001},
    'M->I': {'beta': None, 'p': None},
    'I->F': {'beta': None, 'p': None},
    'F->M': {'beta': None, 'p': None},
    'M->F': {'beta': None, 'p': None},
}


def build_triplets(merged):
    """
    Build consecutive-wave triplets for lead-lag analysis.
    Returns DataFrame with idauniq, wave_t, age_t, axes at t0, t1, and deltas.
    Also keeps diabetes/highbp at wave t for subgroup filtering.
    """
    # Subset to complete 3-axis visits
    df = merged[merged['complete_3axis']].copy()
    df = df.sort_values(['idauniq', 'wave'])

    meta_cols = ['diabetes', 'highbp']
    keep_cols = ['idauniq', 'wave', 'age'] + list(AXES.values()) + [
        c for c in meta_cols if c in df.columns]
    df = df[keep_cols].copy()

    rows = []
    for subj, grp in df.groupby('idauniq', sort=False):
        grp = grp.sort_values('wave')
        for i in range(len(grp) - 1):
            r0 = grp.iloc[i]
            r1 = grp.iloc[i + 1]
            if any(pd.isna(r0[ax]) or pd.isna(r1[ax]) for ax in AXES.values()):
                continue
            if pd.isna(r0['age']):
                continue
            row = {
                'idauniq': subj,
                'wave_t': int(r0['wave']),
                'wave_t1': int(r1['wave']),
                'age_t': float(r0['age']),
            }
            for name, col in AXES.items():
                row[f'{name}_t0'] = float(r0[col])
                row[f'{name}_t1'] = float(r1[col])
                row[f'd_{name}'] = float(r1[col] - r0[col])
            for m in meta_cols:
                if m in df.columns:
                    row[m] = r0[m]
            rows.append(row)
    return pd.DataFrame(rows)


def cross_lagged_beta(triplets, from_name, to_name, n_boot=N_BOOT, rng=None):
    """
    d_{to} ~ beta * {from}_t0 + gamma * {to}_t0 + delta * age_t + intercept

    Bootstrap resamples subjects (idauniq), not rows — preserves within-person
    structure. Returns (beta, ci_lo, ci_hi, p_value, n, n_subjects).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    y = triplets[f'd_{to_name}'].values
    x_from = triplets[f'{from_name}_t0'].values
    x_auto = triplets[f'{to_name}_t0'].values
    x_age = triplets['age_t'].values
    subj = triplets['idauniq'].values

    valid = (np.isfinite(y) & np.isfinite(x_from) &
             np.isfinite(x_auto) & np.isfinite(x_age))
    y, x_from, x_auto, x_age, subj = (
        y[valid], x_from[valid], x_auto[valid], x_age[valid], subj[valid])
    n = len(y)
    n_subj = len(np.unique(subj))

    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, n, n_subj

    X = np.column_stack([np.ones(n), x_from, x_auto, x_age])
    try:
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan, n, n_subj

    beta_cross = float(beta_hat[1])

    # Analytic p-value (OLS SE)
    resid = y - X @ beta_hat
    sigma2 = np.sum(resid ** 2) / max(n - 4, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
        t_stat = beta_cross / se if se > 0 else np.nan
        p_val = float(2 * t_dist.sf(abs(t_stat), df=max(n - 4, 1))) \
            if np.isfinite(t_stat) else np.nan
    except np.linalg.LinAlgError:
        p_val = np.nan

    # Cluster-bootstrap by subject
    subj_to_rows = {}
    for i, s in enumerate(subj):
        subj_to_rows.setdefault(s, []).append(i)
    subj_ids = np.array(list(subj_to_rows.keys()))
    subj_row_arrays = [np.asarray(subj_to_rows[s], dtype=np.int64)
                       for s in subj_ids]

    boots = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, len(subj_ids), size=len(subj_ids))
        idx = np.concatenate([subj_row_arrays[p] for p in picks])
        try:
            bh, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
            boots[b] = bh[1]
        except np.linalg.LinAlgError:
            boots[b] = np.nan

    ci_lo = float(np.nanpercentile(boots, 2.5))
    ci_hi = float(np.nanpercentile(boots, 97.5))

    return beta_cross, ci_lo, ci_hi, p_val, n, n_subj


def run_lead_lag(triplets, label, n_boot=N_BOOT):
    """Run the 6-pair analysis for a given triplet set. Returns list of dicts."""
    print(f"\n{'=' * 60}")
    print(f"Subgroup: {label}")
    print(f"  N pairs: {len(triplets):,}, N subjects: "
          f"{triplets['idauniq'].nunique():,}")
    print(f"  Age range: {triplets['age_t'].min():.0f}–{triplets['age_t'].max():.0f}")
    print('=' * 60)

    if len(triplets) < 20:
        print(f"  Too few pairs — skipping.")
        return []

    print(f"\n{'Pair':<8s} {'beta':>9s} {'95% CI':>22s} {'p':>10s} "
          f"{'N':>7s} {'Npers':>7s} {'Pred':>5s} {'Obs':>5s} {'Conc':>5s}")
    print('-' * 82)

    results = []
    rng = np.random.default_rng(SEED)
    for (from_name, to_name), j_sign in J_SIGNS.items():
        beta, ci_lo, ci_hi, p_val, n, n_subj = cross_lagged_beta(
            triplets, from_name, to_name, n_boot=n_boot, rng=rng)

        pair_str = f'{from_name}->{to_name}'
        if np.isnan(beta):
            obs_sign = 0
            concordant = False
        else:
            obs_sign = 1 if beta > 0 else -1
            concordant = obs_sign == 1   # Convention B: all predict +

        obs_str = '+' if obs_sign > 0 else ('-' if obs_sign < 0 else '?')
        ci_str = f'[{ci_lo:+.4f},{ci_hi:+.4f}]' if np.isfinite(ci_lo) else 'N/A'
        p_str = f'{p_val:.4f}' if np.isfinite(p_val) else 'N/A'

        print(f"{pair_str:<8s} {beta:>+9.4f} {ci_str:>22s} {p_str:>10s} "
              f"{n:>7,d} {n_subj:>7,d} {'+':>5s} {obs_str:>5s} "
              f"{'YES' if concordant else 'NO':>5s}")

        results.append({
            'subgroup': label,
            'from_axis': from_name,
            'to_axis': to_name,
            'pair': pair_str,
            'beta': beta,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'p_value': p_val,
            'n_pairs': n,
            'n_subjects': n_subj,
            'j_sign': int(j_sign),
            'predicted_sign': 1,
            'observed_sign': int(obs_sign),
            'concordant': bool(concordant),
        })

    # Concordance summary (Convention B)
    n_tested = sum(1 for r in results if not np.isnan(r['beta']))
    n_conc = sum(1 for r in results if r['concordant'])

    if n_tested > 0:
        p_conc = binomtest(n_conc, n_tested, 0.5,
                           alternative='greater').pvalue
        print(f"\n  Concordance (Convention B): {n_conc}/{n_tested} "
              f"({100 * n_conc / n_tested:.0f}%), binomial p={p_conc:.4f}")

    return results


def apply_fdr_bh(results, key_pair='pair', key_p='p_value',
                 key_out='p_fdr', alpha=0.05):
    """In-place Benjamini-Hochberg FDR on a list of result dicts.

    Writes `p_fdr`, `fdr_reject` onto each dict. Entries with NaN p-values
    are skipped (keep `p_fdr` = NaN, `fdr_reject` = False) and do not
    consume an FDR correction slot.
    """
    idx = [i for i, r in enumerate(results)
           if np.isfinite(r.get(key_p, np.nan))]
    if not idx:
        return
    pvals = np.array([results[i][key_p] for i in idx])
    reject, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    for k, i in enumerate(idx):
        results[i][key_out] = float(p_fdr[k])
        results[i]['fdr_reject'] = bool(reject[k])
    for i in range(len(results)):
        results[i].setdefault(key_out, None)
        results[i].setdefault('fdr_reject', False)


def build_combined_fdr(elsa_full_results, inchianti_results,
                       output_path, alpha=0.05):
    """Compile FDR-adjusted p-values across both cohorts (separate
    corrections per cohort) and persist as a single JSON for reviewers.
    """
    elsa_rows = [
        {
            'cohort': 'ELSA', 'pair': r['pair'],
            'from_axis': r['from_axis'], 'to_axis': r['to_axis'],
            'beta': r['beta'], 'ci_lower': r.get('ci_lower'),
            'ci_upper': r.get('ci_upper'),
            'p_raw': r['p_value'], 'p_fdr': r.get('p_fdr'),
            'fdr_reject_05': r.get('fdr_reject'),
            'n_pairs': r.get('n_pairs'), 'n_subjects': r.get('n_subjects'),
            'predicted_sign': r.get('predicted_sign', 1),
            'observed_sign': r.get('observed_sign'),
            'concordant': r.get('concordant'),
        }
        for r in elsa_full_results
    ]

    # Apply BH FDR to the 12 InCHIANTI pairs independently.
    inchianti_copy = [dict(r) for r in inchianti_results]
    apply_fdr_bh(inchianti_copy, alpha=alpha)
    inchianti_rows = [
        {
            'cohort': 'InCHIANTI', 'pair': r['pair'],
            'from_axis': r['from_axis'], 'to_axis': r['to_axis'],
            'beta': r['beta'], 'ci_lower': r.get('ci_lower'),
            'ci_upper': r.get('ci_upper'),
            'p_raw': r['p_value'], 'p_fdr': r.get('p_fdr'),
            'fdr_reject_05': r.get('fdr_reject'),
            'n_pairs': r.get('n', r.get('n_pairs')),
            'predicted_sign': r.get('predicted_sign', 1),
            'observed_sign': r.get('observed_sign'),
            'concordant': r.get('concordant'),
        }
        for r in inchianti_copy
    ]

    n_elsa_sig = sum(1 for r in elsa_rows if r['fdr_reject_05'])
    n_inch_sig = sum(1 for r in inchianti_rows if r['fdr_reject_05'])

    payload = {
        'description': (
            'Benjamini-Hochberg FDR-adjusted p-values for cross-lagged '
            'lead-lag coefficients in ELSA (3-axis, 6 ordered pairs) and '
            'InCHIANTI (4-axis, 12 ordered pairs). FDR is applied '
            'independently within each cohort.'
        ),
        'method': {
            'correction': 'Benjamini-Hochberg (statsmodels multipletests, '
                          "method='fdr_bh')",
            'alpha': alpha,
            'elsa_n_tests': len(elsa_rows),
            'inchianti_n_tests': len(inchianti_rows),
            'note': ('ELSA FDR uses the full-sample (non-stratified) 6 pairs '
                     'only; subgroup analyses are not included in the FDR '
                     'family.'),
        },
        'summary': {
            'elsa_n_fdr_05': n_elsa_sig,
            'elsa_n_tests': len(elsa_rows),
            'inchianti_n_fdr_05': n_inch_sig,
            'inchianti_n_tests': len(inchianti_rows),
        },
        'elsa': elsa_rows,
        'inchianti': inchianti_rows,
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Combined FDR JSON: {output_path}")
    return payload


def make_heatmap(all_results, output_path):
    """3×3 cross-lagged β heatmap.

    Rows = source axis level at t; columns = target axis change t→t+1.
    Cell annotation: β, raw p, FDR-adjusted p (BH across the 6 pairs).
    A bold border on the cell indicates FDR<0.05.
    """
    full = [r for r in all_results if r['subgroup'] == 'full']
    axes = list(AXES.keys())
    mat = np.full((3, 3), np.nan)
    pmat = np.full((3, 3), np.nan)
    pfdr_mat = np.full((3, 3), np.nan)
    fdr_ok = np.zeros((3, 3), dtype=bool)
    n_pair = None
    for r in full:
        i = axes.index(r['from_axis'])
        j = axes.index(r['to_axis'])
        mat[i, j] = r['beta']
        pmat[i, j] = r['p_value']
        pfdr = r.get('p_fdr')
        if pfdr is not None and np.isfinite(pfdr):
            pfdr_mat[i, j] = pfdr
        fdr_ok[i, j] = bool(r.get('fdr_reject', False))
        n_pair = r.get('n_pairs', n_pair)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 0.02)
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f'Δ{a}' for a in axes])
    ax.set_yticklabels([f'{a}(t)' for a in axes])
    ax.set_xlabel('Target axis change (t → t+1)')
    ax.set_ylabel('Source axis level (t)')
    title = 'ELSA 3-axis cross-lagged β'
    if n_pair is not None:
        title += f'  (N={n_pair:,} consecutive-wave pairs)'
    ax.set_title(title)

    for i in range(3):
        for j in range(3):
            if not np.isnan(mat[i, j]):
                p_raw = pmat[i, j]
                p_fdr = pfdr_mat[i, j]
                if np.isfinite(p_fdr):
                    label = (f'β={mat[i, j]:+.3f}\n'
                             f'p={p_raw:.2g}\n'
                             f'q={p_fdr:.2g}')
                else:
                    label = f'β={mat[i, j]:+.3f}\np={p_raw:.2g}'
                color = ('white' if abs(mat[i, j]) > 0.6 * vmax else 'black')
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=8.5, color=color,
                        fontweight='bold' if fdr_ok[i, j] else 'normal')
                if fdr_ok[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor='black', linewidth=2.2))
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=10, color='gray')

    cb = plt.colorbar(im, ax=ax, label='β (cross-lagged)')
    cb.ax.tick_params(labelsize=8)
    fig.text(0.01, 0.01, 'q = BH-FDR p across 6 pairs; '
             'bold border = FDR<0.05',
             fontsize=8, color='#555555')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, bbox_inches='tight')
    fig.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description='ELSA 3-axis lead-lag analysis')
    p.add_argument('--n-boot', type=int, default=N_BOOT,
                   help=f'Bootstrap resamples (default {N_BOOT})')
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("ELSA 3-axis lead-lag cross-lagged regression")
    print("=" * 70)
    print(f"Bootstrap resamples: {args.n_boot:,} (cluster by subject)")

    # Build the merged analysis panel (reusing run_elsa_validation pipeline)
    files = load_all_files()
    panel, _ = extract_nurse_biomarkers(files)
    harm = prepare_harmonised(files)
    mort = extract_mortality(files, harm)
    supp = extract_supplementary(files)
    harm_long = harmonised_to_long(harm, waves=[2, 4, 6, 8])
    merged = build_analysis_panel(panel, harm_long, mort, supp)

    # Build consecutive-wave triplets
    triplets = build_triplets(merged)
    print(f"\nTotal consecutive-wave pairs (complete 3-axis): {len(triplets):,}")
    print(f"Unique subjects: {triplets['idauniq'].nunique():,}")

    # ---------------------------------------------------------------
    # Subgroup construction
    # ---------------------------------------------------------------
    subgroups = {'full': triplets}

    # Medication-naive: no diagnosed HTN (highbp==0) AND no diagnosed DM
    # (diabetes==0) at wave t. Matches the Cox adjustment covariate
    # definition in run_elsa_validation.py (highbp, diabetes).
    if 'diabetes' in triplets.columns and 'highbp' in triplets.columns:
        med_naive = triplets[(triplets['diabetes'] == 0) &
                              (triplets['highbp'] == 0)].copy()
        subgroups['med_naive'] = med_naive
        print(f"Medication-naive (no DM, no HTN dx): {len(med_naive):,} pairs, "
              f"{med_naive['idauniq'].nunique():,} subjects")
    else:
        print("WARNING: diabetes/highbp columns missing — "
              "skipping medication-naive subgroup")

    # Age 70+
    age70 = triplets[triplets['age_t'] >= 70].copy()
    subgroups['age_70plus'] = age70
    print(f"Age 70+: {len(age70):,} pairs, "
          f"{age70['idauniq'].nunique():,} subjects")

    # ---------------------------------------------------------------
    # Run analysis across subgroups
    # ---------------------------------------------------------------
    all_results = []
    subgroup_summary = {}
    for name, tp in subgroups.items():
        res = run_lead_lag(tp, label=name, n_boot=args.n_boot)
        all_results.extend(res)

        n_tested = sum(1 for r in res if not np.isnan(r['beta']))
        n_conc = sum(1 for r in res if r['concordant'])
        p_conc = (binomtest(n_conc, n_tested, 0.5,
                            alternative='greater').pvalue
                  if n_tested > 0 else None)
        subgroup_summary[name] = {
            'n_pairs': int(len(tp)),
            'n_subjects': int(tp['idauniq'].nunique()) if len(tp) else 0,
            'n_tested': n_tested,
            'n_concordant': n_conc,
            'concordance_rate': n_conc / n_tested if n_tested else None,
            'binomial_p': float(p_conc) if p_conc is not None else None,
        }

    # ---------------------------------------------------------------
    # FDR adjustment (Benjamini-Hochberg) on the 6 full-sample pairs
    # ---------------------------------------------------------------
    full_results = [r for r in all_results if r['subgroup'] == 'full']
    apply_fdr_bh(full_results)
    # Write back so downstream `all_results` carries the p_fdr field.
    full_by_pair = {r['pair']: r for r in full_results}
    for r in all_results:
        if r['subgroup'] == 'full':
            src = full_by_pair[r['pair']]
            r['p_fdr'] = src.get('p_fdr')
            r['fdr_reject'] = src.get('fdr_reject', False)

    print("\n" + "=" * 70)
    print("Full-sample ELSA lead-lag with FDR (BH)")
    print("=" * 70)
    print(f"{'Pair':<8s} {'beta':>9s} {'p_raw':>10s} {'p_fdr':>10s} "
          f"{'FDR<0.05':>10s}  concordant")
    print('-' * 65)
    for r in full_results:
        pf = r.get('p_fdr')
        pf_str = f"{pf:.4g}" if pf is not None and np.isfinite(pf) else '   N/A '
        print(f"{r['pair']:<8s} {r['beta']:>+9.4f} "
              f"{r['p_value']:>10.4g} {pf_str:>10s} "
              f"{str(r.get('fdr_reject', False)):>10s}  "
              f"{'YES' if r['concordant'] else 'NO '}")
    n_sig = sum(1 for r in full_results if r.get('fdr_reject'))
    print(f"\nELSA: {n_sig}/{len(full_results)} pairs survive FDR<0.05")

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------
    os.makedirs('results', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    df_out = pd.DataFrame(all_results)
    csv_path = 'results/elsa_lead_lag_matrix.csv'
    df_out.to_csv(csv_path, index=False)
    print(f"\nResults CSV: {csv_path}")

    # InCHIANTI comparison table (full-sample)
    full_res = {r['pair']: r for r in all_results if r['subgroup'] == 'full'}
    comparison = {}
    for pair, ref in INCHIANTI_REF.items():
        elsa = full_res.get(pair)
        if elsa is None:
            continue
        entry = {
            'inchianti_beta': ref['beta'],
            'inchianti_p': ref['p'],
            'elsa_beta': elsa['beta'],
            'elsa_p': elsa['p_value'],
            'elsa_ci': [elsa['ci_lower'], elsa['ci_upper']],
            'elsa_n_pairs': elsa['n_pairs'],
            'elsa_n_subjects': elsa['n_subjects'],
        }
        if ref['beta'] is not None:
            same_sign = (np.sign(ref['beta']) == np.sign(elsa['beta']))
            sig = elsa['p_value'] < 0.05 if np.isfinite(elsa['p_value']) else False
            entry['replicates'] = bool(same_sign and sig)
            entry['same_sign'] = bool(same_sign)
        else:
            entry['replicates'] = None
            entry['same_sign'] = None
        comparison[pair] = entry

    summary = {
        'description': 'ELSA 3-axis lead-lag cross-lagged regression, '
                       '6 ordered pairs from {I, M, F}',
        'convention_note': 'Convention B: predicted beta > 0 for all pairs '
                           '(see feedback_leadlag_sign_convention memory).',
        'n_boot': args.n_boot,
        'subgroups': subgroup_summary,
        'comparison': comparison,
        'results': [{k: (None if isinstance(v, float) and np.isnan(v) else v)
                     for k, v in r.items()} for r in all_results],
    }
    json_path = 'results/elsa_lead_lag_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary JSON: {json_path}")

    # ---------------------------------------------------------------
    # Combined FDR across both cohorts
    # ---------------------------------------------------------------
    inch_path = 'results/inchianti_lead_lag_summary.json'
    if os.path.exists(inch_path):
        with open(inch_path) as f:
            inch_summary = json.load(f)
        inch_results = inch_summary.get('results', [])
        combined_path = 'results/lead_lag_fdr_combined.json'
        build_combined_fdr(full_results, inch_results, combined_path)
    else:
        print(f"\n  WARNING: {inch_path} not found — skipping combined FDR.")

    # Figure
    fig_path = 'outputs/figure_elsa_lead_lag.pdf'
    make_heatmap(all_results, fig_path)

    # ---------------------------------------------------------------
    # Final InCHIANTI comparison printout
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Direct comparison to InCHIANTI (full sample)")
    print("=" * 70)
    print(f"{'Pair':<8s} {'InCH β':>10s} {'InCH p':>9s} {'ELSA β':>10s} "
          f"{'ELSA p':>9s} {'Match':>7s}")
    print('-' * 60)
    for pair, c in comparison.items():
        ic_b = f"{c['inchianti_beta']:+.4f}" if c['inchianti_beta'] is not None else '  —   '
        ic_p = f"{c['inchianti_p']:.4f}" if c['inchianti_p'] is not None else '  —  '
        el_b = f"{c['elsa_beta']:+.4f}" if np.isfinite(c['elsa_beta']) else '  —   '
        el_p = f"{c['elsa_p']:.4f}" if np.isfinite(c['elsa_p']) else '  —  '
        if c['replicates'] is True:
            tag = 'YES'
        elif c['replicates'] is False:
            tag = 'NO'
        else:
            tag = '  — '
        print(f"{pair:<8s} {ic_b:>10s} {ic_p:>9s} {el_b:>10s} {el_p:>9s} {tag:>7s}")

    # Key-pair banner
    key_replicates = [p for p in ('I->M', 'F->I')
                      if comparison.get(p, {}).get('replicates')]
    print(f"\nKey-pair replication: {len(key_replicates)}/2 "
          f"(I→M, F→I): {key_replicates or 'none'}")


if __name__ == '__main__':
    main()
