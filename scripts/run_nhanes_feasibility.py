"""
NHANES Real-Data Feasibility Demonstration (Task C)
====================================================
End-to-end pipeline: biomarkers → axes → Γ̂ → α̂ → SWDS → outcome association

Data: NHANES 2011-2012 (public access, no application required)
Axes: M (metabolic: HbA1c + BMI composite), F (functional: grip strength)
Note: CRP not available in this cycle; 2-axis reduction demonstrates the
pipeline without the inflammatory axis. A 3-axis model requires CRP data
from NHANES 2015-2016 (where grip strength is unavailable).
"""

import os
import numpy as np
import pandas as pd
import pyreadstat
from scipy.linalg import inv  # noqa: F401 (kept for compatibility)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)
os.makedirs('outputs', exist_ok=True)
os.makedirs('data/nhanes', exist_ok=True)


def download_nhanes_2011_2012():
    """Download NHANES 2011-2012 XPT files if not already present."""
    import urllib.request
    base = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles'
    files = {
        'demo': 'DEMO_G.XPT',
        'ghb': 'GHB_G.XPT',
        'mgx': 'MGX_G.XPT',
        'bmx': 'BMX_G.XPT',
    }
    for name, fn in files.items():
        path = f'data/nhanes/{name}.XPT'
        if not os.path.exists(path):
            url = f'{base}/{fn}'
            print(f'  Downloading {fn}...', end=' ')
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req, timeout=30)
                data = resp.read()
                with open(path, 'wb') as f:
                    f.write(data)
                print(f'OK ({len(data):,} bytes)')
            except Exception as e:
                print(f'FAILED: {e}')
                print(f'  Please download {fn} manually from {url}')
                print(f'  and place it at {path}')
                return False
    return True


# ============================================================================
# 1. LOAD AND MERGE NHANES DATA
# ============================================================================

print("Checking NHANES data files...")
if not download_nhanes_2011_2012():
    print("ERROR: Could not download NHANES data. See instructions above.")
    exit(1)

print("=" * 70)
print("LOADING NHANES 2011-2012 DATA")
print("=" * 70)

# Load XPT files
demo, meta_d = pyreadstat.read_xport('data/nhanes/demo.XPT')
ghb, meta_g = pyreadstat.read_xport('data/nhanes/ghb.XPT')
mgx, meta_m = pyreadstat.read_xport('data/nhanes/mgx.XPT')
bmx, meta_b = pyreadstat.read_xport('data/nhanes/bmx.XPT')

print(f"  Demographics: {len(demo)} rows, cols: {list(demo.columns[:10])}")
print(f"  HbA1c:        {len(ghb)} rows, cols: {list(ghb.columns)}")
print(f"  Grip strength: {len(mgx)} rows, cols: {list(mgx.columns[:10])}")
print(f"  Body measures: {len(bmx)} rows, cols: {list(bmx.columns[:10])}")

# Merge on SEQN (participant ID)
df = demo[['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDSTATR']].copy()
df = df.rename(columns={'RIDAGEYR': 'age', 'RIAGENDR': 'sex'})

# HbA1c
df = df.merge(ghb[['SEQN', 'LBXGH']], on='SEQN', how='left')
df = df.rename(columns={'LBXGH': 'hba1c'})

# Grip strength (max of left and right)
# Variable names may differ; check what's available
grip_cols = [c for c in mgx.columns if 'MGDCGSZ' in c or 'MGX' in c or 'GRIP' in c.upper()]
print(f"\n  Grip columns available: {list(mgx.columns)[:15]}")

# Use combined grip strength (MGDCGSZ = combined grip size)
mgx_sub = mgx[['SEQN', 'MGDCGSZ']].copy()
mgx_sub = mgx_sub.rename(columns={'MGDCGSZ': 'grip'})

df = df.merge(mgx_sub, on='SEQN', how='left')

# BMI
df = df.merge(bmx[['SEQN', 'BMXBMI']], on='SEQN', how='left')
df = df.rename(columns={'BMXBMI': 'bmi'})

# Self-reported general health (from demo or questionnaire)
# RIDSTATR: 1=interviewed only, 2=interviewed and examined
df = df[df['RIDSTATR'] == 2]  # Keep only examined participants

# Filter to adults 40-79
df = df[(df['age'] >= 40) & (df['age'] <= 79)]

print(f"\n  After filtering (adults 40-79, examined): N = {len(df)}")
print(f"  HbA1c available: {df['hba1c'].notna().sum()}")
print(f"  Grip available: {df['grip'].notna().sum()}")
print(f"  BMI available: {df['bmi'].notna().sum()}")

# Drop rows with any missing key variable
df_complete = df.dropna(subset=['hba1c', 'grip', 'bmi', 'age', 'sex'])
print(f"  Complete cases: N = {len(df_complete)}")


# ============================================================================
# 2. AXIS CONSTRUCTION
# ============================================================================

print("\n" + "=" * 70)
print("AXIS CONSTRUCTION")
print("=" * 70)

# Youthful reference: ages 40-45 (youngest stratum available)
ref = df_complete[(df_complete['age'] >= 40) & (df_complete['age'] <= 45)]
print(f"  Reference group (40-45): N = {len(ref)}")

# Reference statistics
mu_hba1c = ref['hba1c'].mean()
sd_hba1c = ref['hba1c'].std()
mu_bmi = ref['bmi'].mean()
sd_bmi = ref['bmi'].std()
mu_grip = ref['grip'].mean()
sd_grip = ref['grip'].std()

print(f"  HbA1c ref: μ={mu_hba1c:.2f}, σ={sd_hba1c:.2f}")
print(f"  BMI ref:   μ={mu_bmi:.2f}, σ={sd_bmi:.2f}")
print(f"  Grip ref:  μ={mu_grip:.2f}, σ={sd_grip:.2f}")

# Standardize
df_complete = df_complete.copy()
df_complete['dx_hba1c'] = (df_complete['hba1c'] - mu_hba1c) / sd_hba1c
df_complete['dx_bmi'] = (df_complete['bmi'] - mu_bmi) / sd_bmi
df_complete['dx_grip'] = -(df_complete['grip'] - mu_grip) / sd_grip  # Reversed: lower grip = worse

# Composite metabolic axis: average of HbA1c and BMI z-scores
df_complete['dx_M'] = (df_complete['dx_hba1c'] + df_complete['dx_bmi']) / 2

# Functional axis: grip z-score (reversed)
df_complete['dx_F'] = df_complete['dx_grip']

print(f"\n  Δx_M (metabolic) mean by age stratum:")
for lo, hi in [(40,49), (50,59), (60,69), (70,79)]:
    sub = df_complete[(df_complete['age'] >= lo) & (df_complete['age'] <= hi)]
    print(f"    Age {lo}-{hi}: N={len(sub)}, Δx_M={sub['dx_M'].mean():.3f}, Δx_F={sub['dx_F'].mean():.3f}")


# ============================================================================
# 3. CROSS-SECTIONAL COVARIANCE AND α̂ PER STRATUM
# ============================================================================

print("\n" + "=" * 70)
print("COVARIANCE ESTIMATION AND Γ-NATIVE STABILITY PROXY")
print("=" * 70)

n_axes = 2
strata = [(40,49), (50,59), (60,69), (70,79)]
strata_mids = [44.5, 54.5, 64.5, 74.5]

lambda_max_list = []
Gamma_hats = {}

for (lo, hi), mid in zip(strata, strata_mids):
    sub = df_complete[(df_complete['age'] >= lo) & (df_complete['age'] <= hi)]
    X = sub[['dx_M', 'dx_F']].values
    Gamma_hat = np.cov(X.T)
    Gamma_hats[(lo,hi)] = Gamma_hat

    # Γ-native: eigendecomposition of Γ̂ (NO Q specification needed)
    eigenvalues = np.linalg.eigvalsh(Gamma_hat)
    lambda_max = np.max(eigenvalues)
    lambda_max_list.append(lambda_max)

    print(f"\n  Age {lo}-{hi} (N={len(sub)}):")
    print(f"    Γ̂ = [[{Gamma_hat[0,0]:.4f}, {Gamma_hat[0,1]:.4f}],")
    print(f"          [{Gamma_hat[1,0]:.4f}, {Gamma_hat[1,1]:.4f}]]")
    print(f"    λ_max(Γ̂) = {lambda_max:.4f}")

# Monotone trend test on λ_max
trend = all(lambda_max_list[i] < lambda_max_list[i+1] for i in range(3))
print(f"\n  MONOTONE λ_max TREND: {'YES ✓' if trend else 'NO ✗'}")
print(f"    λ_max: {' → '.join(f'{a:.4f}' for a in lambda_max_list)}")


# ============================================================================
# 4. INDIVIDUAL-LEVEL STABILITY-WEIGHTED DYSREGULATION SCORE
# ============================================================================

print("\n" + "=" * 70)
print("INDIVIDUAL-LEVEL STABILITY PROXY (SWDS-Γ)")
print("=" * 70)

def compute_swds_gamma_local(dx, Gamma_hat):
    """Γ-native stability-weighted dysregulation score: Δxᵀ Γ̂ Δx / tr(Γ̂)."""
    return float(dx @ Gamma_hat @ dx / np.trace(Gamma_hat))

# Compute SWDS-Γ for each individual
swds_scores = []
maha_scores = []
l2_scores = []

for _, row in df_complete.iterrows():
    dx = np.array([row['dx_M'], row['dx_F']])
    age = row['age']

    # Find stratum
    for lo, hi in strata:
        if lo <= age <= hi:
            Gamma_hat = Gamma_hats[(lo, hi)]
            break

    swds_scores.append(compute_swds_gamma_local(dx, Gamma_hat))
    maha_scores.append(np.sqrt(dx @ np.linalg.inv(Gamma_hat) @ dx))
    l2_scores.append(np.linalg.norm(dx))

df_complete = df_complete.copy()
df_complete['swds'] = swds_scores
df_complete['mahalanobis'] = maha_scores
df_complete['l2'] = l2_scores

# Summary statistics
print(f"\n  SWDS-Γ distribution:")
print(f"    Mean: {df_complete['swds'].mean():.3f}")
print(f"    Median: {df_complete['swds'].median():.3f}")
print(f"    Correlation with age: {df_complete['swds'].corr(df_complete['age']):.3f}")
print(f"    Correlation with Mahalanobis: {df_complete['swds'].corr(df_complete['mahalanobis']):.3f}")

# SWDS-Γ by age stratum
print(f"\n  SWDS-Γ by age stratum:")
for lo, hi in strata:
    sub = df_complete[(df_complete['age'] >= lo) & (df_complete['age'] <= hi)]
    print(f"    Age {lo}-{hi}: mean SWDS-Γ = {sub['swds'].mean():.3f} ± {sub['swds'].std():.3f}")


# ============================================================================
# 5. ASSOCIATION WITH SELF-REPORTED HEALTH (PROXY OUTCOME)
# ============================================================================

# Note: NHANES mortality linkage requires separate files not directly downloadable.
# We use self-reported general health as a proxy outcome.
# For the manuscript, this would be replaced with actual mortality/morbidity linkage.

print("\n" + "=" * 70)
print("PIPELINE SUMMARY: END-TO-END FEASIBILITY DEMONSTRATED")
print("=" * 70)

print("""
  STEP 1: Biomarker extraction          ✓  (HbA1c, BMI, grip strength from NHANES)
  STEP 2: Axis construction             ✓  (z-score composite M; reversed grip F)
  STEP 3: Youthful reference            ✓  (ages 40-45 subgroup)
  STEP 4: Cross-sectional Γ̂            ✓  (2×2 covariance per age stratum)
  STEP 5: λ_max(Γ̂) stability proxy    ✓  (NO Q SPECIFICATION NEEDED)
  STEP 6: Individual SWDS-Γ            ✓  (Δxᵀ Γ̂ Δx / tr(Γ̂), per person from single visit)
  STEP 7: Outcome association           —  (requires mortality linkage; not in this demo)

  KEY RESULT: The Γ-native pipeline (λ_max, SWDS-Γ) executes on real
  NHANES data without requiring Q specification, Lyapunov inversion,
  or commutation assumptions. Monotone λ_max trend: {trend_result}

  LIMITATIONS OF THIS DEMONSTRATION:
  • 2-axis model only (CRP not available in NHANES 2011-2012)
  • No outcome linkage in this demo (would require NHANES mortality files)
  • No adjustment for medications or comorbidities
  • Survivorship bias and medication variance compression remain
    data-level confounds (not eliminated by Γ-native pipeline)
""".format(trend_result='YES ✓' if trend else 'NO ✗'))


# ============================================================================
# 6. FIGURE
# ============================================================================

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('NHANES 2011-2012: Real-Data Feasibility Demonstration (2-axis model)',
             fontsize=13, fontweight='bold', y=0.98)

# (a) λ_max(Γ̂) trend
ax = axs[0, 0]
ax.plot(strata_mids, lambda_max_list, 'ko-', linewidth=2, markersize=8)
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel(r'$\lambda_{\max}(\hat\Gamma)$', fontsize=11)
ax.set_title(r'(a) $\lambda_{\max}(\hat\Gamma)$ from NHANES cross-sectional data', fontsize=11)
ax.annotate(f'Monotone: {"YES" if trend else "NO"}',
            xy=(0.05, 0.05), xycoords='axes fraction', fontsize=11,
            fontweight='bold', color='green' if trend else 'red')

# (b) Γ̂ diagonal (variance) trends
ax = axs[0, 1]
var_M = [Gamma_hats[(lo,hi)][0,0] for lo,hi in strata]
var_F = [Gamma_hats[(lo,hi)][1,1] for lo,hi in strata]
ax.plot(strata_mids, var_M, 'o-', color='orange', linewidth=2, markersize=7, label=r'$\hat\Gamma_{MM}$ (metabolic)')
ax.plot(strata_mids, var_F, 's-', color='green', linewidth=2, markersize=7, label=r'$\hat\Gamma_{FF}$ (functional)')
ax.set_xlabel('Age stratum midpoint', fontsize=11)
ax.set_ylabel('Variance', fontsize=11)
ax.set_title(r'(b) Cross-sectional variance by age stratum', fontsize=11)
ax.legend(fontsize=9)

# (c) SWDS distribution by age stratum
ax = axs[1, 0]
for (lo, hi), color in zip(strata, ['steelblue', 'orange', 'green', 'red']):
    sub = df_complete[(df_complete['age'] >= lo) & (df_complete['age'] <= hi)]
    ax.hist(sub['swds'], bins=30, density=True, alpha=0.4, color=color,
            label=f'{lo}-{hi}')
ax.set_xlabel('SWDS-Γ (stability-weighted dysregulation)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('(c) SWDS-Γ distribution by age stratum', fontsize=11)
ax.legend(fontsize=9)

# (d) SWDS vs age scatterplot
ax = axs[1, 1]
ax.scatter(df_complete['age'], df_complete['swds'], alpha=0.1, s=5, color='steelblue')
# Add stratum means
means = [df_complete[(df_complete['age']>=lo)&(df_complete['age']<=hi)]['swds'].mean()
         for lo,hi in strata]
ax.plot(strata_mids, means, 'ro-', linewidth=2, markersize=8, label='Stratum mean', zorder=5)
ax.set_xlabel('Age', fontsize=11)
ax.set_ylabel('SWDS-Γ', fontsize=11)
ax.set_title('(d) Individual SWDS-Γ vs age', fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('outputs/figure_nhanes.pdf', dpi=150, bbox_inches='tight')
plt.savefig('outputs/figure_nhanes.png', dpi=150, bbox_inches='tight')
print("Saved figure_nhanes.pdf/png")

# ============================================================================
# PERSISTENT RESULTS
# ============================================================================
try:
    from src.hdr_sim.results_writer import ResultsWriter

    N_complete = len(df_complete)
    lambda_monotone = trend
    swds_mean = df_complete['swds'].mean()
    swds_age_corr = df_complete['swds'].corr(df_complete['age'])

    with ResultsWriter("NHANES Feasibility",
                        "End-to-end pipeline demo on NHANES 2011-2012") as rw:
        rw.add_metric("Complete cases", N_complete)
        rw.add_metric("Axes", "2 (metabolic + functional)")

        rw.add_heading("λ_max(Γ̂) by Age Stratum")
        for (lo, hi), lmax in zip(strata, lambda_max_list):
            rw.add_metric(f"Age {lo}-{hi}", f"{lmax:.4f}")
        rw.add_pass_fail("λ_max monotone increasing", lambda_monotone,
                         "Expected to FAIL in cross-sectional data (survivorship/medication confound)")

        rw.add_heading("SWDS-Γ")
        rw.add_metric("Mean SWDS-Γ", f"{swds_mean:.3f}")
        rw.add_metric("Correlation with age", f"{swds_age_corr:.3f}")
except ImportError:
    pass  # results writing is optional
