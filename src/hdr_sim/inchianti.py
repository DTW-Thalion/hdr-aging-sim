"""
InCHIANTI cohort data loader for HDR 4-axis replication analysis.

Loads 6 waves of InCHIANTI data (baseline + 5 follow-ups), merges
assays, ECG, physical exam, drugs, and disease adjudication into a
long-format panel suitable for the Gamma-native HDR pipeline.

Data source: InCHIANTI_CD_Share (Chianti cohort, Tuscany, ages 20-102)
Waves: Baseline (1998-2000), FU1, FU2, FU3, FU4, FU5

CRITICAL NOTE: RMSSD/HRV time-domain measures are NOT available in the
standard InCHIANTI data release. The N-axis uses resting heart rate
(X_FC) as a proxy, sign-flipped so higher HR = worse autonomic function.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import pyreadstat

# ── Default data root ──────────────────────────────────────────────
_DEFAULT_DATA_ROOT = os.path.join(
    os.path.expanduser("~"), "Downloads", "inCHIANTI", "InCHIANTI_CD_Share"
)

# ── Wave prefix configuration ─────────────────────────────────────
# Each wave uses a different single-letter prefix for variable names.
WAVE_CONFIG = {
    0: {  # Baseline
        "folder": os.path.join("Baseline_V8", "English", "4.Data", "SAS_Datasets"),
        "lab_file": "Assays/labo_raw.sas7bdat",
        "ecg_file": "EKG_ENG_Doppler/mar_raw.sas7bdat",
        "pe_file": "Physical_Exam/per_ana.sas7bdat",
        "drug_file": "Drugs/fmc_ana.sas7bdat",
        "dx_file": "Diseases/adju_ana.sas7bdat",
        "cli_file": "Medical_Exam/cli_rawe.sas7bdat",
        "lab_prefix": "X_",
        "pe_prefix": "PX",
        "drug_prefix": "FX1_",
        "dx_prefix": "AX",
        "il6_var": "X_IL6",
        "il6_ec_var": "X_IL6_EC",
        "glucose_var": "X_GLU",
        "insulin_var": "X_INSULN",
        "crp_var": "X_CRP_HS",
        "fibrinogen_var": "X_FIBRIN",
        "hdl_var": "X_COLHDL",
        "ldl_var": "X_COLLDL",
        "trig_var": "X_TRIGLI",
        "cystc_var": "X_CYSC",
        "albumin_var": "X_ALB",
        "hr_var": "X_FC",
        "rhythm_var": "X_RITMO",
        "age_lab_var": "X_AGEL",
        "age_pe_var": "PXAGE",
        "sppb_var": "PXSPS",
        "sppb_bal_var": "PXSPSB",
        "sppb_chair_var": "PXSPSC",
        "sppb_walk_var": "PXSPSW",
        "grip_var": "PXHGMAX",
        "gait_var": "PXWSPD1A",
    },
    1: {  # Follow-up 1
        "folder": os.path.join("Follow-up1_V5", "4.Data", "SAS_Datasets"),
        "lab_file": "Assays/labf1raw.sas7bdat",
        "ecg_file": "EKG_ENG_Doppler/marf1raw.sas7bdat",
        "pe_file": "Physical_Exam/pef1_ana.sas7bdat",
        "drug_file": "Drugs/fmcf1ana.sas7bdat",
        "dx_file": "Diseases/adjf1ana.sas7bdat",
        "cli_file": None,
        "lab_prefix": "Y_",
        "pe_prefix": "PY",
        "drug_prefix": "FY1_",
        "il6_var": "Y_IL6_E",
        "il6_ec_var": None,
        "glucose_var": "Y_GLU",
        "insulin_var": "Y_INSULA",
        "crp_var": "Y_CRP_HS",
        "fibrinogen_var": "Y_FIBRIN",
        "hdl_var": "Y_COLHDL",
        "ldl_var": "Y_COLLDL",
        "trig_var": "Y_TRIGLI",
        "cystc_var": "Y_CYSC",
        "albumin_var": "Y_ALB",
        "hr_var": "Y_FC",
        "rhythm_var": "Y_RITMO",
        "age_lab_var": "Y_AGEL",
        "age_pe_var": "PYAGE",
        "sppb_var": "PYSPS",
        "sppb_bal_var": "PYSPSB",
        "sppb_chair_var": "PYSPSC",
        "sppb_walk_var": "PYSPSW",
        "grip_var": "PYHGMAX",
        "gait_var": "PYWSPD1A",
    },
    2: {  # Follow-up 2
        "folder": os.path.join("Follow-up2_V4", "4.Data", "SAS_Datasets"),
        "lab_file": "Assays/labf2raw.sas7bdat",
        "ecg_file": "EKG_ENG_Doppler/marf2raw.sas7bdat",
        "pe_file": "Physical_Exam/pef2_ana.sas7bdat",
        "drug_file": "Drugs/fmcf2ana.sas7bdat",
        "dx_file": "Diseases/adjf2ana.sas7bdat",
        "cli_file": None,
        "lab_prefix": "Z_",
        "pe_prefix": "PZ",
        "drug_prefix": "FZ1_",
        "il6_var": "Z_IL6_E",
        "il6_ec_var": None,
        "glucose_var": "Z_GLU",
        "insulin_var": "Z_INSULA",
        "crp_var": "Z_CRP_HS",
        "fibrinogen_var": "Z_FIBRIN",
        "hdl_var": "Z_COLHDL",
        "ldl_var": "Z_COLLDL",
        "trig_var": "Z_TRIGLI",
        "cystc_var": "Z_CYSC",
        "albumin_var": "Z_ALB",
        "hr_var": "Z_FC",
        "rhythm_var": "Z_RITMO",
        "age_lab_var": "Z_AGEL",
        "age_pe_var": "PZAGE",
        "sppb_var": "PZSPS",
        "sppb_bal_var": "PZSPSB",
        "sppb_chair_var": "PZSPSC",
        "sppb_walk_var": "PZSPSW",
        "grip_var": "PZHGMAX",
        "gait_var": "PZWSPD1A",
    },
    3: {  # Follow-up 3
        "folder": os.path.join("Follow-up3_V3", "4.Data", "SAS_Datasets"),
        "lab_file": "Assays/labf3raw.sas7bdat",
        "ecg_file": "EKG_ENG_Doppler/marf3raw.sas7bdat",
        "pe_file": "Physical_Exam/pef3_ana.sas7bdat",
        "drug_file": "Drugs/fmcf3ana.sas7bdat",
        "dx_file": "Diseases/adjf3ana.sas7bdat",
        "cli_file": None,
        "lab_prefix": "Q_",
        "pe_prefix": "PQ",
        "drug_prefix": "FQ1_",
        "il6_var": "Q_IL6_E",
        "il6_ec_var": None,
        "glucose_var": "Q_GLU",
        "insulin_var": None,  # Not available in FU3
        "crp_var": "Q_CRP_HS",
        "fibrinogen_var": "Q_FIBRIN",
        "hdl_var": "Q_COLHDL",
        "ldl_var": "Q_COLLDL",
        "trig_var": "Q_TRIGLI",
        "cystc_var": "Q_CYSC",
        "albumin_var": "Q_ALB",
        "hr_var": "Q_FC",
        "rhythm_var": "Q_RITMO",
        "age_lab_var": "Q_AGEL",
        "age_pe_var": "PQAGE",
        "sppb_var": "PQSPS",
        "sppb_bal_var": "PQSPSB",
        "sppb_chair_var": "PQSPSC",
        "sppb_walk_var": "PQSPSW",
        "grip_var": "PQHGMAX",
        "gait_var": "PQWSPD1A",
    },
    4: {  # Follow-up 4 — no ECG, limited assays
        "folder": os.path.join("Follow-up4_v2", "4.Data", "SAS_Datasets"),
        "lab_file": "Assays/labf4raw.sas7bdat",
        "ecg_file": None,  # No ECG at FU4
        "pe_file": "Physical_Exam/pef4_ana.sas7bdat",
        "drug_file": "Drugs/fmcf4ana.sas7bdat",
        "dx_file": "Diseases/adjf4ana.sas7bdat",
        "cli_file": None,
        "lab_prefix": "C_",
        "pe_prefix": "PC",
        "drug_prefix": "FC1_",
        "il6_var": None,  # Not available in FU4
        "il6_ec_var": None,
        "glucose_var": "C_GLU",
        "insulin_var": None,  # Not available in FU4
        "crp_var": None,
        "fibrinogen_var": "C_FIBRIN",
        "hdl_var": "C_COLHDL",
        "ldl_var": "C_COLLDL",
        "trig_var": "C_TRIGLI",
        "cystc_var": None,
        "albumin_var": "C_ALB",
        "hr_var": None,
        "rhythm_var": None,
        "age_lab_var": "C_AGEL",
        "age_pe_var": "PCAGE",
        "sppb_var": "PCSPS",
        "sppb_bal_var": "PCSPSB",
        "sppb_chair_var": "PCSPSC",
        "sppb_walk_var": "PCSPSW",
        "grip_var": "PCHGMAX",
        "gait_var": "PCWSPD1A",
    },
    5: {  # Follow-up 5 — physical exam only
        "folder": os.path.join("Follow-up5_v1", "4.Data", "SAS_Datasets"),
        "lab_file": None,
        "ecg_file": None,
        "pe_file": "Physical_Exam/pef5_ana.sas7bdat",
        "drug_file": None,
        "dx_file": None,
        "cli_file": None,
        "lab_prefix": None,
        "pe_prefix": "PF",
        "drug_prefix": None,
        "il6_var": None,
        "il6_ec_var": None,
        "glucose_var": None,
        "insulin_var": None,
        "crp_var": None,
        "fibrinogen_var": None,
        "hdl_var": None,
        "ldl_var": None,
        "trig_var": None,
        "cystc_var": None,
        "albumin_var": None,
        "hr_var": None,
        "rhythm_var": None,
        "age_lab_var": None,
        "age_pe_var": "PFAGE",
        "sppb_var": "PFSPS",
        "sppb_bal_var": "PFSPSB",
        "sppb_chair_var": "PFSPSC",
        "sppb_walk_var": "PFSPSW",
        "grip_var": "PFHGMAX",
        "gait_var": "PFWSPD1A",
    },
}

# Baseline diagnosis variables (from adju_ana)
DX_VARS = {
    "dx_htn": "AXIPERT2",
    "dx_dm": "AXDIAB2A",
    "dx_metsyn": "AXMETBOL",
    "dx_frailty": "AXALLFRA",
    "dx_park": "AXPARK",
    "dx_dement": "AXDEMENT",
    "dx_stroke": "AXSTROKE",
    "dx_mi": "AXMI",
    "dx_chf": "AXCHF",
    "dx_cancer": "AXCANCER",
    "dx_copd": "AXBPCO",
}

# Antihypertensive drug class variables (baseline prefix FX1_)
ANTIHTN_VARS = ["C2", "C3", "C4", "C5", "C6", "C6A", "C9", "C13"]


def _safe_read_sas(path):
    """Read a SAS7BDAT file, return (DataFrame, meta) or (None, None)."""
    if path is None or not os.path.exists(path):
        return None, None
    try:
        df, meta = pyreadstat.read_sas7bdat(path)
        return df, meta
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {e}")
        return None, None


def _extract_col(df, varname, default_val=np.nan):
    """Safely extract a column, returning NaN Series if not found."""
    if df is None or varname is None or varname not in df.columns:
        if df is not None:
            return pd.Series(default_val, index=df.index)
        return pd.Series(dtype=float)
    return df[varname]


def _read_wave_data(wave_idx, data_root):
    """Read and harmonize one wave's data into standard column names."""
    cfg = WAVE_CONFIG[wave_idx]
    base = os.path.join(data_root, cfg["folder"])

    rows = []

    # 1) Lab assays
    lab_path = os.path.join(base, cfg["lab_file"]) if cfg["lab_file"] else None
    lab, _ = _safe_read_sas(lab_path)

    # 2) ECG
    ecg_path = os.path.join(base, cfg["ecg_file"]) if cfg["ecg_file"] else None
    ecg, _ = _safe_read_sas(ecg_path)

    # 3) Physical exam
    pe_path = os.path.join(base, cfg["pe_file"]) if cfg["pe_file"] else None
    pe, _ = _safe_read_sas(pe_path)

    # 4) Drugs
    drug_path = os.path.join(base, cfg["drug_file"]) if cfg["drug_file"] else None
    drug, _ = _safe_read_sas(drug_path)

    # Collect all CODE98 values present in any dataset
    id_sets = []
    for ds in [lab, ecg, pe, drug]:
        if ds is not None and "CODE98" in ds.columns:
            id_sets.append(set(ds["CODE98"].dropna().values))
    if not id_sets:
        return pd.DataFrame()
    all_ids = sorted(set.union(*id_sets))

    # Build master frame indexed by CODE98
    master = pd.DataFrame({"code98": all_ids})

    def _merge(master_df, source_df, cols_map):
        """Merge selected columns from source onto master."""
        if source_df is None:
            for new_name in cols_map.values():
                master_df[new_name] = np.nan
            return master_df
        rename = {}
        keep = ["CODE98"]
        for src_var, dst_name in cols_map.items():
            if src_var is not None and src_var in source_df.columns:
                rename[src_var] = dst_name
                keep.append(src_var)
            else:
                master_df[dst_name] = np.nan
        if len(rename) == 0:
            return master_df
        subset = source_df[keep].drop_duplicates(subset=["CODE98"]).rename(columns=rename)
        subset = subset.rename(columns={"CODE98": "code98"})
        return master_df.merge(subset, on="code98", how="left")

    # Lab variables
    lab_map = {
        cfg["il6_var"]: "il6",
        cfg["il6_ec_var"]: "il6_ec",
        cfg["glucose_var"]: "glucose",
        cfg["insulin_var"]: "insulin",
        cfg["crp_var"]: "crp_hs",
        cfg["fibrinogen_var"]: "fibrinogen",
        cfg["hdl_var"]: "hdl",
        cfg["ldl_var"]: "ldl",
        cfg["trig_var"]: "triglycerides",
        cfg["cystc_var"]: "cystatin_c",
        cfg["albumin_var"]: "albumin_pct",
        cfg["age_lab_var"]: "age_lab",
    }
    master = _merge(master, lab, lab_map)

    # ECG variables
    ecg_map = {
        cfg["hr_var"]: "resting_hr",
        cfg["rhythm_var"]: "cardiac_rhythm",
    }
    master = _merge(master, ecg, ecg_map)

    # Physical exam
    pe_map = {
        cfg["age_pe_var"]: "age_pe",
        cfg["sppb_var"]: "sppb",
        cfg["sppb_bal_var"]: "sppb_balance",
        cfg["sppb_chair_var"]: "sppb_chair",
        cfg["sppb_walk_var"]: "sppb_walk",
        cfg["grip_var"]: "grip",
        cfg["gait_var"]: "gait_speed",
    }
    master = _merge(master, pe, pe_map)

    # Drug class flags — build antihypertensive, statin, antidiabetic flags
    if drug is not None and cfg["drug_prefix"] is not None:
        pfx = cfg["drug_prefix"]
        # Antihypertensive = any of C2-C9, C13
        htn_cols = [pfx + s for s in ANTIHTN_VARS if pfx + s in drug.columns]
        drug["_any_antihtn"] = drug[htn_cols].max(axis=1) if htn_cols else 0
        # Statin
        statin_col = pfx + "C1"
        drug["_statin"] = drug[statin_col] if statin_col in drug.columns else 0
        # Antidiabetic (oral + insulin)
        dm_cols = [pfx + s for s in ["A9", "A10"] if pfx + s in drug.columns]
        drug["_any_antidm"] = drug[dm_cols].max(axis=1) if dm_cols else 0
        # NSAID
        nsaid_col = pfx + "M1"
        drug["_nsaid"] = drug[nsaid_col] if nsaid_col in drug.columns else 0
        # Glucocorticoid
        gc_col = pfx + "H1"
        drug["_glucocort"] = drug[gc_col] if gc_col in drug.columns else 0

        drug_subset = drug[["CODE98", "_any_antihtn", "_statin", "_any_antidm",
                            "_nsaid", "_glucocort"]].drop_duplicates(subset=["CODE98"])
        drug_subset = drug_subset.rename(columns={
            "CODE98": "code98",
            "_any_antihtn": "med_antihtn",
            "_statin": "med_statin",
            "_any_antidm": "med_antidm",
            "_nsaid": "med_nsaid",
            "_glucocort": "med_glucocort",
        })
        master = master.merge(drug_subset, on="code98", how="left")
    else:
        for c in ["med_antihtn", "med_statin", "med_antidm", "med_nsaid", "med_glucocort"]:
            master[c] = np.nan

    # Sex and site from whichever dataset has them
    for ds in [lab, ecg, pe, drug]:
        if ds is not None and "SEX" in ds.columns and "sex" not in master.columns:
            sex_df = ds[["CODE98", "SEX"]].drop_duplicates(subset=["CODE98"]).rename(
                columns={"CODE98": "code98", "SEX": "sex"})
            master = master.merge(sex_df, on="code98", how="left")
        if ds is not None and "SITE" in ds.columns and "site" not in master.columns:
            site_df = ds[["CODE98", "SITE"]].drop_duplicates(subset=["CODE98"]).rename(
                columns={"CODE98": "code98", "SITE": "site"})
            master = master.merge(site_df, on="code98", how="left")

    master["wave"] = wave_idx

    # Best available age: prefer lab age, fall back to PE age
    master["age"] = master["age_lab"].fillna(master.get("age_pe", np.nan))

    # Compute SPPB total from components if total is missing (e.g. FU5)
    if master["sppb"].isna().all() or master["sppb"].notna().sum() == 0:
        components = ["sppb_balance", "sppb_chair", "sppb_walk"]
        avail = [c for c in components if c in master.columns and master[c].notna().any()]
        if avail:
            master["sppb"] = master[avail].sum(axis=1, min_count=len(avail))

    return master


def load_inchianti_panel(data_root=None, waves=None):
    """
    Load InCHIANTI data as a long-format panel.

    Parameters
    ----------
    data_root : str, optional
        Path to InCHIANTI_CD_Share directory. Defaults to ~/Downloads/inCHIANTI/InCHIANTI_CD_Share.
    waves : list of int, optional
        Which waves to load (0=baseline, 1=FU1, ..., 5=FU5). Default: all.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:
        code98, wave, age, sex, site,
        il6, il6_ec, glucose, insulin, crp_hs, fibrinogen,
        hdl, ldl, triglycerides, cystatin_c, albumin_pct,
        resting_hr, cardiac_rhythm,
        sppb, sppb_balance, sppb_chair, sppb_walk, grip, gait_speed,
        med_antihtn, med_statin, med_antidm, med_nsaid, med_glucocort,
        homa_ir, log_il6, age_decade, n_med_classes
    """
    if data_root is None:
        data_root = _DEFAULT_DATA_ROOT
    if waves is None:
        waves = list(range(6))

    dfs = []
    for w in waves:
        print(f"  Loading wave {w}...", end=" ")
        wdf = _read_wave_data(w, data_root)
        print(f"N={len(wdf)}")
        if len(wdf) > 0:
            dfs.append(wdf)

    if not dfs:
        raise RuntimeError("No wave data loaded. Check data_root path.")

    panel = pd.concat(dfs, ignore_index=True)

    # ── Merge baseline diagnoses ──────────────────────────────────
    dx_path = os.path.join(data_root, "Baseline_V8", "English", "4.Data",
                           "SAS_Datasets", "Diseases", "adju_ana.sas7bdat")
    dx, _ = _safe_read_sas(dx_path)
    if dx is not None:
        dx_cols = {"CODE98": "code98"}
        for new_name, src_var in DX_VARS.items():
            if src_var in dx.columns:
                dx_cols[src_var] = new_name
        dx_sub = dx[list(dx_cols.keys())].drop_duplicates(subset=["CODE98"])
        dx_sub = dx_sub.rename(columns=dx_cols)
        # Binarize: 0/1 with NaN preserved
        for col in DX_VARS.keys():
            if col in dx_sub.columns:
                dx_sub[col] = dx_sub[col].apply(
                    lambda x: 1.0 if x == 1 else (0.0 if x == 0 else np.nan))
        panel = panel.merge(dx_sub, on="code98", how="left")

    # ── Merge vital status ────────────────────────────────────────
    vs_path = os.path.join(data_root, "Vital_Status", "1.Data", "SAS_Datasets",
                           "Master_thru_Follow-up5", "ana_raw.sas7bdat")
    vs, _ = _safe_read_sas(vs_path)
    if vs is not None:
        vs_cols = ["CODE98"]
        vs_rename = {"CODE98": "code98"}
        for c in vs.columns:
            if "DEAD" in c.upper() or "DEATH" in c.upper() or "VITAL" in c.upper() or "STATUS" in c.upper():
                vs_cols.append(c)
                vs_rename[c] = c.lower()
        if len(vs_cols) > 1:
            vs_sub = vs[vs_cols].drop_duplicates(subset=["CODE98"]).rename(columns=vs_rename)
            panel = panel.merge(vs_sub, on="code98", how="left")

    # ── Derived variables ─────────────────────────────────────────
    # HOMA-IR
    panel["homa_ir"] = (panel["glucose"] * panel["insulin"]) / 405.0
    # Log-transform IL-6 (offset for zeros)
    panel["log_il6"] = np.log(panel["il6"] + 0.1)
    # Age decade
    panel["age_decade"] = (panel["age"] // 10) * 10
    # Medication class count
    med_cols = ["med_antihtn", "med_statin", "med_antidm", "med_nsaid", "med_glucocort"]
    panel["n_med_classes"] = panel[med_cols].apply(
        lambda row: row.dropna().sum() if row.notna().any() else np.nan, axis=1)
    # Comorbidity count
    dx_cols_list = [c for c in DX_VARS.keys() if c in panel.columns]
    if dx_cols_list:
        panel["n_comorbidities"] = panel[dx_cols_list].sum(axis=1)
    else:
        panel["n_comorbidities"] = np.nan

    # Drop helper columns
    for c in ["age_lab", "age_pe"]:
        if c in panel.columns:
            panel.drop(columns=[c], inplace=True)

    # Sort
    panel = panel.sort_values(["code98", "wave"]).reset_index(drop=True)

    return panel


def compute_youthful_reference(panel, age_min=20, age_max=30, expand_to=35):
    """
    Compute youthful reference (mean, SD) from healthy young adults.

    Parameters
    ----------
    panel : DataFrame from load_inchianti_panel()
    age_min, age_max : int
        Initial age window for reference group.
    expand_to : int
        If N_healthy < 50 in [age_min, age_max], expand upper bound to this.

    Returns
    -------
    dict : {axis_name: {mean, sd, n, log_transformed}}
    """
    # Baseline only
    bl = panel[panel["wave"] == 0].copy()

    # Healthy = no major diagnoses
    dx_exclude = ["dx_htn", "dx_dm", "dx_metsyn", "dx_frailty",
                  "dx_mi", "dx_chf", "dx_stroke", "dx_dement"]
    mask_healthy = pd.Series(True, index=bl.index)
    for dx in dx_exclude:
        if dx in bl.columns:
            mask_healthy &= (bl[dx] != 1) | bl[dx].isna()

    mask_age = (bl["age"] >= age_min) & (bl["age"] <= age_max)
    ref = bl[mask_healthy & mask_age]

    if len(ref) < 50:
        warnings.warn(f"Only {len(ref)} healthy {age_min}-{age_max}yo; expanding to {age_min}-{expand_to}")
        mask_age = (bl["age"] >= age_min) & (bl["age"] <= expand_to)
        ref = bl[mask_healthy & mask_age]

    axes = {
        "I": {"var": "log_il6", "log": True},
        "M": {"var": "homa_ir", "log": True},
        "N": {"var": "resting_hr", "log": False},
        "F": {"var": "sppb", "log": False},
    }

    result = {}
    for axis, info in axes.items():
        v = info["var"]
        vals = ref[v].dropna()
        if info["log"] and v != "log_il6":
            vals = np.log(vals + 0.1)
        result[axis] = {
            "var": v,
            "mean_ref": float(vals.mean()) if len(vals) > 0 else np.nan,
            "sd_ref": float(vals.std()) if len(vals) > 1 else np.nan,
            "n_ref": int(len(vals)),
            "log_transformed": info["log"],
        }

    # Handle SPPB ceiling effect: if SD=0 in young reference, use SD from
    # full healthy population aged <60 (SPPB has ceiling at 12 in young adults)
    if result["F"]["sd_ref"] == 0 or np.isnan(result["F"]["sd_ref"]):
        wider = bl[mask_healthy & (bl["age"] < 60)]
        sppb_wider = wider["sppb"].dropna()
        if len(sppb_wider) > 1 and sppb_wider.std() > 0:
            result["F"]["sd_ref"] = float(sppb_wider.std())
            result["F"]["sd_source"] = f"healthy <60 (N={len(sppb_wider)})"
        else:
            # Last resort: use full baseline SD
            sppb_all = bl["sppb"].dropna()
            result["F"]["sd_ref"] = float(sppb_all.std())
            result["F"]["sd_source"] = f"all baseline (N={len(sppb_all)})"

    result["_meta"] = {
        "age_range": f"{age_min}-{age_max}" + (f" (expanded to {expand_to})" if len(bl[mask_healthy & ((bl['age'] >= age_min) & (bl['age'] <= age_max))]) < 50 else ""),
        "n_healthy_young": int(len(ref)),
    }

    return result


def standardize_axes(panel, ref):
    """
    Standardize 4 axes to youthful reference with sign convention:
    positive = decline.

    Adds columns: delta_I, delta_M, delta_N, delta_F to the panel.
    """
    panel = panel.copy()

    # I-axis: IL-6 (log). Higher = worse → positive delta = decline ✓
    if ref["I"]["log_transformed"]:
        panel["delta_I"] = (panel["log_il6"] - ref["I"]["mean_ref"]) / ref["I"]["sd_ref"]
    else:
        panel["delta_I"] = (panel["il6"] - ref["I"]["mean_ref"]) / ref["I"]["sd_ref"]

    # M-axis: HOMA-IR (log). Higher = worse → positive delta = decline ✓
    log_homa = np.log(panel["homa_ir"] + 0.1)
    panel["delta_M"] = (log_homa - ref["M"]["mean_ref"]) / ref["M"]["sd_ref"]

    # N-axis: Resting HR. Higher HR = worse autonomic → positive delta = decline ✓
    panel["delta_N"] = (panel["resting_hr"] - ref["N"]["mean_ref"]) / ref["N"]["sd_ref"]

    # F-axis: SPPB. Lower = worse → SIGN FLIP: negative so that positive delta = decline
    panel["delta_F"] = -(panel["sppb"] - ref["F"]["mean_ref"]) / ref["F"]["sd_ref"]

    return panel
