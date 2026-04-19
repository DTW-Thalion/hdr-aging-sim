"""Bootstrap confidence intervals for delta-C (Harrell's C difference
between nested Cox models).

Used by scripts/run_bootstrap_delta_c.py to produce 95% CIs for the
ΔC(M5 - M4) increment reported in the manuscript for both InCHIANTI
and ELSA. Event-stratified resampling preserves the number of deaths
across bootstrap samples, which is important when events are sparse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_cox(data: pd.DataFrame, covs, time_col: str, event_col: str,
             penalizer: float):
    """Fit a Cox model with a small ridge penalty as fallback."""
    from lifelines import CoxPHFitter

    cols = list(dict.fromkeys(list(covs) + [time_col, event_col]))
    sub = data[cols]
    try:
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        return cph.concordance_index_
    except Exception:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(sub, duration_col=time_col, event_col=event_col)
        return cph.concordance_index_


def compute_delta_c(surv_data: pd.DataFrame, m4_covs, m5_covs,
                    time_col: str = "time", event_col: str = "event",
                    penalizer: float = 0.01) -> dict:
    """Single-fit point estimate of C(M4), C(M5), and their difference."""
    c4 = _fit_cox(surv_data, m4_covs, time_col, event_col, penalizer)
    c5 = _fit_cox(surv_data, m5_covs, time_col, event_col, penalizer)
    return {
        "c_m4": float(c4),
        "c_m5": float(c5),
        "delta_c": float(c5 - c4),
        "n": int(len(surv_data)),
        "events": int(surv_data[event_col].sum()),
    }


def bootstrap_delta_c(surv_data: pd.DataFrame, m4_covs, m5_covs,
                      time_col: str = "time", event_col: str = "event",
                      n_boot: int = 1000, seed: int = 42,
                      penalizer: float = 0.01,
                      verbose: bool = True, progress_every: int = 200) -> dict:
    """Event-stratified bootstrap CI for ΔC = C(M5) - C(M4).

    Each bootstrap sample resamples separately within event==0 and event==1
    strata (with replacement, preserving per-stratum size) so the number of
    deaths in every resample matches the original.

    Returns
    -------
    dict with keys:
        delta_c_point      - ΔC on the full (unresampled) sample
        c_m4_point         - C(M4) on the full sample
        c_m5_point         - C(M5) on the full sample
        delta_c_median     - median of bootstrap ΔC
        delta_c_mean       - mean of bootstrap ΔC
        delta_c_se         - SD of bootstrap ΔC (bootstrap standard error)
        delta_c_ci_lo      - 2.5 percentile of bootstrap ΔC
        delta_c_ci_hi      - 97.5 percentile
        p_value            - fraction of bootstrap samples with ΔC <= 0
                             (one-sided test against H0: M5 does not improve M4)
        ci_excludes_zero   - bool (delta_c_ci_lo > 0)
        ci_excludes_0_01   - bool (delta_c_ci_lo > 0.01)
        n                  - original sample size
        events             - original number of events
        n_boot             - resamples requested
        n_successful       - resamples that fit without error
        n_failed           - n_boot - n_successful
        seed, penalizer, m4_covs, m5_covs
    """
    rng = np.random.RandomState(seed)

    # Single-fit point estimate on the full sample
    point = compute_delta_c(surv_data, m4_covs, m5_covs,
                            time_col=time_col, event_col=event_col,
                            penalizer=penalizer)

    # Index by event for stratified resampling
    event_vals = surv_data[event_col].values
    idx0 = np.where(event_vals == 0)[0]
    idx1 = np.where(event_vals == 1)[0]
    n0, n1 = len(idx0), len(idx1)

    if verbose:
        print(f"    Bootstrap: N={len(surv_data)} (events={n1}, censored={n0}), "
              f"n_boot={n_boot}")

    delta_cs = []
    n_failed = 0

    for i in range(n_boot):
        b_idx = np.concatenate([
            rng.choice(idx0, size=n0, replace=True),
            rng.choice(idx1, size=n1, replace=True),
        ])
        boot = surv_data.iloc[b_idx].reset_index(drop=True)

        try:
            c4 = _fit_cox(boot, m4_covs, time_col, event_col, penalizer)
            c5 = _fit_cox(boot, m5_covs, time_col, event_col, penalizer)
            delta_cs.append(c5 - c4)
        except Exception:
            n_failed += 1
            continue

        if verbose and progress_every and (i + 1) % progress_every == 0:
            arr = np.array(delta_cs)
            print(f"    ... {i+1:>5d}/{n_boot} done "
                  f"(failed={n_failed}, running median={np.median(arr):+.4f}, "
                  f"2.5%={np.percentile(arr, 2.5):+.4f}, "
                  f"97.5%={np.percentile(arr, 97.5):+.4f})")

    delta_cs = np.array(delta_cs)
    ci_lo = float(np.percentile(delta_cs, 2.5))
    ci_hi = float(np.percentile(delta_cs, 97.5))

    return {
        "delta_c_point": point["delta_c"],
        "c_m4_point": point["c_m4"],
        "c_m5_point": point["c_m5"],
        "delta_c_median": float(np.median(delta_cs)),
        "delta_c_mean": float(np.mean(delta_cs)),
        "delta_c_se": float(np.std(delta_cs, ddof=1)) if len(delta_cs) > 1 else float("nan"),
        "delta_c_ci_lo": ci_lo,
        "delta_c_ci_hi": ci_hi,
        "ci_95": [ci_lo, ci_hi],
        "p_value": float(np.mean(delta_cs <= 0)),
        "ci_excludes_zero": bool(ci_lo > 0),
        "ci_excludes_0_01": bool(ci_lo > 0.01),
        "n": int(len(surv_data)),
        "events": int(surv_data[event_col].sum()),
        "n_boot": int(n_boot),
        "n_successful": int(len(delta_cs)),
        "n_failed": int(n_failed),
        "seed": int(seed),
        "penalizer": float(penalizer),
        "m4_covs": list(m4_covs),
        "m5_covs": list(m5_covs),
        "stratified_by_event": True,
    }
