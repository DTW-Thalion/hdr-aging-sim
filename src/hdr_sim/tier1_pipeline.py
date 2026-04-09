"""Compute all Tier-1 observables from cohort data.

These are the quantities that the R6 paper validated empirically.
The pipeline works identically on synthetic and real cohort data.

Tier-1 observables (require only sample covariance, no A estimation):
  1. lambda_max(Gamma_change)  per age stratum
  2. lambda_max(Gamma_cross)   per age stratum
  3. SWDS-Gamma                per individual
  4. Primacy ratio Pi          per age stratum
"""

import json
import os
from collections import defaultdict

import numpy as np

from .synthetic_cohort import CohortData


# Default age strata matching ELSA convention
DEFAULT_STRATA = [(50, 60), (60, 70), (70, 80), (80, 90)]
MIN_STRATUM_N = 30


class Tier1Pipeline:
    """Compute all Tier-1 observables from cohort data.

    These are the quantities that the R6 paper validated empirically.
    The pipeline works identically on synthetic and real cohort data.
    """

    def __init__(self, cohort_data):
        self._data = cohort_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _strata_indices(self, visit, age_strata):
        """Return dict mapping (lo, hi) -> list of person indices in stratum."""
        if age_strata is None:
            age_strata = DEFAULT_STRATA
        result = {}
        for lo, hi in age_strata:
            mask = (
                (self._data.visit_ages[:, visit] >= lo)
                & (self._data.visit_ages[:, visit] < hi)
                & self._data.alive[:, visit]
            )
            idx = np.where(mask)[0]
            if len(idx) >= MIN_STRATUM_N:
                result[(lo, hi)] = idx
        return result

    @staticmethod
    def _lambda_max(Gamma):
        """Largest eigenvalue of a symmetric matrix."""
        return float(np.max(np.linalg.eigvalsh(Gamma)))

    # ------------------------------------------------------------------
    # 1. Change covariance
    # ------------------------------------------------------------------

    def compute_gamma_change(self, age_strata=None):
        """Visit-pair change covariance per age stratum.

        For consecutive visits k, k+1:
          delta_y = y(visit_{k+1}) - y(visit_k)
        Pool delta_y within age strata.  Compute sample covariance.
        Extract lambda_max(Gamma_change) per stratum.

        Expected: increases with age.
        """
        if age_strata is None:
            age_strata = DEFAULT_STRATA
        d = self._data
        results = []

        for lo, hi in age_strata:
            deltas = []
            for k in range(d.n_visits - 1):
                mask = (
                    (d.visit_ages[:, k] >= lo)
                    & (d.visit_ages[:, k] < hi)
                    & d.alive[:, k]
                    & d.alive[:, k + 1]
                )
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                dy = d.observed[idx, k + 1] - d.observed[idx, k]
                deltas.append(dy)

            if not deltas:
                continue
            deltas = np.vstack(deltas)
            if len(deltas) < MIN_STRATUM_N:
                continue

            Gamma_c = np.cov(deltas.T)
            results.append({
                "age_lo": lo,
                "age_hi": hi,
                "age_mid": (lo + hi) / 2,
                "n": len(deltas),
                "lambda_max": self._lambda_max(Gamma_c),
                "trace": float(np.trace(Gamma_c)),
            })

        return results

    # ------------------------------------------------------------------
    # 2. Cross-sectional covariance
    # ------------------------------------------------------------------

    def compute_gamma_cross_sectional(self, age_strata=None, visit=0):
        """Cross-sectional covariance per stratum.

        Expected: DECREASES with age when survivorship is present.
        """
        if age_strata is None:
            age_strata = DEFAULT_STRATA
        d = self._data
        strata = self._strata_indices(visit, age_strata)
        results = []

        for (lo, hi), idx in strata.items():
            Y = d.observed[idx, visit]
            Gamma = np.cov(Y.T)
            results.append({
                "age_lo": lo,
                "age_hi": hi,
                "age_mid": (lo + hi) / 2,
                "n": len(idx),
                "lambda_max": self._lambda_max(Gamma),
                "trace": float(np.trace(Gamma)),
            })

        return results

    # ------------------------------------------------------------------
    # 3. SWDS-Gamma
    # ------------------------------------------------------------------

    def compute_swds(self, reference_stratum=None, visit=0):
        """SWDS-Gamma per individual.

        Uses the youngest stratum's Gamma as reference (matching R6).

        Returns: dict with per-stratum SWDS distributions.
        """
        d = self._data
        strata = self._strata_indices(visit, DEFAULT_STRATA)
        if not strata:
            return {}

        # Reference Gamma: youngest stratum
        if reference_stratum is None:
            ref_key = min(strata.keys(), key=lambda k: k[0])
        else:
            ref_key = reference_stratum

        if ref_key not in strata:
            return {}

        Y_ref = d.observed[strata[ref_key], visit]
        Gamma_ref = np.cov(Y_ref.T)
        trace_ref = np.trace(Gamma_ref)

        results = {}
        for (lo, hi), idx in strata.items():
            Y = d.observed[idx, visit]
            # SWDS_i = y_i^T Gamma_ref y_i / tr(Gamma_ref)
            scores = np.sum((Y @ Gamma_ref) * Y, axis=1) / trace_ref
            results[(lo, hi)] = {
                "age_lo": lo,
                "age_hi": hi,
                "age_mid": (lo + hi) / 2,
                "n": len(idx),
                "swds_mean": float(np.mean(scores)),
                "swds_std": float(np.std(scores)),
                "swds_median": float(np.median(scores)),
                "swds_q25": float(np.percentile(scores, 25)),
                "swds_q75": float(np.percentile(scores, 75)),
                "swds_values": scores,
            }

        return results

    # ------------------------------------------------------------------
    # 4. Primacy ratio
    # ------------------------------------------------------------------

    def compute_primacy_ratio(self, age_strata=None, visit=0):
        """Decompose Gamma into V_norm and C_norm.

        V_norm: mean diagonal of Gamma (within-axis variance)
        C_norm: mean |off-diagonal| of correlation matrix (cross-axis)
        Pi(s) = C_norm(s) / V_norm(s)
        """
        if age_strata is None:
            age_strata = DEFAULT_STRATA
        d = self._data
        strata = self._strata_indices(visit, age_strata)
        results = []

        ref_V = None
        ref_C = None

        for (lo, hi), idx in sorted(strata.items()):
            Y = d.observed[idx, visit]
            Gamma = np.cov(Y.T)
            n_ax = Gamma.shape[0]

            # V: mean diagonal variance
            variances = np.diag(Gamma)
            V = float(np.mean(variances))

            # Correlation matrix
            D_inv = np.diag(1.0 / np.sqrt(np.maximum(variances, 1e-12)))
            R_hat = D_inv @ Gamma @ D_inv

            # C: mean |off-diagonal correlation|
            off_diag = []
            for a in range(n_ax):
                for b in range(a + 1, n_ax):
                    off_diag.append(abs(R_hat[a, b]))
            C = float(np.mean(off_diag)) if off_diag else 0.0

            if ref_V is None:
                ref_V = V
                ref_C = C

            V_norm = V / ref_V if ref_V > 0 else 0.0
            C_norm = C / ref_C if ref_C > 0 else 0.0
            Pi = C_norm / V_norm if V_norm > 0 else float("nan")

            results.append({
                "age_lo": lo,
                "age_hi": hi,
                "age_mid": (lo + hi) / 2,
                "n": len(idx),
                "V": V,
                "C": C,
                "V_norm": float(V_norm),
                "C_norm": float(C_norm),
                "Pi": float(Pi),
            })

        return results

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def full_analysis(self, output_dir="results"):
        """Run complete Tier-1 analysis and produce report.

        Returns: dict of all computed quantities.
        """
        gamma_change = self.compute_gamma_change()
        gamma_cross = self.compute_gamma_cross_sectional()
        swds = self.compute_swds()
        primacy = self.compute_primacy_ratio()

        # Serialisable SWDS (strip numpy arrays)
        swds_serial = {}
        for key, val in swds.items():
            v = dict(val)
            v.pop("swds_values", None)
            swds_serial[f"{key[0]}-{key[1]}"] = v

        result = {
            "gamma_change": gamma_change,
            "gamma_cross_sectional": gamma_cross,
            "swds": swds_serial,
            "primacy_ratio": primacy,
            "summary": {
                "gamma_change_increases": self._check_monotone(
                    gamma_change, "lambda_max"
                ),
                "swds_increases": self._check_monotone_swds(swds),
            },
        }

        return result

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_monotone(records, key):
        """Check if key increases across age-sorted records."""
        if len(records) < 2:
            return False
        vals = [r[key] for r in sorted(records, key=lambda r: r["age_mid"])]
        return all(vals[i + 1] > vals[i] for i in range(len(vals) - 1))

    @staticmethod
    def _check_monotone_swds(swds_dict):
        """Check if mean SWDS increases across age strata."""
        if len(swds_dict) < 2:
            return False
        items = sorted(swds_dict.items(), key=lambda kv: kv[0][0])
        means = [v["swds_mean"] for _, v in items]
        return all(means[i + 1] > means[i] for i in range(len(means) - 1))
