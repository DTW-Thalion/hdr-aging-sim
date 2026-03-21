"""
Unit tests for ELSA validation pipeline.
"""

import numpy as np
import pytest
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from hdr_sim.estimation import compute_swds_gamma, compute_swds_gamma_batch, gamma_stability_proxy


# ---------------------------------------------------------------------------
# HbA1c conversion
# ---------------------------------------------------------------------------
class TestHbA1cConversion:
    """Test DCCT ↔ IFCC conversion logic."""

    def test_dcct_to_ifcc_known_values(self):
        """Verify conversion formula: IFCC = (DCCT - 2.15) × 10.929"""
        # 5.0% DCCT → (5.0 - 2.15) * 10.929 = 31.148 mmol/mol
        assert abs((5.0 - 2.15) * 10.929 - 31.148) < 0.01

        # 6.5% DCCT → (6.5 - 2.15) * 10.929 = 47.541 mmol/mol
        assert abs((6.5 - 2.15) * 10.929 - 47.541) < 0.01

        # 7.0% DCCT → (7.0 - 2.15) * 10.929 = 53.006 mmol/mol
        assert abs((7.0 - 2.15) * 10.929 - 53.006) < 0.01

    def test_unit_detection_dcct(self):
        """Median < 15 should be detected as DCCT."""
        from run_elsa_validation import detect_and_convert_hba1c
        import pandas as pd

        dcct_values = pd.Series([5.0, 5.5, 6.0, 6.5, 7.0, 8.0])
        converted, unit = detect_and_convert_hba1c(dcct_values)
        assert unit == 'DCCT'
        # Converted values should all be > 15 (IFCC range)
        assert converted.min() > 15

    def test_unit_detection_ifcc(self):
        """Median ≥ 15 should be detected as already IFCC."""
        from run_elsa_validation import detect_and_convert_hba1c
        import pandas as pd

        ifcc_values = pd.Series([31.0, 35.0, 42.0, 48.0, 53.0])
        converted, unit = detect_and_convert_hba1c(ifcc_values)
        assert unit == 'IFCC'
        # Values should be unchanged
        np.testing.assert_array_almost_equal(converted.values, ifcc_values.values)

    def test_empty_series(self):
        """Empty series should return 'unknown'."""
        from run_elsa_validation import detect_and_convert_hba1c
        import pandas as pd

        empty = pd.Series(dtype=float)
        _, unit = detect_and_convert_hba1c(empty)
        assert unit == 'unknown'


# ---------------------------------------------------------------------------
# SWDS-Γ on known covariance
# ---------------------------------------------------------------------------
class TestSWDSGamma:
    """Test SWDS-Γ computation with hand-computed examples."""

    def test_identity_covariance(self):
        """With Γ = I, SWDS-Γ = ||Δx||² / n."""
        n = 3
        Gamma = np.eye(n)
        delta_x = np.array([1.0, 0.0, 0.0])
        score = compute_swds_gamma(delta_x, Gamma)
        expected = 1.0 / 3.0  # ||[1,0,0]||² / tr(I₃) = 1/3
        assert abs(score - expected) < 1e-10

    def test_diagonal_covariance(self):
        """With Γ = diag(λ₁, λ₂, λ₃), verify quadratic form."""
        Gamma = np.diag([4.0, 2.0, 1.0])
        delta_x = np.array([1.0, 1.0, 1.0])
        score = compute_swds_gamma(delta_x, Gamma)
        # Δxᵀ Γ Δx = 4 + 2 + 1 = 7, tr(Γ) = 7, score = 1.0
        assert abs(score - 1.0) < 1e-10

    def test_batch_matches_individual(self):
        """Batch computation should match individual computation."""
        np.random.seed(42)
        Gamma = np.array([[2.0, 0.5, 0.1],
                          [0.5, 1.5, 0.2],
                          [0.1, 0.2, 1.0]])
        X = np.random.randn(100, 3)

        batch_scores = compute_swds_gamma_batch(X, Gamma)
        individual_scores = np.array([
            compute_swds_gamma(X[i], Gamma) for i in range(100)
        ])
        np.testing.assert_array_almost_equal(batch_scores, individual_scores)

    def test_zero_vector(self):
        """Zero displacement should give zero score."""
        Gamma = np.eye(3) * 2.0
        score = compute_swds_gamma(np.zeros(3), Gamma)
        assert score == 0.0

    def test_nonnegative(self):
        """SWDS-Γ should always be non-negative for PD covariance."""
        np.random.seed(42)
        # Generate a random PD matrix
        A = np.random.randn(3, 3)
        Gamma = A @ A.T + np.eye(3)

        for _ in range(100):
            dx = np.random.randn(3)
            score = compute_swds_gamma(dx, Gamma)
            assert score >= 0


# ---------------------------------------------------------------------------
# Gamma stability proxy
# ---------------------------------------------------------------------------
class TestGammaStabilityProxy:
    """Test gamma_stability_proxy on known matrices."""

    def test_identity_matrix(self):
        """Identity matrix: all eigenvalues = 1, κ = 1."""
        proxy = gamma_stability_proxy(np.eye(3))
        assert abs(proxy['lambda_max'] - 1.0) < 1e-10
        assert abs(proxy['lambda_min'] - 1.0) < 1e-10
        assert abs(proxy['kappa'] - 1.0) < 1e-10
        assert abs(proxy['trace'] - 3.0) < 1e-10

    def test_known_eigenvalues(self):
        """Diagonal matrix with known eigenvalues."""
        Gamma = np.diag([5.0, 2.0, 1.0])
        proxy = gamma_stability_proxy(Gamma)
        assert abs(proxy['lambda_max'] - 5.0) < 1e-10
        assert abs(proxy['lambda_min'] - 1.0) < 1e-10
        assert abs(proxy['kappa'] - 5.0) < 1e-10
        assert abs(proxy['trace'] - 8.0) < 1e-10

    def test_eigenvalues_descending(self):
        """Eigenvalues should be returned in descending order."""
        np.random.seed(42)
        A = np.random.randn(4, 4)
        Gamma = A @ A.T + np.eye(4)
        proxy = gamma_stability_proxy(Gamma)
        evals = proxy['eigenvalues']
        for i in range(len(evals) - 1):
            assert evals[i] >= evals[i + 1]


# ---------------------------------------------------------------------------
# Frailty Index
# ---------------------------------------------------------------------------
class TestFrailtyIndex:
    """Test Rockwood FI range and validity."""

    def test_fi_range(self):
        """FI should be in [0, 1] when valid."""
        from run_elsa_validation import compute_rockwood_fi
        import pandas as pd

        # All deficits present — single merged row with canonical names
        row = pd.Series({
            # Canonical harmonised condition names
            'diabetes': 1, 'highbp': 1, 'heart': 1,
            'stroke': 1, 'lung': 1, 'arthritis': 1,
            'cesd': 6,
            # Supplementary ADL/IADL/mobility/health/meds
            'headlba': 1, 'headlea': 1, 'headlbe': 1,
            'headlwc': 1, 'headldr': 1, 'headlwa': 1,
            'headlda': 1, 'headlpr': 1, 'headlsh': 1,
            'headlph': 1, 'headlco': 1, 'headlme': 1,
            'headlho': 1, 'headlmo': 1,
            'hemobwa': 1, 'hemobsi': 1, 'hemobch': 1,
            'hemobcs': 1, 'hemobcl': 1, 'hemobst': 1,
            'hemobre': 1, 'hemobpu': 1, 'hemobli': 1,
            'hemobpi': 1,
            'hehelf': 5, 'hemda': 1, 'hemdb': 1,
        })
        fi = compute_rockwood_fi(row)
        assert 0.0 <= fi <= 1.0
        # All deficits → FI should be high
        assert fi > 0.8

    def test_fi_zero_deficits(self):
        """No deficits → FI = 0."""
        from run_elsa_validation import compute_rockwood_fi
        import pandas as pd

        row = pd.Series({
            # Canonical harmonised condition names
            'diabetes': 0, 'highbp': 0, 'heart': 0,
            'stroke': 0, 'lung': 0, 'arthritis': 0,
            'cesd': 1,
            # Supplementary ADL/IADL/mobility/health/meds
            'headlba': 0, 'headlea': 0, 'headlbe': 0,
            'headlwc': 0, 'headldr': 0, 'headlwa': 0,
            'headlda': 0, 'headlpr': 0, 'headlsh': 0,
            'headlph': 0, 'headlco': 0, 'headlme': 0,
            'headlho': 0, 'headlmo': 0,
            'hemobwa': 0, 'hemobsi': 0, 'hemobch': 0,
            'hemobcs': 0, 'hemobcl': 0, 'hemobst': 0,
            'hemobre': 0, 'hemobpu': 0, 'hemobli': 0,
            'hemobpi': 0,
            'hehelf': 1, 'hemda': 0, 'hemdb': 0,
        })
        fi = compute_rockwood_fi(row)
        assert fi == 0.0

    def test_fi_insufficient_items(self):
        """FI with < 10 non-missing items should return NaN."""
        from run_elsa_validation import compute_rockwood_fi
        import pandas as pd

        row = pd.Series({'diabetes': 1, 'cesd': 3, 'headlba': 1})  # Only 3 items
        fi = compute_rockwood_fi(row)
        assert np.isnan(fi)


# ---------------------------------------------------------------------------
# Grip max selection
# ---------------------------------------------------------------------------
class TestGripMax:
    """Test max-of-dominant-hand grip logic."""

    def test_grip_max_selection(self):
        """Max of 3 trials should be selected."""
        import pandas as pd
        df = pd.DataFrame({
            'mmgsd1': [30.0, 25.0, np.nan],
            'mmgsd2': [32.0, 28.0, 20.0],
            'mmgsd3': [31.0, np.nan, 22.0],
        })
        grip_max = df[['mmgsd1', 'mmgsd2', 'mmgsd3']].max(axis=1)
        assert grip_max.iloc[0] == 32.0
        assert grip_max.iloc[1] == 28.0
        assert grip_max.iloc[2] == 22.0

    def test_grip_max_all_nan(self):
        """All NaN trials should give NaN."""
        import pandas as pd
        df = pd.DataFrame({
            'mmgsd1': [np.nan],
            'mmgsd2': [np.nan],
            'mmgsd3': [np.nan],
        })
        grip_max = df[['mmgsd1', 'mmgsd2', 'mmgsd3']].max(axis=1)
        assert np.isnan(grip_max.iloc[0])


# ---------------------------------------------------------------------------
# Missing value filtering
# ---------------------------------------------------------------------------
class TestMissingValueFilter:
    """Test ELSA missing code filtering."""

    def test_negative_codes_replaced(self):
        """All negative values should be replaced with NaN."""
        from run_elsa_validation import filter_missing
        import pandas as pd

        df = pd.DataFrame({
            'idauniq': [1, 2, 3],
            'hscrp': [2.5, -1, -9],
            'hba1c': [-2, 5.5, -8],
        })
        result = filter_missing(df, exclude_cols=['idauniq'])
        assert result['idauniq'].tolist() == [1, 2, 3]  # unchanged
        assert result['hscrp'].iloc[0] == 2.5
        assert np.isnan(result['hscrp'].iloc[1])
        assert np.isnan(result['hscrp'].iloc[2])
        assert np.isnan(result['hba1c'].iloc[0])
        assert result['hba1c'].iloc[1] == 5.5
