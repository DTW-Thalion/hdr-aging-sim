"""Observation model mapping 9-dimensional latent HDR state to biomarkers.

Each cohort configuration defines which axes are observed, through which
biomarkers, and with what observation noise.

The observation model is:  y(t_k) = C * Δx(t_k) + v_k,  v_k ~ N(0, R)
C projects the 9-dim latent state to the observable biomarker space.
"""

import numpy as np


class ObservationModel:
    """Maps 9-dimensional latent HDR state to measurable biomarkers.

    Each cohort configuration defines which axes are observed, through
    which biomarkers, and with what observation noise.
    """

    COHORT_CONFIGS = {
        "ELSA_3axis": {
            "observed_axes": [0, 1, 7],  # I, M, F
            "biomarker_names": ["log_CRP", "HbA1c_BMI", "grip_strength"],
            "sign_reversal": [False, False, True],  # grip is reversed
            "observation_noise_std": 0.3,
        },
        "InCHIANTI_4axis": {
            "observed_axes": [0, 1, 6, 7],  # I, M, N, F
            "biomarker_names": ["IL6", "HOMA_IR", "RMSSD", "SPPB_grip_gait"],
            "sign_reversal": [False, False, True, True],
            "observation_noise_std": 0.25,
        },
        "full_9axis": {
            "observed_axes": list(range(9)),
            "biomarker_names": [
                "I", "M", "E", "mito", "P", "C", "N", "F", "B",
            ],
            "sign_reversal": [False] * 9,
            "observation_noise_std": 0.2,
        },
    }

    def __init__(self, cohort="ELSA_3axis"):
        """Build C matrix and R for the specified cohort."""
        if cohort not in self.COHORT_CONFIGS:
            raise ValueError(
                f"Unknown cohort {cohort!r}. "
                f"Choose from {list(self.COHORT_CONFIGS)}"
            )

        cfg = self.COHORT_CONFIGS[cohort]
        self._cohort = cohort
        self._observed_axes = cfg["observed_axes"]
        self._biomarker_names = cfg["biomarker_names"]
        self._sign_reversal = cfg["sign_reversal"]
        self._noise_std = cfg["observation_noise_std"]
        self._n_obs = len(self._observed_axes)
        self._n_latent = 9

        # Build C matrix (n_obs x 9)
        self._C = np.zeros((self._n_obs, self._n_latent))
        for i, ax_idx in enumerate(self._observed_axes):
            sign = -1.0 if self._sign_reversal[i] else 1.0
            self._C[i, ax_idx] = sign

        # Build R matrix (n_obs x n_obs)
        self._R = (self._noise_std ** 2) * np.eye(self._n_obs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def C(self):
        """Observation matrix (n_obs x 9)."""
        return self._C.copy()

    @property
    def R(self):
        """Observation noise covariance (n_obs x n_obs)."""
        return self._R.copy()

    @property
    def n_obs(self):
        """Number of observed biomarkers."""
        return self._n_obs

    @property
    def biomarker_names(self):
        """List of biomarker names."""
        return list(self._biomarker_names)

    @property
    def observed_axes(self):
        """Indices of observed axes in the 9-dim state."""
        return list(self._observed_axes)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, x):
        """Project 9-dim latent state to observable biomarker space.

        Args:
            x: (9,) or (N, 9) array of latent states.
        Returns:
            y: (n_obs,) or (N, n_obs) array of projected observables.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return self._C @ x
        return (self._C @ x.T).T

    def observe(self, x, seed=None):
        """Project and add observation noise.

        y = C @ x + v,  v ~ N(0, R)

        Args:
            x: (9,) or (N, 9) array of latent states.
            seed: random seed for noise generation.
        Returns:
            y: (n_obs,) or (N, n_obs) array of noisy observables.
        """
        rng = np.random.default_rng(seed)
        y_clean = self.project(x)
        if y_clean.ndim == 1:
            noise = rng.multivariate_normal(
                np.zeros(self._n_obs), self._R
            )
        else:
            noise = rng.multivariate_normal(
                np.zeros(self._n_obs), self._R, size=y_clean.shape[0]
            )
        return y_clean + noise

    def project_covariance(self, Gamma):
        """Project latent covariance to observation space.

        Γ_obs = C @ Γ @ C^T + R

        Args:
            Gamma: (9, 9) latent covariance.
        Returns:
            Gamma_obs: (n_obs, n_obs) observed covariance.
        """
        return self._C @ Gamma @ self._C.T + self._R
