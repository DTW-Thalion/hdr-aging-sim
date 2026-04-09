"""Observation model mapping latent HDR state to biomarkers.

Each cohort configuration defines which axes are observed, through which
biomarkers, and with what observation noise.

The observation model is:  y(t_k) = C * Δx(t_k) + v_k,  v_k ~ N(0, R)
C projects the latent state to the observable biomarker space.

Supports both the fast subsystem (7-dim, default) and full system (9-dim).
Axis names are resolved to indices based on the provided axis list.
"""

import numpy as np


# Default fast axes (matches HDRMechanisticModel.FAST_AXES)
_DEFAULT_AXES = ["I", "M", "mito", "P", "C", "N", "F"]
_FULL_AXES = ["I", "M", "E", "mito", "P", "C", "N", "F", "B"]


class ObservationModel:
    """Maps latent HDR state to measurable biomarkers.

    Each cohort configuration defines which axes are observed, through
    which biomarkers, and with what observation noise.
    """

    COHORT_CONFIGS = {
        "ELSA_3axis": {
            "observed_axes": ["I", "M", "F"],
            "biomarker_names": ["log_CRP", "HbA1c_BMI", "grip_strength"],
            "sign_reversal": [False, False, True],
            "observation_noise_std": 0.3,
        },
        "InCHIANTI_4axis": {
            "observed_axes": ["I", "M", "N", "F"],
            "biomarker_names": ["IL6", "HOMA_IR", "RMSSD", "SPPB_grip_gait"],
            "sign_reversal": [False, False, True, True],
            "observation_noise_std": 0.25,
        },
        "full_fast": {
            "observed_axes": ["I", "M", "mito", "P", "C", "N", "F"],
            "biomarker_names": [
                "I", "M", "mito", "P", "C", "N", "F",
            ],
            "sign_reversal": [False] * 7,
            "observation_noise_std": 0.2,
        },
        "full_9axis": {
            "observed_axes": ["I", "M", "E", "mito", "P", "C", "N", "F", "B"],
            "biomarker_names": [
                "I", "M", "E", "mito", "P", "C", "N", "F", "B",
            ],
            "sign_reversal": [False] * 9,
            "observation_noise_std": 0.2,
        },
    }

    def __init__(self, cohort="ELSA_3axis", axes=None):
        """Build C matrix and R for the specified cohort.

        Args:
            cohort: name of the cohort configuration.
            axes: list of axis names defining the latent state ordering.
                  Default: fast subsystem axes (I, M, mito, P, C, N, F).
                  Pass _FULL_AXES or model.AXES for the full 9-dim system.
        """
        if cohort not in self.COHORT_CONFIGS:
            raise ValueError(
                f"Unknown cohort {cohort!r}. "
                f"Choose from {list(self.COHORT_CONFIGS)}"
            )

        if axes is None:
            axes = list(_DEFAULT_AXES)

        cfg = self.COHORT_CONFIGS[cohort]
        self._cohort = cohort
        self._axes = list(axes)
        self._axis_idx = {a: i for i, a in enumerate(self._axes)}
        self._n_latent = len(self._axes)

        # Resolve axis names to indices
        obs_names = cfg["observed_axes"]
        self._observed_axis_names = obs_names
        self._observed_axes = []
        self._biomarker_names = []
        self._sign_reversal = []

        for k, ax_name in enumerate(obs_names):
            if ax_name not in self._axis_idx:
                continue  # skip axes not in the latent state
            self._observed_axes.append(self._axis_idx[ax_name])
            self._biomarker_names.append(cfg["biomarker_names"][k])
            self._sign_reversal.append(cfg["sign_reversal"][k])

        self._noise_std = cfg["observation_noise_std"]
        self._n_obs = len(self._observed_axes)

        # Build C matrix (n_obs x n_latent)
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
        """Observation matrix (n_obs x n_latent)."""
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
    def n_latent(self):
        """Dimension of the latent state."""
        return self._n_latent

    @property
    def biomarker_names(self):
        """List of biomarker names."""
        return list(self._biomarker_names)

    @property
    def observed_axes(self):
        """Indices of observed axes in the latent state."""
        return list(self._observed_axes)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, x):
        """Project latent state to observable biomarker space.

        Args:
            x: (n_latent,) or (N, n_latent) array.
        Returns:
            y: (n_obs,) or (N, n_obs) array.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return self._C @ x
        return (self._C @ x.T).T

    def observe(self, x, seed=None):
        """Project and add observation noise.

        y = C @ x + v,  v ~ N(0, R)
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
        """
        return self._C @ Gamma @ self._C.T + self._R
