"""HDR Aging Dynamics Simulation and Estimation Library.

Part 1 (R1-R6): 4-axis toy simulation and ELSA validation.
Part 2: 9-axis mechanistic-evidence-informed model with two-timescale
decomposition (7-axis fast subsystem + quasi-static E/B forcing),
sensitivity analysis, synthetic cohort generation, intervention
framework, and Bayesian prior updating scaffold.

Key entry points:
    configure(axes=('I','M','F'))      — legacy setup (ages 30-80, linear interp)
    configure_v2(axes=('I','M','F'))   — V2 lit-calibrated (ages 25-120, Gompertz)
    tau_of_age(age), J_of_age(age)     — age-interpolated parameters
    tau_at_age(axis, age)              — V2 single-axis tau at any age
    tau_vector(axes, age)              — V2 multi-axis tau vector
    JMatrixSpec.from_csv(path)         — provenance tracking for J-matrix CSVs
"""
from .csv_loader import (
    load_J_csv, build_J_basin, build_J_basin_imputed,
    get_J_anchors, get_J_anchors_v2,
    TAU_REGISTRY, TAU_REGISTRY_LEGACY, TAU_REGISTRY_V2,
    tau_at_age, tau_vector, J_at_age,
    calibrate_three_point,
    calibrate_fast_subsystem, calibrate_stable_system,
    find_j_blend_amplitude,
    j_blend_fraction, j_at_age_blended, build_system_at_age,
    _extract_submatrix, _ALL_9_AXES, _FAST_7_AXES, _FAST_6_AXES,
    _SLOW_2_AXES, _SLOW_3_AXES,
)
from .aging_params import configure, configure_v2, tau_of_age, J_of_age, get_axis_names, get_axis_colors
from .j_matrix_spec import JMatrixSpec
from .estimation import (
    get_params,
    build_A,
    spectral_abscissa,
    rho_discrete,
    stationary_covariance,
    lyapunov_inversion_symmetric,
    estimate_A_lyapunov_full,
    estimate_A_sign_constrained,
    partial_correlations,
    sign_concordance,
    estimate_tau_from_recovery,
    generate_stratum,
    stability_weighted_score,
    # R4 Γ-native additions
    compute_swds_gamma,
    compute_swds_gamma_batch,
    gamma_stability_proxy,
    covariance_sign_concordance,
    lyapunov_residual_norm,
)

# Part 2: Mechanistic-evidence-informed model classes
from .mechanistic_model import HDRMechanisticModel
from .state_conditioned import StateSwitchedModel
from .observation_model import ObservationModel
from .sensitivity import PriorSensitivityAnalysis, MCResults
from .prior_stress import PriorStressTest
from .synthetic_cohort import SyntheticCohort, CohortData
from .tier1_pipeline import Tier1Pipeline
from .intervention import InterventionModel
from .trial_simulator import TrialSimulator
from .bayesian_update import BayesianPriorUpdate, ABCResults
