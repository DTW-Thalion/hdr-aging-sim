"""HDR Aging Dynamics Simulation and Estimation Library.

Part 1 (R1-R6): 4-axis toy simulation and ELSA validation.
Part 2: 9-axis mechanistic-evidence-informed model with two-timescale
decomposition (7-axis fast subsystem + quasi-static E/B forcing),
sensitivity analysis, synthetic cohort generation, intervention
framework, and Bayesian prior updating scaffold.

Key entry points:
    configure(axes=('I','M','F'))   — set up aging model for an axis subset
    tau_of_age(age), J_of_age(age)  — age-interpolated parameters
    JMatrixSpec.from_csv(path)      — provenance tracking for J-matrix CSVs
"""
from .csv_loader import load_J_csv, build_J_basin, get_J_anchors, TAU_REGISTRY
from .aging_params import configure, tau_of_age, J_of_age, get_axis_names, get_axis_colors
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
