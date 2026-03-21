"""HDR Aging Dynamics Simulation and Estimation Library."""
from .csv_loader import load_J_csv, build_J_basin, get_J_anchors
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
