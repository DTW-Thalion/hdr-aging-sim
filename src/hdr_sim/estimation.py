"""
Reusable estimation functions for HDR aging framework.

Implements Lyapunov-inversion stability estimation, constrained OU drift
estimation, partial correlation computation, and recovery time constant
estimation from perturbation-recovery episodes.

Used by: run_figure_recoverability.py, run_figure_uncertainty.py
Reference: HDR Ontology Manuscript R2, Appendices C-F
"""

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, inv
from scipy.optimize import minimize


def get_params(age, n_axes=4):
    """
    Return (tau, J) at a given age for the 4-axis model (I, M, N, F).
    Linear interpolation from age 30 to age 80.
    Parameters match Section 2.6 of the ontology manuscript.
    """
    t = np.clip((age - 30) / 50, 0, 1)
    tau_30 = np.array([7.0, 0.1, 0.01, 8.0])
    tau_80 = np.array([21.0, 0.4, 0.04, 42.0])
    tau = tau_30 * (1 - t) + tau_80 * t
    J_30 = np.array([
        [0.0,   0.015, 0.005, -0.04],
        [0.02,  0.0,   0.005, -0.06],
        [0.01,  0.008, 0.0,   -0.03],
        [0.01,  0.01,  0.005,  0.0],
    ])
    J_80 = np.array([
        [0.0,   0.08,  0.025, -0.015],
        [0.10,  0.0,   0.025, -0.02],
        [0.05,  0.04,  0.0,   -0.01],
        [0.06,  0.06,  0.03,   0.0],
    ])
    J = J_30 * (1 - t) + J_80 * t
    return tau, J


def build_A(tau, J):
    """Build dynamics matrix A = -D + J."""
    return -np.diag(1.0 / tau) + J


def spectral_abscissa(A):
    """Compute max Re(eigenvalue) of A."""
    return float(np.max(np.real(np.linalg.eigvals(A))))


def rho_discrete(A, dt):
    """Compute spectral radius of discrete transition matrix exp(A*dt)."""
    Phi = expm(A * dt)
    return float(np.max(np.abs(np.linalg.eigvals(Phi))))


def stationary_covariance(A, Q):
    """Compute stationary covariance Gamma from Lyapunov equation A*Gamma + Gamma*A^T = -Q."""
    return solve_continuous_lyapunov(A, -Q)


def lyapunov_inversion_symmetric(Gamma_hat, Q):
    """
    Estimate A from stationary covariance via symmetric Lyapunov approximation.
    A_hat = -Q @ Gamma^{-1} / 2
    Exact when A is symmetric; biased for asymmetric A (underestimates |alpha|).
    Reference: Appendix F of the ontology manuscript.
    """
    P = inv(Gamma_hat)
    return -Q @ P / 2


def estimate_A_lyapunov_full(Gamma_hat, Q, maxiter=5000):
    """
    Estimate A from stationary covariance by minimizing the Lyapunov residual.
    Solves: min_A ||A @ Gamma + Gamma @ A^T + Q||_F^2
    Handles asymmetric A via L-BFGS-B optimization.
    """
    n = Gamma_hat.shape[0]

    def objective(a_flat):
        A = a_flat.reshape(n, n)
        residual = A @ Gamma_hat + Gamma_hat @ A.T + Q
        return 0.5 * np.sum(residual**2)

    def gradient(a_flat):
        A = a_flat.reshape(n, n)
        residual = A @ Gamma_hat + Gamma_hat @ A.T + Q
        return (residual @ Gamma_hat.T + residual.T @ Gamma_hat).flatten()

    A0 = lyapunov_inversion_symmetric(Gamma_hat, Q)
    result = minimize(objective, A0.flatten(), jac=gradient,
                      method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-15})
    return result.x.reshape(n, n)


def estimate_A_sign_constrained(Gamma_hat, Q, J_prior_signs, lambda_reg=0.1):
    """
    Estimate A with sign-constraint regularization from compiled J prior.
    Adds a penalty for off-diagonal entries that disagree with the prior sign structure.
    Reference: Remark D.7 and Tests 3-4 of the ontology manuscript.
    """
    n = Gamma_hat.shape[0]

    def objective(a_flat):
        A = a_flat.reshape(n, n)
        residual = A @ Gamma_hat + Gamma_hat @ A.T + Q
        loss = 0.5 * np.sum(residual**2)
        for i in range(n):
            for j in range(n):
                if i != j and J_prior_signs[i, j] != 0:
                    if np.sign(A[i, j]) != J_prior_signs[i, j]:
                        loss += lambda_reg * A[i, j] ** 2
        return loss

    A0 = lyapunov_inversion_symmetric(Gamma_hat, Q)
    result = minimize(objective, A0.flatten(), method='L-BFGS-B',
                      options={'maxiter': 5000, 'ftol': 1e-15})
    return result.x.reshape(n, n)


def partial_correlations(Y):
    """
    Compute partial correlation matrix from observed data matrix Y (N x n).
    pcor(i,j) = -P_ij / sqrt(P_ii * P_jj) where P = Gamma^{-1}.
    """
    Gamma = np.cov(Y.T)
    P = inv(Gamma)
    n = P.shape[0]
    pcor = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                pcor[i, j] = -P[i, j] / np.sqrt(P[i, i] * P[j, j])
    return pcor


def sign_concordance(J_est, J_true):
    """
    Compute off-diagonal sign concordance between two coupling matrices.
    Returns: (n_agree, n_total, concordance_fraction)
    """
    n = J_est.shape[0]
    n_agree = 0
    n_total = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if np.sign(J_est[i, j]) == np.sign(J_true[i, j]):
                    n_agree += 1
                n_total += 1
    return n_agree, n_total, n_agree / n_total if n_total > 0 else 0.0


def estimate_tau_from_recovery(trajectory, dt, fit_window=None):
    """
    Estimate recovery time constant by fitting exponential decay.
    Fits log(x - x_ss) = log(x_0) - t/tau via linear regression.

    Args:
        trajectory: 1D array of axis values during recovery episode
        dt: time step between samples (days)
        fit_window: max time (days) to include in fit (optional)
    Returns:
        Estimated tau (days), or np.nan if fit fails
    """
    t = np.arange(len(trajectory)) * dt
    if fit_window:
        mask = t <= fit_window
        t = t[mask]
        trajectory = trajectory[mask]
    x_ss = np.mean(trajectory[-max(1, int(len(trajectory) * 0.2)):])
    y = trajectory - x_ss
    pos = y > 0.05
    if np.sum(pos) < 5:
        return np.nan
    t_fit = t[pos]
    log_y = np.log(y[pos])
    A_mat = np.column_stack([np.ones_like(t_fit), t_fit])
    try:
        coeffs = np.linalg.lstsq(A_mat, log_y, rcond=None)[0]
        if coeffs[1] < -0.001:
            return -1.0 / coeffs[1]
    except Exception:
        pass
    return np.nan


def generate_stratum(age_mid, N, q, meas_noise_sd=0.3):
    """
    Generate N cross-sectional samples from the stationary distribution
    at a given age, with measurement noise.

    Args:
        age_mid: age at which to generate data
        N: number of individuals
        q: array of per-axis noise variances (diagonal of Q)
        meas_noise_sd: standard deviation of measurement noise
    Returns:
        (X_true, Y_obs, Gamma_true)
    """
    n_axes = len(q)
    Q = np.diag(q)
    tau, J = get_params(age_mid)
    A = build_A(tau, J)
    Gamma = stationary_covariance(A, Q)
    X_true = np.random.multivariate_normal(np.zeros(n_axes), Gamma, size=N)
    Y_obs = X_true + meas_noise_sd * np.random.randn(N, n_axes)
    return X_true, Y_obs, Gamma


def stability_weighted_score(delta_x, A_hat):
    """
    Compute the stability-weighted dysregulation score (SWDS) for an individual.

    s_p = Σ_k (v_k^T Δx_p)² / |Re(λ_k)|

    Penalizes deviation along slow-recovering (near-unstable) modes more heavily.
    An individual with high load on the dominant eigenvalue direction scores
    higher (worse) than one with the same total dysregulation on fast-recovering axes.

    Args:
        delta_x: individual's state vector Δx (n-dimensional array)
        A_hat: estimated dynamics matrix (n×n)
    Returns:
        SWDS score (scalar, higher = more vulnerable)

    Reference: Eq. (SWDS) in Tests 5-6, HDR Ontology Manuscript R3.
    """
    eigvals, eigvecs = np.linalg.eig(A_hat)
    eigvecs_real = np.real(eigvecs)
    eigvals_real = np.real(eigvals)

    score = 0.0
    for k in range(len(eigvals)):
        projection = np.dot(eigvecs_real[:, k], delta_x)
        weight = 1.0 / max(abs(eigvals_real[k]), 1e-10)
        score += projection**2 * weight
    return score
