"""Core dynamics: A matrix construction, simulation, and spectral analysis."""

import numpy as np
from scipy import linalg


def build_A(tau: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Build dynamics matrix A = -D + J where D = diag(1/tau_i).

    Args:
        tau: array of shape (n,) — recovery time constants for each axis
        J: array of shape (n, n) — coupling matrix (diagonal should be 0)

    Returns:
        A: array of shape (n, n) — system dynamics matrix
    """
    D = np.diag(1.0 / tau)
    return -D + J


def spectral_abscissa(A: np.ndarray) -> float:
    """Return α(A) = max_k Re(λ_k(A)). Negative = stable."""
    eigenvalues = linalg.eig(A, right=False)
    return float(np.max(np.real(eigenvalues)))


def spectral_gap(A: np.ndarray) -> float:
    """Return |λ_1| - |λ_2|, the coherence measure κ̂."""
    eigenvalues = linalg.eig(A, right=False)
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]
    return float(magnitudes[0] - magnitudes[1])


def recovery_timescale(A: np.ndarray) -> float:
    """Return 1/|α(A)| — the dominant recovery timescale in time units."""
    alpha = spectral_abscissa(A)
    if alpha == 0:
        return float('inf')
    return 1.0 / abs(alpha)


def spectral_radius_discrete(A: np.ndarray, dt: float) -> float:
    """Compute ρ(e^{A·Δt}) — the discrete-time spectral radius."""
    Phi = linalg.expm(A * dt)
    eigenvalues = linalg.eig(Phi, right=False)
    return float(np.max(np.abs(eigenvalues)))


def simulate(A: np.ndarray, x0: np.ndarray, dt: float, T: float,
             noise_std: float = 0.0, perturbations: list = None) -> tuple:
    """Euler-Maruyama simulation of dx = A·x·dt + σ·dW.

    Args:
        A: dynamics matrix
        x0: initial state
        dt: time step (use 0.01 for smooth curves)
        T: total simulation time
        noise_std: standard deviation of Wiener process increments
        perturbations: list of (time, axis, magnitude) tuples for impulse perturbations

    Returns:
        t: time array
        x: state trajectory array of shape (n_steps, n_axes)
    """
    n_steps = int(T / dt)
    n = len(x0)
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros((n_steps + 1, n))
    x[0] = x0.copy()

    # Pre-process perturbations into a dict of step_index -> list of (axis, magnitude)
    pert_dict = {}
    if perturbations:
        for p_time, p_axis, p_mag in perturbations:
            step_idx = int(round(p_time / dt))
            step_idx = max(0, min(step_idx, n_steps))
            if step_idx not in pert_dict:
                pert_dict[step_idx] = []
            pert_dict[step_idx].append((p_axis, p_mag))

    rng = np.random.default_rng(42)

    for i in range(n_steps):
        # Apply perturbations at this step
        if i in pert_dict:
            for axis, mag in pert_dict[i]:
                x[i, axis] += mag

        # Euler-Maruyama step
        dx = A @ x[i] * dt
        if noise_std > 0:
            dx += noise_std * np.sqrt(dt) * rng.standard_normal(n)
        x[i + 1] = x[i] + dx

    return t, x
