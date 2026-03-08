"""
Kernel regression estimator for GCI → steady-state relative productivity.

Implements Equation A.1 from Hubbard & Sharma (2016):

    φ(GCI) = Σ K(ωᵢ)·φᵢ / Σ K(ωᵢ)

where K(ωᵢ) = (1/√2π) · exp(-½ · ((GCIᵢ - GCI) / h)²)

The bandwidth h is chosen as the smallest value that ensures
the estimated relationship is monotonically non-decreasing.
"""

import numpy as np


def gaussian_kernel(gci_points: np.ndarray, gci_eval: float, bandwidth: float) -> np.ndarray:
    """Compute Gaussian kernel weights for each comparator country.

    Args:
        gci_points: GCI scores of steady-state comparator countries (n,)
        gci_eval: GCI score at which to evaluate
        bandwidth: Kernel bandwidth h

    Returns:
        Kernel weights (n,)
    """
    omega = (gci_points - gci_eval) / bandwidth
    return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * omega ** 2)


def kernel_estimate(
    gci_points: np.ndarray,
    phi_points: np.ndarray,
    gci_eval: float | np.ndarray,
    bandwidth: float = 0.294,
) -> float | np.ndarray:
    """Estimate steady-state relative productivity for given GCI score(s).

    Implements Equation A.1: Nadaraya-Watson kernel regression.

    Args:
        gci_points: GCI scores of steady-state countries (n,)
        phi_points: Relative productivity of steady-state countries (n,)
        gci_eval: GCI score(s) at which to estimate φ
        bandwidth: Kernel bandwidth h (default 0.294)

    Returns:
        Estimated steady-state relative productivity φ
    """
    gci_points = np.asarray(gci_points, dtype=float)
    phi_points = np.asarray(phi_points, dtype=float)
    scalar = np.isscalar(gci_eval)
    gci_eval = np.atleast_1d(np.asarray(gci_eval, dtype=float))

    results = np.empty(len(gci_eval))
    for j, gci in enumerate(gci_eval):
        weights = gaussian_kernel(gci_points, gci, bandwidth)
        total_weight = weights.sum()
        if total_weight == 0:
            results[j] = 0.0
        else:
            results[j] = (weights * phi_points).sum() / total_weight

    return float(results[0]) if scalar else results


def is_monotonically_nondecreasing(
    gci_points: np.ndarray,
    phi_points: np.ndarray,
    bandwidth: float,
    n_eval: int = 500,
) -> bool:
    """Check if the kernel estimate is monotonically non-decreasing.

    Evaluates the kernel at n_eval equally-spaced GCI values across the
    range of the data and checks that estimates never decrease.
    """
    gci_min, gci_max = gci_points.min(), gci_points.max()
    gci_grid = np.linspace(gci_min, gci_max, n_eval)
    estimates = kernel_estimate(gci_points, phi_points, gci_grid, bandwidth)
    return bool(np.all(np.diff(estimates) >= -1e-10))


def find_optimal_bandwidth(
    gci_points: np.ndarray,
    phi_points: np.ndarray,
    h_min: float = 0.05,
    h_max: float = 2.0,
    tol: float = 0.001,
) -> float:
    """Find the smallest bandwidth ensuring monotonically non-decreasing estimates.

    Uses binary search between h_min and h_max.

    Args:
        gci_points: GCI scores of steady-state countries
        phi_points: Relative productivity of steady-state countries
        h_min: Lower bound for bandwidth search
        h_max: Upper bound for bandwidth search
        tol: Tolerance for binary search convergence

    Returns:
        Optimal bandwidth h
    """
    gci_points = np.asarray(gci_points, dtype=float)
    phi_points = np.asarray(phi_points, dtype=float)

    while h_max - h_min > tol:
        h_mid = (h_min + h_max) / 2
        if is_monotonically_nondecreasing(gci_points, phi_points, h_mid):
            h_max = h_mid
        else:
            h_min = h_mid

    return h_max


def kernel_curve(
    gci_points: np.ndarray,
    phi_points: np.ndarray,
    bandwidth: float = 0.294,
    n_eval: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the full kernel regression curve for plotting.

    Returns:
        Tuple of (gci_grid, phi_estimates)
    """
    gci_points = np.asarray(gci_points, dtype=float)
    gci_min, gci_max = gci_points.min() - 0.2, gci_points.max() + 0.2
    gci_grid = np.linspace(gci_min, gci_max, n_eval)
    phi_estimates = kernel_estimate(gci_points, phi_points, gci_grid, bandwidth)
    return gci_grid, phi_estimates
