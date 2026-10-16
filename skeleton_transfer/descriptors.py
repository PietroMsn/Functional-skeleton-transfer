"""
Spectral point descriptors used to estimate the functional map.

- :func:`wave_kernel_signature` (WKS, Aubry et al. 2011): an intrinsic,
  multi-scale descriptor of each point, invariant to isometries.
- :func:`wave_kernel_map` (WKM): the wave kernel "diffused" from a landmark,
  i.e. a WKS-like descriptor that encodes the distance from a given point.
  Computing it for a few corresponding landmarks breaks the intrinsic
  symmetries (e.g. left/right) that the WKS alone cannot resolve.

Both descriptors work in log-eigenvalue ("energy") space, with a bank of
Gaussian band-pass filters centered at ``n_scales`` energies.
"""

import numpy as np


def _gaussian_filters(log_evals: np.ndarray, energies: np.ndarray, sigma: float) -> np.ndarray:
    """Filter bank ``exp(-(e - log(lambda))^2 / (2 sigma^2))``.

    Returns a ``(n_eigs, n_scales)`` matrix with one filter per column.
    """
    return np.exp(-((energies[None, :] - log_evals[:, None]) ** 2) / (2 * sigma**2))


def wave_kernel_signature(
    evals: np.ndarray,
    evecs: np.ndarray,
    n_scales: int = 200,
    variance: float = 7.0,
) -> np.ndarray:
    """Wave Kernel Signature of every vertex.

    Parameters
    ----------
    evals : (k,) Laplacian eigenvalues, already clamped to be positive
    evecs : (n_vertices, k) eigenfunctions
    n_scales : number of energy scales
    variance : filter width, in multiples of the step between energies

    Returns
    -------
    (n_vertices, n_scales) array; each column is normalized by the total
    energy of its filter.
    """
    log_e = np.log(evals)
    # Energies are spaced linearly in log-space; the first eigenvalue
    # (~0, constant eigenfunction) is skipped.
    energies = np.linspace(log_e[1], np.max(log_e) / 1.02, n_scales)
    sigma = (energies[1] - energies[0]) * variance

    filters = _gaussian_filters(log_e, energies, sigma)  # (k, n_scales)
    wks = (evecs**2) @ filters  # (n_vertices, n_scales)
    return wks / filters.sum(axis=0, keepdims=True)


def wave_kernel_map(
    evals: np.ndarray,
    evecs: np.ndarray,
    landmarks: np.ndarray,
    n_scales: int = 200,
    variance: float = 7.0,
) -> np.ndarray:
    """Wave Kernel Map from each landmark to every vertex.

    For a landmark ``l`` and scale ``e`` the descriptor is

        WKM_l,e(x) = sum_i  f_e(lambda_i) phi_i(l) phi_i(x)

    i.e. the band-pass filtered "delta function" centered at ``l``.

    Parameters
    ----------
    evals : (k,) Laplacian eigenvalues, already clamped to be positive
    evecs : (n_vertices, k) eigenfunctions
    landmarks : (n_landmarks,) vertex indices
    n_scales : number of energy scales per landmark
    variance : filter width, relative to the energy range

    Returns
    -------
    (n_vertices, n_landmarks * n_scales) array; columns are grouped by
    landmark (all scales of landmark 0, then landmark 1, ...).
    """
    log_e = np.log(np.abs(evals))
    e_min, e_max = log_e[1], log_e[-1]
    sigma = variance * (e_max - e_min) / n_scales
    # Leave a margin of 2 sigma at both ends of the energy range.
    energies = np.linspace(e_min + 2 * sigma, e_max - 2 * sigma, n_scales)

    filters = _gaussian_filters(log_e, energies, sigma)  # (k, n_scales)

    blocks = []
    for landmark in np.asarray(landmarks, dtype=int):
        # phi_i(l) for every eigenfunction, then filter and go back to space.
        landmark_coeffs = evecs[landmark, :][:, None]  # (k, 1)
        blocks.append(evecs @ (filters * landmark_coeffs))  # (n_vertices, n_scales)
    return np.hstack(blocks)
