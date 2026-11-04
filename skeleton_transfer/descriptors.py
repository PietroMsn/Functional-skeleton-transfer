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

Not spectral, but also built from the landmarks:

- :func:`landmark_geodesics`: graph geodesic distances from the landmarks;
- :func:`geodesic_descriptors`: those distances and Gaussians of them;
- :func:`voronoi_cells`: indicators of the geodesic Voronoi cell of every
  landmark, a coarse segmentation of the shape into body parts.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


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


def landmark_geodesics(
    vertices: np.ndarray, faces: Optional[np.ndarray], landmarks: np.ndarray, n_neighbors: int = 8
) -> np.ndarray:
    """Shortest-path distances from the landmarks along the mesh edges, or
    along a k-NN graph for a point cloud (``faces is None``; the neighbours
    are doubled until the graph is connected, up to 64).

    Points that no landmark reaches (another connected component) get the
    largest finite distance.

    Returns an ``(n_vertices, n_landmarks)`` array.
    """
    n = len(vertices)

    def edge_graph(rows, cols):
        lengths = np.linalg.norm(vertices[rows] - vertices[cols], axis=1)
        return sp.csr_matrix((lengths, (rows, cols)), shape=(n, n))

    if faces is not None:
        graph = edge_graph(faces.ravel(), faces[:, [1, 2, 0]].ravel())
    else:
        tree = cKDTree(vertices)
        while True:
            idx = tree.query(vertices, n_neighbors + 1)[1][:, 1:]
            graph = edge_graph(np.repeat(np.arange(n), n_neighbors), idx.ravel())
            if n_neighbors >= 64 or connected_components(graph, directed=False)[0] == 1:
                break
            n_neighbors *= 2
    distances = dijkstra(graph, directed=False, indices=landmarks).T
    finite = np.isfinite(distances)
    distances[~finite] = distances[finite].max()
    return distances


def geodesic_descriptors(distances: np.ndarray, scales=(0.1, 0.25, 0.5)) -> np.ndarray:
    """The distances from the landmarks (relative to the square root of the
    area, so that they do not depend on the size) followed by Gaussians of
    them at the given ``scales``, which weigh the points near each landmark
    more."""
    return np.hstack([distances] + [np.exp(-((distances / s) ** 2)) for s in scales])


def voronoi_cells(distances: np.ndarray) -> np.ndarray:
    """``(n_vertices, n_landmarks)`` indicators of the nearest landmark."""
    return (distances.argmin(axis=1)[:, None] == np.arange(distances.shape[1])[None]).astype(float)
