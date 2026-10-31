"""
Functional map estimation between the target and the source shape.

Conventions
-----------
Throughout this module the map goes **from the target to the source**:

- ``C`` is a ``(k, k)`` matrix that converts the spectral coefficients of a
  function defined on the target into the coefficients of the corresponding
  function on the source:  ``f_source_hat ~= C @ f_target_hat``;
- ``p2p`` is the induced point-to-point map, an array with one entry per
  **source** vertex holding the index of the matching **target** vertex.
  It is the pull-back of ``C``: ``evecs_src @ C ~= evecs_tgt[p2p]``.

Pipeline (:func:`estimate_functional_map`)
------------------------------------------
1. WKS + landmark WKM descriptors on both shapes (:mod:`.descriptors`);
2. a small ``n_eigs_init`` map is computed in closed form so that it
   preserves the descriptors and commutes with the Laplacians;
3. ICP refinement: alternate nearest neighbours in the spectral embedding and
   projection of the map onto the orthogonal matrices;
4. ZoomOut (Melzi et al. 2019): iteratively grow the map to ``n_eigs_final``
   by alternating point-to-point and functional representations.

With ``cfg.cmh`` every map of size ``k`` in step 4 (and in the later
refinements) uses the first ``k`` eigenfunctions followed by the 3 CMH
functions of each shape, so ``C`` is ``(k + 3, k + 3)``.
"""

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from .config import FunctionalMapConfig
from .descriptors import wave_kernel_map, wave_kernel_signature
from .spectral import LaplaceBasis

#: Eigenvalues below this value are clamped before taking their logarithm.
MIN_EIGENVALUE = 1e-6


@dataclass
class FunctionalMapResult:
    """Output of :func:`estimate_functional_map`.

    Attributes
    ----------
    C : (k, k) functional map, target coefficients -> source coefficients
    p2p : (n_source_vertices,) index of the target vertex matched to each
        source vertex
    """

    C: np.ndarray
    p2p: np.ndarray


#: Queries per block of the GPU search (a block of distances is
#: ``GPU_BLOCK x n_points`` doubles).
GPU_BLOCK = 2048


def _cuda_device():
    """The CUDA device, or ``None`` without PyTorch, without a GPU, or with
    ``SKELETON_TRANSFER_DEVICE=cpu`` in the environment."""
    if os.environ.get("SKELETON_TRANSFER_DEVICE", "auto").lower() == "cpu":
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def nearest_neighbors(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index of the nearest row of ``points`` for each row of ``queries``.

    Exact search: brute force in double precision on the GPU when PyTorch
    finds one (see :func:`_cuda_device`), otherwise a k-d tree. In the
    100-200 dimensions of the spectral embeddings the k-d tree degenerates
    to an exhaustive search with a large overhead, so the GPU is much faster.
    (The original code used FLANN, which is approximate.)
    """
    device = _cuda_device()
    if device is None:
        _, idx = cKDTree(points).query(queries, k=1)
        return idx
    import torch

    P = torch.as_tensor(np.ascontiguousarray(points, dtype=np.float64), device=device)
    Q = torch.as_tensor(np.ascontiguousarray(queries, dtype=np.float64), device=device)
    norms = (P * P).sum(dim=1)
    idx = torch.cat([torch.argmin(norms - 2 * Q[s : s + GPU_BLOCK] @ P.T, dim=1)
                     for s in range(0, len(Q), GPU_BLOCK)])  # fmt: skip
    return idx.cpu().numpy()


def p2p_to_functional_map(
    p2p: np.ndarray, evecs_src: np.ndarray, evecs_tgt: np.ndarray, k: int
) -> np.ndarray:
    """Least-squares functional map ``C`` such that ``evecs_src C ~= evecs_tgt[p2p]``."""
    return np.linalg.pinv(evecs_src[:, :k]) @ evecs_tgt[p2p, :k]


#: Basis functions of a shape: either the eigenfunctions (sliced to the first
#: ``k`` columns) or a function returning the basis of size ``k``, e.g.
#: ``lambda k: basis.functions(k, cmh=True)``.
Basis = Union[np.ndarray, Callable[[int], np.ndarray]]


def basis_functions(basis: Basis, k: int) -> np.ndarray:
    """The basis of size ``k`` (see :data:`Basis`)."""
    return basis(k) if callable(basis) else basis[:, :k]


#: Optional pair of per-vertex features ``(F_src, F_tgt)`` appended to the
#: spectral embeddings in the nearest-neighbour searches.
Features = Optional[Tuple[np.ndarray, np.ndarray]]
#: Optional pair of corresponding vertex indices ``(landmarks_src, landmarks_tgt)``.
Landmarks = Optional[Tuple[np.ndarray, np.ndarray]]


def functional_map_to_p2p(
    C: np.ndarray,
    evecs_src: np.ndarray,
    evecs_tgt: np.ndarray,
    features: Features = None,
    landmarks: Landmarks = None,
) -> np.ndarray:
    """Point-to-point map (source vertex -> target vertex) induced by ``C``.

    Each target vertex is embedded with ``C`` into the source spectral space
    and matched to the nearest source vertex embedding.

    Parameters
    ----------
    features : extra per-vertex features ``(F_src, F_tgt)``, already scaled,
        concatenated to the spectral embeddings before the search
    landmarks : ``(landmarks_src, landmarks_tgt)``; the source landmarks are
        forced to map onto the target ones
    """
    k_src, k_tgt = C.shape
    points = evecs_tgt[:, :k_tgt] @ C.T
    queries = evecs_src[:, :k_src]
    if features is not None:
        points = np.hstack((points, features[1]))
        queries = np.hstack((queries, features[0]))
    p2p = nearest_neighbors(points, queries)
    if landmarks is not None:
        p2p[landmarks[0]] = landmarks[1]
    return p2p


# ----------------------------------------------------------------------------
# Step 1: descriptors
# ----------------------------------------------------------------------------


def compute_descriptors(basis: LaplaceBasis, landmarks: np.ndarray, cfg: FunctionalMapConfig) -> np.ndarray:
    """WKS + landmark WKM descriptors, subsampled along the scales.

    Returns an ``(n_vertices, n_descriptors)`` array.
    """
    evals = np.maximum(basis.evals, MIN_EIGENVALUE)
    wks = wave_kernel_signature(evals, basis.evecs, cfg.n_descriptor_scales, cfg.wks_variance)
    wkm = wave_kernel_map(evals, basis.evecs, landmarks, cfg.n_descriptor_scales, cfg.wkm_variance)
    descriptors = np.hstack((wks, wkm))
    return descriptors[:, :: cfg.descriptor_subsample]


# ----------------------------------------------------------------------------
# Step 2: initial map from descriptors
# ----------------------------------------------------------------------------


def functional_map_energy(
    C: np.ndarray,
    desc_coeffs_tgt: np.ndarray,
    desc_coeffs_src: np.ndarray,
    evals_tgt: np.ndarray,
    evals_src: np.ndarray,
    cfg: FunctionalMapConfig,
) -> float:
    """Value of the energy minimized by :func:`optimize_initial_map`."""
    descriptor_term = np.sum((C @ desc_coeffs_tgt - desc_coeffs_src) ** 2) / 2
    laplacian_term = np.sum((C * (evals_tgt[None, :] - evals_src[:, None])) ** 2)
    return float(cfg.weight_descriptors * descriptor_term + cfg.weight_laplacian * laplacian_term)


def optimize_initial_map(
    desc_coeffs_tgt: np.ndarray,
    desc_coeffs_src: np.ndarray,
    evals_tgt: np.ndarray,
    evals_src: np.ndarray,
    cfg: FunctionalMapConfig,
) -> np.ndarray:
    """Exact minimizer of the classic functional map energy

        E(C) = w_d / 2 * ||C A_tgt - A_src||^2  +  w_l * ||C L_tgt - L_src C||^2

    where ``A`` are the spectral coefficients of the descriptors (one column per
    descriptor) and ``L`` the diagonal matrices of eigenvalues.

    Since the ``L`` are diagonal, ``(C L_tgt - L_src C)_ij = C_ij (l_tgt_j -
    l_src_i)`` and ``E`` splits over the rows of ``C``. Row ``i`` solves

        (w_d A_tgt A_tgt^T + 2 w_l diag((l_tgt - l_src_i)^2)) c_i = w_d A_tgt a_src_i

    (The original implementation approximated it with 1000 Adam iterations.)
    """
    gram = cfg.weight_descriptors * desc_coeffs_tgt @ desc_coeffs_tgt.T
    rhs = cfg.weight_descriptors * desc_coeffs_tgt @ desc_coeffs_src.T  # column i -> row i of C
    C = np.empty((len(evals_src), len(evals_tgt)))
    for i, l_src in enumerate(evals_src):
        system = gram + np.diag(2 * cfg.weight_laplacian * (evals_tgt - l_src) ** 2)
        C[i] = np.linalg.lstsq(system, rhs[:, i], rcond=None)[0]
    return C


# ----------------------------------------------------------------------------
# Step 3 and 4: refinement
# ----------------------------------------------------------------------------


def icp_refine(
    C: np.ndarray, evecs_src: np.ndarray, evecs_tgt: np.ndarray, n_iterations: int, landmarks: Landmarks = None
) -> np.ndarray:
    """Spectral ICP: alternate nearest neighbours and orthogonal projection.

    Returns the point-to-point map (source -> target) obtained at the last
    nearest-neighbour step.
    """
    k_src, k_tgt = C.shape
    p2p = None
    for _ in range(n_iterations):
        p2p = functional_map_to_p2p(C, evecs_src, evecs_tgt, landmarks=landmarks)
        # Best map for this p2p, then closest orthogonal matrix (via SVD).
        W = np.linalg.lstsq(evecs_src[:, :k_src], evecs_tgt[p2p, :k_tgt], rcond=None)[0]
        U, _, Vt = np.linalg.svd(W)
        C = U @ np.eye(k_src, k_tgt) @ Vt
    return p2p


def zoomout(
    p2p: np.ndarray,
    evecs_src: Basis,
    evecs_tgt: Basis,
    k_init: int,
    k_final: int,
    step: int,
    landmarks: Landmarks = None,
):
    """ZoomOut upsampling of a point-to-point map.

    Starting from ``k_init`` eigenfunctions, at each iteration the map is
    converted to a functional map of size ``k``, then back to a
    point-to-point map, and ``k`` is increased by ``step`` until ``k_final``.

    Returns
    -------
    C : (k_final, k_final) functional map (larger if the bases add functions)
    p2p : point-to-point map used to compute ``C``
    """
    phi_src, phi_tgt = basis_functions(evecs_src, k_init), basis_functions(evecs_tgt, k_init)
    C = p2p_to_functional_map(p2p, phi_src, phi_tgt, phi_src.shape[1])
    for k in range(k_init, k_final, step):
        p2p = functional_map_to_p2p(C, phi_src, phi_tgt, landmarks=landmarks)
        phi_src, phi_tgt = basis_functions(evecs_src, k + step), basis_functions(evecs_tgt, k + step)
        C = p2p_to_functional_map(p2p, phi_src, phi_tgt, phi_src.shape[1])
    return C, p2p


def bijective_zoomout(
    p2p: np.ndarray,
    evecs_src: Basis,
    evecs_tgt: Basis,
    k_init: int,
    k_final: int,
    step: int,
    weight: float = 1.0,
    landmarks: Landmarks = None,
):
    """ZoomOut on both directions coupled by a bijectivity term.

    In the spirit of the discrete optimization of Ren et al. [2021] (as in
    pyCMH): besides the map ``C`` (target -> source coefficients) of the
    source -> target vertex map, it refines a map ``C_rev`` (source -> target
    coefficients) of a target -> source vertex map. Each vertex map is the
    nearest neighbour in embeddings that ask both for ``Phi_src ~ Phi_tgt C^T``
    and for consistency with the other direction (``C_rev C ~ I``), which
    reduces the many-to-one matches of plain ZoomOut. The reverse map starts
    from ``C^T`` (the inverse of an orthogonal map).

    Returns
    -------
    C : (k_final, k_final) functional map (larger if the bases add functions)
    p2p : point-to-point map (source -> target) used to compute ``C``
    """
    phi_src, phi_tgt = basis_functions(evecs_src, k_init), basis_functions(evecs_tgt, k_init)
    C = p2p_to_functional_map(p2p, phi_src, phi_tgt, phi_src.shape[1])
    q2p = nearest_neighbors(phi_src @ C, phi_tgt)  # target -> source
    if landmarks is not None:
        q2p[landmarks[1]] = landmarks[0]
    C_rev = p2p_to_functional_map(q2p, phi_tgt, phi_src, phi_tgt.shape[1])
    for k in range(k_init, k_final, step):
        p2p = nearest_neighbors(np.hstack((phi_tgt @ C.T, weight * phi_tgt)),
                                np.hstack((phi_src, weight * phi_src @ C_rev.T)))  # fmt: skip
        q2p = nearest_neighbors(np.hstack((phi_src @ C_rev.T, weight * phi_src)),
                                np.hstack((phi_tgt, weight * phi_tgt @ C.T)))  # fmt: skip
        if landmarks is not None:
            p2p[landmarks[0]] = landmarks[1]
            q2p[landmarks[1]] = landmarks[0]
        phi_src, phi_tgt = basis_functions(evecs_src, k + step), basis_functions(evecs_tgt, k + step)
        C = p2p_to_functional_map(p2p, phi_src, phi_tgt, phi_src.shape[1])
        C_rev = p2p_to_functional_map(q2p, phi_tgt, phi_src, phi_tgt.shape[1])
    return C, p2p


def refine_with_features(
    p2p: np.ndarray,
    evecs_src: Basis,
    evecs_tgt: Basis,
    k: int,
    n_iterations: int,
    features: Features = None,
    landmarks: Landmarks = None,
):
    """ZoomOut-like iterations at a fixed size ``k``, with extra features.

    Alternates ``C = p2p -> functional map`` and ``p2p = NN([Phi_src, F_src],
    [Phi_tgt C^T, F_tgt])``: the features (e.g. derived from the current
    skeleton) steer the point-to-point map, the functional map keeps it smooth.

    Returns
    -------
    C : (k, k) functional map (larger if the bases add functions)
    p2p : point-to-point map (source -> target)
    """
    phi_src, phi_tgt = basis_functions(evecs_src, k), basis_functions(evecs_tgt, k)
    C = p2p_to_functional_map(p2p, phi_src, phi_tgt, phi_src.shape[1])
    for _ in range(n_iterations):
        p2p = functional_map_to_p2p(C, phi_src, phi_tgt, features, landmarks)
        C = p2p_to_functional_map(p2p, phi_src, phi_tgt, phi_src.shape[1])
    return C, p2p


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------


def cmh_basis(basis: LaplaceBasis, cfg: FunctionalMapConfig) -> Basis:
    """Basis of the ZoomOut and of the later refinements: the eigenfunctions,
    extended with the CMH functions if ``cfg.cmh``."""
    return (lambda k: basis.functions(k, cmh=True)) if cfg.cmh else basis.evecs


def estimate_functional_map(
    basis_src: LaplaceBasis,
    basis_tgt: LaplaceBasis,
    landmarks_src: np.ndarray,
    landmarks_tgt: np.ndarray,
    cfg: FunctionalMapConfig,
    n_eigs_init: Optional[int] = None,
) -> FunctionalMapResult:
    """Estimate the functional map from the target to the source.

    Parameters
    ----------
    basis_src, basis_tgt : Laplace bases of the two shapes
    landmarks_src, landmarks_tgt : corresponding landmark vertex indices
    cfg : functional map parameters
    n_eigs_init : overrides ``cfg.n_eigs_init`` (used to restart from a
        different initial map)
    """
    k = cfg.n_eigs_init if n_eigs_init is None else n_eigs_init
    forced = (landmarks_src, landmarks_tgt) if cfg.enforce_landmarks else None

    # 1) Descriptors. They are normalized on both shapes with the same
    #    per-descriptor norm, measured on the target.
    desc_tgt = compute_descriptors(basis_tgt, landmarks_tgt, cfg)
    desc_src = compute_descriptors(basis_src, landmarks_src, cfg)
    norms = np.sqrt(np.einsum("ij,ij->j", basis_tgt.mass @ desc_tgt, desc_tgt))
    desc_tgt = desc_tgt / norms
    desc_src = desc_src / norms

    # 2) Initial map
    evals_tgt = np.maximum(basis_tgt.evals, MIN_EIGENVALUE)
    evals_src = np.maximum(basis_src.evals, MIN_EIGENVALUE)
    C = optimize_initial_map(
        desc_coeffs_tgt=basis_tgt.project(desc_tgt, k),
        desc_coeffs_src=basis_src.project(desc_src, k),
        evals_tgt=evals_tgt[:k],
        evals_src=evals_src[:k],
        cfg=cfg,
    )

    # 3) ICP refinement -> point-to-point map
    p2p = icp_refine(C, basis_src.evecs, basis_tgt.evecs, cfg.n_icp_iterations, forced)

    # 4) ZoomOut up to the final size
    bases = cmh_basis(basis_src, cfg), cmh_basis(basis_tgt, cfg)
    if cfg.bijective:
        C, p2p = bijective_zoomout(p2p, *bases, k, cfg.n_eigs_final, cfg.zoomout_step, cfg.bijective_weight, forced)
    else:
        C, p2p = zoomout(p2p, *bases, k, cfg.n_eigs_final, cfg.zoomout_step, forced)
    return FunctionalMapResult(C=C, p2p=p2p)
