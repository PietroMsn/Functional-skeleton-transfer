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
2. a small ``n_eigs_init`` map is optimized with Adam so that it preserves
   the descriptors and commutes with the Laplacians;
3. ICP refinement: alternate nearest neighbours in the spectral embedding and
   projection of the map onto the orthogonal matrices;
4. ZoomOut (Melzi et al. 2019): iteratively grow the map to ``n_eigs_final``
   by alternating point-to-point and functional representations.
"""

from dataclasses import dataclass

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


def nearest_neighbors(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Index of the nearest row of ``points`` for each row of ``queries``.

    Exact k-d tree search. (The original code used FLANN, which is
    approximate; any NN backend can be plugged in here.)
    """
    _, idx = cKDTree(points).query(queries, k=1)
    return idx


def p2p_to_functional_map(
    p2p: np.ndarray, evecs_src: np.ndarray, evecs_tgt: np.ndarray, k: int
) -> np.ndarray:
    """Least-squares functional map ``C`` such that ``evecs_src C ~= evecs_tgt[p2p]``."""
    return np.linalg.pinv(evecs_src[:, :k]) @ evecs_tgt[p2p, :k]


def functional_map_to_p2p(
    C: np.ndarray, evecs_src: np.ndarray, evecs_tgt: np.ndarray
) -> np.ndarray:
    """Point-to-point map (source vertex -> target vertex) induced by ``C``.

    Each target vertex is embedded with ``C`` into the source spectral space
    and matched to the nearest source vertex embedding.
    """
    k_src, k_tgt = C.shape
    embedded_tgt = evecs_tgt[:, :k_tgt] @ C.T
    return nearest_neighbors(embedded_tgt, evecs_src[:, :k_src])


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


def optimize_initial_map(
    desc_coeffs_tgt: np.ndarray,
    desc_coeffs_src: np.ndarray,
    evals_tgt: np.ndarray,
    evals_src: np.ndarray,
    C_init: np.ndarray,
    cfg: FunctionalMapConfig,
) -> np.ndarray:
    """Minimize the classic functional map energy with Adam.

        E(C) = w_d / 2 * ||C A_tgt - A_src||^2  +  w_l * ||C L_tgt - L_src C||^2

    where ``A`` are the spectral coefficients of the descriptors (one column per
    descriptor) and ``L`` the diagonal matrices of eigenvalues.
    """
    import tensorflow as tf  # imported lazily: only needed for the optimization

    A_tgt = tf.constant(desc_coeffs_tgt, dtype=tf.float32)
    A_src = tf.constant(desc_coeffs_src, dtype=tf.float32)
    L_tgt = tf.linalg.diag(tf.constant(evals_tgt, dtype=tf.float32))
    L_src = tf.linalg.diag(tf.constant(evals_src, dtype=tf.float32))

    C = tf.Variable(C_init, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(cfg.learning_rate)

    for _ in range(cfg.n_iterations):
        with tf.GradientTape() as tape:
            descriptor_term = tf.reduce_sum((tf.matmul(C, A_tgt) - A_src) ** 2) / 2
            laplacian_term = tf.reduce_sum((tf.matmul(C, L_tgt) - tf.matmul(L_src, C)) ** 2)
            loss = cfg.weight_descriptors * descriptor_term + cfg.weight_laplacian * laplacian_term
        grads = tape.gradient(loss, [C])
        optimizer.apply_gradients(zip(grads, [C]))

    return C.numpy()


# ----------------------------------------------------------------------------
# Step 3 and 4: refinement
# ----------------------------------------------------------------------------


def icp_refine(C: np.ndarray, evecs_src: np.ndarray, evecs_tgt: np.ndarray, n_iterations: int) -> np.ndarray:
    """Spectral ICP: alternate nearest neighbours and orthogonal projection.

    Returns the point-to-point map (source -> target) obtained at the last
    nearest-neighbour step.
    """
    k_src, k_tgt = C.shape
    p2p = None
    for _ in range(n_iterations):
        p2p = functional_map_to_p2p(C, evecs_src, evecs_tgt)
        # Best map for this p2p, then closest orthogonal matrix (via SVD).
        W = np.linalg.lstsq(evecs_src[:, :k_src], evecs_tgt[p2p, :k_tgt], rcond=None)[0]
        U, _, Vt = np.linalg.svd(W)
        C = U @ np.eye(k_src, k_tgt) @ Vt
    return p2p


def zoomout(
    p2p: np.ndarray, evecs_src: np.ndarray, evecs_tgt: np.ndarray, k_init: int, k_final: int, step: int
):
    """ZoomOut upsampling of a point-to-point map.

    Starting from ``k_init`` eigenfunctions, at each iteration the map is
    converted to a functional map of size ``k``, then back to a
    point-to-point map, and ``k`` is increased by ``step`` until ``k_final``.

    Returns
    -------
    C : (k_final, k_final) functional map
    p2p : point-to-point map used to compute ``C``
    """
    C = p2p_to_functional_map(p2p, evecs_src, evecs_tgt, k_init)
    for k in range(k_init, k_final, step):
        p2p = functional_map_to_p2p(C, evecs_src[:, :k], evecs_tgt[:, :k])
        C = p2p_to_functional_map(p2p, evecs_src, evecs_tgt, k + step)
    return C, p2p


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------


def estimate_functional_map(
    basis_src: LaplaceBasis,
    basis_tgt: LaplaceBasis,
    landmarks_src: np.ndarray,
    landmarks_tgt: np.ndarray,
    cfg: FunctionalMapConfig,
) -> FunctionalMapResult:
    """Estimate the functional map from the target to the source.

    Parameters
    ----------
    basis_src, basis_tgt : Laplace bases of the two shapes
    landmarks_src, landmarks_tgt : corresponding landmark vertex indices
    cfg : functional map parameters
    """
    k = cfg.n_eigs_init

    # 1) Descriptors. They are normalized on both shapes with the same
    #    per-descriptor norm, measured on the target.
    desc_tgt = compute_descriptors(basis_tgt, landmarks_tgt, cfg)
    desc_src = compute_descriptors(basis_src, landmarks_src, cfg)
    norms = np.sqrt(np.einsum("ij,ij->j", basis_tgt.mass @ desc_tgt, desc_tgt))
    desc_tgt = desc_tgt / norms
    desc_src = desc_src / norms

    # 2) Initial map. The constant function is mapped exactly: its
    #    coefficient only depends on the ratio of the areas (and on the
    #    arbitrary sign of the first eigenfunctions).
    C_init = np.zeros((k, k))
    C_init[0, 0] = np.sign(basis_tgt.evecs[0, 0] * basis_src.evecs[0, 0]) * np.sqrt(
        basis_src.area / basis_tgt.area
    )
    evals_tgt = np.maximum(basis_tgt.evals, MIN_EIGENVALUE)
    evals_src = np.maximum(basis_src.evals, MIN_EIGENVALUE)
    C = optimize_initial_map(
        desc_coeffs_tgt=basis_tgt.project(desc_tgt, k),
        desc_coeffs_src=basis_src.project(desc_src, k),
        evals_tgt=evals_tgt[:k],
        evals_src=evals_src[:k],
        C_init=C_init,
        cfg=cfg,
    )

    # 3) ICP refinement -> point-to-point map
    print("ICP refinement...")
    p2p = icp_refine(C, basis_src.evecs, basis_tgt.evecs, cfg.n_icp_iterations)

    # 4) ZoomOut up to the final size
    print("ZoomOut...")
    C, p2p = zoomout(
        p2p, basis_src.evecs, basis_tgt.evecs, k, cfg.n_eigs_final, cfg.zoomout_step
    )
    return FunctionalMapResult(C=C, p2p=p2p)
