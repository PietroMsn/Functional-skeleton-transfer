"""
Localized joint regressor on the source shape.

We look for a matrix ``R`` of shape ``(n_joints, n_vertices)`` such that the
joints are a linear combination of the vertices, ``J ~= R @ V``, with two
priors that make ``R`` transferable to other shapes:

- **locality**: joint ``j`` connects bone ``j`` to its parent bone, so it may
  only depend on the vertices that are skinned to *both* bones (the
  "locality mask" ``F``, see :func:`locality_mask`);
- **partition of unity**: every row of ``R`` sums to one, so the regressed
  joint is an (affine) average of vertices and moves rigidly with them.

The energy minimized by :func:`fit_spatial_regressor` is

    E(R) = w_fit * ||R V - J||^2
         + w_loc * ||R * (1 - F)||^2          (coefficients outside the mask)
         + w_pou * ||R 1 - 1||^2

(In the paper this is Eq. 13; its locality and sparsity terms coincide for a
binary mask and are merged here into ``w_loc``.)

Closed-form solution
--------------------
``E`` is quadratic and the rows of ``R`` are independent. For one joint with
row ``r`` write the two data terms as ``||A r - b||^2`` with

    A = [sqrt(w_fit) V^T ; sqrt(w_pou) 1^T]      (4, n_vertices)
    b = [sqrt(w_fit) J_j ; sqrt(w_pou)]          (4,)

and split ``r`` into the coefficients inside (``S``) and outside (``O``) the
mask. For a fixed ``r_S`` the optimal ``r_O`` is a ridge regression,

    r_O = A_O^T (A_O A_O^T + w_loc I)^-1 (b - A_S r_S),

and substituting it back leaves ``(A_S r_S - b)^T K (A_S r_S - b)`` with the
4x4 matrix ``K = w_loc (A_O A_O^T + w_loc I)^-1``. Its minimizer is not unique
(4 equations, ``|S|`` unknowns): we take the minimum-norm one, which spreads
the weight evenly over the vertices of the mask. The original implementation
approximated the minimizer with 1000 Adam iterations started from ``R = 0``.
"""

import numpy as np

from .config import RegressorConfig


def locality_mask(skinning_weights: np.ndarray, parents: np.ndarray, threshold: float) -> np.ndarray:
    """Binary mask of the vertices each joint is allowed to depend on.

    ``mask[j, v] = 1`` iff vertex ``v`` has a skinning weight above
    ``threshold`` both for bone ``j`` and for its parent ``parents[j]``.

    Returns
    -------
    (n_joints, n_vertices) float array of 0/1 values.
    """
    influenced = skinning_weights > threshold  # (n_vertices, n_joints)
    return (influenced & influenced[:, parents]).T.astype(float)


def regressor_energy(
    regressor: np.ndarray, vertices: np.ndarray, joints: np.ndarray, mask: np.ndarray, cfg: RegressorConfig
) -> float:
    """Value of the energy ``E(R)`` minimized by :func:`fit_spatial_regressor`."""
    fit = np.sum((regressor @ vertices - joints) ** 2)
    locality = np.sum((regressor * (1.0 - mask)) ** 2)
    partition_of_unity = np.sum((regressor.sum(axis=1) - 1.0) ** 2)
    return float(
        cfg.weight_fit * fit + cfg.weight_locality * locality + cfg.weight_partition_of_unity * partition_of_unity
    )


def fit_spatial_regressor(
    vertices: np.ndarray,
    joints: np.ndarray,
    mask: np.ndarray,
    cfg: RegressorConfig,
) -> np.ndarray:
    """Exact (minimum-norm) minimizer of the regressor energy.

    Parameters
    ----------
    vertices : (n_vertices, 3) source vertices
    joints : (n_joints, 3) source joints
    mask : (n_joints, n_vertices) locality mask, see :func:`locality_mask`
    cfg : energy weights

    Returns
    -------
    (n_joints, n_vertices) regressor ``R``.
    """
    n_joints, n_vertices = mask.shape
    w_loc = cfg.weight_locality
    A = np.vstack((np.sqrt(cfg.weight_fit) * vertices.T, np.sqrt(cfg.weight_partition_of_unity) * np.ones(n_vertices)))
    # A A^T over all vertices; the part outside the mask is obtained by
    # subtracting the (usually much smaller) part inside it.
    gram_all = A @ A.T

    R = np.zeros((n_joints, n_vertices))
    for j in range(n_joints):
        inside = mask[j] > 0
        A_S = A[:, inside]
        b = np.append(np.sqrt(cfg.weight_fit) * joints[j], np.sqrt(cfg.weight_partition_of_unity))

        gram_out_reg = gram_all - A_S @ A_S.T + w_loc * np.eye(len(b))
        K = w_loc * np.linalg.inv(gram_out_reg)
        L = np.linalg.cholesky(K)  # K = L L^T

        r_S = np.linalg.lstsq(L.T @ A_S, L.T @ b, rcond=None)[0] if inside.any() else np.zeros(0)
        r_O = A[:, ~inside].T @ np.linalg.solve(gram_out_reg, b - A_S @ r_S)

        R[j, inside] = r_S
        R[j, ~inside] = r_O
    return R
