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


def fit_spatial_regressor(
    vertices: np.ndarray,
    joints: np.ndarray,
    mask: np.ndarray,
    cfg: RegressorConfig,
) -> np.ndarray:
    """Fit the localized regressor with Adam, starting from ``R = 0``.

    Parameters
    ----------
    vertices : (n_vertices, 3) source vertices
    joints : (n_joints, 3) source joints
    mask : (n_joints, n_vertices) locality mask, see :func:`locality_mask`
    cfg : weights and optimizer settings

    Returns
    -------
    (n_joints, n_vertices) regressor ``R``.
    """
    import tensorflow as tf  # imported lazily: only needed for the optimization

    n_joints, n_vertices = mask.shape
    V = tf.constant(vertices, dtype=tf.float64)
    J = tf.constant(joints, dtype=tf.float64)
    outside_mask = tf.constant(1.0 - mask, dtype=tf.float64)
    ones_v = tf.ones((n_vertices, 1), dtype=tf.float64)
    ones_j = tf.ones((n_joints, 1), dtype=tf.float64)

    R = tf.Variable(tf.zeros((n_joints, n_vertices), dtype=tf.float64))
    optimizer = tf.keras.optimizers.Adam(cfg.learning_rate)

    def energy():
        fit = tf.reduce_sum((R @ V - J) ** 2)
        locality = tf.reduce_sum((R * outside_mask) ** 2)
        partition_of_unity = tf.reduce_sum((R @ ones_v - ones_j) ** 2)
        return (
            cfg.weight_fit * fit
            + cfg.weight_locality * locality
            + cfg.weight_partition_of_unity * partition_of_unity
        )

    for it in range(cfg.n_iterations):
        with tf.GradientTape() as tape:
            loss = energy()
        grads = tape.gradient(loss, [R])
        optimizer.apply_gradients(zip(grads, [R]))
        if cfg.log_every and it % cfg.log_every == 0:
            print(f"Regressor iteration {it}: loss = {loss.numpy():.6g}")

    return R.numpy()
