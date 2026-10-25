"""The closed-form solvers return the true minimizers of their energies."""

import numpy as np

from skeleton_transfer.config import FunctionalMapConfig, RegressorConfig
from skeleton_transfer.functional_map import functional_map_energy, optimize_initial_map
from skeleton_transfer.regressor import fit_spatial_regressor, locality_mask, regressor_energy


def _regressor_problem(rng, n_vertices=60, n_joints=5):
    vertices = rng.normal(size=(n_vertices, 3))
    parents = np.array([0, 0, 1, 1, 3])[:n_joints]
    # Smooth-ish random skinning weights so that masks overlap between bones.
    weights = rng.dirichlet(np.full(n_joints, 0.5), size=n_vertices)
    mask = locality_mask(weights, parents, threshold=0.1)
    joints = np.stack([vertices[m > 0].mean(axis=0) if m.any() else np.zeros(3) for m in mask])
    return vertices, joints, mask


def test_regressor_reaches_global_minimum(rng):
    cfg = RegressorConfig()
    vertices, joints, mask = _regressor_problem(rng)
    R = fit_spatial_regressor(vertices, joints, mask, cfg)

    # Global minimum of the (least-squares) energy, from a dense solve of the
    # whole stacked system for every joint.
    n_joints, n_vertices = mask.shape
    A = np.vstack((np.sqrt(cfg.weight_fit) * vertices.T, np.sqrt(cfg.weight_partition_of_unity) * np.ones(n_vertices)))
    R_dense = np.zeros_like(R)
    for j in range(n_joints):
        system = np.vstack((A, np.sqrt(cfg.weight_locality) * np.diag(1.0 - mask[j])))
        rhs = np.concatenate(
            (np.sqrt(cfg.weight_fit) * joints[j], [np.sqrt(cfg.weight_partition_of_unity)], np.zeros(n_vertices))
        )
        R_dense[j] = np.linalg.lstsq(system, rhs, rcond=None)[0]

    e_closed = regressor_energy(R, vertices, joints, mask, cfg)
    e_dense = regressor_energy(R_dense, vertices, joints, mask, cfg)
    assert e_closed <= e_dense + 1e-9 * max(1.0, e_dense)
    np.testing.assert_allclose(R @ vertices, joints, atol=1e-6)
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-6)


def test_regressor_is_minimum_norm_inside_mask(rng):
    """Among the minimizers we pick the one with the smallest weights."""
    cfg = RegressorConfig()
    vertices, joints, mask = _regressor_problem(rng)
    R = fit_spatial_regressor(vertices, joints, mask, cfg)
    e0 = regressor_energy(R, vertices, joints, mask, cfg)
    # Any direction inside the mask that leaves R V and R 1 unchanged keeps
    # the energy constant and increases the norm.
    j = int(np.argmax(mask.sum(axis=1)))
    inside = np.flatnonzero(mask[j])
    A_S = np.vstack((vertices[inside].T, np.ones(len(inside))))
    null = np.linalg.svd(A_S)[2][4:]
    R2 = R.copy()
    R2[j, inside] += 0.1 * null[0]
    assert abs(regressor_energy(R2, vertices, joints, mask, cfg) - e0) < 1e-8 * max(1.0, e0)
    assert np.linalg.norm(R2[j]) > np.linalg.norm(R[j])


def test_initial_map_zero_gradient(rng):
    cfg = FunctionalMapConfig()
    k, d = 12, 30
    A_tgt, A_src = rng.normal(size=(k, d)), rng.normal(size=(k, d))
    evals_tgt = np.sort(rng.uniform(0, 50, k))
    evals_src = np.sort(rng.uniform(0, 50, k))
    C = optimize_initial_map(A_tgt, A_src, evals_tgt, evals_src, cfg)

    grad = cfg.weight_descriptors * (C @ A_tgt - A_src) @ A_tgt.T + 2 * cfg.weight_laplacian * C * (
        evals_tgt[None, :] - evals_src[:, None]
    ) ** 2
    assert np.abs(grad).max() < 1e-8
    e0 = functional_map_energy(C, A_tgt, A_src, evals_tgt, evals_src, cfg)
    for _ in range(5):
        Cp = C + 1e-3 * rng.normal(size=C.shape)
        assert functional_map_energy(Cp, A_tgt, A_src, evals_tgt, evals_src, cfg) > e0
