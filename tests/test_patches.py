"""Local CMH functions on body parts."""

import numpy as np
from scipy.spatial.transform import Rotation

from skeleton_transfer.patches import (
    SMPL_BODY_PARTS, joint_to_part, local_cmh_functions, part_weights_from_skeleton, part_weights_from_skinning,
    refine_local_cmh,
)  # fmt: skip
from skeleton_transfer.spectral import compute_laplace_basis

from .conftest import ellipsoid_mesh
from .test_skeleton_priors import JOINTS, PARENTS


def _two_parts(vertices):
    """Soft weights of the halves x < 0 and x > 0."""
    w = 1 / (1 + np.exp(-vertices[:, 0] / 0.2))
    return np.column_stack((1 - w, w))


def test_local_functions_represent_articulated_coordinates():
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.6, 0.5))
    basis = compute_laplace_basis(vertices, faces, 10)
    W = _two_parts(vertices)
    phi = local_cmh_functions(basis, 10, W)
    assert phi.shape == (len(vertices), 17)  # the weights sum to the constant, already in the basis
    np.testing.assert_allclose(phi.T @ (basis.mass @ phi), np.eye(17), atol=1e-8)
    # Linear blend skinning of the two halves, each with its own rigid motion
    R = Rotation.from_rotvec([[0, 0, 0.8], [0.3, -0.5, 0]]).as_matrix()
    t = np.array([[0.1, 0, 0], [0, 0.2, -0.3]])
    posed = sum(W[:, [b]] * (vertices @ R[b].T + t[b]) for b in range(2))
    projected = phi @ (phi.T @ (basis.mass @ posed))
    np.testing.assert_allclose(projected, posed, atol=1e-8)
    # The global CMH functions represent the rest coordinates, not the posed ones
    cmh = basis.functions(10, cmh=True)
    assert np.abs(cmh @ (cmh.T @ (basis.mass @ posed)) - posed).max() > 1e-2


def test_degenerate_parts_are_dropped():
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.6, 0.5))
    basis = compute_laplace_basis(vertices, faces, 10)
    W = np.column_stack((_two_parts(vertices), np.zeros(len(vertices))))  # an empty third part
    assert local_cmh_functions(basis, 10, W).shape[1] == 17


def test_refinement_keeps_the_identity():
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.6, 0.5))
    basis = compute_laplace_basis(vertices, faces, 20)
    phi = local_cmh_functions(basis, 20, _two_parts(vertices))
    identity = np.arange(len(vertices))
    C, p2p = refine_local_cmh(identity, phi, phi, 3)
    np.testing.assert_array_equal(p2p, identity)
    np.testing.assert_allclose(C, np.eye(phi.shape[1]), atol=1e-8)


def test_part_weights():
    part = joint_to_part()
    assert part.max() + 1 == len(SMPL_BODY_PARTS)
    skinning = np.random.default_rng(0).dirichlet(np.ones(24), size=50)
    W = part_weights_from_skinning(skinning, part)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert W.shape == (50, len(SMPL_BODY_PARTS))
    points = np.random.default_rng(1).normal(size=(30, 3)) * 0.3
    W = part_weights_from_skeleton(points, JOINTS, PARENTS, np.arange(len(JOINTS)))
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
