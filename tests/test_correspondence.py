"""Bone affinities, diagnostics and the correspondence refinements."""

import numpy as np
import pytest

from skeleton_transfer.config import FunctionalMapConfig, TransferConfig
from skeleton_transfer.diagnostics import bone_affinity, point_to_bone_distances, transfer_diagnostics
from skeleton_transfer.functional_map import functional_map_to_p2p, refine_with_features
from skeleton_transfer.spectral import compute_laplace_basis

from .conftest import ellipsoid_mesh
from .test_skeleton_priors import JOINTS, PARENTS


def test_point_to_bone_distances():
    # bone 0 goes from the pelvis (0, 0, 0) to the left hip (0.1, -0.1, 0)
    points = np.array([[0.05, -0.05, 0.3], [0.2, -0.2, 0.0], [-0.1, 0.1, 0.0]])
    d = point_to_bone_distances(points, JOINTS, PARENTS)[:, 0]
    np.testing.assert_allclose(d, [0.3, np.hypot(0.1, 0.1), np.hypot(0.1, 0.1)])


def test_bone_affinity():
    points = JOINTS[PARENTS[1:]] * 0.5 + JOINTS[1:] * 0.5  # midpoint of every bone
    a = bone_affinity(points, JOINTS, PARENTS, temperature=0.25)
    np.testing.assert_allclose(a.sum(axis=1), 1.0)
    np.testing.assert_array_equal(a.argmax(axis=1), np.arange(len(points)))
    # scale invariance
    np.testing.assert_allclose(bone_affinity(3 * points, 3 * JOINTS, PARENTS, 0.25), a)


def test_diagnostics_of_perfect_transfer():
    vertices = np.random.default_rng(0).normal(size=(200, 3)) * 0.3
    diag = transfer_diagnostics(
        JOINTS, JOINTS, np.eye(10), np.arange(200), vertices, vertices, JOINTS, PARENTS
    )
    assert diag["bone_ratio"] == pytest.approx(0, abs=1e-9)
    assert diag["symmetry"] == pytest.approx(0, abs=1e-9)
    assert diag["spectral_pointwise_gap"] == 0
    assert diag["orthogonality"] == pytest.approx(0, abs=1e-9)
    assert diag["coverage"] == 1 and diag["segment_consistency"] == 1


def test_landmarks_are_forced():
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.7, 0.5))
    evecs = compute_laplace_basis(vertices, faces, 10).evecs
    landmarks = (np.array([3, 50]), np.array([700, 800]))
    p2p = functional_map_to_p2p(np.eye(10), evecs, evecs, landmarks=landmarks)
    assert p2p[3] == 700 and p2p[50] == 800
    assert np.mean(p2p == np.arange(len(vertices))) > 0.9


def test_features_steer_the_map():
    """Features that single out a reflection make the map follow it."""
    vertices, faces = ellipsoid_mesh(radii=(1.0, 0.7, 0.5))
    evecs = compute_laplace_basis(vertices, faces, 20).evecs
    mirrored = vertices * [-1, 1, 1]
    identity = np.arange(len(vertices))
    # Features: the coordinates on the source, the mirrored ones on the target
    # (a cue that the map should be the reflection x -> -x, which is also an
    # isometry of the ellipsoid). Weak features leave the map unchanged,
    # strong ones flip it.
    for weight, expected in [(0.5, vertices), (5.0, mirrored)]:
        features = (weight * vertices, weight * mirrored)
        _, p2p = refine_with_features(identity, evecs, evecs, 20, 5, features)
        assert np.linalg.norm(vertices[p2p] - expected, axis=1).mean() < 0.01


def test_restart_sizes_validated():
    cfg = TransferConfig(functional_map=FunctionalMapConfig(restart_init_sizes=(15,)))
    with pytest.raises(ValueError):
        cfg.validate()


def test_restart_selection_validated():
    with pytest.raises(ValueError):
        TransferConfig(functional_map=FunctionalMapConfig(restart_selection="other")).validate()
    with pytest.raises(ValueError):  # "refined" needs the iterative refinement
        TransferConfig(functional_map=FunctionalMapConfig(restart_selection="refined")).validate()
