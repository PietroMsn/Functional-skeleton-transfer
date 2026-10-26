import numpy as np

from skeleton_transfer.skeleton_priors import (
    SkeletonPriorConfig,
    apply_skeleton_priors,
    bone_lengths,
    symmetric_joint_pairs,
)

# Toy humanoid: 0 pelvis, 1/2 hips, 3/4 knees, 5 spine, 6/7 shoulders, 8/9 elbows
PARENTS = np.array([0, 0, 0, 1, 2, 0, 5, 5, 6, 7])
JOINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.1, -0.1, 0.0], [-0.1, -0.1, 0.0],
        [0.1, -0.5, 0.0], [-0.1, -0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.2, 0.5, 0.0], [-0.2, 0.5, 0.0],
        [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0],
    ]
)  # fmt: skip


def test_symmetric_pairs():
    assert sorted(symmetric_joint_pairs(JOINTS, PARENTS)) == [(1, 2), (3, 4), (6, 7), (8, 9)]


def test_priors_off_is_identity(rng):
    noisy = JOINTS + 0.02 * rng.normal(size=JOINTS.shape)
    cfg = SkeletonPriorConfig(weight_symmetry=0, weight_proportions=0)
    np.testing.assert_allclose(apply_skeleton_priors(noisy, PARENTS, JOINTS, cfg), noisy, atol=1e-8)


def test_perfect_skeleton_is_fixed_point():
    """A scaled, rotated copy of the source satisfies both priors exactly."""
    angle = 0.7
    rot = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    target = 1.3 * JOINTS @ rot.T + [1, 2, 3]
    np.testing.assert_allclose(apply_skeleton_priors(target, PARENTS, JOINTS), target, atol=1e-6)


def test_symmetry_fixes_a_shortened_leg():
    target = JOINTS.copy()
    target[3, 1] = -0.4  # left shin 10 cm shorter than the right one
    cfg = SkeletonPriorConfig(weight_symmetry=100.0, weight_proportions=0.0)
    out = apply_skeleton_priors(target, PARENTS, JOINTS, cfg)
    lengths = bone_lengths(out, PARENTS)
    # bones are indexed by child joint minus one (joint 0 is the root)
    assert abs(lengths[2] - lengths[3]) < 0.005
    assert abs(bone_lengths(target, PARENTS)[2] - bone_lengths(target, PARENTS)[3]) > 0.09
