"""
Skeleton priors applied on the target after the transfer.

The transferred joints ``J0`` are corrected by minimizing

    E(J, s) = w_data * sum_j ||J_j - J0_j||^2
            + w_sym  * sum_{(a, b) mirrored} (l_a(J) - l_b(J))^2
            + w_prop * sum_b (l_b(J) - s * l_b^src)^2

where ``l_b(J) = ||J_b - J_parent(b)||`` is the length of the bone ending in
joint ``b``:

- **symmetry**: mirrored bones (left / right femur, ...) have the same length;
- **proportions**: the bone lengths are those of the source up to a global
  scale ``s``, which is optimized too (the target can be taller or smaller).

All residuals are in the units of the shapes (metres for SMPL), so the
weights are directly comparable. The problem is small (``3 * n_joints + 1``
unknowns) and is solved with ``scipy.optimize.least_squares``.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SkeletonPriorConfig:
    """Weights of the skeleton priors (0 disables a term)."""

    #: Stay close to the transferred joints.
    weight_data: float = 1.0
    #: Left / right mirrored bones have the same length.
    weight_symmetry: float = 1.0
    #: Bone lengths proportional to the source ones.
    weight_proportions: float = 1.0
    #: Axis of the source orthogonal to its symmetry plane (0 = x for SMPL).
    symmetry_axis: int = 0
    #: Max distance between a joint and its mirrored counterpart on the
    #: source, to be detected as a symmetric pair (source units).
    symmetry_tolerance: float = 0.02


def bones(parents: np.ndarray) -> np.ndarray:
    """Child joint of every bone (all joints except the root)."""
    return np.array([j for j, p in enumerate(parents) if j != p])


def bone_lengths(joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Length of each bone, indexed like :func:`bones`."""
    b = bones(parents)
    return np.linalg.norm(joints[b] - joints[parents[b]], axis=1)


def bone_length_ratios(joints: np.ndarray, parents: np.ndarray, source_joints: np.ndarray) -> np.ndarray:
    """Bone lengths relative to the source, after removing the global scale.

    Values far from 1 reveal a failed correspondence: on synthetic SMPL
    bodies the transfers with errors above 10 cm all have some bone shorter
    than 0.6 or longer than 1.6 times its expected length.
    """
    ratios = bone_lengths(joints, parents) / bone_lengths(source_joints, parents)
    return ratios / (bone_lengths(joints, parents).sum() / bone_lengths(source_joints, parents).sum())


def is_plausible(
    joints: np.ndarray, parents: np.ndarray, source_joints: np.ndarray, low: float = 0.6, high: float = 1.6
) -> bool:
    """Whether every bone ratio (see :func:`bone_length_ratios`) is in ``[low, high]``."""
    ratios = bone_length_ratios(joints, parents, source_joints)
    return bool(ratios.min() >= low and ratios.max() <= high)


def symmetric_joint_pairs(
    joints: np.ndarray, parents: np.ndarray, axis: int = 0, tolerance: float = 0.02
) -> List[Tuple[int, int]]:
    """Detect left / right joint pairs on a (roughly symmetric) source.

    Each joint is mirrored with respect to the plane orthogonal to ``axis``
    through the root, and paired with the nearest other joint if they are
    mutual nearest neighbours closer than ``tolerance``. Only pairs whose
    parents are the same joint or are themselves a pair are kept, so that the
    paired bones are really mirror images.
    """
    mirrored = joints.copy()
    mirrored[:, axis] = 2 * joints[0, axis] - joints[:, axis]
    dist = np.linalg.norm(mirrored[:, None, :] - joints[None, :, :], axis=2)
    nearest = dist.argmin(axis=1)
    pairs = {}
    for a, b in enumerate(nearest):
        if a < b and nearest[b] == a and dist[a, b] < tolerance:
            pairs[a] = b
    partner = {**pairs, **{b: a for a, b in pairs.items()}}

    def is_mirror(a, b):
        return a == b or partner.get(a) == b

    return [(a, b) for a, b in pairs.items() if is_mirror(parents[a], parents[b]) and a != parents[a]]


def apply_skeleton_priors(
    joints: np.ndarray,
    parents: np.ndarray,
    source_joints: np.ndarray,
    cfg: Optional[SkeletonPriorConfig] = None,
    symmetric_pairs: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Correct transferred joints with the symmetry and proportion priors.

    Parameters
    ----------
    joints : (n_joints, 3) transferred joints ``J0`` on the target
    parents : (n_joints,) skeleton hierarchy (root points to itself)
    source_joints : (n_joints, 3) joints of the source, for the reference
        proportions and the detection of the symmetric pairs
    cfg : prior weights
    symmetric_pairs : (left, right) joint pairs; detected on the source if
        not given (see :func:`symmetric_joint_pairs`)

    Returns
    -------
    (n_joints, 3) corrected joints.
    """
    cfg = cfg or SkeletonPriorConfig()
    if symmetric_pairs is None:
        symmetric_pairs = symmetric_joint_pairs(
            source_joints, parents, cfg.symmetry_axis, cfg.symmetry_tolerance
        )
    b = bones(parents)
    bone_index = {child: i for i, child in enumerate(b)}
    sym = np.array([(bone_index[l], bone_index[r]) for l, r in symmetric_pairs], dtype=int).reshape(-1, 2)
    src_lengths = bone_lengths(source_joints, parents)
    scale0 = bone_lengths(joints, parents).sum() / src_lengths.sum()

    w_data, w_sym, w_prop = (np.sqrt(w) for w in (cfg.weight_data, cfg.weight_symmetry, cfg.weight_proportions))
    n = len(joints)

    def residuals(x):
        J = x[: 3 * n].reshape(n, 3)
        lengths = np.linalg.norm(J[b] - J[parents[b]], axis=1)
        return np.concatenate(
            (
                w_data * (J - joints).ravel(),
                w_sym * (lengths[sym[:, 0]] - lengths[sym[:, 1]]),
                w_prop * (lengths - x[-1] * src_lengths),
            )
        )

    x0 = np.append(joints.ravel(), scale0)
    solution = least_squares(residuals, x0, method="trf", x_scale="jac")
    return solution.x[: 3 * n].reshape(n, 3)
