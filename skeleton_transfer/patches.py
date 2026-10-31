"""
Local Coordinates Manifold Harmonics on body parts.

The global CMH functions (:meth:`.spectral.LaplaceBasis.functions`) are the
coordinates of a shape in its own pose, so they differ between two poses of
the same body. Under an articulated motion, however, every body part moves
(almost) rigidly: with soft part weights ``w_b`` (a partition of unity, e.g.
skinning weights) the coordinates of the deformed shape are a linear blend
skinning of the rest coordinates,

    x'(v) = sum_b w_b(v) (R_b x(v) + t_b),

so they lie in the span of the 4 functions ``w_b``, ``w_b x``, ``w_b y``,
``w_b z`` of every part. Appending these functions to the Laplace-Beltrami
basis (:func:`local_cmh_functions`) gives a basis that represents the
coordinates of the shape in any articulated pose; between two shapes, the
functional map acts on them as one affine transformation per part.

The parts are groups of SMPL joints (:data:`SMPL_BODY_PARTS`); their weights
come either from skinning weights (:func:`part_weights_from_skinning`) or from
a skeleton placed in the shape (:func:`part_weights_from_skeleton`).
:func:`refine_local_cmh` refines a point-to-point map in the extended bases.
"""

from typing import Dict, List

import numpy as np

from .diagnostics import bone_affinity
from .functional_map import functional_map_to_p2p
from .skeleton_priors import bones
from .smpl import SMPL_JOINT_NAMES
from .spectral import LaplaceBasis

#: Body parts that move almost rigidly, as groups of the SMPL joints whose
#: transformations they follow (hands with the forearms, feet separate from
#: the shins, collars with the torso).
SMPL_BODY_PARTS: Dict[str, List[str]] = {
    "pelvis": ["pelvis"],
    "torso": ["spine1", "spine2", "spine3", "L_collar", "R_collar"],
    "head": ["neck", "head"],
    "L_thigh": ["L_hip"],
    "R_thigh": ["R_hip"],
    "L_shin": ["L_knee"],
    "R_shin": ["R_knee"],
    "L_foot": ["L_ankle", "L_foot"],
    "R_foot": ["R_ankle", "R_foot"],
    "L_upper_arm": ["L_shoulder"],
    "R_upper_arm": ["R_shoulder"],
    "L_forearm": ["L_elbow", "L_wrist", "L_hand"],
    "R_forearm": ["R_elbow", "R_wrist", "R_hand"],
}

#: Relative eigenvalue below which a direction of the local functions is
#: dropped (the functions of a part with almost no weight are degenerate).
RANK_TOLERANCE = 1e-8


def joint_to_part(parts: Dict[str, List[str]] = SMPL_BODY_PARTS, names: List[str] = SMPL_JOINT_NAMES) -> np.ndarray:
    """Part index of every joint, ``(n_joints,)``."""
    index = {name: p for p, joints in enumerate(parts.values()) for name in joints}
    return np.array([index[name] for name in names])


def part_weights_from_skinning(skinning_weights: np.ndarray, part_of_joint: np.ndarray) -> np.ndarray:
    """Part weights ``(n_vertices, n_parts)``: the skinning weights summed over the joints of each part."""
    W = np.zeros((len(skinning_weights), part_of_joint.max() + 1))
    np.add.at(W.T, part_of_joint, skinning_weights.T)
    return W


def part_weights_from_skeleton(
    points: np.ndarray, joints: np.ndarray, parents: np.ndarray, part_of_joint: np.ndarray, temperature: float = 0.25
) -> np.ndarray:
    """Part weights ``(n_points, n_parts)`` from a skeleton placed in the shape.

    The bone affinities (:func:`.diagnostics.bone_affinity`) summed over the
    bones of each part; a bone belongs to the part of the joint it hangs from
    (its parent), whose transformation moves it.
    """
    affinity = bone_affinity(points, joints, parents, temperature)
    W = np.zeros((len(points), part_of_joint.max() + 1))
    np.add.at(W.T, part_of_joint[parents[bones(parents)]], affinity.T)
    return W


def local_cmh_functions(basis: LaplaceBasis, k: int, weights: np.ndarray) -> np.ndarray:
    """The first ``k`` eigenfunctions followed by the local CMH functions.

    For every part ``b`` the functions ``w_b`` and ``w_b (x - c_b)`` (``c_b``
    the weighted centroid of the part), made ``M``-orthogonal to the
    eigenfunctions and ``M``-orthonormalized together (by a symmetric
    whitening, which drops the degenerate directions).

    Returns
    -------
    (n_vertices, k + m) array, with ``m <= 4 * n_parts``.
    """
    phi = basis.evecs[:, :k]
    mass = basis.mass.diagonal()
    columns = []
    for w in weights.T:
        centroid = (mass * w) @ basis.vertices / max((mass * w).sum(), 1e-12)
        columns += [w, *(w[:, None] * (basis.vertices - centroid)).T]
    F = np.column_stack(columns)
    for _ in range(2):  # twice, for the numerical orthogonality
        F = F - phi @ (phi.T @ (mass[:, None] * F))
    evals, evecs = np.linalg.eigh(F.T @ (mass[:, None] * F))
    keep = evals > RANK_TOLERANCE * evals.max()
    return np.hstack((phi, F @ (evecs[:, keep] / np.sqrt(evals[keep]))))


def refine_local_cmh(p2p: np.ndarray, phi_src: np.ndarray, phi_tgt: np.ndarray, n_iterations: int):
    """ZoomOut-like iterations in two (extended) bases of fixed size.

    The bases may have different sizes (e.g. different numbers of local
    functions), so ``C`` is ``(n_src_functions, n_tgt_functions)``.

    Returns
    -------
    C : functional map, target coefficients -> source coefficients
    p2p : point-to-point map (source vertex -> target vertex)
    """
    pinv_src = np.linalg.pinv(phi_src)
    C = pinv_src @ phi_tgt[p2p]
    for _ in range(n_iterations):
        p2p = functional_map_to_p2p(C, phi_src, phi_tgt)
        C = pinv_src @ phi_tgt[p2p]
    return C, p2p
