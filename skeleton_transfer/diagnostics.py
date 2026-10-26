"""
Quality scores of a transfer that do not need the ground truth.

They are used to detect failures, to choose between restarts and to monitor
the iterative refinement. All of them are "lower is better" except
``coverage`` and ``segment_consistency``. Which of them actually predicts the
joint error is an empirical question: ``tools/benchmark_synthetic.py`` reports
their rank correlation with the ground-truth error.
"""

import numpy as np

from .skeleton_priors import bone_length_ratios, bone_lengths, symmetric_joint_pairs, bones


def point_to_bone_distances(points: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Distance of every point from every bone segment, ``(n_points, n_bones)``."""
    b = bones(parents)
    a, c = joints[parents[b]], joints[b]  # segment endpoints, (n_bones, 3)
    ac = c - a
    t = np.einsum("pbk,bk->pb", points[:, None, :] - a[None], ac) / np.maximum(np.sum(ac**2, axis=1), 1e-12)
    closest = a[None] + np.clip(t, 0, 1)[..., None] * ac[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def bone_affinity(points: np.ndarray, joints: np.ndarray, parents: np.ndarray, temperature: float) -> np.ndarray:
    """Soft assignment of the points to the bones, ``(n_points, n_bones)``.

    ``softmax(-d / tau)`` of the point-to-bone distances, with ``tau`` equal
    to ``temperature`` times the mean bone length (so it is scale invariant).
    """
    d = point_to_bone_distances(points, joints, parents)
    tau = temperature * bone_lengths(joints, parents).mean()
    logits = -(d - d.min(axis=1, keepdims=True)) / tau
    w = np.exp(logits)
    return w / w.sum(axis=1, keepdims=True)


def transfer_diagnostics(
    joints_spectral: np.ndarray,
    joints_pointwise: np.ndarray,
    C: np.ndarray,
    p2p: np.ndarray,
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    source_joints: np.ndarray,
    parents: np.ndarray,
    temperature: float = 0.25,
) -> dict:
    """Unsupervised quality scores of a transfer.

    - ``bone_ratio``: max ``|log|`` of the bone lengths relative to the source
      (after removing the global scale); large if some bone is torn apart;
    - ``symmetry``: mean length difference of mirrored bones, relative to the
      mean bone length;
    - ``spectral_pointwise_gap``: mean distance between the joints of the
      spectral and of the pointwise transfer, relative to the mean bone length
      (the two should agree for a good map);
    - ``orthogonality``: deviation of ``C`` from a (scaled) orthogonal matrix,
      expected for near-isometric shapes;
    - ``coverage``: fraction of target points hit by the point-to-point map
      (a map that collapses parts of the target has low coverage);
    - ``segment_consistency``: fraction of source vertices that map onto a
      target point closest to the same bone, the bones being placed by the
      transferred skeleton. Measures the agreement of map and skeleton.
    """
    mean_bone = bone_lengths(source_joints, parents).mean() * (
        bone_lengths(joints_spectral, parents).sum() / bone_lengths(source_joints, parents).sum()
    )
    ratios = bone_length_ratios(joints_spectral, parents, source_joints)
    b = bones(parents)
    index = {child: i for i, child in enumerate(b)}
    lengths = bone_lengths(joints_spectral, parents)
    pairs = symmetric_joint_pairs(source_joints, parents)
    sym = np.mean([abs(lengths[index[l]] - lengths[index[r]]) for l, r in pairs]) / mean_bone if pairs else 0.0

    gram = C.T @ C
    gram = gram / np.mean(np.diag(gram))
    orthogonality = np.linalg.norm(gram - np.eye(len(gram))) / np.sqrt(len(gram))

    label_src = point_to_bone_distances(source_vertices, source_joints, parents).argmin(axis=1)
    label_tgt = point_to_bone_distances(target_vertices, joints_spectral, parents).argmin(axis=1)

    return {
        "bone_ratio": float(np.abs(np.log(np.maximum(ratios, 1e-6))).max()),
        "symmetry": float(sym),
        "spectral_pointwise_gap": float(np.linalg.norm(joints_spectral - joints_pointwise, axis=1).mean() / mean_bone),
        "orthogonality": float(orthogonality),
        "coverage": float(len(np.unique(p2p)) / len(target_vertices)),
        "segment_consistency": float(np.mean(label_src == label_tgt[p2p])),
    }
