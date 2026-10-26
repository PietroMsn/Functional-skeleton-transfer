"""
Synthetic targets with ground-truth joints, for evaluation.

:func:`make_synthetic_target` samples an SMPL body with random shape and
pose, optionally "dresses" it, and returns it as a point cloud (so that it
shares no connectivity with the SMPL source) together with its true joints.

The clothing models are deliberately simple geometric stress tests, not
cloth simulations:

- ``"shirt"``: the torso and arms are inflated along the normals by a few
  centimetres (smoothly varying), like a loose shirt;
- ``"skirt"``: between the hips and below the knees the two legs are
  replaced by a single flared tube that encloses both, like a skirt or a
  dress. The legs are no longer separated, which is what makes loose clothing
  hard for intrinsic methods.
"""

from typing import Optional, Sequence

import numpy as np

from .shapes import SMPL_LANDMARKS, TargetShape
from .smpl import SMPLModel

UPPER_BODY_JOINTS = [3, 6, 9, 13, 14, 16, 17, 18, 19]  # spine, collars, shoulders, elbows
LEG_JOINTS = [0, 1, 2, 4, 5]  # pelvis, hips, knees


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals of a triangle mesh."""
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _smooth_noise(vertices: np.ndarray, rng: np.random.Generator, n_waves: int = 6, wavelength: float = 0.3):
    """Smooth random field in [-1, 1] (sum of random plane waves)."""
    dirs = rng.normal(size=(n_waves, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    return np.mean(np.sin(2 * np.pi * vertices @ dirs.T / wavelength + phases), axis=1) * np.sqrt(2)


def add_shirt(vertices, faces, weights, rng, thickness=0.04):
    """Inflate the vertices skinned mostly to the upper body."""
    upper = np.isin(weights.argmax(axis=1), UPPER_BODY_JOINTS)
    offset = thickness * (1 + 0.5 * np.clip(_smooth_noise(vertices, rng), -1, 1))
    out = vertices.copy()
    out[upper] += offset[upper, None] * vertex_normals(vertices, faces)[upper]
    return out


def add_skirt(vertices, joints, weights, length=1.2, flare=0.35, margin=0.02, up_axis=1):
    """Replace the legs between the hips and the hem by one flared tube.

    ``length`` is the hem position as a fraction of the hip-knee distance
    (1.2 = a bit below the knees). The skirt is built in horizontal slices
    around the vertical axis ``up_axis``: at each height every leg vertex is
    pushed radially onto a circle that encloses both legs.
    """
    hip_h = joints[[1, 2], up_axis].mean()
    knee_h = joints[[4, 5], up_axis].mean()
    hem_h = hip_h - length * (hip_h - knee_h)
    h = vertices[:, up_axis]
    region = np.isin(weights.argmax(axis=1), LEG_JOINTS) & (h < hip_h) & (h > hem_h)
    if not region.any():
        return vertices.copy()

    horizontal = [a for a in range(3) if a != up_axis]
    out = vertices.copy()
    pts = vertices[region][:, horizontal]
    center = pts.mean(axis=0)
    radial = pts - center
    dist = np.linalg.norm(radial, axis=1)
    radius_top = np.percentile(dist[h[region] > hip_h - 0.05], 95) + margin if np.any(h[region] > hip_h - 0.05) else dist.max()
    radius = radius_top + flare * (hip_h - h[region])
    out_h = out[region]
    out_h[:, horizontal] = center + radial / np.maximum(dist, 1e-9)[:, None] * radius[:, None]
    out[region] = out_h
    return out


def make_synthetic_target(
    model: SMPLModel,
    rng: np.random.Generator,
    pose_std: float = 0.3,
    shape_std: float = 1.0,
    clothing: Sequence[str] = (),
    n_points: Optional[int] = 5000,
    noise: float = 0.0,
    landmarks: np.ndarray = SMPL_LANDMARKS,
):
    """Random SMPL body as a point cloud, with its ground-truth joints.

    Parameters
    ----------
    model : SMPL model
    rng : random generator
    pose_std : std (radians) of the axis-angle rotation of every joint;
        the global orientation is kept fixed (upright body)
    shape_std : std of the shape coefficients (1 = population std)
    clothing : any of ``"shirt"``, ``"skirt"``
    n_points : number of points sampled from the vertices (None = all)
    noise : std of the Gaussian noise added to the points
    landmarks : SMPL vertex indices of the landmarks (always kept)

    Returns
    -------
    target : TargetShape point cloud (``faces=None``)
    joints : (24, 3) ground-truth joints
    """
    betas = rng.normal(0, shape_std, model.n_betas)
    pose = rng.normal(0, pose_std, (24, 3))
    pose[0] = 0
    vertices, joints = model.forward(betas, pose)

    if "shirt" in clothing:
        vertices = add_shirt(vertices, model.faces, model.weights, rng)
    if "skirt" in clothing:
        vertices = add_skirt(vertices, joints, model.weights)
    unknown = set(clothing) - {"shirt", "skirt"}
    if unknown:
        raise ValueError(f"Unknown clothing {unknown}")

    n = len(vertices)
    landmarks = np.asarray(landmarks, dtype=int)
    if n_points is None or n_points >= n:
        keep = np.arange(n)
    else:
        others = rng.choice(np.setdiff1d(np.arange(n), landmarks), n_points - len(landmarks), replace=False)
        keep = np.concatenate((landmarks, np.sort(others)))
    points = vertices[keep] + noise * rng.normal(size=(len(keep), 3))
    new_index = {v: i for i, v in enumerate(keep)}
    target = TargetShape(vertices=points, faces=None, landmarks=np.array([new_index[l] for l in landmarks]))
    return target, joints
