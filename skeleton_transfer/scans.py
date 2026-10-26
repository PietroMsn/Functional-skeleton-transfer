"""
Real scans with a registered body model, for evaluation.

Datasets of clothed people (CAPE, 4D-DRESS, ...) provide, for each frame, a
scan of the dressed subject and a registration of the SMPL body under the
clothes. The registration gives the ground-truth skeleton; the scan is the
target of the transfer. This module holds the dataset-independent part:

- :class:`ScanSample`: one frame in a common format (saved as ``.npz`` by
  the dataset adapters, see :func:`save_sample` / :func:`load_sample`);
- :func:`sample_surface`: uniform point sampling of the scan, so that the
  target shares no connectivity with the SMPL source;
- :func:`transfer_landmarks`: landmarks on the scan from the SMPL landmark
  vertices of the registration (nearest scan point, i.e. what a user would
  click on the visible, possibly clothed, surface);
- :func:`make_target`: everything together, from a sample to a
  ``(TargetShape, ground-truth joints)`` pair.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .shapes import SMPL_LANDMARKS, TargetShape


@dataclass
class ScanSample:
    """One frame of a dataset of scans with a registered SMPL body.

    Attributes
    ----------
    name : unique identifier (e.g. ``"4ddress/00122/Outer/Take2/f00011"``)
    scan_vertices : (n, 3) scan points / vertices, same frame as the registration
    scan_faces : (m, 3) triangles of the scan, or None for a point cloud
    smpl_vertices : (6890, 3) registered SMPL body (under the clothes)
    gender : ``"male"``, ``"female"`` or ``"neutral"``, selects the SMPL
        joint regressor
    category : group used in the reports (e.g. clothing type)
    scan_labels : optional (n,) per-vertex semantic labels of the scan
    joints : optional (24, 3) ground-truth joints. If missing they are
        regressed from ``smpl_vertices`` (the SMPL regressor applied to a
        posed body is accurate within a few centimetres; datasets that
        provide pose and shape should give the forward-kinematics joints)
    metadata : free-form extra information
    """

    name: str
    scan_vertices: np.ndarray
    scan_faces: Optional[np.ndarray]
    smpl_vertices: np.ndarray
    gender: str = "neutral"
    category: str = ""
    scan_labels: Optional[np.ndarray] = None
    joints: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


def save_sample(path: str, sample: ScanSample) -> None:
    """Save a sample in the common ``.npz`` format."""
    arrays = dict(
        name=sample.name,
        scan_vertices=np.asarray(sample.scan_vertices, dtype=np.float32),
        smpl_vertices=np.asarray(sample.smpl_vertices, dtype=np.float32),
        gender=sample.gender,
        category=sample.category,
        metadata=np.array(sample.metadata, dtype=object),
    )
    if sample.scan_faces is not None:
        arrays["scan_faces"] = np.asarray(sample.scan_faces, dtype=np.int32)
    if sample.scan_labels is not None:
        arrays["scan_labels"] = np.asarray(sample.scan_labels)
    if sample.joints is not None:
        arrays["joints"] = np.asarray(sample.joints, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def load_sample(path: str) -> ScanSample:
    """Load a sample saved with :func:`save_sample`."""
    data = np.load(path, allow_pickle=True)
    return ScanSample(
        name=str(data["name"]),
        scan_vertices=data["scan_vertices"].astype(float),
        scan_faces=data["scan_faces"].astype(int) if "scan_faces" in data else None,
        smpl_vertices=data["smpl_vertices"].astype(float),
        gender=str(data["gender"]),
        category=str(data["category"]),
        scan_labels=data["scan_labels"] if "scan_labels" in data else None,
        joints=data["joints"].astype(float) if "joints" in data else None,
        metadata=data["metadata"].item() if "metadata" in data else {},
    )


def sample_surface(vertices: np.ndarray, faces: np.ndarray, n_points: int, rng: np.random.Generator):
    """Uniformly sample points on a triangle mesh (area weighted).

    Returns
    -------
    points : (n_points, 3)
    face_index : (n_points,) triangle of each point
    """
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    face_index = rng.choice(len(faces), size=n_points, p=areas / areas.sum())
    u, v = rng.random(n_points), rng.random(n_points)
    flip = u + v > 1  # reflect into the triangle
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = a[face_index] + u[:, None] * (b - a)[face_index] + v[:, None] * (c - a)[face_index]
    return points, face_index


def transfer_landmarks(body_vertices: np.ndarray, landmark_ids: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Index of the point nearest to each landmark vertex of the registered body."""
    return cKDTree(points).query(body_vertices[landmark_ids])[1]


def make_target(
    sample: ScanSample,
    joint_regressor: np.ndarray,
    n_points: Optional[int] = 8000,
    landmarks: np.ndarray = SMPL_LANDMARKS,
    max_distance: Optional[float] = 0.3,
    rng: Optional[np.random.Generator] = None,
):
    """Target point cloud and ground-truth joints of a scan sample.

    Parameters
    ----------
    sample : the scan and its SMPL registration
    joint_regressor : (24, 6890) SMPL joint regressor (of the sample gender),
        used for the ground-truth joints if ``sample.joints`` is missing
    n_points : points sampled uniformly on the scan surface; ``None`` keeps
        the scan vertices (also used when the scan has no faces, subsampled
        if needed)
    landmarks : SMPL vertex indices of the landmarks, transferred to the
        nearest scan points
    max_distance : drop the scan points farther than this from the
        registered body (floor, capture artefacts); None keeps all
    rng : random generator for the sampling

    Returns
    -------
    target : TargetShape point cloud (``faces=None``)
    joints : (24, 3) ground-truth joints
    """
    rng = rng or np.random.default_rng(0)
    if sample.scan_faces is not None and n_points is not None:
        points, _ = sample_surface(sample.scan_vertices, sample.scan_faces, n_points, rng)
    else:
        points = sample.scan_vertices
        if n_points is not None and len(points) > n_points:
            points = points[rng.choice(len(points), n_points, replace=False)]

    if max_distance is not None:
        dist = cKDTree(sample.smpl_vertices).query(points)[0]
        points = points[dist <= max_distance]

    target = TargetShape(
        vertices=points,
        faces=None,
        landmarks=transfer_landmarks(sample.smpl_vertices, np.asarray(landmarks), points),
    )
    joints = sample.joints if sample.joints is not None else joint_regressor @ sample.smpl_vertices
    return target, joints
