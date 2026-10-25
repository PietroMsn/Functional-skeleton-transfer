"""
Shape containers and data loading.

The pipeline works with two kinds of shapes:

- a :class:`SourceShape`: a rigged template, i.e. a mesh with a skeleton
  (joint positions + hierarchy) and skinning weights. In the provided example
  this is the SMPL template;
- a :class:`TargetShape`: any mesh or point cloud that should receive the
  skeleton (in the example, a remeshed Dynamic FAUST scan).

Both carry a small set of corresponding landmark vertices, used to
disambiguate the intrinsic symmetries when computing the correspondence.

All indices (faces, landmarks) are stored **0-based**, even though the
provided ``.mat`` files use MATLAB's 1-based convention.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.io as sio

#: Indices (0-based) of the 7 landmarks on the SMPL template used in the paper.
#: They must be given in the same order as the landmarks of the target.
SMPL_LANDMARKS = np.array([3509, 3076, 414, 5909, 6683, 3216, 2444]) - 1


@dataclass
class SourceShape:
    """A rigged template shape.

    Attributes
    ----------
    vertices : (n_vertices, 3) float array
    faces : (n_faces, 3) int array, 0-based (may be None for a point cloud)
    joints : (n_joints, 3) float array, joint positions
    parents : (n_joints,) int array, ``parents[i]`` is the parent joint of
        joint ``i``; the root points to itself (``parents[0] == 0``)
    skinning_weights : (n_vertices, n_joints) float array
    landmarks : (n_landmarks,) int array of vertex indices, 0-based
    """

    vertices: np.ndarray
    faces: Optional[np.ndarray]
    joints: np.ndarray
    parents: np.ndarray
    skinning_weights: np.ndarray
    landmarks: np.ndarray

    @property
    def n_joints(self) -> int:
        return len(self.joints)


@dataclass
class TargetShape:
    """A shape that will receive the skeleton.

    Attributes
    ----------
    vertices : (n_vertices, 3) float array
    faces : (n_faces, 3) int array, 0-based (None for a point cloud)
    landmarks : (n_landmarks,) int array of vertex indices, 0-based, in the
        same order as ``SourceShape.landmarks``
    """

    vertices: np.ndarray
    faces: Optional[np.ndarray]
    landmarks: np.ndarray


def load_smpl_source(path: str, landmarks: np.ndarray = SMPL_LANDMARKS) -> SourceShape:
    """Load the SMPL template from a ``.mat`` export of the SMPL model.

    The file must contain the standard SMPL fields: ``v_template``, ``f``,
    ``J_regressor``, ``weights`` and ``kintree_table``. The joints are
    obtained with SMPL's own regressor applied to the rest-pose template.
    """
    data = sio.loadmat(path)

    vertices = np.asarray(data["v_template"], dtype=float)
    faces = np.asarray(data["f"], dtype=int)  # already 0-based in SMPL

    smpl_regressor = np.asarray(data["J_regressor"].todense())
    joints = smpl_regressor @ vertices

    # First row of kintree_table holds the parent of each joint; the root's
    # parent is stored as a sentinel value, we make it point to itself.
    parents = np.array(data["kintree_table"][0, :], dtype=int)
    parents[0] = 0

    return SourceShape(
        vertices=vertices,
        faces=faces,
        joints=joints,
        parents=parents,
        skinning_weights=np.asarray(data["weights"], dtype=float),
        landmarks=np.asarray(landmarks, dtype=int),
    )


def load_target_mat(
    path: str,
    vertices_key: str = "vertex",
    faces_key: Optional[str] = "faces",
    landmarks_key: str = "landmarkRes",
) -> TargetShape:
    """Load a target shape from a ``.mat`` file with 1-based indices.

    Parameters
    ----------
    path : path of the ``.mat`` file
    vertices_key, faces_key, landmarks_key : names of the fields holding the
        vertices, the faces and the landmark indices. Pass ``faces_key=None``
        for point clouds.
    """
    data = sio.loadmat(path)
    faces = None
    if faces_key is not None:
        faces = np.asarray(data[faces_key], dtype=int) - 1
    return TargetShape(
        vertices=np.asarray(data[vertices_key], dtype=float),
        faces=faces,
        landmarks=np.asarray(data[landmarks_key], dtype=int).ravel() - 1,
    )


def read_off(path: str):
    """Read a triangle mesh in OFF format.

    Returns
    -------
    vertices : (n_vertices, 3) float array
    faces : (n_faces, 3) int array, 0-based
    """
    with open(path) as f:
        tokens = [line.split("#")[0].split() for line in f]
    tokens = [t for t in tokens if t]  # drop empty / comment lines

    header = tokens.pop(0)
    if header[0] != "OFF":
        raise ValueError(f"{path} is not an OFF file")
    if len(header) > 1:  # counts on the same line as "OFF"
        counts = header[1:]
    else:
        counts = tokens.pop(0)
    n_vertices, n_faces = int(counts[0]), int(counts[1])

    vertices = np.array(tokens[:n_vertices], dtype=float)[:, :3]
    faces = np.array([t[1:4] for t in tokens[n_vertices : n_vertices + n_faces]], dtype=int)
    return vertices, faces


def read_obj(path: str):
    """Read the vertices and faces of a Wavefront OBJ file.

    Polygons with more than three vertices are fan-triangulated; texture and
    normal indices (``f 1/2/3``) are ignored. A file without faces is
    returned as a point cloud (``faces is None``).

    Returns
    -------
    vertices : (n_vertices, 3) float array
    faces : (n_faces, 3) int array, 0-based, or None
    """
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                # negative indices are relative to the end of the vertex list
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    return np.array(vertices, dtype=float), (np.array(faces, dtype=int) if faces else None)


def load_mesh(path: str):
    """Read a mesh or point cloud from disk.

    ``.off`` and ``.obj`` are read natively; any other format (``.ply``,
    ``.stl``, ...) is delegated to ``trimesh``, if installed.

    Returns
    -------
    vertices : (n_vertices, 3) float array
    faces : (n_faces, 3) int array, 0-based, or None for a point cloud
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".off":
        return read_off(path)
    if ext == ".obj":
        return read_obj(path)
    try:
        import trimesh
    except ImportError as err:
        raise ImportError(f"Reading '{ext}' files requires trimesh: pip install trimesh") from err
    geometry = trimesh.load(path, process=False)
    faces = getattr(geometry, "faces", None)
    if faces is not None and len(faces) == 0:
        faces = None
    return np.asarray(geometry.vertices, dtype=float), (None if faces is None else np.asarray(faces, dtype=int))


def load_landmarks(path: str, one_based: bool = False) -> np.ndarray:
    """Read landmark vertex indices from a text file.

    Indices may be separated by whitespace, commas or newlines; ``#`` starts
    a comment.
    """
    with open(path) as f:
        tokens = [line.split("#")[0].replace(",", " ").split() for line in f]
    landmarks = np.array([int(t) for line in tokens for t in line], dtype=int)
    return landmarks - 1 if one_based else landmarks


def load_target(
    path: str,
    landmarks=None,
    one_based_landmarks: bool = False,
    point_cloud: bool = False,
) -> TargetShape:
    """Load a target shape from any supported file.

    Parameters
    ----------
    path : ``.mat`` (fields ``vertex``, ``faces``, ``landmarkRes``, see
        :func:`load_target_mat`) or a mesh / point cloud file readable by
        :func:`load_mesh`
    landmarks : landmark vertex indices, as an array or as the path of a text
        file. Required for mesh files, optional for ``.mat`` files (overrides
        the ones stored in the file).
    one_based_landmarks : the given landmarks use 1-based (MATLAB) indices
    point_cloud : drop the faces and treat the shape as a point cloud
    """
    if path.lower().endswith(".mat"):
        target = load_target_mat(path)
    else:
        if landmarks is None:
            raise ValueError(f"Landmarks are required for '{path}' (pass an array or a text file).")
        vertices, faces = load_mesh(path)
        target = TargetShape(vertices=vertices, faces=faces, landmarks=np.zeros(0, dtype=int))

    if landmarks is not None:
        if isinstance(landmarks, (str, os.PathLike)):
            target.landmarks = load_landmarks(landmarks, one_based_landmarks)
        else:
            target.landmarks = np.asarray(landmarks, dtype=int).ravel() - (1 if one_based_landmarks else 0)
    if point_cloud:
        target.faces = None
    return target
