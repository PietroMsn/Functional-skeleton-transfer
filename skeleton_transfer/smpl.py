"""
Minimal SMPL model (Loper et al. 2015), NumPy only.

Used to generate shapes with known ground-truth joints (any body shape and
pose) for evaluation, see :mod:`skeleton_transfer.synthetic`. It reads the
``.mat`` export of the SMPL model that is also used as the transfer source.
"""

import os
import pickle
from dataclasses import dataclass

import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation

from .shapes import SMPL_LANDMARKS, SourceShape

#: Names of the 24 SMPL joints. "L"/"R" are the subject's left and right.
SMPL_JOINT_NAMES = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2", "L_ankle",
    "R_ankle", "spine3", "L_foot", "R_foot", "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_hand", "R_hand",
]  # fmt: skip


class _ChumpyStub:
    """Placeholder for the chumpy classes stored in the official SMPL files."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


class _ChumpyFreeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "chumpy":
            return _ChumpyStub
        return super().find_class(module, name)


@dataclass
class SMPLModel:
    """The SMPL parameters needed to pose and shape the template."""

    v_template: np.ndarray  # (n_vertices, 3)
    faces: np.ndarray  # (n_faces, 3)
    shapedirs: np.ndarray  # (n_vertices, 3, n_betas)
    posedirs: np.ndarray  # (n_vertices, 3, 207)
    J_regressor: np.ndarray  # (24, n_vertices)
    weights: np.ndarray  # (n_vertices, 24)
    parents: np.ndarray  # (24,), root points to itself

    @classmethod
    def from_mat(cls, path: str) -> "SMPLModel":
        data = sio.loadmat(path)
        shapedirs = data["shapedirs"]
        if shapedirs.dtype.names:  # chumpy export: struct with field "x"
            shapedirs = shapedirs[0, 0]["x"]
        parents = np.array(data["kintree_table"][0, :], dtype=int)
        parents[0] = 0
        return cls(
            v_template=np.asarray(data["v_template"], dtype=float),
            faces=np.asarray(data["f"], dtype=int),
            shapedirs=np.asarray(shapedirs, dtype=float),
            posedirs=np.asarray(data["posedirs"], dtype=float),
            J_regressor=np.asarray(data["J_regressor"].todense()),
            weights=np.asarray(data["weights"], dtype=float),
            parents=parents,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SMPLModel":
        """Build the model from the fields of an official SMPL file."""

        def array(value):
            value = getattr(value, "x", value)  # chumpy objects keep the array in "x"
            if hasattr(value, "toarray"):  # scipy sparse
                value = value.toarray()
            return np.asarray(value, dtype=float)

        parents = np.array(array(data["kintree_table"])[0, :], dtype=np.int64)
        parents[0] = 0
        return cls(
            v_template=array(data["v_template"]),
            faces=np.asarray(array(data["f"]), dtype=int),
            shapedirs=array(data["shapedirs"]),
            posedirs=array(data["posedirs"]),
            J_regressor=array(data["J_regressor"]),
            weights=array(data["weights"]),
            parents=parents,
        )

    @classmethod
    def from_pkl(cls, path: str) -> "SMPLModel":
        """Load an official SMPL ``.pkl`` (e.g. ``SMPL_MALE.pkl``, v1.0.0 / v1.1.0).

        These files were written with Python 2 and contain ``chumpy``
        objects; they are read without installing chumpy.
        """
        with open(path, "rb") as f:
            data = _ChumpyFreeUnpickler(f, encoding="latin1").load()
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str) -> "SMPLModel":
        """Load a SMPL model from ``.mat``, ``.pkl`` or ``.npz``."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".mat":
            return cls.from_mat(path)
        if ext == ".pkl":
            return cls.from_pkl(path)
        if ext == ".npz":
            return cls.from_dict(dict(np.load(path, allow_pickle=True)))
        raise ValueError(f"Unsupported SMPL model file: {path}")

    @property
    def n_betas(self) -> int:
        return self.shapedirs.shape[2]

    def forward(self, betas=None, pose=None, v_shaped=None, transl=None):
        """Vertices and joints of SMPL for given shape and pose.

        Parameters
        ----------
        betas : (n_betas,) shape coefficients (default: zeros, the template)
        pose : (24, 3) axis-angle rotation of each joint relative to its
            parent; row 0 is the global orientation (default: rest pose)
        v_shaped : (n_vertices, 3) body in the rest pose, used instead of
            ``v_template + shapedirs @ betas`` (e.g. a measured minimally
            clothed body shape, as in CAPE)
        transl : (3,) global translation

        Returns
        -------
        vertices : (n_vertices, 3)
        joints : (24, 3) ground-truth joint positions
        """
        betas = np.zeros(self.n_betas) if betas is None else np.asarray(betas, dtype=float)
        pose = np.zeros((24, 3)) if pose is None else np.asarray(pose, dtype=float).reshape(24, 3)

        if v_shaped is None:
            v_shaped = self.v_template + self.shapedirs[:, :, : len(betas)] @ betas
        J = self.J_regressor @ v_shaped
        rots = Rotation.from_rotvec(pose).as_matrix()
        pose_feature = (rots[1:] - np.eye(3)).ravel()
        v_posed = v_shaped + self.posedirs @ pose_feature

        G = np.zeros((24, 4, 4))
        for j in range(24):
            T = np.eye(4)
            T[:3, :3] = rots[j]
            T[:3, 3] = J[j] - (J[self.parents[j]] if j else 0)
            G[j] = T if j == 0 else G[self.parents[j]] @ T
        joints = G[:, :3, 3].copy()
        G[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)  # relative to the rest pose
        verts_h = np.c_[v_posed, np.ones(len(v_posed))]
        vertices = np.einsum("vj,jab,vb->va", self.weights, G, verts_h)[:, :3]
        if transl is not None:
            vertices = vertices + transl
            joints = joints + transl
        return vertices, joints

    def as_source(self, landmarks: np.ndarray = SMPL_LANDMARKS) -> SourceShape:
        """The rest-pose template as a transfer source."""
        vertices, joints = self.forward()
        return SourceShape(
            vertices=vertices,
            faces=self.faces,
            joints=joints,
            parents=self.parents,
            skinning_weights=self.weights,
            landmarks=np.asarray(landmarks, dtype=int),
        )
