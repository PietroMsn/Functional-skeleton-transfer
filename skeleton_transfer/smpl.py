"""
Minimal SMPL model (Loper et al. 2015), NumPy only.

Used to generate shapes with known ground-truth joints (any body shape and
pose) for evaluation, see :mod:`skeleton_transfer.synthetic`. It reads the
``.mat`` export of the SMPL model that is also used as the transfer source.
"""

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

    @property
    def n_betas(self) -> int:
        return self.shapedirs.shape[2]

    def forward(self, betas=None, pose=None):
        """Vertices and joints of SMPL for given shape and pose.

        Parameters
        ----------
        betas : (n_betas,) shape coefficients (default: zeros, the template)
        pose : (24, 3) axis-angle rotation of each joint relative to its
            parent; row 0 is the global orientation (default: rest pose)

        Returns
        -------
        vertices : (n_vertices, 3)
        joints : (24, 3) ground-truth joint positions
        """
        betas = np.zeros(self.n_betas) if betas is None else np.asarray(betas, dtype=float)
        pose = np.zeros((24, 3)) if pose is None else np.asarray(pose, dtype=float).reshape(24, 3)

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
