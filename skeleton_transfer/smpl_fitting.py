"""
Baseline: fit the SMPL body model to a (clothed) target and read its joints.

When the target is a human and the template is SMPL, the direct alternative
to transferring the skeleton is to register the parametric body to the scan
and take the joints of the fitted model. :func:`fit_smpl` implements a
standard optimization-based registration for clothed scans, with the same
inputs as the transfer (a point cloud and its landmarks):

1. rigid alignment of the template to the landmarks of the torso and legs;
2. pose fitting to all the landmarks, with a strong and then a weaker pose
   prior;
3. ICP-style fitting of pose, shape and translation to the surface:

   - scan -> body: every target point is pulled to its nearest body vertex,
     with robust (Geman-McClure) weights, so that loose garments far from
     the body have little influence;
   - body -> scan: every body vertex is pulled to its nearest target point,
     strongly if it lies outside the scan (the body must stay under the
     clothes) and weakly if it lies inside;
   - landmarks, and L2 priors on the pose and the shape.

The residuals use a fixed subset of the body vertices; correspondences and
robust weights are recomputed at every ICP iteration.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .smpl import SMPLModel
from .synthetic import vertex_normals


@dataclass
class FittingConfig:
    """Weights and schedule of :func:`fit_smpl` (distances in metres, angles in radians)."""

    n_betas: int = 10
    n_body_vertices: int = 3000  # body vertices used in the residuals
    landmark_weight: float = 1.0
    scan_to_body_weight: float = 1.0
    body_to_scan_weight: float = 1.0
    inside_weight: float = 0.1  # relative weight of body vertices inside the scan
    robust_scale: float = 0.03  # Geman-McClure scale of the scan -> body distances
    pose_prior: tuple = (1e-3, 1e-4)  # landmark stages
    surface_pose_prior: float = 1e-5
    shape_prior: float = 1e-5
    point_to_plane: bool = True
    n_icp: int = 8
    max_nfev: int = 30  # per least-squares solve


@dataclass
class FittingResult:
    joints: np.ndarray  # (24, 3) joints of the fitted body
    vertices: np.ndarray  # (6890, 3) fitted body
    pose: np.ndarray  # (24, 3) axis-angle
    betas: np.ndarray
    transl: np.ndarray
    scan_to_body: float  # mean distance of the target points to the fitted body


class _SubsetSMPL:
    """SMPL forward kinematics restricted to a subset of the vertices."""

    def __init__(self, model: SMPLModel, ids: np.ndarray, n_betas: int):
        self.model, self.ids = model, ids
        self.v_template = model.v_template[ids]
        self.shapedirs = model.shapedirs[ids, :, :n_betas]
        self.posedirs = model.posedirs[ids]
        self.weights = model.weights[ids]
        # joints are linear in the shape coefficients
        self.J_template = model.J_regressor @ model.v_template
        self.J_shapedirs = np.einsum("jv,vab->jab", model.J_regressor, model.shapedirs[:, :, :n_betas])

    def __call__(self, pose, betas, transl):
        rots = Rotation.from_rotvec(pose).as_matrix()
        J = self.J_template + self.J_shapedirs @ betas
        v = self.v_template + self.shapedirs @ betas + self.posedirs @ (rots[1:] - np.eye(3)).ravel()
        parents = self.model.parents
        G = np.zeros((24, 4, 4))
        for j in range(24):
            T = np.eye(4)
            T[:3, :3] = rots[j]
            T[:3, 3] = J[j] - (J[parents[j]] if j else 0)
            G[j] = T if j == 0 else G[parents[j]] @ T
        joints = G[:, :3, 3].copy()
        G[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)
        A = np.einsum("vj,jab->vab", self.weights, G[:, :3])
        return np.einsum("vab,vb->va", A[:, :, :3], v) + A[:, :, 3] + transl, joints + transl


def _kabsch(src: np.ndarray, dst: np.ndarray):
    """Rotation R and translation t minimizing |R src + t - dst|."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def fit_smpl(
    model: SMPLModel,
    points: np.ndarray,
    landmark_vertices: np.ndarray,
    landmark_points: np.ndarray,
    alignment_landmarks: np.ndarray,
    config: FittingConfig = FittingConfig(),
    rng: np.random.Generator = None,
) -> FittingResult:
    """Fit SMPL to a target point cloud.

    Parameters
    ----------
    model : SMPL model to fit (e.g. the neutral one)
    points : (n, 3) target point cloud
    landmark_vertices : (l,) SMPL vertices of the landmarks
    landmark_points : (l,) indices of the corresponding target points
    alignment_landmarks : indices (into the landmarks) used for the initial
        rigid alignment: landmarks whose position does not depend much on the
        pose (torso, head, feet)
    """
    cfg = config
    rng = rng or np.random.default_rng(0)
    ids = np.sort(rng.choice(len(model.v_template), cfg.n_body_vertices, replace=False))
    ids = np.union1d(ids, landmark_vertices)
    lm_rows = np.searchsorted(ids, landmark_vertices)
    body = _SubsetSMPL(model, ids, cfg.n_betas)
    lm_target = points[landmark_points]
    tree = cKDTree(points)

    # 1) rigid alignment of the rest pose
    rest, _ = body(np.zeros((24, 3)), np.zeros(cfg.n_betas), np.zeros(3))
    R, t = _kabsch(rest[lm_rows[alignment_landmarks]], lm_target[alignment_landmarks])
    J0 = body.J_template[0]
    pose = np.zeros((24, 3))
    pose[0] = Rotation.from_matrix(R).as_rotvec()
    betas, transl = np.zeros(cfg.n_betas), t + R @ J0 - J0

    def unpack(x):
        return x[:72].reshape(24, 3), x[72 : 72 + cfg.n_betas], x[72 + cfg.n_betas :]

    def solve(x, residuals):
        return least_squares(residuals, x, method="trf", max_nfev=cfg.max_nfev, x_scale="jac").x

    n_lm = len(landmark_vertices)

    # 2) pose from the landmarks (shape fixed)
    for prior in cfg.pose_prior:

        def landmark_residuals(x, prior=prior):
            p, b, tr = unpack(np.r_[x[:72], betas, x[72:]])
            v, _ = body(p, b, tr)
            return np.r_[
                np.sqrt(cfg.landmark_weight / n_lm) * (v[lm_rows] - lm_target).ravel(),
                np.sqrt(prior) * p[1:].ravel(),
            ]

        x = solve(np.r_[pose.ravel(), transl], landmark_residuals)
        pose, transl = x[:72].reshape(24, 3), x[72:]

    # 3) ICP on the surface
    x = np.r_[pose.ravel(), betas, transl]
    for _ in range(cfg.n_icp):
        p, b, tr = unpack(x)
        v, _ = body(p, b, tr)
        normals = vertex_normals(_full_vertices(model, p, b, tr, cfg.n_betas), model.faces)[ids]
        # scan -> body, robust weights
        d, nn_body = cKDTree(v).query(points)
        w_s2b = np.sqrt(cfg.scan_to_body_weight / len(points)) * cfg.robust_scale**2 / (cfg.robust_scale**2 + d**2)
        # body -> scan, stronger outside the scan
        _, nn_scan = tree.query(v)
        q = points[nn_scan]
        outside = np.einsum("ij,ij->i", v - q, normals) > 0
        w_b2s = np.sqrt(cfg.body_to_scan_weight / len(v)) * np.where(outside, 1.0, np.sqrt(cfg.inside_weight))

        n_s2b = normals[nn_body]

        def surface_residuals(x):
            p, b, tr = unpack(x)
            v, _ = body(p, b, tr)
            if cfg.point_to_plane:  # distances along the body normals: no bias from the vertex spacing
                s2b = w_s2b * np.einsum("ij,ij->i", points - v[nn_body], n_s2b)
                b2s = w_b2s * np.einsum("ij,ij->i", v - q, normals)
            else:
                s2b = (w_s2b[:, None] * (points - v[nn_body])).ravel()
                b2s = (w_b2s[:, None] * (v - q)).ravel()
            return np.r_[
                np.sqrt(cfg.landmark_weight / n_lm) * (v[lm_rows] - lm_target).ravel(),
                s2b,
                b2s,
                np.sqrt(cfg.surface_pose_prior) * p[1:].ravel(),
                np.sqrt(cfg.shape_prior) * b,
            ]

        x = solve(x, surface_residuals)

    pose, betas, transl = unpack(x)
    vertices, joints = model.forward(np.r_[betas, np.zeros(model.n_betas - cfg.n_betas)], pose, transl=transl)
    return FittingResult(
        joints=joints,
        vertices=vertices,
        pose=pose,
        betas=betas,
        transl=transl,
        scan_to_body=float(cKDTree(vertices).query(points)[0].mean()),
    )


def _full_vertices(model, pose, betas, transl, n_betas):
    return model.forward(np.r_[betas, np.zeros(model.n_betas - n_betas)], pose, transl=transl)[0]
