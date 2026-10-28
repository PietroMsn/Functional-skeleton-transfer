import os

import numpy as np
import pytest

from skeleton_transfer.evaluation import ALL_LANDMARKS
from skeleton_transfer.smpl import SMPLModel
from skeleton_transfer.smpl_fitting import FittingConfig, fit_smpl

from .conftest import DATA_DIR

SMPL = os.path.join(DATA_DIR, "basicmodel_m_lbs.mat")


@pytest.mark.skipif(not os.path.exists(SMPL), reason="SMPL model not available")
def test_fit_recovers_posed_body(rng):
    """On an unclothed SMPL body, the fit recovers the joints within 2 cm."""
    model = SMPLModel.from_file(SMPL)
    pose = np.zeros((24, 3))
    pose[0] = [0, 0.8, 0]  # turned around the vertical axis
    pose[1:] = rng.normal(scale=0.15, size=(23, 3))
    betas = rng.normal(scale=1.0, size=model.n_betas)
    vertices, joints = model.forward(betas, pose, transl=np.array([0.3, 0.1, -0.2]))
    points = vertices[rng.choice(len(vertices), 4000, replace=False)]
    landmarks = np.argmin(np.linalg.norm(points[None] - vertices[ALL_LANDMARKS, None], axis=2), axis=1)

    fit = fit_smpl(model, points, ALL_LANDMARKS, landmarks, np.array([0, 1, 2, 4, 5, 7, 8]),
                   FittingConfig(n_body_vertices=1500, n_icp=4))  # fmt: skip
    assert np.linalg.norm(fit.joints - joints, axis=1).mean() < 0.02
