import os
from dataclasses import replace

import numpy as np
import pytest

from skeleton_transfer.shapes import SMPL_LANDMARKS, TargetShape
from skeleton_transfer.smpl import SMPLModel
from skeleton_transfer.templates import proportion_cost, select_source

from .conftest import DATA_DIR

SMPL = os.path.join(DATA_DIR, "basicmodel_m_lbs.mat")
pytestmark = pytest.mark.skipif(not os.path.exists(SMPL), reason="SMPL model not available")


@pytest.fixture(scope="module")
def model():
    return SMPLModel.from_mat(SMPL)


def test_as_source_with_betas(model):
    betas = np.zeros(model.n_betas)
    betas[0] = 2.0
    source = model.as_source(SMPL_LANDMARKS, betas)
    vertices, joints = model.forward(betas)
    np.testing.assert_allclose(source.vertices, vertices)
    np.testing.assert_allclose(source.joints, joints)
    assert not np.allclose(source.vertices, model.as_source().vertices)


def test_template_with_the_closest_proportions_is_chosen(model, rng):
    """The landmark proportions ignore the pose and the size, so a posed and
    scaled copy of a template is matched to it rather than to a template
    with other proportions."""
    mean = model.as_source(SMPL_LANDMARKS)
    stretch = np.array([1.0, 1.6, 1.0])  # taller (y up), same width: other proportions
    sources = [mean, replace(mean, vertices=mean.vertices * stretch)]
    pose = rng.normal(0, 0.2, (24, 3))
    pose[0] = 0
    vertices, _ = model.forward(None, pose)
    target = TargetShape(vertices=1.7 * vertices, faces=model.faces, landmarks=SMPL_LANDMARKS)
    assert select_source(sources, target) == 0
    target = replace(target, vertices=model.forward(None, pose, v_shaped=sources[1].vertices)[0])
    assert select_source(sources, target) == 1
    same = TargetShape(vertices=2 * sources[0].vertices, faces=model.faces, landmarks=SMPL_LANDMARKS)
    assert proportion_cost(sources[0], same) < 1e-9
    assert proportion_cost(sources[1], same) > 0
    # without faces (point cloud) the cost is still small for the same shape
    cloud = replace(same, faces=None)
    assert proportion_cost(sources[0], cloud) < proportion_cost(sources[1], cloud)
