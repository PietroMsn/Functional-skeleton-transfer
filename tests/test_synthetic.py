import os

import numpy as np
import pytest

from skeleton_transfer import load_smpl_source
from skeleton_transfer.shapes import SMPL_LANDMARKS
from skeleton_transfer.smpl import SMPLModel
from skeleton_transfer.spectral import compute_laplace_basis
from skeleton_transfer.synthetic import make_synthetic_target

from .conftest import DATA_DIR

SMPL = os.path.join(DATA_DIR, "basicmodel_m_lbs.mat")
pytestmark = pytest.mark.skipif(not os.path.exists(SMPL), reason="SMPL model not available")


@pytest.fixture(scope="module")
def model():
    return SMPLModel.from_mat(SMPL)


def test_rest_pose_matches_source_loader(model):
    source = load_smpl_source(SMPL)
    vertices, joints = model.forward()
    np.testing.assert_allclose(vertices, source.vertices, atol=1e-12)
    np.testing.assert_allclose(joints, source.joints, atol=1e-12)


def test_posed_joints_follow_regressor(model, rng):
    """Without pose blend shapes the SMPL regressor would recover the posed
    joints exactly; with them, it stays within a couple of centimetres."""
    vertices, joints = model.forward(rng.normal(size=10), rng.normal(0, 0.3, (24, 3)))
    assert np.linalg.norm(model.J_regressor @ vertices - joints, axis=1).max() < 0.03


@pytest.mark.parametrize("clothing", [(), ("shirt",), ("skirt",)])
def test_synthetic_target(model, clothing):
    target, joints = make_synthetic_target(model, np.random.default_rng(0), clothing=clothing, n_points=3000)
    assert target.faces is None and target.vertices.shape == (3000, 3) and joints.shape == (24, 3)
    # landmarks point to the same body points as on the source (up to clothing offsets)
    posed, _ = model.forward(*_params(model))
    assert np.linalg.norm(target.vertices[target.landmarks] - posed[SMPL_LANDMARKS], axis=1).max() < 0.1
    # the spectral basis sees a single connected shape (one zero eigenvalue)
    basis = compute_laplace_basis(target.vertices, None, 5, is_point_cloud=True)
    assert basis.evals[1] > 1e-3


def _params(model):
    rng = np.random.default_rng(0)
    betas = rng.normal(0, 1.0, model.n_betas)
    pose = rng.normal(0, 0.3, (24, 3))
    pose[0] = 0
    return betas, pose


def test_iterations_stop_on_segment_consistency(model):
    """The refinement stops when the segment consistency stops improving."""
    from skeleton_transfer import IterativeConfig, TransferConfig, transfer_skeleton

    target, _ = make_synthetic_target(model, np.random.default_rng(2), clothing=("skirt",), n_points=3000)
    cfg = IterativeConfig(n_iterations=8, stop_patience=2, stop_min_gain=0.002, tolerance=0.0)
    result = transfer_skeleton(model.as_source(), target, TransferConfig(iterative=cfg))
    consistency = [h["segment_consistency"] for h in result.history]
    iterations = [h for h in result.history if h["stage"].startswith("iteration")]

    # replay the rule on the recorded history: it must stop exactly where the pipeline did
    best, stalled, expected = consistency[-len(iterations) - 1], 0, len(iterations)
    for i, h in enumerate(iterations, start=1):
        if h["segment_consistency"] > best + cfg.stop_min_gain:
            best, stalled = h["segment_consistency"], 0
        else:
            stalled += 1
        if stalled >= cfg.stop_patience:
            expected = i
            break
    assert len(iterations) == expected <= cfg.n_iterations
    assert result.diagnostics == {k: v for k, v in result.history[-1].items() if k in result.diagnostics}
