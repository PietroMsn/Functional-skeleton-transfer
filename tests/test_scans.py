import os
import pickle
import sys
import types

import numpy as np
import pytest
import scipy.sparse as sparse

from skeleton_transfer.evaluation import ALL_LANDMARKS, Evaluator, completed_targets, default_variants
from skeleton_transfer.scans import ScanSample, load_sample, make_target, sample_surface, save_sample, transfer_landmarks
from skeleton_transfer.smpl import SMPLModel
from skeleton_transfer.synthetic import add_shirt

from .conftest import DATA_DIR

SMPL = os.path.join(DATA_DIR, "basicmodel_m_lbs.mat")
needs_smpl = pytest.mark.skipif(not os.path.exists(SMPL), reason="SMPL model not available")


def test_smpl_pkl_without_chumpy(tmp_path, rng):
    """Official SMPL .pkl files store chumpy objects; they load without chumpy."""
    chumpy = types.ModuleType("chumpy")
    chumpy_ch = types.ModuleType("chumpy.ch")

    class Ch:  # what chumpy pickles: an object whose state holds the array in "x"
        def __init__(self, x):
            self.x = x

    Ch.__module__, Ch.__qualname__ = "chumpy.ch", "Ch"
    chumpy_ch.Ch = Ch
    sys.modules.update({"chumpy": chumpy, "chumpy.ch": chumpy_ch})
    n = 7
    data = {
        "v_template": Ch(rng.normal(size=(n, 3))),
        "f": np.array([[0, 1, 2]], dtype=np.uint32),
        "shapedirs": Ch(rng.normal(size=(n, 3, 10))),
        "posedirs": rng.normal(size=(n, 3, 207)),
        "J_regressor": sparse.csc_matrix(rng.random((24, n))),
        "weights": Ch(rng.random((n, 24))),
        "kintree_table": np.vstack((np.arange(24) - 1, np.arange(24))).astype(np.int64) % 2**32,
    }
    try:
        with open(tmp_path / "smpl.pkl", "wb") as f:
            pickle.dump(data, f, protocol=2)
    finally:
        del sys.modules["chumpy"], sys.modules["chumpy.ch"]

    model = SMPLModel.from_file(str(tmp_path / "smpl.pkl"))
    np.testing.assert_allclose(model.v_template, data["v_template"].x)
    np.testing.assert_allclose(model.shapedirs, data["shapedirs"].x)
    np.testing.assert_allclose(model.J_regressor, data["J_regressor"].toarray())
    assert model.parents[0] == 0 and model.parents[5] == 4 and model.n_betas == 10


def test_sample_surface(rng):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [3, 3, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 4]])  # areas 0.5 and 3
    points, face_index = sample_surface(vertices, faces, 20000, rng)
    assert abs(np.mean(face_index == 0) - 0.5 / 3.5) < 0.01
    first = points[face_index == 0]
    assert np.all(first[:, :2] >= -1e-12) and np.all(first[:, 0] + first[:, 1] <= 1 + 1e-12)
    np.testing.assert_allclose(points[:, 2], 0)


def test_transfer_landmarks():
    body = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    points = np.array([[0.9, 0.1, 0.05], [0, 0, 0.1], [0.1, 0.9, 0.0]])
    np.testing.assert_array_equal(transfer_landmarks(body, np.array([2, 0, 1]), points), [2, 1, 0])


def test_sample_roundtrip(tmp_path, rng):
    sample = ScanSample(
        name="cape/00032/longshort/f001",
        scan_vertices=rng.normal(size=(50, 3)),
        scan_faces=rng.integers(0, 50, (20, 3)),
        smpl_vertices=rng.normal(size=(6890, 3)),
        gender="female",
        category="long-short",
        scan_labels=rng.integers(0, 4, 50),
        joints=rng.normal(size=(24, 3)),
        metadata={"frame": 1},
    )
    save_sample(str(tmp_path / "s.npz"), sample)
    loaded = load_sample(str(tmp_path / "s.npz"))
    assert (loaded.name, loaded.gender, loaded.category, loaded.metadata) == (
        sample.name, sample.gender, sample.category, sample.metadata,
    )  # fmt: skip
    np.testing.assert_allclose(loaded.scan_vertices, sample.scan_vertices, atol=1e-6)
    np.testing.assert_array_equal(loaded.scan_faces, sample.scan_faces)
    np.testing.assert_allclose(loaded.joints, sample.joints, atol=1e-6)


@pytest.fixture(scope="module")
def dressed_sample():
    """A 'scan' made from a posed SMPL body with a synthetic loose shirt."""
    model = SMPLModel.from_file(SMPL)
    rng = np.random.default_rng(1)
    pose = rng.normal(0, 0.3, (24, 3))
    pose[0] = 0
    body, joints = model.forward(rng.normal(size=10), pose, transl=np.array([0.3, 0.9, -0.2]))
    scan = add_shirt(body, model.faces, model.weights, rng)
    scan = np.vstack((scan, [[5.0, 5.0, 5.0]]))  # a capture artefact far from the body
    sample = ScanSample("synthetic/0", scan, model.faces, body, gender="male", category="shirt")
    return model, sample, joints


@needs_smpl
def test_make_target(dressed_sample):
    model, sample, fk_joints = dressed_sample
    target, gt = make_target(sample, model.J_regressor, n_points=6000, landmarks=ALL_LANDMARKS)
    assert target.faces is None and len(target.vertices) == 6000  # the artefact is not on any face
    # regressed ground truth is close to the forward-kinematics joints
    assert np.linalg.norm(gt - fk_joints, axis=1).max() < 0.03
    # landmarks land on the scan near the corresponding body vertices
    assert np.linalg.norm(target.vertices[target.landmarks] - sample.smpl_vertices[ALL_LANDMARKS], axis=1).max() < 0.1

    no_faces = ScanSample("pc", sample.scan_vertices, None, sample.smpl_vertices)
    target, _ = make_target(no_faces, model.J_regressor, n_points=None)
    assert len(target.vertices) == len(sample.scan_vertices) - 1  # artefact dropped by max_distance


@needs_smpl
def test_evaluator_on_scan(dressed_sample):
    model, sample, fk_joints = dressed_sample
    evaluator = Evaluator(model.as_source(), variants=default_variants()[:1])  # baseline only: fast
    target, gt = make_target(sample, model.J_regressor, n_points=5000, landmarks=ALL_LANDMARKS)
    evaluator.evaluate(target, gt, "shirt", sample.name)
    (run,) = evaluator.runs
    assert run["variant"] == "baseline" and run["error"] < 0.1 and run["history"][0]["error"] == run["error"]
    # a variant added later is the only one run again on the same target
    evaluator.variants = default_variants()[:2]
    assert completed_targets(evaluator.runs, evaluator.variants) == set()
    evaluator.evaluate(target, gt, "shirt", sample.name)
    assert [r["variant"] for r in evaluator.runs] == ["baseline", "+ knee/elbow landmarks"]
    assert completed_targets(evaluator.runs, evaluator.variants) == {("shirt", sample.name)}
