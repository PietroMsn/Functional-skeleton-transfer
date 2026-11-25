"""Loading of the SMPL model and exact nearest neighbours."""

import pickle

import numpy as np
import pytest
from scipy import sparse
from scipy.spatial import cKDTree

from skeleton_transfer.functional_map import nearest_neighbors
from skeleton_transfer.shapes import find_smpl_model, load_smpl_source


@pytest.mark.parametrize("dim", [3, 20, 120])
def test_nearest_neighbors_is_exact(rng, dim):
    points, queries = rng.normal(size=(3000, dim)), rng.normal(size=(500, dim))
    expected = cKDTree(points).query(queries)[1]
    np.testing.assert_array_equal(nearest_neighbors(points, queries, chunk_size=128), expected)


def _fake_smpl(path, extra=None):
    """A tiny file with the fields of the official SMPL .pkl."""
    data = {
        "v_template": np.eye(3),
        "f": np.array([[0, 1, 2]]),
        "J_regressor": sparse.csc_matrix(np.full((2, 3), 1 / 3)),
        "weights": np.full((3, 2), 0.5),
        "kintree_table": np.array([[4294967295, 0], [0, 1]]),
    }
    data.update(extra or {})
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_load_smpl_pkl(tmp_path):
    path = tmp_path / "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
    _fake_smpl(path)
    assert find_smpl_model(str(tmp_path)) == str(path)
    source = load_smpl_source(str(path), landmarks=[0, 1])
    np.testing.assert_allclose(source.joints, np.full((2, 3), 1 / 3))
    np.testing.assert_array_equal(source.parents, [0, 0])


class Unexpected:
    pass


def test_smpl_pkl_refuses_other_classes(tmp_path):
    path = tmp_path / "SMPL_NEUTRAL.pkl"
    _fake_smpl(path, {"payload": Unexpected()})
    with pytest.raises(pickle.UnpicklingError):
        load_smpl_source(str(path))


def test_missing_smpl_model_explains_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="smpl.is.tue.mpg.de"):
        find_smpl_model(str(tmp_path))
