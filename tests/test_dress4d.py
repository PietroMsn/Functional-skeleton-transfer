"""The 4D-DRESS exporter on a tiny archive with the same layout as the real one."""

import importlib.util
import io
import os
import pickle
import tarfile

import numpy as np
import pytest

from skeleton_transfer.scans import load_sample
from skeleton_transfer.smpl import SMPLModel

from .conftest import DATA_DIR, REPO_ROOT

SMPL = os.path.join(DATA_DIR, "basicmodel_m_lbs.mat")
pytestmark = pytest.mark.skipif(not os.path.exists(SMPL), reason="SMPL model not available")


def _load_exporter_module(name="dress4d"):
    path = os.path.join(REPO_ROOT, "tools", "datasets", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add(tar, name, obj):
    data = obj if isinstance(obj, bytes) else pickle.dumps(obj)
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_export_from_archive(tmp_path):
    dress4d = _load_exporter_module()
    model = SMPLModel.from_file(SMPL)
    rng = np.random.default_rng(0)
    take = "00123_Outer/Outer/Take1"
    params = {}
    archive = tmp_path / "00123_Outer.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for frame in range(1, 6):
            p = {"global_orient": rng.normal(0, 0.2, 3), "body_pose": rng.normal(0, 0.2, 69),
                 "transl": rng.normal(size=3), "betas": rng.normal(size=10)}  # fmt: skip
            params[frame] = p
            body, _ = model.forward(p["betas"], np.concatenate((p["global_orient"], p["body_pose"])).reshape(24, 3), transl=p["transl"])
            _add(tar, f"{take}/Capture/0004/images/capture-f{frame:05d}.png", b"\x89PNG")
            _add(tar, f"{take}/Meshes_pkl/atlas-f{frame:05d}.pkl", np.zeros((4, 4, 3)))
            _add(tar, f"{take}/Meshes_pkl/mesh-f{frame:05d}.pkl", {"vertices": body.astype(np.float32), "faces": model.faces})
            _add(tar, f"{take}/SMPL/mesh-f{frame:05d}_smpl.pkl", p)
            _add(tar, f"{take}/Semantic/labels/label-f{frame:05d}.pkl", {"scan_labels": np.full(len(body), 5.0)})
        _add(tar, f"{take}/basic_info.pkl", {"gender": "female", "rotation": np.eye(3), "scan_frames": []})

    out = tmp_path / "samples"
    exporter = dress4d.Exporter({"neutral": model}, str(out), stride=2, frames_per_take=2, require_labels=True)
    dress4d.read_archive(str(archive), exporter)
    exporter.finish()

    files = sorted(os.listdir(out))
    assert files == ["00123_Outer_Take1_f00001.npz", "00123_Outer_Take1_f00003.npz"]
    sample = load_sample(str(out / files[1]))
    assert sample.category == "Outer" and sample.gender == "female"
    assert sample.metadata["label_fractions"]["outer"] == 1.0
    assert sample.metadata["source"] == "00123_Outer.tar.gz" and sample.metadata["take"] == "Take1"
    p = params[3]
    _, joints = model.forward(p["betas"], np.concatenate((p["global_orient"], p["body_pose"])).reshape(24, 3), transl=p["transl"])
    np.testing.assert_allclose(sample.joints, joints, atol=1e-5)
    np.testing.assert_allclose(sample.smpl_vertices, sample.scan_vertices, atol=1e-5)


def test_manifest_records_provenance(tmp_path):
    test_export_from_archive(tmp_path)
    manifest = _load_exporter_module("manifest")
    rows = manifest.manifest_rows(str(tmp_path / "samples"))
    assert [(r["source"], r["subject"], r["outfit"], r["take"], r["frame"]) for r in rows] == [
        ("00123_Outer.tar.gz", "00123", "Outer", "Take1", 1),
        ("00123_Outer.tar.gz", "00123", "Outer", "Take1", 3),
    ]


def test_select_frames_one_per_sequence_with_varied_poses():
    select = _load_exporter_module("select_frames").select_frames
    poses = {f: np.full(3, float(f)) for f in range(5)}
    samples = {("s", "Inner", "Take1"): [(f, poses[f]) for f in range(5)],
               ("s", "Inner", "Take2"): [(f + 10, poses[f]) for f in range(5)]}  # fmt: skip
    # with nothing selected, the first sequence gives its middle frame, the next one the farthest pose
    assert select(samples, []) == [2, 10]
    # kept sequences are skipped, and their poses count as selected
    assert select(samples, [(("s", "Inner", "Take1"), "kept", poses[0])]) == [14]
