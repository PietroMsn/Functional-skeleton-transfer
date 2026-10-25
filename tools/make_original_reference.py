"""
Run the ORIGINAL 2021 implementation and save its outputs as a reference.

The original scripts are extracted untouched from git (commit ``a857c61``,
the version released with the paper) into a temporary folder, and executed
with the same steps as the original ``main.py`` (minus the viewer). Only two
compatibility shims are applied, both outside the original files:

- ``pyflann`` (unmaintained, no wheels for recent Python) is replaced by an
  exact k-d tree search with the same ``FLANN().nn`` interface. FLANN's
  default search is approximate, so tiny differences are possible;
- ``eigsh`` receives ``maxiter`` as an int (recent SciPy rejects ``1e9``).

Requires TensorFlow (``pip install tensorflow-cpu``) in addition to the
package dependencies. Usage, from the repository root::

    python tools/make_original_reference.py --out tests/data/original_reference.npz

The SMPL / DFAUST example files are read from ``data/`` if present, otherwise
from the same git commit.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import types

import numpy as np

ORIGINAL_COMMIT = "a857c61"
ORIGINAL_FILES = [
    "calc_regressor.py",
    "functional_maps.py",
    "spatial_regressor.py",
    "goodname/__init__.py",
    "goodname/WKM.py",
]
DATA_FILES = ["data/basicmodel_m_lbs.mat", "data/50007_knees_001806.mat"]


def git_show(repo, path, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if path == "goodname/__init__.py":  # namespace package in the original
        open(dest, "w").close()
        return
    blob = subprocess.run(
        ["git", "-C", repo, "show", f"{ORIGINAL_COMMIT}:{path}"], check=True, capture_output=True
    ).stdout
    with open(dest, "wb") as f:
        f.write(blob)


def install_pyflann_shim():
    from scipy.spatial import cKDTree

    class FLANN:
        def nn(self, dataset, testset, num_neighbors=1):
            dists, idx = cKDTree(dataset).query(testset, k=num_neighbors)
            return idx, dists ** 2

    shim = types.ModuleType("pyflann")
    shim.FLANN = FLANN
    shim.__all__ = ["FLANN"]
    sys.modules["pyflann"] = shim


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="tests/data/original_reference.npz")
    args = parser.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workdir = tempfile.mkdtemp(prefix="fst_original_")
    for path in ORIGINAL_FILES:
        git_show(repo, path, os.path.join(workdir, path))
    data_paths = []
    for path in DATA_FILES:
        local = os.path.join(repo, path)
        if not os.path.exists(local):
            local = os.path.join(workdir, path)
            git_show(repo, path, local)
        data_paths.append(local)

    install_pyflann_shim()
    sys.path.insert(0, workdir)

    import scipy.io as sio
    import scipy.sparse.linalg

    import functional_maps

    def eigsh_int_maxiter(*a, **kw):
        if "maxiter" in kw:
            kw["maxiter"] = int(kw["maxiter"])
        return scipy.sparse.linalg.eigsh(*a, **kw)

    functional_maps.eigsh = eigsh_int_maxiter
    from calc_regressor import get_joints

    # ---- same steps as the original main.py (lines 13-72) ----
    params = {"isPC1": False, "isPC2": False, "kZO1": 20, "kZO2": 20}
    k1, k2 = 120, 120
    source = sio.loadmat(data_paths[0])
    target = sio.loadmat(data_paths[1])
    parents = np.transpose(source["kintree_table"][0, :])
    parents[0] = 0
    landSrc = (np.asarray([3509, 3076, 414, 5909, 6683, 3216, 2444], dtype=int) - 1).T
    fSrcArray = np.asarray(source["f"], dtype=int)
    vSrc = np.asarray(source["v_template"])
    J = np.matmul(np.asarray(source["J_regressor"].todense()), vSrc)
    weights = source["weights"]
    vTar_input = np.asarray(target["vertex"], dtype=float)
    vTar = vTar_input
    fTarArray = np.asarray(target["faces"], dtype=int) - 1
    landTar = np.squeeze(np.asarray(target["landmarkRes"], dtype=int) - 1)

    evalsX, phiX, MX, pinv_phiX = functional_maps.get_coeff(vSrc, fSrcArray, 400, False)
    evalsY, phiY, MY, pinv_phiY = functional_maps.get_coeff(vTar, fTarArray, 400, False)
    scale = np.sqrt(MX.sum() / MY.sum())
    vTar = np.multiply(scale, vTar)
    L1 = {"evalsX": evalsX, "phiX": phiX, "MX": MX, "pinv_phiX": pinv_phiX}
    L2 = {"evalsY": evalsY, "phiY": phiY, "MY": MY, "pinv_phiY": pinv_phiY}

    (R_spatial, R_hat, _, _, _, _, _, Tar_hat, vTar, C, F, matches) = get_joints(
        J, vSrc, vTar, fSrcArray, fTarArray, weights, parents, k1, k2, landSrc, landTar, params, L1, L2
    )
    joints_spectral = R_hat @ np.array(C) @ Tar_hat
    joints_pointwise = R_spatial @ np.squeeze(vTar[matches, :])

    # The original rescales the target twice (main.py and get_joints), so its
    # joints live in a frame scaled by `scale**2`. Store them also in the
    # frame of the input target, which is what the new code returns.
    total_scale = scale ** 2
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        evals_src=evalsX[:120],
        evals_tgt=evalsY[:120],
        regressor=R_spatial.astype(np.float32),
        mask=F.astype(bool),
        C=np.asarray(C),
        p2p=np.squeeze(matches).astype(np.int32),
        joints_spectral=joints_spectral / total_scale,
        joints_pointwise=joints_pointwise / total_scale,
        target_scale=total_scale,
    )
    print(f"Saved reference to {args.out} (target scale factor applied twice: {total_scale:.6f})")


if __name__ == "__main__":
    main()
