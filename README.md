# Functional-skeleton-transfer

**[Project page](https://pietromsn.github.io/Functional-skeleton-transfer/)**: abstract, figures, papers and BibTeX.

Musoni, P. and Marin, R. and Melzi, S. and Castellani, U., A Functional Skeleton Transfer, PACMCGIT 2021

<p align="center">
<img src="teaser.png">
</p>

## Contents
* [Quick-start](#quick-start)
* [How it works](#how-it-works)
* [Code structure](#code-structure)
* [Using it from Python](#using-it-from-python)
* [Differences from the original code](#differences-from-the-original-code)
* [Requirements and tests](#requirements-and-tests)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Quick-start

1. Install (Python 3.9 or newer; a virtual environment is recommended):

   ```bash
   git clone https://github.com/PietroMsn/Functional-skeleton-transfer.git
   cd Functional-skeleton-transfer
   pip install -e ".[viz]"
   ```

2. Download the SMPL model. It is not distributed with this code, because of
   its license: register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de),
   download **SMPL for Python** (version 1.1.0) and copy
   `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` (in the `models` folder of the
   archive) into the `data/` folder. (The `basicmodel_m_lbs.mat` file of the first
   release is a `.mat` export of the same model and still works.)

3. Run:

   ```bash
   skeleton-transfer          # or: python main.py
   ```

This transfers the skeleton of the SMPL template (source) onto the remeshed
Dynamic FAUST shape of the paper (target, in `data/`), in about a minute, and
opens a 3D viewer with the source skeleton and the two transferred skeletons.
Pass `--source PATH` to use a SMPL model stored elsewhere.

Useful options (`python main.py --help` for the full list):

| option | effect |
| --- | --- |
| `--target PATH` | another target: a `.mat` file (fields `vertex`, `faces`, `landmarkRes`, 1-based) or a mesh (`.obj`, `.off`; `.ply` etc. with `trimesh`) |
| `--target-landmarks FILE` | text file with the target landmark indices (required for meshes); add `--one-based-landmarks` for MATLAB indices |
| `--target-point-cloud` | treat the target as a point cloud |
| `--output joints.npz` | save joints, functional map, point-to-point map and regressor |
| `--no-plot` | do not open the viewer |
| `--screenshot view.png` | save the view to an image instead of opening a window |
| `--plot-results joints.npz` | only plot results saved with `--output` (no computation) |
| `-q` | only print warnings and errors |

Sample data:
* SMPL model: http://smpl.is.tue.mpg.de
* Dynamic FAUST dataset: https://dfaust.is.tue.mpg.de/

## How it works

The code follows Algorithm 1 of the paper (Sec. 4.2):

| step | paper | code |
| --- | --- | --- |
| Laplace-Beltrami basis and Fourier coefficients | Sec. 3.1, Eq. 2 | `spectral.py` |
| Step 1: spatial regressor `R` (reconstruction, locality, sparsity, convexity) | Eqs. 9-13 | `regressor.py` |
| Step 2: functional map `C` (WKS + landmark WKM, ICP, ZoomOut 20 → 120) | Sec. 3.2, 4.2 | `descriptors.py`, `functional_map.py` |
| Step 3: spectral regressor `R Φ` | Eq. 8 | `transfer.py` |
| Step 4: transferred skeleton `R Φ C Φ⁺ X` | Eq. 14 | `transfer.py` |

1. **Joint regressor on the source.** A linear regressor `R` (joints = `R @ vertices`)
   is fitted on the source. Each joint may only use the vertices skinned to both
   bones it connects, and each row of `R` sums to one. The energy is quadratic and
   is minimized exactly, in closed form.
2. **Correspondence.** A functional map between the Laplace-Beltrami eigenbases of
   the two shapes is estimated from WKS descriptors plus wave-kernel maps of 7
   landmarks, then refined with spectral ICP and ZoomOut.
3. **Transfer.** The regressor is moved onto the target through the functional
   map (*spectral transfer*, the method of the paper) and, for comparison,
   through the induced point-to-point map (*pointwise transfer*).

## Code structure

```
pyproject.toml               package metadata (pip install -e .)
main.py                      backward-compatible entry point
skeleton_transfer/
    cli.py                   command line interface (skeleton-transfer)
    config.py                all parameters, grouped by stage (dataclasses)
    shapes.py                SourceShape / TargetShape, .mat / .obj / .off loading
    spectral.py              Laplace-Beltrami eigenbasis
    descriptors.py           Wave Kernel Signature / Wave Kernel Map
    functional_map.py        functional map: optimization, ICP, ZoomOut
    regressor.py             localized joint regressor
    transfer.py              end-to-end pipeline (transfer_skeleton)
    visualization.py         PyVista viewer (optional)
data/                        example target (DFAUST); put the SMPL model here
tests/                       unit tests + comparison with the original code
tools/make_original_reference.py   runs the 2021 code to produce that reference
```

Every numeric parameter lives in `skeleton_transfer/config.py`, with a comment
explaining its role; the defaults are the ones used in the paper.

## Using it from Python

```python
from skeleton_transfer import TransferConfig, load_smpl_source, load_target, transfer_skeleton

source = load_smpl_source()   # the SMPL model in data/, or load_smpl_source("path/to/model.pkl")
target = load_target("data/50007_knees_001806.mat")
# or any mesh with its landmarks (same order as the source ones):
# target = load_target("scan.obj", landmarks="scan_landmarks.txt")

config = TransferConfig()
config.functional_map.n_eigs_final = 80   # any parameter can be changed

result = transfer_skeleton(source, target, config)
result.joints_spectral    # (24, 3) joints via the functional map
result.joints_pointwise   # (24, 3) joints via the point-to-point map
```

The source regressor and Laplace basis only depend on the source, so when
processing many targets you can compute them once and pass them to
`transfer_skeleton(..., regressor=..., basis_src=...)`.

To use a different template or target format, build a `SourceShape` /
`TargetShape` directly from numpy arrays (0-based indices).

## Differences from the original code

The code released with the paper in 2021 is preserved at commit `a857c61`.
The current version computes the same pipeline with these differences:

* **No TensorFlow.** Both energies optimized with Adam in the original code
  (the joint regressor and the initial functional map) are quadratic, and are
  now solved exactly in closed form. This is faster and deterministic.
* **Better regressor.** 1000 Adam iterations did not converge: the original
  regressor reconstructs the SMPL joints with errors up to 2.6 cm, the exact one
  reproduces them exactly. On SMPL posed with random poses (rotations of
  0.4 rad std per joint) the mean joint error drops from 15.2 mm to 4.6 mm.
  Consequently, the transferred joints differ from the original ones (by up to
  5 cm on the example).
* **Same functional map.** Given the same regressor, the new functional map
  pipeline gives the same joints as the original one within 1.2 mm.
* **No double rescaling of the target.** The original code rescaled the
  target to the source area twice; results are now returned in the input
  frame of the target.
* The locality and sparsity terms of Eq. 13 coincide for a binary mask and
  are merged into a single term (weight 1000 + 100).
* Exact nearest neighbours replace FLANN (an approximate search), and the
  eigendecomposition uses a fixed starting vector, so runs are reproducible.
* **Faster.** The transfer on the example takes less than a minute on an
  8-core desktop, against about 3 minutes for the original code (the
  nearest-neighbour searches in the spectral embeddings use matrix products
  instead of a k-d tree, with the same results).
* The SMPL model is read from the official `.pkl` file (without installing
  chumpy) and is no longer shipped in `data/`.

## Requirements and tests

The only required packages are NumPy, SciPy and
[robust_laplacian](https://github.com/nmwsharp/robust-laplacians-py);
PyVista is needed for the viewer and trimesh for mesh formats other than
OBJ/OFF. Tested on Windows with Python 3.13 and 3.14 (numpy 2.5, scipy 1.18,
robust_laplacian 1.1, pyvista 0.49).

```bash
pip install -e ".[all]"
pytest
```

The tests comparing with the original code need the SMPL model and the
example target in `data/` (or in the folder pointed by
`SKELETON_TRANSFER_DATA`) and take about a minute; they are skipped if the
files are missing. The reference outputs in `tests/data/` can be regenerated
with `tools/make_original_reference.py`, which runs the original code and
needs TensorFlow.

## License
Please check the license terms (also of third parts software) before downloading and/or using the code, the models and the data.
All code and results obtained from it not already covered by other licenses has to be intendend only for non-commercial scientific research purposes.
Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes including, for example, 3D models, movies, or video games.

## Acknowledgements

RM and SM are supported by the ERC Starting Grant No. 802554 (SPECGEO). This work is partially supported by the project of the Italian Ministry of Education, Universities and Research (MIUR) ”Dipartimenti di Eccellenza 2018-2022” of the Department of Computer Science of Sapienza University and the Department of Computer Science of Verona University

If you use this code in your paper, please cite:
```
@article{functionalskeletontransfer,
  title={A functional skeleton transfer},
  author={Musoni, Pietro and Marin, Riccardo and Melzi, Simone and Castellani, Umberto},
  journal={Proceedings of the ACM on Computer Graphics and Interactive Techniques},
  volume={4},
  number={3},
  pages={1--15},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```
