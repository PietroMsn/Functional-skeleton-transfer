# Functional-skeleton-transfer

Musoni, P. and Marin, R. and Melzi, S. and Castellani, U., A Functional Skeleton Transfer, PACMCGIT 2021

<p align="center">
<img src="teaser.png">
</p>

## Contents
* [Quick-start](#quick-start)
* [How it works](#how-it-works)
* [Code structure](#code-structure)
* [Using it from Python](#using-it-from-python)
* [Skeleton priors and synthetic benchmark](#skeleton-priors-and-synthetic-benchmark)
* [Differences from the original code](#differences-from-the-original-code)
* [Requirements and tests](#requirements-and-tests)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Quick-start

```bash
pip install -e ".[viz]"
skeleton-transfer          # or: python main.py
```

This transfers the skeleton of the SMPL template (source) onto a remeshed
Dynamic FAUST shape (target), both stored in `data/`, and opens a 3D viewer
with the source skeleton and the two transferred skeletons.

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

1. **Joint regressor on the source.** A linear regressor `R` (joints = `R @ vertices`)
   is fitted on the source. Each joint may only use the vertices skinned to both
   bones it connects, and each row of `R` sums to one. The energy is quadratic and
   is minimized exactly, in closed form.
2. **Correspondence.** A functional map between the Laplace-Beltrami eigenbases of
   the two shapes is estimated from WKS descriptors plus wave-kernel maps of 7
   landmarks, then refined with spectral ICP and ZoomOut.
3. **Transfer.** The regressor is moved onto the target either through the
   functional map (*spectral transfer*) or through the induced point-to-point
   map (*pointwise transfer*).

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
data/                        example source (SMPL) and target (DFAUST)
tests/                       unit tests + comparison with the original code
tools/make_original_reference.py   runs the 2021 code to produce that reference
```

Every numeric parameter lives in `skeleton_transfer/config.py`, with a comment
explaining its role; the defaults are the ones used in the paper.

## Using it from Python

```python
from skeleton_transfer import TransferConfig, load_smpl_source, load_target, transfer_skeleton

source = load_smpl_source("data/basicmodel_m_lbs.mat")
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

## Skeleton priors and synthetic benchmark

`--priors` (or `TransferConfig(skeleton_priors=SkeletonPriorConfig())`) corrects
the transferred joints so that mirrored bones have the same length and the bone
proportions match the source up to a global scale
(`skeleton_transfer/skeleton_priors.py`). Independently of it, every transfer
checks the bone proportions and sets `result.plausible = False` (with a warning)
when some bone is shorter than 0.6 or longer than 1.6 times its expected length,
a strong sign that the correspondence failed.

`tools/benchmark_synthetic.py` measures the joint error against ground truth on
random SMPL bodies (random shape, poses with 0.3 rad std per joint, 5000-point
clouds), nude or with two synthetic loose garments: a shirt (torso and arms
inflated by ~4 cm) and a flared skirt that merges the legs. Mean joint error in
mm, 8 bodies per condition:

| | nude | shirt | skirt | shirt + skirt |
| --- | --- | --- | --- | --- |
| spectral transfer | 27.5 | 42.0 | 94.9 | 337.8 |
| failures detected by the plausibility check | 0/8 | 2/8 | 5/8 | 7/8 |
| non-flagged targets, without priors | 27.5 | 33.8 | 59.5 | 76.5 |
| non-flagged targets, with priors | 26.0 | 32.4 | 57.6 | 75.4 |

The priors give a consistent but small gain (~5%). With loose clothing the
error is dominated by failures of the correspondence (whole joints mapped onto
the wrong limb), which the plausibility check detects but no skeleton prior can
repair.

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
* Exact k-d tree nearest neighbours (SciPy) replace FLANN, and the
  eigendecomposition uses a fixed starting vector, so runs are reproducible.

## Requirements and tests

The only required packages are NumPy, SciPy and
[robust_laplacian](https://github.com/nmwsharp/robust-laplacians-py);
PyVista is needed for the viewer and trimesh for mesh formats other than
OBJ/OFF. Tested on Python 3.13 (numpy 2.5, scipy 1.18, robust_laplacian 1.1).

```bash
pip install -e ".[all]"
pytest
```

The tests comparing with the original code need the example files in `data/`
(or in the folder pointed by `SKELETON_TRANSFER_DATA`) and take about two
minutes; they are skipped if the files are missing. The reference outputs in
`tests/data/` can be regenerated with `tools/make_original_reference.py`,
which runs the original code and needs TensorFlow.

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
