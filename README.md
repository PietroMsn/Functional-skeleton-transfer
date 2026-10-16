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
* [Requirements](#requirements)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Quick-start

```bash
pip install -r requirements.txt
python main.py
```

This transfers the skeleton of the SMPL template (source) onto a remeshed
Dynamic FAUST shape (target), both stored in `data/`, and opens a 3D viewer
with the source skeleton and the two transferred skeletons.

Useful options (`python main.py --help` for the full list):

| option | effect |
| --- | --- |
| `--target PATH` | use another target `.mat` file (fields `vertex`, `faces`, `landmarkRes`, 1-based) |
| `--target-point-cloud` | treat the target as a point cloud |
| `--output joints.npz` | save joints, functional map, point-to-point map and regressor |
| `--no-plot` | do not open the viewer |
| `--screenshot view.png` | save the view to an image instead of opening a window |
| `--plot-results joints.npz` | only plot results saved with `--output` (no computation) |

Sample data:
* SMPL model: http://smpl.is.tue.mpg.de
* Dynamic FAUST dataset: https://dfaust.is.tue.mpg.de/

## How it works

1. **Joint regressor on the source.** A linear regressor `R` (joints = `R @ vertices`)
   is fitted on the source. Each joint may only use the vertices skinned to both
   bones it connects, and each row of `R` sums to one.
2. **Correspondence.** A functional map between the Laplace-Beltrami eigenbases of
   the two shapes is estimated from WKS descriptors plus wave-kernel maps of 7
   landmarks, then refined with spectral ICP and ZoomOut.
3. **Transfer.** The regressor is moved onto the target either through the
   functional map (*spectral transfer*) or through the induced point-to-point
   map (*pointwise transfer*).

## Code structure

```
main.py                      command line entry point
skeleton_transfer/
    config.py                all parameters, grouped by stage (dataclasses)
    shapes.py                SourceShape / TargetShape, .mat and .off loading
    spectral.py              Laplace-Beltrami eigenbasis
    descriptors.py           Wave Kernel Signature / Wave Kernel Map
    functional_map.py        functional map: optimization, ICP, ZoomOut
    regressor.py             localized joint regressor
    transfer.py              end-to-end pipeline (transfer_skeleton)
    visualization.py         PyVista viewer (optional)
data/                        example source (SMPL) and target (DFAUST)
```

Every numeric parameter lives in `skeleton_transfer/config.py`, with a comment
explaining its role; the defaults are the ones used in the paper.

## Using it from Python

```python
from skeleton_transfer import TransferConfig, load_smpl_source, load_target_mat, transfer_skeleton

source = load_smpl_source("data/basicmodel_m_lbs.mat")
target = load_target_mat("data/50007_knees_001806.mat")

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

## Requirements

All the required packages are listed in `requirements.txt`. The code was
originally developed on Windows 10 with Python 3.7 and has been tested on
Python 3.13 with recent versions of the packages.

On some Linux setups using software OpenGL (Mesa), loading TensorFlow and VTK
in the same process can crash the viewer. In that case run the computation
with `--output results.npz --no-plot`, then plot in a separate run with
`--plot-results results.npz`.

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
