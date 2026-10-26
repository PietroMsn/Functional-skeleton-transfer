"""
A Functional Skeleton Transfer
==============================

Reference implementation of

    P. Musoni, R. Marin, S. Melzi, U. Castellani,
    "A Functional Skeleton Transfer", PACMCGIT 4(3), 2021.

The method transfers an animation skeleton (the joint positions of a rigged
template such as SMPL) onto a new, unrigged shape. It works in two stages:

1. **Joint regressor on the source.** A sparse linear regressor ``R`` is
   learned on the source template so that ``R @ vertices`` reproduces the
   source joints. Each joint is only allowed to depend on the vertices of the
   two body parts it connects (see :mod:`skeleton_transfer.regressor`).

2. **Correspondence source <-> target.** A functional map ``C`` between the
   Laplace-Beltrami eigenbases of the two shapes is estimated from wave-kernel
   descriptors and a few landmarks, then refined with ICP and ZoomOut
   (see :mod:`skeleton_transfer.functional_map`).

The regressor is then moved onto the target either through the functional map
(spectral transfer) or through the induced point-to-point map (pointwise
transfer). See :func:`skeleton_transfer.transfer.transfer_skeleton`.

Package layout
--------------
- :mod:`~skeleton_transfer.config`          all tunable parameters (dataclasses)
- :mod:`~skeleton_transfer.shapes`          shape containers and data loading
- :mod:`~skeleton_transfer.spectral`        Laplace-Beltrami eigenbasis
- :mod:`~skeleton_transfer.descriptors`     WKS / WKM descriptors
- :mod:`~skeleton_transfer.functional_map`  functional map estimation
- :mod:`~skeleton_transfer.regressor`       localized joint regressor
- :mod:`~skeleton_transfer.skeleton_priors` symmetry / proportion priors on the target
- :mod:`~skeleton_transfer.smpl`            minimal SMPL model (shape + pose)
- :mod:`~skeleton_transfer.synthetic`       synthetic targets with ground truth
- :mod:`~skeleton_transfer.transfer`        end-to-end pipeline
- :mod:`~skeleton_transfer.visualization`   PyVista plotting (optional)
"""

from .config import FunctionalMapConfig, IterativeConfig, RegressorConfig, SpectralConfig, TransferConfig
from .skeleton_priors import SkeletonPriorConfig, apply_skeleton_priors
from .shapes import SourceShape, TargetShape, load_mesh, load_smpl_source, load_target, load_target_mat
from .transfer import TransferResult, transfer_skeleton

__all__ = [
    "FunctionalMapConfig",
    "IterativeConfig",
    "RegressorConfig",
    "SpectralConfig",
    "TransferConfig",
    "SkeletonPriorConfig",
    "apply_skeleton_priors",
    "SourceShape",
    "TargetShape",
    "load_mesh",
    "load_smpl_source",
    "load_target",
    "load_target_mat",
    "TransferResult",
    "transfer_skeleton",
]
