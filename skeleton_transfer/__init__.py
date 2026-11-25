"""
A Functional Skeleton Transfer
==============================

Reference implementation of

    P. Musoni, R. Marin, S. Melzi, U. Castellani,
    "A Functional Skeleton Transfer", PACMCGIT 4(3), 2021.

The method transfers an animation skeleton (the joint positions of a rigged
template such as SMPL) onto a new, unrigged shape, following Algorithm 1 of
the paper (Sec. 4.2):

1. **Spatial regressor on the source** (Step 1, Eqs. 9-13). A sparse linear
   regressor ``R`` is fitted on the source template so that ``R @ vertices``
   reproduces the source joints. Each joint is only allowed to depend on the
   vertices of the two body parts it connects
   (:mod:`skeleton_transfer.regressor`).

2. **Functional map** (Step 2). A functional map ``C`` between the
   Laplace-Beltrami eigenbases of the two shapes (Sec. 3) is estimated from
   wave-kernel descriptors and seven landmarks, then refined with ICP and
   ZoomOut (:mod:`skeleton_transfer.functional_map`).

3. **Transfer** (Steps 3-4, Eqs. 8 and 14). The regressor is converted into
   the spectral regressor ``R @ Phi`` and applied through ``C`` to the
   Fourier coefficients of the target coordinates
   (:func:`skeleton_transfer.transfer.transfer_skeleton`).

Package layout
--------------
- :mod:`~skeleton_transfer.config`          all tunable parameters (dataclasses)
- :mod:`~skeleton_transfer.shapes`          shape containers and data loading
- :mod:`~skeleton_transfer.spectral`        Laplace-Beltrami eigenbasis
- :mod:`~skeleton_transfer.descriptors`     WKS / WKM descriptors
- :mod:`~skeleton_transfer.functional_map`  functional map estimation
- :mod:`~skeleton_transfer.regressor`       localized joint regressor
- :mod:`~skeleton_transfer.transfer`        end-to-end pipeline
- :mod:`~skeleton_transfer.visualization`   PyVista plotting (optional)
"""

from .config import FunctionalMapConfig, RegressorConfig, SpectralConfig, TransferConfig
from .shapes import (
    SourceShape,
    TargetShape,
    find_smpl_model,
    load_mesh,
    load_smpl_source,
    load_target,
    load_target_mat,
)
from .transfer import TransferResult, transfer_skeleton

__all__ = [
    "FunctionalMapConfig",
    "RegressorConfig",
    "SpectralConfig",
    "TransferConfig",
    "SourceShape",
    "TargetShape",
    "find_smpl_model",
    "load_mesh",
    "load_smpl_source",
    "load_target",
    "load_target_mat",
    "TransferResult",
    "transfer_skeleton",
]
