"""
End-to-end skeleton transfer.

Given a rigged :class:`~skeleton_transfer.shapes.SourceShape` and an unrigged
:class:`~skeleton_transfer.shapes.TargetShape`, :func:`transfer_skeleton`

1. fits a localized joint regressor ``R`` on the source;
2. computes the Laplace-Beltrami bases of both shapes;
3. estimates the functional map ``C`` (target -> source) and the induced
   point-to-point map ``p2p`` (source vertex -> target vertex);
4. transfers the joints in two ways:

   - **spectral transfer**: express ``R`` in the source eigenbasis
     (``R_hat = R @ Phi_src``), move the target coordinates into the source
     basis with ``C`` and apply it:

         J_target = R_hat @ C @ (Phi_tgt^+ @ V_target)

   - **pointwise transfer**: apply ``R`` to the target vertices matched to
     each source vertex:

         J_target = R @ V_target[p2p]

Both results are expressed in the coordinate frame of the input target.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import TransferConfig
from .functional_map import estimate_functional_map
from .regressor import fit_spatial_regressor, locality_mask
from .shapes import SourceShape, TargetShape
from .spectral import LaplaceBasis, compute_laplace_basis

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Output of :func:`transfer_skeleton`.

    Attributes
    ----------
    joints_spectral : (n_joints, 3) joints transferred through the functional map
    joints_pointwise : (n_joints, 3) joints transferred through the p2p map
    regressor : (n_joints, n_source_vertices) regressor fitted on the source
    mask : (n_joints, n_source_vertices) locality mask used for the regressor
    C : (k, k) functional map, target -> source
    p2p : (n_source_vertices,) target vertex matched to each source vertex
    """

    joints_spectral: np.ndarray
    joints_pointwise: np.ndarray
    regressor: np.ndarray
    mask: np.ndarray
    C: np.ndarray
    p2p: np.ndarray


def transfer_skeleton(
    source: SourceShape,
    target: TargetShape,
    config: Optional[TransferConfig] = None,
    regressor: Optional[np.ndarray] = None,
    basis_src: Optional[LaplaceBasis] = None,
) -> TransferResult:
    """Transfer the skeleton of ``source`` onto ``target``.

    Parameters
    ----------
    source : rigged template
    target : shape receiving the skeleton
    config : pipeline parameters (defaults to ``TransferConfig()``)
    regressor : optional pre-computed source regressor. The regressor only
        depends on the source, so it can be reused when transferring the same
        template onto many targets.
    basis_src : optional pre-computed source Laplace basis (same reason).
    """
    config = config or TransferConfig()
    config.validate()
    k = config.functional_map.n_eigs_final

    # 1) Localized regressor on the source
    mask = locality_mask(
        source.skinning_weights, source.parents, config.regressor.skinning_weight_threshold
    )
    if regressor is None:
        logger.info("Fitting the joint regressor on the source...")
        regressor = fit_spatial_regressor(source.vertices, source.joints, mask, config.regressor)

    # 2) Laplace-Beltrami bases
    logger.info("Computing the Laplace-Beltrami bases...")
    if basis_src is None:
        basis_src = compute_laplace_basis(
            source.vertices, source.faces, config.spectral.n_eigs, config.source_is_point_cloud or source.faces is None
        )
    basis_tgt = compute_laplace_basis(
        target.vertices, target.faces, config.spectral.n_eigs, config.target_is_point_cloud or target.faces is None
    )

    # 3) Correspondence
    logger.info("Estimating the functional map...")
    fmap = estimate_functional_map(
        basis_src, basis_tgt, source.landmarks, target.landmarks, config.functional_map
    )

    # 4a) Spectral transfer through C
    regressor_hat = regressor @ basis_src.evecs[:, :k]  # (n_joints, k)
    target_coords_hat = basis_tgt.project(target.vertices, k)  # (k, 3)
    joints_spectral = regressor_hat @ fmap.C @ target_coords_hat

    # 4b) Pointwise transfer through p2p
    joints_pointwise = regressor @ target.vertices[fmap.p2p]

    return TransferResult(
        joints_spectral=joints_spectral,
        joints_pointwise=joints_pointwise,
        regressor=regressor,
        mask=mask,
        C=fmap.C,
        p2p=fmap.p2p,
    )
