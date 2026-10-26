"""
Configuration of the skeleton transfer pipeline.

Every numeric constant that used to be hard-coded in the scripts lives here,
grouped by pipeline stage. The defaults reproduce the settings used in the
paper / original code, so ``TransferConfig()`` is a good starting point.

Example
-------
>>> from skeleton_transfer import TransferConfig
>>> cfg = TransferConfig()
>>> cfg.functional_map.n_eigs_final = 80      # smaller final map
>>> cfg.regressor.skinning_weight_threshold = 0.2  # tighter locality mask
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .skeleton_priors import SkeletonPriorConfig


@dataclass
class SpectralConfig:
    """Laplace-Beltrami eigendecomposition."""

    #: Number of eigenpairs computed for each shape. Must be at least
    #: ``FunctionalMapConfig.n_eigs_final``; the extra eigenpairs are used by
    #: the WKS/WKM descriptors, which benefit from a wide frequency range.
    n_eigs: int = 400


@dataclass
class FunctionalMapConfig:
    """Estimation of the functional map between target and source."""

    # --- spectral sizes -----------------------------------------------------
    #: Size of the initial (small) functional map optimized from descriptors.
    n_eigs_init: int = 20
    #: Size of the final functional map after ZoomOut upsampling.
    n_eigs_final: int = 120
    #: Number of eigenfunctions added at each ZoomOut iteration.
    zoomout_step: int = 10

    # --- descriptors --------------------------------------------------------
    #: Number of energy scales of the Wave Kernel Signature / Map.
    n_descriptor_scales: int = 200
    #: Width of the WKS energy filters, in units of the energy step.
    wks_variance: float = 7.0
    #: Width of the WKM energy filters, relative to the energy range
    #: (``sigma = wkm_variance * (e_max - e_min) / n_descriptor_scales``).
    wkm_variance: float = 7.0
    #: Keep one descriptor column every ``descriptor_subsample``.
    descriptor_subsample: int = 10

    # --- energy and optimizer -----------------------------------------------
    #: Weight of the descriptor preservation term ``||C A - B||^2 / 2``.
    weight_descriptors: float = 1e-1
    #: Weight of the Laplacian commutativity term ``||C L_s - L_t C||^2``.
    weight_laplacian: float = 1e-8

    # --- refinement ---------------------------------------------------------
    #: Number of ICP iterations (orthogonal refinement of the initial map).
    n_icp_iterations: int = 5
    #: Force the landmarks to correspond in every point-to-point step.
    enforce_landmarks: bool = False
    #: If the transferred skeleton is implausible, retry with these sizes of
    #: the initial map and keep the most plausible result (empty = no retry).
    #: ``n_eigs_final - size`` must be a multiple of ``zoomout_step``.
    restart_init_sizes: Tuple[int, ...] = ()


@dataclass
class RegressorConfig:
    """Fitting of the localized joint regressor on the source shape."""

    #: A vertex is considered part of a bone if its skinning weight for that
    #: bone is above this threshold (used to build the locality mask).
    skinning_weight_threshold: float = 0.1

    #: Weight of the joint reconstruction term ``||R V - J||^2``.
    weight_fit: float = 1.0
    #: Weight of the locality term, penalizing coefficients outside the mask.
    #: (The original code had two identical terms with weights 1000 and 100.)
    weight_locality: float = 1100.0
    #: Weight of the partition-of-unity term ``||R 1 - 1||^2``.
    weight_partition_of_unity: float = 10.0


@dataclass
class IterativeConfig:
    """Alternate skeleton estimation and correspondence refinement.

    At each iteration the current target skeleton defines, for every target
    point, its affinity to each bone (softmax of the negative distances to the
    bone segments); the same is computed on the source with its skeleton.
    These affinities are appended to the spectral embeddings in the
    point-to-point searches, so that the correspondence respects the skeleton.
    """

    #: Maximum number of skeleton -> correspondence iterations.
    n_iterations: int = 8
    #: Weight of the bone affinities relative to the spectral embedding.
    feature_weight: float = 1.0
    #: Softmax temperature of the affinities, relative to the mean bone length.
    temperature: float = 0.25
    #: Refinement steps (p2p <-> functional map) per iteration.
    n_refine_steps: int = 3
    #: Stop when no joint moves more than this fraction of the mean bone length.
    tolerance: float = 0.01


@dataclass
class TransferConfig:
    """Top-level configuration grouping all pipeline stages."""

    #: Set to True if the source is a point cloud (no faces available).
    source_is_point_cloud: bool = False
    #: Set to True if the target is a point cloud (no faces available).
    target_is_point_cloud: bool = False

    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    functional_map: FunctionalMapConfig = field(default_factory=FunctionalMapConfig)
    regressor: RegressorConfig = field(default_factory=RegressorConfig)
    #: Symmetry / proportion priors applied to the transferred joints
    #: (None = off). See :mod:`skeleton_transfer.skeleton_priors`.
    skeleton_priors: Optional[SkeletonPriorConfig] = None
    #: Iterative skeleton <-> correspondence refinement (None = off).
    iterative: Optional[IterativeConfig] = None

    def validate(self) -> None:
        """Raise ``ValueError`` if the parameters are inconsistent."""
        fm = self.functional_map
        if fm.n_eigs_final > self.spectral.n_eigs:
            raise ValueError(
                f"functional_map.n_eigs_final ({fm.n_eigs_final}) cannot exceed "
                f"spectral.n_eigs ({self.spectral.n_eigs})."
            )
        if fm.n_eigs_init > fm.n_eigs_final:
            raise ValueError("functional_map.n_eigs_init must be <= n_eigs_final.")
        for size in fm.restart_init_sizes:
            if size > fm.n_eigs_final or (fm.n_eigs_final - size) % fm.zoomout_step != 0:
                raise ValueError(f"Invalid restart size {size}: see FunctionalMapConfig.restart_init_sizes.")
        if (fm.n_eigs_final - fm.n_eigs_init) % fm.zoomout_step != 0:
            raise ValueError(
                "n_eigs_final - n_eigs_init must be a multiple of zoomout_step, "
                "otherwise ZoomOut does not end exactly at n_eigs_final."
            )
