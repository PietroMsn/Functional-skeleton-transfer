"""
Configuration of the skeleton transfer pipeline.

Every numeric constant that used to be hard-coded in the scripts lives here,
grouped by pipeline stage. The defaults are those of the code released with
the paper, so ``TransferConfig()`` reproduces its results.

Example
-------
>>> from skeleton_transfer import TransferConfig
>>> cfg = TransferConfig()
>>> cfg.functional_map.n_eigs_final = 80      # smaller final map
>>> cfg.regressor.skinning_weight_threshold = 0.2  # tighter locality mask
"""

from dataclasses import dataclass, field


@dataclass
class SpectralConfig:
    """Laplace-Beltrami eigendecomposition."""

    #: Number of eigenpairs computed for each shape. Must be at least
    #: ``FunctionalMapConfig.n_eigs_final``; the extra eigenpairs are used by
    #: the WKS/WKM descriptors, which benefit from a wide frequency range.
    n_eigs: int = 400


@dataclass
class FunctionalMapConfig:
    """Estimation of the functional map between target and source (Sec. 4.2, Step 2)."""

    # --- spectral sizes -----------------------------------------------------
    #: Size of the initial (small) functional map optimized from descriptors
    #: (20 x 20 in the paper).
    n_eigs_init: int = 20
    #: Size of the final functional map after ZoomOut upsampling (120 x 120 in
    #: the paper); also the size of the spectral regressor of Eq. (8).
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


@dataclass
class RegressorConfig:
    """Fitting of the localized joint regressor on the source shape (Sec. 4.2, Step 1)."""

    #: A vertex is considered part of a bone if its skinning weight for that
    #: bone is above this threshold (used to build the mask ``F`` of Eq. 10).
    skinning_weight_threshold: float = 0.1

    #: Weight of the reconstruction term ``||R V - J||^2`` (Eq. 9).
    weight_fit: float = 1.0
    #: Weight of the coefficients outside the mask: the locality (Eq. 10) and
    #: sparsity (Eq. 11) terms, identical for a binary mask, with the weights
    #: 1000 and 100 of the released code.
    weight_locality: float = 1100.0
    #: Weight of the convexity term ``||R 1 - 1||^2`` (Eq. 12), as in the
    #: released code.
    weight_partition_of_unity: float = 10.0


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
        if (fm.n_eigs_final - fm.n_eigs_init) % fm.zoomout_step != 0:
            raise ValueError(
                "n_eigs_final - n_eigs_init must be a multiple of zoomout_step, "
                "otherwise ZoomOut does not end exactly at n_eigs_final."
            )
