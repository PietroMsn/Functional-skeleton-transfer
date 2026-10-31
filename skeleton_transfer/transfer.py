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

Optional robustness steps (see :class:`~skeleton_transfer.config.TransferConfig`):

- **restarts**: if the skeleton is implausible, the functional map is
  re-estimated from initial maps of different sizes and the most plausible
  result is kept (or, with ``restart_selection="refined"``, every candidate
  is refined and the one with the best segment consistency is kept);
- **iterative refinement**: the transferred skeleton is used to refine the
  correspondence (bone affinities as extra features), and the skeleton is
  transferred again, until the joints stop moving;
- **skeleton priors**: symmetry / proportion correction of the joints.

Every result carries unsupervised quality scores (:mod:`.diagnostics`).
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import TransferConfig
from .diagnostics import bone_affinity, transfer_diagnostics
from .functional_map import cmh_basis, estimate_functional_map, refine_with_features
from .regressor import fit_spatial_regressor, locality_mask
from .shapes import SourceShape, TargetShape
from .skeleton_priors import apply_skeleton_priors, bone_lengths, is_plausible
from .spectral import LaplaceBasis, compute_laplace_basis

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Output of :func:`transfer_skeleton`.

    Attributes
    ----------
    joints_spectral : (n_joints, 3) joints transferred through the functional map
    joints_pointwise : (n_joints, 3) joints transferred through the p2p map
        (both after the skeleton priors, if enabled)
    regressor : (n_joints, n_source_vertices) regressor fitted on the source
    mask : (n_joints, n_source_vertices) locality mask used for the regressor
    C : (k, k) functional map, target -> source
    p2p : (n_source_vertices,) target vertex matched to each source vertex
    plausible : False if the bone proportions of the spectral transfer reveal
        a failed correspondence (see ``skeleton_priors.is_plausible``)
    diagnostics : unsupervised quality scores of the final result
        (see :func:`.diagnostics.transfer_diagnostics`)
    history : joints and diagnostics after each stage (initial map, restarts,
        iterations), for analysis
    """

    joints_spectral: np.ndarray
    joints_pointwise: np.ndarray
    regressor: np.ndarray
    mask: np.ndarray
    C: np.ndarray
    p2p: np.ndarray
    plausible: bool = True
    diagnostics: dict = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)


def transfer_skeleton(
    source: SourceShape,
    target: TargetShape,
    config: Optional[TransferConfig] = None,
    regressor: Optional[np.ndarray] = None,
    basis_src: Optional[LaplaceBasis] = None,
    basis_tgt: Optional[LaplaceBasis] = None,
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
    basis_tgt : optional pre-computed target Laplace basis (e.g. to compare
        several configurations on the same target).
    """
    config = config or TransferConfig()
    config.validate()
    fm_cfg = config.functional_map
    k = fm_cfg.n_eigs_final

    # 1) Localized regressor on the source
    mask = locality_mask(source.skinning_weights, source.parents, config.regressor.skinning_weight_threshold)
    if regressor is None:
        logger.info("Fitting the joint regressor on the source...")
        regressor = fit_spatial_regressor(source.vertices, source.joints, mask, config.regressor)

    # 2) Laplace-Beltrami bases
    logger.info("Computing the Laplace-Beltrami bases...")
    if basis_src is None:
        basis_src = compute_laplace_basis(
            source.vertices, source.faces, config.spectral.n_eigs, config.source_is_point_cloud or source.faces is None
        )
    if basis_tgt is None:
        basis_tgt = compute_laplace_basis(
            target.vertices, target.faces, config.spectral.n_eigs, config.target_is_point_cloud or target.faces is None
        )

    phi_src = basis_src.functions(k, fm_cfg.cmh)  # (n_source_vertices, k), k + 3 with the CMH functions
    regressor_hat = regressor @ phi_src  # (n_joints, k)
    target_coords_hat = basis_tgt.project(target.vertices, k, fm_cfg.cmh)  # (k, 3)
    landmarks = (source.landmarks, target.landmarks) if fm_cfg.enforce_landmarks else None
    history = []

    def evaluate(C, p2p, stage):
        """Joints of both transfers and their diagnostics."""
        spectral = regressor_hat @ C @ target_coords_hat
        pointwise = regressor @ target.vertices[p2p]
        diag = transfer_diagnostics(
            spectral, pointwise, C, p2p, source.vertices, target.vertices, source.joints, source.parents
        )
        history.append({"stage": stage, "joints_spectral": spectral, "joints_pointwise": pointwise, **diag})
        return spectral, pointwise, diag

    it_cfg = config.iterative
    if it_cfg is not None:
        affinity_src = bone_affinity(source.vertices, source.joints, source.parents, it_cfg.temperature)
        # Scale the affinities to the typical norm of the spectral embedding.
        embedding_norm = np.median(np.linalg.norm(phi_src, axis=1))
        scale = it_cfg.feature_weight * embedding_norm / np.median(np.linalg.norm(affinity_src, axis=1))

    def refine(C, p2p, spectral, pointwise, diag, label=""):
        """Iterative refinement: skeleton -> correspondence -> skeleton ..."""
        best_consistency, stalled = diag["segment_consistency"], 0
        for it in range(it_cfg.n_iterations):
            skeleton = spectral
            if config.skeleton_priors is not None:
                skeleton = apply_skeleton_priors(spectral, source.parents, source.joints, config.skeleton_priors)
            affinity_tgt = bone_affinity(target.vertices, skeleton, source.parents, it_cfg.temperature)
            C, p2p = refine_with_features(
                p2p, cmh_basis(basis_src, fm_cfg), cmh_basis(basis_tgt, fm_cfg), k, it_cfg.n_refine_steps,
                (scale * affinity_src, scale * affinity_tgt), landmarks,
            )  # fmt: skip
            previous = spectral
            spectral, pointwise, diag = evaluate(C, p2p, f"iteration {it + 1}{label}")
            step = np.linalg.norm(spectral - previous, axis=1).max() / bone_lengths(spectral, source.parents).mean()
            history[-1]["joint_step"] = float(step)
            if diag["segment_consistency"] > best_consistency + it_cfg.stop_min_gain:
                best_consistency, stalled = diag["segment_consistency"], 0
            else:
                stalled += 1
            if step < it_cfg.tolerance or (it_cfg.stop_patience and stalled >= it_cfg.stop_patience):
                break
        return C, p2p, spectral, pointwise, diag

    # 3) Correspondence, with restarts if the skeleton is implausible
    logger.info("Estimating the functional map...")
    fmap = estimate_functional_map(basis_src, basis_tgt, source.landmarks, target.landmarks, fm_cfg)
    C, p2p = fmap.C, fmap.p2p
    spectral, pointwise, diag = evaluate(C, p2p, f"init k={fm_cfg.n_eigs_init}")
    refined = False
    if not is_plausible(spectral, source.parents, source.joints) and fm_cfg.restart_init_sizes:
        if fm_cfg.restart_selection == "refined":
            # 3-4) Refine every candidate, keep the best final segment consistency.
            logger.info("Implausible skeleton, refining the initial map and %d restarts...", len(fm_cfg.restart_init_sizes))
            candidates = [refine(C, p2p, spectral, pointwise, diag, f" (k={fm_cfg.n_eigs_init})")]
            for size in fm_cfg.restart_init_sizes:
                alt = estimate_functional_map(
                    basis_src, basis_tgt, source.landmarks, target.landmarks, fm_cfg, n_eigs_init=size
                )
                candidates.append(refine(alt.C, alt.p2p, *evaluate(alt.C, alt.p2p, f"restart k={size}"), f" (k={size})"))
            C, p2p, spectral, pointwise, diag = max(candidates, key=lambda c: c[4]["segment_consistency"])
            refined = True
        else:
            for size in fm_cfg.restart_init_sizes:
                logger.info("Implausible skeleton, restarting from an initial map of size %d...", size)
                alt = estimate_functional_map(
                    basis_src, basis_tgt, source.landmarks, target.landmarks, fm_cfg, n_eigs_init=size
                )
                alt_eval = evaluate(alt.C, alt.p2p, f"restart k={size}")
                if alt_eval[2]["bone_ratio"] < diag["bone_ratio"]:
                    C, p2p, (spectral, pointwise, diag) = alt.C, alt.p2p, alt_eval
                if is_plausible(spectral, source.parents, source.joints):
                    break

    # 4) Iterative refinement of the chosen map
    if it_cfg is not None and not refined:
        C, p2p, spectral, pointwise, diag = refine(C, p2p, spectral, pointwise, diag)

    # 5) Sanity check and optional skeleton priors
    plausible = is_plausible(spectral, source.parents, source.joints)
    if not plausible:
        logger.warning(
            "Implausible bone proportions in the transferred skeleton: the correspondence "
            "probably failed (check the landmarks and the target connectivity)."
        )
    if config.skeleton_priors is not None:
        spectral = apply_skeleton_priors(spectral, source.parents, source.joints, config.skeleton_priors)
        pointwise = apply_skeleton_priors(pointwise, source.parents, source.joints, config.skeleton_priors)

    return TransferResult(
        joints_spectral=spectral,
        joints_pointwise=pointwise,
        regressor=regressor,
        mask=mask,
        C=C,
        p2p=p2p,
        plausible=plausible,
        diagnostics=diag,
        history=history,
    )
