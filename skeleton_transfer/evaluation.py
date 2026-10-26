"""
Evaluation of pipeline variants against ground-truth joints.

Shared by the synthetic benchmark (``tools/benchmark_synthetic.py``) and the
evaluation on real scans (``tools/evaluate_scans.py``):

- :func:`default_variants`: the pipeline variants compared in the paper;
- :class:`Evaluator`: runs every variant on a target, reusing the source
  regressor and the Laplace bases;
- :func:`report`: error tables per condition, failures, error per joint
  group, rank correlation of the unsupervised diagnostics with the error,
  and convergence of the iterative refinement.

A run is a plain dict (``condition``, ``target``, ``variant``, ``gt``,
``joints``, ``error``, ``plausible``, ``history``, ``seconds``) so that
results can be pickled and re-reported.
"""

import time
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from .config import IterativeConfig, TransferConfig
from .regressor import fit_spatial_regressor, locality_mask
from .shapes import SMPL_EXTRA_LANDMARKS, SMPL_LANDMARKS, SourceShape, TargetShape
from .skeleton_priors import SkeletonPriorConfig
from .smpl import SMPL_JOINT_NAMES
from .spectral import compute_laplace_basis
from .transfer import transfer_skeleton

#: Landmarks given to every target: the 7 of the paper, then knees and elbows.
ALL_LANDMARKS = np.concatenate((SMPL_LANDMARKS, SMPL_EXTRA_LANDMARKS))
JOINT_GROUPS = {
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "spine": [0, 3, 6, 9, 12, 15],
    "arms": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
}
#: Error above which a transfer is counted as a failure (metres).
FAILURE = 0.1
DIAGNOSTICS = ["bone_ratio", "symmetry", "spectral_pointwise_gap", "orthogonality", "coverage", "segment_consistency"]
HIGHER_IS_BETTER = {"coverage", "segment_consistency"}

Variant = Tuple[str, bool, TransferConfig]  # name, uses the extra landmarks, config


def default_variants() -> List[Variant]:
    """Pipeline variants: (name, uses knee/elbow landmarks, config)."""
    base = TransferConfig()
    fm = base.functional_map
    restarts = replace(fm, restart_init_sizes=(10, 30, 40))
    both = replace(fm, enforce_landmarks=True, restart_init_sizes=(10, 30, 40))
    return [
        ("baseline", False, base),
        ("+ knee/elbow landmarks", True, base),
        ("+ restarts", False, replace(base, functional_map=restarts)),
        ("+ iterative", False, replace(base, iterative=IterativeConfig())),
        ("all", True, replace(base, functional_map=both, iterative=IterativeConfig())),
        ("all + priors", True, replace(base, functional_map=both, iterative=IterativeConfig(),
                                       skeleton_priors=SkeletonPriorConfig())),
    ]  # fmt: skip


class Evaluator:
    """Run pipeline variants on targets with ground-truth joints.

    Parameters
    ----------
    source : the rigged source, with the 7 landmarks of the paper
    variants : defaults to :func:`default_variants`
    """

    def __init__(self, source: SourceShape, variants: Optional[List[Variant]] = None):
        self.variants = variants or default_variants()
        self.config = TransferConfig()
        self.source = source
        self.source_extra = replace(source, landmarks=np.concatenate((source.landmarks, SMPL_EXTRA_LANDMARKS)))
        mask = locality_mask(source.skinning_weights, source.parents, self.config.regressor.skinning_weight_threshold)
        self.regressor = fit_spatial_regressor(source.vertices, source.joints, mask, self.config.regressor)
        self.basis_src = compute_laplace_basis(source.vertices, source.faces, self.config.spectral.n_eigs)
        self.runs: List[dict] = []

    def evaluate(self, target: TargetShape, gt: np.ndarray, condition: str, name) -> str:
        """Run all variants on ``target``, whose landmarks are :data:`ALL_LANDMARKS`.

        Returns a one-line summary of the errors (mm), one per variant.
        """
        n_base = len(self.source.landmarks)
        basis_tgt = compute_laplace_basis(target.vertices, target.faces, self.config.spectral.n_eigs, target.faces is None)
        line = f"{condition:14s} {str(name):>24s}"
        for variant, extra, cfg in self.variants:
            t = time.time()
            src = self.source_extra if extra else self.source
            tgt = target if extra else replace(target, landmarks=target.landmarks[:n_base])
            result = transfer_skeleton(
                src, tgt, cfg, regressor=self.regressor, basis_src=self.basis_src, basis_tgt=basis_tgt
            )
            err = np.linalg.norm(result.joints_spectral - gt, axis=1)
            for h in result.history:
                h["error"] = float(np.linalg.norm(h["joints_spectral"] - gt, axis=1).mean())
            self.runs.append(dict(condition=condition, target=name, variant=variant, gt=gt,
                                  joints=result.joints_spectral, error=float(err.mean()),
                                  plausible=result.plausible, history=result.history,
                                  seconds=time.time() - t))  # fmt: skip
            line += f" | {1000 * err.mean():6.1f}"
        return line


def report(runs: List[dict]) -> None:
    """Print the evaluation tables of a list of runs."""
    names = list(dict.fromkeys(r["variant"] for r in runs))
    conditions = list(dict.fromkeys(r["condition"] for r in runs))

    def select(variant, condition=None):
        return [r for r in runs if r["variant"] == variant and (condition is None or r["condition"] == condition)]

    header = f"{'variant':24s}" + "".join(f"{c:>14s}" for c in conditions) + f"{'failures':>10s}{'time':>7s}"
    print("\nMean joint error (mm), spectral transfer\n" + header + "\n" + "-" * len(header))
    for name in names:
        cells = "".join(f"{1000 * np.mean([r['error'] for r in select(name, c)]):14.1f}" for c in conditions)
        fails = sum(r["error"] > FAILURE for r in select(name))
        secs = np.mean([r["seconds"] for r in select(name)])
        print(f"{name:24s}{cells}{fails:>6d}/{len(select(name)):<3d}{secs:6.0f}s")

    print("\nMedian joint error (mm) - less sensitive to the failures")
    for name in names:
        cells = "".join(f"{1000 * np.median([r['error'] for r in select(name, c)]):14.1f}" for c in conditions)
        print(f"{name:24s}{cells}")

    first, last = names[0], names[-1]
    print(f"\nError per joint group (mm): {first} -> {last}")
    for c in conditions:
        b, a = select(first, c), select(last, c)
        groups = "  ".join(
            f"{g}: {1000 * np.mean([np.linalg.norm(r['joints'][idx] - r['gt'][idx], axis=1).mean() for r in b]):5.1f}"
            f" -> {1000 * np.mean([np.linalg.norm(r['joints'][idx] - r['gt'][idx], axis=1).mean() for r in a]):5.1f}"
            for g, idx in JOINT_GROUPS.items()
        )
        print(f"{c:14s} {groups}")
    err = np.array([np.linalg.norm(r["joints"] - r["gt"], axis=1) for r in select(last)])
    worst = np.argsort(-err.mean(axis=0))[:5]
    print(f"worst joints ({last}):", ", ".join(f"{SMPL_JOINT_NAMES[j]} {1000 * err[:, j].mean():.0f}" for j in worst))

    # Unsupervised diagnostics vs ground-truth error, over every stage of every run
    stages = [h for r in runs for h in r["history"]]
    errors = np.array([h["error"] for h in stages])
    fail = errors > FAILURE
    print(f"\nUnsupervised diagnostics vs true error ({len(stages)} stages of all runs)")
    print(f"{'diagnostic':24s}{'Spearman rho':>14s}{'AUC fail':>10s}")
    for key in DIAGNOSTICS:
        values = np.array([h[key] for h in stages]) * (-1 if key in HIGHER_IS_BETTER else 1)
        rho = spearmanr(values, errors)[0]
        auc = np.mean(values[fail][:, None] > values[~fail][None, :]) if fail.any() and (~fail).any() else np.nan
        print(f"{key:24s}{rho:14.2f}{auc:10.2f}")

    iterative = select("+ iterative")
    if iterative:
        print("\nIterative refinement: mean error (mm) and mean joint step per iteration")
        for s in range(max(len(r["history"]) for r in iterative)):
            hs = [r["history"][min(s, len(r["history"]) - 1)] for r in iterative]
            step = np.mean([h.get("joint_step", np.nan) for h in hs])
            print(f"  stage {s}: error {1000 * np.mean([h['error'] for h in hs]):6.1f}   step {step:.3f}")
