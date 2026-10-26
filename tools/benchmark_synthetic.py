"""
Benchmark on synthetic SMPL targets with ground-truth joints.

For every condition (nude, loose shirt, skirt, ...) random SMPL bodies are
generated (see ``skeleton_transfer.synthetic``), the SMPL skeleton is
transferred onto them, and the joint errors are measured with and without
the skeleton priors. Usage, from the repository root::

    python tools/benchmark_synthetic.py --smpl data/basicmodel_m_lbs.mat -n 8 --out bench.npz

The transfer is the slow part (~30 s per target); the raw transferred
joints are saved, so different prior weights can be evaluated afterwards
without recomputing them::

    python tools/benchmark_synthetic.py --evaluate bench.npz
"""

import argparse
import itertools
import logging
import time

import numpy as np

from skeleton_transfer import TransferConfig, transfer_skeleton
from skeleton_transfer.regressor import fit_spatial_regressor, locality_mask
from skeleton_transfer.skeleton_priors import SkeletonPriorConfig, apply_skeleton_priors
from skeleton_transfer.smpl import SMPL_JOINT_NAMES, SMPLModel
from skeleton_transfer.spectral import compute_laplace_basis
from skeleton_transfer.synthetic import make_synthetic_target

CONDITIONS = {"nude": (), "shirt": ("shirt",), "skirt": ("skirt",), "shirt+skirt": ("shirt", "skirt")}
JOINT_GROUPS = {
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
    "spine": [0, 3, 6, 9, 12, 15],
    "arms": [13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
}


def run(args):
    model = SMPLModel.from_mat(args.smpl)
    source = model.as_source()
    config = TransferConfig()
    mask = locality_mask(source.skinning_weights, source.parents, config.regressor.skinning_weight_threshold)
    regressor = fit_spatial_regressor(source.vertices, source.joints, mask, config.regressor)
    basis_src = compute_laplace_basis(source.vertices, source.faces, config.spectral.n_eigs)

    records = {"condition": [], "gt": [], "spectral": [], "pointwise": []}
    for name in args.conditions:
        rng = np.random.default_rng(args.seed)  # same bodies for every condition
        for i in range(args.n):
            target, gt = make_synthetic_target(
                model, rng, pose_std=args.pose_std, shape_std=args.shape_std,
                clothing=CONDITIONS[name], n_points=args.n_points, noise=args.noise,
            )  # fmt: skip
            t = time.time()
            result = transfer_skeleton(source, target, config, regressor=regressor, basis_src=basis_src)
            err = 1000 * np.linalg.norm(result.joints_spectral - gt, axis=1).mean()
            print(f"{name:12s} {i + 1}/{args.n}  spectral error {err:6.1f} mm  ({time.time() - t:.0f} s)", flush=True)
            records["condition"].append(name)
            records["gt"].append(gt)
            records["spectral"].append(result.joints_spectral)
            records["pointwise"].append(result.joints_pointwise)
    np.savez(
        args.out,
        **{k: np.array(v) for k, v in records.items()},
        source_joints=source.joints,
        parents=source.parents,
    )
    print(f"Saved {args.out}")
    return args.out


def evaluate(path, prior_grid):
    data = np.load(path)
    conditions, gt = data["condition"], data["gt"]
    source_joints, parents = data["source_joints"], data["parents"]

    rows = []
    for method in ["spectral", "pointwise"]:
        raw = data[method]
        rows.append((method, "-", raw))
        for w_sym, w_prop in prior_grid:
            cfg = SkeletonPriorConfig(weight_symmetry=w_sym, weight_proportions=w_prop)
            refined = np.array([apply_skeleton_priors(j, parents, source_joints, cfg) for j in raw])
            rows.append((method, f"sym={w_sym:g} prop={w_prop:g}", refined))

    names = list(dict.fromkeys(conditions))
    header = f"{'method':10s} {'priors':20s} " + " ".join(f"{n:>12s}" for n in names)
    print("\nMean joint error (mm)\n" + header + "\n" + "-" * len(header))
    for method, label, joints in rows:
        err = 1000 * np.linalg.norm(joints - gt, axis=2)  # (n_targets, n_joints)
        print(f"{method:10s} {label:20s} " + " ".join(f"{err[conditions == n].mean():12.1f}" for n in names))

    print("\nSpectral transfer, error per joint group (mm), raw -> best priors in the grid")
    raw_err = 1000 * np.linalg.norm(data["spectral"] - gt, axis=2)
    spectral_rows = [r for r in rows if r[0] == "spectral" and r[1] != "-"]
    best = min(spectral_rows, key=lambda r: np.linalg.norm(r[2] - gt, axis=2).mean())
    best_err = 1000 * np.linalg.norm(best[2] - gt, axis=2)
    print(f"best priors: {best[1]}")
    for n in names:
        sel = conditions == n
        groups = "  ".join(
            f"{g}: {raw_err[sel][:, idx].mean():5.1f} -> {best_err[sel][:, idx].mean():5.1f}"
            for g, idx in JOINT_GROUPS.items()
        )
        print(f"{n:12s} {groups}")
    worst = np.argsort(-raw_err.mean(axis=0))[:5]
    print("worst joints (raw spectral):", ", ".join(f"{SMPL_JOINT_NAMES[j]} {raw_err[:, j].mean():.1f}" for j in worst))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smpl", default="data/basicmodel_m_lbs.mat")
    parser.add_argument("-n", type=int, default=8, help="targets per condition")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--pose-std", type=float, default=0.3)
    parser.add_argument("--shape-std", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark_synthetic.npz")
    parser.add_argument("--evaluate", metavar="NPZ", help="only evaluate saved results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    path = args.evaluate or run(args)
    grid = list(itertools.product([0.0, 1.0, 10.0], [0.0, 0.1, 1.0]))[1:]  # skip (0, 0) = raw
    evaluate(path, grid)


if __name__ == "__main__":
    main()
