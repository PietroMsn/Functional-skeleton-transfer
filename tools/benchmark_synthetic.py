"""
Benchmark on synthetic SMPL targets with ground-truth joints.

Random SMPL bodies are generated for every condition (nude, loose shirt,
skirt, ...; see ``skeleton_transfer.synthetic``), the SMPL skeleton is
transferred onto them with several variants of the pipeline (see
``skeleton_transfer.evaluation``), and the joint errors are compared.
Usage, from the repository root::

    python tools/benchmark_synthetic.py --smpl data/basicmodel_m_lbs.mat -n 8 --out bench.pkl

Every stage of every run (initial map, restarts, iterations) is saved with
its unsupervised diagnostics, after every body: an interrupted run continues
with ``--resume``. ``--report bench.pkl`` re-prints the tables.
"""

import argparse
import logging
import os
import pickle

import numpy as np

from skeleton_transfer.evaluation import ALL_LANDMARKS, Evaluator, report
from skeleton_transfer.smpl import SMPLModel
from skeleton_transfer.synthetic import make_synthetic_target

CONDITIONS = {"nude": (), "shirt": ("shirt",), "skirt": ("skirt",), "shirt+skirt": ("shirt", "skirt")}


def run(args):
    model = SMPLModel.from_file(args.smpl)
    evaluator = Evaluator(model.as_source())
    if args.resume and os.path.exists(args.out):
        with open(args.out, "rb") as f:
            evaluator.runs = pickle.load(f)
        print(f"Resuming from {args.out}: {len(evaluator.runs)} runs already done")
    done = {(r["condition"], r["target"]) for r in evaluator.runs}
    for condition in args.conditions:
        rng = np.random.default_rng(args.seed)  # same bodies for every condition
        for i in range(args.n):
            # Always generate the body, so that the random sequence (and thus
            # every following body) is the same when resuming.
            target, gt = make_synthetic_target(
                model, rng, pose_std=args.pose_std, shape_std=args.shape_std, clothing=CONDITIONS[condition],
                n_points=args.n_points, noise=args.noise, landmarks=ALL_LANDMARKS,
            )  # fmt: skip
            name = f"{i + 1}/{args.n}"
            if (condition, name) in done:
                continue
            print(evaluator.evaluate(target, gt, condition, name), flush=True)
            with open(args.out, "wb") as f:  # saved after every body: the run can be interrupted
                pickle.dump(evaluator.runs, f)
    print(f"Saved {args.out}")
    return evaluator.runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smpl", default="data/basicmodel_m_lbs.mat", help="SMPL model (.mat, .pkl or .npz)")
    parser.add_argument("-n", type=int, default=8, help="targets per condition")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--pose-std", type=float, default=0.3)
    parser.add_argument("--shape-std", type=float, default=1.0)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark_synthetic.pkl")
    parser.add_argument("--resume", action="store_true", help="skip the bodies already saved in --out")
    parser.add_argument("--report", metavar="PKL", help="only report saved results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if args.report:
        with open(args.report, "rb") as f:
            runs = pickle.load(f)
    else:
        runs = run(args)
    report(runs)


if __name__ == "__main__":
    main()
