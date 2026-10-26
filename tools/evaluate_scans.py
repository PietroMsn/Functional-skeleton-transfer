"""
Evaluate the skeleton transfer on real scans of clothed people.

The scans must first be exported to the common sample format of
``skeleton_transfer.scans`` (one ``.npz`` per frame with the scan, the
registered SMPL body, the gender and a category), by the dataset adapters in
``tools/datasets/``. Then, from the repository root::

    python tools/evaluate_scans.py samples/ --source data/basicmodel_m_lbs.mat \\
        --smpl-male SMPL_MALE.pkl --smpl-female SMPL_FEMALE.pkl --out scans.pkl

The source is the SMPL template (as in the paper); the ``--smpl-*`` models
only provide the gender-specific joint regressors used for the ground truth
when a sample does not store its joints. Results are reported per category
(``--group-by`` can also group by gender or by the first path components of
the sample names), with the same tables as the synthetic benchmark.
"""

import argparse
import glob
import logging
import os
import pickle

import numpy as np

from skeleton_transfer.evaluation import ALL_LANDMARKS, Evaluator, report
from skeleton_transfer.scans import load_sample, make_target
from skeleton_transfer.smpl import SMPLModel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="folder with the exported .npz samples (searched recursively)")
    parser.add_argument("--source", default="data/basicmodel_m_lbs.mat", help="SMPL model used as source")
    for gender in ("male", "female", "neutral"):
        parser.add_argument(f"--smpl-{gender}", help=f"SMPL {gender} model (for its joint regressor)")
    parser.add_argument("--n-points", type=int, default=8000, help="points sampled on each scan")
    parser.add_argument("--max-distance", type=float, default=0.3, help="drop scan points farther from the body (m)")
    parser.add_argument("--limit", type=int, help="evaluate at most this many samples")
    parser.add_argument("--group-by", choices=["category", "gender"], default="category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="evaluate_scans.pkl")
    parser.add_argument("--report", metavar="PKL", help="only report saved results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if args.report:
        with open(args.report, "rb") as f:
            report(pickle.load(f))
        return

    source_model = SMPLModel.from_file(args.source)
    regressors = {"neutral": source_model.J_regressor}
    for gender in ("male", "female", "neutral"):
        path = getattr(args, f"smpl_{gender}")
        if path:
            regressors[gender] = SMPLModel.from_file(path).J_regressor

    paths = sorted(glob.glob(os.path.join(args.samples, "**", "*.npz"), recursive=True))[: args.limit]
    print(f"{len(paths)} samples")
    evaluator = Evaluator(source_model.as_source())
    rng = np.random.default_rng(args.seed)
    for path in paths:
        sample = load_sample(path)
        if sample.joints is None and sample.gender not in regressors:
            logging.warning("No %s SMPL model given, using the source regressor for %s", sample.gender, sample.name)
        regressor = regressors.get(sample.gender, regressors["neutral"])
        target, gt = make_target(sample, regressor, args.n_points, ALL_LANDMARKS, args.max_distance, rng)
        group = sample.category if args.group_by == "category" else sample.gender
        print(evaluator.evaluate(target, gt, group or "all", sample.name), flush=True)
        with open(args.out, "wb") as f:  # saved after every sample: long runs can be interrupted
            pickle.dump(evaluator.runs, f)
    report(evaluator.runs)


if __name__ == "__main__":
    main()
