"""
Evaluate the skeleton transfer on real scans of clothed people.

The scans must first be exported to the common sample format of
``skeleton_transfer.scans`` (one ``.npz`` per frame with the scan, the
registered SMPL body, the gender and a category), by the dataset adapters in
``tools/datasets/``. Then, from the repository root::

    python tools/evaluate_scans.py samples/ --source data/basicmodel_m_lbs.mat \\
        --smpl-dir path/to/SMPL_python_v.1.1.0/smpl/models --out scans.pkl

The source is the SMPL neutral template, as in the paper (the example file
``data/basicmodel_m_lbs.mat`` is the neutral model, despite its name). The
gender-specific models, found in ``--smpl-dir`` or given one by one with
``--smpl-male`` etc., only provide the joint regressors used for the ground
truth when a sample does not store its joints. Results are reported per
category (or per gender, with ``--group-by gender``), with the same tables as
the synthetic benchmark.
"""

import argparse
import glob
import logging
import os
import pickle
import zlib

import numpy as np

from skeleton_transfer.evaluation import ALL_LANDMARKS, Evaluator, completed_targets, report
from skeleton_transfer.scans import load_sample, make_target
from skeleton_transfer.smpl import SMPLModel, find_smpl_models


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="folder with the exported .npz samples (searched recursively)")
    parser.add_argument("--source", default="data/basicmodel_m_lbs.mat", help="SMPL model used as source")
    parser.add_argument("--smpl-dir", help="folder with the official SMPL models (male, female, neutral)")
    for gender in ("male", "female", "neutral"):
        parser.add_argument(f"--smpl-{gender}", help=f"SMPL {gender} model (overrides --smpl-dir)")
    parser.add_argument("--n-points", type=int, default=8000, help="points sampled on each scan")
    parser.add_argument("--max-distance", type=float, default=0.3, help="drop scan points farther from the body (m)")
    parser.add_argument("--limit", type=int, help="evaluate at most this many samples")
    parser.add_argument("--group-by", choices=["category", "gender"], default="category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="evaluate_scans.pkl")
    parser.add_argument("--resume", action="store_true", help="skip the runs already saved in --out (only missing variants are run)")
    parser.add_argument("--report", metavar="PKL", help="only report saved results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if args.report:
        with open(args.report, "rb") as f:
            report(pickle.load(f))
        return

    source_model = SMPLModel.from_file(args.source)
    regressors = {"neutral": source_model.J_regressor}
    models = find_smpl_models(args.smpl_dir) if args.smpl_dir else {}
    for gender in ("male", "female", "neutral"):
        path = getattr(args, f"smpl_{gender}") or models.get(gender)
        if path:
            print(f"SMPL {gender}: {path}")
            regressors[gender] = SMPLModel.from_file(path).J_regressor

    files = sorted(glob.glob(os.path.join(args.samples, "**", "*.npz"), recursive=True))[: args.limit]
    print(f"{len(files)} samples")
    evaluator = Evaluator(source_model.as_source())
    if args.resume and os.path.exists(args.out):
        with open(args.out, "rb") as f:
            evaluator.runs = pickle.load(f)
        print(f"Resuming from {args.out}: {len(evaluator.runs)} runs already done")
    done = {name for _, name in completed_targets(evaluator.runs, evaluator.variants)}
    for path in files:
        sample = load_sample(path)
        if sample.name in done:
            continue
        # one generator per sample: the sampled points do not depend on which samples run before
        rng = np.random.default_rng([args.seed, zlib.crc32(sample.name.encode())])
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
