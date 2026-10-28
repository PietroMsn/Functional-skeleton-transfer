"""
Baseline on real scans: fit SMPL to each scan and take the joints of the fit.

Uses the same targets as ``tools/evaluate_scans.py`` (same sampled points,
same landmarks, same random seed per sample), so the errors are directly
comparable. From the repository root::

    python tools/fit_smpl_baseline.py samples/4ddress_eval --model data/basicmodel_m_lbs.mat \\
        --out fit_smpl.pkl
    python tools/fit_smpl_baseline.py --compare fit_smpl.pkl --with eval_s*.pkl

The fitted model is the neutral SMPL by default (the gender is not assumed,
as for the transfer); ``--smpl-dir`` with ``--gender`` fits the model of the
subject gender instead.
"""

import argparse
import glob
import os
import pickle
import time
import zlib

import numpy as np

from skeleton_transfer.evaluation import ALL_LANDMARKS, FAILURE
from skeleton_transfer.scans import load_sample, make_target
from skeleton_transfer.smpl import SMPLModel, find_smpl_models
from skeleton_transfer.smpl_fitting import fit_smpl

#: Landmarks (indices into ALL_LANDMARKS) for the initial rigid alignment:
#: back and front of the torso, head, feet, knees.
ALIGNMENT_LANDMARKS = np.array([0, 1, 2, 4, 5, 7, 8])
VARIANT = "SMPL fitting"


def compare(paths, others, reference):
    runs = {}
    for path in others:
        with open(path, "rb") as f:
            runs.update({(r["target"], r["variant"]): r for r in pickle.load(f)})
    for path in paths:
        with open(path, "rb") as f:
            runs.update({(r["target"], r["variant"]): r for r in pickle.load(f)})
    fits = [r for r in runs.values() if r["variant"] == VARIANT]
    names = [VARIANT] + reference
    print(f"{'variant':34s}{'condition':>10s}{'mean':>8s}{'median':>8s}{'failures':>10s}")
    for condition in sorted({r["condition"] for r in fits}) + ["all"]:
        targets = [r["target"] for r in fits if condition in ("all", r["condition"])]
        for name in names:
            if all((t, name) in runs for t in targets):
                e = 1000 * np.array([runs[(t, name)]["error"] for t in targets])
                print(f"{name:34s}{condition:>10s}{e.mean():8.1f}{np.median(e):8.1f}{(e > 1000 * FAILURE).sum():>6d}/{len(e)}")
    for name in reference:
        targets = [r["target"] for r in fits if (r["target"], name) in runs]
        d = np.array([runs[(t, name)]["error"] - runs[(t, VARIANT)]["error"] for t in targets])
        print(f"{name}: better than the fitting on {(d < 0).sum()}/{len(d)} scans")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", nargs="?", help="folder with the exported .npz samples (searched recursively)")
    parser.add_argument("--model", default="data/basicmodel_m_lbs.mat", help="SMPL model to fit (neutral)")
    parser.add_argument("--smpl-dir", help="official SMPL models, used with --gender")
    parser.add_argument("--gender", action="store_true", help="fit the model of the subject gender")
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--max-distance", type=float, default=0.3)
    parser.add_argument("--shard", metavar="I/N", help="only the I-th of N interleaved parts of the samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="fit_smpl.pkl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare", metavar="PKL", nargs="+", help="only compare saved fits with saved transfers")
    parser.add_argument("--with", dest="others", metavar="PKL", nargs="*", default=[], help="saved transfer runs")
    parser.add_argument("--reference", nargs="*", default=["baseline", "all + priors, refined restarts"])
    args = parser.parse_args()

    if args.compare:
        compare(args.compare, args.others, args.reference)
        return
    if not args.samples:
        parser.error("the samples folder is required")

    models = {"neutral": SMPLModel.from_file(args.model)}
    if args.gender:
        for gender, path in find_smpl_models(args.smpl_dir).items():
            models[gender] = SMPLModel.from_file(path)
    files = sorted(glob.glob(os.path.join(args.samples, "**", "*.npz"), recursive=True))
    if args.shard:
        index, count = map(int, args.shard.split("/"))
        files = files[index - 1 :: count]
    runs = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, "rb") as f:
            runs = pickle.load(f)
    done = {r["target"] for r in runs}
    for path in files:
        sample = load_sample(path)
        if sample.name in done:
            continue
        rng = np.random.default_rng([args.seed, zlib.crc32(sample.name.encode())])  # as in evaluate_scans.py
        target, gt = make_target(sample, models["neutral"].J_regressor, args.n_points, ALL_LANDMARKS,
                                 args.max_distance, rng)  # fmt: skip
        model = models.get(sample.gender, models["neutral"]) if args.gender else models["neutral"]
        t = time.time()
        fit = fit_smpl(model, target.vertices, ALL_LANDMARKS, target.landmarks, ALIGNMENT_LANDMARKS)
        err = np.linalg.norm(fit.joints - gt, axis=1)
        runs.append(dict(condition=sample.category, target=sample.name, variant=VARIANT, gt=gt, joints=fit.joints,
                         error=float(err.mean()), scan_to_body=fit.scan_to_body, pose=fit.pose, betas=fit.betas,
                         transl=fit.transl, seconds=time.time() - t))  # fmt: skip
        print(f"{sample.category:8s} {sample.name:40s} {1000 * err.mean():6.1f} mm  {time.time() - t:5.1f}s", flush=True)
        with open(args.out, "wb") as f:
            pickle.dump(runs, f)


if __name__ == "__main__":
    main()
