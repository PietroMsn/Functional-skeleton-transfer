"""
Benchmark of the skeleton transfer on the unified body dataset.

The dataset (built by the ``unified-body-dataset`` project) holds real scans
(4D-DRESS, CloSe-Di, BUFF) and synthetic bodies (adults, kids, clothed,
animals), one ``.npz`` per sample in the format of ``skeleton_transfer.scans``,
listed in ``manifest.csv`` with their provenance and split. The source is the
SMPL template for the humans (kids included: an adult template on a child)
and the SMAL template for the animals. From the repository root::

    python tools/benchmark_unified.py DATASET_ROOT --split dev --smal path/to/smal_CVPR2017.pkl \\
        --shard 1/8 --out unified_s1.pkl --resume
    python tools/benchmark_unified.py --report unified_s*.pkl

Results are grouped by ``dataset/category``. The errors are in millimetres
and also relative to the mean ground-truth bone length (kids and animals have
other sizes than adults); a failure is a mean error above 10 cm.
"""

import argparse
import csv
import logging
import os
import pickle
import zlib
from collections import defaultdict

import numpy as np

from skeleton_transfer.evaluation import ALL_LANDMARKS, FAILURE, Evaluator, default_variants, map_variants, report
from skeleton_transfer.scans import load_sample, make_target
from skeleton_transfer.shapes import SMAL_EXTRA_LANDMARKS, SMAL_LANDMARKS
from skeleton_transfer.skeleton_priors import bone_lengths
from skeleton_transfer.smpl import SMPLModel

DEFAULT_VARIANTS = [
    "baseline",
    "all + priors, per-chain restarts",
    "all + priors, per-chain restarts, loose clothing",
    "all + priors, per-chain restarts, loose clothing, Voronoi",
]


def read_manifest(root, split, datasets):
    with open(os.path.join(root, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split and (not datasets or r["dataset"] in datasets)]
    # round robin over the groups, so that any prefix of a run covers all of them
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["category"])].append(r)
    ordered = []
    for i in range(max(map(len, groups.values()))):
        ordered += [g[i] for _, g in sorted(groups.items()) if i < len(g)]
    return ordered


def summary(runs):
    """Mean / median error (mm), relative error and failures per group and variant."""
    names = list(dict.fromkeys(r["variant"] for r in runs))
    conditions = sorted(dict.fromkeys(r["condition"] for r in runs))
    datasets = sorted({c.split("/")[0] for c in conditions})
    rows = [(c, lambda r, c=c: r["condition"] == c) for c in conditions]
    rows += [(f"{d} (all)", lambda r, d=d: r["condition"].split("/")[0] == d) for d in datasets]
    rows += [("ALL humans", lambda r: not r["condition"].startswith("synthetic_animals")), ("ALL", lambda r: True)]
    width = max(len(n) for n in names)
    print(f"\nMean (median) joint error in mm, error / mean bone length in %, failures (> {1000 * FAILURE:.0f} mm)")
    for label, keep in rows:
        print(f"\n{label}")
        for name in names:
            sel = [r for r in runs if r["variant"] == name and keep(r)]
            if not sel:
                continue
            err = np.array([r["error"] for r in sel])
            rel = np.array([r["error"] / bone_lengths(r["gt"], r["parents"]).mean() for r in sel])
            print(f"  {name:{width}s} n={len(sel):4d}  {1000 * err.mean():6.1f} ({1000 * np.median(err):5.1f})"
                  f"  {100 * rel.mean():5.1f}%  fail {np.sum(err > FAILURE):3d}")  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", help="root of the unified dataset (with manifest.csv)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--datasets", nargs="+", help="only these datasets (e.g. 4ddress synthetic_animals)")
    parser.add_argument("--source", default="data/basicmodel_m_lbs.mat", help="SMPL model used as source of the humans")
    parser.add_argument("--smal", help="SMAL model (smal_CVPR2017.pkl), source of the animals")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="names from evaluation.default_variants, or '... loose clothing' with ', geodesic' "
                        "or ', geodesic + Voronoi' (evaluation.map_variants)")  # fmt: skip
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--max-distance", type=float, default=0.3)
    parser.add_argument("--limit", type=int, help="only the first N samples (they cover all the groups)")
    parser.add_argument("--shard", metavar="I/N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark_unified.pkl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", metavar="PKL", nargs="+", help="only report saved results (several files are merged)")
    parser.add_argument("--full-report", action="store_true", help="with --report, also the tables of evaluate_scans")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if args.report:
        runs = {}
        for path in args.report:
            with open(path, "rb") as f:
                runs.update({(r["target"], r["variant"]): r for r in pickle.load(f)})
        summary(list(runs.values()))
        if args.full_report:
            report(list(runs.values()))
        return
    if not args.root:
        parser.error("the dataset root is required")

    # map_variants also rebuilds the last default variant (Voronoi), with the same name and config
    pool = {v[0]: v for v in default_variants() + map_variants(default_variants()[-2:-1])}
    unknown = set(args.variants) - set(pool)
    if unknown:
        parser.error(f"unknown variants: {', '.join(sorted(unknown))}")
    variants = [pool[name] for name in args.variants]

    rows = read_manifest(args.root, args.split, args.datasets)[: args.limit]
    if args.shard:
        index, count = map(int, args.shard.split("/"))
        rows = rows[index - 1 :: count]
    print(f"{len(rows)} samples")

    sources = {}  # body model -> (evaluator, target landmarks, regressor)

    def source_for(body_model):
        if body_model not in sources:
            if body_model == "smal":
                if not args.smal:
                    raise SystemExit("--smal is required for the animals")
                model = SMPLModel.from_file(args.smal)
                sources[body_model] = (
                    Evaluator(model.as_source(SMAL_LANDMARKS), variants, SMAL_EXTRA_LANDMARKS),
                    np.concatenate((SMAL_LANDMARKS, SMAL_EXTRA_LANDMARKS)),
                    model.J_regressor,
                )
            else:  # smpl, smpl_kid: same topology
                model = SMPLModel.from_file(args.source)
                sources[body_model] = (Evaluator(model.as_source(), variants), ALL_LANDMARKS, model.J_regressor)
        return sources[body_model]

    saved = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, "rb") as f:
            saved = pickle.load(f)
        print(f"Resuming from {args.out}: {len(saved)} runs already done")
    names = {name for name, _, _ in variants}
    have = defaultdict(set)
    for r in saved:
        have[r["target"]].add(r["variant"])

    for row in rows:
        sample = load_sample(os.path.join(args.root, row["file"]))
        if names <= have[sample.name]:
            continue
        body_model = "smal" if row["dataset"] == "synthetic_animals" else "smpl"
        evaluator, landmarks, regressor = source_for(body_model)
        evaluator.runs = [r for r in saved if r["target"] == sample.name]
        rng = np.random.default_rng([args.seed, zlib.crc32(sample.name.encode())])
        target, gt = make_target(sample, regressor, args.n_points, landmarks, args.max_distance, rng)
        condition = f"{row['dataset']}/{row['category']}"
        print(evaluator.evaluate(target, gt, condition, sample.name), flush=True)
        for r in evaluator.runs:
            r.update(file=row["file"], parents=evaluator.source.parents)
        saved = [r for r in saved if r["target"] != sample.name] + evaluator.runs
        with open(args.out, "wb") as f:  # saved after every sample: long runs can be interrupted
            pickle.dump(saved, f)
    summary(saved)


if __name__ == "__main__":
    main()
