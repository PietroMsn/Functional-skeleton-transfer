"""
Refinement of the correspondence with local CMH functions on body parts,
on real scans with a registered SMPL body (oracle study).

For every scan the map of a pipeline variant (``--initial``) is refined in
several bases, always from the same starting map:

- ``LB``: the Laplace-Beltrami eigenfunctions only (plain ZoomOut iterations);
- ``CMH``: plus the 3 global CMH functions;
- ``local GT``: plus the local CMH functions of the body parts, with the
  part weights of the scan taken from the registered SMPL body (each scan
  point gets the skinning weights of its nearest registered vertex): an
  upper bound for the refinement, which assumes a perfect segmentation;
- ``local``: the same with the part weights of the scan estimated from the
  transferred skeleton, re-estimated after every round of refinement.

The source part weights always come from the SMPL skinning weights. Besides
the joint error, the registration gives the ground-truth correspondence (each
scan point -> its nearest registered vertex, which has the index of the
template vertex): the *map error* is the mean distance, on the source
template, between every source vertex and the ground-truth match of the
target point it is mapped to. The scans are sampled exactly as by
``tools/evaluate_scans.py``. From the repository root::

    python tools/patch_oracle.py samples/4ddress_eval_full --smpl-dir path/to/smpl/models --shard 1/7 --out oracle_s1.pkl
    python tools/patch_oracle.py --report oracle_s*.pkl
"""

import argparse
import glob
import logging
import os
import pickle
import sys
import time
import zlib
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_transfer.evaluation import ALL_LANDMARKS, FAILURE, Evaluator, default_variants  # noqa: E402
from skeleton_transfer.patches import (  # noqa: E402
    joint_to_part, local_cmh_functions, part_weights_from_skeleton, part_weights_from_skinning, refine_local_cmh,
)  # fmt: skip
from skeleton_transfer.scans import load_sample, make_target  # noqa: E402
from skeleton_transfer.smpl import SMPLModel, find_smpl_models  # noqa: E402
from skeleton_transfer.spectral import compute_laplace_basis  # noqa: E402
from skeleton_transfer.transfer import transfer_skeleton  # noqa: E402

INITIAL = {"baseline": "baseline", "complete": "all + priors, refined restarts"}
METHODS = ["initial", "LB", "CMH", "local GT", "local"]


def joints_of(regressor, phi_src, C, phi_tgt, target_basis, target_vertices):
    """Spectral transfer ``R Phi_src C Phi_tgt^T M V`` in the given bases (no skeleton priors)."""
    return regressor @ phi_src @ C @ (phi_tgt.T @ (target_basis.mass @ target_vertices))


def study(evaluator, basis_tgt, target, gt, gt_match, initial, k, n_steps, n_rounds):
    """Errors of every refinement of the map of the ``initial`` variant on one scan."""
    name, extra, cfg = next(v for v in default_variants() if v[0] == INITIAL[initial])
    src = evaluator.source_extra if extra else evaluator.source
    tgt = target if extra else replace(target, landmarks=target.landmarks[: len(evaluator.source.landmarks)])
    t = time.time()
    result = transfer_skeleton(src, tgt, cfg, regressor=evaluator.regressor, basis_src=evaluator.basis_src,
                               basis_tgt=basis_tgt)  # fmt: skip
    seconds = {"initial": time.time() - t}
    R, source = evaluator.regressor, evaluator.source
    basis_src = evaluator.basis_src
    source_vertices = source.vertices

    def errors(joints, p2p):
        return dict(error=float(np.linalg.norm(joints - gt, axis=1).mean()),
                    pointwise=float(np.linalg.norm(R @ target.vertices[p2p] - gt, axis=1).mean()),
                    map=float(np.linalg.norm(source_vertices[gt_match[p2p]] - source_vertices, axis=1).mean()))  # fmt: skip

    phi_lb_src, phi_lb_tgt = basis_src.evecs[:, :k], basis_tgt.evecs[:, :k]
    C0 = result.C[:k, :k]
    raw = joints_of(R, phi_lb_src, C0, phi_lb_tgt, basis_tgt, target.vertices)
    out = {"initial": dict(errors(raw, result.p2p), final=float(np.linalg.norm(result.joints_spectral - gt, axis=1).mean()))}

    part = joint_to_part()
    weights_src = part_weights_from_skinning(source.skinning_weights, part)
    bases = {
        "LB": (phi_lb_src, phi_lb_tgt),
        "CMH": (basis_src.functions(k, cmh=True), basis_tgt.functions(k, cmh=True)),
        "local GT": (local_cmh_functions(basis_src, k, weights_src),
                     local_cmh_functions(basis_tgt, k, part_weights_from_skinning(source.skinning_weights[gt_match], part))),
    }  # fmt: skip
    for method, (phi_src, phi_tgt) in bases.items():
        t = time.time()
        C, p2p = refine_local_cmh(result.p2p, phi_src, phi_tgt, n_steps)
        out[method] = errors(joints_of(R, phi_src, C, phi_tgt, basis_tgt, target.vertices), p2p)
        seconds[method] = time.time() - t

    # Estimated parts: skeleton -> part weights -> map -> skeleton ...
    t = time.time()
    phi_src = bases["local GT"][0]
    joints, p2p, rounds = raw, result.p2p, []
    for _ in range(n_rounds):
        phi_tgt = local_cmh_functions(basis_tgt, k, part_weights_from_skeleton(target.vertices, joints, source.parents, part))
        C, p2p = refine_local_cmh(p2p, phi_src, phi_tgt, n_steps)
        joints = joints_of(R, phi_src, C, phi_tgt, basis_tgt, target.vertices)
        rounds.append(errors(joints, p2p))
    out["local"] = dict(rounds[-1], rounds=rounds)
    seconds["local"] = time.time() - t
    return out, seconds


def report(rows):
    """Mean (median) errors in mm per initial map, method and condition."""
    conditions = sorted({r["condition"] for r in rows})
    for initial in INITIAL:
        sel = [r for r in rows if r["initial"] == initial]
        if not sel:
            continue
        print(f"\nInitial map: {initial} ({len(sel)} scans); mean (median) mm, failures > {1000 * FAILURE:.0f} mm")
        print(f"{'method':10s}" + "".join(f"{c:>16s}" for c in conditions) + f"{'all':>16s} fail {'pointwise':>16s} {'map':>14s}")
        for m in METHODS:
            e = lambda rs, key="error": np.array([r["methods"][m][key] for r in rs])  # noqa: E731
            cells = "".join(f"{1000 * e([r for r in sel if r['condition'] == c]).mean():8.1f} ({1000 * np.median(e([r for r in sel if r['condition'] == c])):5.1f})" for c in conditions)  # fmt: skip
            a = e(sel)
            print(f"{m:10s}{cells}{1000 * a.mean():8.1f} ({1000 * np.median(a):5.1f}) {int((a > FAILURE).sum()):4d} "
                  f"{1000 * e(sel, 'pointwise').mean():8.1f} ({1000 * np.median(e(sel, 'pointwise')):5.1f}) "
                  f"{1000 * e(sel, 'map').mean():6.1f} ({1000 * np.median(e(sel, 'map')):5.1f})")  # fmt: skip
        final = np.array([r["methods"]["initial"]["final"] for r in sel])
        print(f"(the full pipeline, with its skeleton priors: {1000 * final.mean():.1f} ({1000 * np.median(final):.1f}) mm)")
        for m in METHODS[1:]:
            d = np.array([r["methods"][m]["error"] - r["methods"]["initial"]["error"] for r in sel])
            print(f"  {m:9s} vs initial: better {int((d < -1e-3).sum())}, worse {int((d > 1e-3).sum())}")
        rounds = np.array([[x["error"] for x in r["methods"]["local"]["rounds"]] for r in sel])
        print("  local, error per round: " + ", ".join(f"{1000 * x:.1f}" for x in rounds.mean(axis=0)))
        print("  seconds: " + ", ".join(f"{m} {np.mean([r['seconds'][m] for r in sel]):.1f}" for m in METHODS))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", nargs="?", help="folder with the exported .npz samples (searched recursively)")
    parser.add_argument("--source", default="data/basicmodel_m_lbs.mat", help="SMPL model used as source")
    parser.add_argument("--smpl-dir", help="folder with the official SMPL models (ground-truth joint regressors)")
    parser.add_argument("--initial", nargs="+", default=list(INITIAL), choices=list(INITIAL))
    parser.add_argument("--steps", type=int, default=5, help="ZoomOut-like iterations of every refinement")
    parser.add_argument("--rounds", type=int, default=3, help="skeleton -> parts -> map rounds with estimated parts")
    parser.add_argument("--n-points", type=int, default=8000)
    parser.add_argument("--max-distance", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", metavar="I/N")
    parser.add_argument("--out", default="patch_oracle.pkl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", nargs="+", metavar="PKL", help="only report saved results (several files are merged)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)

    if args.report:
        rows = {}
        for pattern in args.report:
            for path in glob.glob(pattern):
                with open(path, "rb") as f:
                    rows.update({(r["target"], r["initial"]): r for r in pickle.load(f)})
        report(list(rows.values()))
        return
    if not args.samples:
        parser.error("the samples folder is required")

    source_model = SMPLModel.from_file(args.source)
    regressors = {"neutral": source_model.J_regressor}
    for gender, path in (find_smpl_models(args.smpl_dir) if args.smpl_dir else {}).items():
        regressors[gender] = SMPLModel.from_file(path).J_regressor
    files = sorted(glob.glob(os.path.join(args.samples, "**", "*.npz"), recursive=True))
    if args.shard:
        index, count = map(int, args.shard.split("/"))
        files = files[index - 1 :: count]
    rows = []
    if args.resume and os.path.exists(args.out):
        with open(args.out, "rb") as f:
            rows = pickle.load(f)
    done = {(r["target"], r["initial"]) for r in rows}
    evaluator = Evaluator(source_model.as_source())
    k = evaluator.config.functional_map.n_eigs_final
    print(f"{len(files)} samples", flush=True)
    for path in files:
        sample = load_sample(path)
        todo = [i for i in args.initial if (sample.name, i) not in done]
        if not todo:
            continue
        rng = np.random.default_rng([args.seed, zlib.crc32(sample.name.encode())])  # as tools/evaluate_scans.py
        regressor = regressors.get(sample.gender, regressors["neutral"])
        target, gt = make_target(sample, regressor, args.n_points, ALL_LANDMARKS, args.max_distance, rng)
        gt_match = cKDTree(sample.smpl_vertices).query(target.vertices)[1]
        basis_tgt = compute_laplace_basis(target.vertices, None, evaluator.config.spectral.n_eigs, True)
        for initial in todo:
            methods, seconds = study(evaluator, basis_tgt, target, gt, gt_match, initial, k, args.steps, args.rounds)
            rows.append(dict(target=sample.name, condition=sample.category, initial=initial, methods=methods, seconds=seconds))
            print(f"{sample.name} {initial:8s} " + " | ".join(f"{m} {1000 * methods[m]['error']:6.1f}" for m in METHODS), flush=True)
            with open(args.out, "wb") as f:
                pickle.dump(rows, f)
    report(rows)



if __name__ == "__main__":
    main()
