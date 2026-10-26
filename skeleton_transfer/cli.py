"""
A Functional Skeleton Transfer - command line entry point.

Transfers the SMPL skeleton onto a target shape and shows the result.

Examples
--------
Run on the example data (``data/`` folder)::

    skeleton-transfer

Run on another target, save the joints and skip the viewer::

    skeleton-transfer --target data/my_shape.mat --output joints.npz --no-plot

Any mesh format, with the 7 landmark indices in a text file::

    skeleton-transfer --target scan.obj --target-landmarks scan_landmarks.txt

Target point cloud (no faces)::

    skeleton-transfer --target data/my_scan.mat --target-point-cloud

``python main.py`` from the repository root is equivalent.
"""

import argparse
import logging

import numpy as np

from . import SkeletonPriorConfig, TransferConfig, load_smpl_source, load_target, transfer_skeleton


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/basicmodel_m_lbs.mat", help="SMPL model in .mat format")
    parser.add_argument("--target", default="data/50007_knees_001806.mat", help="target shape: .mat file or mesh (.obj, .off, .ply, ...)")
    parser.add_argument(
        "--target-landmarks",
        help="text file with the target landmark vertex indices, same order as the source ones "
        "(required for mesh files, overrides the ones stored in a .mat file)",
    )
    parser.add_argument("--one-based-landmarks", action="store_true", help="landmark indices start from 1")
    parser.add_argument(
        "--target-point-cloud", action="store_true", help="treat the target as a point cloud (ignore faces)"
    )
    parser.add_argument(
        "--priors", action="store_true", help="apply left/right symmetry and proportion priors to the joints"
    )
    parser.add_argument("--output", help="save joints, maps and regressor to this .npz file")
    parser.add_argument("--no-plot", action="store_true", help="do not open the 3D viewer")
    parser.add_argument("--screenshot", help="save the 3D view to this image instead of opening a window")
    parser.add_argument(
        "--plot-results",
        metavar="NPZ",
        help="skip the computation and only plot results previously saved with --output",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only print warnings and errors")
    return parser.parse_args(argv)


def _load_target(args):
    return load_target(
        args.target,
        landmarks=args.target_landmarks,
        one_based_landmarks=args.one_based_landmarks,
        point_cloud=args.target_point_cloud,
    )


def plot_saved_results(args):
    """Plot a ``.npz`` file written by ``--output`` (no computation)."""
    from .transfer import TransferResult
    from .visualization import plot_transfer

    source = load_smpl_source(args.source)
    target = _load_target(args)
    saved = np.load(args.plot_results)
    result = TransferResult(
        joints_spectral=saved["joints_spectral"],
        joints_pointwise=saved["joints_pointwise"],
        regressor=saved["regressor"],
        mask=None,
        C=saved["C"],
        p2p=saved["p2p"],
    )
    plot_transfer(source, target, result, screenshot=args.screenshot)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    if args.plot_results:
        plot_saved_results(args)
        return

    config = TransferConfig(
        target_is_point_cloud=args.target_point_cloud,
        skeleton_priors=SkeletonPriorConfig() if args.priors else None,
    )
    # Every parameter can be changed here, e.g.:
    # config.functional_map.n_eigs_final = 80
    # config.regressor.skinning_weight_threshold = 0.2

    source = load_smpl_source(args.source)
    target = _load_target(args)

    result = transfer_skeleton(source, target, config)

    if args.output:
        np.savez(
            args.output,
            joints_spectral=result.joints_spectral,
            joints_pointwise=result.joints_pointwise,
            parents=source.parents,
            C=result.C,
            p2p=result.p2p,
            regressor=result.regressor,
        )
        logging.info("Results saved to %s", args.output)

    if not args.no_plot or args.screenshot:
        from .visualization import plot_transfer

        plot_transfer(source, target, result, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
