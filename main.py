"""
A Functional Skeleton Transfer - command line entry point.

Transfers the SMPL skeleton onto a target shape and shows the result.

Examples
--------
Run on the example data shipped with the repository::

    python main.py

Run on another target, save the joints and skip the viewer::

    python main.py --target data/my_shape.mat --output joints.npz --no-plot

Target point cloud (no faces)::

    python main.py --target data/my_scan.mat --target-point-cloud
"""

import argparse

import numpy as np

from skeleton_transfer import TransferConfig, load_smpl_source, load_target_mat, transfer_skeleton


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/basicmodel_m_lbs.mat", help="SMPL model in .mat format")
    parser.add_argument("--target", default="data/50007_knees_001806.mat", help="target shape in .mat format")
    parser.add_argument(
        "--target-point-cloud", action="store_true", help="treat the target as a point cloud (ignore faces)"
    )
    parser.add_argument("--output", help="save joints, maps and regressor to this .npz file")
    parser.add_argument("--no-plot", action="store_true", help="do not open the 3D viewer")
    parser.add_argument("--screenshot", help="save the 3D view to this image instead of opening a window")
    parser.add_argument(
        "--plot-results",
        metavar="NPZ",
        help="skip the computation and only plot results previously saved with --output",
    )
    return parser.parse_args()


def plot_saved_results(args):
    """Plot a ``.npz`` file written by ``--output`` (no TensorFlow needed)."""
    from skeleton_transfer.transfer import TransferResult
    from skeleton_transfer.visualization import plot_transfer

    source = load_smpl_source(args.source)
    target = load_target_mat(args.target, faces_key=None if args.target_point_cloud else "faces")
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


def main():
    args = parse_args()

    if args.plot_results:
        plot_saved_results(args)
        return

    config = TransferConfig(target_is_point_cloud=args.target_point_cloud)
    # Every parameter can be changed here, e.g.:
    # config.functional_map.n_eigs_final = 80
    # config.regressor.n_iterations = 500

    source = load_smpl_source(args.source)
    target = load_target_mat(args.target, faces_key=None if args.target_point_cloud else "faces")

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
        print(f"Results saved to {args.output}")

    if not args.no_plot or args.screenshot:
        from skeleton_transfer.visualization import plot_transfer

        plot_transfer(source, target, result, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
