"""
Build an evaluation subset with one frame per sequence and varied poses.

The frames exported from a sequence are consecutive poses of the same motion,
so a test set keeps one frame per sequence (subject, outfit, take). Among the
exported frames of a sequence it picks the pose farthest from the poses
already selected (farthest-point sampling on the SMPL body pose, root
excluded), so that the subset covers as many different poses as possible.
Frames of an existing subset (``--keep``) are kept as they are and count as
already selected:

    python tools/datasets/select_frames.py samples/4ddress --out samples/4ddress_eval \\
        --keep samples/4ddress_1pertake

The chosen samples are copied to ``--out``, which receives a ``manifest.csv``
with the provenance of every sample (see ``manifest.py``).
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skeleton_transfer.scans import load_sample  # noqa: E402
from manifest import manifest_rows, summary, write_manifest  # noqa: E402


def body_pose(metadata: dict) -> np.ndarray:
    """(69,) axis-angle rotations of the 23 non-root joints."""
    return np.ravel(metadata["smpl_params"]["body_pose"]).astype(float)


def select_frames(samples: dict, kept: list) -> list:
    """Pick one file per sequence by farthest-point sampling on the body pose.

    Parameters
    ----------
    samples : {sequence: [(file, pose), ...]} candidate frames
    kept : [(sequence, file, pose)] frames already selected

    Returns the files chosen for the sequences not in ``kept``, in order.
    """
    chosen = [pose for _, _, pose in kept]
    done = {sequence for sequence, _, _ in kept}
    selected = []
    for sequence in sorted(samples):
        if sequence in done:
            continue
        candidates = samples[sequence]
        if chosen:
            score = [min(np.linalg.norm(pose - c) for c in chosen) for _, pose in candidates]
            best = int(np.argmax(score))
        else:
            best = len(candidates) // 2  # first sequence: its middle frame
        selected.append(candidates[best][0])
        chosen.append(candidates[best][1])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("samples", help="folder with the exported samples")
    parser.add_argument("--out", required=True, help="folder of the subset")
    parser.add_argument("--keep", help="existing subset whose frames are kept")
    args = parser.parse_args()

    def sequence(metadata):
        return (metadata["subject"], metadata["outfit"], metadata["take"])

    samples = {}
    for path in sorted(glob.glob(os.path.join(args.samples, "*.npz"))):
        m = load_sample(path).metadata
        samples.setdefault(sequence(m), []).append((path, body_pose(m)))
    kept = []
    if args.keep:
        for path in sorted(glob.glob(os.path.join(args.keep, "*.npz"))):
            m = load_sample(path).metadata
            kept.append((sequence(m), path, body_pose(m)))

    os.makedirs(args.out, exist_ok=True)
    files = [path for _, path, _ in kept] + select_frames(samples, kept)
    for path in files:
        target = os.path.join(args.out, os.path.basename(path))
        if not os.path.exists(target):
            shutil.copy2(path, target)
    manifest = write_manifest(args.out)
    print(f"{len(files)} samples in {args.out} ({len(kept)} kept), provenance in {manifest}")
    print(summary(manifest_rows(args.out)))


if __name__ == "__main__":
    main()
