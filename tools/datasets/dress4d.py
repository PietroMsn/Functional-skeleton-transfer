"""
Export a subset of 4D-DRESS frames to the common sample format.

4D-DRESS (Wang et al. 2024) is distributed as one ``.tar.gz`` per subject and
outfit (``00123_Inner.tar.gz``, ``00127_Outer.tar.gz``, ... 7-15 GB each),
mostly filled with camera images and textures. This script extracts only a
few frames per sequence and converts each of them to a
``skeleton_transfer.scans.ScanSample`` (scan + registered SMPL body + exact
ground-truth joints + semantic labels), a few MB per frame:

    python tools/datasets/dress4d.py "D:/4dDress/00123_Outer.tar.gz" "D:/4dDress/00123_Inner" \\
        --smpl-dir path/to/SMPL_python_v.1.1.0/smpl/models --out samples/4ddress

Sources can be archives (read as a stream: nothing is extracted to disk) or
folders already extracted (``<subject>_<outfit>/<Outfit>/Take*/``). Frames are
chosen deterministically: frame numbers ``1, 1 + stride, 1 + 2 stride, ...``,
at most ``--frames-per-take`` per sequence.

Files read for every chosen frame of a take (``Take*/``):

- ``basic_info.pkl``: ``gender`` (selects the SMPL model) and ``rotation``;
- ``Meshes_pkl/mesh-fXXXXX.pkl``: scan ``vertices`` and ``faces``;
- ``SMPL/mesh-fXXXXX_smpl.pkl``: SMPL ``betas``, ``global_orient``,
  ``body_pose``, ``transl``; the ground-truth joints are obtained by forward
  kinematics (they reproduce the released ``_smpl.ply`` registrations exactly);
- ``Semantic/labels/label-fXXXXX.pkl``: per-vertex ``scan_labels``
  (skin 0, hair 1, shoe 2, upper 3, lower 4, outer 5), stored with the sample.

The category of a sample is its outfit (``Inner`` / ``Outer``); the fraction of
scan vertices of each label is stored in the metadata.
"""

import argparse
import io
import os
import pickle
import re
import sys
import tarfile
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from skeleton_transfer.scans import ScanSample, save_sample  # noqa: E402
from skeleton_transfer.smpl import SMPLModel, find_smpl_models  # noqa: E402

SURFACE_LABELS = ["skin", "hair", "shoe", "upper", "lower", "outer"]

#: <...>/<Outfit>/<Take>/<relative path inside the take>
MEMBER = re.compile(r"(?:^|/)(?P<outfit>Inner|Outer)/(?P<take>Take\d+)/(?P<rel>.+)$")
FRAME_FILES = {
    "mesh": re.compile(r"^Meshes_pkl/mesh-f(\d+)\.pkl$"),
    "smpl": re.compile(r"^SMPL/mesh-f(\d+)_smpl\.pkl$"),
    "labels": re.compile(r"^Semantic/labels/label-f(\d+)\.pkl$"),
}


@dataclass
class Take:
    """Frames of one sequence collected while reading a source."""

    subject: str
    outfit: str
    take: str
    info: Optional[dict] = None
    chosen: list = field(default_factory=list)  # frame numbers, in order of discovery
    frames: Dict[int, dict] = field(default_factory=dict)  # frame -> {"mesh": ..., "smpl": ..., "labels": ...}
    saved: set = field(default_factory=set)

    @property
    def name(self):
        return f"{self.subject}_{self.outfit}_{self.take}"


class Exporter:
    """Select frames, collect their files and save the samples."""

    def __init__(self, models: Dict[str, SMPLModel], out: str, stride: int, frames_per_take: int, require_labels: bool):
        self.models, self.out = models, out
        self.stride, self.frames_per_take, self.require_labels = stride, frames_per_take, require_labels
        self.takes: Dict[str, Take] = {}
        self.n_saved = 0
        os.makedirs(out, exist_ok=True)

    def wants(self, take: Take, frame: int) -> bool:
        """Deterministic selection, consistent across the files of a frame."""
        if frame in take.chosen:
            return True
        if (frame - 1) % self.stride != 0 or len(take.chosen) >= self.frames_per_take:
            return False
        take.chosen.append(frame)
        return True

    def take(self, subject: str, outfit: str, name: str) -> Take:
        key = f"{subject}_{outfit}_{name}"
        if key not in self.takes:
            self.takes[key] = Take(subject, outfit, name)
        return self.takes[key]

    def needs(self, subject: str, rel_path: str, outfit: str, take_name: str) -> bool:
        """Whether a file of a take must be read (before reading it)."""
        take = self.take(subject, outfit, take_name)
        if rel_path == "basic_info.pkl":
            return take.info is None
        for kind, pattern in FRAME_FILES.items():
            match = pattern.match(rel_path)
            if match:
                frame = int(match.group(1))
                return frame not in take.saved and self.wants(take, frame)
        return False

    def add(self, subject: str, outfit: str, take_name: str, rel_path: str, data) -> None:
        take = self.take(subject, outfit, take_name)
        if rel_path == "basic_info.pkl":
            take.info = data
        else:
            for kind, pattern in FRAME_FILES.items():
                match = pattern.match(rel_path)
                if match:
                    take.frames.setdefault(int(match.group(1)), {})[kind] = data
        self.flush(take)

    def flush(self, take: Take, final: bool = False) -> None:
        """Save the frames of ``take`` whose files are all available."""
        if take.info is None:
            return
        for frame, parts in list(take.frames.items()):
            complete = "mesh" in parts and "smpl" in parts and ("labels" in parts or not self.require_labels or final)
            if complete:
                self.save(take, frame, parts)
                take.saved.add(frame)
                del take.frames[frame]

    def save(self, take: Take, frame: int, parts: dict) -> None:
        gender = str(take.info.get("gender", "neutral"))
        model = self.models.get(gender) or self.models["neutral"]
        p = parts["smpl"]
        pose = np.concatenate((np.ravel(p["global_orient"]), np.ravel(p["body_pose"]))).reshape(24, 3)
        body, joints = model.forward(np.ravel(p["betas"]), pose, transl=np.ravel(p["transl"]))

        labels = None
        metadata = {"subject": take.subject, "outfit": take.outfit, "take": take.take, "frame": frame,
                    "rotation": np.asarray(take.info.get("rotation")), "smpl_params": {k: np.asarray(v) for k, v in p.items()}}  # fmt: skip
        if "labels" in parts:
            labels = np.asarray(parts["labels"]["scan_labels"]).astype(np.int8)
            counts = np.bincount(labels, minlength=len(SURFACE_LABELS)) / len(labels)
            metadata["label_fractions"] = dict(zip(SURFACE_LABELS, np.round(counts, 4).tolist()))

        mesh = parts["mesh"]
        sample = ScanSample(
            name=f"4ddress/{take.subject}/{take.outfit}/{take.take}/f{frame:05d}",
            scan_vertices=np.asarray(mesh["vertices"], dtype=float),
            scan_faces=np.asarray(mesh["faces"], dtype=int),
            smpl_vertices=body,
            gender=gender,
            category=take.outfit,
            scan_labels=labels,
            joints=joints,
            metadata=metadata,
        )
        path = os.path.join(self.out, f"{take.name}_f{frame:05d}.npz")
        save_sample(path, sample)
        self.n_saved += 1
        print(f"  saved {os.path.basename(path)} ({len(sample.scan_vertices)} scan vertices, {gender})", flush=True)

    def finish(self) -> None:
        for take in self.takes.values():
            self.flush(take, final=True)
            for frame in take.frames:
                print(f"  incomplete frame skipped: {take.name} f{frame:05d} (missing {sorted({'mesh', 'smpl'} - set(take.frames[frame]))})")


def subject_of(path: str) -> str:
    match = re.search(r"(\d{5})", os.path.basename(os.path.normpath(path)))
    return match.group(1) if match else "unknown"


def read_archive(path: str, exporter: Exporter) -> None:
    """Stream a .tar.gz, reading only the members the exporter needs."""
    subject = subject_of(path)
    size = os.path.getsize(path)
    t0 = time.time()
    with open(path, "rb") as raw, tarfile.open(fileobj=raw, mode="r|gz") as tar:
        for i, member in enumerate(tar):
            if i % 2000 == 0:
                done = raw.tell() / size
                print(f"  {100 * done:5.1f}% of {os.path.basename(path)} ({time.time() - t0:.0f} s)", flush=True)
            if not member.isfile():
                continue
            match = MEMBER.search(member.name)
            if not match or not exporter.needs(subject, match["rel"], match["outfit"], match["take"]):
                continue
            data = pickle.load(io.BytesIO(tar.extractfile(member).read()))
            exporter.add(subject, match["outfit"], match["take"], match["rel"], data)


def read_folder(path: str, exporter: Exporter) -> None:
    """Read the chosen frames of an extracted ``<subject>_<outfit>`` folder."""
    subject = subject_of(path)
    for root, _, files in os.walk(path):
        rel_root = root.replace("\\", "/")
        if not re.search(r"/(Inner|Outer)/Take\d+$", rel_root):
            continue
        outfit, take = rel_root.split("/")[-2:]
        candidates = ["basic_info.pkl"]
        mesh_dir = os.path.join(root, "Meshes_pkl")
        frames = sorted(int(m.group(1)) for f in os.listdir(mesh_dir) if (m := FRAME_FILES["mesh"].match("Meshes_pkl/" + f)))
        for frame in frames:
            candidates += [f"Meshes_pkl/mesh-f{frame:05d}.pkl", f"SMPL/mesh-f{frame:05d}_smpl.pkl",
                           f"Semantic/labels/label-f{frame:05d}.pkl"]  # fmt: skip
        for rel in candidates:
            full = os.path.join(root, rel)
            if os.path.exists(full) and exporter.needs(subject, rel, outfit, take):
                with open(full, "rb") as f:
                    exporter.add(subject, outfit, take, rel, pickle.load(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help=".tar.gz archives and/or extracted <subject>_<outfit> folders")
    parser.add_argument("--smpl-dir", required=True, help="folder with the official SMPL models (male, female, neutral)")
    parser.add_argument("--out", default="samples/4ddress", help="output folder for the .npz samples")
    parser.add_argument("--stride", type=int, default=20, help="keep frames 1, 1 + stride, ...")
    parser.add_argument("--frames-per-take", type=int, default=5, help="max frames per sequence")
    parser.add_argument("--allow-missing-labels", action="store_true", help="save frames without semantic labels")
    args = parser.parse_args()

    paths = find_smpl_models(args.smpl_dir)
    if not paths:
        parser.error(f"No SMPL models found in {args.smpl_dir}")
    models = {gender: SMPLModel.from_file(path) for gender, path in paths.items()}
    models.setdefault("neutral", next(iter(models.values())))
    exporter = Exporter(models, args.out, args.stride, args.frames_per_take, not args.allow_missing_labels)

    for source in args.sources:
        print(f"Reading {source}")
        if os.path.isdir(source):
            read_folder(source, exporter)
        elif source.endswith((".tar.gz", ".tgz")):
            read_archive(source, exporter)
        else:
            print(f"  skipped: not a folder or .tar.gz archive")
    exporter.finish()
    print(f"{exporter.n_saved} samples saved to {args.out}")


if __name__ == "__main__":
    main()
