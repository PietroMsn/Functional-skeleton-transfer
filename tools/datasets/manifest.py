"""
Provenance of exported samples.

Every folder of samples carries a ``manifest.csv`` with one row per sample:
the file, the source it was read from (archive or extracted folder), subject,
outfit, sequence (take), frame, gender and the fraction of the scan covered by
each garment label. The manifest is rebuilt from the samples themselves, so it
stays correct after samples are added, copied or removed, and the source
archives can be deleted once a subject is exported:

    python tools/datasets/manifest.py samples/4ddress

Samples exported before the source was recorded get ``<subject>_<outfit>``
(the name shared by the archive and its extracted folder).
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from skeleton_transfer.scans import load_sample  # noqa: E402

LABELS = ["skin", "hair", "shoe", "upper", "lower", "outer"]
COLUMNS = ["file", "source", "subject", "outfit", "take", "frame", "gender", "n_scan_vertices"] + LABELS


def manifest_rows(folder: str):
    """One provenance row per ``.npz`` sample of ``folder``."""
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        sample = load_sample(path)
        m = sample.metadata
        fractions = m.get("label_fractions", {})
        rows.append({
            "file": os.path.basename(path),
            "source": m.get("source") or f"{m.get('subject')}_{m.get('outfit')}",
            "subject": m.get("subject"),
            "outfit": m.get("outfit"),
            "take": m.get("take"),
            "frame": m.get("frame"),
            "gender": sample.gender,
            "n_scan_vertices": len(sample.scan_vertices),
            **{label: fractions.get(label, "") for label in LABELS},
        })  # fmt: skip
    return rows


def write_manifest(folder: str) -> str:
    """Write ``folder/manifest.csv`` and return its path."""
    path = os.path.join(folder, "manifest.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows(folder))
    return path


def summary(rows) -> str:
    """Number of samples and sequences per subject and outfit."""
    groups = {}
    for r in rows:
        groups.setdefault((r["subject"], r["outfit"], r["gender"]), []).append(r["take"])
    return "\n".join(
        f"  {subject} {outfit:5s} {gender:6s} {len(set(takes)):3d} sequences {len(takes):4d} frames"
        for (subject, outfit, gender), takes in sorted(groups.items())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="folders of exported samples")
    args = parser.parse_args()
    for folder in args.folders:
        path = write_manifest(folder)
        print(f"{path}\n{summary(manifest_rows(folder))}")


if __name__ == "__main__":
    main()
