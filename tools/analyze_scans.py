"""
Offline statistics of saved real-scan runs, beyond the tables of ``--report``.

Reads the pickles written by ``tools/evaluate_scans.py`` (several files are
merged, e.g. the shards of a run) and reports, for the complete pipeline
against the baseline: the error per gender and per subject, the error on a
subset of the targets against the others, how often the plausibility check
flags the failures, the number of iterations of the guided refinement, and
what the refined restarts change where they are triggered (with the oracle
that would pick the best candidate). From the repository root::

    python tools/analyze_scans.py samples/eval_4ddress_eval_s*.pkl --samples samples/4ddress \\
        --subset samples/4ddress_eval

Targets are named ``<dataset>/<subject>/...`` as by the 4D-DRESS exporter; the
gender of each subject is read from the ``manifest.csv`` of ``--samples``.
``--subset`` takes folders of samples (their ``manifest.csv``) or pickles of
runs, and compares their targets with the remaining ones.
"""

import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton_transfer.evaluation import FAILURE  # noqa: E402

BASELINE = "baseline"
RESTARTS = "+ restarts"
ITERATIVE = "+ iterative"
PRIORS = "all + priors"
COMPLETE = "all + priors, refined restarts"
#: Changes of error below this (m) count neither as improvements nor as worsenings.
TOLERANCE = 1e-3


def load_runs(paths):
    """{(target, variant): run} merged from several pickles (later files win)."""
    runs = {}
    for path in paths:
        with open(path, "rb") as f:
            runs.update({(r["target"], r["variant"]): r for r in pickle.load(f)})
    return runs


def manifest_targets(folder):
    """Target names of the samples listed in ``folder/manifest.csv`` (4D-DRESS naming)."""
    with open(os.path.join(folder, "manifest.csv"), newline="") as f:
        return {f"4ddress/{r['subject']}/{r['outfit']}/{r['take']}/f{int(r['frame']):05d}" for r in csv.DictReader(f)}


def manifest_genders(folder):
    """{subject: gender} from ``folder/manifest.csv``."""
    with open(os.path.join(folder, "manifest.csv"), newline="") as f:
        return {r["subject"]: r["gender"] for r in csv.DictReader(f)}


def candidate_errors(history):
    """Final error of each candidate (first map and restarts) of a refined-restarts history."""
    blocks = []
    for stage in history:
        if stage["stage"].startswith(("init", "restart")):
            blocks.append([stage])
        elif blocks:
            blocks[-1].append(stage)
    return [block[-1]["error"] for block in blocks]


def analyze(runs, genders=None, subset=None):
    """Statistics of the targets that have the complete pipeline, as a dict (errors in m)."""
    targets = sorted({t for t, v in runs if v == COMPLETE})
    error = {v: np.array([runs[(t, v)]["error"] for t in targets]) for v in (BASELINE, COMPLETE)}
    subjects = np.array([t.split("/")[1] for t in targets])
    out = {"n": len(targets)}

    def errors(mask):
        return {v: (float(e[mask].mean()), int((e[mask] > FAILURE).sum())) for v, e in error.items()}

    out["subjects"] = {s: dict(n=int((subjects == s).sum()), **errors(subjects == s)) for s in sorted(set(subjects))}
    if genders:
        g = np.array([genders.get(s, "unknown") for s in subjects])
        out["genders"] = {name: dict(n=int((g == name).sum()), subjects=len(set(subjects[g == name])), **errors(g == name))
                          for name in sorted(set(g))}  # fmt: skip
    if subset is not None:
        inside = np.array([t in subset for t in targets])
        out["subset"] = {"inside": dict(n=int(inside.sum()), **errors(inside)),
                         "outside": dict(n=int((~inside).sum()), **errors(~inside))}  # fmt: skip

    out["plausibility"] = {}
    for variant in (RESTARTS, COMPLETE):
        present = [t for t in targets if (t, variant) in runs]
        fail = np.array([runs[(t, variant)]["error"] > FAILURE for t in present], dtype=bool)
        flag = np.array([not runs[(t, variant)]["plausible"] for t in present], dtype=bool)
        out["plausibility"][variant] = dict(failures=int(fail.sum()), flagged=int((flag & fail).sum()),
                                            false_alarms=int((flag & ~fail).sum()))  # fmt: skip

    iterative = [t for t in targets if (t, ITERATIVE) in runs]
    if iterative:
        steps = [sum(s["stage"].startswith("iteration") for s in runs[(t, ITERATIVE)]["history"]) for t in iterative]
        out["iterations"] = float(np.mean(steps))

    triggered = [t for t in targets
                 if (t, PRIORS) in runs and any(s["stage"].startswith("restart") for s in runs[(t, COMPLETE)]["history"])]  # fmt: skip
    if triggered:
        before = np.array([runs[(t, PRIORS)]["error"] for t in triggered])
        after = np.array([runs[(t, COMPLETE)]["error"] for t in triggered])
        oracle = np.array([min(candidate_errors(runs[(t, COMPLETE)]["history"])) for t in triggered])
        out["restarts"] = dict(n=len(triggered), before=float(before.mean()), after=float(after.mean()),
                               failures_before=int((before > FAILURE).sum()), failures_after=int((after > FAILURE).sum()),
                               improved=int((after < before - TOLERANCE).sum()), worsened=int((after > before + TOLERANCE).sum()),
                               oracle=float(oracle.mean()), oracle_failures=int((oracle > FAILURE).sum()))  # fmt: skip
    return out


def print_report(stats):
    mm = lambda e: f"{1000 * e:6.1f}"  # noqa: E731

    def row(label, d):
        (b, bf), (c, cf) = d[BASELINE], d[COMPLETE]
        return f"  {label:24s}{d['n']:5d}  {mm(b)} {bf:3d}  {mm(c)} {cf:3d}"

    print(f"{stats['n']} targets; mean error (mm) and failures: baseline -> {COMPLETE}")
    for name, d in stats.get("genders", {}).items():
        print(row(f"{name} ({d['subjects']} subjects)", d))
    for name, d in stats.get("subset", {}).items():
        print(row(f"subset: {name}", d))
    print("per subject")
    for name, d in stats["subjects"].items():
        print(row(name, d))
    for variant, d in stats["plausibility"].items():
        print(f"plausibility check ({variant}): flags {d['flagged']}/{d['failures']} failures, {d['false_alarms']} false alarms")
    if "iterations" in stats:
        print(f"guided refinement ({ITERATIVE}): {stats['iterations']:.1f} iterations on average")
    if "restarts" in stats:
        r = stats["restarts"]
        print(f"restarts triggered on {r['n']} targets: {PRIORS} {mm(r['before']).strip()} -> {mm(r['after']).strip()} mm, "
              f"failures {r['failures_before']} -> {r['failures_after']}, improved {r['improved']}, worsened {r['worsened']} "
              f"(by more than {1000 * TOLERANCE:.0f} mm); oracle {mm(r['oracle']).strip()} mm with {r['oracle_failures']} failures")  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="pickles written by tools/evaluate_scans.py (merged)")
    parser.add_argument("--samples", help="folder of exported samples whose manifest.csv gives the subject genders")
    parser.add_argument("--subset", nargs="+", help="folders of samples or pickles of runs: their targets against the rest")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.runs for p in glob.glob(pattern)})  # the shell may not expand patterns
    subset = None
    if args.subset:
        subset = set()
        for item in args.subset:
            subset |= manifest_targets(item) if os.path.isdir(item) else {t for t, _ in load_runs([item])}
    genders = manifest_genders(args.samples) if args.samples else None
    print_report(analyze(load_runs(paths), genders, subset))


if __name__ == "__main__":
    main()
