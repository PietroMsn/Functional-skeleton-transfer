"""Offline statistics of saved real-scan runs (tools/analyze_scans.py) on hand-made runs."""

import importlib.util
import os
import pickle

import pytest

from .conftest import REPO_ROOT


def _load_tool():
    path = os.path.join(REPO_ROOT, "tools", "analyze_scans.py")
    spec = importlib.util.spec_from_file_location("analyze_scans", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _run(target, variant, error, plausible=True, history=None):
    return dict(target=target, variant=variant, error=error, plausible=plausible, history=history or [], condition="Inner")


def _stages(*blocks):
    """History with one block per candidate: (first stage name, [errors of the stage and its iterations])."""
    return [{"stage": name if i == 0 else f"iteration {i}", "error": e} for name, errors in blocks for i, e in enumerate(errors)]


@pytest.fixture
def runs():
    a, b = "4ddress/00001/Inner/Take1/f00001", "4ddress/00002/Inner/Take1/f00001"
    return [
        _run(a, tool.BASELINE, 0.20),
        _run(b, tool.BASELINE, 0.04),
        _run(a, tool.RESTARTS, 0.15, plausible=False),
        _run(b, tool.RESTARTS, 0.03, plausible=False),
        _run(a, tool.ITERATIVE, 0.10, history=_stages(("init k=20", [0.2, 0.15, 0.1]))),
        _run(b, tool.ITERATIVE, 0.03, history=_stages(("init k=20", [0.04, 0.03]))),
        _run(a, tool.PRIORS, 0.12),
        _run(b, tool.PRIORS, 0.03),
        # restarts on a: the selection keeps the k=30 candidate (0.05), the best one is k=10 (0.04)
        _run(a, tool.COMPLETE, 0.05, history=_stages(("init k=20", [0.2, 0.12]), ("restart k=10", [0.1, 0.04]),
                                                      ("restart k=30", [0.09, 0.05]))),  # fmt: skip
        _run(b, tool.COMPLETE, 0.03, history=_stages(("init k=20", [0.04, 0.03]))),
    ]


def test_candidate_errors():
    history = _stages(("init k=20", [0.3, 0.2]), ("restart k=10", [0.1]), ("restart k=30", [0.2, 0.1, 0.05]))
    assert tool.candidate_errors(history) == [0.2, 0.1, 0.05]


def test_analyze(runs):
    stats = tool.analyze({(r["target"], r["variant"]): r for r in runs}, {"00001": "female", "00002": "male"},
                         subset={"4ddress/00001/Inner/Take1/f00001"})  # fmt: skip
    assert stats["n"] == 2
    assert stats["subjects"]["00001"][tool.BASELINE] == (0.20, 1)
    assert stats["genders"]["male"]["n"] == 1 and stats["genders"]["male"][tool.COMPLETE] == (0.03, 0)
    assert stats["subset"]["inside"]["n"] == 1 and stats["subset"]["outside"][tool.BASELINE] == (0.04, 0)
    assert stats["plausibility"][tool.RESTARTS] == dict(failures=1, flagged=1, false_alarms=1)
    assert stats["iterations"] == pytest.approx(1.5)
    r = stats["restarts"]
    assert r["n"] == 1 and r["before"] == pytest.approx(0.12) and r["after"] == pytest.approx(0.05)
    assert (r["failures_before"], r["failures_after"], r["improved"], r["worsened"]) == (1, 0, 1, 0)
    assert r["oracle"] == pytest.approx(0.04)


def test_load_runs_merges_shards(tmp_path, runs):
    for i, part in enumerate((runs[:5], runs[5:])):
        with open(tmp_path / f"s{i}.pkl", "wb") as f:
            pickle.dump(part, f)
    merged = tool.load_runs([tmp_path / "s0.pkl", tmp_path / "s1.pkl"])
    assert len(merged) == len(runs)
