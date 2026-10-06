"""pytest plugin: load a JSONL eval set and print a metric report.

Enable it with `-p evals_engineer.plugin` or by installing the package, which
registers the pytest11 entry point. Cases are parametrized only when
`--eval-dataset` is set, so ordinary unit tests are left alone.

A test records a row with the `eval_recorder` fixture:

    def test_case(eval_case, eval_recorder):
        passed = answer(eval_case) == eval_case["expected"]
        eval_recorder(case_id=eval_case["id"], correct=passed)
        assert passed
"""

from __future__ import annotations

import pytest

from evals_engineer.datasets import load_jsonl
from evals_engineer.metrics import mean, precision_recall_f1, wilson_interval


def pytest_addoption(parser):
    group = parser.getgroup("evals")
    group.addoption("--eval-dataset", action="store", default=None, help="JSONL eval dataset")


def pytest_configure(config):
    config._eval_rows = []


def pytest_generate_tests(metafunc):
    dataset = metafunc.config.getoption("--eval-dataset")
    if not dataset or "eval_case" not in metafunc.fixturenames:
        return
    cases = load_jsonl(dataset)
    metafunc.parametrize(
        "eval_case",
        cases,
        ids=[str(case.get("id", index)) for index, case in enumerate(cases)],
    )


@pytest.fixture
def eval_recorder(request):
    def record(case_id: str, correct: bool, score: float | None = None, **extra):
        row = {
            "case_id": case_id,
            "correct": correct,
            "score": (1.0 if correct else 0.0) if score is None else score,
        }
        row.update(extra)
        request.config._eval_rows.append(row)

    return record


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    del exitstatus
    rows = getattr(config, "_eval_rows", [])
    if not rows:
        return
    terminalreporter.write_sep("=", "eval metric report")
    scores = [row["score"] for row in rows if "score" in row]
    if scores:
        binary = all(score in (0.0, 1.0) for score in scores)
        terminalreporter.write_line(f"cases: {len(rows)}")
        terminalreporter.write_line(f"mean score: {mean(scores):.4f}")
        if binary:
            interval = wilson_interval(sum(1 for score in scores if score >= 1.0), len(scores))
            terminalreporter.write_line(f"success Wilson 95% CI: [{interval[0]:.4f}, {interval[1]:.4f}]")
    if rows and all("correct" in row for row in rows):
        truth = [True] * len(rows)
        pred = [bool(row["correct"]) for row in rows]
        stats = precision_recall_f1(truth, pred, True)
        terminalreporter.write_line(f"pass rate: {stats['precision']:.4f}  (correct {stats['tp']} / {len(rows)})")
    for row in rows:
        if row.get("correct") is False:
            terminalreporter.write_line(f"fail {row.get('case_id')}")
