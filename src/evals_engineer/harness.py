"""One CLI surface over three local framework adapters.

The adapters follow the assertion styles of promptfoo (exact checks), DeepEval
(rubric criteria), and RAGAS (faithfulness, answer relevancy, context precision).
They are local implementations. A metric is never averaged across frameworks.
"""

from __future__ import annotations

from evals_engineer.report import Metric, Report
from evals_engineer.rubrics import CRITERIA, weighted_overall
from evals_engineer.scoring import contains_match, exact_match, regex_match
from evals_engineer.textutil import tokenize

FRAMEWORKS = ("promptfoo", "deepeval", "ragas")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_promptfoo(cases: list[dict], predictions: dict[str, dict]) -> dict:
    """promptfoo-style assertions: equals, contains, regex. All must pass."""
    rows = []
    for case in cases:
        pred = predictions[case["id"]]
        text = pred.get("text", "")
        checks = []
        for assertion in case.get("assertions", []):
            kind = assertion["type"]
            if kind == "equals":
                ok = exact_match(text, assertion["value"]) == 1.0
            elif kind == "contains":
                ok = contains_match(text, assertion["value"]) == 1.0
            elif kind == "regex":
                ok = regex_match(text, assertion["value"]) == 1.0
            else:
                raise ValueError(f"unknown assertion {kind}")
            checks.append({"type": kind, "passed": ok})
        rows.append({"id": case["id"], "passed": all(item["passed"] for item in checks) if checks else False, "checks": checks})
    return {"framework": "promptfoo", "implementation": "local", "pass_rate": _mean([1.0 if row["passed"] else 0.0 for row in rows]), "rows": rows}


def evaluate_deepeval(cases: list[dict], predictions: dict[str, dict]) -> dict:
    """DeepEval-style criteria. The case supplies criterion scores or a rubric function result."""
    rows = []
    for case in cases:
        pred = predictions[case["id"]]
        criteria = pred.get("criteria") or {}
        missing = [name for name in CRITERIA if name not in criteria]
        if missing:
            raise ValueError(f"{case['id']} is missing criteria {missing}")
        overall = weighted_overall({name: int(criteria[name]) for name in CRITERIA})
        minimum = int(case.get("min_overall", 4))
        rows.append({"id": case["id"], "overall": overall, "passed": overall >= minimum})
    return {
        "framework": "deepeval",
        "implementation": "local",
        "pass_rate": _mean([1.0 if row["passed"] else 0.0 for row in rows]),
        "mean_overall": _mean([row["overall"] for row in rows]),
        "rows": rows,
    }


def _context_precision(answer: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0
    answer_terms = set(tokenize(answer, drop_stop=True))
    if not answer_terms:
        return 0.0
    useful = 0
    for context in contexts:
        terms = set(tokenize(context, drop_stop=True))
        if answer_terms & terms:
            useful += 1
    return useful / len(contexts)


def _answer_relevancy(question: str, answer: str) -> float:
    q = set(tokenize(question, drop_stop=True))
    a = set(tokenize(answer, drop_stop=True))
    if not q or not a:
        return 0.0
    return len(q & a) / len(q)


def evaluate_ragas(cases: list[dict], predictions: dict[str, dict]) -> dict:
    """RAGAS-style faithfulness, answer relevancy, and context precision."""
    rows = []
    for case in cases:
        pred = predictions[case["id"]]
        answer = pred.get("text", "")
        contexts = pred.get("contexts") or case.get("contexts") or []
        cited = pred.get("cited_contexts") or contexts
        if pred.get("abstained"):
            faith = 1.0
        elif not cited:
            faith = 0.0
        else:
            terms = set(tokenize(answer, drop_stop=True))
            supported = set()
            for context in cited:
                supported |= set(tokenize(context, drop_stop=True))
            faith = (len(terms & supported) / len(terms)) if terms else 0.0
        relevancy = _answer_relevancy(case.get("question", ""), answer)
        precision = _context_precision(answer, contexts)
        passed = faith >= 0.5 and (pred.get("abstained") or relevancy > 0 or not case.get("question"))
        rows.append(
            {
                "id": case["id"],
                "faithfulness": faith,
                "answer_relevancy": relevancy,
                "context_precision": precision,
                "passed": passed,
            }
        )
    return {
        "framework": "ragas",
        "implementation": "local",
        "pass_rate": _mean([1.0 if row["passed"] else 0.0 for row in rows]),
        "faithfulness": _mean([row["faithfulness"] for row in rows]),
        "answer_relevancy": _mean([row["answer_relevancy"] for row in rows]),
        "context_precision": _mean([row["context_precision"] for row in rows]),
        "rows": rows,
    }


_ADAPTERS = {
    "promptfoo": evaluate_promptfoo,
    "deepeval": evaluate_deepeval,
    "ragas": evaluate_ragas,
}


def run_harness(cases: list[dict], predictions: dict[str, dict], frameworks: tuple[str, ...] = FRAMEWORKS) -> dict:
    unknown = [name for name in frameworks if name not in _ADAPTERS]
    if unknown:
        raise ValueError(f"unknown frameworks: {unknown}")
    sections = [_ADAPTERS[name](cases, predictions) for name in frameworks]
    return {
        "frameworks": sections,
        "averaged": False,
        "note": "Framework scores are reported separately and are not averaged.",
    }


def harness_report(result: dict) -> Report:
    metrics = []
    lines = [result["note"]]
    all_pass = True
    for section in result["frameworks"]:
        metrics.append(Metric(f"{section['framework']}_pass_rate", section["pass_rate"], section["implementation"]))
        if section["pass_rate"] < 1.0:
            all_pass = False
        for key, value in section.items():
            if key in {"framework", "implementation", "pass_rate", "rows"}:
                continue
            if isinstance(value, float):
                metrics.append(Metric(f"{section['framework']}_{key}", value))
    return Report(
        title="Unified harness",
        passed=all_pass,
        summary="Local promptfoo, DeepEval, and RAGAS adapters ran behind one report.",
        metrics=metrics,
        lines=lines,
    )
