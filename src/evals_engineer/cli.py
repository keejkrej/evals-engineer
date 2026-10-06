"""Command line for the Harbor eval systems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals_engineer.context_memory import FifoPolicy, SaliencePolicy, context_report, run_eviction
from evals_engineer.failures import failure_report, run_failure_suite
from evals_engineer.gate import run_gate, summarize_run
from evals_engineer.golden import write_golden
from evals_engineer.judge import ScriptedJudge, calibrate, judge_report
from evals_engineer.pareto import example_points, pareto_report, summarize, write_dashboard
from evals_engineer.portfolio import run_portfolio
from evals_engineer.rag import evaluate_rag, grounded_generate, rag_report
from evals_engineer.redteam import guarded_answer, injection_results, looping_agent, redteam_report, run_loop, schema_results
from evals_engineer.replay import decision_llm, decision_tools, refund_trace, replay, replay_report
from evals_engineer.synthetic import EdgeCaseGenerator, synthetic_report
from evals_engineer.teardown import run_teardown
from evals_engineer.trajectory import evaluate_trajectories, trajectory_report
from evals_engineer.compare import compare_report


def _print(report) -> None:
    sys.stdout.write(report.to_markdown())


def _load_run(path: str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "success_rate" in payload and "cases" in payload:
        return payload
    return summarize_run(payload.get("name", Path(path).stem), payload["cases"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description="Harbor eval harness")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run every system and write a portfolio report")
    demo.add_argument("--out", default="reports/demo")
    demo.add_argument("--boot", type=int, default=2000)

    golden = sub.add_parser("golden", help="write the versioned 300-case golden set")
    golden.add_argument("--out", default="data/golden/v1")

    sub.add_parser("judge", help="calibrate the scripted judge against 500 anchors")
    sub.add_parser("rag", help="run retrieval and generation gates")
    sub.add_parser("trajectory", help="grade bundled tool-call trajectories")
    sub.add_parser("failures", help="classify 500 scripted adversarial outputs")
    sub.add_parser("teardown", help="compare the careful agent with the naive baseline")

    gate = sub.add_parser("gate", help="compare a run with a baseline and set the exit code")
    gate.add_argument("--current", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--state", default="reports/gate-state.json")
    gate.add_argument("--comment", default="reports/pr-comment.md")

    pareto = sub.add_parser("pareto", help="write the cost-quality dashboard")
    pareto.add_argument("--out", default="reports/pareto.html")

    sub.add_parser("replay", help="swap the lookup hop and replay the refund trace")
    sub.add_parser("context", help="compare salience eviction with FIFO")
    sub.add_parser("fuzz", help="run canary, schema, and loop-guard checks")
    sub.add_parser("synthetic", help="generate template edge cases")

    args = parser.parse_args(argv)
    if args.command == "demo":
        payload = run_portfolio(args.out, n_boot=args.boot)
        sys.stdout.write(f"wrote {args.out} (passed={payload['passed']})\n")
        return 0 if payload["passed"] else 1
    if args.command == "golden":
        manifest = write_golden(args.out)
        sys.stdout.write(f"wrote {args.out} kappa={manifest['kappa']:.4f} sha256={manifest['sha256']}\n")
        return 0
    if args.command == "judge":
        result = calibrate(ScriptedJudge(name="calibrated"))
        _print(judge_report(result))
        return 0 if result["calibrated"] else 1
    if args.command == "rag":
        result = evaluate_rag(generate=grounded_generate)
        report = rag_report(result)
        _print(report)
        return 0 if report.passed else 1
    if args.command == "trajectory":
        report = trajectory_report(evaluate_trajectories())
        _print(report)
        return 0 if report.passed else 1
    if args.command == "failures":
        report = failure_report(run_failure_suite())
        _print(report)
        return 0 if report.passed else 1
    if args.command == "teardown":
        result = run_teardown()
        report = compare_report(result["comparison"])
        _print(report)
        return 0
    if args.command == "gate":
        decision = run_gate(
            _load_run(args.current),
            _load_run(args.baseline),
            state_path=args.state,
            comment_path=args.comment,
        )
        sys.stdout.write(decision["comment"])
        return 0 if decision["passed"] else 1
    if args.command == "pareto":
        result = summarize(example_points())
        write_dashboard(result, args.out)
        _print(pareto_report(result))
        sys.stdout.write(f"wrote {args.out}\n")
        return 0
    if args.command == "replay":
        tools = decision_tools({"A1000": {"status": "delivered", "days_ago": 90}})
        result = replay(refund_trace(), "lookup", decision_llm, tools)
        report = replay_report(result)
        _print(report)
        return 0 if result["outputs"]["act"]["refunded"] is False else 1
    if args.command == "context":
        report = context_report(run_eviction(SaliencePolicy()), run_eviction(FifoPolicy()))
        _print(report)
        return 0 if report.passed else 1
    if args.command == "fuzz":
        report = redteam_report(
            injection_results(guarded_answer),
            schema_results(),
            run_loop(looping_agent, "refund A1000"),
        )
        _print(report)
        return 0 if report.passed else 1
    if args.command == "synthetic":
        cases = EdgeCaseGenerator().generate(
            [{"id": "seed-0", "input": {"query": "What is the refund window?", "order_id": "A1000"}}]
        )
        report = synthetic_report(cases)
        _print(report)
        return 0 if report.passed else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
