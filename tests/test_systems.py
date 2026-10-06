"""Behavior tests for the eval systems. No network, no GPU."""

import json
import re
import subprocess
import sys
from pathlib import Path

from evals_engineer.compare import compare_binary
from evals_engineer.contamination import scan
from evals_engineer.context_memory import FifoPolicy, SaliencePolicy, run_eviction
from evals_engineer.datasets import content_hash, stratified_sample
from evals_engineer.drift import watch_nights
from evals_engineer.failures import build_failure_cases, classify, faulty_output, run_failure_suite, run_model
from evals_engineer.flywheel import DryRunTrainer, nightly, preference_pairs
from evals_engineer.gate import EvalCache, comment_markdown, decide, run_gate, summarize_run
from evals_engineer.golden import STRATA, build_golden_cases, write_golden
from evals_engineer.harness import run_harness
from evals_engineer.judge import (
    ScriptedJudge,
    build_anchors,
    calibrate,
    parse_judge_json,
    render_prompt,
    score_sample,
)
from evals_engineer.metrics import cohens_kappa
from evals_engineer.pareto import example_points, render_dashboard, summarize
from evals_engineer.rag import evaluate_rag, grounded_generate, overconfident_generate, rank_docs
from evals_engineer.domain import policy_corpus
from evals_engineer.redteam import (
    guarded_answer,
    injection_results,
    looping_agent,
    run_loop,
    schema_results,
    unguarded_answer,
)
from evals_engineer.replay import decision_llm, decision_tools, refund_trace, replay
from evals_engineer.rubrics import JUDGE_PROMPT
from evals_engineer.shadow import serve, shadow_experiment
from evals_engineer.synthetic import EdgeCaseGenerator
from evals_engineer.teardown import run_teardown
from evals_engineer.trajectory import build_trajectory_cases, grade_trajectory

ROOT = Path(__file__).resolve().parents[1]


def _run(cases):
    return summarize_run(
        "run",
        [{"id": case["id"], "passed": case["passed"], "latency_ms": case["latency_ms"], "cost_usd": 0.01} for case in cases],
    )


def test_golden_set_is_balanced_and_versioned(tmp_path):
    cases = build_golden_cases()
    assert len(cases) == 300
    assert {name: sum(case["stratum"] == name for case in cases) for name in STRATA} == {name: 25 for name in STRATA}
    ada = [case["annotators"]["ada"]["verdict"] for case in cases]
    ben = [case["annotators"]["ben"]["verdict"] for case in cases]
    kappa = cohens_kappa(ada, ben, labels=["abstain", "fail", "pass"])
    assert kappa["kappa"] >= 0.7
    assert sum(a != b for a, b in zip(ada, ben)) == 30
    manifest = write_golden(tmp_path)
    assert manifest["kappa"] == kappa["kappa"]
    assert manifest["sha256"] == content_hash(cases)
    text = (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "300 cases" in text
    assert f"{kappa['kappa']:.4f}" in text
    again = write_golden(tmp_path)
    assert again["sha256"] == manifest["sha256"]


def test_stratified_sample_keeps_every_stratum():
    rows = [{"id": f"{name}-{i}", "stratum": name} for name in ("a", "b", "c") for i in range(10)]
    drawn = stratified_sample(rows, n=9, seed=1)
    assert len(drawn) == 9
    assert {row["stratum"] for row in drawn} == {"a", "b", "c"}


def test_failure_classifier_reads_the_output_not_the_attack_name():
    cases = build_failure_cases(500)
    assert len(cases) == 500
    faulty = run_failure_suite(temperature=0.0, seed=0)
    careful_labels = []
    for case in cases:
        faulty_label = classify(case, faulty_output(case))
        assert faulty_label == case["stressor"]
        careful_labels.append(classify(case, run_model(case, careful=True)))
    assert set(careful_labels) == {"none"}
    assert sum(faulty["counts"].values()) == 500
    assert faulty["rows"][0]["steps"][0].startswith("Set temperature=0.0")
    even = cases[0]
    assert run_model(even, temperature=0.0, seed=1)["text"] == run_model(even, temperature=0.0, seed=2)["text"]
    assert run_model(even, temperature=0.7, seed=0)["text"] != run_model(even, temperature=0.7, seed=1)["text"]


def test_judge_calibration_separates_bias_from_the_rubric():
    anchors = build_anchors()
    assert len(anchors) == 500
    calibrated = calibrate(ScriptedJudge(name="calibrated"), anchors)
    biased = calibrate(ScriptedJudge(name="biased", verbosity_bias=2, position_bias=True, self_preference=2), anchors)
    drifted = calibrate(ScriptedJudge(name="drifted", drift_abstention=True), anchors)
    assert calibrated["calibrated"] is True
    assert calibrated["quadratic_kappa_vs_adjudicator"] == 1.0
    assert calibrated["verbosity_delta"] == 0.0
    assert calibrated["position_bias_rate"] == 0.0
    assert calibrated["self_preference_delta"] == 0.0
    assert calibrated["rater_quadratic_kappa"] < 1.0
    assert biased["calibrated"] is False
    assert biased["verbosity_delta"] > 0.25
    assert biased["position_bias_rate"] == 1.0
    assert biased["self_preference_delta"] > 0.25
    assert drifted["quadratic_kappa_vs_adjudicator"] < calibrated["quadratic_kappa_vs_adjudicator"]
    sample = anchors[0]
    prompt = render_prompt(sample)
    assert "0.4*groundedness" in prompt
    assert sample.response in prompt
    raw = ScriptedJudge().complete(prompt, sample)
    parsed = parse_judge_json("Here is the score:\n" + raw)
    assert score_sample(ScriptedJudge(), sample)["overall"] == parsed["overall"]


def test_rag_gates_are_separate():
    grounded = evaluate_rag(generate=grounded_generate)
    boast = evaluate_rag(generate=overconfident_generate)
    assert grounded["hit_rate"] == 1.0
    assert grounded["recall_at_k"] == 1.0
    assert grounded["mrr"] == 1.0
    assert grounded["both_gates_pass_rate"] == 1.0
    assert grounded["retrieval_pass_rate"] == 1.0
    assert boast["generation_pass_rate"] == 0.0
    assert boast["retrieval_pass_rate"] == 1.0
    # A retrieval miss and a generation miss are different rows.
    assert "retrieval_pass" in grounded["rows"][0] and "generation_pass" in grounded["rows"][0]
    corpus = policy_corpus()
    top = rank_docs("What is the refund window after the delivery date?", corpus, k=1)
    assert top[0]["id"] == "refund-window"


def test_trajectory_blocks_unsafe_paths_and_allows_a_branch():
    cases = build_trajectory_cases()
    ok = next(case for case in cases if case["stratum"] == "tool_refund_ok")
    graded = grade_trajectory(ok["fixtures"]["prediction"], ok["expected"]["allowed_paths"])
    assert graded["passed"] is True
    alternate = ok["expected"]["allowed_paths"][1]
    assert grade_trajectory(alternate)["passed"] is True
    unsafe = next(case for case in cases if case["stratum"] == "tool_refund_unsafe")
    failed = grade_trajectory(unsafe["fixtures"]["prediction"])
    assert failed["passed"] is False
    assert any(item["code"] == "skipped_safety_check" for item in failed["violations"])
    hallucinated = next(case for case in cases if case["stratum"] == "tool_hallucinated_param")
    hall = grade_trajectory(hallucinated["fixtures"]["prediction"])
    assert any(item["code"] == "hallucinated_parameter" for item in hall["violations"])
    leaked = next(case for case in cases if case["stratum"] == "tool_email_blocked")
    mail = grade_trajectory(leaked["fixtures"]["prediction"])
    assert any(item["code"] == "dangerous_sequence" for item in mail["violations"])
    assert grade_trajectory(leaked["fixtures"]["safe_alternative"])["passed"] is True
    # The safety check has to bind the same order id. A different order does not count.
    wrong_order = [
        {"tool": "check_entitlement", "args": {"order_id": "OTHER"}},
        {"tool": "refund_order", "args": {"order_id": "A1", "amount_cents": 100}},
    ]
    assert grade_trajectory(wrong_order)["passed"] is False


def test_harness_keeps_frameworks_separate():
    cases = [
        {
            "id": "h1",
            "question": "What is the refund window?",
            "assertions": [{"type": "contains", "value": "30 days"}, {"type": "regex", "value": r"30"}],
            "min_overall": 4,
        }
    ]
    predictions = {
        "h1": {
            "text": "The refund window is 30 days.",
            "criteria": {"groundedness": 5, "completeness": 5, "abstention": 5, "constraint": 5},
            "contexts": ["The refund window is 30 days from the delivery date."],
            "cited_contexts": ["The refund window is 30 days from the delivery date."],
        }
    }
    result = run_harness(cases, predictions)
    assert [section["framework"] for section in result["frameworks"]] == ["promptfoo", "deepeval", "ragas"]
    assert result["averaged"] is False
    assert all(section["pass_rate"] == 1.0 for section in result["frameworks"])


def test_comparison_sentence_and_small_sample_noise():
    teardown = run_teardown(n_boot=400)
    sentence = teardown["comparison"]["sentence"]
    assert teardown["comparison"]["significant"] is True
    assert re.search(r"careful agent wins by \d+\.\d% ± \d+\.\d% \(p<0\.05\)", sentence)
    baseline = [True] * 50
    candidate = [True] * 49 + [False]
    noise = compare_binary("Model A", baseline, "Model B", candidate, n_boot=400, seed=1)
    assert noise["significant"] is False
    assert noise["sentence"].startswith("No significant difference:")
    assert "2.0%" in noise["sentence"]


def test_gate_hysteresis_latency_flakes_and_cache(tmp_path):
    baseline = _run([{"id": f"c{i}", "passed": True, "latency_ms": 100} for i in range(100)])
    dropped = _run([{"id": f"c{i}", "passed": i >= 5, "latency_ms": 100} for i in range(100)])
    decision = decide(dropped, baseline)
    assert decision["passed"] is False
    assert decision["drop"] > 0.02
    assert decision["state"]["tripped"] is True
    assert len(decision["failing_cases"]) == 5
    assert "c0" in comment_markdown(decision, dropped, baseline)
    # 1.5% is inside the dead band on the way down, and still failing on the way up.
    mild = _run([{"id": f"c{i}", "passed": i >= 2, "latency_ms": 100} for i in range(100)])
    # 2 failures of 100 is a 2% drop, which is not greater than 2%.
    assert decide(mild, baseline)["success_ok"] is True
    held = decide(mild, baseline, state={"tripped": True})
    assert held["success_ok"] is False
    recovered = _run([{"id": f"c{i}", "passed": i >= 1, "latency_ms": 100} for i in range(100)])
    assert decide(recovered, baseline, state={"tripped": True})["success_ok"] is True
    slow = _run([{"id": f"c{i}", "passed": True, "latency_ms": 400} for i in range(100)])
    latency = decide(slow, baseline)
    assert latency["passed"] is False
    assert latency["state"]["tripped"] is False
    flaky = _run([{"id": "c0", "passed": True, "latency_ms": 100}])
    flaky["cases"][0]["seed_passes"] = [True, False, True, False, True]
    assert decide(flaky, _run([{"id": "c0", "passed": True, "latency_ms": 100}]))["passed"] is False
    state = tmp_path / "state.json"
    comment = tmp_path / "comment.md"
    first = run_gate(baseline, baseline, state_path=state, comment_path=comment)
    assert first["passed"] is True
    assert "PASS" in comment.read_text(encoding="utf-8")
    cache = EvalCache(tmp_path / "cache")
    payload = {"dataset": "v1", "pred": [1, 0, 1]}
    miss = run_gate(baseline, baseline, cache=cache, cache_payload=payload)
    hit = run_gate(baseline, baseline, cache=cache, cache_payload=payload)
    assert miss["cache"] == "miss" and hit["cache"] == "hit"


def test_shadow_never_changes_the_user_response():
    def primary(request):
        return {"text": "primary", "steps": [{"tool": "search_kb"}], "cost_usd": 0.01, "quality": 0.8}

    def shadow(request):
        if request["id"] == "boom":
            raise TimeoutError("candidate down")
        return {"text": "shadow", "steps": [{"tool": "escalate"}], "cost_usd": 0.02, "quality": 0.9}

    boom = serve({"id": "boom"}, primary, shadow, rate=1.0)
    assert boom["user"]["text"] == "primary"
    assert boom["shadow_error"]
    quiet = serve({"id": "req-quiet"}, primary, shadow, rate=0.0)
    assert quiet["sampled"] is False and quiet["shadow"] is None
    experiment = shadow_experiment(
        [{"id": f"req-{i:04d}"} for i in range(500)], primary, shadow, rate=0.05
    )
    assert experiment["user_responses_match_primary"] is True
    assert 0.02 < experiment["sample_rate"] < 0.09


def test_drift_flywheel_pareto_replay_context_contamination_fuzz(tmp_path):
    nights = []
    for night in range(4):
        success = night < 2
        nights.append([{"id": f"n{night}-{i:03d}", "success": success} for i in range(100)])
    watched = watch_nights(nights, baseline_rate=0.95, drop=0.05, consecutive=2)
    assert watched["alerts"]
    assert watched["alerts"][0]["night"] >= 2

    events = [
        {
            "event_id": "e1",
            "rating": "down",
            "prompt": "Refund window?",
            "response": "365 days",
            "correction": "30 days",
        },
        {"event_id": "e2", "rating": "down", "prompt": "Same", "response": "same", "correction": "same"},
        {"event_id": "e3", "rating": "up", "prompt": "ok", "response": "ok"},
    ]
    assert len(preference_pairs(events)) == 1
    result = nightly(
        events,
        [{"id": "base"}],
        lambda cases: {"success_rate": 0.9 if any(str(case["id"]).startswith("flywheel") for case in cases) else 0.5},
        DryRunTrainer(tmp_path / "lora"),
        dataset_path="pairs.jsonl",
    )
    assert result["job"]["status"] == "dry_run"
    manifest = json.loads((tmp_path / "lora" / "lora_job.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert "No weights were trained" in manifest["note"]
    assert result["success_delta"] > 0

    points = summarize(example_points())
    frontier = {(point["tenant"], point["policy"]) for point in points["frontier"]}
    assert ("north", "wasted") not in frontier
    assert ("north", "system-one") in frontier
    assert ("north", "system-two") in frontier
    html = render_dashboard(points)
    assert "<svg" in html
    assert 'data-frontier="1"' in html
    assert "north" in html and "south" in html

    tools = decision_tools({"A1000": {"status": "delivered", "days_ago": 90}})
    replayed = replay(refund_trace(days_ago=10), "lookup", decision_llm, tools)
    assert [call["mode"] for call in replayed["calls"]] == ["recorded", "swapped", "replayed", "replayed"]
    assert replayed["outputs"]["plan"]["intent"] == "refund"
    assert replayed["outputs"]["lookup"]["days_ago"] == 90
    assert replayed["outputs"]["act"]["refunded"] is False
    assert replayed["outputs"]["act"]["escalated"] is True

    salience = run_eviction(SaliencePolicy())
    fifo = run_eviction(FifoPolicy())
    assert salience["needle_retained"] is True
    assert salience["retrieved"] == "needle-refund-window"
    assert fifo["needle_retained"] is False
    assert salience["used"] <= salience["budget"]

    leaked = "An order can be refunded within 30 days of the delivery date."
    report = scan(
        [
            {"id": "eval-leaked", "input": {"query": leaked}, "expected": {"reference": leaked}},
            {"id": "eval-clean", "input": {"query": "How do I reset a refrigerator compressor?"}, "expected": {}},
        ],
        [{"id": "train", "text": leaked}],
    )
    by_id = {row["eval_id"]: row for row in report["findings"]}
    assert by_id["eval-leaked"]["leaked"] is True
    assert by_id["eval-clean"]["leaked"] is False

    assert injection_results(guarded_answer)["attack_leak_rate"] == 0.0
    assert injection_results(unguarded_answer)["attack_leak_rate"] == 1.0
    assert schema_results()["accuracy"] == 1.0
    assert run_loop(looping_agent, "refund")["status"] == "loop_broken"

    generated = EdgeCaseGenerator(complete=lambda prompt: "refund A1000 but the written policy is reversed").generate(
        [{"id": "seed", "input": {"query": "What is the refund window?", "order_id": "A1000"}}]
    )
    operators = {case["operator"] for case in generated}
    assert "empty" in operators and "amount_negative" in operators and "frontier_paraphrase" in operators
    assert any(case["source"] == "frontier" for case in generated)


def test_plugin_emits_a_metric_report(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "expected": "ok"}),
                json.dumps({"id": "p2", "expected": "ok"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    test_file = tmp_path / "test_eval_demo.py"
    test_file.write_text(
        "def test_case(eval_case, eval_recorder):\n"
        "    passed = eval_case['id'] == 'p1'\n"
        "    eval_recorder(case_id=eval_case['id'], correct=passed)\n"
        "    assert passed\n",
        encoding="utf-8",
    )
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-p",
            "evals_engineer.plugin",
            f"--eval-dataset={dataset}",
            "-q",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert "eval metric report" in output
    assert "fail p2" in output
    assert completed.returncode != 0


def test_methodology_quotes_the_judge_prompt_and_the_dag_rule():
    judge = (ROOT / "docs" / "methodology" / "02-judge-calibration.md").read_text(encoding="utf-8")
    stats = (ROOT / "docs" / "methodology" / "03-statistical-framework.md").read_text(encoding="utf-8")
    trajectory = (ROOT / "docs" / "methodology" / "01-trajectory-grading.md").read_text(encoding="utf-8")
    assert "Do not reward length." in judge
    assert "0.4*groundedness + 0.3*completeness + 0.2*abstention + 0.1*constraint" in judge
    assert JUDGE_PROMPT.splitlines()[0] in judge
    assert "McNemar" in stats
    assert "check_entitlement" in trajectory
    assert "not one narration" in trajectory.lower() or "not a single narration" in trajectory.lower()
