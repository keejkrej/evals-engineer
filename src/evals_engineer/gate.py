"""CI regression gate with hysteresis, flaky detection, cost budget, and caching.

Success fails closed when it drops more than 2 points against the baseline.
Once tripped, the gate stays tripped until the drop is back within 1 point.
Latency, cost, and flaky cases are reported on the same decision.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from evals_engineer.metrics import percentile


def case_key(payload) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(raw.encode("utf-8")).hexdigest()


class EvalCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get(self, key: str):
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / f"{key}.json").write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


DEFAULT_CONFIG = {
    "max_success_drop": 0.02,
    "recover_drop": 0.01,
    "latency_ratio": 1.25,
    "latency_min_delta_ms": 50,
    "max_cost_usd": 25.0,
    "flaky_low": 0.2,
    "flaky_high": 0.8,
    "min_flake_seeds": 5,
    "fail_on_flaky": True,
}


def summarize_run(name: str, cases: list[dict]) -> dict:
    """cases: id, passed, latency_ms, cost_usd, optional seed_passes list."""
    if not cases:
        raise ValueError("run has no cases")
    successes = sum(1 for case in cases if case["passed"])
    latencies = [case["latency_ms"] for case in cases]
    return {
        "name": name,
        "n": len(cases),
        "success_rate": successes / len(cases),
        "latency_p95_ms": percentile(latencies, 0.95),
        "cost_usd": sum(case["cost_usd"] for case in cases),
        "cases": cases,
    }


def flaky_cases(cases: list[dict], config: dict | None = None) -> list[dict]:
    config = {**DEFAULT_CONFIG, **(config or {})}
    found = []
    for case in cases:
        seeds = case.get("seed_passes")
        if not seeds or len(seeds) < config["min_flake_seeds"]:
            continue
        rate = sum(1 for value in seeds if value) / len(seeds)
        if config["flaky_low"] < rate < config["flaky_high"]:
            found.append({"id": case["id"], "pass_rate": rate, "seeds": len(seeds)})
    return found


def load_state(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {"tripped": False}
    return json.loads(file_path.read_text(encoding="utf-8"))


def save_state(path: str | Path, state: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decide(current: dict, baseline: dict, state: dict | None = None, config: dict | None = None) -> dict:
    config = {**DEFAULT_CONFIG, **(config or {})}
    state = {"tripped": False} if state is None else dict(state)
    drop = baseline["success_rate"] - current["success_rate"]
    was_tripped = bool(state.get("tripped"))
    reasons = []
    failing_cases = []
    baseline_by_id = {case["id"]: case for case in baseline["cases"]}
    for case in current["cases"]:
        previous = baseline_by_id.get(case["id"])
        if previous and previous["passed"] and not case["passed"]:
            failing_cases.append(
                {
                    "id": case["id"],
                    "baseline": "pass",
                    "current": "fail",
                    "latency_ms": case["latency_ms"],
                }
            )
    # Hysteresis applies to the success drop only. Latency and cost fail the
    # current run without latching the success band. The epsilon keeps a drop
    # of exactly 2/100 from tripping on binary float noise.
    if was_tripped:
        success_ok = drop <= config["recover_drop"] + 1e-9
    else:
        success_ok = drop <= config["max_success_drop"] + 1e-9
    if not success_ok:
        band = config["recover_drop"] if was_tripped else config["max_success_drop"]
        reasons.append(
            f"Task success {current['success_rate']:.1%} vs baseline {baseline['success_rate']:.1%} "
            f"(drop {drop:.1%}, limit {band:.1%})."
        )

    latency_limit = max(
        baseline["latency_p95_ms"] * config["latency_ratio"],
        baseline["latency_p95_ms"] + config["latency_min_delta_ms"],
    )
    # A spike is a relative jump and an absolute jump. Small noise does not trip.
    latency_spike = (
        current["latency_p95_ms"] > baseline["latency_p95_ms"] * config["latency_ratio"]
        and current["latency_p95_ms"] > baseline["latency_p95_ms"] + config["latency_min_delta_ms"]
    )
    if latency_spike:
        reasons.append(
            f"Latency p95 {current['latency_p95_ms']:.0f} ms vs baseline {baseline['latency_p95_ms']:.0f} ms "
            f"(spike limit {latency_limit:.0f} ms)."
        )
    if current["cost_usd"] > config["max_cost_usd"]:
        reasons.append(
            f"Cost ${current['cost_usd']:.2f} exceeds budget ${config['max_cost_usd']:.2f}."
        )
    flaky = flaky_cases(current["cases"], config)
    if flaky and config["fail_on_flaky"]:
        reasons.append(f"Flaky cases: {', '.join(item['id'] for item in flaky)}.")
    passed = not reasons
    return {
        "passed": passed,
        "drop": drop,
        "reasons": reasons,
        "failing_cases": failing_cases,
        "flaky": flaky,
        "state": {"tripped": not success_ok},
        "was_tripped": was_tripped,
        "success_ok": success_ok,
    }


def comment_markdown(decision: dict, current: dict, baseline: dict) -> str:
    status = "PASS" if decision["passed"] else "FAIL"
    lines = [
        f"## Eval gate: {status}",
        "",
        f"Task success {current['success_rate']:.1%} vs baseline {baseline['success_rate']:.1%} "
        f"(drop {decision['drop']:.1%}).",
        f"Latency p95 {current['latency_p95_ms']:.0f} ms vs baseline {baseline['latency_p95_ms']:.0f} ms.",
        f"Cost ${current['cost_usd']:.2f}.",
        "",
    ]
    if decision["reasons"]:
        lines.append("### Reasons")
        lines.extend(f"- {reason}" for reason in decision["reasons"])
        lines.append("")
    if decision["failing_cases"]:
        lines.append("### Failing cases")
        lines.append("| id | baseline | current | latency_ms |")
        lines.append("| --- | --- | --- | ---: |")
        for case in decision["failing_cases"]:
            lines.append(
                f"| {case['id']} | {case['baseline']} | {case['current']} | {case['latency_ms']} |"
            )
        lines.append("")
    if decision["flaky"]:
        lines.append("### Flaky cases")
        for item in decision["flaky"]:
            lines.append(f"- {item['id']}: pass rate {item['pass_rate']:.0%} across {item['seeds']} seeds")
        lines.append("")
    lines.append(
        "Hysteresis: a drop above 2% trips the gate. It clears only when the drop is at or under 1%."
    )
    return "\n".join(lines) + "\n"


def run_gate(
    current: dict,
    baseline: dict,
    state_path: str | Path | None = None,
    comment_path: str | Path | None = None,
    config: dict | None = None,
    cache: EvalCache | None = None,
    cache_payload=None,
) -> dict:
    if cache is not None and cache_payload is not None:
        key = case_key(cache_payload)
        cached = cache.get(key)
        if cached is not None:
            cached["cache"] = "hit"
            return cached
    state = load_state(state_path) if state_path else {"tripped": False}
    decision = decide(current, baseline, state=state, config=config)
    decision["comment"] = comment_markdown(decision, current, baseline)
    decision["cache"] = "miss"
    if state_path is not None:
        save_state(state_path, decision["state"])
    if comment_path is not None:
        path = Path(comment_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(decision["comment"], encoding="utf-8")
    if cache is not None and cache_payload is not None:
        cache.put(case_key(cache_payload), decision)
    return decision
