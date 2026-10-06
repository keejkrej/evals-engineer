"""Shadow router.

The user always receives the primary response. A stable 5% sample is also sent
to the candidate model. A candidate exception does not change the user response
and does not fail the request.
"""

from __future__ import annotations

from evals_engineer.report import Metric, Report
from evals_engineer.textutil import normalized_similarity, should_sample


def serve(request: dict, primary, shadow, rate: float = 0.05) -> dict:
    user = primary(request)
    mirrored = None
    error = None
    sampled = should_sample(str(request["id"]), rate)
    if sampled:
        try:
            mirrored = shadow(request)
        except Exception as exc:  # a shadow failure must not reach the user
            error = f"{type(exc).__name__}: {exc}"
    return {
        "request_id": request["id"],
        "user": user,
        "sampled": sampled,
        "shadow": mirrored,
        "shadow_error": error,
    }


def _steps(response: dict) -> list:
    return [tuple(sorted(step.items())) if isinstance(step, dict) else step for step in response.get("steps", [])]


def compare_pair(primary_response: dict, shadow_response: dict) -> dict:
    text_same = primary_response.get("text") == shadow_response.get("text")
    similarity = normalized_similarity(_steps(primary_response), _steps(shadow_response))
    return {
        "text_same": text_same,
        "trajectory_similarity": similarity,
        "cost_delta_usd": float(shadow_response.get("cost_usd", 0)) - float(primary_response.get("cost_usd", 0)),
        "quality_delta": float(shadow_response.get("quality", 0)) - float(primary_response.get("quality", 0)),
    }


def shadow_experiment(requests: list[dict], primary, shadow, rate: float = 0.05) -> dict:
    served = [serve(request, primary, shadow, rate=rate) for request in requests]
    diffs = []
    for item in served:
        if not item["sampled"] or item["shadow"] is None:
            continue
        diffs.append(compare_pair(item["user"], item["shadow"]))
    n = len(served)
    sampled = sum(1 for item in served if item["sampled"])
    errors = sum(1 for item in served if item["shadow_error"])
    user_intact = all(item["user"] == primary(request) for item, request in zip(served, requests))

    def avg(key: str) -> float:
        if not diffs:
            return 0.0
        return sum(item[key] for item in diffs) / len(diffs)

    return {
        "n": n,
        "sampled": sampled,
        "sample_rate": sampled / n if n else 0.0,
        "shadow_errors": errors,
        "user_responses_match_primary": user_intact,
        "compared": len(diffs),
        "text_agreement": avg("text_same"),
        "mean_trajectory_similarity": avg("trajectory_similarity"),
        "mean_cost_delta_usd": avg("cost_delta_usd"),
        "mean_quality_delta": avg("quality_delta"),
        "rows": served,
    }


def shadow_report(result: dict) -> Report:
    return Report(
        title="Shadow routing comparator",
        passed=result["user_responses_match_primary"],
        summary=(
            f"Mirrored {result['sampled']} of {result['n']} requests "
            f"({result['sample_rate']:.1%}). User responses match the primary model. "
            f"Contained shadow errors: {result['shadow_errors']}."
        ),
        metrics=[
            Metric("sample_rate", result["sample_rate"]),
            Metric("text_agreement", result["text_agreement"]),
            Metric("trajectory_similarity", result["mean_trajectory_similarity"]),
            Metric("cost_delta_usd", result["mean_cost_delta_usd"]),
            Metric("quality_delta", result["mean_quality_delta"]),
        ],
        lines=["Shadow exceptions are recorded and are not returned to the user."],
    )
