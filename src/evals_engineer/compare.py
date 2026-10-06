"""Paired comparison of two models.

The published sentence is a paired risk difference with a percentile bootstrap
interval and an exact McNemar p-value. A two-point gap on a tiny discordant
set stays non-significant.
"""

from __future__ import annotations

from evals_engineer.metrics import (
    cohens_h,
    mcnemar_exact,
    mean,
    paired_bootstrap_diff,
    sample_size_two_proportions,
    seed_variance,
)
from evals_engineer.report import Metric, Report


def compare_binary(
    name_a: str,
    outcomes_a: list[bool],
    name_b: str,
    outcomes_b: list[bool],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    if len(outcomes_a) != len(outcomes_b) or not outcomes_a:
        raise ValueError("outcomes must be aligned and non-empty")
    a = [1.0 if value else 0.0 for value in outcomes_a]
    b = [1.0 if value else 0.0 for value in outcomes_b]
    boot = paired_bootstrap_diff(a, b, n_boot=n_boot, seed=seed)
    test = mcnemar_exact(outcomes_a, outcomes_b)
    p_value = test["p"]
    rate_a = mean(a)
    rate_b = mean(b)
    diff = boot["point"]
    half = (boot["high"] - boot["low"]) / 2
    winner = name_b if diff >= 0 else name_a
    magnitude = abs(diff)
    if p_value < alpha:
        p_text = f"p<{alpha:.2f}"
        significant = True
    else:
        p_text = f"p={p_value:.2f}"
        significant = False
    sentence = f"{winner} wins by {magnitude * 100:.1f}% ± {half * 100:.1f}% ({p_text})"
    if not significant:
        sentence = f"No significant difference: {sentence}."
    baseline_rate = min(0.97, max(0.03, rate_a))
    alternate = baseline_rate - 0.02 if baseline_rate >= 0.5 else baseline_rate + 0.02
    n_needed = sample_size_two_proportions(baseline_rate, alternate)
    return {
        "name_a": name_a,
        "name_b": name_b,
        "n": len(a),
        "rate_a": rate_a,
        "rate_b": rate_b,
        "diff": diff,
        "ci_low": boot["low"],
        "ci_high": boot["high"],
        "half_width": half,
        "p": p_value,
        "p_bootstrap": boot["p_bootstrap"],
        "significant": significant,
        "effect_h": cohens_h(rate_b, rate_a),
        "discordant": test,
        "sentence": sentence,
        "sample_size_per_model_for_2pt": n_needed,
        "seed": seed,
        "n_boot": n_boot,
    }


def compare_report(result: dict) -> Report:
    return Report(
        title="Statistical comparison",
        passed=True,
        summary=result["sentence"],
        metrics=[
            Metric("diff", result["diff"]),
            Metric("ci_low", result["ci_low"]),
            Metric("ci_high", result["ci_high"]),
            Metric("p_value", result["p"]),
            Metric("cohens_h", result["effect_h"]),
        ],
        lines=[
            f"McNemar discordant pairs: {result['discordant']['candidate_only']} candidate-only, {result['discordant']['baseline_only']} baseline-only.",
            "The p-value is the exact McNemar test. The ± term is half the width of the 95% paired bootstrap interval.",
        ],
    )


def measure_seed_variance(score_at_seed, seeds: list[int]) -> dict:
    return seed_variance(score_at_seed, seeds)
