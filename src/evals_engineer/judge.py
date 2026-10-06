"""LLM-as-a-judge with a fixed rubric and bias calibration.

`complete(prompt, sample)` is the model boundary. The bundled models are
scripted so calibration runs offline. A hosted model implements the same
method and ignores `sample` if it only reads the prompt. The harness
recomputes the weighted overall score from the four criteria.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from evals_engineer.domain import ABSTAIN_TEXT, POLICIES
from evals_engineer.metrics import mean, quadratic_weighted_kappa
from evals_engineer.report import Metric, Report
from evals_engineer.rubrics import CRITERIA, JUDGE_PROMPT, PAIR_PROMPT, SCORE_LABELS, weighted_overall
from evals_engineer.textutil import rough_tokens

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# A judge is an instrument only when these all hold.
CALIBRATION_GATES = {
    "min_quadratic_kappa": 0.8,
    "max_abs_verbosity_delta": 0.25,
    "max_position_bias_rate": 0.1,
    "max_abs_self_preference_delta": 0.25,
}


@dataclass
class Sample:
    sample_id: str
    task: str
    context: str
    response: str
    constraints: str = ""
    facts: tuple[str, ...] = ()
    must_abstain: bool = False
    author: str = "other"
    pair_id: str = ""
    bias_tag: str = "neutral"
    human_score: int = 0
    human_criteria: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.sample_id,
            "task": self.task,
            "context": self.context,
            "response": self.response,
            "constraints": self.constraints,
            "facts": list(self.facts),
            "must_abstain": self.must_abstain,
            "author": self.author,
            "pair_id": self.pair_id,
            "bias_tag": self.bias_tag,
            "human_score": self.human_score,
            "human_criteria": self.human_criteria,
        }


def render_prompt(sample: Sample) -> str:
    return JUDGE_PROMPT.format(
        task=sample.task,
        context=sample.context,
        constraints=sample.constraints or "None.",
        response=sample.response,
    )


def render_pair_prompt(task: str, context: str, candidate_a: str, candidate_b: str) -> str:
    return PAIR_PROMPT.format(
        task=task, context=context, candidate_a=candidate_a, candidate_b=candidate_b
    )


def parse_judge_json(raw: str) -> dict:
    match = _JSON_OBJECT.search(raw or "")
    if not match:
        raise ValueError("judge output did not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge JSON must be an object")
    return payload


def rubric_criteria(sample: Sample) -> dict[str, int]:
    """Deterministic criterion scores from the response and the context.

    These are the adjudicator labels. A calibrated model reproduces them.
    A drifted or biased model does not.
    """
    text = sample.response or ""
    lowered = text.lower()
    abstained = ABSTAIN_TEXT.lower() in lowered or "i don't have enough" in lowered
    if sample.must_abstain:
        groundedness = 5 if abstained else 1
        completeness = 5 if abstained else 2
        abstention = 5 if abstained else 1
    else:
        if sample.facts:
            hits = sum(1 for fact in sample.facts if str(fact).lower() in lowered)
            coverage = hits / len(sample.facts)
        else:
            coverage = 1.0 if text.strip() else 0.0
        invented = "definitely allows" in lowered or "already refunded" in lowered
        groundedness = 1 if invented else max(1, min(5, round(1 + 4 * coverage)))
        completeness = max(1, min(5, round(1 + 4 * coverage)))
        abstention = 2 if abstained else 5
    constraint = 5
    if "canary-leak" in lowered:
        constraint = 1
    if sample.constraints and "three bullets" in sample.constraints.lower():
        bullets = [line for line in text.splitlines() if line.strip().startswith("- ")]
        constraint = 5 if len(bullets) == 3 else 2
    return {
        "groundedness": groundedness,
        "completeness": completeness,
        "abstention": abstention,
        "constraint": constraint,
    }


def _clamp_criterion(value: int) -> int:
    return max(1, min(5, int(value)))


class ScriptedJudge:
    """Offline judge. Biases are explicit knobs so the calibration report can see them."""

    def __init__(
        self,
        verbosity_bias: int = 0,
        position_bias: bool = False,
        self_preference: int = 0,
        drift_abstention: bool = False,
        name: str = "calibrated",
    ):
        self.verbosity_bias = verbosity_bias
        self.position_bias = position_bias
        self.self_preference = self_preference
        self.drift_abstention = drift_abstention
        self.name = name

    def complete(self, prompt: str, sample: Sample) -> str:
        if not prompt.strip():
            raise ValueError("prompt is empty")
        criteria = rubric_criteria(sample)
        if self.drift_abstention:
            # A drifted judge marks every criterion as perfect and stops reading.
            for name in criteria:
                criteria[name] = 5
        # Penalise the short response and the "other" author so the bias is visible
        # when the favored side is already at the top of the 1-5 scale.
        if self.verbosity_bias and rough_tokens(sample.response) < 80:
            criteria["groundedness"] = _clamp_criterion(criteria["groundedness"] - self.verbosity_bias)
        if self.self_preference and sample.author == "other":
            criteria["completeness"] = _clamp_criterion(criteria["completeness"] - self.self_preference)
        criteria["overall"] = weighted_overall(criteria)
        criteria["rationale"] = f"{self.name} scored the rubric from the response and the context."
        return json.dumps(criteria)

    def prefer(self, prompt: str, left: Sample, right: Sample) -> str:
        if not prompt.strip():
            raise ValueError("prompt is empty")
        if self.position_bias:
            return json.dumps({"preference": "A", "rationale": "position bias prefers the first candidate"})
        left_score = weighted_overall(rubric_criteria(left))
        right_score = weighted_overall(rubric_criteria(right))
        if left_score > right_score:
            choice = "A"
        elif right_score > left_score:
            choice = "B"
        else:
            choice = "tie"
        return json.dumps({"preference": choice, "rationale": "preference follows the weighted rubric"})


def score_sample(judge: ScriptedJudge, sample: Sample) -> dict:
    prompt = render_prompt(sample)
    raw = judge.complete(prompt, sample)
    parsed = parse_judge_json(raw)
    criteria = {name: _clamp_criterion(parsed[name]) for name in CRITERIA}
    return {
        "id": sample.sample_id,
        "criteria": criteria,
        "overall": weighted_overall(criteria),
        "model_overall": parsed.get("overall"),
        "prompt_has_weights": "0.4*groundedness" in prompt,
        "rationale": parsed.get("rationale", ""),
    }


def _pad_verbose(text: str) -> str:
    filler = " ".join(["This restates the same grounded fact without adding a new claim."] * 12)
    return text.rstrip() + "\n" + filler


def build_anchors(n: int = 500) -> list[Sample]:
    """500 anchors: quality labels, verbosity pairs, position pairs, self/other pairs.

    Human scores are the rubric adjudicator (rater ada). Rater ben shifts one
    point on every 7th neutral item. That noise is recorded on the sample only
    through `human_score` (ada) and a parallel list from `ben_scores`.
    """
    if n != 500:
        raise ValueError("the published anchor set is 500 items")
    samples: list[Sample] = []
    # 200 neutral quality items: 25 policies x 8 variants.
    variants = (
        ("good", False),
        ("good", False),
        ("partial", False),
        ("wrong", False),
        ("abstain_ok", True),
        ("abstain_bad", True),
        ("good", False),
        ("wrong", False),
    )
    for index in range(200):
        policy = POLICIES[index % len(POLICIES)]
        kind, must = variants[index % len(variants)]
        # Abstain variants are unanswerable tasks, not the policy question.
        if must:
            task = f"Unanswerable support question {index}."
            context = "The knowledge base has no document about this question."
            facts: tuple[str, ...] = ()
        else:
            task = policy.question
            context = policy.as_doc()["text"]
            facts = tuple(policy.slots.values())
        if kind == "good":
            response = policy.text
        elif kind == "partial":
            response = policy.title + "."
        elif kind == "wrong":
            response = "Yes. The policy definitely allows this and the order is already refunded."
        elif kind == "abstain_ok":
            response = ABSTAIN_TEXT
        else:
            response = "The warehouse cutoff promise is 4 days and the order is already refunded."
        sample = Sample(
            sample_id=f"anchor-neutral-{index:03d}",
            task=task,
            context=context,
            response=response,
            facts=facts,
            must_abstain=must,
            bias_tag="neutral",
            pair_id=f"neutral-{index:03d}",
        )
        criteria = rubric_criteria(sample)
        sample.human_criteria = criteria
        sample.human_score = weighted_overall(criteria)
        samples.append(sample)

    # 50 verbosity pairs = 100 items. Same facts, short and long.
    for index in range(50):
        policy = POLICIES[index % len(POLICIES)]
        short = Sample(
            sample_id=f"anchor-terse-{index:03d}",
            task=policy.question,
            context=policy.as_doc()["text"],
            response=policy.text,
            facts=tuple(policy.slots.values()),
            author="rater",
            bias_tag="terse",
            pair_id=f"verbose-{index:03d}",
        )
        long = Sample(
            sample_id=f"anchor-verbose-{index:03d}",
            task=policy.question,
            context=policy.as_doc()["text"],
            response=_pad_verbose(policy.text),
            facts=tuple(policy.slots.values()),
            author="rater",
            bias_tag="verbose",
            pair_id=f"verbose-{index:03d}",
        )
        for sample in (short, long):
            criteria = rubric_criteria(sample)
            sample.human_criteria = criteria
            sample.human_score = weighted_overall(criteria)
            samples.append(sample)

    # 50 position pairs = 100 items. One good response, one wrong response.
    for index in range(50):
        policy = POLICIES[index % len(POLICIES)]
        good = policy.text
        bad = "Yes. The policy definitely allows this and the order is already refunded."
        for side, response, tag in (
            ("a", good, "position_good_first"),
            ("b", bad, "position_bad_second"),
        ):
            sample = Sample(
                sample_id=f"anchor-position-{index:03d}-{side}",
                task=policy.question,
                context=policy.as_doc()["text"],
                response=response,
                facts=tuple(policy.slots.values()),
                bias_tag=tag,
                pair_id=f"position-{index:03d}",
            )
            criteria = rubric_criteria(sample)
            sample.human_criteria = criteria
            sample.human_score = weighted_overall(criteria)
            samples.append(sample)

    # 50 self/other pairs = 100 items. Identical text, different author tag.
    for index in range(50):
        policy = POLICIES[index % len(POLICIES)]
        for author in ("self", "other"):
            sample = Sample(
                sample_id=f"anchor-{author}-{index:03d}",
                task=policy.question,
                context=policy.as_doc()["text"],
                response=_pad_verbose(policy.text),
                facts=tuple(policy.slots.values()),
                author=author,
                bias_tag=author,
                pair_id=f"self-{index:03d}",
            )
            criteria = rubric_criteria(sample)
            sample.human_criteria = criteria
            sample.human_score = weighted_overall(criteria)
            samples.append(sample)

    if len(samples) != 500:
        raise RuntimeError(f"expected 500 anchors, built {len(samples)}")
    return samples


def ben_scores(samples: list[Sample]) -> list[int]:
    """Second rater: agrees with ada except a one-point drop on every 7th neutral item."""
    scores = []
    for sample in samples:
        score = sample.human_score
        if sample.bias_tag == "neutral":
            number = int(sample.sample_id.rsplit("-", 1)[-1])
            if number % 7 == 0:
                score = max(1, score - 1)
        scores.append(score)
    return scores


def _grouped(samples: list[Sample], tag_prefix: str | None = None) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        if tag_prefix and not sample.pair_id.startswith(tag_prefix):
            continue
        groups.setdefault(sample.pair_id, []).append(sample)
    return groups


def verbosity_delta(judge: ScriptedJudge, samples: list[Sample]) -> float:
    deltas = []
    for pair in _grouped(samples, "verbose").values():
        by_tag = {sample.bias_tag: sample for sample in pair}
        if "verbose" not in by_tag or "terse" not in by_tag:
            continue
        long_score = score_sample(judge, by_tag["verbose"])["overall"]
        short_score = score_sample(judge, by_tag["terse"])["overall"]
        deltas.append(long_score - short_score)
    return mean(deltas) if deltas else 0.0


def self_preference_delta(judge: ScriptedJudge, samples: list[Sample]) -> float:
    deltas = []
    for pair in _grouped(samples, "self").values():
        by_author = {sample.author: sample for sample in pair}
        self_score = score_sample(judge, by_author["self"])["overall"]
        other_score = score_sample(judge, by_author["other"])["overall"]
        deltas.append(self_score - other_score)
    return mean(deltas) if deltas else 0.0


def position_bias_rate(judge: ScriptedJudge, samples: list[Sample]) -> float:
    """Fraction of pairs where the judge picks whichever candidate is shown first, in both orders."""
    biased = 0
    total = 0
    for pair in _grouped(samples, "position").values():
        by_tag = {sample.bias_tag: sample for sample in pair}
        good = by_tag["position_good_first"]
        bad = by_tag["position_bad_second"]
        forward = render_pair_prompt(good.task, good.context, good.response, bad.response)
        reverse = render_pair_prompt(good.task, good.context, bad.response, good.response)
        first = parse_judge_json(judge.prefer(forward, good, bad))["preference"]
        # In the reversed prompt, candidate A is the bad response.
        second = parse_judge_json(judge.prefer(reverse, bad, good))["preference"]
        total += 1
        if first == "A" and second == "A":
            biased += 1
    return biased / total if total else 0.0


def calibrate(judge: ScriptedJudge, samples: list[Sample] | None = None) -> dict:
    samples = build_anchors() if samples is None else samples
    neutrals = [sample for sample in samples if sample.bias_tag == "neutral"]
    judged = [score_sample(judge, sample)["overall"] for sample in neutrals]
    human = [sample.human_score for sample in neutrals]
    second = [score for score, sample in zip(ben_scores(samples), samples) if sample.bias_tag == "neutral"]
    qwk_human = quadratic_weighted_kappa(human, judged, SCORE_LABELS)
    qwk_raters = quadratic_weighted_kappa(human, second, SCORE_LABELS)
    qwk_second = quadratic_weighted_kappa(second, judged, SCORE_LABELS)
    verb = verbosity_delta(judge, samples)
    position = position_bias_rate(judge, samples)
    self_delta = self_preference_delta(judge, samples)
    gates = {
        "quadratic_kappa": qwk_human["kappa"] >= CALIBRATION_GATES["min_quadratic_kappa"],
        "verbosity": abs(verb) <= CALIBRATION_GATES["max_abs_verbosity_delta"],
        "position": position <= CALIBRATION_GATES["max_position_bias_rate"],
        "self_preference": abs(self_delta) <= CALIBRATION_GATES["max_abs_self_preference_delta"],
    }
    return {
        "judge": judge.name,
        "n_anchors": len(samples),
        "n_neutral": len(neutrals),
        "quadratic_kappa_vs_adjudicator": qwk_human["kappa"],
        "quadratic_kappa_vs_second_rater": qwk_second["kappa"],
        "rater_quadratic_kappa": qwk_raters["kappa"],
        "verbosity_delta": verb,
        "position_bias_rate": position,
        "self_preference_delta": self_delta,
        "gates": gates,
        "calibrated": all(gates.values()),
    }


def judge_report(result: dict) -> Report:
    return Report(
        title=f"Judge calibration: {result['judge']}",
        passed=result["calibrated"],
        summary=(
            f"Quadratic weighted kappa vs adjudicator {result['quadratic_kappa_vs_adjudicator']:.3f}. "
            f"Verbosity delta {result['verbosity_delta']:.3f}. "
            f"Position bias {result['position_bias_rate']:.3f}. "
            f"Self-preference delta {result['self_preference_delta']:.3f}."
        ),
        metrics=[
            Metric("quadratic_kappa_vs_adjudicator", result["quadratic_kappa_vs_adjudicator"]),
            Metric("quadratic_kappa_vs_second_rater", result["quadratic_kappa_vs_second_rater"]),
            Metric("rater_quadratic_kappa", result["rater_quadratic_kappa"]),
            Metric("verbosity_delta", result["verbosity_delta"]),
            Metric("position_bias_rate", result["position_bias_rate"]),
            Metric("self_preference_delta", result["self_preference_delta"]),
        ],
        lines=[
            "Overall is recomputed as 0.4*groundedness + 0.3*completeness + 0.2*abstention + 0.1*constraint.",
            "The second rater drops one point on every 7th neutral anchor. That disagreement is the rater kappa, not judge drift.",
        ],
    )
