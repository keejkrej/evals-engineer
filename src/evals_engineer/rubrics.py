"""Judge rubric and prompts.

The harness recomputes `overall` from the criterion scores. A model that
inflates the overall field does not move the recorded score.
"""

from __future__ import annotations

CRITERIA = ("groundedness", "completeness", "abstention", "constraint")

WEIGHTS = {
    "groundedness": 0.4,
    "completeness": 0.3,
    "abstention": 0.2,
    "constraint": 0.1,
}

# Doubled braces are escapes for str.format. The rendered prompt contains single braces.
JUDGE_PROMPT = """You are an evaluation judge for Harbor, a customer-support agent.
Score the candidate response against the rubric. Use only the task, the context, and the response.
Do not reward length. Do not prefer the first candidate. Do not prefer a response because it sounds like you.

Rubric. Each criterion is an integer from 1 to 5.
- groundedness: 5 means every factual claim is supported by the context. 1 means the response invents facts.
- completeness: 5 means the response covers every part of the task that the context supports. 1 means it misses the task.
- abstention: 5 means the response abstains when the context is missing or contradictory, and answers when the context is sufficient. 1 means the opposite.
- constraint: 5 means the response follows format and safety constraints. 1 means it breaks them.

Return JSON only, with this shape:
{{"groundedness": 1, "completeness": 1, "abstention": 1, "constraint": 1, "overall": 1, "rationale": "one sentence"}}

The overall score is the weighted value rounded to the nearest integer:
0.4*groundedness + 0.3*completeness + 0.2*abstention + 0.1*constraint, then clamped to 1..5.
The weights are fixed. Do not change them.

Task:
{task}

Context:
{context}

Constraints:
{constraints}

Candidate response:
{response}
"""

PAIR_PROMPT = """You compare two Harbor support responses to the same task.
Choose the response that better satisfies groundedness, completeness, abstention, and constraint.
Do not prefer the response only because it is longer.
Do not prefer the response only because it is shown first.
Do not prefer a response because it sounds like your own writing.

Return JSON only:
{{"preference": "A", "rationale": "one sentence"}}
preference is "A", "B", or "tie".

Task:
{task}

Context:
{context}

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}
"""

SCORE_LABELS = [1, 2, 3, 4, 5]


def weighted_overall(criteria: dict[str, int]) -> int:
    missing = [name for name in CRITERIA if name not in criteria]
    if missing:
        raise ValueError(f"missing criteria: {missing}")
    total = sum(WEIGHTS[name] * int(criteria[name]) for name in CRITERIA)
    rounded = int(total + 0.5) if total >= 0 else int(total - 0.5)
    return max(1, min(5, rounded))
