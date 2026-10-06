# Judge calibration

An uncalibrated judge is a vibe with a temperature. This note is the instrument: the rubric, the prompt, the score the harness actually stores, and the three bias probes.

## Rubric

Each criterion is an integer from 1 to 5.

- groundedness: every factual claim is supported by the context.
- completeness: the response covers the part of the task the context supports.
- abstention: the response abstains when the context is missing or contradictory, and answers when it is sufficient.
- constraint: the response keeps format and safety constraints.

The stored overall score is not the model's `overall` field. The harness recomputes it and clamps it to 1..5:

```
0.4*groundedness + 0.3*completeness + 0.2*abstention + 0.1*constraint
```

Weights are fixed. A model that inflates `overall` does not move the recorded score.

## Prompt

The single-response prompt is the constant `JUDGE_PROMPT`.

```
You are an evaluation judge for Harbor, a customer-support agent.
Score the candidate response against the rubric. Use only the task, the context, and the response.
Do not reward length. Do not prefer the first candidate. Do not prefer a response because it sounds like you.

Rubric. Each criterion is an integer from 1 to 5.
- groundedness: 5 means every factual claim is supported by the context. 1 means the response invents facts.
- completeness: 5 means the response covers every part of the task that the context supports. 1 means it misses the task.
- abstention: 5 means the response abstains when the context is missing or contradictory, and answers when the context is sufficient. 1 means the opposite.
- constraint: 5 means the response follows format and safety constraints. 1 means it breaks them.

Return JSON only, with this shape:
{"groundedness": 1, "completeness": 1, "abstention": 1, "constraint": 1, "overall": 1, "rationale": "one sentence"}

The overall score is the weighted value rounded to the nearest integer:
0.4*groundedness + 0.3*completeness + 0.2*abstention + 0.1*constraint, then clamped to 1..5.
The weights are fixed. Do not change them.
```

The pairwise prompt asks for `{"preference": "A"|"B"|"tie"}` and tells the model not to prefer the candidate shown first.

## Anchors

`build_anchors()` returns 500 items.

- 200 neutral items, with adjudicator scores from the rubric features.
- 50 verbosity pairs (100 items): the same facts, a short answer and a padded answer.
- 50 position pairs (100 items): one supported answer and one invented answer.
- 50 self/other pairs (100 items): the same text, tagged as the judge's own writing or someone else's.

Ada is the adjudicator. Ben drops one point on every 7th neutral item. Rater disagreement is reported separately from judge error. Both raters are simulated. Point a real judge at the same functions by implementing `complete(prompt, sample)`.

## Probes and gates

A judge is calibrated only when all of these hold on this anchor set:

- quadratic weighted kappa against the adjudicator is at least 0.8
- mean(score of the long answer minus score of the short answer) is at most 0.25 in absolute value
- the rate at which the judge picks whichever candidate is shown first, in both orders, is at most 0.1
- mean(score when the author tag is `self` minus score when it is `other`) is at most 0.25 in absolute value

The bundled calibrated model meets the gates. The biased model (length penalty on the short answer, first-position preference, penalty when the author is `other`) does not. The drifted model marks every criterion as 5 and loses kappa. Publish those three numbers together. A kappa on neutral items, with no bias probe, is not a calibration.

## Agreement

Quadratic weighted kappa uses labels `[1, 2, 3, 4, 5]` and weights `(i - j)^2 / 16`.
