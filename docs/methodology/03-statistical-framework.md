# Statistical framework

A two-point gap on 50 samples is not a routing decision. This harness reports a paired difference, a 95% interval, and a p-value, and it refuses the win when the test does not clear the bar.

## Comparison sentence

For paired binary outcomes the sentence looks like this:

```
careful agent wins by 12.0% ± 4.0% (p<0.05)
```

or, when the test misses:

```
No significant difference: Model A wins by 2.0% ± 6.0% (p=1.00).
```

The pieces are:

- The winner is the model with the higher success rate. The percentage is the absolute paired risk difference, `mean(candidate) - mean(baseline)`, shown as a positive number for the winner.
- The `±` term is half the width of the 95% percentile bootstrap interval of that paired difference. Default is 2000 resamples. The seed is part of the result.
- The p-value is an exact two-sided McNemar test on the discordant pairs, not the bootstrap tail. With no discordant pairs the p-value is 1. One discordant pair cannot be significant.
- `p<0.05` is printed only when the McNemar p-value is below 0.05. Otherwise the sentence prints `p=` to two digits and a "no significant difference" prefix.

Cohen's h is reported next to the risk difference as an effect size. It is not a substitute for the interval.

## Sample size

`sample_size_two_proportions` is the usual two-sided normal approximation for two independent proportions: per-model `n` to detect a 2-point gap near the baseline rate, at alpha 0.05 and power 0.80. Use it before you collect the run, not as a p-value after the fact.

## Seed variance

At temperature 0 the scripted failure model ignores the seed. At temperature 0.7, even-numbered cases flip with the seed. `seed_variance` reports the mean and sample standard deviation of a metric across seeds. A gate that treats one warm seed as the truth will ship noise.

## Regression gate

The merge gate compares a candidate run with the baseline run on the same cases.

- Trip when task success drops by more than 2%.
- Once the success band is tripped, clear it only when the drop is at or under 1%. A 1.5% drop does not trip a healthy gate, and it does not clear a tripped one.
- Latency trips when p95 exceeds both 1.25 times the baseline and the baseline plus 50 ms. Latency does not latch the success band.
- Cost trips above the configured budget. The default budget is $25 for a run.
- A case with at least 5 seeds and a pass rate strictly between 20% and 80% is flaky. Flaky cases fail the gate by default.
- The comment posted to the pull request lists the reason and the cases that passed on the baseline and fail on the candidate.
- An eval cache keys on the hash of the dataset, the config, and the predictions. A hit returns the stored decision.

## Contamination and abstention

Do not read a win off a set that overlaps the training corpus. The contamination scanner flags an eval case when character 5-gram containment is at least 0.6 or embedding cosine is at least 0.92. The default embedder is a hashed lexical vector. Pass a neural embedder when a paraphrase with no shared wording would still be a leak.

Retrieval metrics (hit rate, MRR, nDCG, recall@k) are computed on cases that have relevant documents. Generation metrics (faithfulness, citation grounding, abstention precision and recall) are a second gate. The run passes only when both pass. They are not averaged.

## Reproducibility

| Artifact | Seed |
| --- | --- |
| Golden dataset rater rule | 20261004 |
| Teardown comparison | 20261006 |
| Bootstrap interval | the `seed` field of the comparison result |

`python -m evals_engineer.cli demo` rewrites `reports/demo/portfolio.md` from these seeds.
