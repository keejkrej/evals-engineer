"""Statistics for eval scores.

Binary and multiclass agreement, Wilson intervals, percentile bootstrap
intervals, exact McNemar tests, effect size, and ranking metrics. Given a
seed, every resampling result is repeatable.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter
from collections.abc import Callable, Sequence


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Peter Acklam's approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sample")
    return sum(values) / len(values)


def sample_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def precision_recall_f1(y_true: Sequence, y_pred: Sequence, positive) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = fn = tn = 0
    for truth, pred in zip(y_true, y_pred):
        if pred == positive and truth == positive:
            tp += 1
        elif pred == positive and truth != positive:
            fp += 1
        elif pred != positive and truth == positive:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence) -> float:
    if not labels:
        raise ValueError("labels are required")
    scores = [precision_recall_f1(y_true, y_pred, label)["f1"] for label in labels]
    return mean(scores)


def confusion_table(
    y_a: Sequence, y_b: Sequence, labels: Sequence | None = None
) -> tuple[list, list[list[int]]]:
    if len(y_a) != len(y_b):
        raise ValueError("ratings must be aligned")
    if labels is None:
        labels = sorted(set(y_a) | set(y_b), key=str)
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    table = [[0 for _ in labels] for _ in labels]
    for left, right in zip(y_a, y_b):
        table[index[left]][index[right]] += 1
    return labels, table


def cohens_kappa(
    y_a: Sequence, y_b: Sequence, labels: Sequence | None = None
) -> dict:
    """Unweighted Cohen's kappa for two raters."""
    if not y_a:
        raise ValueError("ratings must be non-empty")
    labels, table = confusion_table(y_a, y_b, labels)
    n = sum(sum(row) for row in table)
    observed = sum(table[i][i] for i in range(len(labels))) / n
    row_mass = [sum(row) / n for row in table]
    col_mass = [
        sum(table[i][j] for i in range(len(labels))) / n for j in range(len(labels))
    ]
    expected = sum(row * col for row, col in zip(row_mass, col_mass))
    kappa = 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "kappa": kappa,
        "po": observed,
        "pe": expected,
        "n": n,
        "labels": labels,
    }


def quadratic_weighted_kappa(
    y_a: Sequence, y_b: Sequence, labels: Sequence
) -> dict:
    """Quadratic weighted kappa. `labels` is the ordered category scale."""
    if len(labels) < 2:
        raise ValueError("need at least two ordered labels")
    labels, table = confusion_table(y_a, y_b, labels)
    n_cat = len(labels)
    n = sum(sum(row) for row in table)
    denom = (n_cat - 1) ** 2
    row_mass = [sum(row) for row in table]
    col_mass = [sum(table[i][j] for i in range(n_cat)) for j in range(n_cat)]
    observed = 0.0
    expected = 0.0
    for i in range(n_cat):
        for j in range(n_cat):
            weight = ((i - j) ** 2) / denom
            observed += weight * table[i][j] / n
            expected += weight * (row_mass[i] * col_mass[j]) / (n * n)
    kappa = 1.0 if expected == 0.0 else 1.0 - observed / expected
    return {"kappa": kappa, "n": n, "labels": list(labels)}


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = norm_ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_stat(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    if not values:
        raise ValueError("empty sample")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(float(stat(sample)))
    stats.sort()
    alpha = 1 - confidence
    low_i = int(math.floor((alpha / 2) * (n_boot - 1)))
    high_i = int(math.ceil((1 - alpha / 2) * (n_boot - 1)))
    high_i = min(high_i, n_boot - 1)
    return {
        "point": float(stat(values)),
        "low": stats[low_i],
        "high": stats[high_i],
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
    }


def paired_bootstrap_diff(
    baseline: Sequence[float],
    candidate: Sequence[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile interval for mean(candidate) - mean(baseline), paired by index."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples must be aligned and non-empty")
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    rng = random.Random(seed)
    n = len(baseline)
    diffs = []
    for _ in range(n_boot):
        indexes = [rng.randrange(n) for _ in range(n)]
        base = sum(baseline[i] for i in indexes) / n
        cand = sum(candidate[i] for i in indexes) / n
        diffs.append(cand - base)
    diffs.sort()
    alpha = 1 - confidence
    low_i = int(math.floor((alpha / 2) * (n_boot - 1)))
    high_i = min(n_boot - 1, int(math.ceil((1 - alpha / 2) * (n_boot - 1))))
    point = mean(candidate) - mean(baseline)
    if point >= 0:
        one_sided = sum(1 for diff in diffs if diff <= 0) / n_boot
    else:
        one_sided = sum(1 for diff in diffs if diff >= 0) / n_boot
    return {
        "point": point,
        "low": diffs[low_i],
        "high": diffs[high_i],
        "p_bootstrap": min(1.0, 2 * one_sided),
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
    }


def mcnemar_exact(baseline_correct: Sequence[bool], candidate_correct: Sequence[bool]) -> dict:
    """Two-sided exact McNemar test on paired binary outcomes.

    The null is that the two models disagree in either direction equally often.
    With no discordant pairs the p-value is 1.
    """
    if len(baseline_correct) != len(candidate_correct):
        raise ValueError("paired outcomes must be aligned")
    candidate_only = 0
    baseline_only = 0
    for base, cand in zip(baseline_correct, candidate_correct):
        if cand and not base:
            candidate_only += 1
        elif base and not cand:
            baseline_only += 1
    n = candidate_only + baseline_only
    if n == 0:
        return {"candidate_only": candidate_only, "baseline_only": baseline_only, "p": 1.0, "n_discordant": 0}
    gap = abs(candidate_only - baseline_only)
    if n > 2000:
        # Continuity-corrected chi-square tail, chi-square with 1 df.
        stat = (abs(candidate_only - baseline_only) - 1) ** 2 / n
        p_value = math.erfc(math.sqrt(stat / 2))
    else:
        total = 0
        for i in range(n + 1):
            if abs(2 * i - n) >= gap:
                total += math.comb(n, i)
        p_value = total / (2**n)
    return {
        "candidate_only": candidate_only,
        "baseline_only": baseline_only,
        "p": min(1.0, float(p_value)),
        "n_discordant": n,
    }


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two independent proportions."""

    def _phi(p: float) -> float:
        if not 0.0 <= p <= 1.0:
            raise ValueError("proportions must be in [0, 1]")
        return 2 * math.asin(math.sqrt(p))

    return _phi(p1) - _phi(p2)


def sample_size_two_proportions(
    p1: float, p2: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Per-group sample size for a two-sided test of two independent proportions."""
    if p1 == p2:
        raise ValueError("proportions must differ")
    for p in (p1, p2):
        if not 0.0 < p < 1.0:
            raise ValueError("proportions must be in (0, 1)")
    z_alpha = norm_ppf(1 - alpha / 2)
    z_beta = norm_ppf(power)
    diff = p1 - p2
    variance = p1 * (1 - p1) + p2 * (1 - p2)
    n = (z_alpha + z_beta) ** 2 * variance / (diff * diff)
    return math.ceil(n)


def dcg_at_k(relevances: Sequence[float], k: int | None = None) -> float:
    rows = list(relevances if k is None else relevances[:k])
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(rows))


def ndcg_at_k(relevances: Sequence[float], k: int | None = None) -> float:
    if not relevances:
        return 0.0
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal


def reciprocal_rank(ranked_ids: Sequence, relevant: set) -> float:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(ranked_lists: Sequence[Sequence], relevant_sets: Sequence[set]) -> float:
    if len(ranked_lists) != len(relevant_sets):
        raise ValueError("ranked lists and relevance sets must be aligned")
    if not ranked_lists:
        return 0.0
    return mean(
        [reciprocal_rank(ranked, relevant) for ranked, relevant in zip(ranked_lists, relevant_sets)]
    )


def recall_at_k(ranked_ids: Sequence, relevant: set, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in list(ranked_ids)[:k] if doc_id in relevant)
    return found / len(relevant)


def hit_at_k(ranked_ids: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(doc_id in relevant for doc_id in list(ranked_ids)[:k]) else 0.0


def seed_variance(score_at_seed: Callable[[int], float], seeds: Sequence[int]) -> dict:
    if len(seeds) < 2:
        raise ValueError("need at least two seeds")
    values = [float(score_at_seed(seed)) for seed in seeds]
    return {
        "seeds": list(seeds),
        "values": values,
        "mean": mean(values),
        "stdev": sample_stdev(values),
        "low": min(values),
        "high": max(values),
    }


def counts(labels: Sequence) -> dict:
    return dict(sorted(Counter(labels).items(), key=lambda item: str(item[0])))
