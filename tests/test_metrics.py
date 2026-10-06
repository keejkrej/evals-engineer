"""Known values for the statistics the rest of the harness reports."""

from evals_engineer.metrics import (
    cohens_h,
    cohens_kappa,
    hit_at_k,
    macro_f1,
    mcnemar_exact,
    mean_reciprocal_rank,
    ndcg_at_k,
    norm_ppf,
    paired_bootstrap_diff,
    precision_recall_f1,
    quadratic_weighted_kappa,
    recall_at_k,
    sample_size_two_proportions,
    seed_variance,
    wilson_interval,
)
from evals_engineer.scoring import exact_match, facts_covered, fuzzy_match
from evals_engineer.textutil import containment, cosine, hashed_embedding


def test_norm_ppf_matches_the_common_z_values():
    assert abs(norm_ppf(0.975) - 1.95996398454) < 1e-6
    assert abs(norm_ppf(0.5)) < 1e-9
    assert abs(norm_ppf(0.8) - 0.84162123357) < 1e-5


def test_precision_recall_f1_and_kappa():
    stats = precision_recall_f1(["yes", "yes", "no", "no"], ["yes", "no", "no", "yes"], "yes")
    assert stats["tp"] == 1 and stats["fp"] == 1 and stats["fn"] == 1
    assert abs(stats["precision"] - 0.5) < 1e-9
    assert abs(stats["recall"] - 0.5) < 1e-9
    assert abs(stats["f1"] - 0.5) < 1e-9
    # yes/yes, yes/no, no/no, no/no. po=0.75, pe=0.5, kappa=0.5
    kappa = cohens_kappa(["yes", "yes", "no", "no"], ["yes", "no", "no", "no"], labels=["no", "yes"])
    assert abs(kappa["kappa"] - 0.5) < 1e-9
    perfect = quadratic_weighted_kappa([1, 2, 3, 5], [1, 2, 3, 5], [1, 2, 3, 4, 5])
    assert perfect["kappa"] == 1.0


def test_macro_f1_and_wilson():
    score = macro_f1(["a", "b", "a"], ["a", "a", "a"], ["a", "b"])
    assert 0.0 < score < 1.0
    low, high = wilson_interval(80, 100)
    assert low < 0.8 < high
    assert 0.0 <= low and high <= 1.0


def test_mcnemar_known_tail():
    # 9 vs 1 discordant pairs. Two-sided exact p = 22/1024.
    baseline = [False] * 9 + [True]
    candidate = [True] * 9 + [False]
    result = mcnemar_exact(baseline, candidate)
    assert result["n_discordant"] == 10
    assert abs(result["p"] - (22 / 1024)) < 1e-12
    assert mcnemar_exact([True, False], [True, False])["p"] == 1.0


def test_bootstrap_is_seeded_and_ranking_metrics():
    first = paired_bootstrap_diff([0, 1, 1, 0], [1, 1, 1, 0], n_boot=200, seed=3)
    second = paired_bootstrap_diff([0, 1, 1, 0], [1, 1, 1, 0], n_boot=200, seed=3)
    assert first == second
    assert ndcg_at_k([3, 2, 1]) == 1.0
    assert ndcg_at_k([1, 2, 3]) < 1.0
    assert recall_at_k(["d3", "d1", "d4"], {"d1", "d2"}, 2) == 0.5
    assert hit_at_k(["d3", "d1"], {"d1"}, 2) == 1.0
    assert mean_reciprocal_rank([["d3", "d1"], ["d2"]], [{"d1"}, {"d2"}]) == (0.5 + 1.0) / 2


def test_effect_size_sample_size_and_seed_spread():
    assert abs(cohens_h(0.5, 0.5)) < 1e-12
    n = sample_size_two_proportions(0.90, 0.88)
    assert n > 1000
    spread = seed_variance(lambda seed: float(seed), [1, 2, 3])
    assert spread["low"] == 1 and spread["high"] == 3


def test_scorers_and_containment():
    assert exact_match("The Refund Window.", "the refund window") == 1.0
    assert facts_covered("Refunds take 30 days.", ["30 days"]) == 1.0
    assert fuzzy_match("refund window is 30 days", "the refund window is 30 days") > 0.7
    leaked = "An order can be refunded within 30 days of the delivery date."
    assert containment(leaked, leaked + " Extra.") > 0.9
    assert containment(leaked, "toaster crumb trays") < 0.2
    assert abs(cosine(hashed_embedding(leaked), hashed_embedding(leaked)) - 1.0) < 1e-9
