"""Exact, fuzzy, and embedding scorers.

The embedding scorer defaults to a hashed lexical vector so the suite runs
offline. Pass any function `(text) -> list[float]` to swap in a model.
"""

from __future__ import annotations

import re

from evals_engineer.textutil import cosine, hashed_embedding, levenshtein, token_f1, tokenize


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    if normalize:
        return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0
    return 1.0 if prediction == reference else 0.0


def contains_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(reference) in normalize_text(prediction) else 0.0


def regex_match(prediction: str, pattern: str) -> float:
    return 1.0 if re.search(pattern, prediction or "") else 0.0


def fuzzy_match(prediction: str, reference: str) -> float:
    """Blend of token F1 and a character-edit ratio, both in [0, 1]."""
    tokens = token_f1(reference, prediction)
    left = normalize_text(reference)
    right = normalize_text(prediction)
    if not left and not right:
        edit_ratio = 1.0
    else:
        distance = levenshtein(left, right)
        edit_ratio = 1.0 - distance / max(len(left), len(right))
    return 0.5 * tokens + 0.5 * edit_ratio


def facts_covered(prediction: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    haystack = normalize_text(prediction)
    hits = sum(1 for fact in facts if normalize_text(fact) in haystack)
    return hits / len(facts)


def embedding_similarity(
    prediction: str, reference: str, embed=hashed_embedding
) -> float:
    return cosine(embed(prediction), embed(reference))


def score_prediction(case: dict, prediction: str) -> dict:
    """Score one golden case that carries an expected answer or fact list."""
    expected = case.get("expected", {})
    reference = expected.get("reference", "")
    facts = expected.get("facts", [])
    result = {
        "exact": exact_match(prediction, reference) if reference else None,
        "fuzzy": fuzzy_match(prediction, reference) if reference else None,
        "facts": facts_covered(prediction, facts) if facts else None,
        "embedding": embedding_similarity(prediction, reference) if reference else None,
    }
    if expected.get("match") == "exact":
        result["passed"] = result["exact"] == 1.0
    elif facts:
        result["passed"] = result["facts"] == 1.0
    elif reference:
        threshold = float(expected.get("fuzzy_threshold", 0.8))
        result["passed"] = (result["fuzzy"] or 0.0) >= threshold
    else:
        result["passed"] = None
    return result
