"""Token, hash, edit-distance, and hashing-embedder helpers.

Python's builtin hash is salted per process. Dataset hashes and traffic
sampling use FNV-1a so a run can be repeated.
"""

from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "of",
        "to",
        "and",
        "or",
        "for",
        "in",
        "on",
        "within",
        "after",
        "before",
        "can",
        "i",
        "my",
        "do",
        "does",
        "how",
        "what",
        "when",
        "where",
        "who",
        "if",
        "it",
        "be",
        "this",
        "that",
        "with",
        "from",
        "at",
        "by",
        "as",
        "was",
        "were",
        "will",
        "would",
        "should",
        "could",
        "please",
        "me",
        "we",
        "you",
        "your",
        "our",
    }
)


def fnv1a(text: str) -> int:
    value = 1469598103934665603
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def tokenize(text: str, drop_stop: bool = False) -> list[str]:
    tokens = [match.group(0).lower() for match in _TOKEN.finditer(text or "")]
    if drop_stop:
        tokens = [tok for tok in tokens if tok not in STOPWORDS]
    return tokens


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.findall(text or "")]
    return [part for part in parts if part]


def token_f1(left: str, right: str) -> float:
    a = tokenize(left)
    b = tokenize(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(cb.values())
    recall = overlap / sum(ca.values())
    return 2 * precision * recall / (precision + recall)


def overlap_coefficient(left: str, right: str) -> float:
    a = set(tokenize(left, drop_stop=True))
    b = set(tokenize(right, drop_stop=True))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, ca in enumerate(left, start=1):
        current = [i]
        for j, cb in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def sequence_edit_distance(left: list, right: list) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, ca in enumerate(left, start=1):
        current = [i]
        for j, cb in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def normalized_similarity(left: list, right: list) -> float:
    if not left and not right:
        return 1.0
    denom = max(len(left), len(right))
    return 1.0 - sequence_edit_distance(left, right) / denom


def char_ngrams(text: str, n: int = 5) -> set[str]:
    compact = " ".join((text or "").lower().split())
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def containment(needle: str, haystack: str, n: int = 5) -> float:
    """Fraction of needle character n-grams that also appear in haystack."""
    left = char_ngrams(needle, n)
    if not left:
        return 0.0
    right = char_ngrams(haystack, n)
    return len(left & right) / len(left)


def hashed_embedding(text: str, dims: int = 256) -> list[float]:
    """Bag-of-tokens plus character n-grams, hashed into a unit vector.

    This is a lexical embedder. Pass a neural embedder into the contamination
    scanner when you need semantic near-duplicates that share no wording.
    """
    vector = [0.0] * dims
    tokens = tokenize(text)
    for token in tokens:
        digest = fnv1a(token)
        index = digest % dims
        sign = 1.0 if ((digest // dims) % 2) == 0 else -1.0
        vector[index] += sign
    compact = "".join(tokens)
    for start in range(max(0, len(compact) - 4)):
        digest = fnv1a(compact[start : start + 5])
        index = digest % dims
        sign = 1.0 if ((digest // dims) % 2) == 0 else -1.0
        vector[index] += 0.5 * sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("vectors must be the same non-zero length")
    return sum(a * b for a, b in zip(left, right))


def bucket(text: str, modulo: int = 10_000) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    return fnv1a(text) % modulo


def should_sample(text: str, rate: float, modulo: int = 10_000) -> bool:
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    threshold = int(round(rate * modulo))
    return bucket(text, modulo) < threshold


def rough_tokens(text: str) -> int:
    """Whitespace token count, with a floor of one for any non-empty string."""
    if not text or not text.strip():
        return 0
    return max(1, len(text.split()))
