"""N-gram and embedding overlap between an eval set and a training corpus.

The default embedder is lexical. Pass a neural embedder to catch paraphrases
that share no word pieces. A finding does not prove a document was in the
training mix. It is a leak candidate for a person to review.
"""

from __future__ import annotations

from evals_engineer.report import Metric, Report
from evals_engineer.textutil import containment, cosine, hashed_embedding


def case_text(case: dict) -> str:
    parts = [case.get("id", "")]
    parts.append(_flatten(case.get("input", "")))
    parts.append(_flatten(case.get("expected", "")))
    return "\n".join(part for part in parts if part)


def _flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def scan(
    eval_cases: list[dict],
    train_docs: list[dict],
    n: int = 5,
    containment_threshold: float = 0.6,
    cosine_threshold: float = 0.92,
    embed=hashed_embedding,
) -> dict:
    findings = []
    for case in eval_cases:
        text = case_text(case)
        left = embed(text)
        best = {"containment": 0.0, "cosine": 0.0, "train_id": None}
        for doc in train_docs:
            doc_text = doc.get("text", "")
            contained = containment(text, doc_text, n=n)
            similarity = cosine(left, embed(doc_text))
            if contained > best["containment"] or (
                contained == best["containment"] and similarity > best["cosine"]
            ):
                best = {"containment": contained, "cosine": similarity, "train_id": doc.get("id")}
        leaked = best["containment"] >= containment_threshold or best["cosine"] >= cosine_threshold
        findings.append(
            {
                "eval_id": case.get("id"),
                "train_id": best["train_id"],
                "containment": best["containment"],
                "cosine": best["cosine"],
                "leaked": leaked,
            }
        )
    leaked = [row for row in findings if row["leaked"]]
    return {
        "n_eval": len(eval_cases),
        "n_train": len(train_docs),
        "n_leaked": len(leaked),
        "leak_rate": (len(leaked) / len(eval_cases)) if eval_cases else 0.0,
        "findings": findings,
        "thresholds": {"containment": containment_threshold, "cosine": cosine_threshold, "n": n},
    }


def contamination_report(result: dict) -> Report:
    return Report(
        title="Dataset contamination",
        passed=True,
        summary=f"{result['n_leaked']} of {result['n_eval']} eval cases overlap the training corpus.",
        metrics=[
            Metric("leak_rate", result["leak_rate"]),
            Metric("n_leaked", float(result["n_leaked"])),
        ],
        lines=[
            "Containment is the fraction of eval character n-grams found in a training doc.",
            "Review every leaked id before publishing a score.",
        ],
    )
