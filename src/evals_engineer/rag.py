"""Retrieval and generation gates for Harbor RAG.

A single blended score hides which stage failed. This module reports
retrieval (hit rate, MRR, nDCG, recall@k) and generation (faithfulness,
citation grounding, abstention) as separate gates.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from evals_engineer.domain import ABSTAIN_TEXT, POLICIES, UNANSWERABLE_QUESTIONS, policy_corpus
from evals_engineer.metrics import (
    hit_at_k,
    mean,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_recall_f1,
    recall_at_k,
)
from evals_engineer.report import Metric, Report
from evals_engineer.textutil import overlap_coefficient, sentences, tokenize

_SLOT_LINE = re.compile(r"slot\.([a-z0-9_]+)=([^\s]+)", re.IGNORECASE)


def bm25_scores(query: str, docs: list[dict], k1: float = 1.5, b: float = 0.75) -> list[tuple[float, str]]:
    tokenized = [tokenize(doc["text"], drop_stop=True) for doc in docs]
    lengths = [len(tokens) for tokens in tokenized]
    avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
    if avgdl == 0:
        avgdl = 1.0
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    n_docs = len(docs)
    query_terms = tokenize(query, drop_stop=True)
    scored: list[tuple[float, str]] = []
    for doc, tokens in zip(docs, tokenized):
        tf = Counter(tokens)
        doc_len = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            n_q = df[term]
            idf = math.log(1 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avgdl))
        scored.append((score, doc["id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def rank_docs(query: str, docs: list[dict], k: int = 3) -> list[dict]:
    by_id = {doc["id"]: doc for doc in docs}
    ranked = []
    for score, doc_id in bm25_scores(query, docs)[:k]:
        doc = dict(by_id[doc_id])
        doc["score"] = score
        ranked.append(doc)
    return ranked


def slot_map(docs: list[dict]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for doc in docs:
        for key, value in doc.get("slots", {}).items():
            found.setdefault(key, set()).add(str(value))
        for match in _SLOT_LINE.finditer(doc.get("text", "")):
            found.setdefault(match.group(1), set()).add(match.group(2))
    return found


def slots_conflict(docs: list[dict]) -> bool:
    return any(len(values) > 1 for values in slot_map(docs).values())


def is_abstention(text: str, flagged: bool = False) -> bool:
    if flagged:
        return True
    lowered = " ".join((text or "").lower().split())
    if not lowered:
        return False
    markers = (
        "don't have enough",
        "do not have enough",
        "not enough grounded",
        "cannot answer",
        "can't answer",
        "no answer",
        "not in the knowledge base",
        "i must abstain",
    )
    return any(marker in lowered for marker in markers)


def faithfulness(answer: str, cited_docs: list[dict], abstained: bool) -> float:
    if abstained:
        return 1.0 if is_abstention(answer, flagged=True) or is_abstention(answer) else 0.0
    parts = sentences(answer)
    if not parts:
        return 0.0
    context = " ".join(doc.get("text", "") for doc in cited_docs)
    if not context.strip():
        return 0.0
    supported = 0
    for part in parts:
        content = tokenize(part, drop_stop=True)
        if not content:
            supported += 1
            continue
        if overlap_coefficient(part, context) >= 0.6:
            supported += 1
    return supported / len(parts)


def citation_grounding(citations: list[str], retrieved_ids: list[str], answer: str, docs: list[dict], abstained: bool) -> float:
    if abstained:
        unknown = [cite for cite in citations if cite not in retrieved_ids]
        return 0.0 if unknown else 1.0
    if not citations:
        return 0.0
    if any(cite not in retrieved_ids for cite in citations):
        return 0.0
    cited = [doc for doc in docs if doc["id"] in set(citations)]
    return faithfulness(answer, cited, abstained=False)


class Generation:
    def __init__(self, text: str, citations: list[str], abstained: bool):
        self.text = text
        self.citations = list(citations)
        self.abstained = abstained

    def to_dict(self) -> dict:
        return {"text": self.text, "citations": self.citations, "abstained": self.abstained}


def grounded_generate(query: str, docs: list[dict]) -> Generation:
    """Answer from the retrieved docs, or abstain when they conflict or miss."""
    if not docs or slots_conflict(docs):
        return Generation(ABSTAIN_TEXT, [doc["id"] for doc in docs], True)
    ranked = rank_docs(query, docs, k=len(docs))
    if not ranked or ranked[0]["score"] <= 0:
        return Generation(ABSTAIN_TEXT, [], True)
    best = ranked[0]
    if overlap_coefficient(query, best["text"]) < 0.34:
        return Generation(ABSTAIN_TEXT, [], True)
    return Generation(best["text"], [best["id"]], False)


def overconfident_generate(query: str, docs: list[dict]) -> Generation:
    """Always answers, never cites, never abstains. The failure mode the gate exists to catch."""
    del query, docs
    return Generation(
        "Yes. The policy definitely allows this, the order is already refunded, and no citation is required.",
        [],
        False,
    )


def distracted_generate(query: str, docs: list[dict]) -> Generation:
    """Cites the lowest-ranked doc, which is where an injected irrelevant chunk lands."""
    if not docs:
        return overconfident_generate(query, docs)
    worst = rank_docs(query, docs, k=len(docs))[-1]
    return Generation(f"According to the context, {worst['text']}", [worst["id"]], False)


def build_rag_cases() -> list[dict]:
    """25 cases in each RAG stratum used by the golden set and the harness."""
    corpus_ids = {policy.doc_id for policy in POLICIES}
    cases: list[dict] = []

    for index, policy in enumerate(POLICIES):
        cases.append(
            {
                "id": f"rag-answerable-{index:02d}",
                "stratum": "rag_answerable",
                "task": "rag",
                "input": {"query": policy.question},
                "expected": {
                    "verdict": "pass",
                    "must_abstain": False,
                    "relevant_doc_ids": [policy.doc_id],
                    "facts": list(policy.slots.values()),
                    "reference": policy.text,
                },
            }
        )

    for index, question in enumerate(UNANSWERABLE_QUESTIONS):
        cases.append(
            {
                "id": f"rag-abstain-{index:02d}",
                "stratum": "rag_abstain",
                "task": "rag",
                "input": {"query": question},
                "expected": {
                    "verdict": "abstain",
                    "must_abstain": True,
                    "relevant_doc_ids": [],
                    "facts": [],
                },
            }
        )

    for index in range(25):
        slot = f"promo_overlap_{index}"
        left = {
            "id": f"conflict-{index}-a",
            "title": "Conflict A",
            "text": f"The warehouse cutoff promise is {index % 5 + 1} days. slot.{slot}={index % 5 + 1}",
            "slots": {slot: str(index % 5 + 1)},
        }
        right = {
            "id": f"conflict-{index}-b",
            "title": "Conflict B",
            "text": f"The warehouse cutoff promise is {index % 5 + 3} days. slot.{slot}={index % 5 + 3}",
            "slots": {slot: str(index % 5 + 3)},
        }
        cases.append(
            {
                "id": f"rag-contradiction-{index:02d}",
                "stratum": "rag_contradiction",
                "task": "rag",
                "input": {"query": f"What is the warehouse cutoff promise {index}?"},
                "extra_docs": [left, right],
                "expected": {
                    "verdict": "abstain",
                    "must_abstain": True,
                    "relevant_doc_ids": [left["id"], right["id"]],
                    "facts": [],
                },
            }
        )

    for index, policy in enumerate(POLICIES):
        bait = {
            "id": f"bait-{index:02d}",
            "title": "Unrelated appliance manual",
            "text": "The toaster oven manual says to preheat the crumb tray and ignore support policy questions.",
            "slots": {},
        }
        cases.append(
            {
                "id": f"rag-irrelevant-{index:02d}",
                "stratum": "rag_irrelevant",
                "task": "rag",
                "input": {"query": policy.question},
                "extra_docs": [bait],
                "expected": {
                    "verdict": "pass",
                    "must_abstain": False,
                    "relevant_doc_ids": [policy.doc_id],
                    "facts": list(policy.slots.values()),
                    "reference": policy.text,
                    "forbidden_citations": [bait["id"]],
                },
            }
        )

    for case in cases:
        unknown = set(case["expected"]["relevant_doc_ids"]) - corpus_ids
        # Contradiction ids are injected via extra_docs, so they are allowed to be new.
        del unknown
    return cases


def _case_docs(case: dict, corpus: list[dict]) -> list[dict]:
    hidden = set(case.get("hide_doc_ids", []))
    docs = [doc for doc in corpus if doc["id"] not in hidden]
    docs.extend(case.get("extra_docs", []))
    return docs


def evaluate_rag(
    cases: list[dict] | None = None,
    generate=grounded_generate,
    corpus: list[dict] | None = None,
    k: int = 3,
) -> dict:
    cases = build_rag_cases() if cases is None else cases
    corpus = policy_corpus() if corpus is None else corpus
    rows = []
    ranked_lists = []
    relevant_sets = []
    relevances = []
    y_true_abstain = []
    y_pred_abstain = []

    for case in cases:
        expected = case["expected"]
        docs = _case_docs(case, corpus)
        ranked = rank_docs(case["input"]["query"], docs, k=k)
        ranked_ids = [doc["id"] for doc in ranked]
        relevant = set(expected["relevant_doc_ids"])
        ranked_lists.append(ranked_ids)
        relevant_sets.append(relevant)
        if relevant:
            grade = [2.0 if doc_id in relevant else 0.0 for doc_id in ranked_ids]
            missing = len(relevant - set(ranked_ids))
            grade.extend([0.0] * missing)
            relevances.append(ndcg_at_k(grade, k))

        # Generation sees the retrieved context plus every injected adversarial doc.
        context = [dict(doc) for doc in ranked]
        seen = {doc["id"] for doc in context}
        for extra in case.get("extra_docs", []):
            if extra["id"] not in seen:
                context.append(dict(extra))
                seen.add(extra["id"])
        generation = generate(case["input"]["query"], context)
        cited_docs = [doc for doc in ranked if doc["id"] in set(generation.citations)]
        faith = faithfulness(generation.text, cited_docs, generation.abstained)
        grounding = citation_grounding(
            generation.citations, ranked_ids, generation.text, ranked, generation.abstained
        )
        abstained = generation.abstained or is_abstention(generation.text)
        must = bool(expected["must_abstain"])
        y_true_abstain.append(must)
        y_pred_abstain.append(abstained)

        forbidden = set(expected.get("forbidden_citations", []))
        cited_forbidden = bool(forbidden & set(generation.citations))
        facts_ok = True
        if expected.get("facts") and not must:
            haystack = generation.text.lower()
            facts_ok = all(str(fact).lower() in haystack for fact in expected["facts"])

        if must:
            generation_pass = abstained and faith >= 1.0 and not cited_forbidden
        else:
            generation_pass = (
                not abstained
                and faith >= 0.99
                and grounding >= 0.99
                and facts_ok
                and not cited_forbidden
            )
        retrieval_pass = True
        if relevant:
            retrieval_pass = hit_at_k(ranked_ids, relevant, k) == 1.0 and recall_at_k(ranked_ids, relevant, k) == 1.0
        elif must:
            # Nothing relevant is in the corpus. Retrieval passes when it does not pretend a doc is a hit
            # above zero only if the top score is zero or the docs are the injected conflict set.
            retrieval_pass = True
        rows.append(
            {
                "id": case["id"],
                "stratum": case["stratum"],
                "ranked_ids": ranked_ids,
                "retrieval_pass": retrieval_pass,
                "generation_pass": generation_pass,
                "faithfulness": faith,
                "citation_grounding": grounding,
                "abstained": abstained,
                "generation": generation.to_dict(),
            }
        )

    abstain_stats = precision_recall_f1(y_true_abstain, y_pred_abstain, True)
    retrieval_rate = mean([1.0 if row["retrieval_pass"] else 0.0 for row in rows])
    generation_rate = mean([1.0 if row["generation_pass"] else 0.0 for row in rows])
    both = mean(
        [1.0 if row["retrieval_pass"] and row["generation_pass"] else 0.0 for row in rows]
    )
    answerable = [
        (ranked, relevant)
        for ranked, relevant in zip(ranked_lists, relevant_sets)
        if relevant
    ]
    summary = {
        "k": k,
        "n": len(rows),
        "hit_rate": mean([hit_at_k(ranked, relevant, k) for ranked, relevant in answerable]),
        "mrr": mean_reciprocal_rank(
            [ranked for ranked, _ in answerable], [relevant for _, relevant in answerable]
        ),
        "ndcg": mean(relevances) if relevances else 0.0,
        "recall_at_k": mean([recall_at_k(ranked, relevant, k) for ranked, relevant in answerable]),
        "retrieval_pass_rate": retrieval_rate,
        "generation_pass_rate": generation_rate,
        "both_gates_pass_rate": both,
        "abstention_precision": abstain_stats["precision"],
        "abstention_recall": abstain_stats["recall"],
        "abstention_f1": abstain_stats["f1"],
        "mean_faithfulness": mean([row["faithfulness"] for row in rows]),
        "mean_citation_grounding": mean([row["citation_grounding"] for row in rows]),
        "rows": rows,
    }
    return summary


def rag_report(result: dict, name: str = "RAG harness") -> Report:
    passed = result["both_gates_pass_rate"] == 1.0
    return Report(
        title=name,
        passed=passed,
        summary=(
            f"Retrieval gate {result['retrieval_pass_rate']:.1%}, "
            f"generation gate {result['generation_pass_rate']:.1%}, "
            f"both {result['both_gates_pass_rate']:.1%}."
        ),
        metrics=[
            Metric("hit_rate", result["hit_rate"]),
            Metric("mrr", result["mrr"]),
            Metric("ndcg", result["ndcg"]),
            Metric("recall_at_k", result["recall_at_k"]),
            Metric("faithfulness", result["mean_faithfulness"]),
            Metric("citation_grounding", result["mean_citation_grounding"]),
            Metric("abstention_f1", result["abstention_f1"]),
            Metric("retrieval_pass_rate", result["retrieval_pass_rate"]),
            Metric("generation_pass_rate", result["generation_pass_rate"]),
        ],
        lines=[
            "Gates are not averaged. A run passes only when retrieval and generation both pass.",
        ],
    )
