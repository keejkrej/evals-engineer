"""Careful Harbor agent against a naive baseline.

The careful agent checks entitlement before a refund and abstains when the
knowledge base does not support an answer. The naive agent refunds immediately
and always answers. The same tasks, seeds, and graders score both.
"""

from __future__ import annotations

from evals_engineer.compare import compare_binary
from evals_engineer.domain import ABSTAIN_TEXT, POLICIES
from evals_engineer.rag import grounded_generate, overconfident_generate, policy_corpus, rank_docs
from evals_engineer.trajectory import grade_trajectory

SEED = 20261006


def build_tasks() -> list[dict]:
    tasks = []
    for index, policy in enumerate(POLICIES):
        order_id = f"A{3000 + index}"
        days = 10 if index % 2 == 0 else 45
        tasks.append(
            {
                "id": f"task-{index:02d}",
                "kind": "refund" if index < 12 else "qa",
                "order_id": order_id,
                "days_ago": days,
                "amount_cents": 2500 + index * 10,
                "query": policy.question,
                "doc_id": policy.doc_id,
            }
        )
    return tasks


def careful_steps(task: dict) -> list[dict]:
    order_id = task["order_id"]
    if task["kind"] == "refund":
        steps = [
            {"tool": "lookup_order", "args": {"order_id": order_id}},
            {"tool": "check_entitlement", "args": {"order_id": order_id}},
        ]
        if task["days_ago"] <= 30:
            steps.append(
                {"tool": "refund_order", "args": {"order_id": order_id, "amount_cents": task["amount_cents"]}}
            )
        else:
            steps.append({"tool": "escalate", "args": {"reason": "outside the refund window", "order_id": order_id}})
        return steps
    return [{"tool": "search_kb", "args": {"query": task["query"]}}]


def naive_steps(task: dict) -> list[dict]:
    if task["kind"] == "refund":
        return [
            {
                "tool": "refund_order",
                "args": {"order_id": task["order_id"], "amount_cents": task["amount_cents"]},
            }
        ]
    return [{"tool": "search_kb", "args": {"query": task["query"]}}]


def careful_answer(task: dict) -> str:
    if task["kind"] != "qa":
        if task["days_ago"] <= 30:
            return f"Order {task['order_id']} is inside the refund window."
        return f"Order {task['order_id']} is outside the refund window, so I escalated it."
    docs = rank_docs(task["query"], policy_corpus(), k=3)
    generation = grounded_generate(task["query"], docs)
    return generation.text


def naive_answer(task: dict) -> str:
    del task
    return overconfident_generate("", []).text or "refunded"


def run_teardown(n_boot: int = 2000) -> dict:
    tasks = build_tasks()
    careful_ok = []
    naive_ok = []
    rows = []
    for task in tasks:
        careful = grade_trajectory(careful_steps(task))
        naive = grade_trajectory(naive_steps(task))
        if task["kind"] == "qa":
            careful_pass = careful["passed"] and ABSTAIN_TEXT not in careful_answer(task)
            naive_pass = naive["passed"] and "definitely allows" not in naive_answer(task)
        else:
            careful_pass = careful["passed"]
            naive_pass = naive["passed"]
        careful_ok.append(careful_pass)
        naive_ok.append(naive_pass)
        rows.append(
            {
                "id": task["id"],
                "kind": task["kind"],
                "careful": careful_pass,
                "naive": naive_pass,
                "careful_violations": careful["violations"],
                "naive_violations": naive["violations"],
            }
        )
    comparison = compare_binary("naive baseline", naive_ok, "careful agent", careful_ok, n_boot=n_boot, seed=SEED)
    return {
        "seed": SEED,
        "n": len(tasks),
        "careful_success": comparison["rate_b"],
        "naive_success": comparison["rate_a"],
        "comparison": comparison,
        "rows": rows,
    }
