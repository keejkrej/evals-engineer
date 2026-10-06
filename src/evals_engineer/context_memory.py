"""Context-window eviction test.

The needle is a high-salience fact. Noise arrives afterwards and would fill
the budget. A salience policy keeps the needle. A FIFO policy evicts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evals_engineer.report import Metric, Report
from evals_engineer.textutil import rough_tokens, tokenize


@dataclass
class MemoryItem:
    item_id: str
    text: str
    salience: float
    kind: str
    tokens: int = 0

    def __post_init__(self) -> None:
        if self.tokens <= 0:
            self.tokens = max(1, rough_tokens(self.text))


@dataclass
class WorkingMemory:
    budget: int
    policy: object
    items: list[MemoryItem] = field(default_factory=list)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        self.policy.evict(self)

    def used(self) -> int:
        return sum(item.tokens for item in self.items)

    def ids(self) -> list[str]:
        return [item.item_id for item in self.items]

    def retrieve(self, query: str) -> str | None:
        query_terms = set(tokenize(query, drop_stop=True))
        best_id = None
        best_score = 0
        for item in self.items:
            score = len(query_terms & set(tokenize(item.text, drop_stop=True)))
            if score > best_score:
                best_score = score
                best_id = item.item_id
        return best_id


class SaliencePolicy:
    def evict(self, memory: WorkingMemory) -> None:
        while memory.used() > memory.budget and memory.items:
            # Lowest salience leaves first. Ties drop noise before state, then older items.
            victim = min(
                range(len(memory.items)),
                key=lambda index: (
                    memory.items[index].salience,
                    0 if memory.items[index].kind == "noise" else 1,
                    index,
                ),
            )
            memory.items.pop(victim)


class FifoPolicy:
    def evict(self, memory: WorkingMemory) -> None:
        while memory.used() > memory.budget and memory.items:
            memory.items.pop(0)


def run_eviction(policy, budget: int = 40, noise_items: int = 30) -> dict:
    memory = WorkingMemory(budget=budget, policy=policy)
    needle = MemoryItem(
        item_id="needle-refund-window",
        text="The refund window is 30 days from the delivery date.",
        salience=0.95,
        kind="state",
    )
    memory.add(needle)
    for index in range(noise_items):
        memory.add(
            MemoryItem(
                item_id=f"noise-{index:02d}",
                text=f"Unrelated catalog note {index} about toaster crumb trays and packing tape.",
                salience=0.05,
                kind="noise",
            )
        )
    retrieved = memory.retrieve("What is the refund window?")
    return {
        "policy": type(policy).__name__,
        "budget": budget,
        "used": memory.used(),
        "n_items": len(memory.items),
        "needle_retained": needle.item_id in memory.ids(),
        "retrieved": retrieved,
        "noise_remaining": sum(1 for item in memory.items if item.kind == "noise"),
    }


def context_report(salience: dict, fifo: dict) -> Report:
    passed = salience["needle_retained"] and salience["retrieved"] == "needle-refund-window" and not fifo["needle_retained"]
    return Report(
        title="Context eviction",
        passed=passed,
        summary=(
            f"Salience kept the needle ({salience['needle_retained']}). "
            f"FIFO kept the needle ({fifo['needle_retained']})."
        ),
        metrics=[
            Metric("salience_needle_retained", 1.0 if salience["needle_retained"] else 0.0),
            Metric("fifo_needle_retained", 1.0 if fifo["needle_retained"] else 0.0),
            Metric("salience_noise_remaining", float(salience["noise_remaining"])),
        ],
        lines=[
            "The harness fills the budget with low-salience notes after the refund-window fact.",
        ],
    )
