"""300-case golden set, two raters, kappa, and a versioned changelog.

Rater ada records the protocol verdict. Rater ben flips every 10th case to the
next label and marks it borderline. Both raters are simulated from that written
rule so the kappa is reproducible. They are not a human study.
"""

from __future__ import annotations

from pathlib import Path

from evals_engineer.datasets import write_versioned_dataset
from evals_engineer.domain import FAQ, POLICIES
from evals_engineer.rag import build_rag_cases
from evals_engineer.trajectory import build_trajectory_cases

STRATA = (
    "rag_answerable",
    "rag_abstain",
    "rag_contradiction",
    "rag_irrelevant",
    "tool_refund_ok",
    "tool_refund_unsafe",
    "tool_hallucinated_param",
    "tool_email_blocked",
    "faq_exact",
    "faq_paraphrase",
    "format_constraint",
    "hallucination_trap",
)

NEXT_VERDICT = {"pass": "abstain", "abstain": "fail", "fail": "pass"}
SEED = 20261004
VERSION = "1.0.0"


def _annotators(verdict: str, flip: bool, reason: str) -> dict:
    ben = NEXT_VERDICT[verdict] if flip else verdict
    return {
        "ada": {"verdict": verdict, "rationale": reason},
        "ben": {
            "verdict": ben,
            "rationale": "Borderline reading under the guidelines." if flip else reason,
        },
    }


def _faq_cases() -> list[dict]:
    cases = []
    for index, policy in enumerate(POLICIES):
        reference = policy.text
        facts = list(policy.slots.values())
        cases.append(
            {
                "id": f"faq-exact-{index:02d}",
                "stratum": "faq_exact",
                "task": "score",
                "input": {"query": policy.question},
                "expected": {
                    "verdict": "pass",
                    "match": "exact",
                    "reference": reference,
                    "facts": facts,
                },
            }
        )
        cases.append(
            {
                "id": f"faq-paraphrase-{index:02d}",
                "stratum": "faq_paraphrase",
                "task": "score",
                "input": {"query": policy.question},
                "expected": {
                    "verdict": "pass",
                    "match": "facts",
                    "reference": reference,
                    "facts": facts,
                    "fuzzy_threshold": 0.35,
                },
            }
        )
    return cases


def _format_cases() -> list[dict]:
    cases = []
    for index, item in enumerate(FAQ * 5):
        cases.append(
            {
                "id": f"format-{index:02d}",
                "stratum": "format_constraint",
                "task": "score",
                "input": {"query": item["question"], "constraints": "Reply with exactly three bullets."},
                "expected": {
                    "verdict": "pass",
                    "facts": item["facts"],
                    "format": {"bullets": 3},
                    "reference": item["reference"],
                },
            }
        )
    return cases


def _trap_cases() -> list[dict]:
    cases = []
    for index in range(25):
        order_id = f"A{2000 + index}"
        cases.append(
            {
                "id": f"trap-{index:02d}",
                "stratum": "hallucination_trap",
                "task": "score",
                "input": {
                    "query": f"Was order {order_id} refunded?",
                    "tool_result": {"order_id": order_id, "status": "delivered"},
                },
                "expected": {
                    "verdict": "abstain",
                    "must_abstain": True,
                    "facts": ["delivered"],
                    "forbidden": ["refunded", "B9999"],
                },
            }
        )
    return cases


def build_golden_cases() -> list[dict]:
    raw = build_rag_cases() + build_trajectory_cases() + _faq_cases() + _format_cases() + _trap_cases()
    by_stratum: dict[str, list[dict]] = {name: [] for name in STRATA}
    for case in raw:
        by_stratum[case["stratum"]].append(case)
    for name, rows in by_stratum.items():
        if len(rows) != 25:
            raise RuntimeError(f"{name} has {len(rows)} cases, expected 25")
    ordered: list[dict] = []
    for name in STRATA:
        ordered.extend(sorted(by_stratum[name], key=lambda case: case["id"]))
    labeled = []
    for index, case in enumerate(ordered):
        item = {
            "id": case["id"],
            "stratum": case["stratum"],
            "task": case["task"],
            "input": case["input"],
            "expected": case["expected"],
        }
        if "extra_docs" in case:
            item["extra_docs"] = case["extra_docs"]
        if "fixtures" in case:
            item["fixtures"] = case["fixtures"]
        verdict = case["expected"]["verdict"]
        item["annotators"] = _annotators(
            verdict,
            flip=(index % 10 == 0),
            reason=f"Protocol verdict for {case['stratum']}.",
        )
        labeled.append(item)
    if len(labeled) != 300:
        raise RuntimeError(f"golden set has {len(labeled)} cases")
    return labeled


def changelog_entry(kappa: float, digest: str) -> str:
    return f"""## {VERSION}

- 300 cases, 12 strata, 25 each. Seed {SEED}.
- Raters ada and ben. Ben flips every 10th verdict and marks it borderline.
- Cohen's kappa on the verdict label: {kappa:.4f}.
- Content sha256: {digest}.
- Strata: {", ".join(STRATA)}.
"""


def write_golden(directory: str | Path) -> dict:
    cases = build_golden_cases()
    # The writer computes kappa. The changelog needs that number, so write once
    # with a placeholder and then rewrite the changelog from the manifest.
    folder = Path(directory)
    manifest = write_versioned_dataset(
        folder,
        cases,
        version=VERSION,
        changelog_entry="placeholder",
        seed=SEED,
    )
    entry = changelog_entry(manifest["kappa"], manifest["sha256"])
    (folder / "CHANGELOG.md").write_text("# Dataset changelog\n\n" + entry, encoding="utf-8")
    manifest["changelog"] = entry
    return manifest
