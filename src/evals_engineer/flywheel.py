"""Thumbs-down to preference-pair to golden-set flywheel.

The trainer boundary writes a LoRA job manifest and validates the pairs.
The bundled trainer is a dry run: it does not load weights or start a GPU job.
"""

from __future__ import annotations

import json
from pathlib import Path

from evals_engineer.report import Metric, Report


def preference_pairs(events: list[dict]) -> list[dict]:
    pairs = []
    skipped = []
    for event in events:
        if event.get("rating") != "down":
            continue
        chosen = (event.get("correction") or "").strip()
        rejected = (event.get("response") or "").strip()
        prompt = (event.get("prompt") or "").strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            skipped.append(event.get("event_id"))
            continue
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source_event_id": event["event_id"],
                "reason": "thumbs_down",
            }
        )
    return pairs


def cases_from_pairs(pairs: list[dict]) -> list[dict]:
    cases = []
    for pair in pairs:
        cases.append(
            {
                "id": f"flywheel-{pair['source_event_id']}",
                "stratum": "flywheel",
                "task": "score",
                "input": {"query": pair["prompt"]},
                "expected": {
                    "verdict": "pass",
                    "match": "facts",
                    "reference": pair["chosen"],
                    "facts": [],
                    "rejected": pair["rejected"],
                },
                "provenance": {"event_id": pair["source_event_id"], "reason": pair["reason"]},
            }
        )
    return cases


class DryRunTrainer:
    """Validate pairs and write a manifest. No weights are updated."""

    def __init__(self, output_dir: str | Path, base_model: str = "harbor-support-base", rank: int = 16):
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.rank = rank

    def submit(self, pairs: list[dict], dataset_path: str) -> dict:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        for pair in pairs:
            if pair["chosen"] == pair["rejected"] or not pair["chosen"] or not pair["rejected"]:
                raise ValueError(f"invalid preference pair {pair.get('source_event_id')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "job": "lora",
            "status": "dry_run",
            "base_model": self.base_model,
            "rank": self.rank,
            "n_pairs": len(pairs),
            "dataset": dataset_path,
            "note": "No weights were trained. Submit this manifest to a trainer to run the job.",
        }
        path = self.output_dir / "lora_job.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["manifest_path"] = str(path)
        return manifest


def nightly(
    events: list[dict],
    golden: list[dict],
    eval_fn,
    trainer: DryRunTrainer,
    dataset_path: str,
) -> dict:
    """eval_fn(cases) -> {"success_rate": float}. Called before and after new cases are appended."""
    pairs = preference_pairs(events)
    new_cases = cases_from_pairs(pairs)
    before = eval_fn(golden)
    after_cases = list(golden) + new_cases
    after = eval_fn(after_cases)
    job = trainer.submit(pairs, dataset_path) if pairs else {
        "job": "lora",
        "status": "skipped",
        "n_pairs": 0,
        "note": "No valid thumbs-down pairs.",
    }
    return {
        "pairs": pairs,
        "new_cases": new_cases,
        "before": before,
        "after": after,
        "success_delta": after["success_rate"] - before["success_rate"],
        "job": job,
    }


def flywheel_report(result: dict) -> Report:
    status = result["job"]["status"]
    return Report(
        title="DPO flywheel",
        passed=status in {"dry_run", "skipped"},
        summary=(
            f"{len(result['pairs'])} preference pairs. "
            f"Eval success {result['before']['success_rate']:.1%} -> {result['after']['success_rate']:.1%}. "
            f"Trainer status: {status}."
        ),
        metrics=[
            Metric("pairs", float(len(result["pairs"]))),
            Metric("new_cases", float(len(result["new_cases"]))),
            Metric("success_delta", result["success_delta"]),
        ],
        lines=[result["job"].get("note", "")],
    )
