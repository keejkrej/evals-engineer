"""Nightly production drift monitor.

A stable 5% of each night's traffic is scored offline. An alert fires after
the success rate stays more than `drop` below the baseline for `consecutive` nights.
"""

from __future__ import annotations

from evals_engineer.report import Metric, Report
from evals_engineer.textutil import should_sample


def sample_traffic(events: list[dict], rate: float = 0.05) -> list[dict]:
    return [event for event in events if should_sample(str(event["id"]), rate)]


def success_rate(events: list[dict]) -> float:
    if not events:
        return 0.0
    return sum(1.0 for event in events if event.get("success")) / len(events)


def watch_nights(
    nights: list[list[dict]],
    baseline_rate: float,
    rate: float = 0.05,
    drop: float = 0.05,
    consecutive: int = 2,
) -> dict:
    rows = []
    streak = 0
    alerts = []
    for index, night in enumerate(nights):
        sample = sample_traffic(night, rate=rate)
        current = success_rate(sample)
        decay = baseline_rate - current
        if decay > drop:
            streak += 1
        else:
            streak = 0
        alert = streak >= consecutive
        row = {
            "night": index,
            "n": len(night),
            "sampled": len(sample),
            "success_rate": current,
            "decay": decay,
            "streak": streak,
            "alert": alert,
        }
        rows.append(row)
        if alert:
            alerts.append(row)
    return {"nights": rows, "alerts": alerts, "baseline_rate": baseline_rate}


def drift_report(result: dict) -> Report:
    latest = result["nights"][-1] if result["nights"] else None
    latest_rate = 0.0 if latest is None else latest["success_rate"]
    return Report(
        title="Production drift monitor",
        passed=not result["alerts"],
        summary=(
            f"{len(result['alerts'])} alert(s) across {len(result['nights'])} nights. "
            f"Latest sampled success {latest_rate:.1%} against baseline {result['baseline_rate']:.1%}."
        ),
        metrics=[
            Metric("alerts", float(len(result["alerts"]))),
            Metric("latest_success_rate", latest_rate),
            Metric("baseline_rate", result["baseline_rate"]),
        ],
        lines=[
            "Sampling uses a stable hash of the event id, so the same traffic night draws the same 5%.",
        ],
    )
