"""Cost against task success, per tenant and routing policy.

A point is on the frontier when no other point is both cheaper and better,
with at least one of those strict. System-one and system-two routes are
just policy names on the points.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from evals_engineer.report import Metric, Report


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Minimize cost_usd and maximize success_rate."""
    frontier = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            cheaper_or_eq = other["cost_usd"] <= point["cost_usd"]
            better_or_eq = other["success_rate"] >= point["success_rate"]
            strict = other["cost_usd"] < point["cost_usd"] or other["success_rate"] > point["success_rate"]
            if cheaper_or_eq and better_or_eq and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda item: (item["cost_usd"], -item["success_rate"], item["tenant"], item["policy"]))


def summarize(points: list[dict]) -> dict:
    frontier = pareto_frontier(points)
    frontier_ids = {(item["tenant"], item["policy"]) for item in frontier}
    rows = []
    for point in points:
        row = dict(point)
        row["on_frontier"] = (point["tenant"], point["policy"]) in frontier_ids
        rows.append(row)
    return {"points": rows, "frontier": frontier}


def render_dashboard(result: dict, title: str = "Cost-quality Pareto") -> str:
    points = result["points"]
    width, height = 800, 480
    left, right, top, bottom = 70, 760, 30, 430
    max_cost = max(point["cost_usd"] for point in points) * 1.15 or 1.0

    def x_of(cost: float) -> float:
        return left + (right - left) * (cost / max_cost)

    def y_of(success: float) -> float:
        return bottom - (bottom - top) * success

    circles = []
    for point in points:
        x = x_of(point["cost_usd"])
        y = y_of(point["success_rate"])
        frontier = "1" if point["on_frontier"] else "0"
        color = "#0f6f4a" if point["on_frontier"] else "#8a8f98"
        label = escape(f"{point['tenant']} {point['policy']}")
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" data-frontier="{frontier}" '
            f'data-tenant="{escape(point["tenant"])}" data-policy="{escape(point["policy"])}">'
            f"<title>{label}: cost {point['cost_usd']:.4f}, success {point['success_rate']:.1%}</title></circle>"
        )
    frontier = sorted(result["frontier"], key=lambda item: item["cost_usd"])
    if len(frontier) >= 2:
        poly = " ".join(f"{x_of(point['cost_usd']):.1f},{y_of(point['success_rate']):.1f}" for point in frontier)
        line = f'<polyline fill="none" stroke="#0f6f4a" stroke-width="2" points="{poly}" />'
    else:
        line = ""
    legend = []
    tenants = sorted({point["tenant"] for point in points})
    for index, tenant in enumerate(tenants):
        legend.append(
            f'<text x="70" y="{455 + index * 0}" fill="#222">{escape(tenant)}</text>'
        )
    # Keep the legend inside the svg by listing tenants in one line.
    legend_text = "Tenants: " + ", ".join(tenants) + ". Green points are on the frontier."
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
  <rect width="{width}" height="{height}" fill="#f7f4ee" />
  <text x="70" y="24" font-family="Georgia, serif" font-size="18" fill="#1c1a16">{escape(title)}</text>
  <line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#1c1a16" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#1c1a16" />
  <text x="{left}" y="{height - 18}" font-family="Consolas, monospace" font-size="12" fill="#1c1a16">cost (USD per task) →</text>
  <text x="16" y="240" font-family="Consolas, monospace" font-size="12" fill="#1c1a16" transform="rotate(-90 16 240)">task success →</text>
  {line}
  {''.join(circles)}
  <text x="70" y="460" font-family="Consolas, monospace" font-size="12" fill="#1c1a16">{escape(legend_text)}</text>
</svg>
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{escape(title)}</title>
  <style>
    body {{ margin: 0; background: #f7f4ee; color: #1c1a16; font-family: Georgia, serif; }}
    main {{ max-width: 860px; margin: 24px auto; padding: 0 16px; }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p>Each point is one tenant on one routing policy. Frontier points are not dominated on cost and success.</p>
    {svg}
  </main>
</body>
</html>
"""


def write_dashboard(result: dict, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(render_dashboard(result), encoding="utf-8")
    return file_path


def pareto_report(result: dict) -> Report:
    return Report(
        title="Cost-quality Pareto",
        passed=len(result["frontier"]) >= 1,
        summary=f"{len(result['frontier'])} of {len(result['points'])} routing points are on the frontier.",
        metrics=[
            Metric("points", float(len(result["points"]))),
            Metric("frontier", float(len(result["frontier"]))),
        ],
        lines=[
            "Frontier: "
            + ", ".join(f"{point['tenant']}/{point['policy']}" for point in result["frontier"])
        ],
    )


def example_points() -> list[dict]:
    """System-one is cheap. System-two spends more and succeeds more often. One dominated point is included."""
    return [
        {"tenant": "north", "policy": "system-one", "cost_usd": 0.002, "success_rate": 0.81},
        {"tenant": "north", "policy": "system-two", "cost_usd": 0.018, "success_rate": 0.93},
        {"tenant": "north", "policy": "wasted", "cost_usd": 0.030, "success_rate": 0.84},
        {"tenant": "south", "policy": "system-one", "cost_usd": 0.003, "success_rate": 0.76},
        {"tenant": "south", "policy": "system-two", "cost_usd": 0.021, "success_rate": 0.91},
        {"tenant": "east", "policy": "routed", "cost_usd": 0.009, "success_rate": 0.90},
    ]
