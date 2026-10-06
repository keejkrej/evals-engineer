"""Shared report objects. Systems return these instead of printing ad hoc text."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metric:
    name: str
    value: float
    detail: str = ""

    def to_dict(self) -> dict:
        payload = {"name": self.name, "value": self.value}
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class Report:
    title: str
    passed: bool
    summary: str
    metrics: list[Metric] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "passed": self.passed,
            "summary": self.summary,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "lines": list(self.lines),
        }

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        rows = [f"## {self.title}", "", f"Status: **{status}**. {self.summary}", ""]
        if self.metrics:
            rows.append("| Metric | Value | Detail |")
            rows.append("| --- | ---: | --- |")
            for metric in self.metrics:
                detail = metric.detail.replace("|", "\\|")
                rows.append(f"| {metric.name} | {metric.value:.4g} | {detail} |")
            rows.append("")
        rows.extend(self.lines)
        return "\n".join(rows).rstrip() + "\n"
