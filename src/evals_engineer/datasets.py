"""JSONL datasets, stratified sampling, and content-addressed versions."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from evals_engineer.metrics import cohens_kappa


def canonical_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    file_path = Path(path)
    with file_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{file_path}:{line_no}: each row must be an object")
            rows.append(item)
    return rows


def dump_jsonl(path: str | Path, rows: list[dict]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(canonical_json(row) + "\n" for row in rows)
    file_path.write_text(body, encoding="utf-8")


def content_hash(rows: list[dict]) -> str:
    payload = "\n".join(canonical_json(row) for row in rows)
    return sha256(payload.encode("utf-8")).hexdigest()


def stratified_sample(
    rows: list[dict],
    n: int,
    key: str = "stratum",
    seed: int = 0,
    allocation: str = "equal",
) -> list[dict]:
    """Draw `n` rows stratified by `key`.

    `equal` gives every stratum the same count (leftover rows go to the
    largest strata). `proportional` matches the source mix.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or not rows:
        return []
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    names = sorted(groups)
    rng = random.Random(seed)
    for name in names:
        rng.shuffle(groups[name])
    if n >= len(rows):
        chosen: list[dict] = []
        for name in names:
            chosen.extend(groups[name])
        return chosen

    if allocation == "equal":
        base, remainder = divmod(n, len(names))
        quotas = {name: base for name in names}
        # Largest groups absorb the remainder so small strata are not overdrawn
        # past their size more often than large ones.
        by_size = sorted(names, key=lambda name: (-len(groups[name]), name))
        for name in by_size[:remainder]:
            quotas[name] += 1
    elif allocation == "proportional":
        raw = {name: n * len(groups[name]) / len(rows) for name in names}
        quotas = {name: int(raw[name]) for name in names}
        leftover = n - sum(quotas.values())
        fractional = sorted(names, key=lambda name: (-(raw[name] - quotas[name]), name))
        for name in fractional[:leftover]:
            quotas[name] += 1
    else:
        raise ValueError("allocation must be 'equal' or 'proportional'")

    chosen = []
    shortfall = 0
    for name in names:
        take = min(quotas[name], len(groups[name]))
        shortfall += quotas[name] - take
        chosen.extend(groups[name][:take])
        groups[name] = groups[name][take:]
    if shortfall:
        pool = [row for name in names for row in groups[name]]
        rng.shuffle(pool)
        chosen.extend(pool[:shortfall])
    return chosen[:n]


def annotator_verdicts(rows: list[dict], left: str = "ada", right: str = "ben") -> dict:
    y_left = [row["annotators"][left]["verdict"] for row in rows]
    y_right = [row["annotators"][right]["verdict"] for row in rows]
    labels = ["abstain", "fail", "pass"]
    stats = cohens_kappa(y_left, y_right, labels=labels)
    disagreed = [
        row["id"]
        for row in rows
        if row["annotators"][left]["verdict"] != row["annotators"][right]["verdict"]
    ]
    return {
        "kappa": stats["kappa"],
        "po": stats["po"],
        "pe": stats["pe"],
        "n": stats["n"],
        "disagreements": disagreed,
        "raters": [left, right],
    }


def write_versioned_dataset(
    directory: str | Path,
    rows: list[dict],
    version: str,
    changelog_entry: str,
    seed: int,
) -> dict:
    """Write cases, a content hash, and a changelog entry."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    dump_jsonl(folder / "cases.jsonl", rows)
    digest = content_hash(rows)
    agreement = annotator_verdicts(rows) if rows and "annotators" in rows[0] else None
    manifest = {
        "version": version,
        "sha256": digest,
        "n": len(rows),
        "seed": seed,
        "kappa": None if agreement is None else agreement["kappa"],
    }
    (folder / "VERSION.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    changelog = folder / "CHANGELOG.md"
    block = changelog_entry.strip() + "\n"
    if changelog.exists():
        existing = changelog.read_text(encoding="utf-8")
        if block.strip() not in existing:
            changelog.write_text(existing.rstrip() + "\n\n" + block, encoding="utf-8")
    else:
        changelog.write_text("# Dataset changelog\n\n" + block, encoding="utf-8")
    manifest["agreement"] = agreement
    return manifest
