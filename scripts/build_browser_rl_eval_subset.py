#!/usr/bin/env python3
"""Build family-balanced browser-RL eval subsets.

The input is a task JSONL file. The output keeps whole task specs, sampled by
family and balanced across templates inside each family when possible.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--families", default="", help="Comma-separated family allowlist; empty means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    allow = {item.strip() for item in args.families.split(",") if item.strip()}
    tasks = read_jsonl(Path(args.input))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        family = family_of(task)
        if allow and family not in allow:
            continue
        by_family[family].append(task)

    selected: list[dict[str, Any]] = []
    family_summary: dict[str, Any] = {}
    for family in sorted(by_family):
        picked = pick_template_balanced(by_family[family], per_family=args.per_family, rng=rng)
        selected.extend(picked)
        family_summary[family] = {
            "available": len(by_family[family]),
            "selected": len(picked),
            "templates": dict(Counter(str(task.get("template") or "unknown") for task in picked)),
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for task in selected:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    summary = {
        "input": args.input,
        "output": args.output,
        "per_family": args.per_family,
        "seed": args.seed,
        "families": family_summary,
        "total": len(selected),
    }
    (out.with_suffix(out.suffix + ".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def family_of(task: dict[str, Any]) -> str:
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    return str(metadata.get("family") or task.get("template") or "unknown")


def pick_template_balanced(tasks: list[dict[str, Any]], *, per_family: int, rng: random.Random) -> list[dict[str, Any]]:
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_template[str(task.get("template") or "unknown")].append(task)
    queues: dict[str, list[dict[str, Any]]] = {}
    for template, rows in by_template.items():
        rows = list(rows)
        rng.shuffle(rows)
        queues[template] = rows
    picked: list[dict[str, Any]] = []
    templates = sorted(queues)
    while len(picked) < per_family and any(queues.values()):
        for template in templates:
            if len(picked) >= per_family:
                break
            if queues[template]:
                picked.append(queues[template].pop())
    return sorted(picked, key=lambda item: str(item.get("task_id") or ""))


if __name__ == "__main__":
    main()
