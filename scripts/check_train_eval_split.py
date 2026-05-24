#!/usr/bin/env python3
"""Check that an eval JSONL is separated from a training trace JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_SOURCE_COUNTS = {
    "gui_grounding": 500,
    "doc_qa": 250,
    "chart_qa": 200,
    "science_diagram_qa": 50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl")
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-overlap", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(value: Any) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_key(*parts: Any) -> str:
    raw = "||".join(stable_json(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def answer_value(row: dict[str, Any]) -> Any:
    if row.get("answers"):
        return row.get("answers")
    return row.get("answer")


def source_identity(row: dict[str, Any]) -> str | None:
    task = row.get("task_type")
    meta = row.get("meta") or {}
    source = row.get("source")
    if task == "gui_grounding" and meta.get("img_filename"):
        return hash_key(source, task, meta.get("img_filename"), meta.get("bbox_xywh"), norm(row.get("question")))
    if task == "doc_qa" and (meta.get("question_id") or meta.get("questionId")):
        return hash_key(source, task, meta.get("doc_id") or meta.get("docId"), meta.get("question_id") or meta.get("questionId"))
    source_index = meta.get("source_index")
    if source_index is not None:
        return hash_key(source, task, source_index)
    return None


def semantic_identity(row: dict[str, Any]) -> str:
    return hash_key(row.get("source"), row.get("task_type"), norm(row.get("question")), answer_value(row))


def index_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_index = {}
    semantic_index = {}
    for row in rows:
        sid = source_identity(row)
        if sid:
            source_index[sid] = row
        semantic_index[semantic_identity(row)] = row
    return source_index, semantic_index


def source_range_violations(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for row in eval_rows:
        task = row.get("task_type")
        meta = row.get("meta") or {}
        source_index = meta.get("source_index")
        limit = DEFAULT_TRAIN_SOURCE_COUNTS.get(task)
        if source_index is None or limit is None:
            continue
        if int(source_index) < limit:
            violations.append({"id": row.get("id"), "task_type": task, "source_index": source_index, "train_source_limit": limit})
    return violations


def main() -> int:
    args = parse_args()
    train_rows = read_jsonl(Path(args.train))
    eval_rows = read_jsonl(Path(args.eval))
    train_source, train_semantic = index_rows(train_rows)
    eval_source, eval_semantic = index_rows(eval_rows)

    source_overlaps = sorted(set(train_source) & set(eval_source))
    semantic_overlaps = sorted(set(train_semantic) & set(eval_semantic))
    range_violations = source_range_violations(eval_rows)
    split_ok = not source_overlaps and not range_violations
    summary = {
        "train": args.train,
        "eval": args.eval,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "eval_task_counts": dict(Counter(row.get("task_type") for row in eval_rows)),
        "source_identity_overlap_count": len(source_overlaps),
        "semantic_duplicate_warning_count": len(semantic_overlaps),
        "semantic_duplicate_note": "Semantic duplicates can occur across distinct source examples, especially generic ChartQA/DocVQA questions; they are warnings, not split failures.",
        "source_range_violation_count": len(range_violations),
        "source_range_violations_sample": range_violations[:20],
        "split_ok": split_ok,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    if not summary["split_ok"] and not args.allow_overlap:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
