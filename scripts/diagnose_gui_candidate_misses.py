#!/usr/bin/env python3
"""Diagnose GUI candidate coverage misses by split and target type."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Candidate JSONL, e.g. outputs/.../val.jsonl.")
    parser.add_argument(
        "--source-data",
        default="/root/models/datasets/os_atlas_linux_2k/source_os_atlas_linux_2k.jsonl",
        help="Optional source JSONL used to recover OS-Atlas meta fields removed by candidate builders.",
    )
    parser.add_argument("--output", required=True, help="Markdown report path.")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path.")
    parser.add_argument("--miss-jsonl", default=None, help="Optional JSONL with missed/low-IoU rows.")
    parser.add_argument("--low-iou-threshold", type=float, default=0.3)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def attach_source_meta(rows: list[dict[str, Any]], source_path: str | Path | None) -> list[dict[str, Any]]:
    if not source_path:
        return rows
    path = Path(source_path)
    if not path.exists():
        return rows
    source_by_id = {str(row.get("id")): row for row in read_jsonl(path)}
    enriched = []
    for row in rows:
        source = source_by_id.get(str(row.get("id"))) or {}
        if source and not row.get("meta"):
            row = {**row, "meta": source.get("meta") or {}}
        enriched.append(row)
    return enriched


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def data_type(row: dict[str, Any]) -> str:
    return str((row.get("meta") or {}).get("data_type") or "unknown")


def source_of_oracle(row: dict[str, Any]) -> str:
    oracle_id = row.get("oracle_candidate_id")
    for candidate in row.get("candidates") or []:
        if candidate.get("candidate_id") == oracle_id:
            return "+".join(candidate.get("sources") or [candidate.get("source") or "unknown"])
    return "none"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("oracle_metrics") or {} for row in rows]
    return {
        "count": len(rows),
        "avg_candidates": mean([float(row.get("candidate_count") or len(row.get("candidates") or [])) for row in rows]),
        "oracle_hit_rate": mean([float(bool(metric.get("hit"))) for metric in metrics]),
        "oracle_pointing_rate": mean([float(bool(metric.get("pointing"))) for metric in metrics]),
        "oracle_iou50_rate": mean([float(bool(metric.get("iou_50"))) for metric in metrics]),
        "avg_oracle_iou": mean([float(metric.get("iou") or 0.0) for metric in metrics]),
    }


def type_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[data_type(row)].append(row)
    items = []
    for dtype, dtype_rows in by_type.items():
        item = summarize(dtype_rows)
        item["data_type"] = dtype
        items.append(item)
    return sorted(items, key=lambda item: (-int(item["count"]), str(item["data_type"])))


def write_miss_jsonl(path: str | Path, rows: list[dict[str, Any]], low_iou_threshold: float) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            metrics = row.get("oracle_metrics") or {}
            iou = float(metrics.get("iou") or 0.0)
            if bool(metrics.get("hit")) and iou >= low_iou_threshold:
                continue
            record = {
                "id": row.get("id"),
                "image": row.get("image"),
                "instruction": row.get("instruction"),
                "data_type": data_type(row),
                "answer_bbox": row.get("answer_bbox"),
                "candidate_count": row.get("candidate_count") or len(row.get("candidates") or []),
                "oracle_candidate_id": row.get("oracle_candidate_id"),
                "oracle_bbox": row.get("oracle_bbox"),
                "oracle_metrics": metrics,
                "oracle_source": source_of_oracle(row),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(path: str | Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    source_counter: Counter[str] = Counter(source_of_oracle(row) for row in rows if bool((row.get("oracle_metrics") or {}).get("hit")))
    miss_rows = [row for row in rows if not bool((row.get("oracle_metrics") or {}).get("hit"))]
    low_iou_rows = [row for row in rows if float((row.get("oracle_metrics") or {}).get("iou") or 0.0) < args.low_iou_threshold]
    lines = [
        "# GUI Candidate Miss Diagnosis",
        "",
        f"- Data: `{args.data}`",
        f"- Rows: {summary['count']}",
        f"- Low-IoU threshold: `{args.low_iou_threshold}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Avg candidates | {num(summary['avg_candidates'])} |",
        f"| Oracle hit | {pct(summary['oracle_hit_rate'])} |",
        f"| Oracle pointing | {pct(summary['oracle_pointing_rate'])} |",
        f"| Oracle IoU@0.5 | {pct(summary['oracle_iou50_rate'])} |",
        f"| Avg oracle IoU | {num(summary['avg_oracle_iou'])} |",
        f"| Oracle miss rows | {len(miss_rows)} |",
        f"| Low-IoU rows | {len(low_iou_rows)} |",
        "",
        "## By OS-Atlas Data Type",
        "",
        "| Type | Rows | Avg Cand | Oracle Hit | IoU@0.5 | Avg IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in type_summary(rows):
        lines.append(
            f"| `{item['data_type']}` | {item['count']} | {num(item['avg_candidates'])} | "
            f"{pct(item['oracle_hit_rate'])} | {pct(item['oracle_iou50_rate'])} | {num(item['avg_oracle_iou'])} |"
        )
    lines.extend([
        "",
        "## Oracle Hit Sources",
        "",
        "| Source | Count |",
        "|---|---:|",
    ])
    for source, count in source_counter.most_common():
        lines.append(f"| `{source}` | {count} |")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = attach_source_meta(read_jsonl(args.data), args.source_data)
    summary = summarize(rows)
    summary["by_type"] = type_summary(rows)
    summary["data"] = args.data
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.miss_jsonl:
        write_miss_jsonl(args.miss_jsonl, rows, args.low_iou_threshold)
    write_report(args.output, args, rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
