#!/usr/bin/env python3
"""Build a compact markdown table from direct/tool summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="Summary JSON path. Can repeat.")
    parser.add_argument("--output", default="outputs/sft_eval/comparison.md")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def num(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "-"


def infer_label(summary: dict[str, Any]) -> str:
    path = summary.get("_path", "")
    model = str(summary.get("model", ""))
    family = "4B" if "qwen3" in model.lower() or "4b" in model.lower() else "3B"
    mode = "tool" if "avg_tool_calls" in summary or "tool_" in path else "direct"
    stage = "post-sft" if summary.get("adapter") or "sft" in path else "pre-sft"
    return f"{family} {stage} {mode}"


def row(summary: dict[str, Any]) -> list[str]:
    gui = (summary.get("by_task") or {}).get("gui_grounding") or {}
    return [
        infer_label(summary),
        pct(summary.get("text_relaxed_match")),
        pct(gui.get("pointing_accuracy") or summary.get("grounding_pointing_accuracy")),
        pct(gui.get("iou_50") or summary.get("grounding_iou_50")),
        pct(summary.get("evidence_closed_rate")),
        num(summary.get("avg_tool_calls")),
        pct(summary.get("samples_with_protocol_error_rate")),
        pct(summary.get("final_parse_rate")),
        summary.get("_path", ""),
    ]


def main() -> int:
    args = parse_args()
    summaries = [load(Path(path)) for path in args.summary if Path(path).exists()]
    headers = [
        "Run",
        "Text relaxed",
        "GUI pointing",
        "GUI IoU@0.5",
        "Evidence closed",
        "Avg tools",
        "Protocol error",
        "Final parse",
        "Summary",
    ]
    lines = ["# SFT Metrics Comparison", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for summary in summaries:
        lines.append("| " + " | ".join(row(summary)) + " |")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
