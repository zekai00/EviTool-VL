#!/usr/bin/env python3
"""Run a quick smoke test over the local visual tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.runner import run_tool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/eval_mini/images/chartqa/0000.png")
    parser.add_argument("--out-dir", default="outputs/tools_smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = Path(args.image)
    actions = [
        {"tool": "inspect", "args": {}},
        {"tool": "crop", "args": {"bbox": [80, 80, 760, 520]}},
        {"tool": "zoom", "args": {"bbox": [80, 80, 760, 520], "scale": 1.5}},
        {"tool": "ocr", "args": {"bbox": [80, 80, 760, 520], "engine": "auto"}},
        {"tool": "detect", "args": {"mode": "bar", "max_results": 10}},
        {"tool": "measure", "args": {"bbox": [80, 80, 760, 520]}},
        {"tool": "mark", "args": {"bboxes": [[80, 80, 760, 520]], "labels": ["chart_region"]}},
    ]
    for action in actions:
        result = run_tool(image, action, out_dir=args.out_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
