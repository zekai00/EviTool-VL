#!/usr/bin/env python3
"""Convert a small WebArena-Infinity trajectory sample into SFT-style rows."""

from __future__ import annotations

import argparse
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--repo-id", default="webarena-x/webarena-infinity-trajectories")
    parser.add_argument("--timeout-sec", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo_id": args.repo_id,
        "requested_limit": args.limit,
        "converted": 0,
        "blocked": False,
        "blocker": None,
        "note": "This conversion is best-effort because upstream row schemas may vary.",
    }
    rows: list[dict[str, Any]] = []
    try:
        _set_timeout(args.timeout_sec)
        from datasets import load_dataset

        ds = load_dataset(args.repo_id, split="train", streaming=True)
        for idx, row in enumerate(ds):
            if idx >= args.limit:
                break
            converted = convert_row(row, idx, output_dir / "images")
            if converted is not None:
                rows.append(converted)
    except Exception as exc:
        summary["blocked"] = True
        summary["blocker"] = f"{type(exc).__name__}: {exc}"
    finally:
        _clear_timeout()
    summary["converted"] = len(rows)
    (output_dir / "sft_messages.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _set_timeout(seconds: int) -> None:
    if seconds <= 0:
        return

    def handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"timed out after {seconds}s")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)


def _clear_timeout() -> None:
    signal.alarm(0)


def convert_row(row: dict[str, Any], idx: int, image_dir: Path) -> dict[str, Any] | None:
    image_path = None
    image = row.get("image")
    if image is not None and hasattr(image, "save"):
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"webarena_infinity_{idx:05d}.png"
        image.save(image_path)
    text_parts = []
    for key in ("instruction", "task", "goal", "intent", "action", "reasoning", "result", "trajectory"):
        if key in row and row[key] is not None:
            text_parts.append(f"{key}: {row[key]}")
    if not text_parts and image_path is None:
        return None
    prompt = "<image>\n请根据该 WebArena-Infinity 轨迹样本整理下一步 browser action。" if image_path else "请整理该 WebArena-Infinity 轨迹样本。"
    answer = "\n".join(text_parts)[:8000] if text_parts else "{}"
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": [str(image_path)] if image_path else [],
        "source": "webarena-infinity-trajectories",
        "row_index": idx,
        "raw_keys": sorted(row.keys()),
    }


if __name__ == "__main__":
    main()
