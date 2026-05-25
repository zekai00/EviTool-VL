#!/usr/bin/env python3
"""Run direct-answer evals through Alibaba Cloud Bailian OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import getpass
import importlib.util
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "eval_baseline.py"
spec = importlib.util.spec_from_file_location("eval_baseline", BASELINE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Bailian model id, e.g. qwen3.6-plus")
    parser.add_argument("--data", default="data/eval_mini/eval_mini_100.jsonl")
    parser.add_argument("--image-root", default="data/eval_mini")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key-stdin", action="store_true", help="Read API key from stdin without echoing.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-task", type=int, default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_api_key(args: argparse.Namespace) -> str:
    if args.api_key_stdin:
        if sys.stdin.isatty():
            key = getpass.getpass("Bailian API key: ").strip()
        else:
            key = sys.stdin.readline().strip()
    else:
        key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"Set {args.api_key_env} or pass a non-empty key via --api-key-stdin.")
    return key


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            ],
        }
    ]


def chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned: {payload}")
    message = choices[0].get("message") or {}
    return str(message.get("content", "")).strip(), payload


def main() -> int:
    args = parse_args()
    api_key = get_api_key(args)

    data_path = Path(args.data)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = base.select_rows(base.load_rows(data_path), args)
    print(f"Loaded {len(rows)} samples from {data_path}", flush=True)
    print(f"Using Bailian model: {args.model}", flush=True)

    results = []
    usage_totals: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            image_path = image_root / row["image"]
            prompt = base.direct_prompt(row)
            started = time.time()
            error = None
            prediction = ""
            raw_usage = None
            try:
                prediction, raw = chat_completion(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    messages=build_messages(image_path, prompt),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
                raw_usage = raw.get("usage")
                if isinstance(raw_usage, dict):
                    for key, value in raw_usage.items():
                        if isinstance(value, int):
                            usage_totals[key] = usage_totals.get(key, 0) + value
            except Exception as exc:
                error = repr(exc)
            latency = time.time() - started
            metrics = base.score_prediction(prediction, row) if error is None else {"error": error}
            result = {
                "id": row.get("id"),
                "task_type": row.get("task_type"),
                "image": row.get("image"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "answers": row.get("answers"),
                "prediction": prediction,
                "metrics": metrics,
                "latency_sec": round(latency, 4),
                "usage": raw_usage,
                "error": error,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            results.append(result)
            print(f"[{idx}/{len(rows)}] {row.get('id')} {row.get('task_type')} -> {prediction[:120]!r}", flush=True)

    summary = base.summarize(results)
    summary["model"] = args.model
    summary["data"] = str(data_path)
    summary["api_base_url"] = args.base_url
    summary["usage_totals"] = usage_totals
    summary["error_rate"] = base.mean([float(result["error"] is not None) for result in results])
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
