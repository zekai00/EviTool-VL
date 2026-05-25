#!/usr/bin/env python3
"""Evaluate a model or simple policy on fixed GUI candidate-selection data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gui_candidate_env import (
    build_candidate_prompt,
    oracle_candidate,
    parse_candidate_id,
    policy_action,
    score_candidate_action,
)

BASELINE_PATH = PROJECT_ROOT / "eval" / "eval_baseline.py"
spec = importlib.util.spec_from_file_location("eval_baseline", BASELINE_PATH)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/gui_candidate_rl/val.jsonl")
    parser.add_argument("--output", default="outputs/gui_candidate_rl/policy_eval.jsonl")
    parser.add_argument("--report", default="reports/gui_candidate_policy_eval.md")
    parser.add_argument("--policy", choices=["model", "top1", "oracle", "random", "invalid"], default="model")
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-overlay", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def make_model_prompt(row: dict[str, Any]) -> str:
    base = build_candidate_prompt(row, row.get("candidates") or [])
    return (
        f"{base}\n\n"
        "The image is annotated with candidate ids. Select the candidate that best matches the instruction. "
        "Return no explanation."
    )


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_one(model: Any, processor: Any, image_path: Path, prompt: str, max_new_tokens: int, temperature: float) -> str:
    messages = build_messages(image_path, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[:, input_len:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [r["reward"] for r in results]
    return {
        "count": len(results),
        "avg_reward": mean([float(r["total"]) for r in rewards]),
        "parseable_rate": mean([float(r["candidate_id"] is not None) for r in rewards]),
        "valid_rate": mean([float(r["valid"]) for r in rewards]),
        "pointing_rate": mean([float(r["pointing"]) for r in rewards]),
        "iou50_rate": mean([float(r["iou_50"]) for r in rewards]),
        "avg_iou": mean([float(r["metrics"]["iou"]) for r in rewards]),
        "avg_selected_rank": mean([float(r["selected_rank"]) for r in rewards if r.get("selected_rank") is not None]),
        "avg_latency_sec": mean([float(r["latency_sec"]) for r in results]),
    }


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# GUI Candidate Policy Eval",
        "",
        f"- Data: `{args.data}`",
        f"- Policy: `{args.policy}`",
        f"- Model: `{args.model if args.policy == 'model' else '-'}`",
        f"- Adapter: `{args.adapter if args.adapter else '-'}`",
        f"- Samples: {summary['count']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Avg reward | {num(summary['avg_reward'])} |",
        f"| Parseable | {pct(summary['parseable_rate'])} |",
        f"| Valid candidate | {pct(summary['valid_rate'])} |",
        f"| Pointing | {pct(summary['pointing_rate'])} |",
        f"| IoU@0.5 | {pct(summary['iou50_rate'])} |",
        f"| Avg IoU | {num(summary['avg_iou'])} |",
        f"| Avg selected rank | {num(summary['avg_selected_rank'])} |",
        f"| Avg latency(s) | {num(summary['avg_latency_sec'])} |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    records = load_records(Path(args.data), args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor = None
    model = None
    if args.policy == "model":
        print(f"Loading model: {args.model}", flush=True)
        if args.adapter:
            print(f"Loading adapter: {args.adapter}", flush=True)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = baseline.load_model(args.model, args.adapter)
        model.eval()

    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(records, start=1):
            candidates = row.get("candidates") or []
            started = time.time()
            prediction: Any
            error = None
            if args.policy == "model":
                image_field = row.get("overlay_image") if args.use_overlay and row.get("overlay_image") else row.get("image_path")
                prompt = make_model_prompt(row)
                try:
                    assert model is not None and processor is not None
                    prediction = generate_one(model, processor, Path(str(image_field)), prompt, args.max_new_tokens, args.temperature)
                except Exception as exc:
                    prediction = ""
                    error = repr(exc)
            elif args.policy == "oracle":
                best, _ = oracle_candidate(row, candidates)
                prediction = {"candidate_id": best["candidate_id"] if best else "c00"}
            else:
                prediction = policy_action(args.policy, row, candidates, rng)
            latency = time.time() - started
            reward = score_candidate_action(row, candidates, prediction)
            candidate_id, parseable = parse_candidate_id(prediction)
            result = {
                "id": row.get("id"),
                "image": row.get("image"),
                "instruction": row.get("instruction"),
                "answer_bbox": row.get("answer_bbox"),
                "candidate_count": len(candidates),
                "prediction": prediction,
                "candidate_id": candidate_id,
                "parseable": parseable,
                "reward": reward,
                "latency_sec": round(latency, 4),
                "error": error,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            results.append(result)
            print(f"[{idx}/{len(records)}] {row.get('id')} -> {prediction!r} reward={reward['total']}", flush=True)

    summary = summarize(results)
    summary.update({"policy": args.policy, "model": args.model if args.policy == "model" else None, "adapter": args.adapter})
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(Path(args.report), args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
