#!/usr/bin/env python3
"""Run direct-answer VLM baselines on eval_mini_100.

This script loads one model once, runs image-question inference, writes JSONL
predictions, and computes lightweight first-pass metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None


TEXT_TASKS = {"chart_qa", "doc_qa", "science_diagram_qa"}
GROUNDING_TASKS = {"gui_grounding"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model path or HF repo id.")
    parser.add_argument("--adapter", default=None, help="Optional PEFT/LoRA adapter path to load on top of --model.")
    parser.add_argument("--data", default="data/eval_mini/eval_mini_100.jsonl")
    parser.add_argument("--image-root", default="data/eval_mini")
    parser.add_argument("--output", required=True, help="Prediction JSONL path.")
    parser.add_argument("--summary", default=None, help="Metrics JSON path. Defaults to output + .summary.json")
    parser.add_argument("--limit", type=int, default=None, help="Optional total sample limit.")
    parser.add_argument("--sample-per-task", type=int, default=None, help="Optional first N samples per task.")
    parser.add_argument("--task", action="append", default=None, help="Restrict to task type. Can repeat.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task:
        allowed = set(args.task)
        rows = [row for row in rows if row.get("task_type") in allowed]
    if args.sample_per_task is not None:
        counts: Counter[str] = Counter()
        selected = []
        for row in rows:
            task = row.get("task_type", "unknown")
            if counts[task] >= args.sample_per_task:
                continue
            selected.append(row)
            counts[task] += 1
        rows = selected
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def load_model(model_name_or_path: str, adapter_name_or_path: str | None = None):
    lower_name = model_name_or_path.lower()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {"dtype": dtype, "device_map": "auto", "trust_remote_code": True}

    if "qwen3" in lower_name:
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("Current transformers does not expose Qwen3VLForConditionalGeneration.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        if adapter_name_or_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_name_or_path)
        return model

    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Current transformers does not expose Qwen2_5_VLForConditionalGeneration.")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    if adapter_name_or_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_name_or_path)
    return model


def direct_prompt(row: dict[str, Any]) -> str:
    question = str(row.get("question", ""))
    if row.get("task_type") == "gui_grounding":
        return (
            "Locate the UI element described by the instruction. "
            "Return only the bounding box as [x1, y1, x2, y2].\n\n"
            f"Question: {question}"
        )
    return (
        "Answer the question based on the image. Return only the final answer.\n\n"
        f"Question: {question}"
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


def generate_one(model, processor, image_path: Path, prompt: str, max_new_tokens: int, temperature: float) -> str:
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


def normalize_text(text: Any) -> str:
    text = str(text).lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = re.sub(r"^(answer|final answer)\s*[:：]\s*", "", text)
    text = re.sub(r"[。.!?,;:，；：]+$", "", text)
    return text.strip()


def extract_number(text: Any) -> float | None:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(text).replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def relaxed_numeric_match(pred: str, answers: list[Any], rel_tol: float = 0.03) -> bool:
    pred_num = extract_number(pred)
    if pred_num is None:
        return False
    for answer in answers:
        answer_num = extract_number(answer)
        if answer_num is None:
            continue
        denom = max(abs(answer_num), 1.0)
        if abs(pred_num - answer_num) / denom <= rel_tol:
            return True
    return False


def multiple_choice_match(pred: str, row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("task_type") != "science_diagram_qa":
        return None
    choices = row.get("choices") or []
    raw_answer = str((row.get("meta") or {}).get("raw_answer", ""))
    if not raw_answer.isdigit():
        return None
    answer_idx = int(raw_answer)
    if answer_idx < 0 or answer_idx >= len(choices):
        return None

    correct_letter = chr(65 + answer_idx)
    correct_text = str(choices[answer_idx])
    pred_norm = normalize_text(pred)
    correct_letter_norm = correct_letter.lower()
    correct_text_norm = normalize_text(correct_text)

    # Accept either a bare option letter, an option prefix such as "B. D",
    # or the option text itself. This handles direct-answer VLM outputs.
    first_option = re.match(r"^([a-z])\s*[\.\)\:：-]?", pred_norm)
    letter_ok = bool(first_option and first_option.group(1) == correct_letter_norm)
    text_ok = pred_norm == correct_text_norm or pred_norm.endswith(f" {correct_text_norm}")
    exact = letter_ok or text_ok
    return {
        "pred_norm": pred_norm,
        "answer_norms": [correct_letter_norm, correct_text_norm],
        "correct_letter": correct_letter,
        "correct_text": correct_text,
        "exact_match": exact,
        "relaxed_match": exact,
    }


def text_match(pred: str, row: dict[str, Any]) -> dict[str, Any]:
    mc = multiple_choice_match(pred, row)
    if mc is not None:
        return mc

    answers = row.get("answers")
    if not isinstance(answers, list):
        answers = [row.get("answer", "")]
    pred_norm = normalize_text(pred)
    answer_norms = [normalize_text(answer) for answer in answers]
    exact = pred_norm in answer_norms
    relaxed = exact or relaxed_numeric_match(pred, answers)
    return {
        "pred_norm": pred_norm,
        "answer_norms": answer_norms,
        "exact_match": exact,
        "relaxed_match": relaxed,
    }


def parse_numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))]


def parse_bbox_or_point(text: str) -> tuple[list[float] | None, list[float] | None]:
    nums = parse_numbers(text)
    if len(nums) >= 4:
        return nums[:4], None
    if len(nums) >= 2:
        return None, nums[:2]
    return None, None


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def point_in_bbox(point: list[float], bbox: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def grounding_match(pred: str, row: dict[str, Any]) -> dict[str, Any]:
    gt = row.get("answer_bbox") or row.get("answer")
    if not isinstance(gt, list) or len(gt) != 4:
        return {"bbox_parseable": False, "iou": 0.0, "iou_50": False, "pointing": False}
    bbox, point = parse_bbox_or_point(pred)
    if bbox is not None:
        iou = bbox_iou(bbox, [float(x) for x in gt])
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        return {
            "bbox_parseable": True,
            "pred_bbox": bbox,
            "iou": iou,
            "iou_50": iou >= 0.5,
            "pointing": point_in_bbox(center, gt),
        }
    if point is not None:
        return {
            "bbox_parseable": True,
            "pred_point": point,
            "iou": 0.0,
            "iou_50": False,
            "pointing": point_in_bbox(point, gt),
        }
    return {"bbox_parseable": False, "iou": 0.0, "iou_50": False, "pointing": False}


def score_prediction(pred: str, row: dict[str, Any]) -> dict[str, Any]:
    task = row.get("task_type")
    if task in GROUNDING_TASKS:
        return grounding_match(pred, row)
    return text_match(pred, row)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[result["task_type"]].append(result)

    task_summary = {}
    for task, items in by_task.items():
        if task in GROUNDING_TASKS:
            task_summary[task] = {
                "count": len(items),
                "bbox_parse_rate": mean([float(x["metrics"].get("bbox_parseable", False)) for x in items]),
                "iou_50": mean([float(x["metrics"].get("iou_50", False)) for x in items]),
                "pointing_accuracy": mean([float(x["metrics"].get("pointing", False)) for x in items]),
                "mean_iou": mean([float(x["metrics"].get("iou", 0.0)) for x in items]),
            }
        else:
            task_summary[task] = {
                "count": len(items),
                "exact_match": mean([float(x["metrics"].get("exact_match", False)) for x in items]),
                "relaxed_match": mean([float(x["metrics"].get("relaxed_match", False)) for x in items]),
            }

    text_items = [r for r in results if r["task_type"] in TEXT_TASKS]
    grounding_items = [r for r in results if r["task_type"] in GROUNDING_TASKS]
    return {
        "count": len(results),
        "task_counts": dict(Counter(r["task_type"] for r in results)),
        "text_exact_match": mean([float(r["metrics"].get("exact_match", False)) for r in text_items]),
        "text_relaxed_match": mean([float(r["metrics"].get("relaxed_match", False)) for r in text_items]),
        "grounding_iou_50": mean([float(r["metrics"].get("iou_50", False)) for r in grounding_items]),
        "grounding_pointing_accuracy": mean([float(r["metrics"].get("pointing", False)) for r in grounding_items]),
        "empty_output_rate": mean([float(not r["prediction"].strip()) for r in results]),
        "avg_latency_sec": mean([float(r["latency_sec"]) for r in results]),
        "avg_output_chars": mean([float(len(r["prediction"])) for r in results]),
        "by_task": task_summary,
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    data_path = Path(args.data)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_rows(data_path), args)
    print(f"Loaded {len(rows)} samples from {data_path}", flush=True)
    print(f"Loading model: {args.model}", flush=True)
    if args.adapter:
        print(f"Loading adapter: {args.adapter}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.adapter)
    model.eval()

    results = []
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            image_path = image_root / row["image"]
            prompt = direct_prompt(row)
            started = time.time()
            error = None
            prediction = ""
            try:
                prediction = generate_one(model, processor, image_path, prompt, args.max_new_tokens, args.temperature)
            except Exception as exc:  # Keep batch runs debuggable.
                error = repr(exc)
            latency = time.time() - started
            metrics = score_prediction(prediction, row) if error is None else {"error": error}
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
                "error": error,
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            results.append(result)
            print(f"[{idx}/{len(rows)}] {row.get('id')} {row.get('task_type')} -> {prediction[:120]!r}", flush=True)

    summary = summarize(results)
    summary["model"] = args.model
    summary["adapter"] = args.adapter
    summary["data"] = str(data_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
