#!/usr/bin/env python3
"""Augment low-coverage GUI candidate rows with DashScope VLM boxes."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gui_candidate_env import draw_candidate_overlay, oracle_candidate
from tools.common import bbox_area, bbox_iou, clip_bbox, image_size, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["all", "train", "val"])
    parser.add_argument("--models", default="gui-plus-2026-02-26,qwen3.7-max-2026-05-20,qwen3.7-max")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--augment-iou-below", type=float, default=0.5)
    parser.add_argument("--max-vlm-boxes", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=110)
    parser.add_argument(
        "--coordinate-mode",
        choices=("pixel", "qwen1000"),
        default="qwen1000",
        help="DashScope GUI models commonly emit 0-1000 coordinates; convert them to pixels by default.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API workers per split.")
    return parser.parse_args()


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def api_key(args: argparse.Namespace) -> str:
    load_env_file(args.env_file)
    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"Set {args.api_key_env} in environment or {args.env_file}.")
    return key


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def instruction_text(row: dict[str, Any]) -> str:
    return str(row.get("instruction") or (row.get("meta") or {}).get("instruction") or row.get("question") or "")


def should_augment(row: dict[str, Any], threshold: float) -> bool:
    metrics = row.get("oracle_metrics") or {}
    return float(metrics.get("iou") or 0.0) < threshold


def build_prompt(row: dict[str, Any], size: tuple[int, int], max_boxes: int) -> str:
    return (
        "You are proposing candidate bounding boxes for a GUI grounding dataset.\n"
        "Given the screenshot and instruction, return several plausible UI element boxes that could match the instruction.\n"
        "Return absolute pixel coordinates in xyxy format using the screenshot coordinate system.\n"
        "Do not click. Do not explain. Do not return markdown.\n"
        f"Screenshot size: width={size[0]}, height={size[1]}.\n"
        f"Instruction: {instruction_text(row)}\n"
        f"Return JSON only: {{\"bboxes\":[[x1,y1,x2,y2], ...]}}. Return at most {max_boxes} boxes."
    )


def chat_completion(
    *,
    key: str,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
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
    return str(message.get("content") or "").strip(), payload


def parse_json_object(text: str) -> Any:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def to_pixel_box(box: list[Any], size: tuple[int, int], coordinate_mode: str) -> list[float]:
    values = [float(v) for v in box[:4]]
    if coordinate_mode == "qwen1000":
        width, height = size
        return [values[0] / 1000.0 * width, values[1] / 1000.0 * height, values[2] / 1000.0 * width, values[3] / 1000.0 * height]
    return values


def parse_bboxes(text: str, size: tuple[int, int], max_boxes: int, coordinate_mode: str) -> list[list[int]]:
    try:
        payload = parse_json_object(text)
    except Exception:
        nums = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text)]
        payload = {"bboxes": [nums[i : i + 4] for i in range(0, len(nums) - 3, 4)]}
    raw_boxes = payload.get("bboxes") if isinstance(payload, dict) else payload
    if not isinstance(raw_boxes, list):
        return []
    if len(raw_boxes) >= 4 and all(isinstance(value, (int, float)) for value in raw_boxes[:4]):
        raw_boxes = [raw_boxes[:4]]
    elif (
        len(raw_boxes) == 2
        and all(isinstance(point, list) and len(point) >= 2 for point in raw_boxes)
        and all(isinstance(value, (int, float)) for point in raw_boxes for value in point[:2])
    ):
        raw_boxes = [[raw_boxes[0][0], raw_boxes[0][1], raw_boxes[1][0], raw_boxes[1][1]]]
    boxes: list[list[int]] = []
    for item in raw_boxes:
        if isinstance(item, dict):
            box = item.get("bbox") or item.get("bbox_2d") or item.get("box") or item.get("coordinates")
        else:
            box = item
        if (
            isinstance(box, list)
            and len(box) == 2
            and all(isinstance(point, list) and len(point) >= 2 for point in box)
        ):
            box = [box[0][0], box[0][1], box[1][0], box[1][1]]
        if not isinstance(box, list) or len(box) < 4:
            continue
        try:
            clipped = clip_bbox(to_pixel_box(box, size, coordinate_mode), size)
        except Exception:
            continue
        if bbox_area(clipped) < 16:
            continue
        boxes.append(clipped)
        if len(boxes) >= max_boxes:
            break
    return boxes


def duplicate_box(box: list[int], candidates: list[dict[str, Any]]) -> bool:
    for candidate in candidates:
        bbox = candidate.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and bbox_iou(box, bbox) >= 0.9:
            return True
    return False


def add_dashscope_candidates(row: dict[str, Any], boxes: list[list[int]], model: str, max_candidates: int) -> dict[str, Any]:
    candidates = list(row.get("candidates") or [])
    added = 0
    for box in boxes:
        if len(candidates) >= max_candidates:
            break
        if duplicate_box(box, candidates):
            continue
        candidate_id = f"c{len(candidates):02d}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "rank": len(candidates) + 1,
                "bbox": box,
                "score": 0.66,
                "rank_score": 0.66,
                "label": "dashscope_target_augment",
                "source": "dashscope_target_augment",
                "sources": ["dashscope_target_augment", model],
                "text": "",
            }
        )
        added += 1
    oracle, oracle_metrics = oracle_candidate(row, candidates)
    enriched = dict(row)
    enriched["candidates"] = candidates
    enriched["candidate_count"] = len(candidates)
    enriched["oracle_candidate_id"] = oracle.get("candidate_id") if oracle else None
    enriched["oracle_bbox"] = oracle.get("bbox") if oracle else None
    enriched["oracle_rank"] = oracle.get("rank") if oracle else None
    enriched["oracle_metrics"] = oracle_metrics
    meta = dict(enriched.get("candidate_meta") or {})
    dash_meta = dict(meta.get("dashscope_augment") or {})
    dash_meta["added"] = int(dash_meta.get("added") or 0) + added
    dash_meta["source_model"] = model
    meta["dashscope_augment"] = dash_meta
    enriched["candidate_meta"] = meta
    return enriched


def augment_row(row: dict[str, Any], args: argparse.Namespace, key: str, models: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not should_augment(row, args.augment_iou_below):
        return row, {"attempted": False, "added": 0, "model": None, "error": None}
    image_path = Path(str(row.get("image_path") or row.get("image")))
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    size = image_size(load_image(image_path))
    prompt = build_prompt(row, size, args.max_vlm_boxes)
    errors = []
    for model in models:
        try:
            text, raw = chat_completion(
                key=key,
                base_url=args.base_url,
                model=model,
                image_path=image_path,
                prompt=prompt,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            boxes = parse_bboxes(text, size, args.max_vlm_boxes, args.coordinate_mode)
            enriched = add_dashscope_candidates(row, boxes, model, args.max_candidates)
            usage = raw.get("usage") if isinstance(raw, dict) else None
            return enriched, {
                "attempted": True,
                "added": len(enriched.get("candidates") or []) - len(row.get("candidates") or []),
                "parsed_boxes": len(boxes),
                "model": model,
                "error": None,
                "usage": usage,
                "response": text[:1000],
            }
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {exc}")
    return row, {"attempted": True, "added": 0, "model": None, "error": " | ".join(errors)}


def refresh_overlay(row: dict[str, Any], output_dir: Path, split: str, max_candidates: int) -> dict[str, Any]:
    candidates = row.get("candidates") or []
    if not candidates:
        return row
    raw_id = str(row.get("id") or "row")
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw_id)
    overlay_path = output_dir / "overlays" / split / f"{stem}.png"
    image_path = Path(str(row.get("image_path") or row.get("image")))
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    draw_candidate_overlay(image_path, candidates, overlay_path, max_candidates=max_candidates)
    enriched = dict(row)
    enriched["overlay_image"] = str(overlay_path)
    return enriched


def summarize(rows: list[dict[str, Any]], logs: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    metrics = [row.get("oracle_metrics") or {} for row in rows]
    return {
        "count": len(rows),
        "avg_candidates": mean([float(row.get("candidate_count") or len(row.get("candidates") or [])) for row in rows]),
        "oracle_hit_rate": mean([float(bool(metric.get("hit"))) for metric in metrics]),
        "oracle_pointing_rate": mean([float(bool(metric.get("pointing"))) for metric in metrics]),
        "oracle_iou50_rate": mean([float(bool(metric.get("iou_50"))) for metric in metrics]),
        "avg_oracle_iou": mean([float(metric.get("iou") or 0.0) for metric in metrics]),
        "attempted": sum(int(log.get("attempted", False)) for log in logs),
        "added_boxes": sum(int(log.get("added") or 0) for log in logs),
        "errors": sum(int(bool(log.get("error"))) for log in logs),
    }


def main() -> int:
    args = parse_args()
    key = api_key(args)
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    split_summaries: dict[str, Any] = {}
    all_logs: list[dict[str, Any]] = []

    for split in args.splits:
        rows = read_jsonl(input_dir / f"{split}.jsonl", args.limit)
        output_rows: list[dict[str, Any] | None] = [None] * len(rows)
        logs: list[dict[str, Any] | None] = [None] * len(rows)

        def process(index: int, row: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
            before_metrics = row.get("oracle_metrics") or {}
            enriched, log = augment_row(row, args, key, models)
            enriched = refresh_overlay(enriched, output_dir, split, args.max_candidates)
            log = {
                **log,
                "split": split,
                "id": row.get("id"),
                "before_iou": before_metrics.get("iou"),
                "after_iou": (enriched.get("oracle_metrics") or {}).get("iou"),
                "after_hit": (enriched.get("oracle_metrics") or {}).get("hit"),
            }
            return index, enriched, log

        if args.workers <= 1:
            for index, row in enumerate(rows, start=1):
                result_index, enriched, log = process(index, row)
                output_rows[result_index - 1] = enriched
                logs[result_index - 1] = log
                all_logs.append(log)
                if args.sleep_sec > 0 and log.get("attempted"):
                    time.sleep(args.sleep_sec)
                if index % 10 == 0 or index == len(rows):
                    print(f"[{split} {index}/{len(rows)}] augmented", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process, index, row) for index, row in enumerate(rows, start=1)]
                completed = 0
                for future in as_completed(futures):
                    result_index, enriched, log = future.result()
                    output_rows[result_index - 1] = enriched
                    logs[result_index - 1] = log
                    all_logs.append(log)
                    completed += 1
                    if completed % 10 == 0 or completed == len(rows):
                        print(f"[{split} {completed}/{len(rows)}] augmented", flush=True)

        final_rows = [row for row in output_rows if row is not None]
        final_logs = [log for log in logs if log is not None]
        write_jsonl(output_dir / f"{split}.jsonl", final_rows)
        write_jsonl(output_dir / f"{split}_dashscope_logs.jsonl", final_logs)
        split_summaries[split] = summarize(final_rows, final_logs)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": args.splits,
        "models": models,
        "augment_iou_below": args.augment_iou_below,
        "max_vlm_boxes": args.max_vlm_boxes,
        "max_candidates": args.max_candidates,
        "coordinate_mode": args.coordinate_mode,
        "split_summaries": split_summaries,
        "total_attempted": sum(int(log.get("attempted", False)) for log in all_logs),
        "total_added_boxes": sum(int(log.get("added") or 0) for log in all_logs),
        "total_errors": sum(int(bool(log.get("error"))) for log in all_logs),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
