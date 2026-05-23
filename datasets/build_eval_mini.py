#!/usr/bin/env python3
"""Build a small local VLM evaluation set from public HF datasets.

The script streams a limited number of samples, saves images locally, and writes
an EviTool-style JSONL file. It defaults to the Hugging Face mirror because the
main endpoint is often slow from mainland China networks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from PIL import Image


DEFAULT_OUTPUT_DIR = "/root/models/datasets/evitool_eval_mini"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chartqa", type=int, default=30)
    parser.add_argument("--docvqa", type=int, default=25)
    parser.add_argument("--ai2d", type=int, default=20)
    parser.add_argument("--screenspot", type=int, default=25)
    parser.add_argument("--max-scan", type=int, default=500)
    return parser.parse_args()


def save_image(image: Image.Image, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    image.save(image_path)


def first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def take_stream(stream: Iterable[dict[str, Any]], limit: int, max_scan: int) -> list[dict[str, Any]]:
    samples = []
    for idx, example in enumerate(stream):
        if idx >= max_scan or len(samples) >= limit:
            break
        samples.append(example)
    return samples


def build_chartqa(out_dir: Path, limit: int, max_scan: int) -> list[dict[str, Any]]:
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
    records = []
    for i, ex in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/chartqa/{i:04d}.png"
        save_image(ex["image"], out_dir / image_rel)
        answer = first_text(ex.get("label", ""))
        records.append({
            "id": f"chartqa_{i:04d}",
            "source": "HuggingFaceM4/ChartQA",
            "task_type": "chart_qa",
            "image": image_rel,
            "question": str(ex.get("query", "")),
            "answer": answer,
            "answers": ex.get("label", [answer]),
            "meta": {"human_or_machine": ex.get("human_or_machine")},
        })
    return records


def build_docvqa(out_dir: Path, limit: int, max_scan: int) -> list[dict[str, Any]]:
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)
    records = []
    for i, ex in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/docvqa/{i:04d}.png"
        save_image(ex["image"], out_dir / image_rel)
        answer = first_text(ex.get("answers", ""))
        records.append({
            "id": f"docvqa_{i:04d}",
            "source": "lmms-lab/DocVQA",
            "task_type": "doc_qa",
            "image": image_rel,
            "question": str(ex.get("question", "")),
            "answer": answer,
            "answers": ex.get("answers", [answer]),
            "meta": {
                "question_id": ex.get("questionId"),
                "question_types": ex.get("question_types"),
                "doc_id": ex.get("docId"),
            },
        })
    return records


def build_ai2d(out_dir: Path, limit: int, max_scan: int) -> list[dict[str, Any]]:
    ds = load_dataset("lmms-lab/ai2d", split="test", streaming=True)
    records = []
    for i, ex in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/ai2d/{i:04d}.png"
        save_image(ex["image"], out_dir / image_rel)
        options = [str(option) for option in ex.get("options", [])]
        raw_answer = str(ex.get("answer", ""))
        answer = raw_answer
        if raw_answer.isdigit():
            answer_idx = int(raw_answer)
            if 0 <= answer_idx < len(options):
                answer = options[answer_idx]
        option_text = "\n".join(f"{chr(65 + j)}. {option}" for j, option in enumerate(options))
        question = str(ex.get("question", ""))
        if option_text:
            question = f"{question}\n{option_text}"
        records.append({
            "id": f"ai2d_{i:04d}",
            "source": "lmms-lab/ai2d",
            "task_type": "science_diagram_qa",
            "image": image_rel,
            "question": question,
            "answer": answer,
            "choices": options,
            "meta": {"raw_answer": raw_answer},
        })
    return records


def xywh_to_xyxy(bbox: list[Any]) -> list[int]:
    x, y, w, h = [int(round(float(v))) for v in bbox]
    return [x, y, x + w, y + h]


def build_screenspot(out_dir: Path, limit: int, max_scan: int) -> list[dict[str, Any]]:
    ds = load_dataset("lmms-lab/ScreenSpot-v2", split="train", streaming=True)
    records = []
    for i, ex in enumerate(take_stream(ds, limit, max_scan)):
        image_rel = f"images/screenspot/{i:04d}.png"
        save_image(ex["image"], out_dir / image_rel)
        bbox_xyxy = xywh_to_xyxy(ex.get("bbox", [0, 0, 0, 0]))
        instruction = str(ex.get("instruction", ""))
        records.append({
            "id": f"screenspot_{i:04d}",
            "source": "lmms-lab/ScreenSpot-v2",
            "task_type": "gui_grounding",
            "image": image_rel,
            "question": f"Locate the UI element for this instruction: {instruction}. Return the bounding box.",
            "answer": bbox_xyxy,
            "answer_bbox": bbox_xyxy,
            "evidence_regions": [{"id": "gt_ev_1", "bbox": bbox_xyxy, "label": instruction}],
            "meta": {
                "img_filename": ex.get("img_filename"),
                "data_type": ex.get("data_type"),
                "data_source": ex.get("data_source"),
                "bbox_xywh": ex.get("bbox"),
            },
        })
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    plan = [
        ("chartqa", args.chartqa, build_chartqa),
        ("docvqa", args.docvqa, build_docvqa),
        ("ai2d", args.ai2d, build_ai2d),
        ("screenspot", args.screenspot, build_screenspot),
    ]
    summary = {}
    for name, limit, builder in plan:
        if limit <= 0:
            continue
        print(f"Building {name}: target={limit}", flush=True)
        built = builder(out_dir, limit, args.max_scan)
        records.extend(built)
        summary[name] = len(built)
        print(f"Built {name}: {len(built)}", flush=True)

    write_jsonl(out_dir / "eval_mini_100.jsonl", records)
    (out_dir / "summary.json").write_text(
        json.dumps({"total": len(records), "by_source": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} records to {out_dir / 'eval_mini_100.jsonl'}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
