#!/usr/bin/env python3
"""A/B test GUI candidate generators without running a VLM."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import detect
from tools.common import bbox_center, bbox_iou, point_in_bbox
from tools.external_detectors import run_external_detectors

RECALL_KS = (1, 3, 5, 10, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl")
    parser.add_argument("--image-root", default="/root/models/datasets/evitool_traces_1k")
    parser.add_argument("--output", default="reports/gui_candidate_ablation.md")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=["trace_initial", "trace_pipeline", "current", "current_ocr", "omniparser", "current_omniparser", "current_ocr_omniparser"])
    parser.add_argument("--max-results", type=int, default=30)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--omniparser-root", default="third_party/OmniParser")
    parser.add_argument("--omniparser-weights-dir", default="third_party/OmniParser/weights")
    parser.add_argument("--omniparser-use-caption", action="store_true")
    return parser.parse_args()


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("task_type") != "gui_grounding":
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def trace_detect_obs(row: dict[str, Any]) -> list[dict[str, Any]]:
    obs = []
    for step in row.get("trace") or []:
        observation = step.get("observation") or {}
        if observation.get("tool") == "detect":
            obs.append(observation)
    return obs


def get_trace_detections(row: dict[str, Any], pipeline: bool) -> list[dict[str, Any]]:
    obs = trace_detect_obs(row)
    if not obs:
        return []
    if pipeline and (row.get("quality") or {}).get("ocr_fallback_hit") and len(obs) > 1:
        return (obs[1].get("content") or {}).get("detections") or []
    return (obs[0].get("content") or {}).get("detections") or []


def run_variant(row: dict[str, Any], image_path: Path, variant: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    instruction = str((row.get("meta") or {}).get("instruction") or row.get("question") or "")
    started = time.time()
    if variant == "trace_initial":
        detections = get_trace_detections(row, pipeline=False)
        return detections[: args.max_results], {"source": "trace", "latency_sec": 0.0, "available": True}
    if variant == "trace_pipeline":
        detections = get_trace_detections(row, pipeline=True)
        return detections[: args.max_results], {"source": "trace", "latency_sec": 0.0, "available": True}
    if variant == "current":
        content, _, _ = detect.run(image_path, mode="ui", query=instruction, max_results=args.max_results, min_area=args.min_area)
        return content.get("detections") or [], {"source": "current", "latency_sec": round(time.time() - started, 4), "available": True, "meta": content.get("meta")}
    if variant == "current_ocr":
        content, _, _ = detect.run(image_path, mode="ui", query=instruction, max_results=args.max_results, min_area=args.min_area, include_ocr=True)
        return content.get("detections") or [], {"source": "current_ocr", "latency_sec": round(time.time() - started, 4), "available": True, "meta": content.get("meta")}
    if variant == "omniparser":
        items, meta = run_external_detectors(
            image_path,
            providers=["omniparser"],
            max_results=args.max_results,
            cache_dir=args.external_cache_dir,
            omniparser_root=args.omniparser_root,
            omniparser_weights_dir=args.omniparser_weights_dir,
            omniparser_use_caption=args.omniparser_use_caption,
        )
        return items, {**meta, "latency_sec": round(time.time() - started, 4)}
    if variant == "current_omniparser":
        content, _, _ = detect.run(
            image_path,
            mode="ui",
            query=instruction,
            max_results=args.max_results,
            min_area=args.min_area,
            include_omniparser=True,
            omniparser_root=args.omniparser_root,
            omniparser_weights_dir=args.omniparser_weights_dir,
            omniparser_use_caption=args.omniparser_use_caption,
            external_cache_dir=args.external_cache_dir,
        )
        return content.get("detections") or [], {"source": "current_omniparser", "latency_sec": round(time.time() - started, 4), "available": True, "meta": content.get("meta")}
    if variant == "current_ocr_omniparser":
        content, _, _ = detect.run(
            image_path,
            mode="ui",
            query=instruction,
            max_results=args.max_results,
            min_area=args.min_area,
            include_ocr=True,
            include_omniparser=True,
            omniparser_root=args.omniparser_root,
            omniparser_weights_dir=args.omniparser_weights_dir,
            omniparser_use_caption=args.omniparser_use_caption,
            external_cache_dir=args.external_cache_dir,
        )
        return content.get("detections") or [], {"source": "current_ocr_omniparser", "latency_sec": round(time.time() - started, 4), "available": True, "meta": content.get("meta")}
    raise ValueError(f"unknown variant: {variant}")


def candidate_hit(detections: list[dict[str, Any]], gt_bbox: list[float], k: int) -> tuple[bool, float, int | None]:
    best_iou = 0.0
    best_rank = None
    for idx, det in enumerate(detections[:k], start=1):
        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        iou = bbox_iou(bbox, gt_bbox)
        best_iou = max(best_iou, iou)
        center_hit = point_in_bbox(bbox_center(bbox), gt_bbox)
        if iou >= 0.3 or center_hit:
            return True, best_iou, idx
        if best_rank is None or iou == best_iou:
            best_rank = idx
    return False, best_iou, best_rank


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input), args.limit)
    image_root = Path(args.image_root)
    stats: dict[str, dict[str, Any]] = {variant: {"count": 0, "hits": Counter(), "best_ious": [], "hit_ranks": [], "candidate_counts": [], "latencies": [], "errors": Counter(), "sources": Counter()} for variant in args.variants}

    for idx, row in enumerate(rows, start=1):
        image_path = image_root / row["image"]
        gt = row.get("answer_bbox")
        if not isinstance(gt, list) or len(gt) != 4:
            continue
        for variant in args.variants:
            bucket = stats[variant]
            bucket["count"] += 1
            try:
                detections, meta = run_variant(row, image_path, variant, args)
                bucket["candidate_counts"].append(float(len(detections)))
                bucket["latencies"].append(float(meta.get("latency_sec", 0.0)))
                for det in detections:
                    for source in det.get("sources") or [det.get("source")]:
                        if source:
                            bucket["sources"][str(source)] += 1
                for k in RECALL_KS:
                    ok, best_iou, rank = candidate_hit(detections, gt, k)
                    bucket["hits"][k] += int(ok)
                    if k == args.max_results or k == 30:
                        bucket["best_ious"].append(best_iou)
                        if ok and rank is not None:
                            bucket["hit_ranks"].append(float(rank))
            except Exception as exc:
                bucket["errors"][f"{type(exc).__name__}: {exc}"] += 1
        if idx % 25 == 0:
            print(f"[{idx}/{len(rows)}] processed", flush=True)

    summary = {"input": args.input, "count": len(rows), "variants": {}}
    lines = [
        "# GUI Candidate A/B Report",
        "",
        f"- Input: `{args.input}`",
        f"- Samples: {len(rows)}",
        f"- Variants: `{', '.join(args.variants)}`",
        "- Hit definition: IoU >= 0.3 or candidate center inside GT bbox.",
        "",
        "| Variant | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@30 | Oracle Est. | Avg Candidates | Avg Best IoU | Avg Hit Rank | Avg Latency | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, bucket in stats.items():
        count = bucket["count"]
        recalls = {k: (bucket["hits"][k] / count if count else None) for k in RECALL_KS}
        row = {
            "count": count,
            "recall": recalls,
            "oracle_estimate": 1.0 - recalls[30] if recalls[30] is not None else None,
            "avg_candidates": mean(bucket["candidate_counts"]),
            "avg_best_iou": mean(bucket["best_ious"]),
            "avg_hit_rank": mean(bucket["hit_ranks"]),
            "avg_latency_sec": mean(bucket["latencies"]),
            "errors": dict(bucket["errors"]),
            "candidate_sources": dict(bucket["sources"]),
        }
        summary["variants"][variant] = row
        lines.append(
            f"| `{variant}` | {pct(recalls[1])} | {pct(recalls[3])} | {pct(recalls[5])} | {pct(recalls[10])} | {pct(recalls[30])} | "
            f"{pct(row['oracle_estimate'])} | {row['avg_candidates'] or 0:.2f} | {row['avg_best_iou'] or 0:.4f} | {row['avg_hit_rank'] or 0:.2f} | "
            f"{row['avg_latency_sec'] or 0:.3f}s | {sum(bucket['errors'].values())} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `omniparser` variants require local weights under `third_party/OmniParser/weights`; otherwise they report zero candidates or unchanged current-detect behavior.",
        "- Use this same report before and after installing external detectors to decide whether a provider enters the training data pipeline.",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    summary_path = Path(args.summary) if args.summary else output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
