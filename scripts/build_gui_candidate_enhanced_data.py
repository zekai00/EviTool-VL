#!/usr/bin/env python3
"""Build enhanced GUI candidate data with source quotas.

The older fused-ui run globally sorted a mixed candidate pool and then
truncated it, which allowed one source to crowd out useful text/layout boxes.
This builder first asks `detect(mode=ui)` for a large pool, then keeps a
quota-balanced subset before computing oracle coverage.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gui_candidate_env import draw_candidate_overlay, oracle_candidate, summarize_candidate_records
from tools import detect as detect_tool
from tools.common import bbox_area, bbox_iou


DEFAULT_QUOTAS = {
    "base_c60": 60,
    "omniparser": 30,
    "ocr_text": 18,
    "text_expanded": 14,
    "row_container": 12,
    "ui_rect": 12,
    "ui_edge": 10,
    "query_prior": 8,
    "icon_visual": 8,
    "text_visual": 8,
    "layout": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/gui_candidate_rl_os_atlas_linux_2k_c60")
    parser.add_argument("--output-dir", default="outputs/gui_candidate_rl_os_atlas_linux_2k_enhanced_c100")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=240)
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--ocr-engine", default="easyocr")
    parser.add_argument("--include-ocr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-omniparser", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-input-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-aware", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--omniparser-root", default="third_party/OmniParser")
    parser.add_argument("--omniparser-weights-dir", default="third_party/OmniParser/weights")
    parser.add_argument("--val-ratio", type=float, default=None, help="Default reuses input-dir train/val split.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def instruction_text(row: dict[str, Any]) -> str:
    return str(row.get("instruction") or (row.get("meta") or {}).get("instruction") or row.get("question") or "")


def source_key(item: dict[str, Any]) -> str:
    sources = [str(source) for source in item.get("sources") or [item.get("source") or "unknown"]]
    for key in DEFAULT_QUOTAS:
        if key in sources:
            return key
    source = str(item.get("source") or sources[0] if sources else "unknown")
    if source.startswith("omniparser"):
        return "omniparser"
    return source


def input_candidates_as_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the previous candidate pool as a high-priority source.

    The old c60 OmniParser pool has relatively tight icon boxes.  Enhanced
    fused pools add useful text/layout candidates but can contain large row
    containers.  Preserving the input candidates prevents the enhanced dataset
    from improving center hits while destroying IoU@0.5 coverage.
    """
    items: list[dict[str, Any]] = []
    for candidate in row.get("candidates") or []:
        bbox = candidate.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        items.append(
            {
                **candidate,
                "source": "base_c60",
                "sources": ["base_c60", *[str(source) for source in candidate.get("sources") or []]],
                "score": float(candidate.get("score") or 0.0),
                "rank_score": 10.0 - 0.001 * float(candidate.get("rank") or 0),
                "label": candidate.get("label") or "base_c60_candidate",
            }
        )
    return items


def same_box(a: dict[str, Any], b: dict[str, Any]) -> bool:
    abox = a.get("bbox")
    bbox = b.get("bbox")
    if not isinstance(abox, list) or not isinstance(bbox, list):
        return False
    if bbox_iou(abox, bbox) >= 0.92:
        return True
    ax1, ay1, ax2, ay2 = [float(v) for v in abox]
    bx1, by1, bx2, by2 = [float(v) for v in bbox]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    smaller = min(bbox_area(abox), bbox_area(bbox))
    return smaller > 0 and inter / smaller >= 0.95


def add_unique(selected: list[dict[str, Any]], item: dict[str, Any]) -> bool:
    if any(same_box(existing, item) for existing in selected):
        return False
    selected.append(item)
    return True


def select_with_quotas(pool: list[dict[str, Any]], *, max_candidates: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for item in pool:
        by_source.setdefault(source_key(item), []).append(item)

    selected: list[dict[str, Any]] = []
    quota_used: Counter[str] = Counter()
    for source, quota in DEFAULT_QUOTAS.items():
        for item in by_source.get(source, [])[:quota]:
            if len(selected) >= max_candidates:
                break
            if add_unique(selected, item):
                quota_used[source] += 1

    # Fill remaining slots by the original detector ranking so high-confidence
    # candidates from any source still survive after quota balancing.
    for item in pool:
        if len(selected) >= max_candidates:
            break
        source = source_key(item)
        if add_unique(selected, item):
            quota_used[source] += 1

    return selected[:max_candidates], {
        "quotas": DEFAULT_QUOTAS,
        "quota_used": dict(quota_used),
        "source_pool_counts": {source: len(items) for source, items in sorted(by_source.items())},
    }


def normalize_candidates(items: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in items[:max_candidates]:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        sources = [str(source) for source in item.get("sources") or [item.get("source") or "unknown"]]
        candidates.append(
            {
                "candidate_id": f"c{len(candidates):02d}",
                "rank": len(candidates) + 1,
                "bbox": [int(round(float(v))) for v in bbox],
                "score": round(float(item.get("score", item.get("rank_score", 0.0)) or 0.0), 4),
                "rank_score": round(float(item.get("rank_score", item.get("score", 0.0)) or 0.0), 4),
                "label": str(item.get("label") or item.get("source") or "candidate"),
                "source": str(item.get("source") or sources[0] if sources else "unknown"),
                "sources": sources,
                "text": str(item.get("text") or ""),
            }
        )
    return candidates


def generate_candidates(row: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content, _, _ = detect_tool.run(
        row["image_path"],
        mode="ui",
        query=instruction_text(row) if args.query_aware else None,
        max_results=args.pool_size,
        min_area=args.min_area,
        include_ocr=args.include_ocr,
        ocr_engine=args.ocr_engine,
        ocr_languages=["en"],
        include_omniparser=args.include_omniparser,
        external_cache_dir=args.external_cache_dir,
        omniparser_root=args.omniparser_root,
        omniparser_weights_dir=args.omniparser_weights_dir,
        omniparser_use_caption=False,
    )
    pool = []
    if args.include_input_candidates:
        pool.extend(input_candidates_as_items(row))
    pool.extend(content.get("detections") or [])
    selected, quota_meta = select_with_quotas(pool, max_candidates=args.max_candidates)
    return normalize_candidates(selected, args.max_candidates), {
        "provider": "enhanced_quota_ui",
        "available": True,
        "pool_size": len(pool),
        "detect_count": content.get("count"),
        "detect_candidate_pool_size": content.get("candidate_pool_size"),
        "detect_candidate_sources": content.get("candidate_sources") or {},
        "ocr_available": content.get("ocr_available"),
        "ocr_engine": content.get("ocr_engine"),
        "ocr_errors": content.get("ocr_errors") or [],
        "external": content.get("external") or {},
        "query_aware": bool(args.query_aware),
        **quota_meta,
    }


def row_for_output(row: dict[str, Any], candidates: list[dict[str, Any]], meta: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    oracle, oracle_metrics = oracle_candidate(row, candidates)
    raw_id = str(row.get("id") or "row")
    stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw_id)
    overlay_path = output_dir / "overlays" / f"{stem}.png"
    if candidates:
        draw_candidate_overlay(row["image_path"], candidates, overlay_path, max_candidates=args.max_candidates)
    return {
        **{k: v for k, v in row.items() if k not in {"candidates", "candidate_meta", "oracle_candidate_id", "oracle_bbox", "oracle_rank", "oracle_metrics", "overlay_image", "candidate_count"}},
        "overlay_image": str(overlay_path) if candidates else None,
        "candidate_count": len(candidates),
        "candidate_meta": meta,
        "oracle_candidate_id": oracle.get("candidate_id") if oracle else None,
        "oracle_bbox": oracle.get("bbox") if oracle else None,
        "oracle_rank": oracle.get("rank") if oracle else None,
        "oracle_metrics": oracle_metrics,
        "candidates": candidates,
    }


def split_records(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    all_rows = read_jsonl(input_dir / "all.jsonl")
    train_ids = {str(row.get("id")) for row in read_jsonl(input_dir / "train.jsonl")} if (input_dir / "train.jsonl").exists() else set()
    val_ids = {str(row.get("id")) for row in read_jsonl(input_dir / "val.jsonl")} if (input_dir / "val.jsonl").exists() else set()

    cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    cache_hits = 0
    records: list[dict[str, Any]] = []
    for index, row in enumerate(all_rows, start=1):
        cache_key = str(row.get("image_path") or row.get("image"))
        if args.query_aware:
            cache_key += f"::{instruction_text(row)}"
        cached = cache.get(cache_key)
        if cached is None:
            candidates, meta = generate_candidates(row, args)
            cache[cache_key] = (copy.deepcopy(candidates), copy.deepcopy(meta))
        else:
            cache_hits += 1
            candidates, meta = copy.deepcopy(cached)
        records.append(row_for_output(row, candidates, meta, output_dir, args))
        if index % 25 == 0 or index == len(all_rows):
            print(f"[{index}/{len(all_rows)}] built", flush=True)

    if args.val_ratio is not None:
        train, val = split_records(records, args.val_ratio, args.seed)
    else:
        train = [row for row in records if str(row.get("id")) in train_ids]
        val = [row for row in records if str(row.get("id")) in val_ids]

    write_jsonl(output_dir / "all.jsonl", records)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_candidates": args.max_candidates,
        "pool_size": args.pool_size,
        "min_area": args.min_area,
        "include_ocr": args.include_ocr,
        "ocr_engine": args.ocr_engine,
        "include_omniparser": args.include_omniparser,
        "include_input_candidates": args.include_input_candidates,
        "query_aware": args.query_aware,
        "unique_candidate_cache_keys": len(cache),
        "candidate_cache_hits": cache_hits,
        "all": summarize_candidate_records(records),
        "train": summarize_candidate_records(train),
        "val": summarize_candidate_records(val),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
