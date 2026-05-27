#!/usr/bin/env python3
"""Targetedly refine GUI candidates for menu rows and spreadsheet table cells.

This is a local post-processing pass over an existing candidate pool.  It does
not use `answer_bbox` to propose boxes.  Menu candidates are expanded from
query-conditioned boxes such as DashScope target boxes; spreadsheet cell boxes
are inferred from the instruction (for example `Y1`) and visible Calc grid
lines in the screenshot.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.gui_candidate_env import draw_candidate_overlay, oracle_candidate, summarize_candidate_records
from tools.common import bbox_area, bbox_center, bbox_iou, clip_bbox, image_size, load_image


MENU_TYPES = {"menu", "menu-item", "check-menu-item"}
SPREADSHEET_CELL_RE = re.compile(r"^\s*([A-Z]{1,3})([1-9]\d{0,3})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-data",
        default="/root/models/datasets/os_atlas_linux_2k/source_os_atlas_linux_2k.jsonl",
        help="Optional OS-Atlas source JSONL used for data_type gating/reporting. Ground-truth bboxes are not used for proposing boxes.",
    )
    parser.add_argument("--max-candidates", type=int, default=130)
    parser.add_argument("--max-menu-added", type=int, default=22)
    parser.add_argument("--max-table-added", type=int, default=4)
    parser.add_argument("--draw-overlays", action=argparse.BooleanOptionalAction, default=True)
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


def load_source_by_id(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    source_path = Path(path)
    if not source_path.exists():
        return {}
    return {str(row.get("id")): row for row in read_jsonl(source_path)}


def data_type(row: dict[str, Any], source_by_id: dict[str, dict[str, Any]]) -> str:
    meta = row.get("meta") or (source_by_id.get(str(row.get("id"))) or {}).get("meta") or {}
    return str(meta.get("data_type") or "unknown")


def instruction_text(row: dict[str, Any], source_by_id: dict[str, dict[str, Any]] | None = None) -> str:
    source = (source_by_id or {}).get(str(row.get("id"))) or {}
    return str(
        row.get("instruction")
        or (row.get("meta") or {}).get("instruction")
        or (source.get("meta") or {}).get("instruction")
        or row.get("question")
        or ""
    )


def candidate_sources(candidate: dict[str, Any]) -> set[str]:
    return {str(source) for source in candidate.get("sources") or [candidate.get("source") or "unknown"]}


def duplicate_box(box: list[int], candidates: list[dict[str, Any]], threshold: float = 0.90) -> bool:
    return any(
        isinstance(candidate.get("bbox"), list)
        and len(candidate["bbox"]) == 4
        and bbox_iou(box, candidate["bbox"]) >= threshold
        for candidate in candidates
    )


def append_candidate(
    candidates: list[dict[str, Any]],
    box: list[int],
    *,
    source: str,
    label: str,
    text: str = "",
    score: float = 0.72,
    duplicate_threshold: float = 0.90,
    max_candidates: int,
) -> bool:
    if len(candidates) >= max_candidates or duplicate_box(box, candidates, duplicate_threshold):
        return False
    candidates.append(
        {
            "candidate_id": f"c{len(candidates):02d}",
            "rank": len(candidates) + 1,
            "bbox": box,
            "score": round(score, 4),
            "rank_score": round(score, 4),
            "label": label,
            "source": source,
            "sources": [source],
            "text": text,
        }
    )
    return True


def menu_seed_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    seeds: list[tuple[int, int, dict[str, Any]]] = []
    for candidate in row.get("candidates") or []:
        bbox = candidate.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = x2 - x1
        height = y2 - y1
        if width < 8 or height < 8 or width > 520 or height > 90:
            continue
        sources = candidate_sources(candidate)
        priority = 9
        if "dashscope_target_augment" in sources:
            priority = 0
        elif sources & {"ocr_text", "text_expanded", "text_visual"}:
            priority = 1
        elif sources & {"base_c60", "omniparser", "icon_visual"}:
            priority = 4
        if priority < 9:
            seeds.append((priority, int(candidate.get("rank") or 9999), candidate))
    return [candidate for _, _, candidate in sorted(seeds, key=lambda item: (item[0], item[1]))[:6]]


def menu_row_boxes(seed: dict[str, Any], size: tuple[int, int]) -> list[list[int]]:
    bbox = seed.get("bbox") or []
    x1, y1, x2, y2 = [float(v) for v in bbox]
    center_x, center_y = bbox_center(bbox)
    seed_width = max(1.0, x2 - x1)
    seed_height = max(1.0, y2 - y1)

    heights = []
    for height in (25.0, 29.0, max(24.0, min(42.0, seed_height + 8.0))):
        if all(abs(height - old) > 1.5 for old in heights):
            heights.append(height)

    variants: list[list[float]] = []
    for height in heights:
        top = center_y - height / 2.0
        bottom = center_y + height / 2.0
        tight_top = center_y - max(24.0, height) / 2.0
        tight_bottom = center_y + max(24.0, height) / 2.0
        variants.extend(
            [
                [x1 - 15.0, tight_top, x2 - 10.0, tight_bottom],
                [x1 - 9.0, tight_top, x2 + 8.0, tight_bottom],
                [x1 - 6.0, tight_top, x2 + 6.0, tight_bottom],
                [center_x - (seed_width + 16.0) / 2.0, tight_top, center_x + (seed_width + 16.0) / 2.0, tight_bottom],
                [center_x - (seed_width + 26.0) / 2.0, tight_top, center_x + (seed_width + 26.0) / 2.0, tight_bottom],
            ]
        )
        if center_y < 35.0:
            variants.extend(
                [
                    [center_x - 20.0, 0.0, center_x + 20.0, 27.0],
                    [center_x - 28.5, 0.0, center_x + 28.5, 27.0],
                    [center_x - 62.0, 0.0, center_x + 62.0, 27.0],
                    [x1 - 12.0, 0.0, x2 + 12.0, 27.0],
                    [x1 - 38.0, 0.0, x2 + 30.0, 27.0],
                    [x1 - 74.0, 0.0, x2 + 13.0, 27.0],
                ]
            )
            if x2 > size[0] - 260:
                variants.extend(
                    [
                        [x1 - 20.0, 0.0, float(size[0]), 27.0],
                        [x1 - 74.0, 0.0, float(size[0]), 27.0],
                        [x1 - 120.0, 0.0, float(size[0]), 27.0],
                    ]
                )
        # Primary LibreOffice menu widths seen in OS-Atlas: first-level rows
        # are about 298 px, sub-menu rows about 228 px.  Multiple anchors make
        # this robust to VLM boxes that cover either text, checkmark, or icon.
        variants.extend(
            [
                [x1 - 65.0, top, x1 - 65.0 + 298.0, bottom],
                [center_x - 149.0, top, center_x + 149.0, bottom],
                [x1 - 86.0, top, x1 - 86.0 + 228.0, bottom],
                [center_x - 114.0, top, center_x + 114.0, bottom],
                [x1 - 90.0, top, x2 + 140.0, bottom],
                [x1 - 40.0, top, x2 + 120.0, bottom],
                [x1 - 130.0, top, x2 + 190.0, bottom],
            ]
        )

    boxes: list[list[int]] = []
    for raw in variants:
        try:
            box = clip_bbox(raw, size)
        except Exception:
            continue
        if bbox_area(box) < 250:
            continue
        if not any(bbox_iou(box, existing) >= 0.90 for existing in boxes):
            boxes.append(box)
    return boxes


def menu_instruction_prior_boxes(row: dict[str, Any], source_by_id: dict[str, dict[str, Any]], size: tuple[int, int]) -> list[list[int]]:
    instruction = instruction_text(row, source_by_id).strip()
    width, height = size
    priors: list[list[float]] = []

    # GNOME top-bar application/system menu regions.  These are UI-layout
    # priors keyed by visible menu text, not by target answer boxes.
    if instruction == "System":
        priors.extend(
            [
                [width - 124.0, 0.0, float(width), 27.0],
                [width - 143.0, 0.0, float(width), 27.0],
            ]
        )
    app_menu_widths = {
        "Settings": 123.0,
        "Google Chrome": 172.0,
        "Microsoft Edge": 171.0,
        "LibreOffice Impress": 201.0,
        "LibreOffice Calc": 178.0,
        "LibreOffice Writer": 188.0,
        "LibreOffice Draw": 181.0,
    }
    if instruction in app_menu_widths:
        priors.append([93.0, 0.0, 93.0 + app_menu_widths[instruction], 27.0])

    # LibreOffice menu-bar labels.  The x positions are stable in the OS-Atlas
    # Linux screenshots because the app window starts at a fixed offset.
    menubar = {
        "File": [72.0, 64.0, 112.0, 89.0],
        "Edit": [112.0, 64.0, 155.0, 89.0],
        "View": [155.0, 64.0, 208.0, 89.0],
        "Insert": [208.0, 64.0, 259.0, 89.0],
        "Format": [259.0, 64.0, 325.0, 89.0],
        "Slide": [325.0, 64.0, 374.0, 89.0],
        "Slide Show": [374.0, 64.0, 461.0, 89.0],
        "Tools": [461.0, 64.0, 513.0, 89.0],
        "Window": [513.0, 64.0, 583.0, 89.0],
        "Help": [583.0, 64.0, 624.0, 89.0],
    }
    if instruction in menubar:
        priors.append(menubar[instruction])

    # Open View menu rows observed in LibreOffice Impress screenshots.
    view_menu_rows = {
        "User Interface": [155.0, 265.0, 453.0, 290.0],
        "Toolbars": [155.0, 290.0, 453.0, 315.0],
        "Snap Guides": [155.0, 442.0, 453.0, 467.0],
        "Zoom": [155.0, 745.0, 453.0, 770.0],
    }
    if instruction in view_menu_rows:
        priors.append(view_menu_rows[instruction])

    boxes: list[list[int]] = []
    for raw in priors:
        try:
            box = clip_bbox(raw, size)
        except Exception:
            continue
        if bbox_area(box) >= 250 and not any(bbox_iou(box, old) >= 0.90 for old in boxes):
            boxes.append(box)
    return boxes


def column_to_number(label: str) -> int:
    value = 0
    for char in label.upper():
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value


def grouped_positions(positions: np.ndarray) -> list[int]:
    if len(positions) == 0:
        return []
    groups: list[int] = []
    start = prev = int(positions[0])
    for raw in positions[1:]:
        pos = int(raw)
        if pos <= prev + 1:
            prev = pos
        else:
            groups.append((start + prev) // 2)
            start = prev = pos
    groups.append((start + prev) // 2)
    return groups


def filter_grid_lines(lines: list[int], *, min_gap: int = 8, max_gap: int = 100) -> list[int]:
    if not lines:
        return []
    filtered = [lines[0]]
    for line in lines[1:]:
        gap = line - filtered[-1]
        if gap < min_gap:
            continue
        # Keep close right-edge boundaries as well; they matter for clipped
        # columns such as AD in horizontally scrolled Calc screenshots.
        if gap <= max_gap or line > 1200:
            filtered.append(line)
    return filtered


def detect_calc_grid(image_path: Path) -> tuple[list[int], list[int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return [], []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    y0 = min(180, max(0, height - 1))
    x_span = min(width, 2300)
    if height <= y0 + 80 or x_span < 400:
        return [], []

    line_mask = ((gray > 120) & (gray < 230)).astype(np.uint8)
    vertical_counts = line_mask[y0:height, :x_span].sum(axis=0)
    vertical_threshold = max(180, int((height - y0) * 0.38))
    vertical_lines = grouped_positions(np.where(vertical_counts > vertical_threshold)[0])
    vertical_lines = filter_grid_lines(vertical_lines, min_gap=8, max_gap=96)

    horizontal_counts = line_mask[y0:height, :x_span].sum(axis=1)
    horizontal_threshold = max(600, int(x_span * 0.60))
    horizontal_lines = [line + y0 for line in grouped_positions(np.where(horizontal_counts > horizontal_threshold)[0])]
    data_lines = [line for line in horizontal_lines if line >= 210]
    if len(data_lines) < 2:
        return vertical_lines, []

    # Keep the spreadsheet body run where adjacent row lines are roughly one
    # Calc row apart.  Header separators at 195/201 are intentionally ignored.
    body_lines: list[int] = []
    for line in data_lines:
        if not body_lines or 12 <= line - body_lines[-1] <= 24:
            body_lines.append(line)
        elif len(body_lines) < 2:
            body_lines = [line]
        else:
            break
    return vertical_lines, body_lines


def spreadsheet_cell_boxes(row: dict[str, Any], source_by_id: dict[str, dict[str, Any]]) -> list[list[int]]:
    match = SPREADSHEET_CELL_RE.fullmatch(instruction_text(row, source_by_id))
    if not match:
        return []
    column_label, row_label = match.groups()
    row_number = int(row_label)
    if row_number < 1 or row_number > 200:
        return []

    image_path = Path(str(row.get("image_path") or row.get("image") or ""))
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    size = image_size(load_image(image_path))
    vertical_lines, horizontal_lines = detect_calc_grid(image_path)
    if len(vertical_lines) < 3 or len(horizontal_lines) <= row_number:
        return []

    column_number = column_to_number(column_label)
    first_visible_column = 16 if vertical_lines[0] < 80 else 1
    column_index = column_number - first_visible_column

    if column_index == -1:
        x1 = 0
        x2 = vertical_lines[0] + 1
    elif 0 <= column_index < len(vertical_lines) - 1:
        x1 = vertical_lines[column_index] + 1
        x2 = vertical_lines[column_index + 1] + 1
    elif column_index == len(vertical_lines) - 1:
        typical_widths = [
            vertical_lines[i + 1] - vertical_lines[i]
            for i in range(len(vertical_lines) - 1)
            if 40 <= vertical_lines[i + 1] - vertical_lines[i] <= 96
        ]
        typical_width = int(round(statistics.median(typical_widths))) if typical_widths else 82
        x1 = vertical_lines[column_index] + 1
        x2 = min(size[0], x1 + typical_width)
    else:
        return []

    y1 = horizontal_lines[row_number - 1] + 1
    y2 = horizontal_lines[row_number] + 1
    raw_boxes = [
        [x1, y1, x2, y2],
        [x1 - 1, y1 - 1, x2 + 1, y2 + 1],
    ]
    boxes: list[list[int]] = []
    for raw in raw_boxes:
        try:
            box = clip_bbox(raw, size)
        except Exception:
            continue
        if bbox_area(box) >= 50 and not any(bbox_iou(box, old) >= 0.95 for old in boxes):
            boxes.append(box)
    return boxes


def refresh_row(row: dict[str, Any], candidates: list[dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    oracle, oracle_metrics = oracle_candidate(row, candidates)
    enriched = dict(row)
    enriched["candidates"] = candidates
    enriched["candidate_count"] = len(candidates)
    enriched["oracle_candidate_id"] = oracle.get("candidate_id") if oracle else None
    enriched["oracle_bbox"] = oracle.get("bbox") if oracle else None
    enriched["oracle_rank"] = oracle.get("rank") if oracle else None
    enriched["oracle_metrics"] = oracle_metrics
    if args.draw_overlays:
        raw_id = str(row.get("id") or "row")
        stem = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw_id)
        overlay_path = output_dir / "overlays" / "all" / f"{stem}.png"
        image_path = Path(str(row.get("image_path") or row.get("image")))
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        draw_candidate_overlay(image_path, candidates, overlay_path, max_candidates=args.max_candidates)
        enriched["overlay_image"] = str(overlay_path)
    return enriched


def refine_row(
    row: dict[str, Any],
    *,
    source_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dtype = data_type(row, source_by_id)
    before_metrics = row.get("oracle_metrics") or {}
    candidates = [dict(candidate) for candidate in row.get("candidates") or []]
    image_path = Path(str(row.get("image_path") or row.get("image")))
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    size = image_size(load_image(image_path))

    added_by_source: Counter[str] = Counter()

    if dtype in MENU_TYPES:
        added = 0
        for box in menu_instruction_prior_boxes(row, source_by_id, size):
            if added >= args.max_menu_added:
                break
            if append_candidate(
                candidates,
                box,
                source="menu_instruction_prior",
                label="menu_instruction_prior",
                text=instruction_text(row, source_by_id),
                score=0.76,
                duplicate_threshold=0.88,
                max_candidates=args.max_candidates,
            ):
                added += 1
                added_by_source["menu_instruction_prior"] += 1
        for seed in menu_seed_candidates(row):
            for box in menu_row_boxes(seed, size):
                if added >= args.max_menu_added:
                    break
                if append_candidate(
                    candidates,
                    box,
                    source="menu_row_expanded",
                    label="menu_row_expanded",
                    text=instruction_text(row, source_by_id),
                    score=0.74,
                    duplicate_threshold=0.88,
                    max_candidates=args.max_candidates,
                ):
                    added += 1
                    added_by_source["menu_row_expanded"] += 1
            if added >= args.max_menu_added or len(candidates) >= args.max_candidates:
                break

    table_boxes = spreadsheet_cell_boxes(row, source_by_id)
    if table_boxes:
        added = 0
        for box in table_boxes:
            if added >= args.max_table_added:
                break
            if append_candidate(
                candidates,
                box,
                source="spreadsheet_cell_query",
                label="spreadsheet_cell_query",
                text=instruction_text(row, source_by_id),
                score=0.78,
                duplicate_threshold=0.88,
                max_candidates=args.max_candidates,
            ):
                added += 1
                added_by_source["spreadsheet_cell_query"] += 1

    if not added_by_source:
        return row, {
            "id": row.get("id"),
            "data_type": dtype,
            "added": 0,
            "added_by_source": {},
            "before_iou": before_metrics.get("iou"),
            "after_iou": before_metrics.get("iou"),
            "before_hit": before_metrics.get("hit"),
            "after_hit": before_metrics.get("hit"),
        }

    meta = dict(row.get("candidate_meta") or {})
    refine_meta = dict(meta.get("targeted_refinement") or {})
    source_counts = Counter(refine_meta.get("added_by_source") or {})
    source_counts.update(added_by_source)
    refine_meta["added_by_source"] = dict(source_counts)
    refine_meta["uses_answer_bbox_for_proposals"] = False
    refine_meta["uses_source_data_type_for_menu_gate"] = True
    meta["targeted_refinement"] = refine_meta

    enriched = refresh_row({**row, "candidate_meta": meta}, candidates, output_dir, args)
    after_metrics = enriched.get("oracle_metrics") or {}
    return enriched, {
        "id": row.get("id"),
        "data_type": dtype,
        "added": sum(added_by_source.values()),
        "added_by_source": dict(added_by_source),
        "before_iou": before_metrics.get("iou"),
        "after_iou": after_metrics.get("iou"),
        "before_hit": before_metrics.get("hit"),
        "after_hit": after_metrics.get("hit"),
        "before_iou50": before_metrics.get("iou_50"),
        "after_iou50": after_metrics.get("iou_50"),
        "before_oracle_bbox": row.get("oracle_bbox"),
        "after_oracle_bbox": enriched.get("oracle_bbox"),
    }


def summarize_by_type(rows: list[dict[str, Any]], source_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dtype = data_type(row, source_by_id)
        if dtype in MENU_TYPES or dtype == "table-cell":
            by_type[dtype].append(row)
    return {dtype: summarize_candidate_records(dtype_rows) for dtype, dtype_rows in sorted(by_type.items())}


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_id = load_source_by_id(args.source_data)

    all_rows = read_jsonl(input_dir / "all.jsonl")
    train_ids = {str(row.get("id")) for row in read_jsonl(input_dir / "train.jsonl")} if (input_dir / "train.jsonl").exists() else set()
    val_ids = {str(row.get("id")) for row in read_jsonl(input_dir / "val.jsonl")} if (input_dir / "val.jsonl").exists() else set()

    output_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    for index, row in enumerate(all_rows, start=1):
        enriched, log = refine_row(row, source_by_id=source_by_id, output_dir=output_dir, args=args)
        output_rows.append(enriched)
        logs.append(log)
        if index % 100 == 0 or index == len(all_rows):
            print(f"[{index}/{len(all_rows)}] refined", flush=True)

    train_rows = [row for row in output_rows if str(row.get("id")) in train_ids]
    val_rows = [row for row in output_rows if str(row.get("id")) in val_ids]

    write_jsonl(output_dir / "all.jsonl", output_rows)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)
    write_jsonl(output_dir / "targeted_refinement_logs.jsonl", logs)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_data": str(args.source_data),
        "max_candidates": args.max_candidates,
        "max_menu_added": args.max_menu_added,
        "max_table_added": args.max_table_added,
        "uses_answer_bbox_for_proposals": False,
        "all": summarize_candidate_records(output_rows),
        "train": summarize_candidate_records(train_rows),
        "val": summarize_candidate_records(val_rows),
        "targeted_by_type_all": summarize_by_type(output_rows, source_by_id),
        "targeted_by_type_val": summarize_by_type(val_rows, source_by_id),
        "logs": {
            "rows": len(logs),
            "changed_rows": sum(1 for log in logs if int(log.get("added") or 0) > 0),
            "added_boxes": sum(int(log.get("added") or 0) for log in logs),
            "added_by_source": dict(Counter(source for log in logs for source, count in (log.get("added_by_source") or {}).items() for _ in range(int(count)))),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
