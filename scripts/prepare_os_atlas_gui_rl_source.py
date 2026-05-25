#!/usr/bin/env python3
"""Convert OS-Atlas desktop grounding data into EviTool GUI grounding rows.

The candidate-RL builder in this repository expects one JSONL row per target
element with:

* a local image path relative to `--image-root`;
* a natural-language instruction;
* an absolute pixel-space ground-truth bbox named `answer_bbox`.

OS-Atlas desktop JSON stores one row per screenshot chunk, with several UI
elements per row and normalized bboxes in [0, 1].  This script samples target
elements, extracts only the screenshots needed for those elements from the
downloaded image zip, and writes the repository-native source JSONL.
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class Target:
    """A single clickable/locatable UI target from OS-Atlas."""

    source_index: int
    element_index: int
    image_name: str
    instruction: str
    data_type: str
    bbox_norm: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="OS-Atlas *_splited.json file.")
    parser.add_argument("--image-zip", default=None, help="Local zip file that contains the referenced screenshots.")
    parser.add_argument(
        "--image-zip-url",
        default=None,
        help=(
            "Remote zip URL.  When set, the script reads only the selected "
            "screenshots with HTTP Range requests instead of downloading the "
            "full multi-GB archive first."
        ),
    )
    parser.add_argument("--output-root", required=True, help="Directory for extracted images and JSONL output.")
    parser.add_argument("--output-jsonl", default="source_os_atlas_linux_2k.jsonl")
    parser.add_argument("--report-json", default="source_os_atlas_linux_2k.summary.json")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling-mode",
        choices=("element-random", "image-random"),
        default="element-random",
        help=(
            "`element-random` maximizes screenshot diversity but can require "
            "many unique images. `image-random` first samples screenshots and "
            "then takes up to --max-targets-per-image elements per screenshot, "
            "which is more practical for a fast candidate-RL run."
        ),
    )
    parser.add_argument(
        "--min-bbox-area",
        type=float,
        default=16.0,
        help="Drop tiny targets after converting normalized boxes to pixels.",
    )
    parser.add_argument(
        "--max-targets-per-image",
        type=int,
        default=8,
        help="Caps repeated targets from one screenshot so the 2k set is not dominated by a few dense pages.",
    )
    return parser.parse_args()


def read_os_atlas_targets(path: Path) -> list[Target]:
    """Flatten OS-Atlas screenshot chunks into target-level records."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets: list[Target] = []
    for source_index, row in enumerate(payload):
        image_name = str(row.get("img_filename") or "")
        if not image_name:
            continue
        for element_index, element in enumerate(row.get("elements") or []):
            instruction = str(element.get("instruction") or "").strip()
            bbox = element.get("bbox")
            if not instruction or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                bbox_norm = [float(v) for v in bbox]
            except Exception:
                continue
            if any(v < -0.01 or v > 1.01 for v in bbox_norm):
                continue
            targets.append(
                Target(
                    source_index=source_index,
                    element_index=element_index,
                    image_name=image_name,
                    instruction=instruction,
                    data_type=str(element.get("data_type") or ""),
                    bbox_norm=bbox_norm,
                )
            )
    return targets


def select_targets(
    targets: list[Target],
    *,
    limit: int,
    seed: int,
    max_targets_per_image: int,
    sampling_mode: str,
) -> list[Target]:
    """Sample targets while preventing one screenshot from dominating the set."""
    rng = random.Random(seed)
    if sampling_mode == "image-random":
        # Candidate generation is screenshot-bound: all targets on the same
        # screenshot reuse the same OmniParser candidate set.  This mode keeps
        # the row count high while limiting the number of unique screenshots,
        # which makes quick GRPO experiments feasible.
        by_image: dict[str, list[Target]] = {}
        for target in targets:
            by_image.setdefault(target.image_name, []).append(target)
        image_names = list(by_image)
        rng.shuffle(image_names)
        selected: list[Target] = []
        for image_name in image_names:
            image_targets = list(by_image[image_name])
            rng.shuffle(image_targets)
            selected.extend(image_targets[:max_targets_per_image])
            if len(selected) >= limit:
                return selected[:limit]
        return selected

    shuffled = list(targets)
    rng.shuffle(shuffled)
    selected: list[Target] = []
    per_image: Counter[str] = Counter()
    for target in shuffled:
        if per_image[target.image_name] >= max_targets_per_image:
            continue
        selected.append(target)
        per_image[target.image_name] += 1
        if len(selected) >= limit:
            break
    return selected


def open_image_archive(zip_path: Path | None, zip_url: str | None) -> Any:
    """Open either a local zip or a remote zip URL as a zip-like object.

    The OS-Atlas Linux image archive is about 3 GB.  For a 2k RL subset we only
    need the screenshots referenced by the sampled targets, so remote zip access
    is much faster in this environment than waiting for the complete archive.
    """
    if zip_url:
        try:
            from remotezip import RemoteZip
        except ImportError as exc:
            raise RuntimeError("Install `remotezip` or pass a complete local --image-zip.") from exc
        return RemoteZip(zip_url)
    if zip_path is None:
        raise ValueError("Either --image-zip or --image-zip-url is required.")
    return zipfile.ZipFile(zip_path)


def build_zip_index(zip_archive: Any) -> dict[str, str]:
    """Map image basenames to their member paths inside the zip archive."""
    index: dict[str, str] = {}
    for name in zip_archive.namelist():
        if name.endswith("/"):
            continue
        basename = Path(name).name
        if basename and basename not in index:
            index[basename] = name
    return index


def extract_image(zip_archive: Any, member_name: str, output_path: Path) -> tuple[int, int]:
    """Extract one image from the zip and return its pixel size."""
    if output_path.exists():
        with Image.open(output_path) as image:
            return image.size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = zip_archive.read(member_name)
    with Image.open(BytesIO(raw)) as image:
        rgb = image.convert("RGB")
        rgb.save(output_path)
        return rgb.size


def norm_bbox_to_pixels(bbox: list[float], size: tuple[int, int]) -> list[int]:
    """Convert OS-Atlas normalized xyxy bbox to clipped integer pixel xyxy."""
    width, height = size
    x1, y1, x2, y2 = bbox
    vals = [
        max(0.0, min(float(width), x1 * width)),
        max(0.0, min(float(height), y1 * height)),
        max(0.0, min(float(width), x2 * width)),
        max(0.0, min(float(height), y2 * height)),
    ]
    left, top, right, bottom = vals
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]


def bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    json_path = Path(args.json)
    zip_path = Path(args.image_zip) if args.image_zip else None
    output_root = Path(args.output_root)
    image_dir = output_root / "images" / "os_atlas_linux"

    targets = read_os_atlas_targets(json_path)
    # Select a buffer larger than the requested output because some elements can
    # be unusable after pixel conversion: missing zip members, degenerate boxes,
    # or very tiny targets that would make the reward signal noisy.
    selected = select_targets(
        targets,
        limit=min(len(targets), args.limit * 3),
        seed=args.seed,
        max_targets_per_image=args.max_targets_per_image,
        sampling_mode=args.sampling_mode,
    )

    rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    tiny_boxes = 0
    image_size_cache: dict[str, tuple[int, int]] = {}
    with open_image_archive(zip_path, args.image_zip_url) as zip_archive:
        zip_index = build_zip_index(zip_archive)
        for target in selected:
            member_name = zip_index.get(Path(target.image_name).name)
            if not member_name:
                missing_images.append(target.image_name)
                continue
            relative_image = image_dir.relative_to(output_root) / Path(target.image_name).name
            image_path = output_root / relative_image
            size = image_size_cache.get(target.image_name)
            if size is None:
                size = extract_image(zip_archive, member_name, image_path)
                image_size_cache[target.image_name] = size
            bbox = norm_bbox_to_pixels(target.bbox_norm, size)
            if bbox_area(bbox) < args.min_bbox_area:
                tiny_boxes += 1
                continue
            center = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
            row_id = f"os_atlas_linux_{len(rows):06d}"
            rows.append(
                {
                    "id": row_id,
                    "source": "OS-Copilot/OS-Atlas-data",
                    "source_index": target.source_index,
                    "task_type": "gui_grounding",
                    "question": f"Locate the UI element for this instruction: {target.instruction}.",
                    "answer": center,
                    "answer_bbox": bbox,
                    "meta": {
                        "instruction": target.instruction,
                        "data_type": target.data_type,
                        "bbox_norm_xyxy": target.bbox_norm,
                        "img_filename": target.image_name,
                        "image_size": list(size),
                        "domain": "desktop_linux",
                        "element_index": target.element_index,
                    },
                    "image": str(relative_image),
                }
            )
            if len(rows) >= args.limit:
                break

    if len(rows) < args.limit:
        raise RuntimeError(
            f"Prepared {len(rows)} usable rows, below requested {args.limit}. "
            f"Missing images={len(missing_images)}, tiny boxes={tiny_boxes}."
        )

    rows = rows[: args.limit]
    output_jsonl = output_root / args.output_jsonl
    report_json = output_root / args.report_json
    write_jsonl(output_jsonl, rows)
    summary = {
        "source_json": str(json_path),
        "source_image_zip": str(zip_path) if zip_path else None,
        "source_image_zip_url": args.image_zip_url,
        "output_jsonl": str(output_jsonl),
        "output_root": str(output_root),
        "requested_limit": args.limit,
        "sampling_mode": args.sampling_mode,
        "max_targets_per_image": args.max_targets_per_image,
        "rows": len(rows),
        "unique_images": len({row["image"] for row in rows}),
        "missing_images": len(missing_images),
        "tiny_boxes": tiny_boxes,
        "data_type_counts": Counter((row.get("meta") or {}).get("data_type") for row in rows),
    }
    report_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
