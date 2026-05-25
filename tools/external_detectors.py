"""Optional external GUI detector adapters.

External detectors are deliberately optional: this module must import without
OmniParser weights or third-party runtime dependencies. The main detector can
fuse these candidates only when explicitly requested.
"""

from __future__ import annotations

import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .common import bbox_area, bbox_center, clip_bbox, image_size, load_image


DEFAULT_OMNIPARSER_ROOT = Path("third_party/OmniParser")
DEFAULT_OMNIPARSER_WEIGHTS = DEFAULT_OMNIPARSER_ROOT / "weights"


def _is_ratio_bbox(values: list[float]) -> bool:
    return all(-0.05 <= v <= 1.05 for v in values)


def _to_xyxy(value: Any, size: tuple[int, int], fmt: str | None = None) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    vals = [float(v) for v in value[:4]]
    width, height = size
    if _is_ratio_bbox(vals):
        vals = [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    if fmt == "xywh":
        vals = [vals[0], vals[1], vals[0] + vals[2], vals[1] + vals[3]]
    try:
        return clip_bbox(vals, size)
    except Exception:
        return None


def normalize_external_detection(item: dict[str, Any], size: tuple[int, int], provider: str, idx: int) -> dict[str, Any] | None:
    bbox = _to_xyxy(item.get("bbox") or item.get("box") or item.get("xyxy"), size, item.get("bbox_format"))
    if bbox is None:
        return None
    text = item.get("content") or item.get("text") or item.get("caption") or item.get("label") or ""
    kind = item.get("type") or item.get("kind") or "element"
    score = item.get("score")
    if not isinstance(score, (int, float)):
        score = 0.72 if kind == "icon" else 0.66
    det = {
        "bbox": bbox,
        "center": [round(v, 3) for v in bbox_center(bbox)],
        "score": round(float(score), 4),
        "label": f"{provider}_{kind}",
        "text": str(text) if text is not None else "",
        "source": provider,
        "sources": [provider],
        "external_provider": provider,
        "external_id": item.get("id") or f"{provider}_{idx:03d}",
        "area": bbox_area(bbox),
    }
    if "interactivity" in item:
        det["interactivity"] = bool(item.get("interactivity"))
    if item.get("source"):
        det["external_source"] = item.get("source")
    return det


def _read_cache_file(image_path: Path, cache_dir: Path, provider: str) -> dict[str, Any] | None:
    candidates = [
        cache_dir / provider / f"{image_path.stem}.json",
        cache_dir / provider / f"{image_path.name}.json",
        cache_dir / f"{provider}_{image_path.stem}.json",
        cache_dir / f"{image_path.stem}.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def run_cache_provider(image_path: str | Path, provider: str, cache_dir: str | Path, max_results: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    size = image_size(load_image(image_path))
    started = time.time()
    payload = _read_cache_file(image_path, cache_dir, provider)
    if payload is None:
        return [], {"provider": provider, "available": False, "error": f"cache miss under {cache_dir}", "latency_sec": 0.0}
    raw = payload.get("detections") or payload.get("parsed_content_list") or payload.get("items") or []
    detections = []
    for idx, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            det = normalize_external_detection(item, size, provider, idx)
            if det:
                detections.append(det)
    return detections[:max_results], {
        "provider": provider,
        "available": True,
        "raw_count": len(raw),
        "count": len(detections[:max_results]),
        "latency_sec": round(time.time() - started, 4),
        "cache_dir": str(cache_dir),
    }


@functools.lru_cache(maxsize=2)
def _load_omniparser_yolo_model(model_path: str):
    from ultralytics import YOLO  # type: ignore

    return YOLO(model_path)


@functools.lru_cache(maxsize=2)
def _load_omniparser_models(root: str, weights_dir: str, use_caption: bool, box_threshold: float):
    root_path = Path(root).resolve()
    weights_path = Path(weights_dir).resolve()
    if not root_path.exists():
        raise RuntimeError(f"OmniParser root not found: {root_path}")
    icon_model = weights_path / "icon_detect" / "model.pt"
    if not icon_model.exists():
        raise RuntimeError(f"OmniParser icon model not found: {icon_model}")
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    from util.utils import get_caption_model_processor, get_yolo_model  # type: ignore

    yolo_model = get_yolo_model(model_path=str(icon_model))
    caption_model_processor = None
    if use_caption:
        caption_dir = weights_path / "icon_caption_florence"
        if not caption_dir.exists():
            raise RuntimeError(f"OmniParser caption model not found: {caption_dir}")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=str(caption_dir),
        )
    return yolo_model, caption_model_processor


def run_omniparser(
    image_path: str | Path,
    *,
    max_results: int = 120,
    root: str | Path | None = None,
    weights_dir: str | Path | None = None,
    use_caption: bool = False,
    use_paddleocr: bool = False,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    imgsz: int = 640,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run OmniParser if local source and weights are present.

    The returned detections use the same bbox schema as `tools.detect`.
    Missing weights/dependencies return an unavailable meta record instead of
    raising, so A/B scripts can run before installation is complete.
    """
    started = time.time()
    image_path = Path(image_path)
    root_path = Path(root) if root is not None else DEFAULT_OMNIPARSER_ROOT
    weights_path = Path(weights_dir) if weights_dir is not None else DEFAULT_OMNIPARSER_WEIGHTS
    try:
        if not root_path.exists():
            raise RuntimeError(f"OmniParser root not found: {root_path}")
        icon_model = weights_path / "icon_detect" / "model.pt"
        if not icon_model.exists():
            raise RuntimeError(f"OmniParser icon model not found: {icon_model}")
        if omniparser_use_caption := bool(use_caption):
            caption_dir = weights_path / "icon_caption_florence"
            if not caption_dir.exists():
                raise RuntimeError(f"OmniParser caption model not found: {caption_dir}")
        if not omniparser_use_caption:
            image = Image.open(image_path).convert("RGB")
            size = image_size(image)
            yolo_model = _load_omniparser_yolo_model(str(icon_model))
            results = yolo_model.predict(
                str(image_path),
                conf=float(box_threshold),
                iou=float(iou_threshold),
                imgsz=int(imgsz),
                verbose=False,
            )
            detections = []
            boxes = results[0].boxes if results else []
            for idx, box in enumerate(boxes, start=1):
                xyxy = box.xyxy[0].detach().cpu().tolist()
                bbox = clip_bbox([float(v) for v in xyxy], size)
                score = float(box.conf[0].detach().cpu()) if box.conf is not None else 0.0
                detections.append(
                    {
                        "bbox": bbox,
                        "center": [round(v, 3) for v in bbox_center(bbox)],
                        "score": round(score, 4),
                        "label": "omniparser_icon",
                        "text": "",
                        "source": "omniparser",
                        "sources": ["omniparser"],
                        "external_provider": "omniparser",
                        "external_id": f"omniparser_{idx:03d}",
                        "area": bbox_area(bbox),
                    }
                )
            detections = sorted(detections, key=lambda d: (-float(d.get("score", 0.0)), d["bbox"][1], d["bbox"][0]))
            return detections[:max_results], {
                "provider": "omniparser",
                "available": True,
                "root": str(root_path),
                "weights_dir": str(weights_path),
                "use_caption": False,
                "raw_count": len(boxes),
                "count": len(detections[:max_results]),
                "latency_sec": round(time.time() - started, 4),
            }
        if str(root_path.resolve()) not in sys.path:
            sys.path.insert(0, str(root_path.resolve()))
        from util.utils import check_ocr_box, get_som_labeled_img  # type: ignore

        yolo_model, caption_model_processor = _load_omniparser_models(
            str(root_path),
            str(weights_path),
            bool(use_caption),
            float(box_threshold),
        )
        image = Image.open(image_path).convert("RGB")
        ocr_result, _ = check_ocr_box(
            image,
            display_img=False,
            output_bb_format="xyxy",
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=use_paddleocr,
        )
        ocr_text, ocr_bbox = ocr_result
        _, _, parsed = get_som_labeled_img(
            image,
            yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            caption_model_processor=caption_model_processor,
            ocr_text=ocr_text,
            use_local_semantics=bool(use_caption),
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )
        size = image_size(image)
        detections = []
        for idx, item in enumerate(parsed, start=1):
            if isinstance(item, dict):
                det = normalize_external_detection(item, size, "omniparser", idx)
                if det:
                    detections.append(det)
        detections = sorted(detections, key=lambda d: (-float(d.get("score", 0.0)), d["bbox"][1], d["bbox"][0]))
        return detections[:max_results], {
            "provider": "omniparser",
            "available": True,
            "root": str(root_path),
            "weights_dir": str(weights_path),
            "use_caption": bool(use_caption),
            "raw_count": len(parsed),
            "count": len(detections[:max_results]),
            "latency_sec": round(time.time() - started, 4),
        }
    except Exception as exc:
        return [], {
            "provider": "omniparser",
            "available": False,
            "root": str(root_path),
            "weights_dir": str(weights_path),
            "error": f"{type(exc).__name__}: {exc}",
            "latency_sec": round(time.time() - started, 4),
        }


def run_external_detectors(
    image_path: str | Path,
    *,
    providers: list[str] | tuple[str, ...],
    max_results: int = 120,
    cache_dir: str | Path | None = None,
    omniparser_root: str | Path | None = None,
    omniparser_weights_dir: str | Path | None = None,
    omniparser_use_caption: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"providers": {}}
    for provider in providers:
        provider = provider.strip().lower()
        if not provider:
            continue
        if provider == "omniparser":
            items, item_meta = run_omniparser(
                image_path,
                max_results=max_results,
                root=omniparser_root,
                weights_dir=omniparser_weights_dir,
                use_caption=omniparser_use_caption,
            )
        elif provider.startswith("cache:"):
            if cache_dir is None:
                items, item_meta = [], {"provider": provider, "available": False, "error": "cache_dir is required"}
            else:
                cache_provider = provider.split(":", 1)[1]
                items, item_meta = run_cache_provider(image_path, cache_provider, cache_dir, max_results)
        else:
            items, item_meta = [], {"provider": provider, "available": False, "error": "unknown external provider"}
        detections.extend(items)
        meta["providers"][provider] = item_meta
    return detections[:max_results], meta
