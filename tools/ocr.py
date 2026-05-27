"""OCR tool with optional backends and a safe fallback."""

from __future__ import annotations

import shutil
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .common import crop_pil, image_size, load_image, pil_to_cv, sort_boxes_reading_order


_EASYOCR_READERS: dict[tuple[tuple[str, ...], bool, str], Any] = {}


def _candidate_text_regions(image_bgr: np.ndarray, offset: tuple[int, int] = (0, 0), max_regions: int = 80) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Text-like regions tend to have high local contrast and horizontal strokes.
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    ox, oy = offset
    h, w = gray.shape[:2]
    for contour in contours:
        x, y, bw_, bh = cv2.boundingRect(contour)
        area = bw_ * bh
        if area < 20 or bw_ < 4 or bh < 4:
            continue
        if area > 0.8 * w * h:
            continue
        boxes.append({"text": "", "bbox": [x + ox, y + oy, x + bw_ + ox, y + bh + oy], "score": 0.0, "source": "candidate_region"})
    boxes = sort_boxes_reading_order(boxes)
    return boxes[:max_regions]


def _run_tesseract(image, offset: tuple[int, int]) -> tuple[list[dict[str, Any]], str]:
    import pytesseract

    if shutil.which("tesseract") is None:
        raise RuntimeError("tesseract binary not found")
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    spans = []
    ox, oy = offset
    for i, text in enumerate(data.get("text", [])):
        text = str(text).strip()
        if not text:
            continue
        conf = float(data.get("conf", [0])[i])
        if conf < 0:
            conf = 0.0
        x = int(data["left"][i]) + ox
        y = int(data["top"][i]) + oy
        w = int(data["width"][i])
        h = int(data["height"][i])
        spans.append({"text": text, "bbox": [x, y, x + w, y + h], "score": conf / 100.0, "source": "tesseract"})
    return spans, "tesseract"


def _run_easyocr(image_bgr: np.ndarray, offset: tuple[int, int], languages: list[str]) -> tuple[list[dict[str, Any]], str]:
    import easyocr
    import torch

    model_dir = os.environ.get("EASYOCR_MODULE_PATH", "/root/models/easyocr")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    key = (tuple(languages), gpu, str(Path(model_dir).resolve()))
    reader = _EASYOCR_READERS.get(key)
    if reader is None:
        reader = easyocr.Reader(languages, gpu=gpu, model_storage_directory=model_dir)
        _EASYOCR_READERS[key] = reader
    result = reader.readtext(image_bgr)
    ox, oy = offset
    spans = []
    for polygon, text, score in result:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        spans.append({
            "text": str(text),
            "bbox": [int(min(xs)) + ox, int(min(ys)) + oy, int(max(xs)) + ox, int(max(ys)) + oy],
            "score": float(score),
            "source": "easyocr",
        })
    return spans, "easyocr"


def _poly_to_bbox(poly: Any, offset: tuple[int, int]) -> list[int]:
    arr = np.asarray(poly)
    if arr.ndim == 1 and arr.size >= 4:
        xs = [arr[0], arr[2]]
        ys = [arr[1], arr[3]]
    else:
        xs = arr[:, 0]
        ys = arr[:, 1]
    ox, oy = offset
    return [int(np.min(xs)) + ox, int(np.min(ys)) + oy, int(np.max(xs)) + ox, int(np.max(ys)) + oy]


def _run_paddleocr(image_bgr: np.ndarray, offset: tuple[int, int], languages: list[str]) -> tuple[list[dict[str, Any]], str]:
    # PaddleOCR 3.x downloads official OCR models on first use. Keep the cache
    # under /root/models and prefer ModelScope for mainland China networks.
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/root/models/paddleocr")
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "modelscope")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("FLAGS_use_mkldnn", "false")

    from paddleocr import PaddleOCR

    lang = "ch" if any(lang.startswith("ch") or lang.startswith("zh") for lang in languages) else "en"
    ocr = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    spans: list[dict[str, Any]] = []

    if hasattr(ocr, "predict"):
        results = ocr.predict(image_bgr)
        for result in results or []:
            if not isinstance(result, dict):
                try:
                    result = dict(result)
                except Exception:
                    result = getattr(result, "json", {}) or {}
            texts = result.get("rec_texts") or []
            scores = result.get("rec_scores") or []
            polys = result.get("rec_polys") or result.get("dt_polys") or result.get("rec_boxes") or []
            for idx, text in enumerate(texts):
                if not str(text).strip():
                    continue
                poly = polys[idx] if idx < len(polys) else [0, 0, 0, 0]
                score = float(scores[idx]) if idx < len(scores) else 0.0
                spans.append({
                    "text": str(text),
                    "bbox": _poly_to_bbox(poly, offset),
                    "score": score,
                    "source": "paddleocr",
                })
        return spans, "paddleocr"

    # Backward-compatible PaddleOCR 2.x path.
    result = ocr.ocr(image_bgr, cls=True)
    for line in result or []:
        for item in line or []:
            polygon, text_score = item
            text, score = text_score
            spans.append({
                "text": str(text),
                "bbox": _poly_to_bbox(polygon, offset),
                "score": float(score),
                "source": "paddleocr",
            })
    return spans, "paddleocr"


def run(
    image_path: str | Path,
    bbox: list[float] | None = None,
    engine: str = "auto",
    languages: list[str] | None = None,
    max_regions: int = 80,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[int] | None]:
    languages = languages or ["en"]
    image = load_image(image_path)
    offset = (0, 0)
    clipped = None
    if bbox is not None:
        image, clipped = crop_pil(image, bbox)
        offset = (clipped[0], clipped[1])
    image_bgr = pil_to_cv(image)

    engines = [engine] if engine != "auto" else ["easyocr", "paddleocr", "tesseract"]
    errors = []
    spans = []
    used_engine = None
    for candidate in engines:
        try:
            if candidate == "tesseract":
                spans, used_engine = _run_tesseract(image, offset)
            elif candidate == "easyocr":
                spans, used_engine = _run_easyocr(image_bgr, offset, languages)
            elif candidate == "paddleocr":
                spans, used_engine = _run_paddleocr(image_bgr, offset, languages)
            else:
                raise ValueError(f"unknown OCR engine: {candidate}")
            break
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")

    available = used_engine is not None
    candidate_regions = []
    if not available:
        candidate_regions = _candidate_text_regions(image_bgr, offset=offset, max_regions=max_regions)

    content = {
        "engine": used_engine,
        "available": available,
        "languages": languages,
        "spans": sort_boxes_reading_order(spans),
        "candidate_regions": candidate_regions,
        "image_size": list(image_size(image)),
        "errors": errors,
    }
    if not available:
        content["message"] = "No OCR backend is installed. Install paddleocr, easyocr, or pytesseract+tesseract to read text. Candidate regions are visual boxes only."
    return content, {}, clipped
