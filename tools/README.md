# Visual Tools

Local visual tools share a JSON action interface and JSON evidence output.

Example:

```bash
python -m tools.runner \
  --image data/eval_mini/images/chartqa/0000.png \
  --action '{"tool":"crop","args":{"bbox":[80,80,760,520]}}' \
  --pretty
```

Supported tools:

| Tool | Purpose |
|---|---|
| `inspect` | Return image metadata and basic statistics. |
| `crop` | Crop a region and optionally save it. |
| `zoom` | Crop then resize a region. |
| `ocr` | Read text with optional PaddleOCR/EasyOCR/Tesseract backends; falls back to text-like candidate boxes. |
| `detect` | Detect layout, text, fused UI candidates, chart bars, or color regions with OpenCV heuristics; `mode="ui"` can optionally fuse OCR boxes with `include_ocr=true`. |
| `measure` | Measure bbox sizes, centers, relative positions, distances, and bar heights. |
| `mark` / `visualize` | Draw boxes and points on an image. |
| `trace` | Execute a list of tool actions and return evidence observations. |
| `click` | Virtual GUI select/click target recorder; returns selected point/bbox as evidence. |

OCR backend status:

- `easyocr` is installed and used first in `engine=auto` mode.
- EasyOCR weights are cached under `/root/models/easyocr`.
- `paddleocr` and `paddlepaddle` are installed. PaddleOCR weights are cached under `/root/models/paddleocr`.
- PaddleOCR is higher quality on some document/chart OCR, but CPU inference is much slower on large images. Use `engine="paddleocr"` when quality matters more than speed.
- `pytesseract` is not installed, and the system `tesseract` binary is currently unavailable.

Install/update OCR backends:

```bash
python3 -m pip install easyocr
python3 -m pip install paddleocr paddlepaddle
# optional, requires system binary too
python3 -m pip install pytesseract
```
