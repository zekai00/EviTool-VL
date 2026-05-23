"""CLI and dispatcher for local EviTool visual tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import click, crop, detect, measure, ocr, visualize, zoom
from . import inspect_image
from .schema import make_evidence_id, parse_action, run_with_timing


TOOL_IMPLS = {
    "inspect": inspect_image.run,
    "crop": crop.run,
    "zoom": zoom.run,
    "ocr": ocr.run,
    "detect": detect.run,
    "measure": measure.run,
    "mark": visualize.run,
    "visualize": visualize.run,
    "click": click.run,
}


def run_tool(image_path: str | Path, action: str | dict[str, Any], out_dir: str | Path | None = None, evidence_id: str | None = None) -> dict[str, Any]:
    tool, args = parse_action(action)
    impl = TOOL_IMPLS[tool]
    evidence_id = evidence_id or make_evidence_id(tool, image_path, args)
    if out_dir is not None and tool in {"crop", "zoom", "mark", "visualize"}:
        args = {**args, "out_dir": out_dir}
    result = run_with_timing(
        impl,
        evidence_id=evidence_id,
        tool=tool,
        image=str(image_path),
        image_path=image_path,
        **args,
    )
    return result.to_dict()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--action", required=True, help="Tool action JSON string.")
    parser.add_argument("--out-dir", default="outputs/tools", help="Directory for image artifacts.")
    parser.add_argument("--evidence-id", default=None)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_tool(args.image, args.action, out_dir=args.out_dir, evidence_id=args.evidence_id)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
