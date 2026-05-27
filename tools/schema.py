"""JSON schema helpers for EviTool visual tool calls."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import stable_hash


SUPPORTED_TOOLS = {
    "inspect",
    "crop",
    "zoom",
    "ocr",
    "detect",
    "measure",
    "mark",
    "visualize",
    "click",
    "select_candidate",
}


@dataclass
class EvidenceIdGenerator:
    prefix: str = "ev"
    counter: int = 0

    def next(self) -> str:
        self.counter += 1
        return f"{self.prefix}_{self.counter:03d}"


@dataclass
class ToolResult:
    evidence_id: str
    tool: str
    image: str | None = None
    bbox: list[int] | None = None
    content: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None
    elapsed_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "evidence_id": self.evidence_id,
            "tool": self.tool,
            "image": self.image,
            "bbox": self.bbox,
            "content": self.content,
            "artifacts": self.artifacts,
            "ok": self.ok,
            "error": self.error,
        }
        if self.elapsed_ms is not None:
            data["elapsed_ms"] = round(self.elapsed_ms, 3)
        return data


def make_evidence_id(tool: str, image_path: str | Path, args: dict[str, Any] | None = None) -> str:
    return "ev_" + stable_hash({"tool": tool, "image": str(image_path), "args": args or {}}, length=8)


def parse_action(action: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(action, str):
        action = json.loads(action)
    if not isinstance(action, dict):
        raise ValueError("action must be a JSON object")
    tool = action.get("tool") or action.get("action", {}).get("tool")
    args = action.get("args") or action.get("action", {}).get("args") or {}
    if tool not in SUPPORTED_TOOLS:
        raise ValueError(f"unsupported tool: {tool}")
    if not isinstance(args, dict):
        raise ValueError("action args must be an object")
    return str(tool), args


def run_with_timing(func, *, evidence_id: str, tool: str, image: str | None, bbox=None, **kwargs) -> ToolResult:
    started = time.time()
    try:
        if bbox is not None and "bbox" not in kwargs:
            kwargs["bbox"] = bbox
        content, artifacts, out_bbox = func(**kwargs)
        return ToolResult(
            evidence_id=evidence_id,
            tool=tool,
            image=image,
            bbox=out_bbox if out_bbox is not None else bbox,
            content=content,
            artifacts=artifacts,
            elapsed_ms=(time.time() - started) * 1000,
        )
    except Exception as exc:
        return ToolResult(
            evidence_id=evidence_id,
            tool=tool,
            image=image,
            bbox=bbox,
            content={},
            artifacts={},
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            elapsed_ms=(time.time() - started) * 1000,
        )
