"""Interactive rollout environment for EviTool visual tools."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.runner import run_tool
from rl.rewards import score_rollout

EVAL_TOOL_PATH = PROJECT_ROOT / "eval" / "eval_tool_baseline.py"
spec = importlib.util.spec_from_file_location("eval_tool_baseline", EVAL_TOOL_PATH)
eval_tool = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(eval_tool)


class EviToolRolloutEnvironment:
    """Stateful tool environment compatible with TRL experiments.

    The class provides Python tool methods for future `environment_factory`
    use, and also exposes `run_scripted` for deterministic dry-runs.
    """

    def __init__(self, image_root: str | Path, tool_out_dir: str | Path = "outputs/rl_tool_artifacts"):
        self.image_root = Path(image_root)
        self.tool_out_dir = Path(tool_out_dir)
        self.row: dict[str, Any] | None = None
        self.image_path: Path | None = None
        self.trace: list[dict[str, Any]] = []
        self.evidence_ids: list[str] = []

    def reset(self, **kwargs: Any) -> str:
        """Reset environment state and return task context for the model."""
        self.row = dict(kwargs.get("row") or kwargs)
        image_rel = self.row.get("image")
        self.image_path = self.image_root / str(image_rel)
        self.trace = []
        self.evidence_ids = []
        return eval_tool.initial_prompt(self.row, self.image_path, allow_direct_final=False)

    def _call_tool(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.image_path is None:
            raise RuntimeError("environment must be reset before calling tools")
        action = {"tool": tool, "args": args or {}}
        observation = run_tool(self.image_path, action, out_dir=self.tool_out_dir)
        self.trace.append({"action": action, "observation": observation})
        if observation.get("evidence_id"):
            self.evidence_ids.append(str(observation["evidence_id"]))
        return observation

    def detect(self, mode: str = "ui", query: str = "", max_results: int = 30, min_area: float = 20.0) -> str:
        """Find visual candidate regions in the current image.

        Args:
            mode: Detection mode such as ui, text, diagram, bar, layout, or color.
            query: Optional natural-language target description.
            max_results: Maximum number of candidates.
            min_area: Minimum candidate area in pixels.

        Returns:
            JSON observation with evidence id and candidate detections.
        """
        return json.dumps(self._call_tool("detect", {"mode": mode, "query": query, "max_results": max_results, "min_area": min_area}), ensure_ascii=False)

    def ocr(self, bbox: list[float] | None = None, engine: str = "easyocr") -> str:
        """Read text in the current image or a bbox region.

        Args:
            bbox: Optional [x1, y1, x2, y2] crop region.
            engine: OCR backend name.

        Returns:
            JSON observation with OCR spans.
        """
        args: dict[str, Any] = {"engine": engine, "languages": ["en"]}
        if bbox is not None:
            args["bbox"] = bbox
        return json.dumps(self._call_tool("ocr", args), ensure_ascii=False)

    def click(self, bbox: list[float] | None = None, point: list[float] | None = None, label: str = "") -> str:
        """Record a GUI click or selected bbox as evidence.

        Args:
            bbox: Optional [x1, y1, x2, y2] selected target box.
            point: Optional [x, y] selected point.
            label: Short target label.

        Returns:
            JSON observation with selected target evidence.
        """
        args: dict[str, Any] = {"label": label}
        if bbox is not None:
            args["bbox"] = bbox
        if point is not None:
            args["point"] = point
        return json.dumps(self._call_tool("click", args), ensure_ascii=False)

    def run_scripted(self, actions: list[dict[str, Any]], final_answer: Any, final_evidence: list[str] | None = None) -> dict[str, Any]:
        if self.row is None:
            raise RuntimeError("environment must be reset before scripted rollout")
        started = time.time()
        for action in actions:
            self._call_tool(str(action.get("tool")), dict(action.get("args") or {}))
        final_evidence = final_evidence if final_evidence is not None else list(self.evidence_ids)
        final = {"reasoning": [{"step": "scripted rollout sanity check", "evidence": final_evidence}], "answer": final_answer}
        prediction = json.dumps(final_answer, ensure_ascii=False) if isinstance(final_answer, (list, dict)) else str(final_answer)
        missing = [eid for eid in final_evidence if eid not in self.evidence_ids]
        rollout = {
            "prediction": prediction,
            "final": final,
            "tool_trace": self.trace,
            "tool_call_count": len(self.trace),
            "tool_success_count": sum(1 for step in self.trace if step["observation"].get("ok")),
            "parse_errors": [],
            "protocol_errors": [],
            "final_parseable": True,
            "referenced_evidence_ids": final_evidence,
            "missing_evidence_ids": missing,
            "evidence_closed": bool(final_evidence) and not missing,
            "latency_sec": round(time.time() - started, 4),
        }
        rollout["reward"] = score_rollout(self.row, rollout)
        return rollout


def extract_json_object(text: str) -> dict[str, Any]:
    return eval_tool.extract_json_object(text)


def classify_model_json(obj: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return eval_tool.classify_model_json(obj)
