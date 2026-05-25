"""Reward functions for evidence-grounded tool reasoning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

BASELINE_PATH = Path(__file__).resolve().parents[1] / "eval" / "eval_baseline.py"
spec = importlib.util.spec_from_file_location("eval_baseline", BASELINE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def answer_reward(metrics: dict[str, Any], task_type: str) -> float:
    if task_type == "gui_grounding":
        iou = float(metrics.get("iou", 0.0) or 0.0)
        return 0.7 * _bool(metrics.get("pointing")) + 0.3 * min(1.0, iou / 0.5)
    return 0.75 * _bool(metrics.get("relaxed_match")) + 0.25 * _bool(metrics.get("exact_match"))


def score_prediction(prediction: str, row: dict[str, Any]) -> dict[str, Any]:
    return base.score_prediction(prediction, row)


def score_rollout(row: dict[str, Any], rollout: dict[str, Any]) -> dict[str, float | dict[str, Any]]:
    """Return decomposed reward components and weighted total.

    This reward is intentionally inspectable. It is suitable for smoke runs and
    early GRPO experiments; later papers should report each component and its
    ablations rather than only the weighted sum.
    """
    prediction = str(rollout.get("prediction") or "")
    metrics = score_prediction(prediction, row)
    task_type = str(row.get("task_type"))
    tool_count = int(rollout.get("tool_call_count") or 0)
    success_count = int(rollout.get("tool_success_count") or 0)
    parse_errors = rollout.get("parse_errors") or []
    protocol_errors = rollout.get("protocol_errors") or []
    referenced_ids = rollout.get("referenced_evidence_ids") or []
    missing_ids = rollout.get("missing_evidence_ids") or []

    format_component = 0.5 * _bool(rollout.get("final_parseable")) + 0.25 * _bool(not parse_errors) + 0.25 * _bool(not protocol_errors)
    tool_component = 0.0 if tool_count <= 0 else success_count / max(1, tool_count)
    evidence_component = 0.6 * _bool(referenced_ids) + 0.4 * _bool(referenced_ids and not missing_ids)
    ans_component = answer_reward(metrics, task_type)
    cost_penalty = min(0.2, 0.03 * max(0, tool_count - 2))
    empty_penalty = 0.25 if not prediction.strip() else 0.0

    total = (
        0.45 * ans_component
        + 0.20 * evidence_component
        + 0.20 * format_component
        + 0.15 * tool_component
        - cost_penalty
        - empty_penalty
    )
    return {
        "total": round(float(total), 6),
        "answer": round(float(ans_component), 6),
        "evidence": round(float(evidence_component), 6),
        "format": round(float(format_component), 6),
        "tool": round(float(tool_component), 6),
        "cost_penalty": round(float(cost_penalty), 6),
        "empty_penalty": round(float(empty_penalty), 6),
        "metrics": metrics,
    }


def grpo_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """TRL-compatible reward function for single-turn smoke tests.

    Full tool-interactive RL should use `rollout_func` or `environment_factory`;
    this function only validates reward plumbing on generated final JSON/text.
    """
    rows = kwargs.get("rows") or kwargs.get("row") or []
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
            text = str(completion[-1].get("content", ""))
        else:
            text = str(completion)
        row = rows[idx] if isinstance(rows, list) and idx < len(rows) else kwargs
        rollout = {
            "prediction": text,
            "final_parseable": bool(text.strip()),
            "tool_call_count": 0,
            "tool_success_count": 0,
            "parse_errors": [],
            "protocol_errors": [],
            "referenced_evidence_ids": [],
            "missing_evidence_ids": [],
        }
        rewards.append(float(score_rollout(row, rollout)["total"]))
    return rewards
