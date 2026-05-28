"""Programmatic verifiers for browser GUI-RL tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .task_spec import BrowserTaskSpec


@dataclass
class VerifierResult:
    success: bool
    reward: float
    progress: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reward": self.reward,
            "progress": self.progress,
            "error": self.error,
        }


def evaluate_verifier(page: Any, task: BrowserTaskSpec) -> VerifierResult:
    verifier = task.verifier or {}
    success_js = verifier.get("success_js")
    progress_js = verifier.get("progress_js") or {}
    progress: dict[str, bool] = {}
    try:
        success = bool(page.evaluate(success_js)) if success_js else False
        for name, expr in progress_js.items():
            try:
                progress[str(name)] = bool(page.evaluate(expr))
            except Exception:
                progress[str(name)] = False
        if success:
            reward = 1.0
        elif progress:
            reward = 0.2 * (sum(1 for value in progress.values() if value) / max(1, len(progress)))
        else:
            reward = 0.0
        return VerifierResult(success=success, reward=reward, progress=progress)
    except Exception as exc:
        return VerifierResult(success=False, reward=0.0, progress=progress, error=f"{type(exc).__name__}: {exc}")
