"""Task specification helpers for browser GUI-RL."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BrowserTaskSpec:
    task_id: str
    goal: str
    html: str | None = None
    reset_url: str | None = None
    app: str = "local_smoke"
    template: str = "generic"
    seed: int = 0
    split: str = "train"
    difficulty: int = 1
    viewport: tuple[int, int] = (1280, 720)
    max_steps: int = 8
    action_space: list[str] = field(default_factory=lambda: ["click", "type", "press", "scroll", "wait", "finish"])
    verifier: dict[str, Any] = field(default_factory=dict)
    oracle_actions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrowserTaskSpec":
        payload = dict(data)
        if "reset" in payload and isinstance(payload["reset"], dict):
            reset = payload.pop("reset")
            payload.setdefault("reset_url", reset.get("url"))
            payload.setdefault("viewport", reset.get("viewport", payload.get("viewport", (1280, 720))))
        viewport = payload.get("viewport", (1280, 720))
        payload["viewport"] = (int(viewport[0]), int(viewport[1]))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["viewport"] = list(self.viewport)
        return data


def load_tasks(path: str | Path, limit: int | None = None) -> list[BrowserTaskSpec]:
    tasks: list[BrowserTaskSpec] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(BrowserTaskSpec.from_dict(json.loads(line)))
                if limit is not None and len(tasks) >= limit:
                    break
    return tasks


def write_tasks(path: str | Path, tasks: list[BrowserTaskSpec]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
