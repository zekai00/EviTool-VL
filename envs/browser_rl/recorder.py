"""Rollout recorder for browser GUI-RL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RolloutRecorder:
    output_dir: Path
    policy_version: str = "scripted_oracle"
    rollout_id: str | None = None
    task_id: str | None = None
    goal: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)

    def start(self, rollout_id: str, task_id: str, goal: str) -> None:
        self.rollout_id = rollout_id
        self.task_id = task_id
        self.goal = goal
        self.steps = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_step(self, step: dict[str, Any]) -> None:
        self.steps.append(step)

    def finish(self, *, success: bool, total_reward: float, final_info: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "rollout_id": self.rollout_id,
            "policy_version": self.policy_version,
            "task_id": self.task_id,
            "goal": self.goal,
            "trajectory": self.steps,
            "success": success,
            "total_reward": total_reward,
            "num_steps": len(self.steps),
            "final_info": final_info or {},
        }
        return payload


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
