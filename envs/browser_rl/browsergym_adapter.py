"""Adapter for BrowserGym MiniWoB tasks.

Adapter means a small translation layer. BrowserGym has its own observation,
action, and reward format; this class converts it into the browser_rl format
used by this repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .actions import DirectAction, normalized_to_pixels, parse_action


class BrowserGymMiniwobAdapter:
    """Run MiniWoB++ tasks through BrowserGym and expose our rollout schema."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        miniwob_url: str | None = None,
        headless: bool = True,
        screenshot_subdir: str = "screenshots",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.screenshot_dir = self.output_dir / screenshot_subdir
        self.miniwob_url = miniwob_url
        self.headless = headless
        self.env = None
        self.task_id: str | None = None
        self.goal: str = ""
        self.step_index = 0
        self.history: list[dict[str, Any]] = []
        self.viewport: tuple[int, int] = (0, 0)
        self.last_obs: dict[str, Any] | None = None

    def __enter__(self) -> "BrowserGymMiniwobAdapter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self, task_id: str, *, seed: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
        import os

        if self.miniwob_url:
            os.environ["MINIWOB_URL"] = self.miniwob_url
        import browsergym.miniwob  # noqa: F401
        import gymnasium as gym
        from browsergym.core.action.highlevel import HighLevelActionSet

        self.close()
        self.task_id = task_id
        self.step_index = 0
        self.history = []
        action_mapping = HighLevelActionSet(subsets=["miniwob_all"], multiaction=False).to_python_code
        self.env = gym.make(task_id, headless=self.headless, action_mapping=action_mapping)
        obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        self.goal = str(obs.get("goal") or "")
        screenshot = self._write_screenshot(obs)
        observation = self._observation(obs, screenshot)
        return observation, {"task_id": task_id, "goal": self.goal, "browsergym_info": self._jsonable(info)}

    def step(self, action_payload: str | dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.env is None or self.task_id is None:
            raise RuntimeError("call reset() before step()")
        parsed = parse_action(action_payload)
        action_string = "noop()"
        materialized: dict[str, Any] | str = action_payload
        exec_status = "invalid_action"
        exec_error = parsed.error
        if parsed.ok and parsed.action is not None:
            try:
                action_string = self._to_browsergym_action(parsed.action)
                materialized = parsed.action.to_json()
                exec_status = "ok"
                exec_error = None
            except Exception as exc:
                exec_status = "action_map_error"
                exec_error = f"{type(exc).__name__}: {exc}"
        obs, reward, terminated, truncated, bg_info = self.env.step(action_string)
        self.last_obs = obs
        self.step_index += 1
        screenshot = self._write_screenshot(obs)
        observation = self._observation(obs, screenshot)
        success = bool(float(reward) > 0 and (terminated or truncated))
        info = {
            "task_id": self.task_id,
            "step": self.step_index,
            "valid_action": parsed.ok,
            "exec_status": exec_status,
            "exec_error": exec_error,
            "action": materialized,
            "browsergym_action": action_string,
            "verifier": {"success": success, "reward": float(reward), "progress": {}, "error": None},
            "browsergym_info": self._jsonable(bg_info),
            "screenshot": screenshot,
        }
        self.history.append(info)
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _to_browsergym_action(self, action: DirectAction) -> str:
        if action.action == "wait":
            return "noop(200)"
        if action.action == "finish":
            return "noop(200)"
        if action.action in {"click", "double_click"}:
            x, y = normalized_to_pixels(float(action.x), float(action.y), self.viewport)
            fn = "mouse_dblclick" if action.action == "double_click" else "mouse_click"
            return f"{fn}({x}, {y})"
        if action.action == "type":
            return f"keyboard_type({action.text!r})"
        if action.action == "press":
            return f"keyboard_press({action.key!r})"
        if action.action == "scroll":
            return f"scroll({float(action.dx)}, {float(action.dy)})"
        if action.action == "drag":
            x1, y1 = normalized_to_pixels(float(action.x1), float(action.y1), self.viewport)
            x2, y2 = normalized_to_pixels(float(action.x2), float(action.y2), self.viewport)
            return f"mouse_drag_and_drop({x1}, {y1}, {x2}, {y2})"
        if action.action == "hotkey":
            return f"keyboard_press({'+'.join(action.keys)!r})"
        raise ValueError(f"unsupported browsergym action: {action.action}")

    def _write_screenshot(self, obs: dict[str, Any]) -> str:
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot = obs.get("screenshot")
        if screenshot is None:
            raise RuntimeError("BrowserGym observation does not contain screenshot")
        image = Image.fromarray(screenshot)
        self.viewport = (int(image.width), int(image.height))
        safe_task_id = str(self.task_id).replace("/", "_")
        path = self.screenshot_dir / f"{safe_task_id}_step{self.step_index:02d}.png"
        image.save(path)
        return str(path)

    def _observation(self, obs: dict[str, Any], screenshot: str) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": str(obs.get("goal") or self.goal),
            "screenshot": screenshot,
            "viewport": list(self.viewport),
            "step": self.step_index,
            "history": self.history,
            "action_space": ["click", "double_click", "type", "press", "scroll", "drag", "wait", "finish"],
            "max_steps": None,
            "source": "browsergym_miniwob",
            "url": str(obs.get("url") or ""),
            "last_action": str(obs.get("last_action") or ""),
            "last_action_error": str(obs.get("last_action_error") or ""),
        }

    def _jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        return str(value)
