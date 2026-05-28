"""Playwright-backed browser environment for direct-action GUI-RL."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .actions import DirectAction, normalized_to_pixels, parse_action, pixels_to_normalized
from .task_spec import BrowserTaskSpec
from .verifier import VerifierResult, evaluate_verifier


class PlaywrightBrowserEnv:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        headless: bool = True,
        screenshot_subdir: str = "screenshots",
        reuse_context: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.screenshot_dir = self.output_dir / screenshot_subdir
        self.headless = headless
        self.reuse_context = reuse_context
        self._playwright = None
        self._browser = None
        self._context = None
        self._context_viewport: tuple[int, int] | None = None
        self.page = None
        self.task: BrowserTaskSpec | None = None
        self.step_index = 0
        self.history: list[dict[str, Any]] = []
        self.screenshot_prefix = ""

    def __enter__(self) -> "PlaywrightBrowserEnv":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._playwright is not None:
            return
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None
        self.page = None

    def reset(self, task: BrowserTaskSpec) -> tuple[dict[str, Any], dict[str, Any]]:
        self.start()
        width, height = task.viewport
        can_reuse = self.reuse_context and self._context is not None and self._context_viewport == task.viewport
        if self._context is not None and not can_reuse:
            self._context.close()
            self._context = None
            self.page = None
            self._context_viewport = None
        self.task = task
        self.step_index = 0
        self.history = []
        if self._context is None:
            self._context = self._browser.new_context(viewport={"width": width, "height": height})
            self._context_viewport = task.viewport
            self.page = self._context.new_page()
        elif self.page is None:
            self.page = self._context.new_page()
        else:
            self.page.goto("about:blank")
        if task.reset_url:
            self.page.goto(task.reset_url, wait_until="networkidle")
        elif task.html is not None:
            self.page.set_content(task.html, wait_until="load")
        else:
            raise ValueError("task requires either reset_url or html")
        screenshot = self._screenshot()
        observation = self._observation(screenshot)
        return observation, {"task_id": task.task_id, "goal": task.goal}

    def step(self, action_payload: str | dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.task is None or self.page is None:
            raise RuntimeError("call reset() before step()")
        parsed = parse_action(action_payload)
        exec_status = "invalid_action"
        exec_error = parsed.error
        materialized: dict[str, Any] | None = None
        if parsed.ok and parsed.action is not None:
            try:
                action = self.materialize_action(parsed.action)
                materialized = action.to_json()
                self._execute(action)
                exec_status = "ok"
                exec_error = None
            except Exception as exc:
                exec_status = "exec_error"
                exec_error = f"{type(exc).__name__}: {exc}"
        self.step_index += 1
        verifier = evaluate_verifier(self.page, self.task)
        terminated = bool(verifier.success)
        if parsed.ok and parsed.action is not None and parsed.action.action == "finish":
            terminated = True
        truncated = self.step_index >= self.task.max_steps and not terminated
        screenshot = self._screenshot()
        observation = self._observation(screenshot)
        info = {
            "task_id": self.task.task_id,
            "step": self.step_index,
            "valid_action": parsed.ok,
            "exec_status": exec_status,
            "exec_error": exec_error,
            "action": materialized if materialized is not None else action_payload,
            "verifier": verifier.to_dict(),
            "screenshot": screenshot,
        }
        self.history.append(info)
        return observation, verifier.reward, terminated, truncated, info

    def materialize_action(self, action: DirectAction) -> DirectAction:
        if action.selector and action.action in {"click", "double_click"}:
            center = self.selector_center(action.selector)
            x, y = pixels_to_normalized(center[0], center[1], self.task.viewport)
            data = action.to_json()
            data.pop("selector", None)
            data["x"] = x
            data["y"] = y
            return parse_action(data).action
        return action

    def selector_center(self, selector: str) -> tuple[float, float]:
        locator = self.page.locator(selector).first
        locator.wait_for(state="visible", timeout=5000)
        box = locator.bounding_box()
        if box is None:
            raise RuntimeError(f"selector has no bounding box: {selector}")
        return (float(box["x"] + box["width"] / 2), float(box["y"] + box["height"] / 2))

    def _execute(self, action: DirectAction) -> None:
        if action.action == "wait":
            time.sleep(0.2)
            return
        if action.action == "finish":
            return
        viewport = self.task.viewport
        if action.action in {"click", "double_click"}:
            x, y = normalized_to_pixels(float(action.x), float(action.y), viewport)
            if action.action == "double_click":
                self.page.mouse.dblclick(x, y)
            else:
                self.page.mouse.click(x, y)
            return
        if action.action == "type":
            self.page.keyboard.type(action.text or "")
            return
        if action.action == "press":
            self.page.keyboard.press(action.key or "")
            return
        if action.action == "hotkey":
            for key in action.keys:
                self.page.keyboard.down(key)
            for key in reversed(action.keys):
                self.page.keyboard.up(key)
            return
        if action.action == "scroll":
            self.page.mouse.wheel(action.dx, action.dy)
            return
        if action.action == "drag":
            x1, y1 = normalized_to_pixels(float(action.x1), float(action.y1), viewport)
            x2, y2 = normalized_to_pixels(float(action.x2), float(action.y2), viewport)
            self.page.mouse.move(x1, y1)
            self.page.mouse.down()
            self.page.mouse.move(x2, y2)
            self.page.mouse.up()
            return
        raise ValueError(f"unsupported action: {action.action}")

    def _screenshot(self) -> str:
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        task_id = self.task.task_id if self.task is not None else "unknown"
        base = self.screenshot_prefix or task_id
        path = self.screenshot_dir / f"{base}_step{self.step_index:02d}.png"
        self.page.screenshot(path=str(path), full_page=False)
        return str(path)

    def _observation(self, screenshot: str) -> dict[str, Any]:
        task = self.task
        return {
            "task_id": task.task_id,
            "goal": task.goal,
            "screenshot": screenshot,
            "viewport": list(task.viewport),
            "step": self.step_index,
            "history": self.history,
            "action_space": task.action_space,
            "max_steps": task.max_steps,
        }
