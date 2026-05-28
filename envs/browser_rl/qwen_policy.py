"""DashScope Qwen teacher/baseline policy for browser GUI-RL rollouts.

Policy means the component that chooses the next action. This module sends a
screenshot and a task goal to a DashScope OpenAI-compatible Qwen endpoint, then
normalizes the model output into one direct GUI action JSON object.

DashScope models used here are teacher/baseline policies. They are not the
trainable current model for on-policy RL; the current model must be a local
Qwen2.5-VL/Qwen3-VL checkpoint whose parameters we update.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import SUPPORTED_ACTIONS, parse_action


DEFAULT_ACTION = {"action": "wait"}
ACTION_ALIASES = {
    "tap": "click",
    "left_click": "click",
    "input": "type",
    "enter_text": "type",
    "keyboard_type": "type",
    "key": "press",
    "keyboard_press": "press",
    "done": "finish",
    "stop": "finish",
}


@dataclass
class QwenPolicyResult:
    action: dict[str, Any]
    info: dict[str, Any]


class QwenDashScopePolicy:
    """DashScope-backed Qwen teacher/baseline policy for GUI actions."""

    def __init__(
        self,
        *,
        models: list[str],
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env: str = "DASHSCOPE_API_KEY",
        env_file: str | Path = ".env",
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 120.0,
        retries: int = 1,
        max_history: int = 4,
        on_error_action: dict[str, Any] | None = None,
    ) -> None:
        self.models = [model.strip() for model in models if model.strip()]
        if not self.models:
            raise ValueError("at least one Qwen model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.env_file = Path(env_file)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.max_history = max_history
        self.on_error_action = on_error_action or DEFAULT_ACTION
        self.api_key = self._load_api_key()

    def act(self, observation: dict[str, Any]) -> QwenPolicyResult:
        prompt = build_prompt(observation, max_history=self.max_history)
        screenshot = Path(str(observation.get("screenshot") or ""))
        errors: list[str] = []
        for model in self.models:
            for attempt in range(self.retries + 1):
                try:
                    raw_text, raw_payload = self._chat_completion(model=model, screenshot=screenshot, prompt=prompt)
                    action, parse_info = parse_model_action(raw_text)
                    return QwenPolicyResult(
                        action=action,
                        info={
                            "policy": "qwen_dashscope",
                            "provider": "dashscope",
                            "model": model,
                            "attempt": attempt,
                            "raw_text": raw_text,
                            "raw_usage": raw_payload.get("usage"),
                            **parse_info,
                        },
                    )
                except Exception as exc:
                    message = f"{model}: attempt={attempt}: {type(exc).__name__}: {exc}"
                    errors.append(message)
                    if attempt < self.retries:
                        time.sleep(min(2.0, 0.5 * (attempt + 1)))
        return QwenPolicyResult(
            action=dict(self.on_error_action),
            info={
                "policy": "qwen_dashscope",
                "provider": "dashscope",
                "model": None,
                "raw_text": "",
                "valid_json": False,
                "valid_action": False,
                "error": errors[-1] if errors else "unknown policy error",
                "errors": errors[-5:],
                "fallback_action": self.on_error_action,
            },
        )

    def _chat_completion(self, *, model: str, screenshot: Path, prompt: str) -> tuple[str, dict[str, Any]]:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url(screenshot)}},
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(f"no choices returned: {payload}")
        message = choices[0].get("message") or {}
        return str(message.get("content") or "").strip(), payload

    def _load_api_key(self) -> str:
        load_env_file(self.env_file)
        # Also check the workspace root because this repo often runs from
        # /root/Workspace/VLM/EviTool-VL while project-level files live one
        # directory above.
        load_env_file(Path(__file__).resolve().parents[3] / ".env")
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Set {self.api_key_env} in environment or .env.")
        return key


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def image_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"screenshot not found: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_prompt(observation: dict[str, Any], *, max_history: int = 4) -> str:
    goal = str(observation.get("goal") or "")
    step = observation.get("step")
    max_steps = observation.get("max_steps")
    viewport = observation.get("viewport") or [0, 0]
    action_space = observation.get("action_space") or sorted(SUPPORTED_ACTIONS)
    history = compact_history(observation.get("history") or [], max_items=max_history)
    progress = current_progress(history)
    coord_hint = coordinate_hint(viewport)
    viewport_text = json.dumps(viewport, ensure_ascii=False)
    action_space_text = json.dumps(action_space, ensure_ascii=False)
    return (
        "你是一个浏览器 GUI 操作智能体。GUI 指图形用户界面，你只能看截图和任务目标，然后输出下一步动作。\n"
        "必须只输出一个 JSON 对象，不要输出 Markdown，不要解释。\n"
        "坐标必须使用 0-1000 归一化截图坐标：左上角是 (0,0)，右下角是 (1000,1000)。不要输出像素坐标。\n"
        f"{coord_hint}\n"
        "可用动作：\n"
        "- click: {\"action\":\"click\",\"x\":500,\"y\":500}\n"
        "- double_click: {\"action\":\"double_click\",\"x\":500,\"y\":500}\n"
        "- type: {\"action\":\"type\",\"text\":\"要输入的文本\"}\n"
        "- press: {\"action\":\"press\",\"key\":\"Enter\"}\n"
        "- hotkey: {\"action\":\"hotkey\",\"keys\":[\"Control\",\"A\"]}\n"
        "- scroll: {\"action\":\"scroll\",\"dy\":250} 或 {\"action\":\"scroll\",\"dy\":-250}\n"
        "- drag: {\"action\":\"drag\",\"x1\":100,\"y1\":100,\"x2\":900,\"y2\":900}\n"
        "- wait: {\"action\":\"wait\"}\n"
        "- finish: {\"action\":\"finish\"}\n"
        "如果需要输入文字，通常先 click 到输入框，再 type。任务完成后输出 finish。\n"
        f"任务目标：{goal}\n"
        f"当前 step：{step}\n"
        f"最大 step：{max_steps}\n"
        f"viewport：{viewport_text}\n"
        f"允许动作：{action_space_text}\n"
        f"当前 verifier progress：{json.dumps(progress, ensure_ascii=False)}\n"
        f"最近历史：{json.dumps(history, ensure_ascii=False)}\n"
        "根据截图和历史判断下一步，不要重复已经无效或没有进展的动作。\n"
        "现在只输出下一步 GUI action JSON。"
    )


def coordinate_hint(viewport: Any) -> str:
    try:
        width = max(1, int(viewport[0]))
        height = max(1, int(viewport[1]))
    except Exception:
        return "如果你估计目标像素点为 (px, py)，必须先换算成 x=round(px/width*1000), y=round(py/height*1000)。"
    px = max(1, int(round(width * 0.2)))
    py = max(1, int(round(height * 0.8)))
    nx = int(round(px / width * 1000))
    ny = int(round(py / height * 1000))
    return (
        f"当前截图尺寸是 width={width}, height={height}。"
        f"如果你估计像素点是 ({px},{py})，必须输出 x={nx}, y={ny}。"
        f"换算公式：x=round(px/{width}*1000), y=round(py/{height}*1000)。"
    )


def compact_history(history: list[Any], *, max_items: int) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in history[-max_items:]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "step": item.get("step"),
                "action": item.get("action"),
                "exec_status": item.get("exec_status"),
                "exec_error": item.get("exec_error"),
                "verifier": item.get("verifier"),
            }
        )
    return compact


def current_progress(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    verifier = history[-1].get("verifier") or {}
    progress = verifier.get("progress") if isinstance(verifier, dict) else None
    return progress if isinstance(progress, dict) else {}


def parse_model_action(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    info: dict[str, Any] = {"valid_json": False, "valid_action": False, "error": None}
    try:
        payload = parse_json_object(text)
        info["valid_json"] = True
    except Exception as exc:
        info["error"] = f"json_parse_error: {type(exc).__name__}: {exc}"
        return dict(DEFAULT_ACTION), info
    action = normalize_action_payload(payload)
    parsed = parse_action(action)
    if not parsed.ok or parsed.action is None:
        info["error"] = parsed.error
        return dict(DEFAULT_ACTION), info
    info["valid_action"] = True
    return parsed.action.to_json(), info


def parse_json_object(text: str) -> Any:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_action_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return dict(DEFAULT_ACTION)
    data = dict(payload.get("action") if isinstance(payload.get("action"), dict) else payload)
    name = str(data.get("action") or data.get("name") or data.get("type") or "").strip()
    name = ACTION_ALIASES.get(name, name)
    data["action"] = name
    data.pop("name", None)
    data.pop("type", None)
    if "coordinate" in data and ("x" not in data or "y" not in data):
        apply_point(data, data.get("coordinate"))
    if "point" in data and ("x" not in data or "y" not in data):
        apply_point(data, data.get("point"))
    if "position" in data and ("x" not in data or "y" not in data):
        apply_point(data, data.get("position"))
    if "bbox" in data and ("x" not in data or "y" not in data):
        apply_bbox_center(data, data.get("bbox"))
    if "content" in data and "text" not in data:
        data["text"] = data.get("content")
    for key in ("x", "y", "x1", "y1", "x2", "y2", "dx", "dy"):
        if key in data:
            data[key] = unwrap_scalar(data[key])
    if name not in SUPPORTED_ACTIONS:
        data["action"] = name
    return data


def unwrap_scalar(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def apply_point(data: dict[str, Any], point: Any) -> None:
    if isinstance(point, dict):
        data["x"] = point.get("x")
        data["y"] = point.get("y")
    elif isinstance(point, list) and len(point) >= 2:
        data["x"] = point[0]
        data["y"] = point[1]


def apply_bbox_center(data: dict[str, Any], bbox: Any) -> None:
    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        data["x"] = (x1 + x2) / 2.0
        data["y"] = (y1 + y2) / 2.0
