"""Direct GUI action parsing for browser RL rollouts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_ACTIONS = {
    "click",
    "double_click",
    "type",
    "press",
    "hotkey",
    "scroll",
    "drag",
    "wait",
    "finish",
}


@dataclass
class DirectAction:
    action: str
    x: float | None = None
    y: float | None = None
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    text: str | None = None
    key: str | None = None
    keys: list[str] = field(default_factory=list)
    dx: float = 0.0
    dy: float = 0.0
    selector: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"action": self.action}
        for name in ("x", "y", "x1", "y1", "x2", "y2", "text", "key", "selector"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.keys:
            data["keys"] = self.keys
        if self.dx:
            data["dx"] = self.dx
        if self.dy:
            data["dy"] = self.dy
        return data


@dataclass
class ParsedAction:
    action: DirectAction | None
    ok: bool
    error: str | None = None
    input: Any = None


def _as_dict(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("action must be a JSON object")
    if isinstance(payload.get("action"), dict):
        nested = dict(payload["action"])
        if "name" in nested and "action" not in nested:
            nested["action"] = nested.pop("name")
        return nested
    if "name" in payload and "action" not in payload:
        payload = dict(payload)
        payload["action"] = payload.pop("name")
    return payload


def parse_action(payload: str | dict[str, Any]) -> ParsedAction:
    try:
        data = _as_dict(payload)
        name = str(data.get("action") or "").strip()
        if name not in SUPPORTED_ACTIONS:
            raise ValueError(f"unsupported action: {name}")
        action = DirectAction(
            action=name,
            x=_maybe_float(data.get("x")),
            y=_maybe_float(data.get("y")),
            x1=_maybe_float(data.get("x1")),
            y1=_maybe_float(data.get("y1")),
            x2=_maybe_float(data.get("x2")),
            y2=_maybe_float(data.get("y2")),
            text=str(data["text"]) if "text" in data else None,
            key=str(data["key"]) if "key" in data else None,
            keys=[str(item) for item in data.get("keys") or []],
            dx=float(data.get("dx") or 0.0),
            dy=float(data.get("dy") or 0.0),
            selector=str(data["selector"]) if "selector" in data else None,
            raw=data,
        )
        _validate(action)
        return ParsedAction(action=action, ok=True, input=payload)
    except Exception as exc:
        return ParsedAction(action=None, ok=False, error=f"{type(exc).__name__}: {exc}", input=payload)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _validate(action: DirectAction) -> None:
    if action.action in {"click", "double_click"} and not action.selector:
        if action.x is None or action.y is None:
            raise ValueError(f"{action.action} requires x/y or selector")
    if action.action == "type" and action.text is None:
        raise ValueError("type requires text")
    if action.action == "press" and action.key is None:
        raise ValueError("press requires key")
    if action.action == "hotkey" and not action.keys:
        raise ValueError("hotkey requires keys")
    if action.action == "drag":
        required = [action.x1, action.y1, action.x2, action.y2]
        if any(value is None for value in required):
            raise ValueError("drag requires x1/y1/x2/y2")


def normalized_to_pixels(x: float, y: float, viewport: tuple[int, int]) -> tuple[int, int]:
    width, height = viewport
    return (
        int(round(max(0.0, min(1000.0, x)) / 1000.0 * width)),
        int(round(max(0.0, min(1000.0, y)) / 1000.0 * height)),
    )


def pixels_to_normalized(x: float, y: float, viewport: tuple[int, int]) -> tuple[int, int]:
    width, height = viewport
    return (
        int(round(float(x) / max(1, width) * 1000.0)),
        int(round(float(y) / max(1, height) * 1000.0)),
    )
