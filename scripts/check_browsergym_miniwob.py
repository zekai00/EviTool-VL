#!/usr/bin/env python3
"""Check BrowserGym/MiniWoB availability and run minimal reset/step probes."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


DEFAULT_TASKS = [
    "browsergym/miniwob.click-button",
    "browsergym/miniwob.click-checkboxes",
    "browsergym/miniwob.enter-text",
    "browsergym/miniwob.choose-list",
    "browsergym/miniwob.focus-text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--miniwob-url", default=os.environ.get("MINIWOB_URL"))
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.miniwob_url:
        os.environ["MINIWOB_URL"] = args.miniwob_url
    results = []
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "miniwob_url": os.environ.get("MINIWOB_URL"),
        "attempted": len(args.tasks),
        "reset_ok": 0,
        "step_ok": 0,
        "blocked": False,
        "blocker": None,
    }
    try:
        import browsergym.miniwob  # noqa: F401
        import gymnasium as gym
    except Exception as exc:
        summary["blocked"] = True
        summary["blocker"] = f"import_failed: {type(exc).__name__}: {exc}"
        _write(args.output, summary, results)
        return
    for task_id in args.tasks:
        item = {"task_id": task_id, "reset_ok": False, "step_ok": False, "error": None}
        env = None
        try:
            env = gym.make(task_id)
            obs, info = env.reset()
            item["reset_ok"] = True
            item["obs_keys"] = sorted(list(obs.keys())) if isinstance(obs, dict) else []
            _, reward, terminated, truncated, step_info = env.step("noop()")
            item["step_ok"] = True
            item["reward"] = reward
            item["terminated"] = terminated
            item["truncated"] = truncated
            item["info_keys"] = sorted(list(step_info.keys())) if isinstance(step_info, dict) else []
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        results.append(item)
    summary["reset_ok"] = sum(1 for item in results if item["reset_ok"])
    summary["step_ok"] = sum(1 for item in results if item["step_ok"])
    if summary["reset_ok"] == 0:
        summary["blocked"] = True
        summary["blocker"] = results[0]["error"] if results else "no_tasks"
    _write(args.output, summary, results)


def _write(path: str, summary: dict, results: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": results}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
