"""Weakly isolated CUA-Gym mock_web setup/reward runner.

Sandbox here means container-internal weak isolation: each task runs in its own
work directory, untrusted task Python files run in subprocesses with a small
environment whitelist, and reward execution has timeout/resource limits. This
does not provide the same boundary as a separate Docker/VM sandbox.
"""

from __future__ import annotations

import copy
import json
import os
import re
import resource
import shutil
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PLACEHOLDER_RE = re.compile(r"__CUA_GYM_([A-Z0-9_]+)_URL__")
REWARD_RE = re.compile(r"REWARD:\s*([-+]?\d+(?:\.\d+)?)")


@dataclass
class SandboxCommandResult:
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    reward: float | None = None

    def to_dict(self, *, max_chars: int = 4000) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_tail": self.stdout[-max_chars:],
            "stderr_tail": self.stderr[-max_chars:],
            "reward": self.reward,
        }


@dataclass
class SandboxTaskResult:
    task_id: str
    app_type: str | None
    goal: str | None
    work_dir: str
    setup: SandboxCommandResult
    reward: SandboxCommandResult
    state_keys: list[str] = field(default_factory=list)
    browser_log: str = ""

    @property
    def setup_ok(self) -> bool:
        return self.setup.returncode == 0 and not self.setup.timed_out

    @property
    def reward_ok(self) -> bool:
        return self.reward.returncode == 0 and not self.reward.timed_out and self.reward.reward is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "app_type": self.app_type,
            "goal": self.goal,
            "work_dir": self.work_dir,
            "setup_ok": self.setup_ok,
            "reward_ok": self.reward_ok,
            "reward": self.reward.reward,
            "setup": self.setup.to_dict(),
            "reward_exec": self.reward.to_dict(),
            "state_keys": self.state_keys,
            "browser_log": self.browser_log[-4000:],
        }


class MockWebStateStore:
    """Thread-safe in-memory state API compatible with CUA-Gym mock scripts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[tuple[str, str], dict[str, Any]] = {}

    def set_state(self, app: str, sid: str, state: Any) -> None:
        state_copy = copy.deepcopy(state)
        with self._lock:
            self._states[(app, sid)] = {
                "initial_state": copy.deepcopy(state_copy),
                "current_state": copy.deepcopy(state_copy),
            }

    def set_current_state(self, app: str, sid: str, state: Any) -> None:
        with self._lock:
            item = self._states.setdefault((app, sid), {"initial_state": None, "current_state": None})
            item["current_state"] = copy.deepcopy(state)

    def get_state(self, app: str, sid: str) -> dict[str, Any]:
        with self._lock:
            item = self._states.get((app, sid), {"initial_state": None, "current_state": None})
            return copy.deepcopy(item)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {f"{app}:{sid}": copy.deepcopy(value) for (app, sid), value in self._states.items()}

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


class MockWebStateServer:
    """Local HTTP state server for CUA-Gym mock app setup/reward scripts."""

    def __init__(self, host: str = "127.0.0.1", port: int | None = None) -> None:
        self.host = host
        self.port = port or find_free_port(host)
        self.store = MockWebStateStore()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "MockWebStateServer":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._server is not None:
            return
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                app, endpoint = parse_app_endpoint(parsed.path)
                sid = parse_qs(parsed.query).get("sid", ["default"])[0]
                if endpoint == "go":
                    self._write_json(store.get_state(app, sid))
                    return
                if endpoint in {"", "index"} or endpoint not in {"post", "go"}:
                    state = store.get_state(app, sid)
                    html = render_debug_html(app, sid, state)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(html.encode("utf-8"))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                app, endpoint = parse_app_endpoint(parsed.path)
                sid = parse_qs(parsed.query).get("sid", ["default"])[0]
                if endpoint != "post":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception:
                    self.send_error(HTTPStatus.BAD_REQUEST, "invalid json")
                    return
                action = payload.get("action")
                if action == "set":
                    store.set_state(app, sid, payload.get("state"))
                elif action in {"set_current", "set_current_state"}:
                    store.set_current_state(app, sid, payload.get("state"))
                else:
                    self.send_error(HTTPStatus.BAD_REQUEST, f"unsupported action: {action}")
                    return
                self._write_json({"ok": True, "app": app, "sid": sid})

            def log_message(self, *_: Any) -> None:
                return

            def _write_json(self, payload: Any) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class CuaGymMockWebSandbox:
    """Runs copied CUA-Gym task setup/reward scripts with weak isolation."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        timeout_sec: int = 45,
        cpu_seconds: int = 20,
        memory_mb: int = 2048,
        file_mb: int = 128,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.timeout_sec = timeout_sec
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb
        self.file_mb = file_mb

    def run_task(self, task_row: dict[str, Any], bundle_dir: str | Path, server: MockWebStateServer) -> SandboxTaskResult:
        server.store.clear()
        task_id = str(task_row.get("task_id") or task_row.get("id") or Path(bundle_dir).name)
        task_out = self.output_dir / task_id
        work_dir = task_out / "work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        copy_bundle(Path(bundle_dir), work_dir)
        patch_cua_scripts(work_dir, server.base_url)
        fake_bin, browser_log_path = make_fake_browser_bin(task_out)
        env = safe_subprocess_env(fake_bin, work_dir)

        setup = self._run_python(work_dir / "initial_setup.py", work_dir, env)
        reward = self._run_python(work_dir / "reward.py", work_dir, env)
        state_snapshot = server.store.snapshot()
        task_out.mkdir(parents=True, exist_ok=True)
        (task_out / "state_snapshot.json").write_text(json.dumps(state_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        browser_log = browser_log_path.read_text(encoding="utf-8") if browser_log_path.exists() else ""
        result = SandboxTaskResult(
            task_id=task_id,
            app_type=task_row.get("app_type"),
            goal=task_row.get("goal") or task_row.get("instruction"),
            work_dir=str(work_dir),
            setup=setup,
            reward=reward,
            state_keys=sorted(state_snapshot.keys()),
            browser_log=browser_log,
        )
        (task_out / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (task_out / "setup_stdout.txt").write_text(setup.stdout, encoding="utf-8")
        (task_out / "setup_stderr.txt").write_text(setup.stderr, encoding="utf-8")
        (task_out / "reward_stdout.txt").write_text(reward.stdout, encoding="utf-8")
        (task_out / "reward_stderr.txt").write_text(reward.stderr, encoding="utf-8")
        return result

    def _run_python(self, script: Path, cwd: Path, env: dict[str, str]) -> SandboxCommandResult:
        if not script.exists():
            return SandboxCommandResult(returncode=None, timed_out=False, stdout="", stderr=f"missing script: {script}")
        try:
            completed = subprocess.run(
                [sys.executable, str(script.resolve())],
                cwd=str(cwd),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                preexec_fn=lambda: limit_child_resources(self.cpu_seconds, self.memory_mb, self.file_mb),
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            return SandboxCommandResult(
                returncode=completed.returncode,
                timed_out=False,
                stdout=stdout,
                stderr=stderr,
                reward=parse_reward(stdout + "\n" + stderr),
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxCommandResult(
                returncode=None,
                timed_out=True,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                reward=parse_reward((exc.stdout or "") + "\n" + (exc.stderr or "")),
            )


def parse_app_endpoint(path: str) -> tuple[str, str]:
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        return "default", ""
    if parts[-1] in {"post", "go"}:
        app = "/".join(parts[:-1]) or "default"
        return app, parts[-1]
    return parts[0], parts[-1]


def render_debug_html(app: str, sid: str, state: dict[str, Any]) -> str:
    body = json.dumps(state, ensure_ascii=False, indent=2)
    escaped = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html>
  <head><meta charset="utf-8"><title>CUA-Gym Mock State</title></head>
  <body>
    <h1>CUA-Gym Mock State</h1>
    <p>app={app} sid={sid}</p>
    <pre>{escaped}</pre>
  </body>
</html>
"""


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def copy_bundle(src: Path, dst: Path) -> None:
    def ignore(_: str, names: list[str]) -> set[str]:
        return {name for name in names if name.startswith("._") or name == "__pycache__"}

    shutil.copytree(src, dst, ignore=ignore)


def patch_cua_scripts(work_dir: Path, base_url: str) -> None:
    for script in work_dir.glob("*.py"):
        text = script.read_text(encoding="utf-8")

        def replace_placeholder(match: re.Match[str]) -> str:
            app_key = match.group(1).lower()
            return f"{base_url}/{app_key}"

        text = PLACEHOLDER_RE.sub(replace_placeholder, text)
        isolated_tmp = str((work_dir / "tmp").resolve())
        Path(isolated_tmp).mkdir(parents=True, exist_ok=True)
        text = text.replace("/tmp/task_web_sid", f"{isolated_tmp}/task_web_sid")
        text = text.replace("/tmp/task_golden_sid", f"{isolated_tmp}/task_golden_sid")
        script.write_text(text, encoding="utf-8")


def make_fake_browser_bin(task_out: Path) -> tuple[Path, Path]:
    fake_bin = task_out / "fake_bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    browser_log = task_out / "fake_browser.log"
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = fake_bin / name
        path.write_text(
            "#!/bin/sh\n"
            "printf '%s %s\\n' \"$(date '+%Y-%m-%d %H:%M:%S')\" \"$*\" >> \"$CUA_SANDBOX_BROWSER_LOG\"\n"
            "exit 0\n",
            encoding="utf-8",
        )
        path.chmod(0o755)
    return fake_bin, browser_log


def safe_subprocess_env(fake_bin: Path, work_dir: Path) -> dict[str, str]:
    fake_bin = fake_bin.resolve()
    work_dir = work_dir.resolve()
    keep = {
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HOME": str(work_dir),
        "TMPDIR": str(work_dir / "tmp"),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "CUA_SANDBOX_BROWSER_LOG": str((fake_bin.parent / "fake_browser.log").resolve()),
    }
    return keep


def limit_child_resources(cpu_seconds: int, memory_mb: int, file_mb: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))
    memory_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    file_bytes = file_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def parse_reward(text: str) -> float | None:
    matches = REWARD_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def write_sandbox_summary(output_dir: Path, results: list[SandboxTaskResult]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": len(results),
        "setup_ok": sum(1 for item in results if item.setup_ok),
        "reward_ok": sum(1 for item in results if item.reward_ok),
        "avg_initial_reward": (
            sum(float(item.reward.reward or 0.0) for item in results if item.reward.reward is not None)
            / max(1, sum(1 for item in results if item.reward.reward is not None))
        ),
        "isolation": "subprocess_weak_container_internal",
        "notes": [
            "No host Docker permission is required.",
            "This validates setup/reward execution against a local state server, not the real mock web frontend.",
            "API keys and .env variables are not passed to task subprocesses.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "runs.jsonl").write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    return summary
