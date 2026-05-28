#!/usr/bin/env python3
"""Inspect CUA-Gym mock_web tasks for the next browser-RL phase.

mock_web means simulated web applications. They are lighter than desktop tasks
such as VSCode or LibreOffice, so they are the first CUA-Gym subset to try.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo": "xlangai/CUA-Gym",
        "metadata_rows": 0,
        "mock_web_rows": 0,
        "sample_limit": args.limit,
        "inspected_bundles": 0,
        "reward_py_compile_ok": 0,
        "blocked": False,
        "blockers": [],
    }
    try:
        from datasets import load_dataset

        ds = load_dataset("xlangai/CUA-Gym", "tasks", split="train")
        rows = [dict(row) for row in ds]
    except Exception as exc:
        summary["blocked"] = True
        summary["blockers"].append(f"metadata_load_failed: {type(exc).__name__}: {exc}")
        write_outputs(output_dir, summary, [], [])
        return

    mock_rows = [row for row in rows if row.get("app_family") == "mock_web" or str(row.get("app_type", "")).endswith("_mock")]
    summary["metadata_rows"] = len(rows)
    summary["mock_web_rows"] = len(mock_rows)
    summary["mock_web_by_app_type"] = dict(Counter(str(row.get("app_type")) for row in mock_rows).most_common())
    sample = mock_rows[: args.limit]
    normalized = [normalize_row(row) for row in sample]
    bundle_results: list[dict[str, Any]] = []
    try:
        archive = Path(args.artifact) if args.artifact else download_artifact(output_dir)
        tar_path = decompressed_tar_path(archive)
        bundle_results = inspect_bundles(tar_path, sample, output_dir / "bundles")
        summary["inspected_bundles"] = len(bundle_results)
        summary["reward_py_compile_ok"] = sum(1 for item in bundle_results if item.get("reward_py_compile_ok"))
    except Exception as exc:
        summary["blocked"] = True
        summary["blockers"].append(f"bundle_inspection_failed: {type(exc).__name__}: {exc}")
    write_outputs(output_dir, summary, normalized, bundle_results)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("id"),
        "goal": row.get("instruction"),
        "app_type": row.get("app_type"),
        "app_family": row.get("app_family"),
        "platform": row.get("platform"),
        "difficulty": row.get("difficulty"),
        "archive_member": row.get("archive_member"),
        "task_json_member": row.get("task_json_member"),
        "reward_member": row.get("reward_member"),
        "setup_file_members": parse_jsonish_list(row.get("setup_file_members")),
    }


def parse_jsonish_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


def download_artifact(output_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id="xlangai/CUA-Gym",
            repo_type="dataset",
            filename="artifacts/cua_gym_tasks_v1.tar.zst",
            local_dir=str(output_dir / "hf_cache"),
        )
    )


def decompressed_tar_path(archive: Path) -> Path:
    if archive.suffix != ".zst":
        return archive
    tar_path = archive.with_suffix("")
    if tar_path.exists() and tar_path.stat().st_size > 0:
        return tar_path
    import zstandard as zstd

    with archive.open("rb") as src, tar_path.open("wb") as dst:
        reader = zstd.ZstdDecompressor().stream_reader(src)
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return tar_path


def inspect_bundles(tar_path: Path, rows: list[dict[str, Any]], extract_dir: Path) -> list[dict[str, Any]]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with tarfile.open(tar_path) as tar:
        members_by_name = {member.name: member for member in tar.getmembers()}
        for row in rows:
            archive_member = str(row.get("archive_member"))
            prefix = archive_member.rstrip("/") + "/"
            wanted = [member for name, member in members_by_name.items() if name.startswith(prefix)]
            safe_extract(tar, wanted, extract_dir)
            task_dir = extract_dir / archive_member
            task_json = task_dir / "task.json"
            reward_py = task_dir / "reward.py"
            item = {
                "task_id": row.get("id"),
                "app_type": row.get("app_type"),
                "difficulty": row.get("difficulty"),
                "task_json_exists": task_json.exists(),
                "reward_py_exists": reward_py.exists(),
                "reward_py_compile_ok": False,
                "archive_member": archive_member,
            }
            if task_json.exists():
                try:
                    item["task_json_keys"] = sorted(json.loads(task_json.read_text(encoding="utf-8")).keys())
                except Exception as exc:
                    item["task_json_error"] = f"{type(exc).__name__}: {exc}"
            if reward_py.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(reward_py)],
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                item["reward_py_compile_ok"] = result.returncode == 0
                if result.returncode != 0:
                    item["reward_py_compile_error"] = (result.stderr or result.stdout)[-1000:]
            results.append(item)
    return results


def safe_extract(tar: tarfile.TarFile, members: list[tarfile.TarInfo], target_dir: Path) -> None:
    target_root = target_dir.resolve()
    safe_members: list[tarfile.TarInfo] = []
    for member in members:
        resolved = (target_dir / member.name).resolve()
        if target_root not in resolved.parents and resolved != target_root:
            raise RuntimeError(f"unsafe archive member path: {member.name}")
        safe_members.append(member)
    tar.extractall(target_dir, members=safe_members)


def write_outputs(output_dir: Path, summary: dict[str, Any], normalized: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "mock_web_tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized),
        encoding="utf-8",
    )
    (output_dir / "bundle_inspection.json").write_text(json.dumps(bundles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "sample_count": len(normalized), "bundle_count": len(bundles)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
