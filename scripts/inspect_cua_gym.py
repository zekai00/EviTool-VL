#!/usr/bin/env python3
"""Inspect CUA-Gym metadata and task bundles without running untrusted rewards."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--download-artifact", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "repo": "xlangai/CUA-Gym",
        "metadata_rows": 0,
        "inspected_bundles": 0,
        "reward_py_compile_ok": 0,
        "blocked": False,
        "blockers": [],
    }
    rows: list[dict] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("xlangai/CUA-Gym", "tasks", split="train")
        summary["metadata_rows"] = len(ds)
        for row in ds.select(range(min(args.limit, len(ds)))):
            rows.append(dict(row))
    except Exception as exc:
        summary["blocked"] = True
        summary["blockers"].append(f"metadata_load_failed: {type(exc).__name__}: {exc}")
        _write(output_dir, summary, rows, [])
        return

    bundles: list[dict] = []
    if args.download_artifact:
        try:
            from huggingface_hub import hf_hub_download

            archive = hf_hub_download(
                repo_id="xlangai/CUA-Gym",
                repo_type="dataset",
                filename="artifacts/cua_gym_tasks_v1.tar.zst",
                local_dir=str(output_dir / "hf_cache"),
            )
            extract_dir = output_dir / "bundles"
            extract_dir.mkdir(parents=True, exist_ok=True)
            names = _list_archive_members(Path(archive))
            target_dirs = _target_dirs(rows, names)
            _extract_members(Path(archive), extract_dir, target_dirs)
            for row in rows:
                member = row.get("archive_member")
                task_dir = extract_dir / str(member)
                task_json = task_dir / "task.json"
                reward_py = task_dir / "reward.py"
                item = {
                    "task_id": row.get("task_id") or row.get("id"),
                    "archive_member": member,
                    "task_json_exists": task_json.exists(),
                    "reward_py_exists": reward_py.exists(),
                    "reward_py_compile_ok": False,
                    "instruction": row.get("instruction"),
                    "app_type": row.get("app_type"),
                    "difficulty": row.get("difficulty"),
                }
                if task_json.exists():
                    try:
                        task_data = json.loads(task_json.read_text(encoding="utf-8"))
                        item["task_json_keys"] = sorted(task_data.keys())
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
                bundles.append(item)
            summary["inspected_bundles"] = len(bundles)
            summary["reward_py_compile_ok"] = sum(1 for item in bundles if item["reward_py_compile_ok"])
        except Exception as exc:
            summary["blocked"] = True
            summary["blockers"].append(f"artifact_inspection_failed: {type(exc).__name__}: {exc}")
    _write(output_dir, summary, rows, bundles)


def _list_archive_members(archive: Path) -> list[str]:
    result = subprocess.run(
        ["tar", "--zstd", "-tf", str(archive)],
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode == 0:
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    tar_path = _decompressed_tar_path(archive)
    with tarfile.open(tar_path) as tar:
        return tar.getnames()


def _target_dirs(rows: list[dict], names: list[str]) -> list[str]:
    dirs = []
    name_set = set(names)
    for row in rows:
        member = str(row.get("archive_member") or "").rstrip("/")
        if not member:
            continue
        prefix = member + "/"
        if any(name.startswith(prefix) for name in name_set):
            dirs.append(member)
    return dirs


def _extract_members(archive: Path, extract_dir: Path, dirs: list[str]) -> None:
    wanted: list[str] = []
    all_members = _list_archive_members(archive)
    for dirname in dirs:
        prefix = dirname.rstrip("/") + "/"
        wanted.extend([name for name in all_members if name.startswith(prefix)])
    if not wanted:
        return
    cmd = ["tar", "--zstd", "-xf", str(archive), "-C", str(extract_dir), *wanted]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    if result.returncode == 0:
        return
    tar_path = _decompressed_tar_path(archive)
    with tarfile.open(tar_path) as tar:
        members = [member for member in tar.getmembers() if member.name in set(wanted)]
        tar.extractall(extract_dir, members=members)


def _decompressed_tar_path(archive: Path) -> Path:
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


def _write(output_dir: Path, summary: dict, rows: list[dict], bundles: list[dict]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "metadata_sample.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "bundle_inspection.json").write_text(json.dumps(bundles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "bundle_inspection": bundles}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
