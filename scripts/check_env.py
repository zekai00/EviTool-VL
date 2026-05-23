#!/usr/bin/env python3
"""Check the minimum runtime environment for EviTool-VL."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "datasets",
    "trl",
    "PIL",
    "cv2",
    "numpy",
    "pandas",
]


def run_command(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return "not found"
    return completed.stdout.strip()


def check_packages() -> bool:
    ok = True
    print("Python:", sys.version.replace("\n", " "))
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
        except Exception as exc:
            ok = False
            print(f"[missing] {package}: {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"[ok] {package}: {version}")
    return ok


def check_torch() -> bool:
    try:
        import torch
    except Exception as exc:
        print(f"[missing] torch: {exc}")
        return False

    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / 1024**3
        print(f"GPU {idx}: {props.name}, {mem_gb:.1f} GB")
    return torch.cuda.is_available() and torch.cuda.device_count() >= 1


def check_paths() -> None:
    model_dir = Path(os.environ.get("EVITOOL_MODEL_DIR", "/root/models"))
    print("Model directory:", model_dir)
    print("Model directory exists:", model_dir.exists())
    print("Project root:", Path.cwd())
    print("git:", shutil.which("git") or "not found")
    print("nvidia-smi:")
    print(run_command(["nvidia-smi"]))


def main() -> int:
    packages_ok = check_packages()
    torch_ok = check_torch()
    check_paths()
    return 0 if packages_ok and torch_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
