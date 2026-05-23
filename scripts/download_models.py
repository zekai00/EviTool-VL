#!/usr/bin/env python3
"""Download the starting VLM checkpoints to /root/models by default.

The downloader tries Hugging Face first and falls back to ModelScope. Use
--source modelscope when Hugging Face is slow or unavailable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download as hf_snapshot_download


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("EVITOOL_MODEL_DIR", "/root/models"),
        help="Directory used for model snapshots.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model repo id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint or mirror.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "hf", "modelscope"],
        default="auto",
        help="Download source. auto tries Hugging Face first, then ModelScope.",
    )
    return parser.parse_args()


def local_model_dir(model_dir: Path, repo_id: str) -> Path:
    return model_dir / repo_id.split("/")[-1]


def download_hf(repo_id: str, local_dir: Path, endpoint: str | None) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    hf_snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_modelscope(repo_id: str, local_dir: Path) -> None:
    from modelscope import snapshot_download as ms_snapshot_download

    ms_snapshot_download(repo_id, local_dir=str(local_dir))


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    models = args.model or DEFAULT_MODELS
    for repo_id in models:
        local_dir = local_model_dir(model_dir, repo_id)
        if (local_dir / "config.json").exists() and any(local_dir.glob("*.safetensors")):
            print(f"Already exists: {local_dir}")
            continue

        print(f"Downloading {repo_id} -> {local_dir} [{args.source}]")
        if args.source == "hf":
            download_hf(repo_id, local_dir, args.endpoint)
        elif args.source == "modelscope":
            download_modelscope(repo_id, local_dir)
        else:
            try:
                download_hf(repo_id, local_dir, args.endpoint)
            except Exception as exc:
                print(f"Hugging Face download failed for {repo_id}: {exc}")
                print("Falling back to ModelScope...")
                download_modelscope(repo_id, local_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
