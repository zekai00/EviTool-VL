#!/usr/bin/env python3
"""Evaluate candidate-constrained model scores.

This evaluator does not generate free-form text.  It scores a finite list of
candidate-id completions, picks the highest-probability id, and then computes
the same GUI selection metrics.  This directly measures the CC-GRPO policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.candidate_constrained import build_action_ids, build_cc_prompt, compute_candidate_completion_logprobs
from rl.gui_candidate_env import score_candidate_action_v2

BASELINE_PATH = PROJECT_ROOT / "eval" / "eval_baseline.py"
spec = importlib.util.spec_from_file_location("eval_baseline", BASELINE_PATH)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None, help="Default scores all candidates. Set for debug speed.")
    parser.add_argument("--score-batch-size", type=int, default=12, help="Chunk candidate scoring to avoid eval-time OOM.")
    parser.add_argument("--use-cc-action-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_records(path: str | Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def image_path_for_row(row: dict[str, Any], use_overlay: bool) -> Path:
    image_field = row.get("overlay_image") if use_overlay and row.get("overlay_image") else row.get("image_path") or row.get("image")
    path = Path(str(image_field))
    return path if path.is_absolute() else PROJECT_ROOT / path


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [result["reward"] for result in results]
    return {
        "count": len(results),
        "avg_reward": mean([float(reward["total"]) for reward in rewards]),
        "valid_rate": mean([float(reward["valid"]) for reward in rewards]),
        "pointing_rate": mean([float(reward["pointing"]) for reward in rewards]),
        "iou50_rate": mean([float(reward["iou_50"]) for reward in rewards]),
        "avg_iou": mean([float(reward["metrics"]["iou"]) for reward in rewards]),
        "avg_selected_rank": mean([float(reward["selected_rank"]) for reward in rewards if reward.get("selected_rank") is not None]),
        "avg_candidate_count": mean([float(result["candidate_count"]) for result in results]),
        "avg_latency_sec": mean([float(result["latency_sec"]) for result in results]),
    }


def score_candidates_in_chunks(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    action_ids: list[str],
    *,
    chunk_size: int,
) -> torch.Tensor:
    """Score a long candidate list in smaller chunks.

    Training intentionally scores one compact action set at once.  Evaluation on
    all candidates can be much larger, so chunking keeps memory bounded while
    preserving the same candidate log-probability definition.
    """
    chunks: list[torch.Tensor] = []
    for start in range(0, len(action_ids), max(1, chunk_size)):
        chunk_ids = action_ids[start : start + max(1, chunk_size)]
        chunk_scores = compute_candidate_completion_logprobs(
            model,
            processor,
            image_path,
            prompt,
            chunk_ids,
            require_grad=False,
        )
        chunks.append(chunk_scores.detach().cpu())
    return torch.cat(chunks, dim=0)


def write_report(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# GUI Candidate Score Eval",
        "",
        f"- Data: `{args.data}`",
        f"- Model: `{args.model}`",
        f"- Adapter: `{args.adapter if args.adapter else '-'}`",
        f"- Use cc_action_ids: `{args.use_cc_action_ids}`",
        f"- Max actions: `{args.max_actions if args.max_actions else 'all'}`",
        f"- Samples: {summary['count']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Avg reward(v2 parser metric) | {num(summary['avg_reward'])} |",
        f"| Valid candidate | {pct(summary['valid_rate'])} |",
        f"| Pointing | {pct(summary['pointing_rate'])} |",
        f"| IoU@0.5 | {pct(summary['iou50_rate'])} |",
        f"| Avg IoU | {num(summary['avg_iou'])} |",
        f"| Avg selected rank | {num(summary['avg_selected_rank'])} |",
        f"| Avg candidate count | {num(summary['avg_candidate_count'])} |",
        f"| Avg latency(s) | {num(summary['avg_latency_sec'])} |",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_records(args.data, args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, max_pixels=args.image_max_pixels)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = baseline.load_model(args.model, args.adapter)
    model.eval()

    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            candidates = row.get("candidates") or []
            if args.use_cc_action_ids:
                action_ids = [str(candidate_id) for candidate_id in row.get("cc_action_ids") or []]
                if not action_ids:
                    action_ids = build_action_ids(row, candidates, max_actions=args.max_actions or 12, seed=args.seed)
            else:
                action_ids = [str(candidate["candidate_id"]) for candidate in candidates]
            if args.max_actions is not None:
                action_ids = action_ids[: args.max_actions]
            started = time.time()
            if action_ids:
                with torch.inference_mode():
                    scores = score_candidates_in_chunks(
                        model,
                        processor,
                        image_path_for_row(row, args.use_overlay),
                        build_cc_prompt(row, candidates),
                        action_ids,
                        chunk_size=args.score_batch_size,
                    )
                    probs = torch.softmax(scores.float() / max(args.policy_temperature, 1e-6), dim=0)
                    best_idx = int(torch.argmax(probs).item())
                candidate_id = action_ids[best_idx]
                prediction = {"candidate_id": candidate_id}
                score_map = {candidate_id: round(float(score), 6) for candidate_id, score in zip(action_ids, scores.tolist(), strict=False)}
            else:
                candidate_id = None
                prediction = {"candidate_id": "c00"}
                score_map = {}
            latency = time.time() - started
            reward = score_candidate_action_v2(row, candidates, prediction)
            result = {
                "id": row.get("id"),
                "candidate_count": len(candidates),
                "action_count": len(action_ids),
                "prediction": prediction,
                "candidate_id": candidate_id,
                "reward": reward,
                "scores": score_map,
                "latency_sec": round(latency, 4),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            results.append(result)
            print(f"[{idx}/{len(rows)}] {row.get('id')} -> {prediction} reward={reward['total']}", flush=True)

    summary = summarize(results)
    summary.update({"model": args.model, "adapter": args.adapter})
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(Path(args.report), args, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
