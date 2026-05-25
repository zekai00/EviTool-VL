"""Candidate-constrained policy helpers for GUI grounding RL.

The first GUI GRPO trial treated candidate selection as free-form text
generation.  That wastes RL capacity on JSON formatting and often gives zero
reward variance.  This module keeps the VLM interface intact, but constrains
the action space to the finite candidate ids already present in each sample.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info

from rl.gui_candidate_env import (
    build_candidate_prompt,
    candidate_metrics,
    center_distance_score,
)


def candidate_completion(candidate_id: str) -> str:
    """Return the fixed completion used to score one discrete candidate action.

    Candidate ids are still represented as the model's native output text, but
    the training algorithm only compares this finite set of strings instead of
    asking the model to explore arbitrary JSON completions.
    """
    return json.dumps({"candidate_id": candidate_id}, separators=(",", ":"))


def build_cc_prompt(row: dict[str, Any], candidates: list[dict[str, Any]] | None = None) -> str:
    """Build the shared prompt for candidate scoring.

    The model sees the same overlay image and candidate list as the old GRPO
    pipeline.  Only the action selection changes: candidate completions are
    scored teacher-forced one by one and normalized into a discrete policy.
    """
    return (
        f"{build_candidate_prompt(row, candidates or row.get('candidates') or [])}\n\n"
        "The image is annotated with candidate ids. Select the candidate that best matches the instruction. "
        "Return only JSON. Do not explain."
    )


def candidate_reward_v3(row: dict[str, Any], candidate: dict[str, Any] | None) -> dict[str, Any]:
    """Score a candidate action without JSON/validity components.

    In candidate-constrained RL every sampled action is a valid candidate id, so
    format and validity rewards would be constant noise.  V3 keeps only geometry
    and selection-quality terms that can rank candidates within the same prompt.
    """
    metrics = candidate_metrics(row, candidate)
    iou_shaped = min(1.0, float(metrics["iou"]) / 0.5)
    center_shaped = center_distance_score(row.get("answer_bbox"), candidate)
    rank = int(candidate.get("rank") or 0) if candidate else 0
    top1_wrong_penalty = 0.05 if rank == 1 and not metrics["hit"] else 0.0
    total = (
        0.35 * float(metrics["pointing"])
        + 0.30 * float(metrics["iou_50"])
        + 0.20 * iou_shaped
        + 0.15 * center_shaped
        - top1_wrong_penalty
    )
    return {
        "total": round(float(total), 6),
        "pointing": round(float(metrics["pointing"]), 6),
        "iou_50": round(float(metrics["iou_50"]), 6),
        "iou_shaped": round(float(iou_shaped), 6),
        "center_shaped": round(float(center_shaped), 6),
        "top1_wrong_penalty": round(float(top1_wrong_penalty), 6),
        "metrics": metrics,
    }


def candidate_reward_table(row: dict[str, Any], candidates: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Return candidate-id -> reward_v3 for all candidates in a row."""
    table: dict[str, float] = {}
    for candidate in candidates or row.get("candidates") or []:
        table[str(candidate["candidate_id"])] = float(candidate_reward_v3(row, candidate)["total"])
    return table


def rank_bucket(rank: int | None) -> str:
    """Bucket oracle rank so training can monitor early/middle/late bias."""
    if rank is None or rank <= 0:
        return "unknown"
    if rank <= 5:
        return "early"
    if rank <= 20:
        return "middle"
    return "late"


def rank_balanced_indices(rows: list[dict[str, Any]], *, seed: int, epoch: int = 0) -> list[int]:
    """Return row indices interleaved by oracle-rank bucket.

    The previous candidate policy overused early candidates such as `c00`.  A
    plain shuffled dataloader can still over-represent one oracle-rank bucket in
    short runs, so this helper round-robins early/middle/late rows after
    shuffling each bucket.  It is intentionally deterministic per epoch.
    """
    buckets: dict[str, list[int]] = {"early": [], "middle": [], "late": [], "unknown": []}
    for index, row in enumerate(rows):
        bucket = str(row.get("oracle_rank_bucket") or rank_bucket(row.get("oracle_rank")))
        if bucket not in buckets:
            bucket = "unknown"
        buckets[bucket].append(index)

    rng = random.Random(f"rank-balanced:{seed}:{epoch}")
    for indices in buckets.values():
        rng.shuffle(indices)

    order: list[int] = []
    active = ["early", "middle", "late", "unknown"]
    while any(buckets[name] for name in active):
        for name in active:
            if buckets[name]:
                order.append(buckets[name].pop())
    return order


def _candidate_by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(candidate["candidate_id"]): candidate for candidate in candidates}


def build_hard_negative_ids(
    row: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    *,
    oracle_id: str | None = None,
    max_hard: int = 8,
    seed: int = 42,
) -> list[str]:
    """Select diverse hard negatives for one candidate-selection prompt.

    GRPO needs reward differences inside a group.  This selector deliberately
    mixes several wrong-candidate types: detector-top candidates, near-center
    misses, high-IoU-but-not-enough misses, and random wrong candidates.
    """
    cand_list = list(candidates or row.get("candidates") or [])
    if not cand_list:
        return []
    by_id = _candidate_by_id(cand_list)
    reward_table = candidate_reward_table(row, cand_list)
    oracle_id = oracle_id or row.get("oracle_candidate_id")
    if not oracle_id or oracle_id not in by_id:
        oracle_id = max(reward_table, key=lambda cid: reward_table[cid])

    selected: list[str] = []
    seen = {oracle_id}

    def add(candidate_id: str | None) -> None:
        if candidate_id and candidate_id in by_id and candidate_id not in seen and len(selected) < max_hard:
            selected.append(candidate_id)
            seen.add(candidate_id)

    wrong = [candidate for candidate in cand_list if candidate["candidate_id"] != oracle_id]

    # 1) Front-ranked wrong candidates expose the model's c00/c01 shortcut.
    for candidate in sorted(wrong, key=lambda item: int(item.get("rank") or 9999)):
        metrics = candidate_metrics(row, candidate)
        if not metrics["hit"]:
            add(candidate["candidate_id"])
        if len(selected) >= max_hard:
            return selected

    # 2) High-IoU misses are visually plausible but still not good enough.
    for candidate in sorted(wrong, key=lambda item: candidate_metrics(row, item)["iou"], reverse=True):
        metrics = candidate_metrics(row, candidate)
        if not metrics["iou_50"]:
            add(candidate["candidate_id"])
        if len(selected) >= max_hard:
            return selected

    # 3) Near-center misses teach the selector fine spatial discrimination.
    for candidate in sorted(wrong, key=lambda item: center_distance_score(row.get("answer_bbox"), item), reverse=True):
        metrics = candidate_metrics(row, candidate)
        if not metrics["hit"]:
            add(candidate["candidate_id"])
        if len(selected) >= max_hard:
            return selected

    # 4) Random wrongs keep the policy from overfitting only one negative type.
    rng = random.Random(f"{seed}:{row.get('id')}")
    shuffled = wrong[:]
    rng.shuffle(shuffled)
    for candidate in shuffled:
        add(candidate["candidate_id"])
        if len(selected) >= max_hard:
            return selected
    return selected


def build_action_ids(
    row: dict[str, Any],
    candidates: list[dict[str, Any]] | None = None,
    *,
    max_actions: int = 12,
    max_hard: int = 8,
    seed: int = 42,
) -> list[str]:
    """Build the finite action subset used by CC-GRPO for one row.

    The full candidate list may contain up to 60 boxes.  For each optimization
    step we use a compact action set containing the oracle plus hard negatives,
    which keeps GPU memory manageable and guarantees useful reward contrast.
    """
    cand_list = list(candidates or row.get("candidates") or [])
    if not cand_list:
        return []
    reward_table = candidate_reward_table(row, cand_list)
    oracle_id = row.get("oracle_candidate_id")
    if not oracle_id or oracle_id not in reward_table:
        oracle_id = max(reward_table, key=lambda cid: reward_table[cid])

    action_ids: list[str] = [str(oracle_id)]
    for candidate_id in build_hard_negative_ids(row, cand_list, oracle_id=str(oracle_id), max_hard=max_hard, seed=seed):
        if candidate_id not in action_ids:
            action_ids.append(candidate_id)

    # Fill remaining slots with deterministic random candidates so every sample
    # has enough legal actions even when hard-negative categories are sparse.
    rng = random.Random(f"actions:{seed}:{row.get('id')}")
    remaining = [candidate["candidate_id"] for candidate in cand_list if candidate["candidate_id"] not in action_ids]
    rng.shuffle(remaining)
    for candidate_id in remaining:
        if len(action_ids) >= max_actions:
            break
        action_ids.append(str(candidate_id))
    return action_ids[:max_actions]


def action_reward_std(action_ids: list[str], reward_table: dict[str, float]) -> float:
    values = torch.tensor([float(reward_table[cid]) for cid in action_ids if cid in reward_table], dtype=torch.float32)
    return float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0


def build_messages(image_path: str | Path, prompt: str) -> list[dict[str, Any]]:
    """Build Qwen-VL style multimodal messages for one overlay image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _repeat_vision_inputs(values: Any, count: int) -> Any:
    if values is None:
        return None
    if isinstance(values, list):
        return values * count if len(values) <= 1 else values
    return values


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def compute_candidate_completion_logprobs(
    model: torch.nn.Module,
    processor: Any,
    image_path: str | Path,
    prompt: str,
    candidate_ids: list[str],
    *,
    require_grad: bool,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Teacher-force each candidate completion and return log-prob scores.

    The returned tensor has shape `[num_candidates]`.  It is differentiable when
    `require_grad=True`, so training can optimize the model's relative
    probabilities over candidate ids without ever sampling arbitrary text.
    """
    if not candidate_ids:
        raise ValueError("candidate_ids must be non-empty")

    messages = build_messages(image_path, prompt)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    completions = [candidate_completion(candidate_id) for candidate_id in candidate_ids]
    full_texts = [prompt_text + completion for completion in completions]

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())

    inputs = processor(
        text=full_texts,
        images=_repeat_vision_inputs(image_inputs, len(full_texts)),
        videos=_repeat_vision_inputs(video_inputs, len(full_texts)),
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = _move_batch_to_device(inputs, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    completion_lengths = (attention_mask.sum(dim=-1) - prompt_len).clamp_min(1)
    max_completion_len = int(completion_lengths.max().item())

    with torch.set_grad_enabled(require_grad):
        # Qwen2.5-VL supports `logits_to_keep`.  Keeping only completion-token
        # logits avoids materializing `[batch, prompt_len, vocab]`, which was the
        # main memory spike in dual-card CC-GRPO.  We keep `L + 1` positions
        # because the first completion token is predicted by the hidden state
        # immediately before the completion starts.
        outputs = model(**inputs, logits_to_keep=max_completion_len + 1)
        logits = outputs.logits[:, :max_completion_len, :].float()
        target_labels = labels[:, prompt_len : prompt_len + max_completion_len]
        token_positions = torch.arange(max_completion_len, device=attention_mask.device).unsqueeze(0)
        token_mask = token_positions < completion_lengths.unsqueeze(1)
        target_labels = target_labels.masked_fill(~token_mask, -100)
        safe_labels = target_labels.masked_fill(~token_mask, 0)
        token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * token_mask
        sequence_logprobs = token_logprobs.sum(dim=-1)
        if length_normalize:
            lengths = token_mask.sum(dim=-1).clamp_min(1)
            sequence_logprobs = sequence_logprobs / lengths
    return sequence_logprobs
