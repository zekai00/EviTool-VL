#!/usr/bin/env python3
"""Small-scale verifier-guided on-policy RL for browser GUI actions.

This is the first direct-action browser RL path:

1. Use the current local Qwen-VL policy to sample several actions for the same
   screenshot state.
2. Replay the same browser state and let the verifier score each sampled action.
3. Apply a group-relative policy-gradient update to the current LoRA adapter.

It is intentionally small and deterministic enough for smoke experiments.  The
environment is local Playwright HTML tasks, so branch evaluation is implemented
by reset + replay of the committed action prefix.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.browser_rl import LocalQwenPolicy, PlaywrightBrowserEnv, load_tasks
from envs.browser_rl.actions import pixels_to_normalized
from envs.browser_rl.qwen_policy import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/train_tasks.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--shuffle-tasks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-groups", type=int, default=24)
    parser.add_argument("--target-trainable-groups", type=int, default=None)
    parser.add_argument("--train-max-groups", type=int, default=0, help="Optional cap for trainable groups when training from an existing groups.jsonl.")
    parser.add_argument(
        "--family-quotas-json",
        default=None,
        help='JSON object of target trainable groups per family, e.g. {"form":20,"search":20}.',
    )
    parser.add_argument(
        "--template-quotas-json",
        default=None,
        help='JSON object of target trainable groups per task template, e.g. {"advanced_scroll":20}.',
    )
    parser.add_argument("--include-families", default=None, help="Comma-separated family allowlist before shuffle/limit.")
    parser.add_argument("--include-templates", default=None, help="Comma-separated template allowlist before shuffle/limit.")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=3)
    parser.add_argument("--dedupe-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-sample-attempts", type=int, default=8)
    parser.add_argument(
        "--continue-on-single-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue the rollout when a state yields only one unique VLM action; no GRPO group is created for that state.",
    )
    parser.add_argument(
        "--inject-scroll-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly add rule-based scroll candidates for advanced_scroll exploration. Off by default to keep collection VLM-sampled.",
    )
    parser.add_argument(
        "--inject-target-click-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly add DOM-center target click candidates after target_visible. Off by default to avoid leaking verifier/DOM priors.",
    )
    parser.add_argument(
        "--scroll-candidate-dys",
        default="600,900,1200",
        help="Comma-separated dy values injected as verifier-guided exploration candidates for advanced_scroll.",
    )
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-history", type=int, default=4)
    parser.add_argument("--prompt-style", default="full", choices=["full", "sft_minimal"])
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--commit-strategy", choices=["first", "best"], default="first")
    parser.add_argument("--invalid-penalty", type=float, default=0.2)
    parser.add_argument("--exec-error-penalty", type=float, default=0.2)
    parser.add_argument("--success-bonus", type=float, default=0.0)
    parser.add_argument("--step-cost", type=float, default=0.0)
    parser.add_argument(
        "--target-distance-reward-weight",
        type=float,
        default=0.0,
        help="Extra reward for click proximity to visible #target. Default 0 keeps previous reward behavior.",
    )
    parser.add_argument(
        "--target-distance-reward-sigma",
        type=float,
        default=80.0,
        help="Gaussian sigma in 0-1000 normalized coordinate units for target distance reward.",
    )
    parser.add_argument(
        "--target-distance-reward-templates",
        default="advanced_scroll",
        help="Comma-separated task templates that may receive target distance reward.",
    )
    parser.add_argument(
        "--repeat-click-penalty",
        type=float,
        default=0.0,
        help="Penalty when a click repeats a previous click within repeat-click-radius. Default 0 disables it.",
    )
    parser.add_argument(
        "--repeat-click-radius",
        type=float,
        default=25.0,
        help="Normalized coordinate radius used by repeat-click-penalty.",
    )
    parser.add_argument("--min-reward-std", type=float, default=1e-6)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--logprob-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--grpo-loss-type", choices=["vanilla", "clipped"], default="vanilla")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--old-logprob-key", default="old_logprob")
    parser.add_argument("--reference-logprob-key", default="reference_logprob")
    parser.add_argument(
        "--skip-logprob-cache",
        action="store_true",
        help="Do not precompute old/reference logprobs. Only valid for vanilla loss or pre-cached groups.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--replay-sft-jsonl", default=None, help="History-aware SFT JSONL used as a conservative replay anchor.")
    parser.add_argument("--replay-loss-weight", type=float, default=0.0)
    parser.add_argument("--replay-ratio", type=float, default=0.0, help="Expected replay SFT rows per trainable RL group.")
    parser.add_argument("--replay-max-rows", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--stream-collect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream-flush-every", type=int, default=5)
    parser.add_argument("--resume-collect", action="store_true", help="Resume collection from groups.jsonl.tmp and rollouts.jsonl.tmp if present.")
    parser.add_argument("--train-only-from", default=None, help="Existing groups.jsonl to train from without collecting.")
    parser.add_argument("--dry-run", action="store_true", help="Validate task replay and verifier scoring without loading Qwen.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_only_from:
        groups = read_jsonl(Path(args.train_only_from))
        rollouts: list[dict[str, Any]] = []
        collect_summary: dict[str, Any] = {"mode": "train_only", "groups": len(groups)}
    elif args.dry_run:
        groups, rollouts, collect_summary = dry_run_collect(args, output_dir)
    else:
        groups, rollouts, collect_summary = collect_onpolicy_groups(args, output_dir)

    write_jsonl(output_dir / "groups.jsonl", groups)
    write_jsonl(output_dir / "rollouts.jsonl", rollouts)
    (output_dir / "collect_summary.json").write_text(
        json.dumps(collect_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.dry_run or args.collect_only:
        summary = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "dry_run" if args.dry_run else "collect_only",
            "collect": collect_summary,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    train_summary = train_group_relative_policy(args, output_dir, groups)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "collect_and_train" if not args.train_only_from else "train_only",
        "collect": collect_summary,
        "train": train_summary,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def dry_run_collect(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks = load_task_subset(args, limit=min(2, args.limit))
    groups: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    with PlaywrightBrowserEnv(output_dir / "artifacts", headless=args.headless, reuse_context=True) as env:
        for task_index, task in enumerate(tasks):
            prefix: list[dict[str, Any]] = []
            obs, replay = reset_and_replay(env, task, prefix, screenshot_prefix=f"dry_state_{task.task_id}_0")
            samples: list[dict[str, Any]] = []
            candidates = []
            if task.oracle_actions:
                candidates.append(("oracle", task.oracle_actions[0]))
            candidates.append(("wait", {"action": "wait"}))
            for sample_index, (label, action) in enumerate(candidates):
                branch = evaluate_branch(
                    env,
                    task,
                    prefix,
                    action,
                    policy_info={"policy": f"dry_{label}", "valid_json": True, "valid_action": True, "raw_text": json.dumps(action)},
                    screenshot_prefix=f"dry_branch_{task.task_id}_{sample_index}",
                    args=args,
                )
                samples.append(branch)
            group = build_group(task, obs, samples, group_index=task_index, prefix_actions=prefix, committed_index=0, args=args)
            groups.append(group)
            rollouts.append(
                {
                    "rollout_id": f"dry_{task.task_id}",
                    "policy_version": "dry_run",
                    "task_id": task.task_id,
                    "goal": task.goal,
                    "success": bool(samples and samples[0].get("success")),
                    "total_reward": float(samples[0].get("env_reward", 0.0) if samples else 0.0),
                    "num_steps": 1,
                    "trajectory": [],
                    "final_info": {"replay": replay},
                }
            )
    return groups, rollouts, summarize_groups(groups, rollouts, extra={"dry_run": True})


def collect_onpolicy_groups(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks = load_task_subset(args, limit=args.limit)
    family_quotas = parse_family_quotas(args.family_quotas_json)
    template_quotas = parse_family_quotas(args.template_quotas_json)
    target_trainable_groups = args.target_trainable_groups
    if (family_quotas or template_quotas) and target_trainable_groups is None:
        target_trainable_groups = max(sum(family_quotas.values()), sum(template_quotas.values()))
    trainable_by_family: Counter[str] = Counter()
    trainable_by_template: Counter[str] = Counter()
    groups: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    stream = CollectStreamWriter(output_dir, enabled=args.stream_collect, resume=args.resume_collect)
    if args.resume_collect:
        groups = stream.load_groups()
        rollouts = stream.load_rollouts()
        for group in groups:
            if group.get("trainable"):
                trainable_by_family[str(group.get("family") or "unknown")] += 1
                trainable_by_template[str(group.get("template") or "unknown")] += 1
    completed_task_ids = {str(row.get("task_id")) for row in rollouts if row.get("task_id")}
    completed_task_ids.update(str(group.get("task_id")) for group in groups if group.get("task_id"))
    if collection_target_reached(target_trainable_groups, trainable_by_family, trainable_by_template, family_quotas, template_quotas):
        stream.write_summary(groups, rollouts, extra=live_collect_extra(args, len(groups), len(rollouts), completed=True))
        return groups, rollouts, summarize_groups(
            groups,
            rollouts,
            extra=live_collect_extra(args, len(groups), len(rollouts), completed=True),
        )
    policy = LocalQwenPolicy(
        model_path=args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.sampling_temperature,
        max_history=args.max_history,
        prompt_style=args.prompt_style,
        system_prompt=args.system_prompt,
        image_max_pixels=args.image_max_pixels,
    )
    resumed_groups = len(groups)
    resumed_rollouts = len(rollouts)
    with PlaywrightBrowserEnv(output_dir / "artifacts", headless=args.headless, reuse_context=True) as env:
        stop_collection = collection_target_reached(target_trainable_groups, trainable_by_family, trainable_by_template, family_quotas, template_quotas)
        for task in tasks:
            if stop_collection:
                break
            if str(task.task_id) in completed_task_ids:
                continue
            family = str((task.metadata or {}).get("family") or task.template)
            template = str(task.template)
            if family_quotas and trainable_by_family[family] >= family_quotas.get(family, 0):
                continue
            if template_quotas and trainable_by_template[template] >= template_quotas.get(template, 0):
                continue
            prefix: list[dict[str, Any]] = []
            trajectory: list[dict[str, Any]] = []
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            task_success = False
            max_steps = min(int(task.max_steps), args.max_steps)
            for step_index in range(max_steps):
                group_index = len(groups)
                obs, replay = reset_and_replay(
                    env,
                    task,
                    prefix,
                    screenshot_prefix=f"state_{task.task_id}_{step_index}_{group_index}",
                )
                if replay.get("terminated") or replay.get("truncated"):
                    final_info = replay.get("final_info") or {}
                    task_success = bool(((final_info.get("verifier") or {}).get("success")))
                    break

                samples: list[dict[str, Any]] = []
                seen_actions: set[str] = set()
                attempts = 0
                while len(samples) < args.num_generations and attempts < max(args.num_generations, args.max_sample_attempts):
                    sample_index = attempts
                    attempts += 1
                    result = policy.act(obs)
                    key = action_key(result.action)
                    if args.dedupe_actions and key in seen_actions:
                        continue
                    seen_actions.add(key)
                    branch = evaluate_branch(
                        env,
                        task,
                        prefix,
                        result.action,
                        policy_info=result.info,
                        screenshot_prefix=f"branch_{task.task_id}_{step_index}_{group_index}_{sample_index}",
                        args=args,
                    )
                    samples.append(branch)
                for injected_index, action in enumerate(injected_candidate_actions(env, task, obs, seen_actions, args)):
                    branch = evaluate_branch(
                        env,
                        task,
                        prefix,
                        action,
                        policy_info=injected_policy_info(action, source="scroll_candidate"),
                        screenshot_prefix=f"branch_{task.task_id}_{step_index}_{group_index}_inject{injected_index}",
                        args=args,
                    )
                    samples.append(branch)
                if not samples:
                    final_info = {"error": "not_enough_unique_samples", "sample_count": len(samples), "attempts": attempts}
                    break
                if len(samples) < 2 and not args.continue_on_single_sample:
                    final_info = {"error": "not_enough_unique_samples", "sample_count": len(samples), "attempts": attempts}
                    break

                committed_index = choose_commit_index(samples, args.commit_strategy)
                group: dict[str, Any] | None = None
                if len(samples) >= 2:
                    group = build_group(task, obs, samples, group_index=group_index, prefix_actions=prefix, committed_index=committed_index, args=args)
                    groups.append(group)
                    stream.append_group(group)
                    if group.get("trainable"):
                        trainable_by_family[str(group.get("family") or "unknown")] += 1
                        trainable_by_template[str(group.get("template") or "unknown")] += 1
                    stream.maybe_write_summary(
                        groups,
                        rollouts,
                        extra=live_collect_extra(args, resumed_groups, resumed_rollouts, completed=False),
                        every=args.stream_flush_every,
                    )
                committed = samples[committed_index]
                prefix.append(committed["action"])
                total_reward += float(committed.get("env_reward", 0.0))
                final_info = committed.get("info") or {}
                task_success = bool(committed.get("success"))
                trajectory.append(
                    {
                        "t": step_index,
                        "screenshot": obs.get("screenshot"),
                        "goal": task.goal,
                        "policy_input": {"goal": task.goal, "screenshot": obs.get("screenshot")},
                        "policy_output": committed.get("action"),
                        "policy_info": committed.get("policy_info") or {},
                        "action": committed.get("action"),
                        "exec_status": committed.get("exec_status"),
                        "exec_error": committed.get("exec_error"),
                        "reward_step": committed.get("env_reward"),
                        "verifier": committed.get("verifier"),
                        "sample_group_id": group.get("group_id") if group is not None else None,
                    }
                )
                if len(groups) >= args.max_groups:
                    stop_collection = True
                    break
                if collection_target_reached(target_trainable_groups, trainable_by_family, trainable_by_template, family_quotas, template_quotas):
                    stop_collection = True
                    break
                if task_success:
                    break
            rollout = {
                "rollout_id": f"onpolicy_{task.task_id}",
                "policy_version": "local_qwen_onpolicy_collect",
                "task_id": task.task_id,
                "goal": task.goal,
                "success": task_success,
                "total_reward": total_reward,
                "num_steps": len(trajectory),
                "trajectory": trajectory,
                "final_info": final_info,
            }
            rollouts.append(rollout)
            completed_task_ids.add(str(task.task_id))
            stream.append_rollout(rollout)
            stream.write_summary(groups, rollouts, extra=live_collect_extra(args, resumed_groups, resumed_rollouts, completed=False))
    unload_policy(policy)
    stream.write_summary(groups, rollouts, extra=live_collect_extra(args, resumed_groups, resumed_rollouts, completed=True))
    return groups, rollouts, summarize_groups(groups, rollouts, extra=live_collect_extra(args, resumed_groups, resumed_rollouts, completed=True))


def collection_target_reached(
    target_trainable_groups: int | None,
    trainable_by_family: Counter[str],
    trainable_by_template: Counter[str],
    family_quotas: dict[str, int],
    template_quotas: dict[str, int],
) -> bool:
    if target_trainable_groups is not None and sum(trainable_by_family.values()) >= target_trainable_groups:
        return True
    if family_quotas and all(trainable_by_family[family] >= target for family, target in family_quotas.items()):
        return True
    if template_quotas and all(trainable_by_template[template] >= target for template, target in template_quotas.items()):
        return True
    return False


def live_collect_extra(args: argparse.Namespace, resumed_groups: int, resumed_rollouts: int, *, completed: bool) -> dict[str, Any]:
    return {
        "stream_collect": bool(args.stream_collect),
        "resume_collect": bool(args.resume_collect),
        "resumed_groups": resumed_groups,
        "resumed_rollouts": resumed_rollouts,
        "completed": completed,
    }


class CollectStreamWriter:
    def __init__(self, output_dir: Path, *, enabled: bool, resume: bool) -> None:
        self.output_dir = output_dir
        self.enabled = enabled
        self.groups_tmp = output_dir / "groups.jsonl.tmp"
        self.rollouts_tmp = output_dir / "rollouts.jsonl.tmp"
        self.live_summary = output_dir / "live_summary.json"
        self.appended_groups = 0
        if self.enabled:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not resume:
                for path in (self.groups_tmp, self.rollouts_tmp, self.live_summary):
                    if path.exists():
                        path.unlink()

    def load_groups(self) -> list[dict[str, Any]]:
        return self._load_rows(self.groups_tmp, self.output_dir / "groups.jsonl")

    def load_rollouts(self) -> list[dict[str, Any]]:
        return self._load_rows(self.rollouts_tmp, self.output_dir / "rollouts.jsonl")

    def append_group(self, group: dict[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(self.groups_tmp, group)
        self.appended_groups += 1

    def append_rollout(self, rollout: dict[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(self.rollouts_tmp, rollout)

    def maybe_write_summary(
        self,
        groups: list[dict[str, Any]],
        rollouts: list[dict[str, Any]],
        *,
        extra: dict[str, Any],
        every: int,
    ) -> None:
        if self.enabled and max(1, every) > 0 and self.appended_groups % max(1, every) == 0:
            self.write_summary(groups, rollouts, extra=extra)

    def write_summary(self, groups: list[dict[str, Any]], rollouts: list[dict[str, Any]], *, extra: dict[str, Any]) -> None:
        if not self.enabled:
            return
        summary = summarize_groups(groups, rollouts, extra=extra)
        summary["groups_tmp"] = str(self.groups_tmp)
        summary["rollouts_tmp"] = str(self.rollouts_tmp)
        atomic_write_json(self.live_summary, summary)

    @staticmethod
    def _load_rows(primary: Path, fallback: Path) -> list[dict[str, Any]]:
        if primary.exists():
            return read_jsonl(primary)
        if fallback.exists():
            return read_jsonl(fallback)
        return []


def parse_family_quotas(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--family-quotas-json must be a JSON object")
    return {str(key): int(val) for key, val in payload.items() if int(val) > 0}


def action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def injected_candidate_actions(
    env: PlaywrightBrowserEnv,
    task: Any,
    obs: dict[str, Any],
    seen_actions: set[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if str(getattr(task, "template", "")) != "advanced_scroll":
        return []
    actions: list[dict[str, Any]] = []
    if args.inject_scroll_candidates and not target_already_visible(obs):
        for dy in parse_scroll_candidate_dys(args.scroll_candidate_dys):
            action = {"action": "scroll", "dy": dy}
            append_injected_action(actions, seen_actions, action, args)
    if args.inject_target_click_candidates and target_already_visible(obs):
        for action in target_click_candidates(env, task):
            append_injected_action(actions, seen_actions, action, args)
    return actions


def append_injected_action(
    actions: list[dict[str, Any]],
    seen_actions: set[str],
    action: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    key = action_key(action)
    if args.dedupe_actions and key in seen_actions:
        return
    seen_actions.add(key)
    actions.append(action)


def target_click_candidates(env: PlaywrightBrowserEnv, task: Any) -> list[dict[str, Any]]:
    try:
        center_x, center_y = env.selector_center("#target")
    except Exception:
        return []
    offsets = [(0, 0), (-20, 0), (20, 0), (0, -12), (0, 12)]
    candidates: list[dict[str, Any]] = []
    for dx, dy in offsets:
        x, y = pixels_to_normalized(center_x + dx, center_y + dy, task.viewport)
        candidates.append({"action": "click", "x": x, "y": y})
    return candidates


def parse_scroll_candidate_dys(value: str) -> list[int]:
    dys: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        dys.append(int(float(item)))
    return dys or [600, 900, 1200]


def target_already_visible(obs: dict[str, Any]) -> bool:
    history = obs.get("history") or []
    if not history:
        return False
    progress = ((history[-1].get("verifier") or {}).get("progress") or {}) if isinstance(history[-1], dict) else {}
    return bool(progress.get("target_visible"))


def injected_policy_info(action: dict[str, Any], *, source: str) -> dict[str, Any]:
    raw_text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    return {
        "policy": "verifier_guided_injected_candidate",
        "provider": "local_rule",
        "source": source,
        "raw_text": raw_text,
        "valid_json": True,
        "valid_action": True,
        "error": None,
    }


def load_task_subset(args: argparse.Namespace, *, limit: int | None) -> list[Any]:
    tasks = load_tasks(args.tasks, limit=None)
    include_families = parse_csv_set(args.include_families)
    include_templates = parse_csv_set(args.include_templates)
    if include_families:
        tasks = [task for task in tasks if str((task.metadata or {}).get("family") or task.template) in include_families]
    if include_templates:
        tasks = [task for task in tasks if str(task.template) in include_templates]
    if args.shuffle_tasks:
        random.Random(args.seed).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def parse_csv_set(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def reset_and_replay(
    env: PlaywrightBrowserEnv,
    task: Any,
    prefix_actions: list[dict[str, Any]],
    *,
    screenshot_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env.screenshot_prefix = screenshot_prefix
    obs, info = env.reset(task)
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    for action in prefix_actions:
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return obs, {
                "reset_info": info,
                "terminated": terminated,
                "truncated": truncated,
                "total_reward": total_reward,
                "final_info": final_info,
            }
    return obs, {"reset_info": info, "terminated": False, "truncated": False, "total_reward": total_reward, "final_info": final_info}


def evaluate_branch(
    env: PlaywrightBrowserEnv,
    task: Any,
    prefix_actions: list[dict[str, Any]],
    action: dict[str, Any],
    *,
    policy_info: dict[str, Any],
    screenshot_prefix: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _, replay = reset_and_replay(env, task, prefix_actions, screenshot_prefix=screenshot_prefix)
    if replay.get("terminated") or replay.get("truncated"):
        info = replay.get("final_info") or {}
        reward = shaped_reward(0.0, info, policy_info, args, reward_components={})
        return sample_record(action, policy_info, info, env_reward=0.0, shaped=reward, terminated=True, truncated=False, reward_components={})
    reward_components = branch_reward_components(env, task, prefix_actions, action, args)
    _, env_reward, terminated, truncated, info = env.step(action)
    reward = shaped_reward(float(env_reward), info, policy_info, args, reward_components=reward_components)
    return sample_record(
        action,
        policy_info,
        info,
        env_reward=float(env_reward),
        shaped=reward,
        terminated=terminated,
        truncated=truncated,
        reward_components=reward_components,
    )


def branch_reward_components(
    env: PlaywrightBrowserEnv,
    task: Any,
    prefix_actions: list[dict[str, Any]],
    action: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    templates = parse_csv_set(args.target_distance_reward_templates)
    if templates and str(getattr(task, "template", "")) not in templates:
        return {}
    components: dict[str, Any] = {}
    distance = click_target_distance(env, task, action)
    if distance is not None and args.target_distance_reward_weight > 0:
        sigma = max(1e-6, float(args.target_distance_reward_sigma))
        bonus = float(args.target_distance_reward_weight) * math.exp(-(distance * distance) / (2.0 * sigma * sigma))
        components["target_distance"] = distance
        components["target_distance_bonus"] = bonus
    penalty = repeated_click_penalty(prefix_actions, action, args)
    if penalty:
        components["repeat_click_penalty"] = penalty
    return components


def click_target_distance(env: PlaywrightBrowserEnv, task: Any, action: dict[str, Any]) -> float | None:
    if str((action or {}).get("action")) not in {"click", "double_click"}:
        return None
    if "x" not in action or "y" not in action:
        return None
    center = visible_selector_center(env, "#target")
    if center is None:
        return None
    target_x, target_y = pixels_to_normalized(center[0], center[1], task.viewport)
    try:
        click_x = float(action["x"])
        click_y = float(action["y"])
    except Exception:
        return None
    return math.hypot(click_x - float(target_x), click_y - float(target_y))


def visible_selector_center(env: PlaywrightBrowserEnv, selector: str) -> tuple[float, float] | None:
    page = getattr(env, "page", None)
    if page is None:
        return None
    try:
        result = page.evaluate(
            """
(selector) => {
  const element = document.querySelector(selector);
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  if (rect.bottom <= 0 || rect.top >= window.innerHeight) return null;
  if (rect.right <= 0 || rect.left >= window.innerWidth) return null;
  return {x: rect.left + rect.width / 2, y: rect.top + rect.height / 2};
}
""",
            selector,
        )
    except Exception:
        return None
    if not isinstance(result, dict) or "x" not in result or "y" not in result:
        return None
    return float(result["x"]), float(result["y"])


def repeated_click_penalty(prefix_actions: list[dict[str, Any]], action: dict[str, Any], args: argparse.Namespace) -> float:
    penalty = float(args.repeat_click_penalty)
    if penalty <= 0 or str((action or {}).get("action")) not in {"click", "double_click"}:
        return 0.0
    try:
        click_x = float(action["x"])
        click_y = float(action["y"])
    except Exception:
        return 0.0
    radius = max(0.0, float(args.repeat_click_radius))
    for previous in reversed(prefix_actions):
        if str((previous or {}).get("action")) not in {"click", "double_click"}:
            continue
        if "x" not in previous or "y" not in previous:
            continue
        try:
            previous_x = float(previous["x"])
            previous_y = float(previous["y"])
        except Exception:
            continue
        if math.hypot(click_x - previous_x, click_y - previous_y) <= radius:
            return -penalty
    return 0.0


def shaped_reward(
    env_reward: float,
    info: dict[str, Any],
    policy_info: dict[str, Any],
    args: argparse.Namespace,
    *,
    reward_components: dict[str, Any],
) -> float:
    reward = float(env_reward) - float(args.step_cost)
    reward += float(reward_components.get("target_distance_bonus") or 0.0)
    reward += float(reward_components.get("repeat_click_penalty") or 0.0)
    if bool((info.get("verifier") or {}).get("success")):
        reward += float(args.success_bonus)
    if policy_info.get("valid_json") is False or policy_info.get("valid_action") is False:
        reward -= float(args.invalid_penalty)
    if info.get("exec_status") != "ok":
        reward -= float(args.exec_error_penalty)
    return reward


def sample_record(
    action: dict[str, Any],
    policy_info: dict[str, Any],
    info: dict[str, Any],
    *,
    env_reward: float,
    shaped: float,
    terminated: bool,
    truncated: bool,
    reward_components: dict[str, Any],
) -> dict[str, Any]:
    raw_text = str(policy_info.get("raw_text") or "").strip()
    completion_text = raw_text or json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    return {
        "completion_text": completion_text,
        "action": action,
        "policy_info": policy_info,
        "env_reward": env_reward,
        "reward": shaped,
        "success": bool((info.get("verifier") or {}).get("success")),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "exec_status": info.get("exec_status"),
        "exec_error": info.get("exec_error"),
        "verifier": info.get("verifier"),
        "reward_components": reward_components,
        "info": info,
    }


def choose_commit_index(samples: list[dict[str, Any]], strategy: str) -> int:
    if not samples:
        return 0
    if strategy == "best":
        return max(range(len(samples)), key=lambda index: float(samples[index].get("reward", 0.0)))
    return 0


def build_group(
    task: Any,
    obs: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    group_index: int,
    prefix_actions: list[dict[str, Any]],
    committed_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rewards = [float(sample.get("reward", 0.0)) for sample in samples]
    reward_mean = sum(rewards) / max(1, len(rewards))
    reward_std = std(rewards)
    prompt = build_prompt(obs, max_history=args.max_history) if args.prompt_style == "full" else "任务：" + task.goal + "\n请输出下一步 GUI action JSON。"
    return {
        "group_id": f"{task.task_id}_g{group_index:05d}",
        "task_id": task.task_id,
        "goal": task.goal,
        "template": task.template,
        "family": (task.metadata or {}).get("family"),
        "split": task.split,
        "step": obs.get("step"),
        "screenshot": obs.get("screenshot"),
        "viewport": obs.get("viewport"),
        "history": obs.get("history"),
        "action_space": obs.get("action_space"),
        "max_steps": obs.get("max_steps"),
        "prompt": prompt,
        "prefix_actions": list(prefix_actions),
        "samples": samples,
        "committed_index": committed_index,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "trainable": reward_std > args.min_reward_std and len(samples) >= 2,
    }


def summarize_groups(groups: list[dict[str, Any]], rollouts: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    reward_stds = [float(group.get("reward_std", 0.0)) for group in groups]
    trainable_groups = [group for group in groups if group.get("trainable")]
    samples = [sample for group in groups for sample in group.get("samples", [])]
    action_counts = Counter(str((sample.get("action") or {}).get("action")) for sample in samples)
    family_counts = Counter(str(group.get("family") or "unknown") for group in groups)
    family_trainable = Counter(str(group.get("family") or "unknown") for group in trainable_groups)
    template_counts = Counter(str(group.get("template") or "unknown") for group in groups)
    template_trainable = Counter(str(group.get("template") or "unknown") for group in trainable_groups)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "groups": len(groups),
        "trainable_groups": len(trainable_groups),
        "zero_std_groups": sum(1 for value in reward_stds if value <= 1e-8),
        "avg_reward_std": sum(reward_stds) / max(1, len(reward_stds)),
        "samples": len(samples),
        "rollouts": len(rollouts),
        "rollout_success_rate": sum(1 for row in rollouts if row.get("success")) / max(1, len(rollouts)),
        "action_distribution": dict(action_counts),
        "groups_by_family": dict(family_counts),
        "trainable_groups_by_family": dict(family_trainable),
        "groups_by_template": dict(template_counts),
        "trainable_groups_by_template": dict(template_trainable),
    }
    if extra:
        summary.update(extra)
    return summary


def train_group_relative_policy(args: argparse.Namespace, output_dir: Path, groups: list[dict[str, Any]]) -> dict[str, Any]:
    train_groups = [group for group in groups if group.get("trainable")]
    if args.train_max_groups and len(train_groups) > args.train_max_groups:
        train_groups = train_groups[: args.train_max_groups]
    if not train_groups:
        return {"trained": False, "reason": "no_trainable_groups", "groups": len(groups)}

    processor, model = load_trainable_qwen(args)
    needs_cached_logprobs = args.grpo_loss_type == "clipped" or float(args.kl_beta) > 0
    if needs_cached_logprobs and not args.skip_logprob_cache:
        model.eval()
        cache_sample_logprobs(model, processor, train_groups, args)
    model.train()
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.learning_rate)
    trainable, total = parameter_counts(model)
    replay_rows = load_sft_replay_rows(args)
    replay_rng = random.Random(args.seed + 1701)
    losses: list[float] = []
    policy_losses: list[float] = []
    replay_losses: list[float] = []
    reward_means: list[float] = []
    reward_stds: list[float] = []
    approx_kls: list[float] = []
    clip_fracs: list[float] = []
    ratio_means: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    update_steps = 0
    micro_steps = 0

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_groups)
        for group in train_groups:
            policy_loss, metrics = group_loss(model, processor, group, args)
            if policy_loss is None:
                continue
            loss = policy_loss
            replay_count = sample_replay_count(args.replay_ratio, replay_rng) if replay_rows and args.replay_loss_weight > 0 else 0
            if replay_count > 0:
                sampled_replay_losses: list[torch.Tensor] = []
                for _ in range(replay_count):
                    row = replay_rows[replay_rng.randrange(len(replay_rows))]
                    sampled_replay_losses.append(sft_replay_loss(model, processor, row, args))
                replay_loss = torch.stack(sampled_replay_losses).mean()
                loss = loss + float(args.replay_loss_weight) * replay_loss
                replay_losses.append(float(replay_loss.detach().cpu().item()))
            (loss / max(1, args.gradient_accumulation_steps)).backward()
            losses.append(float(loss.detach().cpu().item()))
            policy_losses.append(float(policy_loss.detach().cpu().item()))
            reward_means.append(metrics["reward_mean"])
            reward_stds.append(metrics["reward_std"])
            approx_kls.append(metrics.get("approx_kl", 0.0))
            clip_fracs.append(metrics.get("clip_frac", 0.0))
            ratio_means.append(metrics.get("ratio_mean", 1.0))
            micro_steps += 1
            if micro_steps % max(1, args.gradient_accumulation_steps) == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_steps += 1
    if micro_steps % max(1, args.gradient_accumulation_steps) != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_steps += 1

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    if needs_cached_logprobs:
        write_jsonl(output_dir / "train_groups_with_logprobs.jsonl", train_groups)
    return {
        "trained": True,
        "adapter_dir": str(adapter_dir),
        "trainable_groups": len(train_groups),
        "micro_steps": micro_steps,
        "optimizer_steps": update_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "grpo_loss_type": args.grpo_loss_type,
        "clip_epsilon": args.clip_epsilon,
        "kl_beta": args.kl_beta,
        "old_logprob_key": args.old_logprob_key,
        "reference_logprob_key": args.reference_logprob_key,
        "cached_logprobs": needs_cached_logprobs and not args.skip_logprob_cache,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / max(1, total),
        "loss_mean": sum(losses) / max(1, len(losses)),
        "loss_last": losses[-1] if losses else None,
        "policy_loss_mean": sum(policy_losses) / max(1, len(policy_losses)),
        "replay_loss_mean": sum(replay_losses) / max(1, len(replay_losses)) if replay_losses else None,
        "replay_rows": len(replay_rows),
        "replay_ratio": args.replay_ratio,
        "replay_loss_weight": args.replay_loss_weight,
        "reward_mean": sum(reward_means) / max(1, len(reward_means)),
        "reward_std_mean": sum(reward_stds) / max(1, len(reward_stds)),
        "approx_kl_mean": sum(approx_kls) / max(1, len(approx_kls)),
        "clip_frac_mean": sum(clip_fracs) / max(1, len(clip_fracs)),
        "ratio_mean": sum(ratio_means) / max(1, len(ratio_means)),
    }


def load_sft_replay_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.replay_sft_jsonl or args.replay_loss_weight <= 0 or args.replay_ratio <= 0:
        return []
    path = Path(args.replay_sft_jsonl)
    rows = []
    for row in read_jsonl(path):
        replay = parse_sft_replay_row(row)
        if replay:
            rows.append(replay)
    if args.replay_max_rows and len(rows) > args.replay_max_rows:
        rng = random.Random(args.seed + 1702)
        rows = rng.sample(rows, args.replay_max_rows)
    return rows


def parse_sft_replay_row(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages") or []
    images = row.get("images") or []
    if not isinstance(messages, list) or not images:
        return None
    user_text = None
    assistant_text = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user" and user_text is None:
            user_text = message_text(message.get("content"))
        if message.get("role") == "assistant" and assistant_text is None:
            assistant_text = message_text(message.get("content"))
    if not user_text or not assistant_text:
        return None
    return {
        "image": str(images[0]),
        "prompt": strip_image_marker(user_text),
        "completion": assistant_text.strip(),
        "task_id": row.get("task_id"),
        "step": row.get("step"),
    }


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return ""


def strip_image_marker(text: str) -> str:
    text = text.strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip()
    return text


def sample_replay_count(ratio: float, rng: random.Random) -> int:
    if ratio <= 0:
        return 0
    base = int(math.floor(ratio))
    frac = float(ratio) - base
    return base + (1 if rng.random() < frac else 0)


def sft_replay_loss(model: Any, processor: Any, row: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    return -completion_logprob(
        model,
        processor,
        image_path=resolve_path(row["image"]),
        prompt=str(row["prompt"]),
        completion=str(row["completion"]),
        system_prompt=args.system_prompt,
        reduction=args.logprob_reduction,
    )


def group_loss(model: Any, processor: Any, group: dict[str, Any], args: argparse.Namespace) -> tuple[torch.Tensor | None, dict[str, float]]:
    samples = list(group.get("samples") or [])
    rewards = torch.tensor([float(sample.get("reward", 0.0)) for sample in samples], device=infer_input_device(model))
    if rewards.numel() < 2:
        return None, {}
    reward_std = rewards.std(unbiased=False)
    if float(reward_std.detach().cpu().item()) <= args.min_reward_std:
        return None, {}
    advantages = (rewards - rewards.mean()) / reward_std.clamp_min(1e-8)
    logps: list[torch.Tensor] = []
    for sample in samples:
        logps.append(
            completion_logprob(
                model,
                processor,
                image_path=resolve_path(group["screenshot"]),
                prompt=str(group["prompt"]),
                completion=str(sample.get("completion_text") or json.dumps(sample.get("action") or {}, ensure_ascii=False)),
                system_prompt=args.system_prompt,
                reduction=args.logprob_reduction,
            )
        )
    logp_tensor = torch.stack(logps)
    metrics = {"reward_mean": float(rewards.mean().detach().cpu().item()), "reward_std": float(reward_std.detach().cpu().item())}
    if args.grpo_loss_type == "vanilla" and float(args.kl_beta) <= 0:
        loss = -(advantages.detach() * logp_tensor).mean()
        return loss, metrics

    old_logps = cached_sample_logprobs(group, args.old_logprob_key, logp_tensor.device, logp_tensor.dtype)
    ref_logps = cached_sample_logprobs(group, args.reference_logprob_key, logp_tensor.device, logp_tensor.dtype)
    log_ratio = logp_tensor - old_logps
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - float(args.clip_epsilon), 1.0 + float(args.clip_epsilon))
    objective = torch.minimum(ratio * advantages.detach(), clipped_ratio * advantages.detach())
    policy_loss = -objective.mean()
    approx_kl = 0.5 * ((logp_tensor - ref_logps) ** 2).mean()
    loss = policy_loss + float(args.kl_beta) * approx_kl
    clip_frac = (torch.abs(ratio - 1.0) > float(args.clip_epsilon)).float().mean()
    metrics.update(
        {
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "clip_frac": float(clip_frac.detach().cpu().item()),
            "ratio_mean": float(ratio.detach().mean().cpu().item()),
        }
    )
    return loss, metrics


def cache_sample_logprobs(model: Any, processor: Any, groups: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for group_index, group in enumerate(groups):
        for sample in group.get("samples") or []:
            completion = str(sample.get("completion_text") or json.dumps(sample.get("action") or {}, ensure_ascii=False))
            with torch.no_grad():
                logp = completion_logprob(
                    model,
                    processor,
                    image_path=resolve_path(group["screenshot"]),
                    prompt=str(group["prompt"]),
                    completion=completion,
                    system_prompt=args.system_prompt,
                    reduction=args.logprob_reduction,
                )
            value = float(logp.detach().cpu().item())
            sample.setdefault(args.old_logprob_key, value)
            sample.setdefault(args.reference_logprob_key, value)
        if (group_index + 1) % 10 == 0:
            print(json.dumps({"sample_logprob_cached_groups": group_index + 1, "total": len(groups)}, ensure_ascii=False), flush=True)


def cached_sample_logprobs(group: dict[str, Any], key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = []
    missing = []
    for index, sample in enumerate(group.get("samples") or []):
        if key not in sample:
            missing.append(index)
            values.append(0.0)
        else:
            values.append(float(sample[key]))
    if missing:
        raise RuntimeError(f"missing cached sample logprobs key={key} group={group.get('group_id')} sample_indices={missing[:5]}")
    return torch.tensor(values, device=device, dtype=dtype)


def completion_logprob(
    model: Any,
    processor: Any,
    *,
    image_path: Path,
    prompt: str,
    completion: str,
    system_prompt: str,
    reduction: str = "mean",
) -> torch.Tensor:
    from qwen_vl_utils import process_vision_info

    user_content = [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]
    prompt_messages: list[dict[str, Any]] = []
    full_messages: list[dict[str, Any]] = []
    if system_prompt:
        prompt_messages.append({"role": "system", "content": system_prompt})
        full_messages.append({"role": "system", "content": system_prompt})
    prompt_messages.append({"role": "user", "content": user_content})
    full_messages.append({"role": "user", "content": user_content})
    full_messages.append({"role": "assistant", "content": completion})

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(prompt_messages)
    prompt_inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = infer_input_device(model)
    prompt_len = int(prompt_inputs.input_ids.shape[1])
    labels = full_inputs.input_ids.clone()
    labels[:, : min(prompt_len, labels.shape[1])] = -100
    inputs = full_inputs.to(device)
    labels = labels.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    if int(mask.sum().detach().cpu().item()) == 0:
        return token_logps.sum() * 0.0
    total = (token_logps * mask).sum()
    if reduction == "mean":
        return total / mask.sum().clamp_min(1)
    return total


def load_trainable_qwen(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    dtype = parse_torch_dtype(args.torch_dtype)
    processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.image_max_pixels:
        processor_kwargs["max_pixels"] = args.image_max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    model_kwargs["torch_dtype"] = dtype if dtype != "auto" else "auto"
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if dtype == "auto" else dtype
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return processor, model


def parse_torch_dtype(name: str) -> Any:
    normalized = str(name or "auto").lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def infer_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parameter_counts(model: Any) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def unload_policy(policy: LocalQwenPolicy) -> None:
    try:
        policy.model = None
        policy.processor = None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


if __name__ == "__main__":
    raise SystemExit(main())
