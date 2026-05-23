#!/usr/bin/env python3
"""Run prompt-only visual tool-use baselines.

The model emits JSON text. This script parses the JSON, executes local visual
tools, appends observations back into the chat, and asks the model for a final
JSON answer. No native function-calling API is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.runner import run_tool

BASELINE_PATH = Path(__file__).resolve().parent / "eval_baseline.py"
spec = importlib.util.spec_from_file_location("eval_baseline", BASELINE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


TOOL_DESCRIPTIONS = """Available visual tools. Coordinates are pixel coordinates [x1, y1, x2, y2].

1. inspect: get image size and simple image statistics.
   {"thought":"need image metadata","action":{"tool":"inspect","args":{}}}

2. crop: crop a region.
   {"thought":"inspect a local region","action":{"tool":"crop","args":{"bbox":[x1,y1,x2,y2]}}}

3. zoom: crop and enlarge a region.
   {"thought":"read small visual details","action":{"tool":"zoom","args":{"bbox":[x1,y1,x2,y2],"scale":2.0}}}

4. ocr: read text in the whole image or a region. Use engine "easyocr" by default; use "paddleocr" only if necessary.
   {"thought":"read text in a region","action":{"tool":"ocr","args":{"bbox":[x1,y1,x2,y2],"engine":"easyocr","languages":["en"]}}}

5. detect: find candidate regions. mode can be "layout", "text", "ui", "bar", or "color".
   {"thought":"find UI candidates","action":{"tool":"detect","args":{"mode":"ui","max_results":10}}}

6. measure: measure boxes, centers, relative position, IoU, and bar heights.
   {"thought":"compare visual boxes","action":{"tool":"measure","args":{"bboxes":[[x1,y1,x2,y2],[x1,y1,x2,y2]],"mode":"compare_height"}}}

7. click: virtual GUI select/click target recorder. Use only for GUI grounding if you want to record a selected point or bbox as evidence.
   {"thought":"select the close button","action":{"tool":"click","args":{"bbox":[x1,y1,x2,y2],"label":"close button"}}}

8. mark: draw boxes/points for visualization. Use this only for debugging, not for answering.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model path or HF repo id.")
    parser.add_argument("--data", default="data/eval_mini/eval_mini_100.jsonl")
    parser.add_argument("--image-root", default="data/eval_mini")
    parser.add_argument("--output", required=True, help="Prediction JSONL path.")
    parser.add_argument("--summary", default=None, help="Metrics JSON path. Defaults to output + .summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-per-task", type=int, default=None)
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--max-tool-steps", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tool-out-dir", default="outputs/tool_artifacts")
    parser.add_argument("--max-observation-spans", type=int, default=20)
    parser.add_argument("--max-observation-chars", type=int, default=3500)
    parser.add_argument("--allow-direct-final", action="store_true", help="Prompt says direct final is allowed before tool use.")
    parser.add_argument("--force-tool-first", action="store_true", help="Reject direct final before at least one tool call.")
    parser.add_argument("--repair-missing-evidence", action="store_true", help="Ask the model to rewrite final JSON if it omits evidence ids.")
    return parser.parse_args()


def load_model(model_name_or_path: str):
    return base.load_model(model_name_or_path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return base.load_rows(path)


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    return base.select_rows(rows, args)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.width, img.height


def task_guidance(row: dict[str, Any]) -> str:
    task = row.get("task_type")
    if task == "gui_grounding":
        return (
            "This is a GUI grounding task. You need return a bounding box. "
            "Useful tools: detect(mode=\"ui\"), crop/zoom around candidates, measure. "
            "Final answer must be a bbox [x1, y1, x2, y2]."
        )
    if task == "chart_qa":
        return (
            "This is a chart/table visual question. Use crop/zoom/OCR/detect(mode=\"bar\")/measure if values, labels, or bars are local. "
            "Final answer should be short."
        )
    if task == "doc_qa":
        return (
            "This is a document or figure question. Use OCR on the relevant region when reading text or numbers. "
            "Prefer easyocr for speed. Final answer should be short."
        )
    if task == "science_diagram_qa":
        return (
            "This is a science diagram multiple-choice question. Use inspect/crop/zoom if the relevant part is local. "
            "Final answer may be the option letter or option text."
        )
    return "Use visual tools to gather evidence before answering."


def initial_prompt(row: dict[str, Any], image_path: Path, allow_direct_final: bool) -> str:
    width, height = image_size(image_path)
    final_rule = (
        "Your first response MUST be an ACTION JSON that calls one tool. Do not answer directly in the first response."
        if not allow_direct_final
        else "You may either call one tool with ACTION JSON or answer with FINAL JSON if no tool is needed."
    )
    return f"""You are EviTool-VL, a visual reasoning model that can request local visual tools.

Question:
{row.get('question')}

Task type: {row.get('task_type')}
Ground truth is hidden during prediction.
Image size: width={width}, height={height}.
{task_guidance(row)}

{TOOL_DESCRIPTIONS}

Output protocol:
- {final_rule}
- Return exactly one valid JSON object. Do not use markdown fences.
- Never mix ACTION fields and FINAL fields in the same JSON object.
- If you output ACTION JSON, do not include `answer` or `reasoning`.
- If you output FINAL JSON, do not include `action`.
- For tool calls, output:
  {{"thought":"...","action":{{"tool":"tool_name","args":{{...}}}}}}
- After receiving observations, either call another tool or output FINAL JSON:
  {{"reasoning":[{{"step":"...","evidence":["ev_001"]}}],"answer":"..."}}
- Every final reasoning step should cite evidence ids that appeared in observations.
- FINAL JSON must contain a non-empty `answer`. For GUI grounding, `answer` must be [x1, y1, x2, y2].
""".strip()


def continue_prompt(observation: dict[str, Any], remaining_steps: int) -> str:
    compact = compact_observation(observation)
    return f"""Tool observation:
{json.dumps(compact, ensure_ascii=False)}

Remaining tool calls: {remaining_steps}.
Return exactly one JSON object. Either call another tool with ACTION JSON or provide FINAL JSON.
Do not mix action and answer in one object. If you have enough evidence, provide FINAL JSON now with a non-empty answer.
""".strip()


def force_final_prompt(evidence_ids: list[str]) -> str:
    return f"""You have reached the maximum number of tool calls.
Available evidence ids: {evidence_ids}
Return exactly one FINAL JSON object now:
{{"reasoning":[{{"step":"...","evidence":["ev_001"]}}],"answer":"..."}}
""".strip()


def build_initial_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def append_text_turn(messages: list[dict[str, Any]], role: str, text: str) -> None:
    messages.append({"role": role, "content": [{"type": "text", "text": text}]})


def generate_message(model, processor, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[:, input_len:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def strip_json_fence(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    raw = strip_json_fence(text)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(raw) if ch == "{"]
    last_error = None
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(raw[start:])
            if isinstance(obj, dict):
                return obj
        except Exception as exc:
            last_error = exc
    raise ValueError(f"no valid JSON object found: {last_error}; raw={text[:500]!r}")


def classify_model_json(obj: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "action" in obj and isinstance(obj["action"], dict):
        action = obj["action"]
        if "tool" in action:
            return "action", {"thought": obj.get("thought"), "action": action}
    if "action" in obj and isinstance(obj["action"], str):
        return "action", {"thought": obj.get("thought"), "action": {"tool": obj.get("action"), "args": obj.get("args", {})}}
    if "tool" in obj:
        return "action", {"thought": obj.get("thought"), "action": {"tool": obj.get("tool"), "args": obj.get("args", {})}}
    if "final" in obj and isinstance(obj["final"], dict):
        return "final", obj["final"]
    if "answer" in obj:
        return "final", obj
    raise ValueError(f"JSON is neither action nor final: {obj}")


def compact_observation(observation: dict[str, Any], max_spans: int = 20, max_chars: int = 3500) -> dict[str, Any]:
    content = observation.get("content") or {}
    compact: dict[str, Any] = {
        "evidence_id": observation.get("evidence_id"),
        "tool": observation.get("tool"),
        "bbox": observation.get("bbox"),
        "ok": observation.get("ok"),
        "error": observation.get("error"),
    }
    if observation.get("artifacts"):
        compact["artifacts"] = observation.get("artifacts")

    tool = observation.get("tool")
    if tool == "ocr":
        spans = content.get("spans") or []
        compact["content"] = {
            "engine": content.get("engine"),
            "available": content.get("available"),
            "spans": spans[:max_spans],
            "span_count": len(spans),
            "errors": content.get("errors", [])[:3],
        }
    elif tool == "detect":
        detections = content.get("detections") or []
        compact["content"] = {
            "mode": content.get("mode"),
            "detections": detections[:max_spans],
            "count": content.get("count", len(detections)),
        }
    elif tool in {"crop", "zoom", "inspect", "measure", "mark", "visualize"}:
        compact["content"] = content
    else:
        compact["content"] = content

    text = json.dumps(compact, ensure_ascii=False)
    if len(text) > max_chars:
        compact["content_truncated"] = True
        compact["content"] = json.loads(json.dumps(compact["content"], ensure_ascii=False)[: max_chars // 2] + '"') if False else compact["content"]
        # Safe truncation by reducing long lists rather than producing invalid JSON.
        if isinstance(compact.get("content"), dict):
            for key in ["spans", "detections", "candidate_regions"]:
                if isinstance(compact["content"].get(key), list):
                    compact["content"][key] = compact["content"][key][: max(3, max_spans // 2)]
    return compact


def final_answer_text(final_obj: dict[str, Any], raw_output: str) -> str:
    if isinstance(final_obj.get("answer"), (list, dict)):
        return json.dumps(final_obj.get("answer"), ensure_ascii=False)
    if final_obj.get("answer") is not None:
        return str(final_obj.get("answer"))
    return raw_output


def recover_answer_from_raw(raw_output: str) -> str:
    """Best-effort answer extraction from malformed FINAL JSON.

    This does not make final_parseable true; it only gives scoring something
    useful when the model almost followed the protocol.
    """
    if not raw_output:
        return ""
    text = raw_output.strip()
    # Common malformed pattern: "answer": "0.28" inside an otherwise broken object/list.
    match = re.search(r'"answer"\s*:\s*(\[[^\]]+\]|"(?:[^"\\]|\\.)*"|[^,}\]]+)', text, flags=re.IGNORECASE)
    if match:
        value = match.group(1).strip()
        if value.startswith('"') and value.endswith('"'):
            try:
                return json.loads(value)
            except Exception:
                return value.strip('"')
        return value.strip()
    # Last resort for GUI bboxes.
    bbox = re.search(r'\[[\s\d.,+-]+\]', text)
    if bbox:
        return bbox.group(0)
    return text


def make_error_observation(image_path: Path, action: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "evidence_id": f"ev_error_{abs(hash(json.dumps(action, sort_keys=True, default=str))) % 100000:05d}",
        "tool": action.get("tool"),
        "image": str(image_path),
        "bbox": (action.get("args") or {}).get("bbox") if isinstance(action.get("args"), dict) else None,
        "content": {"message": "Tool call failed before execution."},
        "artifacts": {},
        "ok": False,
        "error": error,
    }


def referenced_evidence_ids(final_obj: dict[str, Any]) -> list[str]:
    ids = []
    reasoning = final_obj.get("reasoning") or []
    if isinstance(reasoning, list):
        for step in reasoning:
            if isinstance(step, dict):
                evidence = step.get("evidence") or []
                if isinstance(evidence, str):
                    evidence = [evidence]
                ids.extend(str(x) for x in evidence)
    return ids


def run_one_sample(model, processor, row: dict[str, Any], image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    messages = build_initial_messages(image_path, initial_prompt(row, image_path, args.allow_direct_final))
    trace: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    parse_errors: list[str] = []
    protocol_errors: list[str] = []
    evidence_repair_attempted = False
    final_obj: dict[str, Any] | None = None
    final_raw = ""
    error = None
    started = time.time()

    try:
        for step_idx in range(args.max_tool_steps + 1):
            raw = generate_message(model, processor, messages, args.max_new_tokens, args.temperature)
            append_text_turn(messages, "assistant", raw)
            try:
                obj = extract_json_object(raw)
                kind, payload = classify_model_json(obj)
            except Exception as exc:
                parse_errors.append(f"step {step_idx}: {type(exc).__name__}: {exc}")
                append_text_turn(
                    messages,
                    "user",
                    "Your previous response was not valid protocol JSON. Return exactly one JSON object using ACTION JSON or FINAL JSON.",
                )
                if step_idx >= args.max_tool_steps:
                    final_raw = raw
                    break
                continue

            if kind == "final":
                if not args.allow_direct_final and not trace:
                    protocol_errors.append(f"step {step_idx}: model answered before calling a tool")
                    if args.force_tool_first and step_idx < args.max_tool_steps:
                        append_text_turn(
                            messages,
                            "user",
                            "Protocol violation: your first response must call one visual tool before answering. Return ACTION JSON only now.",
                        )
                        continue
                if args.repair_missing_evidence and trace and not referenced_evidence_ids(payload) and not evidence_repair_attempted:
                    evidence_repair_attempted = True
                    protocol_errors.append(f"step {step_idx}: final JSON did not cite evidence ids")
                    append_text_turn(
                        messages,
                        "user",
                        f"Your FINAL JSON did not cite evidence ids. Rewrite FINAL JSON only, using evidence ids from this list: {evidence_ids}. Do not call another tool.",
                    )
                    continue
                elif trace and not referenced_evidence_ids(payload):
                    protocol_errors.append(f"step {step_idx}: final JSON did not cite evidence ids")
                final_obj = payload
                final_raw = raw
                break

            if step_idx >= args.max_tool_steps:
                append_text_turn(messages, "user", force_final_prompt(evidence_ids))
                raw = generate_message(model, processor, messages, args.max_new_tokens, args.temperature)
                append_text_turn(messages, "assistant", raw)
                final_raw = raw
                try:
                    obj = extract_json_object(raw)
                    kind, payload = classify_model_json(obj)
                    if kind == "final":
                        final_obj = payload
                    else:
                        parse_errors.append("forced final produced another action")
                except Exception as exc:
                    parse_errors.append(f"forced final: {type(exc).__name__}: {exc}")
                break

            action = payload["action"]
            if "args" not in action or action["args"] is None:
                action["args"] = {}
            try:
                observation = run_tool(image_path, action, out_dir=args.tool_out_dir)
            except Exception as exc:
                observation = make_error_observation(image_path, action, f"{type(exc).__name__}: {exc}")
            evidence_id = str(observation.get("evidence_id"))
            evidence_ids.append(evidence_id)
            trace.append(
                {
                    "step": step_idx,
                    "raw_output": raw,
                    "thought": payload.get("thought"),
                    "action": action,
                    "observation": observation,
                }
            )
            append_text_turn(
                messages,
                "user",
                continue_prompt(observation, remaining_steps=max(0, args.max_tool_steps - step_idx - 1)),
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency = time.time() - started
    final_recovered = False
    prediction = final_answer_text(final_obj or {}, final_raw)
    if not prediction.strip() and final_raw:
        prediction = recover_answer_from_raw(final_raw)
        final_recovered = bool(prediction.strip())
    elif final_obj is None and final_raw:
        recovered = recover_answer_from_raw(final_raw)
        if recovered and recovered != final_raw:
            prediction = recovered
            final_recovered = True
    metrics = base.score_prediction(prediction, row) if error is None else {"error": error}
    referenced_ids = referenced_evidence_ids(final_obj or {})
    missing_ids = [eid for eid in referenced_ids if eid not in evidence_ids]
    tool_successes = [bool(t["observation"].get("ok")) for t in trace]
    return {
        "id": row.get("id"),
        "task_type": row.get("task_type"),
        "image": row.get("image"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "answers": row.get("answers"),
        "prediction": prediction,
        "final": final_obj,
        "final_raw": final_raw,
        "tool_trace": trace,
        "tool_call_count": len(trace),
        "tool_success_count": sum(tool_successes),
        "tool_success_rate": (sum(tool_successes) / len(tool_successes)) if tool_successes else None,
        "parse_errors": parse_errors,
        "protocol_errors": protocol_errors,
        "final_parseable": final_obj is not None,
        "final_recovered": final_recovered,
        "referenced_evidence_ids": referenced_ids,
        "missing_evidence_ids": missing_ids,
        "evidence_closed": len(missing_ids) == 0 if referenced_ids else False,
        "metrics": metrics,
        "latency_sec": round(latency, 4),
        "error": error,
    }


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize_tool_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    base_summary = base.summarize(results)
    total_tool_calls = sum(r.get("tool_call_count", 0) for r in results)
    observations = [step["observation"] for r in results for step in r.get("tool_trace", [])]
    tool_counter = Counter(step["action"].get("tool") for r in results for step in r.get("tool_trace", []))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_task[result["task_type"]].append(result)

    base_summary.update(
        {
            "final_parse_rate": mean([float(r.get("final_parseable", False)) for r in results]),
            "final_recovery_rate": mean([float(r.get("final_recovered", False)) for r in results]),
            "samples_with_parse_error_rate": mean([float(bool(r.get("parse_errors"))) for r in results]),
            "samples_with_protocol_error_rate": mean([float(bool(r.get("protocol_errors"))) for r in results]),
            "avg_tool_calls": mean([float(r.get("tool_call_count", 0)) for r in results]),
            "max_tool_calls": max([r.get("tool_call_count", 0) for r in results], default=0),
            "total_tool_calls": total_tool_calls,
            "tool_success_rate": mean([float(obs.get("ok", False)) for obs in observations]),
            "tool_counts": dict(tool_counter),
            "evidence_reference_rate": mean([float(bool(r.get("referenced_evidence_ids"))) for r in results]),
            "evidence_closed_rate": mean([float(r.get("evidence_closed", False)) for r in results]),
            "avg_missing_evidence_ids": mean([float(len(r.get("missing_evidence_ids", []))) for r in results]),
            "by_task_tool": {
                task: {
                    "avg_tool_calls": mean([float(r.get("tool_call_count", 0)) for r in items]),
                    "final_parse_rate": mean([float(r.get("final_parseable", False)) for r in items]),
                    "parse_error_rate": mean([float(bool(r.get("parse_errors"))) for r in items]),
                    "protocol_error_rate": mean([float(bool(r.get("protocol_errors"))) for r in items]),
                    "evidence_closed_rate": mean([float(r.get("evidence_closed", False)) for r in items]),
                }
                for task, items in by_task.items()
            },
        }
    )
    return base_summary


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    data_path = Path(args.data)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output_path.with_suffix(output_path.suffix + ".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.tool_out_dir).mkdir(parents=True, exist_ok=True)

    rows = select_rows(load_rows(data_path), args)
    print(f"Loaded {len(rows)} samples from {data_path}", flush=True)
    print(f"Loading model: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model)
    model.eval()

    results = []
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            image_path = image_root / row["image"]
            result = run_one_sample(model, processor, row, image_path, args)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            results.append(result)
            print(
                f"[{idx}/{len(rows)}] {row.get('id')} {row.get('task_type')} "
                f"tools={result['tool_call_count']} final={result['final_parseable']} pred={result['prediction'][:100]!r}",
                flush=True,
            )

    summary = summarize_tool_results(results)
    summary["model"] = args.model
    summary["data"] = str(data_path)
    summary["max_tool_steps"] = args.max_tool_steps
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
