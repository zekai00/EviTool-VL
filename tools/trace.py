"""Execute multi-step visual tool traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runner import run_tool
from .schema import EvidenceIdGenerator


def _extract_action(step: dict[str, Any]) -> dict[str, Any]:
    if "tool" in step:
        return step
    if "action" in step and isinstance(step["action"], dict):
        return step["action"]
    raise ValueError(f"trace step has no action: {step}")


def run_trace(image_path: str | Path, trace: list[dict[str, Any]], out_dir: str | Path | None = None, prefix: str = "ev") -> list[dict[str, Any]]:
    generator = EvidenceIdGenerator(prefix=prefix)
    observations = []
    for index, step in enumerate(trace):
        action = _extract_action(step)
        evidence_id = step.get("evidence_id") or generator.next()
        result = run_tool(image_path, action, out_dir=out_dir, evidence_id=evidence_id)
        observation = {
            "index": index,
            "thought": step.get("thought"),
            "action": action,
            "observation": result,
        }
        observations.append(observation)
    return observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--trace", required=True, help="Path to a JSON file containing a list of actions or trace steps.")
    parser.add_argument("--out-dir", default="outputs/tools_trace")
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_path = Path(args.trace)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(trace, list):
        raise ValueError("trace JSON must be a list")
    observations = run_trace(args.image, trace, out_dir=args.out_dir)
    ok = all(item["observation"].get("ok") for item in observations)
    print(json.dumps(observations, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
