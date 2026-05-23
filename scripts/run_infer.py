#!/usr/bin/env python3
"""Minimal single-image inference script for Qwen VL checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo id or local model path.")
    parser.add_argument("--image", required=True, help="Path to an image.")
    parser.add_argument("--question", required=True, help="Question about the image.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(model_name_or_path: str):
    lower_name = model_name_or_path.lower()
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if "qwen3" in lower_name:
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("Current transformers does not expose Qwen3VLForConditionalGeneration.")
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Current transformers does not expose Qwen2_5_VLForConditionalGeneration.")
    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )


def build_messages(image_path: Path, question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        }
    ]


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    image_path = Path(args.image)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model)

    messages = build_messages(image_path, args.question)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[:, input_len:]
    answer = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(json.dumps({"answer": answer}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
