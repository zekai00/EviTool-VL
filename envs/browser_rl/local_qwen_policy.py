"""Local Qwen2.5-VL/Qwen3-VL policy for browser GUI-RL rollouts.

Local policy means the action chooser runs from a local checkpoint whose
parameters we can fine-tune. This is the "current model" path for later
on-policy RL. DashScope Qwen models remain teacher/baseline policies only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .qwen_policy import DEFAULT_ACTION, QwenPolicyResult, build_prompt, parse_model_action


@dataclass
class LocalQwenLoadInfo:
    model_path: str
    adapter_path: str | None
    device_map: str
    load_in_4bit: bool
    torch_dtype: str


class LocalQwenPolicy:
    """Run a local trainable Qwen-VL checkpoint as a direct GUI action policy."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        adapter_path: str | Path | None = None,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        max_history: int = 4,
        prompt_style: str = "sft_minimal",
        system_prompt: str = "You are a helpful assistant.",
        trust_remote_code: bool = True,
        image_max_pixels: int | None = 262144,
        on_error_action: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.torch_dtype_name = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_history = max_history
        self.prompt_style = prompt_style
        self.system_prompt = system_prompt
        self.trust_remote_code = trust_remote_code
        self.image_max_pixels = image_max_pixels
        self.on_error_action = on_error_action or DEFAULT_ACTION
        self.processor = None
        self.model = None
        self.load_info = LocalQwenLoadInfo(
            model_path=self.model_path,
            adapter_path=self.adapter_path,
            device_map=self.device_map,
            load_in_4bit=self.load_in_4bit,
            torch_dtype=self.torch_dtype_name,
        )
        self._load()

    def act(self, observation: dict[str, Any]) -> QwenPolicyResult:
        prompt = local_qwen_prompt(observation, max_history=self.max_history, style=self.prompt_style)
        screenshot = Path(str(observation.get("screenshot") or ""))
        try:
            raw_text = self._generate(screenshot=screenshot, prompt=prompt)
            action, parse_info = parse_model_action(raw_text)
            return QwenPolicyResult(
                action=action,
                info={
                    "policy": "local_qwen",
                    "provider": "local",
                    "model": self.model_path,
                    "adapter": self.adapter_path,
                    "prompt_style": self.prompt_style,
                    "raw_text": raw_text,
                    **parse_info,
                },
            )
        except Exception as exc:
            return QwenPolicyResult(
                action=dict(self.on_error_action),
                info={
                    "policy": "local_qwen",
                    "provider": "local",
                    "model": self.model_path,
                    "adapter": self.adapter_path,
                    "prompt_style": self.prompt_style,
                    "raw_text": "",
                    "valid_json": False,
                    "valid_action": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_action": self.on_error_action,
                },
            )

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        dtype = parse_torch_dtype(torch, self.torch_dtype_name)
        processor_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.image_max_pixels:
            processor_kwargs["max_pixels"] = self.image_max_pixels
        self.processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = "auto"
        if self.load_in_4bit:
            compute_dtype = torch.bfloat16 if dtype == "auto" else dtype
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs)
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = False
            if self.temperature <= 0.0:
                self.model.generation_config.temperature = None

    def _generate(self, *, screenshot: Path, prompt: str) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("local Qwen policy is not loaded")
        if not screenshot.exists():
            raise FileNotFoundError(f"screenshot not found: {screenshot}")

        import torch
        from qwen_vl_utils import process_vision_info

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(screenshot)},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(infer_input_device(self.model))
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generate_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_trimmed = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0] if decoded else "").strip()


def parse_torch_dtype(torch_module: Any, name: str) -> Any:
    normalized = str(name or "auto").lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def local_qwen_prompt(observation: dict[str, Any], *, max_history: int, style: str) -> str:
    if style == "sft_minimal":
        goal = str(observation.get("goal") or "")
        return "任务：" + goal + "\n请输出下一步 GUI action JSON。"
    if style == "full":
        return build_prompt(observation, max_history=max_history)
    raise ValueError(f"unsupported local prompt style: {style}")


def infer_input_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return "cuda"
