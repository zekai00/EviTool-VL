# OmniParser 与 RL Rollout 准备报告

## 1. 本轮完成

- 已克隆 `third_party/OmniParser` 源码，仅作为本地依赖，不提交到 Git。
- 新增 `tools/external_detectors.py`，提供可选外部 detector adapter。
- `detect(mode="ui")` 新增可选参数：`include_omniparser`、`omniparser_root`、`omniparser_weights_dir`、`omniparser_use_caption`、`external_cache_dir`。
- 默认行为不变：不设置 `include_omniparser=true` 时，当前评测和工具输出不受影响。
- 新增 `scripts/ablate_gui_candidates.py`，用于统一比较 `trace_initial`、`trace_pipeline`、`current`、`current_ocr`、`omniparser`、`current_omniparser`。
- 新增 `rl/rollout_env.py` 与 `rl/rewards.py`，用于工具交互 rollout 和可解释 reward 分解。
- 新增 `scripts/rl_rollout_smoke.py` 和 `scripts/run_trl_grpo_smoke.py`。

## 2. 已验证

```bash
python3 -m py_compile tools/external_detectors.py tools/detect.py scripts/ablate_gui_candidates.py rl/rewards.py rl/rollout_env.py scripts/rl_rollout_smoke.py scripts/run_trl_grpo_smoke.py
python3 scripts/ablate_gui_candidates.py --limit 5 --variants trace_initial trace_pipeline current omniparser current_omniparser --output outputs/ablate_gui_candidates_omni_smoke.md
CUDA_VISIBLE_DEVICES= python3 scripts/rl_rollout_smoke.py --limit 2 --output outputs/rl_smoke/rollout_dryrun.jsonl
python3 scripts/run_trl_grpo_smoke.py --dry-run --limit 4
```

结果：

- OmniParser 权重未安装时，`include_omniparser=true` 会快速返回 unavailable，不破坏 `detect(ui)` 当前候选。
- GUI 候选 A/B runner 已能输出报告和 summary。
- RL rollout dry-run 能执行本地工具、闭合 evidence id，并输出 reward 分解。
- TRL GRPO smoke dry-run 能验证数据和 reward plumbing。

## 3. OmniParser 后续执行

当前还没有下载 OmniParser 权重。等 SFT v2 tool eval 结束后，再下载权重并跑完整 A/B：

```bash
cd third_party/OmniParser
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do   huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done
mv weights/icon_caption weights/icon_caption_florence
cd ../..
```

候选 A/B：

```bash
python3 scripts/ablate_gui_candidates.py   --input /root/models/datasets/evitool_traces_1k/train_traces_1000.jsonl   --image-root /root/models/datasets/evitool_traces_1k   --variants trace_pipeline current current_ocr omniparser current_omniparser   --output reports/gui_candidate_ablation_omniparser.md
```

进入训练数据 pipeline 的标准：

- `current_omniparser` 相比 `current_ocr` 至少提升 Recall@30，或
- Oracle GT 注入估计下降到 15% 以下，或
- Recall@1/3 明显改善且 latency 可接受。

## 4. RL 后续执行

当前两张 GPU 仍在跑 `eval_sft_v2_validation.sh`，所以没有启动真实 GRPO。等评测结束后，先跑极小 3B smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_trl_grpo_smoke.py   --model /root/models/Qwen2.5-VL-3B-Instruct   --adapter checkpoints/qwen25vl_3b_evitool_sft_v2_tool_gui_lora   --limit 8   --num-generations 2   --max-steps 1
```

注意：这个 TRL 脚本在传入 `--adapter` 时会从 SFT v2 LoRA 起步，但当前仍是训练循环 plumbing smoke，不是最终工具交互 RL。真正的 tool-RL 需要把 `rl/rollout_env.py` 接到 TRL `rollout_func` 或 `environment_factory`，让模型在 rollout 中多轮调用本地工具。
