# EviTool-VL

Evidence-grounded visual tool reasoning for small vision-language models.

EviTool-VL is a research workspace for studying whether small VLMs can answer visual questions more reliably when they are trained or prompted to use explicit local visual tools. The repository currently focuses on Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct, a compact mixed evaluation set, deterministic tool execution, and first-pass direct-answer versus prompt-only tool-use baselines.

EviTool-VL 是一个面向小型视觉语言模型的证据驱动视觉工具推理研究项目。项目目标是研究小模型在回答图表、文档、科学图示和 GUI 定位问题时，能否通过显式调用本地视觉工具获得更可靠、可审计的中间证据。本仓库当前围绕 Qwen2.5-VL-3B-Instruct 与 Qwen3-VL-4B-Instruct，构建小规模混合评测集、确定性工具层，以及直接回答和 prompt-only 工具调用两类基线。

## Project Status / 项目状态

This repository is an early research implementation, not a polished package. The committed code covers the first milestone:

- environment and model-loading scripts;
- local mini-evaluation data construction;
- direct-answer VLM baseline evaluation;
- prompt-only JSON action-observation-final tool-use evaluation;
- local visual tools for crop, zoom, OCR, detection, measurement, marking, tracing, and virtual GUI clicking;
- first reports for baseline behavior, tool-use behavior, and remaining failure cases.

本仓库目前是早期研究实现，不是完整工程化发布包。已提交内容覆盖第一阶段里程碑：

- 环境检查与模型下载脚本；
- 本地 mini evaluation 数据构建；
- 直接回答 VLM 基线评测；
- 基于纯文本 JSON 协议的 action-observation-final 工具调用评测；
- crop、zoom、OCR、detect、measure、mark、trace、click 等本地视觉工具；
- 直接回答、工具调用和失败模式的初步报告。

Large model weights, generated outputs, local dataset caches, third-party repositories, and runtime artifacts are intentionally excluded from Git.

模型权重、生成输出、本地数据缓存、第三方仓库和运行时产物都不会提交到 Git。

## Research Goal / 研究目标

The central hypothesis is that small VLMs should not be evaluated only by final-answer accuracy. For visually grounded tasks, a useful model should also expose which image regions, text spans, detected boxes, or measured quantities support the answer.

核心假设是：小型 VLM 不应只按最终答案正确率评估。在需要视觉定位和局部读数的任务中，一个有用的模型还应给出答案所依赖的图像区域、OCR 文本片段、检测框或测量结果。

The intended EviTool-VL loop is:

1. receive an image and question;
2. decide whether local visual evidence is needed;
3. call deterministic tools through a fixed JSON schema;
4. receive structured observations with evidence ids;
5. produce a concise final answer that cites the evidence.

预期的 EviTool-VL 推理流程是：

1. 接收图像和问题；
2. 判断是否需要局部视觉证据；
3. 通过固定 JSON schema 调用确定性工具；
4. 接收带 evidence id 的结构化 observation；
5. 输出简洁答案，并引用支撑答案的证据。

## Repository Layout / 仓库结构

```text
.
├── configs/                  # SFT and GRPO experiment configuration drafts
├── datasets/                 # Evaluation-set construction scripts
├── eval/                     # Direct and tool-use baseline evaluators
├── reports/                  # Current result tables and analysis notes
├── scripts/                  # Environment checks, model downloads, smoke tests, inference helpers
├── tools/                    # Local visual tool implementation and JSON runner
├── requirements.txt          # Core Python dependencies
├── requirements-ocr.txt      # OCR backend dependencies
└── README.md
```

The top-level planning notes outside this directory are not part of the repository and are not required to run the committed code.

仓库目录外的中文规划文档没有纳入本仓库；运行当前已提交代码不依赖这些外部文档。

## Models / 模型

The first supported model family is Qwen-VL:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

Model files should be stored outside the Git repository, by default under:

```bash
/root/models
```

模型文件应放在 Git 仓库外，默认位置为：

```bash
/root/models
```

Download helper:

```bash
python3 scripts/download_models.py
```

Download a single model:

```bash
python3 scripts/download_models.py --model Qwen/Qwen2.5-VL-3B-Instruct
```

The downloader uses Hugging Face by default and can fall back to ModelScope when configured.

下载脚本默认使用 Hugging Face，也支持在配置后回退到 ModelScope。

## Environment / 环境

Recommended setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-ocr.txt
```

If your environment already provides CUDA, PyTorch, and OCR packages, install only the missing dependencies. OCR backends are optional at runtime; the tool layer has fallback behavior, but OCR quality depends strongly on the available backend.

如果环境中已经安装 CUDA、PyTorch 和 OCR 相关包，只需补齐缺失依赖。OCR 后端在运行时是可选的；工具层有 fallback 逻辑，但 OCR 质量会明显受后端影响。

Check the local environment:

```bash
python3 scripts/check_env.py
```

## Data / 数据

The mini evaluation set is built from public datasets and saved locally. By default, `datasets/build_eval_mini.py` writes to:

```bash
/root/models/datasets/evitool_eval_mini
```

The expected JSONL file is:

```text
eval_mini_100.jsonl
```

It contains a 100-sample mix:

| Task | Count | Purpose |
|---|---:|---|
| ChartQA | 30 | chart/table reading and numeric reasoning |
| DocVQA | 25 | document and figure text reading |
| AI2D | 20 | science diagram multiple-choice QA |
| ScreenSpot-v2 | 25 | GUI element grounding |

构建脚本会从公开数据集中抽取样本并保存到本地，默认生成 100 条混合评测数据：ChartQA 30 条、DocVQA 25 条、AI2D 20 条、ScreenSpot-v2 25 条。数据文件和图像不提交到 Git。

Build the dataset:

```bash
python3 datasets/build_eval_mini.py --output-dir /root/models/datasets/evitool_eval_mini
```

Create the local symlink expected by the evaluation scripts:

```bash
mkdir -p data
ln -s /root/models/datasets/evitool_eval_mini data/eval_mini
```

`data/eval_mini` is ignored by Git because it is machine-local.

`data/eval_mini` 已加入 `.gitignore`，因为它是本机数据链接。

## Visual Tool Layer / 视觉工具层

All tools use JSON actions and return JSON observations. Coordinates use pixel-space `[x1, y1, x2, y2]` boxes unless otherwise stated.

所有工具都使用 JSON action，并返回 JSON observation。除非另有说明，坐标格式为像素坐标 `[x1, y1, x2, y2]`。

Example:

```bash
python3 -m tools.runner \
  --image data/eval_mini/images/chartqa/0000.png \
  --action '{"tool":"crop","args":{"bbox":[80,80,760,520]}}' \
  --pretty
```

Supported tools:

| Tool | Purpose / 用途 |
|---|---|
| `inspect` | Image metadata and basic statistics / 图像尺寸和基础统计信息 |
| `crop` | Crop a region / 裁剪局部区域 |
| `zoom` | Crop and enlarge a region / 裁剪并放大局部区域 |
| `ocr` | Read text with EasyOCR, PaddleOCR, Tesseract, or fallback heuristics / 读取文本 |
| `detect` | Detect layout, text, UI, chart bars, or color regions / 检测版面、文本、UI、柱状图或颜色区域 |
| `measure` | Measure sizes, centers, distances, IoU, and bar heights / 测量尺寸、中心、距离、IoU 和柱高 |
| `mark` | Draw visual annotations / 绘制标注 |
| `visualize` | Render boxes and points / 可视化框和点 |
| `trace` | Execute a sequence of tool actions / 执行工具调用序列 |
| `click` | Record a virtual GUI click or selected bbox as evidence / 记录 GUI 点击点或目标框 |

See `tools/README.md` for backend notes and lower-level usage.

更多后端说明和底层调用方式见 `tools/README.md`。

## Evaluation / 评测

### Direct-Answer Baseline / 直接回答基线

Run Qwen2.5-VL-3B:

```bash
python3 eval/eval_baseline.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --data data/eval_mini/eval_mini_100.jsonl \
  --image-root data/eval_mini \
  --output outputs/baseline_3b_direct_eval_mini_100.jsonl \
  --max-new-tokens 128
```

Run Qwen3-VL-4B:

```bash
python3 eval/eval_baseline.py \
  --model /root/models/Qwen3-VL-4B-Instruct \
  --data data/eval_mini/eval_mini_100.jsonl \
  --image-root data/eval_mini \
  --output outputs/baseline_4b_direct_eval_mini_100.jsonl \
  --max-new-tokens 128
```

The evaluator writes JSONL predictions and a `.summary.json` metrics file next to the output.

评测脚本会写出 JSONL 预测文件，并在同目录生成 `.summary.json` 指标文件。

### Prompt-Only Tool-Use Baseline / 纯 Prompt 工具调用基线

This baseline does not use native function calling. The model emits JSON text, `eval/eval_tool_baseline.py` parses it, executes local tools, appends observations to the chat, and asks for the next action or final answer.

该基线不使用原生 function calling。模型输出 JSON 文本，由 `eval/eval_tool_baseline.py` 解析并执行本地工具，再把 observation 追加回对话，让模型继续输出下一步 action 或 final answer。

Example:

```bash
python3 eval/eval_tool_baseline.py \
  --model /root/models/Qwen3-VL-4B-Instruct \
  --data data/eval_mini/eval_mini_100.jsonl \
  --image-root data/eval_mini \
  --output outputs/tool_4b_eval_mini_100.jsonl \
  --max-tool-steps 2 \
  --max-new-tokens 384
```

Smoke test:

```bash
python3 eval/eval_tool_baseline.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --sample-per-task 1 \
  --max-tool-steps 2 \
  --output outputs/tool_smoke_3b_eval_mini.jsonl \
  --max-new-tokens 384
```

## Current Results / 当前结果

Current results are first-pass baselines on `data/eval_mini/eval_mini_100.jsonl`. They should be treated as diagnostic numbers, not final model rankings.

当前结果来自 `data/eval_mini/eval_mini_100.jsonl` 的第一轮基线实验，应视为诊断指标，而不是最终模型排名。

Direct-answer baseline:

| Model | Text Relaxed | ChartQA | DocVQA | AI2D | ScreenSpot IoU@0.5 | ScreenSpot Pointing |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct | 81.33% | 63.33% | 92.00% | 95.00% | 0.00% | 12.00% |
| Qwen3-VL-4B-Instruct | 76.00% | 66.67% | 80.00% | 85.00% | 0.00% | 8.00% |

Prompt-only tool-use baseline with `max_tool_steps=2`:

| Model | Text Relaxed | ChartQA | DocVQA | AI2D | ScreenSpot IoU@0.5 | ScreenSpot Pointing | Protocol Error | Evidence Closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct | 56.00% | 36.67% | 72.00% | 65.00% | 12.00% | 56.00% | 73.00% | 18.00% |
| Qwen3-VL-4B-Instruct | 77.33% | 73.33% | 80.00% | 80.00% | 12.00% | 24.00% | 0.00% | 100.00% |

Main interpretation:

- Direct answering can perform well on some text QA tasks but gives no auditable evidence.
- Direct GUI grounding is weak for both models, with 0% IoU@0.5 on this mini set.
- Prompt-only tools improve GUI pointing and expose evidence, but the smaller 3B model has many protocol errors.
- Tool use changes the failure mode; it does not solve grounding by prompt engineering alone.

主要结论：

- 直接回答在部分文本 QA 上表现尚可，但没有可审计证据；
- 两个模型的直接 GUI grounding 都很弱，在该 mini set 上 IoU@0.5 为 0%；
- prompt-only 工具调用改善了 GUI pointing，并能暴露 evidence，但 3B 模型协议错误率很高；
- 工具调用改变了错误形态，但仅靠 prompt 不能彻底解决 grounding 问题。

Detailed reports:

- `reports/baseline_table.md`
- `reports/baseline_findings.md`
- `reports/tool_baseline_full.md`
- `reports/tool_baseline_smoke.md`
- `reports/failure_cases.md`

## Training Plan / 训练计划

Configuration drafts are provided for:

- QLoRA supervised fine-tuning on tool traces;
- GRPO-style optimization with answer, grounding, evidence, protocol, and efficiency rewards.

配置草案包括：

- 基于工具轨迹的 QLoRA SFT；
- 使用 answer、grounding、evidence、protocol 和 efficiency 奖励的 GRPO 类优化。

The next practical step is to build supervised action-observation-final traces from correct direct and tool-assisted examples, then train models to follow the fixed tool schema before applying reward optimization.

下一步更实际的工作是从正确的直接回答样本和工具辅助样本中构造 action-observation-final 监督轨迹，先让模型学会稳定遵循工具 schema，再进行奖励优化。

## Reproducibility Notes / 可复现性说明

- Generated outputs under `outputs/` are ignored by Git.
- Dataset caches and symlinks under `data/` are ignored by Git.
- Model checkpoints are expected under `/root/models` or a user-provided path.
- Evaluation results depend on model revision, `transformers` version, OCR backend, GPU availability, and decoding settings.
- The current scripts use deterministic greedy decoding by default when `temperature=0`.

可复现性注意事项：

- `outputs/` 下的生成结果不会提交；
- `data/` 下的数据缓存和符号链接不会提交；
- 模型 checkpoint 默认位于 `/root/models`，也可以通过参数传入其它路径；
- 结果会受模型版本、`transformers` 版本、OCR 后端、GPU 可用性和解码参数影响；
- 当前脚本在 `temperature=0` 时默认使用确定性 greedy decoding。

## Limitations / 局限

- The mini evaluation set has only 100 samples and is intended for iteration speed, not final benchmarking.
- OCR and detection tools are heuristic-heavy and not yet tuned per dataset.
- Prompt-only JSON control is brittle for weaker models.
- GUI grounding remains weak even after tool use.
- Training and reward optimization configs are present, but full training runs are not yet committed as reproducible experiments.

局限包括：

- mini evaluation 只有 100 条样本，适合快速迭代，不适合作为最终 benchmark；
- OCR 和 detection 仍以启发式为主，尚未针对各数据集充分调参；
- 对较弱模型来说，纯 prompt JSON 控制不稳定；
- 即使加入工具调用，GUI grounding 仍然较弱；
- 仓库已有训练与奖励优化配置草案，但完整可复现实验尚未提交。

## License / 许可

No explicit license file is currently included. Add a license before redistribution or external reuse.

当前仓库尚未包含明确的 license 文件。在对外分发或复用前应补充 license。
