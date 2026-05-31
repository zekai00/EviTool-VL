# EviTool-VL

<p align="center">
  <a href="#中文"><kbd>中文</kbd></a>
  <a href="#english"><kbd>English</kbd></a>
</p>

## 中文

EviTool-VL 是一个面向视觉语言模型的 GUI tool-call reinforcement learning 项目。

项目目标不是让模型只做静态截图问答，而是训练一个 VLM agent：它能够观察浏览器页面截图，输出可执行动作 JSON，由 Playwright 浏览器环境真实执行动作，再通过 verifier reward 进行 SFT 和 on-policy GRPO 训练。

核心任务定义：

```text
Executable GUI Tool-Call RL for Vision-Language Agents
```

也就是：给定一个可重置、可执行、可自动验证的浏览器任务，模型需要多步操作页面并完成目标。

### 为什么做这个项目

普通 VLM benchmark 往往只检查最终文本答案。但真实 GUI agent 需要具备更强的闭环能力：

```text
observe screenshot -> emit action JSON -> execute in browser -> receive verifier feedback -> continue or finish
```

本项目重点研究：

- VLM action JSON 协议学习；
- history-aware trajectory SFT；
- 可 reset/step/verifier 的浏览器 RL 环境；
- verifier-guided on-policy GRPO；
- clipped ratio、reference KL、SFT replay 等稳定化策略；
- 小参数 VLM 在真实可执行 GUI 任务上的能力边界。

### 当前能力

当前主线模型是 Qwen2.5-VL-3B-Instruct 加 LoRA adapter。模型能在本地浏览器任务中完成多类 GUI 操作：

- 填写表单；
- 搜索并选择结果；
- 点击菜单项；
- 操作表格按钮；
- 添加 todo；
- 选择 checkbox/radio/select；
- 处理 tab 和 dialog；
- 部分处理 scroll 后点击任务。

当前最强 adapter 在 200 条 held-out test tasks 上达到：

| model | test success | avg steps | valid JSON | valid action |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B + staged GRPO adapter | **0.890** | 3.465 | 1.000 | 1.000 |

按任务类型拆分：

| family | test count | success |
|---|---:|---:|
| todo | 20 | **1.000** |
| menu | 25 | 0.920 |
| form | 45 | 0.911 |
| search | 40 | 0.900 |
| choice | 33 | 0.879 |
| table | 25 | 0.840 |
| advanced | 12 | 0.667 |

主要短板是 `advanced_scroll`，在 test split 中仍未稳定解决。

### BrowserRL 任务环境

项目构建了一个本地 BrowserRL task suite，包含 2000 个可执行任务：

| split | tasks |
|---|---:|
| train | 1600 |
| validation | 200 |
| test | 200 |
| total | 2000 |

任务类型分布：

| family | total |
|---|---:|
| form | 450 |
| search | 400 |
| choice | 350 |
| menu | 250 |
| table | 250 |
| todo | 200 |
| advanced | 100 |

所有任务都通过 scripted oracle 校验，oracle success rate 为 1.000。这说明任务、浏览器 reset/step、动作执行和 verifier 判分链路是闭环可解的。

### SFT 结果

原始 instruct 底模并不会自动适配本项目的 GUI action JSON 协议。在同一套 prompt 和执行环境下，原始 Qwen2.5-VL-3B 在 balanced validation subset 上 success 为 0。

SFT 阶段结果：

| stage | data scale | eval set | success | note |
|---|---:|---|---:|---|
| raw Qwen2.5-VL-3B | 0 | val balanced | 0.000 | 未学会可执行动作协议 |
| history-aware SFT v2 | 300 tasks | test30 | 0.333 | 初步学会多步状态 |
| history-aware SFT v3 | 1000 tasks + augmentation | test100 | 0.820 | 达到可用 warmup |
| history-aware SFT v3 | 1000 tasks + augmentation | val200 | 0.745 | 后续 RL 起点 |
| form/choice repair SFT | mixed replay | test100 | 0.760 | 局部修复但整体退化，未采用 |

v3 SFT 使用坐标扰动和失败恢复示范，训练样本约 7.9k rows，valid JSON rate 和 valid action rate 均达到 1.000。

### On-Policy GRPO 结果

RL 阶段从 history-aware SFT v3 出发，使用模型自己在环境中采样得到的 on-policy groups 训练。Verifier 根据真实浏览器状态给 reward。

正式 val200 对比：

| method | trainable groups | val200 success | avg steps |
|---|---:|---:|---:|
| SFT v3 baseline | 0 | 0.745 | 3.810 |
| step-wise GRPO 100tg | 100 | 0.790 | 3.680 |
| step-wise GRPO + replay 213tg | 213 | 0.850 | 3.560 |
| table/advanced repair GRPO 289tg | 289 | **0.910** | **3.445** |

最终 289tg adapter 首次打开 test200 后达到 0.890 success，说明提升不是只在 validation 上成立。

### GRPO 消融

项目进一步比较了不同 GRPO 变体。

同一 SFT v3 起点、同一批 60 trainable groups 的 step-wise 对比：

| method | val balanced | val200 | note |
|---|---:|---:|---|
| SFT v3 baseline | 0.826 | 0.745 | no RL |
| vanilla step-wise GRPO | 0.841 | 0.750 | 仅组内 reward 标准化 |
| clipped + reference KL GRPO | **0.884** | **0.770** | epsilon=0.2, beta=0.01 |
| staged 289tg GRPO | 0.942 | 0.910 | 分阶段累计 on-policy groups |

Trajectory-level GRPO v1 也已跑通：

| method | val balanced | val200 | decision |
|---|---:|---:|---|
| trajectory-level GRPO v1 | 0.913 | 0.900 | 低于 289tg staged adapter，未升级为主线 |
| staged 289tg step-wise GRPO | **0.942** | **0.910** | 当前主线 |

结论：trajectory-level RL 链路可行，但当前阶段 step-wise verifier-guided GRPO 配合 replay 更稳定；clipped ratio 和 reference KL 在小规模 probe 中有正向信号，值得作为后续主线补强。

### GUI Candidate Selection 前置实验

在真正 BrowserRL 之前，项目也做过 GUI candidate selection 预研：给定候选框，让模型选择正确目标区域。这个方向现在不是主线，但它帮助明确了 GUI grounding 的瓶颈。

OS-Atlas Linux 2k candidate pool：

| policy | validation count | pointing | IoU@0.5 | avg IoU |
|---|---:|---:|---:|---:|
| random candidate | 423 | 0.021 | 0.017 | 0.015 |
| top-1 heuristic | 423 | 0.026 | 0.021 | 0.020 |
| oracle upper bound | 423 | **0.617** | **0.522** | **0.443** |

Candidate-constrained GRPO ablation on val100:

| method | avg reward | pointing | IoU@0.5 | avg IoU |
|---|---:|---:|---:|---:|
| SFT warmup | 0.255 | 0.110 | 0.090 | 0.081 |
| old CC-GRPO | 0.346 | 0.200 | 0.170 | 0.138 |
| listwise warmup | 0.337 | 0.200 | 0.160 | 0.135 |
| listwise + KL + CC-GRPO | **0.366** | **0.240** | **0.180** | **0.157** |

这个结果说明：候选约束下的 RL 确实能改善目标选择，但最终 GUI agent 不能只停留在“从候选框里选答案”，因此项目主线已转向可执行 BrowserRL。

### 早期视觉工具基线

项目早期还实现了 crop、zoom、OCR、detect、measure、click 等本地视觉工具，并在混合 VQA/GUI benchmark 上做了 direct-answer 与 prompt-only tool-use 对比。

Medium-600 evaluation selected results：

| model / setting | text exact | text relaxed | GUI pointing | evidence closed |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B direct, no adapter | 0.780 | 0.830 | 0.215 | - |
| Qwen2.5-VL-3B tool-use, no adapter | 0.398 | 0.478 | 0.100 | 0.178 |
| Qwen2.5-VL-3B tool-use after SFT | 0.643 | 0.720 | **0.450** | 0.365 |
| Qwen3-VL-4B direct, no adapter | 0.808 | 0.855 | 0.010 | - |
| Qwen3-VL-4B tool-use after SFT | 0.718 | 0.805 | 0.110 | **0.988** |

这些早期结果显示：prompt-only tool use 会牺牲部分文本 QA 分数，但能显著提高 evidence-closed 行为和部分 GUI grounding 能力。后续 BrowserRL 因此转向“可执行环境 + verifier reward”，而不是只做 prompt 工具调用。

### 方法概览

当前主线训练流程：

1. Build executable tasks：构建可 reset、可 step、可 verifier 的浏览器任务。
2. Scripted oracle rollout：用专家策略验证任务可解，并生成初始轨迹。
3. History-aware SFT：让 VLM 学会根据截图、历史动作和 verifier progress 输出动作 JSON。
4. On-policy rollout collection：当前模型自己在 Playwright 环境中采样动作。
5. Verifier-guided GRPO：同一状态下多次采样，使用组内 reward 相对优势训练。
6. Replay stabilization：混入 SFT replay 或旧 on-policy groups，降低能力遗忘。
7. Held-out evaluation：使用 validation/test success rate、valid JSON、valid action 和 family-level metrics 评估。

### 代码结构

```text
configs/      Training and experiment configs
envs/         BrowserRL and sandbox environments
eval/         Baseline and tool-use evaluators
rl/           RL environment utilities
scripts/      Dataset construction, rollout, SFT, GRPO, evaluation scripts
tools/        Deterministic visual tools and JSON runner
training/     Training helpers
```

Large datasets, checkpoints, generated rollouts, model weights, and private experiment logs are intentionally kept outside the public repository.

### Next Steps

- Make clipped-ratio + reference-KL GRPO the default step-wise RL variant.
- Improve exploration for `advanced_scroll`.
- Run a larger trajectory-level GRPO comparison after the environment is stable.
- Evaluate larger VLM backbones such as Qwen2.5-VL-7B under the same BrowserRL suite.
- Publish a cleaned benchmark subset and reproducible evaluation script.

## English

EviTool-VL is a GUI tool-call reinforcement learning project for vision-language models.

The goal is not static screenshot question answering. The goal is to train a VLM agent that observes browser screenshots, emits executable action JSON, interacts with a real Playwright browser environment, and learns from verifier rewards through trajectory SFT and on-policy GRPO.

Core task:

```text
Executable GUI Tool-Call RL for Vision-Language Agents
```

Given a resettable, executable, and automatically verifiable browser task, the model must operate the page through multiple steps and complete the goal.

### Motivation

Standard VLM benchmarks usually evaluate only the final text answer. A GUI agent needs a closed interaction loop:

```text
observe screenshot -> emit action JSON -> execute in browser -> receive verifier feedback -> continue or finish
```

This project studies:

- VLM action-JSON protocol learning;
- history-aware trajectory SFT;
- reset/step/verifier browser RL environments;
- verifier-guided on-policy GRPO;
- stabilization with clipped ratio, reference KL, and SFT replay;
- the capability boundary of compact VLMs on executable GUI tasks.

### Current Capability

The current mainline model is Qwen2.5-VL-3B-Instruct with a LoRA adapter. It can complete several browser GUI task families:

- form filling;
- search and result selection;
- menu selection;
- table actions;
- todo creation;
- checkbox/radio/select choices;
- tab and dialog interaction;
- partial scroll-and-click tasks.

The strongest adapter reaches the following held-out test performance:

| model | test success | avg steps | valid JSON | valid action |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B + staged GRPO adapter | **0.890** | 3.465 | 1.000 | 1.000 |

Breakdown by task family:

| family | test count | success |
|---|---:|---:|
| todo | 20 | **1.000** |
| menu | 25 | 0.920 |
| form | 45 | 0.911 |
| search | 40 | 0.900 |
| choice | 33 | 0.879 |
| table | 25 | 0.840 |
| advanced | 12 | 0.667 |

The main unresolved weakness is `advanced_scroll`.

### BrowserRL Environment

The project builds a local BrowserRL task suite with 2000 executable tasks:

| split | tasks |
|---|---:|
| train | 1600 |
| validation | 200 |
| test | 200 |
| total | 2000 |

Task-family distribution:

| family | total |
|---|---:|
| form | 450 |
| search | 400 |
| choice | 350 |
| menu | 250 |
| table | 250 |
| todo | 200 |
| advanced | 100 |

All 2000 tasks are solved by a scripted oracle with 1.000 success rate, validating the task definitions, browser execution loop, and verifier.

### SFT Results

The raw instruct model does not naturally follow this project's executable GUI action protocol. Under the same prompt and environment, raw Qwen2.5-VL-3B has 0 success on a balanced validation subset.

SFT progression:

| stage | data scale | eval set | success | note |
|---|---:|---|---:|---|
| raw Qwen2.5-VL-3B | 0 | val balanced | 0.000 | no executable-action protocol |
| history-aware SFT v2 | 300 tasks | test30 | 0.333 | initial multi-step learning |
| history-aware SFT v3 | 1000 tasks + augmentation | test100 | 0.820 | usable warmup |
| history-aware SFT v3 | 1000 tasks + augmentation | val200 | 0.745 | RL starting point |
| form/choice repair SFT | mixed replay | test100 | 0.760 | local repair but overall regression |

The adopted SFT v3 stage uses coordinate jittering and recovery demonstrations. It contains about 7.9k training rows and reaches 1.000 valid-JSON and valid-action rates.

### On-Policy GRPO Results

The RL stage starts from history-aware SFT v3. The model collects its own on-policy groups in the browser environment, and a verifier scores the resulting states.

Formal val200 comparison:

| method | trainable groups | val200 success | avg steps |
|---|---:|---:|---:|
| SFT v3 baseline | 0 | 0.745 | 3.810 |
| step-wise GRPO 100tg | 100 | 0.790 | 3.680 |
| step-wise GRPO + replay 213tg | 213 | 0.850 | 3.560 |
| table/advanced repair GRPO 289tg | 289 | **0.910** | **3.445** |

The final 289tg adapter reaches 0.890 success on the held-out test200 split.

### GRPO Ablations

Same SFT v3 starting point, same 60 trainable groups:

| method | val balanced | val200 | note |
|---|---:|---:|---|
| SFT v3 baseline | 0.826 | 0.745 | no RL |
| vanilla step-wise GRPO | 0.841 | 0.750 | group-normalized reward only |
| clipped + reference KL GRPO | **0.884** | **0.770** | epsilon=0.2, beta=0.01 |
| staged 289tg GRPO | 0.942 | 0.910 | accumulated on-policy groups |

Trajectory-level GRPO v1:

| method | val balanced | val200 | decision |
|---|---:|---:|---|
| trajectory-level GRPO v1 | 0.913 | 0.900 | not promoted |
| staged 289tg step-wise GRPO | **0.942** | **0.910** | current mainline |

The trajectory-level pipeline works, but step-wise verifier-guided GRPO with replay is currently more stable. Clipped ratio and reference KL show positive signal and are planned for the next mainline version.

### GUI Candidate Selection Pre-Study

Before the current BrowserRL direction, the project studied GUI candidate selection: given candidate boxes, choose the correct target. This is no longer the mainline task, but it clarified the grounding bottleneck.

OS-Atlas Linux 2k candidate pool:

| policy | validation count | pointing | IoU@0.5 | avg IoU |
|---|---:|---:|---:|---:|
| random candidate | 423 | 0.021 | 0.017 | 0.015 |
| top-1 heuristic | 423 | 0.026 | 0.021 | 0.020 |
| oracle upper bound | 423 | **0.617** | **0.522** | **0.443** |

Candidate-constrained GRPO ablation on val100:

| method | avg reward | pointing | IoU@0.5 | avg IoU |
|---|---:|---:|---:|---:|
| SFT warmup | 0.255 | 0.110 | 0.090 | 0.081 |
| old CC-GRPO | 0.346 | 0.200 | 0.170 | 0.138 |
| listwise warmup | 0.337 | 0.200 | 0.160 | 0.135 |
| listwise + KL + CC-GRPO | **0.366** | **0.240** | **0.180** | **0.157** |

The candidate-selection experiments showed that RL can improve target selection, but an agent should not be limited to selecting from precomputed boxes. This motivated the shift to executable BrowserRL.

### Early Visual Tool Baselines

The early project stage implemented deterministic visual tools such as crop, zoom, OCR, detect, measure, and click, then compared direct-answer VLMs with prompt-only tool use.

Medium-600 selected results:

| model / setting | text exact | text relaxed | GUI pointing | evidence closed |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B direct, no adapter | 0.780 | 0.830 | 0.215 | - |
| Qwen2.5-VL-3B tool-use, no adapter | 0.398 | 0.478 | 0.100 | 0.178 |
| Qwen2.5-VL-3B tool-use after SFT | 0.643 | 0.720 | **0.450** | 0.365 |
| Qwen3-VL-4B direct, no adapter | 0.808 | 0.855 | 0.010 | - |
| Qwen3-VL-4B tool-use after SFT | 0.718 | 0.805 | 0.110 | **0.988** |

These results motivated the move from prompt-only tool use to executable environments and verifier reward.

### Method Overview

The current pipeline:

1. Build executable tasks.
2. Validate them with scripted oracle rollouts.
3. Train history-aware trajectory SFT.
4. Collect on-policy rollouts from the current model.
5. Train with verifier-guided GRPO.
6. Stabilize with SFT replay and old on-policy groups.
7. Evaluate on held-out validation and test splits.

### Repository Layout

```text
configs/      Training and experiment configs
envs/         BrowserRL and sandbox environments
eval/         Baseline and tool-use evaluators
rl/           RL environment utilities
scripts/      Dataset construction, rollout, SFT, GRPO, evaluation scripts
tools/        Deterministic visual tools and JSON runner
training/     Training helpers
```

Large datasets, checkpoints, generated rollouts, model weights, and private experiment logs are intentionally kept outside the public repository.

### Next Steps

- Make clipped-ratio + reference-KL GRPO the default step-wise RL variant.
- Improve exploration for `advanced_scroll`.
- Run a larger trajectory-level GRPO comparison after the environment is stable.
- Evaluate larger VLM backbones such as Qwen2.5-VL-7B under the same BrowserRL suite.
- Publish a cleaned benchmark subset and reproducible evaluation script.
