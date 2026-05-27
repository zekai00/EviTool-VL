# Repository Guidelines

## 默认工作偏好

- 默认用中文回答。
- 修改代码前先说明计划；开始任务时先看 `git status --short`，避免覆盖用户改动。
- 开始较长任务前阅读 `docs/codex-worklog.md`，了解最近实验状态；如果文件不存在则继续执行，不要阻塞。
- 重要阶段完成后更新 `docs/codex-worklog.md`，记录完成内容、改动文件、实验命令、未解决问题和下一步计划。
- 写代码时保持注释清楚但克制：解释不明显的算法逻辑、数据 schema、训练假设；不要写空泛注释。
- 修改后尽量运行相关检查，例如 `py_compile`、dry-run、小样本评测或数据质量脚本。
- 不要擅自引入新依赖；确实需要新依赖时先说明原因、影响和替代方案。
- 对不确定的实验假设、数据来源、模型路径或指标口径要明确说明。

## 项目上下文

- 工作区根目录通常是 `/root/Workspace/VLM`，仓库根目录是 `/root/Workspace/VLM/EviTool-VL`。
- 当前主线已经切换为真正的 on-policy VLM agentic RL：训练 Qwen2.5-VL / Qwen3-VL 在可执行 GUI 环境中看截图、输出 direct GUI action、执行动作、接收新截图和 verifier reward，并通过在线 rollout 做 RL。
- 主动作空间应优先使用 direct action，而不是候选框选择：
  - `click(x, y)`
  - `double_click(x, y)`
  - `type(text)`
  - `press(key)`
  - `hotkey(keys)`
  - `scroll(dx, dy)`
  - `drag(x1, y1, x2, y2)`
  - `wait`
  - `finish`
- 坐标默认使用 `0-1000` 归一化坐标，执行时映射到当前 viewport 或截图像素。
- 当前 candidate / bbox / full-aware 数据不再是主线 RL 目标，只保留为：
  - trajectory SFT warmup
  - 单步 grounding auxiliary loss
  - debug 可视化和评测对照
  - 可选工具/teacher，不作为默认 action space
- 新增 GUI-RL 环境时，优先从本地 browser Playwright Gym 做起，再接 BrowserGym/MiniWoB++，不要一开始直接上 OSWorld/真实桌面重环境。
- 真正 on-policy RL 数据应来自当前模型在线 rollout，而不是静态截图候选框数据。每条 RL 任务必须尽量具备 `reset`、`step`、`screenshot`、`verifier` 和 `reward`。
- 大型数据、模型权重、训练输出和评测输出应放在 `outputs/`、`checkpoints/` 或 `/root/models`，不要提交到 Git。

## 项目结构与模块组织

- `tools/`：本地视觉工具，提供 JSON action/observation 风格接口，例如 `detect`、`ocr`、`crop`、`zoom`、`measure`、`visualize`、`click`。
- `rl/`：RL 环境、reward、policy helper、trajectory/reward 构造逻辑。历史上包含 GUI candidate selection；新增工作应优先面向 direct-action agentic RL。
- `envs/`：如果新增可执行环境，优先放 browser/GUI RL Gym，例如 Playwright reset/step/screenshot/verifier/recorder。
- `scripts/`：实验脚本，包括 direct-action trajectory 构建、Playwright task 构建、on-policy rollout、GRPO 训练、candidate 数据构建和历史 full-aware/candidate 评测。
- `datasets/`：trace/SFT 数据转换与评测子集构建。
- `eval/`：direct/tool baseline 评测入口。
- `configs/`：SFT/GRPO/Lora 训练配置。
- `reports/`：实验报告、指标表和阶段总结。
- `docs/`：项目工作日志和给 agent 的长期上下文。
- `third_party/`：OmniParser、TRL、LlamaFactory、Qwen3-VL 等外部代码。

## 常用命令

安装依赖：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-ocr.txt
```

检查环境：

```bash
python3 scripts/check_env.py
```

运行工具冒烟测试：

```bash
python3 scripts/smoke_tools.py \
  --image data/eval_mini/images/chartqa/0000.png \
  --out-dir outputs/tools_smoke
```

运行直接回答或工具调用评测：

```bash
python3 eval/eval_baseline.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --output outputs/direct.jsonl

python3 eval/eval_tool_baseline.py \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --output outputs/tool.jsonl
```

对改动过的 Python 文件做语法检查：

```bash
python3 -m py_compile path/to/changed_file.py
```

构建当前 full-aware candidate RL 数据：

```bash
python3 scripts/build_gui_candidate_full_aware_data.py \
  --input-dir outputs/gui_candidate_rl_os_atlas_linux_2k_c60 \
  --output-dir outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware \
  --report reports/gui_candidate_full_aware_dataset_os_atlas_linux_2k_c60.md \
  --max-actions 16 \
  --min-train-oracle-iou 0.05 \
  --require-train-oracle-hit \
  --seed 42
```

构建或训练新的 on-policy GUI agentic RL 时，应优先新增 direct-action/browser 环境命令；历史 full-aware candidate 命令只作为兼容和对照。

CC-GRPO 数据加载 dry-run：

```bash
python3 scripts/train_gui_candidate_cc_grpo.py \
  --train-data outputs/gui_candidate_rl_os_atlas_linux_2k_c60_full_aware/train.jsonl \
  --output-dir /tmp/full_aware_cc_grpo_dryrun \
  --max-train-rows 8 \
  --dry-run
```

## 编码风格与命名约定

- 使用 Python 3、4 空格缩进、类型标注和清晰函数名。
- JSON schema 使用 snake_case 字段名。
- 像素框统一使用 `[x1, y1, x2, y2]`，不要混入 xywh，除非字段名明确说明。
- 数据构建、评测和训练采样要尽量 deterministic，显式传入 `--seed`。
- 保持新脚本可复现：参数写入 `summary.json` 或报告，输出路径稳定。
- 修改已有实验脚本时保持向后兼容；如果字段重命名，尽量保留旧字段别名，例如 `cc_action_ids`。

## 测试与质量检查

- 当前没有正式 pytest 套件，优先使用：
  - `python3 -m py_compile ...`
  - 工具 smoke test
  - 训练脚本 `--dry-run`
  - 小样本 eval
  - 数据质量统计和 split leakage 检查
- 修改 `tools/` 时，同时验证模块 runner 和下游 evaluator。
- 修改 candidate RL 数据构建时，至少检查：
  - train/val image overlap
  - candidate/action id 是否存在于完整候选池
  - reward std 是否为 0
  - oracle hit / IoU@0.5 / avg IoU
  - 输出 JSONL 行数与 summary 是否一致
- 修改 on-policy GUI-RL 环境时，至少检查：
  - task reset 是否确定性可复现
  - scripted oracle success rate
  - screenshot 是否成功保存并可被模型读取
  - action JSON parse rate / valid action rate
  - verifier 是否只读环境状态，不向模型泄露答案
  - train/val/test 是否按 task template、app 或 seed 做防泄露切分
  - rollout 记录是否包含 policy version、task id、每步 observation/action/reward/logprob

## 当前实验注意事项

- 当前推荐的 warmup 数据是 strict v2 direct/tool trajectory，可从 `/root/models/datasets/gui_tool_trajectory_v2_os_atlas_linux_1k_unique_strict_20260527_1933` 转换或使用。
- 候选框补框、full-candidate score cache、candidate-only GRPO 都不是当前第一优先级。
- 下一阶段优先实现本地 Playwright GUI-RL Gym、可 verifier 任务集、scripted oracle demo、on-policy rollout recorder 和 Verifier-Guided GRPO。
- 静态截图数据不能替代 on-policy RL 环境；没有 `reset/step/verifier` 的数据只应用于 SFT warmup 或辅助 loss。

## 提交与 PR 指南

- 提交摘要使用简短祈使句，例如 `Add full-aware candidate data builder`。
- 每个提交保持范围清晰，不要把代码修改和大型生成产物混在一起。
- PR 或阶段报告应说明：
  - 变更目标
  - 修改文件
  - 运行命令
  - 关键指标
  - 未解决问题
  - 后续计划
- 大型 JSONL、模型 adapter、checkpoint、图片 overlay、评测 raw output 通常不提交；必要时只提交报告和小型 summary。

## 安全与配置

- 不要提交 API key、模型权重、数据集缓存或私有输出。
- 供应商访问凭据使用环境变量，例如 `DASHSCOPE_API_KEY`；报告时只说明是否配置，不要输出具体值。
- 不要运行破坏性 git 命令，例如 `git reset --hard` 或 `git checkout --`，除非用户明确要求。
