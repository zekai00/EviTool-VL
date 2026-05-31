# Repository Guidelines

## 默认工作偏好

- 默认用中文回答。
- 凡是写给用户看的回复、项目文档、阶段报告、路线规划和实验说明，英文术语或缩写第一次出现必须先用中文解释，再继续使用。不要假设用户已经知道这些英文词在本项目里的具体含义。例如：
  - `Adapter`：适配器，把外部环境的输入输出格式翻译成我们项目内部格式。
  - `Rollout`：一次完整尝试轨迹，从任务开始、执行多步动作，到成功、失败或超时结束。
  - `Schema`：数据格式规范，说明一条记录有哪些字段、字段叫什么、值是什么类型。
  - `Policy`：策略，也就是决定下一步动作的主体，可以是脚本、随机规则或 Qwen 模型。
  - `Verifier`：验证器，用程序检查任务是否完成，并返回 reward。
  - `Reward`：奖励分数，用来告诉 RL 这一步或这条轨迹做得好不好。
  - `On-policy RL`：在线策略强化学习，用当前模型自己产生的新轨迹训练当前模型。
  - `Teacher policy`：教师策略，用更强或远程模型生成示范或做基线评测；它不是要被 RL 更新的当前模型。
  - `Current model`：当前模型，指本轮训练真正会更新参数的本地 Qwen2.5-VL/Qwen3-VL checkpoint。
- 面向用户说明时不要过度省略。尤其是涉及 BrowserGym、MiniWoB++、CUA-Gym、Playwright、GRPO、SFT、rollout、reward、verifier、sandbox 等内容时，要解释“它是什么、为什么要做、输入是什么、输出是什么、当前做到哪一步、还没做到什么”。
- 如果同一段里有多个英文名词，要逐个解释；如果一个名词在不同上下文里含义会变，比如 `policy` 既可能指随机脚本也可能指 Qwen 模型，要明确当前说的是哪一种。
- 报告实验状态时要区分“环境已经可执行”“脚本已经接入”“模型已经训练”“模型已经学会任务”这几个层级，不要用过短的结论跳过中间状态。
- 如果某个步骤只是验证可接入，而不是完成训练或证明模型能力，必须明确说清楚，避免把环境验证误说成模型已经学会任务。
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
- DashScope 上的 `qwen3.6-flash`、`qwen3.7-max` 等远程模型只能作为 teacher/baseline policy，用于验证环境、生成少量 warmup 轨迹或做对照。它们不是本项目要训练的 current model，不能把它们的 rollout 误称为本地 Qwen2.5-VL/Qwen3-VL 的 on-policy RL。
- 真正的 current model 必须是本地可训练 checkpoint，例如 `/root/models/Qwen2.5-VL-3B-Instruct`、本地 Qwen3-VL 或其 LoRA/SFT/RL checkpoint。只有这个模型自己 rollout 后再被 verifier reward 更新，才算严格 on-policy RL。
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
- 正式可复用 BrowserRL 数据集统一放在 `/root/datasets/browser_rl/`；`outputs/` 只放临时实验输出，`checkpoints/` 放训练 checkpoint，`/root/models` 放底模权重。不要把大型数据、模型权重、训练输出或评测 raw output 提交到 Git。

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
- 当前已经有 `local_qwen` 本地策略接入口：它能加载本地 Qwen2.5-VL/Qwen3-VL base checkpoint 和 LoRA adapter，并在 browser rollout 中输出 direct GUI action JSON。
- 截至 2026-05-27 23:02 CST，`local_qwen` 接口已跑通，但本地 Qwen2.5-VL LoRA 在 smoke_form_01 上 success_rate 仍为 0.0；这代表模型尚未学会任务，不应误写成已经完成 on-policy RL。
- 截至 2026-05-27 23:56 CST，`history-aware trajectory SFT` v1 已实现并训练，本地 Qwen2.5-VL 在 20 个 local smoke tasks 上 success_rate=0.3、valid_action_rate=1.0。
- `history-aware` 指提示词里包含 step、历史动作、上一轮 verifier progress、动作空间和坐标规则，用来让模型学习多步状态判断。
- 截至 2026-05-28 00:38 CST，300 条 local browser tasks 已构建并通过 scripted oracle 校验，oracle success_rate=1.0；history-aware SFT v2 已训练，本地 Qwen2.5-VL 在独立 30 个 test tasks 上 success_rate=0.333、valid_action_rate=1.0。
- 当前 300 任务 v2 结果说明模型已经学会合法 JSON action 格式，但还没有稳定学会精确坐标 grounding。失败主要集中在 form/search/menu/table 的点击高度和结果定位。
- 截至 2026-05-28 03:45 CST，1000 条 local browser tasks 已构建，加入坐标扰动增强和失败恢复示范后训练 history-aware SFT v3。本地 Qwen2.5-VL-3B LoRA 在独立 test100 上 success_rate=0.82、valid_json_rate=1.0、valid_action_rate=1.0、policy_error_rate=0.0。
- 这次 0.82 是 SFT warmup 后的可执行环境评测，不是严格 on-policy RL；它说明 current model 已达到进入小规模 verifier-guided RL 的门槛。
- 当前剩余弱项主要是 form 和 choice：form success_rate=0.6364，choice success_rate=0.6667；table/todo 已达到 1.0，search/menu 已超过 0.82。
- Qwen2.5-VL-7B-Instruct 已下载到 `/root/models/Qwen2.5-VL-7B-Instruct`，截至 2026-05-28 04:34 CST 已完成 5/5 个 safetensors 分片，目录大小 17G；`transformers.AutoConfig` 本地读取验证通过，`model_type=qwen2_5_vl`。
- 截至 2026-05-28 12:19 CST，form/choice 定向 SFT repair v1 已完成但不采用：`qwen25vl_3b_history_aware_sft_v3_form_choice_repair_v1_lora` 在 test100 上 best/root success_rate=0.76，低于 v3 baseline 0.82。它把 choice 从 0.6667 提到 0.8889，但 search 从 0.8636 掉到 0.5909，table 从 1.0 掉到 0.8824，说明发生了能力干扰。
- 后续不要默认使用 repair v1 adapter 作为 current model；current model 仍应优先使用 `checkpoints/qwen25vl_3b_history_aware_sft_v3_1000_aug_lora`。repair 数据只作为分析材料，除非重新做更保守的 loss weight/replay/early-stop 方案。
- 截至 2026-05-28 13:26 CST，`scripts/train_browser_rl_onpolicy_grpo.py` 已跑通 verifier-guided on-policy Browser RL 小规模闭环：当前模型采样动作，Playwright reset/replay 分支执行，verifier reward 组内归一化后更新 LoRA。激进版 test30 success_rate=0.5667，安全版 test30 success_rate=0.70，等于 v3 baseline first30=0.70。
- 本轮 on-policy adapter 只证明工程链路可用，不代表性能超过 v3 baseline；默认 current model 仍不要切到 `outputs/onpolicy_browser_rl_grpo_smoke_v1_safe_20260528_1318/adapter`，除非后续扩大训练并通过 test100/test set evaluation gate。
- 截至 2026-05-28 14:38 CST，新的 BrowserRL 2000 task suite 已建立并迁移到正式数据目录：`/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344`，train/val/test=1600/200/200。旧中间目录 `outputs/browser_rl_task_suite_2000_20260528_1338` 的切分不采用，因为 choice/advanced 子模板覆盖有偏。
- 当前正式 validation gate 是 v3 baseline 在新 val200 上的 `success_rate=0.745`，路径：`outputs/rollouts_local_qwen25vl_history_aware_v3_2000_val200_20260528_1412/merged/family_metrics.json`。后续 RL adapter 必须至少超过这个 val200 baseline，且主 family 不应明显退化。
- `test200` 暂时保留作最终验收，不要频繁用于调参或选择 adapter。快速 gate 可用 `/root/datasets/browser_rl/task_suites/browser_rl_task_suite_2000_20260528_1344/val_balanced_70_tasks.jsonl`，但最终接受 adapter 仍看 val200。
- 候选框补框、full-candidate score cache、candidate-only GRPO 都不是当前第一优先级。
- 截至 2026-05-28 17:30 CST，100 trainable groups 的安全版 verifier-guided on-policy GRPO 已通过 val200 gate：adapter=`outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter`，val200 success_rate=0.790，高于 v3 baseline 0.745；validation=`outputs/onpolicy_browser_rl_grpo_100tg_safe_validation_20260528_1729`，444/444 valid，error_count=0。
- 后续 on-policy RL 的 current model 可以切到 `outputs/onpolicy_browser_rl_grpo_100tg_safe_20260528_1646/adapter`；但它还没有经过 `test200` 最终验收，不要把它写成最终模型。
- 下一阶段应扩大到 200-300 trainable groups，并加入 v3 replay 或 KL 稳定项。重点补 advanced_scroll/dialog、choice_checkbox、table_action，避免 advanced 和 table 在 val200 上继续退化。
- 截至 2026-05-28 21:32 CST，213 trainable groups 的 replay 版 GRPO 已完成：adapter=`outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter`，val_balanced_70=0.8406，val200=0.850，validation=`outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_validation_20260528_2132`，606/606 valid。
- 213tg replay adapter 总体优于 100tg adapter，但 table 明显退化：v3 SFT table=0.88，100tg table=0.84，213tg table=0.76；advanced 仍为 0.6667，低于 v3 SFT 0.7778。因此它只能作为下一轮 RL candidate/current model，不能跑通用最终验收，不要跑或宣称 `test200` 结果。
- 截至 2026-05-28 22:46 CST，`scripts/train_browser_rl_onpolicy_grpo.py` 的 collect-only 工程问题已修：默认 `--stream-collect` 会边采边写 `groups.jsonl.tmp`、`rollouts.jsonl.tmp` 和 `live_summary.json`；`--resume-collect` 会读取临时文件、跳过已覆盖 task，且目标已满足时不加载本地 Qwen 直接退出。验证目录：`outputs/onpolicy_collect_stream_smoke_20260528_check`。
- 截至 2026-05-29 00:33 CST，正式可复用 BrowserRL 数据已归档到 `/root/datasets/browser_rl/`：`task_suites/` 放固定任务套件，`sft/` 放 LLaMA-Factory 可读 SFT 数据，`onpolicy/` 放可复用 on-policy group 数据。后续新数据优先写入这个目录，仓库内 `outputs/` 只作为临时中间产物。
- 截至 2026-05-29 02:16 CST，table/advanced repair 版 289 trainable groups 已完成：adapter=`outputs/onpolicy_browser_rl_grpo_table_advanced_repair_289tg_safe_20260529_0101/adapter`，val_balanced_70=0.9420，val200=0.9100，test200=0.8900。它是当前阶段最强 adapter，可以作为下一轮 current model。
- 本轮 test200 已首次用于阶段性最终验收，后续不要继续用 test200 频繁调参。主要剩余问题是 advanced_scroll：val advanced_scroll=0.6667，但 test advanced_scroll=0.0000；下一轮应改 scroll action sampling 或构建 scroll recovery 数据，不要只盲采。
- 截至 2026-05-29 05:40 CST，advanced_scroll 滚动点击专项修复已完成两轮。第一轮 409tg adapter=`outputs/onpolicy_browser_rl_grpo_advanced_scroll_repair_409tg_safe_20260529_0358/adapter` 在新 advanced_scroll progress val30 上 success_rate=0.3000，val_balanced_70=0.9710；第二轮 220tg scroll-click adapter=`outputs/onpolicy_browser_rl_grpo_advanced_scroll_click_repair_220tg_safe_20260529_0501/adapter` 在同一专项 val30 上 success_rate=0.4000，val_balanced_70=0.9420。
- 不要把 220tg scroll-click adapter 直接写成新的全局主线。它证明 target-click 候选注入有效，但还没有达到 0.75 目标，也没有通过 val200；当前保守主线仍是已通过 val200/test200 的 289tg adapter。409tg 可作为下一轮 scroll 修复候选基座，但也需要 val200 验证后才能替换主线。
- advanced_scroll 当前主要错误已经从“不会滚动”变成“滚动后目标可见但点击坐标偏高/重复点击同一错误位置”。后续应优先做 after-scroll click SFT/replay、小规模坐标距离 reward、重复无效点击惩罚，再继续扩大 RL。
- 截至 2026-05-29 15:00 CST，`scripts/train_browser_rl_onpolicy_grpo.py` 中 `--inject-scroll-candidates` 和 `--inject-target-click-candidates` 默认已改为关闭。默认采集必须让 VLM 自己采样动作；只有做探索增强或消融实验时才显式打开 injected candidates。带 injected candidates 的实验不能称为严格纯 on-policy。
- 截至 2026-05-29 16:46 CST，pure VLM advanced_scroll 20tg 小实验完成：采集 `/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_pure_vlm_collect_20260529_1611`，15 rollouts、20 trainable groups、0 injected samples；训练 adapter=`outputs/onpolicy_browser_rl_grpo_advanced_scroll_pure_vlm_20tg_safe_20260529_1637/adapter`；advanced_scroll progress val30=0.1667，与 289tg baseline 持平，不采用。重要工程修正：单一唯一动作状态现在可以继续 rollout，但不形成 GRPO group。
- pure VLM 实验说明：模型会稳定先 scroll，但 val 失败仍集中在目标可见后的点击偏差。下一步应改 reward，引入目标可见后的点击距离奖励和重复失败点击惩罚，而不是恢复默认 injected candidates。
- 截至 2026-05-29 18:25 CST，advanced_scroll 距离 reward 小实验完成：在 `scripts/train_browser_rl_onpolicy_grpo.py` 中加入 `--target-distance-reward-*` 和 `--repeat-click-*` 参数；采集 `/root/datasets/browser_rl/onpolicy/onpolicy_browser_rl_advanced_scroll_distance_reward_collect_20260529_1756`，23 groups、20 trainable groups、12 rollouts，validation error_count=0；训练 adapter=`outputs/onpolicy_browser_rl_grpo_advanced_scroll_distance_reward_20tg_safe_20260529_1816/adapter`；advanced_scroll progress val30=0.1667，未超过 289tg baseline，不采用。
- 距离 reward 实验说明：reward shaping 确实增加了组内 reward 差异，但 20tg/5 optimizer steps 没有改变推理时的 after-scroll 点击失败形态。后续若继续纯 VLM on-policy 路线，应扩大到至少 80-150 trainable groups，或加入不泄露目标候选的 after-scroll click SFT replay。
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
