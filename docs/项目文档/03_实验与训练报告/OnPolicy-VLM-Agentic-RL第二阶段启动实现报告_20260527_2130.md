# OnPolicy-VLM-Agentic-RL 第二阶段启动实现报告

- 生成时间：2026-05-27 21:30 CST
- 工作区：`/root/Workspace/VLM`
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 当前主线：真正的 `On-policy RL`（在线策略强化学习，即用当前模型自己在环境里产生的新轨迹训练当前模型）+ VLM GUI agent（视觉语言模型图形界面智能体）。

## 1. 本阶段结论

本阶段完成的是“环境接入与轨迹采集底座”，不是 Qwen 模型训练完成。

现在已经可以把本地 Playwright smoke tasks 和 BrowserGym/MiniWoB++ 任务都转换成同一种 rollout 记录格式，并能把 CUA-Gym 的 `mock_web`（模拟网页任务子集）筛出来做下一步沙箱执行。也就是说，我们已经从“静态截图候选框选择”进一步走到“可 reset、可 step、可记录 reward 的可执行环境接入”。

## 2. 关键名词

- `BrowserGym`：浏览器强化学习任务集合和接口。它把 MiniWoB++ 等网页任务包装成类似 Gym 的 `reset/step` 形式，让智能体能反复尝试网页操作。
- `MiniWoB++`：一组轻量网页操作任务，例如点击按钮、勾选框、输入文字。它偏合成，但很适合先跑通在线 RL 闭环。
- `Adapter`：适配器，把外部环境的 observation/action/reward 翻译成我们项目内部统一格式。本轮新增的 BrowserGym adapter 就负责把 BrowserGym 的截图、目标、动作执行结果和奖励转换成我们的 rollout schema。
- `Rollout`：一次完整尝试轨迹，从初始截图开始，策略输出动作，环境执行动作，返回新截图和 reward，直到成功、失败或达到最大步数。
- `Schema`：数据格式规范，说明一条 rollout 里必须有哪些字段，例如 `task_id`、`goal`、`trajectory`、`action`、`reward_step`、`success`。
- `Policy`：策略，也就是决定下一步动作的主体。本阶段验证了脚本 oracle、随机策略和 noop 策略；后续要把这里替换成 Qwen2.5-VL/Qwen3-VL 或 DashScope 上的 Qwen 模型。
- `CUA-Gym mock_web`：CUA-Gym 里的模拟网页任务集合，任务目标和 reward 文件来自数据集，不需要一开始直接跑真实桌面环境。

## 3. 已完成内容

1. 更新 `AGENTS.md`。
   - 明确要求：英文术语或缩写第一次出现必须先用中文解释。
   - 明确报告要解释“是什么、为什么做、输入是什么、输出是什么、做到哪一步、还没做到什么”。
   - 明确要区分环境接入、脚本接入、模型训练、模型学会任务这几个层级。

2. 新增 BrowserGym adapter。
   - 文件：`envs/browser_rl/browsergym_adapter.py`
   - 作用：把 BrowserGym/MiniWoB++ 的 `reset/step/obs/reward` 转成我们自己的 observation 和 rollout schema。
   - 当前支持动作：`click`、`double_click`、`type`、`press`、`hotkey`、`scroll`、`drag`、`wait`、`finish`。
   - 坐标继续使用 `0-1000` 归一化坐标，执行时映射到浏览器 viewport 像素。

3. 新增统一 rollout collector。
   - 文件：`scripts/run_browser_rl_rollouts.py`
   - 作用：用同一个入口采集本地 Playwright smoke tasks 和 BrowserGym/MiniWoB++ 的 rollout。
   - 输出：
     - `rollouts.jsonl`：完整轨迹。
     - `sft_messages.json`：成功轨迹中可用于 SFT warmup 的单步消息。
     - `summary.json`：成功率、平均步数、合法动作率等统计。

4. 新增 CUA-Gym mock_web inspector。
   - 文件：`scripts/inspect_cua_gym_mock_web.py`
   - 作用：从 CUA-Gym metadata 中筛出 mock_web 任务，解包小样本 task bundle，检查 `task.json` 和 `reward.py`。
   - 当前做到：静态检查和 `reward.py` 语法编译。
   - 尚未做到：在隔离沙箱里动态运行 `reward.py`。

5. 新增手动交互脚本。
   - 文件：`scripts/play_browser_rl_task.py`
   - 作用：让人直接输入 action JSON，亲手验证 `reset/step/screenshot/verifier/reward` 是否按预期工作。
   - 服务器没有图形化界面时需要用 `--headless`，`--no-headless` 会因为没有 XServer 失败，这是环境限制，不是任务逻辑错误。

## 4. 本轮验证结果

| 验证项 | 输出目录 | 数量 | 成功率 | 合法动作率 | 说明 |
|---|---|---:|---:|---:|---|
| 本地 smoke task + scripted oracle | `outputs/rollouts_local_oracle_20260527_2125` | 5 rollouts | 100% | 100% | oracle 是脚本答案，用来验证环境和记录器正确 |
| BrowserGym/MiniWoB++ + noop policy | `outputs/rollouts_miniwob_noop_20260527_2125` | 3 rollouts | 0% | 100% | noop 策略只等待，不会完成任务，成功率 0% 符合预期 |
| BrowserGym/MiniWoB++ + random policy | `outputs/rollouts_miniwob_random_20260527_2125` | 2 rollouts | 0% | 100% | 随机策略只验证 action schema 和 step 能跑通，不代表模型能力 |
| CUA-Gym mock_web inspection | `outputs/cua_gym_mock_web_inspection_20260527_2125` | 10 bundles | - | - | metadata 7897 条，mock_web 1068 条，`reward.py` 编译通过 10/10 |

额外统计：

- 本地 smoke task scripted oracle 平均步数：4.6。
- 本地 smoke task 生成 SFT rows：23。
- BrowserGym noop/random 平均步数：2。
- CUA-Gym mock_web 检查没有 blocker。

## 5. 执行过的主要命令

```bash
python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy scripted_oracle \
  --output-dir outputs/rollouts_local_oracle_20260527_2125 \
  --limit 5 \
  --headless
```

```bash
python3 scripts/run_browser_rl_rollouts.py \
  --env-source browsergym_miniwob \
  --policy noop \
  --output-dir outputs/rollouts_miniwob_noop_20260527_2125 \
  --max-steps 2 \
  --limit 3 \
  --headless
```

```bash
python3 scripts/run_browser_rl_rollouts.py \
  --env-source browsergym_miniwob \
  --policy random \
  --output-dir outputs/rollouts_miniwob_random_20260527_2125 \
  --max-steps 2 \
  --limit 2 \
  --headless \
  --seed 7
```

```bash
python3 scripts/inspect_cua_gym_mock_web.py \
  --output-dir outputs/cua_gym_mock_web_inspection_20260527_2125 \
  --limit 10
```

```bash
python3 -m py_compile \
  envs/browser_rl/actions.py \
  envs/browser_rl/task_spec.py \
  envs/browser_rl/verifier.py \
  envs/browser_rl/recorder.py \
  envs/browser_rl/playwright_env.py \
  envs/browser_rl/browsergym_adapter.py \
  envs/browser_rl/__init__.py \
  scripts/build_browser_rl_smoke_tasks.py \
  scripts/run_browser_rl_smoke.py \
  scripts/check_browsergym_miniwob.py \
  scripts/inspect_cua_gym.py \
  scripts/inspect_cua_gym_mock_web.py \
  scripts/convert_webarena_infinity_trajectories.py \
  scripts/play_browser_rl_task.py \
  scripts/run_browser_rl_rollouts.py
```

结果：语法检查通过。

## 6. 当前还没有完成什么

1. 还没有做 Qwen on-policy rollout。
   - 现在的 policy 还是 scripted oracle、random、noop。
   - 下一步要接 Qwen 输出 action JSON。

2. BrowserGym/MiniWoB++ 还没有成功策略。
   - 当前 random/noop 只是验证环境可执行和记录格式正确。
   - 下一步需要 scripted policy、SFT policy 或 Qwen policy。

3. CUA-Gym mock_web 还没有动态运行 reward。
   - 本轮只证明 metadata、bundle、`reward.py` 静态语法没问题。
   - 下一步要做 sandbox runner，在隔离目录或容器里执行 setup/reward。

4. WebArena-Infinity 仍然被 Hugging Face repo tree 访问超时阻塞。
   - 后续应改成离线分片下载或镜像同步。

## 7. 下一步建议

优先顺序：

1. 做 CUA-Gym mock_web sandbox runner。
   - 目标：让 mock_web 任务真正支持 `reset/step/reward`，而不是只静态检查文件。

2. 做 Qwen policy wrapper。
   - 目标：把截图和 goal 发给 Qwen，让模型直接输出 `{"action": "...", ...}`。
   - 可以先支持 DashScope 模型，例如 `qwen3.6-plus` 或 `qwen3.7-max`，再接本地 Qwen2.5-VL/Qwen3-VL。

3. 做 rollout schema validator。
   - 目标：自动检查每条 rollout 是否包含截图、动作、reward、终止状态、错误信息和 policy version。

4. 跑 20 个本地 smoke tasks 的 action JSON SFT smoke。
   - 目标：先让模型稳定输出合法 action JSON，再进入 RL。

5. 启动 Verifier-Guided GRPO。
   - `Verifier-Guided GRPO`：由 verifier reward 引导的 GRPO 强化学习，重点优化成功率、合法动作率和轨迹长度。
