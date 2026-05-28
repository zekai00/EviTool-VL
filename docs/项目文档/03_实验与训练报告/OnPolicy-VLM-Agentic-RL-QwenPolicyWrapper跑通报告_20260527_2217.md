# OnPolicy-VLM-Agentic-RL Qwen Policy Wrapper 跑通报告

- 生成时间：2026-05-27 22:17 CST
- 工作区：`/root/Workspace/VLM`
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 当前状态：已能让 DashScope Qwen teacher/baseline policy 看截图、读任务目标、输出 direct GUI action JSON，并在可执行环境里 rollout。

## 1. 本轮结论

`Qwen policy wrapper` 指 Qwen 策略封装器：把环境截图和任务目标发给 Qwen 模型，再把模型输出解析成 `click/type/press/scroll/finish` 等 GUI action JSON。

本轮已经跑通第一版：`qwen3.6-flash-2026-04-16` 可以作为 `teacher/baseline policy`（教师/基线策略，即远程模型负责示范或评测对照，但不被我们训练更新），直接在本地 Playwright smoke tasks 和 BrowserGym/MiniWoB++ 上产生动作并执行。这里已经不是静态 SFT 数据，也不是候选框选择，而是“观察截图 -> 远程 teacher 输出动作 -> 环境执行 -> verifier/reward 返回结果”的 rollout。

这还不是本地 Qwen2.5-VL/Qwen3-VL 的 on-policy RL。`Current model`（当前模型）必须是本地可训练 checkpoint，例如本地 Qwen2.5-VL/Qwen3-VL 或其 LoRA checkpoint；只有它自己 rollout 后再被 reward 更新，才算严格 on-policy RL。本轮只是把 teacher/baseline 策略入口和环境链路跑通。

## 2. 本轮新增/修改文件

1. `envs/browser_rl/qwen_policy.py`
   - 新增 DashScope Qwen teacher/baseline policy wrapper。
   - 支持读取 `.env` 中的 `DASHSCOPE_API_KEY`，但不会把 key 写入日志。
   - 使用 DashScope OpenAI-compatible chat completions 接口。
   - 输入：截图、任务目标、viewport、最近动作历史。
   - 输出：单步 GUI action JSON。
   - 兼容模型常见输出变体，例如 `tap` -> `click`、`input` -> `type`、`x:[90]` -> `x:90`。

2. `scripts/run_browser_rl_rollouts.py`
   - 新增 `--policy qwen_dashscope`。
   - 新增 Qwen 相关参数：
     - `--qwen-models`
     - `--qwen-base-url`
     - `--qwen-api-key-env`
     - `--qwen-env-file`
     - `--qwen-temperature`
     - `--qwen-max-tokens`
     - `--qwen-timeout`
     - `--qwen-retries`
     - `--qwen-max-history`
   - summary 增加：
     - `qwen_valid_json_rate`
     - `qwen_valid_action_rate`
     - `qwen_error_rate`
     - `qwen_models`

3. `envs/browser_rl/__init__.py`
   - 导出 `QwenDashScopePolicy`。

## 3. 关键设计

1. 动作空间仍然是 direct GUI action。
   - 不走候选框。
   - 不输出 `candidate_id`。
   - 直接输出 `{"action":"click","x":...,"y":...}` 这类动作。

2. 坐标统一用 0-1000 归一化坐标。
   - prompt 中明确说明不要输出像素坐标。
   - 对 MiniWoB++ 这种小截图，额外加入当前 viewport 的换算公式，避免模型把像素坐标误当归一化坐标。

3. API 错误不会污染环境。
   - 如果模型调用失败或动作解析失败，默认 fallback 为 `{"action":"wait"}`。
   - 错误信息写入 `policy_info`，不影响整个 rollout collector 继续运行。

4. 支持 teacher/baseline 模型 fallback。
   - 默认模型列表：
     - `qwen3.6-flash-2026-04-16`
     - `qwen3.7-max-2026-05-20`
     - `qwen3.7-max`
   - 本轮实际成功使用的是 `qwen3.6-flash-2026-04-16`。
   - 注意：这些远程 DashScope 模型不是要训练的 current model。

## 4. 验证命令与结果

### 4.1 语法检查

```bash
python3 -m py_compile \
  envs/browser_rl/qwen_policy.py \
  envs/browser_rl/__init__.py \
  scripts/run_browser_rl_rollouts.py \
  envs/browser_rl/cua_gym_sandbox.py \
  scripts/run_cua_gym_mock_web_sandbox.py
```

结果：通过。

### 4.2 本地 Playwright smoke tasks 小评测

```bash
python3 scripts/run_browser_rl_rollouts.py \
  --env-source local_smoke \
  --policy qwen_dashscope \
  --output-dir outputs/rollouts_local_qwen_dashscope_eval5_20260527_2206 \
  --limit 5 \
  --max-steps 8 \
  --headless \
  --qwen-models qwen3.6-flash-2026-04-16,qwen3.7-max-2026-05-20,qwen3.7-max \
  --qwen-timeout 120 \
  --qwen-retries 0
```

结果：

| 指标 | 数值 |
|---|---:|
| rollouts | 5 |
| success_rate | 1.0 |
| avg_steps | 4.8 |
| avg_valid_action_rate | 1.0 |
| qwen_valid_json_rate | 1.0 |
| qwen_valid_action_rate | 1.0 |
| qwen_error_rate | 0.0 |

逐任务结果：

| task_id | success | steps | total_reward |
|---|---:|---:|---:|
| `smoke_form_01` | true | 5 | 1.4 |
| `smoke_form_02` | true | 5 | 1.4 |
| `smoke_form_03` | true | 5 | 1.4 |
| `smoke_form_04` | true | 5 | 1.4 |
| `smoke_todo_01` | true | 4 | 1.4 |

### 4.3 BrowserGym/MiniWoB++ 小评测

```bash
python3 scripts/run_browser_rl_rollouts.py \
  --env-source browsergym_miniwob \
  --policy qwen_dashscope \
  --output-dir outputs/rollouts_miniwob_qwen_dashscope_eval5_20260527_2206 \
  --limit 5 \
  --max-steps 5 \
  --headless \
  --qwen-models qwen3.6-flash-2026-04-16,qwen3.7-max-2026-05-20,qwen3.7-max \
  --qwen-timeout 120 \
  --qwen-retries 0
```

结果：

| 指标 | 数值 |
|---|---:|
| rollouts | 5 |
| success_rate | 0.4 |
| avg_steps | 4.2 |
| avg_valid_action_rate | 1.0 |
| qwen_valid_json_rate | 1.0 |
| qwen_valid_action_rate | 1.0 |
| qwen_error_rate | 0.0 |

逐任务结果：

| task_id | success | steps | total_reward |
|---|---:|---:|---:|
| `browsergym/miniwob.click-button` | true | 1 | 1.0 |
| `browsergym/miniwob.click-checkboxes` | false | 5 | 0.0 |
| `browsergym/miniwob.enter-text` | false | 5 | 0.0 |
| `browsergym/miniwob.choose-list` | false | 5 | 0.0 |
| `browsergym/miniwob.focus-text` | true | 5 | 1.0 |

## 5. 当前判断

本轮已经证明：

1. DashScope Qwen 视觉模型可以作为 teacher/baseline policy 接入可执行环境 rollout。
2. 模型能稳定输出合法 GUI action JSON。
3. 本地可控任务已经能端到端成功。
4. MiniWoB++ 上已经有非零成功率，但还不稳定，需要 prompt、动作后处理和任务特化策略继续增强。
5. 这一步可以为本地 Qwen2.5-VL/Qwen3-VL 的 SFT warmup 提供少量高质量示范，但不能替代本地模型的 on-policy rollout。

## 6. 当前限制

1. MiniWoB++ 成功率还低。
   - `click-checkboxes`、`enter-text`、`choose-list` 没过。
   - 主要问题可能是小控件坐标、输入框 focus、select/list 操作和任务完成条件判断。

2. rollout 速度偏慢。
   - 每步都要调用一次远程 VLM。
   - 5 个 MiniWoB++ 任务耗时明显长于本地 smoke tasks。

3. 还没有本地模型 RL 更新。
   - 当前只是用远程 DashScope Qwen teacher/baseline policy 采集 rollout。
   - 下一步要让本地 Qwen2.5-VL/Qwen3-VL 输出 action JSON，并用它自己的 rollout 接 GRPO 或其他 on-policy RL 算法。

## 7. 下一步计划

1. 增加 rollout schema validator。
   - 确保每条轨迹都有截图、动作、reward、policy_info、终止状态和错误信息。

2. 增强 Qwen policy prompt 和动作后处理。
   - 特别处理 checkbox、list/select、text input focus。
   - 加入失败后反思规则：如果上一步 reward 仍为 0，下一步不要重复点击同一位置。

3. 扩大评测到 20 个本地 smoke tasks 和 20 个 MiniWoB++ tasks。
   - 先得到稳定 baseline。
   - 再决定是否做 SFT warmup 或直接小规模 GRPO。

4. 启动第一版本地 current model rollout。
   - 先用 teacher 成功轨迹做少量 action JSON SFT warmup。
   - 再让本地 Qwen2.5-VL/Qwen3-VL checkpoint 自己 rollout。
   - 用 verifier reward 训练本地 Qwen2.5-VL/Qwen3-VL LoRA。
