# 真正 On-Policy VLM Agentic RL 浏览器环境与算法规划

- 生成时间：2026-05-27 20:08 CST
- 项目根目录：`/root/Workspace/VLM`
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 当前主线：从 candidate selection / bbox 补框转向 direct-action、可执行、可验证的 on-policy VLM agentic RL。

## 0. 一句话结论

真正的 VLM agentic RL 不应该以“给定候选框后选择 `candidate_id`”为主线，而应该训练模型在真实可执行 GUI 环境中：

```text
观察截图 + 任务目标
-> 输出 GUI action JSON
-> 环境执行鼠标/键盘动作
-> 返回新截图和执行状态
-> verifier 给 reward
-> 用当前模型 on-policy rollout 做 RL 更新
```

候选框数据只保留为 warmup、辅助 grounding loss、debug 可视化和对照实验，不再作为主训练目标。

## 1. “本地 GUI-RL Gym，先从 browser Playwright 环境做”是什么意思

这里的 Gym 不是指 OpenAI Gym 的具体包，而是指一个标准化的可交互 RL 环境接口：

```python
obs, info = env.reset(task_id, seed)
obs, reward, terminated, truncated, info = env.step(action)
```

对本项目来说，`obs` 主要是浏览器截图，`action` 是鼠标/键盘 GUI action，`reward` 来自程序化 verifier。

### 为什么先做 browser Playwright

浏览器环境比直接做 OSWorld/真实桌面更适合作为第一阶段：

1. 可 reset。
   - 每个任务可以打开固定本地 URL，重置 localStorage、数据库、页面状态和随机 seed。

2. 可 verifier。
   - Playwright 可以读取 DOM、表单值、URL、页面状态、数据库 mock 状态。
   - reward 不需要靠人工看截图判断。

3. 可自动生成大量任务。
   - 同一个模板可以随机化文本、按钮位置、表格行、表单字段、颜色、弹窗、菜单层级。

4. 可控且成本低。
   - 不需要完整桌面虚拟机。
   - 不需要 LibreOffice/GNOME 这种重环境。
   - 2x3090 的主要成本可以留给 VLM 生成和训练。

5. 接近真实 GUI agent 的核心问题。
   - 模型仍然只能看截图。
   - 动作仍然是 click/type/scroll/press。
   - verifier 只在环境侧使用，不泄露给模型。

### 这个环境要提供什么

最小接口：

```json
{
  "task_id": "form_login_0001",
  "goal": "把用户名填成 Alice，密码填成 123456，然后点击登录",
  "observation": {
    "screenshot": "obs_000.png",
    "viewport": [1280, 720],
    "step": 0,
    "history": []
  },
  "action_space": [
    "click",
    "double_click",
    "type",
    "press",
    "hotkey",
    "scroll",
    "drag",
    "wait",
    "finish"
  ],
  "max_steps": 8
}
```

模型输出：

```json
{"action": "click", "x": 412, "y": 318}
```

或：

```json
{"action": "type", "text": "Alice"}
```

坐标建议统一使用 `0-1000` 归一化坐标，执行时映射到当前 viewport 像素：

```text
x_px = round(x_norm / 1000 * viewport_width)
y_px = round(y_norm / 1000 * viewport_height)
```

## 2. 有没有可以直接用的数据集

有，但要区分两类：可执行环境和静态轨迹数据。

### 2.1 可用于 on-policy RL 的可执行环境

这些可以直接或半直接用于 RL，因为它们支持环境交互、reset 或 reward/verifier。

| 名称 | 是否适合第一阶段 | 用途 | 主要问题 |
|---|---:|---|---|
| BrowserGym | 高 | 统一 browser agent Gym 接口，可接 MiniWoB/WebArena/VisualWebArena/WorkArena 等 | 需要适配我们的 VLM 输入、action JSON 和训练 loop |
| MiniWoB++ | 高 | 轻量浏览器 RL 任务，适合先跑通 on-policy | 页面偏合成，视觉复杂度有限 |
| WebArena | 中 | 真实网站任务，适合第二阶段评测 | 自托管复杂，任务长，RL 成本高 |
| VisualWebArena | 中 | 更贴近 VLM web agent，含视觉 grounding | 设置和评测成本高，不适合第一天就上 |
| WorkArena/OpenApps | 中 | 类企业应用任务 | 依赖和环境较重 |
| AndroidWorld | 低到中 | 移动端 agent，任务多且可随机化 | 需要 Android emulator，不建议先做 |
| OSWorld | 低到中 | 桌面 OS agent 评测 | 环境重，任务长，先做会拖慢闭环 |

推荐第一阶段：

1. 先接入 CUA-Gym、BrowserGym/MiniWoB++、WebArena-Infinity 这类已有可执行环境或轨迹资源，验证它们能否落到本项目的 direct-action rollout schema。
2. 同时做少量本地 Playwright smoke tasks，用来控制变量、调试 action parser、screenshot recorder、verifier 和 reward，不再先手写大规模任务集。
3. 最后用 WebArena/VisualWebArena/OSWorld 做泛化评测，而不是一开始就拿重环境训练。

参考：

- BrowserGym GitHub: `https://github.com/ServiceNow/BrowserGym`
- BrowserGym paper: `https://arxiv.org/abs/2412.05467`
- WebArena paper: `https://arxiv.org/abs/2307.13854`
- VisualWebArena paper: `https://arxiv.org/abs/2401.13649`
- AndroidWorld GitHub: `https://github.com/google-research/android_world`

### 2.2 可用于 SFT / warmup 的静态数据

这些数据有 demo、截图、HTML 或历史动作，但不能天然做 on-policy RL，因为不能稳定 reset 和执行 verifier。

| 名称 | 用途 | 局限 |
|---|---|---|
| Mind2Web | web agent 行为克隆、HTML/action grounding | 多数场景不是可执行 RL 环境 |
| WebLINX | 真实网页交互轨迹 SFT | 更偏离线 demo，不等于在线环境 |
| OS-Atlas | GUI grounding / 坐标 warmup | 静态截图，无法 step 和 verifier |
| AITW | 移动端动作轨迹 SFT | Android 环境复现成本高 |
| 当前 v2 strict trajectory | action JSON 格式 warmup、单步 click grounding | 仍是静态截图，不是 on-policy |

结论：可以直接用现成数据做 SFT/warmup，但真正 on-policy RL 需要可执行环境。没有 verifier 的静态数据不能替代 RL 环境。

## 3. 是否必须自己构建 300-500 个任务

不是“必须从零手写 300-500 个”，但必须拥有一批可 reset、可 verifier、可批量 rollout 的任务。

推荐做法是混合四类任务：

| 来源 | 第一阶段数量 | 作用 |
|---|---:|---|
| CUA-Gym 任务 | 先验证 3-5 个，随后扩展 | 最接近 RLVR / verifiable reward 的 computer-use 任务池 |
| MiniWoB++ / BrowserGym 任务 | 先验证 5-10 个，随后扩展 | 快速获得现成可执行 browser RL 任务，减少自造偏差 |
| WebArena-Infinity 轨迹/环境 | 先转 100-500 条 SFT，再验证 1 个 app | 提供成功轨迹和自动生成可验证 web app 的参考 |
| 自建 Playwright smoke tasks | 20 个 | 完全可控，用于调试 action schema、recorder、verifier 和 reward |
| 现有 OS-Atlas/v2 转换的 direct-click demo | 500-1000 demo | 只用于 SFT warmup，不作为 RL task |

自建任务的优势不是“数据量”，而是 verifier 质量和调试可控性。真正 RL 最怕 reward 不可信。宁可先做 20 个 verifier 很强的 smoke tasks，把接口和记录格式跑通，再从 CUA-Gym/BrowserGym/WebArena-Infinity 扩展。

## 4. Task Dataset 应该长什么样

任务定义集不是截图数据集，而是环境任务 spec：

```json
{
  "task_id": "todo_add_item_0007",
  "env_type": "playwright_browser",
  "app": "todo",
  "template": "add_todo_item",
  "seed": 7,
  "reset": {
    "url": "http://127.0.0.1:8210/apps/todo?seed=7",
    "storage_state": null,
    "viewport": [1280, 720]
  },
  "goal": "新建一个待办事项：buy milk",
  "max_steps": 6,
  "action_space": ["click", "type", "press", "wait", "finish"],
  "verifier": {
    "type": "dom_state",
    "success": "todo_items contains 'buy milk'",
    "progress": ["input_focused", "input_value_match", "item_added"]
  },
  "split": "train",
  "difficulty": 1
}
```

每个任务必须有：

- `reset`：如何回到初始状态。
- `goal`：给模型看的自然语言任务。
- `max_steps`：控制 horizon。
- `action_space`：允许哪些动作。
- `verifier`：环境侧如何判定成功和中间进度。
- `split`：按模板、应用和 seed 做切分，避免训练/验证泄露。

## 5. Rollout 数据应该长什么样

RL 过程中在线生成 trajectory：

```json
{
  "rollout_id": "r_step0120_task_todo_add_item_0007_k03",
  "policy_version": "sft_step_120",
  "task_id": "todo_add_item_0007",
  "goal": "新建一个待办事项：buy milk",
  "trajectory": [
    {
      "t": 0,
      "screenshot": "obs_000.png",
      "prompt": "...",
      "model_output": "{\"action\":\"click\",\"x\":274,\"y\":188}",
      "action": {"action": "click", "x": 274, "y": 188},
      "exec_status": "ok",
      "reward_step": 0.1,
      "info": {"valid_json": true, "valid_action": true}
    },
    {
      "t": 1,
      "screenshot": "obs_001.png",
      "model_output": "{\"action\":\"type\",\"text\":\"buy milk\"}",
      "action": {"action": "type", "text": "buy milk"},
      "exec_status": "ok",
      "reward_step": 0.3
    }
  ],
  "success": true,
  "final_reward": 1.0,
  "total_reward": 1.35,
  "num_steps": 4
}
```

这才是 on-policy RL 的训练数据。它不是提前固定的数据集，而是随着 `policy_version` 不断产生。

## 6. Demo / SFT 数据应该长什么样

SFT 只做 warmup，目标是让模型先会输出合法 action JSON 和基本 GUI 操作。

```json
{
  "task_id": "todo_add_item_0007",
  "messages": [
    {
      "role": "user",
      "content": "<image>\n任务：新建一个待办事项：buy milk\n请输出下一步 GUI action JSON。"
    },
    {
      "role": "assistant",
      "content": "{\"action\":\"click\",\"x\":274,\"y\":188}"
    }
  ],
  "images": ["obs_000.png"]
}
```

来源：

1. Playwright scripted oracle 自动生成。
2. MiniWoB++/BrowserGym 可执行任务的 expert/heuristic 轨迹。
3. 当前 v2 strict 数据转换为 direct `click(x,y)` 单步 warmup。
4. 少量强模型 teacher 轨迹，但必须由本地 verifier 复核成功。

## 7. 算法主线

### 7.1 POMDP 建模

GUI agent 是部分可观测 MDP：

```text
s_t = (o_t, h_t, g)
```

其中：

- `o_t`：当前截图。
- `h_t`：历史 action 和 observation。
- `g`：自然语言任务目标。

策略：

```text
a_t ~ π_θ(a_t | o_t, h_t, g)
```

轨迹：

```text
τ = (s_0, a_0, r_0, s_1, ..., s_T)
```

目标：

```text
max_θ E_{τ~π_θ}[R(τ)] - β KL(π_θ || π_ref)
```

### 7.2 Direct Action JSON

主动作空间：

```json
{"action": "click", "x": 512, "y": 384}
{"action": "double_click", "x": 512, "y": 384}
{"action": "type", "text": "Alice"}
{"action": "press", "key": "ENTER"}
{"action": "hotkey", "keys": ["CTRL", "L"]}
{"action": "scroll", "dy": -500}
{"action": "drag", "x1": 100, "y1": 300, "x2": 600, "y2": 300}
{"action": "wait"}
{"action": "finish"}
```

原则：

- 不以 `select_candidate` 为主线。
- 不把 DOM 暴露给模型，DOM 只用于 verifier。
- 允许未来加工具观察动作，例如 `ocr(region)`、`crop(region)`，但第一阶段先把鼠标键盘闭环跑通。

## 8. 数学改进方向

### 8.1 Verifier-Guided Potential GRPO

终局成功 reward 很稀疏，使用 verifier 构造 potential shaping：

```text
r'_t = r_t + γ Φ(s_{t+1}) - Φ(s_t)
```

`Φ(s)` 是任务进度函数，例如：

- 输入框是否聚焦。
- 输入值是否正确。
- 是否打开目标菜单。
- 是否到达目标页面。
- 是否完成提交。

总回报：

```text
R_i = Σ_t γ^t r'_{i,t}
```

对同一任务采样 `K` 条 rollout，组内归一化 advantage：

```text
A_i = (R_i - mean(R_1,...,R_K)) / (std(R_1,...,R_K) + ε)
```

GRPO clipped objective：

```text
L_GRPO =
E_i Σ_t min(
  ρ_{i,t} A_i,
  clip(ρ_{i,t}, 1-ε, 1+ε) A_i
) - β KL(π_θ || π_ref)
```

其中：

```text
ρ_{i,t} = exp(log π_θ(a_{i,t}|s_{i,t}) - log π_old(a_{i,t}|s_{i,t}))
```

意义：保留 GRPO 不需要 value model 的优点，同时用 verifier 进度缓解 sparse reward。

### 8.2 Step-Credit GRPO

只用整条轨迹 reward 会导致 credit assignment 粗糙。对每一步构造 reward-to-go：

```text
G_{i,t} = Σ_{k=t}^{T} γ^{k-t} r'_{i,k}
```

再在同一任务、同一 step bucket 中归一化：

```text
A_{i,t} =
(G_{i,t} - mean_j G_{j,t}) / (std_j G_{j,t} + ε)
```

训练目标改为 step-level advantage：

```text
L_step =
E_i Σ_t min(
  ρ_{i,t} A_{i,t},
  clip(ρ_{i,t}, 1-ε, 1+ε) A_{i,t}
)
```

意义：让“错误的第一步点击”和“最后一步忘记 finish”获得不同 credit，而不是整条轨迹一起背锅。

### 8.3 Typed Action Validity Constraint

GUI action 有强 schema 约束。定义非法动作成本：

```text
C(τ) =
λ_json C_json
+ λ_schema C_schema
+ λ_range C_range
+ λ_exec C_exec
+ λ_loop C_loop
```

约束优化：

```text
max_θ E[R(τ)] - β KL(π_θ || π_ref)
s.t. E[C(τ)] <= c
```

用拉格朗日形式：

```text
L = L_GRPO - η max(0, E[C(τ)] - c)
```

意义：防止模型通过乱输出、越界点击、重复点击等方式制造虚假探索。

### 8.4 Coordinate-Aware Reward Kernel

对于 click 类动作，除了最终 success，还可以给连续坐标奖励：

```text
r_click =
exp(- ||p - p*||² / (2σ²))
```

其中 `p` 是模型点击点，`p*` 是 verifier 或 oracle 的目标点。若存在目标区域 `B*`：

```text
r_point = 1[p ∈ B*]
```

组合：

```text
r_ground = α r_point + (1-α) r_click
```

意义：比 binary hit 更平滑，尤其适合早期训练。

### 8.5 Self-Evolving Curriculum

对每个任务维护最近成功率 `p_i` 和平均回报方差 `v_i`。采样权重：

```text
w_i ∝ exp(- (p_i - p_target)^2 / τ_p) · (v_i + ε)^α · d_i^β
```

其中：

- `p_target` 可设为 0.3-0.7，让训练集中在可学习边界。
- `v_i` 高说明策略还不稳定。
- `d_i` 是任务难度。

意义：避免一直采太简单或完全不可学的任务。

## 9. 训练阶段规划

### Phase 0：冻结 candidate 主线

目标：

- 不再继续大规模补框。
- 保留 strict v2 数据作为 warmup。
- 把工程主线切到 direct-action agent。

产物：

- AGENTS.md 更新。
- 本规划文档。

### Phase 1：实现本地 Playwright GUI-RL Gym

新增模块建议：

```text
envs/browser_rl/
├── task_spec.py
├── playwright_env.py
├── actions.py
├── verifier.py
├── recorder.py
├── apps/
│   ├── todo/
│   ├── forms/
│   ├── tables/
│   ├── menus/
│   └── shopping/
└── tasks/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

验收：

- `reset` 后截图稳定。
- `step(click/type/press/scroll)` 可执行。
- 每条任务有 success verifier。
- scripted oracle 成功率 >= 95%。

### Phase 2：构建 100-200 个高可信 verifier 任务

优先模板：

| 模板 | 示例 | verifier |
|---|---|---|
| form fill | 填用户名/密码/邮箱并提交 | DOM input value + submit state |
| search/select | 搜索商品并点击目标结果 | URL/state/selected item |
| todo | 新增、编辑、删除、勾选事项 | todo state |
| table | 找到某行某列并点击按钮 | DOM row id / action log |
| menu/modal | 打开菜单、选择项、关闭弹窗 | menu state / modal state |
| pagination/filter | 筛选列表、翻页找目标 | filter state / target visible |
| drag/slider | 调整滑块或拖放项目 | numeric state |

每个模板随机化：

- 文案。
- 布局。
- 按钮位置。
- 干扰项。
- 颜色/主题。
- viewport。
- 目标难度。

### Phase 3：Trajectory SFT warmup

数据来源：

1. Playwright scripted oracle 轨迹。
2. 当前 strict v2 转 direct-click 单步轨迹。
3. 少量 BrowserGym/MiniWoB demo。

训练目标：

- 学会 action JSON。
- 学会多轮历史格式。
- 学会 `finish`。
- 降低无效动作率。

不以 SFT 成败作为最终结论，只用它初始化 RL。

### Phase 4：On-policy rollout

每轮：

```text
sample task batch
for each task sample K rollouts with current policy
execute actions in Playwright
record screenshots/actions/rewards/logprobs
compute group advantages
update LoRA policy
evaluate held-out tasks
```

建议起步参数：

- base model：`Qwen2.5-VL-3B-Instruct`
- LoRA rank：16 或 32
- `K=4`
- max steps：4-6
- train tasks：100-200
- val tasks：30-50
- 每次 update 前 rollout：128-256 trajectories
- KL reference：SFT checkpoint

### Phase 5：外部环境泛化

顺序：

1. MiniWoB++ / BrowserGym。
2. VisualWebArena 小子集。
3. WebArena/WebArena-Lite。
4. AndroidWorld 或 OSWorld。

注意：外部环境先做 eval，再决定是否训练。不要一开始让重环境拖慢算法迭代。

## 10. 评测指标

主指标：

- `success_rate`：任务成功率。
- `avg_return`：平均总 reward。
- `avg_steps_success`：成功任务平均步数。
- `valid_action_rate`：合法 action 比例。
- `json_parse_rate`：JSON 可解析比例。
- `exec_error_rate`：执行失败比例。
- `finish_correct_rate`：正确终止比例。

泛化指标：

- held-out seed success。
- held-out template success。
- held-out app success。
- 不同 viewport success。
- 文案扰动 success。

安全/反作弊指标：

- verifier 是否只依赖环境内部状态，不暴露给模型。
- 是否存在靠重复点击刷 reward。
- 是否存在不完成任务也 `finish` 的高分。
- 是否存在越界/无效 action 获益。

## 11. 与当前候选框工作的关系

当前候选框工作保留价值：

1. 转为 direct-click SFT warmup。
2. 作为单步 grounding auxiliary task。
3. 作为 debug 可视化：判断模型点击点是否接近 UI 元素。
4. 作为 ablation：candidate action vs direct action。

但不再继续把主资源投入：

- 大规模补框。
- full-candidate score cache。
- candidate-only GRPO。
- 针对单一 data_type 堆规则。

## 12. 近期执行清单

第一周目标：

1. 新增 `envs/browser_rl/` 统一接口，覆盖 `reset/step/screenshot/verifier/recorder`，并采用 direct GUI action JSON。
2. 先接 MiniWoB++ 或 BrowserGym，跑通 5-10 个最小 browser RL 任务，确认外部 Gym 任务能映射到本项目 rollout schema。
3. 下载并检查 `xlangai/CUA-Gym`，在隔离目录中验证 3-5 个 task bundle，确认 `task.json`、setup 配置和 `reward.py` 的执行入口、依赖和安全约束。
4. 接 WebArena-Infinity：先把 Hugging Face trajectories 转成 SFT/warmup 样本，再尝试本地启动 1 个 app 或记录无法启动的依赖阻塞。
5. 做 20 个自建 Playwright smoke tasks，只用于控制变量和调试 action parser、screenshot、verifier、recorder，不再先手写大规模任务集。

第二周目标：

1. 基于第一周验证结果选择主训练任务源，优先级为 CUA-Gym > BrowserGym/MiniWoB++ > WebArena-Infinity > 自建 smoke tasks。
2. 扩展可执行任务到 100-200 个，并统一 action/observation/reward schema。
3. 实现 Verifier-Guided GRPO 最小版本。
4. 跑 `K=4`、max_steps=4 的小规模 on-policy RL。
5. 输出 held-out success rate、valid action rate、reward hacking 检查报告。

第三周目标：

1. 加 step-credit advantage。
2. 加 typed action validity constraint。
3. 接 BrowserGym/MiniWoB++。
4. 做 direct-action RL vs SFT-only vs candidate-warmup 的对照。

## 13. 风险与约束

1. VLM rollout 成本高。
   - 先用短 horizon、小任务、低分辨率截图。

2. Reward hacking。
   - verifier 只读环境状态。
   - 进度 reward 必须和真实任务完成相关。

3. JSON/action 不稳定。
   - SFT warmup + schema parser + invalid action penalty。

4. 长轨迹 credit assignment 难。
   - 先做 4-6 步任务。
   - 再逐步 curriculum 到 8-12 步。

5. 浏览器任务和桌面任务有 domain gap。
   - 第一阶段目标是跑通 on-policy VLM agent RL 闭环。
   - 泛化到 OSWorld/桌面是第二阶段，不应混成同一个问题。

## 14. 最终目标

最终模型不应只是“从候选框里选答案”，而应能在 GUI 环境里执行：

```text
理解任务 -> 看截图 -> 规划下一步 -> 点击/输入/滚动 -> 根据新截图修正 -> 完成任务
```

训练目标也不再是 oracle IoU，而是执行成功率、动作合法性、效率和跨任务泛化能力。
