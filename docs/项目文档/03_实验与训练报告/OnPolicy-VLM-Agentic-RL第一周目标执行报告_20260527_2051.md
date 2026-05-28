# OnPolicy VLM Agentic RL 第一周目标执行报告

- 生成时间：2026-05-27 20:51 CST
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 主线：direct-action、可执行、可验证的 on-policy VLM agentic RL
- 结论：第一周目标的本地闭环已经跑通；CUA-Gym 和 BrowserGym/MiniWoB++ 已确认可接入；WebArena-Infinity 已实现转换入口，但本机访问 HF repo tree 超时，暂未完成本地样本转换和 app 启动。

## 1. 规划文档更新

已更新：

- `/root/Workspace/VLM/项目文档/01_规划与路线/真正OnPolicy-VLM-Agentic-RL浏览器环境与算法规划_20260527_2008.md`
- `docs/项目文档/01_规划与路线/真正OnPolicy-VLM-Agentic-RL浏览器环境与算法规划_20260527_2008.md`

第一周目标已从“先自建大批 Playwright 任务 + SFT smoke”修正为：

1. 新增 `envs/browser_rl/` 统一接口。
2. 先接 BrowserGym/MiniWoB++，验证 5-10 个最小 browser RL 任务。
3. 下载并检查 `xlangai/CUA-Gym`，验证 3-5 个 task bundle。
4. 接 WebArena-Infinity，先转 SFT/warmup 样本，再尝试启动 1 个 app 或记录依赖阻塞。
5. 做 20 个自建 Playwright smoke tasks，用于控制变量和调试。

## 2. 新增代码

新增 browser RL 环境：

- `envs/__init__.py`
- `envs/browser_rl/__init__.py`
- `envs/browser_rl/actions.py`
- `envs/browser_rl/task_spec.py`
- `envs/browser_rl/verifier.py`
- `envs/browser_rl/recorder.py`
- `envs/browser_rl/playwright_env.py`

新增脚本：

- `scripts/build_browser_rl_smoke_tasks.py`
- `scripts/run_browser_rl_smoke.py`
- `scripts/check_browsergym_miniwob.py`
- `scripts/inspect_cua_gym.py`
- `scripts/convert_webarena_infinity_trajectories.py`

新增依赖说明：

- `requirements-browser-rl.txt`

本机已安装运行依赖：

- `playwright`
- `browsergym-miniwob`
- `gymnasium`
- `zstandard`
- Playwright Chromium browser

## 3. 本地 Playwright Smoke Tasks

构建命令：

```bash
python3 scripts/build_browser_rl_smoke_tasks.py \
  --output-dir outputs/browser_rl_smoke_tasks_20260527_1w \
  --timestamp 20260527_1w \
  --count-per-template 4
```

执行命令：

```bash
python3 scripts/run_browser_rl_smoke.py \
  --tasks outputs/browser_rl_smoke_tasks_20260527_1w/all_tasks.jsonl \
  --output-dir outputs/browser_rl_smoke_rollouts_20260527_1w \
  --headless
```

结果：

| 指标 | 数值 |
|---|---:|
| smoke tasks | 20 |
| templates | 5 |
| rollouts | 20 |
| scripted oracle success rate | 100.00% |
| avg steps | 3.0 |
| generated SFT rows | 60 |

任务模板：

- `form_fill`
- `todo_add`
- `search_select`
- `menu_select`
- `table_action`

输出：

- `outputs/browser_rl_smoke_tasks_20260527_1w`
- `outputs/browser_rl_smoke_rollouts_20260527_1w`

说明：这 20 个任务不是主训练集，而是 action parser、Playwright execution、screenshot、verifier、recorder 的控制变量测试集。

## 4. BrowserGym / MiniWoB++

安装后 BrowserGym/MiniWoB++ 成功注册 125 个 MiniWoB task。

MiniWoB++ HTML 获取方式：

- Git clone 直连 GitHub 超时。
- 改用 `https://codeload.github.com/Farama-Foundation/miniwob-plusplus/zip/refs/heads/main` 下载 zip 成功。
- 本地路径：`/root/models/datasets/miniwob-plusplus-main/miniwob/html/miniwob/`

检查命令：

```bash
python3 scripts/check_browsergym_miniwob.py \
  --output outputs/browsergym_miniwob_check_20260527_1w.json \
  --miniwob-url file:///root/models/datasets/miniwob-plusplus-main/miniwob/html/miniwob/
```

结果：

| 指标 | 数值 |
|---|---:|
| attempted tasks | 5 |
| reset_ok | 5 |
| step_ok | 5 |
| blocked | false |

验证任务：

- `browsergym/miniwob.click-button`
- `browsergym/miniwob.click-checkboxes`
- `browsergym/miniwob.enter-text`
- `browsergym/miniwob.choose-list`
- `browsergym/miniwob.focus-text`

说明：本轮只验证 reset/step 和 observation schema，没有训练 agent，也没有要求 noop 成功，因此 reward 为 0 是正常现象。

## 5. CUA-Gym

直连 Hugging Face 在本机超时，改用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com python3 scripts/inspect_cua_gym.py \
  --output-dir outputs/cua_gym_inspection_20260527_1w_mirror \
  --limit 5 \
  --download-artifact
```

结果：

| 指标 | 数值 |
|---|---:|
| metadata rows | 7897 |
| inspected bundles | 5 |
| task_json exists | 5/5 |
| reward.py exists | 5/5 |
| reward.py py_compile ok | 5/5 |
| blocked | false |

抽样 app 类型：

- `instagram_mock`
- `vscode`
- `libreoffice_calc`
- `libreoffice_writer`
- `trello_mock`

说明：本轮没有直接执行 `reward.py`，只做静态解包和语法编译。原因是 `reward.py` 属于外部可执行代码，后续必须放到隔离容器或 VM，并配套对应 app/server/desktop 环境后再做动态执行。

## 6. WebArena-Infinity

已实现转换入口：

```bash
HF_ENDPOINT=https://hf-mirror.com HUGGINGFACE_HUB_HTTP_TIMEOUT=20 \
python3 scripts/convert_webarena_infinity_trajectories.py \
  --output-dir outputs/webarena_infinity_sft_sample_20260527_1w \
  --limit 10 \
  --timeout-sec 60
```

结果：

| 指标 | 数值 |
|---|---:|
| requested limit | 10 |
| converted | 0 |
| blocked | true |

阻塞原因：

- 本机 `datasets` streaming 在遍历 `webarena-x/webarena-infinity-trajectories` 的 Hugging Face repo tree 时反复超时。
- 当前没有本地 app bundle，因此无法启动 WebArena-Infinity app。

结论：WebArena-Infinity 数据源确认存在，转换脚本已写好，但本轮没有完成本地样本转换和 app 启动。下一步应使用离线下载、`huggingface-cli` 分片下载，或改走可访问镜像/手动同步。

## 7. 验证命令

语法检查：

```bash
python3 -m py_compile \
  envs/browser_rl/actions.py \
  envs/browser_rl/task_spec.py \
  envs/browser_rl/verifier.py \
  envs/browser_rl/recorder.py \
  envs/browser_rl/playwright_env.py \
  scripts/build_browser_rl_smoke_tasks.py \
  scripts/run_browser_rl_smoke.py \
  scripts/check_browsergym_miniwob.py \
  scripts/inspect_cua_gym.py \
  scripts/convert_webarena_infinity_trajectories.py
```

结果：通过。

## 8. 当前判断

第一周最关键的闭环已经成立：

```text
task spec -> reset -> screenshot -> direct action -> step -> verifier -> reward -> rollout record -> SFT warmup row
```

后续不需要再围绕候选框补框推进。下一步应该把 `PlaywrightBrowserEnv` 和 BrowserGym/MiniWoB++ 包装成统一 rollout source，然后优先把 CUA-Gym 的 `mock_web` 子集放进隔离环境动态执行。

## 9. 下一步

1. 新增 CUA-Gym sandbox runner，只允许在隔离目录/容器中执行 `setup.py` 和 `reward.py`。
2. 优先筛选 CUA-Gym 的 `mock_web` 子集，避免一开始引入 LibreOffice/VSCode/桌面重环境。
3. 把 BrowserGym/MiniWoB observation 转成本项目 `BrowserTaskSpec` / rollout schema。
4. 用 20 个 smoke tasks 生成的 60 条 SFT rows 做 action JSON smoke SFT。
5. 实现 on-policy rollout collector，先支持 scripted/oracle/random policy，再接 Qwen2.5-VL policy。
