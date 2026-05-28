# OnPolicy-VLM-Agentic-RL 弱隔离 CUA-Gym Sandbox 跑通报告

- 生成时间：2026-05-27 21:51 CST
- 工作区：`/root/Workspace/VLM`
- 仓库：`/root/Workspace/VLM/EviTool-VL`
- 输出目录：`outputs/cua_gym_mock_web_sandbox_20260527_2151`

## 1. 本轮结论

在没有宿主机 Docker 权限的条件下，已经跑通 CUA-Gym mock_web 的第一版弱隔离执行链路。

这里的 `Sandbox` 指隔离沙箱。本轮不是新开宿主机容器，而是在当前 Docker 容器内部做弱隔离：每个任务复制到独立工作目录，`initial_setup.py` 和 `reward.py` 用子进程执行，只传白名单环境变量，不传 `.env` 和 API key，并设置超时和资源限制。

本轮跑通的是 CUA-Gym 的 setup/reward 动态闭环，还不是完整的可交互网页前端环境。也就是说，现在已经能让 CUA-Gym 任务完成“初始化状态 -> reward 脚本读取状态 -> 输出分数”，但还没有真实 mock app 前端给 VLM 点击操作。

## 2. 关键名词

- `CUA-Gym`：computer-use agent 任务集合，里面包含网页、桌面和多应用任务。本项目先用其中的 mock_web 子集。
- `mock_web`：模拟网页任务，状态由脚本写入本地服务，reward 脚本再读取状态判断是否完成任务。
- `initial_setup.py`：任务初始化脚本，负责创建一个 sid，并把任务初始状态写入 mock app。
- `reward.py`：奖励脚本，读取当前状态并输出 `REWARD: x`，其中 x 是 0 到 1 之间的任务完成分数。
- `State server`：本轮新增的本地状态服务，用来接住 CUA-Gym 脚本里的 `/post?sid=...` 和 `/go?sid=...` 请求。
- `Weak isolation`：弱隔离，指当前容器内的子进程隔离，不等价于单独 Docker、gVisor 或虚拟机。

## 3. 本轮新增文件

1. `envs/browser_rl/cua_gym_sandbox.py`
   - 实现本地状态服务。
   - 实现 CUA-Gym task bundle 复制、脚本 patch、子进程执行、reward 解析和结果落盘。
   - 支持单 app 和 multi-app placeholder，例如 `__CUA_GYM_HUBSPOT_URL__` 会映射成本地 `/hubspot` 状态命名空间。

2. `scripts/run_cua_gym_mock_web_sandbox.py`
   - 命令行入口。
   - 从 `mock_web_tasks.jsonl` 读取任务，从已解包 bundle 目录执行样本。

3. `requirements-browser-rl.txt`
   - 增加 `requests`，因为 CUA-Gym 的 setup/reward 脚本依赖它访问本地状态服务。

## 4. 隔离策略

当前实现的隔离措施：

1. 独立工作目录。
   - 每个 task bundle 会复制到 `outputs/cua_gym_mock_web_sandbox_*/<task_id>/work`。

2. 子进程执行。
   - 不在主进程 import 外部 `reward.py`，避免它直接污染主进程状态。

3. 白名单环境变量。
   - 子进程只收到 `PATH`、`HOME`、`TMPDIR`、`NO_PROXY` 等必要变量。
   - 不传 `DASHSCOPE_API_KEY`、`.env`、HF token 或其他私有变量。

4. 临时文件路径重写。
   - CUA-Gym 脚本里常见的 `/tmp/task_web_sid` 被重写到任务工作目录下的 `work/tmp/task_web_sid`。

5. 资源限制。
   - CPU 时间、虚拟内存、文件大小和打开文件数都有基础限制。

6. fake browser。
   - 因为服务器没有图形界面，本轮用 fake `google-chrome` 截获 CUA-Gym 初始化脚本的浏览器启动命令，只记录 URL，不真的打开图形浏览器。

## 5. 验证命令

```bash
python3 -m py_compile \
  envs/browser_rl/cua_gym_sandbox.py \
  envs/browser_rl/__init__.py \
  scripts/run_cua_gym_mock_web_sandbox.py
```

结果：通过。

```bash
python3 scripts/run_cua_gym_mock_web_sandbox.py \
  --output-dir outputs/cua_gym_mock_web_sandbox_20260527_2145 \
  --limit 2 \
  --timeout-sec 45
```

结果：setup 2/2，reward 2/2。

```bash
python3 scripts/run_cua_gym_mock_web_sandbox.py \
  --output-dir outputs/cua_gym_mock_web_sandbox_20260527_2151 \
  --limit 10 \
  --timeout-sec 60
```

结果：

| 指标 | 数值 |
|---|---:|
| runs | 10 |
| setup_ok | 10 |
| reward_ok | 10 |
| avg_initial_reward | 0.0 |

`avg_initial_reward` 为 0.0 是预期结果，因为当前只执行初始化和 reward，没有 agent 去完成任务。

## 6. 10 个样本结果

| task_id | app_type | setup | reward | 初始分数 | 状态命名空间数 |
|---|---|---:|---:|---:|---:|
| `0018392e-cfed-5e2f-8278-a4580d64a00a` | instagram_mock | true | true | 0.0 | 1 |
| `003ee5e8-5ff5-551f-95fc-1758879fd4c0` | trello_mock | true | true | 0.0 | 1 |
| `0077a580-d847-507b-a7e1-8cf2b024a23e` | slack_mock,notion_mock,asana_mock,gmail_mock | true | true | 0.0 | 8 |
| `00ca6afd-f5ec-543e-bf5f-694a62bdbb55` | outlook_web_mock | true | true | 0.0 | 1 |
| `011983c3-acd2-52e2-afaa-3226004e9a1e` | gmail_mock,salesforce_mock,notion_mock,slack_mock,linkedin_mock | true | true | 0.0 | 5 |
| `01385c1b-4bb8-5e41-a699-d786069b04b5` | jira_mock,slack_mock,notion_mock | true | true | 0.0 | 3 |
| `0141cfc7-752b-5d58-9ac5-2376ab528b76` | instacart_mock | true | true | 0.0 | 1 |
| `014ff5a6-88a8-53f0-a599-1efefe0c8d61` | hubspot_mock | true | true | 0.0 | 1 |
| `0174cbda-3c7b-515d-b7b2-c6069009b2fc` | reddit_mock | true | true | 0.0 | 2 |
| `01b69b03-034a-55f3-91e2-8346b36e1c33` | instacart_mock | true | true | 0.0 | 1 |

## 7. 当前还没有完成什么

1. 还没有真实 mock web 前端。
   - 本轮的 state server 只提供状态 API 和 debug HTML，不是 CUA-Gym 原始前端。
   - 因此 VLM 还不能在这些任务上通过截图点击真实控件。

2. 还没有 step action 改写 state。
   - 现在只验证初始状态和 reward。
   - 下一步需要要么接真实 mock app 前端，要么先做 state-level action bridge。

3. 还不是 on-policy RL。
   - 当前没有让 Qwen 模型产生动作，也没有用当前模型 rollout 做训练。
   - 这是 on-policy RL 前的环境验证步骤。

## 8. 下一步建议

优先做两条并行线：

1. 接 Qwen policy wrapper 到本地 Playwright smoke tasks 和 BrowserGym/MiniWoB++。
   - 这条线最快能进入真正 rollout。
   - 目标是让 Qwen 看截图和任务目标，输出合法 GUI action JSON。

2. 继续补 CUA-Gym mock_web 前端或 state-level bridge。
   - 如果能拿到 mock app 前端代码，就接真实网页 step。
   - 如果短期拿不到前端，就先做 state-level bridge，只把 CUA-Gym 用作 reward/verifier 任务池，而不是完整 GUI 环境。

当前优先级建议：先接 Qwen policy wrapper，因为可执行 GUI 环境已经有本地 smoke tasks 和 BrowserGym/MiniWoB++，可以更快进入真正 on-policy rollout。
