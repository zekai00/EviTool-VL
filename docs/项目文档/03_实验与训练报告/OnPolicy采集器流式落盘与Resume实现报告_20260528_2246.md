# OnPolicy 采集器流式落盘与 Resume 实现报告

- 生成时间：2026-05-28 22:46 CST
- 目标：修复 on-policy collect 长时间运行时无法实时看到 trainable group 数量、被中断后丢失 partial groups 的工程问题。
- 结论：已实现边采边写、实时 summary、resume 跳过已完成 task、恢复后目标已满足则不加载模型直接退出。

## 术语说明

- `采集器`：`scripts/train_browser_rl_onpolicy_grpo.py` 中的 on-policy collect 部分，负责让当前模型在浏览器环境中试动作，并把 verifier reward 整理成 GRPO 可训练的 group。
- `stream collect`：边采边写。每产生一个 group 或 rollout，就追加写入临时 JSONL 文件。
- `resume`：恢复采集。从已有临时文件读取已采数据，继续未完成的采集目标。
- `live_summary`：实时统计文件，记录当前 groups、trainable_groups、family/template 分布等。

## 代码变更

修改文件：

- `scripts/train_browser_rl_onpolicy_grpo.py`

新增参数：

```bash
--stream-collect / --no-stream-collect
--stream-flush-every 5
--resume-collect
```

新增输出：

```text
groups.jsonl.tmp
rollouts.jsonl.tmp
live_summary.json
```

行为：

1. 默认开启 `--stream-collect`。
2. 每采到一个 group，立即 append 到 `groups.jsonl.tmp`。
3. 每完成一个 rollout，立即 append 到 `rollouts.jsonl.tmp`。
4. 每 `--stream-flush-every` 个 group 更新一次 `live_summary.json`，每个 rollout 后也更新一次。
5. 正常结束时仍会写最终 `groups.jsonl`、`rollouts.jsonl`、`collect_summary.json`、`summary.json`。
6. 使用 `--resume-collect` 时，会优先读取 `groups.jsonl.tmp` 和 `rollouts.jsonl.tmp`，没有临时文件时回退读取最终文件。
7. resume 后会跳过已有 group/rollout 覆盖到的 task，避免重复采同一 task。
8. 如果 resume 后目标已经满足，脚本直接输出 summary，不再加载本地 Qwen 模型。

## 验证

语法检查：

```bash
/opt/conda/envs/llama/bin/python3 -m py_compile scripts/train_browser_rl_onpolicy_grpo.py
```

真实小采集：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  --tasks outputs/browser_rl_task_suite_2000_20260528_1344/train_tasks.jsonl \
  --output-dir outputs/onpolicy_collect_stream_smoke_20260528_check \
  --model /root/models/Qwen2.5-VL-3B-Instruct \
  --adapter outputs/onpolicy_browser_rl_grpo_replay_213tg_safe_20260528_2041/adapter \
  --include-templates table_action \
  --template-quotas-json '{"table_action":1}' \
  --target-trainable-groups 1 \
  --limit 20 \
  --max-groups 4 \
  --max-steps 2 \
  --num-generations 2 \
  --collect-only \
  --stream-collect \
  --stream-flush-every 1
```

结果：

| 文件 | 行数 |
| --- | ---: |
| groups.jsonl.tmp | 1 |
| rollouts.jsonl.tmp | 1 |
| groups.jsonl | 1 |
| rollouts.jsonl | 1 |

`live_summary.json` 正常记录：

- groups=1
- trainable_groups=1
- groups_tmp 路径
- rollouts_tmp 路径
- completed=true

resume 验证：

```bash
同一命令追加 --resume-collect
```

结果：

- resumed_groups=1
- resumed_rollouts=1
- 目标已满足，直接退出。
- 没有重新加载本地 Qwen 模型。
- 没有重复追加 group/rollout。

## 后续用法

长采集建议命令模式：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  ... \
  --collect-only \
  --stream-collect \
  --stream-flush-every 5
```

中断后恢复：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/llama/bin/python3 scripts/train_browser_rl_onpolicy_grpo.py \
  ... \
  --collect-only \
  --stream-collect \
  --resume-collect
```

实时查看：

```bash
cat outputs/某次采集/live_summary.json
wc -l outputs/某次采集/groups.jsonl.tmp
wc -l outputs/某次采集/rollouts.jsonl.tmp
```

## 仍需注意

1. 当前 resume 粒度是 task 级：已有 group 或 rollout 的 task 会被跳过，避免重复采样。
2. 如果未来希望精确恢复到同一 task 的同一步，需要进一步记录 task cursor、prefix_actions 和 RNG 状态。
3. 当前实现已经足够解决长采集被中断后无法拿到 partial groups 的问题。
