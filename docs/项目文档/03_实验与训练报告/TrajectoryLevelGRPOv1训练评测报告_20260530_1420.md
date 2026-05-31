# Trajectory-level GRPO v1 训练评测报告

主报告位置：

```text
/root/Workspace/VLM/项目文档/03_实验与训练报告/TrajectoryLevelGRPOv1训练评测报告_20260530_1420.md
```

核心结论：训练链路跑通；`val_balanced_70=0.9130` 低于 289tg 的 `0.9420`，补跑 `val200=0.9000` 也低于 289tg 的 `0.9100`，因此本次 v1 adapter 不升级为主线 current model。

关键产物：

- 训练脚本：`scripts/train_browser_rl_trajectory_grpo.py`
- v1 adapter：`outputs/trajectory_level_grpo_v1_50x8_20260530_1323/adapter`
- val 输出：`outputs/rollouts_local_qwen25vl_trajectory_grpo_v1_50x8_val_balanced70_20260530_1358`
- val200 输出：`outputs/rollouts_local_qwen25vl_trajectory_grpo_v1_50x8_val200_20260530_1427/merged`
- 当前继续保留的主线 adapter：`/root/models/adapters/browser_rl/qwen25vl_3b_browserrl_289tg_table_advanced_repair_adapter_backup_20260530_1313`
