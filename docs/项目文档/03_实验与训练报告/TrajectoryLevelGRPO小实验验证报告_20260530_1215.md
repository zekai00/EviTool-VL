# Trajectory-level GRPO 小实验验证报告 2026-05-30 12:15 CST

同名报告已同步到根项目文档：

```text
/root/Workspace/VLM/项目文档/03_实验与训练报告/TrajectoryLevelGRPO小实验验证报告_20260530_1215.md
```

核心结果：

- 50 task groups
- 400 full trajectories
- 42 trainable groups
- trainable group rate = 0.8400
- all-zero reward groups = 0
- rollout success rate = 0.8725

结论：当前 289tg adapter 已经足够强，trajectory-level GRPO 小规模训练有实际信号，不是“为了做而做”。

