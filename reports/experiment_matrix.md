# AlgoAce Experiment Matrix

所有实验固定：

- Test split：`taco_split/test`
- Seed：`42`
- Base model：`Qwen/Qwen2.5-Coder-7B-Instruct`
- 每组使用独立报告文件，禁止复用不匹配的 partial 文件。

| ID | Model | Adapter | Candidates | Repair turns | Temperature |
|---|---|---|---:|---:|---:|
| E1 | Base | none | 1 | 0 | 0.0 |
| E2 | Base | none | 4 | 0 | 0.2 |
| E3 | Base | none | 1 | 3 | 0.0 |
| E4 | Base | none | 4 | 3 | 0.2 |
| E5 | Base | SFT | 1 | 0 | 0.0 |
| E6 | Base | SFT | 4 | 3 | 0.2 |
| E7 | Base | SFT + GRPO | 1 | 0 | 0.0 |
| E8 | Base | SFT + GRPO | 4 | 3 | 0.2 |

## 已验证阶段结论

- Base Dev100 pass@1 verified success rate：22%。
- SFT-v1 Dev100：12%。
- SFT-v2 checkpoint-500：10%；SFT-v2 final：12%。
- SFT 未进入后续主线 checkpoint；保留为“监督模仿目标与执行正确率不一致”的负向消融。

主结果优先报告 `verified_success_rate`，并同时报告 `repair_gain`、运行错误率、超时率和平均尝试轮数。
