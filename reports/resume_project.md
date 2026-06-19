# AlgoAce Resume Draft

**AlgoAce：基于可执行反馈的算法题代码生成系统**

- 基于 TACO-verified 构造并验证 Python 代码训练数据，使用 QLoRA 对代码模型进行监督微调。
- 设计可执行 reward，以语法、可见测试、奖励测试、超时和运行时错误作为可验证训练信号。
- 实现上下文受控修复流程，在不泄漏留出测试的情况下，根据失败类型和可见反例迭代修复代码。
- 使用 pass@1、pass@k、verified success rate 和 repair gain 对 Base、SFT、RL 与修复流程进行消融评测。

