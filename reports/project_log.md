# AlgoAce Project Log

## 2026-06-19：项目重新定位

- 新项目只优化算法题 Python 代码正确率。
- 模型输出仅包含一个 Python 代码块。
- 移除中文解释、复杂度说明和用户追问训练目标。
- 核心技术路线：Code SFT、可执行 reward、best-of-N、上下文受控修复。
- 核心指标：pass@1、pass@k、verified success rate、repair gain。

## 2026-06-19：正式实验隔离与断点续跑

- 增加固定 seed 的 train/dev/test 题目级划分与 split 指纹。
- test split 不参与 SFT、GRPO、prompt 修改或阈值选择。
- visible tests 可反馈具体反例；reward tests 仅反馈抽象失败；eval tests 最终只执行一次。
- 转换、oracle 验证、SFT/GRPO 数据生成和评测支持断点续跑。
- 评测按 problem id 独立设随机种子，保证中断恢复不改变剩余样本的生成结果。
