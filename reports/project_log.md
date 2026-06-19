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

## 2026-06-19：TACO Callable Harness 修复

- 100 题 oracle 初次验证为 81/100，失败中 18 题来自 callable harness，只有 1 题来自 stdin 空白差异。
- callable 解法可能把入口方法定义在 `class Solution` 中，不能只查找全局函数。
- TACO callable 输出存在单元素包装，例如 `["MAS"]` 实际表示函数返回字符串 `"MAS"`。
- 标准输出比较需要忽略每行两侧空白，避免把缩进差异误判为 Wrong Answer。
- 数据集的 verified 标签依赖原始执行 harness；移植数据时应先复现 harness，再判断 oracle 质量。

## 2026-06-19：题内测试隔离修复

- 初版按 visible、reward、eval 顺序填满上限，导致测试数不足 24 的题经常没有 eval tests。
- 新策略在题目至少有 3 个测试时，强制为 visible、reward、eval 各预留至少一个且不重复。
- 剩余测试优先保证合理的 eval 规模，再将其他测试分配给 reward，避免正式划分大量丢题。
