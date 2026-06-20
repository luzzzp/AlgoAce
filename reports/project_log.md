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

## 2026-06-20：训练数据与 Benchmark 资格分离

- 100 题中有 34 题原始测试少于 3 个，无法同时构造 visible/reward/eval，但 oracle 均已验证。
- 这些题不进入 dev/test benchmark，但追加到 train 用于 Code SFT，避免浪费正确代码数据。
- 只有具备完整三组测试的题参与严格 dev/test 划分；GRPO prompts 自动跳过缺少 reward tests 的训练题。

## 2026-06-20：Code SFT 训练目标修正

- SFT loss 只计算 assistant 输出的 Python 代码，不再训练模型复述 system prompt 和题面。
- 长题面采用 token 级首尾保留策略，优先完整保留目标代码，避免右侧截断损坏训练答案。
- 训练前新增数据审计，检查 verified oracle、代码块格式、Python 语法、重复样本、测试组完整性和数据指纹。
- Oracle 验证新增题目级并行 worker，默认并行验证 8 道题，并保持逐题落盘和断点续跑。
