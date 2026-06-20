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
- Oracle 候选改为首个全测试通过后立即停止，避免继续执行无用候选；编译阶段静默处理 `SyntaxWarning`，并支持中断时取消尚未开始的并行任务。
- 正式数据审计发现重复题面和百万字符级异常样本；划分改为按规范化题面与调用签名去重，优先保留测试更完整的代表题，SFT 在 tokenizer 前过滤极端字符长度样本。
- SFT preflight 默认剔除会触发 `SyntaxWarning` 的目标代码并报告数量，兼容旧 verifier 已生成的数据，无需重跑全量 oracle。

## 2026-06-20：首轮 SFT Dev100 退化分析

- Base verified success rate 为 22%，SFT 为 12%；visible/reward/eval pass rate 均下降。
- 两者 syntax error rate 均为 1%、runtime error rate 均为 11%，说明退化主要来自算法语义而非输出格式。
- 暂停 GRPO，新增逐题报告对比，统计 Base-only/SFT-only 转移、IO 模式差异、失败原因和代码长度，避免直接在退化 checkpoint 上继续强化学习。
- 配对结果为 Base-only 11 题、SFT-only 1 题；SFT 输出代码中位长度从 502.5 降至 235，stdin 与 callable 均退化。
- 下一轮采用 1 epoch、`5e-5` 学习率、3% warmup，并保留多个 checkpoint 评测学习曲线，控制变量验证是否为更新过强。
- SFT-v2 checkpoint-500 verified success rate 为 10%，最终模型为 12%，仍显著低于 Base 的 22%。
- 降低学习率和训练轮数没有恢复正确率，因此排除“仅由更新过强导致”的解释；更可能是单参考代码的 token imitation 诱导短代码捷径，与执行正确率目标不一致。
- SFT 作为负向消融保留，不再继续调参。后续从 Base 出发验证 best-of-N、上下文修复与执行奖励优化。

## 2026-06-21：Base best-of-4 初始结果与候选重排

- temperature 0.2 的 best-of-4 将 verified success rate 从 22% 提升至 26%，visible pass@4 为 31%。
- pass@4 仅比 pass@1 visible success 27% 高 4 个百分点，说明候选高度相关，低温采样多样性不足。
- 新增 `top_p` 采样参数与 `reward_rerank` 消融：reward tests 只用于内部候选排序，不向模型泄漏测试内容；eval tests 仍只运行最终候选一次。

## 2026-06-21：执行奖励驱动的后训练主线

- SFT 负向消融后，将核心训练方法改为从 Base 直接进行 GRPO，不沿用退化 adapter。
- Reward 重新归一到 1.0：语法 0.05、可见测试 0.15、隐藏 reward tests 0.60、全部通过奖励 0.20；超时、运行错误、无输入硬编码和超长输出单独惩罚。
- 删除“答案字符串出现在代码中即硬编码”的误判规则，避免惩罚合法的 `print("YES")` 等程序。
- GRPO prompt 改为与推理一致的 system/user chat 模板，随机抽取且要求至少 5 个 reward tests。
- 200 题 smoke 使用 4 个 generation、25 个更新步、`1e-6` 学习率和 `beta=0.04` KL 约束；reward tests 4 路并行执行。
- 新增 reward 审计与训练日志汇总，重点监控 oracle/bad 排序、零方差 group、完整通过率、超时和硬编码惩罚，防止 reward hacking。
