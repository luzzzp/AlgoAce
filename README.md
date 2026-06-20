# AlgoAce

AlgoAce 是一个面向算法题的可验证 Python 代码生成系统。项目只优化代码正确率：

```text
题面 -> 多候选代码生成 -> 可见测试筛选 -> 上下文受控修复 -> 独立留出测试
```

模型输出只包含一个 `python` 代码块。核心实验比较 Base、Code SFT、GRPO、best-of-N 和上下文修复。

## 项目结构

```text
algoace/    schema、执行器、上下文构造、reward 和求解控制器
scripts/    TACO 转换、验证、数据划分、SFT/GRPO 数据生成和评测
training/   QLoRA SFT 与 GRPO 训练入口
tests/      单元测试和流水线 smoke
data/       数据占位目录，大数据不提交 Git
reports/    项目日志、实验矩阵和简历材料
```

## 服务器初始化

```bash
git clone -b codex/code-only https://github.com/luzzzp/AlgoAce.git
cd AlgoAce
conda create -n algoace python=3.10 -y
conda activate algoace
pip install -r requirements-train.txt
python -m unittest discover tests
```

长任务的数据和输出建议放到 `/root/autodl-tmp/algoace`。

## 100 题 Smoke

```bash
mkdir -p /root/autodl-tmp/algoace/data

python scripts/convert_taco.py \
  --dataset likaixin/TACO-verified \
  --split train \
  --out-dir /root/autodl-tmp/algoace/data/taco_100 \
  --limit 100 \
  --max-visible-tests 3 \
  --max-reward-tests 20 \
  --max-eval-tests 20 \
  --resume

python scripts/verify_oracles.py \
  --problems /root/autodl-tmp/algoace/data/taco_100 \
  --out-dir /root/autodl-tmp/algoace/data/taco_100_verified \
  --max-solutions-per-problem 3 \
  --resume

python scripts/make_code_sft.py \
  --problems /root/autodl-tmp/algoace/data/taco_100_verified \
  --out /root/autodl-tmp/algoace/data/taco_100_code_sft.jsonl \
  --resume
```

Smoke 只验证流水线，不作为最终准确率结论。

## 正式数据

### 1. 转换与验证

```bash
python scripts/convert_taco.py \
  --dataset likaixin/TACO-verified \
  --split train \
  --out-dir /root/autodl-tmp/algoace/data/taco_15000 \
  --limit 15000 \
  --max-visible-tests 3 \
  --max-reward-tests 20 \
  --max-eval-tests 20 \
  --resume

python scripts/verify_oracles.py \
  --problems /root/autodl-tmp/algoace/data/taco_15000 \
  --out-dir /root/autodl-tmp/algoace/data/taco_15000_verified \
  --max-solutions-per-problem 3 \
  --workers 8 \
  --resume
```

### 2. 按题目划分

```bash
python scripts/split_problems.py \
  --problems /root/autodl-tmp/algoace/data/taco_15000_verified \
  --out-dir /root/autodl-tmp/algoace/data/taco_split \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

同一道题完整进入一个 split。`test/` 不得参与 SFT、GRPO、阈值调整或 prompt 修改。
划分前会按规范化题面、IO 模式和 callable entry point 去重，并优先保留测试更完整的代表题，防止等价题跨 train/dev/test 泄漏。
划分脚本默认排除没有 verified oracle 的题。缺少完整测试组的 verified 题只追加到 train，不进入 dev/test，既保留 SFT 数据量，又确保正式 benchmark 的成功定义一致。
每个 split 会写 `_split_metadata.json`；训练数据脚本检测到 dev/test 角色时会直接拒绝，防止误用测试集。

### 3. 训练数据

```bash
python scripts/make_code_sft.py \
  --problems /root/autodl-tmp/algoace/data/taco_split/train \
  --out /root/autodl-tmp/algoace/data/train_code_sft.jsonl \
  --resume

python scripts/make_grpo_prompts.py \
  --problems /root/autodl-tmp/algoace/data/taco_split/train \
  --out /root/autodl-tmp/algoace/data/grpo_prompts_200.jsonl \
  --limit 200 \
  --resume

python scripts/audit_training_data.py \
  --problems /root/autodl-tmp/algoace/data/taco_split/train \
  --dataset /root/autodl-tmp/algoace/data/train_code_sft.jsonl \
  --out /root/autodl-tmp/algoace/reports/train_data_audit.json \
  --strict
```

## QLoRA SFT

```bash
python training/sft_train.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset /root/autodl-tmp/algoace/data/train_code_sft.jsonl \
  --output-dir /root/autodl-tmp/algoace/outputs/qwen25-coder-7b-code-sft \
  --max-seq-length 4096 \
  --max-prompt-chars 100000 \
  --max-completion-chars 50000 \
  --epochs 2 \
  --learning-rate 2e-4 \
  --warmup-ratio 0.03 \
  --save-steps 250 \
  --save-total-limit 10 \
  --seed 42
```

SFT 只对 assistant 的 Python 代码计算 loss。若题面过长，训练脚本会保留 prompt 首尾并完整保留代码；无法在上下文中为题面预留最小 token 数的超长代码样本会在 preflight 阶段报告并跳过。

中断后追加 `--resume-from-checkpoint`。

## GRPO Smoke

```bash
python training/grpo_train.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter /root/autodl-tmp/algoace/outputs/qwen25-coder-7b-code-sft \
  --prompts /root/autodl-tmp/algoace/data/grpo_prompts_200.jsonl \
  --problems /root/autodl-tmp/algoace/data/taco_split/train \
  --output-dir /root/autodl-tmp/algoace/outputs/qwen25-coder-7b-grpo \
  --num-generations 4 \
  --max-completion-length 2048 \
  --seed 42
```

## 评测

Base pass@1 示例：

```bash
python scripts/evaluate_hf_model.py \
  --problems /root/autodl-tmp/algoace/data/taco_split/test \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --out /root/autodl-tmp/algoace/reports/base_pass1.json \
  --split-name test \
  --max-repair-turns 0 \
  --candidates-per-turn 1 \
  --temperature 0 \
  --seed 42 \
  --load-in-4bit \
  --resume
```

不同实验必须使用相同 test 目录和 seed；每个实验使用独立报告文件。评测会写 `.partial.jsonl` 和 `.partial.meta.json`，配置不一致时拒绝错误续跑。

## 核心指标

- `initial_visible_success_rate`
- `initial_candidate_visible_pass_at_k`
- `initial_verified_success_rate`
- `final_visible_success_rate`
- `verified_success_rate`
- `visible_test_pass_rate`
- `reward_test_pass_rate`
- `eval_test_pass_rate`
- `avg_attempts`
- `syntax_error_rate`
- `runtime_error_rate`
- `timeout_rate`
- `repair_gain`

评测约束：visible tests 可提供具体反例；reward tests 只提供抽象失败信号；eval tests 最终只执行一次，失败后不得继续修复。
