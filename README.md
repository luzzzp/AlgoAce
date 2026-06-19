# AlgoAce

AlgoAce 是一个面向算法题的可验证 Python 代码生成系统。项目只优化代码正确率：

```text
题面 -> 候选代码生成 -> 可见测试筛选 -> 上下文受控修复 -> 留出测试验证
```

模型输出只包含一个 `python` 代码块。核心实验比较：

- Base model
- Code SFT
- Code SFT + context-guided repair
- Code SFT + GRPO
- Code SFT + GRPO + context-guided repair

## 项目结构

```text
algoace/    核心 schema、执行器、上下文构造、reward 和求解控制器
scripts/    TACO 转换、oracle 验证、SFT 数据生成和模型评测
training/   SFT 与 GRPO 训练入口
tests/      单元测试
data/       本地数据占位目录，大数据不提交 Git
reports/    实验日志和简历材料
```

## 本地验证

```bash
python -m unittest discover tests
```

## 数据流水线

### 1. 转换 TACO-verified

```bash
python scripts/convert_taco.py \
  --dataset likaixin/TACO-verified \
  --split train \
  --out-dir data/problems/taco_100 \
  --limit 100
```

### 2. 验证 Python oracle

```bash
python scripts/verify_oracles.py \
  --problems data/problems/taco_100 \
  --out-dir data/problems/taco_100_verified \
  --max-solutions-per-problem 3
```

### 3. 构造 Code SFT

```bash
python scripts/make_code_sft.py \
  --problems data/problems/taco_100_verified \
  --out data/processed/taco_100_code_sft.jsonl
```

### 4. QLoRA SFT

```bash
python training/sft_train.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset data/processed/taco_100_code_sft.jsonl \
  --output-dir outputs/qwen25-coder-7b-code-sft
```

### 5. 端到端评测

```bash
python scripts/evaluate_hf_model.py \
  --problems data/problems/taco_100_verified \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --adapter outputs/qwen25-coder-7b-code-sft \
  --out reports/sft_agent_eval.json \
  --max-repair-turns 3 \
  --candidates-per-turn 4
```

## 核心指标

- `initial_visible_success_rate`
- `initial_verified_success_rate`
- `final_visible_success_rate`
- `verified_success_rate`
- `visible_test_pass_rate`
- `eval_test_pass_rate`
- `avg_attempts`
- `syntax_error_rate`
- `runtime_error_rate`
- `timeout_rate`
- `repair_gain`

