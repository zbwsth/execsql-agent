<div align="center">

# ExecSQL-Agent

### Execution-grounded Text-to-SQL Agent with Tool Calling, QLoRA SFT and Verifiable Evaluation

让模型不只是生成 SQL，而是主动探索数据库、执行查询、观察反馈、修复错误，并用真实执行结果验证答案。

<p>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Model-Qwen3--8B-6C5CE7?style=flat-square" alt="Qwen3-8B">
  <img src="https://img.shields.io/badge/Inference-vLLM-00A67E?style=flat-square" alt="vLLM">
  <img src="https://img.shields.io/badge/SFT-4--bit%20QLoRA-F39C12?style=flat-square" alt="4-bit QLoRA">
  <img src="https://img.shields.io/badge/Database-SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white" alt="SQLite">
</p>

<img src="docs/assets/execsql-agent-hero.png" width="100%" alt="ExecSQL-Agent multi-database agent illustration">

<p>
  <a href="#项目概览">项目概览</a> ·
  <a href="#核心能力">核心能力</a> ·
  <a href="#实验结果">实验结果</a> ·
  <a href="#sft-数据与训练">SFT</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="#离线评测">离线评测</a>
</p>

</div>

> [!IMPORTANT]
> **当前稳定主线为 SQL Agent + QLoRA SFT + execution-based evaluation。**
>
> Agentic GRPO / RLVR 已具备 verifier、数据适配与 VERL diagnostic 链路，但尚未形成正式 Qwen3-8B GRPO benchmark，因此本仓库不声明 GRPO 模型效果。

## 项目概览

ExecSQL-Agent 不是一次性完成“问题到 SQL”映射的脚本，而是一个面向真实数据库交互的多轮 SQL Agent。模型会根据问题与当前上下文动态发现表、检索 Schema、生成并校验 SQL，再利用真实 SQLite 执行结果继续回答或修复错误。

项目使用 Qwen3-8B 作为基础模型，通过 vLLM 的 OpenAI-compatible API 提供推理服务。在线 Agent、数据库工具、离线评测、SFT 数据构造与 RLVR verifier 复用相同的 ToolRegistry 和 SQLite 执行边界，避免训练、评测与运行时各自维护一套数据库逻辑。

核心实现：

- [FunctionCallingAgent](src/execsql_agent/agents/function_calling.py)：多轮 Tool Calling 与 Agent 状态机
- [ToolRegistry](src/execsql_agent/tools/registry.py)：工具注册、参数校验、调用调度与结果回填
- [SQLValidator](src/execsql_agent/tools/sql_validator.py)：只读 SQL 安全校验
- [SQLExecutor](src/execsql_agent/tools/sql_executor.py)：受限 SQLite 执行
- [Evaluator](src/execsql_agent/evaluation/evaluator.py)：离线评测与报告生成

## 核心能力

### Dynamic Agent Loop

模型每轮都会接收 System Prompt、用户问题、Session Context、历史 Tool Calls 和真实 Tool Observations，并动态决定下一步操作。运行时并不强制所有问题采用同一条固定流水线。

Agent Loop 支持：

- native Tool Calling 与 JSON fallback；
- 多轮、多工具调用及 Observation 回填；
- 无状态重复 Tool Call 与重复 SQL 检测；
- 不安全 SQL 和最大 Agent step 终止；
- **Completion Guard**：模型尚未取得数据库证据便提前回答时，引导其继续使用工具；
- 基于 Session ID 隔离的短期多轮记忆。

### 四个结构化数据库工具

| Tool | 作用 |
|---|---|
| `list_tables` | 在未知数据库中发现可用表 |
| `inspect_schema` | 查看字段、主键与外键关系 |
| `validate_sql` | 对单条只读 SQLite 查询进行静态与编译期校验 |
| `execute_sql` | 在受限只读连接中执行 SQL，并返回结构化结果或错误 |

所有模型可见工具均由 ToolRegistry 提供严格 JSON Schema、参数校验和统一调度。工具 Observation 会进入下一轮模型上下文，但可信数据库路径和评分侧信息不会暴露给模型。

### Execution Feedback 与修复

SQLite 执行结果、SQL 异常和工具参数错误会作为 Observation 回传。Agent 可以据此重新检查 Schema、修复字段或表名、调整 SQL 并再次执行，而不是一次失败即结束。

系统分别记录首次执行、最终执行、有效修复、重复调用、协议完成和结果正确性，因此“SQL 能够执行”与“SQL 语义正确”不会被混为同一指标。

### 只读执行边界

Agent 只能执行单条只读 `SELECT` / `WITH` 查询。运行时同时使用静态校验、SQLite read-only URI、`query_only`、authorizer、progress handler 与 wall-clock timeout。DDL、DML、`ATTACH` 和危险 `PRAGMA` 不会进入正常执行路径。

## 系统设计

项目按职责划分为三个相互复用的层次：

1. **Agent Runtime**：Qwen3 通过 OpenAI-compatible 接口完成多轮 Tool Calling，Session Memory 维护隔离的会话上下文。
2. **Structured Tool Layer**：ToolRegistry 统一提供表发现、Schema 检索、SQL 校验与只读执行，将结构化 Observation 回填给 Agent。
3. **Offline Evaluation & Post-training**：Trajectory Logger 记录完整交互，Evaluator 通过私有评分信息完成结果等价性评测；SFT 与 RLVR 数据构造复用真实工具输出。

Gold SQL、expected result、数据库路径和 scorer metadata 均位于私有评分侧，不会注入 Agent 的 system/user prompt。训练、评测与推理共享同一套工具协议，使离线学习目标与线上 Agent 行为保持一致。

## 实验结果

正式 Base / SFT 对照使用从 BIRD Train 按数据库级隔离构造的 held-out evaluation split，共 **943 个样本，覆盖 7 个训练阶段未见数据库**。

两组实验使用完全相同的 case 顺序、system prompt、Tool Schema、ToolRegistry、SQLite databases、decoding configuration 和 evaluator。唯一主要变量是是否加载正式 SFT LoRA Adapter。

| Metric | Qwen3-8B Base | Qwen3-8B + SFT | Change |
|---|---:|---:|---:|
| **SQL Execution Accuracy** | 23.44% (221/943) | **37.43% (353/943)** | **+14.00pp** |
| Protocol Completion | 63.84% | **81.55%** | **+17.71pp** |
| First Execution Success | 57.79% | **81.97%** | **+24.18pp** |
| Final Execution Success | 67.44% | **88.02%** | **+20.57pp** |
| no-final-SQL | 149 | **14** | **-90.6%** |

> [!NOTE]
> SFT 不仅提升最终准确率，也显著改善了工具协议遵循、Schema 交互和 SQL 可执行性。训练后，剩余错误更多集中在真正困难的 SQL semantic mismatch，而不是 Agent 基础设施失败。

<details>
<summary><strong>查看指标定义</strong></summary>

- **SQL Execution Accuracy**：预测 SQL 与 scorer-side reference SQL 分别在真实 SQLite 上执行，并比较完整结果集合是否等价。
- **Protocol Completion**：Agent 正常完成多轮工具协议并到达有效最终状态。
- **First Execution Success**：第一次 `execute_sql` 即成功执行。
- **Final Execution Success**：trajectory 结束前存在最终成功执行结果，代表 SQL 可以执行，但不等同于语义一定正确。

</details>

以上结果来自项目冻结的 database-level held-out split，不是 BIRD 官方 Test leaderboard 成绩。

## SFT 数据与训练

### 数据隔离与 Expert Trajectories

训练数据基于 BIRD Train 构建，并按 `database_id` 完成数据库级划分，而不是对问题随机切分：

- 61 个数据库用于训练数据候选池；
- 7 个完整数据库作为 held-out evaluation split；
- 训练数据库与 held-out 数据库不重叠；
- BIRD Mini-Dev 不参与 SFT 或 GRPO 参数更新。

| Split | Expert Trajectories | Assistant-turn Samples |
|---|---:|---:|
| Train | **2,500** | **12,246** |
| Dev | **110** | **540** |

每条 Expert Agent Trajectory 的 Schema、validation 与 execution observations 均来自真实 ToolRegistry 和 SQLite。数据构造执行 physical-table check、Gold SQL execution、tool-call/result pairing、oracle leakage scan 与 4,096-token hard-stop；任一硬条件失败时不会生成最终训练样本。

### Assistant-only supervision

一条多轮 trajectory 会按 assistant turn 展开为多个监督样本：

- system、user 和 tool observation tokens 全部 mask；
- assistant tool-call、SQL action 与 final-answer tokens 参与 loss；
- 后续 assistant turn 可以读取此前真实工具结果，但这些上下文 token 不参与监督损失。

### QLoRA 配置

正式 SFT 使用 Transformers、PEFT 与 bitsandbytes 对 Qwen3-8B 进行 4-bit QLoRA，训练 1 epoch，共 3,062 optimizer steps。

<details>
<summary><strong>查看训练配置与结果</strong></summary>

| Configuration | Value |
|---|---|
| Quantization | NF4 + double quantization |
| Compute dtype | BF16 |
| LoRA | r=16, alpha=32, dropout=0.05, all-linear |
| Max length | 4,096 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| Optimizer | paged_adamw_8bit |
| Scheduler | cosine |
| Training loss | 0.06396 |
| Final Dev loss | 0.04622 |
| Peak allocated VRAM | 18.39 GiB |
| Peak reserved VRAM | 30.44 GiB |

</details>

训练流程同时记录 Git revision、输入数据 SHA256、Tool Schema SHA256、基础模型信息、超参数和运行环境。模型权重与正式 LoRA Adapter 不包含在本仓库中。

## 快速开始

### 安装

```bash
git clone https://github.com/zbwsth/execsql-agent.git
cd execsql-agent

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -e ".[dev]"
```

环境要求为 Python 3.11+ 与 SQLite。本地 deterministic demo 和完整 CPU 测试不需要 GPU。

### 运行 deterministic demo

```bash
python scripts/create_demo_database.py

python -m execsql_agent.cli run \
  --agent-mode function-calling \
  --database data/demo.db \
  --question "查询已完成订单中消费金额最高的五位客户"
```

该模式使用 deterministic FakeLLM，适合验证 Agent Loop、错误回填、SQL 修复和 trajectory logging，不代表真实模型能力。

<details>
<summary><strong>接入 vLLM / OpenAI-compatible 推理服务</strong></summary>

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_MODEL="your-served-model"

python -m execsql_agent.cli run \
  --agent-mode function-calling \
  --database /path/to/database.sqlite \
  --question "统计每个地区已完成订单的总金额" \
  --llm openai \
  --max-steps 6
```

LLM client 支持请求超时、有限重试、native Tool Calling、JSON fallback 与显式关闭 Qwen thinking mode。

</details>

## 离线评测

Evaluator 直接消费完整 Agent trajectory，提取候选 SQL，并在真实 SQLite 上比较预测结果与 scorer-side reference result。它不使用 LLM judge。

支持能力包括：

- real SQLite result-based scoring；
- 多数据库路径解析与 database fingerprint validation；
- scorer-side Gold Cache；
- checkpoint / resume 与配置指纹检查；
- failure taxonomy、数据库维度与工具使用统计；
- JSON、CSV 与 Markdown 报告。

Gold SQL、expected result、database path 和 scorer metadata 始终位于私有评分侧。

<details>
<summary><strong>运行 synthetic evaluation</strong></summary>

```bash
python scripts/create_demo_database.py

python -m execsql_agent.cli evaluate \
  --agent-mode both \
  --database data/demo.db \
  --dataset data/synthetic/eval_questions.json \
  --dataset-format native \
  --output-dir outputs/synthetic
```

</details>

<details>
<summary><strong>运行 BIRD evaluation</strong></summary>

BIRD annotations 与 SQLite databases 需要从官方来源自行准备。本仓库不分发 BIRD databases、Gold SQL、private Gold Cache 或完整 benchmark outputs。

```bash
python -m execsql_agent.cli evaluate \
  --agent-mode function-calling \
  --database-root /path/to/bird/databases \
  --dataset /path/to/bird_eval.json \
  --dataset-format bird \
  --bird-gold-cache /path/to/private_gold_cache.jsonl \
  --bird-preflight-report /path/to/preflight_report.json \
  --bird-protocol-config config/bird_eval_protocol.json \
  --output-dir outputs/bird_eval \
  --real-model
```

冻结评测协议见 [config/bird_eval_protocol.json](config/bird_eval_protocol.json)。

</details>

## 训练入口

### 通用 SFT trajectory builder

```bash
python training/build_sft_dataset.py \
  --database /path/to/database.sqlite \
  --cases /path/to/cases.json \
  --split train \
  --output /path/to/train.jsonl \
  --tools-output /path/to/tools.json
```

### Assistant-only preprocessing 检查

```bash
python training/assistant_turn_preprocessing.py \
  --model /path/to/Qwen3-8B \
  --train /path/to/train.jsonl \
  --dev /path/to/dev.jsonl
```

### QLoRA SFT

```bash
python training/train_qlora_sft_full.py \
  --model /path/to/Qwen3-8B \
  --train /path/to/train.jsonl \
  --dev /path/to/dev.jsonl \
  --tools /path/to/tools.json \
  --output /path/to/new-adapter-output \
  --epochs 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4
```

核心 Agent 依赖与 GPU training dependencies 有意分离。训练环境需要自行安装与目标 CUDA 版本兼容的 PyTorch、Transformers、PEFT、bitsandbytes 和 Accelerate。

## Agentic RLVR / GRPO

Agentic RLVR 是当前实验方向，而不是仓库已经完成的模型效果结论。当前已有 SQL response parser、execution verifier、BIRD agentic dataset adapter、VERL ToolAgentLoop integration、execution-based reward 与 Qwen3-0.6B single-step diagnostic。

当前保留两种 diagnostic reward view：

- **R1**：结果完全等价为 1，否则为 0；
- **R2**：结果等价为 1，可执行但错误为 0.2，其他为 0。

目前没有正式 Qwen3-8B Agentic GRPO checkpoint 或 benchmark improvement claim。更多信息见 [Qwen3-0.6B GRPO diagnostic](docs/experiments/qwen3_0.6b_grpo_diagnostic.md)。

## 项目结构

<details>
<summary><strong>展开项目目录</strong></summary>

```text
execsql-agent/
├── src/execsql_agent/
│   ├── agents/          # Tool Calling Agent 与 pipeline baseline
│   ├── llm/             # FakeLLM 与 OpenAI-compatible client
│   ├── tools/           # ToolRegistry、Schema、Validator、Executor
│   ├── evaluation/      # Dataset adapter、scorer、metrics、reports
│   ├── trajectory/      # Structured logging 与 Session Memory
│   └── rlvr/            # SQL response parser 与 execution verifier
├── training/            # SFT/GRPO 数据构造、预处理与训练入口
├── scripts/             # Demo、BIRD preflight、rescore 与 diagnostics
├── config/              # 冻结评测协议
├── configs/             # Runtime / VERL 配置
├── tests/               # CPU regression 与 contract tests
├── data/synthetic/      # 可公开的合成评测 fixture
└── data/grpo_smoke/     # DEV-only GRPO diagnostic fixture
```

</details>

## 质量保证

```bash
pytest
ruff check .
mypy
git diff --check
```

当前公开版本包含 210 个通过的测试与 4 个跳过项。测试覆盖 Tool Schema、参数校验、SQL 安全边界、多轮工具协议、Completion Guard、Session 隔离、BIRD scoring、评测续跑、Assistant-only masking、数据泄漏检查和 RLVR verifier。

## 范围与限制

- 当前数据库后端为 SQLite；
- Agent 只有 read-only access；
- CLI 当前为同步执行；
- 仓库不包含 Qwen3 权重、正式 SFT Adapter 或 BIRD 原始数据；
- 正式结果来自项目自建 database-level held-out split，不代表 BIRD 官方 Test leaderboard；
- Agentic GRPO 尚未形成正式效果结论。

## Acknowledgements

ExecSQL-Agent builds on Qwen3, vLLM, BIRD, Transformers, PEFT, bitsandbytes and VERL.

---

<div align="center">

**ExecSQL-Agent · Tool-augmented Text-to-SQL with execution feedback**

<sub>Execute, observe, repair and verify.</sub>

</div>
