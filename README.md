# AIGC Detector Toolkit

中文 AI 生成文本 (AIGC) **检测 + 降重**工具，六引擎融合检测，一键降 AIGC 率。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 特性

- **六引擎融合检测**
  - **FengCi0** — 12 维文本特征 + 随机森林，毫秒级响应，无需 API
  - **HC3+m3e** — m3e-base 语义编码 + CNN 分类器，本地运行
  - **OpenAI 兼容 API** — 利用 LLM 的 logprobs 计算困惑度
  - **Binoculars** — 双视角 perplexity 比值（ICLR 2024 论文方法）
  - **Local Logprob** — 本地 causal LM 的 LogRank / perplexity（默认关闭）
  - **LastDe** — 多尺度分布熵检测（ICLR 2025）
- **一键降 AIGC 率（refine）** — 检测 → 自动分类 → LLM 润色 → 单段循环复测
- **模型训练** — 支持 FengCi0 RF 和 HC3 CNN 重新训练，含数据增强
- **数据生成** — 通过 LLM API 批量生成 AI 训练数据
- **多格式支持** — `.docx` / `.md` / `.txt`
- **多格式输出** — 终端 / TXT / JSON / HTML 可视化报告

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/clolckliang/aigc-detector-toolkit.git
cd aigc-detector-toolkit

# 安装依赖
uv sync

# 可选：HC3 深度学习引擎
uv sync --extra hc3

# 可选：本地 Logprob 引擎
uv sync --extra local-logprob

# 全部可选引擎
uv sync --extra all
```

### 检测

```bash
# 快速测试（内置样本）
uv run python main.py test

# 检测单个文件
uv run python main.py detect 论文.docx

# 仅用 FengCi0 引擎（最快）
uv run python main.py detect 论文.docx --engine fengci

# 批量检测目录
uv run python main.py batch ./docs/

# 评测标注集
uv run python main.py eval examples/eval_sample.jsonl --engine fengci

# 查看引擎状态
uv run python main.py status
```

### 降 AIGC 率（refine）

```bash
# 检测模型 API（用于 OpenAI/Binoculars/LastDe 等检测引擎）
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"   # 可选
export OPENAI_MODEL="gpt-4o-mini"                     # 可选

# 润色模型 API（可与检测模型不同；留空则复用 OPENAI_*）
export REFINER_API_KEY="your-refiner-key"
export REFINER_BASE_URL="https://api.deepseek.com/v1" # 可选
export REFINER_MODEL="deepseek-chat"                  # 可选

# 默认阈值 0.4（AIGC > 40% 的段落进入循环润色）
uv run python main.py refine 论文.docx

# 更严格的阈值
uv run python main.py refine 论文.docx --threshold 0.3

# 单段最多循环 3 轮，并发复测 4 个段落
uv run python main.py refine 论文.docx --max-rounds 3 --detect-concurrency 4

# 命令行临时指定润色模型
uv run python main.py refine 论文.docx \
  --refiner-api-base https://api.deepseek.com/v1 \
  --refiner-api-key "$DEEPSEEK_API_KEY" \
  --refiner-model deepseek-chat

# 跳过复测
uv run python main.py refine 论文.docx --no-recheck
```

**refine 工作流：**

1. 提取文档段落
2. 六引擎融合逐段检测
3. 自动分类并选择润色策略：
   - `refine_cn` — 表达润色：修复语病、去除口语，克制修改
   - `deai_cn` — 去 AI 味：消除套话、增加个人化表达、细节具体化
   - `deai_en` — 英文去 AI 化：重写为自然学术表达
4. 对每个高风险段落执行“润色 → 检测 → 未达标继续调整”的独立循环
5. 输出报告 + 润色后 DOCX

检测、融合、润色和循环复测统一使用 `core.progress` 管理进度条；交互式终端会优先显示 Rich 面板，非 TTY/CI 环境会自动退化为简洁输出。润色阶段会显示已完成段落、已修改段落和累计循环轮数。

**输出文件：**

| 文件 | 说明 |
|------|------|
| `降AIGC报告_*.txt` | 逐段润色详情 + 润色前后对比 |
| `润色后文本_*.docx` | 润色后完整文本（`[已润色]` 标注） |
| `润色结果_*.json` | 结构化数据 |

## 配置

### OpenAI 兼容 API

```bash
# 方式一：环境变量
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"

# 方式二：编辑 configs/default.yaml
```

支持任何 OpenAI 兼容 API：OpenAI、vLLM、Ollama、LiteLLM、SGLang、DeepSeek 等。

### 润色模型 API

`refine` 可以使用不同的大模型分别负责检测和润色：

```yaml
openai_api:
  api_base: "https://token-plan-cn.xiaomimimo.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "mimo-v2.5"
  strategy: "perplexity"

refiner_api:
  api_base: "https://api.deepseek.com/v1"
  api_key: "${REFINER_API_KEY}"
  model: "deepseek-chat"
  temperature: 0.3
```

环境变量优先级：

```bash
# 检测模型
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."
export OPENAI_MODEL="..."

# 润色模型
export REFINER_API_KEY="..."
export REFINER_BASE_URL="..."
export REFINER_MODEL="..."
export REFINER_TEMPERATURE="0.3"
```

如果 `refiner_api` 和 `REFINER_*` 都没有配置，润色会回退复用 `openai_api`。

### 引擎权重与阈值

编辑 `configs/default.yaml`：

```yaml
engine:
  default: "ensemble"
  ensemble:
    fengci_weight: 0.30
    hc3_weight: 0.20
    openai_weight: 0.25
    binoculars_weight: 0.25
    local_logprob_weight: 0.0
    lastde_weight: 0.0

threshold:
  aigc_threshold: 0.5
```

不配置 API key 时自动退化为 FengCi0 + HC3 双引擎。

### 本地 Logprob 引擎（可选）

```yaml
engine:
  ensemble:
    local_logprob_weight: 0.20

local_logprob:
  model_name_or_path: "/path/to/local/causal-lm"
  device: "cpu"
  method: "logrank"
```

## 模型训练

### 准备训练数据

```text
data/train/
├── ai/          # AI 生成的文本，每篇一个 .txt 文件
│   ├── 0001.txt
│   └── ...
└── human/       # 人工编写的文本，每篇一个 .txt 文件
    ├── 0001.txt
    └── ...
```

### 生成 AI 训练数据

```bash
# 生成 200 条 AI 样本
uv run python scripts/generate_sample_data.py --count 200

# 指定主题和输出目录
uv run python scripts/generate_sample_data.py --count 100 --topics "科技,教育,历史" --output-dir data/train
```

人工文本需手动收集，建议来源：个人博客、新闻稿件、学生作文、学术论文等。

### 训练 FengCi0 随机森林

```bash
uv run python train.py --engine fengci --data-dir data/train/
```

训练流程：加载文本 → 12 维特征提取 → 数据增强 → 训练 RF/LogReg/GB 三个候选 → 自动选最佳 → 阈值校准 → 保存模型。

输出：
- `models/fengci/aigc_detector_model.joblib` — 模型文件
- `models/fengci/aigc_detector_model.metadata.json` — 训练元数据与指标

### 训练 HC3 CNN

```bash
uv run python train.py --engine hc3 --data-dir data/train/ --epochs 30
```

训练流程：加载文本 → m3e-base 编码 → 训练 OptimizedCNN → 早停 → 保存权重。

输出：
- `models/hc3/OptimizedCNN_aigc_detector.pth` — 模型权重

### 训练参数

```bash
# 使用已有评测集作为验证集
uv run python train.py --engine fengci --data-dir data/train/ --eval-data examples/eval_sample.jsonl

# 禁用数据增强
uv run python train.py --engine fengci --data-dir data/train/ --no-augment

# HC3 自定义参数
uv run python train.py --engine hc3 --data-dir data/train/ --epochs 50 --batch-size 64 --lr 0.0005 --patience 8
```

## 引擎说明

| 引擎 | 速度 | 本地模型 | API | 原理 |
| ---- | ---- | :------: | :--: | ---- |
| FengCi0 | 毫秒 | 1MB 自带 | - | 12 维文本特征 + 随机森林 |
| HC3+m3e | 秒级 | 25MB 自带 | - | m3e 语义编码 + CNN 分类 |
| OpenAI 兼容 API | ~0.5s/段 | - | 需要 | LLM logprobs 困惑度 |
| Binoculars | ~0.5s/段 | - | 需要 | 双视角 ppl/x_ppl 比值（ICLR 2024） |
| Local Logprob | 取决于模型 | 用户配置 | - | 本地 LM LogRank / perplexity |

默认融合权重：FengCi0 30% + HC3 20% + OpenAI API 25% + Binoculars 25%。

## 输出示例

`detect` 默认生成三类报告：

- `AIGC检测报告_*.txt` — 快速查看整体结果
- `AIGC检测结果_*.json` — 脚本处理和后续对比
- `AIGC可视化报告_*.html` — 浏览器查看，红色下划线标出高风险段落

```text
============================================================
              AIGC 查重检测报告
============================================================
检测引擎:   ensemble
  FengCi0       : ✓ (权重 37%)
  HC3+m3e       : ✓ (权重 33%)
  OpenAI 兼容 API: ✗ (权重 0%)
  Binoculars    : ✗ (权重 0%)
  Local Logprob : ✗ (权重 0%)
------------------------------------------------------------
总段落数:     454
判定为人工撰写: 444 (97.8%)
判定为AI生成:    10 ( 2.2%)
平均AIGC分数:   24.7%
============================================================
```

## Python API

### 检测

```python
from core.engine import DetectionEngine

engine = DetectionEngine(mode="ensemble")

# 单段检测
result = engine.detect_single("待检测的文本...")
print(result["aigc_score"], result["label"])

# 批量检测
results = engine.detect_batch(["文本1", "文本2", "文本3"])
for r in results:
    print(r["aigc_score"], r["label"])
```

### 降 AIGC 率

```python
from core.refiner import RefinementEngine, generate_refinement_report, export_refined_docx

refiner = RefinementEngine(
    api_base="https://api.deepseek.com/v1",
    api_key="sk-...",
    model="deepseek-chat",
    temperature=0.3,
    concurrency=8,
)

# 单段润色
refined, note = refiner.refine_paragraph("综上所述，...", strategy="deai_cn")

# 批量润色
ref_results = refiner.refine_batch(paragraphs, detection_results, score_threshold=0.4)

# 循环润色：每段润色后立即复测，未达标继续调整
ref_results = await refiner.refine_batch_iterative_async(
    paragraphs,
    detection_results,
    detect_fn=engine.detect_single,
    score_threshold=0.4,
    max_rounds=3,
    detect_concurrency=4,
)

# 生成报告
generate_refinement_report(ref_results, "报告.txt")
export_refined_docx(paragraphs, ref_results, "润色后.docx")
```

## 结果解读

| AIGC 分数 | 含义 | 建议 |
| --------- | ---- | ---- |
| 0-20% | 极低 AI 概率，人工风格明确 | 无需处理 |
| 20-40% | 低 AI 概率，偏人工 | 可选择性润色 |
| 40-50% | 接近阈值 | 建议润色 |
| 50-70% | 偏高 AI 概率 | 强烈建议润色 |
| 70-100% | 高 AI 概率 | 必须润色 |

## 项目结构

```text
aigc-detector-toolkit/
├── main.py                          # CLI 主入口（detect/batch/refine/eval/status/test）
├── train.py                         # 模型训练入口
├── pyproject.toml                   # uv 项目配置，含可选依赖组
├── requirements.txt
├── configs/
│   └── default.yaml                 # 默认配置（引擎权重、API、阈值）
├── core/
│   ├── engine.py                    # 多引擎融合检测器
│   ├── refiner.py                   # 降 AIGC 率润色引擎（三策略 prompt）
│   ├── trainer.py                   # 模型训练（FengCiTrainer / HC3Trainer）
│   ├── evaluation.py                # 评测指标、阈值扫描
│   ├── fengci_adapter.py            # FengCi0 特征提取 + RF 推理（自包含）
│   ├── hc3_adapter.py               # HC3 m3e + CNN 推理（自包含）
│   ├── openai_adapter.py            # OpenAI 兼容 API logprobs
│   ├── binoculars_adapter.py        # Binoculars 双视角检测
│   ├── local_logprob_adapter.py     # 可选本地 LM LogRank
│   └── progress.py                  # 进度条工具
├── extractors/                      # 文档段落提取（docx/md/txt）
├── reporters/                       # 输出报告（终端/txt/json/html）
├── models/
│   ├── fengci/
│   │   ├── aigc_detector_model.joblib
│   │   └── aigc_detector_model.metadata.json
│   └── hc3/
│       └── OptimizedCNN_aigc_detector.pth
├── scripts/
│   └── generate_sample_data.py      # LLM API 生成训练数据
├── examples/
│   └── eval_sample.jsonl            # 评测样本
├── skills/
│   └── aigc-detect/
│       ├── SKILL.md                 # AI Agent Skill 定义
│       └── INSTALL.md               # Skill 安装说明
└── output/                          # 检测输出（git 忽略）
```

## Claude Code / Agent Skill

项目内置 `aigc-detect` Skill，支持 Claude Code、Cursor、Hermes、Codex 等 AI Agent 框架。

```bash
# Claude Code 项目级安装
mkdir -p .claude/skills/aigc-detect
cp skills/aigc-detect/SKILL.md .claude/skills/aigc-detect/
```

自然语言触发：

```text
"帮我检测这篇论文是否有 AI 生成内容"
"降低这篇论文的 AIGC 率"
"训练一个新的 FengCi0 模型"
"生成 200 条 AI 训练数据"
```

详见 [skills/aigc-detect/INSTALL.md](skills/aigc-detect/INSTALL.md)。

## 许可证

MIT License
