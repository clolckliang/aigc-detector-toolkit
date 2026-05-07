# AIGC Detector Toolkit

中文 AI 生成文本 (AIGC) 检测工具，五引擎融合，支持模型训练与数据生成。

## 特性

- **五引擎融合检测**
  - **FengCi0** — 12 维文本特征 + 随机森林，毫秒级响应，无需 API
  - **HC3+m3e** — m3e-base 语义编码 + CNN 分类器，本地运行
  - **OpenAI 兼容 API** — 利用 LLM 的 logprobs 计算困惑度
  - **Binoculars** — 双视角 perplexity 比值（ICLR 2024 论文方法）
  - **Local Logprob** — 本地 causal LM 的 LogRank / perplexity（默认关闭）
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

## 配置

### OpenAI 兼容 API（可选，提升准确率）

```bash
# 方式一：环境变量
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"

# 方式二：编辑 configs/default.yaml
```

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

项目支持重新训练 FengCi0 和 HC3 模型以提升准确率。

### 准备训练数据

数据目录结构：

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

使用 LLM API 批量生成：

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

训练流程：加载文本 -> 12 维特征提取 -> 数据增强 -> 训练 RF/LogReg/GB 三个候选 -> 自动选最佳 -> 阈值校准 -> 保存模型。

输出：

- `models/fengci/aigc_detector_model.joblib` — 模型文件
- `models/fengci/aigc_detector_model.metadata.json` — 训练元数据与指标

### 训练 HC3 CNN

```bash
uv run python train.py --engine hc3 --data-dir data/train/ --epochs 30
```

训练流程：加载文本 -> m3e-base 编码 -> 训练 OptimizedCNN -> 早停 -> 保存权重。

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
| FengCi0 | 毫秒 | 1MB | - | 12 维文本特征 + 随机森林 |
| HC3+m3e | 秒级 | 25MB | - | m3e 语义编码 + CNN 分类 |
| OpenAI 兼容 API | ~0.5s/段 | - | 需要 | LLM logprobs 困惑度 |
| Binoculars | ~0.5s/段 | - | 需要 | 双视角 ppl/x_ppl 比值 |
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
  FengCi0:  ✓ (权重 37%)
  HC3+m3e:  ✓ (权重 33%)
  OpenAI 兼容 API: ✗ (权重 0%)
  Binoculars: ✗ (权重 0%)
  Local Logprob: ✗ (权重 0%)
------------------------------------------------------------
总段落数:     454
判定为人工撰写: 444 (97.8%)
判定为AI生成:    10 ( 2.2%)
平均AIGC分数:   24.7%
============================================================
```

## Python API

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

## 结果解读

| AIGC 分数 | 含义 |
| --------- | ---- |
| 0-20% | 极低 AI 概率，人工风格明确 |
| 20-40% | 低 AI 概率，偏人工 |
| 40-50% | 接近阈值，需结合上下文判断 |
| 50-70% | 偏高 AI 概率，建议改写 |
| 70-100% | 高 AI 概率，强烈建议改写 |

## 项目结构

```text
aigc-detector-toolkit/
├── main.py                          # CLI 主入口（检测/评测/状态）
├── train.py                         # 模型训练入口
├── requirements.txt
├── configs/
│   └── default.yaml                 # 默认配置（引擎权重、API、阈值）
├── core/
│   ├── engine.py                    # 多引擎融合检测器
│   ├── trainer.py                   # 模型训练逻辑（FengCiTrainer / HC3Trainer）
│   ├── evaluation.py                # 评测指标、阈值扫描
│   ├── fengci_adapter.py            # FengCi0 特征提取 + RF 推理
│   ├── hc3_adapter.py               # HC3 m3e + CNN 推理
│   ├── openai_adapter.py            # OpenAI 兼容 API logprobs
│   ├── binoculars_adapter.py        # Binoculars 双视角检测
│   └── local_logprob_adapter.py     # 可选本地 LM LogRank
├── scripts/
│   └── generate_sample_data.py      # LLM API 生成训练数据
├── extractors/                      # 文档段落提取（docx/md/txt）
├── reporters/                       # 输出报告（终端/txt/json/html）
├── models/
│   ├── fengci/
│   │   ├── aigc_detector_model.joblib
│   │   └── aigc_detector_model.metadata.json
│   └── hc3/
│       └── OptimizedCNN_aigc_detector.pth
├── data/
│   └── train/                       # 训练数据（ai/ + human/）
├── examples/
│   └── eval_sample.jsonl            # 评测样本
└── output/                          # 检测输出（git 忽略）
```

## 许可证

MIT License
