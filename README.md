# AIGC Detector Toolkit

中文 AIGC 文本检测、逐段解释和降重润色工具。项目提供命令行、WebUI、训练脚本和多格式报告导出，适合论文、报告、课程文档和长文本批量审阅。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 核心能力

- **多引擎融合检测**：FengCi0、HC3+m3e、OpenAI 兼容 API、Binoculars、Local Logprob、LastDe。
- **逐段解释**：输出综合分、标签、引擎明细、风险原因和分数分布。
- **降 AIGC 率润色**：检测高风险段落后，按段自动选择润色策略，并支持“润色 → 复测 → 继续调整”的循环。
- **润色前后对比**：WebUI 和 HTML 报告展示每段润色前分数、润色后分数和变化值。
- **多格式输入输出**：支持 `.docx`、`.md`、`.txt`，输出 TXT、JSON、CSV、HTML、DOCX。
- **WebUI 工作台**：支持文件上传、文本粘贴、单段复检、单段重新润色、配置编辑和结果导出。
- **模型训练与评测**：支持 FengCi0、HC3 训练，支持 JSONL/CSV 标注集评测。

## 安装

推荐使用 `uv` 管理 Python 依赖。

```bash
git clone https://github.com/clolckliang/aigc-detector-toolkit.git
cd aigc-detector-toolkit

# 基础依赖
uv sync

# 开发依赖，包含 pytest
uv sync --group dev

# 可选：HC3 本地深度学习引擎
uv sync --extra hc3

# 可选：本地 Logprob 引擎
uv sync --extra local-logprob

# 安装全部可选引擎
uv sync --extra all --group dev
```

WebUI 前端开发需要 Node.js 20+ 和 npm。普通使用 `webui.py` 不需要重新构建前端，仓库已包含生产构建产物。

## 快速使用

### 命令行检测

```bash
# 内置样本快速测试
uv run python main.py test

# 检测单个文件
uv run python main.py detect 论文.docx

# 指定检测引擎
uv run python main.py detect 论文.docx --engine fengci

# 批量检测目录
uv run python main.py batch ./docs

# 评测标注集
uv run python main.py eval examples/eval_sample.jsonl --engine fengci

# 查看引擎状态
uv run python main.py status
```

`detect` 默认会在输入文件同目录生成：

- `AIGC检测报告_*.txt`
- `AIGC检测结果_*.json`
- `AIGC可视化报告_*.html`

### WebUI

```bash
uv run python webui.py --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。

WebUI 支持：

- 检测 / 润色两种任务模式
- 上传 `.docx`、`.md`、`.txt` 或直接粘贴文本
- 配置检测阈值、最小段长、润色阈值、最大轮数和复测并发
- 查看分数分布、平均分、AI 判定数、已修改数和引擎状态
- 逐段展开查看引擎贡献、风险原因、原文 / 改写 / Diff
- 润色结果按段展示 `润色前 → 润色后` 分数和变化值
- 单段手动修改、单段复检、单段重新润色
- JSON / CSV 导出，CSV 包含 `score_before`、`score_after`、`score_delta`

### 降 AIGC 率润色

```bash
# 检测模型 API
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"

# 润色模型 API，可与检测模型不同；不配置时复用 OPENAI_*
export REFINER_API_KEY="your-refiner-key"
export REFINER_BASE_URL="https://api.deepseek.com/v1"
export REFINER_MODEL="deepseek-chat"

# 默认阈值 0.4，高于该分数的段落进入润色流程
uv run python main.py refine 论文.docx

# 更严格阈值
uv run python main.py refine 论文.docx --threshold 0.3

# 每段最多 3 轮，复测并发 4
uv run python main.py refine 论文.docx --max-rounds 3 --detect-concurrency 4

# 临时指定润色模型
uv run python main.py refine 论文.docx \
  --refiner-api-base https://api.deepseek.com/v1 \
  --refiner-api-key "$DEEPSEEK_API_KEY" \
  --refiner-model deepseek-chat

# 跳过润色后复测
uv run python main.py refine 论文.docx --no-recheck
```

`refine` 工作流：

1. 提取文档段落。
2. 使用检测引擎逐段计算 AIGC 分数。
3. 对超过阈值的段落自动选择策略：
   - `refine_cn`：中文表达润色，修复语病和口语化表达。
   - `deai_cn`：中文去 AI 味，减少套话和模板化句式。
   - `deai_en`：英文去 AI 化，改写为自然学术表达。
4. 每段独立执行润色和复测，未达标可继续下一轮。
5. 导出报告和润色后文档。

`refine` 输出文件：

| 文件 | 用途 |
| ---- | ---- |
| `降AIGC报告_*.txt` | 纯文本润色报告 |
| `降AIGC可视化报告_*.html` | 浏览器查看的润色前后对比报告 |
| `润色后文本_*.docx` | 可直接替换使用的润色后文本 |
| `润色结果_*.json` | 结构化润色结果 |

## 配置

默认配置位于 `configs/default.yaml`。环境变量会覆盖配置中的 API 字段。

### 检测 API

```yaml
openai_api:
  api_base: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  strategy: "perplexity"
```

支持 OpenAI 兼容 API，包括 OpenAI、DeepSeek、vLLM、Ollama、LiteLLM、SGLang 等。

可用环境变量：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."
export OPENAI_MODEL="..."
```

### 润色 API

```yaml
refiner_api:
  api_base: "https://api.deepseek.com/v1"
  api_key: "${REFINER_API_KEY}"
  model: "deepseek-chat"
  temperature: 0.3
```

可用环境变量：

```bash
export REFINER_API_KEY="..."
export REFINER_BASE_URL="..."
export REFINER_MODEL="..."
export REFINER_TEMPERATURE="0.3"
```

### 引擎权重和阈值

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

未配置 API key 时，项目会使用可用的本地引擎。基础安装通常至少可用 FengCi0。

## 引擎说明

| 引擎 | 运行方式 | 说明 |
| ---- | -------- | ---- |
| FengCi0 | 本地 | 12 维文本特征 + 随机森林，响应快，无 API 依赖 |
| HC3+m3e | 本地 | m3e-base 语义编码 + CNN 分类器 |
| OpenAI API | API | 通过 OpenAI 兼容接口计算 logprobs / perplexity |
| Binoculars | API | 双视角 perplexity 比值方法 |
| Local Logprob | 本地可选 | 使用本地 causal LM 计算 LogRank / perplexity |
| LastDe | API | 多尺度分布熵检测 |

## WebUI 开发

前端源码位于 `webui_src/`，生产构建输出到 `webui/`，由 `webui.py` 托管。

```bash
npm install

# 终端 1：Python API 和静态文件服务
uv run python webui.py --host 127.0.0.1 --port 8765

# 终端 2：Vite 开发服务
npm run dev

# 生产构建
npm run build
```

Vite 会将 `/api/*` 代理到 `http://127.0.0.1:8765`。

## 测试与质量检查

```bash
# Python 单元测试
uv run python -m pytest

# 指定 reporter 测试
uv run python -m pytest tests/test_reporters.py

# 前端生产构建
npm run build

# Python 语法检查
python -m py_compile main.py webui.py core/refiner.py reporters/refinement_html_reporter.py
```

在受限环境中如果 `uv` 无法写入默认缓存，可以临时指定缓存目录：

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m pytest
```

## 训练与评测

训练数据目录格式：

```text
data/train/
├── ai/
│   ├── 0001.txt
│   └── ...
└── human/
    ├── 0001.txt
    └── ...
```

生成 AI 样本：

```bash
uv run python scripts/generate_sample_data.py --count 200
uv run python scripts/generate_sample_data.py --count 100 --topics "科技,教育,历史" --output-dir data/train
```

训练 FengCi0：

```bash
uv run python train.py --engine fengci --data-dir data/train
```

训练 HC3：

```bash
uv run python train.py --engine hc3 --data-dir data/train --epochs 30
```

常用训练参数：

```bash
uv run python train.py --engine fengci --data-dir data/train --eval-data examples/eval_sample.jsonl
uv run python train.py --engine fengci --data-dir data/train --no-augment
uv run python train.py --engine hc3 --data-dir data/train --epochs 50 --batch-size 64 --lr 0.0005 --patience 8
```

## Python API

### 检测

```python
from core.engine import DetectionEngine

engine = DetectionEngine(mode="ensemble")

result = engine.detect_single("待检测文本")
print(result["aigc_score"], result["label"])

results = engine.detect_batch(["文本1", "文本2"])
```

### 润色

```python
from core.refiner import RefinementEngine, generate_refinement_report, export_refined_docx
from reporters.refinement_html_reporter import generate_refinement_html_report

refiner = RefinementEngine(
    api_base="https://api.deepseek.com/v1",
    api_key="sk-...",
    model="deepseek-chat",
    temperature=0.3,
    concurrency=8,
)

refined, note = refiner.refine_paragraph("综上所述，...", strategy="deai_cn")

ref_results = await refiner.refine_batch_iterative_async(
    paragraphs,
    detection_results,
    detect_fn=engine.detect_single,
    score_threshold=0.4,
    max_rounds=3,
    detect_concurrency=4,
)

generate_refinement_report(ref_results, "报告.txt")
generate_refinement_html_report(ref_results, "报告.html")
export_refined_docx(paragraphs, ref_results, "润色后.docx")
```

## 项目结构

```text
aigc-detector-toolkit/
├── main.py                         # CLI 入口
├── webui.py                        # WebUI API 和静态文件服务
├── train.py                        # 模型训练入口
├── configs/default.yaml            # 默认配置
├── core/                           # 检测、润色、训练、评测核心逻辑
├── extractors/                     # docx/md/txt 段落提取
├── reporters/                      # TXT/JSON/HTML 报告生成器
├── webui_src/                      # React + Vite 前端源码
├── webui/                          # 前端生产构建产物
├── models/                         # 自带或训练得到的模型文件
├── scripts/                        # 数据生成脚本
├── tests/                          # 单元测试
└── skills/aigc-detect/             # Agent Skill 定义
```

## 结果解读

| AIGC 分数 | 含义 | 建议 |
| --------- | ---- | ---- |
| 0-20% | 极低风险 | 通常无需处理 |
| 20-40% | 低风险 | 可抽查 |
| 40-50% | 接近阈值 | 建议复核或轻度润色 |
| 50-70% | 偏高风险 | 建议重点润色 |
| 70-100% | 高风险 | 建议重写或多轮润色 |

检测结果是辅助审阅信号，不应作为唯一判断依据。建议结合文本来源、写作过程、引用和上下文进行人工复核。

## Agent Skill

项目内置 `aigc-detect` Skill，可用于 Claude Code、Cursor、Codex 等 Agent 工作流。

```bash
mkdir -p .claude/skills/aigc-detect
cp skills/aigc-detect/SKILL.md .claude/skills/aigc-detect/
```

示例自然语言指令：

```text
帮我检测这篇论文是否有 AI 生成内容
降低这篇论文的 AIGC 率
训练一个新的 FengCi0 模型
生成 200 条 AI 训练数据
```

详见 [skills/aigc-detect/INSTALL.md](skills/aigc-detect/INSTALL.md)。

## License

MIT License
