# AIGC Detector Toolkit

中文 AI 生成文本 (AIGC) 检测工具，四引擎融合，**自包含、无需克隆外部仓库**。

## 特性

- **四引擎融合检测**
  - 🌲 **FengCi0** — 12维文本特征 + 随机森林（自带预训练模型，毫秒级）
  - 🧠 **HC3+m3e** — m3e-base 编码 + OptimizedCNN 分类器（自带模型权重）
  - 🔮 **MiMo API** — 利用 LLM 的 logprobs 计算困惑度（不需本地大模型）
  - 🔭 **Binoculars** — 双模型 perplexity/entropy 比值（ICLR 2024）
- **多格式支持**：`.docx` / `.md` / `.txt`
- **自包含**：预训练模型已内嵌，`git clone` 即可使用
- **可扩展**：支持配置权重、阈值、API key

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 快速测试（使用内置样本，不需要文件）
python main.py test

# 3. 检测单个文件
python main.py detect 论文.docx

# 4. 仅用 FengCi0 引擎（最快）
python main.py detect 论文.docx --engine fengci

# 5. 批量检测目录
python main.py batch ./docs/

# 6. 查看引擎状态
python main.py status
```

### 配置 MiMo API（可选，提升准确率）

```bash
# 方式一：环境变量
export MIMO_API_KEY="your-key-here"
export MIMO_API_BASE="https://your-api-endpoint/v1"

# 方式二：编辑 configs/default.yaml
```

## 引擎说明

| 引擎 | 速度 | 需要本地模型 | 需要 API | 原理 |
|------|------|:----------:|:--------:|------|
| FengCi0 | ⚡ 毫秒 | ✅ 1MB | ❌ | 12维文本特征 + 随机森林 |
| HC3+m3e | 🚀 秒级 | ✅ 25MB | ❌ | m3e 文本编码 + CNN 分类 |
| MiMo API | 🐢 ~0.5s/段 | ❌ | ✅ | LLM logprobs 困惑度 |
| Binoculars | 🐢 ~0.5s/段 | ❌ | ✅ | 双模型 ppl/x_ppl 比值 |

默认权重：FengCi0 30% + HC3 20% + MiMo 25% + Binoculars 25%
不配置 API key 时自动退化为 FengCi0 + HC3 双引擎。

## 输出示例

```
============================================================
              AIGC 查重检测报告
============================================================
检测引擎:   ensemble
  FengCi0:  ✓ (权重 37%)
  HC3+m3e:  ✓ (权重 33%)
  MiMo API: ✗ (权重 0%)
  Binoculars: ✗ (权重 0%)
------------------------------------------------------------
总段落数:     454
判定为人工撰写: 444 (97.8%)
判定为AI生成:    10 ( 2.2%)
平均AIGC分数:   24.7%
============================================================
```

## 项目结构

```
aigc-detector-toolkit/
├── main.py                          # CLI 主入口
├── requirements.txt                 # Python 依赖
├── README.md
├── configs/
│   └── default.yaml                 # 默认配置
├── core/
│   ├── engine.py                    # 四引擎融合检测器
│   ├── fengci_adapter.py            # FengCi0 适配器（自包含特征提取器）
│   ├── hc3_adapter.py               # HC3+m3e 适配器（自包含 CNN 模型）
│   ├── mimo_api_adapter.py          # MiMo API logprobs 适配器
│   └── binoculars_adapter.py        # Binoculars API 适配器
├── extractors/
│   ├── docx_extractor.py            # DOCX 段落提取
│   ├── md_extractor.py              # Markdown 段落提取
│   └── text_extractor.py            # 纯文本提取
├── reporters/
│   ├── console_reporter.py          # 终端报告
│   ├── text_reporter.py             # 文本文件报告
│   └── json_reporter.py             # JSON 数据报告
└── models/
    ├── fengci/
    │   └── aigc_detector_model.joblib  # FengCi0 预训练模型（1MB）
    └── hc3/
        └── OptimizedCNN_aigc_detector.pth  # HC3 CNN 模型（25MB）
```

## Python API 调用

```python
from core.engine import DetectionEngine

engine = DetectionEngine(mode="fengci")  # 或 "ensemble"

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
|-----------|------|
| 0–20% | 极低 AI 概率，人工风格明确 |
| 20–40% | 低 AI 概率，偏人工 |
| 40–50% | 接近阈值，需结合上下文判断 |
| 50–70% | 偏高 AI 概率，建议改写 |
| 70–100% | 高 AI 概率，强烈建议改写 |

## 许可证

MIT License
