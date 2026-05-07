---
name: aigc-detect
description: >
  Use this skill when the user asks to "detect AI text", "check if text is AI generated",
  "run AIGC detection", "analyze document for AI content", "train detection model",
  "generate training data", "reduce AIGC score", "refine paper to lower AI detection",
  "降AIGC率", "降重", "去AI味", or mentions AIGC/AI-generated content detection/refinement.
  This skill provides full access to the AIGC Detector Toolkit: five-engine detection,
  one-click refinement, model training, data generation, and evaluation.
version: 2.0.0
tools: Read, Bash, Edit, Write, Glob, Grep
---

# AIGC Detector Toolkit — Skill Reference

Chinese AI-generated text (AIGC) detection and refinement toolkit with five engines,
model training, data generation, evaluation, and a one-click **降AIGC率** refinement workflow.

## Project Root

All commands assume the working directory is the toolkit root:

```
/home/shaowenliang/Desktop/aigc-detector-toolkit/
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    main.py (CLI)                     │
│  detect | batch | refine | eval | train | status | test │
├─────────────────────────────────────────────────────┤
│                  core/engine.py                      │
│         DetectionEngine (5-engine fusion)            │
│  ┌──────────┬────────┬────────┬─────────┬─────────┐ │
│  │ FengCi0  │  HC3   │ OpenAI │Bino-    │ Local   │ │
│  │ (RF)     │ (CNN)  │ (API)  │culars   │Logprob  │ │
│  │ 12-dim   │ m3e    │ logprobs│(API)   │(opt-in) │ │
│  │ features │ 768-dim│ ppl    │ppl/x_ppl│LogRank  │ │
│  └──────────┴────────┴────────┴─────────┴─────────┘ │
├──────────────────┬──────────────────────────────────┤
│  core/refiner.py │  core/trainer.py                 │
│  降AIGC率工作流   │  FengCiTrainer / HC3Trainer      │
│  3策略:           │  数据增强 / 阈值校准              │
│  refine_cn        │  generate_sample_data.py         │
│  deai_cn          │                                  │
│  deai_en          │                                  │
├──────────────────┴──────────────────────────────────┤
│  extractors/          │  reporters/                  │
│  docx / md / txt      │  console / txt / json / html │
├─────────────────────────────────────────────────────┤
│  models/  (pretrained) │  configs/  │  examples/    │
│  fengci/ (1MB RF)      │  default   │  eval_sample  │
│  hc3/    (25MB CNN)    │  .yaml     │  .jsonl       │
└─────────────────────────────────────────────────────┘
```

## CLI Commands

### detect — 单文件 AIGC 检测

```bash
uv run python main.py detect <file> [--engine ENGINE] [--threshold FLOAT] [-o DIR]
```

- Supported formats: `.docx`, `.md`, `.txt`
- Auto-extracts paragraphs, skips headings/captions/short text
- Generates: `AIGC检测报告_*.txt` + `AIGC检测结果_*.json` + `AIGC可视化报告_*.html`

### batch — 批量检测目录

```bash
uv run python main.py batch <directory> [--engine ENGINE] [--threshold FLOAT] [-o DIR]
```

- Recursively finds all `.docx`/`.md`/`.txt` files
- Generates per-file reports + summary

### refine — 降AIGC率工作流（核心差异化功能）

```bash
uv run python main.py refine <file> [--threshold FLOAT] [--no-recheck] [-o DIR]
```

**Requires `OPENAI_API_KEY`** for LLM-based text refinement.

Workflow:
1. Extract paragraphs from document
2. Run AIGC detection (5-engine fusion) on every paragraph
3. Classify high-risk paragraphs (score > threshold, default 0.4)
4. Auto-select refinement strategy per paragraph:
   - `refine_cn` — Chinese expression polishing (fix grammar, remove colloquialisms)
   - `deai_cn` — Chinese AI-style removal (eliminate AI clichés, add personal touches)
   - `deai_en` — English AI-style removal (natural academic rewriting)
5. Re-detect refined paragraphs (unless `--no-recheck`)
6. Generate: `降AIGC报告_*.txt` + `润色后文本_*.docx` + `润色结果_*.json`

### eval — 评测标注数据集

```bash
uv run python main.py eval <dataset.jsonl> [--engine ENGINE] [--threshold FLOAT] [-o DIR]
```

Dataset format (JSONL, one per line):
```json
{"label": "human", "text": "..."}
{"label": "ai", "text": "..."}
```

Outputs: accuracy, precision, recall, F1, AUC, confusion matrix, threshold scan.

### train — 训练/微调检测模型

```bash
# FengCi0 Random Forest
uv run python train.py --engine fengci --data-dir data/train/

# HC3 CNN
uv run python train.py --engine hc3 --data-dir data/train/ --epochs 30

# With data augmentation
uv run python train.py --engine fengci --data-dir data/train/ --augment

# Generate AI training data via LLM
uv run python scripts/generate_sample_data.py --count 200
```

Training data structure:
```
data/train/
├── ai/          # AI-generated texts, one .txt per sample
│   ├── 0001.txt
│   └── ...
└── human/       # Human-written texts, one .txt per sample
    ├── 0001.txt
    └── ...
```

### status — 引擎状态

```bash
uv run python main.py status
```

### test — 快速自测

```bash
uv run python main.py test
```

## Five Engines — Deep Dive

### 1. FengCi0 (Feature Engineering + Random Forest)

**Core file:** `core/fengci_adapter.py`

**12-dimensional text features** (all self-contained, no external repo needed):

| # | Feature | Description |
|---|---------|-------------|
| 1 | `char_entropy_norm` | Normalized character entropy (diversity of characters) |
| 2 | `avg_sentence_length_norm` | Average sentence length / 60 |
| 3 | `sentence_length_cv_norm` | Coefficient of variation of sentence lengths |
| 4 | `lexical_diversity` | Unique words / total words |
| 5 | `hapax_ratio` | Words appearing exactly once / total words |
| 6 | `repetition_ratio` | 1 - lexical_diversity |
| 7 | `bigram_repetition_ratio` | Repeated bigrams / total bigrams |
| 8 | `function_word_ratio` | Function words (的了是在和...) / total words |
| 9 | `punctuation_ratio` | Punctuation chars / total chars |
| 10 | `long_word_ratio` | Words with len >= 3 / total words |
| 11 | `pos_diversity` | Normalized entropy of POS tag distribution |
| 12 | `noun_verb_balance` | Nouns / (nouns + verbs) |

**Algorithm:** DC resistance — human text has higher entropy, more diverse vocabulary, more varied sentence lengths.

**Model:** sklearn RandomForestClassifier (default), with LogReg and GradientBoosting as candidates. Best selected automatically.

### 2. HC3 + m3e (Deep Learning)

**Core file:** `core/hc3_adapter.py`

- Encoder: `moka-ai/m3e-base` SentenceTransformer (768-dim)
- Classifier: Custom OptimizedCNN (Conv1d → BN → Pool → Conv1d → BN → Pool → FC → FC)
- Training data: HC3-Chinese dataset (Hello-SimpleAI/HC3-Chinese)

### 3. OpenAI-Compatible API (Logprobs)

**Core file:** `core/openai_adapter.py`

Uses any OpenAI-compatible API's `logprobs` feature to compute text perplexity.

**Strategy:** `perplexity` (default) — splits text into prefix/suffix, asks model to continue from prefix, measures surprise of actual suffix.

**Fallback:** If API lacks logprobs, degrades to basic text statistics (sentence CV, avg length).

### 4. Binoculars (ICLR 2024)

**Core file:** `core/binoculars_adapter.py`

Based on paper: "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"

**Formula:** `score = perplexity / cross_perplexity`

- observer mode: "Please reproduce this text verbatim"
- performer mode: "Please rewrite this text in your own words"

Lower ratio → more likely AI-generated.

### 5. Local Logprob (Opt-in)

**Core file:** `core/local_logprob_adapter.py`

Loads a local HuggingFace causal LM and computes LogRank/perplexity directly.

Requires `transformers` and a local model path. **Disabled by default** (weight=0).

## Refinement Strategies (降AIGC率)

### refine_cn — 表达润色

- Target: AIGC-scored paragraphs with moderate AI probability
- Philosophy: **克制修改** — minimal intervention
- Actions: fix grammar, remove colloquialisms, preserve author's style
- Prompt: "如果你原文表达已经清晰、准确且符合学术规范，请务必保留原样"

### deai_cn — 去AI味（中文）

- Target: Paragraphs with 2+ AI cliché markers (综上所述, 值得注意的是, 从本质上讲...)
- Actions:
  - Eliminate template phrases
  - Diversify sentence structures
  - Add personal markers (本文认为, 实验发现, 在调试过程中)
  - Replace vague generalizations with specific technical details

### deai_en — 去AI味（英文）

- Target: English paragraphs
- Actions:
  - Replace overused words (leverage → use, delve into → investigate)
  - Remove mechanical transitions (First and foremost → direct connection)
  - Convert bullet lists to flowing prose
  - Vary sentence length

### Paragraph Classification Logic

```python
def classify_paragraph(text):
    if len(text) < 30:          return 'skip'
    if is_code_or_formula:       return 'skip'
    if is_reference:             return 'skip'
    if is_caption:               return 'skip'
    if is_english:               return 'deai_en'
    if ai_marker_count >= 2:     return 'deai_cn'
    return 'refine_cn'
```

## Configuration Reference

### configs/default.yaml

```yaml
engine:
  default: "ensemble"
  ensemble:
    fengci_weight: 0.30
    hc3_weight: 0.20
    openai_weight: 0.25
    binoculars_weight: 0.25
    local_logprob_weight: 0.0

openai_api:
  api_base: "https://api.openai.com/v1"
  api_key: ""
  model: "gpt-4o-mini"
  strategy: "perplexity"

binoculars:
  api_base: ""    # empty → reuse openai_api
  api_key: ""
  model: ""

local_logprob:
  model_name_or_path: ""  # empty → disabled
  device: "cpu"
  max_length: 512
  stride: 256
  method: "logrank"

extraction:
  min_paragraph_length: 30
  skip_headings: true
  skip_captions: true
  skip_code_blocks: true

threshold:
  aigc_threshold: 0.5

performance:
  api_concurrency: 1

output:
  dir: null    # null → same dir as input file
  json: true
  text: true
  html: true
  console: true
  verbose: false
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI-compatible API key | (empty) |
| `OPENAI_BASE_URL` | API base URL | `https://api.openai.com/v1` |
| `OPENAI_API_BASE` | Alias for OPENAI_BASE_URL | — |
| `OPENAI_MODEL` | Model name | `gpt-4o-mini` |
| `MIMO_API_KEY` | Legacy alias for OPENAI_API_KEY | — |
| `MIMO_API_BASE` | Legacy alias for OPENAI_BASE_URL | — |
| `MIMO_MODEL` | Legacy alias for OPENAI_MODEL | — |

## Python API

### Detection

```python
from core.engine import DetectionEngine

engine = DetectionEngine(mode="ensemble")  # or "fengci", "hc3", "openai", etc.

# Single paragraph
result = engine.detect_single("待检测文本...")
print(result["aigc_score"])     # 0.0 - 1.0
print(result["label"])          # "ai" or "human"
print(result["confidence"])     # 0.0 - 1.0
print(result["engine_results"]) # per-engine breakdown

# Batch
results = engine.detect_batch(["text1", "text2", "text3"])
```

### Refinement

```python
from core.refiner import RefinementEngine, generate_refinement_report, export_refined_docx

refiner = RefinementEngine(
    api_base="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o-mini",
)

# Single paragraph
refined, note = refiner.refine_paragraph("综上所述，...", strategy="deai_cn")

# Batch (with detection results from DetectionEngine)
ref_results = refiner.refine_batch(paragraphs, detection_results, score_threshold=0.4)

# Generate reports
generate_refinement_report(ref_results, "报告.txt")
export_refined_docx(paragraphs, ref_results, "润色后.docx")
```

### Evaluation

```python
from core.evaluation import load_labeled_dataset, summarize_engine_metrics, write_eval_report

dataset = load_labeled_dataset("eval.jsonl")
texts = [d["text"] for d in dataset]
results = engine.detect_batch(texts)
summary = summarize_engine_metrics(dataset, results, threshold=0.5)
write_eval_report(summary, "eval_report.json")
```

### Training

```python
from core.trainer import FengCiTrainer, HC3Trainer

# FengCi0
trainer = FengCiTrainer(output_dir="models/fengci")
metrics = trainer.train(texts, labels, augment=True)
print(metrics["best_model"], metrics["test_metrics"])

# HC3
trainer = HC3Trainer(output_dir="models/hc3")
metrics = trainer.train(texts, labels, epochs=30)
```

## Score Interpretation

| Score | Meaning | Recommended Action |
|-------|---------|-------------------|
| 0-20% | Very low AI probability | No action needed |
| 20-40% | Low AI probability | Optional refinement |
| 40-50% | Borderline | Suggest refinement |
| 50-70% | High AI probability | Strongly suggest refinement |
| 70-100% | Very high AI probability | Must refine |

## Workflow: Improve Detection Accuracy

```
1. Generate AI data
   uv run python scripts/generate_sample_data.py --count 500

2. Collect human texts → data/train/human/

3. Retrain
   uv run python train.py --engine fengci --data-dir data/train/

4. Evaluate
   uv run python main.py eval examples/eval_sample.jsonl

5. Tune threshold in configs/default.yaml
```

## Workflow: Reduce AIGC Score (降AIGC率)

```
1. Detect current state
   uv run python main.py detect 论文.docx

2. Set API key for refinement
   export OPENAI_API_KEY="sk-..."

3. One-click refinement
   uv run python main.py refine 论文.docx --threshold 0.3

4. Check results
   - 降AIGC报告_*.txt — before/after comparison
   - 润色后文本_*.docx — refined document
   - Re-detect refined doc to verify improvement
```

## Extending the Toolkit

### Add a New Detection Engine

1. Create `core/your_engine_adapter.py` with a class implementing:
   ```python
   class YourEngineDetector:
       available: bool
       def detect(self, text: str) -> Dict  # returns {"aigc_score": float, "label": str, ...}
       def detect_batch(self, texts: List[str]) -> List[Dict]
   ```

2. Register in `core/engine.py`:
   - Add import
   - Add to `ENGINE_NAMES`
   - Add constructor parameter
   - Add initialization block

3. Add CLI argument in `main.py`:
   - Add to `choices` list
   - Add to `build_engine()` parameter passing

4. Add config section in `configs/default.yaml`

### Add a New Refinement Strategy

1. Define prompt template in `core/refiner.py`:
   ```python
   PROMPT_YOUR_STRATEGY = """Your prompt here... {paragraph}"""
   ```

2. Add to `classify_paragraph()` logic
3. Add to `refine_paragraph()` strategy dispatch

## Key Files Reference

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point (detect/batch/refine/eval/status/test) |
| `train.py` | Model training entry point |
| `core/engine.py` | Multi-engine fusion detector (orchestrator) |
| `core/fengci_adapter.py` | FengCi0: 12-dim features + RF (fully self-contained) |
| `core/hc3_adapter.py` | HC3: m3e encoding + OptimizedCNN (self-contained) |
| `core/openai_adapter.py` | OpenAI-compatible API logprobs detector |
| `core/binoculars_adapter.py` | Binoculars ppl/x_ppl ratio (API mode) |
| `core/local_logprob_adapter.py` | Local causal LM LogRank/perplexity |
| `core/refiner.py` | Refinement engine + 3 prompt strategies |
| `core/trainer.py` | FengCiTrainer + HC3Trainer + data augmentation |
| `core/evaluation.py` | Metrics, threshold scan, eval report |
| `core/progress.py` | Progress bar utility |
| `extractors/*.py` | Paragraph extraction from docx/md/txt |
| `reporters/*.py` | Console/txt/json/html report generation |
| `configs/default.yaml` | Default configuration |
| `models/fengci/*.joblib` | Pretrained FengCi0 RF model (1MB) |
| `models/hc3/*.pth` | Pretrained HC3 CNN weights (25MB) |
| `examples/eval_sample.jsonl` | Sample evaluation dataset |
| `scripts/generate_sample_data.py` | LLM-based training data generation |

## Known Limitations & Notes

- **HC3 engine** may fail with certain `transformers` versions due to `is_torch_npu_available` import error. FengCi0 + OpenAI API + Binoculars still work.
- **FengCi0** uses `jieba` for Chinese word segmentation. Ensure jieba is installed.
- **Refinement** requires an active OpenAI-compatible API with `OPENAI_API_KEY` set.
- **Local Logprob** engine is opt-in and disabled by default (weight=0). Set `local_logprob_weight > 0` and provide a model path to enable.
- **Binoculars** API mode requires API key. Without it, the engine is skipped.
- **Score interpretation** varies by document type. Academic papers naturally have lower AIGC scores than product descriptions.
