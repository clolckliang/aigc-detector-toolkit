# aigc-detect Skill — 安装与集成指南

## 快速开始

```bash
cd /home/shaowenliang/Desktop/aigc-detector-toolkit
uv sync          # 安装基础依赖（FengCi0 + OpenAI API 引擎可用）
uv sync --extra all  # 安装全部可选引擎（HC3 + Local Logprob）
```

## Claude Code（推荐）

### 项目级安装

```bash
mkdir -p .claude/skills/aigc-detect
cp skills/aigc-detect/SKILL.md .claude/skills/aigc-detect/
```

### 全局安装

```bash
mkdir -p ~/.claude/skills/aigc-detect
cp skills/aigc-detect/SKILL.md ~/.claude/skills/aigc-detect/
```

触发方式（自然语言）：
- "帮我检测这篇论文是否有 AI 生成内容"
- "训练一个新的 FengCi0 模型"
- "用 refine 降低这篇论文的 AIGC 率"

## Cursor

```bash
mkdir -p .cursor/rules
cp skills/aigc-detect/SKILL.md .cursor/rules/aigc-detect.md
```

## Hermes Agent

```bash
mkdir -p ~/.hermes/skills/aigc-detect
cp skills/aigc-detect/SKILL.md ~/.hermes/skills/aigc-detect/
```

## Codex (OpenAI)

```bash
mkdir -p .codex
cp skills/aigc-detect/SKILL.md .codex/instructions.md
```

## 验证安装

```bash
uv run python main.py status
```

如果返回引擎状态信息，说明 Skill 已正确加载。
