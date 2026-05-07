"""Generate synthetic training data using an OpenAI-compatible API.

Usage:
    python scripts/generate_sample_data.py --count 200
    python scripts/generate_sample_data.py --count 100 --output-dir data/train
    python scripts/generate_sample_data.py --count 50 --topics "科技,教育,历史"
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TOPICS = [
    "科技发展", "教育改革", "环境保护", "健康生活", "历史文化",
    "经济趋势", "社会现象", "人工智能", "传统文化", "城市规划",
    "食品安全", "心理健康", "旅游文化", "体育运动", "文学艺术",
]

PROMPT_TEMPLATES = [
    "请写一段关于「{topic}」的中文文章，约200-400字，要求语言自然流畅，有个人见解。",
    "请以「{topic}」为主题，写一篇简短的中文评论，约200-400字，观点明确。",
    "请写一段关于「{topic}」的中文分析，约200-400字，要有逻辑性和深度。",
    "请以「{topic}」为话题，写一段中文随笔，约200-400字，风格自然。",
]


def load_config():
    """Load API config from environment or configs/default.yaml."""
    api_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")

    if not (api_base and api_key and model):
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            api_cfg = cfg.get("openai_api", {})
            api_base = api_base or api_cfg.get("api_base", "")
            api_key = api_key or api_cfg.get("api_key", "")
            model = model or api_cfg.get("model", "")

    if not api_key:
        logger.error("未找到 API key, 请设置 OPENAI_API_KEY 环境变量或在 configs/default.yaml 中配置")
        sys.exit(1)

    return api_base, api_key, model


def generate_one(client, model: str, prompt: str, max_retries: int = 3) -> str:
    """Generate a single text using the API."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.9,
            )
            text = resp.choices[0].message.content.strip()
            if len(text) >= 50:
                return text
        except Exception as e:
            logger.warning("API 调用失败 (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return ""


def main():
    parser = argparse.ArgumentParser(description="使用 LLM API 生成 AI 训练数据")
    parser.add_argument("--count", type=int, default=200,
                        help="生成样本数量 (默认: 200)")
    parser.add_argument("--output-dir", default="data/train",
                        help="输出目录 (默认: data/train)")
    parser.add_argument("--topics", default=None,
                        help="逗号分隔的主题列表")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="API 调用间隔秒数 (默认: 0.5)")
    args = parser.parse_args()

    topics = args.topics.split(",") if args.topics else DEFAULT_TOPICS

    # Setup output
    ai_dir = Path(args.output_dir) / "ai"
    human_dir = Path(args.output_dir) / "human"
    ai_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    # Load API
    api_base, api_key, model = load_config()
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key=api_key)
    logger.info("API: %s  model: %s", api_base, model)

    # Generate
    import random
    rng = random.Random(42)
    success = 0

    for i in range(args.count):
        topic = rng.choice(topics)
        template = rng.choice(PROMPT_TEMPLATES)
        prompt = template.format(topic=topic)

        text = generate_one(client, model, prompt)
        if text:
            out_path = ai_dir / f"ai_{i:04d}.txt"
            out_path.write_text(text, encoding="utf-8")
            success += 1
            if success % 20 == 0:
                logger.info("已生成 %d/%d 条", success, args.count)

        if args.delay > 0:
            time.sleep(args.delay)

    logger.info("AI 样本生成完成: %d/%d 条 -> %s", success, args.count, ai_dir)

    # Create placeholder for human data
    readme = human_dir / "README.txt"
    readme.write_text(
        "请将人工编写的中文文本放入此目录，每篇一个 .txt 文件。\n"
        "建议来源：个人博客、新闻稿件、学术论文、学生作文等。\n"
        f"当前 AI 样本数量: {success} 条，建议收集至少同等数量的人工文本。\n",
        encoding="utf-8",
    )
    logger.info("人工文本目录: %s (请手动添加人工编写的文本)", human_dir)


if __name__ == "__main__":
    main()
