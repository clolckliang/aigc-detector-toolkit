"""CLI entry point for training AIGC detection models."""
import argparse
import logging
import sys

from core.trainer import FengCiTrainer, HC3Trainer, load_text_dir, load_labeled_jsonl
from core.progress import ProgressManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="AIGC 检测模型训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练 FengCi0 随机森林模型
  python train.py --engine fengci --data-dir data/train/

  # 训练 HC3 CNN 模型
  python train.py --engine hc3 --data-dir data/train/

  # 使用已有评测集作为验证集
  python train.py --engine fengci --data-dir data/train/ --eval-data examples/eval_sample.jsonl
        """,
    )
    parser.add_argument("--engine", choices=["fengci", "hc3"], required=True,
                        help="训练引擎: fengci 或 hc3")
    parser.add_argument("--data-dir", required=True,
                        help="训练数据目录, 需包含 ai/ 和 human/ 子目录")
    parser.add_argument("--eval-data", default=None,
                        help="验证集路径 (.jsonl 或 .csv), 不指定则自动从训练集划分")
    parser.add_argument("--output-dir", default=None,
                        help="模型输出目录 (默认: models/<engine>/)")
    parser.add_argument("--no-augment", action="store_true",
                        help="禁用数据增强")

    # HC3 specific
    parser.add_argument("--epochs", type=int, default=30, help="HC3 训练轮数 (默认: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="HC3 批大小 (默认: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="HC3 学习率 (默认: 1e-3)")
    parser.add_argument("--patience", type=int, default=5, help="HC3 早停耐心 (默认: 5)")
    parser.add_argument("--embedding-model", default="moka-ai/m3e-base",
                        help="HC3 embedding 模型 (默认: moka-ai/m3e-base)")

    args = parser.parse_args()

    with ProgressManager(enabled=True) as progress:
        load_task = progress.add_task("加载训练数据", total=1, unit="项")
        logger.info("加载训练数据: %s", args.data_dir)
        texts, labels = load_text_dir(args.data_dir)
        progress.advance(load_task, 1, status=f"{len(texts)} 条")

        eval_texts, eval_labels = None, None
        if args.eval_data:
            eval_task = progress.add_task("加载验证集", total=1, unit="项")
            logger.info("加载验证集: %s", args.eval_data)
            eval_texts, eval_labels = load_labeled_jsonl(args.eval_data)
            progress.advance(eval_task, 1, status=f"{len(eval_texts)} 条")

    # Train
    if args.engine == "fengci":
        output_dir = args.output_dir or "models/fengci"
        trainer = FengCiTrainer(output_dir=output_dir)
        with ProgressManager(enabled=True) as progress:
            train_task = progress.add_task("训练 FengCi0", total=1, unit="项")
            result = trainer.train(
                texts, labels,
                eval_texts=eval_texts, eval_labels=eval_labels,
                augment=not args.no_augment,
            )
            progress.advance(train_task, 1, status="完成")
        logger.info("=" * 50)
        logger.info("FengCi0 训练完成!")
        logger.info("最佳模型: %s", result["best_model"])
        logger.info("推荐阈值: %.4f", result["threshold"])
        m = result["metrics"]
        logger.info("验证指标: acc=%.4f f1=%.4f auc=%s",
                     m["accuracy"], m["f1"],
                     f"{m['roc_auc']:.4f}" if m["roc_auc"] else "N/A")

    elif args.engine == "hc3":
        output_dir = args.output_dir or "models/hc3"
        trainer = HC3Trainer(output_dir=output_dir,
                             embedding_model=args.embedding_model)
        with ProgressManager(enabled=True) as progress:
            train_task = progress.add_task("训练 HC3", total=args.epochs, unit="轮")
            result = trainer.train(
                texts, labels,
                eval_texts=eval_texts, eval_labels=eval_labels,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
            )
            progress.update(train_task, completed=args.epochs, status="完成")
        logger.info("=" * 50)
        logger.info("HC3 CNN 训练完成!")
        m = result["metrics"]
        logger.info("验证指标: acc=%.4f f1=%.4f auc=%s",
                     m["accuracy"], m["f1"],
                     f"{m['roc_auc']:.4f}" if m["auc"] else "N/A")
        logger.info("模型路径: %s", result["model_path"])


if __name__ == "__main__":
    main()
