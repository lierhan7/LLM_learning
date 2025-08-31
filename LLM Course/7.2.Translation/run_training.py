"""
英中翻译训练命令行接口
提供完整的翻译模型训练、评估和推理功能
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import torch

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from config import (
    load_config, save_config, get_default_config,
    quick_test_config, production_config, research_config
)
from trainer import create_translation_trainer


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log', encoding='utf-8')
        ]
    )


def train_command(args):
    """训练命令"""
    logger = logging.getLogger(__name__)
    logger.info("开始翻译模型训练...")
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
        logger.info(f"已加载配置文件: {args.config}")
    else:
        # 使用预设配置
        if args.preset == "quick_test":
            config = quick_test_config()
            logger.info("使用快速测试配置")
        elif args.preset == "production":
            config = production_config()
            logger.info("使用生产环境配置")
        elif args.preset == "research":
            config = research_config()
            logger.info("使用研究配置")
        else:
            config = get_default_config()
            logger.info("使用默认配置")
    
    # 覆盖配置参数
    if args.model_name:
        config.model.model_name = args.model_name
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.max_samples:
        config.data.max_train_samples = args.max_samples
        config.data.max_eval_samples = min(args.max_samples // 10, 1000)
    
    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # 保存配置
    config_path = Path(config.training.output_dir) / "config.yaml"
    save_config(config, config_path)
    logger.info(f"配置已保存至: {config_path}")
    
    # 创建训练器
    trainer = create_translation_trainer(config)
    
    try:
        # 准备数据
        trainer.prepare_data()
        
        # 开始训练
        train_result = trainer.train()
        
        # 自动评估
        if trainer.eval_dataset:
            logger.info("开始验证集评估...")
            eval_results = trainer.evaluate("eval")
            
            # 打印关键指标
            if 'basic_metrics' in eval_results:
                metrics = eval_results['basic_metrics']
                logger.info(f"验证集BLEU分数: {metrics.get('eval_bleu', 'N/A'):.4f}")
                logger.info(f"验证集损失: {metrics.get('eval_loss', 'N/A'):.4f}")
        
        # 生成样本翻译
        logger.info("生成样本翻译...")
        samples = trainer.generate_sample_translations(num_samples=3)
        
        logger.info("\n=== 样本翻译结果 ===")
        for i, sample in enumerate(samples, 1):
            logger.info(f"\n样本 {i}:")
            logger.info(f"源文: {sample['source']}")
            logger.info(f"参考: {sample['reference']}")
            logger.info(f"翻译: {sample['generated']}")
        
        logger.info("训练完成!")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise


def evaluate_command(args):
    """评估命令"""
    logger = logging.getLogger(__name__)
    logger.info("开始模型评估...")
    
    # 加载配置
    config_path = Path(args.model_path) / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        logger.info(f"已加载模型配置: {config_path}")
    else:
        logger.warning("未找到模型配置文件，使用默认配置")
        config = get_default_config()
        config.training.output_dir = args.model_path
    
    # 创建训练器
    trainer = create_translation_trainer(config)
    
    # 加载模型
    trainer.load_model(args.model_path)
    
    # 加载数据（仅用于评估）
    trainer.prepare_data()
    
    # 进行评估
    dataset_name = args.dataset if args.dataset else "eval"
    results = trainer.evaluate(dataset_name)
    
    # 打印结果
    if 'basic_metrics' in results:
        metrics = results['basic_metrics']
        logger.info(f"\n=== 评估结果 ({dataset_name}) ===")
        logger.info(f"BLEU分数: {metrics.get('eval_bleu', 'N/A'):.4f}")
        logger.info(f"ROUGE-L: {metrics.get('eval_rouge_l', 'N/A'):.4f}")
        logger.info(f"损失: {metrics.get('eval_loss', 'N/A'):.4f}")
    
    if 'detailed_analysis' in results:
        analysis = results['detailed_analysis']
        logger.info(f"\n=== 详细分析 ===")
        logger.info(f"平均翻译长度: {analysis.get('avg_translation_length', 'N/A'):.2f}")
        logger.info(f"平均参考长度: {analysis.get('avg_reference_length', 'N/A'):.2f}")
        
        if 'quality_distribution' in analysis:
            quality = analysis['quality_distribution']
            logger.info(f"翻译质量分布:")
            for level, count in quality.items():
                logger.info(f"  {level}: {count}")
    
    logger.info("评估完成!")


def translate_command(args):
    """翻译命令"""
    logger = logging.getLogger(__name__)
    logger.info("开始文本翻译...")
    
    # 使用默认配置
    config = get_default_config()
    config.training.output_dir = args.model_path
    
    # 创建训练器
    trainer = create_translation_trainer(config)
    
    # 加载模型
    trainer.load_model(args.model_path)
    
    # 准备翻译文本
    if args.text:
        texts = [args.text]
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        logger.error("请提供要翻译的文本或输入文件")
        return
    
    # 进行翻译
    logger.info(f"翻译 {len(texts)} 个文本...")
    translations = []
    
    for text in texts:
        # 预处理输入
        task_prefix = "translate English to Chinese: "
        input_text = task_prefix + text
        
        # Tokenize
        inputs = trainer.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=config.data.max_source_length,
            truncation=True,
            padding=True
        ).to(trainer.device)
        
        # 生成翻译
        with torch.no_grad():
            generated_tokens = trainer.model.generate(
                **inputs,
                max_length=config.model.max_target_length,
                num_beams=config.model.num_beams,
                length_penalty=config.model.length_penalty,
                early_stopping=config.model.early_stopping,
                no_repeat_ngram_size=config.model.no_repeat_ngram_size,
                pad_token_id=trainer.tokenizer.pad_token_id,
                eos_token_id=trainer.tokenizer.eos_token_id
            )
        
        # 解码
        translation = trainer.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )
        translations.append(translation)
    
    # 输出结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for source, target in zip(texts, translations):
                f.write(f"源文: {source}\n")
                f.write(f"译文: {target}\n")
                f.write("-" * 50 + "\n")
        logger.info(f"翻译结果已保存至: {args.output_file}")
    else:
        logger.info("\n=== 翻译结果 ===")
        for source, target in zip(texts, translations):
            logger.info(f"源文: {source}")
            logger.info(f"译文: {target}")
            logger.info("-" * 30)


def interactive_command(args):
    """交互式翻译"""
    logger = logging.getLogger(__name__)
    logger.info("启动交互式翻译模式...")
    
    # 使用默认配置
    config = get_default_config()
    config.training.output_dir = args.model_path
    
    # 创建训练器
    trainer = create_translation_trainer(config)
    
    # 加载模型
    trainer.load_model(args.model_path)
    
    logger.info("模型加载完成！输入英文文本进行翻译，输入 'quit' 退出")
    logger.info("-" * 50)
    
    while True:
        try:
            text = input("\n请输入英文文本: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                logger.info("退出交互模式")
                break
            
            if not text:
                continue
            
            # 翻译
            task_prefix = "translate English to Chinese: "
            input_text = task_prefix + text
            
            inputs = trainer.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=config.data.max_source_length,
                truncation=True,
                padding=True
            ).to(trainer.device)
            
            with torch.no_grad():
                generated_tokens = trainer.model.generate(
                    **inputs,
                    max_length=config.model.max_target_length,
                    num_beams=config.model.num_beams,
                    length_penalty=config.model.length_penalty,
                    early_stopping=config.model.early_stopping,
                    pad_token_id=trainer.tokenizer.pad_token_id,
                    eos_token_id=trainer.tokenizer.eos_token_id
                )
            
            translation = trainer.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )
            
            print(f"翻译结果: {translation}")
            
        except KeyboardInterrupt:
            logger.info("\n退出交互模式")
            break
        except Exception as e:
            logger.error(f"翻译出错: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="英中翻译模型训练工具")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练翻译模型")
    train_parser.add_argument("--config", help="配置文件路径")
    train_parser.add_argument("--preset", choices=["quick_test", "production", "research"],
                             default="quick_test", help="预设配置")
    train_parser.add_argument("--model-name", help="模型名称")
    train_parser.add_argument("--output-dir", help="输出目录")
    train_parser.add_argument("--epochs", type=int, help="训练轮数")
    train_parser.add_argument("--batch-size", type=int, help="批次大小")
    train_parser.add_argument("--learning-rate", type=float, help="学习率")
    train_parser.add_argument("--max-samples", type=int, help="最大样本数")
    
    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("model_path", help="模型路径")
    eval_parser.add_argument("--dataset", choices=["eval", "test"], help="评估数据集")
    
    # 翻译命令
    translate_parser = subparsers.add_parser("translate", help="翻译文本")
    translate_parser.add_argument("model_path", help="模型路径")
    translate_parser.add_argument("--text", help="要翻译的文本")
    translate_parser.add_argument("--input-file", help="输入文件路径")
    translate_parser.add_argument("--output-file", help="输出文件路径")
    
    # 交互式翻译
    interactive_parser = subparsers.add_parser("interactive", help="交互式翻译")
    interactive_parser.add_argument("model_path", help="模型路径")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "translate":
        translate_command(args)
    elif args.command == "interactive":
        interactive_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
