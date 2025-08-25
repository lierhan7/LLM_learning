"""
训练运行脚本
提供命令行接口来运行训练
"""

import argparse
import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from trainer import TextClassificationTrainer
from config import TrainingConfig, ConfigTemplates
from utils import set_seed, print_gpu_utilization


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BERT文本分类微调训练")
    
    # 配置相关
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--template", type=str, choices=["quick_test", "production", "large_model"],
                       help="使用预定义配置模板")
    
    # 模型相关
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="预训练模型名称")
    parser.add_argument("--num_labels", type=int, default=2,
                       help="分类标签数量")
    
    # 训练相关
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=0,
                       help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    
    # 数据相关
    parser.add_argument("--dataset_name", type=str, default="glue",
                       help="数据集名称")
    parser.add_argument("--dataset_config", type=str, default="mrpc",
                       help="数据集配置")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    
    # 输出相关
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="输出目录")
    parser.add_argument("--save_model", action="store_true", default=True,
                       help="是否保存模型")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="评估间隔步数")
    parser.add_argument("--fp16", action="store_true",
                       help="启用混合精度训练")
    
    return parser.parse_args()


def create_config_from_args(args) -> TrainingConfig:
    """从命令行参数创建配置"""
    # 如果指定了配置文件，从文件加载
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        print(f"从配置文件加载: {args.config}")
        return config
    
    # 如果指定了模板，使用预定义模板
    if args.template:
        if args.template == "quick_test":
            config = ConfigTemplates.quick_test()
        elif args.template == "production":
            config = ConfigTemplates.production()
        elif args.template == "large_model":
            config = ConfigTemplates.large_model()
        print(f"使用预定义模板: {args.template}")
        return config
    
    # 否则从命令行参数创建配置
    config = TrainingConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        output_dir=args.output_dir,
        save_model=args.save_model,
        seed=args.seed,
        eval_steps=args.eval_steps,
        fp16=args.fp16
    )
    
    print("使用命令行参数创建配置")
    return config


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = create_config_from_args(args)
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 打印系统信息
    print("=== 系统信息 ===")
    print_gpu_utilization()
    print()
    
    # 打印配置信息
    print("=== 训练配置 ===")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    print()
    
    # 保存配置
    os.makedirs(config.output_dir, exist_ok=True)
    config_save_path = os.path.join(config.output_dir, "training_config.yaml")
    config.save_config(config_save_path)
    print(f"训练配置已保存到: {config_save_path}")
    print()
    
    try:
        # 创建训练器并开始训练
        trainer = TextClassificationTrainer(config)
        final_metrics = trainer.train()
        
        print("=== 训练完成 ===")
        print("最终指标:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
