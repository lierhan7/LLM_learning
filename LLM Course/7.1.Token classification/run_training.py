"""
Token分类训练命令行接口
提供完整的命令行参数支持和模板配置
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Optional

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from config import ExperimentConfig, ConfigTemplates
from trainer import TokenClassificationTrainer

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Token分类模型训练框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用预定义模板快速开始
  python run_training.py --template quick_test
  
  # 使用自定义配置文件
  python run_training.py --config config.yaml
  
  # 使用命令行参数自定义训练
  python run_training.py --model_name bert-base-cased --num_epochs 5 --learning_rate 2e-5
  
  # 训练生产环境模型
  python run_training.py --template production --output_dir ./production_model
        """
    )
    
    # 基础参数
    parser.add_argument("--config", type=str, help="配置文件路径 (YAML格式)")
    parser.add_argument("--template", type=str, 
                       choices=["quick_test", "production", "research"],
                       help="使用预定义配置模板")
    parser.add_argument("--experiment_name", type=str, 
                       help="实验名称")
    parser.add_argument("--description", type=str, 
                       help="实验描述")
    
    # 模型相关参数
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument("--model_name", type=str, 
                           help="预训练模型名称或路径")
    model_group.add_argument("--cache_dir", type=str, 
                           help="模型缓存目录")
    model_group.add_argument("--trust_remote_code", action="store_true",
                           help="是否信任远程代码")
    
    # 数据相关参数
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument("--dataset_name", type=str,
                          help="数据集名称")
    data_group.add_argument("--dataset_config", type=str,
                          help="数据集配置")
    data_group.add_argument("--train_file", type=str,
                          help="训练文件路径")
    data_group.add_argument("--validation_file", type=str,
                          help="验证文件路径")
    data_group.add_argument("--test_file", type=str,
                          help="测试文件路径")
    data_group.add_argument("--max_length", type=int,
                          help="最大序列长度")
    data_group.add_argument("--label_all_tokens", action="store_true",
                          help="是否标记所有子词token")
    
    # 训练相关参数
    training_group = parser.add_argument_group("训练参数")
    training_group.add_argument("--output_dir", type=str,
                              help="输出目录")
    training_group.add_argument("--num_epochs", type=int, dest="num_train_epochs",
                              help="训练轮数")
    training_group.add_argument("--batch_size", type=int, dest="per_device_train_batch_size",
                              help="训练批次大小")
    training_group.add_argument("--eval_batch_size", type=int, dest="per_device_eval_batch_size",
                              help="评估批次大小")
    training_group.add_argument("--learning_rate", type=float,
                              help="学习率")
    training_group.add_argument("--weight_decay", type=float,
                              help="权重衰减")
    training_group.add_argument("--warmup_ratio", type=float,
                              help="预热比例")
    training_group.add_argument("--warmup_steps", type=int,
                              help="预热步数")
    training_group.add_argument("--lr_scheduler_type", type=str,
                              help="学习率调度器类型")
    
    # 评估和保存参数
    eval_group = parser.add_argument_group("评估和保存参数")
    eval_group.add_argument("--evaluation_strategy", type=str,
                          choices=["no", "steps", "epoch"],
                          help="评估策略")
    eval_group.add_argument("--eval_steps", type=int,
                          help="评估间隔步数")
    eval_group.add_argument("--save_strategy", type=str,
                          choices=["no", "steps", "epoch"],
                          help="保存策略")
    eval_group.add_argument("--save_steps", type=int,
                          help="保存间隔步数")
    eval_group.add_argument("--save_total_limit", type=int,
                          help="最大保存模型数")
    eval_group.add_argument("--load_best_model_at_end", action="store_true",
                          help="训练结束时加载最佳模型")
    eval_group.add_argument("--metric_for_best_model", type=str,
                          help="最佳模型评估指标")
    
    # 优化参数
    optim_group = parser.add_argument_group("优化参数")
    optim_group.add_argument("--fp16", action="store_true",
                           help="启用FP16混合精度训练")
    optim_group.add_argument("--bf16", action="store_true",
                           help="启用BF16混合精度训练")
    optim_group.add_argument("--gradient_accumulation_steps", type=int,
                           help="梯度累积步数")
    optim_group.add_argument("--max_grad_norm", type=float,
                           help="梯度裁剪最大范数")
    optim_group.add_argument("--early_stopping_patience", type=int,
                           help="早停耐心值")
    
    # 日志和报告参数
    log_group = parser.add_argument_group("日志和报告参数")
    log_group.add_argument("--logging_dir", type=str,
                         help="日志目录")
    log_group.add_argument("--logging_steps", type=int,
                         help="日志记录间隔步数")
    log_group.add_argument("--report_to", type=str, nargs="+",
                         choices=["tensorboard", "wandb", "comet_ml", "neptune"],
                         help="报告目标")
    
    # 其他参数
    other_group = parser.add_argument_group("其他参数")
    other_group.add_argument("--seed", type=int,
                           help="随机种子")
    other_group.add_argument("--local_rank", type=int, default=-1,
                           help="本地进程排名")
    other_group.add_argument("--cuda_visible_devices", type=str,
                           help="可见CUDA设备")
    other_group.add_argument("--do_train", action="store_true", default=True,
                           help="是否执行训练")
    other_group.add_argument("--do_eval", action="store_true", default=True,
                           help="是否执行评估")
    other_group.add_argument("--do_predict", action="store_true",
                           help="是否执行预测")
    
    # Hub相关参数
    hub_group = parser.add_argument_group("Hugging Face Hub参数")
    hub_group.add_argument("--push_to_hub", action="store_true",
                         help="是否推送到Hub")
    hub_group.add_argument("--hub_model_id", type=str,
                         help="Hub模型ID")
    hub_group.add_argument("--hub_strategy", type=str,
                         choices=["end", "every_save", "checkpoint", "all_checkpoints"],
                         help="Hub推送策略")
    hub_group.add_argument("--hub_token", type=str,
                         help="Hub访问token")
    hub_group.add_argument("--hub_private_repo", action="store_true",
                         help="是否为私有仓库")
    
    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """从命令行参数创建配置"""
    
    # 如果指定了配置文件，从文件加载
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"配置文件不存在: {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
        logger.info(f"从配置文件加载: {args.config}")
    
    # 如果指定了模板，使用预定义模板
    elif args.template:
        if args.template == "quick_test":
            config = ConfigTemplates.quick_test()
        elif args.template == "production":
            config = ConfigTemplates.production()
        elif args.template == "research":
            config = ConfigTemplates.research()
        else:
            raise ValueError(f"未知模板: {args.template}")
        logger.info(f"使用预定义模板: {args.template}")
    
    # 否则使用默认配置
    else:
        config = ExperimentConfig()
        logger.info("使用默认配置")
    
    # 更新配置
    config.update_from_args(args)
    
    return config


def setup_logging(config: ExperimentConfig):
    """设置日志"""
    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # 配置日志
    log_file = os.path.join(config.training.output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_config_summary(config: ExperimentConfig):
    """打印配置摘要"""
    print("\n" + "="*60)
    print("实验配置摘要")
    print("="*60)
    print(f"实验名称: {config.experiment_name}")
    print(f"描述: {config.description}")
    print(f"输出目录: {config.training.output_dir}")
    print(f"模型: {config.model.model_name}")
    print(f"数据集: {config.data.dataset_name}")
    print(f"训练轮数: {config.training.num_train_epochs}")
    print(f"批次大小: {config.training.per_device_train_batch_size}")
    print(f"学习率: {config.training.learning_rate}")
    print(f"最大长度: {config.data.max_length}")
    print(f"随机种子: {config.training.seed}")
    print("="*60)


def validate_config(config: ExperimentConfig):
    """验证配置"""
    errors = []
    
    # 检查必要参数
    if not config.data.dataset_name and not config.data.train_file:
        errors.append("必须提供dataset_name或train_file")
    
    if config.training.num_train_epochs <= 0:
        errors.append("num_train_epochs必须大于0")
    
    if config.training.learning_rate <= 0:
        errors.append("learning_rate必须大于0")
    
    if config.training.per_device_train_batch_size <= 0:
        errors.append("per_device_train_batch_size必须大于0")
    
    # 检查文件路径
    if config.data.train_file and not os.path.exists(config.data.train_file):
        errors.append(f"训练文件不存在: {config.data.train_file}")
    
    if config.data.validation_file and not os.path.exists(config.data.validation_file):
        errors.append(f"验证文件不存在: {config.data.validation_file}")
    
    if config.data.test_file and not os.path.exists(config.data.test_file):
        errors.append(f"测试文件不存在: {config.data.test_file}")
    
    if errors:
        logger.error("配置验证失败:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 创建配置
        config = create_config_from_args(args)
        
        # 验证配置
        validate_config(config)
        
        # 设置日志
        setup_logging(config)
        
        # 打印配置摘要
        print_config_summary(config)
        
        # 创建训练器
        trainer = TokenClassificationTrainer(config)
        
        # 执行训练
        if args.do_train:
            logger.info("开始训练...")
            train_result = trainer.train()
            logger.info("训练完成")
        
        # 执行评估
        if args.do_eval:
            logger.info("开始评估...")
            if trainer.eval_dataset:
                eval_results = trainer.evaluate("eval")
                logger.info("验证集评估完成")
            
            if trainer.test_dataset:
                test_results = trainer.evaluate("test")
                logger.info("测试集评估完成")
        
        logger.info("实验完成！")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
