"""
英中翻译框架使用示例
演示如何使用框架进行训练、评估和翻译
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from config import (
    get_default_config, quick_test_config, 
    production_config, research_config
)
from trainer import create_translation_trainer


def example_1_basic_usage():
    """示例1: 基础使用方法"""
    print("=" * 60)
    print("示例1: 基础配置和训练器创建")
    print("=" * 60)
    
    # 创建默认配置
    config = get_default_config()
    print(f"默认模型: {config.model.model_name}")
    print(f"训练轮数: {config.training.num_train_epochs}")
    print(f"批次大小: {config.training.per_device_train_batch_size}")
    
    # 创建训练器
    trainer = create_translation_trainer(config)
    print("✓ 训练器创建成功")
    print()


def example_2_preset_configs():
    """示例2: 预设配置对比"""
    print("=" * 60)
    print("示例2: 不同预设配置对比")
    print("=" * 60)
    
    configs = {
        "快速测试": quick_test_config(),
        "生产环境": production_config(),
        "研究实验": research_config()
    }
    
    for name, config in configs.items():
        print(f"\n{name}配置:")
        print(f"  - 模型: {config.model.model_name}")
        print(f"  - 训练轮数: {config.training.num_train_epochs}")
        print(f"  - 训练样本: {config.data.max_train_samples}")
        print(f"  - 评估样本: {config.data.max_eval_samples}")
        print(f"  - 学习率: {config.training.learning_rate}")
    print()


def example_3_custom_config():
    """示例3: 自定义配置"""
    print("=" * 60)
    print("示例3: 创建自定义配置")
    print("=" * 60)
    
    # 从默认配置开始
    config = get_default_config()
    
    # 自定义设置
    config.experiment_name = "my_custom_translation"
    config.description = "自定义英中翻译实验"
    
    # 模型设置
    config.model.model_name = "Helsinki-NLP/opus-mt-en-zh"
    
    # 训练设置
    config.training.num_train_epochs = 3
    config.training.per_device_train_batch_size = 4
    config.training.learning_rate = 2e-4
    config.training.warmup_ratio = 0.1
    
    # 数据设置
    config.data.max_train_samples = 10000
    config.data.max_eval_samples = 1000
    config.data.max_source_length = 128
    config.data.max_target_length = 128
    
    print("自定义配置创建完成:")
    print(f"  - 实验名称: {config.experiment_name}")
    print(f"  - 模型: {config.model.model_name}")
    print(f"  - 训练轮数: {config.training.num_train_epochs}")
    print(f"  - 批次大小: {config.training.per_device_train_batch_size}")
    print(f"  - 学习率: {config.training.learning_rate}")
    
    # 保存配置
    config.save_config("custom_config.yaml")
    print("✓ 配置已保存到 custom_config.yaml")
    print()


def example_4_memory_optimization():
    """示例4: 8GB显存优化设置"""
    print("=" * 60)
    print("示例4: 8GB显存优化配置")
    print("=" * 60)
    
    config = get_default_config()
    
    # 8GB显存优化设置
    print("应用8GB显存优化设置...")
    
    # 小批次 + 梯度累积
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 8
    print(f"  ✓ 实际批次: {config.training.per_device_train_batch_size}")
    print(f"  ✓ 梯度累积: {config.training.gradient_accumulation_steps}")
    print(f"  ✓ 等效批次: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    
    # 梯度检查点
    config.training.gradient_checkpointing = True
    print("  ✓ 梯度检查点: 已启用")
    
    # 减少序列长度
    config.data.max_source_length = 128
    config.data.max_target_length = 128
    print(f"  ✓ 序列长度: {config.data.max_source_length}")
    
    # 优化DataLoader
    config.training.dataloader_num_workers = 0
    config.training.dataloader_pin_memory = False
    print("  ✓ DataLoader: 已优化")
    
    print("\n8GB显存优化配置完成！")
    print()


def example_5_evaluation_setup():
    """示例5: 评估配置"""
    print("=" * 60)
    print("示例5: 评估指标配置")
    print("=" * 60)
    
    config = get_default_config()
    
    print("评估指标设置:")
    print(f"  - BLEU计算: {config.evaluation.compute_bleu}")
    print(f"  - ROUGE计算: {config.evaluation.compute_rouge}")
    print(f"  - 早停策略: {config.evaluation.early_stopping}")
    print(f"  - 早停耐心: {config.training.early_stopping_patience}")
    print(f"  - 评估步数: {config.training.eval_steps}")
    
    # 自定义评估设置
    config.evaluation.bleu_tokenize = "zh"  # 中文分词
    config.evaluation.length_penalty = 1.0
    config.evaluation.output_predictions = True
    
    print("\n自定义评估设置:")
    print(f"  ✓ BLEU分词: {config.evaluation.bleu_tokenize}")
    print(f"  ✓ 长度惩罚: {config.evaluation.length_penalty}")
    print(f"  ✓ 输出预测: {config.evaluation.output_predictions}")
    print()


def example_6_training_pipeline():
    """示例6: 完整训练流水线"""
    print("=" * 60)
    print("示例6: 完整训练流水线示例")
    print("=" * 60)
    
    print("训练流水线步骤:")
    print("1. 环境检查")
    print("   python setup_and_test.py")
    print()
    
    print("2. 快速测试")
    print("   python run_training.py train --preset quick_test --max-samples 100")
    print()
    
    print("3. 生产训练")
    print("   python run_training.py train --preset production")
    print()
    
    print("4. 模型评估")
    print("   python run_training.py evaluate ./results --dataset eval")
    print()
    
    print("5. 翻译测试")
    print("   python run_training.py translate ./results --text 'Hello world'")
    print()
    
    print("6. 交互式使用")
    print("   python run_training.py interactive ./results")
    print()


def main():
    """运行所有示例"""
    print("🎯 英中翻译框架使用示例")
    print("=" * 60)
    print("本示例将展示框架的各种使用方法")
    print()
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_preset_configs()
        example_3_custom_config()
        example_4_memory_optimization()
        example_5_evaluation_setup()
        example_6_training_pipeline()
        
        print("🎉 所有示例运行完成！")
        print()
        print("接下来您可以:")
        print("1. 运行 python setup_and_test.py 检查环境")
        print("2. 运行 python run_training.py train --preset quick_test 开始训练")
        print("3. 查看 README.md 了解更多功能")
        
    except Exception as e:
        print(f"❌ 示例运行出错: {e}")
        print("请检查环境配置或查看错误信息")


if __name__ == "__main__":
    main()
