"""
训练配置文件
可以通过修改这个文件来调整训练参数，而不需要修改主代码
"""

import yaml
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 模型相关
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    
    # 数据相关
    dataset_name: str = "glue"
    dataset_config: str = "mrpc"
    max_length: int = 512
    
    # 训练相关
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 0
    weight_decay: float = 0.01
    
    # 评估相关
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps" or "epoch"
    
    # 保存相关
    output_dir: str = "./results"
    save_model: bool = True
    save_total_limit: int = 2
    
    # 日志相关
    logging_steps: int = 100
    logging_dir: str = "./logs"
    
    # 其他
    seed: int = 42
    fp16: bool = False  # 混合精度训练
    gradient_accumulation_steps: int = 1
    
    def save_config(self, path: str):
        """保存配置到YAML文件"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, path: str):
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# 预定义的配置模板
class ConfigTemplates:
    """预定义的配置模板"""
    
    @staticmethod
    def quick_test():
        """快速测试配置"""
        return TrainingConfig(
            batch_size=16,
            learning_rate=5e-5,
            num_epochs=1,
            eval_steps=50,
            output_dir="./quick_test_results"
        )
    
    @staticmethod
    def production():
        """生产环境配置"""
        return TrainingConfig(
            batch_size=16,
            learning_rate=2e-5,
            num_epochs=5,
            warmup_steps=100,
            weight_decay=0.01,
            eval_steps=200,
            fp16=True,
            output_dir="./production_results"
        )
    
    @staticmethod
    def large_model():
        """大模型配置（需要更多GPU内存）"""
        return TrainingConfig(
            model_name="bert-large-uncased",
            batch_size=4,
            learning_rate=1e-5,
            num_epochs=3,
            warmup_steps=200,
            gradient_accumulation_steps=2,
            fp16=True,
            output_dir="./large_model_results"
        )


if __name__ == "__main__":
    # 创建默认配置文件
    config = TrainingConfig()
    config.save_config("config.yaml")
    print("默认配置已保存到 config.yaml")
    
    # 创建快速测试配置
    quick_config = ConfigTemplates.quick_test()
    quick_config.save_config("config_quick_test.yaml")
    print("快速测试配置已保存到 config_quick_test.yaml")
    
    # 创建生产环境配置
    prod_config = ConfigTemplates.production()
    prod_config.save_config("config_production.yaml")
    print("生产环境配置已保存到 config_production.yaml")
