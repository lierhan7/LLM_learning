"""
机器翻译训练配置管理
支持英语到简体中文的翻译任务配置
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """模型相关配置"""
    model_name: str = "google/mt5-small"  # 适合8GB显存
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    model_revision: str = "main"
    use_fast_tokenizer: bool = True
    
    # 模型架构参数
    max_source_length: int = 512
    max_target_length: int = 512
    num_beams: int = 4
    length_penalty: float = 1.0
    
    # Dropout参数
    dropout_rate: float = 0.1
    
    # 生成参数
    early_stopping: bool = True
    no_repeat_ngram_size: int = 2


@dataclass
class DataConfig:
    """数据相关配置"""
    dataset_name: str = "opus100"
    dataset_config: str = "en-zh"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # 数据预处理参数
    max_source_length: int = 512
    max_target_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    ignore_pad_token_for_loss: bool = True
    
    # 语言对配置
    source_lang: str = "en"
    target_lang: str = "zh"
    source_prefix: str = "translate English to Chinese: "
    
    # 数据采样
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_predict_samples: Optional[int] = None
    
    # 数据分割
    train_split_name: str = "train"
    validation_split_name: str = "validation"
    test_split_name: str = "test"
    
    # 缓存
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None


@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基础训练参数
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # 适合8GB显存
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-4  # seq2seq任务通常需要稍高的学习率
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    
    # 学习率调度
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    # 评估和保存策略
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: Optional[int] = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_bleu"
    greater_is_better: bool = True
    
    # 早停
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # 梯度相关
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4  # 模拟更大的批次
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0  # 避免Windows多进程问题
    dataloader_pin_memory: bool = False  # 避免显存占用
    
    # 梯度检查点（节省显存）
    gradient_checkpointing: bool = True
    
    # 日志和报告
    logging_dir: Optional[str] = None
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: [])
    
    # 随机种子
    seed: int = 42
    data_seed: Optional[int] = None
    
    # 预测相关
    predict_with_generate: bool = True
    generation_max_length: int = 512
    generation_num_beams: int = 4
    
    # 推送到Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_token: Optional[str] = None
    hub_private_repo: bool = False


@dataclass
class EvaluationConfig:
    """评估相关配置"""
    metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge"])
    compute_metrics_each_eval: bool = True
    prediction_loss_only: bool = False
    
    # BLEU评估参数
    bleu_tokenize: str = "zh"  # 中文分词
    bleu_lowercase: bool = False
    
    # ROUGE评估参数
    rouge_use_stemmer: bool = True
    rouge_lang: str = "chinese"
    
    # 生成评估参数
    num_beams: int = 4
    max_length: int = 512
    length_penalty: float = 1.0
    early_stopping: bool = True
    
    # 输出配置
    output_predictions: bool = True
    predictions_file: str = "predictions.txt"
    output_results: bool = True
    results_file: str = "results.json"


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # 实验基本信息
    experiment_name: str = "en_zh_translation"
    description: str = "English to Chinese translation fine-tuning"
    tags: List[str] = field(default_factory=lambda: ["translation", "en-zh", "mt5"])
    
    # 各模块配置
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # 环境配置
    cuda_visible_devices: Optional[str] = None
    local_rank: int = -1
    
    def save_config(self, path: Union[str, Path]):
        """保存配置到YAML文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典格式
        config_dict = {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "cuda_visible_devices": self.cuda_visible_devices,
            "local_rank": self.local_rank
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]):
        """从YAML文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 重构嵌套配置对象
        model_config = ModelConfig(**config_dict.pop("model", {}))
        data_config = DataConfig(**config_dict.pop("data", {}))
        training_config = TrainingConfig(**config_dict.pop("training", {}))
        evaluation_config = EvaluationConfig(**config_dict.pop("evaluation", {}))
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            **config_dict
        )
    
    def update_from_args(self, args):
        """从命令行参数更新配置"""
        # 更新各个配置模块
        for key, value in vars(args).items():
            if value is None:
                continue
                
            # 根据参数前缀更新对应配置
            if key.startswith("model_"):
                attr_name = key[6:]  # 移除 "model_" 前缀
                if hasattr(self.model, attr_name):
                    setattr(self.model, attr_name, value)
            elif key.startswith("data_"):
                attr_name = key[5:]  # 移除 "data_" 前缀
                if hasattr(self.data, attr_name):
                    setattr(self.data, attr_name, value)
            elif key.startswith("training_"):
                attr_name = key[9:]  # 移除 "training_" 前缀
                if hasattr(self.training, attr_name):
                    setattr(self.training, attr_name, value)
            elif key.startswith("eval_"):
                attr_name = key[5:]  # 移除 "eval_" 前缀
                if hasattr(self.evaluation, attr_name):
                    setattr(self.evaluation, attr_name, value)
            else:
                # 直接设置在实验配置上
                if hasattr(self, key):
                    setattr(self, key, value)


class ConfigTemplates:
    """预定义配置模板"""
    
    @staticmethod
    def quick_test():
        """快速测试配置 - 8GB显存优化"""
        config = ExperimentConfig(
            experiment_name="quick_test_translation",
            description="快速测试英中翻译模型"
        )
        
        # 模型配置 - 小模型快速测试
        config.model.model_name = "t5-small"  # 使用更通用的T5模型
        config.model.max_source_length = 256
        config.model.max_target_length = 256
        
        # 数据配置 - 限制数据量
        config.data.max_source_length = 256
        config.data.max_target_length = 256
        config.data.max_train_samples = 1000
        config.data.max_eval_samples = 100
        
        # 训练配置 - 快速训练
        config.training.num_train_epochs = 1
        config.training.per_device_train_batch_size = 2
        config.training.per_device_eval_batch_size = 2
        config.training.gradient_accumulation_steps = 8
        config.training.eval_steps = 100
        config.training.save_steps = 100
        config.training.logging_steps = 50
        config.training.warmup_steps = 100
        config.training.gradient_checkpointing = True
        
        return config
    
    @staticmethod
    def production():
        """生产环境配置 - 8GB显存优化"""
        config = ExperimentConfig(
            experiment_name="production_translation",
            description="生产环境英中翻译模型"
        )
        
        # 模型配置 - 平衡性能和资源
        config.model.model_name = "Helsinki-NLP/opus-mt-en-zh"  # 使用更兼容的模型
        
        # 训练配置 - 完整训练
        config.training.num_train_epochs = 5
        config.training.per_device_train_batch_size = 4
        config.training.per_device_eval_batch_size = 4
        config.training.gradient_accumulation_steps = 4
        config.training.learning_rate = 3e-4
        config.training.warmup_ratio = 0.1
        config.training.gradient_checkpointing = True
        config.training.early_stopping_patience = 3
        config.training.max_grad_norm = 0.5  # 更小的梯度裁剪阈值
        config.training.dataloader_pin_memory = False  # 避免显存问题
        
        return config
    
    @staticmethod
    def research():
        """研究实验配置 - 最大化性能"""
        config = ExperimentConfig(
            experiment_name="research_translation",
            description="研究用英中翻译模型"
        )
        
        # 模型配置 - 如果显存允许，可以尝试base模型
        config.model.model_name = "google/mt5-small"  # 保守选择
        
        # 训练配置 - 详细实验
        config.training.num_train_epochs = 10
        config.training.per_device_train_batch_size = 2
        config.training.gradient_accumulation_steps = 8
        config.training.learning_rate = 1e-4
        config.training.warmup_ratio = 0.06
        config.training.weight_decay = 0.01
        config.training.logging_steps = 50
        config.training.eval_steps = 200
        config.training.save_steps = 200
        config.training.gradient_checkpointing = True
        
        return config


# 便利函数 - 直接使用ConfigTemplates类的方法
def quick_test_config():
    """快速测试配置"""
    return ConfigTemplates.quick_test()


def production_config():
    """生产环境配置"""
    return ConfigTemplates.production()


def research_config():
    """研究实验配置"""
    return ConfigTemplates.research()


def get_default_config():
    """获取默认配置"""
    return ExperimentConfig()


def save_config(config: ExperimentConfig, config_path: str):
    """保存配置到YAML文件"""
    config.save_config(config_path)


def load_config(config_path: str) -> ExperimentConfig:
    """从YAML文件加载配置"""
    return ExperimentConfig.from_yaml(config_path)


if __name__ == "__main__":
    # 创建示例配置文件
    templates = [
        ("config_quick_test.yaml", quick_test_config()),
        ("config_production.yaml", production_config()),
        ("config_research.yaml", research_config()),
        ("config_default.yaml", get_default_config())
    ]
    
    for filename, config in templates:
        config.save_config(filename)
        print(f"已保存配置: {filename}")
