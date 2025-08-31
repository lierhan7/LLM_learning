"""
Token分类训练配置管理
支持NER、POS标注、命名实体识别等Token级别任务的配置
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelConfig:
    """模型相关配置"""
    model_name: str = "bert-base-cased"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    model_revision: str = "main"
    use_fast_tokenizer: bool = True
    
    # 模型架构参数
    num_labels: Optional[int] = None  # 自动从数据集推断
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: Optional[float] = None


@dataclass
class DataConfig:
    """数据相关配置"""
    dataset_name: str = "bc2gm_corpus"
    dataset_config: Optional[str] = "bc2gm_corpus"
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # 数据预处理参数
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    label_all_tokens: bool = False
    return_entity_level_metrics: bool = True
    
    # 标签相关
    text_column_name: str = "tokens"
    label_column_name: str = "ner_tags"
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None
    
    # 数据分割
    train_split_name: str = "train"
    validation_split_name: str = "validation"
    test_split_name: str = "test"


@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 基础训练参数
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 学习率调度
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    
    # 评估和保存策略
    evaluation_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    save_strategy: str = "epoch"
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    
    # 早停
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # 混合精度训练
    fp16: bool = False
    fp16_opt_level: str = "O1"
    fp16_backend: str = "auto"
    bf16: bool = False
    
    # 其他训练参数
    gradient_accumulation_steps: int = 1
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    group_by_length: bool = False
    length_column_name: str = "length"
    
    # 日志和报告
    logging_dir: Optional[str] = None
    logging_strategy: str = "steps"
    logging_steps: int = 100
    report_to: List[str] = field(default_factory=lambda: [])
    
    # 随机种子
    seed: int = 42
    data_seed: Optional[int] = None
    
    # 推送到Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"
    hub_token: Optional[str] = None
    hub_private_repo: bool = False


@dataclass
class EvaluationConfig:
    """评估相关配置"""
    metrics: List[str] = field(default_factory=lambda: ["seqeval"])
    compute_metrics_each_eval: bool = True
    prediction_loss_only: bool = False
    
    # 测试配置
    do_predict: bool = True
    test_file: Optional[str] = None
    
    # 输出配置
    output_predictions: bool = True
    predictions_file: str = "predictions.txt"
    output_results: bool = True
    results_file: str = "results.json"


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # 实验基本信息
    experiment_name: str = "token_classification_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
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
        """快速测试配置"""
        config = ExperimentConfig(
            experiment_name="quick_test_ner",
            description="快速测试NER模型训练"
        )
        
        # 模型配置 - 使用小模型快速测试
        config.model.model_name = "distilbert-base-cased"
        
        # 数据配置 - 使用少量数据
        config.data.max_length = 128
        
        # 训练配置 - 快速训练
        config.training.num_train_epochs = 1
        config.training.per_device_train_batch_size = 8
        config.training.per_device_eval_batch_size = 8
        config.training.evaluation_strategy = "steps"
        config.training.eval_steps = 50
        config.training.save_strategy = "steps"
        config.training.save_steps = 50
        config.training.logging_steps = 25
        config.training.warmup_steps = 10
        
        return config
    
    @staticmethod
    def production():
        """生产环境配置"""
        config = ExperimentConfig(
            experiment_name="production_ner",
            description="生产环境NER模型训练"
        )
        
        # 模型配置 - 使用高性能模型
        config.model.model_name = "bert-large-cased"
        
        # 训练配置 - 完整训练
        config.training.num_train_epochs = 5
        config.training.per_device_train_batch_size = 16
        config.training.learning_rate = 2e-5
        config.training.warmup_ratio = 0.1
        config.training.fp16 = True
        config.training.gradient_accumulation_steps = 2
        config.training.early_stopping_patience = 3
        
        return config
    
    @staticmethod
    def research():
        """研究实验配置"""
        config = ExperimentConfig(
            experiment_name="research_ner",
            description="研究用NER模型训练"
        )
        
        # 模型配置 - 使用最新模型
        config.model.model_name = "roberta-large"
        
        # 训练配置 - 详细记录
        config.training.num_train_epochs = 10
        config.training.learning_rate = 1e-5
        config.training.warmup_ratio = 0.06
        config.training.weight_decay = 0.01
        config.training.logging_steps = 50
        config.training.eval_steps = 100
        config.training.save_steps = 100
        config.training.report_to = ["tensorboard", "wandb"]
        
        return config


if __name__ == "__main__":
    # 创建示例配置文件
    templates = [
        ("config_quick_test.yaml", ConfigTemplates.quick_test()),
        ("config_production.yaml", ConfigTemplates.production()),
        ("config_research.yaml", ConfigTemplates.research()),
        ("config_default.yaml", ExperimentConfig())
    ]
    
    for filename, config in templates:
        config.save_config(filename)
        print(f"已保存配置: {filename}")
