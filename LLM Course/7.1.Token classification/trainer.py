"""
Token分类专业训练器
集成数据处理、模型管理、训练和评估的完整训练框架
"""

import logging
import os
import json
import time
from typing import Dict, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    get_scheduler
)
from transformers.trainer_utils import set_seed
import numpy as np

from config import ExperimentConfig
from data_processor import TokenClassificationDataProcessor, DatasetStats
from model_manager import TokenClassificationModel, ModelOptimizer
from evaluator import TokenClassificationEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenClassificationTrainer:
    """专业Token分类训练器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.start_time = None
        
        # 核心组件
        self.model_manager = None
        self.data_processor = None
        self.evaluator = None
        self.trainer = None
        
        # 训练数据
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        
        # 设置随机种子
        set_seed(self.config.training.seed)
        
        # 设置GPU
        self._setup_device()
        
    def _setup_device(self):
        """设置计算设备"""
        if self.config.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        
        if torch.cuda.is_available():
            logger.info(f"使用GPU训练，可用设备: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("使用CPU训练")
    
    def prepare_data(self):
        """准备训练数据"""
        logger.info("开始准备数据...")
        
        # 初始化模型管理器和tokenizer
        self.model_manager = TokenClassificationModel(self.config)
        self.tokenizer = self.model_manager.load_tokenizer()
        
        # 初始化数据处理器
        self.data_processor = TokenClassificationDataProcessor(
            self.config, self.tokenizer
        )
        
        # 加载和预处理数据
        raw_dataset = self.data_processor.load_dataset()
        tokenized_dataset = self.data_processor.preprocess_dataset(raw_dataset)
        
        # 获取各个数据集分割
        self.train_dataset = self.data_processor.get_train_dataset(tokenized_dataset)
        self.eval_dataset = self.data_processor.get_eval_dataset(tokenized_dataset)
        self.test_dataset = self.data_processor.get_test_dataset(tokenized_dataset)
        
        # 打印数据统计
        stats = DatasetStats.compute_stats(tokenized_dataset, self.data_processor.label_list)
        DatasetStats.print_stats(stats)
        
        # 保存标签映射
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        self.data_processor.save_label_mapping(self.config.training.output_dir)
        
        logger.info("数据准备完成")
    
    def prepare_model(self):
        """准备模型"""
        logger.info("开始准备模型...")
        
        # 加载模型
        self.model = self.model_manager.load_model(
            num_labels=self.data_processor.num_labels,
            label2id=self.data_processor.label2id,
            id2label=self.data_processor.id2label
        )
        
        # 打印模型信息
        self.model_manager.print_model_summary()
        
        # 应用模型优化
        if self.config.training.gradient_accumulation_steps > 1:
            ModelOptimizer.apply_gradient_checkpointing(self.model)
        
        # 打印内存使用情况
        ModelOptimizer.print_memory_usage()
        
        logger.info("模型准备完成")
    
    def prepare_evaluator(self):
        """准备评估器"""
        self.evaluator = TokenClassificationEvaluator(
            label_list=self.data_processor.label_list,
            id2label=self.data_processor.id2label
        )
    
    def create_training_arguments(self) -> TrainingArguments:
        """创建训练参数"""
        # 设置日志目录
        if self.config.training.logging_dir is None:
            self.config.training.logging_dir = os.path.join(
                self.config.training.output_dir, "logs"
            )
        
        training_args = TrainingArguments(
            # 基础参数
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            adam_epsilon=self.config.training.adam_epsilon,
            max_grad_norm=self.config.training.max_grad_norm,
            
            # 学习率调度
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            warmup_ratio=self.config.training.warmup_ratio,
            warmup_steps=self.config.training.warmup_steps,
            
            # 评估和保存
            eval_strategy=self.config.training.evaluation_strategy,
            eval_steps=self.config.training.eval_steps,
            save_strategy=self.config.training.save_strategy,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            
            # 混合精度
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            
            # 其他参数
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            group_by_length=self.config.training.group_by_length,
            
            # 日志和报告
            logging_dir=self.config.training.logging_dir,
            log_level="info",
            logging_steps=self.config.training.logging_steps,
            report_to=self.config.training.report_to,
            
            # 随机种子
            seed=self.config.training.seed,
            data_seed=self.config.training.data_seed,
            
            # Hub相关
            push_to_hub=self.config.training.push_to_hub,
            hub_model_id=self.config.training.hub_model_id,
            hub_strategy=self.config.training.hub_strategy,
            hub_token=self.config.training.hub_token,
            hub_private_repo=self.config.training.hub_private_repo,
            
            # 移除未使用的列
            remove_unused_columns=False,
            
            # 确保可以使用eval数据集
            do_eval=self.eval_dataset is not None,
        )
        
        return training_args
    
    def create_trainer(self) -> Trainer:
        """创建Trainer"""
        # 创建训练参数
        training_args = self.create_training_arguments()
        
        # 数据整理器
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        # 回调函数
        callbacks = []
        if self.config.training.early_stopping_patience > 0:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.training.early_stopping_patience,
                early_stopping_threshold=self.config.training.early_stopping_threshold
            )
            callbacks.append(early_stopping)
        
        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer,
            compute_metrics=self.evaluator.compute_metrics,
            callbacks=callbacks
        )
        
        return trainer
    
    def train(self):
        """执行训练"""
        logger.info("开始训练...")
        self.start_time = time.time()
        
        # 准备所有组件
        self.prepare_data()
        self.prepare_model()
        self.prepare_evaluator()
        
        # 创建trainer
        self.trainer = self.create_trainer()
        
        # 保存配置
        self.save_config()
        
        # 开始训练
        train_result = self.trainer.train()
        
        # 保存最终模型
        self.trainer.save_model()
        
        # 计算训练时间
        training_time = time.time() - self.start_time
        logger.info(f"训练完成，总耗时: {training_time:.2f}秒")
        
        # 保存训练指标
        self.save_training_metrics(train_result, training_time)
        
        return train_result
    
    def evaluate(self, dataset_name: str = "eval") -> Dict:
        """评估模型"""
        if self.trainer is None:
            raise ValueError("请先完成训练或加载已训练的模型")
        
        logger.info(f"开始评估 {dataset_name} 数据集...")
        
        # 选择数据集
        if dataset_name == "eval" and self.eval_dataset:
            eval_dataset = self.eval_dataset
        elif dataset_name == "test" and self.test_dataset:
            eval_dataset = self.test_dataset
        else:
            raise ValueError(f"数据集 {dataset_name} 不可用")
        
        # 进行预测
        predictions = self.trainer.predict(eval_dataset)
        
        # 转换预测结果
        y_pred = np.argmax(predictions.predictions, axis=2)
        y_true = predictions.label_ids
        
        # 转换为标签序列
        pred_labels = []
        true_labels = []
        
        for pred_seq, true_seq in zip(y_pred, y_true):
            pred_seq_labels = []
            true_seq_labels = []
            
            for pred_label, true_label in zip(pred_seq, true_seq):
                if true_label != -100:
                    pred_seq_labels.append(self.data_processor.id2label[pred_label])
                    true_seq_labels.append(self.data_processor.id2label[true_label])
            
            pred_labels.append(pred_seq_labels)
            true_labels.append(true_seq_labels)
        
        # 详细评估
        results = self.evaluator.evaluate_detailed(pred_labels, true_labels)
        
        # 保存评估结果
        self.save_evaluation_results(results, dataset_name)
        
        # 打印评估摘要
        self.evaluator.print_evaluation_summary(results)
        
        # 生成可视化图表
        self.generate_evaluation_plots(pred_labels, true_labels, dataset_name)
        
        return results
    
    def save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.config.training.output_dir, "experiment_config.yaml")
        self.config.save_config(config_path)
        logger.info(f"实验配置已保存到: {config_path}")
    
    def save_training_metrics(self, train_result, training_time: float):
        """保存训练指标"""
        metrics = {
            "training_time_seconds": training_time,
            "training_time_formatted": f"{training_time:.2f}s",
            "train_loss": train_result.training_loss,
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
            "total_flos": train_result.metrics.get("total_flos"),
            "train_runtime": train_result.metrics.get("train_runtime"),
            "num_parameters": self.model_manager.count_parameters()
        }
        
        metrics_path = os.path.join(self.config.training.output_dir, "training_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"训练指标已保存到: {metrics_path}")
    
    def save_evaluation_results(self, results: Dict, dataset_name: str):
        """保存评估结果"""
        results_path = os.path.join(
            self.config.training.output_dir, 
            f"evaluation_results_{dataset_name}.json"
        )
        self.evaluator.save_evaluation_report(results, results_path)
    
    def generate_evaluation_plots(self, pred_labels: List[List[str]], 
                                 true_labels: List[List[str]], dataset_name: str):
        """生成评估可视化图表"""
        plots_dir = os.path.join(self.config.training.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 混淆矩阵
        cm_path = os.path.join(plots_dir, f"confusion_matrix_{dataset_name}.png")
        self.evaluator.plot_confusion_matrix(pred_labels, true_labels, cm_path)
        
        # 标签分布
        dist_path = os.path.join(plots_dir, f"label_distribution_{dataset_name}.png")
        self.evaluator.plot_label_distribution(true_labels, dist_path)
    
    def load_model_for_inference(self, model_path: str):
        """加载已训练模型用于推理"""
        logger.info(f"从 {model_path} 加载模型用于推理...")
        
        # 加载标签映射
        self.data_processor = TokenClassificationDataProcessor(self.config, None)
        (self.data_processor.label_list, 
         self.data_processor.label2id, 
         self.data_processor.id2label, 
         self.data_processor.num_labels) = self.data_processor.load_label_mapping(model_path)
        
        # 加载模型和tokenizer
        self.model_manager = TokenClassificationModel(self.config)
        self.tokenizer = self.model_manager.load_tokenizer()
        
        # 从保存路径加载模型
        from transformers import AutoModelForTokenClassification
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # 优化模型用于推理
        ModelOptimizer.optimize_for_inference(self.model)
        
        logger.info("模型加载完成，可用于推理")
    
    def predict(self, texts: List[str]) -> List[List[str]]:
        """对文本进行预测"""
        if self.model is None:
            raise ValueError("请先训练模型或加载已训练的模型")
        
        # 预处理文本
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_length,
            return_tensors="pt",
            is_split_into_words=False
        )
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # 转换为标签
        predicted_labels = []
        for i, prediction in enumerate(predictions):
            labels = []
            for j, pred_id in enumerate(prediction):
                # 跳过特殊token
                if inputs["attention_mask"][i][j] == 0:
                    continue
                labels.append(self.data_processor.id2label[pred_id.item()])
            predicted_labels.append(labels)
        
        return predicted_labels


def main():
    """主函数示例"""
    from config import ConfigTemplates
    
    # 使用快速测试配置
    config = ConfigTemplates.quick_test()
    config.data.dataset_name = "conll2003"
    config.training.output_dir = "./ner_experiment_output"
    
    # 创建训练器
    trainer = TokenClassificationTrainer(config)
    
    # 训练模型
    train_result = trainer.train()
    
    # 评估模型
    eval_results = trainer.evaluate("eval")
    
    # 如果有测试集，也进行评估
    if trainer.test_dataset:
        test_results = trainer.evaluate("test")
    
    logger.info("实验完成！")


if __name__ == "__main__":
    main()
