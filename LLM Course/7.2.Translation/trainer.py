"""
机器翻译训练模块
整合所有组件，提供完整的翻译模型训练流程
"""

import logging
import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer
)
from datasets import DatasetDict

from model_manager import TranslationModel
from data_processor import TranslationDataProcessor
from evaluator import TranslationEvaluator

logger = logging.getLogger(__name__)


class TranslationTrainer:
    """翻译训练器"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.data_processor = None
        self.evaluator = None
        self.trainer = None
        
        # 检测设备
        self.device = self._get_device()
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_count = torch.cuda.device_count()
            logger.info(f"使用GPU训练，可用设备: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name}, 显存: {gpu_memory:.1f}GB")
        else:
            device = torch.device("cpu")
            logger.info("使用CPU训练")
        
        return device
    
    def prepare_data(self) -> DatasetDict:
        """准备数据"""
        logger.info("开始准备数据...")
        
        # 创建数据处理器
        logger.info("正在加载tokenizer...")
        model_manager = TranslationModel(self.config)
        self.model, self.tokenizer = model_manager.load_model_and_tokenizer()
        
        # 内存优化
        model_manager.optimize_for_memory()
        
        # 创建数据处理器
        self.data_processor = TranslationDataProcessor(self.config, self.tokenizer)
        
        # 加载原始数据集
        raw_dataset = self.data_processor.load_dataset()
        
        # 预处理数据集
        processed_dataset = self.data_processor.preprocess_dataset(raw_dataset)
        
        # 分割数据集
        self.train_dataset = processed_dataset.get("train")
        self.eval_dataset = processed_dataset.get("validation") 
        self.test_dataset = processed_dataset.get("test")
        
        logger.info("数据准备完成")
        logger.info(f"训练集: {len(self.train_dataset) if self.train_dataset else 0} 条")
        logger.info(f"验证集: {len(self.eval_dataset) if self.eval_dataset else 0} 条")
        logger.info(f"测试集: {len(self.test_dataset) if self.test_dataset else 0} 条")
        
        # 创建评估器
        self.evaluator = TranslationEvaluator(self.config, self.tokenizer)
        
        return processed_dataset
    
    def create_training_arguments(self) -> Seq2SeqTrainingArguments:
        """创建训练参数"""
        # 设置日志目录
        if self.config.training.logging_dir is None:
            self.config.training.logging_dir = os.path.join(
                self.config.training.output_dir, "logs"
            )
        
        training_args = Seq2SeqTrainingArguments(
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
            
            # 其他参数
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            dataloader_drop_last=self.config.training.dataloader_drop_last,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            dataloader_pin_memory=self.config.training.dataloader_pin_memory,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            
            # 日志和报告
            logging_dir=self.config.training.logging_dir,
            log_level="info",
            logging_steps=self.config.training.logging_steps,
            report_to=self.config.training.report_to,
            
            # 随机种子
            seed=self.config.training.seed,
            data_seed=self.config.training.data_seed,
            
            # 生成参数
            predict_with_generate=self.config.training.predict_with_generate,
            generation_max_length=self.config.training.generation_max_length,
            generation_num_beams=self.config.training.generation_num_beams,
            
            # Hub相关
            push_to_hub=self.config.training.push_to_hub,
            hub_model_id=self.config.training.hub_model_id,
            hub_strategy=self.config.training.hub_strategy,
            hub_token=self.config.training.hub_token,
            hub_private_repo=self.config.training.hub_private_repo,
            
            # 移除未使用的列
            remove_unused_columns=False,
            
            # 包含输入用于评估指标
            include_for_metrics=["input_ids", "attention_mask"],
        )
        
        return training_args
    
    def create_trainer(self) -> Seq2SeqTrainer:
        """创建Trainer"""
        # 创建训练参数
        training_args = self.create_training_arguments()
        
        # 获取数据整理器
        data_collator = self.data_processor.get_data_collator()
        
        # 回调函数
        callbacks = []
        if self.config.training.early_stopping_patience > 0:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.training.early_stopping_patience,
                early_stopping_threshold=self.config.training.early_stopping_threshold
            )
            callbacks.append(early_stopping)
        
        # 创建Trainer
        trainer = Seq2SeqTrainer(
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
    
    def train(self) -> Dict[str, Any]:
        """执行训练"""
        logger.info("开始训练...")
        
        # 准备数据
        if self.model is None:
            self.prepare_data()
        
        logger.info("模型准备完成")
        
        # 创建trainer
        self.trainer = self.create_trainer()
        
        # 开始训练
        train_result = self.trainer.train()
        
        # 保存模型
        logger.info("保存最终模型...")
        self.trainer.save_model()
        self.trainer.save_state()
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(self.config.training.output_dir)
        
        # 保存训练指标
        train_metrics = train_result.metrics
        self._save_metrics(train_metrics, "train_results.json")
        
        logger.info("训练完成")
        return train_result
    
    def evaluate(self, dataset_name: str = "eval") -> Dict[str, Any]:
        """评估模型"""
        logger.info(f"开始评估 {dataset_name} 集...")
        
        if self.trainer is None:
            raise ValueError("训练器未初始化，请先运行训练或加载模型")
        
        # 选择数据集
        if dataset_name == "eval" and self.eval_dataset:
            eval_dataset = self.eval_dataset
        elif dataset_name == "test" and self.test_dataset:
            eval_dataset = self.test_dataset
        else:
            logger.warning(f"数据集 {dataset_name} 不存在")
            return {}
        
        # 进行评估
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        # 生成预测
        logger.info("生成预测结果...")
        predictions = self.trainer.predict(eval_dataset)
        
        # 解码预测结果
        decoded_preds = self.tokenizer.batch_decode(
            predictions.predictions, skip_special_tokens=True
        )
        
        # 解码标签
        labels = predictions.label_ids
        # 确保 labels 是 torch tensor
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        labels = torch.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 详细评估
        logger.info("进行详细评估分析...")
        detailed_results = self.evaluator.evaluate_detailed(
            decoded_preds, decoded_labels
        )
        
        # 合并结果
        final_results = {
            "basic_metrics": eval_results,
            "detailed_analysis": detailed_results
        }
        
        # 保存结果
        self._save_metrics(final_results, f"{dataset_name}_results.json")
        
        # 保存预测结果
        if self.config.evaluation.output_predictions:
            self._save_predictions(
                decoded_preds, decoded_labels, 
                f"{dataset_name}_predictions.txt"
            )
        
        logger.info(f"{dataset_name}集评估完成")
        return final_results
    
    def generate_sample_translations(self, num_samples: int = 5) -> List[Dict[str, str]]:
        """生成样本翻译"""
        if not self.eval_dataset:
            logger.warning("没有可用的评估数据集")
            return []
        
        logger.info(f"生成 {num_samples} 个样本翻译...")
        
        samples = []
        sample_indices = list(range(min(num_samples, len(self.eval_dataset))))
        
        for idx in sample_indices:
            example = self.eval_dataset[idx]
            
            # 获取输入
            input_ids = torch.tensor([example['input_ids']]).to(self.device)
            attention_mask = torch.tensor([example['attention_mask']]).to(self.device)
            
            # 生成翻译
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.model.max_target_length,
                    num_beams=self.config.model.num_beams,
                    length_penalty=self.config.model.length_penalty,
                    early_stopping=self.config.model.early_stopping,
                    no_repeat_ngram_size=self.config.model.no_repeat_ngram_size,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            generated_text = self.tokenizer.decode(
                generated_tokens[0], skip_special_tokens=True
            )
            
            # 解码标签（参考答案）
            labels = example['labels']
            labels = [label if label != -100 else self.tokenizer.pad_token_id for label in labels]
            reference_text = self.tokenizer.decode(labels, skip_special_tokens=True)
            
            # 解码输入（源文本）
            source_text = self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            
            samples.append({
                'source': source_text,
                'reference': reference_text,
                'generated': generated_text
            })
        
        return samples
    
    def _save_metrics(self, metrics: Dict[str, Any], filename: str):
        """保存指标"""
        output_path = Path(self.config.training.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保所有值都可以JSON序列化
        serializable_metrics = self._make_serializable(metrics)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"指标已保存至: {output_path}")
    
    def _save_predictions(self, predictions: List[str], references: List[str], filename: str):
        """保存预测结果"""
        output_path = Path(self.config.training.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                f.write(f"样本 {i+1}:\n")
                f.write(f"预测: {pred}\n")
                f.write(f"参考: {ref}\n")
                f.write("-" * 50 + "\n")
        
        logger.info(f"预测结果已保存至: {output_path}")
    
    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj
    
    def save_config(self):
        """保存配置"""
        config_path = Path(self.config.training.output_dir) / "config.yaml"
        self.config.save_config(config_path)
        logger.info(f"配置已保存至: {config_path}")
    
    def load_model(self, model_path: str):
        """加载已训练的模型"""
        logger.info(f"正在加载模型: {model_path}")
        
        # 创建模型管理器
        model_manager = TranslationModel(self.config)
        
        # 加载模型和tokenizer
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 创建评估器
        self.evaluator = TranslationEvaluator(self.config, self.tokenizer)
        
        logger.info("模型加载完成")


def create_translation_trainer(config) -> TranslationTrainer:
    """创建翻译训练器"""
    trainer = TranslationTrainer(config)
    return trainer
