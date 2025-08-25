"""
BERT微调训练脚本 - 文本分类任务
基于GLUE MRPC数据集进行微调训练
"""

import os
import logging
import time
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding, 
    AutoModelForSequenceClassification, 
    get_scheduler
)
from tqdm.auto import tqdm

from config import TrainingConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TextClassificationTrainer:
    """文本分类训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None
        self.lr_scheduler = None
        
    def tokenize_function(self, examples):
        """数据预处理函数"""
        return self.tokenizer(
            examples["sentence1"], 
            examples["sentence2"], 
            truncation=True,
            max_length=self.config.max_length,
            padding=False  # 使用动态padding
        )
    
    def prepare_data(self):
        """准备数据"""
        logger.info("加载数据集...")
        raw_datasets = load_dataset(self.config.dataset_name, self.config.dataset_config)
        
        logger.info("初始化tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        logger.info("数据预处理...")
        tokenized_datasets = raw_datasets.map(
            self.tokenize_function, 
            batched=True,
            desc="Tokenizing"
        )
        
        # 数据清理和格式化
        tokenized_datasets = tokenized_datasets.remove_columns(
            ["sentence1", "sentence2", "idx"]
        )
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        
        # 创建数据整理器
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            tokenized_datasets["train"], 
            shuffle=True, 
            batch_size=self.config.batch_size, 
            collate_fn=data_collator
        )
        self.eval_dataloader = DataLoader(
            tokenized_datasets["validation"], 
            batch_size=self.config.batch_size, 
            collate_fn=data_collator
        )
        
        logger.info(f"训练集大小: {len(tokenized_datasets['train'])}")
        logger.info(f"验证集大小: {len(tokenized_datasets['validation'])}")
    
    def prepare_model(self):
        """准备模型和优化器"""
        logger.info("初始化模型...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, 
            num_labels=self.config.num_labels
        )
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        num_training_steps = self.config.num_epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"总训练步数: {num_training_steps}")
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        logger.info("开始评估...")
        metric = evaluate.load(self.config.dataset_name, self.config.dataset_config)
        
        self.model.eval()
        total_eval_loss = 0
        num_eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="评估中"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                total_eval_loss += outputs.loss.item()
                num_eval_steps += 1
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(
                    predictions=predictions, 
                    references=batch["labels"]
                )
        
        eval_metrics = metric.compute()
        eval_metrics["eval_loss"] = total_eval_loss / num_eval_steps
        
        return eval_metrics
    
    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        
        # 准备数据和模型
        self.prepare_data()
        self.prepare_model()
        
        # 训练循环
        num_training_steps = self.config.num_epochs * len(self.train_dataloader)
        progress_bar = tqdm(range(num_training_steps), desc="训练进度")
        
        global_step = 0
        best_accuracy = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            self.model.train()
            total_train_loss = 0
            num_train_steps = 0
            
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                
                # 优化步骤
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # 记录损失
                total_train_loss += loss.item()
                num_train_steps += 1
                global_step += 1
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.lr_scheduler.get_last_lr()[0]:.2e}'
                })
                
                # 定期评估
                if global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    logger.info(f"Step {global_step} - 评估结果: {eval_metrics}")
                    
                    # 保存最佳模型
                    if eval_metrics.get("accuracy", 0) > best_accuracy:
                        best_accuracy = eval_metrics["accuracy"]
                        if self.config.save_model:
                            self.save_model("best_model")
                    
                    self.model.train()  # 回到训练模式
            
            # Epoch结束后的评估
            avg_train_loss = total_train_loss / num_train_steps
            eval_metrics = self.evaluate()
            
            logger.info(f"Epoch {epoch + 1} 完成")
            logger.info(f"平均训练损失: {avg_train_loss:.4f}")
            logger.info(f"评估结果: {eval_metrics}")
        
        # 最终评估
        final_metrics = self.evaluate()
        logger.info(f"最终评估结果: {final_metrics}")
        
        # 保存最终模型
        if self.config.save_model:
            self.save_model("final_model")
        
        return final_metrics
    
    def save_model(self, save_name: str):
        """保存模型"""
        save_path = os.path.join(self.config.output_dir, save_name)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"模型已保存到: {save_path}")


def main():
    """主函数"""
    try:
        # 创建配置
        config = TrainingConfig(
            batch_size=8,
            learning_rate=5e-5,
            num_epochs=3,
            output_dir="./bert_mrpc_results"
        )
        
        # 创建训练器并开始训练
        trainer = TextClassificationTrainer(config)
        start_time = time.time()
        
        final_metrics = trainer.train()
        
        training_time = time.time() - start_time
        logger.info(f"训练完成! 总耗时: {training_time:.2f}秒")
        logger.info(f"最终指标: {final_metrics}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()