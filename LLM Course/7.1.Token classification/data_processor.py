"""
Token分类数据处理模块
支持多种NER数据集和自定义数据格式的处理
"""

import logging
import os
from typing import Dict, List, Optional, Union, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class TokenClassificationDataProcessor:
    """Token分类数据处理器"""
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.label_list = None
        self.label2id = None
        self.id2label = None
        self.num_labels = None
        
    def load_dataset(self) -> DatasetDict:
        """加载数据集"""
        logger.info("正在加载数据集...")
        
        if self.config.data.dataset_name:
            # 从HuggingFace Hub加载
            dataset = load_dataset(
                self.config.data.dataset_name,
                self.config.data.dataset_config,
                cache_dir=self.config.model.cache_dir
            )
        else:
            # 从本地文件加载
            data_files = {}
            if self.config.data.train_file:
                data_files["train"] = self.config.data.train_file
            if self.config.data.validation_file:
                data_files["validation"] = self.config.data.validation_file
            if self.config.data.test_file:
                data_files["test"] = self.config.data.test_file
                
            if not data_files:
                raise ValueError("必须提供dataset_name或数据文件路径")
                
            dataset = load_dataset("json", data_files=data_files)
        
        logger.info(f"数据集加载完成: {dataset}")
        return dataset
    
    def extract_labels(self, dataset: DatasetDict) -> Tuple[List[str], Dict, Dict, int]:
        """提取标签信息"""
        logger.info("正在提取标签信息...")
        
        # 获取标签列名
        label_column = self.config.data.label_column_name
        
        # 从训练集获取标签特征
        if hasattr(dataset["train"].features[label_column], 'feature'):
            # 如果是Sequence特征（如ConLL2003）
            label_feature = dataset["train"].features[label_column].feature
            label_list = label_feature.names
        else:
            # 如果是普通列表，需要手动提取所有可能的标签
            all_labels = set()
            for example in dataset["train"]:
                all_labels.update(example[label_column])
            label_list = sorted(list(all_labels))
        
        # 创建标签映射
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}
        num_labels = len(label_list)
        
        logger.info(f"发现 {num_labels} 个标签: {label_list}")
        
        return label_list, label2id, id2label, num_labels
    
    def align_labels_with_tokens(self, labels: List[int], word_ids: List[Optional[int]]) -> List[int]:
        """对齐标签与子词token"""
        new_labels = []
        current_word = None
        
        for word_id in word_ids:
            if word_id != current_word:
                # 新单词的开始
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # 特殊token
                new_labels.append(-100)
            else:
                # 同一单词的后续token
                if self.config.data.label_all_tokens:
                    # 标记所有token
                    label = labels[word_id]
                    new_labels.append(label)
                else:
                    # 只标记第一个token，其他用-100
                    label = labels[word_id]
                    # 如果是B-标签，转换为I-标签
                    if label % 2 == 1:
                        label += 1
                    new_labels.append(label)
        
        return new_labels
    
    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        """对文本进行tokenization并对齐标签"""
        text_column = self.config.data.text_column_name
        label_column = self.config.data.label_column_name
        
        # 检查输入格式
        if isinstance(examples[text_column][0], list):
            # 文本已经是token列表
            tokenized_inputs = self.tokenizer(
                examples[text_column],
                truncation=self.config.data.truncation,
                max_length=self.config.data.max_length,
                padding=False,  # 使用DataCollator进行动态padding
                is_split_into_words=True
            )
        else:
            # 文本是字符串，需要先分词
            tokenized_inputs = self.tokenizer(
                examples[text_column],
                truncation=self.config.data.truncation,
                max_length=self.config.data.max_length,
                padding=False
            )
        
        # 对齐标签
        all_labels = examples[label_column]
        new_labels = []
        
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))
        
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """预处理整个数据集"""
        logger.info("正在预处理数据集...")
        
        # 提取标签信息
        self.label_list, self.label2id, self.id2label, self.num_labels = self.extract_labels(dataset)
        
        # 应用tokenization和标签对齐
        tokenized_dataset = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers,
            remove_columns=dataset["train"].column_names,
            desc="对数据进行tokenization"
        )
        
        logger.info("数据预处理完成")
        return tokenized_dataset
    
    def get_train_dataset(self, dataset: DatasetDict) -> Dataset:
        """获取训练数据集"""
        split_name = self.config.data.train_split_name
        if split_name not in dataset:
            raise ValueError(f"数据集中不存在训练分割: {split_name}")
        return dataset[split_name]
    
    def get_eval_dataset(self, dataset: DatasetDict) -> Optional[Dataset]:
        """获取验证数据集"""
        split_name = self.config.data.validation_split_name
        if split_name in dataset:
            return dataset[split_name]
        else:
            logger.warning(f"数据集中不存在验证分割: {split_name}")
            return None
    
    def get_test_dataset(self, dataset: DatasetDict) -> Optional[Dataset]:
        """获取测试数据集"""
        split_name = self.config.data.test_split_name
        if split_name in dataset:
            return dataset[split_name]
        else:
            logger.warning(f"数据集中不存在测试分割: {split_name}")
            return None
    
    def save_label_mapping(self, output_dir: str):
        """保存标签映射到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存标签列表
        with open(os.path.join(output_dir, "label_list.txt"), "w", encoding="utf-8") as f:
            for label in self.label_list:
                f.write(f"{label}\n")
        
        # 保存标签映射
        import json
        with open(os.path.join(output_dir, "label2id.json"), "w", encoding="utf-8") as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(output_dir, "id2label.json"), "w", encoding="utf-8") as f:
            json.dump(self.id2label, f, ensure_ascii=False, indent=2)
        
        logger.info(f"标签映射已保存到 {output_dir}")
    
    @staticmethod
    def load_label_mapping(input_dir: str) -> Tuple[List[str], Dict, Dict, int]:
        """从文件加载标签映射"""
        import json
        
        # 加载标签列表
        with open(os.path.join(input_dir, "label_list.txt"), "r", encoding="utf-8") as f:
            label_list = [line.strip() for line in f.readlines()]
        
        # 加载标签映射
        with open(os.path.join(input_dir, "label2id.json"), "r", encoding="utf-8") as f:
            label2id = json.load(f)
        
        with open(os.path.join(input_dir, "id2label.json"), "r", encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        
        num_labels = len(label_list)
        
        return label_list, label2id, id2label, num_labels


class DatasetStats:
    """数据集统计工具"""
    
    @staticmethod
    def compute_stats(dataset: DatasetDict, label_list: List[str]) -> Dict:
        """计算数据集统计信息"""
        stats = {}
        
        for split_name, split_data in dataset.items():
            split_stats = {
                "num_examples": len(split_data),
                "avg_tokens_per_example": 0,
                "max_tokens": 0,
                "min_tokens": float('inf'),
                "label_distribution": {label: 0 for label in label_list}
            }
            
            token_lengths = []
            
            for example in split_data:
                # 计算token长度（排除特殊token）
                tokens = [token for token in example["input_ids"] if token not in [0, 1, 2]]  # 排除PAD, CLS, SEP
                token_length = len(tokens)
                token_lengths.append(token_length)
                
                split_stats["max_tokens"] = max(split_stats["max_tokens"], token_length)
                split_stats["min_tokens"] = min(split_stats["min_tokens"], token_length)
                
                # 统计标签分布
                labels = example["labels"]
                for label_id in labels:
                    if label_id != -100 and label_id < len(label_list):
                        label_name = label_list[label_id]
                        split_stats["label_distribution"][label_name] += 1
            
            if token_lengths:
                split_stats["avg_tokens_per_example"] = sum(token_lengths) / len(token_lengths)
            
            stats[split_name] = split_stats
        
        return stats
    
    @staticmethod
    def print_stats(stats: Dict):
        """打印数据集统计信息"""
        for split_name, split_stats in stats.items():
            print(f"\n=== {split_name.upper()} 数据集统计 ===")
            print(f"样本数量: {split_stats['num_examples']}")
            print(f"平均token数: {split_stats['avg_tokens_per_example']:.2f}")
            print(f"最大token数: {split_stats['max_tokens']}")
            print(f"最小token数: {split_stats['min_tokens']}")
            
            print("\n标签分布:")
            total_labels = sum(split_stats['label_distribution'].values())
            for label, count in split_stats['label_distribution'].items():
                if count > 0:
                    percentage = (count / total_labels) * 100
                    print(f"  {label}: {count} ({percentage:.2f}%)")


if __name__ == "__main__":
    # 测试数据处理器
    from config import ExperimentConfig, ConfigTemplates
    from transformers import AutoTokenizer
    
    # 使用快速测试配置
    config = ConfigTemplates.quick_test()
    config.data.dataset_name = "conll2003"
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    # 创建数据处理器
    processor = TokenClassificationDataProcessor(config, tokenizer)
    
    # 加载和预处理数据集
    raw_dataset = processor.load_dataset()
    tokenized_dataset = processor.preprocess_dataset(raw_dataset)
    
    # 计算和打印统计信息
    stats = DatasetStats.compute_stats(tokenized_dataset, processor.label_list)
    DatasetStats.print_stats(stats)
    
    print(f"\n标签数量: {processor.num_labels}")
    print(f"标签列表: {processor.label_list[:10]}...")  # 只显示前10个标签
