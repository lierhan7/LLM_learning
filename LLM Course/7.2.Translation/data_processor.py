"""
机器翻译数据处理模块
处理英语到简体中文的翻译数据
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch

logger = logging.getLogger(__name__)


class TranslationDataProcessor:
    """翻译数据处理器"""
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None
        
    def load_dataset(self) -> DatasetDict:
        """加载数据集"""
        try:
            if self.config.data.train_file:
                # 加载本地文件
                logger.info("正在加载本地数据集...")
                data_files = {}
                if self.config.data.train_file:
                    data_files["train"] = self.config.data.train_file
                if self.config.data.validation_file:
                    data_files["validation"] = self.config.data.validation_file
                if self.config.data.test_file:
                    data_files["test"] = self.config.data.test_file
                
                # 根据文件扩展名判断格式
                extension = self.config.data.train_file.split(".")[-1]
                if extension == "json":
                    self.dataset = load_dataset("json", data_files=data_files)
                elif extension == "csv":
                    self.dataset = load_dataset("csv", data_files=data_files)
                else:
                    raise ValueError(f"不支持的文件格式: {extension}")
            else:
                # 加载Hugging Face数据集
                logger.info(f"正在加载数据集: {self.config.data.dataset_name}")
                self.dataset = load_dataset(
                    self.config.data.dataset_name,
                    self.config.data.dataset_config,
                    cache_dir=self.config.model.cache_dir
                )
            
            logger.info("数据集加载完成")
            self._print_dataset_info()
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"数据集加载失败: {str(e)}")
            raise
    
    def _print_dataset_info(self):
        """打印数据集信息"""
        if self.dataset:
            logger.info("数据集信息:")
            for split_name, split_data in self.dataset.items():
                logger.info(f"  {split_name}: {len(split_data)} 条数据")
                if len(split_data) > 0:
                    # 显示示例
                    example = split_data[0]
                    if 'translation' in example:
                        trans = example['translation']
                        logger.info(f"    示例 - 源: {trans[self.config.data.source_lang][:50]}...")
                        logger.info(f"    示例 - 目标: {trans[self.config.data.target_lang][:50]}...")
                    else:
                        logger.info(f"    示例: {str(example)[:100]}...")
    
    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """预处理数据集"""
        logger.info("开始预处理数据集...")
        
        # 预处理每个分割
        processed_dataset = {}
        for split_name, split_data in dataset.items():
            logger.info(f"预处理 {split_name} 集...")
            
            # 限制样本数量
            if split_name == "train" and self.config.data.max_train_samples:
                split_data = split_data.select(range(min(len(split_data), self.config.data.max_train_samples)))
            elif split_name == "validation" and self.config.data.max_eval_samples:
                split_data = split_data.select(range(min(len(split_data), self.config.data.max_eval_samples)))
            elif split_name == "test" and self.config.data.max_predict_samples:
                split_data = split_data.select(range(min(len(split_data), self.config.data.max_predict_samples)))
            
            # 应用tokenization
            processed_data = split_data.map(
                self._preprocess_function,
                batched=True,
                remove_columns=split_data.column_names,
                num_proc=self.config.data.preprocessing_num_workers,
                load_from_cache_file=not self.config.data.overwrite_cache,
                desc=f"预处理{split_name}数据"
            )
            
            processed_dataset[split_name] = processed_data
            logger.info(f"{split_name}集预处理完成: {len(processed_data)} 条数据")
        
        logger.info("数据集预处理完成")
        return DatasetDict(processed_dataset)
    
    def _preprocess_function(self, examples):
        """预处理函数"""
        # 提取源语言和目标语言文本
        if 'translation' in examples:
            # OPUS100格式
            sources = [ex[self.config.data.source_lang] for ex in examples['translation']]
            targets = [ex[self.config.data.target_lang] for ex in examples['translation']]
        else:
            # 自定义格式，假设有source和target字段
            sources = examples.get('source', examples.get(self.config.data.source_lang, []))
            targets = examples.get('target', examples.get(self.config.data.target_lang, []))
        
        # 为mT5添加任务前缀
        if self.config.data.source_prefix:
            sources = [self.config.data.source_prefix + source for source in sources]
        
        # Tokenize源语言
        model_inputs = self.tokenizer(
            sources,
            max_length=self.config.data.max_source_length,
            truncation=True,
            padding='max_length' if self.config.data.padding == 'max_length' else False,
            return_tensors=None
        )
        
        # Tokenize目标语言
        labels = self.tokenizer(
            targets,
            max_length=self.config.data.max_target_length,
            truncation=True,
            padding='max_length' if self.config.data.padding == 'max_length' else False,
            return_tensors=None
        )
        
        # 设置labels
        model_inputs["labels"] = labels["input_ids"]
        
        # 如果忽略pad token的loss，将pad token的label设置为-100
        if self.config.data.ignore_pad_token_for_loss:
            model_inputs["labels"] = [
                [(label if mask == 1 else -100) for label, mask in zip(label_ids, attention_mask)]
                for label_ids, attention_mask in zip(labels["input_ids"], labels["attention_mask"])
            ]
        
        return model_inputs
    
    def get_data_collator(self):
        """获取数据整理器"""
        from transformers import DataCollatorForSeq2Seq
        
        return DataCollatorForSeq2Seq(
            self.tokenizer,
            model=None,  # 将在trainer中设置
            label_pad_token_id=-100 if self.config.data.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            pad_to_multiple_of=None,
        )
    
    def extract_dataset_stats(self, dataset: DatasetDict) -> Dict[str, Any]:
        """提取数据集统计信息"""
        stats = {}
        
        for split_name, split_data in dataset.items():
            split_stats = {
                "num_examples": len(split_data),
                "avg_source_length": 0,
                "avg_target_length": 0,
                "max_source_length": 0,
                "max_target_length": 0
            }
            
            if len(split_data) > 0:
                # 计算长度统计
                source_lengths = []
                target_lengths = []
                
                # 采样一部分数据计算统计信息
                sample_size = min(1000, len(split_data))
                sample_data = split_data.select(range(sample_size))
                
                for example in sample_data:
                    if 'translation' in example:
                        source_text = example['translation'][self.config.data.source_lang]
                        target_text = example['translation'][self.config.data.target_lang]
                    else:
                        source_text = example.get('source', example.get(self.config.data.source_lang, ''))
                        target_text = example.get('target', example.get(self.config.data.target_lang, ''))
                    
                    source_tokens = self.tokenizer.tokenize(source_text)
                    target_tokens = self.tokenizer.tokenize(target_text)
                    
                    source_lengths.append(len(source_tokens))
                    target_lengths.append(len(target_tokens))
                
                if source_lengths:
                    split_stats["avg_source_length"] = np.mean(source_lengths)
                    split_stats["max_source_length"] = max(source_lengths)
                    split_stats["avg_target_length"] = np.mean(target_lengths)
                    split_stats["max_target_length"] = max(target_lengths)
            
            stats[split_name] = split_stats
            logger.info(f"{split_name}集统计:")
            logger.info(f"  样本数: {split_stats['num_examples']}")
            logger.info(f"  平均源长度: {split_stats['avg_source_length']:.1f}")
            logger.info(f"  平均目标长度: {split_stats['avg_target_length']:.1f}")
            logger.info(f"  最大源长度: {split_stats['max_source_length']}")
            logger.info(f"  最大目标长度: {split_stats['max_target_length']}")
        
        return stats
    
    def get_sample_data(self, split: str = "train", num_samples: int = 3) -> List[Dict[str, str]]:
        """获取样本数据用于检查"""
        if not self.dataset or split not in self.dataset:
            return []
        
        samples = []
        split_data = self.dataset[split]
        
        for i in range(min(num_samples, len(split_data))):
            example = split_data[i]
            if 'translation' in example:
                source = example['translation'][self.config.data.source_lang]
                target = example['translation'][self.config.data.target_lang]
            else:
                source = example.get('source', example.get(self.config.data.source_lang, ''))
                target = example.get('target', example.get(self.config.data.target_lang, ''))
            
            samples.append({
                'source': source,
                'target': target,
                'source_with_prefix': self.config.data.source_prefix + source if self.config.data.source_prefix else source
            })
        
        return samples


def create_translation_data_processor(config) -> TranslationDataProcessor:
    """创建翻译数据处理器"""
    # 加载tokenizer
    logger.info(f"正在加载tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        cache_dir=config.model.cache_dir,
        use_fast=config.model.use_fast_tokenizer,
        revision=config.model.model_revision,
        use_auth_token=config.model.use_auth_token,
        trust_remote_code=config.model.trust_remote_code
    )
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Tokenizer加载完成")
    
    # 创建数据处理器
    processor = TranslationDataProcessor(config, tokenizer)
    
    return processor
