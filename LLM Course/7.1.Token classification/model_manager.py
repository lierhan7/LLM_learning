"""
Token分类模型管理模块
处理模型初始化、配置和优化
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoConfig, 
    AutoModelForTokenClassification, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.modeling_outputs import TokenClassifierOutput

logger = logging.getLogger(__name__)


class TokenClassificationModel:
    """Token分类模型管理器"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """加载tokenizer"""
        logger.info(f"正在加载tokenizer: {self.config.model.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir,
            use_fast=self.config.model.use_fast_tokenizer,
            revision=self.config.model.model_revision,
            use_auth_token=self.config.model.use_auth_token,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # 确保tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.tokenizer = tokenizer
        logger.info("Tokenizer加载完成")
        return tokenizer
    
    def load_model(self, num_labels: int, label2id: Dict, id2label: Dict) -> PreTrainedModel:
        """加载和配置模型"""
        logger.info(f"正在加载模型: {self.config.model.model_name}")
        
        # 加载模型配置
        model_config = AutoConfig.from_pretrained(
            self.config.model.model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            cache_dir=self.config.model.cache_dir,
            revision=self.config.model.model_revision,
            use_auth_token=self.config.model.use_auth_token,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # 更新模型配置
        if self.config.model.hidden_dropout_prob is not None:
            model_config.hidden_dropout_prob = self.config.model.hidden_dropout_prob
        
        if self.config.model.attention_probs_dropout_prob is not None:
            model_config.attention_probs_dropout_prob = self.config.model.attention_probs_dropout_prob
        
        if self.config.model.classifier_dropout is not None:
            model_config.classifier_dropout = self.config.model.classifier_dropout
        
        # 加载模型
        model = AutoModelForTokenClassification.from_pretrained(
            self.config.model.model_name,
            config=model_config,
            cache_dir=self.config.model.cache_dir,
            revision=self.config.model.model_revision,
            use_auth_token=self.config.model.use_auth_token,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        self.model = model
        self.model_config = model_config
        logger.info(f"模型加载完成，参数量: {self.count_parameters()}")
        return model
    
    def count_parameters(self) -> Dict[str, int]:
        """统计模型参数"""
        if self.model is None:
            return {"total": 0, "trainable": 0}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "total_M": total_params / 1e6,
            "trainable_M": trainable_params / 1e6
        }
    
    def get_model_info(self) -> Dict:
        """获取模型详细信息"""
        if self.model is None:
            return {}
        
        param_stats = self.count_parameters()
        
        info = {
            "model_name": self.config.model.model_name,
            "model_type": self.model_config.model_type,
            "num_labels": self.model_config.num_labels,
            "hidden_size": getattr(self.model_config, 'hidden_size', 'N/A'),
            "num_hidden_layers": getattr(self.model_config, 'num_hidden_layers', 'N/A'),
            "num_attention_heads": getattr(self.model_config, 'num_attention_heads', 'N/A'),
            "max_position_embeddings": getattr(self.model_config, 'max_position_embeddings', 'N/A'),
            "vocab_size": getattr(self.model_config, 'vocab_size', 'N/A'),
            "parameters": param_stats
        }
        
        return info
    
    def freeze_embeddings(self):
        """冻结嵌入层参数"""
        if self.model is None:
            logger.warning("模型未加载，无法冻结嵌入层")
            return
        
        # 根据不同模型类型冻结嵌入层
        if hasattr(self.model, 'bert'):
            # BERT系列模型
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            logger.info("已冻结BERT嵌入层")
        elif hasattr(self.model, 'roberta'):
            # RoBERTa系列模型
            for param in self.model.roberta.embeddings.parameters():
                param.requires_grad = False
            logger.info("已冻结RoBERTa嵌入层")
        elif hasattr(self.model, 'distilbert'):
            # DistilBERT系列模型
            for param in self.model.distilbert.embeddings.parameters():
                param.requires_grad = False
            logger.info("已冻结DistilBERT嵌入层")
        else:
            logger.warning("未识别的模型类型，无法冻结嵌入层")
    
    def freeze_encoder_layers(self, num_layers: int):
        """冻结前N层编码器"""
        if self.model is None:
            logger.warning("模型未加载，无法冻结编码器层")
            return
        
        frozen_count = 0
        
        # 根据不同模型类型冻结编码器层
        if hasattr(self.model, 'bert'):
            # BERT系列模型
            encoder_layers = self.model.bert.encoder.layer
            for i in range(min(num_layers, len(encoder_layers))):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = False
                frozen_count += 1
        elif hasattr(self.model, 'roberta'):
            # RoBERTa系列模型
            encoder_layers = self.model.roberta.encoder.layer
            for i in range(min(num_layers, len(encoder_layers))):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = False
                frozen_count += 1
        elif hasattr(self.model, 'distilbert'):
            # DistilBERT系列模型
            transformer_layers = self.model.distilbert.transformer.layer
            for i in range(min(num_layers, len(transformer_layers))):
                for param in transformer_layers[i].parameters():
                    param.requires_grad = False
                frozen_count += 1
        else:
            logger.warning("未识别的模型类型，无法冻结编码器层")
            return
        
        logger.info(f"已冻结前 {frozen_count} 层编码器")
    
    def print_model_summary(self):
        """打印模型摘要"""
        if self.model is None:
            logger.warning("模型未加载")
            return
        
        info = self.get_model_info()
        
        print("\n" + "="*50)
        print("模型信息摘要")
        print("="*50)
        print(f"模型名称: {info['model_name']}")
        print(f"模型类型: {info['model_type']}")
        print(f"标签数量: {info['num_labels']}")
        print(f"隐藏层大小: {info['hidden_size']}")
        print(f"隐藏层数量: {info['num_hidden_layers']}")
        print(f"注意力头数: {info['num_attention_heads']}")
        print(f"最大位置编码: {info['max_position_embeddings']}")
        print(f"词汇表大小: {info['vocab_size']}")
        print(f"总参数量: {info['parameters']['total']:,} ({info['parameters']['total_M']:.2f}M)")
        print(f"可训练参数: {info['parameters']['trainable']:,} ({info['parameters']['trainable_M']:.2f}M)")
        print("="*50)


class ModelOptimizer:
    """模型优化工具"""
    
    @staticmethod
    def apply_gradient_checkpointing(model: PreTrainedModel):
        """应用梯度检查点以节省内存"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点")
        else:
            logger.warning("模型不支持梯度检查点")
    
    @staticmethod
    def optimize_for_inference(model: PreTrainedModel):
        """为推理优化模型"""
        model.eval()
        
        # 禁用dropout
        for module in model.modules():
            if hasattr(module, 'dropout'):
                module.dropout.p = 0.0
        
        logger.info("已优化模型用于推理")
    
    @staticmethod
    def get_memory_usage():
        """获取GPU内存使用情况"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        memory_info = {}
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            
            memory_info[f"gpu_{i}"] = {
                "allocated": f"{memory_allocated:.2f} GB",
                "reserved": f"{memory_reserved:.2f} GB", 
                "total": f"{memory_total:.2f} GB",
                "utilization": f"{(memory_allocated/memory_total)*100:.1f}%"
            }
        
        return memory_info
    
    @staticmethod
    def print_memory_usage():
        """打印内存使用情况"""
        memory_info = ModelOptimizer.get_memory_usage()
        
        if not memory_info.get("gpu_available", True):
            print("GPU 不可用")
            return
        
        print("\n" + "="*40)
        print("GPU 内存使用情况")
        print("="*40)
        for gpu_id, info in memory_info.items():
            print(f"{gpu_id.upper()}:")
            print(f"  已分配: {info['allocated']}")
            print(f"  已保留: {info['reserved']}")
            print(f"  总内存: {info['total']}")
            print(f"  使用率: {info['utilization']}")
        print("="*40)


if __name__ == "__main__":
    # 测试模型管理器
    from config import ConfigTemplates
    
    # 使用快速测试配置
    config = ConfigTemplates.quick_test()
    
    # 创建模型管理器
    model_manager = TokenClassificationModel(config)
    
    # 加载tokenizer
    tokenizer = model_manager.load_tokenizer()
    print(f"Tokenizer词汇量: {len(tokenizer)}")
    
    # 模拟标签映射
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    
    # 加载模型
    model = model_manager.load_model(len(label_list), label2id, id2label)
    
    # 打印模型摘要
    model_manager.print_model_summary()
    
    # 打印内存使用情况
    ModelOptimizer.print_memory_usage()
