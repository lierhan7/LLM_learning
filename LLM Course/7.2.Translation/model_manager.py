"""
机器翻译模型管理模块
管理seq2seq翻译模型的加载、配置和优化
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import psutil
import gc

logger = logging.getLogger(__name__)


class TranslationModel:
    """翻译模型管理器"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
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
    
    def load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """加载模型和tokenizer"""
        # 加载tokenizer
        logger.info(f"正在加载tokenizer: {self.config.model.model_name}")
        
        try:
            # 先尝试使用AutoTokenizer with use_fast=False
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                cache_dir=self.config.model.cache_dir,
                use_fast=False,  # 使用slow tokenizer避免兼容性问题
                revision=self.config.model.model_revision,
                use_auth_token=self.config.model.use_auth_token,
                trust_remote_code=self.config.model.trust_remote_code
            )
            logger.info("使用AutoTokenizer加载成功")
        except Exception as e:
            logger.warning(f"使用AutoTokenizer加载失败: {e}")
            # 如果是marian模型，尝试使用MarianTokenizer
            if "opus-mt" in self.config.model.model_name.lower():
                try:
                    from transformers import MarianTokenizer
                    self.tokenizer = MarianTokenizer.from_pretrained(
                        self.config.model.model_name,
                        cache_dir=self.config.model.cache_dir
                    )
                    logger.info("使用MarianTokenizer加载成功")
                except Exception as e2:
                    logger.error(f"MarianTokenizer也失败: {e2}")
                    raise e2
            else:
                raise e
        
        # 确保tokenizer有必要的特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("设置pad_token为eos_token")
        
        logger.info("Tokenizer加载完成")
        
        # 加载模型
        logger.info(f"正在加载模型: {self.config.model.model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir,
            revision=self.config.model.model_revision,
            use_auth_token=self.config.model.use_auth_token,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True  # 减少CPU内存使用
        )
        
        # 调整模型配置
        self._configure_model()
        
        # 移动到设备
        self.model = self.model.to(self.device)
        
        logger.info("模型加载完成")
        self._print_model_info()
        
        return self.model, self.tokenizer
    
    def _configure_model(self):
        """配置模型参数"""
        if hasattr(self.model.config, 'dropout_rate'):
            self.model.config.dropout_rate = self.config.model.dropout_rate
        
        # 设置生成参数
        self.model.config.max_length = self.config.model.max_target_length
        self.model.config.num_beams = self.config.model.num_beams
        self.model.config.length_penalty = self.config.model.length_penalty
        self.model.config.early_stopping = self.config.model.early_stopping
        self.model.config.no_repeat_ngram_size = self.config.model.no_repeat_ngram_size
        
        # 确保模型可以使用tokenizer的pad_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
    
    def _print_model_info(self):
        """打印模型信息"""
        if self.model:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info("模型信息:")
            logger.info(f"  总参数量: {total_params:,}")
            logger.info(f"  可训练参数量: {trainable_params:,}")
            logger.info(f"  参数大小: {total_params * 4 / 1024**2:.1f}MB")
            
            # 估算显存使用
            if self.device.type == "cuda":
                self._print_memory_info()
    
    def _print_memory_info(self):
        """打印显存使用信息"""
        if torch.cuda.is_available():
            device_id = self.device.index if self.device.index is not None else 0
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            reserved_memory = torch.cuda.memory_reserved(device_id) / 1024**3
            allocated_memory = torch.cuda.memory_allocated(device_id) / 1024**3
            
            logger.info("显存使用:")
            logger.info(f"  总显存: {total_memory:.1f}GB")
            logger.info(f"  已分配: {allocated_memory:.1f}GB")
            logger.info(f"  已保留: {reserved_memory:.1f}GB")
            logger.info(f"  可用显存: {total_memory - reserved_memory:.1f}GB")
    
    def optimize_for_memory(self):
        """内存优化"""
        if self.model is None:
            return
        
        logger.info("执行内存优化...")
        
        # 启用梯度检查点
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点")
        
        # 清理不必要的缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("已清理GPU缓存")
        
        # 打印优化后的内存信息
        if self.device.type == "cuda":
            self._print_memory_info()
    
    def count_parameters(self) -> Dict[str, int]:
        """统计模型参数"""
        if self.model is None:
            return {}
        
        stats = {
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "frozen_params": sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        }
        
        # 按层统计
        layer_stats = {}
        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in layer_stats:
                layer_stats[layer_name] = {"params": 0, "trainable": 0}
            layer_stats[layer_name]["params"] += param.numel()
            if param.requires_grad:
                layer_stats[layer_name]["trainable"] += param.numel()
        
        stats["layer_stats"] = layer_stats
        return stats
    
    def freeze_encoder(self):
        """冻结编码器参数"""
        if self.model is None:
            return
        
        logger.info("冻结编码器参数...")
        if hasattr(self.model, 'encoder'):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            logger.info("编码器参数已冻结")
        else:
            logger.warning("模型没有encoder属性")
    
    def freeze_decoder_except_last_layers(self, num_layers: int = 2):
        """冻结解码器除最后几层外的所有参数"""
        if self.model is None:
            return
        
        logger.info(f"冻结解码器前N-{num_layers}层...")
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
            total_layers = len(self.model.decoder.layers)
            freeze_until = max(0, total_layers - num_layers)
            
            for i, layer in enumerate(self.model.decoder.layers):
                if i < freeze_until:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            logger.info(f"已冻结解码器前{freeze_until}层")
        else:
            logger.warning("模型没有decoder.layers属性")
    
    def get_generation_config(self) -> Dict[str, Any]:
        """获取生成配置"""
        return {
            "max_length": self.config.model.max_target_length,
            "num_beams": self.config.model.num_beams,
            "length_penalty": self.config.model.length_penalty,
            "early_stopping": self.config.model.early_stopping,
            "no_repeat_ngram_size": self.config.model.no_repeat_ngram_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "forced_eos_token_id": self.tokenizer.eos_token_id,
        }
    
    def generate_translation(self, input_text: str) -> str:
        """生成翻译"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型或tokenizer未加载")
        
        # 添加任务前缀
        if self.config.data.source_prefix:
            input_text = self.config.data.source_prefix + input_text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.model.max_source_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                **self.get_generation_config()
            )
        
        # Decode
        translation = self.tokenizer.decode(
            generated_tokens[0],
            skip_special_tokens=True
        )
        
        return translation


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, model: PreTrainedModel, config):
        self.model = model
        self.config = config
    
    def apply_optimizations(self):
        """应用各种优化技术"""
        logger.info("应用模型优化...")
        
        # 启用梯度检查点
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("✓ 梯度检查点已启用")
        
        # 其他优化可以在这里添加
        logger.info("模型优化完成")
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """获取内存占用"""
        memory_info = {}
        
        # 模型参数内存
        total_params = sum(p.numel() for p in self.model.parameters())
        param_memory = total_params * 4 / 1024**2  # 假设float32
        memory_info["model_params_mb"] = param_memory
        
        # GPU内存
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            memory_info["gpu_allocated_mb"] = allocated
            memory_info["gpu_reserved_mb"] = reserved
        
        # CPU内存
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**2
        memory_info["cpu_memory_mb"] = cpu_memory
        
        return memory_info


def create_translation_model(config) -> TranslationModel:
    """创建翻译模型管理器"""
    model_manager = TranslationModel(config)
    return model_manager
