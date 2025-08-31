"""
机器翻译评估模块
提供BLEU、ROUGE等翻译质量评估指标
"""

import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import evaluate
import jieba  # 中文分词

logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """翻译评估器"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metrics = {}
        self._load_metrics()
    
    def _load_metrics(self):
        """加载评估指标"""
        logger.info("正在加载评估指标...")
        
        try:
            # 加载BLEU指标
            if "bleu" in self.config.evaluation.metrics:
                self.metrics["bleu"] = evaluate.load("bleu")
                logger.info("✓ BLEU指标加载完成")
            
            # 加载ROUGE指标
            if "rouge" in self.config.evaluation.metrics:
                try:
                    self.metrics["rouge"] = evaluate.load("rouge")
                    logger.info("✓ ROUGE指标加载完成")
                except Exception as e:
                    logger.warning(f"ROUGE指标加载失败: {e}，请安装：pip install rouge-score")
            
            # 加载METEOR指标（如果可用）
            if "meteor" in self.config.evaluation.metrics:
                try:
                    self.metrics["meteor"] = evaluate.load("meteor")
                    logger.info("✓ METEOR指标加载完成")
                except Exception as e:
                    logger.warning(f"METEOR指标加载失败: {e}")
            
            # 加载BERTScore指标（如果可用）
            if "bertscore" in self.config.evaluation.metrics:
                try:
                    self.metrics["bertscore"] = evaluate.load("bertscore")
                    logger.info("✓ BERTScore指标加载完成")
                except Exception as e:
                    logger.warning(f"BERTScore指标加载失败: {e}")
                    
        except Exception as e:
            logger.error(f"评估指标加载失败: {str(e)}")
            # 确保至少有BLEU评估器作为后备
            try:
                self.metrics["bleu"] = evaluate.load("bleu")
                logger.info("✓ 后备BLEU指标加载完成")
            except Exception as e2:
                logger.error(f"连BLEU指标都无法加载: {e2}")
                self.metrics = {}  # 清空，让其他部分处理
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """计算评估指标（用于Trainer）"""
        predictions, labels = eval_preds
        
        # 解码预测和标签
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # 将-100替换为pad_token_id以便解码
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 清理文本
        decoded_preds = [self._clean_text(pred) for pred in decoded_preds]
        decoded_labels = [self._clean_text(label) for label in decoded_labels]
        
        # 调试信息
        logger.info(f"评估样本数: {len(decoded_preds)}")
        logger.info(f"预测示例: {decoded_preds[:3] if decoded_preds else '无'}")
        logger.info(f"标签示例: {decoded_labels[:3] if decoded_labels else '无'}")
        
        # 过滤空预测和标签
        valid_pairs = [(pred, label) for pred, label in zip(decoded_preds, decoded_labels) 
                      if pred.strip() and label.strip()]
        
        if not valid_pairs:
            logger.warning("没有有效的预测-标签对，返回零分")
            return {"bleu": 0.0, "rouge_l": 0.0}
        
        valid_preds, valid_labels = zip(*valid_pairs)
        logger.info(f"有效样本数: {len(valid_pairs)}")
        
        # 计算指标
        result = {}
        
        # BLEU评分
        if "bleu" in self.metrics:
            try:
                # 对中文进行分词，然后重新连接为字符串
                tokenized_preds = [" ".join(jieba.cut(pred)) for pred in valid_preds]
                tokenized_labels = [" ".join(jieba.cut(label)) for label in valid_labels]
                
                bleu_result = self.metrics["bleu"].compute(
                    predictions=tokenized_preds,
                    references=tokenized_labels
                )
                result["bleu"] = bleu_result["bleu"]
            except Exception as e:
                logger.warning(f"BLEU计算失败: {e}")
                result["bleu"] = 0.0
        
        # ROUGE评分
        if "rouge" in self.metrics:
            try:
                rouge_result = self.metrics["rouge"].compute(
                    predictions=list(valid_preds),
                    references=list(valid_labels),
                    use_stemmer=self.config.evaluation.rouge_use_stemmer
                )
                result["rouge_l"] = rouge_result["rougeL"]
            except Exception as e:
                logger.warning(f"ROUGE计算失败: {e}")
                result["rouge_l"] = 0.0
            result["rouge1"] = rouge_result["rouge1"]
            result["rouge2"] = rouge_result["rouge2"]
            result["rougeL"] = rouge_result["rougeL"]
        
        # 计算长度统计
        pred_lens = [len(pred.split()) for pred in decoded_preds]
        result["gen_len"] = np.mean(pred_lens)
        
        return result
    
    def evaluate_detailed(self, predictions: List[str], references: List[str], 
                         sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """详细评估分析"""
        logger.info("进行详细评估分析...")
        
        # 清理文本
        predictions = [self._clean_text(pred) for pred in predictions]
        references = [self._clean_text(ref) for ref in references]
        
        results = {}
        
        # 基础指标
        results.update(self._compute_basic_metrics(predictions, references))
        
        # 高级指标
        results.update(self._compute_advanced_metrics(predictions, references))
        
        # 质量分析
        results["quality_analysis"] = self._analyze_translation_quality(
            predictions, references, sources
        )
        
        # 错误分析
        results["error_analysis"] = self._analyze_errors(predictions, references)
        
        # 长度分析
        results["length_analysis"] = self._analyze_lengths(predictions, references)
        
        logger.info("详细评估分析完成")
        return results
    
    def _compute_basic_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算基础指标"""
        metrics = {}
        
        # BLEU分数
        if "bleu" in self.metrics:
            # 中文分词，转换为字符串格式
            tokenized_preds = [" ".join(jieba.cut(pred)) for pred in predictions]
            tokenized_refs = [" ".join(jieba.cut(ref)) for ref in references]
            
            # 计算不同n-gram的BLEU
            for n in range(1, 5):
                try:
                    bleu_score = self.metrics["bleu"].compute(
                        predictions=tokenized_preds,
                        references=tokenized_refs,
                        max_order=n
                    )
                    metrics[f"bleu_{n}"] = bleu_score["bleu"]
                except:
                    metrics[f"bleu_{n}"] = 0.0
        
        # ROUGE分数
        if "rouge" in self.metrics:
            rouge_scores = self.metrics["rouge"].compute(
                predictions=predictions,
                references=references,
                use_stemmer=self.config.evaluation.rouge_use_stemmer
            )
            metrics.update(rouge_scores)
        
        # METEOR分数
        if "meteor" in self.metrics:
            try:
                meteor_score = self.metrics["meteor"].compute(
                    predictions=predictions,
                    references=references
                )
                metrics["meteor"] = meteor_score["meteor"]
            except:
                logger.warning("METEOR计算失败")
        
        return metrics
    
    def _compute_advanced_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算高级指标"""
        metrics = {}
        
        # BERTScore
        if "bertscore" in self.metrics:
            try:
                bertscore = self.metrics["bertscore"].compute(
                    predictions=predictions,
                    references=references,
                    lang="zh"
                )
                metrics["bertscore_precision"] = np.mean(bertscore["precision"])
                metrics["bertscore_recall"] = np.mean(bertscore["recall"])
                metrics["bertscore_f1"] = np.mean(bertscore["f1"])
            except:
                logger.warning("BERTScore计算失败")
        
        # 字符级别的指标
        char_level_scores = self._compute_char_level_metrics(predictions, references)
        metrics.update(char_level_scores)
        
        return metrics
    
    def _compute_char_level_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算字符级别指标"""
        char_bleu_scores = []
        char_precision_scores = []
        char_recall_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_chars = list(pred.replace(" ", ""))
            ref_chars = list(ref.replace(" ", ""))
            
            # 字符级BLEU
            if ref_chars:
                try:
                    char_bleu = self.metrics["bleu"].compute(
                        predictions=[pred_chars],
                        references=[[ref_chars]]
                    )
                    char_bleu_scores.append(char_bleu["bleu"])
                except:
                    char_bleu_scores.append(0.0)
            
            # 字符级精确率和召回率
            if pred_chars and ref_chars:
                pred_set = set(pred_chars)
                ref_set = set(ref_chars)
                
                intersection = pred_set & ref_set
                precision = len(intersection) / len(pred_set) if pred_set else 0
                recall = len(intersection) / len(ref_set) if ref_set else 0
                
                char_precision_scores.append(precision)
                char_recall_scores.append(recall)
        
        return {
            "char_bleu": np.mean(char_bleu_scores) if char_bleu_scores else 0,
            "char_precision": np.mean(char_precision_scores) if char_precision_scores else 0,
            "char_recall": np.mean(char_recall_scores) if char_recall_scores else 0
        }
    
    def _analyze_translation_quality(self, predictions: List[str], references: List[str], 
                                   sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """翻译质量分析"""
        quality_bins = {"excellent": [], "good": [], "fair": [], "poor": []}
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # 计算单个样本的BLEU分数
            try:
                pred_tokens = list(jieba.cut(pred))
                ref_tokens = [list(jieba.cut(ref))]
                
                bleu_score = self.metrics["bleu"].compute(
                    predictions=[pred_tokens],
                    references=[ref_tokens]
                )["bleu"]
                
                # 质量分级
                if bleu_score >= 0.7:
                    quality_bins["excellent"].append(i)
                elif bleu_score >= 0.5:
                    quality_bins["good"].append(i)
                elif bleu_score >= 0.3:
                    quality_bins["fair"].append(i)
                else:
                    quality_bins["poor"].append(i)
            except:
                quality_bins["poor"].append(i)
        
        # 统计各质量等级的样本数量和比例
        total_samples = len(predictions)
        quality_stats = {}
        for quality, indices in quality_bins.items():
            count = len(indices)
            percentage = count / total_samples * 100 if total_samples > 0 else 0
            quality_stats[quality] = {
                "count": count,
                "percentage": percentage,
                "sample_indices": indices[:5]  # 只保存前5个样本索引作为例子
            }
        
        return quality_stats
    
    def _analyze_errors(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """错误分析"""
        error_analysis = {
            "empty_predictions": [],
            "length_mismatches": [],
            "repetition_errors": [],
            "untranslated_content": []
        }
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # 空预测
            if not pred.strip():
                error_analysis["empty_predictions"].append(i)
            
            # 长度不匹配（长度差异超过50%）
            if ref.strip():
                length_ratio = len(pred) / len(ref)
                if length_ratio < 0.5 or length_ratio > 2.0:
                    error_analysis["length_mismatches"].append({
                        "index": i,
                        "pred_length": len(pred),
                        "ref_length": len(ref),
                        "ratio": length_ratio
                    })
            
            # 重复错误（检测重复的n-gram）
            pred_words = pred.split()
            if len(pred_words) > 4:
                trigrams = [" ".join(pred_words[j:j+3]) for j in range(len(pred_words)-2)]
                if len(trigrams) != len(set(trigrams)):  # 有重复
                    error_analysis["repetition_errors"].append(i)
            
            # 未翻译内容（包含英文字符的比例过高）
            english_chars = sum(1 for c in pred if c.isascii() and c.isalpha())
            if english_chars / len(pred) > 0.3 if pred else 0:
                error_analysis["untranslated_content"].append(i)
        
        return error_analysis
    
    def _analyze_lengths(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """长度分析"""
        pred_lengths = [len(pred) for pred in predictions]
        ref_lengths = [len(ref) for ref in references]
        
        pred_word_lengths = [len(pred.split()) for pred in predictions]
        ref_word_lengths = [len(ref.split()) for ref in references]
        
        return {
            "char_lengths": {
                "pred_mean": np.mean(pred_lengths),
                "pred_std": np.std(pred_lengths),
                "ref_mean": np.mean(ref_lengths),
                "ref_std": np.std(ref_lengths),
                "correlation": np.corrcoef(pred_lengths, ref_lengths)[0, 1] if len(pred_lengths) > 1 else 0
            },
            "word_lengths": {
                "pred_mean": np.mean(pred_word_lengths),
                "pred_std": np.std(pred_word_lengths),
                "ref_mean": np.mean(ref_word_lengths),
                "ref_std": np.std(ref_word_lengths),
                "correlation": np.corrcoef(pred_word_lengths, ref_word_lengths)[0, 1] if len(pred_word_lengths) > 1 else 0
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not isinstance(text, str):
            return ""
        
        # 移除多余的空格
        text = " ".join(text.split())
        
        # 移除可能的特殊token
        text = text.replace("<pad>", "").replace("<unk>", "").replace("<s>", "").replace("</s>", "")
        
        return text.strip()
    
    def plot_score_distribution(self, scores: Dict[str, List[float]], 
                               save_path: Optional[str] = None):
        """绘制分数分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        score_names = list(scores.keys())[:4]  # 最多显示4个指标
        
        for i, score_name in enumerate(score_names):
            if i < len(axes):
                axes[i].hist(scores[score_name], bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{score_name.upper()} Distribution')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(score_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分数分布图已保存至: {save_path}")
        
        plt.close()
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                               output_dir: str, filename: str = "evaluation_results.json"):
        """保存评估结果"""
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保所有值都可以JSON序列化
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存至: {output_path}")
    
    def _make_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj


def create_translation_evaluator(config, tokenizer) -> TranslationEvaluator:
    """创建翻译评估器"""
    evaluator = TranslationEvaluator(config, tokenizer)
    return evaluator
