"""
Token分类评估模块
提供全面的模型评估指标和分析工具
"""

import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter
import evaluate
from pathlib import Path

# 设置matplotlib后端，避免在无GUI环境中出错
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class TokenClassificationEvaluator:
    """Token分类评估器"""
    
    def __init__(self, label_list: List[str], id2label: Dict[int, str]):
        self.label_list = label_list
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        
        # 加载评估指标
        self.seqeval_metric = evaluate.load("seqeval")
        
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """计算评估指标（用于Trainer）"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # 移除ignored index并转换为标签
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # 使用seqeval计算指标
        results = self.seqeval_metric.compute(
            predictions=true_predictions, 
            references=true_labels
        )
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def evaluate_detailed(self, predictions: List[List[str]], 
                         references: List[List[str]]) -> Dict:
        """详细评估分析"""
        logger.info("进行详细评估分析...")
        
        # 基础指标
        basic_metrics = self.seqeval_metric.compute(
            predictions=predictions, 
            references=references
        )
        
        # 实体级别分析
        entity_metrics = self._compute_entity_metrics(predictions, references)
        
        # 标签级别分析
        label_metrics = self._compute_label_metrics(predictions, references)
        
        # 错误分析
        error_analysis = self._analyze_errors(predictions, references)
        
        # 合并所有结果
        detailed_results = {
            "basic_metrics": basic_metrics,
            "entity_metrics": entity_metrics,
            "label_metrics": label_metrics,
            "error_analysis": error_analysis
        }
        
        return detailed_results
    
    def _compute_entity_metrics(self, predictions: List[List[str]], 
                               references: List[List[str]]) -> Dict:
        """计算实体级别指标"""
        pred_entities = self._extract_entities(predictions)
        true_entities = self._extract_entities(references)
        
        # 统计实体数量
        entity_counts = {
            "predicted": len(pred_entities),
            "true": len(true_entities),
            "correct": len(pred_entities.intersection(true_entities))
        }
        
        # 计算指标
        precision = entity_counts["correct"] / max(entity_counts["predicted"], 1)
        recall = entity_counts["correct"] / max(entity_counts["true"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "entity_counts": entity_counts
        }
    
    def _compute_label_metrics(self, predictions: List[List[str]], 
                              references: List[List[str]]) -> Dict:
        """计算标签级别指标"""
        # 展平所有标签
        flat_predictions = [label for seq in predictions for label in seq]
        flat_references = [label for seq in references for label in seq]
        
        # 获取所有标签（除了O）
        all_labels = [label for label in self.label_list if label != "O"]
        
        # 计算分类报告
        report = classification_report(
            flat_references, 
            flat_predictions, 
            labels=all_labels,
            output_dict=True,
            zero_division=0
        )
        
        return report
    
    def _analyze_errors(self, predictions: List[List[str]], 
                       references: List[List[str]]) -> Dict:
        """错误分析"""
        errors = {
            "false_positives": [],  # 预测为实体但实际不是
            "false_negatives": [],  # 实际为实体但预测不是
            "boundary_errors": [],  # 边界错误
            "type_errors": []       # 类型错误
        }
        
        for pred_seq, true_seq in zip(predictions, references):
            pred_entities = self._extract_entities_with_positions([pred_seq])
            true_entities = self._extract_entities_with_positions([true_seq])
            
            # 分析错误类型
            self._classify_errors(pred_entities, true_entities, errors)
        
        # 统计错误数量
        error_stats = {
            error_type: len(error_list) 
            for error_type, error_list in errors.items()
        }
        
        return {
            "error_examples": errors,
            "error_counts": error_stats
        }
    
    def _extract_entities(self, sequences: List[List[str]]) -> set:
        """提取实体集合"""
        entities = set()
        
        for seq_idx, sequence in enumerate(sequences):
            current_entity = None
            start_idx = None
            
            for token_idx, label in enumerate(sequence):
                if label.startswith("B-"):
                    # 保存之前的实体
                    if current_entity:
                        entities.add((seq_idx, start_idx, token_idx - 1, current_entity))
                    # 开始新实体
                    current_entity = label[2:]
                    start_idx = token_idx
                elif label.startswith("I-"):
                    # 继续当前实体
                    if current_entity and label[2:] == current_entity:
                        continue
                    else:
                        # 不一致的I标签，重新开始
                        if current_entity:
                            entities.add((seq_idx, start_idx, token_idx - 1, current_entity))
                        current_entity = label[2:]
                        start_idx = token_idx
                else:
                    # O标签，结束当前实体
                    if current_entity:
                        entities.add((seq_idx, start_idx, token_idx - 1, current_entity))
                        current_entity = None
            
            # 序列结束时的实体
            if current_entity:
                entities.add((seq_idx, start_idx, len(sequence) - 1, current_entity))
        
        return entities
    
    def _extract_entities_with_positions(self, sequences: List[List[str]]) -> List[Tuple]:
        """提取实体及其位置信息"""
        entities = []
        
        for seq_idx, sequence in enumerate(sequences):
            current_entity = None
            start_idx = None
            
            for token_idx, label in enumerate(sequence):
                if label.startswith("B-"):
                    if current_entity:
                        entities.append((seq_idx, start_idx, token_idx - 1, current_entity))
                    current_entity = label[2:]
                    start_idx = token_idx
                elif label.startswith("I-"):
                    if current_entity and label[2:] == current_entity:
                        continue
                    else:
                        if current_entity:
                            entities.append((seq_idx, start_idx, token_idx - 1, current_entity))
                        current_entity = label[2:]
                        start_idx = token_idx
                else:
                    if current_entity:
                        entities.append((seq_idx, start_idx, token_idx - 1, current_entity))
                        current_entity = None
            
            if current_entity:
                entities.append((seq_idx, start_idx, len(sequence) - 1, current_entity))
        
        return entities
    
    def _classify_errors(self, pred_entities: List[Tuple], 
                        true_entities: List[Tuple], errors: Dict):
        """分类错误类型"""
        pred_set = set(pred_entities)
        true_set = set(true_entities)
        
        # False positives: 预测了但实际没有
        false_positives = pred_set - true_set
        
        # False negatives: 实际有但没预测到
        false_negatives = true_set - pred_set
        
        # 进一步分析边界错误和类型错误
        for fp in false_positives:
            # 检查是否是边界错误或类型错误
            seq_idx, start, end, entity_type = fp
            
            # 查找重叠的真实实体
            overlapping_true = [
                te for te in true_entities 
                if te[0] == seq_idx and self._has_overlap((start, end), (te[1], te[2]))
            ]
            
            if overlapping_true:
                # 有重叠，可能是边界错误或类型错误
                true_entity = overlapping_true[0]
                if entity_type != true_entity[3]:
                    errors["type_errors"].append((fp, true_entity))
                else:
                    errors["boundary_errors"].append((fp, true_entity))
            else:
                errors["false_positives"].append(fp)
        
        for fn in false_negatives:
            # 检查是否已经被归类为边界错误或类型错误
            already_classified = False
            for boundary_error in errors["boundary_errors"]:
                if fn == boundary_error[1]:  # boundary_error是(pred, true)的元组
                    already_classified = True
                    break
            for type_error in errors["type_errors"]:
                if fn == type_error[1]:  # type_error是(pred, true)的元组
                    already_classified = True
                    break
            
            if not already_classified:
                errors["false_negatives"].append(fn)
    
    def _has_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """检查两个span是否有重叠"""
        return not (span1[1] < span2[0] or span2[1] < span1[0])
    
    def plot_confusion_matrix(self, predictions: List[List[str]], 
                             references: List[List[str]], 
                             save_path: Optional[str] = None):
        """绘制混淆矩阵"""
        # 展平标签
        flat_predictions = [label for seq in predictions for label in seq]
        flat_references = [label for seq in references for label in seq]
        
        # 计算混淆矩阵
        cm = confusion_matrix(flat_references, flat_predictions, labels=self.label_list)
        
        # 绘制
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_list, yticklabels=self.label_list)
        plt.title('Token Classification Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    def plot_label_distribution(self, references: List[List[str]], 
                               save_path: Optional[str] = None):
        """绘制标签分布"""
        # 统计标签频次
        flat_labels = [label for seq in references for label in seq]
        label_counts = Counter(flat_labels)
        
        # 绘制
        plt.figure(figsize=(12, 6))
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        bars = plt.bar(labels, counts)
        plt.title('Label Distribution')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"标签分布图已保存到: {save_path}")
        
        plt.show()
    
    def save_evaluation_report(self, results: Dict, output_path: str):
        """保存评估报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"评估报告已保存到: {output_path}")
    
    def print_evaluation_summary(self, results: Dict):
        """打印评估摘要"""
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        
        # 基础指标
        basic = results["basic_metrics"]
        print(f"整体精确率: {basic['overall_precision']:.4f}")
        print(f"整体召回率: {basic['overall_recall']:.4f}")
        print(f"整体F1分数: {basic['overall_f1']:.4f}")
        print(f"整体准确率: {basic['overall_accuracy']:.4f}")
        
        # 实体级别指标
        entity = results["entity_metrics"]
        print(f"\n实体级别:")
        print(f"  精确率: {entity['entity_precision']:.4f}")
        print(f"  召回率: {entity['entity_recall']:.4f}")
        print(f"  F1分数: {entity['entity_f1']:.4f}")
        
        # 错误统计
        errors = results["error_analysis"]["error_counts"]
        print(f"\n错误分析:")
        print(f"  假阳性: {errors['false_positives']}")
        print(f"  假阴性: {errors['false_negatives']}")
        print(f"  边界错误: {errors['boundary_errors']}")
        print(f"  类型错误: {errors['type_errors']}")
        
        # 各类别F1分数
        label_metrics = results["label_metrics"]
        print(f"\n各类别F1分数:")
        for label in self.label_list:
            if label != "O" and label in label_metrics:
                f1 = label_metrics[label].get('f1-score', 0)
                print(f"  {label}: {f1:.4f}")
        
        print("="*60)


if __name__ == "__main__":
    # 测试评估器
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    id2label = {i: label for i, label in enumerate(label_list)}
    
    evaluator = TokenClassificationEvaluator(label_list, id2label)
    
    # 模拟预测和真实标签
    predictions = [
        ["O", "B-PER", "I-PER", "O", "B-ORG", "I-ORG"],
        ["B-LOC", "I-LOC", "O", "B-PER", "O", "O"]
    ]
    references = [
        ["O", "B-PER", "I-PER", "O", "B-ORG", "I-ORG"],
        ["B-LOC", "O", "O", "B-PER", "I-PER", "O"]
    ]
    
    # 详细评估
    results = evaluator.evaluate_detailed(predictions, references)
    evaluator.print_evaluation_summary(results)
