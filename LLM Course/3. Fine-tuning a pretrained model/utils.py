"""
训练实用工具函数
包含模型评估、可视化、数据处理等辅助功能
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def set_seed(seed: int):
    """设置随机种子以确保可重现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6
    }


def plot_training_history(train_losses: List[float], eval_losses: List[float] = None, 
                         eval_accuracies: List[float] = None, save_path: str = None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2 if eval_losses or eval_accuracies else 1, figsize=(15, 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # 训练损失
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    if eval_losses:
        axes[0].plot(eval_losses, label='Validation Loss', color='red')
    axes[0].set_title('Training/Validation Loss')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 评估指标
    if eval_accuracies and len(axes) > 1:
        axes[1].plot(eval_accuracies, label='Validation Accuracy', color='green')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_xlabel('Evaluation Steps')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path: str = None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()


def detailed_evaluation_report(y_true, y_pred, labels=None) -> Dict[str, Any]:
    """生成详细的评估报告"""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    print("=== 详细评估报告 ===")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    return report


def save_metrics(metrics: Dict[str, Any], save_path: str):
    """保存评估指标到JSON文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估指标已保存到: {save_path}")


def load_metrics(load_path: str) -> Dict[str, Any]:
    """从JSON文件加载评估指标"""
    with open(load_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前评估分数（越高越好）
            model: 当前模型
            
        Returns:
            bool: 是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        
        return False


class MetricsTracker:
    """指标跟踪器"""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.eval_f1_scores = []
        
    def add_train_loss(self, loss: float):
        """添加训练损失"""
        self.train_losses.append(loss)
    
    def add_eval_metrics(self, loss: float, accuracy: float, f1_score: float = None):
        """添加评估指标"""
        self.eval_losses.append(loss)
        self.eval_accuracies.append(accuracy)
        if f1_score is not None:
            self.eval_f1_scores.append(f1_score)
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """获取最佳指标"""
        best_metrics = {}
        
        if self.eval_accuracies:
            best_acc_idx = np.argmax(self.eval_accuracies)
            best_metrics['best_accuracy'] = self.eval_accuracies[best_acc_idx]
            best_metrics['best_accuracy_step'] = best_acc_idx
        
        if self.eval_losses:
            best_loss_idx = np.argmin(self.eval_losses)
            best_metrics['best_eval_loss'] = self.eval_losses[best_loss_idx]
            best_metrics['best_eval_loss_step'] = best_loss_idx
        
        if self.eval_f1_scores:
            best_f1_idx = np.argmax(self.eval_f1_scores)
            best_metrics['best_f1_score'] = self.eval_f1_scores[best_f1_idx]
            best_metrics['best_f1_step'] = best_f1_idx
        
        return best_metrics
    
    def plot_history(self, save_path: str = None):
        """绘制训练历史"""
        plot_training_history(
            self.train_losses, 
            self.eval_losses, 
            self.eval_accuracies, 
            save_path
        )
    
    def save_history(self, save_path: str):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'eval_accuracies': self.eval_accuracies,
            'eval_f1_scores': self.eval_f1_scores
        }
        
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"训练历史已保存到: {save_path}")


def print_gpu_utilization():
    """打印GPU使用情况"""
    if torch.cuda.is_available():
        print(f"GPU 设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  内存总量: {props.total_memory / 1024**3:.1f} GB")
            print(f"  已分配内存: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
            print(f"  缓存内存: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")
    else:
        print("CUDA 不可用")


if __name__ == "__main__":
    # 测试工具函数
    print("=== GPU 使用情况 ===")
    print_gpu_utilization()
    
    # 测试指标跟踪器
    tracker = MetricsTracker()
    
    # 模拟一些训练数据
    for i in range(10):
        tracker.add_train_loss(1.0 - i * 0.1)
        if i % 2 == 0:
            tracker.add_eval_metrics(0.8 - i * 0.05, 0.6 + i * 0.03)
    
    print("\n=== 最佳指标 ===")
    print(tracker.get_best_metrics())
