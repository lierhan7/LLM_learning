# BERT微调训练项目

这是一个完整的BERT模型微调训练项目，用于文本分类任务。代码经过优化，具有良好的结构化设计和丰富的功能。

## 项目结构

```
├── trainer.py              # 主训练脚本（优化版本）
├── config.py               # 配置管理
├── utils.py                # 实用工具函数
├── run_training.py         # 命令行运行脚本
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明
```

## 主要特性

### 🚀 核心功能
- **完整的训练流程**: 数据预处理、模型训练、评估和保存
- **配置化管理**: 灵活的配置系统，支持YAML配置文件
- **详细的日志记录**: 完整的训练过程记录
- **模型评估**: 多种评估指标和可视化

### 📊 高级功能
- **早停机制**: 防止过拟合
- **学习率调度**: 线性学习率衰减
- **混合精度训练**: 节省内存，加速训练
- **梯度累积**: 支持大批次训练
- **指标跟踪**: 训练过程可视化

### 🛠️ 实用工具
- **GPU监控**: 实时显示GPU使用情况
- **参数统计**: 模型参数数量统计
- **可视化工具**: 训练曲线和混淆矩阵
- **模型保存**: 自动保存最佳模型

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基础训练

```bash
# 使用默认参数训练
python run_training.py

# 使用快速测试模板
python run_training.py --template quick_test

# 使用生产环境模板
python run_training.py --template production
```

### 3. 自定义训练

```bash
# 自定义参数
python run_training.py --batch_size 16 --learning_rate 2e-5 --num_epochs 5

# 使用配置文件
python run_training.py --config config_production.yaml
```

## 配置系统

### 使用预定义模板

```python
from config import ConfigTemplates

# 快速测试配置
config = ConfigTemplates.quick_test()

# 生产环境配置
config = ConfigTemplates.production()

# 大模型配置
config = ConfigTemplates.large_model()
```

### 创建自定义配置

```python
from config import TrainingConfig

config = TrainingConfig(
    model_name="bert-base-uncased",
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=5,
    output_dir="./my_results"
)

# 保存配置
config.save_config("my_config.yaml")
```

### 从YAML文件加载配置

```python
config = TrainingConfig.from_yaml("my_config.yaml")
```

## 高级使用

### 1. 直接使用训练器类

```python
from trainer import TextClassificationTrainer
from config import TrainingConfig

config = TrainingConfig(
    model_name="bert-base-uncased",
    batch_size=8,
    num_epochs=3
)

trainer = TextClassificationTrainer(config)
metrics = trainer.train()
```

### 2. 使用实用工具

```python
from utils import (
    set_seed, 
    count_parameters, 
    plot_training_history,
    EarlyStopping,
    MetricsTracker
)

# 设置随机种子
set_seed(42)

# 统计模型参数
params_info = count_parameters(model)
print(f"模型参数数量: {params_info['total_params_M']:.1f}M")

# 使用指标跟踪器
tracker = MetricsTracker()
# ... 训练过程中添加指标 ...
tracker.plot_history("training_history.png")
```

### 3. 早停机制

```python
from utils import EarlyStopping

early_stopping = EarlyStopping(patience=3, min_delta=0.001)

for epoch in range(num_epochs):
    # ... 训练代码 ...
    eval_score = evaluate_model()
    
    if early_stopping(eval_score, model):
        print("早停触发，停止训练")
        break
```

## 命令行参数

### 基本参数
- `--model_name`: 预训练模型名称 (默认: bert-base-uncased)
- `--batch_size`: 批次大小 (默认: 8)
- `--learning_rate`: 学习率 (默认: 5e-5)
- `--num_epochs`: 训练轮数 (默认: 3)

### 数据参数
- `--dataset_name`: 数据集名称 (默认: glue)
- `--dataset_config`: 数据集配置 (默认: mrpc)
- `--max_length`: 最大序列长度 (默认: 512)

### 高级参数
- `--fp16`: 启用混合精度训练
- `--warmup_steps`: 预热步数
- `--eval_steps`: 评估间隔步数
- `--seed`: 随机种子

## 输出文件

训练完成后，会在输出目录生成以下文件：

```
results/
├── best_model/              # 最佳模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
├── final_model/             # 最终模型
├── training_config.yaml     # 训练配置
├── training.log            # 训练日志
└── metrics.json            # 评估指标
```

## 性能优化建议

### 内存优化
- 使用较小的batch_size
- 启用混合精度训练 (`--fp16`)
- 使用梯度累积

### 训练加速
- 使用GPU训练
- 增加batch_size（在内存允许的情况下）
- 使用预热和学习率调度

### 模型效果
- 增加训练轮数
- 调整学习率
- 使用更大的预训练模型

## 常见问题

### Q: 内存不足怎么办？
A: 
1. 减小batch_size
2. 启用混合精度训练 (`--fp16`)
3. 使用梯度累积
4. 减小max_length

### Q: 训练速度慢怎么办？
A:
1. 确保使用GPU
2. 增加batch_size
3. 启用混合精度训练
4. 使用更快的数据加载

### Q: 如何选择超参数？
A:
1. 从预定义模板开始
2. 使用快速测试模板进行初步验证
3. 根据验证集结果调整参数

## 扩展建议

这个框架可以很容易扩展到其他任务：

1. **其他分类任务**: 修改数据集和标签数量
2. **多语言模型**: 使用多语言预训练模型
3. **大型模型**: 使用BERT-large或其他大型模型
4. **自定义数据**: 添加自定义数据加载器

## 许可证

MIT License
