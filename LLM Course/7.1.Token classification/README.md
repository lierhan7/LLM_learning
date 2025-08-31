# Token分类训练框架

一个功能完整、结构清晰的专业级Token分类训练框架，支持命名实体识别(NER)、词性标注(POS)等Token级别的自然语言处理任务。

## 🌟 特性

- **模块化设计**: 清晰的代码架构，易于维护和扩展
- **配置管理**: 支持YAML配置文件和预定义模板
- **多数据集支持**: 支持Hugging Face数据集和本地文件
- **灵活的训练配置**: 支持混合精度、早停、学习率调度等
- **详细的评估分析**: 提供F1、精确率、召回率等多种评估指标
- **错误分析**: 自动分析预测错误类型和原因
- **命令行接口**: 完整的CLI支持，方便批处理和脚本化
- **GPU加速**: 自动检测和使用可用的GPU设备

## 📁 项目结构

```
Token_Classification/
├── config.py              # 配置管理模块
├── data_processor.py      # 数据处理模块
├── model_manager.py       # 模型管理模块
├── evaluator.py           # 评估模块
├── trainer.py             # 训练模块
├── run_training.py        # 命令行接口
├── requirements.txt       # 依赖包列表
├── check_dataset.py       # 数据集检查工具
└── README.md              # 项目文档
```

## 🚀 快速开始

### 1. 环境准备

确保您的环境中安装了Python 3.8+和必要的依赖包：

```bash
pip install -r requirements.txt
```

### 2. 快速测试

使用预定义的快速测试模板：

```bash
python run_training.py --template quick_test
```

### 3. 生产环境训练

使用生产环境配置：

```bash
python run_training.py --template production --output_dir ./production_model
```

### 4. 自定义训练

指定自定义参数：

```bash
python run_training.py --model_name bert-base-cased --num_epochs 5 --learning_rate 2e-5 --batch_size 16
```

## 📖 详细使用指南

### 配置模板

框架提供三种预定义配置模板：

1. **quick_test**: 快速测试配置
   - 模型: DistilBERT
   - 训练轮数: 1
   - 批次大小: 8
   - 适用于: 快速验证和调试

2. **production**: 生产环境配置
   - 模型: BERT-Large
   - 训练轮数: 5
   - 混合精度: 启用
   - 适用于: 生产部署

3. **research**: 研究实验配置
   - 模型: RoBERTa-Large
   - 训练轮数: 10
   - 详细日志: 启用
   - 适用于: 学术研究

### 命令行参数

#### 基础参数
- `--config`: 指定YAML配置文件路径
- `--template`: 使用预定义模板 (quick_test|production|research)
- `--experiment_name`: 实验名称
- `--description`: 实验描述

#### 模型参数
- `--model_name`: 预训练模型名称或路径
- `--cache_dir`: 模型缓存目录
- `--trust_remote_code`: 是否信任远程代码

#### 数据参数
- `--dataset_name`: 数据集名称 (如: bc2gm_corpus, conll2003)
- `--dataset_config`: 数据集配置
- `--train_file`: 本地训练文件路径
- `--validation_file`: 本地验证文件路径
- `--test_file`: 本地测试文件路径
- `--max_length`: 最大序列长度
- `--label_all_tokens`: 是否标记所有子词token

#### 训练参数
- `--output_dir`: 输出目录
- `--num_epochs`: 训练轮数
- `--batch_size`: 训练批次大小
- `--eval_batch_size`: 评估批次大小
- `--learning_rate`: 学习率
- `--weight_decay`: 权重衰减
- `--warmup_ratio`: 预热比例
- `--fp16`: 启用FP16混合精度训练
- `--gradient_accumulation_steps`: 梯度累积步数

#### 评估参数
- `--evaluation_strategy`: 评估策略 (no|steps|epoch)
- `--eval_steps`: 评估间隔步数
- `--save_strategy`: 保存策略 (no|steps|epoch)
- `--save_steps`: 保存间隔步数
- `--load_best_model_at_end`: 训练结束时加载最佳模型
- `--early_stopping_patience`: 早停耐心值

### 配置文件示例

创建YAML配置文件 `my_config.yaml`:

```yaml
experiment_name: "my_ner_experiment"
description: "自定义NER实验"

model:
  model_name: "bert-base-cased"
  num_labels: null  # 自动推断

data:
  dataset_name: "bc2gm_corpus"
  dataset_config: "bc2gm_corpus"
  max_length: 512
  text_column_name: "tokens"
  label_column_name: "ner_tags"

training:
  output_dir: "./my_results"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  learning_rate: 5e-5
  weight_decay: 0.01
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  fp16: true

evaluation:
  metrics: ["seqeval"]
  output_predictions: true
  output_results: true
```

使用配置文件:
```bash
python run_training.py --config my_config.yaml
```

## 🗃️ 支持的数据集

### Hugging Face数据集
- `bc2gm_corpus`: 生物医学基因/蛋白质实体识别
- `conll2003`: CoNLL-2003 NER共享任务数据集
- `wnut_17`: WNUT-2017新兴实体识别
- 其他兼容的Token分类数据集

### 本地数据集
支持以下格式的本地文件：
- JSON格式: `{"tokens": [...], "ner_tags": [...]}`
- CSV格式: 带有tokens和labels列
- CoNLL格式: 每行一个token和标签，空行分隔句子

## 📊 评估指标

框架提供全面的评估指标：

### 基础指标
- **F1分数**: 精确率和召回率的调和平均
- **精确率**: 预测正确的实体占所有预测实体的比例
- **召回率**: 预测正确的实体占所有真实实体的比例
- **准确率**: Token级别的分类准确率

### 详细分析
- **按实体类型的指标**: 每种实体类型的详细评估
- **混淆矩阵**: 可视化分类错误情况
- **错误分析**: 自动分类错误类型
  - False Positives: 误报
  - False Negatives: 漏报
  - Boundary Errors: 边界错误
  - Type Errors: 类型错误

## 🛠️ 模块说明

### config.py
配置管理模块，提供：
- 层次化配置结构
- YAML序列化支持
- 预定义配置模板
- 命令行参数更新

### data_processor.py
数据处理模块，负责：
- 数据集加载和预处理
- Token对齐和标签处理
- 数据统计和可视化

### model_manager.py
模型管理模块，包含：
- 预训练模型加载
- 模型配置和优化
- 参数统计和内存监控

### evaluator.py
评估模块，提供：
- 多种评估指标计算
- 详细的错误分析
- 结果可视化

### trainer.py
训练模块，是框架的核心：
- 训练流程管理
- 模型训练和评估
- 检查点保存和加载

### run_training.py
命令行接口，提供：
- 完整的CLI参数支持
- 配置验证和错误检查
- 实验管理和日志记录

## 🔧 高级功能

### 混合精度训练
启用FP16或BF16混合精度训练以加速训练和节省内存：

```bash
python run_training.py --fp16 --template production
```

### 梯度累积
当GPU内存有限时，使用梯度累积模拟大批次：

```bash
python run_training.py --batch_size 8 --gradient_accumulation_steps 4
```

### 早停机制
自动停止过拟合的训练：

```bash
python run_training.py --early_stopping_patience 3
```

### 学习率调度
支持多种学习率调度策略：

```bash
python run_training.py --lr_scheduler_type cosine --warmup_ratio 0.1
```

## 📈 训练监控

### 日志文件
训练日志保存在 `{output_dir}/training.log`，包含：
- 训练进度和指标
- 模型性能变化
- 错误和警告信息

### 结果文件
训练结果保存在输出目录中：
- `results.json`: 评估指标
- `predictions.txt`: 模型预测结果
- `config.yaml`: 实际使用的配置
- `pytorch_model.bin`: 训练好的模型

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python run_training.py --batch_size 4
   
   # 使用梯度累积
   python run_training.py --batch_size 4 --gradient_accumulation_steps 4
   ```

2. **数据集加载失败**
   ```bash
   # 检查数据集格式
   python check_dataset.py
   
   # 指定正确的列名
   python run_training.py --dataset_name your_dataset --text_column_name tokens --label_column_name ner_tags
   ```

3. **模型性能不佳**
   - 增加训练轮数
   - 调整学习率
   - 尝试不同的预训练模型
   - 检查数据质量

### 调试模式
使用快速测试模板进行调试：

```bash
python run_training.py --template quick_test --num_epochs 1 --batch_size 2
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个框架！

### 开发环境设置
1. 克隆仓库
2. 安装依赖: `pip install -r requirements.txt`
3. 运行测试: `python run_training.py --template quick_test`

### 代码规范
- 遵循PEP 8代码风格
- 添加必要的文档字符串
- 编写单元测试

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: 在项目页面提交Issue
- Email: [您的邮箱]

## 🙏 致谢

感谢以下开源项目的支持：
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)

---

**Happy Training! 🚀**
