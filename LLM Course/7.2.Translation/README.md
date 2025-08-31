# 英中翻译模型训练框架

一个专业的英语到中文机器翻译训练框架，基于Transformer架构，针对8GB GPU内存优化设计。

## 🌟 特性

- 🔄 **完整的训练流水线**: 从数据处理到模型评估的端到端解决方案
- 🎯 **模块化设计**: 清晰的代码架构，易于维护和扩展
- 💾 **内存优化**: 专为8GB GPU设计，支持梯度检查点和智能批处理
- 📊 **多维度评估**: 集成BLEU和ROUGE评估指标
- ⚙️ **灵活配置**: 预设配置模板，支持快速测试和生产部署
- 🚀 **易于使用**: 简单的命令行界面和详细的文档

## 📁 项目结构

```
7.2.Translation/
├── config.py              # 配置管理模块
├── trainer.py             # 训练器核心逻辑
├── model_manager.py       # 模型加载和内存管理
├── data_processor.py      # 数据处理和预处理
├── evaluator.py           # 评估指标计算
├── run_training.py        # 训练主程序
├── setup_and_test.py      # 环境设置和测试
├── example.py             # 使用示例
├── requirements.txt       # 依赖包列表
├── test_config.yaml       # 测试配置文件
├── README.md              # 项目说明文档
├── QUICKSTART.md          # 快速开始指南
└── PROJECT_STATUS.md      # 项目状态报告
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖包
pip install -r requirements.txt

# 运行环境检查和测试
python setup_and_test.py
```

### 2. 快速训练

```bash
# 使用快速测试配置（推荐首次使用）
python run_training.py --template quick_test

# 使用生产配置
python run_training.py --template production

# 使用研究配置
python run_training.py --template research
```

### 3. 自定义训练

```bash
# 自定义参数训练
python run_training.py \
    --max_length 256 \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 3e-4
```

## ⚙️ 配置选项

### 预设模板

| 模板 | 用途 | 参数特点 |
|------|------|----------|
| `quick_test` | 快速测试 | 小批次，少轮数，快速验证 |
| `production` | 生产部署 | 平衡的参数设置，适合实际使用 |
| `research` | 研究实验 | 大模型，高精度，适合研究 |

### 主要参数说明

- `--max_length`: 最大序列长度（默认: 256）
- `--num_epochs`: 训练轮数（默认: 5）
- `--batch_size`: 批处理大小（默认: 8）
- `--learning_rate`: 学习率（默认: 3e-4）
- `--template`: 预设配置模板

## 📊 评估指标

框架集成了两种主要的翻译质量评估指标：

### BLEU Score
- 基于n-gram的精确率评估
- 支持中文分词（使用jieba）
- 范围：0-100，越高越好

### ROUGE Score
- 基于召回率的评估指标
- 包含ROUGE-1, ROUGE-2, ROUGE-L
- 范围：0-1，越高越好

## 🔧 核心模块

### 1. 配置管理 (config.py)
```python
from config import TrainingConfig

# 创建配置
config = TrainingConfig.quick_test()
config = TrainingConfig.production()
config = TrainingConfig.research()
```

### 2. 训练器 (trainer.py)
```python
from trainer import create_translation_trainer

# 创建训练器
trainer = create_translation_trainer(config)
trainer.train()
```

### 3. 评估器 (evaluator.py)
```python
from evaluator import TranslationEvaluator

evaluator = TranslationEvaluator()
scores = evaluator.evaluate_batch(predictions, references)
```

## 💾 内存优化策略

- **梯度检查点**: 减少激活内存占用
- **动态批处理**: 根据可用内存调整批次大小
- **FP32训练**: 确保兼容性和稳定性
- **智能缓存**: 优化数据加载和预处理

## 🎯 使用场景

### 学习和研究
```bash
# 快速验证概念
python run_training.py --template quick_test

# 深入研究实验
python run_training.py --template research --num_epochs 10
```

### 生产部署
```bash
# 标准生产训练
python run_training.py --template production

# 自定义生产参数
python run_training.py --template production --batch_size 16
```

## 📈 性能监控

训练过程中会实时显示：
- 训练损失变化
- 验证损失趋势
- BLEU评分提升
- GPU内存使用情况
- 训练速度统计

## 🔍 故障排除

### 常见问题

**1. CUDA内存不足**
```bash
# 减小批处理大小
python run_training.py --batch_size 4

# 使用快速测试配置
python run_training.py --template quick_test
```

**2. 数据加载错误**
```bash
# 检查网络连接，重新运行设置
python setup_and_test.py
```

**3. 依赖包问题**
```bash
# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

### 调试模式

```python
# 在代码中启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 详细文档

- [快速开始指南](QUICKSTART.md) - 5分钟快速上手
- [项目状态报告](PROJECT_STATUS.md) - 开发进度和功能清单
- [使用示例](example.py) - 详细的代码示例

## 🤝 贡献指南

1. Fork 项目仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 🙏 致谢

- Hugging Face Transformers - 提供优秀的模型和工具
- Opus100 数据集 - 提供高质量的翻译数据
- BLEU/ROUGE 评估工具 - 提供标准化评估指标

## 📞 支持

如果您在使用过程中遇到问题，请：

1. 查看 [故障排除](#-故障排除) 部分
2. 检查 [项目状态报告](PROJECT_STATUS.md)
3. 提交 Issue 或联系维护者

---

*最后更新: 2024年8月31日*
