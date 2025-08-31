# 快速入门指南 🚀

## 1. 环境准备 (2分钟)

```bash
# 安装依赖
pip install -r requirements.txt

# 激活conda环境 (如果使用conda)
conda activate your_env_name
```

## 2. 立即开始训练 (1分钟)

### 快速测试 ⚡
```bash
python run_training.py --template quick_test
```
*使用DistilBERT在bc2gm_corpus数据集上进行1轮快速训练*

### 生产环境训练 🏭
```bash
python run_training.py --template production --output_dir ./my_model
```
*使用BERT-Large进行完整的5轮训练*

### 自定义训练 🎯
```bash
python run_training.py \
  --model_name bert-base-cased \
  --dataset_name bc2gm_corpus \
  --num_epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --output_dir ./custom_model
```

## 3. 查看结果 📊

训练完成后，检查输出目录：
- `results.json`: 评估指标
- `predictions.txt`: 预测结果
- `training.log`: 训练日志
- `pytorch_model.bin`: 训练好的模型

## 4. 常用命令组合 💡

### 调试训练
```bash
python run_training.py --template quick_test --num_epochs 1 --batch_size 4
```

### GPU内存优化
```bash
python run_training.py --template production --fp16 --gradient_accumulation_steps 2
```

### 使用本地数据
```bash
python run_training.py \
  --train_file ./data/train.json \
  --validation_file ./data/dev.json \
  --model_name bert-base-cased
```

### 完整实验设置
```bash
python run_training.py \
  --template production \
  --experiment_name "my_ner_exp_v1" \
  --description "Testing BERT-Large on biomedical data" \
  --output_dir ./experiments/exp_v1 \
  --fp16 \
  --early_stopping_patience 3
```

## 需要帮助？ 🆘

- 查看完整文档: [README.md](README.md)
- 检查数据集格式: `python check_dataset.py`
- 查看所有参数: `python run_training.py --help`

---
**30秒开始训练，10分钟得到结果！** ⏰
