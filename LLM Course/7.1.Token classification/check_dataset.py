#!/usr/bin/env python3
"""
数据集格式检查脚本
用于检查bc2gm_corpus数据集的具体格式和列名
"""

def check_bc2gm_dataset():
    """检查BC2GM数据集格式"""
    try:
        from datasets import load_dataset, get_dataset_config_names
        
        print("正在检查BC2GM数据集...")
        
        # 检查可用配置
        try:
            configs = get_dataset_config_names('bc2gm_corpus')
            print(f"可用配置: {configs}")
        except Exception as e:
            print(f"无法获取配置列表: {e}")
        
        # 尝试加载数据集
        try:
            # 不指定配置，使用默认配置
            dataset = load_dataset('bc2gm_corpus', split='train[:5]')
            print("\n✅ 数据集加载成功!")
            print(f"列名: {dataset.column_names}")
            
            # 显示示例数据
            print("\n示例数据:")
            for i, example in enumerate(dataset):
                print(f"\n样本 {i+1}:")
                for key, value in example.items():
                    print(f"  {key}: {value}")
                if i >= 2:  # 只显示前3个样本
                    break
                    
            return dataset.column_names
            
        except Exception as e:
            print(f"❌ 加载默认配置失败: {e}")
            
            # 如果有多个配置，尝试每个配置
            if 'configs' in locals() and configs:
                for config in configs:
                    try:
                        print(f"\n尝试配置: {config}")
                        dataset = load_dataset('bc2gm_corpus', config, split='train[:3]')
                        print(f"✅ 配置 {config} 加载成功!")
                        print(f"列名: {dataset.column_names}")
                        
                        # 显示示例
                        print("\n示例数据:")
                        example = dataset[0]
                        for key, value in example.items():
                            print(f"  {key}: {value}")
                            
                        return dataset.column_names, config
                        
                    except Exception as e2:
                        print(f"❌ 配置 {config} 失败: {e2}")
            
            return None
            
    except ImportError:
        print("❌ 需要安装datasets库: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ 检查数据集时出错: {e}")
        return None

def suggest_config_changes(column_names, config_name=None):
    """根据数据集列名建议配置修改"""
    if not column_names:
        print("\n推荐使用通用配置...")
        return
        
    print(f"\n📋 根据数据集列名建议的配置修改:")
    print(f"数据集列名: {column_names}")
    
    # 常见的文本列名
    text_column_candidates = ['tokens', 'text', 'words', 'sentence', 'input_text']
    # 常见的标签列名
    label_column_candidates = ['ner_tags', 'labels', 'tags', 'entities', 'bio_tags']
    
    text_column = None
    label_column = None
    
    for col in column_names:
        if col.lower() in [c.lower() for c in text_column_candidates]:
            text_column = col
        if col.lower() in [c.lower() for c in label_column_candidates]:
            label_column = col
    
    print(f"\n建议的配置:")
    if config_name:
        print(f"dataset_config: '{config_name}'")
    if text_column:
        print(f"text_column_name: '{text_column}'")
    else:
        print(f"text_column_name: '{column_names[0]}' # 请手动确认")
    
    if label_column:
        print(f"label_column_name: '{label_column}'")
    else:
        print(f"label_column_name: '{column_names[-1]}' # 请手动确认")

if __name__ == "__main__":
    result = check_bc2gm_dataset()
    
    if isinstance(result, tuple):
        column_names, config_name = result
        suggest_config_changes(column_names, config_name)
    elif isinstance(result, list):
        suggest_config_changes(result)
    else:
        print("\n由于无法检查数据集，建议使用默认配置或手动指定列名")
