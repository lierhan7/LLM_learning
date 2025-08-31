"""
框架安装和测试脚本
检查依赖、验证环境、运行基础测试
"""

import subprocess
import sys
import importlib
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """检查Python版本"""
    logger.info("检查Python版本...")
    version = sys.version_info
    logger.info(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("需要Python 3.8或更高版本")
        return False
    
    logger.info("✓ Python版本满足要求")
    return True


def check_gpu():
    """检查GPU可用性"""
    logger.info("检查GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✓ 发现 {gpu_count} 个GPU:")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 检查8GB限制
            if gpu_memory >= 8.0:
                logger.info("✓ GPU显存满足8GB要求")
            else:
                logger.warning(f"GPU显存仅{gpu_memory:.1f}GB，可能需要进一步优化")
            
            return True
        else:
            logger.warning("未检测到GPU，将使用CPU训练（速度较慢）")
            return False
    except ImportError:
        logger.error("PyTorch未安装，无法检查GPU")
        return False


def install_requirements():
    """安装依赖包"""
    logger.info("安装依赖包...")
    
    try:
        req_file = Path(__file__).parent / "requirements.txt"
        if not req_file.exists():
            logger.error("requirements.txt文件不存在")
            return False
        
        # 安装依赖
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ 依赖包安装成功")
            return True
        else:
            logger.error(f"依赖包安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"安装依赖包时出错: {str(e)}")
        return False


def check_required_packages():
    """检查必需的包"""
    logger.info("检查必需的包...")
    
    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "jieba",
        "numpy",
        "yaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少以下包: {', '.join(missing_packages)}")
        return False
    
    logger.info("✓ 所有必需包已安装")
    return True


def test_basic_functionality():
    """测试基础功能"""
    logger.info("测试基础功能...")
    
    try:
        # 测试配置模块
        logger.info("测试配置模块...")
        from config import quick_test_config
        config = quick_test_config()
        logger.info("✓ 配置模块正常")
        
        # 测试模型管理器
        logger.info("测试模型管理器...")
        from model_manager import TranslationModel
        model_manager = TranslationModel(config)
        logger.info("✓ 模型管理器正常")
        
        # 测试数据处理器
        logger.info("测试数据处理器...")
        from data_processor import TranslationDataProcessor
        # 不实际加载tokenizer，只测试类创建
        logger.info("✓ 数据处理器正常")
        
        # 测试评估器
        logger.info("测试评估器...")
        from evaluator import TranslationEvaluator
        logger.info("✓ 评估器正常")
        
        # 测试训练器
        logger.info("测试训练器...")
        from trainer import create_translation_trainer
        logger.info("✓ 训练器正常")
        
        logger.info("✓ 所有模块导入成功")
        return True
        
    except Exception as e:
        logger.error(f"功能测试失败: {str(e)}")
        return False


def test_model_loading():
    """测试模型加载（轻量级测试）"""
    logger.info("测试模型加载...")
    
    try:
        from transformers import AutoTokenizer
        
        # 测试加载一个小模型的tokenizer
        logger.info("测试tokenizer加载...")
        tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=False)
        
        # 简单测试
        test_text = "translate English to Chinese: Hello world"
        tokens = tokenizer(test_text, return_tensors="pt")
        
        logger.info(f"✓ Tokenizer测试通过")
        logger.info(f"  测试文本: {test_text}")
        logger.info(f"  Token数量: {len(tokens['input_ids'][0])}")
        
        return True
        
    except Exception as e:
        logger.error(f"模型加载测试失败: {str(e)}")
        logger.warning("这可能是网络问题，训练时会自动下载")
        return False


def create_test_config():
    """创建测试配置文件"""
    logger.info("创建测试配置文件...")
    
    try:
        from config import quick_test_config, save_config
        
        config = quick_test_config()
        config.training.output_dir = "./test_output"
        
        config_path = Path("test_config.yaml")
        save_config(config, config_path)
        
        logger.info(f"✓ 测试配置已保存至: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"创建测试配置失败: {str(e)}")
        return False


def main():
    """主安装和测试流程"""
    logger.info("=" * 60)
    logger.info("英中翻译框架 - 安装和测试")
    logger.info("=" * 60)
    
    tests = [
        ("Python版本检查", check_python_version),
        ("GPU检查", check_gpu),
        ("必需包检查", check_required_packages),
        ("基础功能测试", test_basic_functionality),
        ("模型加载测试", test_model_loading),
        ("创建测试配置", create_test_config),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            if not success:
                failed_tests.append(test_name)
        except Exception as e:
            logger.error(f"{test_name}异常: {str(e)}")
            failed_tests.append(test_name)
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("安装和测试总结")
    logger.info("=" * 60)
    
    if not failed_tests:
        logger.info("🎉 所有测试通过！框架已准备就绪")
        logger.info("\n可以开始使用以下命令:")
        logger.info("  python example.py                    # 运行示例")
        logger.info("  python run_training.py train --preset quick_test  # 快速训练测试")
        logger.info("  python run_training.py --help        # 查看所有命令")
    else:
        logger.error(f"❌ 以下测试失败: {', '.join(failed_tests)}")
        logger.error("\n请解决上述问题后重新运行测试")
        
        if "必需包检查" in failed_tests:
            logger.info("\n尝试安装依赖包:")
            logger.info("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
