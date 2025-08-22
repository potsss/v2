#!/usr/bin/env python3
"""
调试位置特征处理问题的脚本
"""
import sys
import os
import torch
import gc
import psutil
from config import Config

def check_system_resources():
    """检查系统资源"""
    print("=== 系统资源检查 ===")
    print(f"CPU核心数: {psutil.cpu_count()}")
    print(f"总内存: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"可用内存: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"内存使用率: {psutil.virtual_memory().percent}%")
    
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
        print(f"CUDA内存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"CUDA可用内存: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    else:
        print("CUDA不可用，将使用CPU")
    print()

def test_sentence_transformer():
    """测试Sentence Transformer模型"""
    try:
        from sentence_transformers import SentenceTransformer
        print("=== 测试Sentence Transformer ===")
        
        # 使用较小的模型进行测试
        test_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 更轻量的模型
        print(f"加载测试模型: {test_model_name}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(test_model_name, device=device)
        
        # 测试小批量编码
        test_texts = ["测试文本1", "测试文本2", "测试文本3"]
        print("测试编码...")
        embeddings = model.encode(test_texts, show_progress_bar=True)
        print(f"编码成功，形状: {embeddings.shape}")
        
        # 清理内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Sentence Transformer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def optimize_config_for_memory():
    """为内存优化配置"""
    print("=== 优化配置 ===")
    
    # 检查可用内存
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    if available_memory_gb < 4:
        print("内存不足4GB，建议关闭文本特征")
        Config.ENABLE_TEXT_FEATURES = False
        Config.LOCATION_TEXT_BATCH_SIZE = 16
    elif available_memory_gb < 8:
        print("内存不足8GB，使用小批次处理")
        Config.LOCATION_TEXT_BATCH_SIZE = 32
        # 使用更轻量的模型
        Config.LOCATION_TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    else:
        print("内存充足，使用默认配置")
        Config.LOCATION_TEXT_BATCH_SIZE = 64
    
    print(f"文本特征启用: {getattr(Config, 'ENABLE_TEXT_FEATURES', True)}")
    print(f"批次大小: {Config.LOCATION_TEXT_BATCH_SIZE}")
    print(f"文本模型: {Config.LOCATION_TEXT_EMBEDDING_MODEL}")
    print()

def create_lightweight_alternative():
    """创建轻量级替代方案"""
    print("=== 创建轻量级配置文件 ===")
    
    lightweight_config = """
# 轻量级位置特征配置
# 如果遇到内存问题，请使用此配置

# 关闭文本特征（最大的内存消耗源）
ENABLE_TEXT_FEATURES = False

# 或者使用更轻量的文本处理
LOCATION_TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCATION_TEXT_BATCH_SIZE = 16
LOCATION_TEXT_MAX_LENGTH = 64

# 减少位置特征维度
LOCATION_FEATURE_EMBEDDING_DIM = 32
LOCATION_GEOGRAPHIC_EMBEDDING_DIM = 16
LOCATION_SEMANTIC_EMBEDDING_DIM = 32
LOCATION_CATEGORICAL_EMBEDDING_DIM = 16

# 减少地理网格大小
COORDINATE_GRID_SIZE = 50
"""
    
    with open("lightweight_config.py", "w", encoding="utf-8") as f:
        f.write(lightweight_config)
    
    print("已创建 lightweight_config.py 文件")
    print("使用方法：")
    print("1. 将内容复制到 config.py 中覆盖相应配置")
    print("2. 或者在代码中导入: from lightweight_config import *")
    print()

def main():
    print("位置特征处理调试工具")
    print("=" * 50)
    
    # 检查系统资源
    check_system_resources()
    
    # 测试Sentence Transformer
    st_works = test_sentence_transformer()
    
    # 优化配置
    optimize_config_for_memory()
    
    # 创建轻量级替代方案
    create_lightweight_alternative()
    
    print("=== 建议 ===")
    if not st_works:
        print("❌ Sentence Transformer无法正常工作")
        print("   建议：设置 Config.ENABLE_TEXT_FEATURES = False")
    else:
        print("✅ Sentence Transformer工作正常")
    
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if available_memory_gb < 4:
        print("❌ 内存不足，建议关闭文本特征或使用更小的批次大小")
        print("   建议：Config.ENABLE_TEXT_FEATURES = False")
        print("   或者：Config.LOCATION_TEXT_BATCH_SIZE = 16")
    elif available_memory_gb < 8:
        print("⚠️  内存有限，建议使用轻量级模型和小批次")
        print("   建议：使用 all-MiniLM-L6-v2 模型")
        print("   建议：Config.LOCATION_TEXT_BATCH_SIZE = 32")
    else:
        print("✅ 内存充足，可以使用默认配置")
    
    print("\n如果问题仍然存在，请尝试：")
    print("1. 设置 Config.ENABLE_TEXT_FEATURES = False 跳过文本特征")
    print("2. 减小 Config.LOCATION_TEXT_BATCH_SIZE 到 16 或更小")
    print("3. 使用更轻量的模型 all-MiniLM-L6-v2")
    print("4. 监控内存使用情况，必要时重启程序")

if __name__ == "__main__":
    main()
