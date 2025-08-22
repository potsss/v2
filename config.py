"""
v2 配置与实验路径管理
"""
import os
from datetime import datetime
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "v2", "data")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "v2", "experiments")


class Config:
    # 实验名与设备
    EXPERIMENT_NAME = "location_feature"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 行为数据
    DATA_PATH = os.path.join(DATA_DIR, "user_behavior.csv")
    EMBEDDING_DIM = 128
    WINDOW_SIZE = 5
    MIN_COUNT = 5
    NEGATIVE_SAMPLES = 5
    
    # 权重处理参数
    MAX_WEIGHT_REPEAT = 10     # URL权重最大重复次数
    DURATION_WEIGHT_SCALE = 60 # Duration权重缩放因子（秒转换为重复次数）
    USE_WEIGHTS_IN_TRAINING = True  # 是否在item2vec/node2vec训练中使用权重

    # 训练
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    BATCH_SIZE = 256
    RANDOM_SEED = 42
    NUM_WORKERS = 0
    PIN_MEMORY = True if DEVICE == "cuda" else False
    EARLY_STOPPING_PATIENCE = 3
    EVAL_INTERVAL = 1

    # 模型类型
    MODEL_TYPE = "matrix_factorization"  # "item2vec", "node2vec", or "matrix_factorization"


    # Node2Vec 参数
    P_PARAM = 1.0
    Q_PARAM = 1.0
    WALK_LENGTH = 20
    NUM_WALKS = 4
    
    # 矩阵分解参数
    MF_FACTORS = 128           # 矩阵分解隐因子维度
    MF_REGULARIZATION = 0.01   # 正则化参数
    MF_ITERATIONS = 10         # ALS迭代次数
    MF_ALPHA = 40.0           # 隐式反馈置信度参数
    MF_USER_AGGREGATION = "weighted_avg"  # "avg", "weighted_avg"
    MF_MIN_INTERACTIONS = 5    # 最小交互次数阈值
    
    # 矩阵分解位置特征增强参数
    MF_ENABLE_LOCATION_FEATURES = True  # 是否在矩阵分解中启用位置特征增强
    MF_FEATURE_ENHANCEMENT_WEIGHT = 0.3  # 位置特征增强权重 (0.0-1.0)

    # 属性
    ENABLE_ATTRIBUTES = True
    ATTRIBUTE_DATA_PATH = os.path.join(DATA_DIR, "user_attributes.tsv")
    ATTRIBUTE_EMBEDDING_DIM = 64
    FUSION_HIDDEN_DIM = 256
    FINAL_USER_EMBEDDING_DIM = 256
    FUSION_EPOCHS = 10  # 增加epochs，配合学习率调度和早停
    FUSION_LEARNING_RATE = 5e-4  # 降低初始学习率
    FUSION_BATCH_SIZE = 128  # 增大批次大小提升稳定性
    FUSION_WARMUP_EPOCHS = 2  # 学习率预热epochs
    FUSION_EARLY_STOPPING_PATIENCE = 3  # 融合训练早停耐心
    FUSION_LR_DECAY_FACTOR = 0.5  # 学习率衰减因子
    FUSION_LR_DECAY_PATIENCE = 2  # 学习率衰减耐心
    FUSION_WEIGHT_DECAY = 1e-5  # 权重衰减（L2正则化）
    FUSION_GRADIENT_CLIP_NORM = 1.0  # 梯度裁剪
    ATTRIBUTE_LEARNING_RATE = 1e-3
    ATTRIBUTE_EPOCHS = 1
    ATTRIBUTE_BATCH_SIZE = 512
    MASKING_RATIO = 0.15
    ATTRIBUTE_EARLY_STOPPING_PATIENCE = 5
    NUMERICAL_STANDARDIZATION = True
    CATEGORICAL_MIN_FREQ = 5
    
    # 对比学习参数
    ENABLE_CONTRASTIVE_LEARNING = True
    CONTRASTIVE_TEMPERATURE = 0.07  # 稍微降低温度参数，增强对比度
    CONTRASTIVE_LOSS_WEIGHT = 0.5  # 降低对比学习权重，避免过度优化
    BEHAVIOR_DROPOUT_RATE = 0.1    # 行为向量数据增强dropout率
    MAP_LOSS_WEIGHT = 1.0          # 掩码属性预测损失权重

    # 位置
    ENABLE_LOCATION = True
    LOCATION_DATA_PATH = os.path.join(DATA_DIR, "user_base_stations.tsv")
    LOCATION_FEATURES_PATH = os.path.join(DATA_DIR, "location_features.csv")  # 更新为新的特征文件
    LOCATION_EMBEDDING_DIM = 128
    LOCATION_MIN_CONNECTIONS = 2
    LOCATION_MODEL_TYPE = "item2vec"  # or "node2vec"
    
    # 位置特征处理参数
    ENABLE_LOCATION_FEATURES = True  # 是否启用丰富的位置特征
    LOCATION_FEATURE_EMBEDDING_DIM = 64  # 位置特征嵌入维度
    LOCATION_GEOGRAPHIC_EMBEDDING_DIM = 32  # 地理坐标嵌入维度
    LOCATION_SEMANTIC_EMBEDDING_DIM = 64  # 语义特征嵌入维度（基于文本）
    LOCATION_CATEGORICAL_EMBEDDING_DIM = 32  # 类别特征嵌入维度
    LOCATION_FEATURE_FUSION_DIM = 128  # 位置特征融合后的维度
    
    # 地理特征处理
    COORDINATE_NORMALIZATION = "minmax"  # "minmax", "standard", "none"
    COORDINATE_GRID_SIZE = 100  # 地理网格划分大小（用于离散化坐标）
    
    # 语义特征处理
    LOCATION_TEXT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 支持中文的模型
    LOCATION_TEXT_MAX_LENGTH = 128
    LOCATION_TEXT_BATCH_SIZE = 64  # 文本编码批次大小，减少内存使用
    ENABLE_TEXT_FEATURES = True  # 是否启用文本特征（可以关闭以跳过耗时的文本编码）
    
    # 类别特征处理
    LOCATION_CATEGORICAL_MIN_FREQ = 3  # 类别特征最小频次
    # 位置训练专属超参（None 表示沿用通用超参）
    LOCATION_LEARNING_RATE = None
    LOCATION_EPOCHS = 10
    LOCATION_BATCH_SIZE = None
    LOCATION_WINDOW_SIZE = None
    LOCATION_NEGATIVE_SAMPLES = None

    # 文本特征（可选）
    BASE_STATION_FEATURE_MODE = "none"  # or "text_embedding"
    TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_EMBEDDING_DIM = 384

    # 新用户
    NEW_USER_BEHAVIOR_PATH = os.path.join(DATA_DIR, "new_user_behavior.csv")
    NEW_USER_ATTRIBUTE_PATH = os.path.join(DATA_DIR, "new_user_attributes.tsv")
    NEW_USER_LOCATION_PATH = os.path.join(DATA_DIR, "new_user_base_stations.tsv")

    # 运行时路径（由 init_paths 设置）
    PROCESSED_DATA_PATH = None
    CHECKPOINT_DIR = None
    MODEL_SAVE_PATH = None
    LOG_DIR = None
    TENSORBOARD_DIR = None
    VISUALIZATION_DIR = None

    # Runtime 对象
    DEVICE_OBJ = torch.device(DEVICE)


def init_paths(experiment_name: str, mode: str = None):
    """
    初始化实验目录结构并写入 Config 路径。
    训练模式下如果已有同名目录，则自动追加时间戳；推理/可视化模式复用目录。
    """
    if not experiment_name:
        experiment_name = Config.EXPERIMENT_NAME
    base = os.path.join(EXPERIMENTS_DIR, experiment_name)

    # 训练/推理阶段默认复用同名实验目录，避免切换到时间戳目录导致读不到 preprocess 产物
    # 如需强制新开实验，请手动更换 experiment_name

    os.makedirs(base, exist_ok=True)

    Config.EXPERIMENT_NAME = os.path.basename(base)
    Config.PROCESSED_DATA_PATH = os.path.join(base, "processed_data")
    Config.CHECKPOINT_DIR = os.path.join(base, "checkpoints")
    Config.MODEL_SAVE_PATH = os.path.join(base, "models")
    Config.LOG_DIR = os.path.join(base, "logs")
    Config.TENSORBOARD_DIR = os.path.join(base, "runs")
    Config.VISUALIZATION_DIR = os.path.join(base, "visualizations")

    for d in [
        Config.PROCESSED_DATA_PATH,
        Config.CHECKPOINT_DIR,
        Config.MODEL_SAVE_PATH,
        Config.LOG_DIR,
        Config.TENSORBOARD_DIR,
        Config.VISUALIZATION_DIR,
    ]:
        os.makedirs(d, exist_ok=True)

    # 刷新设备对象
    Config.DEVICE_OBJ = torch.device(Config.DEVICE)

    return base




