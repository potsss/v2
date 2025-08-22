# 位置特征增强使用说明

## 概述

您的 `location_features.csv` 文件包含了丰富的位置信息，包括经纬度、城市信息、基站类型、覆盖信息等。我们已经为您实现了一套完整的位置特征处理和集成方案。

## 特征类型

### 1. 地理特征
- **经纬度坐标**: `longitude`, `latitude`, `longitude_gps`, `latitude_gps`
- **地理网格**: 自动计算基于坐标的网格ID，用于空间聚类

### 2. 类别特征  
- **行政区域**: `city_name`, `county_name`, `town_name`
- **基站属性**: `site_net_type_name`, `cover_type`, `cover_info`
- **环境特征**: `along_railway`, `along_freeway`

### 3. 语义特征
- **文本信息**: `site_name`, `site_address`, `cover_area`
- **语义编码**: 支持中文的Sentence Transformer或TF-IDF

## 配置参数

在 `config.py` 中，我们添加了以下配置选项：

```python
# 位置特征处理参数
ENABLE_LOCATION_FEATURES = True  # 是否启用丰富的位置特征
LOCATION_FEATURE_EMBEDDING_DIM = 64  # 位置特征嵌入维度
LOCATION_GEOGRAPHIC_EMBEDDING_DIM = 32  # 地理坐标嵌入维度
LOCATION_SEMANTIC_EMBEDDING_DIM = 64  # 语义特征嵌入维度
LOCATION_CATEGORICAL_EMBEDDING_DIM = 32  # 类别特征嵌入维度
LOCATION_FEATURE_FUSION_DIM = 128  # 位置特征融合后的维度

# 地理特征处理
COORDINATE_NORMALIZATION = "minmax"  # "minmax", "standard", "none"
COORDINATE_GRID_SIZE = 100  # 地理网格划分大小

# 语义特征处理
LOCATION_TEXT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCATION_TEXT_MAX_LENGTH = 128

# 类别特征处理
LOCATION_CATEGORICAL_MIN_FREQ = 3  # 类别特征最小频次
```

## 使用方法

### 1. 数据预处理
```bash
python main.py --mode preprocess --enable_location
```

这将：
- 处理用户位置序列
- 加载和处理 `location_features.csv` 
- 提取地理、类别、语义特征
- 保存处理后的特征到 `location_features_processed.pkl`

### 2. 训练模型
```bash
python main.py --mode train --enable_location
```

这将：
- 训练行为嵌入模型
- 训练位置嵌入模型
- 自动集成位置特征处理

### 3. 融合训练
```bash
python main.py --mode train_fusion --enable_location
```

这将：
- 使用增强的位置特征嵌入
- 融合行为、属性、位置特征
- 生成最终的用户表示

### 4. 计算融合嵌入
```bash
python main.py --mode compute_fused_embeddings --enable_location
```

## 特征处理流程

### 1. 地理特征处理
- **坐标标准化**: MinMax或标准化处理经纬度
- **地理网格**: 将连续坐标离散化为网格ID
- **MLP编码**: 通过神经网络编码为固定维度向量

### 2. 类别特征处理
- **频次过滤**: 过滤低频类别，归并为"其他"
- **标签编码**: LabelEncoder处理类别变量
- **嵌入层**: 通过MLP编码为向量表示

### 3. 语义特征处理
- **文本合并**: 合并站点名称、地址、覆盖区域信息
- **语义编码**: 
  - 优先使用Sentence Transformer（支持中文）
  - 回退到TF-IDF（如果Sentence Transformer不可用）
- **维度对齐**: 统一到指定维度

### 4. 特征融合
- **多模态融合**: 地理+类别+语义特征拼接
- **MLP压缩**: 融合后通过MLP压缩到目标维度
- **归一化**: L2归一化确保特征稳定性

## 增强效果

使用位置特征增强后，用户的位置表示将包含：

1. **基础序列嵌入**: 基于访问序列的Item2Vec/Node2Vec嵌入
2. **地理语义信息**: 基于坐标、地址、POI的空间语义
3. **环境上下文**: 基于基站类型、覆盖信息的环境特征
4. **空间聚类**: 基于地理网格的空间邻近性

最终的用户位置嵌入 = [基础序列嵌入] + [增强特征嵌入]

## 依赖安装

如需使用Sentence Transformer进行语义编码：

```bash
pip install sentence-transformers
```

如果不安装，系统会自动回退到TF-IDF方法。

## 文件结构

```
v2/
├── location_features.py          # 位置特征处理核心模块
├── config.py                     # 配置文件（已更新）
├── data.py                       # 数据预处理器（已扩展）
├── main.py                       # 主程序（已集成）
└── data/
    └── location_features.csv     # 您的位置特征文件
```

## 注意事项

1. **文件路径**: 确保 `location_features.csv` 在 `data/` 目录下
2. **列名匹配**: 系统会自动识别常见的列名变体
3. **内存使用**: 大量位置特征可能占用较多内存
4. **性能优化**: 可调整批次大小和特征维度以平衡性能

## 效果验证

您可以通过以下方式验证特征增强效果：

1. **对比实验**: 启用/禁用位置特征对比效果
2. **可视化**: 使用t-SNE可视化增强前后的用户嵌入
3. **下游任务**: 在分类/推荐任务上评估性能提升

通过这套完整的位置特征处理方案，您的用户嵌入将更好地利用丰富的位置语义信息，提升模型的表达能力和下游任务性能。

