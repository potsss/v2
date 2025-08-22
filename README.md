### 项目概览

本项目是多模态用户表征学习框架，三路信息共同参与：
- 行为 URL/domain（item2vec/node2vec）
- 位置（基站序列，独立 item2vec/node2vec）
- 属性（列级类别嵌入 + 数值拼接，经 MLP 压缩）

最终通过融合网络将三路向量拼接后非线性压缩为统一的用户表示；支持新用户兼容与 t-SNE 可视化。

- 入口：`v2/main.py`
- 配置：`v2/config.py`
- 预处理：`v2/data.py`
- 模型：`v2/model.py`
- 训练器：`v2/trainer.py`
- Node2Vec 工具：`v2/node2vec_utils.py`
- 新用户：`v2/new_users.py`
- 可视化：`v2/visualize.py`


### 运行逻辑（端到端）

1) 预处理（`DataPreprocessorV2.preprocess`）
- 行为：抽取 `domain` 建立 `url_to_id/id_to_url` 映射，按时间构造用户 `domain_id` 序列（支持 `weight` 重复）。
- 属性：自动识别类别/数值列，类别频次裁剪+LabelEncoder；数值缺失填充与标准化（可配）。
- 位置：识别基站列并按时间构造 `bs_id` 序列。
- 产物写入 `v2/experiments/<experiment_name>/processed_data/`。

2) 训练（`TrainerV2.train`）
- 行为：item2vec/node2vec 在用户 `domain_id` 序列上训练（进度条 `Epoch i/EPOCHS`）。
- 位置（若启用）：独立训练位置 item2vec/node2vec（进度条 `Location Epoch i/EPOCHS`），并保存位置模型。

3) 用户向量与可视化
- 行为用户向量：物品向量 mean/max 聚合 + L2 归一化（`UserEmbedding.compute`）。
- 位置用户向量：基站向量 mean 聚合 + L2 归一化。
- 可选：t-SNE 可视化输出至 `visualizations/`。

4) 融合训练与导出
- 属性向量：`AttributeEmbeddingModel` 做列级类别嵌入 + 数值拼接，经 MLP 压缩为属性向量。
- 融合模型：`UserFusionModel` 将 行为/属性/位置 向量拼接，经两层 MLP 压缩到 `FINAL_USER_EMBEDDING_DIM` 并 L2 归一化。
- 训练目标：优先“掩码属性预测（MAP）”，随机遮盖一个类别属性并预测其真实类别（交叉熵）；无类别属性回退“身份对齐（行为-融合）”。
- 导出：`compute_fused_embeddings` 输出 `fused_user_embeddings.pkl`。

5) 新用户（`new_users.py`）
- 读取新行为，过滤未知 URL（可基于训练实体集），与旧模型保持兼容；
- 输出新用户向量与兼容性报告（未知 URL 列表、已知 URL 总量等）。


### 使用到的信息

- 行为：`user_id, url, timestamp_str|timestamp[, weight]`（TSV 推荐）。
- 属性：第一列为用户 ID，后续为属性列（类别/数值自动识别与编码）。
- 位置：包含 `user_id` 与基站列（如 `base_station_id/bs_id/station_id/cell_id`），可选时间列。


### 目录结构与产物

- 数据根：`v2/data/`
  - 行为：`user_behavior.csv`
  - 属性：`user_attributes.tsv`
  - 位置：`user_base_stations.tsv`
- 实验根：`v2/experiments/<experiment_name>/`
  - `processed_data/`：`url_mappings.pkl`、`user_sequences.pkl`、`user_attributes.pkl`、`attribute_info.pkl`、`location_mappings.pkl`、`location_sequences.pkl`
  - `checkpoints/`：`latest_checkpoint.pth`、`best_model.pth`
  - `models/`：行为 `item2vec_model.pth/node2vec_model.pth`；位置 `location_item2vec_model.pth/location_node2vec_model.pth`；融合 `fusion_model.pth`；向量 `user_embeddings.pkl/fused_user_embeddings.pkl`
  - `visualizations/`：t-SNE 图


### 快速开始（脚本与模块均可）

- 预处理
```bash
python main.py --mode preprocess --experiment_name my_exp --enable_attributes --enable_location
# 或
python -m v2.main --mode preprocess --experiment_name my_exp --enable_attributes --enable_location
```

- 训练（行为 + 可选位置）
```bash
python main.py --mode train --experiment_name my_exp --model_type item2vec --enable_location
python main.py --mode train --experiment_name my_exp --model_type node2vec --enable_location
```

- 融合训练与导出
```bash
python main.py --mode train_fusion --experiment_name my_exp --enable_attributes --enable_location
python main.py --mode compute_fused_embeddings --experiment_name my_exp --enable_attributes --enable_location
```

- 可视化与导出基础行为向量
```bash
python main.py --mode visualize --experiment_name my_exp
python main.py --mode compute_embeddings --experiment_name my_exp
```

- 新用户向量
```bash
python main.py --mode compute_new_users --experiment_name my_exp
```


### 训练与融合细节

- 行为/位置：item2vec/node2vec 统一用 Skip-gram + 负采样；node2vec 通过二阶随机游走生成“句子”。
- 属性列级嵌入：每个类别列各自一个 Embedding，数值列直接拼接，经 MLP 压缩为属性向量。
- 融合：拼接 行为/属性/位置 向量，经两层 MLP 压缩输出最终用户向量（L2 归一化）。
- 融合训练目标：MAP（默认）或回退到身份对齐。


### 高级配置（`v2/config.py`）

- 通用：`EMBEDDING_DIM, WINDOW_SIZE, NEGATIVE_SAMPLES, LEARNING_RATE, EPOCHS, BATCH_SIZE, EARLY_STOPPING_PATIENCE`。
- Node2Vec：`P_PARAM, Q_PARAM, WALK_LENGTH, NUM_WALKS`。
- 属性与融合：`ENABLE_ATTRIBUTES, ATTRIBUTE_EMBEDDING_DIM, FUSION_HIDDEN_DIM, FINAL_USER_EMBEDDING_DIM, FUSION_EPOCHS, FUSION_LEARNING_RATE, FUSION_BATCH_SIZE`。
- 位置专属：`ENABLE_LOCATION, LOCATION_MODEL_TYPE, LOCATION_EMBEDDING_DIM, LOCATION_MIN_CONNECTIONS`，以及可独立覆盖的 `LOCATION_LEARNING_RATE/EPOCHS/BATCH_SIZE/WINDOW_SIZE/NEGATIVE_SAMPLES`。
- 实验路径：`v2/experiments/<experiment_name>/` 自动创建 `processed_data/checkpoints/models/visualizations` 等子目录；train 默认复用 preprocess 目录。


### 常见问题与排错

- 分隔符：默认 `sep="\t"`，若为逗号分隔，请改为 TSV 或调整 `pd.read_csv(..., sep=",")`。
- t-SNE：样本少于 3 时跳过可视化。
- 运行方式：支持脚本与模块两种方式；脚本方式已自动兼容相对导入。
- 训练显示：train 阶段会显示“Epoch”和“Location Epoch”；融合阶段显示“Fusion/Fusion MAP Epoch”。


### 许可证

内部项目示例，按需调整。



