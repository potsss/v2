"""
位置特征处理模块
处理丰富的位置特征信息，包括地理坐标、语义信息和类别特征
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers未安装，将使用TF-IDF作为文本特征替代")

try:
    from .config import Config
except ImportError:
    from config import Config


class LocationFeatureProcessor:
    """位置特征处理器"""
    
    def __init__(self):
        self.location_features = {}  # base_station_id -> features dict
        self.geographic_scaler = None
        self.categorical_encoders = {}
        self.text_encoder = None
        self.feature_stats = {}
        
        # 特征列定义
        self.geographic_cols = ['longitude', 'latitude', 'longitude_gps', 'latitude_gps']
        self.categorical_cols = ['city_name', 'county_name', 'town_name', 'site_net_type_name', 
                                'cover_type', 'cover_info', 'along_railway', 'along_freeway']
        self.text_cols = ['site_name', 'site_address', 'cover_area']
        
    def load_and_process_features(self, features_path: str = None) -> bool:
        """加载并处理位置特征文件"""
        if features_path is None:
            features_path = Config.LOCATION_FEATURES_PATH
            
        if not os.path.exists(features_path):
            print(f"位置特征文件不存在: {features_path}")
            return False
            
        print(f"加载位置特征文件: {features_path}")
        try:
            # 读取CSV文件
            df = pd.read_csv(features_path, sep='\t', low_memory=False)
            print(f"成功加载 {len(df)} 条位置特征记录")
            
            # 数据清洗
            df = self._clean_data(df)
            print(f"清洗后剩余 {len(df)} 条记录")
            
            # 处理各类特征
            self._process_geographic_features(df)
            self._process_categorical_features(df)
            self._process_text_features(df)
            
            # 构建特征字典
            self._build_feature_dict(df)
            
            print("位置特征处理完成")
            return True
            
        except Exception as e:
            print(f"处理位置特征时出错: {e}")
            return False
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 确保base_station_id列存在
        if 'base_station_id' not in df.columns:
            raise ValueError("位置特征文件必须包含 'base_station_id' 列")
        
        # 去重
        df = df.drop_duplicates(subset=['base_station_id'])
        
        # 处理缺失值
        for col in self.geographic_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 填充文本列的缺失值
        for col in self.text_cols:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        # 填充类别列的缺失值
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('未知').astype(str)
        
        return df
    
    def _process_geographic_features(self, df: pd.DataFrame):
        """处理地理坐标特征"""
        print("处理地理坐标特征...")
        
        # 收集所有可用的坐标列
        available_geo_cols = [col for col in self.geographic_cols if col in df.columns]
        
        if not available_geo_cols:
            print("未找到地理坐标列")
            return
        
        # 提取坐标数据
        geo_data = df[available_geo_cols].copy()
        
        # 去除无效坐标
        geo_data = geo_data.dropna()
        
        if len(geo_data) == 0:
            print("未找到有效的地理坐标数据")
            return
        
        # 坐标标准化
        if Config.COORDINATE_NORMALIZATION == "standard":
            self.geographic_scaler = StandardScaler()
        elif Config.COORDINATE_NORMALIZATION == "minmax":
            self.geographic_scaler = MinMaxScaler()
        else:
            self.geographic_scaler = None
        
        if self.geographic_scaler:
            geo_data_scaled = self.geographic_scaler.fit_transform(geo_data)
            # 将标准化后的数据放回DataFrame
            for i, col in enumerate(available_geo_cols):
                df[f"{col}_scaled"] = np.nan
                df.loc[geo_data.index, f"{col}_scaled"] = geo_data_scaled[:, i]
        
        # 计算地理网格ID（用于离散化坐标）
        if 'longitude' in df.columns and 'latitude' in df.columns:
            self._compute_geographic_grid(df)
        
        print(f"处理了 {len(available_geo_cols)} 个地理坐标特征")
    
    def _compute_geographic_grid(self, df: pd.DataFrame):
        """计算地理网格ID"""
        valid_coords = df[['longitude', 'latitude']].dropna()
        
        if len(valid_coords) == 0:
            return
        
        # 计算坐标范围
        lon_min, lon_max = valid_coords['longitude'].min(), valid_coords['longitude'].max()
        lat_min, lat_max = valid_coords['latitude'].min(), valid_coords['latitude'].max()
        
        # 计算网格步长
        grid_size = Config.COORDINATE_GRID_SIZE
        lon_step = (lon_max - lon_min) / grid_size
        lat_step = (lat_max - lat_min) / grid_size
        
        # 计算网格ID
        df['geo_grid_x'] = np.nan
        df['geo_grid_y'] = np.nan
        df['geo_grid_id'] = np.nan
        
        valid_indices = valid_coords.index
        df.loc[valid_indices, 'geo_grid_x'] = ((valid_coords['longitude'] - lon_min) / lon_step).astype(int)
        df.loc[valid_indices, 'geo_grid_y'] = ((valid_coords['latitude'] - lat_min) / lat_step).astype(int)
        df.loc[valid_indices, 'geo_grid_id'] = (df.loc[valid_indices, 'geo_grid_x'] * grid_size + 
                                               df.loc[valid_indices, 'geo_grid_y']).astype(int)
        
        print(f"生成了 {len(valid_coords)} 个地理网格ID")
    
    def _process_categorical_features(self, df: pd.DataFrame):
        """处理类别特征"""
        print("处理类别特征...")
        
        available_cat_cols = [col for col in self.categorical_cols if col in df.columns]
        
        for col in available_cat_cols:
            # 统计类别频次
            value_counts = df[col].value_counts()
            
            # 过滤低频类别
            valid_values = value_counts[value_counts >= Config.LOCATION_CATEGORICAL_MIN_FREQ].index
            df[col] = df[col].apply(lambda x: x if x in valid_values else '其他')
            
            # 标签编码
            encoder = LabelEncoder()
            df[f"{col}_encoded"] = encoder.fit_transform(df[col])
            self.categorical_encoders[col] = encoder
            
            print(f"  {col}: {len(encoder.classes_)} 个类别")
        
        print(f"处理了 {len(available_cat_cols)} 个类别特征")
    
    def _process_text_features(self, df: pd.DataFrame):
        """处理文本特征"""
        print("处理文本特征...")
        
        available_text_cols = [col for col in self.text_cols if col in df.columns]
        
        if not available_text_cols:
            print("未找到文本特征列")
            return
        
        # 合并所有文本列
        text_data = []
        for idx, row in df.iterrows():
            combined_text = ' '.join([str(row[col]) for col in available_text_cols if pd.notna(row[col])])
            text_data.append(combined_text.strip())
        
        df['combined_text'] = text_data
        
        # 选择文本编码方式
        if SENTENCE_TRANSFORMERS_AVAILABLE and Config.ENABLE_LOCATION_FEATURES:
            self._encode_text_with_sentence_transformer(df)
        else:
            self._encode_text_with_tfidf(df)
        
        print(f"处理了 {len(available_text_cols)} 个文本特征")
    
    def _encode_text_with_sentence_transformer(self, df: pd.DataFrame):
        """使用Sentence Transformer编码文本"""
        try:
            print("使用Sentence Transformer编码文本...")
            model = SentenceTransformer(Config.LOCATION_TEXT_EMBEDDING_MODEL)
            
            # 过滤空文本
            valid_texts = [text for text in df['combined_text'] if text.strip()]
            
            if valid_texts:
                embeddings = model.encode(valid_texts, 
                                        max_seq_length=Config.LOCATION_TEXT_MAX_LENGTH,
                                        show_progress_bar=True)
                
                # 将嵌入向量保存到DataFrame
                embedding_dim = embeddings.shape[1]
                for i in range(embedding_dim):
                    df[f'text_emb_{i}'] = 0.0
                
                valid_idx = 0
                for idx, text in enumerate(df['combined_text']):
                    if text.strip():
                        for i in range(embedding_dim):
                            df.at[idx, f'text_emb_{i}'] = embeddings[valid_idx][i]
                        valid_idx += 1
                
                self.text_encoder = {'type': 'sentence_transformer', 
                                   'model_name': Config.LOCATION_TEXT_EMBEDDING_MODEL,
                                   'embedding_dim': embedding_dim}
                
                print(f"生成了 {embedding_dim} 维的文本嵌入")
            
        except Exception as e:
            print(f"Sentence Transformer编码失败，回退到TF-IDF: {e}")
            self._encode_text_with_tfidf(df)
    
    def _encode_text_with_tfidf(self, df: pd.DataFrame):
        """使用TF-IDF编码文本"""
        print("使用TF-IDF编码文本...")
        
        # 过滤空文本
        valid_texts = [text if text.strip() else '空文本' for text in df['combined_text']]
        
        # TF-IDF向量化
        vectorizer = TfidfVectorizer(
            max_features=Config.LOCATION_SEMANTIC_EMBEDDING_DIM,
            stop_words=None,  # 保留中文停用词处理
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        
        # 将TF-IDF特征保存到DataFrame
        feature_names = vectorizer.get_feature_names_out()
        tfidf_dense = tfidf_matrix.toarray()
        
        for i, feature_name in enumerate(feature_names):
            df[f'tfidf_{i}'] = tfidf_dense[:, i]
        
        self.text_encoder = {'type': 'tfidf', 
                           'vectorizer': vectorizer,
                           'feature_dim': len(feature_names)}
        
        print(f"生成了 {len(feature_names)} 维的TF-IDF特征")
    
    def _build_feature_dict(self, df: pd.DataFrame):
        """构建特征字典"""
        print("构建位置特征字典...")
        
        for idx, row in df.iterrows():
            bs_id = str(row['base_station_id'])
            features = {}
            
            # 地理特征
            geo_features = []
            for col in df.columns:
                if any(geo_col in col for geo_col in ['longitude', 'latitude', 'geo_grid']):
                    if pd.notna(row[col]):
                        geo_features.append(float(row[col]))
                    else:
                        geo_features.append(0.0)
            features['geographic'] = np.array(geo_features) if geo_features else np.array([])
            
            # 类别特征
            cat_features = []
            for col in df.columns:
                if col.endswith('_encoded'):
                    cat_features.append(int(row[col]))
            features['categorical'] = np.array(cat_features) if cat_features else np.array([])
            
            # 文本特征
            text_features = []
            for col in df.columns:
                if col.startswith('text_emb_') or col.startswith('tfidf_'):
                    text_features.append(float(row[col]))
            features['textual'] = np.array(text_features) if text_features else np.array([])
            
            self.location_features[bs_id] = features
        
        # 记录特征统计信息
        self.feature_stats = {
            'total_locations': len(self.location_features),
            'geographic_dim': len(list(self.location_features.values())[0]['geographic']) if self.location_features else 0,
            'categorical_dim': len(list(self.location_features.values())[0]['categorical']) if self.location_features else 0,
            'textual_dim': len(list(self.location_features.values())[0]['textual']) if self.location_features else 0
        }
        
        print(f"构建了 {self.feature_stats['total_locations']} 个位置的特征字典")
        print(f"特征维度 - 地理: {self.feature_stats['geographic_dim']}, "
              f"类别: {self.feature_stats['categorical_dim']}, "
              f"文本: {self.feature_stats['textual_dim']}")
    
    def get_location_features(self, base_station_id: str) -> Dict[str, np.ndarray]:
        """获取指定基站的特征"""
        bs_id = str(base_station_id)
        if bs_id in self.location_features:
            return self.location_features[bs_id]
        else:
            # 返回零特征
            return {
                'geographic': np.zeros(self.feature_stats.get('geographic_dim', 0)),
                'categorical': np.zeros(self.feature_stats.get('categorical_dim', 0)),
                'textual': np.zeros(self.feature_stats.get('textual_dim', 0))
            }
    
    def save_processed_features(self, save_path: str):
        """保存处理后的特征"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_data = {
            'location_features': self.location_features,
            'geographic_scaler': self.geographic_scaler,
            'categorical_encoders': self.categorical_encoders,
            'text_encoder': self.text_encoder,
            'feature_stats': self.feature_stats
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"位置特征已保存到: {save_path}")
    
    def load_processed_features(self, load_path: str) -> bool:
        """加载处理后的特征"""
        if not os.path.exists(load_path):
            return False
        
        try:
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.location_features = save_data['location_features']
            self.geographic_scaler = save_data['geographic_scaler']
            self.categorical_encoders = save_data['categorical_encoders']
            self.text_encoder = save_data['text_encoder']
            self.feature_stats = save_data['feature_stats']
            
            print(f"从 {load_path} 加载了 {len(self.location_features)} 个位置的特征")
            return True
            
        except Exception as e:
            print(f"加载位置特征失败: {e}")
            return False


class LocationFeatureEmbedding(nn.Module):
    """位置特征嵌入模型"""
    
    def __init__(self, geographic_dim: int, categorical_dim: int, textual_dim: int):
        super().__init__()
        
        self.geographic_dim = geographic_dim
        self.categorical_dim = categorical_dim
        self.textual_dim = textual_dim
        
        # 地理特征处理
        if geographic_dim > 0:
            self.geographic_mlp = nn.Sequential(
                nn.Linear(geographic_dim, Config.LOCATION_GEOGRAPHIC_EMBEDDING_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(Config.LOCATION_GEOGRAPHIC_EMBEDDING_DIM, Config.LOCATION_GEOGRAPHIC_EMBEDDING_DIM)
            )
        
        # 类别特征处理
        if categorical_dim > 0:
            self.categorical_mlp = nn.Sequential(
                nn.Linear(categorical_dim, Config.LOCATION_CATEGORICAL_EMBEDDING_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(Config.LOCATION_CATEGORICAL_EMBEDDING_DIM, Config.LOCATION_CATEGORICAL_EMBEDDING_DIM)
            )
        
        # 文本特征处理
        if textual_dim > 0:
            self.textual_mlp = nn.Sequential(
                nn.Linear(textual_dim, Config.LOCATION_SEMANTIC_EMBEDDING_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(Config.LOCATION_SEMANTIC_EMBEDDING_DIM, Config.LOCATION_SEMANTIC_EMBEDDING_DIM)
            )
        
        # 特征融合
        fusion_input_dim = 0
        if geographic_dim > 0:
            fusion_input_dim += Config.LOCATION_GEOGRAPHIC_EMBEDDING_DIM
        if categorical_dim > 0:
            fusion_input_dim += Config.LOCATION_CATEGORICAL_EMBEDDING_DIM
        if textual_dim > 0:
            fusion_input_dim += Config.LOCATION_SEMANTIC_EMBEDDING_DIM
        
        if fusion_input_dim > 0:
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim, Config.LOCATION_FEATURE_FUSION_DIM),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(Config.LOCATION_FEATURE_FUSION_DIM, Config.LOCATION_FEATURE_EMBEDDING_DIM)
            )
    
    def forward(self, geographic_features: torch.Tensor, 
                categorical_features: torch.Tensor, 
                textual_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        embeddings = []
        
        # 处理地理特征
        if self.geographic_dim > 0 and geographic_features is not None:
            geo_emb = self.geographic_mlp(geographic_features)
            embeddings.append(geo_emb)
        
        # 处理类别特征
        if self.categorical_dim > 0 and categorical_features is not None:
            cat_emb = self.categorical_mlp(categorical_features)
            embeddings.append(cat_emb)
        
        # 处理文本特征
        if self.textual_dim > 0 and textual_features is not None:
            text_emb = self.textual_mlp(textual_features)
            embeddings.append(text_emb)
        
        # 特征融合
        if embeddings:
            fused_features = torch.cat(embeddings, dim=-1)
            return self.fusion_mlp(fused_features)
        else:
            # 如果没有特征，返回零向量
            batch_size = geographic_features.size(0) if geographic_features is not None else 1
            return torch.zeros(batch_size, Config.LOCATION_FEATURE_EMBEDDING_DIM, 
                             device=geographic_features.device if geographic_features is not None else 'cpu')


class EnhancedLocationEmbedding:
    """增强的位置嵌入，结合序列嵌入和特征嵌入"""
    
    def __init__(self, location_model, location_mappings, feature_processor: LocationFeatureProcessor):
        self.location_model = location_model
        self.location_mappings = location_mappings
        self.feature_processor = feature_processor
        
        # 创建特征嵌入模型
        stats = feature_processor.feature_stats
        self.feature_embedding_model = LocationFeatureEmbedding(
            geographic_dim=stats.get('geographic_dim', 0),
            categorical_dim=stats.get('categorical_dim', 0),
            textual_dim=stats.get('textual_dim', 0)
        ).to(Config.DEVICE_OBJ)
    
    def compute_enhanced_user_location_embeddings(self, user_location_sequences: Dict[str, List]) -> Dict[str, np.ndarray]:
        """计算增强的用户位置嵌入"""
        print("计算增强的用户位置嵌入...")
        
        # 获取基础序列嵌入
        base_embeddings = self._get_base_sequence_embeddings(user_location_sequences)
        print(f"基础序列嵌入用户数: {len(base_embeddings)}")
        
        # 获取特征增强嵌入
        feature_embeddings = self._get_feature_enhanced_embeddings(user_location_sequences)
        print(f"特征增强嵌入用户数: {len(feature_embeddings)}")
        
        # 融合两种嵌入
        enhanced_embeddings = {}
        
        # 优先使用基础嵌入，如果有特征嵌入则进行融合
        for user_id in user_location_sequences:
            if user_id in base_embeddings:
                base_emb = base_embeddings[user_id]
                
                if user_id in feature_embeddings:
                    # 有特征嵌入，进行融合
                    feature_emb = feature_embeddings[user_id]
                    
                    # 拼接方式
                    enhanced_emb = np.concatenate([base_emb, feature_emb])
                    
                    # 归一化
                    norm = np.linalg.norm(enhanced_emb)
                    if norm > 0:
                        enhanced_emb = enhanced_emb / norm
                    
                    enhanced_embeddings[user_id] = enhanced_emb
                else:
                    # 只有基础嵌入，直接使用
                    enhanced_embeddings[user_id] = base_emb
                    
            elif user_id in feature_embeddings:
                # 只有特征嵌入，直接使用
                enhanced_embeddings[user_id] = feature_embeddings[user_id]
        
        print(f"计算了 {len(enhanced_embeddings)} 个用户的增强位置嵌入")
        return enhanced_embeddings
    
    def _get_base_sequence_embeddings(self, user_location_sequences: Dict[str, List]) -> Dict[str, np.ndarray]:
        """获取基础序列嵌入"""
        # 检查模型类型
        if hasattr(self.location_model, 'get_embeddings'):
            # Item2Vec/Node2Vec模型
            try:
                from .model import UserEmbedding
            except ImportError:
                from model import UserEmbedding
            
            # 转换位置映射格式以适配UserEmbedding类
            adapted_mappings = {
                "id_to_url": self.location_mappings.get("id_to_bs", {}),
                "url_to_id": self.location_mappings.get("bs_to_id", {})
            }
            
            # 使用现有的UserEmbedding类计算基础嵌入
            user_embedding = UserEmbedding(self.location_model, user_location_sequences, adapted_mappings)
            return user_embedding.compute()
        
        elif hasattr(self.location_model, 'get_item_embeddings'):
            # 矩阵分解模型 - 直接计算用户嵌入，不使用MatrixFactorizationUserEmbedding类
            item_embeddings = self.location_model.get_item_embeddings(normalize=True)
            user_embeddings = {}
            
            # 加载统一物品映射
            unified_mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "unified_item_mappings.pkl")
            unified_item_to_id = {}
            if os.path.exists(unified_mappings_path):
                import pickle
                with open(unified_mappings_path, 'rb') as f:
                    unified_mappings = pickle.load(f)
                unified_item_to_id = unified_mappings.get('item_to_id', {})
                print(f"加载了统一物品映射，包含{len(unified_item_to_id)}个物品")
            else:
                print(f"警告: 统一物品映射文件不存在: {unified_mappings_path}")
            
            # 构建映射关系
            id_to_bs = self.location_mappings.get("id_to_bs", {})  # 内部ID -> 基站ID
            print(f"位置映射关系: id_to_bs有{len(id_to_bs)}个映射")
            
            for user_id, sequence in user_location_sequences.items():
                if not sequence:
                    continue
                
                # 收集该用户的物品向量和权重
                vectors = []
                weights = []
                from collections import Counter
                item_counts = Counter(sequence)
                
                for internal_id, count in item_counts.items():
                    # 将内部ID转换为基站ID
                    if internal_id in id_to_bs:
                        bs_id = id_to_bs[internal_id]
                        
                        # 在统一物品空间中查找基站ID对应的索引
                        if bs_id in unified_item_to_id:
                            matrix_idx = unified_item_to_id[bs_id]
                            if matrix_idx < len(item_embeddings):
                                vectors.append(item_embeddings[matrix_idx])
                                weights.append(float(count))
                            else:
                                print(f"警告: 基站ID {bs_id} 的矩阵索引 {matrix_idx} 超出范围 {len(item_embeddings)}")
                        else:
                            print(f"警告: 基站ID {bs_id} 不在统一物品映射中")
                    else:
                        print(f"警告: 内部ID {internal_id} 不在id_to_bs映射中")
                
                if vectors:
                    import numpy as np
                    vectors = np.array(vectors)
                    weights = np.array(weights)
                    weights = weights / weights.sum()  # 归一化权重
                    
                    # 加权平均
                    user_vector = np.average(vectors, axis=0, weights=weights)
                    
                    # 归一化
                    norm = np.linalg.norm(user_vector)
                    if norm > 0:
                        user_vector = user_vector / norm
                    
                    user_embeddings[user_id] = user_vector
            
            return user_embeddings
        
        else:
            print("警告: 无法识别的位置模型类型，返回空嵌入")
            return {}
    
    def _get_feature_enhanced_embeddings(self, user_location_sequences: Dict[str, List]) -> Dict[str, np.ndarray]:
        """获取特征增强嵌入"""
        feature_embeddings = {}
        
        self.feature_embedding_model.eval()
        with torch.no_grad():
            for user_id, sequence in user_location_sequences.items():
                if not sequence:
                    continue
                
                # 收集用户访问位置的特征
                location_features = []
                location_weights = []
                
                from collections import Counter
                location_counts = Counter(sequence)
                
                for location_id, count in location_counts.items():
                    # 获取位置特征
                    if location_id in self.location_mappings.get('id_to_bs', {}):
                        bs_id = self.location_mappings['id_to_bs'][location_id]
                        features = self.feature_processor.get_location_features(bs_id)
                        location_features.append(features)
                        location_weights.append(count)
                
                if location_features:
                    # 准备特征张量
                    geo_features = []
                    cat_features = []
                    text_features = []
                    
                    for features in location_features:
                        geo_features.append(features['geographic'])
                        cat_features.append(features['categorical'])
                        text_features.append(features['textual'])
                    
                    # 转换为张量
                    if geo_features[0].size > 0:
                        geo_tensor = torch.tensor(np.stack(geo_features), dtype=torch.float32).to(Config.DEVICE_OBJ)
                    else:
                        geo_tensor = None
                    
                    if cat_features[0].size > 0:
                        cat_tensor = torch.tensor(np.stack(cat_features), dtype=torch.float32).to(Config.DEVICE_OBJ)
                    else:
                        cat_tensor = None
                    
                    if text_features[0].size > 0:
                        text_tensor = torch.tensor(np.stack(text_features), dtype=torch.float32).to(Config.DEVICE_OBJ)
                    else:
                        text_tensor = None
                    
                    # 计算特征嵌入
                    feature_embs = self.feature_embedding_model(geo_tensor, cat_tensor, text_tensor)
                    
                    # 加权平均
                    weights = np.array(location_weights, dtype=np.float32)
                    weights = weights / weights.sum()
                    weights_tensor = torch.tensor(weights).to(Config.DEVICE_OBJ).unsqueeze(-1)
                    
                    user_feature_emb = (feature_embs * weights_tensor).sum(dim=0)
                    
                    # 归一化
                    user_feature_emb = user_feature_emb / torch.norm(user_feature_emb)
                    
                    feature_embeddings[user_id] = user_feature_emb.cpu().numpy()
        
        return feature_embeddings

