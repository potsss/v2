"""
矩阵分解模型实现 - 使用ALS算法进行隐式反馈学习
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pickle
import os
from tqdm import tqdm
from typing import Dict, List
try:
    from .config import Config
except ImportError:
    from config import Config


class ALSMatrixFactorization:
    """
    使用ALS (Alternating Least Squares) 算法的矩阵分解模型
    专门针对隐式反馈数据优化
    """
    
    def __init__(self, factors=None, regularization=None, iterations=None, alpha=None):
        self.factors = factors or Config.MF_FACTORS
        self.regularization = regularization or Config.MF_REGULARIZATION
        self.iterations = iterations or Config.MF_ITERATIONS
        self.alpha = alpha or Config.MF_ALPHA
        
        # 模型参数
        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_bias = None
        
        # 映射关系
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def _build_confidence_matrix(self, interaction_matrix):
        """
        构建置信度矩阵 C = 1 + alpha * R
        其中R是交互矩阵，alpha是置信度参数
        """
        # 对于稀疏矩阵，我们需要显式处理
        confidence = interaction_matrix.copy().astype(np.float32)
        confidence.data = 1 + self.alpha * confidence.data
        return confidence
    
    def _als_step_user(self, user_id, interaction_matrix, item_factors, regularization):
        """
        ALS算法中更新用户因子的步骤
        """
        # 获取用户的交互记录
        user_row = interaction_matrix[user_id, :].toarray().flatten()
        
        # 找到非零交互的物品
        nonzero_items = np.where(user_row > 0)[0]
        if len(nonzero_items) == 0:
            return np.zeros(self.factors)
        
        # 获取相关的物品因子和置信度
        Y_u = item_factors[nonzero_items]  # [n_items, factors]
        c_u = 1 + self.alpha * user_row[nonzero_items]  # [n_items]
        
        # 构建系统方程: (Y_u^T * diag(c_u) * Y_u + reg * I) * x_u = Y_u^T * c_u
        YtCuY = Y_u.T.dot(np.diag(c_u)).dot(Y_u) + regularization * np.eye(self.factors)
        YtCup = Y_u.T.dot(c_u)
        
        # 求解线性方程组
        try:
            return np.linalg.solve(YtCuY, YtCup)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用最小二乘解
            return np.linalg.lstsq(YtCuY, YtCup, rcond=None)[0]
    
    def _als_step_item(self, item_id, interaction_matrix, user_factors, regularization):
        """
        ALS算法中更新物品因子的步骤
        """
        # 获取物品的交互记录
        item_col = interaction_matrix[:, item_id].toarray().flatten()
        
        # 找到非零交互的用户
        nonzero_users = np.where(item_col > 0)[0]
        if len(nonzero_users) == 0:
            return np.zeros(self.factors)
        
        # 获取相关的用户因子和置信度
        X_i = user_factors[nonzero_users]  # [n_users, factors]
        c_i = 1 + self.alpha * item_col[nonzero_users]  # [n_users]
        
        # 构建系统方程: (X_i^T * diag(c_i) * X_i + reg * I) * y_i = X_i^T * c_i
        XtCiX = X_i.T.dot(np.diag(c_i)).dot(X_i) + regularization * np.eye(self.factors)
        XtCip = X_i.T.dot(c_i)
        
        # 求解线性方程组
        try:
            return np.linalg.solve(XtCiX, XtCip)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用最小二乘解
            return np.linalg.lstsq(XtCiX, XtCip, rcond=None)[0]
    
    def fit(self, interaction_matrix, verbose=True):
        """
        训练ALS矩阵分解模型
        
        Args:
            interaction_matrix: scipy.sparse.csr_matrix 用户-物品交互矩阵
            verbose: bool 是否显示训练进度
        """
        n_users, n_items = interaction_matrix.shape
        
        # 初始化因子矩阵
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.factors))
        
        # ALS迭代训练
        for iteration in range(self.iterations):
            if verbose:
                print(f"ALS迭代 {iteration + 1}/{self.iterations}")
            
            # 更新用户因子
            for user_id in tqdm(range(n_users), desc="更新用户因子", disable=not verbose):
                self.user_factors[user_id] = self._als_step_user(
                    user_id, interaction_matrix, self.item_factors, self.regularization
                )
            
            # 更新物品因子  
            for item_id in tqdm(range(n_items), desc="更新物品因子", disable=not verbose):
                self.item_factors[item_id] = self._als_step_item(
                    item_id, interaction_matrix, self.user_factors, self.regularization
                )
            
            # 计算损失（可选，用于监控收敛）
            if verbose and iteration % 2 == 0:
                loss = self._calculate_loss(interaction_matrix)
                print(f"  损失: {loss:.6f}")
    
    def _calculate_loss(self, interaction_matrix):
        """
        计算ALS损失函数（简化版）
        """
        # 重构矩阵
        reconstructed = self.user_factors.dot(self.item_factors.T)
        
        # 计算在非零位置的重构误差
        loss = 0
        rows, cols = interaction_matrix.nonzero()
        for i, j in zip(rows, cols):
            confidence = 1 + self.alpha * interaction_matrix[i, j]
            preference = 1.0  # 隐式反馈中偏好为1
            diff = preference - reconstructed[i, j]
            loss += confidence * (diff ** 2)
        
        # 添加正则化项
        reg_loss = self.regularization * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
        )
        
        return loss + reg_loss
    
    def get_item_embeddings(self, normalize=True):
        """
        获取物品嵌入向量
        
        Args:
            normalize: bool 是否归一化
            
        Returns:
            np.ndarray: 物品嵌入矩阵 [n_items, factors]
        """
        embeddings = self.item_factors.copy()
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
            
        return embeddings
    
    def get_user_factors(self, normalize=True):
        """
        获取用户因子矩阵
        
        Args:
            normalize: bool 是否归一化
            
        Returns:
            np.ndarray: 用户因子矩阵 [n_users, factors]
        """
        factors = self.user_factors.copy()
        
        if normalize:
            norms = np.linalg.norm(factors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            factors = factors / norms
            
        return factors
    
    def save_model(self, filepath):
        """
        保存模型到文件
        """
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        从文件加载模型
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.factors = model_data['factors']
        self.regularization = model_data['regularization']
        self.iterations = model_data['iterations']
        self.alpha = model_data['alpha']
        self.user_to_idx = model_data['user_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_item = model_data['idx_to_item']


class UnifiedItemProcessor:
    """
    统一物品处理器 - 将URL和位置合并为统一的物品空间
    支持位置特征增强
    """
    
    def __init__(self, location_feature_processor=None):
        self.item_to_id = {}  # 物品 -> ID
        self.id_to_item = {}  # ID -> 物品
        self.item_types = {}  # 物品 -> 类型 ("url" or "location")
        self.location_feature_processor = location_feature_processor  # 位置特征处理器
        self.next_id = 0
    
    def add_items(self, items, item_type):
        """
        添加物品到统一词典
        
        Args:
            items: list 物品列表
            item_type: str 物品类型 ("url" or "location")
        """
        for item in items:
            if item not in self.item_to_id:
                self.item_to_id[item] = self.next_id
                self.id_to_item[self.next_id] = item
                self.item_types[item] = item_type
                self.next_id += 1
    
    def build_interaction_matrix(self, user_sequences, location_sequences=None):
        """
        构建用户-物品交互矩阵
        
        Args:
            user_sequences: dict {user_id: [url_sequence]}
            location_sequences: dict {user_id: [location_sequence]} (可选)
            
        Returns:
            tuple: (interaction_matrix, user_to_idx, idx_to_user)
        """
        # 构建用户映射
        all_users = set(user_sequences.keys())
        if location_sequences:
            all_users.update(location_sequences.keys())
        
        user_to_idx = {user: idx for idx, user in enumerate(sorted(all_users))}
        idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        
        # 构建交互数据
        interactions = []
        
        # 处理URL交互
        for user, sequence in user_sequences.items():
            user_idx = user_to_idx[user]
            for item in sequence:
                if item in self.item_to_id:
                    item_idx = self.item_to_id[item]
                    interactions.append((user_idx, item_idx))
        
        # 处理位置交互
        if location_sequences:
            for user, sequence in location_sequences.items():
                user_idx = user_to_idx[user]
                for item in sequence:
                    if item in self.item_to_id:
                        item_idx = self.item_to_id[item]
                        interactions.append((user_idx, item_idx))
        
        # 统计交互次数
        interaction_counts = {}
        for user_idx, item_idx in interactions:
            key = (user_idx, item_idx)
            interaction_counts[key] = interaction_counts.get(key, 0) + 1
        
        # 构建稀疏矩阵
        rows, cols, data = [], [], []
        min_interactions = getattr(Config, 'MF_MIN_INTERACTIONS', 1)  # 默认值为1
        for (user_idx, item_idx), count in interaction_counts.items():
            if count >= min_interactions:  # 过滤低频交互
                rows.append(user_idx)
                cols.append(item_idx)
                data.append(count)
        
        n_users = len(user_to_idx)
        n_items = len(self.item_to_id)
        
        interaction_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(n_users, n_items),
            dtype=np.float32
        )
        
        return interaction_matrix, user_to_idx, idx_to_user
    
    def save_mappings(self, filepath):
        """
        保存物品映射关系
        """
        mappings = {
            'item_to_id': self.item_to_id,
            'id_to_item': self.id_to_item,
            'item_types': self.item_types,
            'next_id': self.next_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(mappings, f)
    
    def load_mappings(self, filepath):
        """
        加载物品映射关系
        """
        with open(filepath, 'rb') as f:
            mappings = pickle.load(f)
        
        self.item_to_id = mappings['item_to_id']
        self.id_to_item = mappings['id_to_item']
        self.item_types = mappings['item_types']
        self.next_id = mappings['next_id']
        # 位置特征处理器需要单独设置
        if 'location_feature_processor' in mappings:
            self.location_feature_processor = mappings['location_feature_processor']


class MatrixFactorizationUserEmbedding:
    """
    基于矩阵分解的用户向量生成器
    """
    
    def __init__(self, model, item_processor, user_to_idx):
        self.model = model
        self.item_processor = item_processor
        self.user_to_idx = user_to_idx
        self.user_embeddings = {}
    
    def compute_from_interactions(self, user_sequences, location_sequences=None, aggregation=None):
        """
        从用户交互序列计算用户向量
        
        Args:
            user_sequences: dict {user_id: [url_sequence]}
            location_sequences: dict {user_id: [location_sequence]} (可选)
            aggregation: str 聚合方式 ("avg", "weighted_avg")
            
        Returns:
            dict: {user_id: embedding_vector}
        """
        if aggregation is None:
            aggregation = Config.MF_USER_AGGREGATION
        
        item_embeddings = self.model.get_item_embeddings(normalize=True)
        
        for user_id in self.user_to_idx.keys():
            user_items = []
            user_weights = []
            
            # 收集URL交互（基于序列中的重复次数计算权重）
            if user_id in user_sequences:
                from collections import Counter
                item_counts = Counter(user_sequences[user_id])
                for item, count in item_counts.items():
                    if item in self.item_processor.item_to_id:
                        item_idx = self.item_processor.item_to_id[item]
                        user_items.append(item_idx)
                        user_weights.append(float(count))  # 使用访问频次作为权重
            
            # 收集位置交互（基于序列中的重复次数计算权重）
            if location_sequences and user_id in location_sequences:
                from collections import Counter
                item_counts = Counter(location_sequences[user_id])
                for item, count in item_counts.items():
                    if item in self.item_processor.item_to_id:
                        item_idx = self.item_processor.item_to_id[item]
                        user_items.append(item_idx)
                        user_weights.append(float(count))
            
            if not user_items:
                continue
            
            # 获取物品向量
            item_vecs = item_embeddings[user_items]
            
            # 聚合用户向量
            if aggregation == "weighted_avg" and len(user_weights) > 0:
                weights = np.array(user_weights)
                weights = weights / weights.sum()  # 归一化权重
                user_vec = np.average(item_vecs, axis=0, weights=weights)
            else:
                user_vec = np.mean(item_vecs, axis=0)
            
            # 归一化
            norm = np.linalg.norm(user_vec)
            if norm > 0:
                user_vec = user_vec / norm
            
            self.user_embeddings[user_id] = user_vec
        
        return self.user_embeddings
    
    def compute_from_factors(self):
        """
        直接从矩阵分解的用户因子获取用户向量
        
        Returns:
            dict: {user_id: embedding_vector}
        """
        user_factors = self.model.get_user_factors(normalize=True)
        
        for user_id, user_idx in self.user_to_idx.items():
            self.user_embeddings[user_id] = user_factors[user_idx]
        
        return self.user_embeddings
    
    def compute_from_sequences(self, user_sequences: Dict[str, List], item_mappings: Dict) -> Dict[str, np.ndarray]:
        """
        基于用户序列计算用户嵌入（用于位置特征增强）
        
        Args:
            user_sequences: dict {user_id: [item_sequence]}
            item_mappings: dict 物品映射关系
            
        Returns:
            dict: {user_id: embedding_vector}
        """
        item_embeddings = self.model.get_item_embeddings(normalize=True)
        user_embeddings = {}
        
        # 构建物品ID到矩阵索引的映射
        bs_to_id = item_mappings.get("bs_to_id", {})
        
        for user_id, sequence in user_sequences.items():
            if not sequence:
                continue
            
            # 收集该用户的物品向量和权重
            vectors = []
            weights = []
            from collections import Counter
            item_counts = Counter(sequence)
            
            for item_id, count in item_counts.items():
                # 将位置ID转换为基站ID，再转换为矩阵索引
                if item_id in bs_to_id:
                    matrix_idx = bs_to_id[item_id]
                    if matrix_idx < len(item_embeddings):
                        vectors.append(item_embeddings[matrix_idx])
                        weights.append(float(count))
            
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


class EnhancedMatrixFactorization(ALSMatrixFactorization):
    """
    增强的矩阵分解模型 - 支持位置特征增强
    """
    
    def __init__(self, location_feature_processor=None, **kwargs):
        super().__init__(**kwargs)
        self.location_feature_processor = location_feature_processor
        self.location_item_features = {}  # 位置物品ID -> 特征向量
        
    def _extract_location_features(self, item_processor):
        """提取位置物品的特征向量"""
        if not self.location_feature_processor:
            return
            
        print("提取位置物品特征...")
        location_count = 0
        
        for item, item_id in item_processor.item_to_id.items():
            if item_processor.item_types.get(item) == "location":
                # item是位置基站ID，需要获取其特征
                features = self.location_feature_processor.get_location_features(str(item))
                
                # 拼接所有特征类型
                feature_vector = []
                if features['geographic'].size > 0:
                    feature_vector.extend(features['geographic'])
                if features['categorical'].size > 0:
                    feature_vector.extend(features['categorical'])
                if features['textual'].size > 0:
                    feature_vector.extend(features['textual'])
                
                if feature_vector:
                    self.location_item_features[item_id] = np.array(feature_vector)
                    location_count += 1
        
        print(f"提取了 {location_count} 个位置物品的特征")
    
    def fit(self, interaction_matrix, item_processor=None, verbose=False):
        """
        训练增强的矩阵分解模型
        """
        # 提取位置特征
        if item_processor and self.location_feature_processor:
            self._extract_location_features(item_processor)
        
        # 调用父类的训练方法
        super().fit(interaction_matrix, verbose=verbose)
        
        # 如果有位置特征，进行特征增强
        if self.location_item_features:
            self._enhance_location_embeddings()
    
    def _enhance_location_embeddings(self):
        """使用位置特征增强位置物品的嵌入向量"""
        print("使用位置特征增强嵌入向量...")
        
        if not self.location_item_features:
            return
        
        # 获取位置特征维度
        feature_dims = []
        for features in self.location_item_features.values():
            feature_dims.append(len(features))
        
        if not feature_dims:
            return
            
        max_feature_dim = max(feature_dims)
        
        # 创建特征矩阵
        location_features = np.zeros((len(self.location_item_features), max_feature_dim))
        location_item_ids = list(self.location_item_features.keys())
        
        for i, item_id in enumerate(location_item_ids):
            features = self.location_item_features[item_id]
            location_features[i, :len(features)] = features
        
        # 特征标准化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        location_features_scaled = scaler.fit_transform(location_features)
        
        # 使用PCA或简单的线性变换将特征映射到嵌入空间
        feature_to_embedding_dim = min(self.factors, max_feature_dim)
        
        # 简单的线性变换矩阵
        np.random.seed(42)
        transform_matrix = np.random.normal(0, 0.1, (max_feature_dim, feature_to_embedding_dim))
        
        # 计算特征增强的嵌入
        feature_embeddings = location_features_scaled.dot(transform_matrix)
        
        # 与原有嵌入进行融合
        alpha = getattr(Config, 'MF_FEATURE_ENHANCEMENT_WEIGHT', 0.3)  # 特征增强权重
        for i, item_id in enumerate(location_item_ids):
            if item_id < len(self.item_factors):
                # 加权融合：原嵌入 + 特征嵌入
                original_emb = self.item_factors[item_id][:feature_to_embedding_dim]
                enhanced_emb = (1 - alpha) * original_emb + alpha * feature_embeddings[i]
                self.item_factors[item_id][:feature_to_embedding_dim] = enhanced_emb
        
        print(f"增强了 {len(location_item_ids)} 个位置物品的嵌入向量")
    
    def get_enhanced_item_embeddings(self, normalize=True):
        """获取增强后的物品嵌入"""
        embeddings = self.get_item_embeddings(normalize=normalize)
        return embeddings
    
    def save_model(self, filepath):
        """保存增强模型"""
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'factors': self.factors,
            'regularization': self.regularization,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'location_item_features': self.location_item_features,  # 保存位置特征
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """加载增强模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.factors = model_data['factors']
        self.regularization = model_data['regularization']
        self.iterations = model_data['iterations']
        self.alpha = model_data['alpha']
        self.user_to_idx = model_data['user_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_item = model_data['idx_to_item']
        
        # 加载位置特征（如果存在）
        if 'location_item_features' in model_data:
            self.location_item_features = model_data['location_item_features']
