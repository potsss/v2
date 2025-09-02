#!/usr/bin/env python3
"""
用户扩量（Look-alike Modeling）系统
基于多模态用户向量和种子用户进行相似用户发现
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# 可选的深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch未安装，将使用传统机器学习方法")


class UserExpansionDataset:
    """用户扩量数据集管理器"""
    
    def __init__(self, embeddings_path: str, seed_users_path: str):
        """
        初始化数据集
        
        Args:
            embeddings_path: 综合用户向量文件路径
            seed_users_path: 种子用户名单文件路径
        """
        self.embeddings_path = embeddings_path
        self.seed_users_path = seed_users_path
        self.user_embeddings = {}
        self.seed_users = set()
        self.features = None
        self.labels = None
        self.user_ids = None
        
    def load_data(self):
        """加载用户向量和种子用户数据"""
        print("加载用户向量数据...")
        
        # 加载用户向量
        try:
            with open(self.embeddings_path, 'rb') as f:
                self.user_embeddings = pickle.load(f)
            print(f"成功加载 {len(self.user_embeddings)} 个用户的向量数据")
        except Exception as e:
            raise ValueError(f"无法加载用户向量文件: {e}")
        
        # 加载种子用户
        try:
            if self.seed_users_path.endswith('.csv'):
                seed_df = pd.read_csv(self.seed_users_path)
                # 假设第一列是用户ID
                self.seed_users = set(seed_df.iloc[:, 0].astype(str).tolist())
            elif self.seed_users_path.endswith('.txt'):
                with open(self.seed_users_path, 'r', encoding='utf-8') as f:
                    self.seed_users = set(line.strip() for line in f if line.strip())
            else:
                # 尝试pickle格式
                with open(self.seed_users_path, 'rb') as f:
                    seed_data = pickle.load(f)
                    if isinstance(seed_data, (list, set)):
                        self.seed_users = set(str(uid) for uid in seed_data)
                    else:
                        raise ValueError("种子用户文件格式不支持")
            
            print(f"成功加载 {len(self.seed_users)} 个种子用户")
        except Exception as e:
            raise ValueError(f"无法加载种子用户文件: {e}")
        
        # 检查数据一致性
        available_seed_users = self.seed_users.intersection(set(self.user_embeddings.keys()))
        print(f"在向量数据中找到 {len(available_seed_users)} 个种子用户")
        
        if len(available_seed_users) == 0:
            raise ValueError("种子用户与向量数据没有交集，请检查用户ID格式")
        
        self.seed_users = available_seed_users
        return True
    
    def prepare_features(self, use_feature_engineering=True):
        """准备训练特征"""
        print("准备训练特征...")
        
        features_list = []
        labels_list = []
        user_ids_list = []
        
        for user_id, user_data in self.user_embeddings.items():
            # 提取各种向量
            fused_emb = user_data.get('fused_embedding', None)
            behavior_emb = user_data.get('behavior_embedding', None)
            location_emb = user_data.get('location_embedding', None)
            attribute_emb = user_data.get('attribute_embedding', None)
            
            # 构建特征向量
            feature_vector = []
            
            # 融合向量（主要特征）
            if fused_emb is not None:
                if isinstance(fused_emb, np.ndarray):
                    feature_vector.extend(fused_emb.flatten())
                else:
                    feature_vector.extend(np.array(fused_emb).flatten())
            
            # 行为向量
            if behavior_emb is not None:
                if isinstance(behavior_emb, np.ndarray):
                    feature_vector.extend(behavior_emb.flatten())
                else:
                    feature_vector.extend(np.array(behavior_emb).flatten())
            
            # 位置向量
            if location_emb is not None:
                if isinstance(location_emb, np.ndarray):
                    feature_vector.extend(location_emb.flatten())
                else:
                    feature_vector.extend(np.array(location_emb).flatten())
            
            # 属性向量
            if attribute_emb is not None:
                if isinstance(attribute_emb, np.ndarray):
                    feature_vector.extend(attribute_emb.flatten())
                else:
                    feature_vector.extend(np.array(attribute_emb).flatten())
            
            if len(feature_vector) == 0:
                continue
            
            # 特征工程
            if use_feature_engineering:
                # 添加统计特征
                feature_array = np.array(feature_vector)
                
                # L2范数
                l2_norm = np.linalg.norm(feature_array)
                feature_vector.append(l2_norm)
                
                # 均值和标准差
                feature_vector.append(np.mean(feature_array))
                feature_vector.append(np.std(feature_array))
                
                # 最大值和最小值
                feature_vector.append(np.max(feature_array))
                feature_vector.append(np.min(feature_array))
            
            features_list.append(feature_vector)
            labels_list.append(1 if user_id in self.seed_users else 0)
            user_ids_list.append(user_id)
        
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)
        self.user_ids = np.array(user_ids_list)
        
        print(f"特征维度: {self.features.shape}")
        print(f"正样本数量: {np.sum(self.labels == 1)}")
        print(f"负样本数量: {np.sum(self.labels == 0)}")
        
        return True
    
    def balance_dataset(self, strategy='undersample', ratio=5):
        """平衡数据集"""
        positive_indices = np.where(self.labels == 1)[0]
        negative_indices = np.where(self.labels == 0)[0]
        
        if strategy == 'undersample':
            # 下采样负样本
            target_negative_size = len(positive_indices) * ratio
            if len(negative_indices) > target_negative_size:
                selected_negative = np.random.choice(
                    negative_indices, 
                    size=target_negative_size, 
                    replace=False
                )
                selected_indices = np.concatenate([positive_indices, selected_negative])
            else:
                selected_indices = np.concatenate([positive_indices, negative_indices])
        
        elif strategy == 'oversample':
            # 上采样正样本（简单重复）
            target_positive_size = len(negative_indices) // ratio
            if len(positive_indices) < target_positive_size:
                additional_positive = np.random.choice(
                    positive_indices,
                    size=target_positive_size - len(positive_indices),
                    replace=True
                )
                selected_positive = np.concatenate([positive_indices, additional_positive])
            else:
                selected_positive = positive_indices
            selected_indices = np.concatenate([selected_positive, negative_indices])
        
        else:
            selected_indices = np.arange(len(self.labels))
        
        # 重新排序
        np.random.shuffle(selected_indices)
        
        self.features = self.features[selected_indices]
        self.labels = self.labels[selected_indices]
        self.user_ids = self.user_ids[selected_indices]
        
        print(f"平衡后数据集大小: {len(self.labels)}")
        print(f"正样本数量: {np.sum(self.labels == 1)}")
        print(f"负样本数量: {np.sum(self.labels == 0)}")


class DeepLookalikeModel(nn.Module):
    """深度学习Look-alike模型"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


class UserExpansionModel:
    """用户扩量模型训练器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_importance = None
        
    def create_traditional_models(self):
        """创建传统机器学习模型"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state
            ),
            
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            'mlp': MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=500,
                random_state=self.random_state
            )
        }
        
        return models
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """训练传统机器学习模型"""
        print("训练传统机器学习模型...")
        
        # 数据标准化
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        models = self.create_traditional_models()
        results = {}
        
        for name, model in models.items():
            print(f"训练 {name}...")
            
            try:
                # 训练模型
                model.fit(X_train_scaled, y_train)
                
                # 预测
                y_pred = model.predict(X_val_scaled)
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                
                # 评估
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted')
                recall = recall_score(y_val, y_pred, average='weighted')
                f1 = f1_score(y_val, y_pred, average='weighted')
                auc = roc_auc_score(y_val, y_pred_proba)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'predictions': y_pred_proba
                }
                
                print(f"{name} - AUC: {auc:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"{name} 训练失败: {e}")
        
        self.models = results
        
        # 选择最佳模型（基于AUC）
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
            self.best_model = results[best_model_name]['model']
            print(f"\n最佳传统模型: {best_model_name} (AUC: {results[best_model_name]['auc']:.4f})")
        
        return results
    
    def train_deep_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=256):
        """训练深度学习模型"""
        if not TORCH_AVAILABLE:
            print("PyTorch不可用，跳过深度学习模型")
            return None
        
        print("训练深度学习模型...")
        
        # 数据准备
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_val_scaled = self.scaler.transform(X_val)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 创建模型
        model = DeepLookalikeModel(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练循环
        best_val_auc = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_pred_proba = val_outputs.cpu().numpy()
                val_pred = (val_pred_proba > 0.5).astype(int)
                
                val_auc = roc_auc_score(y_val, val_pred_proba)
                val_f1 = f1_score(y_val, val_pred, average='weighted')
            
            scheduler.step(val_loss)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_deep_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
            
            if patience_counter >= patience:
                print(f"早停于第 {epoch} 轮")
                break
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_deep_model.pth'))
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            final_pred_proba = model(X_val_tensor).cpu().numpy()
            final_pred = (final_pred_proba > 0.5).astype(int)
            
            final_auc = roc_auc_score(y_val, final_pred_proba)
            final_f1 = f1_score(y_val, final_pred, average='weighted')
        
        print(f"深度学习模型最终性能 - AUC: {final_auc:.4f}, F1: {final_f1:.4f}")
        
        return {
            'model': model,
            'auc': final_auc,
            'f1': final_f1,
            'predictions': final_pred_proba
        }
    
    def predict_all_users(self, dataset: UserExpansionDataset):
        """对所有用户进行预测"""
        print("对所有用户进行预测...")
        
        if self.best_model is None:
            raise ValueError("没有训练好的模型")
        
        # 准备所有用户的特征
        X_all = self.scaler.transform(dataset.features)
        
        # 预测
        if hasattr(self.best_model, 'predict_proba'):
            predictions = self.best_model.predict_proba(X_all)[:, 1]
        else:
            # 深度学习模型
            self.best_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_all)
                predictions = self.best_model(X_tensor).cpu().numpy()
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'user_id': dataset.user_ids,
            'score': predictions,
            'is_seed': dataset.labels
        })
        
        # 按分数排序
        results_df = results_df.sort_values('score', ascending=False)
        
        return results_df
    
    def compute_group_similarities(self, dataset: UserExpansionDataset, expansion_users: List[str], 
                                 seed_users: List[str]) -> Dict[str, float]:
        """计算扩量群体与种子用户群体在各维度上的整体相似度"""
        print("计算扩量群体与种子用户群体的整体相似度...")
        
        group_similarities = {}
        
        # 对每种向量类型计算群体相似度
        for emb_type in ['fused_embedding', 'behavior_embedding', 'location_embedding', 'attribute_embedding']:
            print(f"计算 {emb_type} 维度相似度...")
            
            # 收集扩量群体的向量
            expansion_embeddings = []
            for user_id in expansion_users:
                if user_id in dataset.user_embeddings:
                    user_emb = dataset.user_embeddings[user_id].get(emb_type, None)
                    if user_emb is not None:
                        expansion_embeddings.append(np.array(user_emb).flatten())
            
            # 收集种子用户群体的向量
            seed_embeddings = []
            for user_id in seed_users:
                if user_id in dataset.user_embeddings:
                    user_emb = dataset.user_embeddings[user_id].get(emb_type, None)
                    if user_emb is not None:
                        seed_embeddings.append(np.array(user_emb).flatten())
            
            if len(expansion_embeddings) == 0 or len(seed_embeddings) == 0:
                print(f"警告: {emb_type} 维度数据不足，跳过")
                group_similarities[f'{emb_type}_group_similarity'] = None
                continue
            
            # 计算扩量群体中心向量（均值）
            expansion_center = np.mean(expansion_embeddings, axis=0).reshape(1, -1)
            
            # 计算种子用户群体中心向量（均值）
            seed_center = np.mean(seed_embeddings, axis=0).reshape(1, -1)
            
            # 计算两个群体中心向量的余弦相似度
            group_similarity = cosine_similarity(expansion_center, seed_center)[0][0]
            group_similarities[f'{emb_type}_group_similarity'] = group_similarity
            
            print(f"{emb_type} 群体相似度: {group_similarity:.4f}")
        
        return group_similarities
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'models': self.models
        }
        
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.models = model_data.get('models', {})
        print(f"模型已从 {filepath} 加载")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="用户扩量系统")
    parser.add_argument("--embeddings_path", type=str, required=True,
                       help="综合用户向量文件路径")
    parser.add_argument("--seed_users_path", type=str, required=True,
                       help="种子用户名单文件路径")
    parser.add_argument("--output_dir", type=str, default="./expansion_results",
                       help="结果输出目录")
    parser.add_argument("--top_k", type=int, default=1000,
                       help="输出前K个扩量用户")
    parser.add_argument("--balance_strategy", type=str, default="undersample",
                       choices=["undersample", "oversample", "none"],
                       help="数据平衡策略")
    parser.add_argument("--use_deep_learning", action="store_true",
                       help="是否使用深度学习模型")
    parser.add_argument("--save_model", type=str, default=None,
                       help="保存模型的路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载数据
    dataset = UserExpansionDataset(args.embeddings_path, args.seed_users_path)
    dataset.load_data()
    dataset.prepare_features(use_feature_engineering=True)
    
    # 2. 数据平衡
    if args.balance_strategy != "none":
        dataset.balance_dataset(strategy=args.balance_strategy)
    
    # 3. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        dataset.features, dataset.labels,
        test_size=0.2, stratify=dataset.labels,
        random_state=42
    )
    
    # 4. 训练模型
    model = UserExpansionModel(random_state=42)
    
    # 传统机器学习模型
    traditional_results = model.train_traditional_models(X_train, y_train, X_val, y_val)
    
    # 深度学习模型（可选）
    if args.use_deep_learning:
        deep_result = model.train_deep_model(X_train, y_train, X_val, y_val)
        if deep_result and deep_result['auc'] > max([r['auc'] for r in traditional_results.values()]):
            model.best_model = deep_result['model']
            print("选择深度学习模型作为最佳模型")
    
    # 5. 预测所有用户
    predictions_df = model.predict_all_users(dataset)
    
    # 6. 获取top-k扩量用户（排除种子用户）
    expansion_users = predictions_df[predictions_df['is_seed'] == 0].head(args.top_k)
    
    # 7. 计算群体相似度
    seed_user_list = list(dataset.seed_users)
    top_user_list = expansion_users['user_id'].tolist()
    
    group_similarities = model.compute_group_similarities(dataset, top_user_list, seed_user_list)
    
    # 8. 保存结果
    predictions_df.to_csv(os.path.join(args.output_dir, "all_users_predictions.csv"), index=False)
    expansion_users.to_csv(os.path.join(args.output_dir, "expansion_results.csv"), index=False)
    
    # 保存群体相似度结果
    group_similarities_df = pd.DataFrame([group_similarities])
    group_similarities_df.to_csv(os.path.join(args.output_dir, "group_similarities.csv"), index=False)
    
    # 10. 保存模型（可选）
    if args.save_model:
        model.save_model(args.save_model)
    
    # 11. 输出统计信息
    print("\n=== 扩量结果统计 ===")
    print(f"总用户数: {len(predictions_df)}")
    print(f"种子用户数: {len(dataset.seed_users)}")
    print(f"扩量用户数: {len(expansion_users)}")
    print(f"平均扩量分数: {expansion_users['score'].mean():.4f}")
    print(f"扩量分数标准差: {expansion_users['score'].std():.4f}")
    
    # 群体相似度统计
    print("\n=== 群体相似度统计 ===")
    for emb_type in ['fused_embedding', 'behavior_embedding', 'location_embedding', 'attribute_embedding']:
        sim_key = f'{emb_type}_group_similarity'
        if sim_key in group_similarities and group_similarities[sim_key] is not None:
            print(f"{emb_type} 群体相似度: {group_similarities[sim_key]:.4f}")
    
    print(f"\n结果已保存到: {args.output_dir}")
    print("- all_users_predictions.csv: 所有用户的预测分数")
    print("- expansion_results.csv: 扩量用户详细结果")
    print("- group_similarities.csv: 群体相似度结果")


if __name__ == "__main__":
    main()
