import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib
import warnings
warnings.filterwarnings('ignore')


class EnhancedLabelDiscriminator:
    """增强版标签判别器，支持多种模型、特征工程和集成学习"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        self.label_encoder = None
        self.best_model = None
        self.ensemble_model = None
        self.models = {}
        self.results = {}
        
    def load_data(self, embeddings_path, labels_path):
        """加载嵌入向量和标签数据"""
        try:
            with open(embeddings_path, 'rb') as f:
                user_embeddings_dict = pickle.load(f)
            print(f"成功从 {embeddings_path} 加载嵌入向量")
            print(f"包含嵌入向量的用户数量: {len(user_embeddings_dict)}")
        except FileNotFoundError:
            print(f"错误: 在 {embeddings_path} 未找到嵌入向量文件")
            return None, None, None
        except Exception as e:
            print(f"加载嵌入向量时出错: {e}")
            return None, None, None

        try:
            labels_df = pd.read_csv(labels_path, encoding='utf-8', sep='\t')
            print(f"成功从 {labels_path} 加载标签")
            print(f"包含标签的用户数量: {len(labels_df)}")
            if 'user_id' not in labels_df.columns or 'label' not in labels_df.columns:
                print("错误: 标签文件必须包含 'user_id' 和 'label' 列。")
                return None, None, None
        except FileNotFoundError:
            print(f"错误: 在 {labels_path} 未找到标签文件")
            return None, None, None
        except Exception as e:
            print(f"加载标签时出错: {e}")
            return None, None, None

        # 转换user_id为字符串
        labels_df['user_id'] = labels_df['user_id'].astype(str)

        X_list = []
        y_list = []
        user_ids = []
        
        missing_embeddings_count = 0
        processed_users = set()

        for index, row in labels_df.iterrows():
            user_id = row['user_id']
            label = row['label']
            
            if user_id in processed_users:
                print(f"警告: 用户 {user_id} 在标签文件中重复出现，将只使用第一次出现的标签。")
                continue
            processed_users.add(user_id)

            if user_id in user_embeddings_dict:
                X_list.append(user_embeddings_dict[user_id])
                y_list.append(label)
                user_ids.append(user_id)
            else:
                missing_embeddings_count += 1
        
        if missing_embeddings_count > 0:
            print(f"警告: 来自标签文件的 {missing_embeddings_count} 个用户在嵌入向量文件中未找到，已被跳过。")

        if not X_list:
            print("错误: 在嵌入向量和标签数据之间未找到共同用户，或合并后数据集为空。")
            return None, None, None
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 标签编码
        if y.size > 0 and not np.issubdtype(y.dtype, np.number):
            print("标签不是数字类型。正在应用 LabelEncoder。")
            self.label_encoder = LabelEncoder()
            y_numeric = self.label_encoder.fit_transform(y)
            print(f"标签映射: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        elif y.size > 0 and np.issubdtype(y.dtype, np.number):
            print("标签是数字类型。")
            y_numeric = y.astype(int)
        else:
            print("错误：没有可用的标签数据。")
            return None, None, None

        print(f"准备好的数据包含 {X.shape[0]} 个样本和 {X.shape[1]} 个特征。")
        print(f"标签分布: {np.bincount(y_numeric)}")
        
        return X, y_numeric, user_ids
    
    def preprocess_features(self, X_train, X_test, y_train, scaler_type='standard', 
                          use_pca=True, pca_components=0.95, 
                          use_feature_selection=True, k_best=None):
        """特征预处理：标准化、降维、特征选择"""
        print("开始特征预处理...")
        
        # 特征标准化
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
            
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"使用 {scaler_type} 标准化完成")
        
        # PCA降维
        if use_pca:
            if isinstance(pca_components, float) and pca_components < 1.0:
                # 保留指定比例的方差
                self.pca = PCA(n_components=pca_components, random_state=self.random_state)
            else:
                # 保留指定数量的主成分
                n_components = min(int(pca_components), X_train_scaled.shape[1])
                self.pca = PCA(n_components=n_components, random_state=self.random_state)
            
            X_train_pca = self.pca.fit_transform(X_train_scaled)
            X_test_pca = self.pca.transform(X_test_scaled)
            print(f"PCA降维完成: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]}")
            print(f"保留方差比例: {self.pca.explained_variance_ratio_.sum():.4f}")
            X_train_processed = X_train_pca
            X_test_processed = X_test_pca
        else:
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled
            
        # 特征选择
        if use_feature_selection:
            if k_best is None:
                k_best = min(50, X_train_processed.shape[1])  # 默认选择50个最佳特征
            
            self.feature_selector = SelectKBest(f_classif, k=k_best)
            X_train_selected = self.feature_selector.fit_transform(X_train_processed, y_train)
            X_test_selected = self.feature_selector.transform(X_test_processed)
            print(f"特征选择完成: {X_train_processed.shape[1]} -> {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected
        
        return X_train_processed, X_test_processed
    
    def create_models(self, class_weights=None):
        """创建多种机器学习模型"""
        models = {}
        
        # 逻辑回归
        models['logistic'] = LogisticRegression(
            random_state=self.random_state, 
            max_iter=1000,
            class_weight=class_weights
        )
        
        # 随机森林
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight=class_weights
        )
        
        # 梯度提升
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            random_state=self.random_state
        )
        
        # 支持向量机
        models['svm'] = SVC(
            random_state=self.random_state,
            probability=True,
            class_weight=class_weights
        )
        
        # 神经网络
        models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            random_state=self.random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        return models
    
    def hyperparameter_tuning(self, model, X_train, y_train, cv_folds=5):
        """超参数调优"""
        param_grids = {
            'logistic': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'mlp': {
                'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        model_name = None
        for name, clf in self.models.items():
            if type(clf) == type(model):
                model_name = name
                break
        
        if model_name and model_name in param_grids:
            print(f"正在对 {model_name} 进行超参数调优...")
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=cv_folds, scoring='f1_weighted', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            print(f"{model_name} 最佳参数: {grid_search.best_params_}")
            return grid_search.best_estimator_
        
        return model
    
    def evaluate_models(self, X_train, X_test, y_train, y_test, use_cv=True, cv_folds=5):
        """评估多个模型的性能"""
        print("开始模型评估...")
        
        # 计算类别权重以处理不平衡数据
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f"类别权重: {class_weight_dict}")
        
        self.models = self.create_models(class_weight_dict)
        
        for name, model in self.models.items():
            print(f"\n评估模型: {name}")
            
            # 超参数调优
            tuned_model = self.hyperparameter_tuning(model, X_train, y_train, cv_folds)
            
            # 交叉验证
            if use_cv:
                cv_scores = cross_val_score(
                    tuned_model, X_train, y_train, 
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                    scoring='f1_weighted'
                )
                print(f"交叉验证 F1 分数: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 训练模型
            tuned_model.fit(X_train, y_train)
            
            # 预测
            y_pred = tuned_model.predict(X_test)
            y_pred_proba = tuned_model.predict_proba(X_test) if hasattr(tuned_model, 'predict_proba') else None
            
            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results = {
                'model': tuned_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            # 如果是二分类问题，计算AUC
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                results['auc'] = auc
                print(f"AUC: {auc:.4f}")
            
            self.results[name] = results
            
            print(f"准确率: {accuracy:.4f}")
            print(f"精确率: {precision:.4f}")
            print(f"召回率: {recall:.4f}")
            print(f"F1分数: {f1:.4f}")
        
        # 找到最佳模型
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        self.best_model = self.results[best_model_name]['model']
        print(f"\n最佳模型: {best_model_name} (F1: {self.results[best_model_name]['f1']:.4f})")
        
        return self.results
    
    def create_ensemble(self, X_train, y_train):
        """创建集成模型"""
        print("创建集成模型...")
        
        # 选择性能较好的模型进行集成
        top_models = sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        
        ensemble_estimators = []
        for name, result in top_models:
            ensemble_estimators.append((name, result['model']))
            print(f"集成模型包含: {name} (F1: {result['f1']:.4f})")
        
        # 创建投票分类器
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'  # 使用概率投票
        )
        
        self.ensemble_model.fit(X_train, y_train)
        return self.ensemble_model
    
    def evaluate_ensemble(self, X_test, y_test):
        """评估集成模型"""
        if self.ensemble_model is None:
            print("集成模型未创建")
            return None
        
        print("评估集成模型...")
        y_pred_ensemble = self.ensemble_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred_ensemble)
        precision = precision_score(y_test, y_pred_ensemble, average='weighted')
        recall = recall_score(y_test, y_pred_ensemble, average='weighted')
        f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
        
        print(f"集成模型性能:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 如果是二分类问题，计算AUC
        if len(np.unique(y_test)) == 2:
            y_pred_proba_ensemble = self.ensemble_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba_ensemble)
            print(f"AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred_ensemble
        }
    
    def print_detailed_report(self, X_test, y_test, model_name='best'):
        """打印详细的分类报告"""
        if model_name == 'ensemble' and self.ensemble_model is not None:
            y_pred = self.ensemble_model.predict(X_test)
        elif model_name == 'best' and self.best_model is not None:
            y_pred = self.best_model.predict(X_test)
        else:
            print(f"模型 {model_name} 不可用")
            return
        
        print(f"\n{model_name} 模型详细报告:")
        print("=" * 50)
        
        # 分类报告
        if self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = None
            
        print("分类报告:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # 混淆矩阵
        print("混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    
    def save_model(self, save_path):
        """保存训练好的模型"""
        model_data = {
            'best_model': self.best_model,
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_selector': self.feature_selector,
            'label_encoder': self.label_encoder,
            'results': self.results
        }
        
        joblib.dump(model_data, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path):
        """加载训练好的模型"""
        model_data = joblib.load(load_path)
        
        self.best_model = model_data['best_model']
        self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.feature_selector = model_data['feature_selector']
        self.label_encoder = model_data['label_encoder']
        self.results = model_data['results']
        
        print(f"模型已从 {load_path} 加载")


def main():
    parser = argparse.ArgumentParser(description="增强版用户嵌入向量标签判别器")
    parser.add_argument("--embeddings_path", type=str, required=True,
                       help="包含用户嵌入向量的 .pkl 文件路径")
    parser.add_argument("--labels_path", type=str, required=True,
                       help="包含用户标签的 .csv 文件路径")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="测试集比例 (默认: 0.2)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--scaler_type", type=str, default='standard',
                       choices=['standard', 'robust', 'minmax'],
                       help="特征缩放方法 (默认: standard)")
    parser.add_argument("--use_pca", action='store_true',
                       help="是否使用PCA降维")
    parser.add_argument("--pca_components", type=float, default=0.95,
                       help="PCA保留的方差比例或主成分数量 (默认: 0.95)")
    parser.add_argument("--use_feature_selection", action='store_true',
                       help="是否使用特征选择")
    parser.add_argument("--k_best", type=int, default=50,
                       help="特征选择保留的特征数量 (默认: 50)")
    parser.add_argument("--use_ensemble", action='store_true',
                       help="是否使用集成学习")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="交叉验证折数 (默认: 5)")
    parser.add_argument("--save_model", type=str,
                       help="保存模型的路径")
    
    args = parser.parse_args()
    
    print("开始增强版判别器训练流程...")
    
    # 初始化判别器
    discriminator = EnhancedLabelDiscriminator(random_state=args.random_state)
    
    # 加载数据
    X, y, user_ids = discriminator.load_data(args.embeddings_path, args.labels_path)
    
    if X is None or y is None:
        print("数据加载失败，程序退出")
        return
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, 
        stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 特征预处理
    X_train_processed, X_test_processed = discriminator.preprocess_features(
        X_train, X_test, y_train,
        scaler_type=args.scaler_type,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        use_feature_selection=args.use_feature_selection,
        k_best=args.k_best
    )
    
    # 模型评估
    results = discriminator.evaluate_models(
        X_train_processed, X_test_processed, y_train, y_test,
        cv_folds=args.cv_folds
    )
    
    # 打印详细报告
    discriminator.print_detailed_report(X_test_processed, y_test, 'best')
    
    # 集成学习
    if args.use_ensemble:
        discriminator.create_ensemble(X_train_processed, y_train)
        ensemble_results = discriminator.evaluate_ensemble(X_test_processed, y_test)
        discriminator.print_detailed_report(X_test_processed, y_test, 'ensemble')
    
    # 保存模型
    if args.save_model:
        discriminator.save_model(args.save_model)
    
    print("增强版判别器训练流程完成！")


if __name__ == "__main__":
    # 修复全局变量问题
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # 如果没有命令行参数，显示帮助信息
        print("使用示例:")
        print("python enhanced_label_discriminator.py --embeddings_path experiments/v2_default/models/fused_user_embeddings.pkl --labels_path data/label.csv --use_pca --use_feature_selection --use_ensemble --save_model enhanced_discriminator.pkl")
