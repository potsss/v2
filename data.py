"""
v2 数据预处理（统一简化版）
保留原有功能：行为序列 -> id 映射与序列；可选属性/位置处理（轻量包装）。
"""
import os
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from .config import Config
from .location_features import LocationFeatureProcessor
from .matrix_factorization import UnifiedItemProcessor, ALSMatrixFactorization, MatrixFactorizationUserEmbedding


def _extract_domain(url: str):
    if not isinstance(url, str):
        return None
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    try:
        domain = urlparse(url).netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


class AttributeProcessorV2:
    def __init__(self):
        self.categorical_encoders = {}
        self.numerical_scaler = StandardScaler()
        self.attribute_info = {}
        self.processed_attributes = {}

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"属性数据文件不存在: {path}")
            return None
        df = pd.read_csv(path, sep="\t")
        return df

    def analyze(self, df: pd.DataFrame):
        user_id_col = df.columns[0]
        attribute_cols = df.columns[1:]
        info = {}
        for col in attribute_cols:
            if df[col].dtype in ["int64", "float64"]:
                unique_values = df[col].nunique()
                if unique_values <= 20 and str(df[col].dtype) == "int64":
                    info[col] = {"type": "categorical", "vocab_size": unique_values}
                else:
                    info[col] = {"type": "numerical"}
            else:
                info[col] = {"type": "categorical", "vocab_size": df[col].nunique()}
        self.attribute_info = info
        return info, user_id_col

    def preprocess(self, path: str):
        df = self.load(path)
        if df is None:
            return None, None
        info, user_id_col = self.analyze(df)

        categorical_cols = [k for k, v in info.items() if v["type"] == "categorical"]
        numerical_cols = [k for k, v in info.items() if v["type"] == "numerical"]

        if categorical_cols:
            for col in categorical_cols:
                df[col] = df[col].fillna("Unknown")
                vc = df[col].value_counts()
                rare = vc[vc < Config.CATEGORICAL_MIN_FREQ].index
                df[col] = df[col].replace(rare, "Other")
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str))
                self.categorical_encoders[col] = enc
                if "vocab_size" in info[col]:
                    info[col]["vocab_size"] = len(enc.classes_)

        if numerical_cols:
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].mean())
            if Config.NUMERICAL_STANDARDIZATION:
                df[numerical_cols] = self.numerical_scaler.fit_transform(df[numerical_cols])

        processed = {}
        for _, row in df.iterrows():
            uid = row[user_id_col]
            attrs = {}
            for col in df.columns:
                if col != user_id_col:
                    attrs[col] = row[col]
            processed[uid] = attrs

        self.processed_attributes = processed
        return processed, info

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "user_attributes.pkl"), "wb") as f:
            pickle.dump(self.processed_attributes, f)
        with open(os.path.join(out_dir, "attribute_info.pkl"), "wb") as f:
            pickle.dump(self.attribute_info, f)
        with open(os.path.join(out_dir, "attribute_encoders.pkl"), "wb") as f:
            pickle.dump({
                "categorical_encoders": self.categorical_encoders,
                "numerical_scaler": self.numerical_scaler,
            }, f)


class DataPreprocessorV2:
    def __init__(self):
        self.url_to_id = {}
        self.id_to_url = {}
        self.user_sequences = defaultdict(list)
        self.attribute_processor = AttributeProcessorV2() if Config.ENABLE_ATTRIBUTES else None
        # 位置侧
        self.bs_to_id = {}
        self.id_to_bs = {}
        # 位置特征处理器
        self.location_feature_processor = LocationFeatureProcessor() if Config.ENABLE_LOCATION_FEATURES else None
        self.user_location_sequences = defaultdict(list)

    def preprocess(self, behavior_path: str = None):
        if behavior_path is None:
            behavior_path = Config.DATA_PATH
        print(f"读取数据: {behavior_path}")
        df = pd.read_csv(behavior_path, sep="\t")
        if "timestamp_str" not in df.columns:
            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "timestamp_str"})
            else:
                raise ValueError("缺少 timestamp_str/timestamp 列")

        # 清洗
        df = df.dropna()
        df = df.drop_duplicates()
        if "weight" in df.columns:
            df = df[df["weight"] > 0]

        # 处理 domain
        tqdm.pandas(desc="提取domain")
        df["domain"] = df["url"].progress_apply(_extract_domain)
        df = df.dropna(subset=["domain"])

        unique_domains = df["domain"].unique()
        self.url_to_id = {d: i for i, d in enumerate(unique_domains)}
        self.id_to_url = {i: d for d, i in self.url_to_id.items()}
        df["domain_id"] = df["domain"].map(self.url_to_id)

        # 构造序列
        for uid, group in df.groupby("user_id"):
            # 使用 datetime 排序，更稳健的时序
            try:
                group = group.assign(_ts=pd.to_datetime(group["timestamp_str"], errors="coerce")).sort_values("_ts").drop(columns=["_ts"])
            except Exception:
                group = group.sort_values("timestamp_str")
            seq = []
            for _, row in group.iterrows():
                repeat_count = 1
                if "weight" in row:
                    repeat_count = min(max(1, int(row["weight"])), Config.MAX_WEIGHT_REPEAT)
                seq.extend([row["domain_id"]] * repeat_count)
            if len(seq) >= Config.MIN_COUNT:
                self.user_sequences[uid] = seq

        # 保存
        os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
        with open(os.path.join(Config.PROCESSED_DATA_PATH, "url_mappings.pkl"), "wb") as f:
            pickle.dump({"url_to_id": self.url_to_id, "id_to_url": self.id_to_url}, f)
        with open(os.path.join(Config.PROCESSED_DATA_PATH, "user_sequences.pkl"), "wb") as f:
            pickle.dump(dict(self.user_sequences), f)

        # 属性（可选）
        user_attributes = None
        attribute_info = None
        if Config.ENABLE_ATTRIBUTES and self.attribute_processor:
            user_attributes, attribute_info = self.attribute_processor.preprocess(Config.ATTRIBUTE_DATA_PATH)
            if user_attributes is not None:
                self.attribute_processor.save(Config.PROCESSED_DATA_PATH)

        # 位置（可选）
        if Config.ENABLE_LOCATION:
            self._preprocess_location()

        return self.user_sequences, {"url_to_id": self.url_to_id, "id_to_url": self.id_to_url}, user_attributes, attribute_info

    def load_processed(self):
        with open(os.path.join(Config.PROCESSED_DATA_PATH, "url_mappings.pkl"), "rb") as f:
            m = pickle.load(f)
        self.url_to_id = m["url_to_id"]
        self.id_to_url = m["id_to_url"]
        with open(os.path.join(Config.PROCESSED_DATA_PATH, "user_sequences.pkl"), "rb") as f:
            self.user_sequences = pickle.load(f)
        return self.user_sequences, {"url_to_id": self.url_to_id, "id_to_url": self.id_to_url}

    def _preprocess_location(self):
        path = Config.LOCATION_DATA_PATH
        if not path or not os.path.exists(path):
            return
        try:
            print(f"读取位置数据: {path}")
            ldf = pd.read_csv(path, sep="\t")
        except Exception:
            # 若为逗号分隔可自行调整
            ldf = pd.read_csv(path)
        if "user_id" not in ldf.columns:
            return
        # 列名兼容：基站 id 列名可能不同
        bs_col = None
        for cand in ["base_station_id", "bs_id", "station_id", "cell_id"]:
            if cand in ldf.columns:
                bs_col = cand
                break
        if bs_col is None:
            return
        # 可选时间列
        ts_col = None
        for cand in ["timestamp_str", "timestamp", "time"]:
            if cand in ldf.columns:
                ts_col = cand
                break
        # 建立映射
        unique_bs = ldf[bs_col].astype(str).unique()
        self.bs_to_id = {b: i for i, b in enumerate(unique_bs)}
        self.id_to_bs = {i: b for b, i in self.bs_to_id.items()}
        ldf["bs_id_int"] = ldf[bs_col].astype(str).map(self.bs_to_id)
        # 构造位置序列（支持duration权重）
        for uid, group in ldf.groupby("user_id"):
            if ts_col is not None:
                try:
                    group = group.assign(_ts=pd.to_datetime(group[ts_col], errors="coerce")).sort_values("_ts").drop(columns=["_ts"])
                except Exception:
                    group = group.sort_values(ts_col)
            
            seq = []
            for _, row in group.iterrows():
                repeat_count = 1
                # 检查是否有duration列作为权重
                if "duration" in row and pd.notna(row["duration"]):
                    # 将duration转换为重复次数，限制在合理范围内
                    duration_weight = float(row["duration"])
                    if duration_weight > 0:
                        # 根据duration计算重复次数，使用配置中的缩放因子
                        repeat_count = min(max(1, int(duration_weight / Config.DURATION_WEIGHT_SCALE)), Config.MAX_WEIGHT_REPEAT)
                seq.extend([row["bs_id_int"]] * repeat_count)
            
            if len(seq) >= Config.LOCATION_MIN_CONNECTIONS:
                self.user_location_sequences[uid] = seq
        # 处理位置特征（如果启用）
        if Config.ENABLE_LOCATION_FEATURES and self.location_feature_processor:
            print("处理位置特征...")
            success = self.location_feature_processor.load_and_process_features()
            if success:
                # 保存位置特征
                feature_save_path = os.path.join(Config.PROCESSED_DATA_PATH, "location_features_processed.pkl")
                self.location_feature_processor.save_processed_features(feature_save_path)
            else:
                print("位置特征处理失败，将跳过特征增强")
        
        # 保存
        with open(os.path.join(Config.PROCESSED_DATA_PATH, "location_mappings.pkl"), "wb") as f:
            pickle.dump({"bs_to_id": self.bs_to_id, "id_to_bs": self.id_to_bs}, f)
        with open(os.path.join(Config.PROCESSED_DATA_PATH, "location_sequences.pkl"), "wb") as f:
            pickle.dump(dict(self.user_location_sequences), f)

    def load_location_processed(self):
        mp = os.path.join(Config.PROCESSED_DATA_PATH, "location_mappings.pkl")
        sp = os.path.join(Config.PROCESSED_DATA_PATH, "location_sequences.pkl")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            return None, None
        with open(mp, "rb") as f:
            m = pickle.load(f)
        with open(sp, "rb") as f:
            seq = pickle.load(f)
        
        # 将加载的位置序列赋值给实例变量，这样矩阵分解可以访问
        self.user_location_sequences = seq
        self.bs_to_id = m.get("bs_to_id", {})
        self.id_to_bs = m.get("id_to_bs", {})
        
        # 加载位置特征（如果启用且存在）
        if Config.ENABLE_LOCATION_FEATURES and self.location_feature_processor:
            feature_path = os.path.join(Config.PROCESSED_DATA_PATH, "location_features_processed.pkl")
            if os.path.exists(feature_path):
                success = self.location_feature_processor.load_processed_features(feature_path)
                if success:
                    print("已加载位置特征数据")
        
        return seq, m
    
    def process_location(self):
        """处理位置数据的公开方法"""
        self._preprocess_location()

    def process_matrix_factorization(self):
        """
        处理矩阵分解数据：构建统一物品空间和交互矩阵
        """
        print("开始处理矩阵分解数据...")
        
        # 确保已有基础数据
        if not hasattr(self, 'user_sequences') or not self.user_sequences:
            print("请先运行 process_behavior() 处理行为数据")
            return False
        
        # 确保位置数据已处理（如果启用位置）
        if Config.ENABLE_LOCATION:
            if not hasattr(self, 'user_location_sequences') or not self.user_location_sequences:
                print("位置数据未处理，开始处理位置数据...")
                self.process_location()
        
        # 初始化统一物品处理器（传入位置特征处理器）
        item_processor = UnifiedItemProcessor(self.location_feature_processor)
        
        # 添加URL物品
        all_urls = set()
        for seq in self.user_sequences.values():
            all_urls.update(seq)
        item_processor.add_items(list(all_urls), "url")
        print(f"添加了 {len(all_urls)} 个URL物品")
        
        # 添加位置物品（如果启用）
        location_sequences = {}
        if Config.ENABLE_LOCATION and hasattr(self, 'user_location_sequences') and self.user_location_sequences:
            all_locations = set()
            for seq in self.user_location_sequences.values():
                all_locations.update(seq)
            
            # 需要将位置ID转换为基站ID用于物品处理器
            location_items = []
            for loc_id in all_locations:
                if loc_id in self.id_to_bs:
                    bs_id = self.id_to_bs[loc_id]
                    location_items.append(bs_id)
            
            if location_items:
                item_processor.add_items(location_items, "location")
                # 转换位置序列中的ID为基站ID
                converted_sequences = {}
                for user_id, seq in self.user_location_sequences.items():
                    converted_seq = []
                    for loc_id in seq:
                        if loc_id in self.id_to_bs:
                            bs_id = self.id_to_bs[loc_id]
                            converted_seq.append(bs_id)
                    if converted_seq:
                        converted_sequences[user_id] = converted_seq
                location_sequences = converted_sequences
                
            print(f"添加了 {len(location_items)} 个位置物品")
        
        print(f"统一物品空间包含 {len(item_processor.item_to_id)} 个物品")
        
        # 构建交互矩阵
        print("构建用户-物品交互矩阵...")
        interaction_matrix, user_to_idx, idx_to_user = item_processor.build_interaction_matrix(
            self.user_sequences, location_sequences
        )
        
        print(f"交互矩阵维度: {interaction_matrix.shape}")
        print(f"非零交互数: {interaction_matrix.nnz}")
        print(f"稀疏度: {1 - interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]):.6f}")
        
        # 训练ALS模型
        print("开始训练ALS矩阵分解模型...")
        if (Config.ENABLE_LOCATION_FEATURES and 
            getattr(Config, 'MF_ENABLE_LOCATION_FEATURES', True) and
            self.location_feature_processor and 
            self.location_feature_processor.location_features):
            # 使用增强的矩阵分解模型
            print("使用位置特征增强的矩阵分解模型...")
            from .matrix_factorization import EnhancedMatrixFactorization
            als_model = EnhancedMatrixFactorization(location_feature_processor=self.location_feature_processor)
            als_model.fit(interaction_matrix, item_processor=item_processor, verbose=True)
        else:
            # 使用基础矩阵分解模型
            als_model = ALSMatrixFactorization()
            als_model.fit(interaction_matrix, verbose=True)
        
        # 保存模型和映射
        os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
        
        # 保存ALS模型
        model_path = os.path.join(Config.PROCESSED_DATA_PATH, "als_model.pkl")
        als_model.save_model(model_path)
        print(f"ALS模型已保存: {model_path}")
        
        # 保存物品映射
        mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "unified_item_mappings.pkl")
        item_processor.save_mappings(mappings_path)
        print(f"物品映射已保存: {mappings_path}")
        
        # 保存用户映射
        user_mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "matrix_user_mappings.pkl")
        with open(user_mappings_path, 'wb') as f:
            pickle.dump({
                'user_to_idx': user_to_idx,
                'idx_to_user': idx_to_user
            }, f)
        print(f"用户映射已保存: {user_mappings_path}")
        
        # 保存交互矩阵（可选，用于调试）
        matrix_path = os.path.join(Config.PROCESSED_DATA_PATH, "interaction_matrix.pkl")
        with open(matrix_path, 'wb') as f:
            pickle.dump(interaction_matrix, f)
        print(f"交互矩阵已保存: {matrix_path}")
        
        return True
    
    def load_matrix_factorization_data(self):
        """
        加载矩阵分解相关数据
        
        Returns:
            tuple: (als_model, item_processor, user_mappings) or (None, None, None)
        """
        model_path = os.path.join(Config.PROCESSED_DATA_PATH, "als_model.pkl")
        mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "unified_item_mappings.pkl")
        user_mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "matrix_user_mappings.pkl")
        
        if not all(os.path.exists(p) for p in [model_path, mappings_path, user_mappings_path]):
            return None, None, None
        
        # 加载ALS模型
        als_model = ALSMatrixFactorization()
        als_model.load_model(model_path)
        
        # 加载物品处理器
        item_processor = UnifiedItemProcessor()
        item_processor.load_mappings(mappings_path)
        
        # 加载用户映射
        with open(user_mappings_path, 'rb') as f:
            user_mappings = pickle.load(f)
        
        return als_model, item_processor, user_mappings
    
    def compute_mf_user_embeddings(self, method="from_interactions"):
        """
        计算基于矩阵分解的用户向量
        
        Args:
            method: str "from_interactions" 或 "from_factors"
            
        Returns:
            dict: {user_id: embedding_vector} or None
        """
        # 加载矩阵分解数据
        als_model, item_processor, user_mappings = self.load_matrix_factorization_data()
        if als_model is None:
            print("未找到矩阵分解模型，请先运行 process_matrix_factorization()")
            return None
        
        # 加载用户序列数据
        user_sequences, _ = self.load_processed()
        location_sequences = None
        if Config.ENABLE_LOCATION:
            location_sequences, _ = self.load_location_processed()
        
        if user_sequences is None:
            print("未找到用户序列数据")
            return None
        
        # 创建用户向量生成器
        user_embedding_generator = MatrixFactorizationUserEmbedding(
            als_model, item_processor, user_mappings['user_to_idx']
        )
        
        # 计算用户向量
        if method == "from_factors":
            print("从用户因子矩阵计算用户向量...")
            user_embeddings = user_embedding_generator.compute_from_factors()
        else:
            print("从交互序列聚合计算用户向量...")
            user_embeddings = user_embedding_generator.compute_from_interactions(
                user_sequences, location_sequences
            )
        
        print(f"生成了 {len(user_embeddings)} 个用户向量")
        return user_embeddings




