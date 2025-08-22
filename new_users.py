"""
v2 新用户向量计算（支持融合推理：行为+属性+位置）
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from .config import Config
from .model import UserEmbedding, Item2Vec, Node2Vec
from .trainer import FusionTrainer


def save_training_entities(url_mappings, processed_dir):
    entities = {
        "urls": set(url_mappings["url_to_id"].keys()),
        "url_to_id": url_mappings["url_to_id"],
        "id_to_url": url_mappings["id_to_url"],
    }
    os.makedirs(processed_dir, exist_ok=True)
    with open(os.path.join(processed_dir, "training_entities.pkl"), "wb") as f:
        pickle.dump(entities, f)


def load_training_entities(processed_dir):
    path = os.path.join(processed_dir, "training_entities.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_new_user_behavior(behavior_path, url_mappings, training_entities=None):
    if not behavior_path or not os.path.exists(behavior_path):
        return {}, set()
    df = pd.read_csv(behavior_path, sep="\t")
    user_sequences = {}
    unknown = set()
    url_to_id = url_mappings["url_to_id"]
    
    # 先提取domain（与训练时保持一致）
    from urllib.parse import urlparse
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
    
    df["domain"] = df["url"].apply(_extract_domain)
    df = df.dropna(subset=["domain"])
    
    for uid, group in df.groupby("user_id"):
        seq = []
        for _, row in group.iterrows():
            domain = row["domain"]
            if training_entities and domain not in training_entities.get("urls", set()):
                unknown.add(domain)
                continue
            if domain in url_to_id:
                # 应用权重
                repeat_count = 1
                if "weight" in row and pd.notna(row["weight"]):
                    repeat_count = min(max(1, int(row["weight"])), Config.MAX_WEIGHT_REPEAT)
                seq.extend([url_to_id[domain]] * repeat_count)
        if seq:
            user_sequences[uid] = seq
    return user_sequences, unknown


def compute_new_user_embeddings(behavior_model, url_mappings, new_behavior_path=None, save_path=None):
    if new_behavior_path is None:
        new_behavior_path = Config.NEW_USER_BEHAVIOR_PATH
    training_entities = load_training_entities(Config.PROCESSED_DATA_PATH)
    new_sequences, unknown_urls = load_new_user_behavior(new_behavior_path, url_mappings, training_entities)
    if not new_sequences:
        print("没有可用的新用户行为数据")
        return {}
    ue = UserEmbedding(behavior_model, new_sequences, url_mappings).compute()
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(ue, f)
    # 简要报告
    report = {
        "total_new_users": len(new_sequences),
        "unknown_urls": sorted(list(unknown_urls)),
        "known_url_count": len(training_entities["urls"]) if training_entities else 0,
    }
    with open(os.path.join(Config.PROCESSED_DATA_PATH, "new_user_compatibility_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return ue


def _load_attribute_resources():
    aip = os.path.join(Config.PROCESSED_DATA_PATH, "attribute_info.pkl")
    aep = os.path.join(Config.PROCESSED_DATA_PATH, "attribute_encoders.pkl")
    if not (os.path.exists(aip) and os.path.exists(aep)):
        return None, None
    with open(aip, "rb") as f:
        attr_info = pickle.load(f)
    with open(aep, "rb") as f:
        enc_pack = pickle.load(f)
    return attr_info, enc_pack


def _transform_new_attributes(attr_path, attr_info, enc_pack):
    if not attr_path or not os.path.exists(attr_path):
        return None
    df = pd.read_csv(attr_path, sep="\t")
    if df.shape[1] < 2:
        return None
    user_id_col = df.columns[0]
    processed = {}
    # 解析列集合，保持与训练时 attribute_info 的列顺序一致
    categorical_cols = [k for k, v in attr_info.items() if v.get("type") == "categorical" and k in df.columns]
    numerical_cols = [k for k, v in attr_info.items() if v.get("type") == "numerical" and k in df.columns]

    # 类别编码
    cat_encoders = (enc_pack or {}).get("categorical_encoders", {}) if enc_pack else {}
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
        enc = cat_encoders.get(col)
        if enc is None:
            # 兜底：新建映射（不理想，但避免中断）
            classes = sorted(df[col].unique().tolist())
            class_to_id = {c: i for i, c in enumerate(classes)}
        else:
            # 使用训练时的 classes 做安全映射，未知值映射到 Other/Unknown/首类
            classes = list(enc.classes_)
            class_to_id = {c: i for i, c in enumerate(classes)}
            fallback = None
            if "Other" in class_to_id:
                fallback = class_to_id["Other"]
            elif "Unknown" in class_to_id:
                fallback = class_to_id["Unknown"]
            else:
                fallback = 0
            df[col] = df[col].map(lambda x: class_to_id.get(x, fallback))
            continue
        df[col] = df[col].map(lambda x: class_to_id.get(x, 0))

    # 数值标准化
    if len(numerical_cols) > 0:
        # 先按训练均值填充再做标准化（若训练进行了标准化）
        scaler = (enc_pack or {}).get("numerical_scaler") if enc_pack else None
        if scaler is not None and getattr(scaler, "mean_", None) is not None and getattr(scaler, "scale_", None) is not None:
            # 构建每列的填充值映射
            means = scaler.mean_
            fill_map = {c: means[i] for i, c in enumerate(numerical_cols)}
            df[numerical_cols] = df[numerical_cols].fillna(value=fill_map)
            # 严格按列顺序取值并变换
            X = df[numerical_cols].values.astype(np.float32)
            X = scaler.transform(X)
            df.loc[:, numerical_cols] = X
        else:
            # 未提供标准化器则仅用 0 填充
            df[numerical_cols] = df[numerical_cols].fillna(0.0)

    # 组装 attr_raw
    for _, row in df.iterrows():
        uid = row[user_id_col]
        attrs = {}
        for col in categorical_cols + numerical_cols:
            attrs[col] = row[col]
        processed[uid] = attrs
    return processed


def _compute_new_location_embeddings():
    """基于已训练位置模型与映射，计算新用户位置向量（均值聚合）。"""
    loc_path = Config.NEW_USER_LOCATION_PATH
    if not Config.ENABLE_LOCATION or not loc_path or not os.path.exists(loc_path):
        return None
    # 加载训练映射与位置模型
    mp = os.path.join(Config.PROCESSED_DATA_PATH, "location_mappings.pkl")
    if not os.path.exists(mp):
        return None
    with open(mp, "rb") as f:
        m = pickle.load(f)
    bs_to_id = m.get("bs_to_id", {})
    if not bs_to_id:
        return None
    # 加载位置模型
    model_name = f"location_{'node2vec' if Config.LOCATION_MODEL_TYPE=='node2vec' else 'item2vec'}_model.pth"
    loc_model_path = os.path.join(Config.MODEL_SAVE_PATH, model_name)
    if not os.path.exists(loc_model_path):
        return None
    bs_vocab = len(bs_to_id)
    loc_model = Item2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM) if Config.LOCATION_MODEL_TYPE == "item2vec" else Node2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM)
    ckpt = torch.load(loc_model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
    loc_model.load_state_dict(ckpt["model_state_dict"])
    item_emb = loc_model.get_embeddings(normalize=True)

    # 读取新位置数据并编码
    try:
        ldf = pd.read_csv(loc_path, sep="\t")
    except Exception:
        ldf = pd.read_csv(loc_path)
    if "user_id" not in ldf.columns:
        return None
    bs_col = None
    for cand in ["base_station_id", "bs_id", "station_id", "cell_id"]:
        if cand in ldf.columns:
            bs_col = cand
            break
    if bs_col is None:
        return None
    # 可选时间列
    ts_col = None
    for cand in ["timestamp_str", "timestamp", "time"]:
        if cand in ldf.columns:
            ts_col = cand
            break
    if ts_col is not None:
        try:
            ldf = ldf.assign(_ts=pd.to_datetime(ldf[ts_col], errors="coerce")).sort_values(["user_id", "_ts"]).drop(columns=["_ts"])
        except Exception:
            ldf = ldf.sort_values(["user_id", ts_col])

    # 聚合用户位置向量（支持duration权重）
    loc_embeddings = {}
    for uid, group in ldf.groupby("user_id"):
        ids = []
        weights = []
        for _, r in group.iterrows():
            b = str(r[bs_col])
            if b in bs_to_id:
                ids.append(bs_to_id[b])
                # 检查是否有duration列作为权重
                weight = 1.0
                if "duration" in r and pd.notna(r["duration"]):
                    duration_weight = float(r["duration"])
                    if duration_weight > 0:
                        # 使用与训练时相同的缩放逻辑
                        weight = duration_weight / Config.DURATION_WEIGHT_SCALE
                weights.append(weight)
        if not ids:
            continue
        
        # 使用权重进行加权平均
        if len(set(ids)) == len(ids):  # 没有重复的基站
            emb = item_emb[ids]
            weights = np.array(weights)
            weights = weights / weights.sum()  # 归一化权重
            v = np.average(emb, axis=0, weights=weights)
        else:
            # 有重复基站，需要先聚合权重
            id_weight_map = {}
            for id_, weight in zip(ids, weights):
                id_weight_map[id_] = id_weight_map.get(id_, 0) + weight
            
            unique_ids = list(id_weight_map.keys())
            unique_weights = np.array([id_weight_map[id_] for id_ in unique_ids])
            unique_weights = unique_weights / unique_weights.sum()  # 归一化权重
            
            emb = item_emb[unique_ids]
            v = np.average(emb, axis=0, weights=unique_weights)
        
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        loc_embeddings[uid] = v
    return loc_embeddings


def compute_new_user_fused_embeddings(behavior_model, url_mappings, save_path=None):
    """
    使用融合模型推理新用户向量：行为(必需)+属性(可选)+位置(可选)。
    若融合模型或所需资源缺失，将尽最大努力使用可用模态。
    """
    # 1) 行为新用户序列与用户行为向量
    training_entities = load_training_entities(Config.PROCESSED_DATA_PATH)
    new_sequences, unknown_urls = load_new_user_behavior(Config.NEW_USER_BEHAVIOR_PATH, url_mappings, training_entities)
    if not new_sequences:
        print("没有可用的新用户行为数据")
        return {}
    ue = UserEmbedding(behavior_model, new_sequences, url_mappings).compute()

    # 2) 属性原始编码（使用训练时 encoders/scaler）
    attr_raw = None
    attr_info = None
    if Config.ENABLE_ATTRIBUTES:
        attr_info, enc_pack = _load_attribute_resources()
        if attr_info is not None:
            attr_raw = _transform_new_attributes(Config.NEW_USER_ATTRIBUTE_PATH, attr_info, enc_pack)

    # 3) 位置向量
    loc_embeddings = _compute_new_location_embeddings() if Config.ENABLE_LOCATION else None

    # 4) 融合推理
    behavior_dim = Config.EMBEDDING_DIM
    loc_dim = None if loc_embeddings is None else Config.LOCATION_EMBEDDING_DIM
    ft = FusionTrainer(behavior_dim, attribute_info=attr_info, location_dim=loc_dim)
    # 加载融合模型
    fm = os.path.join(Config.MODEL_SAVE_PATH, "fusion_model.pth")
    if os.path.exists(fm):
        try:
            ft.model.load_state_dict(torch.load(fm, map_location=Config.DEVICE_OBJ, weights_only=False))
        except Exception:
            pass
    else:
        print(f"未找到融合模型: {fm}，将回退为行为向量。")
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(ue, f)
        return ue

    fused = ft.export_embeddings(ue, attribute_raw=attr_raw, location_embeddings=loc_embeddings)

    # 5) 保存与报告
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(fused, f)
    report = {
        "total_new_users": len(new_sequences),
        "unknown_urls": sorted(list(unknown_urls)),
        "known_url_count": len(training_entities["urls"]) if training_entities else 0,
        "used_attributes": bool(attr_raw is not None),
        "used_location": bool(loc_embeddings is not None),
        "fusion_model": os.path.exists(fm),
    }
    with open(os.path.join(Config.PROCESSED_DATA_PATH, "new_user_compatibility_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return fused




