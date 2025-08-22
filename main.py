"""
v2 主入口：预处理、训练、可视化、计算嵌入、新用户
"""
import os
import argparse
import pickle
import torch
import numpy as np

# 兼容直接脚本运行：支持 `python main.py` 或 `python v2/main.py`
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "v2"

from .config import Config, init_paths
from .data import DataPreprocessorV2
from .model import Item2Vec, Node2Vec, UserEmbedding
from .trainer import TrainerV2
from .trainer import FusionTrainer
from .visualize import tsne_scatter
from .node2vec_utils import build_item_graph, generate_walks
from .new_users import save_training_entities, compute_new_user_embeddings


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess():
    pre = DataPreprocessorV2()
    user_sequences, url_mappings, user_attrs, attr_info = pre.preprocess()
    save_training_entities(url_mappings, Config.PROCESSED_DATA_PATH)
    print(f"预处理完成：用户={len(user_sequences)}, 物品={len(url_mappings['url_to_id'])}")
    return user_sequences, url_mappings, user_attrs, attr_info


def train(user_sequences, url_mappings):
    vocab_size = len(url_mappings["url_to_id"])
    model = Item2Vec(vocab_size, Config.EMBEDDING_DIM) if Config.MODEL_TYPE == "item2vec" else Node2Vec(vocab_size, Config.EMBEDDING_DIM)
    trainer = TrainerV2(model)
    if Config.MODEL_TYPE == "node2vec":
        graph = build_item_graph(user_sequences, directed=False)
        walks = generate_walks(graph, Config.NUM_WALKS, Config.WALK_LENGTH, Config.P_PARAM, Config.Q_PARAM)
        trainer.train(walks, save_prefix="")
    else:
        trainer.train(user_sequences, save_prefix="")
    trainer.save_model(model_type=Config.MODEL_TYPE)
    return model


def visualize(model, user_sequences, url_mappings):
    ue = UserEmbedding(model, user_sequences, url_mappings).compute()
    emb = list(ue.values())
    labels = list(ue.keys())
    if emb:
        path = tsne_scatter(
            embeddings=torch.tensor(emb).numpy(),
            labels=labels,
            title="用户嵌入向量_t-sne_可视化",
            sample_size=500,
        )
        print(f"可视化已保存: {path}")


def compute_embeddings(model, user_sequences, url_mappings):
    ue = UserEmbedding(model, user_sequences, url_mappings).compute()
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    out = os.path.join(Config.MODEL_SAVE_PATH, "user_embeddings.pkl")
    with open(out, 'wb') as f:
        pickle.dump(ue, f)
    print(f"用户嵌入已保存: {out}")


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["preprocess", "train", "visualize", "compute_embeddings", "compute_new_users", "train_fusion", "compute_fused_embeddings", "matrix_factorization", "all"], default="all")
    p.add_argument("--experiment_name", type=str, default=None)
    p.add_argument("--model_type", choices=["item2vec", "node2vec"], default=None)
    p.add_argument("--enable_attributes", action="store_true")
    p.add_argument("--enable_location", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    base = init_paths(args.experiment_name or Config.EXPERIMENT_NAME, mode=args.mode)
    set_seed(Config.RANDOM_SEED)

    if args.model_type:
        Config.MODEL_TYPE = args.model_type
    if args.enable_attributes:
        Config.ENABLE_ATTRIBUTES = True
    if args.enable_location:
        Config.ENABLE_LOCATION = True

    user_sequences = None
    url_mappings = None
    model = None

    if args.mode in ("preprocess", "all"):
        user_sequences, url_mappings, _, _ = preprocess()

    if args.mode not in ("preprocess",) and user_sequences is None:
        dp = DataPreprocessorV2()
        user_sequences, url_mappings = dp.load_processed()

    if args.mode in ("train", "all"):
        # 训练行为模型
        # 若需要断点续训，尝试加载行为检查点
        model = train(user_sequences, url_mappings)
        # 若启用位置，同步训练位置模型
        if Config.ENABLE_LOCATION:
            dp = DataPreprocessorV2()
            loc_seq, loc_map = dp.load_location_processed()
            if loc_seq and len(loc_seq) > 0:
                bs_vocab = len(loc_map["bs_to_id"]) if loc_map and "bs_to_id" in loc_map else 0
                if bs_vocab > 0:
                    loc_model = Item2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM) if Config.LOCATION_MODEL_TYPE == "item2vec" else Node2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM)
                    # 专属超参覆盖
                    ori_lr, ori_epochs, ori_bs, ori_ws, ori_neg = Config.LEARNING_RATE, Config.EPOCHS, Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.NEGATIVE_SAMPLES
                    try:
                        if Config.LOCATION_LEARNING_RATE is not None:
                            Config.LEARNING_RATE = Config.LOCATION_LEARNING_RATE
                        if Config.LOCATION_EPOCHS is not None:
                            Config.EPOCHS = Config.LOCATION_EPOCHS
                        if Config.LOCATION_BATCH_SIZE is not None:
                            Config.BATCH_SIZE = Config.LOCATION_BATCH_SIZE
                        if Config.LOCATION_WINDOW_SIZE is not None:
                            Config.WINDOW_SIZE = Config.LOCATION_WINDOW_SIZE
                        if Config.LOCATION_NEGATIVE_SAMPLES is not None:
                            Config.NEGATIVE_SAMPLES = Config.LOCATION_NEGATIVE_SAMPLES
                        loc_trainer = TrainerV2(loc_model)
                        if Config.LOCATION_MODEL_TYPE == "node2vec":
                            g = build_item_graph(loc_seq, directed=False)
                            walks = generate_walks(g, Config.NUM_WALKS, Config.WALK_LENGTH, Config.P_PARAM, Config.Q_PARAM)
                            loc_trainer.train(walks, save_prefix="location_", desc="Location Epoch", resume=args.resume)
                        else:
                            loc_trainer.train(loc_seq.values(), save_prefix="location_", desc="Location Epoch", resume=args.resume)
                        # 保存位置模型
                        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
                        torch.save({
                            "model_state_dict": loc_model.state_dict(),
                            "vocab_size": bs_vocab,
                            "embedding_dim": Config.LOCATION_EMBEDDING_DIM,
                            "model_type": Config.LOCATION_MODEL_TYPE,
                        }, os.path.join(Config.MODEL_SAVE_PATH, f"location_{'node2vec' if Config.LOCATION_MODEL_TYPE=='node2vec' else 'item2vec'}_model.pth"))
                    finally:
                        Config.LEARNING_RATE, Config.EPOCHS, Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.NEGATIVE_SAMPLES = ori_lr, ori_epochs, ori_bs, ori_ws, ori_neg

    if args.mode in ("visualize", "all"):
        if model is None:
            # 尝试加载已训练模型
            vocab = len(url_mappings["url_to_id"]) if url_mappings else 0
            model = Item2Vec(vocab, Config.EMBEDDING_DIM)
            model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{'item2vec' if Config.MODEL_TYPE=='item2vec' else 'node2vec'}_model.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
        if model is not None:
            visualize(model, user_sequences, url_mappings)

    if args.mode in ("compute_embeddings", "all"):
        if model is None:
            vocab = len(url_mappings["url_to_id"]) if url_mappings else 0
            model = Item2Vec(vocab, Config.EMBEDDING_DIM)
            model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{'item2vec' if Config.MODEL_TYPE=='item2vec' else 'node2vec'}_model.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
        if model is not None:
            compute_embeddings(model, user_sequences, url_mappings)

    if args.mode == "compute_new_users":
        if model is None:
            vocab = len(url_mappings["url_to_id"]) if url_mappings else 0
            model = Item2Vec(vocab, Config.EMBEDDING_DIM)
            model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{'item2vec' if Config.MODEL_TYPE=='item2vec' else 'node2vec'}_model.pth")
            if os.path.exists(model_path):
                ckpt = torch.load(model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
        out = os.path.join(Config.MODEL_SAVE_PATH, f"new_user_embeddings_{Config.MODEL_TYPE}.pkl")
        res = compute_new_user_embeddings(model, url_mappings, new_behavior_path=Config.NEW_USER_BEHAVIOR_PATH, save_path=out)
        print(f"新用户向量数量: {len(res)}")

    # 训练融合（无监督对齐头：让融合向量贴近行为向量）
    if args.mode == "train_fusion":
        # 1) 确保有行为向量与映射
        if user_sequences is None or url_mappings is None:
            dp = DataPreprocessorV2()
            user_sequences, url_mappings = dp.load_processed()
        vocab = len(url_mappings["url_to_id"]) if url_mappings else 0
        # 2) 行为用户向量
        if Config.MODEL_TYPE == "matrix_factorization":
            # 使用矩阵分解用户向量
            mf_path = os.path.join(Config.MODEL_SAVE_PATH, "matrix_factorization_user_embeddings.pkl")
            if os.path.exists(mf_path):
                import pickle as _p
                with open(mf_path, "rb") as f:
                    ue = _p.load(f)
                print("已加载矩阵分解用户向量")
            else:
                print("未找到矩阵分解用户向量，请先运行 --mode matrix_factorization")
                return
        else:
            # 使用Item2Vec/Node2Vec用户向量
            if model is None:
                model = Item2Vec(vocab, Config.EMBEDDING_DIM)
                model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{'item2vec' if Config.MODEL_TYPE=='item2vec' else 'node2vec'}_model.pth")
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
            ue = UserEmbedding(model, user_sequences, url_mappings).compute()
        # 3) 属性原始编码（若启用）：交给 AttributeEmbeddingModel 做列级嵌入+MLP 压缩
        attr_raw = None
        attr_info = None
        if Config.ENABLE_ATTRIBUTES:
            import pickle as _p
            ap = os.path.join(Config.PROCESSED_DATA_PATH, "user_attributes.pkl")
            aip = os.path.join(Config.PROCESSED_DATA_PATH, "attribute_info.pkl")
            if os.path.exists(ap) and os.path.exists(aip):
                with open(ap, "rb") as f:
                    attr_raw = _p.load(f)  # dict[uid] -> dict[col->encoded val]
                with open(aip, "rb") as f:
                    attr_info = _p.load(f)
        # 4) 位置用户向量（若启用）
        loc_embeddings = None
        if Config.ENABLE_LOCATION:
            dp = DataPreprocessorV2()
            loc_seq, loc_map = dp.load_location_processed()
            
            if loc_seq and len(loc_seq) > 0:
                if Config.ENABLE_LOCATION_FEATURES and dp.location_feature_processor and dp.location_feature_processor.location_features:
                    # 使用增强的位置特征嵌入
                    print("使用增强的位置特征嵌入...")
                    from .location_features import EnhancedLocationEmbedding
                    
                    if Config.MODEL_TYPE == "matrix_factorization":
                        # 矩阵分解模式：从矩阵分解模型中提取位置向量作为基础
                        als_model_path = os.path.join(Config.PROCESSED_DATA_PATH, "als_model.pkl")
                        mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "unified_item_mappings.pkl")
                        
                        if os.path.exists(als_model_path) and os.path.exists(mappings_path):
                            from .matrix_factorization import ALSMatrixFactorization, UnifiedItemProcessor
                            import pickle as _p
                            
                            # 加载ALS模型
                            als_model = ALSMatrixFactorization()
                            als_model.load_model(als_model_path)
                            
                            # 创建虚拟的位置模型用于增强嵌入
                            enhanced_loc_embedding = EnhancedLocationEmbedding(
                                als_model, loc_map, dp.location_feature_processor
                            )
                            loc_embeddings = enhanced_loc_embedding.compute_enhanced_user_location_embeddings(loc_seq)
                        else:
                            print(f"未找到矩阵分解模型，回退到基础位置处理")
                    else:
                        # Item2Vec/Node2Vec模式：使用独立的位置模型
                        bs_vocab = len(loc_map["bs_to_id"]) if loc_map and "bs_to_id" in loc_map else 0
                        if bs_vocab > 0:
                            model_name = f"location_{'node2vec' if Config.LOCATION_MODEL_TYPE=='node2vec' else 'item2vec'}_model.pth"
                            loc_model_path = os.path.join(Config.MODEL_SAVE_PATH, model_name)
                            if os.path.exists(loc_model_path):
                                loc_model = Item2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM) if Config.LOCATION_MODEL_TYPE == "item2vec" else Node2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM)
                                ckpt = torch.load(loc_model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                                loc_model.load_state_dict(ckpt["model_state_dict"])
                                
                                # 使用增强的位置嵌入
                                enhanced_loc_embedding = EnhancedLocationEmbedding(
                                    loc_model, loc_map, dp.location_feature_processor
                                )
                                loc_embeddings = enhanced_loc_embedding.compute_enhanced_user_location_embeddings(loc_seq)
                            else:
                                print(f"未找到已训练的位置模型: {loc_model_path}")
                else:
                    # 回退到原有的位置处理逻辑
                    print("使用基础位置嵌入...")
                    if Config.MODEL_TYPE == "matrix_factorization":
                        # 矩阵分解模式：从矩阵分解模型中提取位置向量
                        als_model_path = os.path.join(Config.PROCESSED_DATA_PATH, "als_model.pkl")
                        mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "unified_item_mappings.pkl")
                        
                        if os.path.exists(als_model_path) and os.path.exists(mappings_path):
                            from .matrix_factorization import ALSMatrixFactorization, UnifiedItemProcessor
                            import pickle as _p
                            
                            # 加载ALS模型
                            als_model = ALSMatrixFactorization()
                            als_model.load_model(als_model_path)
                            
                            # 加载物品映射
                            item_processor = UnifiedItemProcessor()
                            item_processor.load_mappings(mappings_path)
                            
                            # 提取位置相关的物品向量
                            item_embeddings = als_model.get_item_embeddings(normalize=True)
                            loc_embeddings = {}
                            
                            for uid, seq in loc_seq.items():
                                if not seq:
                                    continue
                                
                                # 收集该用户的位置物品向量和权重
                                vectors = []
                                weights = []
                                from collections import Counter
                                location_counts = Counter(seq)
                                
                                for location_id, count in location_counts.items():
                                    if location_id in item_processor.item_to_id:
                                        item_idx = item_processor.item_to_id[location_id]
                                        vectors.append(item_embeddings[item_idx])
                                        weights.append(float(count))
                                
                                if vectors:
                                    vectors = np.array(vectors)
                                    weights = np.array(weights)
                                    weights = weights / weights.sum()  # 归一化权重
                                    
                                    # 加权平均
                                    user_vector = np.average(vectors, axis=0, weights=weights)
                                    
                                    # 归一化
                                    norm = np.linalg.norm(user_vector)
                                    if norm > 0:
                                        user_vector = user_vector / norm
                                    
                                    loc_embeddings[uid] = user_vector
                            
                            print(f"从矩阵分解模型中提取了 {len(loc_embeddings)} 个用户的位置向量")
                        else:
                            print(f"未找到矩阵分解模型或映射文件，融合阶段将跳过位置特征")
                    else:
                        # Item2Vec/Node2Vec模式：使用独立的位置模型
                        bs_vocab = len(loc_map["bs_to_id"]) if loc_map and "bs_to_id" in loc_map else 0
                        if bs_vocab > 0:
                            # 仅加载已训练好的位置模型，不在融合阶段重复训练
                            model_name = f"location_{'node2vec' if Config.LOCATION_MODEL_TYPE=='node2vec' else 'item2vec'}_model.pth"
                            loc_model_path = os.path.join(Config.MODEL_SAVE_PATH, model_name)
                            if os.path.exists(loc_model_path):
                                loc_model = Item2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM) if Config.LOCATION_MODEL_TYPE == "item2vec" else Node2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM)
                                ckpt = torch.load(loc_model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                                loc_model.load_state_dict(ckpt["model_state_dict"])
                                # 用户位置向量聚合（支持权重）
                                item_emb = loc_model.get_embeddings(normalize=True)
                                loc_embeddings = {}
                                for uid, seq in loc_seq.items():
                                    if not seq:
                                        continue
                                    
                                    # 使用权重进行加权平均
                                    from collections import Counter
                                    item_counts = Counter(seq)
                                    items = list(item_counts.keys())
                                    weights = np.array([item_counts[item] for item in items])
                                    weights = weights / weights.sum()  # 归一化权重
                                    
                                    emb = item_emb[items]
                                    v = np.average(emb, axis=0, weights=weights)
                                    n = np.linalg.norm(v)
                                    if n > 0:
                                        v = v / n
                                    loc_embeddings[uid] = v
                                print(f"从独立位置模型中提取了 {len(loc_embeddings)} 个用户的位置向量")
                            else:
                                print(f"未找到已训练的位置模型: {loc_model_path}，融合阶段将跳过位置特征。请先在 --mode train 阶段启用位置训练。")
        # 5) 融合训练
        if Config.MODEL_TYPE == "matrix_factorization":
            behavior_dim = Config.MF_FACTORS
        else:
            behavior_dim = Config.EMBEDDING_DIM
        
        # 动态计算位置嵌入维度
        loc_dim = None
        if loc_embeddings is not None:
            # 从实际的位置嵌入中获取维度
            sample_embedding = next(iter(loc_embeddings.values()))
            loc_dim = sample_embedding.shape[0]
            print(f"检测到位置嵌入维度: {loc_dim}")
        
        ft = FusionTrainer(behavior_dim, attribute_info=attr_info, location_dim=loc_dim)
        # 优先使用对比学习+掩码属性预测混合训练
        if Config.ENABLE_CONTRASTIVE_LEARNING:
            ft.train_contrastive_map(ue, attribute_raw=attr_raw, location_embeddings=loc_embeddings, resume=args.resume)
        elif attr_info and any(v.get("type") == "categorical" for v in attr_info.values()):
            # 回退到原始掩码属性预测
            ft.train_masked_attribute_prediction(ue, attribute_raw=attr_raw, location_embeddings=loc_embeddings, resume=args.resume)
        else:
            # 回退到身份对齐
            ft.train_identity_alignment(ue, attribute_raw=attr_raw, location_embeddings=loc_embeddings, resume=args.resume)
        # 加载最佳模型并保存
        ft.load_best_model()
        torch.save(ft.model.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, "fusion_model.pth"))
        print("融合训练完成，已保存最佳模型为 fusion_model.pth")

    # 导出融合后的最终用户表示
    if args.mode == "compute_fused_embeddings":
        # 载入行为模型
        if user_sequences is None or url_mappings is None:
            dp = DataPreprocessorV2()
            user_sequences, url_mappings = dp.load_processed()
        vocab = len(url_mappings["url_to_id"]) if url_mappings else 0
        
        # 根据模型类型加载用户向量
        if Config.MODEL_TYPE == "matrix_factorization":
            # 使用矩阵分解用户向量
            mf_path = os.path.join(Config.MODEL_SAVE_PATH, "matrix_factorization_user_embeddings.pkl")
            if os.path.exists(mf_path):
                import pickle as _p
                with open(mf_path, "rb") as f:
                    ue = _p.load(f)
                print("已加载矩阵分解用户向量")
            else:
                print("未找到矩阵分解用户向量，请先运行 --mode matrix_factorization")
                return
        else:
            # 使用Item2Vec/Node2Vec用户向量
            if model is None:
                model = Item2Vec(vocab, Config.EMBEDDING_DIM)
                model_path = os.path.join(Config.MODEL_SAVE_PATH, f"{'item2vec' if Config.MODEL_TYPE=='item2vec' else 'node2vec'}_model.pth")
                if os.path.exists(model_path):
                    ckpt = torch.load(model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                    model.load_state_dict(ckpt["model_state_dict"])
            ue = UserEmbedding(model, user_sequences, url_mappings).compute()
        # 载入属性/位置
        attr_raw = None
        attr_info = None
        if Config.ENABLE_ATTRIBUTES:
            import pickle as _p
            ap = os.path.join(Config.PROCESSED_DATA_PATH, "user_attributes.pkl")
            aip = os.path.join(Config.PROCESSED_DATA_PATH, "attribute_info.pkl")
            if os.path.exists(ap) and os.path.exists(aip):
                with open(ap, "rb") as f:
                    attr_raw = _p.load(f)
                with open(aip, "rb") as f:
                    attr_info = _p.load(f)
        # 4) 位置用户向量（若启用）- 与train_fusion保持一致的逻辑
        loc_embeddings = None
        if Config.ENABLE_LOCATION:
            dp = DataPreprocessorV2()
            loc_seq, loc_map = dp.load_location_processed()
            
            if loc_seq and len(loc_seq) > 0:
                if Config.ENABLE_LOCATION_FEATURES and dp.location_feature_processor and dp.location_feature_processor.location_features:
                    # 使用增强的位置特征嵌入（与train_fusion保持一致）
                    print("使用增强的位置特征嵌入...")
                    from .location_features import EnhancedLocationEmbedding
                    
                    if Config.MODEL_TYPE == "matrix_factorization":
                        # 矩阵分解模式：从矩阵分解模型中提取位置向量作为基础
                        als_model_path = os.path.join(Config.PROCESSED_DATA_PATH, "als_model.pkl")
                        mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "unified_item_mappings.pkl")
                        
                        if os.path.exists(als_model_path) and os.path.exists(mappings_path):
                            from .matrix_factorization import ALSMatrixFactorization, UnifiedItemProcessor
                            import pickle as _p
                            
                            # 加载ALS模型
                            als_model = ALSMatrixFactorization()
                            als_model.load_model(als_model_path)
                            
                            # 创建虚拟的位置模型用于增强嵌入
                            enhanced_loc_embedding = EnhancedLocationEmbedding(
                                als_model, loc_map, dp.location_feature_processor
                            )
                            loc_embeddings = enhanced_loc_embedding.compute_enhanced_user_location_embeddings(loc_seq)
                        else:
                            print(f"未找到矩阵分解模型，回退到基础位置处理")
                    else:
                        # Item2Vec/Node2Vec模式：使用独立的位置模型
                        if loc_map and "bs_to_id" in loc_map:
                            bs_vocab = len(loc_map["bs_to_id"])
                            model_name = f"location_{'node2vec' if Config.LOCATION_MODEL_TYPE=='node2vec' else 'item2vec'}_model.pth"
                            loc_model_path = os.path.join(Config.MODEL_SAVE_PATH, model_name)
                            if os.path.exists(loc_model_path):
                                loc_model = Item2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM) if Config.LOCATION_MODEL_TYPE == "item2vec" else Node2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM)
                                ckpt = torch.load(loc_model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                                loc_model.load_state_dict(ckpt["model_state_dict"])
                                
                                # 使用增强的位置嵌入
                                enhanced_loc_embedding = EnhancedLocationEmbedding(
                                    loc_model, loc_map, dp.location_feature_processor
                                )
                                loc_embeddings = enhanced_loc_embedding.compute_enhanced_user_location_embeddings(loc_seq)
                            else:
                                print(f"未找到已训练的位置模型: {loc_model_path}")
                else:
                    # 回退到原有的位置处理逻辑
                    print("使用基础位置嵌入...")
                    if Config.MODEL_TYPE == "matrix_factorization":
                        # 矩阵分解模式：从矩阵分解模型中提取位置向量（与train_fusion保持一致）
                        als_model_path = os.path.join(Config.PROCESSED_DATA_PATH, "als_model.pkl")
                        mappings_path = os.path.join(Config.PROCESSED_DATA_PATH, "unified_item_mappings.pkl")
                        
                        if os.path.exists(als_model_path) and os.path.exists(mappings_path):
                            from .matrix_factorization import ALSMatrixFactorization, UnifiedItemProcessor
                            import pickle as _p
                            
                            # 加载ALS模型
                            als_model = ALSMatrixFactorization()
                            als_model.load_model(als_model_path)
                            
                            # 加载物品映射
                            item_processor = UnifiedItemProcessor()
                            item_processor.load_mappings(mappings_path)
                            
                            # 提取位置相关的物品向量
                            item_embeddings = als_model.get_item_embeddings(normalize=True)
                            loc_embeddings = {}
                            
                            for uid, seq in loc_seq.items():
                                if not seq:
                                    continue
                                
                                # 收集该用户的位置物品向量和权重
                                vectors = []
                                weights = []
                                from collections import Counter
                                location_counts = Counter(seq)
                                
                                for location_id, count in location_counts.items():
                                    if location_id in item_processor.item_to_id:
                                        item_idx = item_processor.item_to_id[location_id]
                                        vectors.append(item_embeddings[item_idx])
                                        weights.append(float(count))
                                
                                if vectors:
                                    vectors = np.array(vectors)
                                    weights = np.array(weights)
                                    weights = weights / weights.sum()  # 归一化权重
                                    
                                    # 加权平均
                                    user_vector = np.average(vectors, axis=0, weights=weights)
                                    
                                    # 归一化
                                    norm = np.linalg.norm(user_vector)
                                    if norm > 0:
                                        user_vector = user_vector / norm
                                    
                                    loc_embeddings[uid] = user_vector
                            
                            print(f"从矩阵分解模型中提取了 {len(loc_embeddings)} 个用户的位置向量")
                        else:
                            print(f"未找到矩阵分解模型或映射文件，融合阶段将跳过位置特征")
                    else:
                        # Item2Vec/Node2Vec模式：使用独立的位置模型
                        if loc_map and "bs_to_id" in loc_map:
                            bs_vocab = len(loc_map["bs_to_id"])
                            # 仅加载已训练好的位置模型，不在此阶段训练
                            model_name = f"location_{'node2vec' if Config.LOCATION_MODEL_TYPE=='node2vec' else 'item2vec'}_model.pth"
                            loc_model_path = os.path.join(Config.MODEL_SAVE_PATH, model_name)
                            if os.path.exists(loc_model_path):
                                loc_model = Item2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM) if Config.LOCATION_MODEL_TYPE == "item2vec" else Node2Vec(bs_vocab, Config.LOCATION_EMBEDDING_DIM)
                                ckpt = torch.load(loc_model_path, map_location=Config.DEVICE_OBJ, weights_only=False)
                                loc_model.load_state_dict(ckpt["model_state_dict"])
                                # 用户位置向量聚合（支持权重）
                                item_emb = loc_model.get_embeddings(normalize=True)
                                loc_embeddings = {}
                                for uid, seq in loc_seq.items():
                                    if not seq:
                                        continue
                                    
                                    # 使用权重进行加权平均
                                    from collections import Counter
                                    item_counts = Counter(seq)
                                    items = list(item_counts.keys())
                                    weights = np.array([item_counts[item] for item in items])
                                    weights = weights / weights.sum()  # 归一化权重
                                    
                                    emb = item_emb[items]
                                    v = np.average(emb, axis=0, weights=weights)
                                    n = np.linalg.norm(v)
                                    if n > 0:
                                        v = v / n
                                    loc_embeddings[uid] = v
                                print(f"从独立位置模型中提取了 {len(loc_embeddings)} 个用户的位置向量")
                            else:
                                print(f"未找到已训练的位置模型: {loc_model_path}，将跳过位置特征。")
        # 载入融合模型
        if Config.MODEL_TYPE == "matrix_factorization":
            behavior_dim = Config.MF_FACTORS
        else:
            behavior_dim = Config.EMBEDDING_DIM
        
        # 动态计算位置嵌入维度
        loc_dim = None
        if loc_embeddings is not None:
            # 从实际的位置嵌入中获取维度
            sample_embedding = next(iter(loc_embeddings.values()))
            loc_dim = sample_embedding.shape[0]
            print(f"检测到位置嵌入维度: {loc_dim}")
        
        ft = FusionTrainer(behavior_dim, attribute_info=attr_info, location_dim=loc_dim)
        fm = os.path.join(Config.MODEL_SAVE_PATH, "fusion_model.pth")
        if os.path.exists(fm):
            ft.model.load_state_dict(torch.load(fm, map_location=Config.DEVICE_OBJ, weights_only=False))
        fused = ft.export_embeddings(ue, attribute_raw=attr_raw, location_embeddings=loc_embeddings)
        out = os.path.join(Config.MODEL_SAVE_PATH, "fused_user_embeddings.pkl")
        import pickle as _p
        with open(out, "wb") as f:
            _p.dump(fused, f)
        print(f"融合用户向量已保存: {out}")

    # 矩阵分解模式
    if args.mode == "matrix_factorization":
        dp = DataPreprocessorV2()
        
        # 确保基础数据已处理
        user_sequences, url_mappings = dp.load_processed()
        if user_sequences is None:
            print("未找到已处理的行为数据，开始预处理...")
            user_sequences, url_mappings, _, _ = dp.preprocess()
            # 保存预处理结果
            os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
            import pickle
            with open(os.path.join(Config.PROCESSED_DATA_PATH, "url_mappings.pkl"), "wb") as f:
                pickle.dump({"url_to_id": dp.url_to_id, "id_to_url": dp.id_to_url}, f)
            with open(os.path.join(Config.PROCESSED_DATA_PATH, "user_sequences.pkl"), "wb") as f:
                pickle.dump(dict(dp.user_sequences), f)
        
        # 处理位置数据（如果启用）
        if Config.ENABLE_LOCATION:
            location_sequences, location_mappings = dp.load_location_processed()
            if location_sequences is None:
                print("未找到已处理的位置数据，开始处理...")
                dp.process_location()
                # 重新加载位置数据
                location_sequences, location_mappings = dp.load_location_processed()
        
        # 处理矩阵分解
        success = dp.process_matrix_factorization()
        if success:
            print("矩阵分解处理完成！")
            
            # 计算用户向量
            print("\n计算基于矩阵分解的用户向量...")
            user_embeddings = dp.compute_mf_user_embeddings(method=Config.MF_USER_AGGREGATION)
            
            if user_embeddings:
                # 保存用户向量
                output_path = os.path.join(Config.MODEL_SAVE_PATH, "matrix_factorization_user_embeddings.pkl")
                os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
                import pickle as _p
                with open(output_path, "wb") as f:
                    _p.dump(user_embeddings, f)
                print(f"矩阵分解用户向量已保存: {output_path}")
        else:
            print("矩阵分解处理失败")


if __name__ == "__main__":
    cli()





