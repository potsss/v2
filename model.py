"""
v2 模型定义：Item2Vec/Node2Vec + 用户聚合 + （可选）属性/融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import Config


def behavior_dropout_augmentation(behavior_embeddings, dropout_rate=None):
    """
    对行为向量进行dropout数据增强，生成两个略有不同的视图
    
    Args:
        behavior_embeddings: tensor [B, D] 行为向量
        dropout_rate: float dropout率，默认使用配置中的值
    
    Returns:
        tuple: (view1, view2) 两个增强后的视图
    """
    if dropout_rate is None:
        dropout_rate = Config.BEHAVIOR_DROPOUT_RATE
    
    # 创建两个不同的dropout掩码
    mask1 = torch.rand_like(behavior_embeddings) > dropout_rate
    mask2 = torch.rand_like(behavior_embeddings) > dropout_rate
    
    # 应用dropout并重新归一化
    view1 = behavior_embeddings * mask1.float()
    view2 = behavior_embeddings * mask2.float()
    
    # 重新归一化，避免因为dropout导致向量长度变化
    view1 = F.normalize(view1, p=2, dim=-1)
    view2 = F.normalize(view2, p=2, dim=-1)
    
    return view1, view2


def infonce_loss(anchor, positive, negatives, temperature=None):
    """
    计算InfoNCE对比学习损失
    
    Args:
        anchor: tensor [B, D] 锚点向量
        positive: tensor [B, D] 正样本向量  
        negatives: tensor [B, N, D] 负样本向量
        temperature: float 温度参数
    
    Returns:
        tensor: InfoNCE损失值
    """
    if temperature is None:
        temperature = Config.CONTRASTIVE_TEMPERATURE
    
    # 计算正样本相似度 [B]
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature
    
    # 计算负样本相似度 [B, N]  
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / temperature
    
    # 计算InfoNCE损失
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # 正样本标签为0
    
    loss = F.cross_entropy(logits, labels)
    return loss


class Item2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._init_weights()

    def _init_weights(self):
        init_range = 0.5 / self.embedding_dim
        nn.init.uniform_(self.in_embeddings.weight, -init_range, init_range)
        nn.init.constant_(self.out_embeddings.weight, 0)

    def forward(self, center_words, context_words, negative_words):
        center = self.in_embeddings(center_words)
        pos = self.out_embeddings(context_words)
        pos_score = torch.sum(center * pos, dim=1)
        pos_loss = F.logsigmoid(pos_score)
        neg = self.out_embeddings(negative_words)
        neg_score = torch.sum(neg * center.unsqueeze(1), dim=2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self, normalize=True):
        emb = self.in_embeddings.weight.data.cpu().numpy()
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb = emb / norms
        return emb


class Node2Vec(Item2Vec):
    pass


class UserEmbedding:
    def __init__(self, model, user_sequences, url_mappings):
        self.model = model
        self.user_sequences = user_sequences
        self.id_to_url = url_mappings["id_to_url"]
        self.user_embeddings = {}

    def compute(self, aggregation="mean"):
        item_embeddings = self.model.get_embeddings(normalize=True)
        for uid, seq in self.user_sequences.items():
            if not seq:
                continue
            
            if aggregation == "max":
                # 对于max聚合，我们仍然只考虑唯一项目
                items = list(set(seq))
                item_embeds = item_embeddings[items]
                ue = np.max(item_embeds, axis=0)
            else:
                # 对于mean聚合，考虑权重（序列中的重复次数）
                from collections import Counter
                item_counts = Counter(seq)
                items = list(item_counts.keys())
                weights = np.array([item_counts[item] for item in items])
                
                # 归一化权重
                weights = weights / weights.sum()
                
                # 加权平均
                item_embeds = item_embeddings[items]
                ue = np.average(item_embeds, axis=0, weights=weights)
            
            n = np.linalg.norm(ue)
            if n > 0:
                ue = ue / n
            self.user_embeddings[uid] = ue
        return self.user_embeddings


class AttributeEmbeddingModel(nn.Module):
    def __init__(self, attribute_info):
        super().__init__()
        self.attribute_info = attribute_info
        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_attrs = []
        self.numerical_attrs = []
        for name, info in attribute_info.items():
            if info.get("type") == "categorical":
                vocab = info.get("vocab_size", 1)
                self.categorical_embeddings[name] = nn.Embedding(vocab, Config.ATTRIBUTE_EMBEDDING_DIM)
                self.categorical_attrs.append(name)
            else:
                self.numerical_attrs.append(name)
        in_dim = len(self.categorical_attrs) * Config.ATTRIBUTE_EMBEDDING_DIM + len(self.numerical_attrs)
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, Config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(Config.FUSION_HIDDEN_DIM, Config.ATTRIBUTE_EMBEDDING_DIM),
        ) if in_dim > 0 else None

    def forward(self, categorical_inputs, numerical_inputs):
        parts = []
        # 计算 batch 大小与设备，用于缺失列补零
        batch_size = None
        device = None
        if categorical_inputs and len(categorical_inputs) > 0:
            any_tensor = next(iter(categorical_inputs.values()))
            batch_size = any_tensor.shape[0]
            device = any_tensor.device
        elif numerical_inputs is not None:
            batch_size = numerical_inputs.shape[0]
            device = numerical_inputs.device
        # 始终按 self.categorical_attrs 的顺序拼接，缺失列补零向量，避免维度漂移
        for name in self.categorical_attrs:
            if categorical_inputs is not None and name in categorical_inputs:
                parts.append(self.categorical_embeddings[name](categorical_inputs[name]))
            else:
                if batch_size is None:
                    # 无任何输入，直接返回 None
                    return None
                parts.append(torch.zeros(batch_size, Config.ATTRIBUTE_EMBEDDING_DIM, device=device))
        if self.numerical_attrs and numerical_inputs is not None:
            parts.append(numerical_inputs)
        if not parts:
            return None
        x = torch.cat(parts, dim=-1)
        if self.fusion is not None:
            x = self.fusion(x)
        return x


class UserFusionModel(nn.Module):
    def __init__(self, behavior_dim, attribute_dim=None, location_dim=None):
        super().__init__()
        in_dim = behavior_dim
        self.attribute_dim = attribute_dim
        self.location_dim = location_dim
        if attribute_dim is not None:
            in_dim += attribute_dim
        if location_dim is not None:
            in_dim += location_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, Config.FUSION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(Config.FUSION_HIDDEN_DIM, Config.FINAL_USER_EMBEDDING_DIM),
        )

    def forward(self, behavior, attribute=None, location=None):
        parts = [behavior]
        # 若该批次没有属性/位置输入，但模型初始化包含对应维度，则补零，保证输入维度恒定
        if attribute is not None:
            parts.append(attribute)
        elif self.attribute_dim is not None:
            parts.append(torch.zeros(behavior.shape[0], self.attribute_dim, device=behavior.device, dtype=behavior.dtype))
        if location is not None:
            parts.append(location)
        elif self.location_dim is not None:
            parts.append(torch.zeros(behavior.shape[0], self.location_dim, device=behavior.device, dtype=behavior.dtype))
        x = torch.cat(parts, dim=-1)
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)




