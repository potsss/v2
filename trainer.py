"""
v2 训练器（统一 Skip-gram 训练 + 早停 + 检查点）
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from .config import Config
from .model import UserFusionModel, AttributeEmbeddingModel, behavior_dropout_augmentation, infonce_loss
import random


class SkipGramDataset(Dataset):
    def __init__(self, sequences, window_size, negative_samples, vocab_size, use_weights=True):
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.vocab_size = vocab_size
        self.use_weights = use_weights
        self.samples = []
        self.weights = []
        
        for seq in sequences:
            if self.use_weights:
                # 计算序列中每个项目的权重（基于出现频次）
                from collections import Counter
                item_counts = Counter(seq)
                total_count = len(seq)
                
                # 为每个唯一的项目对生成样本，权重基于共现频次
                unique_seq = list(set(seq))
                for i, c in enumerate(unique_seq):
                    for j, ctx in enumerate(unique_seq):
                        if i == j:
                            continue
                        # 权重基于两个项目在序列中的共现概率
                        weight = (item_counts[c] * item_counts[ctx]) / (total_count * total_count)
                        self.samples.append((c, ctx))
                        self.weights.append(weight)
            else:
                # 原始逻辑：基于滑动窗口
                for i, c in enumerate(seq):
                    left = max(0, i - window_size)
                    right = min(len(seq), i + window_size + 1)
                    for j in range(left, right):
                        if i == j:
                            continue
                        self.samples.append((c, seq[j]))
                        self.weights.append(1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center, context = self.samples[idx]
        weight = self.weights[idx] if self.use_weights else 1.0
        negatives = np.random.randint(0, self.vocab_size, size=(self.negative_samples,))
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
            torch.tensor(weight, dtype=torch.float),
        )


class TrainerV2:
    def __init__(self, model):
        self.model = model.to(Config.DEVICE_OBJ)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.best_loss = float("inf")

    def _build_loader(self, sequences, use_weights=None):
        if use_weights is None:
            use_weights = Config.USE_WEIGHTS_IN_TRAINING
        if isinstance(sequences, dict):
            seqs = list(sequences.values())
        else:
            seqs = sequences
        ds = SkipGramDataset(seqs, Config.WINDOW_SIZE, Config.NEGATIVE_SAMPLES, self.model.vocab_size, use_weights)
        return DataLoader(ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    def _checkpoint_path(self, prefix: str):
        return os.path.join(Config.CHECKPOINT_DIR, f"{prefix}latest_checkpoint.pth")

    def load_checkpoint(self, prefix: str = ""):
        path = self._checkpoint_path(prefix)
        if not os.path.exists(path):
            return -1
        ckpt = torch.load(path, map_location=Config.DEVICE_OBJ, weights_only=False)
        self.model.load_state_dict(ckpt.get("model_state_dict", {}))
        if "opt_state_dict" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["opt_state_dict"])
            except Exception:
                pass
        self.best_loss = ckpt.get("loss", float("inf"))
        return ckpt.get("epoch", -1)

    def train(self, sequences, save_prefix="", desc="Epoch", resume: bool = False):
        dl = self._build_loader(sequences)
        start_epoch = 0
        if resume:
            last = self.load_checkpoint(save_prefix)
            if last >= 0:
                start_epoch = last + 1
        patience = 0
        for epoch in range(start_epoch, Config.EPOCHS):
            self.model.train()
            total = 0.0
            n = 0
            pbar = tqdm(dl, desc=f"{desc} {epoch+1}/{Config.EPOCHS}")
            for batch in pbar:
                if len(batch) == 4:  # 带权重的批次
                    center, context, neg, weights = batch
                    weights = weights.to(Config.DEVICE_OBJ)
                else:  # 不带权重的批次（向后兼容）
                    center, context, neg = batch
                    weights = torch.ones(center.size(0), device=Config.DEVICE_OBJ)
                
                center = center.to(Config.DEVICE_OBJ)
                context = context.to(Config.DEVICE_OBJ)
                neg = neg.to(Config.DEVICE_OBJ)
                
                loss = self.model(center, context, neg)
                # 应用权重
                weighted_loss = (loss * weights).mean()
                
                self.opt.zero_grad()
                weighted_loss.backward()
                self.opt.step()
                total += weighted_loss.item()
                n += 1
                pbar.set_postfix(loss=weighted_loss.item())
            avg = total / max(n, 1)
            is_best = avg < self.best_loss
            if is_best:
                self.best_loss = avg
                patience = 0
            else:
                patience += 1
            self._save_checkpoint(epoch, avg, is_best, prefix=save_prefix)
            if patience >= Config.EARLY_STOPPING_PATIENCE:
                break

    def _save_checkpoint(self, epoch, loss, is_best=False, prefix=""):
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss": loss,
            "vocab_size": self.model.vocab_size,
            "embedding_dim": self.model.embedding_dim,
            "opt_state_dict": self.opt.state_dict(),
        }
        path = os.path.join(Config.CHECKPOINT_DIR, f"{prefix}latest_checkpoint.pth")
        torch.save(ckpt, path)
        if is_best:
            best = os.path.join(Config.CHECKPOINT_DIR, f"{prefix}best_model.pth")
            torch.save(ckpt, best)

    def save_model(self, model_type="item2vec", prefix=""):
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        name = f"{prefix}{'node2vec' if model_type=='node2vec' else 'item2vec'}_model.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocab_size": self.model.vocab_size,
            "embedding_dim": self.model.embedding_dim,
            "model_type": model_type,
        }, os.path.join(Config.MODEL_SAVE_PATH, name))


class FusionDataset(Dataset):
    def __init__(self, behavior_embeddings, attribute_raw=None, location_embeddings=None, user_ids=None):
        # behavior_embeddings: dict[user_id] -> np.ndarray
        self.user_ids = user_ids or list(behavior_embeddings.keys())
        self.behavior = behavior_embeddings
        self.attr_raw = attribute_raw  # dict[user_id] -> dict[col->encoded value]
        self.loc = location_embeddings  # dict[user_id] -> np.ndarray

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        b = torch.tensor(self.behavior[uid], dtype=torch.float32)
        a_raw = None
        l = None
        if self.attr_raw is not None and uid in self.attr_raw:
            a_raw = self.attr_raw[uid]
        if self.loc is not None and uid in self.loc:
            l = torch.tensor(self.loc[uid], dtype=torch.float32)
        return uid, b, a_raw, l


class FusionTrainer:
    def __init__(self, behavior_dim, attribute_info=None, location_dim=None):
        self.attribute_info = attribute_info
        self.categorical_attrs = []
        self.numerical_attrs = []
        attribute_dim = None
        if attribute_info is not None:
            for name, info in attribute_info.items():
                if info.get("type") == "categorical":
                    self.categorical_attrs.append(name)
                else:
                    self.numerical_attrs.append(name)
            self.attribute_model = AttributeEmbeddingModel(attribute_info).to(Config.DEVICE_OBJ)
            attribute_dim = Config.ATTRIBUTE_EMBEDDING_DIM
        else:
            self.attribute_model = None
        self.model = UserFusionModel(behavior_dim, attribute_dim, location_dim).to(Config.DEVICE_OBJ)
        # 初始化属性预测头（MAP 用）。为所有类别列建立线性头，便于断点恢复加载
        self.attr_heads = None
        if self.attribute_info is not None and len(self.categorical_attrs) > 0:
            self.attr_heads = nn.ModuleDict()
            for name in self.categorical_attrs:
                vocab = self.attribute_info.get(name, {}).get("vocab_size", 1)
                self.attr_heads[name] = nn.Linear(Config.FINAL_USER_EMBEDDING_DIM, vocab).to(Config.DEVICE_OBJ)

        params = list(self.model.parameters())
        if self.attribute_model is not None:
            params += list(self.attribute_model.parameters())
        if self.attr_heads is not None:
            params += list(self.attr_heads.parameters())
        
        # 使用Adam优化器，加入权重衰减
        self.opt = torch.optim.Adam(
            params, 
            lr=Config.FUSION_LEARNING_RATE,
            weight_decay=Config.FUSION_WEIGHT_DECAY
        )
        
        # 学习率调度器：ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=Config.FUSION_LR_DECAY_FACTOR,
            patience=Config.FUSION_LR_DECAY_PATIENCE,
            verbose=False,  # 关闭verbose避免警告
            min_lr=1e-6
        )
        
        # 早停相关
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 初始化融合网络权重
        self._init_fusion_weights()

    def _init_fusion_weights(self):
        """
        初始化融合网络权重，使用Xavier初始化
        """
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        
        self.model.apply(init_weights)
        if self.attribute_model is not None:
            self.attribute_model.apply(init_weights)
        if self.attr_heads is not None:
            for head in self.attr_heads.values():
                init_weights(head)

    def _get_warmup_lr(self, epoch):
        """
        计算预热阶段的学习率
        """
        if epoch < Config.FUSION_WARMUP_EPOCHS:
            # 线性预热
            return Config.FUSION_LEARNING_RATE * (epoch + 1) / Config.FUSION_WARMUP_EPOCHS
        return Config.FUSION_LEARNING_RATE

    def _update_learning_rate(self, epoch):
        """
        更新学习率（预热阶段）
        """
        if epoch < Config.FUSION_WARMUP_EPOCHS:
            lr = self._get_warmup_lr(epoch)
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr

    def _build_loader(self, behavior_embeddings, attribute_raw=None, location_embeddings=None, user_ids=None):
        ds = FusionDataset(behavior_embeddings, attribute_raw, location_embeddings, user_ids)
        return DataLoader(ds, batch_size=Config.FUSION_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False, collate_fn=self._collate)

    def _collate(self, batch):
        # batch: [(uid, b, a_raw(dict) or None, l or None), ...]
        uids = [x[0] for x in batch]
        b = torch.stack([x[1] for x in batch], dim=0)
        # attributes
        cat_inputs = None
        cat_labels = None
        num_inputs = None
        if self.attribute_info is not None:
            # 构造类别张量 dict[name]->LongTensor[B]
            cat_inputs = {}
            cat_labels = {}
            for name in self.categorical_attrs:
                vals = []
                for _, _, a_raw, _ in batch:
                    v = 0
                    if a_raw is not None and name in a_raw:
                        v = int(a_raw[name])
                    vals.append(v)
                t = torch.tensor(vals, dtype=torch.long)
                cat_inputs[name] = t
                cat_labels[name] = t.clone()
            # 数值张量 [B, num_numerical]
            if len(self.numerical_attrs) > 0:
                rows = []
                for _, _, a_raw, _ in batch:
                    row = []
                    for name in self.numerical_attrs:
                        v = 0.0
                        if a_raw is not None and name in a_raw:
                            v = float(a_raw[name])
                        row.append(v)
                    rows.append(row)
                num_inputs = torch.tensor(rows, dtype=torch.float32)
        # location
        l_list = [x[3] for x in batch]
        l_tensor = None
        if all(v is not None for v in l_list) and len(l_list) > 0:
            l_tensor = torch.stack(l_list, dim=0)
        return uids, b, (cat_inputs, num_inputs, cat_labels), l_tensor

    def _fusion_ckpt_path(self):
        return os.path.join(Config.CHECKPOINT_DIR, "fusion_latest_checkpoint.pth")

    def _save_fusion_checkpoint(self, epoch, loss, is_best=False):
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "attribute_model_state_dict": (self.attribute_model.state_dict() if self.attribute_model is not None else None),
            "attr_heads_state_dict": (self.attr_heads.state_dict() if self.attr_heads is not None else None),
            "opt_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss),
            "best_loss": self.best_loss,
            "patience_counter": self.patience_counter,
        }
        
        # 保存最新检查点
        torch.save(ckpt, self._fusion_ckpt_path())
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(Config.CHECKPOINT_DIR, "fusion_best_checkpoint.pth")
            torch.save(ckpt, best_path)

    def _load_fusion_checkpoint(self):
        path = self._fusion_ckpt_path()
        if not os.path.exists(path):
            return -1
        ckpt = torch.load(path, map_location=Config.DEVICE_OBJ, weights_only=False)
        self.model.load_state_dict(ckpt.get("model_state_dict", {}))
        
        if self.attribute_model is not None and ckpt.get("attribute_model_state_dict"):
            try:
                self.attribute_model.load_state_dict(ckpt["attribute_model_state_dict"])
            except Exception:
                pass
                
        if self.attr_heads is not None and ckpt.get("attr_heads_state_dict"):
            try:
                self.attr_heads.load_state_dict(ckpt["attr_heads_state_dict"])
            except Exception:
                pass
                
        if ckpt.get("opt_state_dict"):
            try:
                self.opt.load_state_dict(ckpt["opt_state_dict"])
            except Exception:
                pass
                
        if ckpt.get("scheduler_state_dict"):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        
        # 恢复早停相关状态
        self.best_loss = ckpt.get("best_loss", float('inf'))
        self.patience_counter = ckpt.get("patience_counter", 0)
        
        return ckpt.get("epoch", -1)

    def load_best_model(self):
        """
        加载最佳模型权重（用于推理）
        """
        best_path = os.path.join(Config.CHECKPOINT_DIR, "fusion_best_checkpoint.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=Config.DEVICE_OBJ, weights_only=False)
            self.model.load_state_dict(ckpt.get("model_state_dict", {}))
            if self.attribute_model is not None and ckpt.get("attribute_model_state_dict"):
                self.attribute_model.load_state_dict(ckpt["attribute_model_state_dict"])
            print(f"已加载最佳模型，损失: {ckpt.get('loss', 'N/A'):.6f}")
            return True
        else:
            print("未找到最佳模型检查点，使用当前模型")
            return False

    def train_identity_alignment(self, behavior_embeddings, attribute_raw=None, location_embeddings=None, user_ids=None, resume: bool = False):
        """
        最小可用无监督头：身份对齐（InfoNCE 的温和替代）。
        目标：让同一用户的行为向量与融合向量相似（最大化 cos 相似），
             与 batch 内其他用户的融合向量不相似（最小化平均负相似）。
        """
        dl = self._build_loader(behavior_embeddings, attribute_raw, location_embeddings, user_ids)
        self.model.train()
        if self.attribute_model is not None:
            self.attribute_model.train()
        start_epoch = 0
        if resume:
            last = self._load_fusion_checkpoint()
            if last >= 0:
                start_epoch = last + 1
        for epoch in range(start_epoch, Config.FUSION_EPOCHS):
            pbar = tqdm(dl, desc=f"Fusion Epoch {epoch+1}/{Config.FUSION_EPOCHS}")
            for uids, b, attr_pack, l in pbar:
                b = b.to(Config.DEVICE_OBJ)
                a_vec = None
                if self.attribute_model is not None:
                    cat_inputs, num_inputs, _ = attr_pack
                    # move to device
                    for k in cat_inputs.keys():
                        cat_inputs[k] = cat_inputs[k].to(Config.DEVICE_OBJ)
                    if num_inputs is not None:
                        num_inputs = num_inputs.to(Config.DEVICE_OBJ)
                    a_vec = self.attribute_model(cat_inputs, num_inputs)
                if l is not None:
                    l = l.to(Config.DEVICE_OBJ)
                z = self.model(b, a_vec, l)  # fused vec
                b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
                # 正样本：对角项相似度；负样本：其他项
                sim = b_norm @ z.t()  # [B, B]
                pos = torch.diag(sim)
                neg = (sim.sum(dim=1) - pos) / (sim.size(1) - 1)
                loss = -pos.mean() + neg.mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                pbar.set_postfix(loss=float(loss.detach().cpu()))
            self._save_fusion_checkpoint(epoch, loss)

    def train_masked_attribute_prediction(self, behavior_embeddings, attribute_raw=None, location_embeddings=None, user_ids=None, resume: bool = False):
        """
        掩码属性预测（MAP）：随机遮盖一个类别属性，不将其输入属性编码器，
        使用行为+其余属性(+位置)的融合表示预测被遮盖属性的类别，
        用交叉熵训练属性列嵌入、融合网络（自监督）。
        """
        if self.attribute_model is None or len(self.categorical_attrs) == 0:
            # 回退到身份对齐
            return self.train_identity_alignment(behavior_embeddings, attribute_raw, location_embeddings, user_ids)
        dl = self._build_loader(behavior_embeddings, attribute_raw, location_embeddings, user_ids)
        self.model.train()
        self.attribute_model.train()
        criterion = nn.CrossEntropyLoss()
        start_epoch = 0
        if resume:
            last = self._load_fusion_checkpoint()
            if last >= 0:
                start_epoch = last + 1
        for epoch in range(start_epoch, Config.FUSION_EPOCHS):
            pbar = tqdm(dl, desc=f"Fusion MAP Epoch {epoch+1}/{Config.FUSION_EPOCHS}")
            for uids, b, attr_pack, l in pbar:
                b = b.to(Config.DEVICE_OBJ)
                cat_inputs, num_inputs, cat_labels = attr_pack
                # 随机选择一个存在的类别属性为本 batch 的预测目标
                valid_cats = [n for n in self.categorical_attrs if n in cat_inputs]
                if not valid_cats:
                    continue
                target_name = random.choice(valid_cats)
                # 构造遮盖后的类别输入：去掉目标列
                masked_inputs = {}
                for k, v in cat_inputs.items():
                    if k == target_name:
                        continue
                    masked_inputs[k] = v.to(Config.DEVICE_OBJ)
                if num_inputs is not None:
                    num_inputs = num_inputs.to(Config.DEVICE_OBJ)
                # 前向：属性向量 + 融合
                a_vec = self.attribute_model(masked_inputs, num_inputs)
                if l is not None:
                    l = l.to(Config.DEVICE_OBJ)
                z = self.model(b, a_vec, l)
                # 预测被遮盖属性
                logits = self.attr_heads[target_name](z)
                y = cat_labels[target_name].to(Config.DEVICE_OBJ)
                loss = criterion(logits, y)
                # 反向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                pbar.set_postfix(loss=float(loss.detach().cpu()))
            self._save_fusion_checkpoint(epoch, loss)

    def train_contrastive_map(self, behavior_embeddings, attribute_raw=None, location_embeddings=None, user_ids=None, resume: bool = False):
        """
        混合训练方法：掩码属性预测 + 对比学习
        
        对比学习部分：
        1. 对行为向量进行两次不同的dropout增强，生成两个视图
        2. 分别与属性、位置信息融合，得到两个用户向量
        3. 使用InfoNCE损失拉近正样本对，推远负样本
        
        掩码预测部分：
        1. 随机遮盖一个类别属性
        2. 使用其余信息预测被遮盖属性
        3. 使用交叉熵损失训练
        
        最终损失 = MAP损失 * MAP权重 + 对比损失 * 对比权重
        """
        if not Config.ENABLE_CONTRASTIVE_LEARNING:
            # 如果未启用对比学习，回退到原始MAP训练
            return self.train_masked_attribute_prediction(behavior_embeddings, attribute_raw, location_embeddings, user_ids, resume)
            
        if self.attribute_model is None or len(self.categorical_attrs) == 0:
            # 如果没有类别属性，回退到纯对比学习训练
            return self.train_contrastive_only(behavior_embeddings, attribute_raw, location_embeddings, user_ids, resume)
            
        dl = self._build_loader(behavior_embeddings, attribute_raw, location_embeddings, user_ids)
        self.model.train()
        self.attribute_model.train()
        criterion = nn.CrossEntropyLoss()
        
        start_epoch = 0
        if resume:
            last = self._load_fusion_checkpoint()
            if last >= 0:
                start_epoch = last + 1
                
        for epoch in range(start_epoch, Config.FUSION_EPOCHS):
            # 更新学习率（预热阶段）
            self._update_learning_rate(epoch)
            
            pbar = tqdm(dl, desc=f"Contrastive+MAP Epoch {epoch+1}/{Config.FUSION_EPOCHS}")
            total_map_loss = 0.0
            total_contrastive_loss = 0.0
            total_loss = 0.0
            n_batches = 0
            
            for uids, b, attr_pack, l in pbar:
                b = b.to(Config.DEVICE_OBJ)
                cat_inputs, num_inputs, cat_labels = attr_pack
                
                # ====== 对比学习部分 ======
                contrastive_loss = 0.0
                if Config.ENABLE_CONTRASTIVE_LEARNING and b.size(0) > 1:  # 需要至少2个样本
                    # 1. 对行为向量进行两次不同的dropout增强
                    b_view1, b_view2 = behavior_dropout_augmentation(b)
                    
                    # 2. 处理属性向量（两个视图使用相同的属性）
                    a_vec = None
                    if self.attribute_model is not None:
                        for k in cat_inputs.keys():
                            cat_inputs[k] = cat_inputs[k].to(Config.DEVICE_OBJ)
                        if num_inputs is not None:
                            num_inputs = num_inputs.to(Config.DEVICE_OBJ)
                        a_vec = self.attribute_model(cat_inputs, num_inputs)
                    
                    # 3. 处理位置向量
                    l_vec = None
                    if l is not None:
                        l_vec = l.to(Config.DEVICE_OBJ)
                    
                    # 4. 分别融合两个视图
                    z1 = self.model(b_view1, a_vec, l_vec)  # 视图1的融合向量
                    z2 = self.model(b_view2, a_vec, l_vec)  # 视图2的融合向量
                    
                    # 5. 构造正负样本并计算InfoNCE损失
                    batch_size = z1.size(0)
                    if batch_size > 1:
                        # 对于每个样本，z2是正样本，其他所有z1是负样本
                        negatives = []
                        for i in range(batch_size):
                            # 为第i个样本构造负样本：除了自己之外的所有z1
                            neg_indices = [j for j in range(batch_size) if j != i]
                            neg_samples = z1[neg_indices]  # [N-1, D]
                            negatives.append(neg_samples)
                        
                        # 计算每个样本的InfoNCE损失并平均
                        sample_losses = []
                        for i in range(batch_size):
                            anchor = z1[i:i+1]  # [1, D]
                            positive = z2[i:i+1]  # [1, D] 
                            neg = negatives[i].unsqueeze(0)  # [1, N-1, D]
                            loss_i = infonce_loss(anchor, positive, neg)
                            sample_losses.append(loss_i)
                        contrastive_loss = torch.stack(sample_losses).mean()
                
                # ====== 掩码属性预测部分 ======
                map_loss = 0.0
                valid_cats = [n for n in self.categorical_attrs if n in cat_inputs]
                if valid_cats:
                    target_name = random.choice(valid_cats)
                    
                    # 构造遮盖后的类别输入
                    masked_inputs = {}
                    for k, v in cat_inputs.items():
                        if k == target_name:
                            continue
                        masked_inputs[k] = v.to(Config.DEVICE_OBJ)
                    if num_inputs is not None:
                        num_inputs = num_inputs.to(Config.DEVICE_OBJ)
                    
                    # 前向：属性向量 + 融合
                    a_vec_masked = self.attribute_model(masked_inputs, num_inputs)
                    if l is not None:
                        l = l.to(Config.DEVICE_OBJ)
                    z_masked = self.model(b, a_vec_masked, l)
                    
                    # 预测被遮盖属性
                    logits = self.attr_heads[target_name](z_masked)
                    y = cat_labels[target_name].to(Config.DEVICE_OBJ)
                    map_loss = criterion(logits, y)
                
                # ====== 总损失 ======
                total_batch_loss = (Config.MAP_LOSS_WEIGHT * map_loss + 
                                  Config.CONTRASTIVE_LOSS_WEIGHT * contrastive_loss)
                
                # 反向传播
                self.opt.zero_grad()
                if total_batch_loss > 0:
                    total_batch_loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in self.opt.param_groups for p in group['params']], 
                        Config.FUSION_GRADIENT_CLIP_NORM
                    )
                    self.opt.step()
                
                # 记录损失
                total_map_loss += float(map_loss) if isinstance(map_loss, torch.Tensor) else 0.0
                total_contrastive_loss += float(contrastive_loss) if isinstance(contrastive_loss, torch.Tensor) else 0.0
                total_loss += float(total_batch_loss) if isinstance(total_batch_loss, torch.Tensor) else 0.0
                n_batches += 1
                
                # 获取当前学习率
                current_lr = self.opt.param_groups[0]['lr']
                pbar.set_postfix({
                    'total': f'{total_loss/max(n_batches,1):.4f}',
                    'map': f'{total_map_loss/max(n_batches,1):.4f}',
                    'contrast': f'{total_contrastive_loss/max(n_batches,1):.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # 计算epoch平均损失
            avg_loss = total_loss / max(n_batches, 1)
            
            # 学习率调度（预热后）
            if epoch >= Config.FUSION_WARMUP_EPOCHS:
                self.scheduler.step(avg_loss)
            
            # 早停检查
            is_best = False
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
                is_best = True
                print(f"  → 新的最佳损失: {avg_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"  → 损失未改善，耐心计数: {self.patience_counter}/{Config.FUSION_EARLY_STOPPING_PATIENCE}")
            
            # 保存检查点
            self._save_fusion_checkpoint(epoch, avg_loss, is_best)
            
            # 早停
            if self.patience_counter >= Config.FUSION_EARLY_STOPPING_PATIENCE:
                print(f"早停触发！最佳损失: {self.best_loss:.6f}")
                break

    def train_contrastive_only(self, behavior_embeddings, attribute_raw=None, location_embeddings=None, user_ids=None, resume: bool = False):
        """
        纯对比学习训练（当没有类别属性时使用）
        """
        dl = self._build_loader(behavior_embeddings, attribute_raw, location_embeddings, user_ids)
        self.model.train()
        if self.attribute_model is not None:
            self.attribute_model.train()
            
        start_epoch = 0
        if resume:
            last = self._load_fusion_checkpoint()
            if last >= 0:
                start_epoch = last + 1
                
        for epoch in range(start_epoch, Config.FUSION_EPOCHS):
            # 更新学习率（预热阶段）
            self._update_learning_rate(epoch)
            
            pbar = tqdm(dl, desc=f"Contrastive Only Epoch {epoch+1}/{Config.FUSION_EPOCHS}")
            total_loss = 0.0
            n_batches = 0
            
            for uids, b, attr_pack, l in pbar:
                b = b.to(Config.DEVICE_OBJ)
                
                if b.size(0) <= 1:  # 需要至少2个样本才能做对比学习
                    continue
                    
                # 1. 对行为向量进行两次不同的dropout增强
                b_view1, b_view2 = behavior_dropout_augmentation(b)
                
                # 2. 处理属性向量
                a_vec = None
                if self.attribute_model is not None and attr_pack is not None:
                    cat_inputs, num_inputs, _ = attr_pack
                    if cat_inputs is not None:
                        for k in cat_inputs.keys():
                            cat_inputs[k] = cat_inputs[k].to(Config.DEVICE_OBJ)
                    if num_inputs is not None:
                        num_inputs = num_inputs.to(Config.DEVICE_OBJ)
                    a_vec = self.attribute_model(cat_inputs or {}, num_inputs)
                
                # 3. 处理位置向量
                l_vec = None
                if l is not None:
                    l_vec = l.to(Config.DEVICE_OBJ)
                
                # 4. 分别融合两个视图
                z1 = self.model(b_view1, a_vec, l_vec)
                z2 = self.model(b_view2, a_vec, l_vec)
                
                # 5. 构造正负样本并计算InfoNCE损失
                batch_size = z1.size(0)
                negatives = []
                for i in range(batch_size):
                    neg_indices = [j for j in range(batch_size) if j != i]
                    neg_samples = z1[neg_indices]
                    negatives.append(neg_samples)
                
                # 计算InfoNCE损失
                sample_losses = []
                for i in range(batch_size):
                    anchor = z1[i:i+1]
                    positive = z2[i:i+1]
                    neg = negatives[i].unsqueeze(0)
                    loss_i = infonce_loss(anchor, positive, neg)
                    sample_losses.append(loss_i)
                loss = torch.stack(sample_losses).mean()
                
                # 反向传播
                self.opt.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.opt.param_groups for p in group['params']], 
                    Config.FUSION_GRADIENT_CLIP_NORM
                )
                self.opt.step()
                
                # 记录损失
                total_loss += float(loss.detach().cpu())
                n_batches += 1
                
                # 获取当前学习率
                current_lr = self.opt.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{total_loss/max(n_batches,1):.4f}',
                    'lr': f'{current_lr:.2e}'
                })
            
            # 计算epoch平均损失
            avg_loss = total_loss / max(n_batches, 1)
            
            # 学习率调度（预热后）
            if epoch >= Config.FUSION_WARMUP_EPOCHS:
                self.scheduler.step(avg_loss)
            
            # 早停检查
            is_best = False
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.patience_counter = 0
                is_best = True
                print(f"  → 新的最佳损失: {avg_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"  → 损失未改善，耐心计数: {self.patience_counter}/{Config.FUSION_EARLY_STOPPING_PATIENCE}")
            
            # 保存检查点
            self._save_fusion_checkpoint(epoch, avg_loss, is_best)
            
            # 早停
            if self.patience_counter >= Config.FUSION_EARLY_STOPPING_PATIENCE:
                print(f"早停触发！最佳损失: {self.best_loss:.6f}")
                break

    def export_embeddings(self, behavior_embeddings, attribute_raw=None, location_embeddings=None):
        self.model.eval()
        if self.attribute_model is not None:
            self.attribute_model.eval()
        ds = FusionDataset(behavior_embeddings, attribute_raw, location_embeddings)
        dl = DataLoader(ds, batch_size=Config.FUSION_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False, collate_fn=self._collate)
        fused = {}
        with torch.no_grad():
            for uids, b, attr_pack, l in dl:
                b = b.to(Config.DEVICE_OBJ)
                a_vec = None
                if self.attribute_model is not None and attr_pack is not None:
                    # 兼容 collate 返回的三元组 (cat_inputs, num_inputs, cat_labels)
                    cat_inputs, num_inputs, _ = attr_pack
                    if cat_inputs is not None:
                        for k in cat_inputs.keys():
                            cat_inputs[k] = cat_inputs[k].to(Config.DEVICE_OBJ)
                    if num_inputs is not None:
                        num_inputs = num_inputs.to(Config.DEVICE_OBJ)
                    a_vec = self.attribute_model(cat_inputs or {}, num_inputs)
                if l is not None:
                    l = l.to(Config.DEVICE_OBJ)
                z = self.model(b, a_vec, l)
                z = z.detach().cpu().numpy()
                for uid, vec in zip(uids, z):
                    fused[uid] = vec
        return fused




