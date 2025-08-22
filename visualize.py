"""
v2 可视化（t-SNE）
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from .config import Config


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def tsne_scatter(embeddings, labels, title, sample_size=None, perplexity=30, n_iter=1000):
    if sample_size and len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        emb = embeddings[idx]
        lbl = [labels[i] for i in idx]
    else:
        emb = embeddings
        lbl = labels
    # 小样本保护
    if len(emb) < 3:
        print("样本过少，跳过 t-SNE 可视化（需要至少 3 个样本）")
        return None
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(emb)-1), n_iter=n_iter, random_state=Config.RANDOM_SEED, verbose=1)
    reduced = tsne.fit_transform(emb)
    os.makedirs(Config.VISUALIZATION_DIR, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], s=40, alpha=0.7, c=np.random.rand(len(reduced)))
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(sc)
    path = os.path.join(Config.VISUALIZATION_DIR, title.replace(' ', '_') + '.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path




