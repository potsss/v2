"""
v2 Node2Vec 游走与图构建（简化实现）
"""
import random
from collections import defaultdict


def build_item_graph(user_sequences, directed=False):
    graph = defaultdict(lambda: defaultdict(int))
    for _, seq in (user_sequences.items() if isinstance(user_sequences, dict) else enumerate(user_sequences)):
        if not seq or len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            if a == b:
                continue
            graph[a][b] += 1
            if not directed:
                graph[b][a] += 1
    return graph


def _weighted_choice(neighbors_dict, prev=None, p=1.0, q=1.0, prev_neighbors=None):
    if not neighbors_dict:
        return None
    neighbors = list(neighbors_dict.keys())
    weights = []
    for n in neighbors:
        w = neighbors_dict[n]
        if prev is None:
            weights.append(w)
        else:
            if n == prev:
                weights.append(w / max(p, 1e-8))
            elif prev_neighbors and n in prev_neighbors:
                weights.append(w)
            else:
                weights.append(w / max(q, 1e-8))
    total = float(sum(weights))
    if total <= 0:
        return None
    r = random.random() * total
    cum = 0.0
    for n, w in zip(neighbors, weights):
        cum += w
        if r <= cum:
            return n
    return neighbors[-1]


def generate_walks(graph, num_walks, walk_length, p=1.0, q=1.0):
    nodes = list(graph.keys())
    walks = []
    if not nodes:
        return walks
    neighbor_sets = {n: set(graph[n].keys()) for n in nodes}
    for _ in range(num_walks):
        random.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                if len(walk) == 1:
                    nxt = _weighted_choice(graph[cur])
                else:
                    prev = walk[-2]
                    nxt = _weighted_choice(graph[cur], prev=prev, p=p, q=q, prev_neighbors=neighbor_sets.get(prev))
                if nxt is None:
                    break
                walk.append(nxt)
            walks.append(walk)
    return walks

"""
v2 Node2Vec 随机游走（简化版，复用动态转移思想）
"""
import random
from collections import defaultdict


def build_graph_from_sequences(user_sequences, directed=False):
    graph = defaultdict(lambda: defaultdict(int))
    for _, seq in user_sequences.items() if isinstance(user_sequences, dict) else enumerate(user_sequences):
        if not seq or len(seq) < 2:
            continue
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a == b:
                continue
            graph[a][b] += 1
            if not directed:
                graph[b][a] += 1
    return graph


def generate_node2vec_walks_dynamic(graph, num_walks, walk_length, p, q, start_nodes=None):
    nodes = list(graph.keys()) if start_nodes is None else start_nodes
    if not nodes:
        return []
    neighbors = {n: list(graph[n].keys()) for n in graph.keys()}
    walks = []
    for _ in range(num_walks):
        random.shuffle(nodes)
        for s in nodes:
            walk = [s]
            while len(walk) < walk_length:
                v = walk[-1]
                nbrs = neighbors.get(v, [])
                if not nbrs:
                    break
                if len(walk) == 1:
                    # 均匀选择第一步（按权重也可）
                    nxt = random.choice(nbrs)
                else:
                    t = walk[-2]
                    probs = []
                    total = 0.0
                    for x in nbrs:
                        w = graph[v][x]
                        if x == t:
                            w = w / p
                        elif t in graph[x]:
                            w = w
                        else:
                            w = w / q
                        probs.append(w)
                        total += w
                    if total == 0:
                        break
                    r = random.random() * total
                    acc = 0.0
                    nxt = nbrs[-1]
                    for x, w in zip(nbrs, probs):
                        acc += w
                        if r <= acc:
                            nxt = x
                            break
                walk.append(nxt)
            walks.append(walk)
    return walks


