#通用版本
import os
import glob
import pickle
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from transformers import RobertaTokenizer, RobertaModel

# ===== 1) CodeBERT 静态嵌入查表（与你原来一致）=====
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
_encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
emb_layer = _encoder.get_input_embeddings().cpu()
del _encoder

def codebert_token_embed(tokens):
    if len(tokens) == 0:
        return np.zeros((0, 768), dtype=np.float32)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        emb = emb_layer(input_ids).squeeze(0).detach().cpu().numpy()
    return emb.astype(np.float32, copy=False)

# ===== 2) 多尺度 + 有向共现 + PPMI + bigram 提升 =====
def multiscale_ppmi_edges(tokens,
                          window_sizes=(3, 5, 9),
                          strides=(1, 1, 3),
                          scales=(1.0, 0.6, 0.3),
                          bigram_boost=2.0,
                          eps=1e-8):
    """
    基于原始 BPE token 序列，构建有向共现边并做 PPMI 变换。
    返回：F
      pair_weight: dict[(u,v)] -> float  (PPMI 权重，>=0)
      unique_tokens: List[str]
    """
    assert len(window_sizes) == len(strides) == len(scales)
    L = len(tokens)

    # 1) 去重 token -> 节点
    unique_tokens = []
    token2idx = {}
    for tok in tokens:
        if tok not in token2idx:
            token2idx[tok] = len(unique_tokens)
            unique_tokens.append(tok)

    # 2) 收集多尺度有向共现计数（带距离衰减）
    #    以及 source 出度 / target 入度 统计（用于 PPMI）
    pair_count = defaultdict(float)     # 有向 (u,v)
    src_out = defaultdict(float)        # u 的总“发出”权
    dst_in = defaultdict(float)         # v 的总“接收”权
    min_delta = {}                      # 记录最小距离（可选二次加权）

    for ws, st, sc in zip(window_sizes, strides, scales):
        for start in range(0, L, st):
            end = min(start + ws, L)
            if end - start < 2:  # 少于两个 token 不成边
                continue
            # 用位置对（保持顺序），统计有向 pair
            # 复杂度 O(ws^2)，ws 很小（<=9）几乎不耗时
            for i in range(start, end - 1):
                ui = token2idx[tokens[i]]
                # bigram 提升（紧邻）
                vj = token2idx[tokens[i + 1]]
                pair_count[(ui, vj)] += sc * bigram_boost
                src_out[ui] += sc * bigram_boost
                dst_in[vj] += sc * bigram_boost
                key = (ui, vj)
                d = 1
                if key not in min_delta or d < min_delta[key]:
                    min_delta[key] = d

                # 其它更远的有向 pair（i -> j, j>i+1），带距离衰减
                for j in range(i + 2, end):
                    vj = token2idx[tokens[j]]
                    d = j - i
                    w = sc * (1.0 / (1.0 + d))  # 距离越远，贡献越小
                    pair_count[(ui, vj)] += w
                    src_out[ui] += w
                    dst_in[vj] += w
                    key = (ui, vj)
                    if key not in min_delta or d < min_delta[key]:
                        min_delta[key] = d

    if not pair_count:
        return {}, unique_tokens

    # 3) PPMI（有向版）
    total = sum(pair_count.values()) + eps
    pair_weight = {}
    for (u, v), c_uv in pair_count.items():
        p_uv = c_uv / total
        p_u  = (src_out[u] / total) if src_out[u] > 0 else eps
        p_v  = (dst_in[v] / total) if dst_in[v] > 0 else eps
        pmi  = np.log(max(p_uv / (p_u * p_v + eps), eps))
        ppmi = max(0.0, pmi)  # 正值保留
        # 轻微距离再加权：更近的 pair 略微加分（可不加）
        d = min_delta.get((u, v), 1)
        ppmi *= (1.0 + 0.1 / (1.0 + d))
        if ppmi > 0:
            pair_weight[(u, v)] = float(ppmi)

    return pair_weight, unique_tokens

# ===== 3) 节点特征 =====
def build_node_features(tokens):
    if len(tokens) == 0:
        return np.zeros((0, 768), dtype=np.float32), {}
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        node_features = emb_layer(input_ids).squeeze(0).detach().cpu().numpy()
    token2idx = {tok: idx for idx, tok in enumerate(tokens)}
    return node_features, token2idx

# ===== 4) 单文件处理（替换原 process_code_file）=====
def process_code_file_v2(file_path,
                         window_sizes=(3, 5, 9),
                         strides=(1, 1, 3),
                         scales=(1.0, 0.6, 0.3),
                         bigram_boost=2.0):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    tokens = tokenizer.tokenize(code)
    if not tokens:
        return None

    # 有向多尺度 + PPMI
    pair_weight, unique_tokens = multiscale_ppmi_edges(
        tokens,
        window_sizes=window_sizes,
        strides=strides,
        scales=scales,
        bigram_boost=bigram_boost
    )
    if not pair_weight:
        return None

    # 节点特征（静态查表）
    node_features, token2idx = build_node_features(unique_tokens)

    # 组装 PyG Data
    edges = []
    weights = []
    for (u, v), w in pair_weight.items():
        edges.append([u, v])        # 有向
        weights.append(w)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(weights, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                tokens=list(token2idx.keys()), file=file_path)

# ===== 5) 批量处理入口（用新构图）=====
def batch_process_code_dir_v2(dir_path, label, save_dir,
                              window_sizes=(3, 5, 9),
                              strides=(1, 1, 3),
                              scales=(1.0, 0.6, 0.3),
                              bigram_boost=2.0):
    os.makedirs(save_dir, exist_ok=True)
    file_list = glob.glob(os.path.join(dir_path, "*.c"))
    datas = []
    for fp in tqdm(file_list, desc=f"Processing {dir_path}"):
        data = process_code_file_v2(
            fp,
            window_sizes=window_sizes,
            strides=strides,
            scales=scales,
            bigram_boost=bigram_boost
        )
        if data is not None:
            data.y = torch.tensor([label], dtype=torch.long)
            datas.append(data)
    save_path = os.path.join(save_dir, f'data_label_{label}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(datas, f)
    print(f"Saved {len(datas)} samples to {save_path}")

if __name__ == "__main__":
    save_dir = "/mnt/data/NewReGVD/Devign/Devign_PPMI/"
    # 你可以保留原来的目录结构
    batch_process_code_dir_v2("/mnt/data/Devign/Vul",    1, save_dir)
    batch_process_code_dir_v2("/mnt/data/Devign/No-Vul", 0, save_dir)







# #DiverseVul版本，数据集过大故需要分片处理
# import os
# import glob
# import pickle
# from collections import defaultdict

# import torch
# import numpy as np
# from tqdm import tqdm
# from torch_geometric.data import Data
# from transformers import RobertaTokenizer, RobertaModel

# # ===== 1) CodeBERT 静态嵌入查表（与你原来一致）=====
# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# _encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
# emb_layer = _encoder.get_input_embeddings().cpu()
# del _encoder

# def codebert_token_embed(tokens):
#     if len(tokens) == 0:
#         return np.zeros((0, 768), dtype=np.float32)
#     ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_ids = torch.tensor([ids], dtype=torch.long)
#     with torch.no_grad():
#         emb = emb_layer(input_ids).squeeze(0).detach().cpu().numpy()
#     return emb.astype(np.float32, copy=False)

# # ===== 2) 多尺度 + 有向共现 + PPMI + bigram 提升 =====
# def multiscale_ppmi_edges(tokens,
#                           window_sizes=(3, 5, 9),
#                           strides=(1, 1, 3),
#                           scales=(1.0, 0.6, 0.3),
#                           bigram_boost=2.0,
#                           eps=1e-8):
#     """
#     基于原始 BPE token 序列，构建有向共现边并做 PPMI 变换。
#     返回：F
#       pair_weight: dict[(u,v)] -> float  (PPMI 权重，>=0)
#       unique_tokens: List[str]
#     """
#     assert len(window_sizes) == len(strides) == len(scales)
#     L = len(tokens)

#     # 1) 去重 token -> 节点
#     unique_tokens = []
#     token2idx = {}
#     for tok in tokens:
#         if tok not in token2idx:
#             token2idx[tok] = len(unique_tokens)
#             unique_tokens.append(tok)

#     # 2) 收集多尺度有向共现计数（带距离衰减）
#     #    以及 source 出度 / target 入度 统计（用于 PPMI）
#     pair_count = defaultdict(float)     # 有向 (u,v)
#     src_out = defaultdict(float)        # u 的总“发出”权
#     dst_in = defaultdict(float)         # v 的总“接收”权
#     min_delta = {}                      # 记录最小距离（可选二次加权）

#     for ws, st, sc in zip(window_sizes, strides, scales):
#         for start in range(0, L, st):
#             end = min(start + ws, L)
#             if end - start < 2:  # 少于两个 token 不成边
#                 continue
#             # 用位置对（保持顺序），统计有向 pair
#             # 复杂度 O(ws^2)，ws 很小（<=9）几乎不耗时
#             for i in range(start, end - 1):
#                 ui = token2idx[tokens[i]]
#                 # bigram 提升（紧邻）
#                 vj = token2idx[tokens[i + 1]]
#                 pair_count[(ui, vj)] += sc * bigram_boost
#                 src_out[ui] += sc * bigram_boost
#                 dst_in[vj] += sc * bigram_boost
#                 key = (ui, vj)
#                 d = 1
#                 if key not in min_delta or d < min_delta[key]:
#                     min_delta[key] = d

#                 # 其它更远的有向 pair（i -> j, j>i+1），带距离衰减
#                 for j in range(i + 2, end):
#                     vj = token2idx[tokens[j]]
#                     d = j - i
#                     w = sc * (1.0 / (1.0 + d))  # 距离越远，贡献越小
#                     pair_count[(ui, vj)] += w
#                     src_out[ui] += w
#                     dst_in[vj] += w
#                     key = (ui, vj)
#                     if key not in min_delta or d < min_delta[key]:
#                         min_delta[key] = d

#     if not pair_count:
#         return {}, unique_tokens

#     # 3) PPMI（有向版）
#     total = sum(pair_count.values()) + eps
#     pair_weight = {}
#     for (u, v), c_uv in pair_count.items():
#         p_uv = c_uv / total
#         p_u  = (src_out[u] / total) if src_out[u] > 0 else eps
#         p_v  = (dst_in[v] / total) if dst_in[v] > 0 else eps
#         pmi  = np.log(max(p_uv / (p_u * p_v + eps), eps))
#         ppmi = max(0.0, pmi)  # 正值保留
#         # 轻微距离再加权：更近的 pair 略微加分（可不加）
#         d = min_delta.get((u, v), 1)
#         ppmi *= (1.0 + 0.1 / (1.0 + d))
#         if ppmi > 0:
#             pair_weight[(u, v)] = float(ppmi)

#     return pair_weight, unique_tokens

# # ===== 3) 节点特征 =====
# def build_node_features(tokens):
#     if len(tokens) == 0:
#         return np.zeros((0, 768), dtype=np.float32), {}
#     input_ids = tokenizer.convert_tokens_to_ids(tokens)
#     input_ids = torch.tensor([input_ids], dtype=torch.long)
#     with torch.no_grad():
#         node_features = emb_layer(input_ids).squeeze(0).detach().cpu().numpy()
#     token2idx = {tok: idx for idx, tok in enumerate(tokens)}
#     return node_features, token2idx

# # ===== 4) 单文件处理（替换原 process_code_file）=====
# def process_code_file_v2(file_path,
#                          window_sizes=(3, 5, 9),
#                          strides=(1, 1, 3),
#                          scales=(1.0, 0.6, 0.3),
#                          bigram_boost=2.0):
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
#         code = f.read()
#     tokens = tokenizer.tokenize(code)
#     if not tokens:
#         return None

#     # 有向多尺度 + PPMI
#     pair_weight, unique_tokens = multiscale_ppmi_edges(
#         tokens,
#         window_sizes=window_sizes,
#         strides=strides,
#         scales=scales,
#         bigram_boost=bigram_boost
#     )
#     if not pair_weight:
#         return None

#     # 节点特征（静态查表）
#     node_features, token2idx = build_node_features(unique_tokens)

#     # 组装 PyG Data
#     edges = []
#     weights = []
#     for (u, v), w in pair_weight.items():
#         edges.append([u, v])        # 有向
#         weights.append(w)
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     edge_attr = torch.tensor(weights, dtype=torch.float)
#     x = torch.tensor(node_features, dtype=torch.float)

#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
#                 tokens=list(token2idx.keys()), file=file_path)

# def iter_c_files(dir_path):
#     """递归遍历 dir_path 下所有 .c 文件"""
#     for root, _, files in os.walk(dir_path):
#         for fn in files:
#             if fn.endswith(".c"):
#                 yield os.path.join(root, fn)

# def batch_process_code_dir_v2(dir_path, label, save_dir,
#                               window_sizes=(3, 5, 9),
#                               strides=(1, 1, 3),
#                               scales=(1.0, 0.6, 0.3),
#                               bigram_boost=2.0,
#                               shard_size=787000):   # 每 N 条写一个分片，避免一次性占内存
#     os.makedirs(save_dir, exist_ok=True)

#     # 递归收集并排序，保证可复现
#     file_list = sorted(iter_c_files(dir_path))
#     pbar = tqdm(total=len(file_list), desc=f"Processing {dir_path}", ncols=100)

#     datas, shard_idx, kept = [], 0, 0
#     def flush_shard():
#         nonlocal datas, shard_idx
#         if not datas:
#             return
#         save_path = os.path.join(save_dir, f"data_label_{label}_part{shard_idx:03d}.pkl")
#         with open(save_path, "wb") as f:
#             pickle.dump(datas, f)
#         print(f"[SAVE] {len(datas)} samples -> {save_path}")
#         datas = []
#         shard_idx += 1

#     for fp in file_list:
#         data = process_code_file_v2(
#             fp,
#             window_sizes=window_sizes,
#             strides=strides,
#             scales=scales,
#             bigram_boost=bigram_boost
#         )
#         if data is not None:
#             data.y = torch.tensor([label], dtype=torch.long)
#             datas.append(data)
#             kept += 1
#             if len(datas) >= shard_size:
#                 flush_shard()
#         pbar.update(1)

#     pbar.close()
#     flush_shard()
#     print(f"Done. kept={kept}, shards={shard_idx}")

# if __name__ == "__main__":
#     save_dir = "/mnt/data/NewReGVD/diversevul/diversevul_PPMI" 
#     # 对应目录结构
#     batch_process_code_dir_v2("/mnt/data/NewReGVD/diversevul/Vul",    1, save_dir)
#     batch_process_code_dir_v2("/mnt/data/NewReGVD/diversevul/No-Vul", 0, save_dir)





































































