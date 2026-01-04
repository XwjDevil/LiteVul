import pickle
import random
import os

def split_dataset(input_pkl_list, out_dir, seed=42, ratio=(0.8, 0.1, 0.1)):
    """
    input_pkl_list: List[str], 所有pkl文件路径（可传两个，正负样本各一个）
    out_dir: str, 输出文件夹
    seed: int, 随机种子
    ratio: Tuple[float], 划分比例
    """
    assert abs(sum(ratio) - 1.0) < 1e-6
    # 读取所有数据
    all_data = []
    for pkl in input_pkl_list:
        with open(pkl, "rb") as f:
            all_data += pickle.load(f)
    print(f"总样本数: {len(all_data)}")
    # 随机打乱
    random.seed(seed)
    random.shuffle(all_data)
    N = len(all_data)
    n_train = int(N * ratio[0])
    n_val = int(N * ratio[1])
    n_test = N - n_train - n_val
    train_set = all_data[:n_train]
    val_set = all_data[n_train:n_train+n_val]
    test_set = all_data[n_train+n_val:]
    print(f"Train: {len(train_set)}; Val: {len(val_set)}; Test: {len(test_set)}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_set, f)
    with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_set, f)
    with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_set, f)
    print(f"保存到: {out_dir}")

if __name__ == "__main__":
    # 修改为你实际的输出文件名
    input_pkl_list = [
        "/mnt/data/NewReGVD/Devign/Devign_PPMI/data_label_0_part000.pkl",
        "/mnt/data/NewReGVD/Devign/Devign_PPMI/data_label_1_part000.pkl"
    ]
    out_dir = "/mnt/data/NewReGVD/Devign/Devign_PPMI/pkl"
    split_dataset(input_pkl_list, out_dir, seed=42)













# #适合diversevul
# import os
# import re
# import glob
# import pickle
# import random

# def _label_of_sample(sample, fallback_label=None):
#     """从 PyG Data.y 里拿标签；拿不到就用备用标签。"""
#     try:
#         y = sample.y
#         if hasattr(y, "item"):
#             return int(y.item())
#         return int(y)
#     except Exception:
#         return fallback_label

# def split_dataset_from_dir(in_dir, out_dir, seed=42, ratio=(0.8, 0.1, 0.1),
#                            pattern="data_label_*_part*.pkl"):
#     """
#     in_dir:  保存了 data_label_0_partXXX.pkl / data_label_1_partXXX.pkl 的目录
#     out_dir: 输出 train/val/test 的目录
#     ratio:   (train, val, test)
#     """
#     assert abs(sum(ratio) - 1.0) < 1e-6

#     # 1) 收集所有分片 pkl
#     pkl_files = sorted(glob.glob(os.path.join(in_dir, pattern)))
#     if not pkl_files:
#         # 兼容没有分片名的情况，如 data_label_0.pkl / data_label_1.pkl
#         pkl_files = sorted(glob.glob(os.path.join(in_dir, "data_label_*.pkl")))
#     if not pkl_files:
#         raise FileNotFoundError(f"No pkl files found under {in_dir}")

#     # 2) 读取并按标签聚合（以样本自身的 y 为准）
#     by_label = {0: [], 1: []}
#     total_files = 0
#     for p in pkl_files:
#         # 从文件名里提取一个回退标签（若样本内部无 y 时使用）
#         m = re.search(r"label_(\d+)", os.path.basename(p))
#         fallback_label = int(m.group(1)) if m else None

#         with open(p, "rb") as f:
#             chunk = pickle.load(f)
#         total_files += 1

#         for s in chunk:
#             lab = _label_of_sample(s, fallback_label=fallback_label)
#             if lab not in (0, 1):
#                 # 非法标签样本直接跳过
#                 continue
#             by_label[lab].append(s)

#     n0, n1 = len(by_label[0]), len(by_label[1])
#     print(f"读取完成：neg(0)={n0}, pos(1)={n1}, files={total_files}")

#     # 3) 分层随机划分（每个类别各自按比例切分）
#     rng = random.Random(seed)
#     rng.shuffle(by_label[0])
#     rng.shuffle(by_label[1])

#     def split_one(cls_list):
#         N = len(cls_list)
#         n_train = int(N * ratio[0])
#         n_val   = int(N * ratio[1])
#         n_test  = N - n_train - n_val
#         train = cls_list[:n_train]
#         val   = cls_list[n_train:n_train+n_val]
#         test  = cls_list[n_train+n_val:]
#         return train, val, test

#     t0, v0, s0 = split_one(by_label[0])
#     t1, v1, s1 = split_one(by_label[1])

#     train_set = t0 + t1
#     val_set   = v0 + v1
#     test_set  = s0 + s1

#     # 合并后再各自打乱一次，避免同类聚集
#     rng.shuffle(train_set)
#     rng.shuffle(val_set)
#     rng.shuffle(test_set)

#     print(f"Train: {len(train_set)}  (neg={len(t0)}, pos={len(t1)})")
#     print(f"Val:   {len(val_set)}    (neg={len(v0)}, pos={len(v1)})")
#     print(f"Test:  {len(test_set)}   (neg={len(s0)}, pos={len(s1)})")

#     # 4) 落盘
#     os.makedirs(out_dir, exist_ok=True)
#     with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
#         pickle.dump(train_set, f)
#     with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
#         pickle.dump(val_set, f)
#     with open(os.path.join(out_dir, "test.pkl"), "wb") as f:
#         pickle.dump(test_set, f)
#     print(f"已保存到: {out_dir}")

# if __name__ == "__main__":
#     in_dir  = "/mnt/data/llm/diversevul_graph_data"
#     out_dir = "/mnt/data/llm/diversevul_graph_data/pkl"
#     split_dataset_from_dir(in_dir, out_dir, seed=42, ratio=(0.8, 0.1, 0.1))


