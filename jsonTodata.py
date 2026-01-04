# import os
# import json

# # 配置输入json文件路径
# json_path = '/mnt/data/lzh/xwj/DataSet/diversevul_20230702.json'  # 请替换为你的json文件路径

# # 配置输出文件夹
# vul_dir = '/mnt/data/lzh/xwj/diversevul/Vul'
# novul_dir = '/mnt/data/lzh/xwj/diversevul/No-Vul'
# os.makedirs(vul_dir, exist_ok=True)
# os.makedirs(novul_dir, exist_ok=True)

# # 读取json
# with open(json_path, 'r', encoding='utf-8') as f:
#     try:
#         data = json.load(f)
#         if isinstance(data, dict):  # 兼容单层dict或list
#             data = [data]
#     except json.JSONDecodeError:
#         # 逐行读取（常见于每行为一个json dict的情况）
#         f.seek(0)
#         data = [json.loads(line) for line in f if line.strip()]

# vul_idx, novul_idx = 1, 1

# for item in data:
#     func = item.get('func')
#     target = item.get('target')
#     if func is None or target is None:
#         continue  # 跳过不完整数据

#     if target == 1:
#         filename = f'vul_{vul_idx:05d}.c'
#         filepath = os.path.join(vul_dir, filename)
#         vul_idx += 1
#     elif target == 0:
#         filename = f'nonvul_{novul_idx:05d}.c'
#         filepath = os.path.join(novul_dir, filename)
#         novul_idx += 1
#     else:
#         continue  # 跳过异常target

#     with open(filepath, 'w', encoding='utf-8') as out_f:
#         out_f.write(func)

# print(f'Done! Vul: {vul_idx-1}, No-Vul: {novul_idx-1}')





















# import os
# import json
# from tqdm import tqdm

# # 配置输入json文件路径
# json_path = '/mnt/data/lzh/xwj/DataSet/diversevul_20230702.json'  # 请替换为你的json文件路径

# # 配置输出文件夹
# vul_dir = '/mnt/data/lzh/xwj/diversevul/Vul'
# novul_dir = '/mnt/data/lzh/xwj/diversevul/No-Vul'
# os.makedirs(vul_dir, exist_ok=True)
# os.makedirs(novul_dir, exist_ok=True)

# # 读取json
# with open(json_path, 'r', encoding='utf-8') as f:
#     try:
#         data = json.load(f)
#         if isinstance(data, dict):  # 兼容单层dict或list
#             data = [data]
#     except json.JSONDecodeError:
#         # 逐行读取（常见于每行为一个json dict的情况）
#         f.seek(0)
#         data = [json.loads(line) for line in f if line.strip()]

# vul_idx, novul_idx = 1, 1

# # tqdm可视化
# for item in tqdm(data, desc="Processing"):
#     func = item.get('func')
#     target = item.get('target')
#     if func is None or target is None:
#         continue  # 跳过不完整数据

#     if target == 1:
#         filename = f'vul_{vul_idx:05d}.c'
#         filepath = os.path.join(vul_dir, filename)
#         vul_idx += 1
#     elif target == 0:
#         filename = f'nonvul_{novul_idx:05d}.c'
#         filepath = os.path.join(novul_dir, filename)
#         novul_idx += 1
#     else:
#         continue  # 跳过异常target

#     with open(filepath, 'w', encoding='utf-8') as out_f:
#         out_f.write(func)

# print(f'Done! Vul: {vul_idx-1}, No-Vul: {novul_idx-1}')

















import os 
import json
from tqdm import tqdm

json_path = '/mnt/data/lzh/xwj/DataSet/train_data_no_ffmpeg.json' 
vul_dir = '/mnt/data/lzh/xwj/NewReGVD/clean_diversevul/Vul'
novul_dir = '/mnt/data/lzh/xwj/NewReGVD/clean_diversevul/No-Vul'
os.makedirs(vul_dir, exist_ok=True)
os.makedirs(novul_dir, exist_ok=True)

with open(json_path, 'r', encoding='utf-8') as f:
    try:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        f.seek(0)
        data = [json.loads(line) for line in f if line.strip()]

vul_idx, novul_idx = 1, 1

def make_subfolder(base_dir, idx):
    bucket = (idx - 1) // 10000  # 每1w个文件一个文件夹
    bucket_dir = os.path.join(base_dir, f'{bucket:05d}')
    os.makedirs(bucket_dir, exist_ok=True)
    return bucket_dir

for item in tqdm(data, desc="Processing"):
    func = item.get('func')
    target = item.get('target')
    if func is None or target is None:
        continue

    if target == 1:
        bucket_dir = make_subfolder(vul_dir, vul_idx)
        filename = f'vul_{vul_idx:05d}.c'
        filepath = os.path.join(bucket_dir, filename)
        vul_idx += 1
    elif target == 0:
        bucket_dir = make_subfolder(novul_dir, novul_idx)
        filename = f'nonvul_{novul_idx:05d}.c'
        filepath = os.path.join(bucket_dir, filename)
        novul_idx += 1
    else:
        continue

    with open(filepath, 'w', encoding='utf-8') as out_f:
        out_f.write(func)

print(f'Done! Vul: {vul_idx-1}, No-Vul: {novul_idx-1}')
