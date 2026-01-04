# coding=utf-8
import os
import re
import shutil
import argparse
from clean_gadget import clean_gadget

def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    args = parser.parse_args()
    return args

# def normalize(path):
#     setfolderlist = os.listdir(path)
#     for setfolder in setfolderlist:
#         catefolderlist = os.listdir(path + "//" + setfolder)
#         #print(catefolderlist)
#         for catefolder in catefolderlist:
#             filepath = path + "//" + setfolder + "//" + catefolder
#             print(catefolder)
#             pro_one_file(filepath)

from tqdm import tqdm

def normalize(path):
    # 先收集所有.c文件路径
    c_file_list = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.c'):
                c_file_list.append(os.path.join(root, filename))
    # 进度条遍历
    for filepath in tqdm(c_file_list, desc="Normalizing", ncols=100):
        try:
            pro_one_file(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")



def strip_c_comments(code: str) -> str:
    """
    删除 C/C++ 风格注释，保留字符串/字符常量中的 // 和 /* */。
    不使用正则，线性复杂度，避免卡死。
    """
    n = len(code)
    i = 0
    out = []
    STATE_CODE = 0
    STATE_LINE_COMMENT = 1
    STATE_BLOCK_COMMENT = 2
    STATE_STRING = 3
    STATE_CHAR = 4
    state = STATE_CODE

    while i < n:
        ch = code[i]

        if state == STATE_CODE:
            if ch == '/':
                # 可能进入注释
                if i+1 < n and code[i+1] == '/':
                    state = STATE_LINE_COMMENT
                    i += 2
                    continue
                elif i+1 < n and code[i+1] == '*':
                    state = STATE_BLOCK_COMMENT
                    i += 2
                    continue
                else:
                    out.append(ch)
                    i += 1
            elif ch == '"':
                out.append(ch)
                state = STATE_STRING
                i += 1
            elif ch == "'":
                out.append(ch)
                state = STATE_CHAR
                i += 1
            else:
                out.append(ch)
                i += 1

        elif state == STATE_LINE_COMMENT:
            # 跳到行末，换行符保留（行号不乱）
            if ch == '\n':
                out.append(ch)
                state = STATE_CODE
            i += 1

        elif state == STATE_BLOCK_COMMENT:
            # 跳到下一个 */ 或文件结束
            if ch == '*' and i+1 < n and code[i+1] == '/':
                state = STATE_CODE
                i += 2
            else:
                i += 1

        elif state == STATE_STRING:
            # 处理转义字符
            if ch == '\\':
                if i+1 < n:
                    out.append(ch)
                    out.append(code[i+1])
                    i += 2
                else:
                    out.append(ch)
                    i += 1
            elif ch == '"':
                out.append(ch)
                state = STATE_CODE
                i += 1
            else:
                out.append(ch)
                i += 1

        elif state == STATE_CHAR:
            if ch == '\\':
                if i+1 < n:
                    out.append(ch)
                    out.append(code[i+1])
                    i += 2
                else:
                    out.append(ch)
                    i += 1
            elif ch == "'":
                out.append(ch)
                state = STATE_CODE
                i += 1
            else:
                out.append(ch)
                i += 1

    return ''.join(out)


def pro_one_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception as e:
        print(f"[READ FAIL] {filepath}: {e}")
        return

    # 线性算法去注释，避免正则卡死
    no_comment = strip_c_comments(code).strip()

    # 传给 clean_gadget 前拆成行，保留换行符
    lines = no_comment.splitlines(keepends=True)

    # 防止 clean_gadget 自身异常导致中断
    try:
        nor_lines = clean_gadget(lines)
        if not isinstance(nor_lines, (list, tuple)):
            # 兜底：若返回不是行序列，就按原行写回
            nor_lines = lines
    except Exception as e:
        print(f"[CLEAN FAIL] {filepath}: {e}")
        nor_lines = lines

    try:
        # 统一用 \n 换行，减少平台差异
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(nor_lines)
    except Exception as e:
        print(f"[WRITE FAIL] {filepath}: {e}")

def main():
    args = parse_options()
    normalize(args.input)
    

if __name__ == '__main__':
    main()
 