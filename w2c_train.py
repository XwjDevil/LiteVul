#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2Vec训练代码 - 用于C语言源代码的自定义分词和嵌入训练
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec, FastText
from collections import Counter
import pickle

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeTokenizer:
    """C语言代码分词器"""
    
    def __init__(self, split_camel_case=True, keep_operators=True):
        self.split_camel_case = split_camel_case
        self.keep_operators = keep_operators
        
        # C语言关键字
        self.c_keywords = {
            'auto', 'break', 'case', 'char', 'const', 'continue', 'default',
            'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto',
            'if', 'int', 'long', 'register', 'return', 'short', 'signed',
            'sizeof', 'static', 'struct', 'switch', 'typedef', 'union',
            'unsigned', 'void', 'volatile', 'while', 'NULL', 'true', 'false'
        }
        
        # C语言操作符
        self.c_operators = {
            '+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', 
            '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>',
            '=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '^=', '|=',
            '->', '.', '::', '?', ':', ';', '{', '}', '(', ')', '[', ']', ','
        }
    
    def split_camelcase(self, identifier: str) -> List[str]:
        """分割驼峰命名"""
        # 在大写字母前插入空格
        result = re.sub('([A-Z][a-z]+)', r' \1', identifier)
        result = re.sub('([A-Z]+)', r' \1', result)
        return result.split()
    
    def tokenize_code(self, code: str) -> List[str]:
        """对C代码进行分词"""
        tokens = []
        
        # 移除注释
        # 移除单行注释
        code = re.sub(r'//.*?\n', '\n', code)
        # 移除多行注释
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # 处理字符串字面量（替换为特殊token）
        code = re.sub(r'"([^"\\]|\\.)*"', 'STRING_LITERAL', code)
        code = re.sub(r"'([^'\\]|\\.)'", 'CHAR_LITERAL', code)
        
        # 处理数字字面量
        code = re.sub(r'\b\d+\.?\d*([eE][+-]?\d+)?[fFlLuU]?\b', 'NUMBER_LITERAL', code)
        
        # 使用正则表达式提取token
        # 匹配标识符、操作符和其他符号
        pattern = r'\b\w+\b|[+\-*/%<>=!&|^~]+|[(){}[\];,.:?]|->|::|<<|>>|\+\+|--'
        raw_tokens = re.findall(pattern, code)
        
        for token in raw_tokens:
            if token in self.c_keywords:
                tokens.append(token)
            elif token in self.c_operators and self.keep_operators:
                tokens.append(token)
            elif re.match(r'^[a-zA-Z_]\w*$', token):  # 标识符
                if self.split_camel_case and '_' not in token:
                    # 分割驼峰命名
                    subtokens = self.split_camelcase(token)
                    tokens.extend([st.lower() for st in subtokens if st])
                else:
                    tokens.append(token.lower())
            elif token in ['STRING_LITERAL', 'CHAR_LITERAL', 'NUMBER_LITERAL']:
                tokens.append(token)
        
        return tokens

class Word2VecTrainer:
    """Word2Vec模型训练器"""
    
    def __init__(self, data_path: str, tokenizer: CodeTokenizer):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.corpus = []
        self.vocab_counter = Counter()
        
    def load_c_files(self) -> List[Dict[str, str]]:
        """加载所有C文件"""
        all_files = []
        
        # 读取Vul文件夹
        vul_path = self.data_path / 'Vul'
        if vul_path.exists():
            for c_file in vul_path.glob('*.c'):
                try:
                    with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        all_files.append({
                            'path': str(c_file),
                            'content': content,
                            'label': 'vulnerable'
                        })
                except Exception as e:
                    logger.error(f"Error reading {c_file}: {e}")
        
        # 读取No-Vul文件夹
        no_vul_path = self.data_path / 'No-Vul'
        if no_vul_path.exists():
            for c_file in no_vul_path.glob('*.c'):
                try:
                    with open(c_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        all_files.append({
                            'path': str(c_file),
                            'content': content,
                            'label': 'non-vulnerable'
                        })
                except Exception as e:
                    logger.error(f"Error reading {c_file}: {e}")
        
        logger.info(f"Loaded {len(all_files)} C files")
        return all_files
    
    def prepare_corpus(self, files: List[Dict[str, str]]) -> None:
        """准备训练语料"""
        logger.info("Tokenizing code files...")
        
        for file_info in tqdm(files, desc="Processing files"):
            tokens = self.tokenizer.tokenize_code(file_info['content'])
            if tokens:
                self.corpus.append(tokens)
                self.vocab_counter.update(tokens)
        
        logger.info(f"Corpus size: {len(self.corpus)} documents")
        logger.info(f"Vocabulary size: {len(self.vocab_counter)}")
        logger.info(f"Total tokens: {sum(self.vocab_counter.values())}")
    
    def train_word2vec(self, vector_size=300, window=10, min_count=3, 
                      workers=4, epochs=10, sg=1) -> Word2Vec:
        """训练Word2Vec模型"""
        logger.info("Training Word2Vec model...")
        
        model = Word2Vec(
            sentences=self.corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,  # 1 for skip-gram, 0 for CBOW
            epochs=epochs,
            seed=42
        )
        
        logger.info(f"Model trained. Vocabulary size: {len(model.wv)}")
        return model
    
    def train_fasttext(self, vector_size=300, window=10, min_count=3,
                      workers=4, epochs=10, min_n=3, max_n=6) -> FastText:
        """训练FastText模型（可选，处理OOV更好）"""
        logger.info("Training FastText model...")
        
        model = FastText(
            sentences=self.corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,
            epochs=epochs,
            min_n=min_n,
            max_n=max_n,
            seed=42
        )
        
        logger.info(f"Model trained. Vocabulary size: {len(model.wv)}")
        return model
    
    def save_model(self, model, model_path: str) -> None:
        """保存模型"""
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # 也保存词汇表和词频信息
        vocab_path = model_path.replace('.model', '_vocab.json')
        vocab_info = {
            'vocab_size': len(model.wv),
            'vector_size': model.wv.vector_size,
            'vocab_list': list(model.wv.index_to_key),
            'vocab_freq': {word: self.vocab_counter[word] for word in model.wv.index_to_key}
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary info saved to {vocab_path}")
    
    def analyze_model(self, model: Word2Vec, top_n=20) -> None:
        """分析模型质量"""
        logger.info("\n=== Model Analysis ===")
        
        # 词频统计
        most_common = self.vocab_counter.most_common(top_n)
        logger.info(f"\nTop {top_n} most frequent tokens:")
        for token, freq in most_common:
            logger.info(f"  {token}: {freq}")
        
        # 相似度测试
        test_words = ['malloc', 'free', 'buffer', 'int', 'return', 'if', 'for']
        logger.info("\nSimilarity tests:")
        
        for word in test_words:
            if word in model.wv:
                similar = model.wv.most_similar(word, topn=5)
                logger.info(f"\nMost similar to '{word}':")
                for sim_word, score in similar:
                    logger.info(f"  {sim_word}: {score:.4f}")
            else:
                logger.info(f"\n'{word}' not in vocabulary")

def main():
    """主函数"""
    # 配置参数
    DATA_PATH = "/mnt/data/lzh/xwj/rDevign"
    OUTPUT_DIR = "./w2c_models"
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化分词器
    tokenizer = CodeTokenizer(
        split_camel_case=True,  # 是否分割驼峰命名
        keep_operators=True     # 是否保留操作符
    )
    
    # 初始化训练器
    trainer = Word2VecTrainer(DATA_PATH, tokenizer)
    
    # 加载数据
    files = trainer.load_c_files()
    
    # 准备语料
    trainer.prepare_corpus(files)
    
    # 训练Word2Vec模型
    w2v_model = trainer.train_word2vec(
        vector_size=300,
        window=10,
        min_count=3,
        workers=4,
        epochs=15,
        sg=1  # Skip-gram
    )
    
    # 保存Word2Vec模型
    w2v_path = os.path.join(OUTPUT_DIR, "code_word2vec.model")
    trainer.save_model(w2v_model, w2v_path)
    
    # 分析模型
    trainer.analyze_model(w2v_model)
    
    logger.info("\nTraining completed!")

if __name__ == "__main__":
    main()