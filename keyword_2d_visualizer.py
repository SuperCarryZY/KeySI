#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键词2D位置可视化工具
可以输入关键词查看其在2D空间中的位置，支持多个文件
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk
from collections import Counter
from joblib import Parallel, delayed
import argparse
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置NLTK数据路径
nltk.data.path.append("/Users/yanzhu/nltk_data")
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

class Keyword2DVisualizer:
    def __init__(self, device="cpu"):
        """
        初始化关键词2D可视化器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.embedding_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
        self.kw_model = KeyBERT(model=self.embedding_model)
        self.ps = PorterStemmer()
        
    def extract_keywords_from_text(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        从文本中提取关键词
        
        Args:
            text: 输入文本
            top_n: 提取的关键词数量
            
        Returns:
            关键词列表，每个元素为(关键词, 分数)
        """
        try:
            # 预处理文本
            text = re.sub(r'\d+', '', text).strip()
            if len(text.split()) < 5:
                return []
                
            # 提取名词
            words = word_tokenize(text)
            tagged_words = pos_tag(words)
            nouns = [word for word, pos in tagged_words if pos.startswith("NN")]
            
            if not nouns:
                return []
                
            # 使用KeyBERT提取关键词
            keywords_info = self.kw_model.extract_keywords(
                " ".join(nouns), 
                keyphrase_ngram_range=(1, 1), 
                stop_words='english', 
                top_n=top_n
            )
            
            return keywords_info
            
        except Exception as e:
            print(f"提取关键词时出错: {e}")
            return []
    
    def process_csv_file(self, file_path: str, text_column: int = 1) -> Dict[str, List[Tuple[str, float]]]:
        """
        处理CSV文件，提取所有文本的关键词
        
        Args:
            file_path: CSV文件路径
            text_column: 文本列索引（默认为第1列）
            
        Returns:
            文件关键词字典 {文件名: [(关键词, 分数), ...]}
        """
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) <= text_column:
                print(f"警告: 文件 {file_path} 的列数不足，跳过")
                return {}
                
            all_texts = df.iloc[:, text_column].dropna().astype(str).tolist()
            
            # 并行处理所有文本
            results = Parallel(n_jobs=4)(
                delayed(self.extract_keywords_from_text)(text) for text in all_texts
            )
            
            # 统计关键词频率
            keyword_count = Counter()
            keyword_scores = {}
            
            for result in results:
                for kw, score in result:
                    stemmed = self.ps.stem(kw.lower())
                    keyword_count[stemmed] += 1
                    if stemmed not in keyword_scores or score > keyword_scores[stemmed]:
                        keyword_scores[stemmed] = score
            
            # 返回原始形式的关键词和分数
            file_keywords = []
            for stemmed, count in keyword_count.items():
                if count >= 2:  # 至少出现2次
                    # 找到原始形式
                    original_forms = []
                    for result in results:
                        for kw, score in result:
                            if self.ps.stem(kw.lower()) == stemmed:
                                original_forms.append(kw)
                    
                    if original_forms:
                        # 选择最短的原始形式
                        original_form = min(original_forms, key=len)
                        file_keywords.append((original_form, keyword_scores[stemmed]))
            
            return {os.path.basename(file_path): file_keywords}
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return {}
    
    def visualize_keywords_2d(self, file_keywords: Dict[str, List[Tuple[str, float]]], 
                            target_keywords: List[str] = None,
                            save_path: str = "keyword_2d_visualization.png"):
        """
        可视化关键词的2D位置
        
        Args:
            file_keywords: 文件关键词字典
            target_keywords: 要特别标记的目标关键词列表
            save_path: 保存图片的路径
        """
        # 收集所有关键词
        all_keywords = []
        keyword_to_file = {}
        
        for filename, keywords in file_keywords.items():
            for kw, score in keywords:
                all_keywords.append(kw)
                keyword_to_file[kw] = filename
        
        if not all_keywords:
            print("没有找到关键词")
            return
        
        # 计算关键词的embedding
        print(f"计算 {len(all_keywords)} 个关键词的embedding...")
        keyword_embeddings = self.embedding_model.encode(all_keywords, convert_to_tensor=True).cpu().numpy()
        
        # TSNE降维到2D
        print("使用TSNE进行2D投影...")
        perplexity = min(30, max(5, len(keyword_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        keyword_2d = tsne.fit_transform(keyword_embeddings)
        
        # 创建可视化
        plt.figure(figsize=(15, 10))
        
        # 按文件分组绘制
        colors = plt.cm.Set3(np.linspace(0, 1, len(file_keywords)))
        file_colors = {filename: colors[i] for i, filename in enumerate(file_keywords.keys())}
        
        for filename, keywords in file_keywords.items():
            file_kws = [kw for kw, _ in keywords]
            file_indices = [i for i, kw in enumerate(all_keywords) if kw in file_kws]
            
            if file_indices:
                x_coords = [keyword_2d[i, 0] for i in file_indices]
                y_coords = [keyword_2d[i, 1] for i in file_indices]
                
                plt.scatter(x_coords, y_coords, 
                          c=[file_colors[filename]], 
                          label=filename, 
                          alpha=0.7, s=50)
                
                # 标注关键词
                for i, idx in enumerate(file_indices):
                    kw = all_keywords[idx]
                    plt.annotate(kw, (keyword_2d[idx, 0], keyword_2d[idx, 1]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        # 特别标记目标关键词
        if target_keywords:
            target_indices = [i for i, kw in enumerate(all_keywords) if kw.lower() in [tk.lower() for tk in target_keywords]]
            
            if target_indices:
                target_x = [keyword_2d[i, 0] for i in target_indices]
                target_y = [keyword_2d[i, 1] for i in target_indices]
                target_kws = [all_keywords[i] for i in target_indices]
                
                plt.scatter(target_x, target_y, 
                          c='red', s=200, marker='*', 
                          label='Target Keywords', edgecolors='black', linewidth=2)
                
                # 标注目标关键词
                for i, idx in enumerate(target_indices):
                    kw = all_keywords[idx]
                    plt.annotate(f'★ {kw}', (keyword_2d[idx, 0], keyword_2d[idx, 1]), 
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=12, fontweight='bold', color='red',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.title('关键词2D位置可视化', fontsize=16, fontweight='bold')
        plt.xlabel('TSNE维度1', fontsize=12)
        plt.ylabel('TSNE维度2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        
        # 显示图片
        plt.show()
        
        # 打印目标关键词的位置信息
        if target_keywords:
            print("\n目标关键词位置信息:")
            for target_kw in target_keywords:
                for i, kw in enumerate(all_keywords):
                    if kw.lower() == target_kw.lower():
                        x, y = keyword_2d[i, 0], keyword_2d[i, 1]
                        file_name = keyword_to_file[kw]
                        print(f"  '{kw}' -> 位置: ({x:.3f}, {y:.3f}), 文件: {file_name}")
                        break
                else:
                    print(f"  '{target_kw}' -> 未找到")

def main():
    parser = argparse.ArgumentParser(description='关键词2D位置可视化工具')
    parser.add_argument('--files', nargs='+', required=True, help='CSV文件路径列表')
    parser.add_argument('--keywords', nargs='+', default=[], help='要特别标记的关键词列表')
    parser.add_argument('--text_column', type=int, default=1, help='文本列索引（默认为1）')
    parser.add_argument('--output', default='keyword_2d_visualization.png', help='输出图片路径')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = Keyword2DVisualizer(device=args.device)
    
    # 处理所有文件
    all_file_keywords = {}
    for file_path in args.files:
        if os.path.exists(file_path):
            print(f"处理文件: {file_path}")
            file_keywords = visualizer.process_csv_file(file_path, args.text_column)
            all_file_keywords.update(file_keywords)
        else:
            print(f"文件不存在: {file_path}")
    
    if not all_file_keywords:
        print("没有成功处理任何文件")
        return
    
    # 可视化
    visualizer.visualize_keywords_2d(all_file_keywords, args.keywords, args.output)

if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式界面
    import sys
    
    if len(sys.argv) == 1:
        print("关键词2D位置可视化工具")
        print("=" * 50)
        
        # 获取文件路径
        files_input = input("请输入CSV文件路径（多个文件用空格分隔）: ").strip()
        if not files_input:
            print("未输入文件路径")
            sys.exit(1)
        
        files = files_input.split()
        
        # 获取目标关键词
        keywords_input = input("请输入要特别标记的关键词（多个关键词用空格分隔，直接回车跳过）: ").strip()
        keywords = keywords_input.split() if keywords_input else []
        
        # 获取文本列索引
        text_column_input = input("请输入文本列索引（默认为1）: ").strip()
        text_column = int(text_column_input) if text_column_input.isdigit() else 1
        
        # 设置参数
        sys.argv = [
            'keyword_2d_visualizer.py',
            '--files'] + files + [
            '--text_column', str(text_column)
        ]
        
        if keywords:
            sys.argv.extend(['--keywords'] + keywords)
    
    main() 