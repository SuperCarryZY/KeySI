#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中心点计算方法比较工具
比较关键词组embedding中心点 vs 文档平均embedding中心点
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
import warnings
warnings.filterwarnings('ignore')

# 设置NLTK数据路径
nltk.data.path.append("/Users/yanzhu/nltk_data")
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

class CenterComparison:
    def __init__(self, device="cpu"):
        """
        初始化中心点比较器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.embedding_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
        self.kw_model = KeyBERT(model=self.embedding_model)
        self.ps = PorterStemmer()
        
    def preprocess_text(self, text):
        """
        预处理文本
        """
        if pd.isna(text) or text == '':
            return ''
        
        # 转换为字符串
        text = str(text)
        
        # 移除特殊字符和数字
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # 转换为小写
        text = text.lower()
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def find_documents_with_keywords(self, df, keywords, text_column=1):
        """
        查找包含指定关键词组的文档
        
        Args:
            df: 数据框
            keywords: 关键词列表
            text_column: 文本列索引
            
        Returns:
            包含所有关键词的文档索引列表
        """
        matching_indices = []
        
        for idx, row in df.iterrows():
            text = row.iloc[text_column]
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                continue
                
            # 检查是否包含所有关键词
            contains_all_keywords = True
            for keyword in keywords:
                keyword_lower = keyword.lower()
                keyword_stemmed = self.ps.stem(keyword_lower)
                
                # 分词
                words = word_tokenize(processed_text)
                
                # 检查是否包含当前关键词
                found = False
                for word in words:
                    word_lower = word.lower()
                    word_stemmed = self.ps.stem(word_lower)
                    
                    if (keyword_lower == word_lower or 
                        keyword_stemmed == word_stemmed or
                        keyword_lower in word_lower):
                        found = True
                        break
                
                if not found:
                    contains_all_keywords = False
                    break
            
            if contains_all_keywords:
                matching_indices.append(idx)
        
        return matching_indices
    
    def calculate_keyword_group_center(self, keywords):
        """
        计算关键词组的embedding中心点
        
        Args:
            keywords: 关键词列表
            
        Returns:
            关键词组的embedding中心点
        """
        print(f"计算关键词组 '{', '.join(keywords)}' 的embedding...")
        keyword_embeddings = self.embedding_model.encode(keywords, convert_to_tensor=True).cpu().numpy()
        
        # 计算关键词组的平均embedding作为中心点
        keyword_center = np.mean(keyword_embeddings, axis=0)
        
        return keyword_center, keyword_embeddings
    
    def calculate_document_average_center(self, documents_texts):
        """
        计算文档的平均embedding中心点
        
        Args:
            documents_texts: 文档文本列表
            
        Returns:
            文档的平均embedding中心点
        """
        print(f"计算 {len(documents_texts)} 个文档的embedding...")
        document_embeddings = self.embedding_model.encode(documents_texts, convert_to_tensor=True).cpu().numpy()
        
        # 计算文档的平均embedding作为中心点
        document_center = np.mean(document_embeddings, axis=0)
        
        return document_center, document_embeddings
    
    def compare_centers(self, file_path, keywords, text_column=1, save_path="center_comparison.png"):
        """
        比较两种中心点计算方法
        
        Args:
            file_path: CSV文件路径
            keywords: 关键词列表
            text_column: 文本列索引
            save_path: 保存图片的路径
        """
        print("📄 中心点计算方法比较")
        print("=" * 60)
        print(f"文件: {file_path}")
        print(f"关键词组: {', '.join(keywords)}")
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 查找包含所有关键词的文档
        print(f"\n🔍 查找包含关键词组的文档...")
        matching_indices = self.find_documents_with_keywords(df, keywords, text_column)
        
        if not matching_indices:
            print(f"❌ 未找到同时包含所有关键词 {keywords} 的文档")
            return
        
        print(f"✅ 找到 {len(matching_indices)} 个包含关键词组的文档")
        
        # 获取匹配文档的文本
        documents_texts = [df.iloc[idx, text_column] for idx in matching_indices]
        
        # 计算两种中心点
        print(f"\n📊 计算中心点...")
        
        # 1. 关键词组embedding中心点
        keyword_center, keyword_embeddings = self.calculate_keyword_group_center(keywords)
        
        # 2. 文档平均embedding中心点
        document_center, document_embeddings = self.calculate_document_average_center(documents_texts)
        
        # 计算距离
        keyword_to_doc_distances = []
        doc_to_doc_distances = []
        
        for doc_emb in document_embeddings:
            # 文档到关键词组中心的距离
            keyword_dist = np.linalg.norm(doc_emb - keyword_center)
            keyword_to_doc_distances.append(keyword_dist)
            
            # 文档到文档平均中心的距离
            doc_dist = np.linalg.norm(doc_emb - document_center)
            doc_to_doc_distances.append(doc_dist)
        
        # 统计信息
        print(f"\n📈 距离统计:")
        print(f"关键词组中心到文档的平均距离: {np.mean(keyword_to_doc_distances):.4f} ± {np.std(keyword_to_doc_distances):.4f}")
        print(f"文档平均中心到文档的平均距离: {np.mean(doc_to_doc_distances):.4f} ± {np.std(doc_to_doc_distances):.4f}")
        
        # 比较哪种方法更好
        if np.mean(keyword_to_doc_distances) < np.mean(doc_to_doc_distances):
            print(f"✅ 关键词组embedding中心点更好 (距离更小)")
        else:
            print(f"✅ 文档平均embedding中心点更好 (距离更小)")
        
        # 2D可视化
        print(f"\n🎨 生成2D可视化...")
        
        # 合并所有embedding进行TSNE
        all_embeddings = np.vstack([keyword_embeddings, document_embeddings])
        
        # TSNE降维
        perplexity = min(30, max(5, len(all_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        all_2d = tsne.fit_transform(all_embeddings)
        
        # 分离关键词和文档的2D坐标
        keyword_2d = all_2d[:len(keywords)]
        document_2d = all_2d[len(keywords):]
        
        # 计算中心点的2D坐标
        keyword_center_2d = np.mean(keyword_2d, axis=0)
        document_center_2d = np.mean(document_2d, axis=0)
        
        # 创建可视化
        plt.figure(figsize=(15, 12))
        
        # 绘制关键词
        plt.scatter(keyword_2d[:, 0], keyword_2d[:, 1], 
                   c='red', s=200, marker='s', label='关键词', alpha=0.8)
        
        # 标注关键词
        for i, keyword in enumerate(keywords):
            plt.annotate(keyword, (keyword_2d[i, 0], keyword_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold', color='red')
        
        # 绘制文档
        plt.scatter(document_2d[:, 0], document_2d[:, 1], 
                   c='blue', s=100, alpha=0.6, label='文档')
        
        # 绘制两种中心点
        plt.scatter(keyword_center_2d[0], keyword_center_2d[1], 
                   c='red', s=300, marker='*', edgecolors='black', linewidth=2,
                   label='关键词组中心点')
        
        plt.scatter(document_center_2d[0], document_center_2d[1], 
                   c='green', s=300, marker='*', edgecolors='black', linewidth=2,
                   label='文档平均中心点')
        
        # 标注中心点
        plt.annotate('关键词组中心', (keyword_center_2d[0], keyword_center_2d[1]), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=14, fontweight='bold', color='red',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.annotate('文档平均中心', (document_center_2d[0], document_center_2d[1]), 
                    xytext=(10, -10), textcoords='offset points',
                    fontsize=14, fontweight='bold', color='green',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.title(f'中心点计算方法比较\n关键词组: {", ".join(keywords)}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('TSNE维度1', fontsize=12)
        plt.ylabel('TSNE维度2', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
        
        # 显示图片
        plt.show()
        
        # 返回比较结果
        return {
            'keyword_center_distance': np.mean(keyword_to_doc_distances),
            'document_center_distance': np.mean(doc_to_doc_distances),
            'keyword_center_std': np.std(keyword_to_doc_distances),
            'document_center_std': np.std(doc_to_doc_distances),
            'better_method': 'keyword' if np.mean(keyword_to_doc_distances) < np.mean(doc_to_doc_distances) else 'document'
        }

def main():
    print("🔬 中心点计算方法比较工具")
    print("=" * 60)
    
    # 设置工作目录
    os.chdir("/Users/yanzhu/Box Sync/Yan/KeySI/CSV")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 默认设置
    selected_file = "risk_factors.csv"
    target_keywords = ["smoke", "cancer", "tobacco"]  # 默认关键词组
    text_column = 1  # 默认文本列索引
    
    # 检查文件是否存在
    if not os.path.exists(selected_file):
        print(f"❌ 文件不存在: {selected_file}")
        return
    
    print(f"✅ 使用文件: {selected_file}")
    print(f"🎯 关键词组: {', '.join(target_keywords)}")
    print(f"📊 文本列索引: {text_column}")
    
    # 创建比较器
    print("\n🚀 开始比较...")
    comparator = CenterComparison(device="cpu")
    
    # 执行比较
    keywords_str = "_".join(target_keywords)
    output_path = f"center_comparison_{keywords_str}.png"
    result = comparator.compare_centers(selected_file, target_keywords, text_column, output_path)
    
    if result:
        print(f"\n📊 比较结果:")
        print(f"关键词组中心点平均距离: {result['keyword_center_distance']:.4f} ± {result['keyword_center_std']:.4f}")
        print(f"文档平均中心点平均距离: {result['document_center_distance']:.4f} ± {result['document_center_std']:.4f}")
        print(f"更好的方法: {'关键词组embedding中心点' if result['better_method'] == 'keyword' else '文档平均embedding中心点'}")
    
    print(f"\n✅ 完成！比较结果已保存到: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc() 