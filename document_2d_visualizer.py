#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档2D位置可视化工具
根据关键词查找包含该关键词的文档，然后进行2D可视化
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

class Document2DVisualizer:
    def __init__(self, device="cpu"):
        """
        初始化文档2D可视化器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = device
        self.embedding_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
        self.kw_model = KeyBERT(model=self.embedding_model)
        self.ps = PorterStemmer()
        
    def preprocess_text(self, text):
        """
        预处理文本，类似Final_UI.py中的方法
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
    
    def find_documents_with_keyword(self, df, keyword, text_column=1):
        """
        查找包含指定关键词的文档
        
        Args:
            df: 数据框
            keyword: 要查找的关键词
            text_column: 文本列索引
            
        Returns:
            包含关键词的文档索引列表
        """
        keyword_lower = keyword.lower()
        keyword_stemmed = self.ps.stem(keyword_lower)
        
        matching_indices = []
        
        for idx, row in df.iterrows():
            text = row.iloc[text_column]
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                continue
                
            # 分词
            words = word_tokenize(processed_text)
            
            # 检查是否包含关键词（原始形式、小写形式、词干形式）
            found = False
            for word in words:
                word_lower = word.lower()
                word_stemmed = self.ps.stem(word_lower)
                
                if (keyword_lower == word_lower or 
                    keyword_stemmed == word_stemmed or
                    keyword_lower in word_lower):
                    found = True
                    break
            
            if found:
                matching_indices.append(idx)
        
        return matching_indices
    
    def extract_keywords_from_document(self, text, top_n=10):
        """
        从单个文档中提取关键词
        """
        try:
            processed_text = self.preprocess_text(text)
            if len(processed_text.split()) < 5:
                return []
                
            # 提取名词
            words = word_tokenize(processed_text)
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
    
    def process_documents_with_keyword(self, file_path, target_keyword, text_column=1):
        """
        处理包含指定关键词的文档
        
        Args:
            file_path: CSV文件路径
            target_keyword: 目标关键词
            text_column: 文本列索引
            
        Returns:
            文档信息字典
        """
        try:
            df = pd.read_csv(file_path)
            if len(df.columns) <= text_column:
                print(f"警告: 文件 {file_path} 的列数不足，跳过")
                return {}
            
            print(f"查找包含关键词 '{target_keyword}' 的文档...")
            matching_indices = self.find_documents_with_keyword(df, target_keyword, text_column)
            
            if not matching_indices:
                print(f"未找到包含关键词 '{target_keyword}' 的文档")
                return {}
            
            print(f"找到 {len(matching_indices)} 个包含关键词 '{target_keyword}' 的文档")
            
            # 提取匹配文档的文本和关键词
            documents_info = []
            for idx in matching_indices:
                text = df.iloc[idx, text_column]
                keywords = self.extract_keywords_from_document(text, top_n=5)
                
                # 截取文本前100个字符作为摘要
                text_summary = str(text)[:100] + "..." if len(str(text)) > 100 else str(text)
                
                documents_info.append({
                    'index': idx,
                    'text': text,
                    'text_summary': text_summary,
                    'keywords': keywords
                })
            
            return {
                'file_name': os.path.basename(file_path),
                'target_keyword': target_keyword,
                'matching_documents': documents_info
            }
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return {}
    
    def visualize_documents_2d(self, documents_info, save_path="document_2d_visualization.png"):
        """
        可视化文档的2D位置
        
        Args:
            documents_info: 文档信息字典
            save_path: 保存图片的路径
        """
        if not documents_info:
            print("没有文档信息")
            return
        
        file_name = documents_info['file_name']
        target_keyword = documents_info['target_keyword']
        documents = documents_info['matching_documents']
        
        if not documents:
            print("没有匹配的文档")
            return
        
        # 计算文档的embedding
        print(f"计算 {len(documents)} 个文档的embedding...")
        document_texts = [doc['text'] for doc in documents]
        document_embeddings = self.embedding_model.encode(document_texts, convert_to_tensor=True).cpu().numpy()
        
        # TSNE降维到2D
        print("使用TSNE进行2D投影...")
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        document_2d = tsne.fit_transform(document_embeddings)
        
        # 创建可视化
        plt.figure(figsize=(15, 12))
        
        # 绘制文档点
        plt.scatter(document_2d[:, 0], document_2d[:, 1], 
                   c='blue', alpha=0.7, s=100, label=f'包含关键词 "{target_keyword}" 的文档')
        
        # 标注文档
        for i, doc in enumerate(documents):
            x, y = document_2d[i, 0], document_2d[i, 1]
            
            # 显示文档索引和关键词
            top_keywords = [kw for kw, score in doc['keywords'][:3]]  # 前3个关键词
            label = f"Doc {doc['index']}: {', '.join(top_keywords)}"
            
            plt.annotate(label, (x, y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
        
        plt.title(f'包含关键词 "{target_keyword}" 的文档2D位置可视化\n文件: {file_name}', 
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
        
        # 打印文档详细信息
        print(f"\n包含关键词 '{target_keyword}' 的文档详细信息:")
        print("=" * 80)
        for i, doc in enumerate(documents):
            print(f"\n文档 {i+1} (索引: {doc['index']}):")
            print(f"文本摘要: {doc['text_summary']}")
            print(f"2D位置: ({document_2d[i, 0]:.3f}, {document_2d[i, 1]:.3f})")
            print(f"关键词: {[kw for kw, score in doc['keywords']]}")
            print("-" * 40)

def main():
    print("📄 文档2D位置可视化工具 - 完全自动化版本")
    print("=" * 60)
    
    # 设置工作目录
    os.chdir("/Users/yanzhu/Box Sync/Yan/KeySI/CSV")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 默认设置
    selected_file = "risk_factors.csv"
    target_keyword = "cancer"  # 默认搜索关键词
    text_column = 1  # 默认文本列索引
    
    # 检查文件是否存在
    if not os.path.exists(selected_file):
        print(f"❌ 文件不存在: {selected_file}")
        return
    
    print(f"✅ 使用文件: {selected_file}")
    print(f"🎯 搜索关键词: {target_keyword}")
    print(f"📊 文本列索引: {text_column}")
    
    # 创建可视化器
    print("\n🚀 开始处理...")
    visualizer = Document2DVisualizer(device="cpu")
    
    # 处理文档
    print(f"\n📖 处理文件: {selected_file}")
    documents_info = visualizer.process_documents_with_keyword(selected_file, target_keyword, text_column)
    
    if not documents_info:
        print("❌ 没有找到匹配的文档")
        return
    
    # 显示处理结果摘要
    matching_docs = documents_info['matching_documents']
    print(f"\n📈 处理结果摘要:")
    print(f"  找到 {len(matching_docs)} 个包含关键词 '{target_keyword}' 的文档")
    
    # 显示前几个文档的摘要
    print(f"\n📋 文档摘要:")
    for i, doc in enumerate(matching_docs[:3]):  # 只显示前3个
        print(f"  文档 {i+1} (索引: {doc['index']}): {doc['text_summary']}")
    
    if len(matching_docs) > 3:
        print(f"  ... 还有 {len(matching_docs) - 3} 个文档")
    
    # 可视化
    print("\n🎨 生成2D可视化...")
    output_path = f"document_2d_{target_keyword}.png"
    visualizer.visualize_documents_2d(documents_info, output_path)
    
    print(f"\n✅ 完成！可视化结果已保存到: {output_path}")
    print(f"📊 共处理了 {len(matching_docs)} 个包含关键词 '{target_keyword}' 的文档")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc() 