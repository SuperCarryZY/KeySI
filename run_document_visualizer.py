#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档2D可视化工具 - 完全自动化版本
"""

import os
import sys
from document_2d_visualizer import Document2DVisualizer

def main():
    print("📄 文档2D位置可视化工具 - 完全自动化版本")
    print("=" * 60)
    
    # 设置工作目录
    os.chdir("/Users/yanzhu/Box Sync/Yan/KeySI/CSV")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 默认设置
    selected_file = "risk_factors.csv"
    target_keywords = ["cancer", "covid"]  # 默认搜索多个关键词
    text_column = 1  # 默认文本列索引
    
    # 检查文件是否存在
    if not os.path.exists(selected_file):
        print(f"❌ 文件不存在: {selected_file}")
        return
    
    print(f"✅ 使用文件: {selected_file}")
    print(f"🎯 搜索关键词: {', '.join(target_keywords)}")
    print(f"📊 文本列索引: {text_column}")
    
    # 创建可视化器
    print("\n🚀 开始处理...")
    visualizer = Document2DVisualizer(device="cpu")
    
    # 处理文档 - 支持多关键词
    print(f"\n📖 处理文件: {selected_file}")
    
    # 读取数据
    import pandas as pd
    df = pd.read_csv(selected_file)
    
    # 查找包含所有关键词的文档
    matching_indices = []
    for idx, row in df.iterrows():
        text = row.iloc[text_column]
        processed_text = visualizer.preprocess_text(text)
        
        if not processed_text:
            continue
            
        # 检查是否包含所有关键词
        contains_all_keywords = True
        for keyword in target_keywords:
            keyword_lower = keyword.lower()
            keyword_stemmed = visualizer.ps.stem(keyword_lower)
            
            # 分词
            import nltk
            from nltk.tokenize import word_tokenize
            words = word_tokenize(processed_text)
            
            # 检查是否包含当前关键词
            found = False
            for word in words:
                word_lower = word.lower()
                word_stemmed = visualizer.ps.stem(word_lower)
                
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
    
    if not matching_indices:
        print(f"❌ 未找到同时包含所有关键词 {target_keywords} 的文档")
        return
    
    print(f"找到 {len(matching_indices)} 个同时包含所有关键词的文档")
    
    # 提取匹配文档的文本和关键词
    documents_info = []
    for idx in matching_indices:
        text = df.iloc[idx, text_column]
        keywords = visualizer.extract_keywords_from_document(text, top_n=5)
        
        # 截取文本前100个字符作为摘要
        text_summary = str(text)[:100] + "..." if len(str(text)) > 100 else str(text)
        
        documents_info.append({
            'index': idx,
            'text': text,
            'text_summary': text_summary,
            'keywords': keywords
        })
    
    # 创建文档信息字典 - 使用单关键词格式以兼容现有可视化函数
    documents_info_dict = {
        'file_name': os.path.basename(selected_file),
        'target_keyword': ", ".join(target_keywords),  # 转换为字符串
        'matching_documents': documents_info
    }
    
    # 显示处理结果摘要
    print(f"\n📈 处理结果摘要:")
    print(f"  找到 {len(documents_info)} 个同时包含关键词 {', '.join(target_keywords)} 的文档")
    
    # 显示前几个文档的摘要
    print(f"\n📋 文档摘要:")
    for i, doc in enumerate(documents_info[:3]):  # 只显示前3个
        print(f"  文档 {i+1} (索引: {doc['index']}): {doc['text_summary']}")
    
    if len(documents_info) > 3:
        print(f"  ... 还有 {len(documents_info) - 3} 个文档")
    
    # 可视化
    print("\n🎨 生成2D可视化...")
    keywords_str = "_".join(target_keywords)
    output_path = f"document_2d_{keywords_str}.png"
    visualizer.visualize_documents_2d(documents_info_dict, output_path)
    
    print(f"\n✅ 完成！可视化结果已保存到: {output_path}")
    print(f"📊 共处理了 {len(documents_info)} 个同时包含关键词 {', '.join(target_keywords)} 的文档")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc() 