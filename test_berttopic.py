#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试BertTopic降维功能
"""

import numpy as np
from sklearn.manifold import TSNE

def test_berttopic_import():
    """测试BertTopic导入"""
    try:
        from bertopic import BERTopic
        print("✓ BertTopic导入成功")
        return True
    except ImportError as e:
        print(f"❌ BertTopic导入失败: {e}")
        print("请运行: pip install bertopic")
        return False

def test_berttopic_dimensionality_reduction():
    """测试BertTopic降维功能"""
    if not test_berttopic_import():
        return False
    
    try:
        from bertopic import BERTopic
        
        # 模拟关键词数据
        keywords = [
            "machine learning", "deep learning", "neural networks",
            "data science", "artificial intelligence", "computer vision",
            "natural language processing", "reinforcement learning",
            "supervised learning", "unsupervised learning",
            "clustering", "classification", "regression",
            "big data", "data mining", "statistics",
            "python", "tensorflow", "pytorch", "scikit-learn"
        ]
        
        print(f"测试关键词数量: {len(keywords)}")
        
        # 创建BertTopic模型
        topic_model = BERTopic(
            nr_topics="auto",
            top_n_words=20,
            min_topic_size=2,
            verbose=True
        )
        
        # 将关键词转换为文档格式
        keyword_docs = keywords  # 直接使用字符串列表
        
        print("正在进行BertTopic降维...")
        topic_model = BERTopic(
            nr_topics="auto",
            top_n_words=20,
            min_topic_size=2,
            verbose=True
        )
        # 修复：确保输入是字符串列表，而不是嵌套列表
        keyword_docs = keywords  # 直接使用字符串列表
        topics, probs = topic_model.fit_transform(keyword_docs)
        
        # 获取降维后的坐标 - 使用正确的API
        try:
            # 获取主题嵌入
            topic_embeddings = topic_model.topic_embeddings_
            if topic_embeddings is not None:
                print(f"✓ BertTopic降维完成，主题数量: {len(topic_embeddings)}")
                return True
            else:
                print("⚠ BertTopic降维失败，主题嵌入为空")
                return False
        except Exception as e:
            print(f"⚠ 获取主题嵌入失败: {e}")
            return False
    except Exception as e:
        print(f"⚠ BertTopic降维出错: {e}")
        print("🔄 回退到t-SNE降维...")
        return False

def test_tsne_fallback():
    """测试t-SNE备选方案"""
    print("\n测试t-SNE备选方案...")
    
    try:
        # 模拟关键词嵌入
        keyword_embeddings = np.random.randn(20, 768)  # 768维BERT嵌入
        
        # 使用t-SNE降维
        perplexity = min(30, max(5, len(keyword_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_embeddings = tsne.fit_transform(keyword_embeddings)
        
        print(f"✓ t-SNE降维成功，降维后维度: {reduced_embeddings.shape}")
        return True
        
    except Exception as e:
        print(f"❌ t-SNE降维测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("BertTopic降维功能测试")
    print("=" * 50)
    
    # 测试BertTopic
    berttopic_success = test_berttopic_dimensionality_reduction()
    
    # 测试t-SNE备选方案
    tsne_success = test_tsne_fallback()
    
    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    
    if berttopic_success:
        print("✓ BertTopic降维功能正常")
    else:
        print("❌ BertTopic降维功能异常")
    
    if tsne_success:
        print("✓ t-SNE备选方案正常")
    else:
        print("❌ t-SNE备选方案异常")
    
    if berttopic_success or tsne_success:
        print("\n🎉 至少有一种降维方法可用，系统可以正常运行")
    else:
        print("\n💥 所有降维方法都失败，需要检查依赖安装")

if __name__ == "__main__":
    main()
