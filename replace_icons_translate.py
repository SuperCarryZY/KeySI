#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def replace_icons_and_translate():
    """Remove icons and translate Chinese to English in finaluimodified.py"""
    
    with open('finaluimodified.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove icons/emojis
    icon_pattern = r'[✓🎯🚀📍🌐🔥🔄⚠🔍🟡🔴🟢🔵🟣🟠🔶×]'
    content = re.sub(icon_pattern, '', content)
    
    # Translation mappings
    translations = {
        # Common phrases
        '当前显存使用': 'Current GPU memory usage',
        '显存缓存': 'GPU memory cache',
        '显存容量': 'GPU Memory',
        'GPU型号': 'GPU Model',
        'GPU内存优化已启用': 'GPU memory optimization enabled',
        'GPU内存已清理': 'GPU memory cleared',
        '已在CSV目录中': 'Already in CSV directory',
        '已切换到CSV目录': 'Switched to CSV directory',
        '当前工作目录': 'Current working directory',
        'GPU模式：使用更大的batch_size': 'GPU mode: using larger batch_size',
        'MPS模式：使用batch_size': 'MPS mode: using batch_size',
        'CPU模式：使用batch_size': 'CPU mode: using batch_size',
        
        # Model initialization
        '正在初始化SentenceTransformer模型': 'Initializing SentenceTransformer model',
        '首次运行可能需要下载预训练模型，请耐心等待': 'First run may require downloading pre-trained models, please wait',
        'SentenceTransformer模型初始化完成': 'SentenceTransformer model initialization completed',
        '正在初始化KeyBERT模型': 'Initializing KeyBERT model',
        'KeyBERT模型初始化完成': 'KeyBERT model initialization completed',
        
        # Data processing
        '正在加载数据': 'Loading data',
        '数据加载完成': 'Data loading completed',
        '正在提取关键字': 'Extracting keywords',
        'RTX 5090大显存模式：使用批处理大小': 'RTX 5090 large memory mode: using batch size',
        '使用GPU批处理，batch_size': 'Using GPU batch processing, batch_size',
        'GPU批处理中... 显存使用情况': 'GPU batch processing... memory usage',
        '批次完成，GPU利用率得到优化': 'Batch completed, GPU utilization optimized',
        '使用GPU批处理优化，充分利用RTX 5090性能': 'Using GPU batch processing optimization, fully utilizing RTX 5090 performance',
        '预处理完成，有效文章数': 'Preprocessing completed, valid articles',
        '关键字提取完成': 'Keyword extraction completed',
        '关键字嵌入计算完成': 'Keyword embedding calculation completed',
        
        # Dimensionality reduction
        '使用BertTopic进行关键词降维': 'Using BertTopic for keyword dimensionality reduction',
        'BertTopic降维完成，主题数量': 'BertTopic dimensionality reduction completed, topic count',
        '主题数量.*少于关键词数量.*，回退到t-SNE': 'Topic count is less than keyword count, falling back to t-SNE',
        'BertTopic降维失败，主题嵌入为空': 'BertTopic dimensionality reduction failed, topic embeddings are empty',
        '获取主题嵌入失败': 'Failed to get topic embeddings',
        'BertTopic降维出错': 'BertTopic dimensionality reduction error',
        '回退到t-SNE降维': 'Falling back to t-SNE dimensionality reduction',
        '使用t-SNE进行关键词降维': 'Using t-SNE for keyword dimensionality reduction',
        't-SNE降维完成': 't-SNE dimensionality reduction completed',
        
        # Clustering
        '正在进行层次聚类': 'Performing hierarchical clustering',
        '层次聚类完成': 'Hierarchical clustering completed',
        
        # NLTK
        'NLTK数据包.*已存在': 'NLTK package already exists',
        'NLTK数据包.*不存在，正在下载': 'NLTK package does not exist, downloading',
        'NLTK数据包.*下载完成': 'NLTK package download completed',
        
        # BertTopic
        'BertTopic可用，将使用BertTopic进行关键词降维': 'BertTopic available, will use BertTopic for keyword dimensionality reduction',
        'BertTopic不可用，将使用t-SNE作为备选方案': 'BertTopic not available, will use t-SNE as fallback',
        
        # UI text
        '关键词分组': 'Keyword Grouping',
        '输入组数': 'Enter number of groups',
        '生成组': 'Generate Groups',
        '清空所有': 'Clear All',
        '添加关键词': 'Add Keyword',
        '输入关键词': 'Enter a keyword',
        '查看详情': 'View Details',
        '查看文本': 'View Text',
        '文件详情': 'File Details',
        '操作选项': 'Actions',
        '点击左侧文件以查看详情': 'Click on left files to view details',
        '推荐文件': 'Recommended Files',
        '基于选中组的关键词，显示包含这些关键词的文件': 'Based on selected group keywords, showing files containing these keywords',
        
        # Messages
        '请先选择关键词组': 'Please select a keyword group first',
        '中没有关键词': 'has no keywords',
        '数据未加载': 'Data not loaded',
        '没有找到': 'Not found',
        '个文件': 'files',
        '共找到': 'Found',
        '包含关键词': 'containing keyword',
        '包含组': 'containing group',
        '关键词的文件': 'keyword files',
        '关键词对应的文件': 'files for keywords',
        
        # Debug messages
        'update_keyword_styles 被调用': 'update_keyword_styles called',
        'update_keyword_styles 退出': 'update_keyword_styles exit',
        'update_group_order 被调用': 'update_group_order called',
        'handle_plot_click 被调用': 'handle_plot_click called',
        'handle_plot_click 退出': 'handle_plot_click exit',
        'render_groups 被调用': 'render_groups called',
        'select_group 被调用': 'select_group called',
        'remove_keyword_from_group 被调用': 'remove_keyword_from_group called',
        '点击的关键词': 'Clicked keyword',
        '当前 group_data': 'Current group_data',
        '选中的组': 'Selected group',
        '返回的 new_data': 'Returned new_data',
        '添加关键词但不自动选中，保持显示整组文件': 'Added keyword but not auto-selected, keeping group files display',
        '切换到组': 'Switching to group',
        '清除选中关键词': 'Clear selected keyword',
        '从组管理中选择关键词': 'Select keyword from group management',
        '跳过删除操作：n_clicks为None，可能是自动触发': 'Skip delete operation: n_clicks is None, possibly auto-triggered',
        '删除按钮被真正点击': 'Delete button actually clicked',
        '搜索特定关键词': 'Search specific keyword',
        '搜索整组关键词': 'Search all group keywords',
        
        # File operations
        '已选择文件': 'Selected file',
        '文件内容预览': 'File content preview',
        '点击上方按钮查看完整文本': 'Click button above to view full text',
        '的完整文本': 'complete text',
        '关键词提取失败': 'Keyword extraction failed',
        
        # Error messages
        '生成推荐文件时出错': 'Error generating recommended files',
        '样本数量太少': 'Too few samples',
        '跳过此组': 'skipping this group',
        
        # Comments in Chinese
        '确保关键字在全局作用域中可用': 'Ensure keywords are available in global scope',
        '确保关键字变量已定义': 'Ensure keyword variables are defined',
        '自动检测并选择最佳设备': 'Auto-detect and select best device',
        'GPU内存优化设置': 'GPU memory optimization settings',
        '设置应用布局': 'Set application layout',
        '服务器端回调：更新关键字按钮样式': 'Server-side callback: update keyword button styles',
        '处理图表点击事件，将关键字添加到选中的组': 'Handle plot click events, add keywords to selected group',
        '根据选中关键词生成推荐文件列表': 'Generate recommended file list based on selected keywords',
        '默认显示组内所有关键词对应的文件': 'Default display files for all keywords in group',
        '提取文本的前N个关键词': 'Extract top N keywords from text',
        '显示文件的文本内容': 'Display file text content',
        '从组中移除关键字': 'Remove keyword from group',
        '处理从组管理中选择关键词': 'Handle keyword selection from group management',
        
        # Variable names/labels that appear in UI
        '组': 'Group',
        '关键词': 'keyword',
        '文件': 'File',
        '组别': 'group',
    }
    
    # Apply translations
    for chinese, english in translations.items():
        if '.*' in chinese:  # regex pattern
            content = re.sub(chinese, english, content)
        else:
            content = content.replace(chinese, english)
    
    # Remove any remaining emoji/icon characters
    content = re.sub(r'[\u2600-\u26FF\u2700-\u27BF\u1F300-\u1F5FF\u1F600-\u1F64F\u1F680-\u1F6FF\u1F700-\u1F77F\u1F780-\u1F7FF\u1F800-\u1F8FF\u1F900-\u1F9FF\u1FA00-\u1FA6F\u1FA70-\u1FAFF\u2190-\u21FF]', '', content)
    
    # Clean up extra spaces left by icon removal
    content = re.sub(r'  +', ' ', content)
    content = re.sub(r'^ +', '', content, flags=re.MULTILINE)
    
    with open('finaluimodified.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Icons removed and Chinese text translated to English!")

if __name__ == '__main__':
    replace_icons_and_translate()
