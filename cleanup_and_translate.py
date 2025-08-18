#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def cleanup_and_translate():
    """Remove unnecessary icons, debug info and translate Chinese to English"""
    
    with open('finaluimodified.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove all emoji/icon characters first
    icon_pattern = r'[✓🎯🚀📍🌐🔥🔄⚠🔍🟡🔴🟢🔵🟣🟠🔶×]'
    content = re.sub(icon_pattern, '', content)
    
    # Remove debug print statements
    debug_patterns = [
        r'print\(f?"🟡.*?\)\n',
        r'print\(f?"🔴.*?\)\n',
        r'print\(f?"🟢.*?\)\n',
        r'print\(f?"🔵.*?\)\n',
        r'print\(f?"🟣.*?\)\n',
        r'print\(f?"🟠.*?\)\n',
        r'print\(f?"Debug:.*?\)\n',
        r'print\(".*debug.*"\)\n',
        r'print\(f".*debug.*"\)\n',
        r'# 调试信息[\s\S]*?print\(.*?\)\n',
    ]
    
    for pattern in debug_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Translation mappings for remaining Chinese text
    translations = {
        # Comments
        '# 自动检测并选择最佳设备：CUDA > MPS > CPU': '# Auto-detect and select best device: CUDA > MPS > CPU',
        '# GPU内存优化设置': '# GPU memory optimization settings',
        '# 确保关键字在全局作用域中可用': '# Ensure keywords are available in global scope',
        '# 确保关键字变量已定义': '# Ensure keyword variables are defined',
        '# 设置应用布局': '# Set application layout',
        '# 服务器端回调：更新关键字按钮样式': '# Server-side callback: update keyword button styles',
        '# 处理图表点击事件，将关键字添加到选中的组': '# Handle plot click events, add keywords to selected group',
        '# 根据选中关键词生成推荐文件列表': '# Generate recommended file list based on selected keywords',
        '# 默认显示组内所有关键词对应的文件': '# Default display files for all keywords in group',
        '# 提取文本的前N个关键词': '# Extract top N keywords from text',
        '# 显示文件的文本内容': '# Display file text content',
        '# 从组中移除关键字': '# Remove keyword from group',
        '# 处理从组管理中选择关键词': '# Handle keyword selection from group management',
        '# 顶部控制区域': '# Top control area',
        '# 手动添加关键字区域': '# Manual keyword addition area',
        '# 数据存储': '# Data storage',
        '# 主要内容区域 - 左右分栏布局': '# Main content area - left-right column layout',
        '# 左侧：关键字二维降维可视化': '# Left: keyword 2D visualization',
        '# 右侧：分组管理': '# Right: group management',
        '# 推荐文件显示区域': '# Recommended files display area',
        '# 文件详情区域': '# File details area',
        '# 训练按钮和输出': '# Training button and output',
        '# 训练结果可视化 - 两个并排的大图': '# Training result visualization - two side-by-side large plots',
        '# 文章内容显示区域': '# Article content display area',
        '# 调试输出区域': '# Debug output area',
        
        # UI text
        'Keyword Grouping - 2D Visualization': 'Keyword Grouping - 2D Visualization',
        'Enter number of groups:': 'Enter number of groups:',
        'Generate Groups': 'Generate Groups',
        'Enter a keyword': 'Enter a keyword',
        'Add Keyword': 'Add Keyword',
        'Keywords 2D Visualization': 'Keywords 2D Visualization',
        'Hover over points to see keywords, click to select': 'Hover over points to see keywords, click to select',
        'Group Management': 'Group Management',
        'Recommended Files': 'Recommended Files',
        'File Details': 'File Details',
        'Actions': 'Actions',
        'View Details': 'View Details',
        'View Text': 'View Text',
        'Train': 'Train',
        'Article Content': 'Article Content',
        'Click on a point in the plots above to view article content': 'Click on a point in the plots above to view article content',
        'Click on left files to view details': 'Click on left files to view details',
        
        # System messages (keep important ones, translate to English)
        'BertTopic可用，将使用BertTopic进行关键词降维': 'BertTopic available, using BertTopic for keyword dimensionality reduction',
        'BertTopic不可用，将使用t-SNE作为备选方案': 'BertTopic not available, using t-SNE as fallback',
        'NLTK数据包.*已存在': 'NLTK package already exists',
        'NLTK数据包.*不存在，正在下载': 'NLTK package not found, downloading',
        'NLTK数据包.*下载完成': 'NLTK package download completed',
        '使用 \'punkt\' 作为备选': 'Using \'punkt\' as fallback',
        'NVIDIA GPU detected, using CUDA acceleration': 'NVIDIA GPU detected, using CUDA acceleration',
        'Apple Silicon GPU detected, using MPS acceleration': 'Apple Silicon GPU detected, using MPS acceleration',
        'Using CPU - GPU recommended for better performance': 'Using CPU - GPU recommended for better performance',
        'CUDA memory optimization enabled': 'CUDA memory optimization enabled',
        'GPU memory cleared': 'GPU memory cleared',
        'Already in CSV directory': 'Already in CSV directory',
        'Switched to CSV directory': 'Switched to CSV directory',
        'Current working directory': 'Current working directory',
        'GPU mode: using larger batch_size': 'GPU mode: using larger batch_size',
        'MPS mode: using batch_size': 'MPS mode: using batch_size',
        'CPU mode: using batch_size': 'CPU mode: using batch_size',
        '正在初始化SentenceTransformer模型': 'Initializing SentenceTransformer model',
        '首次运行可能需要下载预训练模型，请耐心等待': 'First run may require downloading pre-trained models, please wait',
        'SentenceTransformer模型初始化完成': 'SentenceTransformer model initialization completed',
        '正在初始化KeyBERT模型': 'Initializing KeyBERT model',
        'KeyBERT模型初始化完成': 'KeyBERT model initialization completed',
        '正在加载数据': 'Loading data',
        '数据加载完成，共': 'Data loading completed, total',
        '篇文章': 'articles',
        '正在提取关键字': 'Extracting keywords',
        '使用GPU批处理优化，充分利用RTX 5090性能': 'Using GPU batch processing optimization, fully utilizing RTX 5090 performance',
        '预处理完成，有效文章数': 'Preprocessing completed, valid articles',
        '使用GPU批处理': 'Using GPU batch processing',
        '处理批次': 'Processing batch',
        '包含': 'containing',
        'GPU批处理中... 显存使用情况': 'GPU batch processing... memory usage',
        '已用显存': 'GPU memory used',
        '缓存显存': 'GPU memory cached',
        '批次完成，GPU利用率得到优化': 'Batch completed, GPU utilization optimized',
        '正在统计关键字': 'Counting keywords',
        '关键字提取完成，共': 'Keyword extraction completed, total',
        '个关键字': 'keywords',
        '正在计算关键字嵌入': 'Computing keyword embeddings',
        '关键字嵌入计算完成': 'Keyword embedding calculation completed',
        '正在进行关键词降维': 'Performing keyword dimensionality reduction',
        '使用BertTopic进行关键词降维': 'Using BertTopic for keyword dimensionality reduction',
        'BertTopic降维完成，主题数量': 'BertTopic dimensionality reduction completed, topic count',
        '主题数量.*少于关键词数量.*，回退到t-SNE': 'Topic count is less than keyword count, falling back to t-SNE',
        'BertTopic降维失败，主题嵌入为空': 'BertTopic dimensionality reduction failed, topic embeddings are empty',
        '获取主题嵌入失败': 'Failed to get topic embeddings',
        'BertTopic降维出错': 'BertTopic dimensionality reduction error',
        '回退到t-SNE降维': 'Falling back to t-SNE dimensionality reduction',
        '使用t-SNE进行关键词降维': 'Using t-SNE for keyword dimensionality reduction',
        't-SNE降维完成': 't-SNE dimensionality reduction completed',
        '正在进行层次聚类': 'Performing hierarchical clustering',
        '层次聚类完成': 'Hierarchical clustering completed',
        'Keyword clustering completed': 'Keyword clustering completed',
        'categories found': 'categories found',
        'Starting Web Application': 'Starting Web Application',
        'Dash应用即将启动': 'Dash app starting',
        '应用将在.*运行': 'App running at http://127.0.0.1:8050',
        '请在浏览器中打开上述地址': 'Please open the above address in your browser',
        
        # Function names and variables
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
        '真正的删除按钮被点击': 'Delete button actually clicked',
        '搜索特定关键词': 'Search specific keyword',
        '搜索整组关键词': 'Search all group keywords',
        
        # Error messages
        '生成推荐文件时出错': 'Error generating recommended files',
        '样本数量太少': 'Too few samples',
        '跳过此组': 'skipping this group',
        '关键词提取失败': 'Keyword extraction failed',
        
        # File operations
        '已选择文件': 'Selected file',
        '文件内容预览': 'File content preview',
        '点击上方按钮查看完整文本': 'Click button above to view full text',
        '的完整文本': 'complete text',
        '没有找到': 'Not found',
        '个文件': 'files',
        '共找到': 'Found',
        '包含关键词': 'containing keyword',
        '包含组': 'containing group',
        '关键词的文件': 'keyword files',
        '关键词对应的文件': 'files for keywords',
        '请先选择关键词组': 'Please select a keyword group first',
        '中没有关键词': 'has no keywords',
        '数据未加载': 'Data not loaded',
        
        # Plot titles
        '2D Projection Before Finetuning': '2D Projection Before Finetuning',
        '2D Projection After Finetuning': '2D Projection After Finetuning',
        
        # Callback and function descriptions
        '服务器端回调：更新关键字按钮样式': 'Server-side callback: update keyword button styles',
        '处理关键字按钮点击 - 修复：只在真正点击关键字按钮时才修改数据': 'Handle keyword button clicks - fix: only modify data when actually clicking keyword buttons',
        '安全检查：如果没有关键字按钮，返回空样式': 'Safety check: if no keyword buttons, return empty styles',
        '默认样式 - 包含完整的按钮样式': 'Default style - includes complete button styles',
        '处理关键字按钮点击': 'Handle keyword button clicks',
        '更新关键字按钮样式': 'Update keyword button styles',
        '创建Plotly图表': 'Create Plotly charts',
        '添加散点图 - 每个类别': 'Add scatter plot - each category',
        '为每个点添加文章索引作为悬停信息': 'Add article index as hover info for each point',
        '确保customdata是正确的格式': 'Ensure customdata is in correct format',
        '每个点一个列表': 'One list per point',
        '如果是After Finetuning图，添加组中心点': 'If After Finetuning plot, add group center points',
        '添加黑色边框': 'Add black border',
        '生成两个图表': 'Generate two charts',
        '处理图表点击事件，显示文章内容': 'Handle chart click events, display article content',
        '确定是哪个图表被点击': 'Determine which chart was clicked',
        '获取点击的文章索引': 'Get clicked article index',
        'customdata现在是一个列表，需要取第一个元素': 'customdata is now a list, need to take first element',
        '检查文章索引是否有效': 'Check if article index is valid',
        '尝试从pointIndex获取索引': 'Try to get index from pointIndex',
        '需要根据trace和pointIndex计算实际的文章索引': 'Need to calculate actual article index based on trace and pointIndex',
        '运行训练并获取两个图表': 'Run training and get two charts',
        '显示训练输出和内容显示区域': 'Show training output and content display area',
        
        # Remove remaining specific debug prints
        'update_keyword_styles 被调用': '',
        'update_keyword_styles 退出': '',
        'update_group_order 被调用': '',
        'handle_plot_click 被调用': '',
        'handle_plot_click 退出': '',
        'render_groups 被调用': '',
        'select_group 被调用': '',
        'remove_keyword_from_group 被调用': '',
        'triggered': '',
        'group_data': '',
        'group_order': '',
        'selected_group': '',
        'selected_keyword': '',
    }
    
    # Apply translations
    for chinese, english in translations.items():
        if '.*' in chinese:  # regex pattern
            content = re.sub(chinese, english, content)
        else:
            content = content.replace(chinese, english)
    
    # Remove empty print statements
    content = re.sub(r'print\(f?""\)\n', '', content)
    content = re.sub(r'print\(""\)\n', '', content)
    content = re.sub(r'print\(\)\n', '', content)
    
    # Remove remaining emoji/icon characters (Unicode ranges)
    content = re.sub(r'[\u2600-\u26FF\u2700-\u27BF\u1F300-\u1F5FF\u1F600-\u1F64F\u1F680-\u1F6FF\u1F700-\u1F77F\u1F780-\u1F7FF\u1F800-\u1F8FF\u1F900-\u1F9FF\u1FA00-\u1FA6F\u1FA70-\u1FAFF\u2190-\u21FF]', '', content)
    
    # Clean up extra spaces and empty lines
    content = re.sub(r'  +', ' ', content)
    content = re.sub(r'^ +', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    with open('finaluimodified.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Icons, debug info removed and Chinese text translated to English!")

if __name__ == '__main__':
    cleanup_and_translate()
