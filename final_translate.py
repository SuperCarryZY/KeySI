#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def final_translate():
    """Final translation of all remaining Chinese text"""
    
    with open('finaluimodified.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Comprehensive translation mappings
    translations = {
        # Batch processing and GPU
        '根据设备类型动态调整batch_size以优化性能': 'Dynamically adjust batch_size based on device type for optimization',
        'RTX 5090可以处理更大的batch size': 'RTX 5090 can handle larger batch size',
        'GPU模式：使用更大的batch_size': 'GPU mode: using larger batch_size',
        'MPS模式：使用batch_size': 'MPS mode: using batch_size',
        'CPU模式：使用batch_size': 'CPU mode: using batch_size',
        'Apple Silicon GPU适中的batch size': 'Apple Silicon GPU moderate batch size',
        'CPU使用较小的batch size': 'CPU uses smaller batch size',
        
        # Model initialization
        '模型设置': 'Model settings',
        '正在初始化SentenceTransformer模型': 'Initializing SentenceTransformer model',
        '首次运行可能需要下载预训练模型，请耐心等待': 'First run may require downloading pre-trained models, please wait',
        'SentenceTransformer模型初始化完成': 'SentenceTransformer model initialization completed',
        '模型初始化失败': 'Model initialization failed',
        '正在初始化KeyBERT模型': 'Initializing KeyBERT model',
        'KeyBERT模型初始化完成': 'KeyBERT model initialization completed',
        
        # Data loading
        '正在加载数据': 'Loading data',
        '数据加载完成，共': 'Data loading completed, total',
        '篇文章': 'articles',
        
        # Keyword extraction
        'GPU优化版本': 'GPU optimized version',
        '批量预处理文章，提取名词': 'Batch preprocess articles, extract nouns',
        '预处理失败': 'Preprocessing failed',
        'GPU批处理关键字提取，充分利用RTX 5090': 'GPU batch keyword extraction, fully utilizing RTX 5090',
        '根据GPU显存动态调整batch_size - RTX 5090有32GB显存！': 'Dynamically adjust batch_size based on GPU memory - RTX 5090 has 32GB memory!',
        'RTX 5090的大显存可以处理更大的batch': 'RTX 5090 large memory can handle larger batch',
        'RTX 5090大显存模式：使用批处理大小': 'RTX 5090 large memory mode: using batch size',
        '使用GPU批处理，batch_size': 'Using GPU batch processing, batch_size',
        '总批次': 'total batches',
        '处理批次': 'Processing batch',
        '包含': 'containing',
        '篇文章': 'articles',
        '真正的GPU批处理关键字提取': 'Real GPU batch keyword extraction',
        'GPU批处理中... 显存使用情况': 'GPU batch processing... memory usage',
        '已用显存': 'GPU memory used',
        '缓存显存': 'GPU memory cached',
        '使用SentenceTransformer批量编码以提高GPU利用率': 'Use SentenceTransformer batch encoding to improve GPU utilization',
        '一次性处理整个batch': 'Process entire batch at once',
        '批量处理关键字提取': 'Batch process keyword extraction',
        '清理GPU内存': 'Clear GPU memory',
        '批次完成，GPU利用率得到优化': 'Batch completed, GPU utilization optimized',
        '批次处理失败': 'Batch processing failed',
        
        # Keyword processing
        '正在提取关键字': 'Extracting keywords',
        '使用GPU批处理优化，充分利用RTX 5090性能': 'Using GPU batch processing optimization, fully utilizing RTX 5090 performance',
        '批量预处理': 'Batch preprocessing',
        '预处理完成，有效文章数': 'Preprocessing completed, valid articles',
        'GPU批处理关键字提取': 'GPU batch keyword extraction',
        '重建完整结果列表': 'Rebuild complete result list',
        '正在统计关键字': 'Counting keywords',
        '关键字提取完成，共': 'Keyword extraction completed, total',
        '个关键字': 'keywords',
        '正在计算关键字嵌入': 'Computing keyword embeddings',
        '关键字嵌入计算完成': 'Keyword embedding calculation completed',
        
        # Dimensionality reduction
        '正在进行关键词降维': 'Performing keyword dimensionality reduction',
        '使用BertTopic进行关键词降维': 'Using BertTopic for keyword dimensionality reduction',
        '使用BertTopic进行降维': 'Using BertTopic for dimensionality reduction',
        '将关键词转换为文档格式（每个关键词作为一个文档）': 'Convert keywords to document format (each keyword as a document)',
        '直接使用字符串列表，不是嵌套列表': 'Use string list directly, not nested list',
        '使用BertTopic进行主题建模和降维': 'Use BertTopic for topic modeling and dimensionality reduction',
        '获取降维后的坐标 - 使用正确的API': 'Get dimensionality reduced coordinates - use correct API',
        '获取主题嵌入': 'Get topic embeddings',
        'BertTopic降维完成，主题数量': 'BertTopic dimensionality reduction completed, topic count',
        '将主题嵌入转换为2D坐标': 'Convert topic embeddings to 2D coordinates',
        '取前两个维度': 'Take first two dimensions',
        '如果主题数量少于关键词数量，需要扩展': 'If topic count is less than keyword count, expansion needed',
        '主题数量.*少于关键词数量.*，回退到t-SNE': 'Topic count is less than keyword count, falling back to t-SNE',
        'BertTopic主题数量不足': 'BertTopic topic count insufficient',
        'BertTopic降维失败，主题嵌入为空': 'BertTopic dimensionality reduction failed, topic embeddings empty',
        'BertTopic降维失败': 'BertTopic dimensionality reduction failed',
        '获取主题嵌入失败': 'Failed to get topic embeddings',
        'BertTopic降维出错': 'BertTopic dimensionality reduction error',
        '回退到t-SNE降维': 'Falling back to t-SNE dimensionality reduction',
        '使用t-SNE进行关键词降维': 'Using t-SNE for keyword dimensionality reduction',
        't-SNE降维完成': 't-SNE dimensionality reduction completed',
        
        # Clustering
        '正在进行层次聚类': 'Performing hierarchical clustering',
        '层次聚类完成': 'Hierarchical clustering completed',
        
        # UI elements
        '创建Dash布局 - 关键字二维降维可视化版本': 'Create Dash layout - keyword 2D dimensionality reduction visualization version',
        '顶部控制区域': 'Top control area',
        '手动添加关键字区域': 'Manual keyword addition area',
        '数据存储': 'Data storage',
        '存储文章数据': 'Store article data',
        '主要内容区域 - 左右分栏布局': 'Main content area - left-right column layout',
        '左侧：关键字二维降维可视化': 'Left: keyword 2D dimensionality reduction visualization',
        '右侧：分组管理': 'Right: group management',
        '推荐文件和详情区域 - 左右分栏布局': 'Recommended files and details area - left-right column layout',
        '左侧：推荐文件列表': 'Left: recommended files list',
        '推荐文件 (Recommended Files)': 'Recommended Files',
        '基于选中组的关键词，显示包含这些关键词的文件': 'Based on selected group keywords, show files containing these keywords',
        '右侧：文件详情和操作': 'Right: file details and actions',
        '文件详情 (File Details)': 'File Details',
        '点击左侧文件以查看详情': 'Click on left files to view details',
        '操作选项 (Actions):': 'Actions:',
        '查看文本 (View Text)': 'View Text',
        '训练按钮和输出': 'Training button and output',
        '训练结果可视化 - 两个并排的大图': 'Training result visualization - two side-by-side large plots',
        '文章内容显示区域': 'Article content display area',
        '调试输出区域': 'Debug output area',
        '设置应用布局': 'Set application layout',
        
        # Button and interaction logic
        '默认样式 - 包含完整的按钮样式': 'Default style - includes complete button styles',
        '处理关键字按钮点击 - 修复：只在真正点击关键字按钮时才修改数据': 'Handle keyword button clicks - fix: only modify data when actually clicking keyword buttons',
        '修复：实现累积添加逻辑 - 总是添加到选中的组，不进行切换': 'Fix: implement cumulative add logic - always add to selected group, no toggle',
        '修复：添加关键词时不自动选中它，保持显示整组文件': 'Fix: do not auto-select when adding keyword, keep showing all group files',
        '如果不是关键词按钮点击，不修改数据': 'If not keyword button click, do not modify data',
        '生成样式数组': 'Generate style array',
        
        # Debug and monitoring
        'update_keyword_styles 被调用': 'update_keyword_styles called',
        'update_group_order 被调用': 'update_group_order called',
        'handle_plot_click 被调用': 'handle_plot_click called',
        'render_groups 被调用': 'render_groups called',
        'select_group 被调用': 'select_group called',
        'remove_keyword_from_group 被调用': 'remove_keyword_from_group called',
        '点击的关键词': 'Clicked keyword',
        '当前 group_data': 'Current group_data',
        '选中的组': 'Selected group',
        '返回的 new_data': 'Returned new_data',
        '添加关键词但不自动选中，保持显示整组文件': 'Added keyword but not auto-selected, keep showing all group files',
        '切换到组': 'Switching to group',
        '清除选中关键词': 'Clear selected keyword',
        '从组管理中选择关键词': 'Select keyword from group management',
        '跳过删除操作：n_clicks为None，可能是自动触发': 'Skip delete operation: n_clicks is None, possibly auto-triggered',
        '真正的删除按钮被点击': 'Delete button actually clicked',
        '删除按钮被真正点击': 'Delete button actually clicked',
        
        # Visualization
        '更新关键字二维降维可视化图表': 'Update keyword 2D dimensionality reduction visualization chart',
        '获取关键字嵌入和降维结果': 'Get keyword embeddings and dimensionality reduction results',
        '使用之前计算的关键字嵌入和t-SNE结果': 'Use previously calculated keyword embeddings and t-SNE results',
        '重新计算t-SNE（或者使用之前的结果）': 'Recalculate t-SNE (or use previous results)',
        '为每个关键字分配颜色（基于聚类类别和分组状态）': 'Assign colors to each keyword (based on cluster category and grouping status)',
        '定义类别颜色（每个类别一个颜色）': 'Define category colors (one color per category)',
        '首先检查是否在用户分组中': 'First check if in user grouping',
        '已分组的关键字 - 使用用户分组的颜色': 'Grouped keywords - use user grouping colors',
        '绿色': 'green',
        '未分组的关键字 - 使用聚类类别颜色': 'Ungrouped keywords - use cluster category colors',
        '找到关键字在哪个聚类中': 'Find which cluster the keyword is in',
        '计算类别索引来获取颜色': 'Calculate category index to get color',
        '如果没找到类别，使用默认颜色': 'If no category found, use default color',
        '蓝色': 'blue',
        '创建散点图': 'Create scatter plot',
        '用于点击事件': 'For click events',
        
        # Group management
        '修复：同步group-data和group-order，确保所有分组的关键词都被正确保存': 'Fix: sync group-data and group-order, ensure all grouped keywords are correctly saved',
        '首先清除所有组的关键词列表': 'First clear all group keyword lists',
        '然后根据group_data重新填充': 'Then refill based on group_data',
        '检查是否是真正的按钮点击（防止自动触发）': 'Check if real button click (prevent auto-trigger)',
        '移除关键字的逻辑已经在remove_keyword_from_group中处理': 'Remove keyword logic already handled in remove_keyword_from_group',
        '这里只需要更新组顺序': 'Only need to update group order here',
        '修复：始终显示所有组的关键词，不依赖于selected_group': 'Fix: always show all group keywords, not dependent on selected_group',
        '检查该关键词是否被选中': 'Check if this keyword is selected',
        
        # File operations
        '根据选中关键词生成推荐文件列表，默认显示组内所有关键词对应的文件': 'Generate recommended file list based on selected keywords, default show files for all keywords in group',
        '请先选择关键词组': 'Please select keyword group first',
        '确定要搜索的关键词': 'Determine keywords to search',
        '如果选中了特定关键词，只搜索这个关键词': 'If specific keyword selected, only search this keyword',
        '包含关键词': 'containing keyword',
        '的文件': 'files',
        '搜索特定关键词': 'Search specific keyword',
        '如果没有选中特定关键词，搜索组内所有关键词': 'If no specific keyword selected, search all keywords in group',
        '包含组': 'containing group',
        '关键词的文件': 'keyword files',
        '搜索整组关键词': 'Search all group keywords',
        '中没有关键词': 'has no keywords',
        '使用全局的df变量来搜索文件': 'Use global df variable to search files',
        '数据未加载': 'Data not loaded',
        '搜索包含关键词的文件': 'Search files containing keywords',
        '文件编号从1开始': 'File numbering starts from 1',
        '检查文件是否包含任何关键词': 'Check if file contains any keywords',
        '提取文件的前5个关键词': 'Extract top 5 keywords from file',
        '没有找到': 'Not found',
        '创建文件列表头部': 'Create file list header',
        '共找到': 'Found total',
        '个文件': 'files',
        '创建关键词标签': 'Create keyword tags',
        '文件': 'File',
        '查看详情': 'View Details',
        '找到': 'Found',
        '个包含关键词的文件': 'files containing keywords',
        '生成推荐文件时出错': 'Error generating recommended files',
        
        # Training and model
        '立即转移到CPU以释放GPU内存': 'Immediately transfer to CPU to free GPU memory',
        '计算perplexity参数 - 确保perplexity < n_samples': 'Calculate perplexity parameter - ensure perplexity < n_samples',
        '样本数量太少': 'Too few samples',
        '跳过此组': 'skipping this group',
        'perplexity必须小于样本数量，并且至少为1': 'perplexity must be less than sample count and at least 1',
        't-SNE参数: 样本数': 't-SNE parameters: sample count',
        'perplexity': 'perplexity',
        '冻结 embedding 层': 'Freeze embedding layer',
        '冻结前8层 encoder': 'Freeze first 8 encoder layers',
        '收集最后4层 encoder 和 pooler 的参数': 'Collect parameters from last 4 encoder layers and pooler',
        'hard negative采样': 'hard negative sampling',
        '没有有效的三元组可训练': 'No valid triplets for training',
        '使用相对路径保存到项目的Keyword_Group目录': 'Use relative path to save to project Keyword_Group directory',
        '计算perplexity参数': 'Calculate perplexity parameters',
        '运行训练并获取两个图表': 'Run training and get two charts',
        '显示训练输出和内容显示区域': 'Show training output and content display area',
        
        # File selection and display
        '文件选择相关回调函数': 'File selection related callback functions',
        '选择文件以在右侧面板显示详情': 'Select file to show details in right panel',
        '如果点击了查看文件按钮': 'If view file button clicked',
        '设置选中的文件': 'Set selected file',
        '显示文件的文本内容': 'Display file text content',
        '获取文件内容': 'Get file content',
        '如果是刚选择文件，显示基本信息': 'If just selected file, show basic info',
        '已选择文件': 'Selected file',
        '文件内容预览': 'File content preview',
        '点击上方按钮查看完整文本': 'Click button above to view full text',
        '显示完整文本': 'Show full text',
        '的完整文本': 'complete text',
        '文件不存在': 'File does not exist',
        '显示文件详情时出错': 'Error displaying file details',
        
        # Debug monitoring
        '添加调试回调来监控数据状态': 'Add debug callback to monitor data state',
        'group-data 内容': 'group-data content',
        'group-order 内容': 'group-order content',
        
        # Function descriptions
        '提取文本的前N个关键词': 'Extract top N keywords from text',
        '使用KeyBERT提取关键词': 'Use KeyBERT to extract keywords',
        '只返回前top_k个关键词的文本部分': 'Only return text part of top_k keywords',
        '如果KeyBERT不可用，返回简单的单词分割': 'If KeyBERT not available, return simple word splitting',
        '过滤掉常见的停用词和短词': 'Filter out common stop words and short words',
        '关键词提取失败': 'Keyword extraction failed',
        '处理图表点击事件，将关键字添加到选中的组': 'Handle chart click events, add keywords to selected group',
        '获取点击的关键字': 'Get clicked keyword',
        '更新分组数据 - 修复：实现累积添加逻辑': 'Update grouping data - fix: implement cumulative add logic',
        '处理从组管理中选择关键词': 'Handle keyword selection from group management',
        '从组中移除关键字': 'Remove keyword from group',
        '获取被点击的按钮信息': 'Get clicked button info',
        '防护：只有当n_clicks有值时才执行删除操作': 'Guard: only execute delete when n_clicks has value',
        '从组顺序中移除关键字': 'Remove keyword from group order',
        '从分组数据中移除': 'Remove from grouping data',
        
        # Clear all remaining
        '确保关键字在全局作用域中可用': 'Ensure keywords are available in global scope',
        '确保关键字变量已定义': 'Ensure keyword variables are defined',
    }
    
    # Apply translations
    for chinese, english in translations.items():
        if '.*' in chinese:  # regex pattern
            content = re.sub(chinese, english, content)
        else:
            content = content.replace(chinese, english)
    
    # Remove any remaining Chinese characters (but be careful not to break comments)
    # This is a more conservative approach - only replace obvious patterns
    patterns_to_remove = [
        r'print\(f?"[^"]*[\u4e00-\u9fff][^"]*"\)',  # Chinese in print statements
        r'print\("[^"]*[\u4e00-\u9fff][^"]*"\)',    # Chinese in print statements
        r'#[^#\n]*[\u4e00-\u9fff][^#\n]*',          # Chinese in comments (but preserve structure)
        r'"""[^"]*[\u4e00-\u9fff][^"]*"""',         # Chinese in docstrings
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
    
    # Clean up extra spaces and newlines
    content = re.sub(r'\n\n\n+', '\n\n', content)
    content = re.sub(r'  +', ' ', content)
    
    with open('finaluimodified.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Final translation completed!")

if __name__ == '__main__':
    final_translate()
