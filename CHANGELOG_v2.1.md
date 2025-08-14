# KeySI v2.1 变更日志

## 🎉 版本概述
KeySI v2.1 是一个重要的功能更新版本，主要集成了BertTopic降维方法，简化了用户界面，并优化了用户体验。

## ✨ 新功能

### 1. BertTopic智能降维
- **新增**: 集成BertTopic进行关键词降维
- **优势**: 基于主题建模的语义降维，比t-SNE更智能
- **自动回退**: 如果BertTopic不可用，自动使用t-SNE备选方案
- **配置**: 支持自动主题数量检测和参数优化

### 2. 界面简化
- **移除**: 上下移动按钮（↑↓）
- **保留**: 删除按钮（×）
- **效果**: 更简洁的分组管理界面

### 3. 关键词选择逻辑优化
- **确认**: 点击2D图表中的关键词会添加到选中组，而不是替换
- **保持**: 原有的添加逻辑完全保留

## 🔧 技术改进

### 1. 降维流程优化
```python
# 新的智能降维流程
if BERTOPIC_AVAILABLE:
    # 使用BertTopic进行主题建模降维
    topic_model = BERTopic(nr_topics="auto", min_topic_size=2)
    topics, probs = topic_model.fit_transform(keyword_docs)
    reduced_embeddings = topic_model.get_representative_docs()
else:
    # 回退到t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(keyword_embeddings)
```

### 2. 回调函数简化
- 移除了`update_group_order`中对上下移动按钮的处理
- 简化了`render_groups`中的按钮布局
- 保持了删除功能的完整性

### 3. 错误处理增强
- BertTopic失败时的优雅回退
- 详细的错误日志和用户提示

## 📁 新增文件

### 1. `requirements_berttopic.txt`
- BertTopic和相关依赖的版本要求
- 包含UMAP和HDBSCAN等必要库

### 2. `test_berttopic.py`
- BertTopic功能的独立测试脚本
- 验证降维功能和备选方案

### 3. `install_berttopic.py`
- 自动化安装助手脚本
- 检查Python版本和依赖安装

### 4. `test_ui_simple.py`
- 简化的UI功能测试
- 验证基本组件和布局

### 5. `requirements_complete.txt`
- 完整的依赖列表
- 由安装脚本自动生成

## 🚀 安装说明

### 方法1: 使用安装脚本（推荐）
```bash
python install_berttopic.py
```

### 方法2: 手动安装
```bash
pip install -r requirements_berttopic.txt
```

### 方法3: 单独安装
```bash
pip install bertopic umap-learn hdbscan
```

## 🧪 测试验证

### 1. 测试BertTopic功能
```bash
python test_berttopic.py
```

### 2. 测试UI功能
```bash
python test_ui_simple.py
```

### 3. 运行主应用
```bash
python finaluimodified.py
```

## 📊 性能影响

### 1. 内存使用
- BertTopic: 中等内存使用，适合中等规模数据集
- t-SNE: 较低内存使用，适合大规模数据集

### 2. 处理速度
- BertTopic: 首次运行较慢（需要训练），后续快速
- t-SNE: 相对稳定，速度适中

### 3. 降维质量
- BertTopic: 更好的语义保持，主题聚类效果
- t-SNE: 良好的局部结构保持

## 🔍 故障排除

### 1. BertTopic安装失败
```bash
# 检查Python版本
python --version

# 升级pip
python -m pip install --upgrade pip

# 安装依赖
pip install umap-learn hdbscan
pip install bertopic
```

### 2. 内存不足
- 系统会自动回退到t-SNE
- 或调整BertTopic的`min_topic_size`参数

### 3. 性能问题
- 对于大数据集，建议使用t-SNE
- 对于语义分析，推荐使用BertTopic

## 🔮 未来计划

### v2.2 计划功能
- [ ] 支持更多降维算法（UMAP, PCA等）
- [ ] 添加降维方法选择器
- [ ] 支持自定义降维参数
- [ ] 添加降维质量评估指标

### v2.3 计划功能
- [ ] 支持实时降维更新
- [ ] 添加降维结果导出功能
- [ ] 支持批量关键词处理
- [ ] 添加降维可视化配置

## 📝 兼容性说明

### Python版本
- **最低要求**: Python 3.8
- **推荐版本**: Python 3.9+
- **测试版本**: Python 3.11.8

### 操作系统
- **Windows**: ✅ 完全支持
- **macOS**: ✅ 完全支持
- **Linux**: ✅ 完全支持

### 依赖库
- **必需**: Dash, Plotly, NumPy, Pandas, scikit-learn
- **可选**: BertTopic, UMAP, HDBSCAN
- **GPU**: PyTorch (CUDA/MPS支持)

## 🙏 致谢

感谢以下开源项目的贡献：
- **BertTopic**: 提供先进的主题建模降维
- **UMAP**: 高效的流形学习算法
- **HDBSCAN**: 层次密度聚类算法
- **Dash**: 优秀的Python Web应用框架

---

**版本**: v2.1  
**发布日期**: 2024年12月  
**维护者**: KeySI开发团队
