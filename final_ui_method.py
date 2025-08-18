import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
from transformers import BertTokenizer, BertModel
import itertools
import warnings
warnings.filterwarnings('ignore')

# ===== 配置参数 =====
# 您已有的3个关键词组搜索后的文档ID
GROUP_IDS = {
    'Medical': [161, 139, 195, 117, 197, 184, 163, 141, 109, 99],
    'Politics': [262, 239, 295, 217, 297, 284, 264, 241, 209, 199],
    'Recreation': [361, 339, 395, 317, 380, 382, 363, 341, 309, 299]
}

# 训练参数 - 完全匹配Final UI
num_epochs = 50
margin_number = 3
learning_rate = 0.00001
batch_size = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TSNE参数
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
# =====================

def load_data_and_groups():
    """加载数据和分组"""
    print("=== Loading Data and Groups ===")
    
    # 读取CSV文件
    csv_path = "CSV/20newsgroups_cleaned.csv"
    df = pd.read_csv(csv_path, header=None)
    
    print(f"Total documents in dataset: {len(df)}")
    print("Group distribution:")
    for group_name, doc_ids in GROUP_IDS.items():
        print(f"  {group_name}: {len(doc_ids)} documents")
    
    return df

def extract_avg_cls_vector(text, model, tokenizer, max_length=512):
    """提取文本的CLS向量 - 完全匹配Final UI"""
    tokens = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    input_ids = tokens['input_ids'][0]
    chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
    cls_vectors = []
    
    for chunk in chunks:
        if len(chunk) < max_length:
            pad_length = max_length - len(chunk)
            chunk = torch.nn.functional.pad(chunk, (0, pad_length), value=tokenizer.pad_token_id)
        
        chunk = chunk.unsqueeze(0).to(DEVICE)
        attention_mask = (chunk != tokenizer.pad_token_id).long()
        outputs = model(chunk, attention_mask=attention_mask)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        cls_vectors.append(cls_vector)
    
    if len(cls_vectors) == 0:
        return torch.zeros(model.config.hidden_size, device=DEVICE)
    
    stacked_cls = torch.cat(cls_vectors, dim=0)
    avg_cls = torch.mean(stacked_cls, dim=0)
    return avg_cls

def get_group_cls_vectors(group_indices, df, model, tokenizer):
    """获取组内所有文档的CLS向量 - 完全匹配Final UI"""
    group_all_cls_vectors = []
    for idx in group_indices:
        if idx < len(df):
            text = df.iloc[idx, 1]
            if pd.notna(text):
                avg_cls = extract_avg_cls_vector(str(text), model, tokenizer)
                group_all_cls_vectors.append(avg_cls)
    return group_all_cls_vectors

def train_triplet_model(group_dict, df, model, tokenizer, triplet_loss_fn, optimizer, scheduler):
    """训练三元组模型 - 完全匹配Final UI的方法"""
    print("=== Training Triplet Model ===")
    
    group_names = list(group_dict.keys())
    anchor_groups = group_names
    
    # 生成训练对
    training_pairs = [
        (anchor_grp, negative_grp)
        for anchor_grp in anchor_groups
        for negative_grp in group_names
        if anchor_grp != negative_grp 
    ]
    
    pair_cycle = itertools.cycle(training_pairs)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0
        triplet_count = 0
        
        for anchor_group in group_names:
            anchor_indices = group_dict[anchor_group]
            anchor_cls_vectors = get_group_cls_vectors(anchor_indices, df, model, tokenizer)
            
            if len(anchor_cls_vectors) < 2:
                continue
            
            # 获取负样本
            negative_cls_vectors = []
            for negative_group in group_names:
                if negative_group == anchor_group:
                    continue
                negative_indices = group_dict[negative_group]
                negative_cls_vectors.extend(get_group_cls_vectors(negative_indices, df, model, tokenizer))
            
            if len(negative_cls_vectors) == 0:
                continue
            
            neg_tensor_all = torch.stack(negative_cls_vectors, dim=0)
            
            # 生成三元组
            for i in range(len(anchor_cls_vectors)):
                for j in range(len(anchor_cls_vectors)):
                    if i == j:
                        continue
                    
                    anchor = anchor_cls_vectors[i]
                    positive = anchor_cls_vectors[j]
                    
                    # Hard negative mining - 完全匹配Final UI
                    anchor_expanded = anchor.unsqueeze(0)
                    distances = torch.norm(anchor_expanded - neg_tensor_all, dim=1)
                    ap_dist = torch.norm(anchor - positive)
                    
                    # 选择困难负样本
                    mask = (distances > ap_dist) & (distances < ap_dist + margin_number)
                    valid_indices = torch.where(mask)[0]
                    
                    if len(valid_indices) == 0:
                        neg_idx = torch.argmin(distances)
                    else:
                        masked_dist = distances[valid_indices]
                        neg_idx = valid_indices[torch.argmin(masked_dist)]
                    
                    negative = neg_tensor_all[neg_idx]
                    
                    # 计算损失
                    loss = triplet_loss_fn(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0))
                    total_loss += loss
                    triplet_count += 1
        
        if triplet_count > 0:
            total_loss = total_loss / triplet_count
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss={total_loss.item():.4f}, Triplets={triplet_count}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: No valid triplets")
    
    return model

def create_tsne_visualization(before_embeddings, after_embeddings, group_dict, df):
    """创建TSNE可视化 - 完全匹配Final UI"""
    print("=== Creating TSNE Visualization ===")
    
    # 应用TSNE降维
    print("Applying TSNE dimensionality reduction...")
    tsne_before = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter_without_progress=TSNE_N_ITER, random_state=42)
    tsne_after = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter_without_progress=TSNE_N_ITER, random_state=42)
    
    projected_2d_before = tsne_before.fit_transform(before_embeddings)
    projected_2d_after = tsne_after.fit_transform(after_embeddings)
    
    # 计算每个组的中心点
    group_centers = {}
    colors = ['red', 'blue', 'green']
    
    for i, (group_name, indices) in enumerate(group_dict.items()):
        if len(indices) > 0:
            # 找到该组文档在embeddings中的索引
            group_emb_indices = []
            all_doc_ids = []
            for group_indices in group_dict.values():
                all_doc_ids.extend(group_indices)
            
            for doc_id in indices:
                if doc_id in all_doc_ids:
                    emb_idx = all_doc_ids.index(doc_id)
                    if emb_idx < len(after_embeddings):
                        group_emb_indices.append(emb_idx)
            
            if len(group_emb_indices) > 0:
                # 计算该组的2D中心点
                group_2d_points = projected_2d_after[group_emb_indices]
                group_center_2d = np.mean(group_2d_points, axis=0)
                group_centers[group_name] = group_center_2d
                print(f"Group {group_name} center: {group_center_2d}")
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 为每个文档分配颜色（基于其所属组）
    doc_colors = []
    all_doc_ids = []
    for group_indices in group_dict.values():
        all_doc_ids.extend(group_indices)
    
    for doc_id in all_doc_ids:
        if doc_id in GROUP_IDS['Medical']:
            doc_colors.append('red')
        elif doc_id in GROUP_IDS['Politics']:
            doc_colors.append('blue')
        elif doc_id in GROUP_IDS['Recreation']:
            doc_colors.append('green')
    
    # Before finetuning
    axes[0].scatter(projected_2d_before[:, 0], projected_2d_before[:, 1], 
                   c=doc_colors, alpha=0.7, s=100)
    axes[0].set_title('2D Projection Before Triplet Training', fontsize=16)
    axes[0].set_xlabel('TSNE 1', fontsize=12)
    axes[0].set_ylabel('TSNE 2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Medical'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Politics'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Recreation')
    ]
    axes[0].legend(handles=legend_elements)
    
    # After finetuning
    axes[1].scatter(projected_2d_after[:, 0], projected_2d_after[:, 1], 
                   c=doc_colors, alpha=0.7, s=100)
    
    # 添加组中心点
    center_colors = ['darkred', 'darkblue', 'darkgreen']
    for i, (group_name, center_2d) in enumerate(group_centers.items()):
        color = center_colors[i % len(center_colors)]
        axes[1].scatter(center_2d[0], center_2d[1], 
                       color=color, s=300, marker='*', 
                       label=f'Center: {group_name}', edgecolors='white', linewidth=2)
        # 添加组名标签
        axes[1].annotate(group_name, (center_2d[0], center_2d[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold', color=color)
    
    axes[1].set_title('2D Projection After Triplet Training', fontsize=16)
    axes[1].set_xlabel('TSNE 1', fontsize=12)
    axes[1].set_ylabel('TSNE 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('CSV/final_ui_method_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存详细结果
    results = {
        'doc_ids': all_doc_ids,
        'group_assignments': GROUP_IDS,
        'tsne_coordinates_before': projected_2d_before.tolist(),
        'tsne_coordinates_after': projected_2d_after.tolist(),
        'group_centers': {k: v.tolist() for k, v in group_centers.items()}
    }
    
    with open('CSV/final_ui_method_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to CSV/final_ui_method_results.json")
    print("Visualization saved to CSV/final_ui_method_visualization.png")

def main():
    print("=== Final UI Method: Training with Given Document Groups ===")
    
    # 1. 加载数据
    df = load_data_and_groups()
    
    # 2. 初始化BERT模型和tokenizer
    print("=== Initializing BERT Model ===")
    model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 3. 获取训练前的嵌入
    print("=== Computing Initial Embeddings ===")
    initial_embeddings = []
    all_doc_ids = []
    
    for group_name, indices in GROUP_IDS.items():
        for doc_id in indices:
            if doc_id < len(df):
                text = df.iloc[doc_id, 1]
                if pd.notna(text):
                    cls_vector = extract_avg_cls_vector(str(text), model, tokenizer)
                    initial_embeddings.append(cls_vector.detach().cpu().numpy())
                    all_doc_ids.append(doc_id)
    
    initial_embeddings = np.array(initial_embeddings)
    print(f"Initial embeddings shape: {initial_embeddings.shape}")
    print(f"Total documents processed: {len(all_doc_ids)}")
    
    # 4. 初始化训练组件 - 完全匹配Final UI
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin_number, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {DEVICE}")
    print(f"Learning rate: {learning_rate}")
    print(f"Margin: {margin_number}")
    
    # 5. 训练模型
    trained_model = train_triplet_model(GROUP_IDS, df, model, tokenizer, triplet_loss_fn, optimizer, scheduler)
    
    # 6. 获取训练后的嵌入
    print("=== Computing Final Embeddings ===")
    final_embeddings = []
    
    for group_name, indices in GROUP_IDS.items():
        for doc_id in indices:
            if doc_id < len(df):
                text = df.iloc[doc_id, 1]
                if pd.notna(text):
                    cls_vector = extract_avg_cls_vector(str(text), trained_model, tokenizer)
                    final_embeddings.append(cls_vector.detach().cpu().numpy())
    
    final_embeddings = np.array(final_embeddings)
    print(f"Final embeddings shape: {final_embeddings.shape}")
    
    # 7. 创建TSNE可视化
    create_tsne_visualization(initial_embeddings, final_embeddings, GROUP_IDS, df)
    
    # 8. 保存模型
    torch.save(trained_model.state_dict(), 'CSV/final_ui_method_model.pth')
    print("Model saved to CSV/final_ui_method_model.pth")
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    main() 