
import os
import re 
import json
import json as json_module
import random
import io
import base64
import math
import gc
import multiprocessing
import socket
import itertools
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
import traceback
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from rank_bm25 import BM25Okapi
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel, AutoTokenizer
from rapidfuzz import fuzz

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score,
    pairwise_distances,
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    adjusted_rand_score, 
    normalized_mutual_info_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
from joblib import Parallel, delayed


OUTPUT_DIR = "KeySI_results"    

training_in_progress = False

FILE_PATHS = {
    # data file
    "csv_path": "CSV/20news_top30_per_class.csv", 
    "final_list_path": f"{OUTPUT_DIR}/final_list.json",
    
    # result
    "bm25_search_results": f"{OUTPUT_DIR}/bm25_search_results.json",
    "filtered_group_assignment": f"{OUTPUT_DIR}/filtered_group_assignment.json", 
    "group_assignment": f"{OUTPUT_DIR}/group_assignment.json",
    "triplet_trained_encoder": f"{OUTPUT_DIR}/triplet_trained_encoder.pth",
    "gap_based_filter_results": f"{OUTPUT_DIR}/gap_based_filter_results.json",
    "clustering_evaluation": f"{OUTPUT_DIR}/clustering_evaluation.json",
    "triplet_run_stats": f"{OUTPUT_DIR}/triplet_run_stats.json",
    "training_group_info": f"{OUTPUT_DIR}/training_group_info.json",
    "embeddings_trained": f"{OUTPUT_DIR}/embeddings_trained.npy",
    "triplet_training_comparison": f"{OUTPUT_DIR}/triplet_training_comparison.png",
    "triplet_tsne_comparison": f"{OUTPUT_DIR}/triplet_tsne_comparison.json",
    "user_finetuned_list": f"{OUTPUT_DIR}/user_finetuned_list.json",
    "bert_finetuned": f"{OUTPUT_DIR}/bert_finetuned.pth",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


TRAINING_CONFIG = {
    "bert_name": "bert-base-uncased",
    "proj_dim": 256,
    "freeze_layers": 6,
    
    "triplet_epochs": 1,        
    "triplet_batch_size": 16,
    "triplet_margin": 1.2,
    "triplet_lr": 2e-5,
    
    "proto_epochs": 1,       
    "proto_batch_size": 64,
    "proto_lr": 2e-5,
    

    "gap_alpha": 0.5,            
    "gap_min_samples": 20,
    "gap_percentile_fallback": 20,
    "gap_floor_threshold": 0.05,
    "gap_mix_ratio": 0.3,
    
    
    "encoding_batch_size": 64,
    "max_length": 256,
    
  
    "tsne_perplexity": 30,
    "tsne_max_iter": 500,
    

    "min_pos_per_group": 2,
    "num_pos_per_anchor": 2,
    "num_neg_per_anchor": 3,
    "min_per_group_prototype": 5,
    "ema_alpha": 0.1,           
}

def get_config(key, default=None):

    return TRAINING_CONFIG.get(key, default)


class SentenceEncoder(nn.Module):

    
    def __init__(self, bert_name=None, proj_dim=None, device='cpu'):
        if bert_name is None:
            bert_name = get_config("bert_name")
        if proj_dim is None:
            proj_dim = get_config("proj_dim")
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_name)
        self.hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(self.hidden, proj_dim) if proj_dim else None
        self.out_dim = proj_dim or self.hidden
        self.ln = nn.LayerNorm(self.out_dim)

        if device != 'cpu':
            self.to(device)

    def encode_tokens(self, tokens):
        out = self.bert(**tokens).last_hidden_state  # (B,L,H)
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        if self.proj is not None: pooled = self.proj(pooled)
        if pooled.dim()==1: pooled = pooled.unsqueeze(0)
        pooled = self.ln(pooled)
        return nn.functional.normalize(pooled, p=2, dim=-1)


def ensure_nltk_data():

    required_packages = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab', 
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng'
    }
    
    for package_name, package_path in required_packages.items():
        try:
            nltk.data.find(package_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception as e:
               
                if package_name == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
                        

    additional_packages = ['stopwords', 'wordnet', 'omw-1.4']
    for package in additional_packages:
        try:
            nltk.download(package, quiet=True)
        except:
            pass  


ensure_nltk_data()






if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


if device == "cuda":

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def clear_gpu_memory():

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

if current_dir.endswith("CSV"):
    csv_dir = current_dir
    print("Already in CSV directory")
elif os.path.exists("CSV"):
    csv_dir = os.path.join(current_dir, "CSV")
    print(f"CSV directory found at: {csv_dir}")
else:
    print("CSV folder does not exist, please check file path")
    print("Please ensure you run this script from the project root directory")
    exit(1)
num_threads=8
top_similiar_file_to_keywords=500
learningrate=1e-4

margin_number = 3
top_keywords=3
early_stop_threshold = 8
mostfequentwords=250
cluster_distance=30
word_count_feq=3
max_d = 30
word_count_threshold= 3
top_similar_files = 3000


if device == "cuda":
    batch_size = 256  
elif device == "mps":
    batch_size = 128  
else:
    batch_size = 64   

clusterthreshold = 25


GROUP_COLORS = {
    "Group 1": "#FF6B6B",  # Red
    "Group 2": "#32CD32",  # Lime Green  
    "Group 3": "#FF8C00",  # Dark Orange
    "Group 4": "#8B4513",  # Saddle Brown
    "Group 5": "#FFD700",  # Gold
    "Group 6": "#8A2BE2",  # Blue Violet
    "Group 7": "#DC143C",  # Crimson
    "Group 8": "#228B22",  # Forest Green
    "Group 9": "#FF1493",  # Deep Pink
    "Group 10": "#800080", # Purple
    "Other": "#A9A9A9",    # Dark Gray for exclusion group
}

def get_group_color(group_name):

    return GROUP_COLORS.get(group_name, "#808080")  


PLOT_STYLES = {

    "background": {
        "color": "#1f77b4",  
        "size": 8,
        "opacity": 0.8,
        "line_width": 0.5,
        "line_color": "white"
    },
    
    "core": {
        "color": "#FFD700",  
        "size": 14,
        "opacity": 1.0,
        "symbol": "star",
        "line_width": 2,
        "line_color": "white"
    },
  
    "gray": {
        "color": "#808080",  
        "size": 12,  
        "opacity": 0.8,
        "symbol": "triangle-up",  
        "line_width": 1.5,
        "line_color": "white"
    },
 
    "center": {
        "size": 20,
        "opacity": 1.0,
        "symbol": "diamond",
        "line_width": 3,
        "line_color": "white"
    },
  
    "layout": {
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "xaxis": {
            "showgrid": False,  
            "zeroline": False,  
            "showline": True,   
            "linecolor": "black",
            "linewidth": 2,
            "mirror": True      
        },
        "yaxis": {
            "showgrid": False,  
            "zeroline": False,  
            "showline": True,   
            "linecolor": "black",
            "linewidth": 2,
            "mirror": True     
        }
    }
}




def ensure_directories(): 
    directories = [
        OUTPUT_DIR,
        "Keyword_Group",
        "CSV"
    ]
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


ensure_directories()


try:
    embedding_model_kw = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
except Exception as e:
    print(f"Model initialization failed: {e}")
    raise


_GLOBAL_DOCUMENT_EMBEDDINGS = None
_GLOBAL_DOCUMENT_TSNE = None
_GLOBAL_DOCUMENT_EMBEDDINGS_READY = False

def safe_encode_batch(batch_texts, model, device, fallback_dim=768):
    try:
        for i, text in enumerate(batch_texts):
            if len(text) > 1000:  
                print(f"    Warning: Text {i} is very long ({len(text)} chars), truncating further")
                batch_texts[i] = truncate_text_for_model(text, max_length=400)
        
        print(f"    Encoding batch of {len(batch_texts)} texts...")
        embeddings = model.encode(batch_texts, convert_to_tensor=True).to(device).cpu().numpy()
        print(f"    Successfully encoded batch, shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"    Error encoding batch: {e}")
        print(f"    Using fallback zero vectors for batch")
        
        batch_size = len(batch_texts)
        fallback_embeddings = np.zeros((batch_size, fallback_dim))
        return fallback_embeddings

def precompute_document_embeddings():
   
    global _GLOBAL_DOCUMENT_EMBEDDINGS, _GLOBAL_DOCUMENT_TSNE, _GLOBAL_DOCUMENT_EMBEDDINGS_READY, df
    
    if _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        print("    Document embeddings already pre-computed, skipping...")
        return
    
    print("    Pre-computing document embeddings for faster response...")
    
    try:
  
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        print(f"    Processing {len(df)} documents...")
        

        all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
        print(f"    Number of articles: {len(all_articles_text)}")
        

        print("    Truncating long texts...")
        truncated_articles = [truncate_text_for_model(text, max_length=500) for text in all_articles_text]
        
 
        batch_size = 64 if device == "cpu" else 128
        all_embeddings = []
        
        print(f"    Computing embeddings with batch size {batch_size}...")
        for i in range(0, len(truncated_articles), batch_size):
            batch_texts = truncated_articles[i:i + batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(truncated_articles) + batch_size - 1)//batch_size}")
            
            batch_embeddings = safe_encode_batch(batch_texts, embedding_model_kw, device)
            all_embeddings.extend(batch_embeddings)
        
        _GLOBAL_DOCUMENT_EMBEDDINGS = np.array(all_embeddings)
        print(f"    Document embeddings computed, shape: {_GLOBAL_DOCUMENT_EMBEDDINGS.shape}")
        
        
        print(f"    Embeddings length: {len(_GLOBAL_DOCUMENT_EMBEDDINGS)}")
        print(f"    df length: {len(df)}")
        

        if len(_GLOBAL_DOCUMENT_EMBEDDINGS) != len(df):
            print(f"    WARNING: Pre-compute length mismatch! Adjusting...")
           
            if len(_GLOBAL_DOCUMENT_EMBEDDINGS) < len(df):
                padding_needed = len(df) - len(_GLOBAL_DOCUMENT_EMBEDDINGS)
                print(f"    Padding embeddings with {padding_needed} zero vectors")
                padding_vectors = np.zeros((padding_needed, _GLOBAL_DOCUMENT_EMBEDDINGS.shape[1]))
                _GLOBAL_DOCUMENT_EMBEDDINGS = np.vstack([_GLOBAL_DOCUMENT_EMBEDDINGS, padding_vectors])

            elif len(_GLOBAL_DOCUMENT_EMBEDDINGS) > len(df):
                print(f"    Truncating embeddings from {len(_GLOBAL_DOCUMENT_EMBEDDINGS)} to {len(df)}")
                _GLOBAL_DOCUMENT_EMBEDDINGS = _GLOBAL_DOCUMENT_EMBEDDINGS[:len(df)]
        
        print(f"    Final embeddings shape: {_GLOBAL_DOCUMENT_EMBEDDINGS.shape}")
        
     
        print("    Computing t-SNE for documents...")
        perplexity = min(30, max(5, len(_GLOBAL_DOCUMENT_EMBEDDINGS) // 3))
        print(f"    Using perplexity: {perplexity}")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        _GLOBAL_DOCUMENT_TSNE = tsne.fit_transform(_GLOBAL_DOCUMENT_EMBEDDINGS)
        print(f"    t-SNE computed, shape: {_GLOBAL_DOCUMENT_TSNE.shape}")
        
        _GLOBAL_DOCUMENT_EMBEDDINGS_READY = True
        print("    Document embeddings and t-SNE pre-computation completed!")
        
        
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"    Error pre-computing document embeddings: {e}")
        _GLOBAL_DOCUMENT_EMBEDDINGS_READY = False

def get_document_embeddings():
    
    global _GLOBAL_DOCUMENT_EMBEDDINGS, _GLOBAL_DOCUMENT_EMBEDDINGS_READY
    
    if not _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        precompute_document_embeddings()
    
    return _GLOBAL_DOCUMENT_EMBEDDINGS

def get_document_tsne():
    
    global _GLOBAL_DOCUMENT_TSNE, _GLOBAL_DOCUMENT_EMBEDDINGS_READY
    
    if not _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        precompute_document_embeddings()
    
    return _GLOBAL_DOCUMENT_TSNE

def truncate_text_for_model(text, max_length=500):
 
    if not text or len(text) <= max_length:
        return text
    
  
    truncated = text[:max_length]
    

    if ' ' in truncated:
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  
            truncated = truncated[:last_space]
    
    return truncated + "..." if len(truncated) < len(text) else truncated



try:
    precompute_document_embeddings()
except Exception as e:
    pass
        


kw_model = KeyBERT(model=embedding_model_kw)

ps = PorterStemmer()
word_count = Counter()
original_form = {}
df = pd.read_csv(FILE_PATHS["csv_path"])
all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
labels = df.iloc[:, 0].values


def preprocess_articles_batch(articles):

    processed_articles = []
    valid_indices = []
    
    for i, article in enumerate(articles):
        try:
            article = re.sub(r'\d+', '', article).strip()
            if len(article.split()) < 5:
                continue
            words = word_tokenize(article)
            tagged_words = pos_tag(words)
            nouns = [word for word, pos in tagged_words if pos.startswith("NN")]
            if nouns:
                processed_articles.append(" ".join(nouns))
                valid_indices.append(i)
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            continue
    
    return processed_articles, valid_indices

def extract_keywords_batch_gpu(articles, batch_size=None):

    if batch_size is None:
    
        if device == "cuda":
            batch_size = 128  
          
        else:
            batch_size = 32
    
    results = []
    total_batches = (len(articles) + batch_size - 1) // batch_size
    
  
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        
        try:
            with torch.no_grad():
                batch_embeddings = embedding_model_kw.encode(
                    batch, 
                    batch_size=len(batch),  
                    convert_to_tensor=True,
                    device=device
                )
            

            batch_results = []
            for i, article in enumerate(batch):
                keywords_info = kw_model.extract_keywords(
                    article, 
                    keyphrase_ngram_range=(1, 1), 
                    stop_words='english', 
                    top_n=8
                )
                result = [(ps.stem(kw), kw) for kw, _ in keywords_info]
                batch_results.append(result if result else None)
            
            results.extend(batch_results)
            
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  Batch processing failed: {e}")
            results.extend([None] * len(batch))
    
    return results

processed_articles, valid_indices = preprocess_articles_batch(all_articles_text)

if processed_articles:
    batch_results = extract_keywords_batch_gpu(processed_articles, batch_size=128)
    
    results = [None] * len(all_articles_text)
    for i, result in enumerate(batch_results):
        if i < len(valid_indices):
            results[valid_indices[i]] = result
else:
    results = [None] * len(all_articles_text)
for res in results:
    if res:
        for stemmed, kw in res:
            word_count[stemmed] += 1
            if stemmed not in original_form or len(kw) < len(original_form[stemmed]):
                original_form[stemmed] = kw

filtered_keywords = [original_form[stem] for stem, count in word_count.items() if count >= word_count_threshold]
if not filtered_keywords:
    raise ValueError("No keywords found with the specified frequency threshold.")

keyword_embeddings = embedding_model_kw.encode(filtered_keywords, convert_to_tensor=True).to(device).cpu().numpy()

perplexity = min(30, max(5, len(keyword_embeddings) // 3))
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
reduced_embeddings = tsne.fit_transform(keyword_embeddings)

linkage_matrix = linkage(reduced_embeddings, method="ward")
labels_hierarchical = fcluster(linkage_matrix, max_d, criterion="distance")

clustered_keywords = {}

for word, label in zip(filtered_keywords, labels_hierarchical):
    clustered_keywords.setdefault(label, []).append(word)
best_k = len(set(labels_hierarchical))
output_dict = {f"cluster{cluster}": clustered_keywords[cluster] for cluster in sorted(clustered_keywords.keys())}
keywords = [kw for cluster in output_dict.values() for kw in cluster]
cluster_names = list(output_dict.keys())
total_clusters = len(cluster_names)
GLOBAL_OUTPUT_DICT = output_dict
GLOBAL_KEYWORDS = keywords  

if 'keywords' not in locals():
    keywords = []
    print("Warning: Keyword list is empty, using default values")

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .training-button {
                animation: pulse 1.5s infinite !important;
            }
            
            /* Article item click animation */
            .article-item-button {
                transition: all 0.2s ease !important;
            }
            
            .article-item-button:active {
                transform: scale(0.95) !important;
                background-color: #e3f2fd !important;
                border-color: #2196F3 !important;
                box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3) !important;
            }
            
            .article-item-button:hover {
                transform: scale(1.02) !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            }
            
            /* Finetune documents list hover effect */
            .finetune-doc-card {
                transition: all 0.3s ease !important;
            }
            
            .finetune-doc-card:hover {
                transform: translateY(-4px) !important;
                box-shadow: 0 6px 16px rgba(0,0,0,0.12) !important;
            }
            
            .finetune-doc-card:active {
                transform: translateY(-2px) !important;
            }
            
            /* Button hover effects */
            button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
            }
            
            /* Input focus effects */
            input:focus {
                outline: none !important;
                border-color: #3498db !important;
                box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1) !important;
            }
            
            /* Modern card styling */
            .modern-card {
                background: white !important;
                border-radius: 10px !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
                border: 1px solid #e9ecef !important;
                transition: all 0.3s ease !important;
            }
            
            .modern-card:hover {
                box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
                transform: translateY(-2px) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def create_layout():
    return html.Div([
        html.Div([
            html.H1("KeySI System", style={
                "textAlign": "center",
                "color": "#2c3e50",
                "fontSize": "2.5rem",
                "fontWeight": "bold",
                "marginBottom": "10px",
                "textShadow": "2px 2px 4px rgba(0,0,0,0.1)"
            }),
            html.P(" Keyword System", style={
                "textAlign": "center",
                "color": "#7f8c8d",
                "fontSize": "1.1rem",
                "marginBottom": "30px",
                "fontStyle": "italic"
            })
        ], style={
            "backgroundColor": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "padding": "20px",
            "borderRadius": "10px",
            "marginBottom": "30px",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.1)"
        }),
        
        
        html.Div([
        
            html.Div([
                html.Label("Number of Groups:", style={
                    "fontWeight": "bold",
                    "color": "#2c3e50",
                    "marginRight": "10px",
                    "fontSize": "1rem"
                }),
                dcc.Input(
                    id="group-count", 
                    type="number", 
                    value=3, 
                    min=1, 
                    step=1,
                    style={
                        "width": "80px",
                        "padding": "8px 12px",
                        "border": "2px solid #e0e0e0",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "marginRight": "15px"
                    }
                ),
                html.Button(
                    "Generate Groups", 
                    id="generate-btn", 
                    n_clicks=0,
                    style={
                        "backgroundColor": "#3498db",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "transition": "all 0.3s ease",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                        "marginRight": "10px",
                        "minWidth": "140px",
                        "flexShrink": "0"
                    }
                ),
                html.Button("Switch to Training View", id="switch-view-btn", n_clicks=0, style={
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "fontSize": "1rem",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "marginRight": "10px",
                    "minWidth": "180px",
                    "flexShrink": "0",
                    "display": "none"
                }),
                html.Button("Train Model", id="train-btn", n_clicks=0, style={
                    "backgroundColor": "#e74c3c",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "fontSize": "1rem",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "minWidth": "120px",
                    "flexShrink": "0"
                }),
                html.Button("Switch to Finetune Mode", id="switch-finetune-btn", n_clicks=0, style={
                    "backgroundColor": "#8e44ad",
                    "color": "white",
                    "border": "none",
                    "padding": "10px 20px",
                    "borderRadius": "6px",
                    "fontSize": "1rem",
                    "fontWeight": "bold",
                    "cursor": "pointer",
                    "transition": "all 0.3s ease",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "minWidth": "180px",
                    "flexShrink": "0",
                    "display": "none"
                })
            ], style={
                "display": "flex",
                "alignItems": "center",
                "padding": "15px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "8px",
                "border": "1px solid #e9ecef",
                "width": "100%",
                "marginBottom": "20px",
                "flexWrap": "nowrap",
                "justifyContent": "flex-start"
            }),
            
           
            html.Div([
                html.Label("Add Custom Keyword:", style={
                    "fontWeight": "bold",
                    "color": "#2c3e50",
                    "marginRight": "10px",
                    "fontSize": "1rem"
                }),
                dcc.Input(
                    id='new-keyword-input',
                    type='text',
                    placeholder='Enter keywords (use comma to separate multiple)...',
                    style={
                        "flex": "1",
                        "padding": "8px 12px",
                        "border": "2px solid #e0e0e0",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "marginRight": "15px"
                    }
                ),
                html.Button(
                    "Add Keyword", 
                    id="add-keyword-btn", 
                    n_clicks=0,
                    style={
                        "backgroundColor": "#27ae60",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "6px",
                        "fontSize": "1rem",
                        "fontWeight": "bold",
                        "cursor": "pointer",
                        "transition": "all 0.3s ease",
                        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
                    }
                )
            ], style={
                "display": "flex",
                "alignItems": "center",
                "padding": "15px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "8px",
                "border": "1px solid #e9ecef",
                "width": "48%",
                "marginLeft": "2%"
            })
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "marginBottom": "30px"
        }),

      
        dcc.Store(id="group-data", data={kw: None for kw in (keywords if 'keywords' in globals() else [])}),
        dcc.Store(id="selected-group", data=None),
        dcc.Store(id="group-order", data={}),
        dcc.Store(id="selected-file", data=None),
        dcc.Store(id="selected-keyword", data=None),
        dcc.Store(id="selected-article", data=None),  # Store selected article index for highlighting
        dcc.Store(id="articles-data", data=[]),  # Store article data
        dcc.Store(id="document-embeddings", data=None),  # Store document embeddings for 2D visualization
        dcc.Store(id="training-status", data={"is_training": False, "status": "idle"}),  # Training status
        dcc.Store(id="display-mode", data="keywords"),  # Display mode: "keywords", "training", or "finetune"
        dcc.Store(id="training-figures", data={"before": None, "after": None}),  # Store training figures
        dcc.Store(id="highlighted-indices", data=[]),  # Store highlighted article indices
        dcc.Store(id="keyword-highlights", data=[]),  # New: store keyword highlights separately
        dcc.Store(id="training-selected-group", data=None),  # Training mode group selection
        dcc.Store(id="training-selected-keyword", data=None),  # Training mode keyword selection
        dcc.Store(id="training-selected-article", data=None),  # Training mode article selection
        # Finetune mode stores
        dcc.Store(id="finetune-figures", data=None),  # Store finetune-specific figures (updated after finetune training)
        dcc.Store(id="finetune-selected-group", data=None),
        dcc.Store(id="finetune-selected-sample", data=None),
        dcc.Store(id="finetune-selected-keyword", data=None),  
        dcc.Store(id="finetune-selected-article-index", data=None),  
        dcc.Store(id="finetune-highlight-core", data=[]),
        dcc.Store(id="finetune-highlight-gray", data=[]),
        dcc.Store(id="finetune-temp-assignments", data={}),
        
        html.Div(id="main-visualization-area", children=[
           
            html.Div([
                html.H4("Keywords 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Click on keywords to view related documents", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='keywords-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '1%'
            }),
            
           
            html.Div([
                html.H4("Documents 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Documents highlighted by selected keyword", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='documents-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '1%'
            })
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
        
        html.Div([
            html.Div([
                html.H4("Group Management", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="group-containers", style={
                    "display": "flex",
                    "flex-direction": "column",
                    "gap": "15px",
                    "margin-bottom": "20px"
                }),
            ], className="modern-card", style={
                'width': '25%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '15px'
            }),
            
        
            html.Div([
                html.H4("Recommended Articles", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="articles-container", style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '40%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'margin': '0 7px'
            }),
            
            
            html.Div([
                html.H4("Article Full Text", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="article-fulltext-container", children=[
                    html.P("Click on an article from the middle panel to view its full content", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '30%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '7px'
            })
        ], id="keywords-group-management-area", style={'display': 'flex', 'marginBottom': '30px', 'height': '600px', 'overflowY': 'auto'}),
        
        
        html.Div(id="training-group-management-area", style={'display': 'none', 'marginBottom': '30px', 'height': '600px', 'overflowY': 'auto'}, children=[
            html.Div([
                html.H4("Training Group Management", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="training-group-containers", children=[
                    html.P("Loading training groups...", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "display": "flex",
                    "flex-direction": "column",
                    "gap": "15px",
                    "margin-bottom": "20px"
                }),
            ], className="modern-card", style={
                'width': '25%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '15px'
            }),
            
            html.Div([
                html.H4("Training Recommended Articles", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="training-articles-container", children=[
                    html.P("Loading training articles...", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '40%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'margin': '0 7px'
            }),
            
            html.Div([
                html.H4("Training Article Full Text", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="training-article-fulltext-container", children=[
                    html.P("Click on an article from the middle panel to view its full content", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '30%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '7px'
            })
        ]),
        
        html.Div(id="finetune-group-management-area", style={'display': 'none', 'marginBottom': '30px', 'height': '600px', 'overflowY': 'auto'}, children=[
        html.Div([
                html.H4("Finetune Group Management", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="finetune-group-containers", children=[
                    html.P("Loading finetune groups...", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "display": "flex",
                    "flex-direction": "column",
                    "gap": "15px",
                    "margin-bottom": "20px"
                }),
            ], className="modern-card", style={
                'width': '25%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '15px'
            }),
            
        html.Div([
                html.H4("Documents List", id="finetune-articles-title", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "15px",
                    "textAlign": "center"
                }),
                html.Div(id="finetune-articles-container", children=[
                    html.P("Select a group or keyword to view documents", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "40px 20px",
                               "fontSize": "1rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "20px",
                    "minHeight": "500px",
                    "maxHeight": "700px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef"
                })
            ], className="modern-card", style={
                'width': '40%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'margin': '0 7px'
            }),
            
            html.Div([
                html.H4("Sample Operations", style={
                    "color": "#2c3e50",
                "fontSize": "1.2rem",
                "fontWeight": "bold",
                    "marginBottom": "10px",
                    "textAlign": "center"
                }),
                html.Div(id="finetune-operation-buttons", children=[], style={"marginBottom": "20px"}),
                
                html.H5("Article Full Text", style={
                    "color": "#2c3e50",
                    "fontSize": "1.1rem",
                    "fontWeight": "bold",
                    "marginBottom": "10px",
                    "textAlign": "center"
                }),
                html.Div(id="finetune-text-container", children=[
                    html.P("Click a document to preview", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "20px",
                               "fontSize": "0.9rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                "borderRadius": "8px",
                    "padding": "15px",
                    "minHeight": "150px",
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef",
                    "marginBottom": "20px",
                    "fontSize": "0.85rem"
                }),
                
                html.H4("Adjustment History", style={
                    "color": "#2c3e50",
                    "fontSize": "1.2rem",
                "fontWeight": "bold",
                    "marginBottom": "10px",
                    "textAlign": "center"
                }),
                html.Div(id="finetune-adjustment-history", children=[
                    html.P("No adjustments yet", 
                           style={
                               "color": "#7f8c8d", 
                               "fontStyle": "italic", 
                               "textAlign": "center", 
                               "padding": "15px",
                               "fontSize": "0.85rem"
                           })
                ], style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "8px",
                    "padding": "15px",
                    "minHeight": "180px",
                    "maxHeight": "250px",
                    "overflowY": "auto",
                    "border": "1px solid #e9ecef",
                    "marginBottom": "15px",
                    "fontSize": "0.85rem"
                }),
                html.Div(id="finetune-history-buttons", children=[], style={"marginTop": "10px"}),
                html.Div([
                    html.H5("Legend:", style={"color": "#2c3e50", "marginBottom": "10px", "fontSize": "1rem"}),
                    html.Ul([
                        html.Li("Gold stars: Core samples", style={"marginBottom": "3px", "fontSize": "0.9rem"}),
                        html.Li("Gray triangles: Need review", style={"marginBottom": "3px", "fontSize": "0.9rem"}),
                        html.Li("Blue dots: Background", style={"marginBottom": "3px", "fontSize": "0.9rem"})
                    ], style={"color": "#2c3e50", "marginBottom": "15px"}),
                    html.Div([
                        html.Button("Run Finetune Training", id="finetune-train-btn", n_clicks=0, style={
                            "backgroundColor": "#9b59b6",
                "color": "white",
                "border": "none",
                            "padding": "12px 24px",
                "borderRadius": "6px",
                "fontSize": "1rem",
                "fontWeight": "bold",
                "cursor": "pointer",
                "transition": "all 0.3s ease",
                            "boxShadow": "0 3px 8px rgba(155, 89, 182, 0.3)",
                            "width": "100%",
                            "marginBottom": "10px"
                        }),
                        html.Div(id="finetune-training-status", style={"marginTop": "10px", "textAlign": "center", "fontWeight": "bold"}),
                        html.Button("Clear Adjustment History", id="finetune-clear-history-btn", n_clicks=0, style={
                            "backgroundColor": "#e74c3c",
                "color": "white",
                "border": "none",
                            "padding": "8px 16px",
                "borderRadius": "6px",
                            "fontSize": "0.9rem",
                            "fontWeight": "bold",
                "cursor": "pointer",
                "transition": "all 0.3s ease",
                            "boxShadow": "0 2px 5px rgba(231, 76, 60, 0.3)",
                            "width": "100%"
                        })
                    ])
                ])
            ], className="modern-card", style={
                'width': '30%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '7px'
            })
        ]),
        

        html.Div(id="debug-output", style={"marginTop": "20px"}),
        

        dcc.Interval(
            id="interval-component",
            interval=1000, 
            n_intervals=0
        )
    ])

app.layout = create_layout()

@app.callback(
    Output("group-order", "data"),
    [
        Input("generate-btn", "n_clicks"),
        Input("group-data", "data"),
    ],
    [
        State("group-count", "value"),
        State("group-order", "data")
    ],
    prevent_initial_call=True
)
def update_group_order(generate_n_clicks, group_data, num_groups, current_order):
    ctx = dash.callback_context
    print(f"        update_group_order called")
    print(f"        triggered: {ctx.triggered}")
    print(f"        group_data: {group_data}")
    if not ctx.triggered:
        raise PreventUpdate
    
    new_order = dict(current_order) if current_order else {}
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "generate-btn":
        if not num_groups or num_groups < 1:
            raise PreventUpdate
        
        groups = {f"Group {i+1}": [] for i in range(num_groups)}
        
        groups["Exclude"] = []
        print(f"        Added 'Exclude' group for exclusion (total groups: {num_groups + 1})")
        
        return groups
    
    elif triggered_id == "group-data":
        for group_name in new_order:
            new_order[group_name] = []
        
        for kw, grp in group_data.items():
            if grp and grp in new_order:
                new_order[grp].append(kw)
        return new_order

    return new_order

@app.callback(
    Output("group-containers", "children"),
    [Input("group-order", "data"),
     Input("selected-group", "data"),
     Input("selected-keyword", "data")], 
    [State("display-mode", "data")]
)
def render_groups(group_order, selected_group, selected_keyword, display_mode):
    print(f"    DEBUG: render_groups CALLBACK TRIGGERED")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   group_order: {group_order}")
    print(f"    DEBUG:   selected_group: {selected_group}")
    print(f"    DEBUG:   selected_keyword: {selected_keyword}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:     PARAMETER TYPES:")
    print(f"    DEBUG:   group_order type: {type(group_order)}")
    print(f"    DEBUG:   selected_group type: {type(selected_group)}")
    print(f"    DEBUG:   selected_keyword type: {type(selected_keyword)}")
    print(f"    DEBUG:   display_mode type: {type(display_mode)}")
    
    if display_mode == "training" and selected_keyword is not None:
        print(f"    DEBUG:     TRAINING MODE WARNING:")
        print(f"    DEBUG:   selected_keyword is {selected_keyword}")
        print(f"    DEBUG:   This might cause conflicts with documents-2d-plot")
        print(f"    DEBUG:   But we're being careful about updates")
    
    if not group_order:
        print(f"    DEBUG:         No group_order, returning empty list")
        return []
    
    print(f"    DEBUG:      Proceeding with group rendering...")

    children = []
    for grp_name, kw_list in group_order.items():
        if grp_name == "Other":
           
            if kw_list: 
                group_display_name = "Exclude"
                group_color = get_group_color(grp_name)
            else:  
                group_display_name = "Other (Exclude)"
                group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Group {group_number}"
            group_color = get_group_color(grp_name)
        
        if grp_name == "Other":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  
            }
        else:
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px solid {group_color}",
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px"
            }
        
        group_header = html.Button(
            group_display_name,
            id={"type": "group-header", "index": grp_name},
            style=header_style
        )

        group_keywords = []
        for i, kw in enumerate(kw_list):
            
            if display_mode == "training":
                is_selected = False  
            else:
                is_selected = selected_keyword and kw == selected_keyword
            
            keyword_button = html.Button(
                kw,
                id={"type": "select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  
                    "color": "white" if is_selected else group_color,  
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  
                }
            )
            
            keyword_item = html.Div([
                keyword_button,
                html.Button("×", id={"type": "remove-keyword", "group": grp_name, "index": i}, 
                           style={"margin": "2px", "padding": "2px 6px", "fontSize": "10px", "color": "red", "float": "right"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "3px"})
            group_keywords.append(keyword_item)

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "8px",
                "minHeight": "50px",
                "maxHeight": "200px",
                "overflowY": "auto",
                "backgroundColor": "#f9f9f9",
                "marginBottom": "10px"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"marginBottom": "15px"}
        )
        children.append(group_container)

    return children


print("    DEBUG: Outputs: selected-group.data, selected-keyword.data")
print("    DEBUG: Input: {'type': 'group-header', 'index': ALL}.n_clicks")
print("    DEBUG: State: display-mode.data")
print("    DEBUG: allow_duplicate: True (for selected-keyword)")
print("    DEBUG: prevent_initial_call: True")

@app.callback(
    [Output("selected-group", "data"),
     Output("selected-keyword", "data", allow_duplicate=True)],
    Input({"type": "group-header", "index": ALL}, "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def select_group(n_clicks, display_mode):
    ctx = dash.callback_context
    
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG: n_clicks: {n_clicks}")
    print(f"    DEBUG: display_mode: {display_mode}")
    print(f"    DEBUG: display_mode type: {type(display_mode)}")
    print(f"    DEBUG: ctx.triggered: {ctx.triggered}")
    print(f"    DEBUG: ctx.triggered length: {len(ctx.triggered) if ctx.triggered else 0}")
    
    if not ctx.triggered:
        print(f"    DEBUG:         No context triggered, preventing update")
        print(f"    DEBUG: This should not happen for a valid group click")
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    triggered_prop_id = ctx.triggered[0].get('prop_id', 'N/A')
    triggered_value = ctx.triggered[0].get('value', 'N/A')

    print(f"    DEBUG:   triggered_id: {triggered_id}")
    print(f"    DEBUG:   triggered_n_clicks: {triggered_n_clicks}")
    print(f"    DEBUG:   triggered_prop_id: {triggered_prop_id}")
    print(f"    DEBUG:   triggered_value: {triggered_value}")
    print(f"    DEBUG:   triggered_id type: {type(triggered_id)}")
    print(f"    DEBUG:   triggered_n_clicks type: {type(triggered_n_clicks)}")
    
    print(f"    DEBUG:   'group-header' in triggered_id: {'group-header' in triggered_id}")
    print(f"    DEBUG:   triggered_n_clicks > 0: {triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0)}")
    print(f"    DEBUG:   triggered_n_clicks is truthy: {bool(triggered_n_clicks)}")
    
    if "group-header" in triggered_id and triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
        print(f"    DEBUG:      Valid group header click detected!")
        try:
            print(f"    DEBUG:     PARSING GROUP ID:")
            print(f"    DEBUG:   Raw triggered_id: {triggered_id}")
            print(f"    DEBUG:   Splitting by '.': {triggered_id.split('.')}")
            print(f"    DEBUG:   First part: {triggered_id.split('.')[0]}")
            
            parsed_id = json.loads(triggered_id.split('.')[0])
            print(f"    DEBUG:   Parsed ID: {parsed_id}")
            print(f"    DEBUG:   Parsed ID type: {type(parsed_id)}")
            
            selected_group = parsed_id["index"]
            print(f"    DEBUG:   Extracted group: {selected_group}")
            print(f"    DEBUG:   Group type: {type(selected_group)}")
            
            print(f"    DEBUG:     PROCESSING GROUP SELECTION:")
            print(f"    DEBUG:   Switching to group: {selected_group}")
            print(f"    DEBUG:   Current display_mode: {display_mode}")
            print(f"    DEBUG:   About to return: selected_group={selected_group}, display_mode={display_mode}")
            
            
            if display_mode == "training":
                print(f"    DEBUG:     TRAINING MODE DETECTED:")
                print(f"    DEBUG:   Returning selected_group={selected_group}")
                print(f"    DEBUG:   Returning dash.no_update for selected-keyword")
                print(f"    DEBUG:   This should prevent documents-2d-plot callback from being triggered")
                print(f"    DEBUG:   Expected behavior: no 'nonexistent object' error")
                return selected_group, dash.no_update
            else:
                print(f"    DEBUG:     KEYWORDS MODE DETECTED:")
                print(f"    DEBUG:   Returning selected_group={selected_group}")
                print(f"    DEBUG:   Returning None for selected-keyword")
                print(f"    DEBUG:   This is normal behavior for keywords mode")
                return selected_group, None
                
        except Exception as e:
            print(f"    DEBUG:         ERROR PARSING GROUP HEADER ID:")
            print(f"    DEBUG:   Error: {e}")
            print(f"    DEBUG:   Error type: {type(e)}")
            print(f"    DEBUG:   Full traceback:")
               
            traceback.print_exc()
            raise PreventUpdate
    else:
        print(f"    DEBUG:         NOT A VALID GROUP HEADER CLICK:")
        print(f"    DEBUG:   'group-header' in triggered_id: {'group-header' in triggered_id}")
        print(f"    DEBUG:   triggered_n_clicks > 0: {triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0)}")
        print(f"    DEBUG:   triggered_id contains: {triggered_id}")
        print(f"    DEBUG:   This might indicate a different type of click or callback")

    print(f"    DEBUG:         No valid conditions met, raising PreventUpdate")
    raise PreventUpdate

@app.callback(
    [Output("selected-keyword", "data", allow_duplicate=True),
     Output("keyword-highlights", "data", allow_duplicate=True)],
    [Input({"type": "select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def select_keyword_from_group(n_clicks, display_mode, group_order):
    print(f"    DEBUG: select_keyword_from_group called")
    print(f"    DEBUG: n_clicks: {n_clicks}")
    print(f"    DEBUG: display_mode: {display_mode}")
    print(f"    DEBUG: group_order: {group_order}")
    print(f"    DEBUG: n_clicks type: {type(n_clicks)}")
    print(f"    DEBUG: display_mode type: {type(display_mode)}")
    print(f"    DEBUG: group_order type: {type(group_order)}")
    
    ctx = dash.callback_context
    print(f"    DEBUG: ctx.triggered: {ctx.triggered}")
    print(f"    DEBUG: ctx.triggered length: {len(ctx.triggered) if ctx.triggered else 0}")
    
    if not ctx.triggered:
        print(f"    DEBUG: No context triggered")
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    print(f"    DEBUG: triggered_id: {triggered_id}")
    print(f"    DEBUG: triggered_n_clicks: {triggered_n_clicks}")
    print(f"    DEBUG: triggered_id type: {type(triggered_id)}")
    print(f"    DEBUG: triggered_n_clicks type: {type(triggered_n_clicks)}")
    
    if "select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            print(f"    DEBUG: Select keyword from group management: {keyword}")
            
            if triggered_n_clicks is None:
                print(f"    DEBUG: Keyword selection triggered by group change, ignoring")
                raise PreventUpdate
            
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                print(f"    DEBUG: Direct keyword click detected, selecting keyword: {keyword}")
                
                if display_mode == "training":
                    print(f"    DEBUG: Training mode: updating keyword-highlights for keyword: {keyword}")
                    keyword_docs = []
                    
                    try:
                        global df
                        if 'df' not in globals():
                            df = pd.read_csv(FILE_PATHS["csv_path"])
                        
                        for i in range(len(df)):
                            text = str(df.iloc[i, 1]).lower()
                            if keyword.lower() in text:
                                keyword_docs.append(i)
                        
                        print(f"    DEBUG: Found {len(keyword_docs)} documents containing keyword '{keyword}': {keyword_docs}")
                        return dash.no_update, keyword_docs
                    except Exception as e:
                        print(f"    DEBUG: Error finding documents for keyword '{keyword}': {e}")
                        return dash.no_update, []
                else:
                    print(f"    DEBUG: Keywords mode: updating selected-keyword for keyword: {keyword}")
                    return keyword, dash.no_update
            else:
                print(f"    DEBUG: Not a direct keyword click, ignoring")
                raise PreventUpdate
            
        except Exception as e:
            print(f"    DEBUG: Error parsing button info: {e}")
            raise PreventUpdate
    
    print(f"    DEBUG: Not a keyword selection")
    raise PreventUpdate

@app.callback(
    [Output("group-order", "data", allow_duplicate=True),
     Output("group-data", "data", allow_duplicate=True)],
    Input({"type": "remove-keyword", "group": ALL, "index": ALL}, "n_clicks"),
    [State("group-order", "data"),
     State("group-data", "data")],
    prevent_initial_call=True
)
def remove_keyword_from_group(n_clicks, group_order, group_data):
    print(f"remove_keyword_from_group called")
    print(f"n_clicks: {n_clicks}")
    
    ctx = dash.callback_context
    print(f"ctx.triggered: {ctx.triggered}")
    
    if not ctx.triggered or not any(n_clicks):
        print("No delete button clicked")
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    if not triggered_id or '.n_clicks' not in triggered_id:
        print("Invalid trigger")
        raise PreventUpdate
    
    try:
        button_id = json.loads(triggered_id.split('.')[0])
        group_name = button_id.get("group")
        keyword_index = button_id.get("index")
        
        print(f"Delete button clicked - Group: {group_name}, Index: {keyword_index}")
        
        if not group_name or keyword_index is None:
            print("Missing group name or index")
            raise PreventUpdate
        
        new_group_order = dict(group_order) if group_order else {}
        
        if group_name in new_group_order:
            keyword_list = list(new_group_order[group_name])
            if 0 <= keyword_index < len(keyword_list):
                removed_keyword = keyword_list.pop(keyword_index)
                new_group_order[group_name] = keyword_list
                
                new_group_data = dict(group_data) if group_data else {}
                if removed_keyword in new_group_data:
                    new_group_data[removed_keyword] = None
                
                print(f"Removed keyword '{removed_keyword}' from group '{group_name}'")
                print(f"Updated group_data: {removed_keyword} = None")
                
                clear_caches()
                return new_group_order, new_group_data
            else:
                print(f"Invalid keyword index {keyword_index} for group '{group_name}'")
        else:
            print(f"Group '{group_name}' not found in group_order")
            
    except Exception as e:
        print(f"Error removing keyword: {e}")
    
    raise PreventUpdate

@app.callback(
    Output("articles-container", "children"),
    [Input("selected-keyword", "data"),
     Input("selected-group", "data")],  
    [State("group-order", "data"),  
     State("display-mode", "data")],
    prevent_initial_call=True
)
def display_recommended_articles(selected_keyword, selected_group, group_order, display_mode):

    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   selected_keyword: {selected_keyword}")
    print(f"    DEBUG:   selected_group: {selected_group}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:     PARAMETER TYPES:")
    print(f"    DEBUG:   selected_keyword type: {type(selected_keyword)}")
    print(f"    DEBUG:   selected_group type: {type(selected_group)}")
    print(f"    DEBUG:   display_mode type: {type(display_mode)}")
    
    if display_mode == "training" and selected_keyword is not None:
        print(f"    DEBUG:     TRAINING MODE WARNING:")
        print(f"    DEBUG:   selected_keyword is {selected_keyword}")
        print(f"    DEBUG:   This might cause conflicts with documents-2d-plot")
        print(f"    DEBUG:   But we're being careful about updates")
    
    try:
        global df, _ARTICLES_CACHE
        if 'df' not in globals():
            print("Data not loaded")
            return html.P("Data not loaded")
        
        cache_key = None
        if selected_keyword:
            cache_key = f"keyword:{selected_keyword}"
        elif selected_group and group_order:
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    cache_key = f"group:{group_name}:{':'.join(sorted(keywords))}"
                    break
        
        if cache_key and cache_key in _ARTICLES_CACHE:
            print(f"Using cached articles for: {cache_key}")
            return _ARTICLES_CACHE[cache_key]
        
        search_keywords = []
        search_title = ""
        
        if selected_keyword:
            search_keywords = [selected_keyword]
            search_title = f"Articles containing '{selected_keyword}'"
            print(f"Searching for articles containing keyword: {selected_keyword}")
        elif selected_group:
            print(f"Group selected: {selected_group}")
            print(f"group_order parameter received: {group_order}")
            
            if group_order:
                search_keywords = []
                for group_name, keywords in group_order.items():
                    print(f"Checking group '{group_name}' vs selected '{selected_group}'")
                    if group_name == selected_group:
                        search_keywords = keywords
                        print(f"Found matching group with keywords: {keywords}")
                        break
                
                if search_keywords:
                    search_title = f"Articles containing keywords from group '{selected_group}'"
                    print(f"Will search for articles containing group '{selected_group}' keywords: {search_keywords}")
                else:
                    print(f"No keywords found for group '{selected_group}' or group is empty")
                    return html.Div([
                        html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.P(f"Group '{selected_group}' has no keywords assigned", 
                               style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                    ])
            else:
                print(f"group_order is empty or None")
                return html.Div([
                    html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                    html.P("No groups have been created yet", 
                           style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                ])
        else:
            return html.Div([
                html.H6("Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P("Please select a keyword or group to view recommended articles", 
                       style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
            ])
        
        matching_articles = []
        
        print(f"    Using BM25 search for keywords: {search_keywords}")
        
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()
            
            stemmer = PorterStemmer()
            
            all_texts = [str(df.iloc[i, 1]) for i in range(len(df))]
            
            def preprocess(text):
                tokens = text.lower().split()
                tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
                return tokens
            
            tokenized_corpus = [preprocess(doc) for doc in all_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            
            query_tokens = []
            for kw in search_keywords:
                query_tokens.extend(preprocess(kw))
            
            scores = bm25.get_scores(query_tokens)
            
            top_indices = np.argsort(scores)[::-1]
            
            for idx in top_indices:
                if scores[idx] > 0:
                    text = str(df.iloc[int(idx), 1])
                    file_keywords = extract_top_keywords(text, 5)
                    matching_articles.append({
                        'file_number': int(idx) + 1,
                        'file_index': int(idx),
                        'text': text,
                        'keywords': file_keywords,
                        'bm25_score': float(scores[idx])
                    })
                    
                    if len(matching_articles) >= 100:
                        break
            
            print(f"    BM25 search found {len(matching_articles)} relevant documents")
            
        except Exception as e:
            print(f"    BM25 search failed: {e}")
            import traceback
            traceback.print_exc()
        
        if not matching_articles:
            result = html.P(f"No articles found for the selected search criteria")
            if cache_key:
                _ARTICLES_CACHE[cache_key] = result
                print(f"Cached 'no articles' result for: {cache_key}")
            return result
        
        article_items = [
            html.H6(f"{search_title} (Found {len(matching_articles)} articles)", 
                   style={"color": "#2c3e50", "marginBottom": "15px"})
        ]
        
        for article_info in matching_articles:
            keyword_tags = []
            for keyword in article_info['keywords']:
                keyword_tag = html.Span(
                    keyword,
                    style={
                        "backgroundColor": "#e3f2fd",
                        "color": "#1976d2",
                        "padding": "2px 6px",
                        "margin": "2px",
                        "borderRadius": "12px",
                        "fontSize": "11px",
                        "display": "inline-block"
                    }
                )
                keyword_tags.append(keyword_tag)
            
            article_item = html.Div([
                html.Button(
                    html.Div([
                        html.H6(f"Article {article_info['file_number']}", 
                               style={"color": "#333", "marginBottom": "8px", "fontSize": "14px", "margin": "0"}),
                        html.Div([
                            html.Span("Top 5 Keywords: ", style={"fontWeight": "bold", "color": "#666"}),
                            html.Div(keyword_tags, style={"display": "inline-block", "marginLeft": "5px"})
                        ], style={"marginBottom": "8px"}),
                    ]),
                    id={"type": "article-item", "index": article_info['file_index']},
                    className="article-item-button",
                    style={
                        "width": "100%",
                        "padding": "12px", 
                        "border": "1px solid #eee",
                        "backgroundColor": "white",
                        "borderRadius": "5px",
                        "marginBottom": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                        "cursor": "pointer",
                        "textAlign": "left",
                        "outline": "none"
                    },
                    n_clicks=0
                ),
                html.Hr(style={"margin": "4px 0", "borderColor": "#ddd"})
            ])
            article_items.append(article_item)
        
        result = html.Div(article_items)
        if cache_key:
            _ARTICLES_CACHE[cache_key] = result
            print(f"Cached articles result for: {cache_key}")
        
        return result
        
    except Exception as e:
        print(f"Error displaying recommended articles: {e}")
        return html.P(f"Error displaying recommended articles: {str(e)}")

def extract_top_keywords(text, top_k=5):
    try:
        global kw_model
        if 'kw_model' in globals() and kw_model:
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), 
                                               stop_words='english')
            return [kw[0] for kw in keywords[:top_k]]
        else:
            words = text.lower().split()
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
            return filtered_words[:top_k]
    except Exception as e:
        print(f"Keyword extraction failed: {e}")
        return ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]

    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']

    if not triggered_n_clicks or triggered_n_clicks is None:
        raise PreventUpdate

    selected_group = json.loads(triggered_id.split('.')[0])["index"]
    print(f"Switch to group: {selected_group}")  
    print(f"        Clear selected keyword") 
    
    return selected_group, None  

def get_all_cls_vectors(df_data, model, tokenizer, device):
    vectors = []
    for i in range(len(df_data)):
        text = str(df_data.iloc[i, 1])
        tokens = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        input_ids = tokens['input_ids'][0]
        max_length = 512

        chunks = [input_ids[j:j + max_length] for j in range(0, len(input_ids), max_length)]
        cls_vectors = []
        for chunk in chunks:
            if len(chunk) < max_length:
                pad_length = max_length - len(chunk)
                chunk = pad(chunk, (0, pad_length), value=tokenizer.pad_token_id)
            chunk = chunk.unsqueeze(0).to(device)
            attention_mask = (chunk != tokenizer.pad_token_id).long()
            outputs = model(chunk, attention_mask=attention_mask)
            cls_vector = outputs.last_hidden_state[:, 0, :]
            cls_vectors.append(cls_vector)
        if len(cls_vectors) == 0:
            vectors.append(torch.zeros(model.config.hidden_size))
        else:
            stacked_cls = torch.cat(cls_vectors, dim=0)
            avg_cls = torch.mean(stacked_cls, dim=0)
            vectors.append(avg_cls.detach().cpu())
    return torch.stack(vectors, dim=0)
def run_training():

    try:
        clear_caches()
        
        print(f"    DEBUG: Loading CSV data from: {FILE_PATHS['csv_path']}")
        df = pd.read_csv(FILE_PATHS["csv_path"])
        print(f"    DEBUG: CSV loaded successfully, shape: {df.shape}")
        
        df_clean = df.dropna(subset=[df.columns[1]])
        print(f"    DEBUG: After dropping NaN, shape: {df_clean.shape}")
        
        all_texts = df_clean.iloc[:,1].astype(str).tolist()
        all_labels = df_clean.iloc[:,0].astype(str).tolist()
        print(f"[INFO] texts: {len(all_texts)}")
        print(f"    DEBUG: Sample text: {all_texts[0][:100]}..." if all_texts else "    DEBUG: No texts found!")
        print(f"    DEBUG: Sample label: {all_labels[0]}" if all_labels else "    DEBUG: No labels found!")
        
    except Exception as e:
        print(f"    ERROR: Failed to load CSV data: {e}")
        traceback.print_exc()
        return None, None

    out_dir = OUTPUT_DIR; os.makedirs(out_dir, exist_ok=True)

    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer('english')
    
    def process_articles_serial(articles):
        tokenized_corpus, valid_indices = [], []
        for i, a in enumerate(articles):
            try:
                a = re.sub(r'\d+', '', str(a)).strip()
                if len(a.split()) >= 5:
                    words = word_tokenize(a)
                    stemmed = [stemmer.stem(w.lower()) for w in words]
                    if stemmed:
                        tokenized_corpus.append(" ".join(stemmed))
                        valid_indices.append(i)
            except Exception:
                pass
        return tokenized_corpus, valid_indices
    
    def custom_tokenizer(text): return text.split()
    def get_main_category(label): return label.split('.')[0] if '.' in label else label
    
    def bm25_search_batch(bm25, query_groups, valid_indices):
        results = {}
        for g, words in query_groups.items():
            q = [stemmer.stem(w.lower()) for w in words]
            print(f"    BM25 '{g}': {words} -> {q}")
            scores = bm25.get_scores(q)
            print(f"    min={min(scores):.4f}, max={max(scores):.4f}, {sum(1 for s in scores if s > 0)}")
            
            idx_corpus = [i for i, s in enumerate(scores) if s > 0.1]  
            if len(idx_corpus) == 0:
                idx_corpus = [i for i, s in enumerate(scores) if s > 0.01]
            
            idx_orig = [valid_indices[i] for i in idx_corpus]
            results[g] = idx_orig[:3000]  

        return results

    tokenized_corpus, valid_indices = process_articles_serial(all_texts)
    bm25 = BM25Okapi([s.split() for s in tokenized_corpus])

    USER_GROUPS_ONLY = True
    ALLOW_EMPTY_GROUPS = False
    
    query_groups = {}
    print(f"    DEBUG: Checking for user groups at: {FILE_PATHS['final_list_path']}")
    if os.path.exists(FILE_PATHS["final_list_path"]):
        try:
            with open(FILE_PATHS["final_list_path"], "r", encoding="utf-8") as f:
                user_groups = json.load(f)
            print(f"    DEBUG: Loaded user groups: {user_groups}")
            for group_name, keywords in user_groups.items():
                if keywords:  
                    query_groups[group_name] = keywords
                    print(f"     {group_name}:  {keywords}")
            print(f"     Loaded user groups for training: {list(query_groups.keys())}")
        except Exception as e:
            print(f"    ERROR: Failed to load user groups: {e}")
            
            traceback.print_exc()
            query_groups = {}
    else:
        print(f"    WARNING: User groups file not found: {FILE_PATHS['final_list_path']}")
        query_groups = {}
    
    if not query_groups and not ALLOW_EMPTY_GROUPS:
        print("    ERROR: No user groups found and ALLOW_EMPTY_GROUPS is False")

        return None, None
    
    print(f"    DEBUG: Starting BM25 search with {len(query_groups)} groups")
    try:
        matched_dict = bm25_search_batch(bm25, query_groups, valid_indices)
        print(f"    DEBUG: BM25 search completed successfully")
    except Exception as e:
        print(f"    ERROR: BM25 search failed: {e}")
        traceback.print_exc()
        return None, None
    for g, idxs in matched_dict.items():
        print(f"[BM25] {g}: {len(idxs)} docs")
    

    if "Other" not in matched_dict:
        matched_dict["Other"] = []
        print(f"[BM25] Other: {len(matched_dict['Other'])} docs")
    else:
        print(f"[BM25] Other: {len(matched_dict['Other'])} docs")
    
    with open(FILE_PATHS["bm25_search_results"], "w", encoding="utf-8") as f:
        json.dump(matched_dict, f, ensure_ascii=False, indent=2)


    print(f"    DEBUG: Initializing tokenizer with model: {get_config('bert_name')}")
    try:
        tokenizer = BertTokenizer.from_pretrained(get_config("bert_name"))
        print(f"    DEBUG: Tokenizer loaded successfully")
    except Exception as e:
        print(f"    ERROR: Failed to load tokenizer: {e}")
        traceback.print_exc()
        return None, None
    

    # 使用全局SentenceEncoder类
    
    print(f"    DEBUG: Initializing encoder on device: {device}")
    try:
        encoder = SentenceEncoder(device=device)
        print(f"    DEBUG: Encoder initialized successfully")
    except Exception as e:
        print(f"    ERROR: Failed to initialize encoder: {e}")
           
        traceback.print_exc()
        return None, None


    class TripletTextDataset(torch.utils.data.Dataset):
        def __init__(self, triplets, texts):
            self.triplets, self.texts = triplets, texts
        def __len__(self): return len(self.triplets)
        def __getitem__(self, i):
            a,p,n = self.triplets[i]
            return self.texts[a], self.texts[p], self.texts[n]

    def collate_triplet(batch, tokenizer, max_len=256, device='cpu'):
        A,P,N = zip(*batch)
        tok = lambda T: tokenizer(list(T), return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        A = {k:v.to(device) for k,v in tok(A).items()}
        P = {k:v.to(device) for k,v in tok(P).items()}
        N = {k:v.to(device) for k,v in tok(N).items()}
        return A,P,N

    def generate_triplets_from_groups(matched_dict, min_pos_per_group=None, num_pos_per_anchor=None, num_neg_per_anchor=None):
        if min_pos_per_group is None: min_pos_per_group = get_config("min_pos_per_group")
        if num_pos_per_anchor is None: num_pos_per_anchor = get_config("num_pos_per_anchor")
        if num_neg_per_anchor is None: num_neg_per_anchor = get_config("num_neg_per_anchor")
        pos_groups = {g:idxs for g,idxs in matched_dict.items() if g!="Other" and len(idxs)>=min_pos_per_group}
        if not pos_groups: return []
        
        triplets = []
        for group_name, idxs in pos_groups.items():
            neg_pool = []
            for g, g_idxs in matched_dict.items():
                if g != group_name:  
                    neg_pool.extend(g_idxs)
                    print(f"[DEBUG]  {g}: {len(g_idxs)} docs")
            
            print(f"[DEBUG] {group_name} : {len(neg_pool)} docs")
            if len(neg_pool) == 0:  
                print(f"[WARN] {group_name} ")
                continue
            
            for a in idxs:
                pos = [x for x in idxs if x!=a]
                if len(pos) > num_pos_per_anchor: 
                    pos = random.sample(pos, num_pos_per_anchor)
                
             
                neg = random.sample(neg_pool, k=min(num_neg_per_anchor, len(neg_pool)))
                
                for p in pos:
                    for n in neg:
                        if n==a or n==p: continue
                        triplets.append((a,p,n))
        
        random.shuffle(triplets)
        print(f"[Triplet] {len(triplets)} triplets")
        return triplets

    def semi_hard_triplet(za, zp, zn, margin=0.8):

        d_ap = torch.norm(za - zp, dim=1)
        d_an = torch.norm(za - zn, dim=1)
        mask = (d_an > d_ap) & (d_an < d_ap + margin)
        if mask.any():
            return nn.TripletMarginLoss(margin=margin, p=2)(za[mask], zp[mask], zn[mask])
        
      
        with torch.no_grad():
            D = torch.cdist(za, zn, p=2)
            n_hard = D.argmin(dim=1)
        return nn.TripletMarginLoss(margin=margin, p=2)(za, zp, zn[n_hard])

    def train_triplet_text(model, tokenizer, triplets, texts, device, epochs=None, bs=None, margin=None, lr=None, freeze_layers=None):

        if epochs is None: epochs = get_config("triplet_epochs")
        if bs is None: bs = get_config("triplet_batch_size")
        if margin is None: margin = get_config("triplet_margin")
        if lr is None: lr = get_config("triplet_lr")
        if freeze_layers is None: freeze_layers = get_config("freeze_layers")
        if hasattr(model,'bert'):
            for p in model.bert.embeddings.parameters(): p.requires_grad=False
            for L in model.bert.encoder.layer[:freeze_layers]:
                for p in L.parameters(): p.requires_grad=False
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"[INFO] Trainable params: {sum(p.numel() for p in params):,}")

        ds = TripletTextDataset(triplets, texts)
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                         collate_fn=lambda b: collate_triplet(b, tokenizer, device=device))
        opt = torch.optim.AdamW(params, lr=lr)

        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            for step, (A,P,N) in enumerate(dl):
                opt.zero_grad()
                za = model.encode_tokens(A)
                zp = model.encode_tokens(P)
                zn = model.encode_tokens(N)
                loss = semi_hard_triplet(za, zp, zn, margin)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                print(f"[Triplet] Epoch {ep+1} Step {step+1}/{len(dl)} Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(dl)
            print(f"[Triplet] Ep {ep+1} Avg {avg_loss:.4f}")
        return model

    def encode_corpus(model, tokenizer, texts, device, bs=None, max_len=None):
        if bs is None: bs = get_config("encoding_batch_size")
        if max_len is None: max_len = get_config("max_length")
        model.eval()
        Z = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                tokens = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
                tokens = {k:v.to(device) for k,v in tokens.items()}
                z = model.encode_tokens(tokens)
                Z.append(z.cpu().numpy())
        return np.vstack(Z)

    def gap_based_group_filtering(Z_raw, matched_dict, label_map, alpha=None):
        if alpha is None:
            alpha = get_config("gap_alpha")
        Z_np = Z_raw.copy()
        Z_norm = Z_np / (np.linalg.norm(Z_np, axis=1, keepdims=True) + 1e-8)
        group_centers = {}
        for group_name, indices in matched_dict.items():
            if group_name == "Other" or len(indices) == 0:
                continue
            valid_indices = [idx for idx in indices if idx < len(Z_norm)]
            if len(valid_indices) > 0:
                group_center = np.mean(Z_norm[valid_indices], axis=0)
                group_center = group_center / (np.linalg.norm(group_center) + 1e-8)
                group_centers[group_name] = group_center
        
        all_similarities = []
        for group_name, center in group_centers.items():
            sim = np.dot(Z_norm, center)
            all_similarities.append(sim)
        
        if len(all_similarities) == 0:

            return matched_dict
        
        all_similarities = np.array(all_similarities).T  
        
        sorted_indices = np.argsort(all_similarities, axis=1)[:, ::-1]  
        s_top1 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 0]]
        s_top2 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 1]] if all_similarities.shape[1] > 1 else s_top1
        gap = s_top1 - s_top2
        
        group_names = list(group_centers.keys())
        arg1 = sorted_indices[:, 0]  
        
        print(f"  {len(gap)} gap")
        print(f"  Gap: [{gap.min():.3f}, {gap.max():.3f}]")
        
        group_thresholds = {}
        
        for group_name, group_idx in label_map.items():
            if group_name == "Other" or group_name not in group_centers:
                continue
                

            if group_name in group_names:
                group_center_idx = group_names.index(group_name)
                
                group_mask = (arg1 == group_center_idx)
                if not group_mask.any():
                    continue
                    
                gaps_group = gap[group_mask]
                mean_gap = gaps_group.mean()
                std_gap = gaps_group.std()
                

                base_threshold = mean_gap - alpha * std_gap
                
                min_samples = get_config("gap_min_samples")
                percentile_fallback = get_config("gap_percentile_fallback")
                if len(gaps_group) < min_samples or std_gap < 1e-6:

                    threshold = np.percentile(gaps_group, percentile_fallback)
                    print(f"    {group_name}: {len(gaps_group)}, std:{std_gap:.6f}")
                else:

                    global_median = np.median(gap)
                    thr_floor = get_config("gap_floor_threshold")
                    mix_ratio = get_config("gap_mix_ratio")
                    threshold = max((1 - mix_ratio) * base_threshold + mix_ratio * global_median, thr_floor)
                
                group_thresholds[group_name] = threshold
                
                print(f"    {group_name}: mean={mean_gap:.3f}, std={std_gap:.3f}, thr={threshold:.3f}")
                print(f"      {np.sum(group_mask)}, gap: [{gaps_group.min():.3f}, {gaps_group.max():.3f}]")
        
        keep_mask = np.ones(len(gap), dtype=bool)
        filtered_by_group = {}
        
        for i, gap_val in enumerate(gap):
            if arg1[i] < len(group_names):
                group_name = group_names[arg1[i]]
                if group_name in group_thresholds:
                    if gap_val < group_thresholds[group_name]:
                        keep_mask[i] = False
                        if group_name not in filtered_by_group:
                            filtered_by_group[group_name] = 0
                        filtered_by_group[group_name] += 1
        
        filtered_count = np.sum(~keep_mask)
        print(f"  {filtered_count} ({filtered_count/len(gap)*100:.1f}%)")
        for group_name, count in filtered_by_group.items():
            print(f"    {group_name}: {count}")
        
        clean_matched_dict = {}
        for group_name, indices in matched_dict.items():
            if group_name == "Other":
                clean_matched_dict[group_name] = indices  
                continue
                
            if group_name in group_centers:
                group_center_idx = group_names.index(group_name)
                orig_idxs = set(indices)  
                group_mask = (arg1 == group_center_idx) & keep_mask
                filtered_indices = [int(i) for i in np.where(group_mask)[0] if int(i) in orig_idxs]
                clean_matched_dict[group_name] = filtered_indices
            else:
                clean_matched_dict[group_name] = indices
        
        return clean_matched_dict

    def build_group_prototypes(encoder, tokenizer, texts, matched_dict, device, bs=None, min_per_group=None):
        if bs is None: bs = get_config("encoding_batch_size")
        if min_per_group is None: min_per_group = get_config("min_per_group_prototype")

        def encode_all():
            encoder.eval()
            Z=[]
            for i in range(0, len(texts), bs):
                toks = tokenizer(texts[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks = {k:v.to(device) for k,v in toks.items()}
                Z.append(encoder.encode_tokens(toks).detach().cpu())
            return torch.vstack(Z)  
        
        Z_all = encode_all()           
        G = {}
        for g, idxs in matched_dict.items():
            if g == "Other": 
                continue
            idxs = [i for i in idxs if 0 <= i < len(Z_all)]
            if len(idxs) >= min_per_group:
                proto = Z_all[idxs].mean(0)
                G[g] = nn.functional.normalize(proto, dim=0)  # (D,)
        return G  # dict[str -> (D,)]

    def prototype_center_training(encoder, tokenizer, all_texts, group_prototypes, device,
                                 epochs=None, bs=None, lr=None, matched_dict=None):

        if epochs is None: epochs = get_config("proto_epochs")
        if bs is None: bs = get_config("proto_batch_size")
        if lr is None: lr = get_config("proto_lr")

        if hasattr(encoder, 'bert'):
            for p in encoder.bert.embeddings.parameters(): 
                p.requires_grad = False
            for layer in encoder.bert.encoder.layer[:4]:
                for p in layer.parameters(): 
                    p.requires_grad = False
        
        global_prototypes = {}
        for group_name, proto_tensor in group_prototypes.items():
            if group_name != "Other":
                global_prototypes[group_name] = proto_tensor.clone().to(device).detach()
        
        params = [p for p in encoder.parameters() if p.requires_grad]
        print(f"Prototype Center Training: {sum(p.numel() for p in params):,}")
        
        opt = torch.optim.AdamW(params, lr=lr)
        
        class BalancedBatchSampler:
            def __init__(self, indices_by_group, groups, m_per_group=4, batch_size=64):
                self.buckets = {g: list(idxs) for g, idxs in indices_by_group.items()}
                self.groups = [g for g in groups if len(self.buckets[g]) >= m_per_group]
                self.m = m_per_group
                self.batch_size = batch_size
                
            def __iter__(self):
                groups_shuffled = self.groups.copy()
                random.shuffle(groups_shuffled)
                
                for i in range(0, len(groups_shuffled), 2):
                    gs = groups_shuffled[i:i+2]
                    if len(gs) < 2:
                        break
                    
                    batch = []
                    for g in gs:
                        group_samples = self.buckets[g].copy()
                        random.shuffle(group_samples)
                        batch.extend(group_samples[:self.m])
                    
                    if len(batch) < self.batch_size:
                        remaining = self.batch_size - len(batch)
                        all_samples = []
                        for g, samples in self.buckets.items():
                            if g not in gs:  
                                all_samples.extend(samples)
                        if all_samples:
                            random.shuffle(all_samples)
                            batch.extend(all_samples[:remaining])
                    
                    yield batch[:self.batch_size]
                    
            def __len__(self):
                return max(1, len(self.groups) // 2)
        
        m_per_group = bs // 4
        sampler = BalancedBatchSampler(matched_dict, list(global_prototypes.keys()), 
                                     m_per_group=m_per_group, batch_size=bs)
        
        print(f"Balanced Batch: {list(global_prototypes.keys())}")
        print(f"Balanced Batch: {m_per_group}")
        
        encoder.train()
        for ep in range(epochs):
            total_center_loss = 0
            total_loss = 0
            steps = 0
            
            for batch_indices in sampler:
                batch_texts = [all_texts[i] for i in batch_indices]
                
                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                                 truncation=True, max_length=256).to(device)
                outputs = encoder.encode_tokens(inputs)
                z_batch = nn.functional.normalize(outputs, p=2, dim=-1)
                
                batch_group_means = {}
                for group_name, indices in matched_dict.items():
                    if group_name == "Other":
                        continue
                    
                    group_mask = torch.tensor([i in indices for i in batch_indices], device=device)
                    if group_mask.sum() > 0:
                        group_embeddings = z_batch[group_mask]
                        batch_group_means[group_name] = group_embeddings.mean(dim=0)
                
                
                center_loss = torch.tensor(0.0, device=device)
                for group_name, indices in matched_dict.items():
                    if group_name == "Other" or group_name not in global_prototypes:
                        continue
                    
                    group_mask = torch.tensor([i in indices for i in batch_indices], device=device)
                    if group_mask.sum() > 0:
                        group_embeddings = z_batch[group_mask]
                        prototype = global_prototypes[group_name]
                        
                        distances = torch.norm(group_embeddings - prototype, p=2, dim=1)
                        center_loss += distances.mean()
                
                total_batch_loss = center_loss
                
                if total_batch_loss > 0:
                    opt.zero_grad()
                    total_batch_loss.backward()
                    opt.step()
                    
                    print(f"[CenterPull] Epoch {ep+1} Step {steps+1} Center Loss: {center_loss.item():.4f}")
                    
                    total_center_loss += center_loss.item()
                    total_loss += total_batch_loss.item()
                    steps += 1
            
            if steps > 0:
                avg_center = total_center_loss / steps
                avg_total = total_loss / steps
                print(f"[PrototypeCenter] Epoch {ep+1}/{epochs} Center={avg_center:.4f} Total={avg_total:.4f}")
            
            print(f"  Epoch {ep+1}...")
            with torch.no_grad():
                encoder.eval()
                Z_all = []
                for i in range(0, len(all_texts), bs):
                    batch_texts = all_texts[i:i+bs]
                    inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                                     truncation=True, max_length=256).to(device)
                    outputs = encoder.encode_tokens(inputs)
                    Z_all.append(outputs.detach())
                Z_all = torch.cat(Z_all, dim=0) 
                Z_all = nn.functional.normalize(Z_all, p=2, dim=-1)
                
                ema_alpha = get_config("ema_alpha")  
                for group_name, indices in matched_dict.items():
                    if group_name != "Other" and group_name in global_prototypes:
                        valid_indices = [i for i in indices if 0 <= i < len(Z_all)]
                        if len(valid_indices) > 0:

                            current_proto = Z_all[valid_indices].mean(dim=0)
                            current_proto = nn.functional.normalize(current_proto, p=2, dim=-1)
                            
                            global_prototypes[group_name] = (1 - ema_alpha) * global_prototypes[group_name] + ema_alpha * current_proto
                            global_prototypes[group_name] = nn.functional.normalize(global_prototypes[group_name], p=2, dim=-1)
                
                encoder.train() 
        
        return encoder

    def evaluate_clustering_quality(Z, true_labels, matched_dict, group_names):
        
       
        
        silhouette = silhouette_score(Z, true_labels)
        
        group_labels = []
        for i in range(len(true_labels)):
            assigned = False
            for group_name in group_names:
                if i in matched_dict.get(group_name, []):
                    group_labels.append(group_name)
                    assigned = True
                    break
            if not assigned:
                group_labels.append("Unassigned")
        
        group_silhouette = silhouette_score(Z, group_labels)
        
        group_stats = {}
        for group_name in group_names:
            if group_name in matched_dict:
                group_indices = matched_dict[group_name]
                if len(group_indices) > 1:
                    group_embeddings = Z[group_indices]
                    
                    intra_distances = []
                    for i in range(len(group_embeddings)):
                        for j in range(i+1, len(group_embeddings)):
                            dist = np.linalg.norm(group_embeddings[i] - group_embeddings[j])
                            intra_distances.append(dist)
                    intra_cohesion = np.mean(intra_distances) if intra_distances else 0
                    
                    other_indices = []
                    for other_group in group_names:
                        if other_group != group_name and other_group in matched_dict:
                            other_indices.extend(matched_dict[other_group])
                    
                    if other_indices:
                        group_center = np.mean(group_embeddings, axis=0)
                        other_embeddings = Z[other_indices]
                        inter_distances = [np.linalg.norm(group_center - other_emb) for other_emb in other_embeddings]
                        inter_separation = np.mean(inter_distances)
                    else:
                        inter_separation = 0
                    
                    separation_ratio = inter_separation / intra_cohesion if intra_cohesion > 0 else 0
                    
                    group_stats[group_name] = {
                        'size': len(group_indices),
                        'intra_cohesion': float(intra_cohesion),
                        'inter_separation': float(inter_separation),
                        'separation_ratio': float(separation_ratio)
                    }
        
        return {
            'silhouette_true_labels': silhouette,
            'silhouette_group_labels': group_silhouette,
            'group_stats': group_stats
        }

    

    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            print(f"{group_name}: {len(indices)} docs ")
            continue
        
        if len(indices) == 0:
            print(f"{group_name}: 0 docs")
            continue
            
        group_labels = [all_labels[i] for i in indices if i < len(all_labels)]
        if len(group_labels) == 0:
            print(f"{group_name}: 0 docs ")
            continue
            
        from collections import Counter
        label_counts = Counter(group_labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        purity = most_common_count / len(group_labels)
        
        print(f"{group_name}: {len(indices)} docs, {purity:.3f} ({most_common_label}, {most_common_count}/{len(group_labels)})")
    

    doc_to_group = {}  
    final_matched_dict = {}
    
    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            continue
            
        final_matched_dict[group_name] = []
        for doc_id in indices:
            if doc_id not in doc_to_group:
                doc_to_group[doc_id] = group_name
                final_matched_dict[group_name].append(doc_id)
            else:
                print(f"Duplicate document ID: {doc_id} found in multiple groups")
    

    if "Other" in matched_dict:
        final_matched_dict["Other"] = []
        for doc_id in matched_dict["Other"]:
            if doc_id in doc_to_group:
                original_group = doc_to_group[doc_id]
                final_matched_dict[original_group].remove(doc_id)
                print(f"{doc_id} {original_group} Other")
            
            doc_to_group[doc_id] = "Other"
            final_matched_dict["Other"].append(doc_id)
    
    matched_dict = final_matched_dict
    

    for group_name, indices in matched_dict.items():
        print(f"{group_name}: {len(indices)} docs")
    

    Z_raw = encode_corpus(encoder, tokenizer, all_texts, device)
    
    label_map, cur = {}, 0
    for g in ["Group 1", "Group 2", "Group 3"]:
        if g in matched_dict:
            label_map[g] = cur; cur += 1
    other_label = cur; label_map["Other"] = other_label
    
    import copy
    bm25_results = copy.deepcopy(matched_dict)
    clean_matched_dict = gap_based_group_filtering(Z_raw, matched_dict, label_map, alpha=0.5)
    

    matched_dict_for_display = copy.deepcopy(matched_dict)  
    matched_dict = clean_matched_dict  
    

    with open(FILE_PATHS["filtered_group_assignment"], "w", encoding="utf-8") as f:
        json.dump(matched_dict, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Filtered group assignment saved to: {FILE_PATHS['filtered_group_assignment']}")
    print(f"[DEBUG] Filtered group sizes: {json.dumps({k: len(v) for k, v in matched_dict.items()}, ensure_ascii=False)}")
    


    with open(FILE_PATHS["gap_based_filter_results"], "w", encoding="utf-8") as f:
        json.dump({
            "original_sizes": {k: len(v) for k, v in bm25_results.items()},
            "clean_sizes": {k: len(v) for k, v in clean_matched_dict.items()},
            "method": "gap_based_group_filtering",
            "alpha": 0.5,
            "threshold_method": "mean_minus_alpha_std_with_fallback"
        }, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Gap-based: {os.path.join(out_dir, 'gap_based_filter_results.json')}")
    

    group_prototypes = build_group_prototypes(encoder, tokenizer, all_texts, matched_dict, device)
    print(f"[INFO] {len(group_prototypes)}: {list(group_prototypes.keys())}")
    


    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            print(f"{group_name}: {len(indices)} docs")
            continue
        
        if len(indices) == 0:
            print(f"{group_name}: 0 docs")
            continue
            
        group_labels = [all_labels[i] for i in indices if i < len(all_labels)]
        if len(group_labels) == 0:
            print(f"{group_name}: 0 docs")
            continue
            
        
        label_counts = Counter(group_labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        purity = most_common_count / len(group_labels)
        
        print(f"{group_name}: {len(indices)} docs, {purity:.3f} ({most_common_label}, {most_common_count}/{len(group_labels)})")

    print(f"[DEBUG] matched_dict keys: {list(matched_dict.keys())}")
    for g, idxs in matched_dict.items():
        print(f"[DEBUG] {g}: {len(idxs)} docs")
    

    try:
        triplets = generate_triplets_from_groups(matched_dict)
        print(f"    DEBUG: Generated {len(triplets)} triplets")
        
        if len(triplets) > 0:
            print(f"[INFO] {len(triplets)} triplets")
            print(f"    DEBUG: Starting triplet training...")
            encoder = train_triplet_text(encoder, tokenizer, triplets, all_texts, device)
            print("[INFO] Triplet embedding")
        else:
            print("[WARN] Triplets Triplet")
    except Exception as e:
        print(f"    ERROR: Triplet generation/training failed: {e}")
        traceback.print_exc()
        return None, None
    


    try:
        encoder = prototype_center_training(encoder, tokenizer, all_texts, group_prototypes, device,
                                          matched_dict=clean_matched_dict)

    except Exception as e:
    
           
        traceback.print_exc()
        return None, None
    
    
    def l2norm(X): 
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    print(f"    DEBUG: Encoding corpus with {len(all_texts)} texts...")
    try:
        Z_current = encode_corpus(encoder, tokenizer, all_texts, device)
        print(f"    DEBUG: Encoded corpus shape: {Z_current.shape}")
        Zn = l2norm(Z_current)
        print(f"    DEBUG: Normalized embeddings shape: {Zn.shape}")
    except Exception as e:
        print(f"    ERROR: Corpus encoding failed: {e}")
        traceback.print_exc()
        return None, None
    
    group_stats = {}
    ema_alpha = get_config("ema_alpha")  
    
    for g, train_idxs in matched_dict.items():
        if g == "Other" or len(train_idxs) < 5: 
            continue
        train_idxs = [i for i in train_idxs if 0 <= i < len(Zn)]
        if len(train_idxs) < 5: 
            continue

  
        proto = Zn[train_idxs].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        

        d_train = np.linalg.norm(Zn[train_idxs] - proto, axis=1)
        r_core = np.quantile(d_train, 0.50)
        r_near = np.quantile(d_train, 0.80)
        r_edge = np.quantile(d_train, 0.90)
        
        group_stats[g] = {
            "proto": proto, 
            "train_idxs": train_idxs, 
            "r_core": float(r_core), 
            "r_near": float(r_near), 
            "r_edge": float(r_edge),
            "ema_proto": proto.copy(),  
            "ema_r_core": float(r_core),  
            "ema_r_near": float(r_near),
            "ema_r_edge": float(r_edge)
        }
    
    
    for g, stat in group_stats.items():
        train_set = set(stat["train_idxs"])
        proto = stat["ema_proto"]
        r_core = stat["ema_r_core"]
        
        d_all = np.linalg.norm(Zn - proto, axis=1)
        
        high_conf_new = []
        for i, d in enumerate(d_all):
            if i not in train_set and d <= r_core:
                high_conf_new.append((i, d))
        
        if len(high_conf_new) > 0:
            print(f"  {g}: {len(high_conf_new)}")
            
            high_conf_indices = [i for i, _ in high_conf_new]
            high_conf_embeddings = Zn[high_conf_indices]
            
            new_proto = high_conf_embeddings.mean(axis=0)
            new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-8)
            
            stat["ema_proto"] = (1 - ema_alpha) * stat["ema_proto"] + ema_alpha * new_proto
            stat["ema_proto"] = stat["ema_proto"] / (np.linalg.norm(stat["ema_proto"]) + 1e-8)
            
            all_relevant_indices = stat["train_idxs"] + high_conf_indices
            d_combined = np.linalg.norm(Zn[all_relevant_indices] - stat["ema_proto"], axis=1)
            
            new_r_core = np.quantile(d_combined, 0.50)
            new_r_near = np.quantile(d_combined, 0.80) 
            new_r_edge = np.quantile(d_combined, 0.90)
            
            stat["ema_r_core"] = (1 - ema_alpha) * stat["ema_r_core"] + ema_alpha * new_r_core
            stat["ema_r_near"] = (1 - ema_alpha) * stat["ema_r_near"] + ema_alpha * new_r_near
            stat["ema_r_edge"] = (1 - ema_alpha) * stat["ema_r_edge"] + ema_alpha * new_r_edge
            
            print(f"    EMA: core={stat['ema_r_core']:.3f}, near={stat['ema_r_near']:.3f}, edge={stat['ema_r_edge']:.3f}")
        else:
            print(f"  {g}:")
    

    try:
        Z_trained = encode_corpus(encoder, tokenizer, all_texts, device)
        print(f"    DEBUG: Final embeddings shape: {Z_trained.shape}")
    except Exception as e:
        print(f"    ERROR: Final embedding encoding failed: {e}")
        traceback.print_exc()
        return None, None

    main_categories = [get_main_category(label) for label in all_labels]
    

    cluster_eval_raw = evaluate_clustering_quality(Z_raw, main_categories, bm25_results, list(bm25_results.keys()))
    
    cluster_eval_trained = evaluate_clustering_quality(Z_trained, main_categories, matched_dict, list(matched_dict.keys()))
    print(f"Silhouette Score : {cluster_eval_raw['silhouette_true_labels']:.4f} -> {cluster_eval_trained['silhouette_true_labels']:.4f}")
    print(f"Silhouette Score (BM25): {cluster_eval_raw['silhouette_group_labels']:.4f} -> {cluster_eval_trained['silhouette_group_labels']:.4f}")

    def convert_numpy_types(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    cluster_eval_raw_serializable = convert_numpy_types(cluster_eval_raw)
    cluster_eval_trained_serializable = convert_numpy_types(cluster_eval_trained)
    
    with open(FILE_PATHS["clustering_evaluation"], "w", encoding="utf-8") as f:
        json.dump({
            "raw_embedding": cluster_eval_raw_serializable,
            "trained_embedding": cluster_eval_trained_serializable,
            "triplets_count": len(triplets),
            "training_epochs": 5,
            "margin": 0.8
        }, f, ensure_ascii=False, indent=2)


    try:

        n = len(Z_raw)
        perp = max(2, min(30, (n - 1) // 3))
        print(f"    DEBUG: Using perplexity: {perp}")
        
 
        print(f"    DEBUG: Computing t-SNE for raw embeddings...")
        X2_raw = TSNE(
            n_components=2, 
            perplexity=perp, 
            random_state=42, 
            max_iter=get_config("tsne_max_iter")
        ).fit_transform(Z_raw)
        print(f"    DEBUG: Raw t-SNE shape: {X2_raw.shape}")
        
        print(f"    DEBUG: Computing t-SNE for trained embeddings...")
        X2_trained = TSNE(
            n_components=2, 
            perplexity=perp, 
            random_state=42, 
            max_iter=get_config("tsne_max_iter")
        ).fit_transform(Z_trained)
        print(f"    DEBUG: Trained t-SNE shape: {X2_trained.shape}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        unique_main_cats = sorted(list(set(main_categories)))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_main_cats)))
        color_map = dict(zip(unique_main_cats, colors))
        
        for main_cat in unique_main_cats:
            cat_mask = [cat == main_cat for cat in main_categories]
            cat_indices = np.where(cat_mask)[0]
            
            if len(cat_indices) > 0:
                cat_2d = X2_raw[cat_indices]
                ax1.scatter(cat_2d[:,0], cat_2d[:,1], 
                           c=[color_map[main_cat]], alpha=0.8, s=60, 
                           marker='o', label=f'{main_cat}')
        
        ax1.set_title(f"BERT Embedding (Silhouette: {cluster_eval_raw['silhouette_true_labels']:.3f})", 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1'); ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        for main_cat in unique_main_cats:
            cat_mask = [cat == main_cat for cat in main_categories]
            cat_indices = np.where(cat_mask)[0]
            
            if len(cat_indices) > 0:
                cat_2d = X2_trained[cat_indices]
                ax2.scatter(cat_2d[:,0], cat_2d[:,1], 
                           c=[color_map[main_cat]], alpha=0.8, s=60, 
                           marker='o', label=f'{main_cat}')
        
        ax2.set_title(f"Triplet Embedding (Silhouette: {cluster_eval_trained['silhouette_true_labels']:.3f})", 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Dimension 1'); ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FILE_PATHS["triplet_training_comparison"], dpi=300, bbox_inches='tight')
        print(f"[SAVE] Triplet: {os.path.join(out_dir, 'triplet_training_comparison.png')}")
        
        with open(FILE_PATHS["triplet_tsne_comparison"], "w", encoding="utf-8") as f:
            json.dump({
                "raw_tsne": {
                    "x": X2_raw[:,0].tolist(),
                    "y": X2_raw[:,1].tolist()
                },
                "trained_tsne": {
                    "x": X2_trained[:,0].tolist(),
                    "y": X2_trained[:,1].tolist()
                },
                "labels": all_labels,
                "main_categories": main_categories,
                "silhouette_scores": {
                    "raw": cluster_eval_raw['silhouette_true_labels'],
                    "trained": cluster_eval_trained['silhouette_true_labels']
                }
            }, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print("[WARN] 2D:", e)

    print(f"    DEBUG: Saving trained embeddings and model...")
    try:
        np.save(FILE_PATHS["embeddings_trained"], Z_trained)
        print(f"    DEBUG: Saved embeddings to embeddings_trained.npy")
        torch.save(encoder.state_dict(), FILE_PATHS["triplet_trained_encoder"])
        print(f"    DEBUG: Saved model state dict to triplet_trained_encoder.pth")
    except Exception as e:
        print(f"    ERROR: Failed to save embeddings or model: {e}")
           
        traceback.print_exc()

    with open(FILE_PATHS["triplet_run_stats"], "w", encoding="utf-8") as f:
        json.dump({
            "bm25_sizes": {k: len(v) for k, v in matched_dict.items()},
            "device": device,
            "proj_dim": encoder.out_dim,
            "triplets_count": len(triplets),
            "training_epochs": 5,
            "margin": 0.8,
            "silhouette_improvement": float(cluster_eval_trained['silhouette_true_labels'] - cluster_eval_raw['silhouette_true_labels'])
        }, f, ensure_ascii=False, indent=2)
    

    print(f"Silhouette Score: {cluster_eval_trained['silhouette_true_labels'] - cluster_eval_raw['silhouette_true_labels']:+.4f}")
    
    
    
    df_articles = df_clean  
    
    model_original = BertModel.from_pretrained('bert-base-uncased').to(device)
    model_original.eval()
    
    class SentenceEncoderAdapter:
        def __init__(self, sentence_encoder):
            self.sentence_encoder = sentence_encoder
            self.config = type('Config', (), {'hidden_size': sentence_encoder.out_dim})()
        
        def __call__(self, input_ids, attention_mask):
            tokens = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            encoded = self.sentence_encoder.encode_tokens(tokens)
            class MockOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state
            batch_size = encoded.shape[0]
            hidden_size = encoded.shape[1]
            seq_len = input_ids.shape[1]
            repeated = encoded.unsqueeze(1).repeat(1, seq_len, 1)
            return MockOutput(repeated)
    
    model_finetuned_adapter = SentenceEncoderAdapter(encoder)
    model_finetuned_adapter.eval = lambda: None  
    
    cls_vectors_before = get_all_cls_vectors(df_articles, model_original, tokenizer, device).cpu()
    cls_vectors_after = get_all_cls_vectors(df_articles, model_finetuned_adapter, tokenizer, device).cpu()
    cls_vectors_after_cpu = cls_vectors_after.cpu().numpy()
    
    perplexity_before = min(30, max(5, len(cls_vectors_before) // 3))
    perplexity_after = min(30, max(5, len(cls_vectors_after_cpu) // 3))
    
    tsne_before = TSNE(
        n_components=2, 
        perplexity=perplexity_before, 
        random_state=42,
        max_iter=get_config("tsne_max_iter")
    )
    tsne_after = TSNE(
        n_components=2, 
        perplexity=perplexity_after, 
        random_state=42,
        max_iter=get_config("tsne_max_iter")
    )
    projected_2d_before = tsne_before.fit_transform(cls_vectors_before.numpy())
    projected_2d_after = tsne_after.fit_transform(cls_vectors_after_cpu)
    
    print(f"    DEBUG: projected_2d_before shape: {projected_2d_before.shape}")
    print(f"    DEBUG: projected_2d_after shape: {projected_2d_after.shape}")
    
    group_centers = {}
    print(f"    DEBUG: matched_dict keys: {list(matched_dict.keys())}")
    print(f"    DEBUG: df_articles shape: {df_articles.shape}, index range: {df_articles.index.min()} - {df_articles.index.max()}")
    print(f"    DEBUG: projected_2d_after shape: {projected_2d_after.shape}")
    
    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            continue  
            
        if len(indices) > 0:
            print(f"    DEBUG: {group_name}: {indices[:5]}...")
            

            valid_indices = [i for i in indices if i < len(projected_2d_after)]
            print(f"    DEBUG: {group_name}: {len(valid_indices)} ({len(indices)})")
            
            if len(valid_indices) > 0:
                group_2d_points = projected_2d_after[valid_indices]
                group_center_2d = np.mean(group_2d_points, axis=0)
                group_centers[group_name] = group_center_2d
                print(f"     Group {group_name} center: {group_center_2d}")
            else:
                print(f"        {group_name} 2D")
        else:
            print(f"        {group_name} ")
    
    print(f"    DEBUG: group_centers: {group_centers}")

    def create_plotly_figure(projected_2d, title, is_after=False, highlighted_indices=None, group_centers=None, matched_dict_param=None):
        print(f"    DEBUG: create_plotly_figure called with projected_2d shape: {projected_2d.shape}")
        print(f"    DEBUG: projected_2d length: {len(projected_2d)}")
        print(f"    DEBUG: projected_2d sample: {projected_2d[:3] if len(projected_2d) >= 3 else projected_2d}")
        
        fig = go.Figure()
        

        article_indices = list(range(len(projected_2d)))
        hover_texts = [f"Article {idx}" for idx in article_indices]
        custom_data = [[idx] for idx in article_indices]
        
        bg_style = PLOT_STYLES["background"]
        fig.add_trace(go.Scatter(
            x=projected_2d[:, 0],
            y=projected_2d[:, 1],
            mode='markers',
            name="All Documents",
            marker=dict(
                color=bg_style["color"],
                size=bg_style["size"],
                opacity=bg_style["opacity"],
                symbol="circle",  
                line=dict(width=bg_style["line_width"], color=bg_style["line_color"])
            ),
            customdata=custom_data,
            hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            hovertext=hover_texts
        ))
        

        if highlighted_indices:

            all_x = projected_2d[:, 0]
            all_y = projected_2d[:, 1]
            
            highlighted_x = [all_x[i] for i in highlighted_indices if i < len(all_x)]
            highlighted_y = [all_y[i] for i in highlighted_indices if i < len(all_y)]
            
            if highlighted_x and highlighted_y:
                core_style = PLOT_STYLES["core"]
                fig.add_trace(go.Scatter(
                    x=highlighted_x,
                    y=highlighted_y,
                    mode='markers',
                    name="Highlighted",
                    marker=dict(
                        color=core_style["color"],
                        size=core_style["size"],
                        symbol=core_style["symbol"],
                        line=dict(width=core_style["line_width"], color=core_style["line_color"])
                    ),
                    customdata=[[i] for i in highlighted_indices if i < len(all_x)],
                    hovertemplate='<b>Article %{customdata[0]}</b><extra></extra>'
                ))
        

        print(f"    DEBUG: is_after={is_after}, group_centers={group_centers}")
        if is_after and group_centers:
            print(f"    DEBUG: {len(group_centers)} ")
            center_style = PLOT_STYLES["center"]
            print(f"    DEBUG: center_style = {center_style}")
            for group_name, center_2d in group_centers.items():

                color = get_group_color(group_name)
                print(f"    DEBUG: {group_name}: {center_2d}, {color}")
                fig.add_trace(go.Scatter(
                    x=[center_2d[0]],
                    y=[center_2d[1]],
                    mode='markers+text',
                    name=f'Center: {group_name}',
                    marker=dict(
                        color=color,
                        size=center_style["size"],
                        symbol=center_style["symbol"],
                        opacity=center_style["opacity"],
                        line=dict(width=center_style["line_width"], color=center_style["line_color"])
                    ),
                    text=[group_name],
                    textposition="top center",
                    textfont=dict(size=12, color=color, family='Arial Black'),
                    hovertemplate=f'<b>Group Center: {group_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ))
        else:
            print(f"    DEBUG: is_after={is_after}, group_centers={group_centers}")
        
        layout_style = PLOT_STYLES["layout"]
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            showlegend=True,
            hovermode='closest',
            plot_bgcolor=layout_style["plot_bgcolor"],
            paper_bgcolor=layout_style["paper_bgcolor"],
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(**layout_style["xaxis"], title="X"),
            yaxis=dict(**layout_style["yaxis"], title="Y")
        )
        
        print(f"    DEBUG: fig.layout.xaxis.showgrid = {fig.layout.xaxis.showgrid}")
        print(f"    DEBUG: fig.layout.xaxis.showline = {fig.layout.xaxis.showline}")
        print(f"    DEBUG: fig.layout.xaxis.mirror = {fig.layout.xaxis.mirror}")
        
        return fig

    fig_before = create_plotly_figure(projected_2d_before, "2D Projection Before Finetuning", False, None, None, None)
    fig_after = create_plotly_figure(projected_2d_after, "2D Projection After Finetuning", True, None, group_centers, matched_dict_for_display)
    

    
    return fig_before, fig_after


def run_training_with_highlights(highlighted_indices):

    global df
    
    if isinstance(highlighted_indices, tuple):
        highlighted_indices = list(highlighted_indices)
    
    model_original = BertModel.from_pretrained('bert-base-uncased').to(device)
    model_finetuned = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model_save_path = FILE_PATHS["bert_finetuned"]
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device)
        model_finetuned.load_state_dict(checkpoint['model_state_dict'])
        model_finetuned.eval()
    
    if "df_global" not in globals():
        df_articles = pd.read_csv(FILE_PATHS["csv_path"])
    else:
        df_articles = df
    
    matched_dict = {}
    bm25_results_path = FILE_PATHS["bm25_search_results"]
    if os.path.exists(bm25_results_path):
        with open(bm25_results_path, "r", encoding="utf-8") as f:
            matched_dict = json.load(f)
        print(f"    DEBUG: Loaded matched_dict with keys: {list(matched_dict.keys())}")
    else:
        print(f"    DEBUG: No bm25_search_results.json found at {bm25_results_path}")
    

    cls_vectors_before = get_all_cls_vectors(df_articles, model_original, tokenizer, device).cpu()
    cls_vectors_after = get_all_cls_vectors(df_articles, model_finetuned, tokenizer, device).cpu()
    cls_vectors_after_cpu = cls_vectors_after.cpu().numpy()
    

    perplexity_before = min(30, max(5, len(cls_vectors_before) // 3))
    perplexity_after = min(30, max(5, len(cls_vectors_after_cpu) // 3))
    
    tsne_before = TSNE(
        n_components=2, 
        perplexity=perplexity_before, 
        random_state=42,
        max_iter=get_config("tsne_max_iter")
    )
    tsne_after = TSNE(
        n_components=2, 
        perplexity=perplexity_after, 
        random_state=42,
        max_iter=get_config("tsne_max_iter")
    )
    projected_2d_before = tsne_before.fit_transform(cls_vectors_before.numpy())
    projected_2d_after = tsne_after.fit_transform(cls_vectors_after_cpu)
    

    group_centers = {}
    if matched_dict:
        print(f"    DEBUG: Calculating group centers for highlights function")
        for group_name, indices in matched_dict.items():
            if group_name == "Other":
                continue  
                
            if len(indices) > 0:
                valid_indices = [i for i in indices if i < len(projected_2d_after)]
                if len(valid_indices) > 0:
                    group_2d_points = projected_2d_after[valid_indices]
                    group_center_2d = np.mean(group_2d_points, axis=0)
                    group_centers[group_name] = group_center_2d
                    print(f"     {group_name} center: {group_center_2d}")
    
    print(f"    DEBUG: Final group_centers for highlights: {group_centers}")
    

    def create_plotly_figure_with_highlights(projected_2d, title, highlighted_indices=None, group_centers=None):
        fig = go.Figure()
        

        article_indices = list(range(len(projected_2d)))
        hover_texts = [f"Article {idx}" for idx in article_indices]
        custom_data = [[idx] for idx in article_indices]
        
        bg_style = PLOT_STYLES["background"]
        fig.add_trace(go.Scatter(
            x=projected_2d[:, 0],
            y=projected_2d[:, 1],
            mode='markers',
            name="Articles",
            marker=dict(
                color=bg_style["color"],
                size=bg_style["size"],
                opacity=bg_style["opacity"],
                symbol="circle",  
                line=dict(width=bg_style["line_width"], color=bg_style["line_color"])
            ),
            customdata=custom_data,
            hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            hovertext=hover_texts
        ))
        

        if highlighted_indices:
            highlighted_x = []
            highlighted_y = []
            highlighted_customdata = []
            highlighted_hovertexts = []
            
            for idx in highlighted_indices:
                if 0 <= idx < len(projected_2d):
                    highlighted_x.append(projected_2d[idx, 0])
                    highlighted_y.append(projected_2d[idx, 1])
                    highlighted_customdata.append([idx])
                    highlighted_hovertexts.append(f"Article {idx}")
            
            if highlighted_x:  
                core_style = PLOT_STYLES["core"]
                fig.add_trace(go.Scatter(
                    x=highlighted_x,
                    y=highlighted_y,
                    mode='markers',
                    name='Highlighted Articles',
                    marker=dict(
                        color=core_style["color"],
                        size=core_style["size"],
                        symbol=core_style["symbol"],
                        line=dict(width=core_style["line_width"], color=core_style["line_color"])
                    ),
                    customdata=highlighted_customdata,
                    hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    hovertext=highlighted_hovertexts
                ))
        

        if "After" in title and group_centers:
            print(f"    DEBUG: Adding {len(group_centers)} prototype centers to {title}")
            center_style = PLOT_STYLES["center"]
            for group_name, center_2d in group_centers.items():
                color = get_group_color(group_name)  
                print(f"    DEBUG: Adding {group_name} center: {center_2d}, color: {color}")
                fig.add_trace(go.Scatter(
                    x=[center_2d[0]],
                    y=[center_2d[1]],
                    mode='markers+text',
                    name=f'Center: {group_name}',
                    marker=dict(
                        color=color,
                        size=center_style["size"],
                        symbol=center_style["symbol"],
                        opacity=center_style["opacity"],
                        line=dict(width=center_style["line_width"], color=center_style["line_color"])
                    ),
                    text=[group_name],
                    textposition="top center",
                    textfont=dict(size=12, color=color, family='Arial Black'),
                    hovertemplate=f'<b>Group Center: {group_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ))
        
        layout_style = PLOT_STYLES["layout"]
        fig.update_layout(
            title=title,
            hovermode='closest',
            showlegend=True,
            plot_bgcolor=layout_style["plot_bgcolor"],
            paper_bgcolor=layout_style["paper_bgcolor"],
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(**layout_style["xaxis"], title='TSNE Dimension 1'),
            yaxis=dict(**layout_style["yaxis"], title='TSNE Dimension 2')
        )
        
        return fig
    
    fig_before = create_plotly_figure_with_highlights(projected_2d_before, "Before Training", highlighted_indices, None)
    fig_after = create_plotly_figure_with_highlights(projected_2d_after, "After Training", highlighted_indices, group_centers)
    
    return fig_before, fig_after


_KEYWORD_TSNE_CACHE = None


_ARTICLES_CACHE = {}


_DOCUMENTS_2D_CACHE = {}

def clear_caches():

    global _ARTICLES_CACHE, _DOCUMENTS_2D_CACHE
    _ARTICLES_CACHE.clear()
    _DOCUMENTS_2D_CACHE.clear()
    print("Cleared all caches due to data change")
    
    try:
        filtered_path = FILE_PATHS.get("filtered_group_assignment")
        if filtered_path and os.path.exists(filtered_path):
            os.remove(filtered_path)
            print(f"    Removed old filtered_group_assignment.json")
    except Exception as e:
        print(f"     Could not remove filtered_group_assignment.json: {e}")
    
 
    try:
        user_finetuned_path = FILE_PATHS.get("user_finetuned_list")
        if user_finetuned_path and os.path.exists(user_finetuned_path):
            os.remove(user_finetuned_path)

    except Exception as e:
        print(f"     Could not remove user_finetuned_list.json: {e}")




@app.callback(
    Output('keywords-2d-plot', 'figure'),
    Input('keywords-2d-plot', 'id')  
)
def update_keywords_2d_plot(plot_id):
    global GLOBAL_OUTPUT_DICT, GLOBAL_KEYWORDS, _KEYWORD_TSNE_CACHE
    
    if not GLOBAL_OUTPUT_DICT or not GLOBAL_KEYWORDS:
        print(f"    Keywords 2D Plot: GLOBAL_OUTPUT_DICT={bool(GLOBAL_OUTPUT_DICT)}, GLOBAL_KEYWORDS={bool(GLOBAL_KEYWORDS)}")
        if GLOBAL_KEYWORDS:
            print(f"    GLOBAL_KEYWORDS length: {len(GLOBAL_KEYWORDS)}")
        if GLOBAL_OUTPUT_DICT:
            print(f"    GLOBAL_OUTPUT_DICT keys: {list(GLOBAL_OUTPUT_DICT.keys())}")
        return {
            'data': [],
            'layout': {
                'title': 'No keywords available',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }
    

    try:
        def adjust_text_positions(x_coords, y_coords, keywords, min_distance=0.12):

            import numpy as np
            adjusted_x = x_coords.copy()
            adjusted_y = y_coords.copy()
            

            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            min_dist = min(x_range, y_range) * min_distance
            
            def find_empty_space(current_x, current_y, all_x, all_y, search_radius=0.3):

                search_dist = min(x_range, y_range) * search_radius
                

                for angle in np.linspace(0, 2*np.pi, 16):  
                    for radius in np.linspace(min_dist, search_dist, 8):  
                        test_x = current_x + radius * np.cos(angle)
                        test_y = current_y + radius * np.sin(angle)
                        
                        too_close = False
                        for other_x, other_y in zip(all_x, all_y):
                            if abs(test_x - other_x) < min_dist and abs(test_y - other_y) < min_dist:
                                too_close = True
                                break
                        
                        if not too_close:
                            return test_x, test_y
                
                return current_x + np.random.normal(0, min_dist*0.5), current_y + np.random.normal(0, min_dist*0.5)
            
            for iteration in range(30):
                overlaps_found = 0
                
                for i in range(len(adjusted_x)):
                    has_overlap = False
                    for j in range(len(adjusted_x)):
                        if i != j:
                            if (abs(adjusted_x[i] - adjusted_x[j]) < min_dist and 
                                abs(adjusted_y[i] - adjusted_y[j]) < min_dist):
                                has_overlap = True
                                break
                    
                    if has_overlap:
                        new_x, new_y = find_empty_space(
                            x_coords[i], y_coords[i],  
                            adjusted_x, adjusted_y
                        )
                        adjusted_x[i] = new_x
                        adjusted_y[i] = new_y
                        overlaps_found += 1
                
                if overlaps_found == 0:
                    print(f"Text positioning converged after {iteration + 1} iterations")
                    break
            
            return adjusted_x, adjusted_y
        
        if _KEYWORD_TSNE_CACHE is None:
            print("Computing t-SNE for keywords (first time)...")
            keyword_embeddings = embedding_model_kw.encode(GLOBAL_KEYWORDS, convert_to_tensor=True).to(device).cpu().numpy()
            
            perplexity = min(30, max(5, len(keyword_embeddings) // 3))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_embeddings = tsne.fit_transform(keyword_embeddings)
            
            x_coords = reduced_embeddings[:, 0]
            y_coords = reduced_embeddings[:, 1]
            x_coords_adjusted, y_coords_adjusted = adjust_text_positions(x_coords, y_coords, GLOBAL_KEYWORDS)
            
            _KEYWORD_TSNE_CACHE = {
                'keywords': GLOBAL_KEYWORDS.copy(),
                'embeddings': reduced_embeddings.copy(),
                'adjusted_x': x_coords_adjusted.copy(),
                'adjusted_y': y_coords_adjusted.copy()
            }
            print("t-SNE computation and text positioning completed and cached")
        else:
            print("Using cached t-SNE results and text positions")
            reduced_embeddings = _KEYWORD_TSNE_CACHE['embeddings']
            x_coords_adjusted = _KEYWORD_TSNE_CACHE['adjusted_x']
            y_coords_adjusted = _KEYWORD_TSNE_CACHE['adjusted_y']
        
        hover_texts = GLOBAL_KEYWORDS
        
        keyword_colors = ['#2196F3'] * len(GLOBAL_KEYWORDS)
        

        
        fig = {
            'data': [{
                'x': x_coords_adjusted,
                'y': y_coords_adjusted,
                'mode': 'markers+text',  
                'type': 'scatter',
                'marker': {
                    'size': 1,  
                    'color': keyword_colors,
                    'opacity': 0.0  
                },
                'text': GLOBAL_KEYWORDS,  
                'textfont': {
                    'size': 10,  
                    'color': keyword_colors  
                },
                'textposition': 'middle center',
                'hovertext': hover_texts,
                'hoverinfo': 'text',
                'customdata': GLOBAL_KEYWORDS,  
                'hovertemplate': '<b>%{hovertext}</b><extra></extra>',
                'opacity': 1.0  
            }],
            'layout': {
                'title': 'Keywords 2D Visualization',
                'xaxis': {
                    'title': 'Dimension 1',
                    'showgrid': True,
                    'gridcolor': '#f0f0f0'
                },
                'yaxis': {
                    'title': 'Dimension 2', 
                    'showgrid': True,
                    'gridcolor': '#f0f0f0'
                },
                'hovermode': 'closest',
                'clickmode': 'event',  
                'dragmode': 'pan',
                'showlegend': False,
                'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},  
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'size': 10}  
            }
        }
        
        return fig
        
    except Exception as e:
        print(f"Error creating 2D plot: {e}")
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {
                    'title': 'X',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'yaxis': {
                    'title': 'Y',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12},
                'title': {
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            }
        }


app.clientside_callback(
    """
    function(group_data) {
        // Define group colors
        const GROUP_COLORS = {
            "Group 1": "#FF6B6B",
            "Group 2": "#32CD32", 
            "Group 3": "#FF8C00",
            "Group 4": "#8B4513",
            "Group 5": "#FFD700",
            "Group 6": "#8A2BE2",
            "Group 7": "#DC143C",
            "Group 8": "#228B22",
            "Group 9": "#FF1493",
            "Group 10": "#800080",
            "Other": "#A9A9A9"
        };
        
        const DEFAULT_COLOR = "#2196F3";
        
        // Get current figure
        const current_figure = window.dash_clientside.callback_context.states['keywords-2d-plot.figure'];
        if (!current_figure || !current_figure.data || !current_figure.data[0]) {
            return window.dash_clientside.no_update;
        }
        
        // Clone the figure to avoid mutation
        const new_figure = JSON.parse(JSON.stringify(current_figure));
        const text_data = new_figure.data[0].text;
        
        if (!text_data) {
            return window.dash_clientside.no_update;
        }
        
        // Update colors based on group assignment
        const new_colors = text_data.map(keyword => {
            if (group_data && group_data[keyword]) {
                return GROUP_COLORS[group_data[keyword]] || DEFAULT_COLOR;
            }
            return DEFAULT_COLOR;
        });
        
        // Update the textfont color and ensure no fading/opacity issues
        new_figure.data[0].textfont.color = new_colors;
        
        // Explicitly set opacity to prevent any fading effects
        new_figure.data[0].opacity = 1.0;
        
        // Ensure marker properties don't cause fading if present
        if (new_figure.data[0].marker) {
            new_figure.data[0].marker.opacity = 1.0;
        }
        
        return new_figure;
    }
    """,
    Output('keywords-2d-plot', 'figure', allow_duplicate=True),
    Input('group-data', 'data'),
    State('keywords-2d-plot', 'figure'),
    prevent_initial_call=True
)





@app.callback(
    Output('documents-2d-plot', 'figure'),
    Input('main-visualization-area', 'children'),  
    [State('display-mode', 'data'),
     State('training-figures', 'data')],  
    prevent_initial_call=True
)
def update_documents_2d_plot_initial(layout_children, display_mode, training_figures):

    global df
    
    
    if display_mode != "keywords":
        print(f"    DEBUG:         NOT IN KEYWORDS MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'keywords'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      KEYWORDS MODE CONFIRMED")
    
    if display_mode == "training":
        print(f"    DEBUG:         TRAINING MODE DETECTED:")
        print(f"    DEBUG:   This should prevent the 'nonexistent object' error")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      NOT IN TRAINING MODE")
    
    if display_mode is None or display_mode not in ["keywords"]:
        print(f"    DEBUG:         UNEXPECTED DISPLAY MODE:")
        print(f"    DEBUG:   display_mode: {display_mode}")
        print(f"    DEBUG:   display_mode is None: {display_mode is None}")
        print(f"    DEBUG:   display_mode not in ['keywords']: {display_mode not in ['keywords']}")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      ALL SAFETY CHECKS PASSED")
    print(f"    DEBUG:     Proceeding with documents 2D visualization...")
    
    if 'df' not in globals():
        return {
            'data': [],
            'layout': {
                'title': 'No data available',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }
    
    try:

        print("    Initial documents 2D visualization calculation...")
        all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
        
        print("    Truncating long texts to fit within model limits...")
        truncated_articles = [truncate_text_for_model(text, max_length=500) for text in all_articles_text]
        
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(truncated_articles), batch_size):
            batch_texts = truncated_articles[i:i + batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(truncated_articles) + batch_size - 1)//batch_size}")
            
            batch_embeddings = safe_encode_batch(batch_texts, embedding_model_kw, device)
            all_embeddings.extend(batch_embeddings)
        
        document_embeddings = np.array(all_embeddings)
        
        print("    Calculating TSNE for initial documents visualization...")
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        document_2d = tsne.fit_transform(document_embeddings)
        document_2d = document_2d.tolist()
        
        print(f"    Initial document_2d length: {len(document_2d)}")
        print(f"    df length: {len(df)}")
        
        if len(document_2d) != len(df):
            print(f"    WARNING: Initial length mismatch! Adjusting...")
            if len(document_2d) < len(df):
                padding_needed = len(df) - len(document_2d)
                print(f"    Padding initial document_2d with {padding_needed} zero points")
                for _ in range(padding_needed):
                    document_2d.append([0.0, 0.0])
            elif len(document_2d) > len(df):
                print(f"    Truncating initial document_2d from {len(document_2d)} to {len(df)}")
                document_2d = document_2d[:len(df)]
        
        print(f"    Final initial document_2d length: {len(document_2d)}")
        
        bg_style = PLOT_STYLES["background"]
        traces = [{
            'x': [document_2d[i][0] for i in range(len(df))],
            'y': [document_2d[i][1] for i in range(len(df))],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'All documents',
            'marker': {
                'size': bg_style["size"],
                'color': bg_style["color"],
                'opacity': bg_style["opacity"],
                'line': {'width': bg_style["line_width"], 'color': bg_style["line_color"]}
            },
            'text': [f'Doc {i+1}' for i in range(len(df))],
            'customdata': [[i] for i in range(len(df))],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        }]
        
        layout_style = PLOT_STYLES["layout"]
        fig = {
            'data': traces,
            'layout': {
                'title': {
                    'text': 'Documents 2D Visualization - All Documents',
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'TSNE Dimension 1',
                    **layout_style["xaxis"]
                },
                'yaxis': {
                    'title': 'TSNE Dimension 2',
                    **layout_style["yaxis"]
                },
                'hovermode': 'closest',
                'showlegend': True,
                'legend': {
                    'x': 0.02,
                    'y': 0.98,
                    'bgcolor': 'rgba(255, 255, 255, 0.8)',
                    'bordercolor': '#2c3e50',
                    'borderwidth': 1
                },
                'plot_bgcolor': layout_style["plot_bgcolor"],
                'paper_bgcolor': layout_style["paper_bgcolor"],
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12}
            }
        }
        
        print("    Initial documents 2D visualization created successfully")
        return fig
        
    except Exception as e:
        print(f"Error creating initial documents 2D plot: {e}")
        layout_style = PLOT_STYLES["layout"]
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {
                    'title': 'X',
                    **layout_style["xaxis"]
                },
                'yaxis': {
                    'title': 'Y',
                    **layout_style["yaxis"]
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12},
                'title': {
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            }
        }



@app.callback(
    [Output('documents-2d-plot', 'figure', allow_duplicate=True),
     Output('highlighted-indices', 'data', allow_duplicate=True)],
    [Input('selected-keyword', 'data'),
     Input('selected-group', 'data'),  
     Input('selected-article', 'data')],  
    State('group-order', 'data'),  
    State('display-mode', 'data'),
    prevent_initial_call=True
)
def update_documents_2d_plot(selected_keyword, selected_group, selected_article, group_order, display_mode):

    global df, _DOCUMENTS_2D_CACHE
    
    if display_mode != "keywords":

        raise PreventUpdate
    
    
    if display_mode == "training":

        raise PreventUpdate
    
    print(f"    DEBUG:      NOT IN TRAINING MODE")
    
    if display_mode is None or display_mode not in ["keywords"]:
        print(f"    DEBUG:         UNEXPECTED DISPLAY MODE:")
        print(f"    DEBUG:   display_mode: {display_mode}")
        print(f"    DEBUG:   display_mode is None: {display_mode is None}")
        print(f"    DEBUG:   display_mode not in ['keywords']: {display_mode not in ['keywords']}")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      ALL SAFETY CHECKS PASSED")
    print(f"    DEBUG:     Proceeding with documents 2D plot update...")
    
    print(f"    update_documents_2d_plot called with:")
    print(f"  selected_keyword: {selected_keyword}")
    print(f"  selected_group: {selected_group}")
    print(f"  selected_article: {selected_article}")
    print(f"  group_order: {group_order}")
    
    if 'df' not in globals():
        print("        No df in globals")
        return {
            'data': [],
            'layout': {
                'title': 'No data available',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }, []
    
    cache_key = None
    if selected_keyword:
        cache_key = f"docs_keyword:{selected_keyword}"
        print(f"    DEBUG: Created cache key for keyword: {cache_key}")
    elif selected_group and group_order:
        for group_name, keywords in group_order.items():
            if group_name == selected_group:
                cache_key = f"docs_group:{group_name}:{':'.join(sorted(keywords))}"
                print(f"    DEBUG: Created cache key for group: {cache_key}")
                break
    else:
        cache_key = "docs_default"
        print(f"    DEBUG: Using default cache key: {cache_key}")
    
    if selected_article is not None:
        cache_key = f"{cache_key}_article:{selected_article}"
        print(f"    DEBUG: Added article to cache key: {cache_key}")
    
    print(f"    DEBUG: Final cache key: {cache_key}")
    print(f"    DEBUG: Cache contains key: {cache_key in _DOCUMENTS_2D_CACHE}")
    
    if cache_key and cache_key in _DOCUMENTS_2D_CACHE:
        print(f"    DEBUG: Using cached documents 2D plot for: {cache_key}")
        cached_fig = _DOCUMENTS_2D_CACHE[cache_key]
        highlighted_indices = []
        if selected_keyword or selected_group or selected_article is not None:
            try:
                for trace in cached_fig.get('data', []):
                    if trace.get('name') in ['Keyword/Group matches', 'Selected Article']:
                        if 'customdata' in trace:
                            for data in trace['customdata']:
                                if isinstance(data, list) and len(data) > 0:
                                    highlighted_indices.append(data[0])
            except:
                pass
        print(f"    DEBUG: Returning cached result with {len(highlighted_indices)} highlighted indices")
        return cached_fig, highlighted_indices
    else:
        print(f"    DEBUG: Cache miss for key: {cache_key}")
    
    try:
        print("    Using pre-computed document embeddings and t-SNE...")
        
        if _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
            document_embeddings = _GLOBAL_DOCUMENT_EMBEDDINGS
            document_2d = _GLOBAL_DOCUMENT_TSNE.tolist()
            print(f"    Using cached embeddings, shape: {document_embeddings.shape}")
            print(f"    Using cached t-SNE, shape: {_GLOBAL_DOCUMENT_TSNE.shape}")
            
            print(f"    Cached document_2d length: {len(document_2d)}")
            print(f"    df length: {len(df)}")
            
            if len(document_2d) != len(df):
                print(f"    WARNING: Cached length mismatch! Adjusting...")
                if len(document_2d) < len(df):
                    padding_needed = len(df) - len(document_2d)
                    print(f"    Padding cached document_2d with {padding_needed} zero points")
                    for _ in range(padding_needed):
                        document_2d.append([0.0, 0.0])
                elif len(document_2d) > len(df):
                    print(f"    Truncating cached document_2d from {len(document_2d)} to {len(df)}")
                    document_2d = document_2d[:len(df)]
            
            print(f"    Final cached document_2d length: {len(document_2d)}")
        else:
            print("    Pre-computed embeddings not available, computing on-demand...")
        print(f"    df shape: {df.shape}")
        all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
        print(f"    Number of articles: {len(all_articles_text)}")
        
        print("    Truncating long texts to fit within model limits...")
        truncated_articles = [truncate_text_for_model(text, max_length=500) for text in all_articles_text]
        
        batch_size = 64 if device == "cpu" else 128
        all_embeddings = []
        
        for i in range(0, len(truncated_articles), batch_size):
            batch_texts = truncated_articles[i:i + batch_size]
            print(f"    Processing batch {i//batch_size + 1} for documents 2D visualization")
            
            batch_embeddings = safe_encode_batch(batch_texts, embedding_model_kw, device)
            all_embeddings.extend(batch_embeddings)
        
        document_embeddings = np.array(all_embeddings)
        
        print("    Calculating TSNE for documents...")
        print(f"    Embeddings shape: {document_embeddings.shape}")
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        print(f"    Perplexity: {perplexity}")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        document_2d = tsne.fit_transform(document_embeddings)
        print(f"    TSNE result shape: {document_2d.shape}")
        print(f"    TSNE result type: {type(document_2d)}")
        print(f"    TSNE result dtype: {document_2d.dtype}")
        document_2d = document_2d.tolist()
        
        print(f"    document_2d length: {len(document_2d)}")
        print(f"    df length: {len(df)}")
        print(f"    all_articles_text length: {len(all_articles_text)}")
        
        if len(document_2d) != len(df):
            print(f"    WARNING: Length mismatch! Adjusting...")
            if len(document_2d) < len(df):
                padding_needed = len(df) - len(document_2d)
                print(f"    Padding document_2d with {padding_needed} zero points")
                for _ in range(padding_needed):
                    document_2d.append([0.0, 0.0])
            elif len(document_2d) > len(df):
                print(f"    Truncating document_2d from {len(document_2d)} to {len(df)}")
                document_2d = document_2d[:len(df)]
        
        print(f"    Final document_2d length: {len(document_2d)}")
        
        highlight_mask = []
        highlight_reason = ""
        
        if selected_keyword:
            print(f"    Processing keyword: '{selected_keyword}'")
            for i in range(len(df)):
                text = str(df.iloc[i, 1]).lower()
                contains_keyword = selected_keyword.lower() in text
                highlight_mask.append(contains_keyword)
                if contains_keyword:
                    print(f"  Document {i+1} contains keyword '{selected_keyword}'")
            highlight_reason = f"Documents containing '{selected_keyword}'"
            print(f"    Found {sum(highlight_mask)} documents containing keyword '{selected_keyword}'")
        
        elif selected_group and group_order:
            group_keywords = []
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    group_keywords = keywords
                    break
            
            print(f"Selected group '{selected_group}' keywords: {group_keywords}")
            
            for i in range(len(df)):
                text = str(df.iloc[i, 1]).lower()
                contains_group_keyword = any(keyword.lower() in text for keyword in group_keywords)
                highlight_mask.append(contains_group_keyword)
            
            highlight_reason = f"Documents containing keywords from group '{selected_group}'"
        
        else:
            highlight_mask = [False] * len(df)
            highlight_reason = ""
        
        selected_article_mask = [False] * len(df)
        if selected_article is not None and selected_article < len(df):
            selected_article_mask[selected_article] = True
        
        traces = []
        
        keyword_group_indices = np.where(np.array(highlight_mask))[0]
        selected_article_indices = np.where(np.array(selected_article_mask))[0]
        
        all_highlighted = np.logical_or(np.array(highlight_mask), np.array(selected_article_mask))
        other_indices = np.where(~all_highlighted)[0]
        
        if len(keyword_group_indices) > 0:
            core_style = PLOT_STYLES["core"]
            traces.append({
                'x': [document_2d[i][0] for i in keyword_group_indices],
                'y': [document_2d[i][1] for i in keyword_group_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Keyword/Group matches',
                'marker': {
                    'size': core_style["size"],
                    'color': core_style["color"],
                    'symbol': core_style["symbol"],
                    'line': {'width': core_style["line_width"], 'color': core_style["line_color"]}
                },
                'text': [f'Doc {i+1}' for i in keyword_group_indices],
                'customdata': [[i] for i in keyword_group_indices],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
        
        if len(selected_article_indices) > 0:
            traces.append({
                'x': [document_2d[i][0] for i in selected_article_indices],
                'y': [document_2d[i][1] for i in selected_article_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Selected Article',
                'marker': {
                    'size': 20,
                    'color': '#FF0000',  
                    'symbol': 'star',
                    'line': {'width': 3, 'color': 'white'}
                },
                'text': [f'Doc {i+1}' for i in selected_article_indices],
                'customdata': [[i] for i in selected_article_indices],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
        
        if len(other_indices) > 0:
            bg_style = PLOT_STYLES["background"]
            traces.append({
                'x': [document_2d[i][0] for i in other_indices],
                'y': [document_2d[i][1] for i in other_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Other documents',
                'marker': {
                    'size': bg_style["size"],
                    'color': bg_style["color"],
                    'opacity': bg_style["opacity"],
                    'line': {'width': bg_style["line_width"], 'color': bg_style["line_color"]}
                },
                'text': [f'Doc {i+1}' for i in other_indices],
                'customdata': [[i] for i in other_indices],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
        if not traces:
            bg_style = PLOT_STYLES["background"]
            traces = [{
                'x': [document_2d[i][0] for i in range(len(df))],
                'y': [document_2d[i][1] for i in range(len(df))],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'All documents',
                'marker': {
                    'size': bg_style["size"],
                    'color': bg_style["color"],
                    'opacity': bg_style["opacity"],
                    'line': {'width': bg_style["line_width"], 'color': bg_style["line_color"]}
                },
                'text': [f'Doc {i+1}' for i in range(len(df))],
                'customdata': [[i] for i in range(len(df))],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            }]
        
        title_parts = []
        
        if selected_keyword:
            title_parts.append(f"Keyword: '{selected_keyword}'")
        elif selected_group and highlight_reason:
            title_parts.append(highlight_reason)
        
        if selected_article is not None:
            title_parts.append(f"Selected Article {selected_article + 1}")
        
        if title_parts:
            title = f"Documents 2D Visualization - {' | '.join(title_parts)}"
        else:
            title = "Documents 2D Visualization"
        
        fig = {
            'data': traces,
            'layout': {
                'title': {
                    'text': title,
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'TSNE Dimension 1',
                    **PLOT_STYLES["layout"]["xaxis"]
                },
                'yaxis': {
                    'title': 'TSNE Dimension 2',
                    **PLOT_STYLES["layout"]["yaxis"]
                },
                'hovermode': 'closest',
                'showlegend': True,
                'legend': {
                    'x': 0.02,
                    'y': 0.98,
                    'bgcolor': 'rgba(255, 255, 255, 0.8)',
                    'bordercolor': '#2c3e50',
                    'borderwidth': 1
                },
                'plot_bgcolor': PLOT_STYLES["layout"]["plot_bgcolor"],
                'paper_bgcolor': PLOT_STYLES["layout"]["paper_bgcolor"],
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12}
            }
        }
        
        highlighted_indices = []
        if len(keyword_group_indices) > 0:
            highlighted_indices.extend(keyword_group_indices.tolist())
        if len(selected_article_indices) > 0:
            highlighted_indices.extend(selected_article_indices.tolist())
        
        print(f"    Final result:")
        print(f"  Number of traces: {len(traces)}")
        print(f"  Highlighted indices: {highlighted_indices}")
        print(f"  Figure keys: {list(fig.keys())}")
        print(f"  First trace keys: {list(traces[0].keys()) if traces else 'No traces'}")
        print(f"  First trace x length: {len(traces[0]['x']) if traces and 'x' in traces[0] else 'No x'}")
        print(f"  First trace y length: {len(traces[0]['y']) if traces and 'y' in traces[0] else 'No y'}")
        print(f"  First trace x sample: {traces[0]['x'][:3] if traces and 'x' in traces[0] and len(traces[0]['x']) > 0 else 'No x data'}")
        print(f"  First trace y sample: {traces[0]['y'][:3] if traces and 'y' in traces[0] and len(traces[0]['y']) > 0 else 'No y data'}")
        
        if cache_key:
            _DOCUMENTS_2D_CACHE[cache_key] = fig
            print(f"Cached documents 2D plot for: {cache_key}")
        
        return fig, highlighted_indices
        
    except Exception as e:
        print(f"Error creating documents 2D plot: {e}")
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {
                    'title': 'X',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'yaxis': {
                    'title': 'Y',
                    'showgrid': True,
                    'gridcolor': '#e1e5e9',
                    'showline': True,
                    'linecolor': '#2c3e50',
                    'linewidth': 1,
                    'mirror': True,
                    'zeroline': True,
                    'zerolinecolor': '#2c3e50',
                    'zerolinewidth': 1
                },
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
                'font': {'size': 12},
                'title': {
                    'font': {'size': 16, 'color': '#2c3e50'},
                    'x': 0.5,
                    'xanchor': 'center'
                }
            }
        }, []

@app.callback(
    [Output("group-data", "data", allow_duplicate=True),
     Output("selected-keyword", "data", allow_duplicate=True)],
    Input("keywords-2d-plot", "clickData"),
    [State("selected-group", "data"),
     State("group-data", "data"),
     State("display-mode", "data")],
    prevent_initial_call=True
)
def handle_plot_click(click_data, selected_group, group_data, display_mode):

    print(f"    DEBUG: handle_plot_click called")
    print(f"    DEBUG: click_data: {click_data}")
    print(f"    DEBUG: selected_group: {selected_group}")
    print(f"    DEBUG: display_mode: {display_mode}")
    print(f"    DEBUG: click_data type: {type(click_data)}")
    print(f"    DEBUG: selected_group type: {type(selected_group)}")
    print(f"    DEBUG: display_mode type: {type(display_mode)}")
    
    if not click_data:
        print(f"    DEBUG: handle_plot_click exit: no click_data")
        raise PreventUpdate
    
    try:
        clicked_keyword = click_data['points'][0]['customdata']
        print(f"Clicked keyword: {clicked_keyword}")
        
        if selected_group:
            new_data = dict(group_data) if group_data else {}
            if clicked_keyword in new_data and new_data[clicked_keyword]:
                if new_data[clicked_keyword] != selected_group:
                    print(f"Moved keyword '{clicked_keyword}' from group '{new_data[clicked_keyword]}' to group '{selected_group}'")
                else:
                    print(f"Keyword '{clicked_keyword}' is already in group '{selected_group}'")
            else:
                print(f"Added keyword '{clicked_keyword}' to group '{selected_group}'")
            new_data[clicked_keyword] = selected_group
            
            if display_mode == "training":
                print(f"    Training mode: not updating selected-keyword to avoid documents-2d-plot error")
                return new_data, dash.no_update
            else:
                return new_data, clicked_keyword  
        else:
            print(f"Selected keyword for highlighting: {clicked_keyword}")
            
            if display_mode == "training":
                print(f"    Training mode: not updating selected-keyword to avoid documents-2d-plot error")
                return group_data, dash.no_update
            else:
                return group_data, clicked_keyword
        
    except Exception as e:
        print(f"Error handling plot click: {e}")
        raise PreventUpdate

@app.callback(
    [Output("train-btn", "children"),
     Output("train-btn", "style"),
     Output("train-btn", "disabled"),
     Output("switch-view-btn", "style"),
     Output("display-mode", "data"),
     Output("training-figures", "data")],
    Input("train-btn", "n_clicks"),
    State("group-order", "data"),
    prevent_initial_call=True
)
def handle_train_button(n_clicks, group_order):
    
    print("=" * 60)
    print("TRAIN BUTTON CALLBACK TRIGGERED!")
    print("=" * 60)
    print(f"Train button clicked, n_clicks: {n_clicks}")
    print(f"Group order data: {group_order}")
    print(f"Current time: {__import__('datetime').datetime.now()}")
    
    import sys
    sys.stdout.flush()
    
    if not n_clicks or n_clicks == 0:
        print("No clicks, preventing update")
        raise PreventUpdate
    
    normal_style = {
        "margin-top": "20px",
        "padding": "10px 20px",
        "fontSize": "16px",
        "backgroundColor": "#4CAF50",
        "color": "white",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "pointer"
    }
    
    training_style = {
        "margin-top": "20px",
        "padding": "10px 20px",
        "fontSize": "16px",
        "backgroundColor": "#FF9800",  
        "color": "white",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "not-allowed",
        "animation": "pulse 1.5s infinite"
    }
    
    if not group_order:
        print("No group data available")
        empty_fig = {
            'data': [],
            'layout': {
                'title': 'No group data available for training',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }
        return "Train", normal_style, False, {"display": "none"}, "keywords", {"before": empty_fig, "after": empty_fig}
    
    try:
        print("Starting training process...")
        print(f"    DEBUG: Group order data: {group_order}")
        print(f"    DEBUG: Final list path: {FILE_PATHS['final_list_path']}")
        
        print("Saving group data to final_list.json...")
        with open(FILE_PATHS["final_list_path"], "w", encoding="utf-8") as f:
            json.dump(group_order, f, indent=4, ensure_ascii=False)
        print(f"Group data saved to {FILE_PATHS['final_list_path']}")
        
        if os.path.exists(FILE_PATHS["final_list_path"]):
            with open(FILE_PATHS["final_list_path"], "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            print(f"    DEBUG: Verified saved data: {saved_data}")
        else:
            print(f"    ERROR: Failed to save group data to {FILE_PATHS['final_list_path']}")
            raise FileNotFoundError(f"Could not save group data to {FILE_PATHS['final_list_path']}")
        
        training_fig = {
            'data': [],
            'layout': {
                'title': 'Training in Progress...',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'},
                'annotations': [{
                    'text': 'Training model in progress...<br>This may take several minutes',
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 18, 'color': '#FF9800'}
                }]
            }
        }
        
        print("Running training function...")
        try:
            fig_before, fig_after = run_training()
            print("Training completed successfully!")
        except Exception as e:
            print(f"    ERROR: Training failed with exception: {e}")
            traceback.print_exc()
            
            error_fig = {
                'data': [],
                'layout': {
                    'title': f'Training Failed: {str(e)}',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': f'Training failed with error:<br>{str(e)}<br><br>Check console for details.',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#f44336'}
                    }]
                }
            }
            return "Train (Failed)", normal_style, False, {"display": "block"}, "keywords", {"before": error_fig, "after": error_fig}
        
        group_info_path = FILE_PATHS["training_group_info"]
        with open(group_info_path, "w", encoding="utf-8") as f:
            json.dump(group_order, f, indent=4, ensure_ascii=False)
        print(f"Group information saved to {group_info_path} for model loading")
        
        print(f"    DEBUG: fig_before type: {type(fig_before)}")
        print(f"    DEBUG: fig_after type: {type(fig_after)}")
        if hasattr(fig_before, 'data') and fig_before.data:
            print(f"    DEBUG: fig_before.data length: {len(fig_before.data)}")
            if len(fig_before.data) > 0:
                first_trace = fig_before.data[0]
                print(f"    DEBUG: fig_before first trace x length: {len(first_trace.x) if hasattr(first_trace, 'x') else 'No x'}")
                print(f"    DEBUG: fig_before first trace y length: {len(first_trace.y) if hasattr(first_trace, 'y') else 'No y'}")
                print(f"    DEBUG: fig_before first trace customdata length: {len(first_trace.customdata) if hasattr(first_trace, 'customdata') else 'No customdata'}")
        
        if hasattr(fig_after, 'data') and fig_after.data:
            print(f"    DEBUG: fig_after.data length: {len(fig_after.data)}")
            if len(fig_after.data) > 0:
                first_trace = fig_after.data[0]
                print(f"    DEBUG: fig_after first trace x length: {len(first_trace.x) if hasattr(first_trace, 'x') else 'No x'}")
                print(f"    DEBUG: fig_after first trace y length: {len(first_trace.y) if hasattr(first_trace, 'y') else 'No y'}")
                print(f"    DEBUG: fig_after first trace customdata length: {len(first_trace.customdata) if hasattr(first_trace, 'customdata') else 'No customdata'}")
        
        completed_style = {
            "margin-top": "20px",
            "padding": "10px 20px",
            "fontSize": "16px",
            "backgroundColor": "#2E7D32",  
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer"
        }
        
        switch_button_style = {
            "margin": "15px auto",
            "padding": "12px 30px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "backgroundColor": "#3498db",
            "color": "white",
            "border": "none",
            "borderRadius": "6px",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 10px rgba(52, 152, 219, 0.3)",
            "display": "block"  
        }
        
        print(f"    DEBUG: Converting figures to dict by manual extraction")
        
        import numpy as np
        
        def fig_to_serializable_dict(fig):
            result = {
            'data': [],
                'layout': {}
            }
            
            if hasattr(fig, 'layout'):
                result['layout'] = fig.layout.to_plotly_json() if hasattr(fig.layout, 'to_plotly_json') else {}
            
            for trace in fig.data:
                trace_dict = {}
                
                for attr in ['x', 'y', 'mode', 'type', 'name', 'text', 'textposition', 'textfont', 'customdata', 'hovertemplate', 'hovertext']:
                    if hasattr(trace, attr):
                        val = getattr(trace, attr)
                        if val is not None:
                 
                            if hasattr(val, 'tolist'):
                                trace_dict[attr] = val.tolist()
                            elif hasattr(val, '__iter__') and not isinstance(val, str):
                                trace_dict[attr] = list(val)
                            else:
                                trace_dict[attr] = val
                
                if hasattr(trace, 'marker'):
                    marker_dict = {}
                    marker = trace.marker
                    for m_attr in ['color', 'size', 'symbol', 'opacity', 'line']:
                        if hasattr(marker, m_attr):
                            m_val = getattr(marker, m_attr)
                            if m_val is not None:
                                if m_attr == 'line' and hasattr(m_val, 'to_plotly_json'):
                                    marker_dict[m_attr] = m_val.to_plotly_json()
                                elif hasattr(m_val, 'tolist'):
                                    marker_dict[m_attr] = m_val.tolist()
                                else:
                                    marker_dict[m_attr] = m_val
                    trace_dict['marker'] = marker_dict
                
                trace_name = trace_dict.get('name', 'Unknown')
                x_len = len(trace_dict.get('x', []))
                print(f"    DEBUG: Extracted trace '{trace_name}': {x_len} points")
                
                result['data'].append(trace_dict)
            
            return result
        
        fig_before_dict = fig_to_serializable_dict(fig_before)
        fig_after_dict = fig_to_serializable_dict(fig_after)
        
        if fig_after_dict.get('data'):
            for trace in fig_after_dict['data']:
                trace_name = trace.get('name', 'Unknown')
                x_len = len(trace.get('x', []))
                print(f"    DEBUG: Trace '{trace_name}': {x_len} points")
                if 'Center:' in trace_name:
                    print(f"    DEBUG: Center trace '{trace_name}' marker:")
                    print(f"  marker: {trace.get('marker', {})}")
                    print(f"  symbol: {trace.get('marker', {}).get('symbol', 'NO SYMBOL')}")
        
        print(f"    DEBUG: Manually built fig_before_dict type: {type(fig_before_dict)}")
        print(f"    DEBUG: Manually built fig_after_dict type: {type(fig_after_dict)}")
        
        if isinstance(fig_before_dict, dict) and 'data' in fig_before_dict:
            print(f"    DEBUG: fig_before_dict data length: {len(fig_before_dict['data'])}")
            if len(fig_before_dict['data']) > 0:
                first_trace = fig_before_dict['data'][0]
                print(f"    DEBUG: fig_before_dict first trace x length: {len(first_trace.get('x', []))}")
                print(f"    DEBUG: fig_before_dict first trace y length: {len(first_trace.get('y', []))}")
                print(f"    DEBUG: fig_before_dict first trace customdata length: {len(first_trace.get('customdata', []))}")
        
        if isinstance(fig_after_dict, dict) and 'data' in fig_after_dict:
            print(f"    DEBUG: fig_after_dict data length: {len(fig_after_dict['data'])}")
            if len(fig_after_dict['data']) > 0:
                first_trace = fig_after_dict['data'][0]
                print(f"    DEBUG: fig_after_dict first trace x length: {len(first_trace.get('x', []))}")
                print(f"    DEBUG: fig_after_dict first trace y length: {len(first_trace.get('y', []))}")
                print(f"    DEBUG: fig_after_dict first trace customdata length: {len(first_trace.get('customdata', []))}")
        
        return "Training Complete", completed_style, False, switch_button_style, "training", {"before": fig_before_dict, "after": fig_after_dict}
        
    except Exception as e:
        print(f"Training failed with error: {e}")
           
        traceback.print_exc()
        
        error_fig = {
            'data': [],
            'layout': {
                'title': f'Training Error: {str(e)}',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'},
                'annotations': [{
                    'text': f'Training failed: {str(e)}',
                    'x': 0.5,
                    'y': 0.5,
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16, 'color': 'red'}
                }]
            }
        }
        
        error_style = {
            "margin-top": "20px",
            "padding": "10px 20px",
            "fontSize": "16px",
            "backgroundColor": "#F44336",  
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer"
        }
        
        switch_button_style = {"display": "none"}
        
        return "Training Failed", error_style, False, switch_button_style, "keywords", {"before": None, "after": None}

@app.callback(
    [Output("train-btn", "children", allow_duplicate=True),
     Output("train-btn", "style", allow_duplicate=True),
     Output("train-btn", "disabled", allow_duplicate=True)],
    Input("train-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_train_button_immediately(n_clicks):

    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    training_style = {
        "margin-top": "20px",
        "padding": "10px 20px",
        "fontSize": "16px",
        "backgroundColor": "#FF9800",  
        "color": "white",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "not-allowed",
        "opacity": "0.8",
        "transform": "scale(0.98)"
    }
    
    return "Training...", training_style, True





@app.callback(
    [Output("article-fulltext-container", "children", allow_duplicate=True),
     Output("highlighted-indices", "data", allow_duplicate=True)],
    [Input("plot-before", "clickData"),
     Input("plot-after", "clickData")],
    prevent_initial_call=True
)
def display_article_content_training(click_data_before, click_data_after):

    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    click_data = None
    if ctx.triggered[0]['prop_id'] == 'plot-before.clickData':
        click_data = click_data_before
    elif ctx.triggered[0]['prop_id'] == 'plot-after.clickData':
        click_data = click_data_after
    
    if not click_data:
        raise PreventUpdate
    
    try:
        article_index = click_data['points'][0]['customdata'][0]
        
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        if article_index is not None and article_index < len(df):
            article_text = str(df.iloc[article_index, 1])
            article_label = str(df.iloc[article_index, 0])
            
            content = html.Div([
                html.H5(f"Article {article_index + 1} (Label: {article_label})", 
                       style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P(article_text, style={
                    "lineHeight": "1.6", 
                    "textAlign": "justify",
                    "fontSize": "14px",
                    "color": "#333"
                })
            ])
            
            return content, [article_index]
        else:
            return html.P("Article not found", style={"color": "red"}), []
    
    except Exception as e:
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"}), []


@app.callback(
    [Output("display-mode", "data", allow_duplicate=True),
     Output("switch-view-btn", "children")],
    Input("switch-view-btn", "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def switch_display_mode(n_clicks, current_mode):

    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    print(f"    DEBUG: switch_display_mode called with current_mode: {current_mode}")
    
    if current_mode == "keywords":
        new_mode = "training"
        button_text = "Switch to Keywords View"
    elif current_mode == "training":
        new_mode = "keywords"
        button_text = "Switch to Training View"
    elif current_mode == "finetune":
        new_mode = "training"
        button_text = "Switch to Training View"
    else:
        new_mode = "keywords"
        button_text = "Switch to Training View"
    
    print(f"    DEBUG: switch_display_mode: switching from {current_mode} to {new_mode}")
    return new_mode, button_text

@app.callback(
    [Output("switch-view-btn", "style", allow_duplicate=True),
     Output("switch-view-btn", "children", allow_duplicate=True)],
    Input("display-mode", "data"),
    prevent_initial_call=True
)
def control_switch_view_btn_visibility(display_mode):
    base_style = {
        "backgroundColor": "#3498db",
        "color": "white",
        "border": "none",
        "padding": "10px 20px",
        "borderRadius": "6px",
        "fontSize": "1rem",
        "fontWeight": "bold",
        "cursor": "pointer",
        "transition": "all 0.3s ease",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
        "marginRight": "10px",
        "minWidth": "180px",
        "flexShrink": "0"
    }
    
    if display_mode == "finetune":
        base_style["display"] = "block"
        button_text = "Switch to Training View"  
    elif display_mode == "training":
        base_style["display"] = "block"
        button_text = "Switch to Keywords View"  
    elif display_mode == "keywords":
        base_style["display"] = "none"
        button_text = "Switch to Training View"  
    else:
        base_style["display"] = "block"
        button_text = "Switch to Training View"  
    
    return base_style, button_text

@app.callback(
    Output("switch-finetune-btn", "style"),
    [Input("display-mode", "data"), Input("training-figures", "data")]
)
def show_switch_finetune_btn(display_mode, training_figures):
    print(f"    DEBUG: show_switch_finetune_btn called with display_mode: {display_mode}")
    print(f"    DEBUG: training_figures: {training_figures}")
    
    base_style = {
        "margin": "15px auto",
        "padding": "12px 30px",
        "fontSize": "1rem",
        "fontWeight": "bold",
        "backgroundColor": "#8e44ad",
        "color": "white",
        "border": "none",
        "borderRadius": "6px",
        "cursor": "pointer",
        "transition": "all 0.3s ease",
        "boxShadow": "0 3px 10px rgba(142, 68, 173, 0.3)",
        "display": "none"
    }
    try:
        has_after = isinstance(training_figures, dict) and bool(training_figures.get("after"))
        print(f"    DEBUG: has_after: {has_after}")
        print(f"    DEBUG: display_mode in ('training', 'finetune'): {display_mode in ('training', 'finetune')}")
        
        if display_mode in ("training", "finetune", "keywords") and has_after:
            base_style["display"] = "block"
            print(f"    DEBUG: Showing finetune button")
        else:
            base_style["display"] = "none"
            print(f"    DEBUG: Hiding finetune button")
    except Exception as e:
        base_style["display"] = "none"
        print(f"    DEBUG: Exception in finetune button logic: {e}")
    
    print(f"    DEBUG: Returning finetune button style: {base_style}")
    return base_style
            

@app.callback(
    [Output("main-visualization-area", "children"),
     Output("training-group-management-area", "style"),
     Output("keywords-group-management-area", "style"),
     Output("finetune-group-management-area", "style")],
    [Input("display-mode", "data"),
     Input("training-figures", "data")],
    prevent_initial_call=True
)
def update_main_visualization_area(display_mode, training_figures):
   
    if training_figures:
        print(f"    DEBUG:   training_figures keys: {list(training_figures.keys()) if isinstance(training_figures, dict) else 'not dict'}")
    
    
    if display_mode == "training":
       
        if training_figures:
            fig_before = training_figures.get("before", {})
            fig_after = training_figures.get("after", {})
            print(f"    Using existing training figures")
        else:
            print(f"    No training figures available, using placeholders")
            fig_before = {
                'data': [],
                'layout': {
                    'title': 'Before Training - No Data Available',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': 'Please run training first to see results',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#666'}
                    }]
                }
            }
            fig_after = {
                'data': [],
                'layout': {
                    'title': 'After Training - No Data Available',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': 'Please run training first to see results',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#666'}
                    }]
                }
            }
        
        training_group_style = {'display': 'flex', 'marginBottom': '30px'}
        
        print(f"    DEBUG:     RETURNING TRAINING MODE LAYOUT:")
        print(f"    DEBUG:   - main-visualization-area: training plots")
        print(f"    DEBUG:   - training-group-management-area: {{'display': 'flex'}}")
        print(f"    DEBUG:   - keywords-group-management-area: {{'display': 'none'}}")
        
        return [
            html.Div([
                html.H4("Before Training", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Training results before model optimization", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='plot-before',
                    figure=fig_before if fig_before else {},
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '1%'
            }),
            
            html.Div([
                html.H4("After Training", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Training results after model optimization", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='plot-after',
                    figure=fig_after if fig_after else {},
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '1%'
            })
        ], training_group_style, {'display': 'none', 'marginBottom': '30px'}, {'display': 'none', 'marginBottom': '30px'}
    elif display_mode == "finetune":
        print(f"    DEBUG:      ENTERING FINETUNE MODE LAYOUT")
        print(f"    DEBUG:   Will return finetune plot and show finetune panels")
        
        if training_figures:
            fig_after = training_figures.get("after", {})
            print(f"    Using existing training after figure for finetune")
        else:
            print(f"    No training figures available, using placeholder")
            fig_after = {
                'data': [],
                'layout': {
                    'title': 'Finetune Mode - No Training Data Available',
                    'xaxis': {'title': 'X'},
                    'yaxis': {'title': 'Y'},
                    'annotations': [{
                        'text': 'Please run training first to access finetune mode',
                        'x': 0.5,
                        'y': 0.5,
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#666'}
                    }]
                }
            }
        
        finetune_group_style = {'display': 'flex', 'marginBottom': '30px'}

        
        return [
            html.Div([
                html.H4("Finetune Mode - Interactive 2D", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Click on points to preview text and reassign samples", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='finetune-2d-plot',
                    figure=fig_after if fig_after else {},
                    style={'height': '800px'},  
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '100%',
                'minHeight': '850px',  
                'padding': '20px',
                'margin': '0 auto',
                'display': 'block'
            })
        ], {'display': 'none', 'marginBottom': '30px'}, {'display': 'none', 'marginBottom': '30px'}, finetune_group_style
    else:

        
        training_group_style = {'display': 'none', 'marginBottom': '30px'}
        keywords_group_style = {'display': 'flex', 'marginBottom': '30px'}
        
        return [
            html.Div([
                html.H4("Keywords 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Click on keywords to highlight related documents", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='keywords-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginRight': '1%'
            }),
            
            html.Div([
                html.H4("Documents 2D Visualization", style={
                    "color": "#2c3e50",
                    "fontSize": "1.3rem",
                    "fontWeight": "bold",
                    "marginBottom": "8px",
                    "textAlign": "center"
                }),
                html.P("Documents highlighted by selected keyword", style={
                    "color": "#7f8c8d",
                    "fontSize": "0.9rem",
                    "textAlign": "center",
                    "marginBottom": "15px",
                    "fontStyle": "italic"
                }),
                dcc.Graph(
                    id='documents-2d-plot',
                    style={'height': '700px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '49%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '20px',
                'marginLeft': '1%'
            })
        ], training_group_style, keywords_group_style, {'display': 'none', 'marginBottom': '30px'}


@app.callback(
    Output('highlighted-indices', 'data', allow_duplicate=True),
    [Input('training-selected-keyword', 'data'),
     Input('training-selected-group', 'data')],
    State('group-order', 'data'),
    State('training-figures', 'data'),
    State('display-mode', 'data'),
    prevent_initial_call=True
)
def update_training_highlights(selected_keyword, selected_group, group_order, training_figures, display_mode):
    global df
    

    if display_mode != "training":
        print(f"    DEBUG:         NOT IN TRAINING MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'training'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      TRAINING MODE CONFIRMED")
    
    if 'df' not in globals() or not training_figures:
        print(f"    DEBUG:         MISSING DATA OR TRAINING FIGURES:")
        print(f"    DEBUG:   df in globals: {'df' in globals()}")
        print(f"    DEBUG:   training_figures: {training_figures}")
        print(f"    DEBUG:   Returning empty")
        return {"type": "none", "indices": []}
    
    print(f"    DEBUG:      DATA AND TRAINING FIGURES AVAILABLE")
    
    
    if selected_keyword:
        print(f"    DEBUG:     KEYWORD SELECTION:")
        print(f"    DEBUG:   Processing keyword: {selected_keyword}")

        keyword_indices = []
        

        filtered_path = FILE_PATHS["filtered_group_assignment"]
        if os.path.exists(filtered_path):
            try:
                print(f"    DEBUG:   Loading filtered results for keyword search")
                with open(filtered_path, "r", encoding="utf-8") as f:
                    filtered_dict = json.load(f)
                

                keyword_group = None
                for grp_name, keywords in group_order.items():
                    if selected_keyword in keywords:
                        keyword_group = grp_name
                        print(f"    DEBUG:   Keyword '{selected_keyword}' belongs to group: {keyword_group}")
                        break
                
                if keyword_group and keyword_group in filtered_dict:
   
                    group_filtered_docs = filtered_dict[keyword_group]
                    print(f"    DEBUG:   Group '{keyword_group}' has {len(group_filtered_docs)} filtered documents")
                    

                    for idx in group_filtered_docs:
                        if idx < len(df):
                            text = str(df.iloc[idx, 1]).lower()
                            if selected_keyword.lower() in text:
                                keyword_indices.append(idx)
                    
                    print(f"    DEBUG:   Found {len(keyword_indices)} documents containing '{selected_keyword}' in filtered group")
                    print(f"    DEBUG:   Document indices: {sorted(keyword_indices)}")
                    return {"type": "keyword", "indices": keyword_indices, "keyword": selected_keyword}
                else:
                    print(f"    DEBUG:   Keyword group '{keyword_group}' not found in filtered results, fallback to full search")
            except Exception as e:
                print(f"    DEBUG:   Error loading filtered results: {e}, fallback to full search")
        else:
            print(f"    DEBUG:   No filtered results file, fallback to full search")
        
        # Fallback: 如果没有过滤文件，使用BM25搜索整个数据集
        try:
            from rank_bm25 import BM25Okapi
            from nltk.stem import PorterStemmer
            from nltk.corpus import stopwords
            
            try:
                stop_words = set(stopwords.words('english'))
            except:
                stop_words = set()
            
            stemmer = PorterStemmer()
            
            def preprocess(text):
                tokens = str(text).lower().split()
                tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
                return tokens
            
            all_texts = [str(df.iloc[i, 1]) for i in range(len(df))]
            tokenized_corpus = [preprocess(doc) for doc in all_texts]
            bm25 = BM25Okapi(tokenized_corpus)
            
            query_tokens = preprocess(selected_keyword)
            scores = bm25.get_scores(query_tokens)
            
            for i, score in enumerate(scores):
                if score > 0:
                    keyword_indices.append(i)
            
            keyword_indices = sorted(keyword_indices, key=lambda x: scores[x], reverse=True)[:100]
            
            print(f"    DEBUG:   BM25 search found {len(keyword_indices)} documents for keyword '{selected_keyword}'")
            print(f"    DEBUG:   Document indices (top scores): {keyword_indices[:20]}")
            
        except Exception as e:
            print(f"    DEBUG:   BM25 failed, using text matching: {e}")
            for i, text in enumerate(df.iloc[:, 1]):
                if selected_keyword.lower() in str(text).lower():
                    keyword_indices.append(i)
            print(f"    DEBUG:   Text matching found {len(keyword_indices)} documents")
        
        return {"type": "keyword", "indices": keyword_indices, "keyword": selected_keyword}
        
    elif selected_group and group_order:
        print(f"    DEBUG:     GROUP SELECTION:")
        print(f"    DEBUG:   Processing group: {selected_group}")
        print(f"    DEBUG:   Full group_order: {group_order}")
        
        if selected_group in group_order:
            group_keywords = group_order[selected_group]
            print(f"    DEBUG:   Group keywords: {group_keywords}")
            
            group_indices = []
            
            filtered_path = FILE_PATHS["filtered_group_assignment"]
            if os.path.exists(filtered_path):
                try:
                    print(f"    DEBUG:   Training mode detected - loading filtered results")
                    with open(filtered_path, "r", encoding="utf-8") as f:
                        filtered_dict = json.load(f)
                    
                    if selected_group in filtered_dict:
                        group_indices = filtered_dict[selected_group]
                        print(f"    DEBUG:   Using filtered documents: {len(group_indices)} documents")
                        print(f"    DEBUG:   Document indices: {group_indices[:20]}")
                        return {"type": "group", "indices": group_indices, "group": selected_group}
                    else:
                        print(f"    DEBUG:   Group '{selected_group}' not in filtered results")
                except Exception as e:
                    print(f"    DEBUG:   Failed to load filtered results: {e}")
            
            print(f"    DEBUG:   No filtered results - using BM25 search")
            
            try:
                from rank_bm25 import BM25Okapi
                from nltk.stem import PorterStemmer
                from nltk.corpus import stopwords
                
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    stop_words = set()
                
                stemmer = PorterStemmer()
                
                def preprocess(text):
                    tokens = str(text).lower().split()
                    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
                    return tokens
                
                all_texts = [str(df.iloc[i, 1]) for i in range(len(df))]
                tokenized_corpus = [preprocess(doc) for doc in all_texts]
                bm25 = BM25Okapi(tokenized_corpus)
                
                query_tokens = []
                for kw in group_keywords:
                    query_tokens.extend(preprocess(kw))
                
                scores = bm25.get_scores(query_tokens)
                
                for i, score in enumerate(scores):
                    if score > 0:
                        group_indices.append(i)
                
                group_indices = sorted(group_indices, key=lambda x: scores[x], reverse=True)[:100]
                
                print(f"    DEBUG:   BM25 search found {len(group_indices)} documents for group '{selected_group}'")
                print(f"    DEBUG:   Document indices (top scores): {group_indices[:20]}")
                
            except Exception as e:
                print(f"    DEBUG:   BM25 failed, using text matching: {e}")
                for i, text in enumerate(df.iloc[:, 1]):
                    text_lower = str(text).lower()
                    if any(keyword.lower() in text_lower for keyword in group_keywords):
                        group_indices.append(i)
                print(f"    DEBUG:   Text matching found {len(group_indices)} documents")
            
            return {"type": "group", "indices": group_indices, "group": selected_group}
        else:
            print(f"    DEBUG:   Group '{selected_group}' not found in group_order")
            return {"type": "group", "indices": [], "group": selected_group}
    
    print(f"    DEBUG:     NO SELECTION:")
    print(f"    DEBUG:   No keyword or group selected, returning empty")
    return {"type": "none", "indices": []}

@app.callback(
    [Output('plot-before', 'figure', allow_duplicate=True),
     Output('plot-after', 'figure', allow_duplicate=True)],
    [Input('highlighted-indices', 'data'),
     Input('training-selected-article', 'data'),
     Input('display-mode', 'data')],
    [State('training-figures', 'data'),
     State('group-order', 'data')],
    prevent_initial_call=True
)
def update_training_plots_with_highlights(highlighted_indices, training_selected_article, display_mode, training_figures, group_order):

    global df
    

    

    if display_mode != "training":
        print(f"    DEBUG:         NOT IN TRAINING MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'training'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      TRAINING MODE CONFIRMED")
    

    if not training_figures:
        print(f"    DEBUG:         NO TRAINING FIGURES:")
        print(f"    DEBUG:   Returning empty figures")
        return {}, {}
    
    print(f"    DEBUG:      TRAINING FIGURES AVAILABLE")
    

    fig_before = training_figures.get("before", {})
    fig_after = training_figures.get("after", {})
    

    keyword_group_highlights = []
    selected_article_highlight = None
    

    if isinstance(highlighted_indices, dict) and 'type' in highlighted_indices:
        highlight_type = highlighted_indices.get('type')
        highlight_indices = highlighted_indices.get('indices', [])
        
        print(f"    DEBUG:     PROCESSING HIGHLIGHTS:")
        print(f"    DEBUG:   Highlight type: {highlight_type}")
        print(f"    DEBUG:   Highlight indices: {highlight_indices}")
        
        if highlight_type == "group":

            keyword_group_highlights = highlight_indices
            print(f"    DEBUG:   Group highlights: {keyword_group_highlights}")
            
        elif highlight_type == "keyword":

            keyword_group_highlights = highlight_indices
            print(f"    DEBUG:   Keyword highlights: {keyword_group_highlights}")
            
        elif highlight_type == "none":

            keyword_group_highlights = []
            print(f"    DEBUG:   No highlights")
    

    if training_selected_article is not None and training_selected_article < len(df):
        selected_article_highlight = training_selected_article
        print(f"    DEBUG:     ARTICLE SELECTION:")
        print(f"    DEBUG:   Selected article: {training_selected_article}")
        
        if keyword_group_highlights and training_selected_article not in keyword_group_highlights:
            print(f"    DEBUG:   Article {training_selected_article} is NOT in current highlights")
            print(f"    DEBUG:   This will show both highlights and article")
        elif keyword_group_highlights and training_selected_article in keyword_group_highlights:
            print(f"    DEBUG:   Article {training_selected_article} IS in current highlights")
            print(f"    DEBUG:   This will show highlights with article highlighted")
        else:
            print(f"    DEBUG:   No current highlights, only showing article")
    
    print(f"    DEBUG:   Keyword/Group highlights: {keyword_group_highlights}")
    print(f"    DEBUG:   Selected article highlight: {selected_article_highlight}")
    
    updated_fig_before = apply_highlights_to_training_plot(fig_before, keyword_group_highlights, selected_article_highlight, "before")
    updated_fig_after = apply_highlights_to_training_plot(fig_after, keyword_group_highlights, selected_article_highlight, "after")
    
    return updated_fig_before, updated_fig_after

def apply_highlights_to_training_plot(fig, keyword_group_highlights, selected_article_highlight, plot_name):

    if not fig or 'data' not in fig:
        return fig
    
    print(f"    DEBUG:     APPLYING HIGHLIGHTS TO {plot_name.upper()} PLOT:")
    print(f"    DEBUG:   Keyword/Group highlights: {keyword_group_highlights}")
    print(f"    DEBUG:   Selected article: {selected_article_highlight}")
    
    updated_fig = fig.copy()
    
    if not updated_fig['data']:
        return updated_fig
    
    traces = []
    main_trace = None
    center_traces = []
    
    print(f"    DEBUG:   Processing {len(updated_fig['data'])} traces from original figure")
    
    for i, trace in enumerate(updated_fig['data']):
        trace_name = trace.get('name', 'Unknown')
        marker = trace.get('marker', {})
        symbol = marker.get('symbol', 'circle')
        x_len = len(trace.get('x', []))
        
        print(f"    DEBUG:     Trace {i}: name='{trace_name}', symbol='{symbol}', points={x_len}")
        
        if symbol == 'diamond' or 'Center' in trace_name:
            center_traces.append(trace)
            print(f"    DEBUG:       Keeping as center trace")
        elif main_trace is None and x_len > 10 and symbol != 'star':
            main_trace = trace
            print(f"    DEBUG:       Keeping as main document trace")
    
    
    if main_trace:
        traces.append(main_trace)
    else:
        print(f"    DEBUG:           No main trace found, returning original figure")
        return fig
    
    traces.extend(center_traces)
    print(f"    DEBUG:   Added {len(center_traces)} center traces")
    
    x_data = main_trace['x'] if isinstance(main_trace['x'], (list, tuple)) else list(main_trace['x'])
    y_data = main_trace['y'] if isinstance(main_trace['y'], (list, tuple)) else list(main_trace['y'])
        
    if keyword_group_highlights:
        highlight_x = [x_data[i] for i in keyword_group_highlights if i < len(x_data)]
        highlight_y = [y_data[i] for i in keyword_group_highlights if i < len(y_data)]
        
        if highlight_x and highlight_y:
            traces.append({
                'x': highlight_x,
                'y': highlight_y,
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Selected Group',
                'marker': {
                    'size': 15,
                    'color': '#FFD700',  
                    'symbol': 'star',
                    'line': {'width': 2, 'color': 'white'}
                },
                'text': [f'Doc {i+1}' for i in keyword_group_highlights if i < len(x_data)],
                'customdata': [[i] for i in keyword_group_highlights if i < len(x_data)],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
            print(f"    DEBUG:   Added keyword/group highlight trace with {len(highlight_x)} points")
    
    if selected_article_highlight is not None and selected_article_highlight < len(x_data):
        article_x = [x_data[selected_article_highlight]]
        article_y = [y_data[selected_article_highlight]]
        
        traces.append({
            'x': article_x,
            'y': article_y,
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Selected Article',
            'marker': {
                'size': 20,
                'color': '#FF0000',  
                'symbol': 'star',
                'line': {'width': 3, 'color': 'white'}
            },
            'text': [f'Doc {selected_article_highlight+1}'],
            'customdata': [[selected_article_highlight]],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        })
        print(f"    DEBUG:   Added selected article highlight trace")
    
    updated_fig['data'] = traces
    
    print(f"    DEBUG:   Original trace count: {len(fig['data'])}")
    print(f"    DEBUG:   Final trace count: {len(traces)}")
    
    return updated_fig

@app.callback(
    Output("training-group-containers", "children"),
    [Input("group-order", "data"),
     Input("training-selected-group", "data"),
     Input("display-mode", "data")],  
    [State("training-selected-keyword", "data")],
    prevent_initial_call=False  
)
def render_training_groups(group_order, selected_group, display_mode, selected_keyword):

    
    if display_mode != "training":
        print(f"    DEBUG:         NOT IN TRAINING MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'training'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      TRAINING MODE CONFIRMED")
    
    if not group_order:
        print(f"    DEBUG:         No group_order, showing placeholder")
        return html.Div([
            html.H6("Training Group Management", style={"color": "#2c3e50", "marginBottom": "10px"}),
            html.P("No groups have been created yet. Please create groups in Keywords mode first.", 
                   style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
        ])
    
    print(f"    DEBUG:      Proceeding with training group rendering...")

    children = []
    for grp_name, kw_list in group_order.items():
        if grp_name == "Other":

            if kw_list:  
                group_display_name = "Exclude"
                group_color = get_group_color(grp_name)
            else:  
                group_display_name = "Other (Exclude)"
                group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Training Group {group_number}"
            group_color = get_group_color(grp_name)
        
        if grp_name == "Other":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  
            }
        else:
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px solid {group_color}",
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px"
            }
        
        group_header = html.Button(
            group_display_name,
            id={"type": "training-group-header", "index": grp_name},
            style=header_style
        )

        group_keywords = []
        for i, kw in enumerate(kw_list):

            is_selected = selected_keyword and kw == selected_keyword
            
            keyword_button = html.Button(
                kw,
                id={"type": "training-select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  
                    "color": "white" if is_selected else group_color,  
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  
                }
            )
            
            keyword_item = html.Div([
                keyword_button,
                html.Button("×", id={"type": "training-remove-keyword", "group": grp_name, "index": i}, 
                           style={"margin": "2px", "padding": "2px 6px", "fontSize": "10px", "color": "red", "float": "right"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "3px"})
            group_keywords.append(keyword_item)

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "8px",
                "minHeight": "50px",
                "maxHeight": "200px",
                "overflowY": "auto",
                "backgroundColor": "#f9f9f9",
                "marginBottom": "10px"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"marginBottom": "15px"}
        )
        children.append(group_container)

    return children

@app.callback(
    [Output("training-selected-group", "data"),
     Output("training-selected-keyword", "data")],
    [Input({"type": "training-group-header", "index": ALL}, "n_clicks"),
     Input("display-mode", "data")],
    prevent_initial_call=True
)
def select_training_group(n_clicks, display_mode):

    ctx = dash.callback_context
    
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG: n_clicks: {n_clicks}")
    print(f"    DEBUG: display_mode: {display_mode}")
    
    if not ctx.triggered:
        print(f"    DEBUG:         No context triggered, preventing update")
        raise PreventUpdate

    print(f"    DEBUG:      Context triggered, analyzing trigger...")
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    print(f"    DEBUG:     TRIGGER ANALYSIS:")
    print(f"    DEBUG:   triggered_id: {triggered_id}")
    print(f"    DEBUG:   triggered_n_clicks: {triggered_n_clicks}")
    
    if "training-group-header" in triggered_id and triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
        try:
            import json
            parsed_id = json.loads(triggered_id.split('.')[0])
            selected_group = parsed_id["index"]
            return selected_group, None  
                
        except Exception as e:
            print(f"    DEBUG:         ERROR PARSING TRAINING GROUP HEADER ID:")
            print(f"    DEBUG:   Error: {e}")
            raise PreventUpdate
    else:
        print(f"    DEBUG:         NOT A VALID TRAINING GROUP HEADER CLICK")
        print(f"    DEBUG:   'training-group-header' in triggered_id: {'training-group-header' in triggered_id}")
        print(f"    DEBUG:   triggered_n_clicks > 0: {triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0)}")

    print(f"    DEBUG:         No valid conditions met, raising PreventUpdate")
    raise PreventUpdate

@app.callback(
    [Output("training-selected-keyword", "data", allow_duplicate=True),
     Output("training-selected-group", "data", allow_duplicate=True)],
    [Input({"type": "training-select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def select_training_keyword_from_group(n_clicks, display_mode, group_order):

    ctx = dash.callback_context
    
    
    if not ctx.triggered:
        print(f"    DEBUG: No context triggered")
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    print(f"    DEBUG: triggered_id: {triggered_id}")
    print(f"    DEBUG: triggered_n_clicks: {triggered_n_clicks}")
    
    if "training-select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            print(f"    DEBUG: Select training keyword from group management: {keyword}")
            
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                print(f"    DEBUG: Direct training keyword click detected, selecting keyword: {keyword}")
                
                print(f"    DEBUG: Following the same logic as keywords mode: keyword selection clears group selection")
                
                keyword_docs = []
                
                if 'df' in globals():
                    for i, text in enumerate(df.iloc[:, 1]):
                        if keyword.lower() in str(text).lower():
                            keyword_docs.append(i)
                    
                    print(f"    DEBUG: Found {len(keyword_docs)} documents containing keyword '{keyword}': {keyword_docs}")
                    print(f"    DEBUG:      SUCCESS: About to return (keyword, None)")
                    print(f"    DEBUG:   This will:")
                    print(f"    DEBUG:     1. Set training-selected-keyword to '{keyword}'")
                    print(f"    DEBUG:     2. Clear training-selected-group to None")
                    print(f"    DEBUG:   Following keywords mode logic: keyword selection clears group selection")
                    
                    return keyword, None
                else:
                    print(f"    DEBUG:         ERROR: No dataframe available")
                    return keyword, None
            else:
                print(f"    DEBUG:         Invalid n_clicks value: {triggered_n_clicks}")
                raise PreventUpdate
            
        except Exception as e:
            print(f"    DEBUG:         ERROR PARSING TRAINING KEYWORD BUTTON ID:")
            print(f"    DEBUG:   Error: {e}")
            raise PreventUpdate
    else:
        print(f"    DEBUG:         NOT A TRAINING KEYWORD SELECTION:")
        print(f"    DEBUG:   'training-select-keyword' in triggered_id: {'training-select-keyword' in triggered_id}")
        print(f"    DEBUG:   triggered_id contains: {triggered_id}")
    
    print(f"    DEBUG:         No valid conditions met, raising PreventUpdate")
    raise PreventUpdate


@app.callback(
    Output("training-articles-container", "children"),
    [Input("training-selected-keyword", "data"),
     Input("training-selected-group", "data"),
     Input("display-mode", "data")],  
    [State("group-order", "data")],
    prevent_initial_call=False  
)
def display_training_recommended_articles(selected_keyword, selected_group, display_mode, group_order):

    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   selected_keyword: {selected_keyword}")
    print(f"    DEBUG:   selected_group: {selected_group}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:       PARAMETER ANALYSIS:")
    print(f"    DEBUG:     selected_keyword type: {type(selected_keyword)}")
    print(f"    DEBUG:     selected_keyword value: {repr(selected_keyword)}")
    print(f"    DEBUG:     selected_group type: {type(selected_group)}")
    print(f"    DEBUG:     selected_group value: {repr(selected_group)}")
    print(f"    DEBUG:     Both parameters present: {selected_keyword is not None and selected_group is not None}")
    print(f"    DEBUG:     This might indicate a callback chain issue")
    
    if display_mode != "training":
        print(f"    DEBUG:         NOT IN TRAINING MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'training'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      TRAINING MODE CONFIRMED")
    
    try:
        global df, _ARTICLES_CACHE
        if 'df' not in globals():
            print("Data not loaded")
            return html.P("Data not loaded")
        
        cache_key = None
        if selected_keyword:
            cache_key = f"training_keyword:{selected_keyword}"
        elif selected_group and group_order:
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    cache_key = f"training_group:{group_name}:{':'.join(sorted(keywords))}"
                    break
        
        if cache_key and cache_key in _ARTICLES_CACHE:
            print(f"Using cached training articles for: {cache_key}")
            return _ARTICLES_CACHE[cache_key]
        
        search_keywords = []
        search_title = ""
        
        if selected_keyword:
            search_keywords = [selected_keyword]
            search_title = f"Training Articles containing '{selected_keyword}'"
            print(f"Searching for training articles containing keyword: {selected_keyword}")
        elif selected_group:
            print(f"Training group selected: {selected_group}")
            print(f"group_order parameter received: {group_order}")
            
            if group_order:
                search_keywords = []
                for group_name, keywords in group_order.items():
                    print(f"Checking training group '{group_name}' vs selected '{selected_group}'")
                    if group_name == selected_group:
                        search_keywords = keywords
                        print(f"Found matching training group with keywords: {keywords}")
                        break
                
                if search_keywords:
                    search_title = f"Training Articles containing keywords from group '{selected_group}'"
                    print(f"Will search for training articles containing group '{selected_group}' keywords: {search_keywords}")
                else:
                    print(f"No keywords found for training group '{selected_group}' or group is empty")
                    return html.Div([
                        html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                        html.P(f"Training Group '{selected_group}' has no keywords assigned", 
                               style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                    ])
            else:
                print(f"group_order is empty or None")
                return html.Div([
                    html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                    html.P("No training groups have been created yet", 
                           style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
                ])
        else:
            return html.Div([
                html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P("Please select a training keyword or group to view recommended articles", 
                       style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
            ])
        
        matching_articles = []
        

        filtered_indices = []
        use_filtered_mode = False
        
        try:
            filtered_path = FILE_PATHS["filtered_group_assignment"]
            print(f"    Checking for filtered results at: {filtered_path}")
            if os.path.exists(filtered_path):
                with open(filtered_path, "r", encoding="utf-8") as f:
                    filtered_dict = json.load(f)
                print(f"    Loaded filtered results with groups: {list(filtered_dict.keys())}")
                print(f"    Group sizes: {json.dumps({k: len(v) for k, v in filtered_dict.items()}, ensure_ascii=False)}")

                if selected_group in filtered_dict:
                    filtered_indices = filtered_dict[selected_group]
                    use_filtered_mode = True
                    print(f"    Using filtered documents for group '{selected_group}': {len(filtered_indices)} documents")
                    print(f"     Will display ALL {len(filtered_indices)} filtered documents (no keyword matching)")
   
                elif selected_keyword and group_order:
         
                    keyword_group = None
                    for grp_name, keywords in group_order.items():
                        if selected_keyword in keywords:
                            keyword_group = grp_name
                            print(f"     Keyword '{selected_keyword}' belongs to group: {keyword_group}")
                            break
                    
                    if keyword_group and keyword_group in filtered_dict:
             
                        group_filtered_docs = filtered_dict[keyword_group]
                        print(f"     Group '{keyword_group}' has {len(group_filtered_docs)} filtered documents")
                        
           
                        for idx in group_filtered_docs:
                            if idx < len(df):
                                text = str(df.iloc[idx, 1]).lower()
                                if selected_keyword.lower() in text:
                                    filtered_indices.append(idx)
                        
                        use_filtered_mode = True
                        print(f"    Found {len(filtered_indices)} documents containing '{selected_keyword}' in filtered group")
                        print(f"    Document indices: {sorted(filtered_indices)}")
                    else:
                        print(f"    Keyword group '{keyword_group}' not found in filtered results, fallback to BM25")
                else:
                    print(f"    Selected item not in filtered results, fallback to BM25")
            else:
                print(f"    No filtered results file, fallback to BM25")
        except Exception as e:
            print(f"    Error loading filtered results: {e}, fallback to BM25")
        
        if use_filtered_mode:
            for idx in filtered_indices:
                if idx < len(df):
                    row = df.iloc[idx]
                    text = str(row.iloc[1]) if len(row) > 1 else ""
                    file_keywords = extract_top_keywords(text, 5)
                    matching_articles.append({
                        'file_number': idx + 1,
                        'file_index': idx,
                        'text': text,
                        'keywords': file_keywords
                    })
        else:
            print(f"    Using BM25 search for training articles")
            try:
                from rank_bm25 import BM25Okapi
                from nltk.stem import PorterStemmer
                from nltk.corpus import stopwords
                
                try:
                    stop_words = set(stopwords.words('english'))
                except:
                    stop_words = set()
                
                stemmer = PorterStemmer()
                
                def preprocess(text):
                    tokens = str(text).lower().split()
                    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
                    return tokens
                
                all_texts = [str(df.iloc[i, 1]) for i in range(len(df))]
                tokenized_corpus = [preprocess(doc) for doc in all_texts]
                bm25 = BM25Okapi(tokenized_corpus)
                
                query_tokens = []
                for kw in search_keywords:
                    query_tokens.extend(preprocess(kw))
                
                scores = bm25.get_scores(query_tokens)
                
                for idx in range(len(df)):
                    if scores[idx] > 0:
                        text = str(df.iloc[idx, 1])
                        file_keywords = extract_top_keywords(text, 5)
                        matching_articles.append({
                            'file_number': idx + 1,
                            'file_index': idx,
                            'text': text,
                            'keywords': file_keywords,
                            'bm25_score': float(scores[idx])
                        })
                
                matching_articles = sorted(matching_articles, key=lambda x: x.get('bm25_score', 0), reverse=True)[:100]
                print(f"    BM25 found {len(matching_articles)} documents")
                
            except Exception as e:
                print(f"    BM25 failed: {e}")
                import traceback
                traceback.print_exc()
        
        if not matching_articles:
            result = html.P(f"No training articles found for the selected search criteria")
            if cache_key:
                _ARTICLES_CACHE[cache_key] = result
                print(f"Cached 'no training articles' result for: {cache_key}")
            return result
        
        article_items = [
            html.H6(f"{search_title} (Found {len(matching_articles)} articles)", 
                   style={"color": "#2c3e50", "marginBottom": "15px"})
        ]
        
        for article_info in matching_articles:
            keyword_tags = []
            for keyword in article_info['keywords']:
                keyword_tag = html.Span(
                    keyword,
                    style={
                        "backgroundColor": "#e3f2fd",
                        "color": "#1976d2",
                        "padding": "2px 6px",
                        "margin": "2px",
                        "borderRadius": "12px",
                        "fontSize": "11px",
                        "display": "inline-block"
                    }
                )
                keyword_tags.append(keyword_tag)
            
            article_item = html.Div([
                html.Button(
                    html.Div([
                        html.H6(f"Training Article {article_info['file_number']}", 
                               style={"color": "#333", "marginBottom": "8px", "fontSize": "14px", "margin": "0"}),
                        html.Div([
                            html.Span("Top 5 Keywords: ", style={"fontWeight": "bold", "color": "#666"}),
                            html.Div(keyword_tags, style={"display": "inline-block", "marginLeft": "5px"})
                        ], style={"marginBottom": "8px"}),
                    ]),
                    id={"type": "training-article-item", "index": article_info['file_index']},
                    className="article-item-button",
                    style={
                        "width": "100%",
                        "padding": "12px", 
                        "border": "1px solid #eee",
                        "backgroundColor": "white",
                        "borderRadius": "5px",
                        "marginBottom": "8px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                        "cursor": "pointer",
                        "textAlign": "left",
                        "outline": "none"
                    },
                    n_clicks=0
                ),
                html.Hr(style={"margin": "4px 0", "borderColor": "#ddd"})
            ])
            article_items.append(article_item)
        
        result = html.Div(article_items)
        if cache_key:
            _ARTICLES_CACHE[cache_key] = result
            print(f"Cached training articles result for: {cache_key}")
        
        return result
        
    except Exception as e:
        print(f"Error displaying training recommended articles: {e}")
        return html.P(f"Error displaying training recommended articles: {str(e)}")

@app.callback(
    [Output("training-article-fulltext-container", "children"),
     Output("training-selected-article", "data")],
    [Input({"type": "training-article-item", "index": ALL}, "n_clicks")],
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def display_training_article_content(article_clicks, display_mode):

    ctx = dash.callback_context
    
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   display_mode: {display_mode}")
    
    if not ctx.triggered:
        print(f"    DEBUG:         No context triggered")
        raise PreventUpdate
    
    article_index = None
    if 'training-article-item' in ctx.triggered[0]['prop_id']:
        try:
            triggered_id = ctx.triggered[0]['prop_id']
            btn_info = json.loads(triggered_id.split('.')[0])
            article_index = btn_info.get("index")
            print(f"    DEBUG: Training article item clicked, index: {article_index}, mode: {display_mode}")
        except Exception as e:
            print(f"    DEBUG:         Error parsing training article item click: {e}")
            raise PreventUpdate
    
    if article_index is None:
        print(f"    DEBUG:         No article index found")
        raise PreventUpdate
    
    try:
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        if article_index is not None and article_index < len(df):
            article_text = str(df.iloc[article_index, 1])
            article_label = str(df.iloc[article_index, 0])
            
            content = html.Div([
                html.H5(f"Training Article {article_index + 1} (Label: {article_label})", 
                       style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P(article_text, style={
                    "lineHeight": "1.6", 
                    "textAlign": "justify",
                    "fontSize": "14px",
                    "color": "#333"
                })
            ])
            
            print(f"    DEBUG:      SUCCESS: Training article content loaded for article {article_index}")
            
            
            return content, article_index
            
        else:
            print(f"    DEBUG:         Article index {article_index} out of range")
            return html.P("Training article not found", style={"color": "red"}), None
    
    except Exception as e:
        print(f"    DEBUG:         Error loading training article: {e}")
        return html.P(f"Error loading training article: {str(e)}", style={"color": "red"}), None





@app.callback(
    [Output("article-fulltext-container", "children"),
     Output("selected-article", "data", allow_duplicate=True)],
    [Input({"type": "article-item", "index": ALL}, "n_clicks")],
    [State("display-mode", "data"),
     State("selected-keyword", "data"),
     State("selected-group", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def display_article_content_smart(article_clicks, display_mode, current_keyword, current_group, group_order):

    ctx = dash.callback_context
    
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:   current_keyword: {current_keyword}")
    print(f"    DEBUG:   current_group: {current_group}")
    print(f"    DEBUG:   group_order: {group_order}")
    
    if not ctx.triggered:
        print(f"    DEBUG:         No context triggered")
        raise PreventUpdate
    
    article_index = None
    if 'article-item' in ctx.triggered[0]['prop_id']:
        try:
            import json
            triggered_id = ctx.triggered[0]['prop_id']
            btn_info = json.loads(triggered_id.split('.')[0])
            article_index = btn_info.get("index")
            print(f"    DEBUG: Article item clicked, index: {article_index}, mode: {display_mode}")
        except Exception as e:
            print(f"    DEBUG:         Error parsing article item click: {e}")
            raise PreventUpdate
    
    if article_index is None:
        print(f"    DEBUG:         No article index found")
        raise PreventUpdate
    
    try:
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        if article_index is not None and article_index < len(df):
            article_text = str(df.iloc[article_index, 1])
            article_label = str(df.iloc[article_index, 0])
            
            content = html.Div([
                html.H5(f"Article {article_index + 1} (Label: {article_label})", 
                       style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P(article_text, style={
                    "lineHeight": "1.6", 
                    "textAlign": "justify",
                    "fontSize": "14px",
                    "color": "#333"
                })
            ])
            
            print(f"    DEBUG:      SUCCESS: Article content loaded for article {article_index}")
            
            
            return content, article_index
            
        else:
            print(f"    DEBUG:         Article index {article_index} out of range")
            return html.P("Article not found", style={"color": "red"}), None
    
    except Exception as e:
        print(f"    DEBUG:         Error loading article: {e}")
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"}), None


@app.callback(
    [Output("display-mode", "data", allow_duplicate=True),
     Output("switch-finetune-btn", "children")],
    Input("switch-finetune-btn", "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def switch_to_finetune_mode(n_clicks, current_mode):
    print(f"    DEBUG: switch_to_finetune_mode called with n_clicks={n_clicks}, current_mode={current_mode}")

    if not n_clicks:
        print(f"    DEBUG: switch_to_finetune_mode: no clicks, raising PreventUpdate")
        raise PreventUpdate
    if current_mode == "training":
        print(f"    DEBUG: switch_to_finetune_mode: switching from training to finetune")
        return "finetune", "Switch to Training View"
    if current_mode == "finetune":
        print(f"    DEBUG: switch_to_finetune_mode: switching from finetune to training")
        return "training", "Switch to Finetune Mode"
    print(f"    DEBUG: switch_to_finetune_mode: unexpected current_mode={current_mode}, raising PreventUpdate")
    raise PreventUpdate

@app.callback(
    Output("finetune-group-containers", "children"),
    [Input("group-order", "data"),
     Input("finetune-selected-group", "data"),  
     Input("finetune-selected-keyword", "data")]  
)
def render_finetune_groups(group_order, selected_group, selected_keyword):
    print(f"   DEBUG: render_finetune_groups called")
    print(f"   DEBUG: group_order = {group_order}")
    print(f"   DEBUG: selected_group = {selected_group}")
    print(f"   DEBUG: selected_keyword = {selected_keyword}")

    if not group_order:
        return []

    children = []
    for grp_name, kw_list in group_order.items():
        print(f"   DEBUG: Processing group: {grp_name}")
        print(f"   DEBUG: Keywords for {grp_name}: {kw_list}")
        if grp_name == "Other":

            if kw_list:  
                group_display_name = "Exclude"
                group_color = get_group_color(grp_name)
            else:  
                group_display_name = "Other (Exclude)"
                group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Group {group_number}"
            group_color = get_group_color(grp_name)
            print(f"   DEBUG: {grp_name} -> display_name: {group_display_name}, color: {group_color}")
        
        if grp_name == "Other":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  
            }
        else:
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px solid {group_color}",
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px"
            }
        
        group_header = html.Button(
            group_display_name,
            id={"type": "finetune-group-header", "index": grp_name},
            style=header_style
        )

        group_keywords = []
        for kw in kw_list:
            is_selected = selected_keyword and kw == selected_keyword
            
            keyword_button = html.Button(
                kw,
                id={"type": "finetune-select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  
                    "color": "white" if is_selected else group_color,  
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  
                }
            )
            group_keywords.append(keyword_button)

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "8px",
                "minHeight": "50px",
                "maxHeight": "200px",
                "overflowY": "auto",
                "backgroundColor": "#f9f9f9",
                "marginBottom": "10px"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"marginBottom": "15px"}
        )
        children.append(group_container)

    return children

@app.callback(
    [Output("finetune-selected-group", "data"),
     Output("finetune-selected-keyword", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    Input({"type": "finetune-group-header", "index": ALL}, "n_clicks"),
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def select_finetune_group(n_clicks, display_mode):
    ctx = dash.callback_context
    
    print(f"\n{'='*80}")
    print(f"   DEBUG: select_finetune_group called")
    print(f"{'='*80}")
    print(f"   display_mode: {display_mode}")
    print(f"   n_clicks: {n_clicks}")
    print(f"   ctx.triggered: {ctx.triggered}")
    
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id']
        triggered_value = ctx.triggered[0]['value']
        print(f"   triggered_id: {triggered_id}")
        print(f"   triggered_value: {triggered_value}")
    
    if display_mode != "finetune":
        print(f"           Not in finetune mode, raising PreventUpdate")
        raise PreventUpdate
    
    if not ctx.triggered:
        print(f"           No trigger, raising PreventUpdate")
        raise PreventUpdate
    
    trig = ctx.triggered[0]['prop_id']
    trig_value = ctx.triggered[0]['value']
    
    print(f"   triggered_id: {trig}")
    print(f"   triggered_value: {trig_value}")
    
    if "finetune-group-header" in trig:
        if trig_value and (isinstance(trig_value, (int, float)) and trig_value > 0):
            try:
                info = json.loads(trig.split('.')[0])
                group_name = info.get("index")
                print(f"   Finetune group header click: {group_name}")
                
                try:
                    user_finetuned_path = FILE_PATHS["user_finetuned_list"]
                    filtered_path = FILE_PATHS["filtered_group_assignment"]
                    bm25_path = FILE_PATHS["bm25_search_results"]
                    
                    loaded_from = None
                    matched_dict = None
                    
                    if os.path.exists(user_finetuned_path):
                        with open(user_finetuned_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "user_finetuned_list.json"
                    elif os.path.exists(filtered_path):
                        with open(filtered_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "filtered_group_assignment.json"
                    elif os.path.exists(bm25_path):
                        with open(bm25_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "bm25_search_results.json"
                    
                    if matched_dict:
                        print(f"   Data source: {loaded_from}")
                        if group_name in matched_dict:
                            doc_count = len(matched_dict[group_name])
                            print(f"   Group '{group_name}': {doc_count} documents")
                            if doc_count <= 20:
                                print(f"   Document indices: {sorted(matched_dict[group_name])}")
                        else:
                            print(f"   Group '{group_name}' not found in file")
                        
                        print(f"   All groups:")
                        total = 0
                        for grp, indices in matched_dict.items():
                            count = len(indices) if isinstance(indices, list) else 0
                            total += count
                            print(f"      {grp}: {count} documents")
                        print(f"      Total: {total} documents")
                except Exception as e:
                    print(f"   Error reading file: {e}")
                
                print(f"   [CLEAN] Clearing keyword selection and selected document")
                return group_name, None, None  
            except Exception as e:
                print(f"           Error parsing group header: {e}")
                raise PreventUpdate
        else:
            print(f"           Invalid n_clicks value: {trig_value}")
    
    print(f"           Not a valid group header click, raising PreventUpdate")
    raise PreventUpdate

@app.callback(
    [Output("finetune-selected-keyword", "data", allow_duplicate=True),
     Output("finetune-selected-group", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    [Input({"type": "finetune-select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def select_finetune_keyword_from_group(n_clicks, display_mode):

    ctx = dash.callback_context
    
    print(f"\n{'='*80}")
    print(f"   DEBUG: select_finetune_keyword_from_group called")
    print(f"{'='*80}")
    print(f"   display_mode: {display_mode}")
    print(f"   n_clicks: {n_clicks}")
    print(f"   ctx.triggered: {ctx.triggered}")
    
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id']
        triggered_value = ctx.triggered[0]['value']
        print(f"   triggered_id: {triggered_id}")
        print(f"   triggered_value: {triggered_value}")
    
    if display_mode != "finetune":
        print(f"           Not in finetune mode, raising PreventUpdate")
        raise PreventUpdate
    
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    if "finetune-select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            group = btn_info.get("group")  
            
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                print(f"   Finetune keyword click: '{keyword}' from group: '{group}'")
                
                try:
                    user_finetuned_path = FILE_PATHS["user_finetuned_list"]
                    filtered_path = FILE_PATHS["filtered_group_assignment"]
                    bm25_path = FILE_PATHS["bm25_search_results"]
                    
                    loaded_from = None
                    matched_dict = None
                    
                    if os.path.exists(user_finetuned_path):
                        with open(user_finetuned_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "user_finetuned_list.json"
                    elif os.path.exists(filtered_path):
                        with open(filtered_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "filtered_group_assignment.json"
                    elif os.path.exists(bm25_path):
                        with open(bm25_path, "r", encoding="utf-8") as f:
                            matched_dict = json.load(f)
                        loaded_from = "bm25_search_results.json"
                    
                    if matched_dict:
                        print(f"   Data source: {loaded_from}")
                        if group in matched_dict:
                            group_docs = matched_dict[group]
                            print(f"   Group '{group}': {len(group_docs)} documents")
                            
                            try:
                                df_local = pd.read_csv(FILE_PATHS["csv_path"])
                                keyword_doc_count = 0
                                keyword_doc_indices = []
                                for idx in group_docs:
                                    if idx < len(df_local):
                                        text = str(df_local.iloc[idx, 1]).lower()
                                        if keyword.lower() in text:
                                            keyword_doc_count += 1
                                            keyword_doc_indices.append(idx)
                                
                                print(f"   Keyword '{keyword}' in group: {keyword_doc_count} documents")
                                if keyword_doc_count <= 20:
                                    print(f"   Document indices: {sorted(keyword_doc_indices)}")
                            except Exception as e:
                                print(f"   Error counting keyword documents: {e}")
                        else:
                            print(f"   Group '{group}' not found in file")
                except Exception as e:
                    print(f"   Error reading file: {e}")
                
                print(f"   Returning: keyword='{keyword}', group='{group}'")
                return keyword, group, None  
        except Exception as e:
            print(f"        Error parsing finetune keyword click: {e}")
            raise PreventUpdate
    
    raise PreventUpdate

@app.callback(
    Output("finetune-articles-container", "children"),
    [Input("finetune-selected-group", "data"),
     Input("finetune-selected-keyword", "data"),
     Input("finetune-highlight-core", "data"),
     Input("finetune-highlight-gray", "data"),
     Input("finetune-selected-article-index", "data")],  
    [State("group-order", "data"),
     State("display-mode", "data")]
)
def display_finetune_articles(selected_group, selected_keyword, core_indices, gray_indices, selected_article_idx, group_order, display_mode):

    if display_mode != "finetune":
        raise PreventUpdate
    
    if not selected_group and not selected_keyword:
        return html.P("Select a group or keyword to view documents", 
                     style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "40px 20px"})
    
    try:
        global df
        if 'df' not in globals():
            return html.P("Data not loaded", style={"color": "#e74c3c", "textAlign": "center"})
        
        doc_indices = []
        doc_types = {}  
        
        if core_indices:
            for idx in core_indices:
                doc_indices.append(idx)
                doc_types[idx] = "core"
        
        if gray_indices:
            for idx in gray_indices:
                doc_indices.append(idx)
                doc_types[idx] = "gray"
        
        if not doc_indices:
            return html.P("No documents to display", 
                         style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
        
        articles = []
        for i, idx in enumerate(sorted(doc_indices)):
            if idx >= len(df):
                continue
            
            text = str(df.iloc[idx, 1])[:200] + "..." if len(str(df.iloc[idx, 1])) > 200 else str(df.iloc[idx, 1])
            
            doc_type = doc_types.get(idx, "background")
            
            
            if doc_type == "core":
                type_label = "Core"
                type_color = "#FFD700"
                border_color = "#FFD700"
            elif doc_type == "gray":
                type_label = " Gray"
                type_color = "#808080"
                border_color = "#808080"
            else:
                type_label = "        Background"
                type_color = "#1f77b4"
                border_color = "#1f77b4"
            
            is_selected = (selected_article_idx is not None and selected_article_idx == idx)
            
            if is_selected:
                card_style = {
                    "padding": "12px",
                    "marginBottom": "10px",
                    "borderRadius": "6px",
                    "border": "2px solid #FF4444",  
                    "backgroundColor": "white",
                    "cursor": "pointer",
                    "transition": "all 0.2s ease",
                    "boxShadow": "0 2px 6px rgba(255, 68, 68, 0.2)",  
                    "scrollMarginTop": "100px"
                }
                doc_label_style = {"fontWeight": "bold", "marginRight": "10px", "color": "#FF4444"}  
            else:
                card_style = {
                    "padding": "12px",
                    "marginBottom": "10px",
                    "borderRadius": "6px",
                    "border": f"1px solid {border_color}",
                    "backgroundColor": "white",
                    "cursor": "pointer",
                    "transition": "all 0.2s ease",
                    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"
                }
                doc_label_style = {"fontWeight": "bold", "marginRight": "10px", "color": "#2c3e50"}
            
            article_card = html.Div([
                html.Div([
                    html.Span(f"Doc {idx+1}", style=doc_label_style),  
                    html.Span(type_label, style={"fontSize": "0.85rem", "color": type_color, "fontWeight": "bold"})
                ], style={"marginBottom": "8px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                html.P(text, style={"color": "#34495e", "fontSize": "0.9rem", "margin": "0", "lineHeight": "1.4"})
            ], id={"type": "finetune-article-card", "index": idx}, className="finetune-doc-card", style=card_style)
            articles.append(article_card)
        
        title_text = ""
        if selected_keyword and selected_group:
            title_text = f"Showing {len(doc_indices)} documents for '{selected_keyword}' in {selected_group}"
        elif selected_keyword:
            title_text = f"Showing {len(doc_indices)} documents for '{selected_keyword}'"
        elif selected_group:
            title_text = f"Showing {len(doc_indices)} documents in {selected_group}"
        
        return [
            html.P(title_text, style={"color": "#2c3e50", "fontWeight": "bold", "marginBottom": "15px", "textAlign": "center", "fontSize": "0.95rem"}),
            html.Div(articles)
        ]
        
    except Exception as e:
        print(f"        Error displaying finetune articles: {e}")
           
        traceback.print_exc()
        return html.P(f"Error: {str(e)}", style={"color": "#e74c3c", "textAlign": "center"})

@app.callback(
    Output("finetune-selected-article-index", "data", allow_duplicate=True),
    Input({"type": "finetune-article-card", "index": ALL}, "n_clicks"),
    [State("display-mode", "data"),
     State("finetune-selected-article-index", "data")],
    prevent_initial_call=True
)
def handle_finetune_article_click(n_clicks, display_mode, current_selected):

    if display_mode != "finetune":
        raise PreventUpdate
    
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_value = ctx.triggered[0]['value']
    
    if not triggered_value or triggered_value == 0:
        raise PreventUpdate
    
    try:
      
        card_info = json.loads(triggered_id.split('.')[0])
        article_idx = card_info.get("index")
        
        print(f"    Finetune article card clicked: Doc {article_idx+1}")
        
        return article_idx
        
    except Exception as e:
        print(f"        Error handling finetune article click: {e}")
           
        traceback.print_exc()
        return current_selected

@app.callback(
    Output("finetune-text-container", "children"),
    Input("finetune-selected-article-index", "data"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def update_finetune_text_preview(selected_idx, display_mode):

    if display_mode != "finetune":
        raise PreventUpdate
    
    if selected_idx is None:
        return html.P("Click a document to preview", 
                     style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "20px", "fontSize": "0.9rem"})
    
    try:
        global df
        if 'df' not in globals() or selected_idx >= len(df):
            return html.P("Document not found", style={"color": "#e74c3c"})
        
        full_text = str(df.iloc[selected_idx, 1])
        
        true_label = "Unknown"
        if len(df.columns) > 0:
            true_label = str(df.iloc[selected_idx, 0])
        
        preview = html.Div([
            html.H5(f"Document {selected_idx+1}", style={"color": "#2c3e50", "marginBottom": "10px", "fontSize": "1rem"}),
            html.Div([
                html.Span("True Label: ", style={"fontWeight": "bold", "color": "#2c3e50"}),
                html.Span(true_label, style={
                    "backgroundColor": "#3498db",
                    "color": "white",
                    "padding": "2px 8px",
                    "borderRadius": "4px",
                    "fontSize": "0.8rem",
                    "fontWeight": "bold"
                })
            ], style={"marginBottom": "15px"}),
            html.P(full_text, style={
                "color": "#34495e", 
                "fontSize": "0.85rem", 
                "lineHeight": "1.5",
                "whiteSpace": "pre-wrap",
                "wordBreak": "break-word"
            })
        ])
        
        return preview
        
    except Exception as e:
        print(f"        Error updating finetune text preview: {e}")
           
        traceback.print_exc()
        return html.P(f"Error: {str(e)}", style={"color": "#e74c3c"})

@app.callback(
    [Output("finetune-highlight-core", "data"),
     Output("finetune-highlight-gray", "data"),
     Output("finetune-operation-buttons", "children")],  
    [Input("finetune-selected-group", "data"),
     Input("finetune-selected-keyword", "data"),
     Input("finetune-selected-article-index", "data")],  
    [State("group-order", "data"),
     State("finetune-temp-assignments", "data")]
)
def compute_finetune_highlights(selected_group, selected_keyword, selected_article_idx, group_order, temp_assignments):
    global df, current_group_order
    core, gray = [], []
    operation_buttons = []
    
    current_group_order = group_order
    
    print(f"   DEBUG: compute_finetune_highlights called:")
    print(f"   selected_group: {selected_group}")
    print(f"   selected_keyword: {selected_keyword}")
    print(f"   selected_article_idx: {selected_article_idx}")
    print(f"   group_order: {group_order}")
    print(f"   DEBUG: group_order details:")
    for grp_name, kw_list in group_order.items():
        print(f"     {grp_name}: {kw_list}")
    print(f"   DEBUG: Set current_group_order = {current_group_order}")
    
    excluded_group = None
    if selected_article_idx is not None and group_order:
        try:
            matched_dict_path = FILE_PATHS["filtered_group_assignment"]
            if os.path.exists(matched_dict_path):
                with open(matched_dict_path, "r", encoding="utf-8") as f:
                    matched_dict = json.load(f)
                
                for grp_name in matched_dict.keys():
                    if isinstance(matched_dict[grp_name], list) and len(matched_dict[grp_name]) > 0:
                        if isinstance(matched_dict[grp_name][0], str):
                            matched_dict[grp_name] = [int(x) for x in matched_dict[grp_name]]
                
                for grp_name, indices in matched_dict.items():
                    try:
                        if isinstance(indices, list):
                            int_indices = []
                            for idx in indices:
                                try:
                                    int_indices.append(int(idx))
                                except (ValueError, TypeError):
                                    continue
                            
                            try:
                                selected_idx = int(selected_article_idx)
                                if selected_idx in int_indices:
                                    excluded_group = grp_name
                                    print(f"           Selected doc {selected_article_idx} belongs to: {excluded_group}")
                                    break
                            except (ValueError, TypeError):
                                print(f"           Invalid selected_article_idx: {selected_article_idx}")
                                continue
                    except Exception as e:
                        print(f"           Error processing indices for group {grp_name}: {e}")
                        continue
        except Exception as e:
            print(f"       Could not determine document's group: {e}")
    
    if temp_assignments and selected_article_idx is not None:
        for idx_str, target_group in temp_assignments.items():
            if idx_str.endswith("_original"):
                continue
            try:
                if int(idx_str) == int(selected_article_idx):
                    excluded_group = target_group
                    print(f"           Selected doc {selected_article_idx} was moved to: {excluded_group}")
                    break
            except (ValueError, TypeError):
                continue
    
    print(f"    DEBUG: Generating operation buttons...")
    print(f"   group_order: {group_order}")
    print(f"   selected_article_idx: {selected_article_idx}")
    
    if group_order and selected_article_idx is not None:
        print(f"        Conditions met, generating buttons for {len(group_order)} groups")
        button_style_base = {
            "color": "white",
            "border": "none",
            "padding": "10px 16px",
            "borderRadius": "6px",
            "fontSize": "0.95rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "width": "100%",
            "marginBottom": "8px"
        }
        
        for group_name in group_order.keys():

            should_exclude = False
            if excluded_group is not None:
                if group_name == excluded_group:
                    should_exclude = True
               
                elif group_name == "Other" and excluded_group == "Exclude":
                    should_exclude = True
                elif group_name == "Exclude" and excluded_group == "Other":
                    should_exclude = True
            
            if should_exclude:
                continue
                

            if group_name == "Group 1":
                bg_color = "#FF6B6B"  
            elif group_name == "Group 2":
                bg_color = "#32CD32"  
            elif group_name == "Other":
                bg_color = "#e74c3c"  
            else:
                bg_color = "#3498db"  
            
            button_style = {**button_style_base, "backgroundColor": bg_color}
            

            if group_name == "Exclude" and group_order.get("Exclude"):
                button_text = "Move to Exclude"
            else:
                button_text = f"Move to {group_name}"
            
            operation_buttons.append(
                html.Button(
                    button_text,
                    id={"type": "finetune-move-to", "target": group_name},
                    n_clicks=0,
                    style=button_style
                )
            )
        
        print(f"        Generated {len(operation_buttons)} operation buttons (excluding {excluded_group})")
    else:
        print(f"       Conditions not met, showing placeholder message")
        operation_buttons = [
            html.P("Select a document to perform operations", 
                   style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "10px"})
        ]
    
    print(f"    DEBUG: Returning {len(operation_buttons)} button(s)")
    
    if selected_keyword and not selected_group:
        try:
            keyword_docs = []
            for i in range(len(df)):
                text_lower = str(df.iloc[i, 1]).lower()
                if selected_keyword.lower() in text_lower:
                    keyword_docs.append(i)
            print(f"    Finetune keyword '{selected_keyword}': {len(keyword_docs)} documents ")
            return keyword_docs, [], operation_buttons
        except Exception as e:
            print(f"        Error highlighting keyword in finetune mode: {e}")
            return [], [], operation_buttons
    
    if not selected_group or not group_order or selected_group not in group_order:
        return core, gray, operation_buttons
    
    try:
 
        user_finetuned_path = FILE_PATHS["user_finetuned_list"]
        matched_dict_path = None
        
        if os.path.exists(user_finetuned_path):
            matched_dict_path = user_finetuned_path
            print(f"      Using user finetuned results from {os.path.basename(matched_dict_path)}")
        elif os.path.exists(FILE_PATHS["filtered_group_assignment"]):
            matched_dict_path = FILE_PATHS["filtered_group_assignment"]
            print(f"      Using filtered group assignment from {os.path.basename(matched_dict_path)}")
        elif os.path.exists(FILE_PATHS["bm25_search_results"]):
            matched_dict_path = FILE_PATHS["bm25_search_results"]
            print(f"      Using BM25 search results from {os.path.basename(matched_dict_path)}")
        else:
            print("      No group assignment data found")
            return core, gray, operation_buttons
        
        print(f"\n{'='*80}")
        print(f"LOADING GROUP ASSIGNMENTS FOR FINETUNE")
        print(f"{'='*80}")
        print(f"   File path: {matched_dict_path}")
        
        with open(matched_dict_path, "r", encoding="utf-8") as f:
            matched_dict = json.load(f)
        
        for grp_name in matched_dict.keys():
            if isinstance(matched_dict[grp_name], list) and len(matched_dict[grp_name]) > 0:
                if isinstance(matched_dict[grp_name][0], str):
                    matched_dict[grp_name] = [int(x) for x in matched_dict[grp_name]]
        
        print(f"   Loaded from file:")
        for grp_name, indices in matched_dict.items():
            print(f"      {grp_name}: {len(indices)} documents")
            if len(indices) <= 15:
                print(f"         索引: {sorted(indices)}")
        print(f"{'='*80}\n")
        

        if temp_assignments:
            print(f"\n{'='*80}")
            print(f"APPLYING USER ADJUSTMENTS (temp_assignments)")
            print(f"{'='*80}")
            print(f"   Total adjustments: {len(temp_assignments)}")
            for idx_str, target_group in temp_assignments.items():
                if not idx_str.endswith("_original"):
                    print(f"      {idx_str} -> {target_group}")
            print(f"{'='*80}\n")
            
            for idx_str, target_group in temp_assignments.items():
                if idx_str.endswith("_original"):
                    continue
                
                try:
                    idx = int(idx_str)
                    
                    removed_from = None
                    for grp_name in matched_dict.keys():
                        if idx in matched_dict[grp_name]:
                            matched_dict[grp_name].remove(idx)
                            removed_from = grp_name
                            print(f"      Removed doc {idx} from {grp_name}")
                            break
                    
                    if target_group in matched_dict:
                        matched_dict[target_group].append(idx)
                        print(f"      Added doc {idx} to {target_group}")
                    elif target_group not in matched_dict:
                        matched_dict[target_group] = [idx]
                        print(f"      Created new group {target_group} with doc {idx}")
                        
                except Exception as e:
                    print(f"      Error applying adjustment for {idx_str}: {e}")
            
            print(f"\n{'='*80}")
            print(f"ADJUSTMENTS APPLIED - UPDATED DISTRIBUTION:")
            print(f"{'='*80}")
            for grp_name, indices in matched_dict.items():
                print(f"   {grp_name}: {len(indices)} documents")
                if len(indices) <= 15:
                    print(f"      Indices: {sorted(indices)}")
            print(f"{'='*80}\n")
        else:
            print(f"   No temp_assignments to apply")
        
        if selected_group == "Exclude":
            exclude_has_keywords = False
            if group_order and "Exclude" in group_order and group_order["Exclude"]:
                exclude_has_keywords = True
                print(f"  Exclude group has user-defined keywords: {group_order['Exclude']}")
                print(f"  Treating Exclude as Exclude group with prototype")
            else:
                print(f"  Exclude group has no user-defined keywords - treating as exclude group")
            
            try:
                if exclude_has_keywords:
                   
                    exclude_keywords = group_order.get("Exclude", [])
                    print(f"  Exclude keywords: {exclude_keywords}")
                    
                    
                    filtered_exclude_indices = []
                    if "Exclude" in matched_dict:
                        filtered_exclude_indices = matched_dict["Exclude"]
                        print(f"  Filtered Exclude documents: {len(filtered_exclude_indices)} documents")
                    
                   
                    manually_moved_indices = []
                    if temp_assignments:
                        for idx_str, target_group in temp_assignments.items():
                            if idx_str.endswith("_original"):
                                continue
                            if target_group == "Exclude":  
                                manually_moved_indices.append(int(idx_str))
                        print(f"  User manually moved to Exclude: {len(manually_moved_indices)} documents")
                    
                    all_exclude_indices = list(set(filtered_exclude_indices + manually_moved_indices))
                    print(f"  Total Exclude documents: {len(all_exclude_indices)} documents")
                    print(f"    All Exclude documents: {sorted(all_exclude_indices) if len(all_exclude_indices) <= 20 else 'too many to display'}")
                    
                    return all_exclude_indices, [], operation_buttons
                else:
                
                    manually_moved_indices = []
                    if temp_assignments:
                        for idx_str, target_group in temp_assignments.items():
                            if idx_str.endswith("_original"):
                                continue
                            if target_group == "Exclude":  
                                manually_moved_indices.append(int(idx_str))
                        print(f"  User manually moved to Exclude (no keywords): {len(manually_moved_indices)} documents")
                    
                    return manually_moved_indices, [], operation_buttons
            except Exception as e:
                print(f"  Error getting Exclude group indices: {e}")
                exclude_indices = matched_dict.get("Exclude", [])
                return exclude_indices, [], operation_buttons

        selected_group_indices = matched_dict.get(selected_group, [])
        
        model_path = FILE_PATHS["triplet_trained_encoder"]
        if not os.path.exists(model_path):

            group_indices = matched_dict.get(selected_group, [])
            return group_indices, [], operation_buttons
        
        print(f"    Computing gap-based highlights for group: {selected_group}")
        
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"        Using device: {device}")
        


        print(f"  Loading state_dict from: {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        

        print(f"  State_dict type: {type(state_dict)}")
        print(f"  State_dict keys count: {len(state_dict.keys())}")
        print(f"  First 10 state_dict keys:")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"    {i+1}. {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
        

        if 'proj.weight' in state_dict:
            proj_weight_shape = state_dict['proj.weight'].shape
            proj_dim = proj_weight_shape[0]  
            bert_hidden_size = proj_weight_shape[1]  
            print(f"  Detected from proj.weight: proj_dim={proj_dim}, bert_hidden_size={bert_hidden_size}")
        else:
            proj_dim = 256
            bert_hidden_size = 768
            print(f"  Using default values: proj_dim={proj_dim}, bert_hidden_size={bert_hidden_size}")

        has_ln = 'ln.weight' in state_dict
        has_norm = 'norm.weight' in state_dict
        print(f"  LayerNorm detection: has_ln={has_ln}, has_norm={has_norm}")

        if bert_hidden_size == 768:
            bert_name = "bert-base-uncased"  
        elif bert_hidden_size == 384:
            bert_name = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            bert_name = get_config("bert_model", "sentence-transformers/all-MiniLM-L6-v2")
        
        print(f"  Selected BERT model: {bert_name}")
        
            
        print(f"  Creating SentenceEncoder with bert_name={bert_name}, proj_dim={proj_dim}, device=cpu")
        encoder = SentenceEncoder(bert_name=bert_name, proj_dim=proj_dim, device="cpu")
        
        
        print(f"  Created model structure:")
        print(f"    - bert config hidden_size: {encoder.bert.config.hidden_size}")
        print(f"    - proj layer: {encoder.proj}")
        print(f"    - ln layer: {encoder.ln}")
        print(f"    - out_dim: {encoder.out_dim}")
        
        
        print(f"  State_dict vs Model parameter matching:")
        model_state_dict = encoder.state_dict()
        print(f"    Model state_dict keys count: {len(model_state_dict.keys())}")
        print(f"    First 10 model keys:")
        for i, key in enumerate(list(model_state_dict.keys())[:10]):
            print(f"      {i+1}. {key}: {model_state_dict[key].shape}")
        
        
        print(f"  Key parameter matching check:")
        for key in ['proj.weight', 'proj.bias', 'ln.weight', 'ln.bias']:
            if key in state_dict and key in model_state_dict:
                state_shape = state_dict[key].shape
                model_shape = model_state_dict[key].shape
                match = state_shape == model_shape
                print(f"    {key}: state_dict{state_shape} vs model{model_shape} -> {'OK' if match else 'MISMATCH'}")
            elif key in state_dict:
                print(f"    {key}: in state_dict only (shape: {state_dict[key].shape})")
            elif key in model_state_dict:
                print(f"    {key}: in model only (shape: {model_state_dict[key].shape})")

        print(f"  Attempting to load state_dict...")
        try:
            missing_keys, unexpected_keys = encoder.load_state_dict(state_dict, strict=False)
            print(f"  State_dict loaded successfully!")
            print(f"    Missing keys: {len(missing_keys)}")
            if missing_keys:
                print(f"    Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            print(f"    Unexpected keys: {len(unexpected_keys)}")
            if unexpected_keys:
                print(f"    Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        except Exception as load_error:
            print(f"  ERROR loading state_dict: {load_error}")
            print(f"  This explains the CUDA index out of bounds error!")
            raise load_error
        
        encoder.eval()
        encoder.bert_model_name = bert_name
        print(f"  Model loaded and ready on CPU")

        if device.type == 'cuda':
            print(f" MOVING MODEL TO GPU DEBUG ")
            try:
                print(f"  Testing CUDA availability...")
                test_tensor = torch.tensor([1.0]).cuda()
                print(f"  CUDA test tensor created: {test_tensor}")
                del test_tensor
                torch.cuda.empty_cache()
                print(f"  CUDA cache cleared successfully")
                
                print(f"  Moving model to GPU...")
                encoder = encoder.cuda()
                print(f"  Model moved to GPU successfully")

                print(f"  Testing model on GPU...")
                test_input = {
                    'input_ids': torch.tensor([[101, 102]]).cuda(),
                    'attention_mask': torch.tensor([[1, 1]]).cuda(),
                    'token_type_ids': torch.tensor([[0, 0]]).cuda()
                }
                with torch.no_grad():
                    test_output = encoder.encode_tokens(test_input)
                print(f"  GPU test successful, output shape: {test_output.shape}")

                
            except Exception as gpu_error:
                print(f"  ERROR moving to GPU: {gpu_error}")
                print(f"  Falling back to CPU...")
                device = torch.device("cpu")
                encoder = encoder.cpu()
                print(f"  Model moved back to CPU")

        else:
            print(f"  Using CPU device as requested")
        

        
        if hasattr(encoder, 'bert_model_name'):
            tokenizer_name = encoder.bert_model_name
        elif hasattr(encoder, 'bert') and hasattr(encoder.bert, 'name_or_path'):
            tokenizer_name = encoder.bert.name_or_path
        else:
            hidden_size = encoder.bert.config.hidden_size if hasattr(encoder, 'bert') else 768
            tokenizer_name = "bert-base-uncased" if hidden_size == 768 else "sentence-transformers/all-MiniLM-L6-v2"
        
        print(f"  Using tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        texts = df.iloc[:, 1].astype(str).tolist()
        batch_size = 32
        all_embeddings = []

        print(f"  Total texts to encode: {len(texts)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
        print(f"  Encoder device: {next(encoder.parameters()).device}")
        
        with torch.no_grad():
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch_num = i//batch_size + 1
                    total_batches = (len(texts)-1)//batch_size + 1

                    print(f"    Batch size: {len(batch)}")
                    print(f"    Sample text: {batch[0][:100]}..." if batch else "Empty batch")

                    print(f"    Tokenizing batch...")
                    inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
                    print(f"    Tokenization completed")

                    vocab_size = tokenizer.vocab_size
                    if 'input_ids' in inputs:

                        min_id = inputs['input_ids'].min().item()
                        max_id = inputs['input_ids'].max().item()
                        print(f"    Token ID range: {min_id} to {max_id} (vocab_size: {vocab_size})")
                        
                        if max_id >= vocab_size:
                            print(f"    WARNING: Found token ID {max_id} >= vocab_size {vocab_size}, clamping...")
                            inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, vocab_size - 1)
                            print(f"    After clamping: {inputs['input_ids'].min().item()} to {inputs['input_ids'].max().item()}")
                        else:
                            print(f"    Token IDs are within valid range")

                    print(f"    Moving inputs to device: {device}")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    print(f"    Inputs moved to device successfully")

                    print(f"    Input shapes and devices:")
                    for k, v in inputs.items():
                        print(f"      {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}")

                    print(f"    Starting encoding...")
                    try:
                        embeds = encoder.encode_tokens(inputs)
                        print(f"    Encoding successful!")
                        print(f"    Output shape: {embeds.shape}")
                        print(f"    Output device: {embeds.device}")
                        print(f"    Output dtype: {embeds.dtype}")

                        if torch.isnan(embeds).any():
                            print(f"    WARNING: Output contains NaN values!")
                        if torch.isinf(embeds).any():
                            print(f"    WARNING: Output contains Inf values!")
                        
                        all_embeddings.append(embeds.cpu().numpy())
                        print(f"    Batch {batch_num} completed successfully")
                        
                    except Exception as encode_error:
                        print(f"    ERROR during encoding: {encode_error}")
                        print(f"    This is likely the source of CUDA index out of bounds!")
                        raise encode_error

            except Exception as encoding_error:
                print(f"CUDA error during encoding, falling back to CPU: {encoding_error}")

                encoder = SentenceEncoder(bert_name=bert_name, proj_dim=proj_dim, device="cpu")
                encoder.load_state_dict(state_dict, strict=False)
                encoder.eval()

                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')

                    vocab_size = tokenizer.vocab_size
                    if 'input_ids' in inputs:

                        min_id = inputs['input_ids'].min().item()
                        max_id = inputs['input_ids'].max().item()
                        print(f"    CPU Token ID range: {min_id} to {max_id} (vocab_size: {vocab_size})")
                        
                        if max_id >= vocab_size:
                            print(f"    CPU WARNING: Found token ID {max_id} >= vocab_size {vocab_size}, clamping...")
                            inputs['input_ids'] = torch.clamp(inputs['input_ids'], 0, vocab_size - 1)
                            print(f"    CPU After clamping: {inputs['input_ids'].min().item()} to {inputs['input_ids'].max().item()}")
                        else:
                            print(f"    CPU Token IDs are within valid range")
                    
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    embeds = encoder.encode_tokens(inputs)
                    all_embeddings.append(embeds.cpu().numpy())
        
        Z_raw = np.vstack(all_embeddings)
        
        Z_norm = Z_raw / (np.linalg.norm(Z_raw, axis=1, keepdims=True) + 1e-8)
        
        group_centers = {}
        group_names_list = []
        for group_name, indices in matched_dict.items():
            if group_name == "Other" or len(indices) == 0:
                continue
            valid_indices = [idx for idx in indices if idx < len(Z_norm)]
            if len(valid_indices) > 0:
                group_center = np.mean(Z_norm[valid_indices], axis=0)
                group_center = group_center / (np.linalg.norm(group_center) + 1e-8)
                group_centers[group_name] = group_center
                group_names_list.append(group_name)
        
        if len(group_centers) == 0:

            return core, gray, operation_buttons
        
        all_similarities = []
        for group_name in group_names_list:
            center = group_centers[group_name]
            sim = np.dot(Z_norm, center)
            all_similarities.append(sim)
        
        all_similarities = np.array(all_similarities).T  
        
        sorted_indices = np.argsort(all_similarities, axis=1)[:, ::-1]  
        s_top1 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 0]]
        s_top2 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 1]] if all_similarities.shape[1] > 1 else s_top1
        gap = s_top1 - s_top2
        
        arg1 = sorted_indices[:, 0]  
        
        if selected_group not in group_names_list:
            return core, gray, operation_buttons
        
        selected_group_indices = matched_dict.get(selected_group, [])
        if len(selected_group_indices) == 0:
            print(f"  {selected_group} ")
            return core, gray, operation_buttons
        
        print(f"  {selected_group} {len(selected_group_indices)} ")
        
        group_mask = np.zeros(len(df), dtype=bool)
        group_mask[selected_group_indices] = True

        valid_selected_indices = [idx for idx in selected_group_indices if idx < len(gap)]
        if len(valid_selected_indices) == 0:
            print(f"  No valid indices for {selected_group}")
            return core, gray, operation_buttons
        
        gaps_group = gap[valid_selected_indices]
        mean_gap = gaps_group.mean()
        std_gap = gaps_group.std()
        
        alpha = get_config("gap_alpha", 1.0)
        min_samples = get_config("gap_min_samples", 10)
        percentile_fallback = get_config("gap_percentile_fallback", 25)
        thr_floor = get_config("gap_floor_threshold", 0.05)
        mix_ratio = get_config("gap_mix_ratio", 0.3)
        

        base_threshold = mean_gap - alpha * std_gap
        

        if len(gaps_group) < min_samples or std_gap < 1e-6:
            gray_threshold = np.percentile(gaps_group, percentile_fallback)
            core_threshold = np.percentile(gaps_group, 50) 
            print(f"    {selected_group}: {len(gaps_group)}, std:{std_gap:.6f}")
        else:

            global_median = np.median(gap)
            gray_threshold = max((1 - mix_ratio) * base_threshold + mix_ratio * global_median, thr_floor)

            core_threshold = mean_gap  
        

        selected_center = group_centers[selected_group]
        selected_center_norm = selected_center / (np.linalg.norm(selected_center) + 1e-8)
        

        distances_to_center = 1 - np.dot(Z_norm, selected_center_norm) 
        distances_group = distances_to_center[group_mask]
        

        mean_dist = distances_group.mean()
        std_dist = distances_group.std()
        

        beta = 1.5 
        prototype_radius = mean_dist + beta * std_dist
        
        print(f"    {selected_group}: mean={mean_gap:.3f}, std={std_gap:.3f}")
        print(f"      core_thr={core_threshold:.3f}, gray_thr={gray_threshold:.3f}")
        print(f"      prototype_radius={prototype_radius:.3f} (mean_dist={mean_dist:.3f}, std_dist={std_dist:.3f})")
        

        print(f"    {len(selected_group_indices)}...")
        for idx in selected_group_indices:
            gap_val = gap[idx]
            dist_to_center = distances_to_center[idx]
            

            if gap_val >= core_threshold and dist_to_center <= prototype_radius:
                core.append(idx)

            else:
                gray.append(idx)
        
        print(f"     {selected_group}: {len(selected_group_indices)} ")
        print(f"   {len(core)} - {core[:10]}")
        print(f"   {len(gray)} - {gray[:10]}")
        print(f"   {len(core) + len(gray)} {len(selected_group_indices)}")
        
        if len(core) + len(gray) != len(selected_group_indices):
            print(f"    ")

        if selected_keyword:
            print(f"    '{selected_keyword}'...")
            keyword_core = []
            keyword_gray = []
            
            for idx in core:
                text_lower = str(df.iloc[idx, 1]).lower()
                if selected_keyword.lower() in text_lower:
                    keyword_core.append(idx)
            
            for idx in gray:
                text_lower = str(df.iloc[idx, 1]).lower()
                if selected_keyword.lower() in text_lower:
                    keyword_gray.append(idx)
            
            print(f"   {len(keyword_core)} ")
            print(f"   {len(keyword_gray)} ")
            print(f"   {len(keyword_core) + len(keyword_gray)} ")
            

            print(f"  Keyword core indices: {len(keyword_core)} - {keyword_core[:5]}")
            print(f"  Keyword gray indices: {len(keyword_gray)} - {keyword_gray[:5]}")
            print(f"  Operation buttons: {len(operation_buttons)}")

            return keyword_core, keyword_gray, operation_buttons
        

        print(f"  Core indices: {len(core)} - {core[:5]}")
        print(f"  Gray indices: {len(gray)} - {gray[:5]}")
        print(f"  Operation buttons: {len(operation_buttons)}")

        return core, gray, operation_buttons
        
    except Exception as e:
        print(f"        Error in gap-based filtering: {e}")
           
        traceback.print_exc()

        if "CUDA" in str(e) or "device-side assert" in str(e):
            print("        CUDA error detected, but skipping cache cleanup to avoid further errors")
        
        return core, gray, operation_buttons


@app.callback(
    Output('finetune-2d-plot', 'figure'),
    [Input('display-mode', 'data'),
     Input('finetune-highlight-core', 'data'),
     Input('finetune-highlight-gray', 'data'),
     Input('finetune-selected-article-index', 'data')],  
    [State('training-figures', 'data'),
     State('finetune-figures', 'data'), 
     State('finetune-selected-group', 'data')]
)
def render_finetune_plot(display_mode, core_indices, gray_indices, selected_article_idx, training_figures, finetune_figures, selected_group):
    if display_mode != "finetune":
        raise PreventUpdate
    

    active_figure = training_figures.get('after') if training_figures else None
    
   
    idx_to_coord = {}
    
    try:
        if isinstance(active_figure, dict):
            after = active_figure
            if after.get('data') and len(after['data']) > 0:
                print(f"      Extracting coordinates from {len(after['data'])} traces in 'after' figure")
                
               
                for trace in after['data']:
                    trace_x = trace.get('x', [])
                    trace_y = trace.get('y', [])
                    trace_customdata = trace.get('customdata', [])
                    
                    if trace_x and trace_y:
                        for i, (x, y) in enumerate(zip(trace_x, trace_y)):
                           
                            if i < len(trace_customdata) and trace_customdata[i]:
                                doc_idx = trace_customdata[i][0] if isinstance(trace_customdata[i], list) else trace_customdata[i]
                                idx_to_coord[doc_idx] = (x, y)
                
                print(f"   Total unique documents: {len(idx_to_coord)}")
    except Exception as e:
        print(f"        Error extracting coordinates: {e}")
           
        traceback.print_exc()


    if not idx_to_coord:
        print("    No coordinates from training figures, using cached t-SNE")
        if _GLOBAL_DOCUMENT_EMBEDDINGS_READY and _GLOBAL_DOCUMENT_TSNE is not None:
            coords = _GLOBAL_DOCUMENT_TSNE
            idx_to_coord = {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}
        else:

            return {
                'data': [],
                'layout': {'title': 'Finetune - No coordinates available'}
            }


    valid_indices = list(idx_to_coord.keys())
    print(f" Using {len(valid_indices)} coordinates for finetune plot")      
    if valid_indices:
        print(f"   Document indices range: {min(valid_indices)} to {max(valid_indices)}")
    
   
    all_idx = set(valid_indices)
    core_set = set(core_indices or []) & all_idx
    gray_set = set(gray_indices or []) & all_idx
    
    print(f"      Index filtering:")
    print(f"   Total valid indices: {len(all_idx)}")
    print(f"   Core indices provided: {len(core_indices or [])}")
    print(f"   Core indices (filtered): {len(core_set)}")
    print(f"   Gray indices provided: {len(gray_indices or [])}")
    print(f"   Gray indices (filtered): {len(gray_set)}")
    

    all_other_idx = list(all_idx - core_set - gray_set)
    print(f"   Background indices: {len(all_other_idx)}")
    

    traces = []

    if isinstance(training_figures, dict) and isinstance(training_figures.get('after'), dict):
        after_data = training_figures['after'].get('data', [])
        for trace in after_data:
           
            traces.append(trace.copy())
    

    if gray_set:
        gidx = list(gray_set)
        gray_style = PLOT_STYLES["gray"]
        traces.append({
            'x': [idx_to_coord[i][0] for i in gidx],
            'y': [idx_to_coord[i][1] for i in gidx],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Need Review',
            'marker': {
                'size': gray_style["size"],
                'color': gray_style["color"],
                'opacity': gray_style["opacity"],
                'symbol': gray_style["symbol"],
                'line': {'width': gray_style["line_width"], 'color': gray_style["line_color"]}
            },
            'text': [f'Doc {i+1}' for i in gidx],
            'customdata': [[i] for i in gidx],
            'hovertemplate': '<b>%{text}</b><br>    Need review<extra></extra>'
        })
    
  
    if core_set:
        cidx = list(core_set)
        core_style = PLOT_STYLES["core"]
        traces.append({
            'x': [idx_to_coord[i][0] for i in cidx],
            'y': [idx_to_coord[i][1] for i in cidx],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Selected Group',
            'marker': {
                'size': core_style["size"],
                'color': core_style["color"],
                'opacity': core_style["opacity"],
                'symbol': core_style["symbol"],
                'line': {'width': core_style["line_width"], 'color': core_style["line_color"]}
            },
            'text': [f'Doc {i+1}' for i in cidx],
            'customdata': [[i] for i in cidx],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        })
    

    if selected_article_idx is not None and selected_article_idx in idx_to_coord:
        print(f"    Highlighting selected document: Doc {selected_article_idx+1}")
        traces.append({
            'x': [idx_to_coord[selected_article_idx][0]],
            'y': [idx_to_coord[selected_article_idx][1]],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Selected Document',
            'marker': {
                'size': 16,  
                'color': '#FF4444',  
                'opacity': 1.0,
                'symbol': 'star',
                'line': {'width': 2, 'color': '#FF0000'} 
            },
            'text': [f'Doc {selected_article_idx+1} (Selected)'],
            'customdata': [[selected_article_idx]],
            'hovertemplate': '<b>%{text}</b><extra></extra>',
            'showlegend': True
        })

    print(f"      Finetune plot rendering:")
    print(f"   Core samples: {len(core_set)}")
    print(f"   Gray samples: {len(gray_set)}")
    print(f"   Background samples: {len(all_other_idx)}")
    print(f"   Selected document: {selected_article_idx if selected_article_idx is not None else 'None'}")
    print(f"   Total points: {len(core_set) + len(gray_set) + len(all_other_idx)}")
    
   
    layout_style = PLOT_STYLES["layout"]
    
    if isinstance(training_figures, dict) and isinstance(training_figures.get('after'), dict):
        
        import copy
        base_layout = copy.deepcopy(training_figures['after'].get('layout', {}))
        
        
        if 'title' in base_layout:
            if isinstance(base_layout['title'], dict):
                base_layout['title']['text'] = 'Finetune Mode - Interactive 2D Visualization'
            else:
                base_layout['title'] = {
                    'text': 'Finetune Mode - Interactive 2D Visualization',
                    'font': {'size': 18, 'color': '#2c3e50'},
                    'x': 0.5
                }
        else:
            base_layout['title'] = {
                'text': 'Finetune Mode - Interactive 2D Visualization',
                'font': {'size': 18, 'color': '#2c3e50'},
                'x': 0.5
            }
        
       
        base_layout['xaxis'] = {
            **base_layout.get('xaxis', {}),
            **layout_style["xaxis"],
            'title': base_layout.get('xaxis', {}).get('title', 'X')  
        }
        base_layout['yaxis'] = {
            **base_layout.get('yaxis', {}),
            **layout_style["yaxis"],
            'title': base_layout.get('yaxis', {}).get('title', 'Y') 
        }
        base_layout['plot_bgcolor'] = layout_style["plot_bgcolor"]
        base_layout['paper_bgcolor'] = layout_style["paper_bgcolor"]
    else:
       
        base_layout = {
            'title': {
                'text': 'Finetune Mode - Interactive 2D Visualization',
                'font': {'size': 18, 'color': '#2c3e50'},
                'x': 0.5
            },
            'xaxis': {**layout_style["xaxis"], 'title': 'X'},
            'yaxis': {**layout_style["yaxis"], 'title': 'Y'},
            'hovermode': 'closest',
            'showlegend': True,
            'plot_bgcolor': layout_style["plot_bgcolor"],
            'paper_bgcolor': layout_style["paper_bgcolor"],
            'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50}
        }
    
    fig = {
        'data': traces,
        'layout': base_layout
    }
    return fig


@app.callback(
    Output("finetune-selected-article-index", "data", allow_duplicate=True),
    Input('finetune-2d-plot', 'clickData'),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def finetune_click_2d_point(click_data, display_mode):

    if display_mode != "finetune":
        raise PreventUpdate
    
    if not click_data:
        raise PreventUpdate
    
    try:
        idx = click_data['points'][0]['customdata'][0]
        print(f"    Finetune 2D plot clicked: Doc {idx}")
        return idx
    except Exception as e:
        print(f"        Error parsing 2D click: {e}")
        raise PreventUpdate


@app.callback(
    [Output("finetune-temp-assignments", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    Input({"type": "finetune-move-to", "target": ALL}, "n_clicks"),
    [State("finetune-selected-article-index", "data"),
     State("finetune-temp-assignments", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def finetune_move_document(n_clicks_list, selected_idx, assignments, group_order):

    print(f"    DEBUG: finetune_move_document called")
    print(f"   n_clicks_list: {n_clicks_list}")
    print(f"   selected_idx: {selected_idx}")
    print(f"   assignments: {assignments}")
    
    if not dash.callback_context.triggered:
        print(f"           No context triggered")
        raise PreventUpdate
    
    triggered = dash.callback_context.triggered[0]
    print(f"   triggered: {triggered}")
    
    if selected_idx is None:
        print("        No document selected from article list")
        raise PreventUpdate
    
  
    triggered_id = triggered['prop_id']
    triggered_value = triggered['value']
    
    print(f"   triggered_id: {triggered_id}")
    print(f"   triggered_value: {triggered_value}")
    
    if triggered_id == '.':
        print(f"           Invalid triggered_id")
        raise PreventUpdate
    
  
    if triggered_value is None or triggered_value == 0:
        print(f"       Button not actually clicked (value={triggered_value}), preventing update")
        raise PreventUpdate
    
    try:
    
        
        button_id = json_module.loads(triggered_id.split('.')[0])
        target_group = button_id['target']
        
        new_map = dict(assignments or {})

        if str(selected_idx) not in assignments:
            try:
                original_group = "Unknown"

                try:
                    with open(FILE_PATHS["filtered_group_assignment"], "r") as f:
                        current_matched_dict = json_module.load(f)
                    for grp_name_file, indices in current_matched_dict.items():
                        if selected_idx in indices:
                            original_group = grp_name_file
                            print(f"     Found original group from filtered_group_assignment.json: {original_group}")
                            break
                except:
                    pass

                if original_group == "Unknown":
                    try:
                        with open(FILE_PATHS["group_assignment"], "r") as f:
                            current_matched_dict = json_module.load(f)
                        for grp_name_file, indices in current_matched_dict.items():
                            if selected_idx in indices:
                                original_group = grp_name_file
                                print(f"     Found original group from group_assignment.json: {original_group}")
                                break
                    except:
                        pass

                if original_group == "Unknown":
                    try:
                        with open(FILE_PATHS["bm25_search_results"], "r") as f:
                            current_matched_dict = json_module.load(f)
                        for grp_name_file, indices in current_matched_dict.items():
                            if selected_idx in indices:
                                original_group = grp_name_file
                                print(f"     Found original group from bm25_search_results.json: {original_group}")
                                break
                    except:
                        pass

                new_map[f"{selected_idx}_original"] = original_group
                print(f"     Recorded original group for Doc {selected_idx+1}: {original_group}")
            except Exception as e:
                print(f"     Could not determine original group: {e}")

                new_map[f"{selected_idx}_original"] = "Unknown"
        
        new_map[str(selected_idx)] = target_group
        print(f"     Moved Doc {selected_idx+1} to {target_group}")
        
      
        print(f"[CLEAN] Clearing selected document after move")
        return new_map, None
    except Exception as e:
        print(f"        Error moving document: {e}")
           
        traceback.print_exc()
        raise PreventUpdate


@app.callback(
    Output("finetune-temp-assignments", "data", allow_duplicate=True),
    Input("finetune-clear-history-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_finetune_history(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    print("Clearing adjustment history")
    return {}


@app.callback(
    [Output("finetune-adjustment-history", "children"),
     Output("finetune-history-buttons", "children")],
    [Input("finetune-temp-assignments", "data")]
)
def update_adjustment_history(temp_assignments):
    global df, current_group_order
    if not temp_assignments or len(temp_assignments) == 0:
        return html.P("No adjustments yet. Click a point and use the buttons to reassign.", 
                     style={
                         "color": "#7f8c8d", 
                         "fontStyle": "italic", 
                         "textAlign": "center", 
                         "padding": "20px",
                         "fontSize": "0.95rem"
                     }), []
    
  
    try:
        matched_dict_path = FILE_PATHS["filtered_group_assignment"]
        if not os.path.exists(matched_dict_path):
            matched_dict_path = FILE_PATHS["bm25_search_results"]
        
        with open(matched_dict_path, "r", encoding="utf-8") as f:
            matched_dict = json.load(f)
        
       
        idx_to_original_group = {}
        for grp_name, indices in matched_dict.items():
            for idx in indices:
                idx_to_original_group[idx] = grp_name
    except Exception as e:
        print(f"Error loading group assignments: {e}")
        idx_to_original_group = {}
    
    if 'df' not in globals():
        try:
            df = pd.read_csv(FILE_PATHS["csv_path"])
        except:
            df = None
    
   
    history_items = []
    history_buttons = []

    if temp_assignments:

        rounds = set()
        for key in temp_assignments.keys():
            if key.startswith('round_'):
                round_num = key.split('_')[1]
                rounds.add(int(round_num))
        
        for round_num in sorted(rounds):
            history_buttons.append(
                html.Button(f"Round {round_num} Adjustments", 
                           id={"type": "history-round", "index": round_num},
                           style={
                               "backgroundColor": "#3498db",
                               "color": "white",
                               "border": "none",
                               "padding": "8px 16px",
                               "borderRadius": "4px",
                               "margin": "5px",
                               "cursor": "pointer",
                               "fontSize": "0.9rem"
                           })
            )
    
    for idx_str, new_group in temp_assignments.items():

        if idx_str.endswith("_original"):
            continue
            
        idx = int(idx_str)

        original_key = f"{idx}_original"
        if original_key in temp_assignments:
            original_group = temp_assignments[original_key]
        else:
            original_group = idx_to_original_group.get(idx, "Unknown")
        
        doc_preview = "..."
        if df is not None and idx < len(df):
            doc_text = str(df.iloc[idx, 1])
            doc_preview = doc_text[:50] + "..." if len(doc_text) > 50 else doc_text
        
        color_from = get_group_color(original_group)
        color_to = get_group_color(new_group)
        


        display_original_group = original_group
        display_new_group = new_group
        

        try:
            print(f"   DEBUG: current_group_order = {current_group_order}")
            if hasattr(globals(), 'current_group_order') and current_group_order:
                print(f"   DEBUG: Exclude group has keywords: {current_group_order.get('Exclude', [])}")

                if "Exclude" in current_group_order and current_group_order["Exclude"]:
                    if display_new_group == "Other":
                        display_new_group = "Exclude"
                        print(f"   DEBUG: Changed new_group from Other to Exclude")
                    if display_original_group == "Other":
                        display_original_group = "Exclude"
                        print(f"   DEBUG: Changed original_group from Other to Exclude")
        except Exception as e:
            print(f"   DEBUG: Error in display logic: {e}")
            pass

        if original_group != new_group and original_group != "Unknown":
            change_text = f"Doc {idx+1}: {display_original_group} -> {display_new_group}"
            change_color = "#27ae60"    
        elif original_group == "Unknown":
            change_text = f"Doc {idx+1}: -> {new_group} (moved to {new_group})"
            change_color = "#e67e22" 
        else:
            change_text = f"Doc {idx+1}: {original_group} (no change)"
            change_color = "#95a5a6"  
        
        history_items.append(
            html.Div([
                html.Div([
                    html.Span(f"Doc {idx+1}", style={  
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                        "marginRight": "10px"
                    }),
                    html.Span("->", style={"margin": "0 5px", "color": "#95a5a6"}),
                ], style={"marginBottom": "5px"}),
                html.Div([
                    html.Span(original_group, style={
                        "backgroundColor": color_from,
                        "color": "white",
                        "padding": "2px 8px",
                        "borderRadius": "4px",
                        "fontSize": "0.85rem",
                        "marginRight": "5px"
                    }),
                    html.Span("->", style={"margin": "0 5px", "color": "#95a5a6"}),
                    html.Span(new_group, style={
                        "backgroundColor": color_to,
                        "color": "white",
                        "padding": "2px 8px",
                        "borderRadius": "4px",
                        "fontSize": "0.85rem"
                    })
                ], style={"marginBottom": "8px"}),
                html.Div(change_text, style={
                    "fontSize": "0.8rem",
                    "color": change_color,
                    "fontWeight": "bold",
                    "marginBottom": "5px"
                }),
                html.Div(doc_preview, style={
                    "fontSize": "0.8rem",
                    "color": "#7f8c8d",
                    "fontStyle": "italic",
                    "paddingLeft": "10px",
                    "borderLeft": "2px solid #ecf0f1"
                })
            ], style={
                "padding": "10px",
                "marginBottom": "10px",
                "backgroundColor": "white",
                "borderRadius": "6px",
                "border": "1px solid #e9ecef"
            })
        )
    
    return [
        html.Div(f"Total adjustments: {len(temp_assignments)}", style={
            "fontWeight": "bold",
            "color": "#2c3e50",
            "marginBottom": "15px",
            "textAlign": "center",
            "fontSize": "1rem"
        }),
        html.Div(history_items, style={"overflowY": "auto"})
    ], history_buttons


@app.callback(
    [Output("finetune-train-btn", "children", allow_duplicate=True),
     Output("finetune-train-btn", "style", allow_duplicate=True),
     Output("finetune-training-status", "children", allow_duplicate=True),
     Output("finetune-selected-group", "data", allow_duplicate=True),
     Output("finetune-selected-keyword", "data", allow_duplicate=True), 
     Output("finetune-temp-assignments", "data", allow_duplicate=True),
     Output("finetune-highlight-core", "data", allow_duplicate=True),  
     Output("finetune-highlight-gray", "data", allow_duplicate=True),  
     Output("training-figures", "data", allow_duplicate=True)],  
    Input("finetune-train-btn", "n_clicks"),
    [State("finetune-temp-assignments", "data"),
     State("group-order", "data"),
     State("training-figures", "data"),
     State("finetune-selected-group", "data")],
    prevent_initial_call=True
)
def run_finetune_training(n_clicks, temp_assignments, group_order, current_training_figures, current_selected_group):
    if not n_clicks:
        raise PreventUpdate
    
    print(f"\n{'='*80}")
    print(f"FINETUNE TRAINING STARTED")
    print(f"{'='*80}")
    print(f"INPUT group_order:")
    for grp_name, kw_list in (group_order or {}).items():
        print(f"   {grp_name}: {kw_list}")
    print(f"INPUT temp_assignments: {temp_assignments}")
    print(f"INPUT current_selected_group: {current_selected_group}")
    print(f"{'='*80}\n")
 
    global training_in_progress
    if training_in_progress:
        running_style = {
            "backgroundColor": "#f39c12",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "not-allowed",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(243, 156, 18, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }
        return "Running...", running_style, "", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    training_style = {
        "backgroundColor": "#f39c12",
        "color": "white",
        "border": "none",
        "padding": "12px 24px",
        "borderRadius": "6px",
        "fontSize": "1rem",
        "fontWeight": "bold",
        "cursor": "not-allowed",
        "transition": "all 0.3s ease",
        "boxShadow": "0 3px 8px rgba(243, 156, 18, 0.3)",
        "width": "100%",
        "marginTop": "10px"
    }
    
    success_style = {
        "backgroundColor": "#27ae60",
        "color": "white",
        "border": "none",
        "padding": "12px 24px",
        "borderRadius": "6px",
        "fontSize": "1rem",
        "fontWeight": "bold",
        "cursor": "pointer",
        "transition": "all 0.3s ease",
        "boxShadow": "0 3px 8px rgba(39, 174, 96, 0.3)",
        "width": "100%",
        "marginTop": "10px"
    }
    
    try:
        print("=" * 60)
        print(" Starting Finetune Training (Triplet + Center Pull)")
        print("=" * 60)
        
      
        training_in_progress = True
        
        global df
        if 'df' not in globals():
            df = pd.read_csv(FILE_PATHS["csv_path"])
        
        user_has_exclude = "Exclude" in group_order and group_order["Exclude"]
        
        print(f"User has defined Exclude group: {user_has_exclude}")
        if user_has_exclude:
            print(f"  Exclude group keywords: {group_order['Exclude']}")
            print("  Using Case 1: Center Pull only")
        else:
            print("  Using Case 2: Center Pull + Exclude Push")
   
        print(f"Applying {len(temp_assignments or {})} user adjustments...")
        print(f"   DEBUG: Original group_order: {group_order}")
        adjusted_group_order = {}
        for grp_name, kw_list in group_order.items():
            adjusted_group_order[grp_name] = kw_list.copy()
        print(f"   DEBUG: adjusted_group_order: {adjusted_group_order}")
        
    


        matched_dict_adjusted = {}
        
        try:
            user_finetuned_path = FILE_PATHS["user_finetuned_list"]
            filtered_path = FILE_PATHS["filtered_group_assignment"]
            
            source_path = None
            if os.path.exists(user_finetuned_path):
                source_path = user_finetuned_path
                print(f"   Loading from user_finetuned_list.json")
            elif os.path.exists(filtered_path):
                source_path = filtered_path
                print(f"   Loading from filtered_group_assignment.json")
            else:
                print(f"   No filtered results found! Cannot run finetune without training first.")
                raise Exception("No filtered results found. Please run training first.")
            
            with open(source_path, "r", encoding="utf-8") as f:
                matched_dict_adjusted = json.load(f)
            
            print(f"   Loaded base group assignments:")
            for grp_name, indices in matched_dict_adjusted.items():
                print(f"      {grp_name}: {len(indices)} documents")
            
            for grp_name in group_order.keys():
                if grp_name not in matched_dict_adjusted:
                    matched_dict_adjusted[grp_name] = []
                    print(f"   Added missing group: {grp_name}")
                    
        except Exception as e:
            print(f"   Failed to load filtered results: {e}")
            raise
        

        if temp_assignments:
            for idx_str, target_group in temp_assignments.items():
                if idx_str.endswith("_original"):
                    continue
                    
                idx = int(idx_str)
              
                for grp_name in matched_dict_adjusted.keys():
                    if idx in matched_dict_adjusted[grp_name]:
                        matched_dict_adjusted[grp_name].remove(idx)
            
                if target_group == "Exclude" and user_has_exclude:
                    matched_dict_adjusted["Exclude"].append(idx)
                elif target_group in matched_dict_adjusted:
                    matched_dict_adjusted[target_group].append(idx)
        

        for grp_name, indices in matched_dict_adjusted.items():
            print(f"   {grp_name}: {len(indices)} samples")


        model_path = FILE_PATHS["triplet_trained_encoder"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"CUDA memory before loading: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
   
        try:
            print(f"  Loading model to CPU first...")
            encoder = torch.load(model_path, map_location="cpu", weights_only=False)
            
            if hasattr(encoder, 'eval'):
                print(f"  Moving model to {device}...")
                if device.type == 'cuda':
                    encoder = encoder.cuda()
                    torch.cuda.synchronize()
                    print(f"  Model moved to GPU successfully")
                else:
                    encoder = encoder.to(device)
                
                encoder.eval()
                print(f"  Model loaded and ready on {device}")
            else:
                raise ValueError("Loaded state_dict instead of model")
        except Exception as e:
            print(f"  Failed to load model object directly: {e}")
            print("  Falling back to loading state_dict...")
            state_dict = torch.load(model_path, map_location="cpu")
            if 'proj.weight' in state_dict:
                proj_weight_shape = state_dict['proj.weight'].shape
                proj_dim = proj_weight_shape[0]
                bert_hidden_size = proj_weight_shape[1]
            else:
                proj_dim = 256
                bert_hidden_size = 768
            
            if bert_hidden_size == 768:
                bert_name = "bert-base-uncased"
            elif bert_hidden_size == 384:
                bert_name = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                bert_name = get_config("bert_model", "sentence-transformers/all-MiniLM-L6-v2")
            
            has_ln = 'ln.weight' in state_dict

            encoder = SentenceEncoder(bert_name=bert_name, proj_dim=proj_dim, device="cpu")
            encoder = encoder.to_empty(device=device)
            encoder.load_state_dict(state_dict, strict=False)
            encoder.eval()
            encoder.bert_model_name = bert_name

       
        if hasattr(encoder, 'bert_model_name'):
            tokenizer_name = encoder.bert_model_name
        else:
            tokenizer_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        texts = df.iloc[:, 1].astype(str).tolist()

        def encode_all_docs():
            encoder.eval()
            Z = []
            batch_size = 16  
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    toks = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
                    vocab_size = tokenizer.vocab_size
                    if 'input_ids' in toks:
                        toks['input_ids'] = torch.clamp(toks['input_ids'], 0, vocab_size - 1)
                    toks = {k: v.to(device) for k, v in toks.items()}
                    with torch.no_grad():
                        Z.append(encoder.encode_tokens(toks).cpu())

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                return torch.vstack(Z)
            except Exception as cuda_error:
                print(f"CUDA error during encoding in finetune training, falling back to CPU: {cuda_error}")
                encoder_cpu = SentenceEncoder(bert_name, proj_dim, device="cpu")
                encoder_cpu = encoder_cpu.to("cpu")
                encoder_cpu.load_state_dict(state_dict, strict=False)
                encoder_cpu.eval()

                Z = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    toks = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
                    vocab_size = tokenizer.vocab_size
                    if 'input_ids' in toks:
                        toks['input_ids'] = torch.clamp(toks['input_ids'], 0, vocab_size - 1)
                    toks = {k: v.to("cpu") for k, v in toks.items()}
                    with torch.no_grad():
                        Z.append(encoder_cpu.encode_tokens(toks).cpu())
                return torch.vstack(Z)
        
        Z_before = encode_all_docs().numpy()
        
      
        group_prototypes = {}
        for grp_name, indices in matched_dict_adjusted.items():
            if len(indices) == 0:
                continue
            grp_embeds = Z_before[indices]
            prototype = grp_embeds.mean(axis=0)
            prototype = prototype / (np.linalg.norm(prototype) + 1e-12)
            group_prototypes[grp_name] = torch.tensor(prototype, device=device, dtype=torch.float32)
        
        print(f"  Computed {len(group_prototypes)} group prototypes")

        if user_has_exclude:
            lr = 5e-6  
            epochs = 10
            print(f"  Case 1: Using lr={lr}, epochs={epochs}")
        else:
            lr = 1e-5
            epochs = 10
            print(f"  Case 2: Using lr={lr}, epochs={epochs}")
 
        encoder.train()
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
        
        batch_size = 16
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
        
            train_samples = []
            exclude_samples = []
            
            for grp_name, indices in matched_dict_adjusted.items():
                if grp_name == "Exclude" and not user_has_exclude:
                    exclude_samples.extend(indices)
                else:
                    for idx in indices:
                        train_samples.append((idx, grp_name))
            
            print(f"  Epoch {epoch+1}/{epochs}:")
            print(f"    Triplet + Center Pull samples: {len(train_samples)}")
            print(f"    Exclude Push samples: {len(exclude_samples)}")
            
            if len(train_samples) == 0:
                print(f"    WARNING: No training samples found!")
                continue
            
            
            random.shuffle(train_samples)
            
            for i in range(0, len(train_samples), batch_size):
                batch = train_samples[i:i+batch_size]
                batch_texts = [texts[idx] for idx, _ in batch]
                batch_groups = [grp for _, grp in batch]
                
                try:
                    toks = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
                    vocab_size = tokenizer.vocab_size
                    if 'input_ids' in toks:
                        toks['input_ids'] = torch.clamp(toks['input_ids'], 0, vocab_size - 1)
                    toks = {k: v.to(device) for k, v in toks.items()}
                    embeds = encoder.encode_tokens(toks)
                except Exception as cuda_error:
                    print(f"CUDA error during finetune training, falling back to CPU: {cuda_error}")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"CUDA error during finetune training, falling back to CPU: {cuda_error}")
                    encoder = SentenceEncoder(bert_name, proj_dim, device="cpu")
                    encoder = encoder.to("cpu")
                    encoder.load_state_dict(state_dict, strict=False)
                    encoder.train()
                    device = torch.device("cpu")
                    toks = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
                    vocab_size = tokenizer.vocab_size
                    if 'input_ids' in toks:
                        toks['input_ids'] = torch.clamp(toks['input_ids'], 0, vocab_size - 1)
                    toks = {k: v.to("cpu") for k, v in toks.items()}
                    embeds = encoder.encode_tokens(toks)
                
                center_loss = torch.tensor(0.0, device=device, requires_grad=True)
                for j, grp in enumerate(batch_groups):
                    if grp in group_prototypes:
                        proto = group_prototypes[grp]
                        similarity = torch.cosine_similarity(embeds[j], proto, dim=0)
                        center_loss = center_loss + (1 - similarity)
                
                center_loss = center_loss / len(batch)
                
                triplet_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if len(batch_groups) >= 2:
                    group_embeddings = {}
                    for j, grp in enumerate(batch_groups):
                        if grp not in group_embeddings:
                            group_embeddings[grp] = []
                        group_embeddings[grp].append(embeds[j])
                    
                    for group_name, group_embeds in group_embeddings.items():
                        if len(group_embeds) < 2:
                            continue
                        
                        for anchor in group_embeds:
                            positive_candidates = [e for e in group_embeds if not torch.equal(e, anchor)]
                            if not positive_candidates:
                                continue
                            positive = positive_candidates[0]
                            
                            negative_candidates = []
                            for other_group, other_embeds in group_embeddings.items():
                                if other_group != group_name:
                                    negative_candidates.extend(other_embeds)
                            
                            if negative_candidates:
                                negative = negative_candidates[0]
                                
                                margin = 0.5
                                ap_dist = torch.norm(anchor - positive, p=2)
                                an_dist = torch.norm(anchor - negative, p=2)
                                triplet_loss = triplet_loss + torch.relu(ap_dist - an_dist + margin)
                
                total_batch_loss = center_loss + 0.5 * triplet_loss
                
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                n_batches += 1
            
            if not user_has_exclude and exclude_samples:
                print(f"    Training {len(exclude_samples)} exclude samples...")
                
                for exclude_idx in exclude_samples:
                    try:
                        exclude_text = texts[exclude_idx]
                        exclude_toks = tokenizer([exclude_text], return_tensors='pt', padding=True, truncation=True, max_length=256)
                        vocab_size = tokenizer.vocab_size
                        if 'input_ids' in exclude_toks:
                            exclude_toks['input_ids'] = torch.clamp(exclude_toks['input_ids'], 0, vocab_size - 1)
                        exclude_toks = {k: v.to(device) for k, v in exclude_toks.items()}
                        exclude_embed = encoder.encode_tokens(exclude_toks)
                        
                        max_similarity = 0
                        for grp_name, prototype in group_prototypes.items():
                            if grp_name != "Other":  
                                sim = torch.cosine_similarity(exclude_embed, prototype.unsqueeze(0))
                                if sim.item() > max_similarity:
                                    max_similarity = sim.item()
                        
                        tau = 0.35
                        exclude_loss = torch.max(torch.tensor(0.0, device=device), 
                                               torch.tensor(max_similarity - tau, device=device))
                        
                        if exclude_loss.item() > 0:
                            optimizer.zero_grad()
                            exclude_loss.backward()
                            optimizer.step()
                            total_loss += exclude_loss.item()
                            n_batches += 1
                        
                    except Exception as e:
                        print(f"    Exclude Push training failed for doc {exclude_idx}: {e}")
                        continue
                
            if user_has_exclude and epoch < epochs - 1: 
                print(f"    Updating prototypes with EMA...")
                alpha = 0.9  
                
                with torch.no_grad():
                    current_embeds = []
                    for grp_name, indices in matched_dict_adjusted.items():
                        if len(indices) == 0:
                            continue
                        grp_embeds = []
                        for idx in indices:
                            text = texts[idx]
                            toks = tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=256)
                            toks = {k: v.to(device) for k, v in toks.items()}
                            embed = encoder.encode_tokens(toks)
                            grp_embeds.append(embed.cpu())
                        
                        if grp_embeds:
                            current_prototype = torch.cat(grp_embeds, dim=0).mean(dim=0)
                            current_prototype = current_prototype / (torch.norm(current_prototype) + 1e-12)
                            
                            if grp_name in group_prototypes:
                                old_prototype = group_prototypes[grp_name]
                                new_prototype = alpha * old_prototype + (1 - alpha) * current_prototype.to(device)
                                new_prototype = new_prototype / (torch.norm(new_prototype) + 1e-12)
                                group_prototypes[grp_name] = new_prototype
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"    Loss: {avg_loss:.4f}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("     Finetune training completed!")

        torch.save(encoder.state_dict(), model_path)
        print(f"        Saved finetuned model to {model_path}")
        
  

 

        ordered_matched_dict = {}
        for grp_name in group_order.keys():
            if grp_name in matched_dict_adjusted:
                ordered_matched_dict[grp_name] = matched_dict_adjusted[grp_name]

        for grp_name, indices in matched_dict_adjusted.items():
            if grp_name not in ordered_matched_dict:
                ordered_matched_dict[grp_name] = indices
                
        
       
        for grp_name, indices in ordered_matched_dict.items():
            print(f"   {grp_name}: {len(indices)} samples")

        print(f"{'='*80}\n")
        
        user_finetuned_path = FILE_PATHS["user_finetuned_list"]
        with open(user_finetuned_path, "w", encoding="utf-8") as f:
            json.dump(ordered_matched_dict, f, ensure_ascii=False, indent=2)
        print(f"        Saved user finetuned results to {user_finetuned_path}")


        filtered_path = FILE_PATHS["filtered_group_assignment"]
        with open(filtered_path, "w", encoding="utf-8") as f:
            json.dump(ordered_matched_dict, f, ensure_ascii=False, indent=2)


        encoder.eval()
        with torch.no_grad():
            all_embeds = []
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                toks = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks = {k: v.to(device) for k, v in toks.items()}
                embeds = encoder.encode_tokens(toks)
                all_embeds.append(embeds.cpu())
            Z_after = torch.cat(all_embeds, dim=0).numpy()
  
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        projected_2d_after = tsne.fit_transform(Z_after)
     

        group_centers = {}
        for grp_name, indices in ordered_matched_dict.items():
            if indices and len(indices) > 0:
                valid_indices = [i for i in indices if i < len(projected_2d_after)]
                if valid_indices:
                    center = projected_2d_after[valid_indices].mean(axis=0)
                    group_centers[grp_name] = center
                    print(f"   {grp_name} : {center}")
 

        
        def create_plotly_figure(projected_2d, title, is_after=False, group_centers_param=None):
            fig = go.Figure()
     
            article_indices = list(range(len(projected_2d)))
            hover_texts = [f"Doc {idx+1}" for idx in article_indices] 
            custom_data = [[idx] for idx in article_indices]
            
            bg_style = PLOT_STYLES["background"]
            fig.add_trace(go.Scatter(
                x=projected_2d[:, 0],
                y=projected_2d[:, 1],
                mode='markers',
                name="All Documents",
                marker=dict(
                    color=bg_style["color"],
                    size=bg_style["size"],
                    opacity=bg_style["opacity"],
                    symbol="circle", 
                    line=dict(width=bg_style["line_width"], color=bg_style["line_color"])
                ),
                customdata=custom_data,
                hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                hovertext=hover_texts
            ))

            if is_after and group_centers_param:
                center_style = PLOT_STYLES["center"]
                for grp_name, center_2d in group_centers_param.items():
                    color = get_group_color(grp_name)
                    fig.add_trace(go.Scatter(
                        x=[center_2d[0]],
                        y=[center_2d[1]],
                        mode='markers+text',
                        name=f'Center: {grp_name}',
                        marker=dict(
                            color=color,
                            size=center_style["size"],
                            symbol=center_style["symbol"],
                            opacity=center_style["opacity"],
                            line=dict(width=center_style["line_width"], color=center_style["line_color"])
                        ),
                        text=[grp_name],
                        textposition="top center",
                        textfont=dict(size=12, color=color, family='Arial Black'),
                        hovertemplate=f'<b>Group Center: {grp_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                    ))

            layout_style = PLOT_STYLES["layout"]
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16)),
                xaxis_title="X",
                yaxis_title="Y",
                showlegend=True,
                hovermode='closest',
                plot_bgcolor=layout_style["plot_bgcolor"],
                paper_bgcolor=layout_style["paper_bgcolor"],
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=layout_style["xaxis"],
                yaxis=layout_style["yaxis"]
            )
            
            return fig

        fig_before_dict = current_training_figures.get("before") if current_training_figures else None
        fig_after = create_plotly_figure(projected_2d_after, "2D Projection After Training (Finetuned)", True, group_centers)
 
        fig_after_dict = {
            "data": [trace.to_plotly_json() for trace in fig_after.data],
            "layout": fig_after.layout.to_plotly_json()
        }
        
        updated_figures = {"before": fig_before_dict, "after": fig_after_dict}
        print(f"     2D visualization generated")

        if not current_selected_group or current_selected_group not in ordered_matched_dict:
            print(f"    No selected group, trying to auto-select from adjustments")

            if temp_assignments:

                target_groups = [new_grp for new_grp in temp_assignments.values()]
                most_common = Counter(target_groups).most_common(1)
                if most_common and most_common[0][0] != "Other":
                    current_selected_group = most_common[0][0]
                    print(f"   Auto-selected group: {current_selected_group} (most adjusted)")

            if not current_selected_group or current_selected_group not in ordered_matched_dict:
                for grp in ordered_matched_dict.keys():
                    if grp != "Other":
                        current_selected_group = grp
                        print(f"   Auto-selected group: {current_selected_group} (first non-Other)")
                        break
        
        print(f"        Training complete. compute_finetune_highlights will recalculate highlights for {current_selected_group}")
        
   
        training_in_progress = False
        
        return "Run Finetune Training", success_style, "", current_selected_group, None, temp_assignments, dash.no_update, dash.no_update, updated_figures
        
    except Exception as e:
        print(f"        Finetune training failed: {e}")

        traceback.print_exc()
        
  
        training_in_progress = False
        
        error_style = {
            "backgroundColor": "#e74c3c",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(231, 76, 60, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }
        
        return "Run Finetune Training", error_style, "", dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update



@app.callback(
    [Output("finetune-train-btn", "children", allow_duplicate=True),
     Output("finetune-train-btn", "style", allow_duplicate=True),
     Output("finetune-training-status", "children", allow_duplicate=True)],
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True
)
def update_training_status(n_intervals):
    global training_in_progress
    
    if training_in_progress:
        running_style = {
            "backgroundColor": "#f39c12",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "not-allowed",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(243, 156, 18, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }
        return "Running...", running_style, ""
    else:
        normal_style = {
            "backgroundColor": "#9b59b6",
            "color": "white",
            "border": "none",
            "padding": "12px 24px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 3px 8px rgba(155, 89, 182, 0.3)",
            "width": "100%",
            "marginTop": "10px"
        }
        return "Run Finetune Training", normal_style, ""



@app.callback(
    Output("train-btn", "style", allow_duplicate=True),
    Input("display-mode", "data"),
    prevent_initial_call=True
)
def control_train_button_visibility(display_mode):
    if display_mode == "keywords":
        return {
            "backgroundColor": "#e74c3c",
            "color": "white",
            "border": "none",
            "padding": "10px 20px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "minWidth": "120px",
            "flexShrink": "0"
        }
    else:
        return {"display": "none"}



@app.callback(
    Output("switch-finetune-btn", "style", allow_duplicate=True),
    Input("display-mode", "data"),
    prevent_initial_call=True
)
def control_finetune_button_visibility(display_mode):
    if display_mode == "training":
        return {
            "backgroundColor": "#8e44ad",
            "color": "white",
            "border": "none",
            "padding": "10px 20px",
            "borderRadius": "6px",
            "fontSize": "1rem",
            "fontWeight": "bold",
            "cursor": "pointer",
            "transition": "all 0.3s ease",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
            "minWidth": "180px",
            "flexShrink": "0"
        }
    else:
        return {"display": "none"}





@app.callback(
    [Output("group-order", "data", allow_duplicate=True),
     Output("new-keyword-input", "value", allow_duplicate=True),
     Output("debug-output", "children", allow_duplicate=True)],
    Input("add-keyword-btn", "n_clicks"),
    [State("new-keyword-input", "value"),
     State("group-order", "data"),
     State("selected-group", "data")],
    prevent_initial_call=True
)
def add_custom_keyword(n_clicks, keyword_value, group_order, selected_group):
    if not n_clicks or not keyword_value or not selected_group:
        raise PreventUpdate
    
    if not group_order:
        group_order = {}
    
    if selected_group not in group_order:
        group_order[selected_group] = []
    

    keywords = [kw.strip() for kw in keyword_value.split(',') if kw.strip()]
    
    added_keywords = []
    already_exists = []
    
    for keyword in keywords:
        if keyword not in group_order[selected_group]:
            group_order[selected_group].append(keyword)
            added_keywords.append(keyword)
            print(f"Added keyword '{keyword}' to group '{selected_group}'")
        else:
            already_exists.append(keyword)
    
   
    if added_keywords and already_exists:
        message = f"Added: {', '.join(added_keywords)} | Already exists: {', '.join(already_exists)}"
    elif added_keywords:
        message = f"Added {len(added_keywords)} keywords: {', '.join(added_keywords)}"
    else:
        message = f"All keywords already exist: {', '.join(already_exists)}"
        return dash.no_update, "", message
    
    return group_order, "", message


if __name__ == "__main__":

    os.environ['FLASK_ENV'] = 'development'
    
    
    try:
      
        app.run(
            debug=True,
            port=8053,
            host='127.0.0.1',
            use_reloader=False,
            threaded=True
        )
    except OSError as e:

        try:
            app.run(
                debug=True, 
                port=8054, 
                host='127.0.0.1', 
                use_reloader=False,
                threaded=True
            )
        except OSError as e2:
            

            
            def find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', 0))
                    s.listen(1)
                    port = s.getsockname()[1]
                return port
            
            try:
                free_port = find_free_port()
                print(f"    DEBUG: Found free port: {free_port}")
                print(f"    DEBUG: Starting on port {free_port}...")
                app.run(
                    debug=True, 
                    port=free_port, 
                    host='127.0.0.1', 
                    use_reloader=False,
                    threaded=True
                )
            except Exception as e3:
                print(" Failed")
