# Import libraries and dependencies for NLP, visualization, modeling, and UI
import re 
import json
import os
import random
import io
import base64
import math
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
import matplotlib
matplotlib.use("Agg") 
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import pad
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from rapidfuzz import fuzz
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import itertools
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
import gc
import multiprocessing
from collections import Counter
import nltk

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer



TRAINING_CONFIG = {

    "bert_name": "bert-base-uncased",
    "proj_dim": 256,
    "freeze_layers": 6,
    

    "triplet_epochs": 1,        
    "triplet_batch_size": 16,
    "triplet_margin": 0.8,
    "triplet_lr": 2e-5,
    

    "proto_epochs": 5,          
    "proto_batch_size": 64,
    "proto_lr": 1e-5,
    "min_separation": 1.0,
    

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
    """[EMOJI]"""
    return TRAINING_CONFIG.get(key, default)


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
num_epochs = 5
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
    """Get color for a specific group"""
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


img_output_dir = "../Keyword_Group/Test"
csv_path = "C:/Users/Super/Box/Yan/KeySI/CSV/20news_top30_per_class.csv"  
final_list_path = "../Keyword_Group/Jupyter/final_list.json"
save_path = "../Keyword_Group/Jupyter/final_list.json"
output_dir = "../Keyword_Group/covidtest"
model_save_path = "../Keyword_Group/bert_finetuned.pth"
output_file = os.path.join(output_dir, "group_results_.csv")
group_dict_path = "../Keyword_Group/BBC/B_indices_output5_T8.json"
output_path = "../Keyword_Group/Jupyter/group_indices_output.json"


def ensure_directories(): 
    directories = [
        img_output_dir,
        os.path.dirname(final_list_path),
        output_dir,
        os.path.dirname(model_save_path),
        os.path.dirname(group_dict_path),
        os.path.dirname(output_path)
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

def precompute_document_embeddings():
   
    global _GLOBAL_DOCUMENT_EMBEDDINGS, _GLOBAL_DOCUMENT_TSNE, _GLOBAL_DOCUMENT_EMBEDDINGS_READY, df
    
    if _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        print("    Document embeddings already pre-computed, skipping...")
        return
    
    print("    Pre-computing document embeddings for faster response...")
    
    try:
  
        if 'df' not in globals():
            df = pd.read_csv(csv_path)
        
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
        
        # Clear GPU cache if using CUDA
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"    Error pre-computing document embeddings: {e}")
        _GLOBAL_DOCUMENT_EMBEDDINGS_READY = False

def get_document_embeddings():
    """Get pre-computed document embeddings"""
    global _GLOBAL_DOCUMENT_EMBEDDINGS, _GLOBAL_DOCUMENT_EMBEDDINGS_READY
    
    if not _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        precompute_document_embeddings()
    
    return _GLOBAL_DOCUMENT_EMBEDDINGS

def get_document_tsne():
    """Get pre-computed document t-SNE results"""
    global _GLOBAL_DOCUMENT_TSNE, _GLOBAL_DOCUMENT_EMBEDDINGS_READY
    
    if not _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
        precompute_document_embeddings()
    
    return _GLOBAL_DOCUMENT_TSNE

def truncate_text_for_model(text, max_length=500):
    """Truncate text to fit within model's maximum sequence length"""
    if not text or len(text) <= max_length:
        return text
    
    # Simple truncation: take the first max_length characters
    # This ensures we don't exceed the model's token limit
    truncated = text[:max_length]
    
    # Try to truncate at a word boundary if possible
    if ' ' in truncated:
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Only truncate at word if it's not too early
            truncated = truncated[:last_space]
    
    return truncated + "..." if len(truncated) < len(text) else truncated

# Pre-compute embeddings when script starts
print("    Initializing document embeddings...")
try:
    precompute_document_embeddings()
    print("         Document embeddings initialization completed successfully!")
    print(f"          Performance improvement: Keyword clicks should now be 3-5x faster")
    print(f"            Memory usage: ~{_GLOBAL_DOCUMENT_EMBEDDINGS.nbytes / 1024 / 1024:.1f} MB for embeddings")
    print_performance_tips()
except Exception as e:
    print(f"    Warning: Could not pre-compute embeddings: {e}")
    print("    Will compute on-demand (slower response)")
    print("           Performance will be slower without pre-computed embeddings")


def safe_encode_batch(batch_texts, model, device, fallback_dim=768):
    """Safely encode a batch of texts with error handling and fallback"""
    try:
        # Check text lengths before encoding
        for i, text in enumerate(batch_texts):
            if len(text) > 1000:  # Additional safety check
                print(f"    Warning: Text {i} is very long ({len(text)} chars), truncating further")
                batch_texts[i] = truncate_text_for_model(text, max_length=400)
        
        # Encode with progress indication
        print(f"    Encoding batch of {len(batch_texts)} texts...")
        embeddings = model.encode(batch_texts, convert_to_tensor=True).to(device).cpu().numpy()
        print(f"    Successfully encoded batch, shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        print(f"    Error encoding batch: {e}")
        print(f"    Using fallback zero vectors for batch")
        
        # Return zero vectors as fallback
        batch_size = len(batch_texts)
        fallback_embeddings = np.zeros((batch_size, fallback_dim))
        return fallback_embeddings

kw_model = KeyBERT(model=embedding_model_kw)

ps = PorterStemmer()
word_count = Counter()
original_form = {}
df = pd.read_csv(csv_path)
all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
labels = df.iloc[:, 0].values

# Extract and count keywords using KeyBERT and NLTK - GPU optimized version
def preprocess_articles_batch(articles):
    """Batch preprocess articles, extract nouns"""
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
            batch_size = 128  # RTX 5090 large memory can handle larger batch
          
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
            
            # Clear GPU memory
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  Batch processing failed: {e}")
            results.extend([None] * len(batch))
    
    return results

# Batch preprocessing
processed_articles, valid_indices = preprocess_articles_batch(all_articles_text)

# GPU batch keyword extraction
if processed_articles:
    batch_results = extract_keywords_batch_gpu(processed_articles, batch_size=128)
    
    # Rebuild complete result list
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
GLOBAL_KEYWORDS = keywords  # Ensure keywords are available in global scope

# Ensure keyword variables are defined
if 'keywords' not in locals():
    keywords = []
    print("Warning: Keyword list is empty, using default values")

# Build Dash layout with UI components like inputs, buttons, and containers
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Add custom CSS for animations
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
    """Create Dash layout - keyword 2D dimensionality reduction visualization version"""
    return html.Div([
        # Header with modern design
        html.Div([
            html.H1("KeySI System", style={
                "textAlign": "center",
                "color": "#2c3e50",
                "fontSize": "2.5rem",
                "fontWeight": "bold",
                "marginBottom": "10px",
                "textShadow": "2px 2px 4px rgba(0,0,0,0.1)"
            }),
            html.P("Intelligent Keyword Analysis & Document Clustering", style={
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
        
        # Top control area with modern styling - side by side layout
        html.Div([
            # Left: Number of Groups
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
                "marginRight": "2%"
            }),
            
            # Right: Add Custom Keyword
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
                    placeholder='Enter a keyword to add...',
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

        # Data storage
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
        dcc.Store(id="finetune-selected-keyword", data=None),  # [EMOJI] selected_keyword store
        dcc.Store(id="finetune-selected-article-index", data=None),  # [EMOJI]
        dcc.Store(id="finetune-highlight-core", data=[]),
        dcc.Store(id="finetune-highlight-gray", data=[]),
        dcc.Store(id="finetune-temp-assignments", data={}),
        
        # Main content area - left-right column layout (dynamic based on display mode)
        html.Div(id="main-visualization-area", children=[
            # Left: keyword 2D visualization with text labels
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
            
            # Right: documents 2D visualization
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
        
        # Group management area (below the 2D visualizations) - three column layout
        html.Div([
            # Left: Group selection and keywords
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
            
            # Middle: Recommended Articles
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
            
            # Right: Article Full Text Display
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
        ], id="keywords-group-management-area", style={'display': 'flex', 'marginBottom': '30px'}),
        
        # Training mode specific Group Management and Recommended Articles (initially hidden)
        html.Div(id="training-group-management-area", style={'display': 'none', 'marginBottom': '30px'}, children=[
            # Left: Training Group Management
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
            
            # Middle: Training Recommended Articles
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
            
            # Right: Training Article Full Text Display
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
        
        # Finetune mode specific Group Management and Operations (initially hidden)
        html.Div(id="finetune-group-management-area", style={'display': 'none', 'marginBottom': '30px'}, children=[
            # Left: Finetune Group Management
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
            
            # Middle: Finetune Document List ([EMOJI] Training Recommended Articles)
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
            
            # Right: Sample Operations + Adjustment History
            html.Div([
                # Sample Operations Section
                html.H4("Sample Operations", style={
                    "color": "#2c3e50",
                    "fontSize": "1.2rem",
                    "fontWeight": "bold",
                    "marginBottom": "10px",
                    "textAlign": "center"
                }),
                # [EMOJI]
                html.Div(id="finetune-operation-buttons", children=[], style={"marginBottom": "20px"}),
                
                # Selected Document Preview with Title
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
                
                # Adjustment History Section
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
        
        # Training button and output
        html.Div([
            html.Button("Train Model", id="train-btn", n_clicks=0, style={
                "margin": "30px auto 15px auto",
                "padding": "15px 40px",
                "fontSize": "1.2rem",
                "fontWeight": "bold",
                "backgroundColor": "#e74c3c",
                "color": "white",
                "border": "none",
                "borderRadius": "8px",
                "cursor": "pointer",
                "transition": "all 0.3s ease",
                "boxShadow": "0 4px 15px rgba(231, 76, 60, 0.3)",
                "display": "block"
            }),
            html.Button("Switch to Training View", id="switch-view-btn", n_clicks=0, style={
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
                "display": "none"  # Initially hidden, shown after training
            }),
            html.Button("Switch to Finetune Mode", id="switch-finetune-btn", n_clicks=0, style={
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
            })
        ], style={"textAlign": "center"}),
        

        

        
        # Debug output area
        html.Div(id="debug-output", style={"marginTop": "20px"})
    ])

# Set application layout
app.layout = create_layout()

# Add necessary callback functions
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
        
        # [EMOJI]
        groups = {f"Group {i+1}": [] for i in range(num_groups)}
        
        # [EMOJI]"Other"[EMOJI]
        groups["Other"] = []
        print(f"        Added 'Other' group for exclusion (total groups: {num_groups + 1})")
        
        return groups
    
    elif triggered_id == "group-data":
        # Fix: sync group-data and group-order, ensure all group keywords are correctly saved
        # First clear all group keyword lists
        for group_name in new_order:
            new_order[group_name] = []
        
        # Then refill based on group_data
        for kw, grp in group_data.items():
            if grp and grp in new_order:
                new_order[grp].append(kw)
        return new_order

    return new_order

@app.callback(
    Output("group-containers", "children"),
    [Input("group-order", "data"),
     Input("selected-group", "data"),
     Input("selected-keyword", "data")],  # Add selected-keyword as Input to trigger re-render
    [State("display-mode", "data")]
)
def render_groups(group_order, selected_group, selected_keyword, display_mode):
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: render_groups CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
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
    
    # In training mode, avoid unnecessary updates that might trigger documents-2d-plot
    if display_mode == "training" and selected_keyword is not None:
        print(f"    DEBUG:     TRAINING MODE WARNING:")
        print(f"    DEBUG:   selected_keyword is {selected_keyword}")
        print(f"    DEBUG:   This might cause conflicts with documents-2d-plot")
        print(f"    DEBUG:   But we're being careful about updates")
        # Still render groups but be more careful about keyword selection highlighting
    
    if not group_order:
        print(f"    DEBUG:         No group_order, returning empty list")
        return []
    
    print(f"    DEBUG:      Proceeding with group rendering...")

    children = []
    for grp_name, kw_list in group_order.items():
        # Group header with number and color
        if grp_name == "Other":
            group_display_name = "Other (Exclude)"
            group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Group {group_number}"
            group_color = get_group_color(grp_name)
        
        # Special styling for Other group
        if grp_name == "Other":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  # Dashed border for exclusion
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  # Slightly transparent
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

        # Keywords list
        group_keywords = []
        for i, kw in enumerate(kw_list):
            # Check if this keyword is selected for Group Management highlighting
            # In training mode, avoid keyword highlighting to prevent conflicts
            if display_mode == "training":
                is_selected = False  # No keyword highlighting in training mode
            else:
                is_selected = selected_keyword and kw == selected_keyword
            
            # Use group color for keywords in this group with selection highlighting
            keyword_button = html.Button(
                kw,
                id={"type": "select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  # Highlight when selected
                    "color": "white" if is_selected else group_color,  # White text when selected
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  # Bold when selected
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

print("    DEBUG: ==========================================")
print("    DEBUG: REGISTERING CALLBACK: select_group")
print("    DEBUG: ==========================================")
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
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: select_group CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
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

    print(f"    DEBUG:      Context triggered, analyzing trigger...")
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    triggered_prop_id = ctx.triggered[0].get('prop_id', 'N/A')
    triggered_value = ctx.triggered[0].get('value', 'N/A')

    print(f"    DEBUG:     TRIGGER ANALYSIS:")
    print(f"    DEBUG:   triggered_id: {triggered_id}")
    print(f"    DEBUG:   triggered_n_clicks: {triggered_n_clicks}")
    print(f"    DEBUG:   triggered_prop_id: {triggered_prop_id}")
    print(f"    DEBUG:   triggered_value: {triggered_value}")
    print(f"    DEBUG:   triggered_id type: {type(triggered_id)}")
    print(f"    DEBUG:   triggered_n_clicks type: {type(triggered_n_clicks)}")
    
    # Check if this is a group header click (only if n_clicks > 0)
    print(f"    DEBUG:     VALIDATION CHECKS:")
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
            
            # Clear selected keyword when switching groups
            # This ensures group selection shows all documents containing group keywords
            # Also add a flag to prevent keyword selection from overriding
            
            # In training mode, don't update selected-keyword to avoid triggering documents-2d-plot
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
            import traceback
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
    """Handle keyword selection from group management"""
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
    
    # Check if this is a keyword selection (even if n_clicks is None initially)
    if "select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            print(f"    DEBUG: Select keyword from group management: {keyword}")
            
            # Check if this is triggered by group selection (not a direct keyword click)
            if triggered_n_clicks is None:
                print(f"    DEBUG: Keyword selection triggered by group change, ignoring")
                raise PreventUpdate
            
            # Check if this is a direct keyword click (n_clicks > 0)
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                print(f"    DEBUG: Direct keyword click detected, selecting keyword: {keyword}")
                
                # In training mode, update keyword-highlights instead of selected-keyword
                if display_mode == "training":
                    print(f"    DEBUG: Training mode: updating keyword-highlights for keyword: {keyword}")
                    # Find documents that contain this keyword
                    keyword_docs = []
                    
                    # Load the dataframe to search for documents containing the keyword
                    try:
                        global df
                        if 'df' not in globals():
                            df = pd.read_csv(csv_path)
                        
                        # Search for documents containing the keyword
                        for i in range(len(df)):
                            text = str(df.iloc[i, 1]).lower()
                            if keyword.lower() in text:
                                keyword_docs.append(i)
                        
                        print(f"    DEBUG: Found {len(keyword_docs)} documents containing keyword '{keyword}': {keyword_docs}")
                        # In training mode, NEVER update selected-keyword to avoid triggering documents-2d-plot
                        return dash.no_update, keyword_docs
                    except Exception as e:
                        print(f"    DEBUG: Error finding documents for keyword '{keyword}': {e}")
                        return dash.no_update, []
                else:
                    # In keywords mode, update selected-keyword normally
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
    """Remove keyword from group when delete button is clicked"""
    print(f"remove_keyword_from_group called")
    print(f"n_clicks: {n_clicks}")
    
    ctx = dash.callback_context
    print(f"ctx.triggered: {ctx.triggered}")
    
    if not ctx.triggered or not any(n_clicks):
        print("No delete button clicked")
        raise PreventUpdate
    
    # Find which delete button was clicked
    triggered_id = ctx.triggered[0]['prop_id']
    if not triggered_id or '.n_clicks' not in triggered_id:
        print("Invalid trigger")
        raise PreventUpdate
    
    try:
        # Parse the button ID
        button_id = json.loads(triggered_id.split('.')[0])
        group_name = button_id.get("group")
        keyword_index = button_id.get("index")
        
        print(f"Delete button clicked - Group: {group_name}, Index: {keyword_index}")
        
        if not group_name or keyword_index is None:
            print("Missing group name or index")
            raise PreventUpdate
        
        # Update group_order by removing the keyword at the specified index
        new_group_order = dict(group_order) if group_order else {}
        
        if group_name in new_group_order:
            keyword_list = list(new_group_order[group_name])
            if 0 <= keyword_index < len(keyword_list):
                removed_keyword = keyword_list.pop(keyword_index)
                new_group_order[group_name] = keyword_list
                
                # Also update group_data to remove the keyword assignment
                new_group_data = dict(group_data) if group_data else {}
                if removed_keyword in new_group_data:
                    new_group_data[removed_keyword] = None
                
                print(f"Removed keyword '{removed_keyword}' from group '{group_name}'")
                print(f"Updated group_data: {removed_keyword} = None")
                
                # Clear caches when groups change
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
     Input("selected-group", "data")],  # Also update when group changes
    [State("group-order", "data"),  # Add group_order as State parameter
     State("display-mode", "data")],
    prevent_initial_call=True
)
def display_recommended_articles(selected_keyword, selected_group, group_order, display_mode):
    """Display recommended articles based on selected keyword or group"""
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: display_recommended_articles CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   selected_keyword: {selected_keyword}")
    print(f"    DEBUG:   selected_group: {selected_group}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:     PARAMETER TYPES:")
    print(f"    DEBUG:   selected_keyword type: {type(selected_keyword)}")
    print(f"    DEBUG:   selected_group type: {type(selected_group)}")
    print(f"    DEBUG:   display_mode type: {type(display_mode)}")
    
    # In training mode, avoid unnecessary updates that might trigger documents-2d-plot
    if display_mode == "training" and selected_keyword is not None:
        print(f"    DEBUG:     TRAINING MODE WARNING:")
        print(f"    DEBUG:   selected_keyword is {selected_keyword}")
        print(f"    DEBUG:   This might cause conflicts with documents-2d-plot")
        print(f"    DEBUG:   But we're being careful about updates")
        # Still display articles but be more careful about keyword selection
    
    try:
        global df, _ARTICLES_CACHE
        if 'df' not in globals():
            print("Data not loaded")
            return html.P("Data not loaded")
        
        # Create cache key based on search criteria
        cache_key = None
        if selected_keyword:
            cache_key = f"keyword:{selected_keyword}"
        elif selected_group and group_order:
            # For groups, create cache key based on group keywords
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    # Sort keywords for consistent cache key
                    cache_key = f"group:{group_name}:{':'.join(sorted(keywords))}"
                    break
        
        # Check cache first
        if cache_key and cache_key in _ARTICLES_CACHE:
            print(f"Using cached articles for: {cache_key}")
            return _ARTICLES_CACHE[cache_key]
        
        # Determine search criteria
        search_keywords = []
        search_title = ""
        
        if selected_keyword:
            # Priority: specific keyword search
            search_keywords = [selected_keyword]
            search_title = f"Articles containing '{selected_keyword}'"
            print(f"Searching for articles containing keyword: {selected_keyword}")
        elif selected_group:
            # Secondary: group keyword search
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
        
        # Search for articles using semantic search if embeddings are available, otherwise use text search
        matching_articles = []
        
        if _GLOBAL_DOCUMENT_EMBEDDINGS_READY and len(search_keywords) > 0:
            try:
                print(f"    Using semantic search for keywords: {search_keywords}")
                
                # Get keyword embeddings
                keyword_texts = " ".join(search_keywords)  # Combine keywords for semantic search
                keyword_embedding = embedding_model_kw.encode([keyword_texts], convert_to_tensor=True).to(device).cpu().numpy()
                
                # Get pre-computed document embeddings
                document_embeddings = get_document_embeddings()
                
                # Calculate similarities
                similarities = cosine_similarity(keyword_embedding, document_embeddings)[0]
                
                # Get top similar documents
                top_indices = np.argsort(similarities)[::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.15:  # Similarity threshold
                        text = str(df.iloc[int(idx), 1]) if len(df.iloc[int(idx)]) > 1 else ""
                        file_keywords = extract_top_keywords(text, 5)
                        matching_articles.append({
                            'file_number': int(idx) + 1,
                            'file_index': int(idx),
                            'text': text,
                            'keywords': file_keywords,
                            'similarity': float(similarities[idx])
                        })
                        
                        if len(matching_articles) >= 50:  # Limit results
                            break
                
                print(f"    Semantic search found {len(matching_articles)} relevant documents")
                
            except Exception as e:
                print(f"    Semantic search failed, falling back to text search: {e}")
                # Fallback to text search
                for idx, row in df.iterrows():
                    text = str(row.iloc[1]) if len(row) > 1 else ""
                    text_lower = text.lower()
                    
                    # Check if any of the search keywords is in the text
                    contains_keyword = any(keyword.lower() in text_lower for keyword in search_keywords)
                    
                    if contains_keyword:
                        file_keywords = extract_top_keywords(text, 5)
                        matching_articles.append({
                            'file_number': idx + 1,
                            'file_index': idx,
                            'text': text,
                            'keywords': file_keywords
                        })
        else:
            # Use traditional text search
            print(f"    Using text search for keywords: {search_keywords}")
        for idx, row in df.iterrows():
            text = str(row.iloc[1]) if len(row) > 1 else ""
            text_lower = text.lower()
            
            # Check if any of the search keywords is in the text
            contains_keyword = any(keyword.lower() in text_lower for keyword in search_keywords)
            
            if contains_keyword:
                file_keywords = extract_top_keywords(text, 5)
                matching_articles.append({
                    'file_number': idx + 1,
                    'file_index': idx,
                    'text': text,
                    'keywords': file_keywords
                })
        
        if not matching_articles:
            result = html.P(f"No articles found for the selected search criteria")
            if cache_key:
                _ARTICLES_CACHE[cache_key] = result
                print(f"Cached 'no articles' result for: {cache_key}")
            return result
        
        # Create article display items
        article_items = [
            html.H6(f"{search_title} (Found {len(matching_articles)} articles)", 
                   style={"color": "#2c3e50", "marginBottom": "15px"})
        ]
        
        # Create article items with file number and keywords
        for article_info in matching_articles:
            # Create keyword tags
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
            
            # Create clickable article item with file number and keywords
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
        
        # Cache the result for future use
        result = html.Div(article_items)
        if cache_key:
            _ARTICLES_CACHE[cache_key] = result
            print(f"Cached articles result for: {cache_key}")
        
        return result
        
    except Exception as e:
        print(f"Error displaying recommended articles: {e}")
        return html.P(f"Error displaying recommended articles: {str(e)}")

def extract_top_keywords(text, top_k=5):
    """Extract top N single-word keywords from text"""
    try:
        global kw_model
        if 'kw_model' in globals() and kw_model:
            # Use KeyBERT to extract keywords - only single words (1, 1)
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), 
                                               stop_words='english')
            # Only return text part of top_k keywords
            return [kw[0] for kw in keywords[:top_k]]
        else:
            # If KeyBERT not available, return simple word splitting
            words = text.lower().split()
            # Filter out common stop words and short words
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
    print(f"Switch to group: {selected_group}")  # Add debug info
    print(f"        Clear selected keyword") 
    
    return selected_group, None  # Clear selected keyword when switching groups

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
# Run training process: match articles by keyword groups, remove outliers, and prepare for fine-tuning
def run_training():
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: STARTING TRAINING PROCESS")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: run_training() called")
    try:
        # Clear caches when training starts as data relationships may change
        clear_caches()
        
        # [EMOJI]combinedloss_onlytriplet.py[EMOJI]
        print(f"    DEBUG: Loading CSV data from: {csv_path}")
        df = pd.read_csv(csv_path)
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
        import traceback
        traceback.print_exc()
        return None, None

    out_dir = "test_results"; os.makedirs(out_dir, exist_ok=True)

    # BM25[EMOJI]
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
            print(f"    BM25[EMOJI] '{g}': [EMOJI] {words} -> [EMOJI] {q}")
            scores = bm25.get_scores(q)
            print(f"    [EMOJI]: min={min(scores):.4f}, max={max(scores):.4f}, [EMOJI]: {sum(1 for s in scores if s > 0)}")
            
            # [EMOJI]
            idx_corpus = [i for i, s in enumerate(scores) if s > 0.1]  # [EMOJI]0[EMOJI]0.1
            if len(idx_corpus) == 0:
                # [EMOJI]
                idx_corpus = [i for i, s in enumerate(scores) if s > 0.01]
            
            idx_orig = [valid_indices[i] for i in idx_corpus]
            results[g] = idx_orig[:3000]  # top_similar_files
            print(f"    [EMOJI] '{g}' [EMOJI] {len(idx_orig)} [EMOJI]")
        return results

    tokenized_corpus, valid_indices = process_articles_serial(all_texts)
    bm25 = BM25Okapi([s.split() for s in tokenized_corpus])

    # [EMOJI]
    USER_GROUPS_ONLY = True
    ALLOW_EMPTY_GROUPS = False
    
    # [EMOJI]
    query_groups = {}
    print(f"    DEBUG: Checking for user groups at: {final_list_path}")
    if os.path.exists(final_list_path):
        try:
            with open(final_list_path, "r", encoding="utf-8") as f:
                user_groups = json.load(f)
            print(f"    DEBUG: Loaded user groups: {user_groups}")
            # [EMOJI]Other[EMOJI]BM25[EMOJI]
            for group_name, keywords in user_groups.items():
                if keywords:  # [EMOJI]Other[EMOJI]
                    query_groups[group_name] = keywords
                    print(f"     {group_name}: [EMOJI] {keywords}")
            print(f"     Loaded user groups for training: {list(query_groups.keys())}")
        except Exception as e:
            print(f"    ERROR: Failed to load user groups: {e}")
            import traceback
            traceback.print_exc()
            query_groups = {}
    else:
        print(f"    WARNING: User groups file not found: {final_list_path}")
        query_groups = {}
    
    # [EMOJI]
    if not query_groups and not ALLOW_EMPTY_GROUPS:
        print("    ERROR: No user groups found and ALLOW_EMPTY_GROUPS is False")
        print("        [EMOJI]UI[EMOJI]")
        return None, None
    
    print(f"    DEBUG: Starting BM25 search with {len(query_groups)} groups")
    try:
        matched_dict = bm25_search_batch(bm25, query_groups, valid_indices)
        print(f"    DEBUG: BM25 search completed successfully")
    except Exception as e:
        print(f"    ERROR: BM25 search failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    for g, idxs in matched_dict.items():
        print(f"[BM25] {g}: {len(idxs)} docs")
    
    # [EMOJI]Other[EMOJI]BM25[EMOJI]
    if "Other" not in matched_dict:
        matched_dict["Other"] = []
        print(f"[BM25] Other: {len(matched_dict['Other'])} docs ([EMOJI])")
    else:
        print(f"[BM25] Other: {len(matched_dict['Other'])} docs ([EMOJI])")
    
    with open(os.path.join(out_dir, "bm25_search_results.json"), "w", encoding="utf-8") as f:
        json.dump(matched_dict, f, ensure_ascii=False, indent=2)

    # [EMOJI]
    print(f"    DEBUG: Initializing tokenizer with model: {get_config('bert_name')}")
    try:
        tokenizer = BertTokenizer.from_pretrained(get_config("bert_name"))
        print(f"    DEBUG: Tokenizer loaded successfully")
    except Exception as e:
        print(f"    ERROR: Failed to load tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # [EMOJI]SentenceEncoder[EMOJI]
    class SentenceEncoder(nn.Module):
        # BERT -> mean-pool -> Linear(proj) -> LayerNorm -> L2
        def __init__(self, bert_name=None, proj_dim=None, device='cpu'):
            if bert_name is None:
                bert_name = get_config("bert_name")
            if proj_dim is None:
                proj_dim = get_config("proj_dim")
            super().__init__()
            # [EMOJI] CPU [EMOJI] meta tensor [EMOJI]
            self.bert = BertModel.from_pretrained(bert_name)
            self.hidden = self.bert.config.hidden_size
            self.proj = nn.Linear(self.hidden, proj_dim) if proj_dim else None
            self.out_dim = proj_dim or self.hidden
            self.ln = nn.LayerNorm(self.out_dim)
            # [EMOJI]
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
    
    print(f"    DEBUG: Initializing encoder on device: {device}")
    try:
        encoder = SentenceEncoder(device=device)
        print(f"    DEBUG: Encoder initialized successfully")
    except Exception as e:
        print(f"    ERROR: Failed to initialize encoder: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # [EMOJI]Triplet[EMOJI]
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
            # [EMOJI]Other[EMOJI]
            neg_pool = []
            for g, g_idxs in matched_dict.items():
                if g != group_name:  # [EMOJI]Other
                    neg_pool.extend(g_idxs)
                    print(f"[DEBUG] [EMOJI] {g}: {len(g_idxs)} docs")
            
            print(f"[DEBUG] {group_name} [EMOJI]: {len(neg_pool)} docs")
            if len(neg_pool) == 0:  # [EMOJI]
                print(f"[WARN] {group_name} [EMOJI]")
                continue
            
            for a in idxs:
                pos = [x for x in idxs if x!=a]
                if len(pos) > num_pos_per_anchor: 
                    pos = random.sample(pos, num_pos_per_anchor)
                
                # [EMOJI]
                neg = random.sample(neg_pool, k=min(num_neg_per_anchor, len(neg_pool)))
                
                for p in pos:
                    for n in neg:
                        if n==a or n==p: continue
                        triplets.append((a,p,n))
        
        random.shuffle(triplets)
        print(f"[Triplet] [EMOJI] {len(triplets)} [EMOJI]triplets[EMOJI]")
        return triplets

    def semi_hard_triplet(za, zp, zn, margin=0.8):
        """
        [EMOJI]fallback[EMOJI]hardest-in-batch
        [EMOJI] semi-hard[EMOJI]d_ap < d_an < d_ap + margin
        """
        d_ap = torch.norm(za - zp, dim=1)
        d_an = torch.norm(za - zn, dim=1)
        # [EMOJI] semi-hard [EMOJI]d_ap < d_an < d_ap + margin
        mask = (d_an > d_ap) & (d_an < d_ap + margin)
        if mask.any():
            return nn.TripletMarginLoss(margin=margin, p=2)(za[mask], zp[mask], zn[mask])
        
        # fallback[EMOJI]hardest-in-batch
        with torch.no_grad():
            D = torch.cdist(za, zn, p=2)
            n_hard = D.argmin(dim=1)
        return nn.TripletMarginLoss(margin=margin, p=2)(za, zp, zn[n_hard])

    def train_triplet_text(model, tokenizer, triplets, texts, device, epochs=None, bs=None, margin=None, lr=None, freeze_layers=None):
        # [EMOJI]
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
                if (step+1) % 20 == 0:
                    print(f"[Triplet] Ep {ep+1} Step {step+1}/{len(dl)} Loss {loss.item():.4f}")
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
        """
        [EMOJI]gap[EMOJI]
        gap_i = s_i,top1 - s_i,top2 ([EMOJI])
        [EMOJI]gap[EMOJI]Mean-Std[EMOJI]
        """
        print("=== Gap-based[EMOJI] ===")
        
        # [EMOJI]
        Z_np = Z_raw.copy()
        Z_norm = Z_np / (np.linalg.norm(Z_np, axis=1, keepdims=True) + 1e-8)
        
        # [EMOJI]
        group_centers = {}
        for group_name, indices in matched_dict.items():
            if group_name == "Other" or len(indices) == 0:
                continue
            valid_indices = [idx for idx in indices if idx < len(Z_norm)]
            if len(valid_indices) > 0:
                group_center = np.mean(Z_norm[valid_indices], axis=0)
                group_center = group_center / (np.linalg.norm(group_center) + 1e-8)
                group_centers[group_name] = group_center
        
        # [EMOJI]
        all_similarities = []
        for group_name, center in group_centers.items():
            sim = np.dot(Z_norm, center)
            all_similarities.append(sim)
        
        if len(all_similarities) == 0:
            print("  [EMOJI]")
            return matched_dict
        
        all_similarities = np.array(all_similarities).T  # [N, num_groups]
        
        # [EMOJI]gap: s_top1 - s_top2
        sorted_indices = np.argsort(all_similarities, axis=1)[:, ::-1]  # [EMOJI]
        s_top1 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 0]]
        s_top2 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 1]] if all_similarities.shape[1] > 1 else s_top1
        gap = s_top1 - s_top2
        
        # [EMOJI]top1[EMOJI]
        group_names = list(group_centers.keys())
        arg1 = sorted_indices[:, 0]  # [EMOJI]
        
        print(f"  [EMOJI] {len(gap)} [EMOJI]gap[EMOJI]")
        print(f"  Gap[EMOJI]: [{gap.min():.3f}, {gap.max():.3f}]")
        
        # [EMOJI]gap[EMOJI]Mean-Std[EMOJI]
        print("  [EMOJI]gap[EMOJI]...")
        group_thresholds = {}
        
        for group_name, group_idx in label_map.items():
            if group_name == "Other" or group_name not in group_centers:
                continue
                
            # [EMOJI]group_centers[EMOJI]
            if group_name in group_names:
                group_center_idx = group_names.index(group_name)
                
                # [EMOJI]gap
                group_mask = (arg1 == group_center_idx)
                if not group_mask.any():
                    continue
                    
                gaps_group = gap[group_mask]
                mean_gap = gaps_group.mean()
                std_gap = gaps_group.std()
                
                # [EMOJI]mean - α*std[EMOJI]
                base_threshold = mean_gap - alpha * std_gap
                
                # [EMOJI]/[EMOJI]
                min_samples = get_config("gap_min_samples")
                percentile_fallback = get_config("gap_percentile_fallback")
                if len(gaps_group) < min_samples or std_gap < 1e-6:
                    # [EMOJI]
                    threshold = np.percentile(gaps_group, percentile_fallback)
                    print(f"    {group_name}: [EMOJI]:{len(gaps_group)}, std:{std_gap:.6f}[EMOJI]")
                else:
                    # [EMOJI]
                    global_median = np.median(gap)
                    thr_floor = get_config("gap_floor_threshold")
                    mix_ratio = get_config("gap_mix_ratio")
                    threshold = max((1 - mix_ratio) * base_threshold + mix_ratio * global_median, thr_floor)
                
                group_thresholds[group_name] = threshold
                
                print(f"    {group_name}: mean={mean_gap:.3f}, std={std_gap:.3f}, thr={threshold:.3f}")
                print(f"      [EMOJI]: {np.sum(group_mask)}, gap[EMOJI]: [{gaps_group.min():.3f}, {gaps_group.max():.3f}]")
        
        # [EMOJI]gap < [EMOJI] → [EMOJI]
        print("  [EMOJI]...")
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
        print(f"  [EMOJI] {filtered_count} [EMOJI] ({filtered_count/len(gap)*100:.1f}%)")
        for group_name, count in filtered_by_group.items():
            print(f"    {group_name}: [EMOJI] {count} [EMOJI]")
        
        # [EMOJI]
        clean_matched_dict = {}
        for group_name, indices in matched_dict.items():
            if group_name == "Other":
                clean_matched_dict[group_name] = indices  # Other[EMOJI]
                continue
                
            if group_name in group_centers:
                group_center_idx = group_names.index(group_name)
                # [EMOJI]BM25[EMOJI]
                orig_idxs = set(indices)  # [EMOJI]BM25[EMOJI]
                group_mask = (arg1 == group_center_idx) & keep_mask
                filtered_indices = [int(i) for i in np.where(group_mask)[0] if int(i) in orig_idxs]
                clean_matched_dict[group_name] = filtered_indices
            else:
                clean_matched_dict[group_name] = indices
        
        return clean_matched_dict

    def build_group_prototypes(encoder, tokenizer, texts, matched_dict, device, bs=None, min_per_group=None):
        if bs is None: bs = get_config("encoding_batch_size")
        if min_per_group is None: min_per_group = get_config("min_per_group_prototype")
        """
        [EMOJI] matched_dict [EMOJI]/[EMOJI]L2 [EMOJI]
        """
        # [EMOJI]
        def encode_all():
            encoder.eval()
            Z=[]
            for i in range(0, len(texts), bs):
                toks = tokenizer(texts[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks = {k:v.to(device) for k,v in toks.items()}
                Z.append(encoder.encode_tokens(toks).detach().cpu())
            return torch.vstack(Z)  # (N,D)
        
        Z_all = encode_all()           # torch.Tensor [N,D], [EMOJI] L2norm
        G = {}
        for g, idxs in matched_dict.items():
            if g == "Other": 
                continue
            idxs = [i for i in idxs if 0 <= i < len(Z_all)]
            if len(idxs) >= min_per_group:
                proto = Z_all[idxs].mean(0)
                G[g] = nn.functional.normalize(proto, dim=0)  # (D,)
        return G  # dict[str -> (D,)]

    def prototype_separation_training(encoder, tokenizer, all_texts, group_prototypes, device,
                                     epochs=None, bs=None, lr=None, min_separation=None, matched_dict=None):
        # [EMOJI]
        if epochs is None: epochs = get_config("proto_epochs")
        if bs is None: bs = get_config("proto_batch_size")
        if lr is None: lr = get_config("proto_lr")
        if min_separation is None: min_separation = get_config("min_separation")
        """
        [EMOJI]EMA memory bank[EMOJI]
        - [EMOJI]group_prototypes[EMOJI]
        - [EMOJI]epoch[EMOJI]
        """
        # [EMOJI]BERT[EMOJI]
        if hasattr(encoder, 'bert'):
            for p in encoder.bert.embeddings.parameters(): 
                p.requires_grad = False
            for layer in encoder.bert.encoder.layer[:4]:
                for p in layer.parameters(): 
                    p.requires_grad = False
        
        # [EMOJI]EMA memory bank[EMOJI]
        global_prototypes = {}
        for group_name, proto_tensor in group_prototypes.items():
            if group_name != "Other":
                global_prototypes[group_name] = proto_tensor.clone().to(device).detach()
        
        params = [p for p in encoder.parameters() if p.requires_grad]
        print(f"[INFO] Prototype Separation[EMOJI]: {sum(p.numel() for p in params):,}")
        
        opt = torch.optim.AdamW(params, lr=lr)
        
        # [EMOJI]balanced batch sampler
        class BalancedBatchSampler:
            def __init__(self, indices_by_group, groups, m_per_group=4, batch_size=64):
                self.buckets = {g: list(idxs) for g, idxs in indices_by_group.items()}
                self.groups = [g for g in groups if len(self.buckets[g]) >= m_per_group]
                self.m = m_per_group
                self.batch_size = batch_size
                
            def __iter__(self):
                # [EMOJI]
                groups_shuffled = self.groups.copy()
                random.shuffle(groups_shuffled)
                
                # [EMOJI]2[EMOJI]batch
                for i in range(0, len(groups_shuffled), 2):
                    gs = groups_shuffled[i:i+2]
                    if len(gs) < 2:
                        break
                    
                    batch = []
                    for g in gs:
                        # [EMOJI]
                        group_samples = self.buckets[g].copy()
                        random.shuffle(group_samples)
                        # [EMOJI]m[EMOJI]
                        batch.extend(group_samples[:self.m])
                    
                    # [EMOJI]batch[EMOJI]
                    if len(batch) < self.batch_size:
                        remaining = self.batch_size - len(batch)
                        all_samples = []
                        for g, samples in self.buckets.items():
                            if g not in gs:  # [EMOJI]
                                all_samples.extend(samples)
                        if all_samples:
                            random.shuffle(all_samples)
                            batch.extend(all_samples[:remaining])
                    
                    yield batch[:self.batch_size]
                    
            def __len__(self):
                return max(1, len(self.groups) // 2)
        
        # [EMOJI]balanced sampler
        m_per_group = bs // 4
        sampler = BalancedBatchSampler(matched_dict, list(global_prototypes.keys()), 
                                     m_per_group=m_per_group, batch_size=bs)
        
        print(f"[INFO] [EMOJI]: {list(global_prototypes.keys())}")
        print(f"[INFO] [EMOJI]Balanced Batch[EMOJI]{m_per_group}[EMOJI]")
        
        encoder.train()
        for ep in range(epochs):
            total_sep_loss = 0
            total_center_loss = 0
            total_loss = 0
            steps = 0
            
            for batch_indices in sampler:
                batch_texts = [all_texts[i] for i in batch_indices]
                
                # [EMOJI]batch
                inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                                 truncation=True, max_length=256).to(device)
                outputs = encoder.encode_tokens(inputs)
                z_batch = nn.functional.normalize(outputs, p=2, dim=-1)
                
                # [EMOJI]batch[EMOJI]
                batch_group_means = {}
                for group_name, indices in matched_dict.items():
                    if group_name == "Other":
                        continue
                    
                    # [EMOJI]batch[EMOJI]
                    group_mask = torch.tensor([i in indices for i in batch_indices], device=device)
                    if group_mask.sum() > 0:
                        group_embeddings = z_batch[group_mask]
                        batch_group_means[group_name] = group_embeddings.mean(dim=0)
                
                # 1. Proto-level separation loss ([EMOJI]batch[EMOJI])
                separation_loss = torch.tensor(0.0, device=device)
                if len(batch_group_means) >= 2:
                    group_names = list(batch_group_means.keys())
                    means = torch.stack([batch_group_means[g] for g in group_names], dim=0)
                    
                    # [EMOJI]pairwise[EMOJI]
                    dists = torch.cdist(means, means, p=2)
                    mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
                    upper_dists = dists[mask]
                    
                    # [EMOJI] → [EMOJI]relu[EMOJI]
                    penalties = F.relu(min_separation - upper_dists)
                    separation_loss = penalties.sum()
                
                # 2. Center-pull loss ([EMOJI])
                center_loss = torch.tensor(0.0, device=device)
                for group_name, indices in matched_dict.items():
                    if group_name == "Other" or group_name not in global_prototypes:
                        continue
                    
                    # [EMOJI]batch[EMOJI]
                    group_mask = torch.tensor([i in indices for i in batch_indices], device=device)
                    if group_mask.sum() > 0:
                        group_embeddings = z_batch[group_mask]
                        prototype = global_prototypes[group_name]
                        
                        # [EMOJI]
                        distances = torch.norm(group_embeddings - prototype, p=2, dim=1)
                        center_loss += distances.mean()
                
                # [EMOJI]loss
                total_batch_loss = separation_loss + 0.1 * center_loss
                
                # [EMOJI]
                if total_batch_loss > 0:
                    opt.zero_grad()
                    total_batch_loss.backward()
                    opt.step()
                    
                    total_sep_loss += separation_loss.item()
                    total_center_loss += center_loss.item()
                    total_loss += total_batch_loss.item()
                    steps += 1
            
            if steps > 0:
                avg_sep = total_sep_loss / steps
                avg_center = total_center_loss / steps
                avg_total = total_loss / steps
                print(f"[PrototypeSep] Epoch {ep+1}/{epochs} Sep={avg_sep:.4f} Center={avg_center:.4f} Total={avg_total:.4f}")
            
            # [EMOJI]epoch[EMOJI]EMA[EMOJI]
            print(f"  Epoch {ep+1}[EMOJI]...")
            with torch.no_grad():
                # [EMOJI]
                encoder.eval()
                Z_all = []
                for i in range(0, len(all_texts), bs):
                    batch_texts = all_texts[i:i+bs]
                    inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, 
                                     truncation=True, max_length=256).to(device)
                    outputs = encoder.encode_tokens(inputs)
                    Z_all.append(outputs.detach())
                Z_all = torch.cat(Z_all, dim=0)  # [N, D]
                Z_all = nn.functional.normalize(Z_all, p=2, dim=-1)
                
                # [EMOJI]EMA[EMOJI]
                ema_alpha = get_config("ema_alpha")  # EMA[EMOJI]
                for group_name, indices in matched_dict.items():
                    if group_name != "Other" and group_name in global_prototypes:
                        valid_indices = [i for i in indices if 0 <= i < len(Z_all)]
                        if len(valid_indices) > 0:
                            # [EMOJI]epoch[EMOJI]
                            current_proto = Z_all[valid_indices].mean(dim=0)
                            current_proto = nn.functional.normalize(current_proto, p=2, dim=-1)
                            
                            # EMA[EMOJI]
                            global_prototypes[group_name] = (1 - ema_alpha) * global_prototypes[group_name] + ema_alpha * current_proto
                            global_prototypes[group_name] = nn.functional.normalize(global_prototypes[group_name], p=2, dim=-1)
                
                encoder.train()  # [EMOJI]
        
        return encoder

    def evaluate_clustering_quality(Z, true_labels, matched_dict, group_names):
        """[EMOJI]"""
        from sklearn.metrics import silhouette_score
        
        # [EMOJI]silhouette score
        silhouette = silhouette_score(Z, true_labels)
        
        # [EMOJI]BM25[EMOJI]silhouette score
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
        
        # [EMOJI]
        group_stats = {}
        for group_name in group_names:
            if group_name in matched_dict:
                group_indices = matched_dict[group_name]
                if len(group_indices) > 1:
                    group_embeddings = Z[group_indices]
                    
                    # [EMOJI]
                    intra_distances = []
                    for i in range(len(group_embeddings)):
                        for j in range(i+1, len(group_embeddings)):
                            dist = np.linalg.norm(group_embeddings[i] - group_embeddings[j])
                            intra_distances.append(dist)
                    intra_cohesion = np.mean(intra_distances) if intra_distances else 0
                    
                    # [EMOJI]
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
                    
                    # [EMOJI]/[EMOJI]
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

    # ===================== [EMOJI] =====================
    
    # [EMOJI]
    print("\n=== [EMOJI] ===")
    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            print(f"{group_name}: {len(indices)} docs ([EMOJI])")
            continue
        
        if len(indices) == 0:
            print(f"{group_name}: 0 docs")
            continue
            
        # [EMOJI]
        group_labels = [all_labels[i] for i in indices if i < len(all_labels)]
        if len(group_labels) == 0:
            print(f"{group_name}: 0 docs ([EMOJI])")
            continue
            
        # [EMOJI]
        from collections import Counter
        label_counts = Counter(group_labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        purity = most_common_count / len(group_labels)
        
        print(f"{group_name}: {len(indices)} docs, [EMOJI]={purity:.3f} ([EMOJI]: {most_common_label}, {most_common_count}/{len(group_labels)})")
    
    # [EMOJI]ID[EMOJI]Other[EMOJI]
    print("\n=== [EMOJI] ===")
    doc_to_group = {}  # [EMOJI]
    final_matched_dict = {}
    
    # [EMOJI]Other[EMOJI]
    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            continue
            
        final_matched_dict[group_name] = []
        for doc_id in indices:
            if doc_id not in doc_to_group:
                doc_to_group[doc_id] = group_name
                final_matched_dict[group_name].append(doc_id)
            else:
                print(f"[[EMOJI]] [EMOJI] {doc_id} [EMOJI] {doc_to_group[doc_id]} [EMOJI]")
    
    # [EMOJI]Other[EMOJI]
    if "Other" in matched_dict:
        final_matched_dict["Other"] = []
        for doc_id in matched_dict["Other"]:
            if doc_id in doc_to_group:
                # [EMOJI]
                original_group = doc_to_group[doc_id]
                final_matched_dict[original_group].remove(doc_id)
                print(f"[[EMOJI]] [EMOJI] {doc_id} [EMOJI] {original_group} [EMOJI] Other [EMOJI]")
            
            doc_to_group[doc_id] = "Other"
            final_matched_dict["Other"].append(doc_id)
    
    # [EMOJI]matched_dict
    matched_dict = final_matched_dict
    
    # [EMOJI]
    print("\n=== [EMOJI] ===")
    for group_name, indices in matched_dict.items():
        print(f"{group_name}: {len(indices)} docs")
    
    # [EMOJI]BERT embedding[EMOJI]
    print("\n=== [EMOJI]BERT embedding ===")
    Z_raw = encode_corpus(encoder, tokenizer, all_texts, device)
    
    # [EMOJI]
    label_map, cur = {}, 0
    for g in ["Group 1", "Group 2", "Group 3"]:
        if g in matched_dict:
            label_map[g] = cur; cur += 1
    other_label = cur; label_map["Other"] = other_label
    
    # [EMOJI]BM25[EMOJI]
    import copy
    bm25_results = copy.deepcopy(matched_dict)
    
    # ===================== [EMOJI]Gap-based[EMOJI] =====================
    print("\n=== [EMOJI]Gap-based[EMOJI] ===")
    clean_matched_dict = gap_based_group_filtering(Z_raw, matched_dict, label_map, alpha=0.5)
    
    # [EMOJI]
    print("\n=== [EMOJI] ===")
    matched_dict_for_display = copy.deepcopy(matched_dict)  # [EMOJI] BM25 [EMOJI]
    matched_dict = clean_matched_dict  # [EMOJI]
    
    # [EMOJI] finetune mode[EMOJI]
    print("\n=== [EMOJI] ===")
    with open(os.path.join(out_dir, "filtered_group_assignment.json"), "w", encoding="utf-8") as f:
        json.dump(matched_dict, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] [EMOJI]: {os.path.join(out_dir, 'filtered_group_assignment.json')}")
    
    # [EMOJI]gap-based[EMOJI]
    print("\n=== [EMOJI]Gap-based[EMOJI] ===")
    with open(os.path.join(out_dir, "gap_based_filter_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "original_sizes": {k: len(v) for k, v in bm25_results.items()},
            "clean_sizes": {k: len(v) for k, v in clean_matched_dict.items()},
            "method": "gap_based_group_filtering",
            "alpha": 0.5,
            "threshold_method": "mean_minus_alpha_std_with_fallback"
        }, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Gap-based[EMOJI]: {os.path.join(out_dir, 'gap_based_filter_results.json')}")
    
    # ===================== [EMOJI] =====================
    print("\n=== [EMOJI] ===")
    group_prototypes = build_group_prototypes(encoder, tokenizer, all_texts, matched_dict, device)
    print(f"[INFO] [EMOJI] {len(group_prototypes)} [EMOJI]: {list(group_prototypes.keys())}")
    
    # [EMOJI]
    print("\n=== [EMOJI] ===")
    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            print(f"{group_name}: {len(indices)} docs ([EMOJI])")
            continue
        
        if len(indices) == 0:
            print(f"{group_name}: 0 docs")
            continue
            
        # [EMOJI]
        group_labels = [all_labels[i] for i in indices if i < len(all_labels)]
        if len(group_labels) == 0:
            print(f"{group_name}: 0 docs ([EMOJI])")
            continue
            
        # [EMOJI]
        from collections import Counter
        label_counts = Counter(group_labels)
        most_common_label, most_common_count = label_counts.most_common(1)[0]
        purity = most_common_count / len(group_labels)
        
        print(f"{group_name}: {len(indices)} docs, [EMOJI]={purity:.3f} ([EMOJI]: {most_common_label}, {most_common_count}/{len(group_labels)})")

    # ===================== [EMOJI]Triplet[EMOJI]=====================
    print("\n=== [EMOJI]Triplet[EMOJI] ===")
    # [EMOJI]matched_dict[EMOJI]Other[EMOJI]
    print(f"[DEBUG] matched_dict keys: {list(matched_dict.keys())}")
    for g, idxs in matched_dict.items():
        print(f"[DEBUG] {g}: {len(idxs)} docs")
    
    print(f"    DEBUG: Generating triplets from groups...")
    try:
        triplets = generate_triplets_from_groups(matched_dict)
        print(f"    DEBUG: Generated {len(triplets)} triplets")
        
        if len(triplets) > 0:
            print(f"[INFO] [EMOJI] {len(triplets)} [EMOJI]triplets[EMOJI]")
            print(f"    DEBUG: Starting triplet training...")
            encoder = train_triplet_text(encoder, tokenizer, triplets, all_texts, device)
            print("[INFO] Triplet[EMOJI]embedding[EMOJI]")
        else:
            print("[WARN] Triplets [EMOJI] Triplet[EMOJI]")
    except Exception as e:
        print(f"    ERROR: Triplet generation/training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # ===================== [EMOJI]Prototype Separation[EMOJI] =====================
    print("\n=== Prototype Separation [EMOJI] ===")
    print(f"    DEBUG: Starting prototype separation training...")
    try:
        encoder = prototype_separation_training(encoder, tokenizer, all_texts, group_prototypes, device,
                                              matched_dict=clean_matched_dict)
        print("[INFO] Prototype Separation[EMOJI]")
    except Exception as e:
        print(f"    ERROR: Prototype separation training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # ===================== [EMOJI]EMA[EMOJI] =====================
    print("\n=== EMA[EMOJI] ===")
    
    def l2norm(X): 
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    
    # [EMOJI]embedding[EMOJI]EMA[EMOJI]
    print(f"    DEBUG: Encoding corpus with {len(all_texts)} texts...")
    try:
        Z_current = encode_corpus(encoder, tokenizer, all_texts, device)
        print(f"    DEBUG: Encoded corpus shape: {Z_current.shape}")
        Zn = l2norm(Z_current)
        print(f"    DEBUG: Normalized embeddings shape: {Zn.shape}")
    except Exception as e:
        print(f"    ERROR: Corpus encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # [EMOJI]
    group_stats = {}
    ema_alpha = get_config("ema_alpha")  # EMA[EMOJI]
    
    for g, train_idxs in matched_dict.items():
        if g == "Other" or len(train_idxs) < 5: 
            continue
        train_idxs = [i for i in train_idxs if 0 <= i < len(Zn)]
        if len(train_idxs) < 5: 
            continue

        # [EMOJI]
        proto = Zn[train_idxs].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        
        # [EMOJI]
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
            "ema_proto": proto.copy(),  # EMA[EMOJI]
            "ema_r_core": float(r_core),  # EMA[EMOJI]
            "ema_r_near": float(r_near),
            "ema_r_edge": float(r_edge)
        }
    
    # [EMOJI]EMA[EMOJI]
    print("[EMOJI]EMA[EMOJI]...")
    
    for g, stat in group_stats.items():
        train_set = set(stat["train_idxs"])
        proto = stat["ema_proto"]
        r_core = stat["ema_r_core"]
        
        # [EMOJI]
        d_all = np.linalg.norm(Zn - proto, axis=1)
        
        # [EMOJI] ≤ r_core [EMOJI]
        high_conf_new = []
        for i, d in enumerate(d_all):
            if i not in train_set and d <= r_core:
                high_conf_new.append((i, d))
        
        if len(high_conf_new) > 0:
            print(f"  {g}: [EMOJI] {len(high_conf_new)} [EMOJI]")
            
            # [EMOJI]embedding
            high_conf_indices = [i for i, _ in high_conf_new]
            high_conf_embeddings = Zn[high_conf_indices]
            
            # EMA[EMOJI]
            new_proto = high_conf_embeddings.mean(axis=0)
            new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-8)
            
            # [EMOJI] + [EMOJI]
            stat["ema_proto"] = (1 - ema_alpha) * stat["ema_proto"] + ema_alpha * new_proto
            stat["ema_proto"] = stat["ema_proto"] / (np.linalg.norm(stat["ema_proto"]) + 1e-8)
            
            # [EMOJI]
            all_relevant_indices = stat["train_idxs"] + high_conf_indices
            d_combined = np.linalg.norm(Zn[all_relevant_indices] - stat["ema_proto"], axis=1)
            
            # EMA[EMOJI]
            new_r_core = np.quantile(d_combined, 0.50)
            new_r_near = np.quantile(d_combined, 0.80) 
            new_r_edge = np.quantile(d_combined, 0.90)
            
            stat["ema_r_core"] = (1 - ema_alpha) * stat["ema_r_core"] + ema_alpha * new_r_core
            stat["ema_r_near"] = (1 - ema_alpha) * stat["ema_r_near"] + ema_alpha * new_r_near
            stat["ema_r_edge"] = (1 - ema_alpha) * stat["ema_r_edge"] + ema_alpha * new_r_edge
            
            print(f"    [EMOJI]EMA[EMOJI]: core={stat['ema_r_core']:.3f}, near={stat['ema_r_near']:.3f}, edge={stat['ema_r_edge']:.3f}")
        else:
            print(f"  {g}: [EMOJI]")
    
    print("[INFO] EMA[EMOJI]")

    # [EMOJI]embedding
    print("\n=== [EMOJI]embedding ===")
    print(f"    DEBUG: Encoding final trained embeddings...")
    try:
        Z_trained = encode_corpus(encoder, tokenizer, all_texts, device)
        print(f"    DEBUG: Final embeddings shape: {Z_trained.shape}")
    except Exception as e:
        print(f"    ERROR: Final embedding encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # [EMOJI]main_categories[EMOJI]
    main_categories = [get_main_category(label) for label in all_labels]
    
    # [EMOJI]embedding[EMOJI]BM25[EMOJI]
    print("\n=== [EMOJI]embedding[EMOJI] ===")
    cluster_eval_raw = evaluate_clustering_quality(Z_raw, main_categories, bm25_results, list(bm25_results.keys()))
    
    # [EMOJI]embedding[EMOJI]
    print("\n=== [EMOJI]embedding[EMOJI] ===")
    cluster_eval_trained = evaluate_clustering_quality(Z_trained, main_categories, matched_dict, list(matched_dict.keys()))

    # [EMOJI]
    print("\n=== [EMOJI] ===")
    print(f"Silhouette Score ([EMOJI]): {cluster_eval_raw['silhouette_true_labels']:.4f} → {cluster_eval_trained['silhouette_true_labels']:.4f}")
    print(f"Silhouette Score (BM25[EMOJI]): {cluster_eval_raw['silhouette_group_labels']:.4f} → {cluster_eval_trained['silhouette_group_labels']:.4f}")
    
    # [EMOJI]numpy[EMOJI]Python[EMOJI]
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
    
    with open(os.path.join(out_dir, "clustering_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump({
            "raw_embedding": cluster_eval_raw_serializable,
            "trained_embedding": cluster_eval_trained_serializable,
            "triplets_count": len(triplets),
            "training_epochs": 5,
            "margin": 0.8
        }, f, ensure_ascii=False, indent=2)

    # 2D[EMOJI]
    print("\n=== 2D[EMOJI] ===")
    print(f"    DEBUG: Starting 2D visualization with {len(Z_raw)} documents...")
    try:
        # [EMOJI]perplexity
        n = len(Z_raw)
        perp = max(2, min(30, (n - 1) // 3))
        print(f"    DEBUG: Using perplexity: {perp}")
        
        # [EMOJI]embedding[EMOJI]t-SNE
        print(f"    DEBUG: Computing t-SNE for raw embeddings...")
        X2_raw = TSNE(
            n_components=2, 
            perplexity=perp, 
            random_state=42, 
            max_iter=get_config("tsne_max_iter")
        ).fit_transform(Z_raw)
        print(f"    DEBUG: Raw t-SNE shape: {X2_raw.shape}")
        
        # [EMOJI]embedding[EMOJI]t-SNE
        print(f"    DEBUG: Computing t-SNE for trained embeddings...")
        X2_trained = TSNE(
            n_components=2, 
            perplexity=perp, 
            random_state=42, 
            max_iter=get_config("tsne_max_iter")
        ).fit_transform(Z_trained)
        print(f"    DEBUG: Trained t-SNE shape: {X2_trained.shape}")
        
        # [EMOJI]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # [EMOJI]
        unique_main_cats = sorted(list(set(main_categories)))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_main_cats)))
        color_map = dict(zip(unique_main_cats, colors))
        
        # [EMOJI]embedding
        for main_cat in unique_main_cats:
            cat_mask = [cat == main_cat for cat in main_categories]
            cat_indices = np.where(cat_mask)[0]
            
            if len(cat_indices) > 0:
                cat_2d = X2_raw[cat_indices]
                ax1.scatter(cat_2d[:,0], cat_2d[:,1], 
                           c=[color_map[main_cat]], alpha=0.8, s=60, 
                           marker='o', label=f'{main_cat}')
        
        ax1.set_title(f"[EMOJI]BERT Embedding (Silhouette: {cluster_eval_raw['silhouette_true_labels']:.3f})", 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1'); ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        # [EMOJI]embedding
        for main_cat in unique_main_cats:
            cat_mask = [cat == main_cat for cat in main_categories]
            cat_indices = np.where(cat_mask)[0]
            
            if len(cat_indices) > 0:
                cat_2d = X2_trained[cat_indices]
                ax2.scatter(cat_2d[:,0], cat_2d[:,1], 
                           c=[color_map[main_cat]], alpha=0.8, s=60, 
                           marker='o', label=f'{main_cat}')
        
        ax2.set_title(f"Triplet[EMOJI]Embedding (Silhouette: {cluster_eval_trained['silhouette_true_labels']:.3f})", 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE Dimension 1'); ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "triplet_training_comparison.png"), dpi=300, bbox_inches='tight')
        print(f"[SAVE] Triplet[EMOJI]: {os.path.join(out_dir, 'triplet_training_comparison.png')}")
        
        # [EMOJI]t-SNE[EMOJI]
        with open(os.path.join(out_dir, "triplet_tsne_comparison.json"), "w", encoding="utf-8") as f:
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
        print("[WARN] 2D[EMOJI]:", e)

    # [EMOJI]embedding
    print(f"    DEBUG: Saving trained embeddings and model...")
    try:
        np.save(os.path.join(out_dir, "embeddings_trained.npy"), Z_trained)
        print(f"    DEBUG: Saved embeddings to embeddings_trained.npy")
        torch.save(encoder.state_dict(), os.path.join(out_dir, "triplet_trained_encoder.pth"))
        print(f"    DEBUG: Saved model state dict to triplet_trained_encoder.pth")
    except Exception as e:
        print(f"    ERROR: Failed to save embeddings or model: {e}")
        import traceback
        traceback.print_exc()

    with open(os.path.join(out_dir, "triplet_run_stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "bm25_sizes": {k: len(v) for k, v in matched_dict.items()},
            "device": device,
            "proj_dim": encoder.out_dim,
            "triplets_count": len(triplets),
            "training_epochs": 5,
            "margin": 0.8,
            "silhouette_improvement": float(cluster_eval_trained['silhouette_true_labels'] - cluster_eval_raw['silhouette_true_labels'])
        }, f, ensure_ascii=False, indent=2)
    
    print("\n=== [EMOJI] ===")
    print(f"Silhouette Score[EMOJI]: {cluster_eval_trained['silhouette_true_labels'] - cluster_eval_raw['silhouette_true_labels']:+.4f}")
    print("=== DONE ===")

    # [EMOJI]2D[EMOJI]UI[EMOJI]
    print("\n=== [EMOJI]2D[EMOJI] ===")
    
    # [EMOJI]embedding - [EMOJI]final[EMOJI]
    df_articles = df_clean  # [EMOJI]
    
    # [EMOJI]
    model_original = BertModel.from_pretrained('bert-base-uncased').to(device)
    model_original.eval()
    
    # [EMOJI]encoder[EMOJI]finetuned[EMOJI]
    # [EMOJI]SentenceEncoder[EMOJI]get_all_cls_vectors[EMOJI]
    class SentenceEncoderAdapter:
        def __init__(self, sentence_encoder):
            self.sentence_encoder = sentence_encoder
            self.config = type('Config', (), {'hidden_size': sentence_encoder.out_dim})()
        
        def __call__(self, input_ids, attention_mask):
            # [EMOJI]input_ids[EMOJI]attention_mask[EMOJI]tokenizer[EMOJI]
            tokens = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            # [EMOJI]SentenceEncoder[EMOJI]encode_tokens[EMOJI]
            encoded = self.sentence_encoder.encode_tokens(tokens)
            # [EMOJI]BERT[EMOJI]
            class MockOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state
            # [EMOJI]get_all_cls_vectors[EMOJI](batch_size, seq_len, hidden_size)[EMOJI]tensor
            # [EMOJI]encoded[EMOJI]
            batch_size = encoded.shape[0]
            hidden_size = encoded.shape[1]
            # [EMOJI]
            seq_len = input_ids.shape[1]
            # [EMOJI]encoded[EMOJI]
            repeated = encoded.unsqueeze(1).repeat(1, seq_len, 1)
            return MockOutput(repeated)
    
    model_finetuned_adapter = SentenceEncoderAdapter(encoder)
    model_finetuned_adapter.eval = lambda: None  # [EMOJI]eval[EMOJI]
    
    # [EMOJI]embedding
    cls_vectors_before = get_all_cls_vectors(df_articles, model_original, tokenizer, device).cpu()
    cls_vectors_after = get_all_cls_vectors(df_articles, model_finetuned_adapter, tokenizer, device).cpu()
    cls_vectors_after_cpu = cls_vectors_after.cpu().numpy()
    
    # [EMOJI]t-SNE[EMOJI]
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
    
    # [EMOJI] - [EMOJI]final[EMOJI]
    group_centers = {}
    print(f"    DEBUG: [EMOJI]matched_dict keys: {list(matched_dict.keys())}")
    print(f"    DEBUG: df_articles shape: {df_articles.shape}, index range: {df_articles.index.min()} - {df_articles.index.max()}")
    print(f"    DEBUG: projected_2d_after shape: {projected_2d_after.shape}")
    
    for group_name, indices in matched_dict.items():
        if group_name == "Other":
            continue  # [EMOJI]Other[EMOJI]
            
        if len(indices) > 0:
            print(f"    DEBUG: [EMOJI] {group_name}[EMOJI]: {indices[:5]}...")
            
            # [EMOJI]2D[EMOJI]
            valid_indices = [i for i in indices if i < len(projected_2d_after)]
            print(f"    DEBUG: {group_name} [EMOJI]: {len(valid_indices)} ([EMOJI]: {len(indices)})")
            
            if len(valid_indices) > 0:
                group_2d_points = projected_2d_after[valid_indices]
                group_center_2d = np.mean(group_2d_points, axis=0)
                group_centers[group_name] = group_center_2d
                print(f"     Group {group_name} center: {group_center_2d}")
            else:
                print(f"        {group_name} [EMOJI]2D[EMOJI]")
        else:
            print(f"        {group_name} [EMOJI]")
    
    print(f"    DEBUG: [EMOJI] group_centers: {group_centers}")

    # [EMOJI]create_plotly_figure[EMOJI] - [EMOJI]
    def create_plotly_figure(projected_2d, title, is_after=False, highlighted_indices=None, group_centers=None, matched_dict_param=None):
        print(f"    DEBUG: create_plotly_figure called with projected_2d shape: {projected_2d.shape}")
        print(f"    DEBUG: projected_2d length: {len(projected_2d)}")
        print(f"    DEBUG: projected_2d sample: {projected_2d[:3] if len(projected_2d) >= 3 else projected_2d}")
        
        fig = go.Figure()
        
        # Before [EMOJI] After training [EMOJI]
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
                symbol="circle",  # [EMOJI] circle
                line=dict(width=bg_style["line_width"], color=bg_style["line_color"])
            ),
            customdata=custom_data,
            hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            hovertext=hover_texts
        ))
        
        # Add highlighted points if provided ([EMOJI])
        if highlighted_indices:
            # [EMOJI]
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
        
        # If After Training plot, add group center points (using group colors)
        print(f"    DEBUG: is_after={is_after}, group_centers={group_centers}")
        if is_after and group_centers:
            print(f"    DEBUG: [EMOJI] {len(group_centers)} [EMOJI]")
            center_style = PLOT_STYLES["center"]
            print(f"    DEBUG: center_style = {center_style}")
            for group_name, center_2d in group_centers.items():
                # [EMOJI]
                color = get_group_color(group_name)
                print(f"    DEBUG: [EMOJI] {group_name} [EMOJI]: {center_2d}, [EMOJI]: {color}")
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
            print(f"    DEBUG: [EMOJI]is_after={is_after}, group_centers={group_centers}")
        
        # [EMOJI]
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
        
        # DEBUG: [EMOJI] layout [EMOJI]
        print(f"    DEBUG: fig.layout.xaxis.showgrid = {fig.layout.xaxis.showgrid}")
        print(f"    DEBUG: fig.layout.xaxis.showline = {fig.layout.xaxis.showline}")
        print(f"    DEBUG: fig.layout.xaxis.mirror = {fig.layout.xaxis.mirror}")
        
        return fig

    # [EMOJI]UI[EMOJI]
    # Before: [EMOJI]
    fig_before = create_plotly_figure(projected_2d_before, "2D Projection Before Finetuning", False, None, None, None)
    # After: [EMOJI] BM25 [EMOJI]
    fig_after = create_plotly_figure(projected_2d_after, "2D Projection After Finetuning", True, None, group_centers, matched_dict_for_display)
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print(f"    DEBUG: ==========================================")
    
    return fig_before, fig_after

# Function to update training figures with highlights
def run_training_with_highlights(highlighted_indices):
    """Update training figures with highlighted indices without re-running training"""
    global df
    
    # Convert highlighted_indices to list if it's a tuple
    if isinstance(highlighted_indices, tuple):
        highlighted_indices = list(highlighted_indices)
    
    # Load the saved models instead of re-running training
    model_original = BertModel.from_pretrained('bert-base-uncased').to(device)
    model_finetuned = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load the saved model
    model_save_path = "../Keyword_Group/bert_finetuned.pth"
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=device)
        model_finetuned.load_state_dict(checkpoint['model_state_dict'])
        model_finetuned.eval()
    
    if "df_global" not in globals():
        df_articles = pd.read_csv(csv_path)
    else:
        df_articles = df
    
    # Load matched_dict from saved file
    matched_dict = {}
    bm25_results_path = os.path.join("test_results", "bm25_search_results.json")
    if os.path.exists(bm25_results_path):
        with open(bm25_results_path, "r", encoding="utf-8") as f:
            matched_dict = json.load(f)
        print(f"    DEBUG: Loaded matched_dict with keys: {list(matched_dict.keys())}")
    else:
        print(f"    DEBUG: No bm25_search_results.json found at {bm25_results_path}")
    
    # Get embeddings
    cls_vectors_before = get_all_cls_vectors(df_articles, model_original, tokenizer, device).cpu()
    cls_vectors_after = get_all_cls_vectors(df_articles, model_finetuned, tokenizer, device).cpu()
    cls_vectors_after_cpu = cls_vectors_after.cpu().numpy()
    
    # Calculate TSNE
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
    
    # Calculate group centers for After Training plot
    group_centers = {}
    if matched_dict:
        print(f"    DEBUG: Calculating group centers for highlights function")
        for group_name, indices in matched_dict.items():
            if group_name == "Other":
                continue  # [EMOJI]Other[EMOJI]
                
            if len(indices) > 0:
                # [EMOJI]2D[EMOJI]
                valid_indices = [i for i in indices if i < len(projected_2d_after)]
                if len(valid_indices) > 0:
                    group_2d_points = projected_2d_after[valid_indices]
                    group_center_2d = np.mean(group_2d_points, axis=0)
                    group_centers[group_name] = group_center_2d
                    print(f"     {group_name} center: {group_center_2d}")
    
    print(f"    DEBUG: Final group_centers for highlights: {group_centers}")
    
    # Create figures with highlights
    def create_plotly_figure_with_highlights(projected_2d, title, highlighted_indices=None, group_centers=None):
        fig = go.Figure()
        
        # Add scatter plot - all points in same color
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
                symbol="circle",  # [EMOJI] circle
                line=dict(width=bg_style["line_width"], color=bg_style["line_color"])
            ),
            customdata=custom_data,
            hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            hovertext=hover_texts
        ))
        
        # Add highlighted points if provided
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
            
            if highlighted_x:  # Only add trace if there are highlighted points
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
        
        # Add group center points for After Training plot (using unified group colors and styles)
        if "After" in title and group_centers:
            print(f"    DEBUG: Adding {len(group_centers)} prototype centers to {title}")
            center_style = PLOT_STYLES["center"]
            for group_name, center_2d in group_centers.items():
                color = get_group_color(group_name)  # [EMOJI]
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
        
        # [EMOJI]
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
    
    # Create figures
    fig_before = create_plotly_figure_with_highlights(projected_2d_before, "Before Training", highlighted_indices, None)
    fig_after = create_plotly_figure_with_highlights(projected_2d_after, "After Training", highlighted_indices, group_centers)
    
    return fig_before, fig_after

# Global cache for keyword embeddings, t-SNE results, and text positions
_KEYWORD_TSNE_CACHE = None

# Global cache for recommended articles search results
_ARTICLES_CACHE = {}

# Global cache for documents 2D visualization results  
_DOCUMENTS_2D_CACHE = {}

def clear_caches():
    """Clear all caches when data changes"""
    global _ARTICLES_CACHE, _DOCUMENTS_2D_CACHE
    _ARTICLES_CACHE.clear()
    _DOCUMENTS_2D_CACHE.clear()
    print("Cleared all caches due to data change")


def print_performance_tips():
    """Print performance optimization tips"""
    print("\n    [EMOJI] PERFORMANCE OPTIMIZATION TIPS:")
    print("    1. Document embeddings are pre-computed for faster keyword response")
    print("    2. t-SNE results are cached to avoid recalculation")
    print("    3. Semantic search is used when embeddings are available")
    print("    4. Batch processing is optimized for your device type")
    print("    5. Cache is automatically managed for optimal performance")
    print("           First keyword click may still be slow due to model loading")
    print("           Subsequent clicks should be 3-5x faster")

# Add necessary 2D visualization callbacks - split into two callbacks for performance
@app.callback(
    Output('keywords-2d-plot', 'figure'),
    Input('keywords-2d-plot', 'id')  # Only initial load
)
def update_keywords_2d_plot(plot_id):
    """Update keyword 2D dimensionality reduction visualization chart with text labels - initial load only"""
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
    
    # Get keyword embeddings and dimensionality reduction results
    try:
        def adjust_text_positions(x_coords, y_coords, keywords, min_distance=0.12):
            """Smart text positioning: move overlapping texts to nearby empty spaces"""
            import numpy as np
            adjusted_x = x_coords.copy()
            adjusted_y = y_coords.copy()
            
            # Calculate relative distances based on data range
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            min_dist = min(x_range, y_range) * min_distance
            
            def find_empty_space(current_x, current_y, all_x, all_y, search_radius=0.3):
                """Find an empty space near the current position"""
                search_dist = min(x_range, y_range) * search_radius
                
                # Try different angles around the original position
                for angle in np.linspace(0, 2*np.pi, 16):  # Try 16 directions
                    for radius in np.linspace(min_dist, search_dist, 8):  # Try different distances
                        test_x = current_x + radius * np.cos(angle)
                        test_y = current_y + radius * np.sin(angle)
                        
                        # Check if this position is far enough from all other texts
                        too_close = False
                        for other_x, other_y in zip(all_x, all_y):
                            if abs(test_x - other_x) < min_dist and abs(test_y - other_y) < min_dist:
                                too_close = True
                                break
                        
                        if not too_close:
                            return test_x, test_y
                
                # If no good position found, return original with small offset
                return current_x + np.random.normal(0, min_dist*0.5), current_y + np.random.normal(0, min_dist*0.5)
            
            # Iteratively resolve overlaps
            for iteration in range(30):
                overlaps_found = 0
                
                for i in range(len(adjusted_x)):
                    # Check if current position overlaps with any other text
                    has_overlap = False
                    for j in range(len(adjusted_x)):
                        if i != j:
                            if (abs(adjusted_x[i] - adjusted_x[j]) < min_dist and 
                                abs(adjusted_y[i] - adjusted_y[j]) < min_dist):
                                has_overlap = True
                                break
                    
                    # If overlap found, move to empty space
                    if has_overlap:
                        new_x, new_y = find_empty_space(
                            x_coords[i], y_coords[i],  # Start from original position
                            adjusted_x, adjusted_y
                        )
                        adjusted_x[i] = new_x
                        adjusted_y[i] = new_y
                        overlaps_found += 1
                
                # If no overlaps found, we're done
                if overlaps_found == 0:
                    print(f"Text positioning converged after {iteration + 1} iterations")
                    break
            
            return adjusted_x, adjusted_y
        
        # Use cache if available to avoid recalculation
        if _KEYWORD_TSNE_CACHE is None:
            print("Computing t-SNE for keywords (first time)...")
            # Calculate keyword embeddings
            keyword_embeddings = embedding_model_kw.encode(GLOBAL_KEYWORDS, convert_to_tensor=True).to(device).cpu().numpy()
            
            # Calculate t-SNE
            perplexity = min(30, max(5, len(keyword_embeddings) // 3))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            reduced_embeddings = tsne.fit_transform(keyword_embeddings)
            
            # Smart text positioning to avoid overlaps
            x_coords = reduced_embeddings[:, 0]
            y_coords = reduced_embeddings[:, 1]
            x_coords_adjusted, y_coords_adjusted = adjust_text_positions(x_coords, y_coords, GLOBAL_KEYWORDS)
            
            # Cache the results including adjusted positions
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
        
        # Simple hover text for all keywords
        hover_texts = GLOBAL_KEYWORDS
        
        # Use default blue color for all keywords in initial load
        keyword_colors = ['#2196F3'] * len(GLOBAL_KEYWORDS)
        

        
        # Create simple scatter plot with text labels
        fig = {
            'data': [{
                'x': x_coords_adjusted,
                'y': y_coords_adjusted,
                'mode': 'markers+text',  # Use markers+text mode for clickable elements
                'type': 'scatter',
                'marker': {
                    'size': 1,  # Very small markers (invisible)
                    'color': keyword_colors,
                    'opacity': 0.0  # Make markers invisible
                },
                'text': GLOBAL_KEYWORDS,  # Display keyword text
                'textfont': {
                    'size': 10,  # Smaller font to reduce overlap
                    'color': keyword_colors  # Dynamic colors based on group assignment
                },
                'textposition': 'middle center',
                'hovertext': hover_texts,
                'hoverinfo': 'text',
                'customdata': GLOBAL_KEYWORDS,  # For click events
                'hovertemplate': '<b>%{hovertext}</b><extra></extra>',
                'opacity': 1.0  # Prevent any fading effects
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
                'clickmode': 'event',  # Remove 'select' to prevent selection-based fading
                'dragmode': 'pan',
                'showlegend': False,
                'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60},  # Larger margins
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': {'size': 10}  # Smaller overall font size
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

# Add a clientside callback for fast color updates without server computation
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



# Add initial callback for documents 2D visualization
print("    DEBUG: ==========================================")
print("    DEBUG: REGISTERING CALLBACK: update_documents_2d_plot_initial")
print("    DEBUG: ==========================================")
print("    DEBUG: Output: documents-2d-plot.figure")
print("    DEBUG: Input: main-visualization-area.children")
print("    DEBUG: States: display-mode.data, training-figures.data")
print("    DEBUG: prevent_initial_call: True")

@app.callback(
    Output('documents-2d-plot', 'figure'),
    Input('main-visualization-area', 'children'),  # Trigger when layout changes
    [State('display-mode', 'data'),
     State('training-figures', 'data')],  # Add training-figures to check if we're in training mode
    prevent_initial_call=True
)
def update_documents_2d_plot_initial(layout_children, display_mode, training_figures):
    """Initial load of documents 2D visualization - show all documents"""
    global df
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: update_documents_2d_plot_initial CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   layout_children type: {type(layout_children)}")
    print(f"    DEBUG:   layout_children length: {len(layout_children) if layout_children else 'None'}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:   training_figures: {training_figures}")
    print(f"    DEBUG:     PARAMETER TYPES:")
    print(f"    DEBUG:   display_mode type: {type(display_mode)}")
    print(f"    DEBUG:   training_figures type: {type(training_figures)}")
    
    print(f"    DEBUG:     SAFETY CHECKS:")
    
    # Only process if we're in keywords mode
    if display_mode != "keywords":
        print(f"    DEBUG:         NOT IN KEYWORDS MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'keywords'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      KEYWORDS MODE CONFIRMED")
    
    # Additional safety check: ensure we're not in training mode
    # This prevents any accidental updates when the component doesn't exist
    if display_mode == "training":
        print(f"    DEBUG:         TRAINING MODE DETECTED:")
        print(f"    DEBUG:   This should prevent the 'nonexistent object' error")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      NOT IN TRAINING MODE")
    
    # Extra safety: check if we're in any mode other than keywords
    # This catches any edge cases where display_mode might be None or unexpected values
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
        # Calculate document embeddings
        print("    Initial documents 2D visualization calculation...")
        all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
        
        # Truncate long texts to prevent token length issues
        print("    Truncating long texts to fit within model limits...")
        truncated_articles = [truncate_text_for_model(text, max_length=500) for text in all_articles_text]
        
        # Calculate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(truncated_articles), batch_size):
            batch_texts = truncated_articles[i:i + batch_size]
            print(f"    Processing batch {i//batch_size + 1}/{(len(truncated_articles) + batch_size - 1)//batch_size}")
            
            # Use safe encoding function with better error handling
            batch_embeddings = safe_encode_batch(batch_texts, embedding_model_kw, device)
            all_embeddings.extend(batch_embeddings)
        
        document_embeddings = np.array(all_embeddings)
        
        # Calculate TSNE for documents
        print("    Calculating TSNE for initial documents visualization...")
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        document_2d = tsne.fit_transform(document_embeddings)
        document_2d = document_2d.tolist()
        
        # Check if document_2d length matches df length
        print(f"    Initial document_2d length: {len(document_2d)}")
        print(f"    df length: {len(df)}")
        
        # Ensure document_2d and df have the same length
        if len(document_2d) != len(df):
            print(f"    WARNING: Initial length mismatch! Adjusting...")
            # If document_2d is shorter, pad with zeros
            if len(document_2d) < len(df):
                padding_needed = len(df) - len(document_2d)
                print(f"    Padding initial document_2d with {padding_needed} zero points")
                for _ in range(padding_needed):
                    document_2d.append([0.0, 0.0])
            # If document_2d is longer, truncate
            elif len(document_2d) > len(df):
                print(f"    Truncating initial document_2d from {len(document_2d)} to {len(df)}")
                document_2d = document_2d[:len(df)]
        
        print(f"    Final initial document_2d length: {len(document_2d)}")
        
        # Show all documents in one color - [EMOJI]
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
        
        # [EMOJI]
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

print("    DEBUG: ==========================================")
print("    DEBUG: REGISTERING CALLBACK: update_documents_2d_plot")
print("    DEBUG: ==========================================")
print("    DEBUG: Outputs: documents-2d-plot.figure, highlighted-indices.data")
print("    DEBUG: Inputs: selected-keyword.data, selected-group.data, selected-article.data")
print("    DEBUG: States: group-order.data, display-mode.data")
print("    DEBUG: allow_duplicate: True (for documents-2d-plot.figure)")
print("    DEBUG: prevent_initial_call: True")
print("    DEBUG:      WARNING: This callback outputs to documents-2d-plot")
print("    DEBUG:      WARNING: This callback should NEVER run in training mode")

@app.callback(
    [Output('documents-2d-plot', 'figure', allow_duplicate=True),
     Output('highlighted-indices', 'data', allow_duplicate=True)],
    [Input('selected-keyword', 'data'),
     Input('selected-group', 'data'),  # Also update when group is selected
     Input('selected-article', 'data')],  # Keep as Input for keywords mode highlighting
    State('group-order', 'data'),  # Add group_order as State parameter
    State('display-mode', 'data'),
    prevent_initial_call=True
)
def update_documents_2d_plot(selected_keyword, selected_group, selected_article, group_order, display_mode):
    """Update documents 2D visualization chart"""
    global df, _DOCUMENTS_2D_CACHE
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: update_documents_2d_plot CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:   selected_keyword: {selected_keyword}")
    print(f"    DEBUG:   selected_group: {selected_group}")
    print(f"    DEBUG:   selected_article: {selected_article}")
    print(f"    DEBUG:   group_order: {group_order}")
    print(f"    DEBUG:     PARAMETER TYPES:")
    print(f"    DEBUG:   display_mode type: {type(display_mode)}")
    print(f"    DEBUG:   selected_keyword type: {type(selected_keyword)}")
    print(f"    DEBUG:   selected_group type: {type(selected_group)}")
    print(f"    DEBUG:   selected_article type: {type(selected_article)}")
    
    print(f"    DEBUG:     SAFETY CHECKS:")
    
    # Only process if we're in keywords mode
    if display_mode != "keywords":
        print(f"    DEBUG:         NOT IN KEYWORDS MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'keywords'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      KEYWORDS MODE CONFIRMED")
    
    # Additional safety check: ensure we're not in training mode
    # This prevents any accidental updates when the component doesn't exist
    if display_mode == "training":
        print(f"    DEBUG:         TRAINING MODE DETECTED:")
        print(f"    DEBUG:   This should prevent the 'nonexistent object' error")
        print(f"    DEBUG:   Preventing update")
        print(f"    DEBUG:   This callback should NEVER run in training mode")
        raise PreventUpdate
    
    print(f"    DEBUG:      NOT IN TRAINING MODE")
    
    # Extra safety: check if we're in any mode other than keywords
    # This catches any edge cases where display_mode might be None or unexpected values
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
    
    # Create cache key for documents 2D visualization
    cache_key = None
    if selected_keyword:
        cache_key = f"docs_keyword:{selected_keyword}"
        print(f"    DEBUG: Created cache key for keyword: {cache_key}")
    elif selected_group and group_order:
        # For groups, create cache key based on group keywords
        for group_name, keywords in group_order.items():
            if group_name == selected_group:
                # Sort keywords for consistent cache key
                cache_key = f"docs_group:{group_name}:{':'.join(sorted(keywords))}"
                print(f"    DEBUG: Created cache key for group: {cache_key}")
                break
    else:
        cache_key = "docs_default"
        print(f"    DEBUG: Using default cache key: {cache_key}")
    
    # Add selected article to cache key
    if selected_article is not None:
        cache_key = f"{cache_key}_article:{selected_article}"
        print(f"    DEBUG: Added article to cache key: {cache_key}")
    
    print(f"    DEBUG: Final cache key: {cache_key}")
    print(f"    DEBUG: Cache contains key: {cache_key in _DOCUMENTS_2D_CACHE}")
    
    # Check cache first
    if cache_key and cache_key in _DOCUMENTS_2D_CACHE:
        print(f"    DEBUG: Using cached documents 2D plot for: {cache_key}")
        cached_fig = _DOCUMENTS_2D_CACHE[cache_key]
        # For cached results, we need to extract highlighted indices
        highlighted_indices = []
        if selected_keyword or selected_group or selected_article is not None:
            # Try to extract indices from the cached figure
            try:
                for trace in cached_fig.get('data', []):
                    if trace.get('name') in ['Keyword/Group matches', 'Selected Article']:
                        # Extract indices from customdata if available
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
        # Use pre-computed document embeddings and t-SNE results
        print("    Using pre-computed document embeddings and t-SNE...")
        
        if _GLOBAL_DOCUMENT_EMBEDDINGS_READY:
            document_embeddings = _GLOBAL_DOCUMENT_EMBEDDINGS
            document_2d = _GLOBAL_DOCUMENT_TSNE.tolist()
            print(f"    Using cached embeddings, shape: {document_embeddings.shape}")
            print(f"    Using cached t-SNE, shape: {_GLOBAL_DOCUMENT_TSNE.shape}")
            
            # Check if cached document_2d length matches df length
            print(f"    Cached document_2d length: {len(document_2d)}")
            print(f"    df length: {len(df)}")
            
            # Ensure document_2d and df have the same length
            if len(document_2d) != len(df):
                print(f"    WARNING: Cached length mismatch! Adjusting...")
                # If document_2d is shorter, pad with zeros
                if len(document_2d) < len(df):
                    padding_needed = len(df) - len(document_2d)
                    print(f"    Padding cached document_2d with {padding_needed} zero points")
                    for _ in range(padding_needed):
                        document_2d.append([0.0, 0.0])
                # If document_2d is longer, truncate
                elif len(document_2d) > len(df):
                    print(f"    Truncating cached document_2d from {len(document_2d)} to {len(df)}")
                    document_2d = document_2d[:len(df)]
            
            print(f"    Final cached document_2d length: {len(document_2d)}")
        else:
            # Fallback to on-demand computation if pre-computation failed
            print("    Pre-computed embeddings not available, computing on-demand...")
        print(f"    df shape: {df.shape}")
        all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
        print(f"    Number of articles: {len(all_articles_text)}")
        
        # Truncate long texts to prevent token length issues
        print("    Truncating long texts to fit within model limits...")
        truncated_articles = [truncate_text_for_model(text, max_length=500) for text in all_articles_text]
        
        # Calculate embeddings in batches
        batch_size = 64 if device == "cpu" else 128
        all_embeddings = []
        
        for i in range(0, len(truncated_articles), batch_size):
            batch_texts = truncated_articles[i:i + batch_size]
            print(f"    Processing batch {i//batch_size + 1} for documents 2D visualization")
            
            # Use safe encoding function with better error handling
            batch_embeddings = safe_encode_batch(batch_texts, embedding_model_kw, device)
            all_embeddings.extend(batch_embeddings)
        
        document_embeddings = np.array(all_embeddings)
        
        # Calculate TSNE for documents
        print("    Calculating TSNE for documents...")
        print(f"    Embeddings shape: {document_embeddings.shape}")
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        print(f"    Perplexity: {perplexity}")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_jobs=-1)
        document_2d = tsne.fit_transform(document_embeddings)
        print(f"    TSNE result shape: {document_2d.shape}")
        print(f"    TSNE result type: {type(document_2d)}")
        print(f"    TSNE result dtype: {document_2d.dtype}")
        # Convert to list for Plotly compatibility
        document_2d = document_2d.tolist()
        
        # Check if document_2d length matches df length
        print(f"    document_2d length: {len(document_2d)}")
        print(f"    df length: {len(df)}")
        print(f"    all_articles_text length: {len(all_articles_text)}")
        
        # Ensure document_2d and df have the same length
        if len(document_2d) != len(df):
            print(f"    WARNING: Length mismatch! Adjusting...")
            # If document_2d is shorter, pad with zeros
            if len(document_2d) < len(df):
                padding_needed = len(df) - len(document_2d)
                print(f"    Padding document_2d with {padding_needed} zero points")
                for _ in range(padding_needed):
                    document_2d.append([0.0, 0.0])
            # If document_2d is longer, truncate
            elif len(document_2d) > len(df):
                print(f"    Truncating document_2d from {len(document_2d)} to {len(df)}")
                document_2d = document_2d[:len(df)]
        
        print(f"    Final document_2d length: {len(document_2d)}")
        
        # Determine which documents to highlight
        highlight_mask = []
        highlight_reason = ""
        
        if selected_keyword:
            # Highlight documents containing the selected keyword
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
            # Highlight documents containing any keywords from the selected group
            group_keywords = []
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    group_keywords = keywords
                    break
            
            print(f"Selected group '{selected_group}' keywords: {group_keywords}")
            
            for i in range(len(df)):
                text = str(df.iloc[i, 1]).lower()
                # Check if any keyword from the group is in the document
                contains_group_keyword = any(keyword.lower() in text for keyword in group_keywords)
                highlight_mask.append(contains_group_keyword)
            
            highlight_reason = f"Documents containing keywords from group '{selected_group}'"
        
        else:
            # No highlighting - show all documents in default color
            highlight_mask = [False] * len(df)
            highlight_reason = ""
        
        # Create selected article mask
        selected_article_mask = [False] * len(df)
        if selected_article is not None and selected_article < len(df):
            selected_article_mask[selected_article] = True
        
        # Create traces
        traces = []
        
        # Get indices for different types of documents
        keyword_group_indices = np.where(np.array(highlight_mask))[0]
        selected_article_indices = np.where(np.array(selected_article_mask))[0]
        
        # Create a combined mask for all highlighted documents
        all_highlighted = np.logical_or(np.array(highlight_mask), np.array(selected_article_mask))
        other_indices = np.where(~all_highlighted)[0]
        
                # Add trace for keyword/group highlighted documents - [EMOJI]
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
        
        # Add trace for selected article (highest priority, will overlay on top)
        if len(selected_article_indices) > 0:
            traces.append({
                'x': [document_2d[i][0] for i in selected_article_indices],
                'y': [document_2d[i][1] for i in selected_article_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Selected Article',
                'marker': {
                    'size': 20,
                    'color': '#FF0000',  # Red color for selected article
                    'symbol': 'star',
                    'line': {'width': 3, 'color': 'white'}
                },
                'text': [f'Doc {i+1}' for i in selected_article_indices],
                'customdata': [[i] for i in selected_article_indices],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
        
        # Add trace for other documents - [EMOJI]
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
        # If no traces created, show all documents in one color - [EMOJI]
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
        
        # Create title
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
        
        # Collect highlighted indices for training plots
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
        
        # Cache the result for future use
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
    """Handle chart click events, select keyword for highlighting documents"""
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
        # Get clicked keyword
        clicked_keyword = click_data['points'][0]['customdata']
        print(f"Clicked keyword: {clicked_keyword}")
        
        # If a group is selected, add keyword to that group
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
            
            # In training mode, don't update selected-keyword to avoid triggering documents-2d-plot
            if display_mode == "training":
                print(f"    Training mode: not updating selected-keyword to avoid documents-2d-plot error")
                return new_data, dash.no_update
            else:
                return new_data, clicked_keyword  # Return both group data and selected keyword
        else:
            # No group selected, just select the keyword for highlighting
            print(f"Selected keyword for highlighting: {clicked_keyword}")
            
            # In training mode, don't update selected-keyword to avoid triggering documents-2d-plot
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
    """Handle Train button click event"""
    print("=" * 60)
    print("TRAIN BUTTON CALLBACK TRIGGERED!")
    print("=" * 60)
    print(f"Train button clicked, n_clicks: {n_clicks}")
    print(f"Group order data: {group_order}")
    print(f"Current time: {__import__('datetime').datetime.now()}")
    
    # Force print to ensure it shows up
    import sys
    sys.stdout.flush()
    
    if not n_clicks or n_clicks == 0:
        print("No clicks, preventing update")
        raise PreventUpdate
    
    # Define button styles
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
        "backgroundColor": "#FF9800",  # Orange color for training
        "color": "white",
        "border": "none",
        "borderRadius": "5px",
        "cursor": "not-allowed",
        "animation": "pulse 1.5s infinite"
    }
    
    if not group_order:
        print("No group data available")
        # Return empty figures and keep training output hidden
        empty_fig = {
            'data': [],
            'layout': {
                'title': 'No group data available for training',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }
        return {"display": "none"}, empty_fig, empty_fig, {"display": "none"}, "Train", normal_style, False
    
    try:
        print("Starting training process...")
        print(f"    DEBUG: Group order data: {group_order}")
        print(f"    DEBUG: Final list path: {final_list_path}")
        
        # Save current group data to final_list.json
        print("Saving group data to final_list.json...")
        with open(final_list_path, "w", encoding="utf-8") as f:
            json.dump(group_order, f, indent=4, ensure_ascii=False)
        print(f"Group data saved to {final_list_path}")
        
        # Verify the file was saved correctly
        if os.path.exists(final_list_path):
            with open(final_list_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            print(f"    DEBUG: Verified saved data: {saved_data}")
        else:
            print(f"    ERROR: Failed to save group data to {final_list_path}")
            raise FileNotFoundError(f"Could not save group data to {final_list_path}")
        
        # Show training in progress state immediately
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
        
        # Run training
        print("Running training function...")
        try:
            fig_before, fig_after = run_training()
            print("Training completed successfully!")
        except Exception as e:
            print(f"    ERROR: Training failed with exception: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error state
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
            return {"display": "block"}, error_fig, error_fig, {"display": "block"}, "Train (Failed)", normal_style, False
        
        # Save group information for model loading
        group_info_path = "test_results/training_group_info.json"
        with open(group_info_path, "w", encoding="utf-8") as f:
            json.dump(group_order, f, indent=4, ensure_ascii=False)
        print(f"Group information saved to {group_info_path} for model loading")
        
        # DEBUG: [EMOJI]
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
        
        # Training completed successfully - reset button to normal state
        completed_style = {
            "margin-top": "20px",
            "padding": "10px 20px",
            "fontSize": "16px",
            "backgroundColor": "#2E7D32",  # Darker green for completed
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer"
        }
        
        # Show switch button after training completion
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
            "display": "block"  # Show the switch button
        }
        
        # [EMOJI] Figure [EMOJI]
        print(f"    DEBUG: Converting figures to dict by manual extraction")
        
        import numpy as np
        
        def fig_to_serializable_dict(fig):
            """[EMOJI] Plotly Figure [EMOJI]"""
            result = {
                'data': [],
                'layout': {}
            }
            
            # [EMOJI] layout
            if hasattr(fig, 'layout'):
                result['layout'] = fig.layout.to_plotly_json() if hasattr(fig.layout, 'to_plotly_json') else {}
            
            # [EMOJI] trace
            for trace in fig.data:
                trace_dict = {}
                
                # [EMOJI] trace [EMOJI]
                for attr in ['x', 'y', 'mode', 'type', 'name', 'text', 'textposition', 'textfont', 'customdata', 'hovertemplate', 'hovertext']:
                    if hasattr(trace, attr):
                        val = getattr(trace, attr)
                        if val is not None:
                            # [EMOJI] numpy [EMOJI]
                            if hasattr(val, 'tolist'):
                                trace_dict[attr] = val.tolist()
                            elif hasattr(val, '__iter__') and not isinstance(val, str):
                                trace_dict[attr] = list(val)
                            else:
                                trace_dict[attr] = val
                
                # [EMOJI] marker
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
                
                # DEBUG
                trace_name = trace_dict.get('name', 'Unknown')
                x_len = len(trace_dict.get('x', []))
                print(f"    DEBUG: Extracted trace '{trace_name}': {x_len} points")
                
                result['data'].append(trace_dict)
            
            return result
        
        # [EMOJI]
        fig_before_dict = fig_to_serializable_dict(fig_before)
        fig_after_dict = fig_to_serializable_dict(fig_after)
        
        # DEBUG: [EMOJI]
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
        
        # DEBUG: [EMOJI]
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
        import traceback
        traceback.print_exc()
        
        # Return error figures
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
        
        # Training failed - show error state
        error_style = {
            "margin-top": "20px",
            "padding": "10px 20px",
            "fontSize": "16px",
            "backgroundColor": "#F44336",  # Red for error
            "color": "white",
            "border": "none",
            "borderRadius": "5px",
            "cursor": "pointer"
        }
        
        # Hide switch button on training failure
        switch_button_style = {"display": "none"}
        
        return "Training Failed", error_style, False, switch_button_style, "keywords", {"before": None, "after": None}

# Add a separate callback to handle button state changes immediately when clicked
@app.callback(
    [Output("train-btn", "children", allow_duplicate=True),
     Output("train-btn", "style", allow_duplicate=True),
     Output("train-btn", "disabled", allow_duplicate=True)],
    Input("train-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_train_button_immediately(n_clicks):
    """Update button state immediately when clicked to show training status"""
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    # Show training state immediately
    training_style = {
        "margin-top": "20px",
        "padding": "10px 20px",
        "fontSize": "16px",
        "backgroundColor": "#FF9800",  # Orange color for training
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
    """Display article content when clicking on training plots (before/after)"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    # Determine which plot was clicked
    click_data = None
    if ctx.triggered[0]['prop_id'] == 'plot-before.clickData':
        click_data = click_data_before
    elif ctx.triggered[0]['prop_id'] == 'plot-after.clickData':
        click_data = click_data_after
    
    if not click_data:
        raise PreventUpdate
    
    try:
        # Get article index from click data
        article_index = click_data['points'][0]['customdata'][0]
        
        # Load article content
        global df
        if 'df' not in globals():
            df = pd.read_csv(csv_path)
        
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
            
            # Return both content and article index for training mode highlighting
            # In training mode, we add the clicked article to highlighted indices for special highlighting
            return content, [article_index]
        else:
            return html.P("Article not found", style={"color": "red"}), []
    
    except Exception as e:
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"}), []

# Load previous model callback removed - no longer needed

@app.callback(
    [Output("display-mode", "data", allow_duplicate=True),
     Output("switch-view-btn", "children")],
    Input("switch-view-btn", "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def switch_display_mode(n_clicks, current_mode):
    """Switch between keywords view and training view"""
    if not n_clicks or n_clicks == 0:
        raise PreventUpdate
    
    if current_mode == "keywords":
        new_mode = "training"
        button_text = "Switch to Keywords View"
    elif current_mode == "training":
        new_mode = "keywords"
        button_text = "Switch to Training View"
    else:
        # [EMOJI] keywords
        new_mode = "keywords"
        button_text = "Switch to Training View"
    
    print(f"Switching display mode from {current_mode} to {new_mode}")
    return new_mode, button_text

# [EMOJI] switch-view-btn [EMOJI] finetune [EMOJI]
@app.callback(
    [Output("switch-view-btn", "style", allow_duplicate=True),
     Output("switch-view-btn", "children", allow_duplicate=True)],
    Input("display-mode", "data"),
    prevent_initial_call=True
)
def control_switch_view_btn_visibility(display_mode):
    base_style = {
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
    }
    
    # [EMOJI] finetune [EMOJI]
    if display_mode == "finetune":
        base_style["display"] = "none"
        button_text = "Switch to Training View"  # [EMOJI]
    elif display_mode == "training":
        base_style["display"] = "block"
        button_text = "Switch to Keywords View"  # [EMOJI] training[EMOJI] keywords [EMOJI]
    elif display_mode == "keywords":
        base_style["display"] = "block"
        button_text = "Switch to Training View"  # [EMOJI] keywords[EMOJI] training [EMOJI]
    else:
        base_style["display"] = "block"
        button_text = "Switch to Training View"  # [EMOJI]
    
    return base_style, button_text

# [EMOJI] Finetune [EMOJI]/[EMOJI]
@app.callback(
    Output("switch-finetune-btn", "style"),
    [Input("display-mode", "data"), Input("training-figures", "data")]
)
def show_switch_finetune_btn(display_mode, training_figures):
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
        # [EMOJI] training [EMOJI] finetune [EMOJI]
        if display_mode in ("training", "finetune") and has_after:
            base_style["display"] = "block"
        else:
            base_style["display"] = "none"
    except Exception:
        base_style["display"] = "none"
    return base_style
            
# Save model callback removed - model is automatically saved after training

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
    """Dynamically update the main visualization area based on display mode"""
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: update_main_visualization_area CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:   display_mode type: {type(display_mode)}")
    print(f"    DEBUG:   training_figures: {training_figures is not None}")
    print(f"    DEBUG:   training_figures type: {type(training_figures)}")
    if training_figures:
        print(f"    DEBUG:   training_figures keys: {list(training_figures.keys()) if isinstance(training_figures, dict) else 'not dict'}")
    
    print(f"    DEBUG:     CALLBACK LOGIC:")
    print(f"    DEBUG:   About to check if display_mode == 'training'")
    print(f"    DEBUG:   display_mode == 'training': {display_mode == 'training'}")
    print(f"    DEBUG:   display_mode == 'keywords': {display_mode == 'keywords'}")
    
    if display_mode == "training":
        print(f"    DEBUG:      ENTERING TRAINING MODE LAYOUT")
        print(f"    DEBUG:   Will return training plots and hide keywords panels")
        print(f"    DEBUG:   training_group_style will be: {{'display': 'flex', 'marginBottom': '30px'}}")
        print(f"    DEBUG:   keywords_group_style will be: {{'display': 'none', 'marginBottom': '30px'}}")
        # Show training plots and show training group management area
        if training_figures:
            fig_before = training_figures.get("before", {})
            fig_after = training_figures.get("after", {})
            print(f"    Using existing training figures")
        else:
            print(f"    No training figures available, using placeholders")
            # Show placeholder figures when no training data is available
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
        
        # Show training group management area
        training_group_style = {'display': 'flex', 'marginBottom': '30px'}
        
        # Force trigger training mode callbacks by updating their inputs
        # This ensures the training interfaces are populated when switching to training mode
        
        print(f"    DEBUG:     RETURNING TRAINING MODE LAYOUT:")
        print(f"    DEBUG:   - main-visualization-area: training plots")
        print(f"    DEBUG:   - training-group-management-area: {{'display': 'flex'}}")
        print(f"    DEBUG:   - keywords-group-management-area: {{'display': 'none'}}")
        
        # training [EMOJI] before/after [EMOJI] training [EMOJI] finetune [EMOJI]
        return [
            # Left: Before Training
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
            
            # Right: After Training
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
        
        # Show finetune plot (using after training figure) and show finetune group management area
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
        
        # Show finetune group management area
        finetune_group_style = {'display': 'flex', 'marginBottom': '30px'}
        
        print(f"    DEBUG:     RETURNING FINETUNE MODE LAYOUT:")
        print(f"    DEBUG:   - main-visualization-area: finetune plot")
        print(f"    DEBUG:   - finetune-group-management-area: {{'display': 'flex'}}")
        print(f"    DEBUG:   - training-group-management-area: {{'display': 'none'}}")
        print(f"    DEBUG:   - keywords-group-management-area: {{'display': 'none'}}")
        
        # finetune [EMOJI] keywords/training [EMOJI] finetune [EMOJI]
        # keywords [EMOJI] keywords [EMOJI] keywords [EMOJI] training/finetune [EMOJI]
        return [
            # Finetune plot - maintain aspect ratio like keywords plot
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
                    style={'height': '800px'},  # Increased height for better aspect ratio
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], className="modern-card", style={
                'width': '100%',
                'minHeight': '850px',  # Set minimum container height
                'padding': '20px',
                'margin': '0 auto',
                'display': 'block'
            })
        ], {'display': 'none', 'marginBottom': '30px'}, {'display': 'none', 'marginBottom': '30px'}, finetune_group_style
    else:
        print(f"    DEBUG:      ENTERING KEYWORDS MODE LAYOUT")
        print(f"    DEBUG:   Will return keywords plots and show keywords panels")
        print(f"    DEBUG:   training_group_style will be: {{'display': 'none', 'marginBottom': '30px'}}")
        print(f"    DEBUG:   keywords_group_style will be: {{'display': 'flex', 'marginBottom': '30px'}}")
        # In keywords mode, restore the original keywords view layout and hide training group management
        training_group_style = {'display': 'none', 'marginBottom': '30px'}
        keywords_group_style = {'display': 'flex', 'marginBottom': '30px'}
        
        print(f"    DEBUG:     RETURNING KEYWORDS MODE LAYOUT:")
        print(f"    DEBUG:   - main-visualization-area: keywords plots")
        print(f"    DEBUG:   - training-group-management-area: {{'display': 'none'}}")
        print(f"    DEBUG:   - keywords-group-management-area: {{'display': 'flex'}}")
        
        return [
            # Left: Keywords 2D Visualization
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
            
            # Right: Documents 2D Visualization
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

# Add callback for training mode highlighting
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
    """Update highlighted indices for training mode - following keywords mode logic"""
    global df
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: update_training_highlights CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   selected_keyword: {selected_keyword}")
    print(f"    DEBUG:   selected_group: {selected_group}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:   group_order: {group_order}")
    print(f"    DEBUG:   training_figures: {training_figures is not None}")
    
    # Only process if we're in training mode
    if display_mode != "training":
        print(f"    DEBUG:         NOT IN TRAINING MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'training'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      TRAINING MODE CONFIRMED")
    
    # Check if we have data
    if 'df' not in globals() or not training_figures:
        print(f"    DEBUG:         MISSING DATA OR TRAINING FIGURES:")
        print(f"    DEBUG:   df in globals: {'df' in globals()}")
        print(f"    DEBUG:   training_figures: {training_figures}")
        print(f"    DEBUG:   Returning empty")
        return {"type": "none", "indices": []}
    
    print(f"    DEBUG:      DATA AND TRAINING FIGURES AVAILABLE")
    
    # Following keywords mode logic: keyword selection has priority over group selection
    # But if no keyword is selected, then group selection should work
    
    if selected_keyword:
        print(f"    DEBUG:     KEYWORD SELECTION:")
        print(f"    DEBUG:   Processing keyword: {selected_keyword}")
        
        # Find documents containing the selected keyword
        keyword_indices = []
        for i, text in enumerate(df.iloc[:, 1]):
            if selected_keyword.lower() in str(text).lower():
                keyword_indices.append(i)
        
        print(f"    DEBUG:   Found {len(keyword_indices)} documents for keyword '{selected_keyword}'")
        print(f"    DEBUG:   Document indices: {keyword_indices}")
        
        return {"type": "keyword", "indices": keyword_indices, "keyword": selected_keyword}
        
    elif selected_group and group_order:
        print(f"    DEBUG:     GROUP SELECTION:")
        print(f"    DEBUG:   Processing group: {selected_group}")
        print(f"    DEBUG:   Full group_order: {group_order}")
        
        # Find documents containing any keyword in the selected group
        if selected_group in group_order:
            group_keywords = group_order[selected_group]
            print(f"    DEBUG:   Group keywords: {group_keywords}")
            print(f"    DEBUG:   Group keywords type: {type(group_keywords)}")
            print(f"    DEBUG:   Group keywords length: {len(group_keywords) if group_keywords else 0}")
            
            group_indices = []
            for i, text in enumerate(df.iloc[:, 1]):
                text_lower = str(text).lower()
                
                # Special debug for document 57 when processing Group 1
                if i == 56 and selected_group == "Group 1":  # Document 57 (0-indexed)
                    print(f"    DEBUG:     DOCUMENT 57 ANALYSIS FOR GROUP 1:")
                    print(f"    DEBUG:   Document index: {i+1} (1-indexed)")
                    print(f"    DEBUG:   Text preview: {str(text)[:300]}...")
                    print(f"    DEBUG:   Text length: {len(str(text))}")
                    print(f"    DEBUG:   Group keywords: {group_keywords}")
                    for kw in group_keywords:
                        contains_kw = kw.lower() in text_lower
                        print(f"    DEBUG:   Contains '{kw}': {contains_kw}")
                        if contains_kw:
                            # Find position of keyword in text
                            pos = text_lower.find(kw.lower())
                            context = text_lower[max(0, pos-50):pos+len(kw)+50]
                            print(f"    DEBUG:     Context: ...{context}...")
                    print(f"    DEBUG:   Overall match: {any(keyword.lower() in text_lower for keyword in group_keywords)}")
                
                # Check if any keyword from the group is in the document
                if any(keyword.lower() in text_lower for keyword in group_keywords):
                        group_indices.append(i)
            
            print(f"    DEBUG:   Found {len(group_indices)} documents for group '{selected_group}'")
            print(f"    DEBUG:   Document indices: {group_indices}")
            
            return {"type": "group", "indices": group_indices, "group": selected_group}
        else:
            print(f"    DEBUG:           Group '{selected_group}' not found in group_order")
            return {"type": "group", "indices": [], "group": selected_group}
    
    # If neither keyword nor group is selected
    print(f"    DEBUG:     NO SELECTION:")
    print(f"    DEBUG:   No keyword or group selected, returning empty")
    return {"type": "none", "indices": []}

# Add callback for updating training plots with highlights
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
    """Update training plots with highlighted indices - following keywords mode logic"""
    global df
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: update_training_plots_with_highlights CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   highlighted_indices: {highlighted_indices}")
    print(f"    DEBUG:   training_selected_article: {training_selected_article}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:   training_figures: {training_figures is not None}")
    print(f"    DEBUG:   group_order: {group_order}")
    
    # Only process if we're in training mode
    if display_mode != "training":
        print(f"    DEBUG:         NOT IN TRAINING MODE:")
        print(f"    DEBUG:   display_mode '{display_mode}' != 'training'")
        print(f"    DEBUG:   Preventing update")
        raise PreventUpdate
    
    print(f"    DEBUG:      TRAINING MODE CONFIRMED")
    
    # Check if we have training figures
    if not training_figures:
        print(f"    DEBUG:         NO TRAINING FIGURES:")
        print(f"    DEBUG:   Returning empty figures")
        return {}, {}
    
    print(f"    DEBUG:      TRAINING FIGURES AVAILABLE")
    
    # Get original training figures
    fig_before = training_figures.get("before", {})
    fig_after = training_figures.get("after", {})
    
    # Initialize highlight variables
    keyword_group_highlights = []
    selected_article_highlight = None
    
    # Process highlights following keywords mode logic
    if isinstance(highlighted_indices, dict) and 'type' in highlighted_indices:
        highlight_type = highlighted_indices.get('type')
        highlight_indices = highlighted_indices.get('indices', [])
        
        print(f"    DEBUG:     PROCESSING HIGHLIGHTS:")
        print(f"    DEBUG:   Highlight type: {highlight_type}")
        print(f"    DEBUG:   Highlight indices: {highlight_indices}")
        
        if highlight_type == "group":
            # Group highlights - show all documents in the group
            keyword_group_highlights = highlight_indices
            print(f"    DEBUG:   Group highlights: {keyword_group_highlights}")
            
        elif highlight_type == "keyword":
            # Keyword highlights - show only documents with this keyword
            keyword_group_highlights = highlight_indices
            print(f"    DEBUG:   Keyword highlights: {keyword_group_highlights}")
            
        elif highlight_type == "none":
            # No highlights
            keyword_group_highlights = []
            print(f"    DEBUG:   No highlights")
    
    # Process article selection (can coexist with group/keyword highlights)
    if training_selected_article is not None and training_selected_article < len(df):
        selected_article_highlight = training_selected_article
        print(f"    DEBUG:     ARTICLE SELECTION:")
        print(f"    DEBUG:   Selected article: {training_selected_article}")
        
        # If we have group/keyword highlights, check if this article is part of them
        if keyword_group_highlights and training_selected_article not in keyword_group_highlights:
            print(f"    DEBUG:   Article {training_selected_article} is NOT in current highlights")
            print(f"    DEBUG:   This will show both highlights and article")
        elif keyword_group_highlights and training_selected_article in keyword_group_highlights:
            print(f"    DEBUG:   Article {training_selected_article} IS in current highlights")
            print(f"    DEBUG:   This will show highlights with article highlighted")
        else:
            print(f"    DEBUG:   No current highlights, only showing article")
    
    print(f"    DEBUG:     FINAL HIGHLIGHT STATE:")
    print(f"    DEBUG:   Keyword/Group highlights: {keyword_group_highlights}")
    print(f"    DEBUG:   Selected article highlight: {selected_article_highlight}")
    
    # Apply highlights to both plots
    updated_fig_before = apply_highlights_to_training_plot(fig_before, keyword_group_highlights, selected_article_highlight, "before")
    updated_fig_after = apply_highlights_to_training_plot(fig_after, keyword_group_highlights, selected_article_highlight, "after")
    
    return updated_fig_before, updated_fig_after

def apply_highlights_to_training_plot(fig, keyword_group_highlights, selected_article_highlight, plot_name):
    """Apply highlights to a training plot - keeps base layer + centers, adds highlights on top"""
    if not fig or 'data' not in fig:
        return fig
    
    print(f"    DEBUG:     APPLYING HIGHLIGHTS TO {plot_name.upper()} PLOT:")
    print(f"    DEBUG:   Keyword/Group highlights: {keyword_group_highlights}")
    print(f"    DEBUG:   Selected article: {selected_article_highlight}")
    
    # Create a copy of the figure
    updated_fig = fig.copy()
    
    if not updated_fig['data']:
        return updated_fig
    
    # [EMOJI]
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
        
        # [EMOJI]diamond[EMOJI] [EMOJI] [EMOJI] 'Center'[EMOJI]
        if symbol == 'diamond' or 'Center' in trace_name:
            center_traces.append(trace)
            print(f"    DEBUG:       → Keeping as center trace")
        # [EMOJI]center trace[EMOJI] 'All Documents'[EMOJI]
        elif main_trace is None and x_len > 10 and symbol != 'star':
            main_trace = trace
            print(f"    DEBUG:       → Keeping as main document trace")
    
    # [EMOJI]
    if main_trace:
        traces.append(main_trace)
    else:
        # [EMOJI]trace[EMOJI]
        print(f"    DEBUG:           No main trace found, returning original figure")
        return fig
    
    # [EMOJI]
    traces.extend(center_traces)
    print(f"    DEBUG:   Added {len(center_traces)} center traces")
    
    # [EMOJI]
    x_data = main_trace['x'] if isinstance(main_trace['x'], (list, tuple)) else list(main_trace['x'])
    y_data = main_trace['y'] if isinstance(main_trace['y'], (list, tuple)) else list(main_trace['y'])
    
    # [EMOJI]/[EMOJI] trace[EMOJI]
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
                    'color': '#FFD700',  # Gold color
                    'symbol': 'star',
                    'line': {'width': 2, 'color': 'white'}
                },
                'text': [f'Doc {i+1}' for i in keyword_group_highlights if i < len(x_data)],
                'customdata': [[i] for i in keyword_group_highlights if i < len(x_data)],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
            print(f"    DEBUG:   Added keyword/group highlight trace with {len(highlight_x)} points")
    
    # [EMOJI] trace[EMOJI]
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
                'color': '#FF0000',  # Red color
                'symbol': 'star',
                'line': {'width': 3, 'color': 'white'}
            },
            'text': [f'Doc {selected_article_highlight+1}'],
            'customdata': [[selected_article_highlight]],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        })
        print(f"    DEBUG:   Added selected article highlight trace")
    
    # [EMOJI] traces
    updated_fig['data'] = traces
    
    print(f"    DEBUG:   Original trace count: {len(fig['data'])}")
    print(f"    DEBUG:   Final trace count: {len(traces)}")
    
    return updated_fig

# Add callback for rendering training group containers
@app.callback(
    Output("training-group-containers", "children"),
    [Input("group-order", "data"),
     Input("training-selected-group", "data"),
     Input("display-mode", "data")],  # Add display-mode as Input to trigger on mode change
    [State("training-selected-keyword", "data")],
    prevent_initial_call=False  # Allow initial call to populate content
)
def render_training_groups(group_order, selected_group, display_mode, selected_keyword):
    """Render training mode group containers - identical to render_groups but independent"""
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: render_training_groups CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   group_order: {group_order}")
    print(f"    DEBUG:   selected_group: {selected_group}")
    print(f"    DEBUG:   selected_keyword: {selected_keyword}")
    print(f"    DEBUG:   display_mode: {display_mode}")
    
    # Only process if we're in training mode
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
        # Group header with number and color
        if grp_name == "Other":
            group_display_name = "Other (Exclude)"
            group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Training Group {group_number}"
            group_color = get_group_color(grp_name)
        
        # Special styling for Other group in training mode
        if grp_name == "Other":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  # Dashed border for exclusion
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  # Slightly transparent
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

        # Keywords list
        group_keywords = []
        for i, kw in enumerate(kw_list):
            # Check if this keyword is selected for Training Group Management highlighting
            is_selected = selected_keyword and kw == selected_keyword
            
            # Use group color for keywords in this group with selection highlighting
            keyword_button = html.Button(
                kw,
                id={"type": "training-select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  # Highlight when selected
                    "color": "white" if is_selected else group_color,  # White text when selected
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  # Bold when selected
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

# Add callback for training group selection
@app.callback(
    [Output("training-selected-group", "data"),
     Output("training-selected-keyword", "data")],
    [Input({"type": "training-group-header", "index": ALL}, "n_clicks"),
     Input("display-mode", "data")],
    prevent_initial_call=True
)
def select_training_group(n_clicks, display_mode):
    """Handle training group selection - following keywords mode logic"""
    ctx = dash.callback_context
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: select_training_group CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
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
    
    # Check if this is a training group header click (only if n_clicks > 0)
    if "training-group-header" in triggered_id and triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
        print(f"    DEBUG:      Valid training group header click detected!")
        try:
            import json
            parsed_id = json.loads(triggered_id.split('.')[0])
            selected_group = parsed_id["index"]
            print(f"    DEBUG:   Extracted training group: {selected_group}")
            
            # Following keywords mode logic: group selection clears keyword selection
            print(f"    DEBUG:   Switching to training group: {selected_group}")
            print(f"    DEBUG:   Clearing training selected keyword")
            print(f"    DEBUG:       RETURN VALUES:")
            print(f"    DEBUG:     training-selected-group: {selected_group}")
            print(f"    DEBUG:     training-selected-keyword: None")
            print(f"    DEBUG:        SUCCESS: About to return (selected_group, None)")
            
            return selected_group, None  # Return group and clear keyword
                
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

# Add callback for training keyword selection
@app.callback(
    [Output("training-selected-keyword", "data", allow_duplicate=True),
     Output("training-selected-group", "data", allow_duplicate=True)],
    [Input({"type": "training-select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data"),
     State("group-order", "data")],
    prevent_initial_call=True
)
def select_training_keyword_from_group(n_clicks, display_mode, group_order):
    """Handle training keyword selection from group management - following keywords mode logic"""
    ctx = dash.callback_context
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: select_training_keyword_from_group CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG: n_clicks: {n_clicks}")
    print(f"    DEBUG: display_mode: {display_mode}")
    print(f"    DEBUG: group_order: {group_order}")
    
    if not ctx.triggered:
        print(f"    DEBUG: No context triggered")
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    print(f"    DEBUG: triggered_id: {triggered_id}")
    print(f"    DEBUG: triggered_n_clicks: {triggered_n_clicks}")
    
    # Check if this is a keyword selection
    if "training-select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            print(f"    DEBUG: Select training keyword from group management: {keyword}")
            
            # Check if this is a direct keyword click (n_clicks > 0)
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                print(f"    DEBUG: Direct training keyword click detected, selecting keyword: {keyword}")
                
                # Following keywords mode logic: keyword selection clears group selection
                print(f"    DEBUG: Following the same logic as keywords mode: keyword selection clears group selection")
                
                # Find documents that contain this keyword
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
                    
                    # Return keyword and clear group selection
                    # This ensures mutual exclusivity between keyword and group selection
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

# Add callback for training recommended articles
@app.callback(
    Output("training-articles-container", "children"),
    [Input("training-selected-keyword", "data"),
     Input("training-selected-group", "data"),
     Input("display-mode", "data")],  # Add display-mode as Input to trigger on mode change
    [State("group-order", "data")],
    prevent_initial_call=False  # Allow initial call to populate content
)
def display_training_recommended_articles(selected_keyword, selected_group, display_mode, group_order):
    """Display training recommended articles - identical to display_recommended_articles but independent"""
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: display_training_recommended_articles CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
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
    
    # Only process if we're in training mode
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
        
        # Create cache key based on search criteria (training specific)
        cache_key = None
        if selected_keyword:
            cache_key = f"training_keyword:{selected_keyword}"
        elif selected_group and group_order:
            # For groups, create cache key based on group keywords
            for group_name, keywords in group_order.items():
                if group_name == selected_group:
                    # Sort keywords for consistent cache key
                    cache_key = f"training_group:{group_name}:{':'.join(sorted(keywords))}"
                    break
        
        # Check cache first
        if cache_key and cache_key in _ARTICLES_CACHE:
            print(f"Using cached training articles for: {cache_key}")
            return _ARTICLES_CACHE[cache_key]
        
        # Determine search criteria
        search_keywords = []
        search_title = ""
        
        if selected_keyword:
            # Priority: specific keyword search
            search_keywords = [selected_keyword]
            search_title = f"Training Articles containing '{selected_keyword}'"
            print(f"Searching for training articles containing keyword: {selected_keyword}")
        elif selected_group:
            # Secondary: group keyword search
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
            # Show default content when no keyword or group is selected
            return html.Div([
                html.H6("Training Recommended Articles", style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P("Please select a training keyword or group to view recommended articles", 
                       style={"color": "#666", "fontStyle": "italic", "textAlign": "center", "padding": "20px"})
            ])
        
        # Search for articles containing any of the search keywords
        matching_articles = []
        for idx, row in df.iterrows():
            text = str(row.iloc[1]) if len(row) > 1 else ""
            text_lower = text.lower()
            
            # Check if any of the search keywords is in the text
            contains_keyword = any(keyword.lower() in text_lower for keyword in search_keywords)
            
            if contains_keyword:
                file_keywords = extract_top_keywords(text, 5)
                matching_articles.append({
                    'file_number': idx + 1,
                    'file_index': idx,
                    'text': text,
                    'keywords': file_keywords
                })
        
        if not matching_articles:
            result = html.P(f"No training articles found for the selected search criteria")
            if cache_key:
                _ARTICLES_CACHE[cache_key] = result
                print(f"Cached 'no training articles' result for: {cache_key}")
            return result
        
        # Create article display items
        article_items = [
            html.H6(f"{search_title} (Found {len(matching_articles)} articles)", 
                   style={"color": "#2c3e50", "marginBottom": "15px"})
        ]
        
        # Create article items with file number and keywords
        for article_info in matching_articles:
            # Create keyword tags
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
            
            # Create clickable article item with file number and keywords
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
        
        # Cache the result for future use
        result = html.Div(article_items)
        if cache_key:
            _ARTICLES_CACHE[cache_key] = result
            print(f"Cached training articles result for: {cache_key}")
        
        return result
        
    except Exception as e:
        print(f"Error displaying training recommended articles: {e}")
        return html.P(f"Error displaying training recommended articles: {str(e)}")

# Add callback for training article content display
@app.callback(
    [Output("training-article-fulltext-container", "children"),
     Output("training-selected-article", "data")],
    [Input({"type": "training-article-item", "index": ALL}, "n_clicks")],
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def display_training_article_content(article_clicks, display_mode):
    """Handle training article clicks - following keywords mode logic"""
    ctx = dash.callback_context
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: display_training_article_content CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   display_mode: {display_mode}")
    
    if not ctx.triggered:
        print(f"    DEBUG:         No context triggered")
        raise PreventUpdate
    
    # Handle training article item clicks
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
        # Load article content
        global df
        if 'df' not in globals():
            df = pd.read_csv(csv_path)
        
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
            print(f"    DEBUG:   Training mode: updating training-selected-article, not triggering keywords mode callbacks")
            
            # Update training-selected-article instead of selected-article to avoid triggering keywords mode callbacks
            return content, article_index
            
        else:
            print(f"    DEBUG:         Article index {article_index} out of range")
            return html.P("Training article not found", style={"color": "red"}), None
    
    except Exception as e:
        print(f"    DEBUG:         Error loading training article: {e}")
        return html.P(f"Error loading training article: {str(e)}", style={"color": "red"}), None





# Smart callback for handling article clicks in both modes
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
    """Smart callback that handles article clicks in both keywords and training modes - following keywords mode logic"""
    ctx = dash.callback_context
    
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: display_article_content_smart CALLBACK TRIGGERED")
    print(f"    DEBUG: ==========================================")
    print(f"    DEBUG: Function called at: {__import__('datetime').datetime.now()}")
    print(f"    DEBUG:     INPUT PARAMETERS:")
    print(f"    DEBUG:   display_mode: {display_mode}")
    print(f"    DEBUG:   current_keyword: {current_keyword}")
    print(f"    DEBUG:   current_group: {current_group}")
    print(f"    DEBUG:   group_order: {group_order}")
    
    if not ctx.triggered:
        print(f"    DEBUG:         No context triggered")
        raise PreventUpdate
    
    # Handle article item clicks from recommended articles
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
        # Load article content
        global df
        if 'df' not in globals():
            df = pd.read_csv(csv_path)
        
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
            print(f"    DEBUG:   Following keywords mode logic: article click only updates selected-article")
            print(f"    DEBUG:   This preserves existing Group/Keyword highlights")
            
            # Following keywords mode logic: article click only updates selected-article
            # This preserves existing Group/Keyword highlights and allows them to coexist
            return content, article_index
            
        else:
            print(f"    DEBUG:         Article index {article_index} out of range")
            return html.P("Article not found", style={"color": "red"}), None
    
    except Exception as e:
        print(f"    DEBUG:         Error loading article: {e}")
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"}), None

# Launch the Dash application
print("    DEBUG: ==========================================")
print("    DEBUG: CALLBACK REGISTRATION COMPLETE")
print("    DEBUG: ==========================================")
print("    DEBUG: All callbacks have been registered")
print("    DEBUG: Application is ready to start")

# ===================== Finetune Page Callbacks =====================

# Show Finetune button after training figures are available
@app.callback(
    [Output("display-mode", "data", allow_duplicate=True),
     Output("switch-finetune-btn", "children")],
    Input("switch-finetune-btn", "n_clicks"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def switch_to_finetune_mode(n_clicks, current_mode):
    """Only toggle between training and finetune; ignore in other modes"""
    if not n_clicks:
        raise PreventUpdate
    if current_mode == "training":
        return "finetune", "Switch to Training Mode"
    if current_mode == "finetune":
        return "training", "Switch to Finetune Mode"
    raise PreventUpdate

# Render finetune group containers
@app.callback(
    Output("finetune-group-containers", "children"),
    [Input("group-order", "data"),
     Input("finetune-selected-group", "data"),  # [EMOJI] Input[EMOJI]
     Input("finetune-selected-keyword", "data")]  # [EMOJI] selected_keyword
)
def render_finetune_groups(group_order, selected_group, selected_keyword):
    """Render finetune groups - identical styling to render_groups"""
    if not group_order:
        return []

    children = []
    for grp_name, kw_list in group_order.items():
        # Group header with number and color
        if grp_name == "Other":
            group_display_name = "Other (Exclude)"
            group_color = get_group_color(grp_name)
        else:
            group_number = grp_name.replace("Group ", "")
            group_display_name = f"Group {group_number}"
            group_color = get_group_color(grp_name)
        
        # Special styling for Other group
        if grp_name == "Other":
            header_style = {
                "width": "100%",
                "background": group_color if grp_name == selected_group else "#f0f0f0",
                "color": "white" if grp_name == selected_group else "black",
                "border": f"2px dashed {group_color}",  # Dashed border for exclusion
                "padding": "10px",
                "cursor": "pointer",
                "fontWeight": "bold",
                "marginBottom": "5px",
                "borderRadius": "5px",
                "opacity": "0.8"  # Slightly transparent
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

        # Keywords list - display as clickable buttons (like Training Mode)
        group_keywords = []
        for kw in kw_list:
            # Check if this keyword is selected for highlighting
            is_selected = selected_keyword and kw == selected_keyword
            
            # Display keyword as clickable button with group color
            keyword_button = html.Button(
                kw,
                id={"type": "finetune-select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 8px", 
                    "margin": "2px", 
                    "border": f"1px solid {group_color}", 
                    "width": "100%",
                    "textAlign": "left",
                    "backgroundColor": group_color if is_selected else f"{group_color}20",  # Highlight when selected
                    "color": "white" if is_selected else group_color,  # White text when selected
                    "cursor": "pointer",
                    "borderRadius": "4px",
                    "fontSize": "12px",
                    "fontWeight": "bold" if is_selected else "normal"  # Bold when selected
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

# Handle finetune group selection
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
    
    print(f"    DEBUG: select_finetune_group called")
    print(f"   display_mode: {display_mode}")
    print(f"   n_clicks: {n_clicks}")
    print(f"   ctx.triggered: {ctx.triggered}")
    
    # Only process if we're in finetune mode
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
        # [EMOJI] n_clicks > 0 [EMOJI]
        if trig_value and (isinstance(trig_value, (int, float)) and trig_value > 0):
            try:
                info = json.loads(trig.split('.')[0])
                group_name = info.get("index")
                print(f"        Finetune group header click: {group_name}")
                print(f"   [CLEAN] Clearing keyword selection and selected document")
                return group_name, None, None  # [EMOJI]
            except Exception as e:
                print(f"           Error parsing group header: {e}")
                raise PreventUpdate
        else:
            print(f"           Invalid n_clicks value: {trig_value}")
    
    print(f"           Not a valid group header click, raising PreventUpdate")
    raise PreventUpdate

# Handle finetune keyword selection
@app.callback(
    [Output("finetune-selected-keyword", "data", allow_duplicate=True),
     Output("finetune-selected-group", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    [Input({"type": "finetune-select-keyword", "keyword": ALL, "group": ALL}, "n_clicks")],
    [State("display-mode", "data")],
    prevent_initial_call=True
)
def select_finetune_keyword_from_group(n_clicks, display_mode):
    """Handle finetune keyword selection from group management - same logic as training mode"""
    ctx = dash.callback_context
    
    print(f"    DEBUG: select_finetune_keyword_from_group called")
    print(f"   display_mode: {display_mode}")
    
    # Only process if we're in finetune mode
    if display_mode != "finetune":
        print(f"           Not in finetune mode, raising PreventUpdate")
        raise PreventUpdate
    
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    # Check if this is a keyword selection
    if "finetune-select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            group = btn_info.get("group")  # [EMOJI]
            
            # Check if this is a direct keyword click (n_clicks > 0)
            if triggered_n_clicks and (isinstance(triggered_n_clicks, (int, float)) and triggered_n_clicks > 0):
                print(f"    Finetune keyword click: {keyword} from group: {group}")
                print(f"   Returning: keyword='{keyword}', group='{group}'")
                print(f"   [EMOJI]")
                print(f"   [CLEAN] Clearing selected document")
                # Finetune Mode: [EMOJI]
                return keyword, group, None  # [EMOJI]
        except Exception as e:
            print(f"        Error parsing finetune keyword click: {e}")
            raise PreventUpdate
    
    raise PreventUpdate

# Display finetune articles list ([EMOJI])
@app.callback(
    Output("finetune-articles-container", "children"),
    [Input("finetune-selected-group", "data"),
     Input("finetune-selected-keyword", "data"),
     Input("finetune-highlight-core", "data"),
     Input("finetune-highlight-gray", "data"),
     Input("finetune-selected-article-index", "data")],  # [EMOJI] Input[EMOJI]
    [State("group-order", "data"),
     State("display-mode", "data")]
)
def display_finetune_articles(selected_group, selected_keyword, core_indices, gray_indices, selected_article_idx, group_order, display_mode):
    """Display articles in finetune mode based on selected group/keyword"""
    if display_mode != "finetune":
        raise PreventUpdate
    
    if not selected_group and not selected_keyword:
        return html.P("Select a group or keyword to view documents", 
                     style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "40px 20px"})
    
    try:
        global df
        if 'df' not in globals():
            return html.P("Data not loaded", style={"color": "#e74c3c", "textAlign": "center"})
        
        # [EMOJI] core [EMOJI] gray [EMOJI]
        doc_indices = []
        doc_types = {}  # {index: "core" or "gray"}
        
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
        
        # [EMOJI]
        articles = []
        for i, idx in enumerate(sorted(doc_indices)):
            if idx >= len(df):
                continue
            
            text = str(df.iloc[idx, 1])[:200] + "..." if len(str(df.iloc[idx, 1])) > 200 else str(df.iloc[idx, 1])
            
            doc_type = doc_types.get(idx, "background")
            
            # [EMOJI]
            if doc_type == "core":
                type_label = "Core"
                type_color = "#FFD700"
                border_color = "#FFD700"
            elif doc_type == "gray":
                type_label = "[EMOJI] Gray"
                type_color = "#808080"
                border_color = "#808080"
            else:
                type_label = "        Background"
                type_color = "#1f77b4"
                border_color = "#1f77b4"
            
            # [EMOJI]
            is_selected = (selected_article_idx is not None and selected_article_idx == idx)
            
            # [EMOJI]
            if is_selected:
                card_style = {
                    "padding": "12px",
                    "marginBottom": "10px",
                    "borderRadius": "6px",
                    "border": "2px solid #FF4444",  # [EMOJI]
                    "backgroundColor": "white",
                    "cursor": "pointer",
                    "transition": "all 0.2s ease",
                    "boxShadow": "0 2px 6px rgba(255, 68, 68, 0.2)",  # [EMOJI]
                    "scrollMarginTop": "100px"
                }
                doc_label_style = {"fontWeight": "bold", "marginRight": "10px", "color": "#FF4444"}  # [EMOJI]
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
                    html.Span(f"Doc {idx+1}", style=doc_label_style),  # [EMOJI] 1 [EMOJI]
                    html.Span(type_label, style={"fontSize": "0.85rem", "color": type_color, "fontWeight": "bold"})
                ], style={"marginBottom": "8px", "display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                html.P(text, style={"color": "#34495e", "fontSize": "0.9rem", "margin": "0", "lineHeight": "1.4"})
            ], id={"type": "finetune-article-card", "index": idx}, className="finetune-doc-card", style=card_style)
            articles.append(article_card)
        
        # [EMOJI]
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
        import traceback
        traceback.print_exc()
        return html.P(f"Error: {str(e)}", style={"color": "#e74c3c", "textAlign": "center"})

# Handle clicking on finetune article cards - [EMOJI]
@app.callback(
    Output("finetune-selected-article-index", "data", allow_duplicate=True),
    Input({"type": "finetune-article-card", "index": ALL}, "n_clicks"),
    [State("display-mode", "data"),
     State("finetune-selected-article-index", "data")],
    prevent_initial_call=True
)
def handle_finetune_article_click(n_clicks, display_mode, current_selected):
    """Handle clicking on article cards in finetune mode"""
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
        # Parse the card ID to get the document index
        import json
        card_info = json.loads(triggered_id.split('.')[0])
        article_idx = card_info.get("index")
        
        print(f"    Finetune article card clicked: Doc {article_idx+1}")
        
        return article_idx
        
    except Exception as e:
        print(f"        Error handling finetune article click: {e}")
        import traceback
        traceback.print_exc()
        return current_selected

# [EMOJI] - [EMOJI]
@app.callback(
    Output("finetune-text-container", "children"),
    Input("finetune-selected-article-index", "data"),
    State("display-mode", "data"),
    prevent_initial_call=True
)
def update_finetune_text_preview(selected_idx, display_mode):
    """[EMOJI]"""
    if display_mode != "finetune":
        raise PreventUpdate
    
    if selected_idx is None:
        return html.P("Click a document to preview", 
                     style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "20px", "fontSize": "0.9rem"})
    
    try:
        global df
        if 'df' not in globals() or selected_idx >= len(df):
            return html.P("Document not found", style={"color": "#e74c3c"})
        
        # [EMOJI]
        full_text = str(df.iloc[selected_idx, 1])
        
        # [EMOJI]
        preview = html.Div([
            html.H5(f"Document {selected_idx+1}", style={"color": "#2c3e50", "marginBottom": "10px", "fontSize": "1rem"}),
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
        import traceback
        traceback.print_exc()
        return html.P(f"Error: {str(e)}", style={"color": "#e74c3c"})

# Compute finetune highlights using gap_based_group_filtering method
@app.callback(
    [Output("finetune-highlight-core", "data"),
     Output("finetune-highlight-gray", "data"),
     Output("finetune-operation-buttons", "children")],  # [EMOJI] dropdown options
    [Input("finetune-selected-group", "data"),
     Input("finetune-selected-keyword", "data"),
     Input("finetune-selected-article-index", "data")],  # [EMOJI] Input[EMOJI]
    [State("group-order", "data")]
)
def compute_finetune_highlights(selected_group, selected_keyword, selected_article_idx, group_order):
    global df
    core, gray = [], []
    operation_buttons = []
    
    print(f"    compute_finetune_highlights called:")
    print(f"   selected_group: {selected_group}")
    print(f"   selected_keyword: {selected_keyword}")
    print(f"   selected_article_idx: {selected_article_idx}")
    print(f"   group_order: {group_order}")
    
    # [EMOJI]
    # [EMOJI]
    excluded_group = None
    if selected_article_idx is not None and group_order:
        # [EMOJI] filtered_group_assignment.json [EMOJI]
        try:
            matched_dict_path = "test_results/filtered_group_assignment.json"
            if os.path.exists(matched_dict_path):
                with open(matched_dict_path, "r", encoding="utf-8") as f:
                    matched_dict = json.load(f)
                
                # [EMOJI]
                for grp_name in matched_dict.keys():
                    if isinstance(matched_dict[grp_name], list) and len(matched_dict[grp_name]) > 0:
                        if isinstance(matched_dict[grp_name][0], str):
                            matched_dict[grp_name] = [int(x) for x in matched_dict[grp_name]]
                
                # [EMOJI]
                for grp_name, indices in matched_dict.items():
                    try:
                        # Safely convert indices to integers if needed
                        if isinstance(indices, list):
                            int_indices = []
                            for idx in indices:
                                try:
                                    int_indices.append(int(idx))
                                except (ValueError, TypeError):
                                    continue
                            
                            # Safely convert selected_article_idx to int
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
    
    # [EMOJI]
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
        
        # [EMOJI] "Move to Group X" [EMOJI]
        for group_name in group_order.keys():
            if group_name != excluded_group:  # [EMOJI]
                # [EMOJI]
                if group_name == "Group 1":
                    bg_color = "#FF6B6B"  # [EMOJI]
                elif group_name == "Group 2":
                    bg_color = "#32CD32"  # [EMOJI]
                elif group_name == "Other":
                    bg_color = "#e74c3c"  # [EMOJI]
                else:
                    bg_color = "#3498db"  # [EMOJI]
                
                button_style = {**button_style_base, "backgroundColor": bg_color}
                
                operation_buttons.append(
                    html.Button(
                        f"Move to {group_name}",
                        id={"type": "finetune-move-to", "target": group_name},
                        n_clicks=0,
                        style=button_style
                    )
                )
        
        print(f"        Generated {len(operation_buttons)} operation buttons (excluding {excluded_group})")
    else:
        print(f"       Conditions not met, showing placeholder message")
        # [EMOJI]
        operation_buttons = [
            html.P("Select a document to perform operations", 
                   style={"color": "#7f8c8d", "fontStyle": "italic", "textAlign": "center", "padding": "10px"})
        ]
    
    print(f"    DEBUG: Returning {len(operation_buttons)} button(s)")
    
    # [EMOJI]
    if selected_keyword and not selected_group:
        try:
            keyword_docs = []
            for i in range(len(df)):
                text_lower = str(df.iloc[i, 1]).lower()
                if selected_keyword.lower() in text_lower:
                    keyword_docs.append(i)
            print(f"    Finetune keyword '{selected_keyword}': {len(keyword_docs)} documents ([EMOJI])")
            return keyword_docs, [], operation_buttons
        except Exception as e:
            print(f"        Error highlighting keyword in finetune mode: {e}")
            return [], [], operation_buttons
    
    if not selected_group or not group_order or selected_group not in group_order:
        return core, gray, operation_buttons
    
    try:
        # Finetune Mode [EMOJI] gap [EMOJI]filtered_group_assignment.json[EMOJI]
        # [EMOJI] filtered_group_assignment.json[EMOJI]
        # Gap [EMOJI] core/gray[EMOJI]
        # [EMOJI] gap [EMOJI]
        matched_dict_path = "test_results/filtered_group_assignment.json"
        if not os.path.exists(matched_dict_path):
            print("      [EMOJI] BM25 [EMOJI]")
            matched_dict_path = "test_results/bm25_search_results.json"
            if not os.path.exists(matched_dict_path):
                print("      [EMOJI]")
                return core, gray, operation_buttons
        
        with open(matched_dict_path, "r", encoding="utf-8") as f:
            matched_dict = json.load(f)
        
        # [EMOJI]
        for grp_name in matched_dict.keys():
            if isinstance(matched_dict[grp_name], list) and len(matched_dict[grp_name]) > 0:
                if isinstance(matched_dict[grp_name][0], str):
                    matched_dict[grp_name] = [int(x) for x in matched_dict[grp_name]]
        
        print(f"      Finetune Mode [EMOJI] {os.path.basename(matched_dict_path)}[EMOJI]:")
        if "filtered" in matched_dict_path:
            print(f"   [EMOJI] gap [EMOJI]")
        else:
            print(f"   [EMOJI]BM25 [EMOJI]")
        print(f"   Gap [EMOJI]")
        for grp_name, indices in matched_dict.items():
            print(f"  {grp_name}: {len(indices)} [EMOJI]")
            if len(indices) <= 10:  # [EMOJI]
                print(f"    [EMOJI]: {sorted(indices)}")
        
        # [EMOJI] Other [EMOJI] - [EMOJI]
        if selected_group == "Other":
            other_indices = matched_dict.get("Other", [])
            print(f"  Other [EMOJI]: {len(other_indices)} [EMOJI]")
            return other_indices, [], operation_buttons
        
        # [EMOJI] matched_dict [EMOJI]
        selected_group_indices = matched_dict.get(selected_group, [])
        
        # [EMOJI]
        model_path = "test_results/triplet_trained_encoder.pth"
        if not os.path.exists(model_path):
            print("      [EMOJI]")
            # [EMOJI]
            group_indices = matched_dict.get(selected_group, [])
            return group_indices, [], options
        
        # [EMOJI] gap_based_group_filtering [EMOJI]
        print(f"    Computing gap-based highlights for group: {selected_group}")
        
        if 'df' not in globals():
            df = pd.read_csv(csv_path)
        
        # [EMOJI]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [EMOJI]
        # [EMOJI] torch.load [EMOJI] weights_only=False [EMOJI]
        try:
            encoder = torch.load(model_path, map_location=device, weights_only=False)
            if hasattr(encoder, 'eval'):
                encoder.eval()
            else:
                # [EMOJI] state_dict[EMOJI]
                raise ValueError("Loaded state_dict instead of model")
        except Exception as e:
            print(f"  Failed to load model object directly: {e}")
            print("  Falling back to loading state_dict...")
            
            # [EMOJI] state_dict [EMOJI]
            state_dict = torch.load(model_path, map_location=device)
            
            # [EMOJI] BERT hidden size[EMOJI] proj.weight [EMOJI]
            if 'proj.weight' in state_dict:
                proj_weight_shape = state_dict['proj.weight'].shape
                proj_dim = proj_weight_shape[0]  # [EMOJI]
                bert_hidden_size = proj_weight_shape[1]  # BERT hidden size
                print(f"  Detected: proj_dim={proj_dim}, bert_hidden_size={bert_hidden_size}")
            else:
                proj_dim = 256
                bert_hidden_size = 768
            
            # [EMOJI] hidden_size [EMOJI] BERT [EMOJI]
            if bert_hidden_size == 768:
                bert_name = "bert-base-uncased"  # [EMOJI] 768 [EMOJI]
            elif bert_hidden_size == 384:
                bert_name = "sentence-transformers/all-MiniLM-L6-v2"
            else:
                bert_name = get_config("bert_model", "sentence-transformers/all-MiniLM-L6-v2")
            
            print(f"  Using BERT model: {bert_name}")
            
            # [EMOJI] LayerNorm [EMOJI]ln [EMOJI] norm[EMOJI]
            has_ln = 'ln.weight' in state_dict
            has_norm = 'norm.weight' in state_dict
            
            # [EMOJI]
            class SentenceEncoder(nn.Module):
                def __init__(self, bert_name, proj_dim, use_ln_name=False):
                    super().__init__()
                    from transformers import AutoModel
                    self.bert = AutoModel.from_pretrained(bert_name)
                    self.proj = nn.Linear(self.bert.config.hidden_size, proj_dim)
                    if use_ln_name:
                        self.ln = nn.LayerNorm(proj_dim)  # [EMOJI]
                    else:
                        self.norm = nn.LayerNorm(proj_dim)  # [EMOJI]
                    self.out_dim = proj_dim
                    self.use_ln_name = use_ln_name
                
                def encode_tokens(self, tokens):
                    out = self.bert(**tokens)
                    cls = out.last_hidden_state[:, 0, :]
                    z = self.proj(cls)
                    if self.use_ln_name:
                        z = self.ln(z)
                    else:
                        z = self.norm(z)
                    z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
                    return z
            
            # [EMOJI]
            # [EMOJI] CPU [EMOJI] meta tensor [EMOJI]
            encoder = SentenceEncoder(bert_name, proj_dim, use_ln_name=has_ln)
            encoder.load_state_dict(state_dict, strict=False)  # strict=False [EMOJI]
            encoder = encoder.to(device)  # [EMOJI]
            encoder.eval()
            
            # [EMOJI] bert_name [EMOJI] tokenizer [EMOJI]
            encoder.bert_model_name = bert_name
        
        # [EMOJI] tokenizer[EMOJI]encoder[EMOJI]BERT[EMOJI]
        from transformers import AutoTokenizer
        if hasattr(encoder, 'bert_model_name'):
            tokenizer_name = encoder.bert_model_name
        elif hasattr(encoder, 'bert') and hasattr(encoder.bert, 'name_or_path'):
            tokenizer_name = encoder.bert.name_or_path
        else:
            # [EMOJI] hidden_size [EMOJI]
            hidden_size = encoder.bert.config.hidden_size if hasattr(encoder, 'bert') else 768
            tokenizer_name = "bert-base-uncased" if hidden_size == 768 else "sentence-transformers/all-MiniLM-L6-v2"
        
        print(f"  Using tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # [EMOJI]
        texts = df.iloc[:, 1].astype(str).tolist()
        batch_size = 32
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                embeds = encoder.encode_tokens(inputs)
                all_embeddings.append(embeds.cpu().numpy())
        
        Z_raw = np.vstack(all_embeddings)
        
        # matched_dict [EMOJI]
        
        # [EMOJI]gap
        Z_norm = Z_raw / (np.linalg.norm(Z_raw, axis=1, keepdims=True) + 1e-8)
        
        # [EMOJI]
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
            print("  [EMOJI]")
            return core, gray, operation_buttons
        
        # [EMOJI]
        all_similarities = []
        for group_name in group_names_list:
            center = group_centers[group_name]
            sim = np.dot(Z_norm, center)
            all_similarities.append(sim)
        
        all_similarities = np.array(all_similarities).T  # [N, num_groups]
        
        # [EMOJI]gap: s_top1 - s_top2
        sorted_indices = np.argsort(all_similarities, axis=1)[:, ::-1]  # [EMOJI]
        s_top1 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 0]]
        s_top2 = all_similarities[np.arange(len(all_similarities)), sorted_indices[:, 1]] if all_similarities.shape[1] > 1 else s_top1
        gap = s_top1 - s_top2
        
        # [EMOJI]top1[EMOJI]
        arg1 = sorted_indices[:, 0]  # [EMOJI]
        
        # [EMOJI]Other[EMOJI]
        if selected_group not in group_names_list:
            return core, gray, operation_buttons
        
        # [EMOJI] matched_dict [EMOJI]
        selected_group_indices = matched_dict.get(selected_group, [])
        if len(selected_group_indices) == 0:
            print(f"  {selected_group} [EMOJI]")
            return core, gray, operation_buttons
        
        print(f"  {selected_group} [EMOJI] {len(selected_group_indices)} [EMOJI]")
        
        # [EMOJI]mask[EMOJI]
        group_mask = np.zeros(len(df), dtype=bool)
        group_mask[selected_group_indices] = True
        
        # [EMOJI]gap[EMOJI]
        gaps_group = gap[selected_group_indices]
        mean_gap = gaps_group.mean()
        std_gap = gaps_group.std()
        
        alpha = get_config("gap_alpha", 1.0)
        min_samples = get_config("gap_min_samples", 10)
        percentile_fallback = get_config("gap_percentile_fallback", 25)
        thr_floor = get_config("gap_floor_threshold", 0.05)
        mix_ratio = get_config("gap_mix_ratio", 0.3)
        
        # [EMOJI]mean - α*std
        base_threshold = mean_gap - alpha * std_gap
        
        # [EMOJI]/[EMOJI]
        if len(gaps_group) < min_samples or std_gap < 1e-6:
            gray_threshold = np.percentile(gaps_group, percentile_fallback)
            core_threshold = np.percentile(gaps_group, 50)  # [EMOJI]
            print(f"    {selected_group}: [EMOJI]:{len(gaps_group)}, std:{std_gap:.6f}[EMOJI]")
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
        

        print(f"    [EMOJI] {len(selected_group_indices)} [EMOJI]...")
        for idx in selected_group_indices:
            gap_val = gap[idx]
            dist_to_center = distances_to_center[idx]
            

            if gap_val >= core_threshold and dist_to_center <= prototype_radius:
                core.append(idx)

            else:
                gray.append(idx)
        
        print(f"     {selected_group}: {len(selected_group_indices)} [EMOJI]")
        print(f"   [EMOJI]: {len(core)} [EMOJI] - [EMOJI]: {core[:10]}")
        print(f"   [EMOJI]: {len(gray)} [EMOJI] - [EMOJI]: {gray[:10]}")
        print(f"   [EMOJI]: {len(core) + len(gray)} [EMOJI] {len(selected_group_indices)}[EMOJI]")
        
        if len(core) + len(gray) != len(selected_group_indices):
            print(f"    [EMOJI]")
        
        # [EMOJI]
        # [EMOJI]/[EMOJI]
        if selected_keyword:
            print(f"    [EMOJI] '{selected_keyword}'...")
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
            
            print(f"   [EMOJI]: {len(keyword_core)} [EMOJI]")
            print(f"   [EMOJI]: {len(keyword_gray)} [EMOJI]")
            print(f"   [EMOJI]: {len(keyword_core) + len(keyword_gray)} [EMOJI]")
            
            return keyword_core, keyword_gray, operation_buttons
        
        return core, gray, operation_buttons
        
    except Exception as e:
        print(f"        Error in gap-based filtering: {e}")
        import traceback
        traceback.print_exc()
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
        import traceback
        traceback.print_exc()

    # Fallback to cached documents TSNE
    if not idx_to_coord:
        print("    No coordinates from training figures, using cached t-SNE")
        if _GLOBAL_DOCUMENT_EMBEDDINGS_READY and _GLOBAL_DOCUMENT_TSNE is not None:
            coords = _GLOBAL_DOCUMENT_TSNE
            idx_to_coord = {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}
        else:
            # Last resort empty figure
            return {
                'data': [],
                'layout': {'title': 'Finetune - No coordinates available'}
            }


    valid_indices = list(idx_to_coord.keys())
    print(f"[EMOJI] Using {len(valid_indices)} coordinates for finetune plot")
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
    """[EMOJI] 2D [EMOJI]"""
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

# Temp assign operations - [EMOJI]
@app.callback(
    [Output("finetune-temp-assignments", "data", allow_duplicate=True),
     Output("finetune-selected-article-index", "data", allow_duplicate=True)],
    Input({"type": "finetune-move-to", "target": ALL}, "n_clicks"),
    [State("finetune-selected-article-index", "data"),
     State("finetune-temp-assignments", "data")],
    prevent_initial_call=True
)
def finetune_move_document(n_clicks_list, selected_idx, assignments):
    """Handle all 'Move to X' button clicks"""
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
    
        import json as json_module
        button_id = json_module.loads(triggered_id.split('.')[0])
        target_group = button_id['target']
        
        new_map = dict(assignments or {})
        new_map[str(selected_idx)] = target_group
        print(f"     Moved Doc {selected_idx+1} to {target_group}")
        
      
        print(f"[CLEAN] Clearing selected document after move")
        return new_map, None
    except Exception as e:
        print(f"        Error moving document: {e}")
        import traceback
        traceback.print_exc()
        raise PreventUpdate

# Clear adjustment history
@app.callback(
    Output("finetune-temp-assignments", "data", allow_duplicate=True),
    Input("finetune-clear-history-btn", "n_clicks"),
    prevent_initial_call=True
)
def clear_finetune_history(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    print("[EMOJI] Clearing adjustment history")
    return {}

# Update adjustment history display
@app.callback(
    Output("finetune-adjustment-history", "children"),
    Input("finetune-temp-assignments", "data")
)
def update_adjustment_history(temp_assignments):
    global df
    if not temp_assignments or len(temp_assignments) == 0:
        return html.P("No adjustments yet. Click a point and use the buttons to reassign.", 
                     style={
                         "color": "#7f8c8d", 
                         "fontStyle": "italic", 
                         "textAlign": "center", 
                         "padding": "20px",
                         "fontSize": "0.95rem"
                     })
    
  
    try:
        matched_dict_path = "test_results/filtered_group_assignment.json"
        if not os.path.exists(matched_dict_path):
            matched_dict_path = "test_results/bm25_search_results.json"
        
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
            df = pd.read_csv(csv_path)
        except:
            df = None
    
   
    history_items = []
    for idx_str, new_group in temp_assignments.items():
        idx = int(idx_str)
        original_group = idx_to_original_group.get(idx, "Unknown")
        
        
        doc_preview = "..."
        if df is not None and idx < len(df):
            doc_text = str(df.iloc[idx, 1])
            doc_preview = doc_text[:50] + "..." if len(doc_text) > 50 else doc_text
        
        
        color_from = get_group_color(original_group)
        color_to = get_group_color(new_group)
        
        history_items.append(
            html.Div([
                html.Div([
                    html.Span(f"Doc {idx+1}", style={  
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                        "marginRight": "10px"
                    }),
                    html.Span("→", style={"margin": "0 5px", "color": "#95a5a6"}),
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
                    html.Span("→", style={"margin": "0 5px", "color": "#95a5a6"}),
                    html.Span(new_group, style={
                        "backgroundColor": color_to,
                        "color": "white",
                        "padding": "2px 8px",
                        "borderRadius": "4px",
                        "fontSize": "0.85rem"
                    })
                ], style={"marginBottom": "8px"}),
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
    ]


@app.callback(
    [Output("finetune-train-btn", "children"),
     Output("finetune-train-btn", "style", allow_duplicate=True),
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
    """Run prototype-based finetune training with user adjustments"""
    if not n_clicks:
        raise PreventUpdate
    
    try:
        print("=" * 60)
        print("[EMOJI] Starting Finetune Training (Prototype-based)")
        print("=" * 60)
        
        global df
        if 'df' not in globals():
            df = pd.read_csv(csv_path)
        
   
        print(f"[EMOJI] Applying {len(temp_assignments or {})} user adjustments...")
        adjusted_group_order = {}
        for grp_name, kw_list in group_order.items():
            adjusted_group_order[grp_name] = kw_list.copy()
        
    
        matched_dict_adjusted = {}
        for grp_name in group_order.keys():
            matched_dict_adjusted[grp_name] = []
        
   
        for i in range(len(df)):
            text_lower = str(df.iloc[i, 1]).lower()
            assigned = False
            for grp_name, kw_list in adjusted_group_order.items():
                if grp_name == "Other":
                    continue
                match_count = sum(1 for kw in kw_list if kw.lower() in text_lower)
                if match_count >= 1:
                    matched_dict_adjusted[grp_name].append(i)
                    assigned = True
                    break
            if not assigned:
                matched_dict_adjusted["Other"].append(i)
        
 
        if temp_assignments:
            for idx_str, target_group in temp_assignments.items():
                idx = int(idx_str)
              
                for grp_name in matched_dict_adjusted.keys():
                    if idx in matched_dict_adjusted[grp_name]:
                        matched_dict_adjusted[grp_name].remove(idx)
            
                if target_group in matched_dict_adjusted:
                    matched_dict_adjusted[target_group].append(idx)
        
        print(f"      Adjusted group distribution:")
        for grp_name, indices in matched_dict_adjusted.items():
            print(f"  {grp_name}: {len(indices)} samples")
        

        model_path = "test_results/triplet_trained_encoder.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
   
        try:
            encoder = torch.load(model_path, map_location=device, weights_only=False)
            if hasattr(encoder, 'eval'):
                encoder.eval()
            else:
                raise ValueError("Loaded state_dict instead of model")
        except Exception:
            state_dict = torch.load(model_path, map_location=device)
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
            
            class SentenceEncoder(nn.Module):
                def __init__(self, bert_name, proj_dim, use_ln_name=False):
                    super().__init__()
                    from transformers import AutoModel
                    self.bert = AutoModel.from_pretrained(bert_name)
                    self.proj = nn.Linear(self.bert.config.hidden_size, proj_dim)
                    if use_ln_name:
                        self.ln = nn.LayerNorm(proj_dim)
                    else:
                        self.norm = nn.LayerNorm(proj_dim)
                    self.out_dim = proj_dim
                    self.use_ln_name = use_ln_name
                
                def encode_tokens(self, tokens):
                    out = self.bert(**tokens)
                    cls = out.last_hidden_state[:, 0, :]
                    z = self.proj(cls)
                    if self.use_ln_name:
                        z = self.ln(z)
                    else:
                        z = self.norm(z)
                    z = z / (z.norm(dim=1, keepdim=True) + 1e-12)
                    return z
            
            encoder = SentenceEncoder(bert_name, proj_dim, use_ln_name=has_ln).to(device)
            encoder.load_state_dict(state_dict, strict=False)
            encoder.eval()
            encoder.bert_model_name = bert_name
        

        from transformers import AutoTokenizer
        if hasattr(encoder, 'bert_model_name'):
            tokenizer_name = encoder.bert_model_name
        else:
            tokenizer_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        print("[EMOJI] Running prototype-based finetune training...")

        texts = df.iloc[:, 1].astype(str).tolist()
        

        def encode_all_docs():
            encoder.eval()
            Z = []
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                toks = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks = {k: v.to(device) for k, v in toks.items()}
                with torch.no_grad():
                    Z.append(encoder.encode_tokens(toks).cpu())
            return torch.vstack(Z)
        
        Z_before = encode_all_docs().numpy()
        
      
        group_prototypes = {}
        for grp_name, indices in matched_dict_adjusted.items():
            if grp_name == "Other" or len(indices) == 0:
                continue
            grp_embeds = Z_before[indices]
            prototype = grp_embeds.mean(axis=0)
            prototype = prototype / (np.linalg.norm(prototype) + 1e-12)
            group_prototypes[grp_name] = torch.tensor(prototype, device=device)
        
        print(f"  Computed {len(group_prototypes)} group prototypes")
        
 
        encoder.train()
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-5)
        
        epochs = 3
        batch_size = 16
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
        
            train_samples = []
            for grp_name, indices in matched_dict_adjusted.items():
                if grp_name == "Other":
                    continue
                for idx in indices:
                    train_samples.append((idx, grp_name))
            
            import random
            random.shuffle(train_samples)
            
            for i in range(0, len(train_samples), batch_size):
                batch = train_samples[i:i+batch_size]
                batch_texts = [texts[idx] for idx, _ in batch]
                batch_groups = [grp for _, grp in batch]
                
      
                toks = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
                toks = {k: v.to(device) for k, v in toks.items()}
                embeds = encoder.encode_tokens(toks)
                
   
                loss = 0
                for j, grp in enumerate(batch_groups):
                    if grp in group_prototypes:
                        proto = group_prototypes[grp]
                        similarity = torch.cosine_similarity(embeds[j], proto, dim=0)
                        loss += (1 - similarity)
                
                loss = loss / len(batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("     Finetune training completed!")
        

        torch.save(encoder.state_dict(), model_path)
        print(f"        Saved finetuned model to {model_path}")
        

        filtered_path = "test_results/filtered_group_assignment.json"
        with open(filtered_path, "w", encoding="utf-8") as f:
            json.dump(matched_dict_adjusted, f, ensure_ascii=False, indent=2)
        print(f"        Saved adjusted group assignment to {filtered_path}")
        print(f"[EMOJI] Saved group distribution:")
        for grp_name, indices in matched_dict_adjusted.items():
            print(f"  {grp_name}: {len(indices)} samples")
            if len(indices) <= 10:
                print(f"    [EMOJI]: {sorted(indices)}")
        

        print(f"      [EMOJI] 2D [EMOJI]...")
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
        
  
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        projected_2d_after = tsne.fit_transform(Z_after)
    
        
     
        group_centers = {}
        for grp_name, indices in matched_dict_adjusted.items():
            if indices and len(indices) > 0:
                valid_indices = [i for i in indices if i < len(projected_2d_after)]
                if valid_indices:
                    center = projected_2d_after[valid_indices].mean(axis=0)
                    group_centers[grp_name] = center
                    print(f"   {grp_name} [EMOJI]: {center}")
        
 
        import plotly.graph_objects as go
        
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
        print(f"     2D [EMOJI]")
        
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
        

        core_indices = []
        gray_indices = []
        
        print(f"    DEBUG: Preparing to return highlight data")
        print(f"   current_selected_group: {current_selected_group}")
        print(f"   matched_dict_adjusted keys: {list(matched_dict_adjusted.keys())}")
        

        if not current_selected_group or current_selected_group not in matched_dict_adjusted:
            print(f"    No selected group, trying to auto-select from adjustments")
            

            if temp_assignments:
                from collections import Counter

                target_groups = [new_grp for new_grp in temp_assignments.values()]
                most_common = Counter(target_groups).most_common(1)
                if most_common and most_common[0][0] != "Other":
                    current_selected_group = most_common[0][0]
                    print(f"   Auto-selected group: {current_selected_group} (most adjusted)")
            

            if not current_selected_group or current_selected_group not in matched_dict_adjusted:
                for grp in matched_dict_adjusted.keys():
                    if grp != "Other":
                        current_selected_group = grp
                        print(f"   Auto-selected group: {current_selected_group} (first non-Other)")
                        break
        

        if not current_selected_group or current_selected_group not in matched_dict_adjusted:
            # [EMOJI]
            if temp_assignments:
                from collections import Counter
      
                target_groups = [new_grp for new_grp in temp_assignments.values()]
                most_common = Counter(target_groups).most_common(1)
                if most_common and most_common[0][0] != "Other":
                    current_selected_group = most_common[0][0]
                    print(f"   Auto-selected group: {current_selected_group} (most adjusted)")
            
        
            if not current_selected_group or current_selected_group not in matched_dict_adjusted:
                for grp in matched_dict_adjusted.keys():
                    if grp != "Other":
                        current_selected_group = grp
                        print(f"   Auto-selected group: {current_selected_group} (first non-Other)")
                        break
        
        print(f"        Iteration complete. compute_finetune_highlights will recalculate highlights for {current_selected_group}")
        print(f"   This ensures consistent gap-based styling")
        

        return "[EMOJI] Iteration Complete - Adjust & Train Again", success_style, current_selected_group, None, {}, dash.no_update, dash.no_update, updated_figures
        
    except Exception as e:
        print(f"        Finetune training failed: {e}")
        import traceback
        traceback.print_exc()
        
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
        
        return f"Training Failed: {str(e)}", error_style, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == "__main__":
    print("    DEBUG: ==========================================")
    print("    DEBUG: DASH APPLICATION STARTUP")
    print("    DEBUG: ==========================================")
    print("    DEBUG: Starting Dash application...")
    print("    DEBUG: Target URL: http://127.0.0.1:8053")
    print("    DEBUG: Current working directory:", os.getcwd())
    print("    DEBUG: Python version:", __import__('sys').version)
    print("    DEBUG: Dash version:", __import__('dash').__version__)
    
    # Disable reloader to prevent threading issues
    import os
    os.environ['FLASK_ENV'] = 'development'
    
    print("    DEBUG: Environment variables set:")
    print("    DEBUG:   FLASK_ENV =", os.environ.get('FLASK_ENV'))
    
    try:
        print("    DEBUG:     ATTEMPTING TO START ON PORT 8053...")
        app.run(
            debug=True,
            port=8053,
            host='127.0.0.1',
            use_reloader=False,
            threaded=True
        )
    except OSError as e:
        print(f"    DEBUG: OSError occurred on port 8053: {e}")
        print(f"    DEBUG: Error type: {type(e)}")
        print(f"    DEBUG: Trying alternative port 8054...")
        
        try:
            app.run(
                debug=True, 
                port=8054, 
                host='127.0.0.1', 
                use_reloader=False,
                threaded=True
            )
        except OSError as e2:
            print(f"    DEBUG: Alternative port 8054 also failed: {e2}")
            print(f"    DEBUG: Please check if ports are available")
            print(f"    DEBUG: You can manually specify a different port")

            import socket
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
                print(f"    DEBUG: All attempts failed: {e3}")
                print(f"    DEBUG: Please restart the application manually")
