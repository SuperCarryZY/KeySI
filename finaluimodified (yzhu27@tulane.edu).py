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
from sklearn.cluster import MiniBatchKMeans, KMeans
import gc
import multiprocessing
from collections import Counter
import nltk

# Use t-SNE for keyword dimensionality reduction (removed BertTopic for simplicity)
# NLTK will automatically find data path on Windows, no manual specification needed
# nltk.data.path.append("/Users/yanzhu/nltk_data")
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

# Auto-detect and download NLTK data packages
def ensure_nltk_data():
    """Ensure NLTK data packages are downloaded"""
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
                # For punkt_tab, if download fails, try punkt as fallback
                if package_name == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                    except:
                        pass
                        
    # Download some additional common packages just in case
    additional_packages = ['stopwords', 'wordnet', 'omw-1.4']
    for package in additional_packages:
        try:
            nltk.download(package, quiet=True)
        except:
            pass  # Ignore if download fails

# Ensure NLTK packages are available
ensure_nltk_data()

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster




# Set device and global parameters
# Auto-detect and select best device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# GPU memory optimization settings
if device == "cuda":
    # Enable CUDA memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def clear_gpu_memory():
    """Clear GPU memory"""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# Ensure CSV directory exists and switch to it
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

if current_dir.endswith("CSV"):
    print("Already in CSV directory")
elif os.path.exists("CSV"):
    os.chdir("CSV")
    print(f"Switched to CSV directory: {os.getcwd()}")
else:
    print("CSV folder does not exist, please check file path")
    print("Please ensure you run this script from the project root directory")
    exit(1)
num_threads=8
top_similiar_file_to_keywords=500
learningrate=1e-4
num_epochs = 1
margin_number = 3
top_keywords=3
early_stop_threshold = 8
mostfequentwords=250
cluster_distance=30
word_count_feq=3
max_d = 50
word_count_threshold= 1
top_similar_files = 3000

# Dynamically adjust batch_size based on device type for optimization
if device == "cuda":
    batch_size = 256  
elif device == "mps":
    batch_size = 128  # Apple Silicon GPU moderate batch size
else:
    batch_size = 64   # CPU uses smaller batch size

clusterthreshold = 25

# Define group colors for different groups - avoiding blue-like colors
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
}

def get_group_color(group_name):
    """Get color for a specific group"""
    return GROUP_COLORS.get(group_name, "#808080")  # Default gray if group not found

# Define relative paths for data, model saving, and output
img_output_dir = "../Keyword_Group/Test"
csv_path = "risk_factors.csv"  
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

# Model settings
try:
    embedding_model_kw = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
except Exception as e:
    print(f"Model initialization failed: {e}")
    raise

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
        dcc.Store(id="display-mode", data="keywords"),  # Display mode: "keywords" or "training"
        dcc.Store(id="training-figures", data={"before": None, "after": None}),  # Store training figures
        dcc.Store(id="highlighted-indices", data=[]),  # Store highlighted article indices
        
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
        ], style={'display': 'flex', 'marginBottom': '30px'}),
        
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
    print(f"🔄 update_group_order called")
    print(f"🔄 triggered: {ctx.triggered}")
    print(f"🔄 group_data: {group_data}")
    if not ctx.triggered:
        raise PreventUpdate
    
    new_order = dict(current_order) if current_order else {}
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "generate-btn":
        if not num_groups or num_groups < 1:
            raise PreventUpdate
        return {f"Group {i+1}": [] for i in range(num_groups)}
    
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
     Input("selected-keyword", "data")]
)
def render_groups(group_order, selected_group, selected_keyword):
    print(f"render_groups called")
    print(f"group_order: {group_order}")
    print(f"selected_group: {selected_group}")
    print(f"selected_keyword: {selected_keyword}")
    if not group_order:
        return []

    children = []
    for grp_name, kw_list in group_order.items():
        # Group header with number and color
        group_number = grp_name.replace("Group ", "")
        group_color = get_group_color(grp_name)
        
        group_header = html.Button(
            f"Group {group_number}",
            id={"type": "group-header", "index": grp_name},
            style={
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
        )

        # Keywords list
        group_keywords = []
        for i, kw in enumerate(kw_list):
            # Check if this keyword is selected for Group Management highlighting
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

@app.callback(
    [Output("selected-group", "data"),
     Output("selected-keyword", "data", allow_duplicate=True)],
    Input({"type": "group-header", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def select_group(n_clicks):
    ctx = dash.callback_context
    print(f"🔵 select_group called")
    print(f"🔵 triggered: {ctx.triggered}")
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']

    print(f"triggered_id: {triggered_id}")
    print(f"triggered_n_clicks: {triggered_n_clicks}")
    
    # Check if this is a group header click (only if n_clicks > 0)
    if "group-header" in triggered_id and triggered_n_clicks and triggered_n_clicks > 0:
        try:
            selected_group = json.loads(triggered_id.split('.')[0])["index"]
            print(f"Switch to group: {selected_group}")  # Add debug info
            print(f"DEBUG: Group selection callback completed")
            print(f"DEBUG: Clearing selected keyword to show all group documents")
            
            # Clear selected keyword when switching groups
            # This ensures group selection shows all documents containing group keywords
            # Also add a flag to prevent keyword selection from overriding
            return selected_group, None
        except Exception as e:
            print(f"Error parsing group header ID: {e}")
            raise PreventUpdate

    raise PreventUpdate

@app.callback(
    Output("selected-keyword", "data", allow_duplicate=True),
    Input({"type": "select-keyword", "keyword": ALL, "group": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def select_keyword_from_group(n_clicks):
    """Handle keyword selection from group management"""
    print(f"select_keyword_from_group called")
    print(f"n_clicks: {n_clicks}")
    
    ctx = dash.callback_context
    print(f"ctx.triggered: {ctx.triggered}")
    
    if not ctx.triggered:
        print("No context triggered")
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    print(f"triggered_id: {triggered_id}")
    print(f"triggered_n_clicks: {triggered_n_clicks}")
    
    # Check if this is a keyword selection (even if n_clicks is None initially)
    if "select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            print(f"Select keyword from group management: {keyword}")
            
            # Check if this is triggered by group selection (not a direct keyword click)
            if triggered_n_clicks is None:
                print(f"Keyword selection triggered by group change, ignoring")
                raise PreventUpdate
            
            # Check if this is a direct keyword click (n_clicks > 0)
            if triggered_n_clicks and triggered_n_clicks > 0:
                print(f"Direct keyword click detected, selecting keyword: {keyword}")
                return keyword
            else:
                print(f"Not a direct keyword click, ignoring")
                raise PreventUpdate
            
        except Exception as e:
            print(f"Error parsing button info: {e}")
            raise PreventUpdate
    
    print("Not a keyword selection")
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
    State("group-order", "data")  # Add group_order as State parameter
)
def display_recommended_articles(selected_keyword, selected_group, group_order):
    """Display recommended articles based on selected keyword or group"""
    print(f"display_recommended_articles called for keyword: {selected_keyword}, group: {selected_group}")
    
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
    print(f"🔵 Clear selected keyword") 
    
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
    # Clear caches when training starts as data relationships may change
    clear_caches()
    
    if not os.path.exists(final_list_path):
        default_groups = {
            "Group 1": ["cancer", "patient", "treatment"],
            "Group 2": ["covid", "infection", "disease"],
            "Group 3": ["mortality", "death", "risk"]
        }
        os.makedirs(os.path.dirname(final_list_path), exist_ok=True)
        with open(final_list_path, "w", encoding="utf-8") as f:
            json.dump(default_groups, f, indent=4, ensure_ascii=False)
        print(f"Created default final_list.json at {final_list_path}")
    
    with open(final_list_path, "r", encoding="utf-8") as f:
        final_dict = json.load(f)
    
    df = pd.read_csv(csv_path)
    df.dropna(subset=[df.columns[1]], inplace=True)
    all_articles = df.iloc[:, 1].astype(str).tolist()
    labels = df.iloc[:, 0].values
    original_indices = df.index.to_list()


    embedding_model_kw = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)


    kw_model = KeyBERT(model=embedding_model_kw)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)


    def stem_keywords_dict(original_dict):
        stemmed_dict = {}
        for group_name, keywords in original_dict.items():
            stemmed_keywords = [ps.stem(word.lower()) for word in keywords]
            stemmed_dict[group_name] = stemmed_keywords
        return stemmed_dict


    ps = PorterStemmer()
    def preprocess(text):
        ps = PorterStemmer()
        return [ps.stem(word) for word in text.lower().split()]
    ps = PorterStemmer()
    tokenized_corpus = Parallel(n_jobs=num_threads)(delayed(preprocess)(doc) for doc in all_articles)
    bm25 = BM25Okapi(tokenized_corpus)

    def bm25_search_batch(query_groups):
        results = {}
        for group_name, query_words in query_groups.items():
            query_stemmed = [PorterStemmer().stem(word) for word in query_words]
            scores = bm25.get_scores(query_stemmed)  
            

            valid_indices = [i for i in range(len(scores)) if scores[i] > 0]
            

            sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)[:top_similar_files]
            

            results[group_name] = [original_indices[i] for i in sorted_indices if 0 <= i < len(original_indices)]
        
        return results

    stemmed_final_dict = stem_keywords_dict(final_dict)
    group_to_indices_map = bm25_search_batch(stemmed_final_dict)
    print(group_to_indices_map)


    def compute_embeddings_batch(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            # Immediately transfer to CPU to free GPU memory
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
        return embeddings


    group_embeddings = []
    group_labels = []
    group_original_indices = []  

    for group_name, indices in group_to_indices_map.items():
        if len(indices) == 0:

            continue
            
        selected_texts = [df.loc[i, df.columns[1]] for i in indices] 

        embeddings = Parallel(n_jobs=num_threads)(
            delayed(compute_embeddings_batch)(selected_texts[i: i + batch_size])
            for i in range(0, len(selected_texts), batch_size)
        )

        if not embeddings:
            print(f"Warning: No embeddings generated for group {group_name}, skipping...")
            continue
            
        embeddings = np.vstack(embeddings)
        group_embeddings.append(embeddings)
        group_labels.extend([group_name] * len(embeddings))
        group_original_indices.extend(indices)

    if not group_embeddings:
        raise ValueError("No valid embeddings found for any group. Please check your keyword groups and data.")
        
    group_embeddings = np.vstack(group_embeddings)
    group_labels = np.array(group_labels)
    group_original_indices = np.array(group_original_indices)  

    print(f" Processed {group_embeddings.shape[0]} articles, embedding size: {group_embeddings.shape[1]}")


    def optimal_kmeans(data, max_k=10):
        best_k = 2
        best_score = -1
        for k in range(2, min(max_k, len(data))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
            score = silhouette_score(data, kmeans.labels_)
            if score > best_score:
                best_k = k
                best_score = score
        return best_k


    group_to_indices_map_cleaned = {}
    unique_groups = np.unique(group_labels)

    for group_name in unique_groups:
        indices = np.where(group_labels == group_name)[0]
        group_embeds = group_embeddings[indices]  
        group_orig_indices = group_original_indices[indices]  
        if len(group_embeds) < 2:
            continue
        query_words = final_dict[group_name]
        query_vector = np.mean(embedding_model_kw.encode([" ".join(query_words)], convert_to_tensor=True).cpu().numpy(), axis=0)


        combined_embeds = np.vstack([group_embeds, query_vector.reshape(1, -1)])
        

        # Calculate perplexity parameter - ensure perplexity < n_samples
        n_samples = len(combined_embeds)
        if n_samples < 2:
            print(f"⚠ Too few samples ({n_samples}), skip this group")
            continue
        
        # perplexity must be less than sample count, and at least 1
        perplexity = min(30, max(1, min(n_samples - 1, n_samples // 3)))
        print(f"  t-SNE parameters: samples={n_samples}, perplexity={perplexity}")
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        projected_2d_after = tsne.fit_transform(combined_embeds)
        
        reduced_group_embeds = projected_2d_after[:-1] 
        query_vector_2d = projected_2d_after[-1]


        best_k = optimal_kmeans(reduced_group_embeds)
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced_group_embeds)


        cluster_avg_vectors = {
            cluster_id: np.mean(reduced_group_embeds[np.where(cluster_labels == cluster_id)[0]], axis=0)
            for cluster_id in np.unique(cluster_labels)
        }



        main_cluster = min(cluster_avg_vectors, key=lambda cluster_id: cosine(cluster_avg_vectors[cluster_id], query_vector_2d))


        cluster_center = cluster_avg_vectors[main_cluster]  

        distances = np.linalg.norm(reduced_group_embeds - cluster_center, axis=1)


        threshold = np.percentile(distances, clusterthreshold)
        valid_indices = group_orig_indices[distances <= threshold]  

        group_to_indices_map_cleaned[group_name] = valid_indices.tolist()


    for group_name, cleaned_indices in group_to_indices_map_cleaned.items():
        print(f"{group_name}: {cleaned_indices}")




    group_to_indices_map2 = group_to_indices_map_cleaned
    group_dict = {grp_name: idx_list for grp_name, idx_list in group_to_indices_map2.items()}



    with open(group_dict_path, "w", encoding="utf-8") as f:
        json.dump(group_dict, f, indent=4, ensure_ascii=False)


    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Group,Index,Label\n")
        for grp, indices in group_dict.items():
            for idx in indices:
                if idx in df.index:
                    label = df.loc[idx, df.columns[0]]
                    f.write(f"{grp},{idx},{label}\n")

    print(f"Results saved to {output_file}")
    labels = df.iloc[:, 0].values
    group_dict = {grp_name: idx_list for grp_name, idx_list in group_to_indices_map.items()}

    for grp, indices in group_dict.items():
        print(f"{grp}:")
        for idx in indices:
            label = labels[idx]
            print(f"    Index {idx}: {label}")
        
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(group_dict, f, indent=4, ensure_ascii=False)
    df_local=df
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Freeze embedding layer
    for param in model.embeddings.parameters():
        param.requires_grad = False

    # Freeze first 8 encoder layers
    for layer in model.encoder.layer[:8]:
        for param in layer.parameters():
            param.requires_grad = False

    # Collect parameters from last 4 encoder layers and pooler
    trainable_params = []
    for layer in model.encoder.layer[8:]:
        trainable_params += list(layer.parameters())
    if hasattr(model, "pooler"):
        trainable_params += list(model.pooler.parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=learningrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin_number, p=2)
    
    group_names = list(group_dict.keys())

    def extract_avg_cls_vector(text, max_length=512):
        tokens = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
        input_ids = tokens['input_ids'][0]
        chunks = [input_ids[i:i + max_length] for i in range(0, len(input_ids), max_length)]
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
            return torch.zeros(model.config.hidden_size, device=device)
        stacked_cls = torch.cat(cls_vectors, dim=0)
        avg_cls = torch.mean(stacked_cls, dim=0)
        return avg_cls

    def get_group_cls_vectors(group_indices):
        group_all_cls_vectors = []
        for idx in group_indices:
            text = df_local.iloc[idx, 1]
            avg_cls = extract_avg_cls_vector(str(text))
            group_all_cls_vectors.append(avg_cls)
        return group_all_cls_vectors

    anchor_groups = [g for g in group_names if g != "dissimilar_group"]
    training_pairs = [
        (anchor_grp, negative_grp)
        for anchor_grp in anchor_groups
        for negative_grp in group_names
        if anchor_grp != negative_grp 
    ]  
    
    pair_cycle = itertools.cycle(training_pairs)  
    early_stop_count = 0 

    model_original = BertModel.from_pretrained('bert-base-uncased').to(device)
    model_original.eval()

    df_articles = df
    labels = df_articles.iloc[:, 0].values
    unique_labels = list(set(labels))
    colors = ['red', 'blue', 'green', 'purple','red']
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    print("group ict",group_dict)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0
        triplet_count = 0

        for anchor_group in group_names:
            anchor_indices = group_dict[anchor_group]
            anchor_cls_vectors = get_group_cls_vectors(anchor_indices)
            if len(anchor_cls_vectors) < 2:
                continue
        
            negative_cls_vectors = []
            for negative_group in group_names:
                if negative_group == anchor_group:
                    continue
                negative_indices = group_dict[negative_group]
                negative_cls_vectors.extend(get_group_cls_vectors(negative_indices))
            if len(negative_cls_vectors) == 0:
                continue
            neg_tensor_all = torch.stack(negative_cls_vectors, dim=0)

           
            for i in range(len(anchor_cls_vectors)):
                for j in range(len(anchor_cls_vectors)):
                    if i == j:
                        continue
                    anchor = anchor_cls_vectors[i]
                    positive = anchor_cls_vectors[j]
                    # hard negative sampling
                    anchor_expanded = anchor.unsqueeze(0)
                    distances = torch.norm(anchor_expanded - neg_tensor_all, dim=1)
                    ap_dist = torch.norm(anchor - positive)
                    mask = (distances > ap_dist) & (distances < ap_dist + margin_number)
                    valid_indices = torch.where(mask)[0]
                    if len(valid_indices) == 0:
                        neg_idx = torch.argmin(distances)
                    else:
                        masked_dist = distances[valid_indices]
                        neg_idx = valid_indices[torch.argmin(masked_dist)]
                    negative = neg_tensor_all[neg_idx]
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
            print("No valid triplets available for training")
        
        # Clear GPU memory
        clear_gpu_memory()
        
        if (epoch + 1) % 10 == 0:
            # Use relative path to save to project's Keyword_Group directory
            model_save_path_epoch = f"../Keyword_Group/bert_finetuned_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item()
            }, model_save_path_epoch)
            print(f"Model saved at epoch {epoch+1} -> {model_save_path_epoch}")
               

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item()
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

    model_original = BertModel.from_pretrained('bert-base-uncased').to(device)
    model_finetuned = BertModel.from_pretrained('bert-base-uncased').to(device)
    checkpoint = torch.load(model_save_path, map_location=device)
    model_finetuned.load_state_dict(checkpoint['model_state_dict'])
    model_finetuned.eval()

    if "df_global" not in globals():
     df_articles = pd.read_csv(csv_path)
    labels = df_articles.iloc[:, 0].values  
    unique_labels = list(set(labels))
    colors = ['red', 'blue', 'green', 'purple','red']
    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    cls_vectors_before = get_all_cls_vectors(df_articles, model_original,tokenizer, device).cpu()
    cls_vectors_after = get_all_cls_vectors(df_articles, model_finetuned,tokenizer, device).cpu()
    cls_vectors_after_cpu = cls_vectors_after.cpu().numpy()
    
    # Calculate perplexity parameter
    perplexity_before = min(30, max(5, len(cls_vectors_before) // 3))
    perplexity_after = min(30, max(5, len(cls_vectors_after_cpu) // 3))
    
    tsne_before = TSNE(n_components=2, perplexity=perplexity_before, random_state=42)
    tsne_after = TSNE(n_components=2, perplexity=perplexity_after, random_state=42)
    projected_2d_before = tsne_before.fit_transform(cls_vectors_before.numpy())
    projected_2d_after = tsne_after.fit_transform(cls_vectors_after_cpu)
    
    # Calculate center point for each keyword group
    group_centers = {}
    for group_name, indices in group_dict.items():
        if len(indices) > 0:
            # Get positions of all article indices in this group within df_articles
            group_article_indices = []
            for idx in indices:
                if idx in df_articles.index:
                    # Find corresponding row index in df_articles
                    article_idx = df_articles.index.get_loc(idx)
                    group_article_indices.append(article_idx)
            
            if len(group_article_indices) > 0:
                # Calculate average embedding of all articles in this group
                group_embeddings = cls_vectors_after_cpu[group_article_indices]
                group_center = np.mean(group_embeddings, axis=0)
                # Project center point to 2D space - use average of all 2D projections in the group
                group_2d_points = projected_2d_after[group_article_indices]
                group_center_2d = np.mean(group_2d_points, axis=0)
                group_centers[group_name] = group_center_2d


    # Create Plotly charts
    def create_plotly_figure(projected_2d, title, is_after=False, highlighted_indices=None):
        fig = go.Figure()
        
        # Add scatter plot - all points in same color (no label colors)
        article_indices = list(range(len(projected_2d)))
        hover_texts = [f"Article {idx}" for idx in article_indices]
        
        # Ensure customdata is in correct format
        custom_data = [[idx] for idx in article_indices]  # One list per point
        
        fig.add_trace(go.Scatter(
            x=projected_2d[:, 0],
            y=projected_2d[:, 1],
            mode='markers',
            name="Articles",
            marker=dict(
                color='lightblue',
                size=8,
                opacity=0.6,
                line=dict(width=0.5, color='white')
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
                fig.add_trace(go.Scatter(
                    x=highlighted_x,
                    y=highlighted_y,
                    mode='markers',
                    name="Highlighted Articles",
                    marker=dict(
                        color='gold',
                        size=12,
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    customdata=highlighted_customdata,
                    hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    hovertext=highlighted_hovertexts
                ))
        
        # If After Finetuning plot, add group center points
        if is_after and group_centers:
            center_colors = ['black', 'red', 'blue', 'green', 'purple', 'orange']
            for i, (group_name, center_2d) in enumerate(group_centers.items()):
                color = center_colors[i % len(center_colors)]
                fig.add_trace(go.Scatter(
                    x=[center_2d[0]],
                    y=[center_2d[1]],
                    mode='markers+text',
                    name=f'Center: {group_name}',
                    marker=dict(
                        color=color,
                        size=20,
                        symbol='star',
                        line=dict(width=2, color='white')
                    ),
                    text=[group_name],
                    textposition="top center",
                    textfont=dict(size=12, color=color),
                    hovertemplate=f'<b>Group Center: {group_name}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50),
            # Add black border
            xaxis=dict(
                linecolor='black',
                linewidth=2,
                mirror=True
            ),
            yaxis=dict(
                linecolor='black',
                linewidth=2,
                mirror=True
            )
        )
        
        return fig
    
    # Get highlighted indices from global state (if available)
    highlighted_indices = []
    try:
        # Try to get highlighted indices from global variables or stores
        # This will be updated by the UI callbacks
        pass
    except:
        pass
    
    # Generate two charts
    fig_before = create_plotly_figure(projected_2d_before, "2D Projection Before Finetuning", False, highlighted_indices)
    fig_after = create_plotly_figure(projected_2d_after, "2D Projection After Finetuning", True, highlighted_indices)
    
    return fig_before, fig_after

# Function to run training with highlighted indices
def run_training_with_highlights(highlighted_indices):
    """Run training and return figures with highlighted indices"""
    # This is a simplified version that reuses the main training function
    # but passes the highlighted indices to the figure creation
    fig_before, fig_after = run_training()
    
    # Update figures with highlights
    if fig_before and fig_after:
        # Add highlighted points to both figures
        for fig in [fig_before, fig_after]:
            if 'data' in fig and len(fig['data']) > 0:
                # Add highlighted trace
                highlighted_x = []
                highlighted_y = []
                highlighted_customdata = []
                highlighted_hovertexts = []
                
                # Get the main scatter trace data
                main_trace = fig['data'][0]
                if 'x' in main_trace and 'y' in main_trace:
                    for idx in highlighted_indices:
                        if 0 <= idx < len(main_trace['x']):
                            highlighted_x.append(main_trace['x'][idx])
                            highlighted_y.append(main_trace['y'][idx])
                            highlighted_customdata.append([idx])
                            highlighted_hovertexts.append(f"Article {idx}")
                    
                    if highlighted_x:  # Only add trace if there are highlighted points
                        fig['data'].append({
                            'x': highlighted_x,
                            'y': highlighted_y,
                            'mode': 'markers',
                            'name': 'Highlighted Articles',
                            'marker': {
                                'color': 'gold',
                                'size': 12,
                                'symbol': 'star',
                                'line': {'width': 2, 'color': 'white'}
                            },
                            'customdata': highlighted_customdata,
                            'hovertemplate': '<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                            'hovertext': highlighted_hovertexts
                        })
    
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

# Add necessary 2D visualization callbacks - split into two callbacks for performance
@app.callback(
    Output('keywords-2d-plot', 'figure'),
    Input('keywords-2d-plot', 'id')  # Only initial load
)
def update_keywords_2d_plot(plot_id):
    """Update keyword 2D dimensionality reduction visualization chart with text labels - initial load only"""
    global GLOBAL_OUTPUT_DICT, GLOBAL_KEYWORDS, _KEYWORD_TSNE_CACHE
    
    if not GLOBAL_OUTPUT_DICT or not GLOBAL_KEYWORDS:
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
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
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
            "Group 10": "#800080"
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
@app.callback(
    Output('documents-2d-plot', 'figure'),
    Input('documents-2d-plot', 'id'),  # Only initial load
    State('display-mode', 'data')
)
def update_documents_2d_plot_initial(plot_id, display_mode):
    """Initial load of documents 2D visualization - show all documents"""
    global df
    
    # Only process if we're in keywords mode
    if display_mode != "keywords":
        raise PreventUpdate
    
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
        print("🔍 Initial documents 2D visualization calculation...")
        all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
        
        # Calculate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(all_articles_text), batch_size):
            batch_texts = all_articles_text[i:i + batch_size]
            batch_embeddings = embedding_model_kw.encode(batch_texts, convert_to_tensor=True).to(device).cpu().numpy()
            all_embeddings.extend(batch_embeddings)
        
        document_embeddings = np.array(all_embeddings)
        
        # Calculate TSNE for documents
        print("🔍 Calculating TSNE for initial documents visualization...")
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        document_2d = tsne.fit_transform(document_embeddings)
        document_2d = document_2d.tolist()
        
        # Show all documents in one color
        traces = [{
            'x': [document_2d[i][0] for i in range(len(df))],
            'y': [document_2d[i][1] for i in range(len(df))],
            'mode': 'markers',
            'type': 'scatter',
            'name': 'All documents',
            'marker': {
                'size': 8,
                'color': '#2196F3',  # Blue color
                'opacity': 0.6,
                'line': {'width': 0.5, 'color': 'white'}
            },
            'text': [f'Doc {i+1}' for i in range(len(df))],
            'customdata': [[i] for i in range(len(df))],
            'hovertemplate': '<b>%{text}</b><extra></extra>'
        }]
        
        fig = {
            'data': traces,
            'layout': {
                'title': 'Documents 2D Visualization - All Documents',
                'xaxis': {'title': 'TSNE Dimension 1'},
                'yaxis': {'title': 'TSNE Dimension 2'},
                'hovermode': 'closest',
                'showlegend': True,
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50}
            }
        }
        
        print("🔍 Initial documents 2D visualization created successfully")
        return fig
        
    except Exception as e:
        print(f"Error creating initial documents 2D plot: {e}")
        return {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }

@app.callback(
    [Output('documents-2d-plot', 'figure', allow_duplicate=True),
     Output('highlighted-indices', 'data')],
    [Input('selected-keyword', 'data'),
     Input('selected-group', 'data'),  # Also update when group is selected
     Input('selected-article', 'data')],  # Also update when article is selected
    State('group-order', 'data'),  # Add group_order as State parameter
    State('display-mode', 'data'),
    prevent_initial_call=True
)
def update_documents_2d_plot(selected_keyword, selected_group, selected_article, display_mode, group_order):
    """Update documents 2D visualization chart"""
    global df, _DOCUMENTS_2D_CACHE
    
    # Only process if we're in keywords mode
    if display_mode != "keywords":
        raise PreventUpdate
    
    print(f"🔍 update_documents_2d_plot called with:")
    print(f"  selected_keyword: {selected_keyword}")
    print(f"  selected_group: {selected_group}")
    print(f"  selected_article: {selected_article}")
    print(f"  group_order: {group_order}")
    
    if 'df' not in globals():
        print("❌ No df in globals")
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
    elif selected_group and group_order:
        # For groups, create cache key based on group keywords
        for group_name, keywords in group_order.items():
            if group_name == selected_group:
                # Sort keywords for consistent cache key
                cache_key = f"docs_group:{group_name}:{':'.join(sorted(keywords))}"
                break
    else:
        cache_key = "docs_default"
    
    # Add selected article to cache key
    if selected_article is not None:
        cache_key = f"{cache_key}_article:{selected_article}"
    
    # Check cache first
    if cache_key and cache_key in _DOCUMENTS_2D_CACHE:
        print(f"Using cached documents 2D plot for: {cache_key}")
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
        return cached_fig, highlighted_indices
    
    try:
        # Calculate document embeddings
        print("🔍 Starting document embeddings calculation...")
        print(f"🔍 df shape: {df.shape}")
        all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
        print(f"🔍 Number of articles: {len(all_articles_text)}")
        
        # Calculate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(all_articles_text), batch_size):
            batch_texts = all_articles_text[i:i + batch_size]
            batch_embeddings = embedding_model_kw.encode(batch_texts, convert_to_tensor=True).to(device).cpu().numpy()
            all_embeddings.extend(batch_embeddings)
        
        document_embeddings = np.array(all_embeddings)
        
        # Calculate TSNE for documents
        print("🔍 Calculating TSNE for documents...")
        print(f"🔍 Embeddings shape: {document_embeddings.shape}")
        perplexity = min(30, max(5, len(document_embeddings) // 3))
        print(f"🔍 Perplexity: {perplexity}")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        document_2d = tsne.fit_transform(document_embeddings)
        print(f"🔍 TSNE result shape: {document_2d.shape}")
        print(f"🔍 TSNE result type: {type(document_2d)}")
        print(f"🔍 TSNE result dtype: {document_2d.dtype}")
        # Convert to list for Plotly compatibility
        document_2d = document_2d.tolist()
        
        # Determine which documents to highlight
        highlight_mask = []
        highlight_reason = ""
        
        if selected_keyword:
            # Highlight documents containing the selected keyword
            print(f"🔍 Processing keyword: '{selected_keyword}'")
            for i in range(len(df)):
                text = str(df.iloc[i, 1]).lower()
                contains_keyword = selected_keyword.lower() in text
                highlight_mask.append(contains_keyword)
                if contains_keyword:
                    print(f"  Document {i+1} contains keyword '{selected_keyword}'")
            highlight_reason = f"Documents containing '{selected_keyword}'"
            print(f"🔍 Found {sum(highlight_mask)} documents containing keyword '{selected_keyword}'")
        
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
        
        # Add trace for keyword/group highlighted documents
        if len(keyword_group_indices) > 0:
            traces.append({
                'x': [document_2d[i][0] for i in keyword_group_indices],
                'y': [document_2d[i][1] for i in keyword_group_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Keyword/Group matches',
                'marker': {
                    'size': 15,
                    'color': '#FFD700',  # Gold color for keyword/group matches
                    'symbol': 'star',
                    'line': {'width': 2, 'color': 'black'}
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
        
        # Add trace for other documents
        if len(other_indices) > 0:
            traces.append({
                'x': [document_2d[i][0] for i in other_indices],
                'y': [document_2d[i][1] for i in other_indices],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'Other documents',
                'marker': {
                    'size': 8,
                    'color': '#2196F3',  # Blue color for other documents
                    'opacity': 0.6,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                'text': [f'Doc {i+1}' for i in other_indices],
                'customdata': [[i] for i in other_indices],
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            })
        # If no traces created, show all documents in one color
        if not traces:
            traces = [{
                'x': [document_2d[i][0] for i in range(len(df))],
                'y': [document_2d[i][1] for i in range(len(df))],
                'mode': 'markers',
                'type': 'scatter',
                'name': 'All documents',
                'marker': {
                    'size': 8,
                    'color': '#2196F3',  # Blue color
                    'opacity': 0.6,
                    'line': {'width': 0.5, 'color': 'white'}
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
                'title': title,
                'xaxis': {'title': 'TSNE Dimension 1'},
                'yaxis': {'title': 'TSNE Dimension 2'},
                'hovermode': 'closest',
                'showlegend': True,
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50}
            }
        }
        
        # Collect highlighted indices for training plots
        highlighted_indices = []
        if len(keyword_group_indices) > 0:
            highlighted_indices.extend(keyword_group_indices.tolist())
        if len(selected_article_indices) > 0:
            highlighted_indices.extend(selected_article_indices.tolist())
        
        print(f"🔍 Final result:")
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
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'}
            }
        }, []

@app.callback(
    [Output("group-data", "data", allow_duplicate=True),
     Output("selected-keyword", "data", allow_duplicate=True)],
    Input("keywords-2d-plot", "clickData"),
    State("selected-group", "data"),
    State("group-data", "data"),
    prevent_initial_call=True
)
def handle_plot_click(click_data, selected_group, group_data):
    """Handle chart click events, select keyword for highlighting documents"""
    print(f"handle_plot_click called")
    if not click_data:
        print(f"handle_plot_click exit: no click_data")
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
            return new_data, clicked_keyword  # Return both group data and selected keyword
        else:
            # No group selected, just select the keyword for highlighting
            print(f"Selected keyword for highlighting: {clicked_keyword}")
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
    print(f"Train button clicked, n_clicks: {n_clicks}")
    
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
        
        # Save current group data to final_list.json
        print("Saving group data to final_list.json...")
        with open(final_list_path, "w", encoding="utf-8") as f:
            json.dump(group_order, f, indent=4, ensure_ascii=False)
        print(f"Group data saved to {final_list_path}")
        
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
        fig_before, fig_after = run_training()
        print("Training completed successfully!")
        
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
        
        return "Training Complete", completed_style, False, switch_button_style, "training", {"before": fig_before, "after": fig_after}
        
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
    [Output("article-fulltext-container", "children"),
     Output("selected-article", "data")],
    [Input("documents-2d-plot", "clickData"),
     Input({"type": "article-item", "index": ALL}, "n_clicks"),
     Input("display-mode", "data")],
    prevent_initial_call=True
)
def display_article_content_fulltext(click_data_docs, article_clicks, display_mode):
    """Display article content when clicking on documents 2D plot or recommended articles"""
    ctx = dash.callback_context
    
    # Only process if we're in keywords mode (documents-2d-plot exists)
    if display_mode != "keywords":
        raise PreventUpdate
    
    if not ctx.triggered:
        raise PreventUpdate
    
    # Determine which plot was clicked or if it's an article click
    click_data = None
    article_index = None
    
    if ctx.triggered[0]['prop_id'] == 'documents-2d-plot.clickData':
        click_data = click_data_docs
    elif 'article-item' in ctx.triggered[0]['prop_id']:
        # Handle article item clicks from recommended articles
        try:
            import json
            triggered_id = ctx.triggered[0]['prop_id']
            btn_info = json.loads(triggered_id.split('.')[0])
            article_index = btn_info.get("index")
            print(f"Article item clicked, index: {article_index}")
        except Exception as e:
            print(f"Error parsing article item click: {e}")
            raise PreventUpdate
    
    if not click_data and article_index is None:
        raise PreventUpdate
    
    try:
        # Get article index from click data or use the parsed article_index
        if click_data and article_index is None:
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
            
            return content, article_index
        else:
            return html.P("Article not found", style={"color": "red"}), None
    
    except Exception as e:
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"}), None

@app.callback(
    Output("article-fulltext-container", "children", allow_duplicate=True),
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
            
            return html.Div([
                html.H5(f"Article {article_index + 1} (Label: {article_label})", 
                       style={"color": "#2c3e50", "marginBottom": "10px"}),
                html.P(article_text, style={
                    "lineHeight": "1.6", 
                    "textAlign": "justify",
                    "fontSize": "14px",
                    "color": "#333"
                })
            ])
        else:
            return html.P("Article not found", style={"color": "red"})
    
    except Exception as e:
        return html.P(f"Error loading article: {str(e)}", style={"color": "red"})

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
    else:
        new_mode = "keywords"
        button_text = "Switch to Training View"
    
    print(f"Switching display mode from {current_mode} to {new_mode}")
    return new_mode, button_text

@app.callback(
    Output("main-visualization-area", "children"),
    [Input("display-mode", "data"),
     Input("training-figures", "data")],
    prevent_initial_call=True
)
def update_main_visualization_area(display_mode, training_figures):
    """Dynamically update the main visualization area based on display mode"""
    print(f"🔍 update_main_visualization_area called:")
    print(f"  display_mode: {display_mode}")
    print(f"  training_figures: {training_figures is not None}")
    
    if display_mode == "training" and training_figures:
        # Show training plots
        fig_before = training_figures.get("before", {})
        fig_after = training_figures.get("after", {})
        

        
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
        ]
    else:
        # In keywords mode, don't update the main visualization area
        # Let the individual callbacks handle the updates
        raise PreventUpdate

# Add callback for training mode highlighting
@app.callback(
    Output('highlighted-indices', 'data', allow_duplicate=True),
    [Input('selected-keyword', 'data'),
     Input('selected-group', 'data')],
    State('group-order', 'data'),
    State('training-figures', 'data'),
    State('display-mode', 'data'),
    prevent_initial_call=True
)
def update_training_highlights(selected_keyword, selected_group, group_order, training_figures, display_mode):
    """Update highlighted indices for training mode"""
    global df
    
    # Only process if we're in training mode
    if display_mode != "training":
        raise PreventUpdate
    
    # Only process if we have data and training figures
    if 'df' not in globals() or not training_figures:
        return []
    
    try:
        highlighted_indices = []
        
        if selected_keyword:
            # Find documents containing the selected keyword
            print(f"🔍 Finding documents for keyword: {selected_keyword}")
            keyword_indices = []
            for i, text in enumerate(df.iloc[:, 1]):
                if selected_keyword.lower() in str(text).lower():
                    keyword_indices.append(i)
            highlighted_indices = keyword_indices
            print(f"🔍 Found {len(keyword_indices)} documents for keyword '{selected_keyword}'")
            
        elif selected_group and group_order:
            # Find documents containing any keyword in the selected group
            print(f"🔍 Finding documents for group: {selected_group}")
            group_keywords = group_order.get(selected_group, [])
            group_indices = []
            for i, text in enumerate(df.iloc[:, 1]):
                text_lower = str(text).lower()
                for keyword in group_keywords:
                    if keyword.lower() in text_lower:
                        group_indices.append(i)
                        break  # Only add once per document
            highlighted_indices = list(set(group_indices))  # Remove duplicates
            print(f"🔍 Found {len(highlighted_indices)} documents for group '{selected_group}'")
        
        return highlighted_indices
        
    except Exception as e:
        print(f"Error updating training highlights: {e}")
        return []

# Add callback for updating training plots with highlights
@app.callback(
    [Output('plot-before', 'figure', allow_duplicate=True),
     Output('plot-after', 'figure', allow_duplicate=True)],
    [Input('highlighted-indices', 'data'),
     Input('display-mode', 'data')],
    State('training-figures', 'data'),
    prevent_initial_call=True
)
def update_training_plots_with_highlights(highlighted_indices, display_mode, training_figures):
    """Update training plots with highlighted indices"""
    
    # Only process if we're in training mode
    if display_mode != "training":
        raise PreventUpdate
    
    # Only process if we have training figures and highlighted indices
    if not training_figures or not highlighted_indices or len(highlighted_indices) == 0:
        # Return original figures
        fig_before = training_figures.get("before", {}) if training_figures else {}
        fig_after = training_figures.get("after", {}) if training_figures else {}
        return fig_before, fig_after
    
    try:
        # Update training figures with highlights
        fig_before, fig_after = run_training_with_highlights(highlighted_indices)
        return fig_before, fig_after
    except Exception as e:
        print(f"Error updating training plots with highlights: {e}")
        # Return original figures if highlighting fails
        fig_before = training_figures.get("before", {})
        fig_after = training_figures.get("after", {})
        return fig_before, fig_after

# Launch the Dash application
if __name__ == "__main__":
    print(" http://127.0.0.1:8053 ")
    app.run(debug=True, dev_tools_hot_reload=False, port=8053)
