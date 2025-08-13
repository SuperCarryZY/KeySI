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
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
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

# Add BertTopic import
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
    print("BertTopic available, using BertTopic for keyword dimensionality reduction")
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("BertTopic not available, using t-SNE as fallback")
    print("To use BertTopic, run: pip install bertopic")
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
            print(f"NLTK package '{package_name}' already exists")
        except LookupError:
            print(f"NLTK package '{package_name}' not found, downloading...")
            try:
                nltk.download(package_name, quiet=True)
                print(f"NLTK package '{package_name}' download completed")
            except Exception as e:
                print(f"Failed to download '{package_name}': {e}")
                # For punkt_tab, if download fails, try punkt as fallback
                if package_name == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                        print("Using 'punkt' as fallback")
                    except:
                        pass
                        
    # Download some additional common packages just in case
    additional_packages = ['stopwords', 'wordnet', 'omw-1.4']
    for package in additional_packages:
        try:
            nltk.download(package, quiet=True)
        except:
            pass  # Ignore if download fails

# Call function to ensure packages exist
ensure_nltk_data()

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster




# Set device and global parameters
# Auto-detect and select best device: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"NVIDIA GPU detected, using CUDA acceleration")
    print(f"  GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif torch.backends.mps.is_available():
    device = "mps"
    print(f"Apple Silicon GPU detected, using MPS acceleration")
else:
    device = "cpu"
    print(f"Using CPU - GPU recommended for better performance")

print(f"Using device: {device}")

# GPU memory optimization settings
if device == "cuda":
    # Enable CUDA memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("CUDA memory optimization enabled")
    
    # Show current GPU memory usage
    if torch.cuda.is_available():
        print(f"  Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  GPU memory cache: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

def clear_gpu_memory():
    """Clear GPU memory"""
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("GPU memory cleared")
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
num_epochs = 50
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
    print(f"GPU mode: using larger batch_size = {batch_size}")
elif device == "mps":
    batch_size = 128  # Apple Silicon GPU moderate batch size
    print(f"MPS mode: using batch_size = {batch_size}")
else:
    batch_size = 64   # CPU uses smaller batch size
    print(f"CPU mode: using batch_size = {batch_size}")

clusterthreshold = 25


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
            print(f"Folder created: {directory}")


ensure_directories()

# Model settings
print("Initializing SentenceTransformer model...")
print("First run may require downloading pre-trained models, please wait...")
try:
    embedding_model_kw = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)
    print("SentenceTransformer model initialization completed")
except Exception as e:
    print(f"Model initialization failed: {e}")
    raise

print("Initializing KeyBERT model...")
kw_model = KeyBERT(model=embedding_model_kw)
print("KeyBERT model initialization completed")

print("Loading data...")
ps = PorterStemmer()
word_count = Counter()
original_form = {}
df = pd.read_csv(csv_path)
all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
labels = df.iloc[:, 0].values 
print(f"Data loading completed, total {len(all_articles_text)} articles")

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
        print(f"  Processing batch {i//batch_size + 1}/{total_batches}, containing {len(batch)} articles")
        
        try:
          
            print(f"    GPU batch processing... memory usage:")
            if device == "cuda":
                print(f"      GPU memory used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"      GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            

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
            
            print(f"    Batch completed, GPU utilization optimized")
            
        except Exception as e:
            print(f"  Batch processing failed: {e}")
            results.extend([None] * len(batch))
    
    return results

print("Extracting keywords...")
print("Using GPU batch processing optimization, fully utilizing RTX 5090 performance...")

# Batch preprocessing
processed_articles, valid_indices = preprocess_articles_batch(all_articles_text)
print(f"Preprocessing completed, valid articles: {len(processed_articles)}")

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

print("Counting keywords...")
for res in results:
    if res:
        for stemmed, kw in res:
            word_count[stemmed] += 1
            if stemmed not in original_form or len(kw) < len(original_form[stemmed]):
                original_form[stemmed] = kw

filtered_keywords = [original_form[stem] for stem, count in word_count.items() if count >= word_count_threshold]
if not filtered_keywords:
    raise ValueError("No keywords found with the specified frequency threshold.")

print(f"Keyword extraction completed, total {len(filtered_keywords)} keywords")

print("Computing keyword embeddings...")
keyword_embeddings = embedding_model_kw.encode(filtered_keywords, convert_to_tensor=True).to(device).cpu().numpy()
print("Keyword embedding calculation completed")

print("Performing keyword dimensionality reduction...")
if BERTOPIC_AVAILABLE:
    try:
        print("Using BertTopic for keyword dimensionality reduction...")
        # Use BertTopic for dimensionality reduction
        topic_model = BERTopic(
            nr_topics="auto",
            top_n_words=20,
            min_topic_size=2,
            verbose=True
        )
        
        # Convert keywords to document format (each keyword as a document)
        keyword_docs = filtered_keywords  # Use string list directly, not nested list
        
        # Use BertTopic for topic modeling and dimensionality reduction
        topics, probs = topic_model.fit_transform(keyword_docs)
        
        # Get dimensionality reduced coordinates - use correct API
        try:
            # Get topic embeddings
            topic_embeddings = topic_model.topic_embeddings_
            if topic_embeddings is not None:
                print(f"BertTopic dimensionality reduction completed, topic count: {len(topic_embeddings)}")
                # Convert topic embeddings to 2D coordinates
                reduced_embeddings = topic_embeddings[:, :2]  # Take first two dimensions
                if len(reduced_embeddings) < len(filtered_keywords):
                    # If topic count is less than keyword count, expansion needed
                    print(f"Topic count ({len(reduced_embeddings)}) is less than keyword count ({len(filtered_keywords)}), falling back to t-SNE...")
                    raise Exception("BertTopic topic count insufficient")
            else:
                print("BertTopic dimensionality reduction failed, topic embeddings are empty")
                raise Exception("BertTopic dimensionality reduction failed")
        except Exception as e:
            print(f"Failed to get topic embeddings: {e}")
            raise Exception("BertTopic dimensionality reduction failed")
            
    except Exception as e:
        print(f"BertTopic dimensionality reduction error: {e}")
        print("Falling back to t-SNE dimensionality reduction...")
        BERTOPIC_AVAILABLE = False

if not BERTOPIC_AVAILABLE:
    print("Using t-SNE for keyword dimensionality reduction...")
    perplexity = min(30, max(5, len(keyword_embeddings) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = tsne.fit_transform(keyword_embeddings)
    print("t-SNE dimensionality reduction completed")

print("Performing hierarchical clustering...")
linkage_matrix = linkage(reduced_embeddings, method="ward")
labels_hierarchical = fcluster(linkage_matrix, max_d, criterion="distance")
print("Hierarchical clustering completed")

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

print(f"Keyword clustering completed, {total_clusters} categories found")
print("Starting Web Application...")

# Ensure keyword variables are defined
if 'keywords' not in locals():
    keywords = []
    print("Warning: Keyword list is empty, using default values")

# Build Dash layout with UI components like inputs, buttons, and containers
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

def create_layout():
    """Create Dash layout - keyword 2D dimensionality reduction visualization version"""
    return html.Div([
        html.H3("Keyword Grouping - 2D Visualization"),
        
        # Top control area
        html.Div([
                    html.Label("Enter number of groups:"),
        dcc.Input(id="group-count", type="number", value=3, min=1, step=1),
        html.Button("Generate Groups", id="generate-btn", n_clicks=0),
        ], style={"marginBottom": "20px"}),
        
        # Manual keyword addition area
        html.Div([
            dcc.Input(
                id='new-keyword-input',
                type='text',
                placeholder='Enter a keyword',
                style={"marginRight": "10px"}
            ),
            html.Button("Add Keyword", id="add-keyword-btn", n_clicks=0)
        ], style={"marginBottom": "20px"}),

        # Data storage
        dcc.Store(id="group-data", data={kw: None for kw in (keywords if 'keywords' in globals() else [])}),
        dcc.Store(id="selected-group", data=None),
        dcc.Store(id="group-order", data={}),
        dcc.Store(id="selected-file", data=None),
        dcc.Store(id="selected-keyword", data=None),
        dcc.Store(id="articles-data", data=[]),  # Store article data
        
        # Main content area - left-right column layout
        html.Div([
            # Left: keyword 2D dimensionality reduction visualization
            html.Div([
                html.H4("Keywords 2D Visualization"),
                html.P("Hover over points to see keywords, click to select"),
                dcc.Graph(
                    id='keywords-2d-plot',
                    style={'height': '600px'},
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            ], style={
                'width': '60%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px',
                'border': '2px solid black'
            }),
            
            # Right: group management
            html.Div([
                html.H4("Group Management"),
                html.Div(id="group-containers", style={
                    "display": "flex",
                    "flex-direction": "column",
                    "gap": "10px",
                    "margin-bottom": "20px"
                }),
            ], style={
                'width': '40%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px'
            })
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Recommended files and details area - left-right column layout
        html.Div([
            # Left: recommended files list
            html.Div([
                html.H4("Recommended Files"),
                html.P("Based on selected group keywords, showing files containing these keywords"),
                html.Div(id="recommended-files-container", style={
                    "border": "1px solid #ddd",
                    "padding": "15px",
                    "backgroundColor": "#f9f9f9",
                    "minHeight": "400px",
                    "maxHeight": "600px",
                    "overflowY": "auto"
                })
            ], style={
                'width': '60%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px'
            }),
            
            # Right: file details and actions
            html.Div([
                html.H4("File Details"),
                html.Div([
                    html.P("Click on left files to view details", style={"color": "#666", "fontStyle": "italic"}),
                    html.H5("Actions:", style={"marginTop": "20px"}),
                    html.Div([
                        html.Button("View Text", 
                                  id="view-text-btn", 
                                  style={"margin": "5px", "padding": "8px 15px", "width": "100%"})
                    ], style={"marginBottom": "15px"}),
                    html.Div(id="file-details-content", style={
                        "marginTop": "15px",
                        "border": "1px solid #ddd",
                        "padding": "15px",
                        "backgroundColor": "#f9f9f9",
                        "minHeight": "300px",
                        "maxHeight": "400px",
                        "overflowY": "auto"
                    })
                ], style={
                    "border": "1px solid #ddd",
                    "padding": "15px",
                    "backgroundColor": "#fff",
                    "minHeight": "400px"
                })
            ], style={
                'width': '40%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px'
            })
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        # Training button and output
        html.Button("Train", id="train-btn", n_clicks=0, style={"margin-top": "20px"}),
        
        # Training result visualization - two side-by-side large plots
        html.Div(id="train-output", children=[
            html.Div([
                dcc.Graph(id="plot-before", style={"height": "70vh", "width": "100%"}),
            ], style={"width": "49%", "display": "inline-block", "vertical-align": "top", "margin-right": "1%"}),
            html.Div([
                dcc.Graph(id="plot-after", style={"height": "70vh", "width": "100%"}),
            ], style={"width": "49%", "display": "inline-block", "vertical-align": "top", "margin-left": "1%"}),
        ], style={"width": "100%", "height": "70vh", "display": "none", "border": "2px solid black", "padding": "10px"}),
        
        # Article content display area
        html.Div([
            html.H4("Article Content", style={"margin-bottom": "10px"}),
            html.Div(id="article-content", children="Click on a point in the plots above to view article content", 
                    style={
                        "border": "1px solid #ddd",
                        "padding": "15px",
                        "background-color": "#f9f9f9",
                        "max-height": "200px",
                        "overflow-y": "auto",
                        "margin-top": "10px"
                    })
        ], id="content-display", style={"display": "none", "margin-top": "20px"}),
        
        # Debug output area
        html.Div(id="debug-output", style={"marginTop": "20px"})
    ])

# Set application layout
app.layout = create_layout()

@app.callback(
    [
        Output("group-data", "data"),
        Output({"type": "keyword-btn", "index": ALL}, "style"),
        Output("selected-keyword", "data")
    ],
    [
        Input({"type": "keyword-btn", "index": ALL}, "n_clicks")
    ],
    [
        State("selected-group", "data"),
        State("group-data", "data")
    ]
)
def update_keyword_styles(keyword_clicks, selected_group, group_data):
    """Server-side callback: update keyword button styles"""
    from dash import callback_context
    
    if not callback_context.triggered:
        raise dash.exceptions.PreventUpdate
    
    # Safety check: if no keyword buttons, return empty styles
    if not keyword_clicks:
        return group_data, []
    
    triggered = callback_context.triggered[0]
    
    new_data = dict(group_data) if group_data else {}
    
    # Default style - includes complete button styling
    base_style = {
        "width": "120px",
        "height": "40px",
        "margin": "40x",
        "textAlign": "center",
        "lineHeight": "40px",
        "borderRadius": "8px",
        "fontSize": "16px",
        "display": "inline-block",
        "verticalAlign": "middle"
    }
    
    default_style = {**base_style, 'backgroundColor': '#f0f0f0', 'color': 'black'}
    selected_style = {**base_style, 'backgroundColor': '#4CAF50', 'color': 'white'}
    
    # Handle keyword button clicks - fix: only modify data when keyword button is actually clicked
    try:
        import json
        btn_id = json.loads(triggered['prop_id'].split('.')[0])
        if btn_id.get('type') == 'keyword-btn':
            kw = btn_id['index']
            # Fix: implement accumulative add logic - always add to selected group, no toggling
            if selected_group:
                if kw in new_data and new_data[kw]:
                    if new_data[kw] != selected_group:
                        print(f"Moved keyword '{kw}' from group '{new_data[kw]}' to group '{selected_group}'")
                    else:
                        print(f"Keyword '{kw}' is already in group '{selected_group}'")
                else:
                    print(f"Added keyword '{kw}' to group '{selected_group}'")
                new_data[kw] = selected_group
                # Fix: when adding keywords, don't auto-select them, keep showing entire group files
                selected_keyword = None
    except (json.JSONDecodeError, KeyError):
        # If not a keyword button click, don't modify data
        selected_keyword = None
    
    # Generate style array
    styles = []
    global_keywords = globals().get('GLOBAL_KEYWORDS', [])
    for i, kw in enumerate(global_keywords):
        if i < len(keyword_clicks):
            style = selected_style if new_data.get(kw) else default_style
            styles.append(style)
    
    return new_data, styles, selected_keyword

@app.callback(
    Output("group-data", "data", allow_duplicate=True),
    Input("add-keyword-btn", "n_clicks"),
    State("group-data", "data"),
    State("new-keyword-input", "value"),
    State("selected-group", "data"),
    prevent_initial_call=True
)
def add_keywords(add_clicks, group_data, new_kw, selected_group):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    if not new_kw or new_kw.strip() == "":
        raise dash.exceptions.PreventUpdate

    if not selected_group:
        raise dash.exceptions.PreventUpdate

    new_data = dict(group_data)
    keywords = re.split(r"[;；]", new_kw)
    
    for kw in keywords:
        cleaned_kw = kw.strip()
        if cleaned_kw:
            new_data[cleaned_kw] = selected_group

    return new_data

@app.callback(
    Output('keywords-2d-plot', 'figure'),
    [Input('group-data', 'data'),
     Input('selected-group', 'data')]
)
def update_keywords_2d_plot(group_data, selected_group):
    """Update keyword 2D dimensionality reduction visualization chart"""
    global GLOBAL_OUTPUT_DICT, GLOBAL_KEYWORDS
    
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
        # Use previously calculated keyword embeddings and t-SNE results
        keyword_embeddings = embedding_model_kw.encode(GLOBAL_KEYWORDS, convert_to_tensor=True).to(device).cpu().numpy()
        
        # Recalculate t-SNE (or use previous results)
        perplexity = min(30, max(5, len(keyword_embeddings) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_embeddings = tsne.fit_transform(keyword_embeddings)
        
        # Assign colors to each keyword (based on cluster category and grouping status)
        colors = []
        hover_texts = []
        
        # Define category colors (one color per category)
        category_colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
        ]
        
        for kw in GLOBAL_KEYWORDS:
            # First check if in user grouping
            if group_data and kw in group_data and group_data[kw]:
                # Grouped keywords - use user group color
                colors.append('#4CAF50')  # Green
                hover_texts.append(f"{kw}<br>User Group: {group_data[kw]}")
            else:
                # Ungrouped keywords - use cluster category color
                # Find which cluster the keyword is in
                category_found = False
                for cluster_name, cluster_keywords in GLOBAL_OUTPUT_DICT.items():
                    if kw in cluster_keywords:
                        # Calculate category index to get color
                        cluster_index = int(cluster_name.replace('cluster', '')) - 1
                        color_index = cluster_index % len(category_colors)
                        colors.append(category_colors[color_index])
                        hover_texts.append(f"{kw}<br>Cluster: {cluster_name}")
                        category_found = True
                        break
                
                if not category_found:
                    # If no category found, use default color
                    colors.append('#2196F3')  # Blue
                    hover_texts.append(f"{kw}<br>Uncategorized")
        
        # Create scatter plot
        fig = {
            'data': [{
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'mode': 'markers',
                'type': 'scatter',
                'marker': {
                    'size': 12,
                    'color': colors,
                    'line': {'width': 1, 'color': 'white'}
                },
                'text': hover_texts,
                'hoverinfo': 'text',
                'customdata': GLOBAL_KEYWORDS,  # For click events
                'hovertemplate': '<b>%{text}</b><extra></extra>'
            }],
            'layout': {
                'title': 'Keywords 2D Visualization - BertTopic + Clustering Colors (Hover to see keywords, click to select)',
                'xaxis': {'title': 'Dimension 1'},
                'yaxis': {'title': 'Dimension 2'},
                'hovermode': 'closest',
                'clickmode': 'event+select',
                'dragmode': 'pan',
                'showlegend': False,
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
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

@app.callback(
    Output("group-order", "data"),
    [
        Input("generate-btn", "n_clicks"),
        Input("group-data", "data"),
        Input({"type": "remove-keyword", "group": ALL, "index": ALL}, "n_clicks"),
    ],
    [
        State("group-count", "value"),
        State("group-order", "data")
    ],
    prevent_initial_call=True
)
def update_group_order(generate_n_clicks, group_data, remove_clicks, num_groups, current_order):
    ctx = dash.callback_context
    print(f"🟠 update_group_order called")
    print(f"🟠 triggered: {ctx.triggered}")
    print(f"🟠 group_data: {group_data}")
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    new_order = dict(current_order) if current_order else {}
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "generate-btn":
        if not num_groups or num_groups < 1:
            raise dash.exceptions.PreventUpdate
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

    elif triggered_id.startswith("{"):
        try:
            # Check if it's a real button click (prevent auto-trigger)
            triggered_value = ctx.triggered[0]['value']
            if triggered_value is None:
                print(f"🟠 Skip delete operation: n_clicks is None, may be auto-triggered")
                raise dash.exceptions.PreventUpdate
                
            btn_info = json.loads(triggered_id)
            grp_name = btn_info.get("group")
            action = btn_info.get("type")
            idx = btn_info.get("index")

            print(f"🟠 Real delete button clicked: group={grp_name}, action={action}, index={idx}")

            if grp_name not in new_order or idx is None:
                raise dash.exceptions.PreventUpdate
            
            kw_list = list(new_order[grp_name])

            if action == "remove-keyword":
                # Keyword removal logic is already handled in remove_keyword_from_group
                # Here just need to update group order
                if idx < len(kw_list):
                    kw_list.pop(idx)
                    new_order[grp_name] = kw_list
                    return new_order

            new_order[grp_name] = kw_list
            return new_order
        except Exception as e:
            print(f"Error handling move button: {e}")
            raise dash.exceptions.PreventUpdate

    return new_order


@app.callback(
    [Output("group-data", "data", allow_duplicate=True),
     Output("selected-keyword", "data", allow_duplicate=True)],
    Input("keywords-2d-plot", "clickData"),
    State("selected-group", "data"),
    State("group-data", "data"),
    prevent_initial_call=True
)
def handle_plot_click(click_data, selected_group, group_data):
    """Handle chart click events, add keywords to selected group"""
    print(f"🔴 handle_plot_click called")
    if not click_data or not selected_group:
        print(f"🔴 handle_plot_click exit: click_data={click_data}, selected_group={selected_group}")
        raise PreventUpdate
    
    try:
        # Get clicked keyword
        clicked_keyword = click_data['points'][0]['customdata']
        print(f"🔴 Clicked keyword: {clicked_keyword}")
        print(f"🔴 Current group_data: {group_data}")
        print(f"🔴 Selected group: {selected_group}")
        
        # Update grouping data - fix: implement accumulative add logic
        new_data = dict(group_data) if group_data else {}
        
        # Fix: implement accumulative add logic - always add to selected group, no toggling
        if clicked_keyword in new_data and new_data[clicked_keyword]:
            if new_data[clicked_keyword] != selected_group:
                print(f"Moved keyword '{clicked_keyword}' from group '{new_data[clicked_keyword]}' to group '{selected_group}'")
            else:
                print(f"Keyword '{clicked_keyword}' is already in group '{selected_group}'")
        else:
            print(f"Added keyword '{clicked_keyword}' to group '{selected_group}'")
        new_data[clicked_keyword] = selected_group
        
        print(f"🔴 Returned new_data: {new_data}")
        print(f"🔴 Added keyword but don't auto-select, keep showing entire group files")
        return new_data, None  # Fix: when adding keywords, don't auto-select them, keep showing entire group files
        
    except Exception as e:
        print(f"Error handling plot click: {e}")
        raise PreventUpdate


@app.callback(
    Output("group-containers", "children"),
    [Input("group-order", "data"),
     Input("selected-group", "data"),
     Input("selected-keyword", "data")]
)
def render_groups(group_order, selected_group, selected_keyword):
    print(f"🟣 render_groups called")
    print(f"🟣 group_order: {group_order}")
    print(f"🟣 selected_group: {selected_group}")
    print(f"🟣 selected_keyword: {selected_keyword}")
    if not group_order:
        return []

    children = []
    for grp_name, kw_list in group_order.items():
        # Fix: always show all group keywords, not dependent on selected_group
        group_header = html.Button(
            grp_name,
            id={"type": "group-header", "index": grp_name},
            style={
                "width": "200px",
                "background": "#4CAF50" if grp_name == selected_group else "#f0f0f0",
                "border": "none",
                "padding": "8px",
                "cursor": "pointer"
            }
        )

        group_keywords = []
        for i, kw in enumerate(kw_list):
            # Check if this keyword is selected
            is_selected = selected_keyword and kw == selected_keyword
            
            keyword_button = html.Button(
                kw,
                id={"type": "select-keyword", "keyword": kw, "group": grp_name},
                style={
                    "padding": "5px 10px", 
                    "margin": "3px", 
                    "border": "1px solid #ddd", 
                    "flex": "1",
                    "backgroundColor": "#007bff" if is_selected else "#f8f9fa",
                    "color": "white" if is_selected else "black",
                    "cursor": "pointer",
                    "borderRadius": "4px"
                }
            )
            
            keyword_item = html.Div([
                keyword_button,
                html.Button("×", id={"type": "remove-keyword", "group": grp_name, "index": i}, 
                           style={"margin": "2px", "padding": "2px 6px", "fontSize": "12px", "color": "red"})
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"})
            group_keywords.append(keyword_item)

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "10px",
                "minHeight": "100px",
                "maxHeight": "300px",
                "overflowY": "auto",
                "backgroundColor": "#f9f9f9"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"display": "inline-block", "margin": "5px"}
        )
        children.append(group_container)

    return children

def extract_top_keywords(text, top_k=5):
    """Extract top N keywords from text"""
    try:
        global kw_model
        if 'kw_model' in globals() and kw_model:
            # Use KeyBERT to extract keywords
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), 
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

@app.callback(
    Output("recommended-files-container", "children"),
    [Input("group-order", "data"),
     Input("selected-group", "data"),
     Input("selected-keyword", "data")]
)
def update_recommended_files(group_order, selected_group, selected_keyword):
    """Generate recommended file list based on selected keywords, default shows files corresponding to all keywords in group"""
    
    if not group_order or not selected_group:
        return html.P("Please select a keyword group first")
    
    # Determine keywords to search
    if selected_keyword:
        # If specific keyword selected, only search this keyword
        search_keywords = [selected_keyword]
        search_info = f"Files containing keyword '{selected_keyword}'"
        print(f"🔍 Search specific keyword: {selected_keyword}")
    else:
        # If no specific keyword selected, search all keywords in group
        search_keywords = group_order.get(selected_group, [])
        search_info = f"Files containing '{selected_group}' group keywords"
        print(f"🔍 Search entire group keywords: {search_keywords}")
    
    if not search_keywords:
        return html.P(f"No keywords in group '{selected_group}'")
    
    try:
        # Use global df variable to search files
        global df
        if 'df' not in globals():
            return html.P("Data not loaded")
        
        # Search for files containing keywords
        recommended_files = []
        for idx, row in df.iterrows():
            file_text = str(row.iloc[1]) if len(row) > 1 else ""
            file_number = idx + 1  # File number starts from 1
            
            # Check if file contains any keywords
            should_include = any(keyword.lower() in file_text.lower() for keyword in search_keywords)
            
            if should_include:
                # Extract top 5 keywords from file
                file_keywords = extract_top_keywords(file_text, 5)
                recommended_files.append({
                    'file_number': file_number,
                    'file_index': idx,
                    'keywords': file_keywords
                })
        
        if not recommended_files:
            return html.P(f"No {search_info} found")
        
      
        file_items = [
            html.H6(f"{search_info} (Found {len(recommended_files)} files)", 
                   style={"color": "#2c3e50", "marginBottom": "15px"})
        ]
        for file_info in recommended_files:
            # Create keyword tags
            keyword_tags = []
            for keyword in file_info['keywords']:
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
            
            file_item = html.Div([
                html.Span(f"File {file_info['file_number']}", 
                         style={"fontWeight": "bold", "marginRight": "10px", "minWidth": "80px"}),
                html.Div(keyword_tags, style={"flex": "1", "display": "flex", "flexWrap": "wrap"}),
                html.Button("View Details", 
                           id={"type": "view-file", "index": file_info['file_index']},
                           style={"marginLeft": "10px", "padding": "2px 8px", "fontSize": "12px"})
            ], style={"padding": "8px", "borderBottom": "1px solid #eee", "display": "flex", "alignItems": "center"})
            file_items.append(file_item)
        
        return html.Div([
            html.P(f"Found {len(recommended_files)} files containing keywords:"),
            html.Div(file_items)
        ])
        
    except Exception as e:
        print(f"Error generating recommended files: {e}")
        return html.P(f"Error generating recommended files: {str(e)}")

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
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']

    if not triggered_n_clicks or triggered_n_clicks is None:
        raise dash.exceptions.PreventUpdate

    selected_group = json.loads(triggered_id.split('.')[0])["index"]
    print(f"Switch to group: {selected_group}")  # Add debug info
    print(f"🔵 Clear selected keyword")
    return selected_group, None  # Clear selected keyword when switching groups


@app.callback(
    Output("selected-keyword", "data", allow_duplicate=True),
    Input({"type": "select-keyword", "keyword": ALL, "group": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def select_keyword_from_group(n_clicks):
    """Handle keyword selection from group management"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']
    
    if triggered_n_clicks and "select-keyword" in triggered_id:
        try:
            import json
            btn_info = json.loads(triggered_id.split('.')[0])
            keyword = btn_info.get("keyword")
            print(f"🔶 Select keyword from group management: {keyword}")
            return keyword
        except:
            raise PreventUpdate
    
    raise PreventUpdate



@app.callback(
    Output("group-data", "data", allow_duplicate=True),
    Input({"type": "remove-keyword", "group": ALL, "index": ALL}, "n_clicks"),
    State("group-order", "data"),
    State("group-data", "data"),
    prevent_initial_call=True
)
def remove_keyword_from_group(remove_clicks, group_order, group_data):
    """Remove keyword from group"""
    ctx = dash.callback_context
    print(f"🟢 remove_keyword_from_group called")
    print(f"🟢 triggered: {ctx.triggered}")
    if not ctx.triggered:
        raise PreventUpdate
    
    try:
        # Get clicked button info
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        triggered_value = ctx.triggered[0]['value']
        
        # Guard: only execute delete operation when n_clicks has value
        if triggered_value is None:
            print(f"🟢 Skip delete operation: n_clicks is None, may be auto-triggered")
            raise PreventUpdate
        
        btn_info = json.loads(triggered_id)
        group_name = btn_info.get("group")
        keyword_index = btn_info.get("index")
        
        print(f"🟢 Delete button really clicked: group={group_name}, index={keyword_index}, n_clicks={triggered_value}")
        
        if not group_name or keyword_index is None:
            raise PreventUpdate
        
        # Remove keyword from group order
        new_order = dict(group_order) if group_order else {}
        if group_name in new_order and keyword_index < len(new_order[group_name]):
            removed_keyword = new_order[group_name].pop(keyword_index)
            
            # Remove from grouping data
            new_data = dict(group_data) if group_data else {}
            if removed_keyword in new_data:
                del new_data[removed_keyword]
            
            print(f"Removed keyword '{removed_keyword}' from group '{group_name}'")
            return new_data
        
        raise PreventUpdate
        
    except Exception as e:
        print(f"🟢 Error removing keyword: {e}")
        raise PreventUpdate

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
            print(f"✅ Model saved at epoch {epoch+1} -> {model_save_path_epoch}")
               

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
    def create_plotly_figure(projected_2d, title, is_after=False):
        fig = go.Figure()
        
        # Add scatter plot - each category
        for label in unique_labels:
            mask = (labels == label)
            if np.any(mask):
                # Add article index as hover info for each point
                article_indices = np.where(mask)[0]
                hover_texts = [f"Article {idx}" for idx in article_indices]
                
                # Ensure customdata is in correct format
                custom_data = [[idx] for idx in article_indices]  # One list per point
                
                fig.add_trace(go.Scatter(
                    x=projected_2d[mask, 0],
                    y=projected_2d[mask, 1],
                    mode='markers',
                    name=f"Class {label}",
                    marker=dict(
                        color=label_to_color[label],
                        size=8,
                        opacity=0.6,
                        line=dict(width=0.5, color='white')
                    ),
                    customdata=custom_data,
                    hovertemplate='<b>%{hovertext}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    hovertext=hover_texts
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
    
    # Generate two charts
    fig_before = create_plotly_figure(projected_2d_before, "2D Projection Before Finetuning", False)
    fig_after = create_plotly_figure(projected_2d_after, "2D Projection After Finetuning", True)
    
    return fig_before, fig_after
#UI-design
@app.callback(
    Output("train-btn", "style"),
    Input("train-btn", "n_clicks")
)
def update_train_btn_style(n_clicks):
    if n_clicks and n_clicks > 0:
        return {"margin-top": "10px", "backgroundColor": "green", "color": "white"}
    return {"margin-top": "10px"}

@app.callback(
    [Output("plot-before", "figure"),
     Output("plot-after", "figure"),
     Output("train-output", "style"),
     Output("content-display", "style")],
    Input("train-btn", "n_clicks"),
    State("group-order", "data")
)
def train_and_plot(n_clicks, group_order):
    if not n_clicks:
        raise PreventUpdate

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            print(group_order)
            json.dump(group_order, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error saving group order:", e)

    # Run training and get two charts
    fig_before, fig_after = run_training()
    
    # Show training output and content display areas
    train_output_style = {"width": "100%", "height": "70vh", "display": "block"}
    content_display_style = {"display": "block", "margin-top": "20px"}
    
    return fig_before, fig_after, train_output_style, content_display_style

# Handle chart click events, display article content
@app.callback(
    Output("article-content", "children"),
    [Input("plot-before", "clickData"),
     Input("plot-after", "clickData")]
)
def display_article_content(click_data_before, click_data_after):
    ctx = callback_context
    if not ctx.triggered:
        return "Click on a point in the plots above to view article content"
    
    # Determine which chart was clicked
    click_data = None
    if ctx.triggered[0]['prop_id'] == 'plot-before.clickData':
        click_data = click_data_before
    elif ctx.triggered[0]['prop_id'] == 'plot-after.clickData':
        click_data = click_data_after
    
    if not click_data or 'points' not in click_data:
        return "Click on a point in the plots above to view article content"
    
    try:
        # Get clicked article index
        point = click_data['points'][0]
        

        
        if 'customdata' in point and point['customdata'] is not None:
            # customdata is now a list, need to take first element
            custom_data = point['customdata']
            if isinstance(custom_data, list) and len(custom_data) > 0:
                article_idx = custom_data[0]
            else:
                article_idx = custom_data
            

            
            # Check if article index is valid
            if isinstance(article_idx, (int, np.integer)) and article_idx < len(all_articles_text):
                article_text = all_articles_text[article_idx]
                return html.Div([
                    html.H5(f"Article {article_idx}", style={"color": "#333"}),
                    html.P(article_text, style={"line-height": "1.6", "text-align": "justify"})
                ])
            else:
                return f"Invalid article index: {article_idx} (total articles: {len(all_articles_text)})"
        else:
            # Try to get index from pointIndex
            if 'pointIndex' in point:
                point_index = point['pointIndex']
                # Need to calculate actual article index based on trace and pointIndex
                trace_index = point.get('curveNumber', 0)
                return html.Div([
                    html.H5(f"Point Info", style={"color": "#333"}),
                    html.P(f"Trace: {trace_index}, Point Index: {point_index}", style={"line-height": "1.6"})
                ])
            else:
                return f"No article data available for this point. Available keys: {list(point.keys())}"
    except Exception as e:
        return f"Error displaying article content: {str(e)}"


# File selection related callback functions
@app.callback(
    Output("selected-file", "data"),
    [Input({"type": "view-file", "index": ALL}, "n_clicks")],
    prevent_initial_call=True
)
def select_file(view_clicks):
    """Select file to display details in right panel"""
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    # If clicked view file button
    if "view-file" in triggered_id:
        try:
            file_index = json.loads(triggered_id.split('.')[0])["index"]
            return file_index  # Set selected file
        except:
            raise PreventUpdate
    
    raise PreventUpdate


@app.callback(
    Output("file-details-content", "children"),
    [Input("selected-file", "data"),
     Input("view-text-btn", "n_clicks")],
    prevent_initial_call=True
)
def show_file_details(selected_file, view_text_clicks):
    """Display file text content"""
    if selected_file is None:
        return html.P("Click on left files to view details", style={"color": "#666", "fontStyle": "italic"})
    
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id']
    
    try:
        global df
        if 'df' not in globals():
            return html.P("Data not loaded")
        
        # Get file content
        if selected_file < len(df):
            file_text = str(df.iloc[selected_file, 1]) if len(df.iloc[selected_file]) > 1 else ""
            file_number = selected_file + 1
            
            # If just selected file, show basic info
            if "selected-file" in triggered_id:
                return html.Div([
                    html.H6(f"Selected File {file_number}"),
                    html.P(f"File content preview: {file_text[:200]}..." if len(file_text) > 200 else file_text, 
                          style={"backgroundColor": "#f8f9fa", "padding": "10px", "borderRadius": "5px"}),
                    html.P("Click button above to view full text", style={"color": "#666", "fontStyle": "italic", "marginTop": "10px"})
                ])
            
            elif "view-text-btn" in triggered_id:
                # Show full text
                return html.Div([
                    html.H6(f"Full text of File {file_number}:"),
                    html.Div(file_text, style={
                        "backgroundColor": "#f0f0f0", 
                        "padding": "10px", 
                        "borderRadius": "5px",
                        "maxHeight": "400px",
                        "overflowY": "auto",
                        "whiteSpace": "pre-wrap"
                    })
                ])
        
        return html.P("File does not exist")
        
    except Exception as e:
        return html.P(f"Error displaying file details: {str(e)}")


# Add debug callback to monitor data state
@app.callback(
    Output("debug-output", "children"),
    [Input("group-data", "data"),
     Input("group-order", "data"),
     Input("selected-group", "data")]
)
def debug_data_monitor(group_data, group_order, selected_group):
    if not callback_context.triggered:
        raise PreventUpdate
    
    debug_info = ""
    
    if group_data:
        debug_info += "<strong>group-data content:</strong><br>"
        for kw, grp in group_data.items():
            if grp:  
                debug_info += f"  {kw} → {grp}<br>"
    
    if group_order:
        debug_info += "<strong>group-order content:</strong><br>"
        for grp, kws in group_order.items():
            if kws:  
                debug_info += f"  {grp}: {', '.join(kws)}<br>"
    
    return html.Div([
        html.Hr(),
        html.Div(debug_info, style={"fontSize": "12px", "backgroundColor": "#f0f0f0", "padding": "10px"})
    ])

# Launch the Dash application
if __name__ == "__main__":
    print(" http://127.0.0.1:8050 ")

    app.run(debug=True, dev_tools_hot_reload=False)