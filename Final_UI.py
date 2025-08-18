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
nltk.data.path.append("/Users/yanzhu/nltk_data")
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster




# Set device and global parameters
device = "mps" if torch.backends.mps.is_available() else "cpu"
device ="cpu"
print(f"Using device: {device}")
os.chdir("/Users/yanzhu/Box Sync/Yan/KeySI/CSV")
num_threads=8
top_similiar_file_to_keywords=500
learningrate=1e-4
num_epochs = 15
margin_number = 3
top_keywords=3
early_stop_threshold = 8
mostfequentwords=250
cluster_distance=30
word_count_feq=3
max_d = 50
word_count_threshold= 1
top_similar_files = 3000
batch_size = 128
clusterthreshold = 25


# Define relative paths for data, model saving, and output
img_output_dir = "../Keyword_Group/Test"
csv_path = "AGnews100.csv"  
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
#model setup
embedding_model_kw = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens', device=device)

kw_model = KeyBERT(model=embedding_model_kw)
ps = PorterStemmer()
word_count = Counter()
original_form = {}
df = pd.read_csv(csv_path)
all_articles_text = df.iloc[:, 1].dropna().astype(str).tolist()
labels = df.iloc[:, 0].values 

# Extract and count keywords using KeyBERT and NLTK
def process_article(article):
    try:
        article = re.sub(r'\d+', '', article).strip()
        if len(article.split()) < 5:
            return None
        words = word_tokenize(article)
        tagged_words = pos_tag(words)
        nouns = [word for word, pos in tagged_words if pos.startswith("NN")]
        if not nouns:
            return None

        keywords_info = kw_model.extract_keywords(" ".join(nouns), keyphrase_ngram_range=(1, 1), stop_words='english', top_n=8)
        result = [(ps.stem(kw), kw) for kw, _ in keywords_info]

        return result if result else None

    except Exception as e:
        print(f"failed to process article: {e}")
        return None

results = Parallel(n_jobs=8)(delayed(process_article)(article) for article in all_articles_text)

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



# Build Dash layout with UI components like inputs, buttons, and containers
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    html.H3("Keyword Grouping"),
    html.Label("Enter number of groups:"),
    dcc.Input(id="group-count", type="number", value=3, min=1, step=1),
    html.Button("Generate Groups", id="generate-btn", n_clicks=0),
    html.Button("Clear All", id="clear-btn", n_clicks=0, style={"margin-left": "10px"}),
   
    html.Div([
        dcc.Input(
            id='new-keyword-input',
            type='text',
            placeholder='Enter a keyword',
            style={"marginRight": "10px"}
        ),
        html.Button("Add Keyword", id="add-keyword-btn", n_clicks=0)
    ], style={"marginBottom": "20px"}),

    dcc.Store(id="group-data", data={kw: None for kw in keywords}),
    dcc.Store(id="selected-group", data=None),
    dcc.Store(id="group-order", data={}),
    dcc.Store(id='current-page', data=0),
    
    html.Div(id="group-containers", style={
        "display": "flex",
        "flex-wrap": "wrap",
        "gap": "10px",
        "margin-bottom": "20px"
    }),
    
    html.H4("Keywords (Click to Move to Selected Group)"),
    html.Div([
        html.Button("←", id="prev-page-btn", style={"margin-right": "10px"}),
        html.Span(id="page-info", style={"margin": "0 10px", "fontSize": "18px", "fontWeight": "bold"}),
        html.Button("→", id="next-page-btn")
    ], style={"margin-top": "10px"}),
    html.Div(id='keywords-container', style={
        "border": "1px solid #ddd",
        "padding": "10px",
        "height": "300px",
        "overflow-y": "auto",
        "margin-top": "10px"
    }),
    
    html.Button("Train", id="train-btn", n_clicks=0, style={"margin-top": "20px"}),
    html.Div(id="train-output", children=[
        html.Img(id="train-plot", src="", style={"width": "100%", "max-width": "800px"})
    ])
])

app.clientside_callback(
        """
        function(keyword_clicks, clear_clicks, selected_group, group_data) {
            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered.length) return [window.dash_clientside.no_update, window.dash_clientside.no_update];
    
            const triggered = ctx.triggered[0];
            const is_clear = triggered.prop_id === 'clear-btn.n_clicks';
            
            let new_data = {...group_data};
            let styles = {};
            
            if (is_clear) {
                Object.keys(new_data).forEach(k => new_data[k] = null);
                Object.keys(new_data).forEach(k => {
                    styles[k] = {backgroundColor: '#f0f0f0', color: 'black'};
                });
                return [new_data, styles];
            }
            
            const btn_id = JSON.parse(triggered.prop_id.split('.')[0]);
            if (btn_id.type !== "keyword-btn") {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
    
            const kw = btn_id.index;
            new_data[kw] = new_data[kw] === null ? selected_group : null;
    
            Object.keys(new_data).forEach(k => {
                styles[k] = new_data[k] ? 
                    {backgroundColor: '#4CAF50', color: 'white'} : 
                    {backgroundColor: '#f0f0f0', color: 'black'};
            });
    
            return [new_data, styles];
        }
    """,
    [
        Output("group-data", "data"),
        Output({"type": "keyword-btn", "index": ALL}, "style")
    ],
    [
        Input({"type": "keyword-btn", "index": ALL}, "n_clicks"), 
        Input("clear-btn", "n_clicks")
    ],
    [
        State("selected-group", "data"),
        State("group-data", "data")
    ]
)

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
    [Output('current-page', 'data'),
     Output('page-info', 'children')],
    [Input('prev-page-btn', 'n_clicks'),
     Input('next-page-btn', 'n_clicks')],
    [State('current-page', 'data')]
)
def update_page(prev_clicks, next_clicks, current_page):
    total_pages = best_k
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_page, f"Category {current_page+1}/{total_pages}"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'prev-page-btn' and current_page > 0:
        current_page -= 1
    elif button_id == 'next-page-btn' and current_page < total_pages-1:
        current_page += 1
        
    return current_page, f"Category {current_page+1}/{total_pages}"

@app.callback(
    Output('keywords-container', 'children'),
    Input('current-page', 'data'),
    State("group-data", "data")
)
def update_keywords(current_page, group_data):
    ctx = callback_context
    if ctx.triggered:
        print(f"🔍 update_keywords triggered by: {ctx.triggered[0]['prop_id']}")

    global GLOBAL_OUTPUT_DICT
    if not GLOBAL_OUTPUT_DICT:
        return html.Div("No Clusters Available", style={"padding": "10px"})

    cluster_names = list(GLOBAL_OUTPUT_DICT.keys())
    total_clusters = len(cluster_names)
    if current_page >= total_clusters:
        current_page = total_clusters - 1

    current_cluster = cluster_names[current_page]
    page_keywords = GLOBAL_OUTPUT_DICT[current_cluster]
    print(f"Page {current_page + 1}/{total_clusters}: Showing cluster '{current_cluster}'")

    btn_base_style = {
        "width": "120px",
        "height": "40px",
        "margin": "40x",
        "textAlign": "center",
        "lineHeight": "40px",
        "borderRadius": "8px",
        "fontSize": "16px",
        "backgroundColor": "#f0f0f0",
        "color": "black",
        "display": "inline-block",
        "verticalAlign": "middle"
    }

    children = []
    for kw in page_keywords:
        is_assigned = bool(group_data.get(kw, False)) if group_data else False
        kw_button = html.Button(
            kw,
            id={"type": "keyword-btn", "index": kw},
            style={**btn_base_style,
                   "backgroundColor":  "#f0f0f0",
                   "color": "black"}
        )
        children.append(kw_button)

    return html.Div(children, key="keywords-container")

@app.callback(
    Output("group-order", "data"),
    [
        Input("generate-btn", "n_clicks"),
        Input("group-data", "data"),
        Input({"type": "move-up", "group": ALL, "index": ALL}, "n_clicks"),
        Input({"type": "move-down", "group": ALL, "index": ALL}, "n_clicks"),
    ],
    [
        State("group-count", "value"),
        State("group-order", "data")
    ],
    prevent_initial_call=True
)
def update_group_order(generate_n_clicks, group_data, up_clicks, down_clicks, num_groups, current_order):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    
    new_order = dict(current_order) if current_order else {}
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "generate-btn":
        if not num_groups or num_groups < 1:
            raise dash.exceptions.PreventUpdate
        return {f"Group {i+1}": [] for i in range(num_groups)}
    
    elif triggered_id == "group-data":
        for grp in new_order:
            new_order[grp] = [kw for kw in new_order[grp] if group_data.get(kw) == grp]
        for kw, grp in group_data.items():
            if grp and grp in new_order and kw not in new_order[grp]:
                new_order[grp].append(kw)
        return new_order

    elif triggered_id.startswith("{"):
        try:
            btn_info = json.loads(triggered_id)
            grp_name = btn_info.get("group")
            action = btn_info.get("type")
            idx = btn_info.get("index")

            if grp_name not in new_order or idx is None:
                raise dash.exceptions.PreventUpdate
            
            kw_list = list(new_order[grp_name])

            if action == "move-up" and idx > 0:
                kw_list[idx], kw_list[idx-1] = kw_list[idx-1], kw_list[idx]
            elif action == "move-down" and idx < len(kw_list)-1:
                kw_list[idx], kw_list[idx+1] = kw_list[idx+1], kw_list[idx]

            new_order[grp_name] = kw_list
            return new_order
        except Exception as e:
            print(f"Error handling move button: {e}")
            raise dash.exceptions.PreventUpdate

    return new_order



@app.callback(
    Output("group-containers", "children"),
    [Input("group-order", "data"),
     Input("selected-group", "data")]
)
def render_groups(group_order, selected_group):
    if not group_order:
        return []

    children = []
    for grp_name, kw_list in group_order.items():
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
        for kw in kw_list:
            group_keywords.append(html.Div(
                html.Span(kw, style={"padding": "5px", "margin": "3px", "border": "1px solid #ddd"}),
                style={"margin-bottom": "5px"}
            ))

        group_body = html.Div(
            group_keywords,
            style={
                "border": "1px solid #ddd",
                "padding": "10px",
                "minHeight": "100px",
                "maxHeight": "300px",
                "overflowY": "auto"
            }
        )

        group_container = html.Div(
            [group_header, group_body],
            style={"display": "inline-block", "margin": "5px"}
        )
        children.append(group_container)

    return children

@app.callback(
    Output("selected-group", "data"),
    Input({"type": "group-header", "index": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def select_group(n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id']
    triggered_n_clicks = ctx.triggered[0]['value']

    if not triggered_n_clicks or triggered_n_clicks is None:
        raise dash.exceptions.PreventUpdate

    selected_group = json.loads(triggered_id.split('.')[0])["index"]
    return selected_group

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
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()


    group_embeddings = []
    group_labels = []
    group_original_indices = []  

    for group_name, indices in group_to_indices_map.items():
        if len(indices) == 0:
            print(f"Warning: Group {group_name} has no matching articles, skipping...")
            continue
            
        selected_texts = [df.loc[i, df.columns[1]] for i in indices] 
        print(f"Group {group_name} has {len(indices)} articles.")
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
        

        # 计算perplexity参数
        perplexity = min(30, max(5, len(combined_embeds) // 3))
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

    print("\n===== Group IDs After Removing Outliers =====")
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
    print("Group Dictionary with Labels:")
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

    # 冻结 embedding 层
    for param in model.embeddings.parameters():
        param.requires_grad = False

    # 冻结前8层 encoder
    for layer in model.encoder.layer[:8]:
        for param in layer.parameters():
            param.requires_grad = False

    # 收集最后4层 encoder 和 pooler 的参数
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
            # 负样本池：所有其他组的所有样本
            negative_cls_vectors = []
            for negative_group in group_names:
                if negative_group == anchor_group:
                    continue
                negative_indices = group_dict[negative_group]
                negative_cls_vectors.extend(get_group_cls_vectors(negative_indices))
            if len(negative_cls_vectors) == 0:
                continue
            neg_tensor_all = torch.stack(negative_cls_vectors, dim=0)

            # anchor-positive两两组合
            for i in range(len(anchor_cls_vectors)):
                for j in range(len(anchor_cls_vectors)):
                    if i == j:
                        continue
                    anchor = anchor_cls_vectors[i]
                    positive = anchor_cls_vectors[j]
                    # hard negative采样
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
            print("没有有效的三元组可训练")
        gc.collect()
        
        if (epoch + 1) % 10 == 0:
            model_save_path_epoch = os.path.expanduser(f"~/Documents/Text/bert_finetuned_epoch_{epoch+1}.pth")
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
    
    # 计算perplexity参数
    perplexity_before = min(30, max(5, len(cls_vectors_before) // 3))
    perplexity_after = min(30, max(5, len(cls_vectors_after_cpu) // 3))
    
    tsne_before = TSNE(n_components=2, perplexity=perplexity_before, random_state=42)
    tsne_after = TSNE(n_components=2, perplexity=perplexity_after, random_state=42)
    projected_2d_before = tsne_before.fit_transform(cls_vectors_before.numpy())
    projected_2d_after = tsne_after.fit_transform(cls_vectors_after_cpu)
    
    # 计算每个关键词组的中心点
    group_centers = {}
    for group_name, indices in group_dict.items():
        if len(indices) > 0:
            # 获取该组所有文章的索引在df_articles中的位置
            group_article_indices = []
            for idx in indices:
                if idx in df_articles.index:
                    # 找到在df_articles中对应的行索引
                    article_idx = df_articles.index.get_loc(idx)
                    group_article_indices.append(article_idx)
            
            if len(group_article_indices) > 0:
                # 计算该组所有文章的embedding平均值
                group_embeddings = cls_vectors_after_cpu[group_article_indices]
                group_center = np.mean(group_embeddings, axis=0)
                # 将中心点投影到2D空间 - 使用该组所有点的2D投影的平均值
                group_2d_points = projected_2d_after[group_article_indices]
                group_center_2d = np.mean(group_2d_points, axis=0)
                group_centers[group_name] = group_center_2d
                print(f"Group {group_name} center: {group_center_2d}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, projected_2d, title in zip(axes, 
                                       [projected_2d_before, projected_2d_after], 
                                       ["2D Projection Before Finetuning", "2D Projection After Finetuning"]):
        for label in unique_labels:
            mask = (labels == label)
            ax.scatter(projected_2d[mask, 0], projected_2d[mask, 1], 
                       color=label_to_color[label], label=f"Class {label}", alpha=0.6)
        
        # 在After Finetuning图中添加关键词组中心点
        if title == "2D Projection After Finetuning":
            center_colors = ['black', 'darkred', 'darkblue', 'darkgreen', 'purple', 'orange']
            for i, (group_name, center_2d) in enumerate(group_centers.items()):
                color = center_colors[i % len(center_colors)]
                ax.scatter(center_2d[0], center_2d[1], 
                          color=color, s=200, marker='*', 
                          label=f'Center: {group_name}', edgecolors='white', linewidth=2)
                # 添加组名标签
                ax.annotate(group_name, (center_2d[0], center_2d[1]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=color)
        
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid()
    

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    

    return encoded_image
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
    Output("train-plot", "src"),
    Input("train-btn", "n_clicks"),
    State("group-order", "data")
)
# Start training and return the final visualization image as base64
def train_and_plot(n_clicks, group_order):
    if not n_clicks:
        raise PreventUpdate

    try:
        with open(save_path, "w", encoding="utf-8") as f:
            print(group_order)
            json.dump(group_order, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error saving group order:", e)

    encoded_image = run_training()
    return "data:image/png;base64," + encoded_image


# Launch the Dash application
if __name__ == "__main__":
    app.run(debug=True, dev_tools_hot_reload=False)