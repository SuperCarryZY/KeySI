# Test imports step by step
print("Testing imports...")

try:
    print("1. Testing basic imports...")
    import re 
    import json
    import os
    import random
    import io
    import base64
    import math
    print("✓ Basic imports successful")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")

try:
    print("2. Testing rank_bm25...")
    from rank_bm25 import BM25Okapi
    print("✓ rank_bm25 successful")
except Exception as e:
    print(f"✗ rank_bm25 failed: {e}")

try:
    print("3. Testing NLTK...")
    from nltk.stem import PorterStemmer
    print("✓ NLTK successful")
except Exception as e:
    print(f"✗ NLTK failed: {e}")

try:
    print("4. Testing matplotlib...")
    import matplotlib
    matplotlib.use("Agg") 
    print("✓ matplotlib successful")
except Exception as e:
    print(f"✗ matplotlib failed: {e}")

try:
    print("5. Testing pandas...")
    import pandas as pd
    print("✓ pandas successful")
except Exception as e:
    print(f"✗ pandas failed: {e}")

try:
    print("6. Testing dash...")
    import dash
    from dash import dcc, html, Input, Output, State, ALL, no_update
    from dash.exceptions import PreventUpdate
    print("✓ dash successful")
except Exception as e:
    print(f"✗ dash failed: {e}")

try:
    print("7. Testing plotly...")
    import plotly.graph_objects as go
    import plotly.express as px
    print("✓ plotly successful")
except Exception as e:
    print(f"✗ plotly failed: {e}")

try:
    print("8. Testing torch...")
    import torch
    import numpy as np
    import torch.nn as nn
    from torch.nn.functional import pad
    print("✓ torch successful")
except Exception as e:
    print(f"✗ torch failed: {e}")

try:
    print("9. Testing transformers...")
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    print("✓ transformers successful")
except Exception as e:
    print(f"✗ transformers failed: {e}")

try:
    print("10. Testing sklearn...")
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    print("✓ sklearn basic successful")
except Exception as e:
    print(f"✗ sklearn basic failed: {e}")

try:
    print("11. Testing sklearn metrics...")
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
    print("✓ sklearn metrics successful")
except Exception as e:
    print(f"✗ sklearn metrics failed: {e}")

try:
    print("12. Testing sklearn cluster...")
    from sklearn.cluster import MiniBatchKMeans, KMeans
    print("✓ sklearn cluster successful")
except Exception as e:
    print(f"✗ sklearn cluster failed: {e}")

try:
    print("13. Testing sklearn manifold...")
    from sklearn.manifold import TSNE
    print("✓ sklearn manifold successful")
except Exception as e:
    print(f"✗ sklearn manifold failed: {e}")

print("All tests completed!")
