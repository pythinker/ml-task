#!/usr/bin/env python3
# Data and feature engineering utilities

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def load_data(train_path='data/ds_project_train_v1.csv', test_path='data/ds_project_test_v1.csv'):
    """Load training and test data"""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")
        
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def get_or_generate_embeddings(data, embedding_model_name, file_path):
    """Generate or load embeddings"""
    if os.path.exists(file_path):
        print(f"Loading existing embeddings from {file_path}...")
        embeddings = np.load(file_path)
    else:
        print(f"Generating embeddings with {embedding_model_name}...")
        model = SentenceTransformer(embedding_model_name)
        
        # Combine transaction and amount for better embeddings
        text_for_embedding = data['transaction'] + ' ' + data['amount'].astype(str)
        embeddings = model.encode(text_for_embedding.tolist(), show_progress_bar=True)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        print(f"Embeddings saved to {file_path}")
        
    return embeddings 