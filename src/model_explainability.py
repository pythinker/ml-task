#!/usr/bin/env python3
# Model Explainability Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import shap
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Add src to path for imports
import sys
sys.path.append('src')

from data_utils import load_data, get_or_generate_embeddings
from model_utils import load_latest_model

# Create directories if they don't exist
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png')
    plt.close()

def analyze_model():
    """Analyze model performance and explainability"""
    # Configuration
    embedding_model_name = 'all-MiniLM-L6-v2'
    train_embeddings_path = f'embeddings/train_embeddings_{embedding_model_name}.npy'
    
    # Load data
    print("Loading data...")
    train_data, _ = load_data()

    # Load embeddings
    X_train_embeddings = get_or_generate_embeddings(train_data, embedding_model_name, train_embeddings_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(train_data['category'])
    num_classes = len(label_encoder.classes_)
    
    # Load model
    print("Loading model...")
    input_size = X_train_embeddings.shape[1]
    hidden_size = 512
    model = load_latest_model(input_size, hidden_size, num_classes)
    
    if model is None:
        print("No model found. Please train a model first.")
        return
        
    # Get a sample for analysis (for speed)
    sample_size = min(1000, len(train_data))
    sample_indices = np.random.choice(train_data.index, sample_size, replace=False)
    sample_data = train_data.loc[sample_indices]
    X_sample_embeddings = torch.tensor(X_train_embeddings[sample_indices], dtype=torch.float32)
    y_sample_encoded = y_encoded[sample_indices]
    
    # Get predictions on sample
    print("Getting predictions for confusion matrix...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_sample_embeddings)
        _, y_pred_encoded = torch.max(outputs, 1)
        
    y_sample_labels = label_encoder.inverse_transform(y_sample_encoded)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded.numpy())
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_sample_labels, y_pred_labels, classes=label_encoder.classes_)
    
    # SHAP Explainability Analysis
    print("\nGenerating SHAP values for model explainability...")
    try:
        # Use a smaller background and explanation set for SHAP to manage memory
        background_size = 100
        explain_size = 10
        
        background_indices = np.random.choice(X_train_embeddings.shape[0], background_size, replace=False)
        background = torch.tensor(X_train_embeddings[background_indices], dtype=torch.float32)
        
        explain_indices = np.random.choice(X_train_embeddings.shape[0], explain_size, replace=False)
        X_explain = torch.tensor(X_train_embeddings[explain_indices], dtype=torch.float32)
        
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_explain)
        
        # We can't easily plot a summary for high-dimensional embeddings.
        # Instead, let's explain a few individual predictions.
        print(f"SHAP analysis for {explain_size} individual predictions:")
        
        # Save a force plot for the first prediction
        # Note: a high-res plot of all features is not practical for embeddings
        try:
            shap.initjs()
            force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0], feature_names="Embeddings", show=False)
            shap.save_html('figures/shap_force_plot.html', force_plot)
            print("SHAP force plot for one prediction saved to figures/shap_force_plot.html")
        except Exception as e:
            print(f"Could not save SHAP force plot: {e}")
            
    except Exception as e:
        print(f"Error generating SHAP values: {e}")
    
    print("\nModel analysis complete!")

if __name__ == "__main__":
    analyze_model() 