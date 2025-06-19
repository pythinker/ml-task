#!/usr/bin/env python3
# PyTorch model definition and utilities

import torch
import torch.nn as nn
import os
from datetime import datetime

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def save_model(model, base_filename="models/nn_classifier"):
    """Save the model with a timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{base_filename}_{timestamp}.pth"
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_filename)
    print(f"\nModel saved to {model_filename}")
    return model_filename

def load_latest_model(input_size, hidden_size, num_classes, model_dir='models'):
    """Load the most recently trained model"""
    if not os.path.exists(model_dir):
        print("No models directory found.")
        return None
    
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        print("No model files found.")
        return None
    
    latest_model_path = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model_path}")
    
    model = SimpleNN(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(latest_model_path))
    model.eval()
    return model 