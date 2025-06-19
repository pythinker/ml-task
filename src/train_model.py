#!/usr/bin/env python3
# Transaction Category Prediction System

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Import from custom modules
from data_utils import load_data, get_or_generate_embeddings
from model_utils import SimpleNN, save_model

def main():
    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # Configuration
    embedding_model_name = 'all-MiniLM-L6-v2'
    train_embeddings_path = f'embeddings/train_embeddings_{embedding_model_name}.npy'
    test_embeddings_path = f'embeddings/test_embeddings_{embedding_model_name}.npy'
    
    # Load data
    print("Loading data...")
    train_data, test_data = load_data()

    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Generate or load embeddings
    X_train_embeddings = get_or_generate_embeddings(train_data, embedding_model_name, train_embeddings_path)
    X_test_embeddings = get_or_generate_embeddings(test_data, embedding_model_name, test_embeddings_path)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(train_data['category'])
    num_classes = len(label_encoder.classes_)

    # Save label encoder for later use
    np.save('results/label_encoder.npy', label_encoder.classes_)

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_embeddings,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Convert data to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_embeddings, dtype=torch.float32)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model parameters
    input_size = X_train.shape[1]
    hidden_size = 512
    model = SimpleNN(input_size, hidden_size, num_classes)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Train the model
    print("\nTraining model...")
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # Plotting the loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/loss_plot.png')
    print("\nLoss plot saved to figures/loss_plot.png")

    # Evaluate on validation set
    print("\nEvaluating model...")
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_list.extend(predicted.numpy())

    y_pred = np.array(y_pred_list)
    y_val_labels = label_encoder.inverse_transform(y_val)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    val_report = classification_report(y_val_labels, y_pred_labels, zero_division=0)
    val_f1 = f1_score(y_val_labels, y_pred_labels, average='weighted', zero_division=0)

    print(f"Validation F1 Score (weighted): {val_f1:.4f}")
    print("\nClassification Report:")
    print(val_report)

    # Save the classification report
    with open('results/validation_report.txt', 'w') as f:
        f.write(f"Validation F1 Score (weighted): {val_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(val_report)

    # Save the model
    save_model(model)

    # Generate predictions for test data
    print("\nGenerating predictions for test data...")
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predictions_encoded = torch.max(test_outputs.data, 1)

    test_predictions = label_encoder.inverse_transform(test_predictions_encoded.numpy())

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'transaction_id': test_data['transaction_id'],
        'category': test_predictions
    })

    # Save predictions to CSV
    predictions_df.to_csv('predictions.csv', index=False)
    print("\nPredictions saved to predictions.csv")

    print("\nDone!")

if __name__ == '__main__':
    main() 