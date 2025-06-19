#!/usr/bin/env python3
# Simple CI test script

import os
import pandas as pd
import numpy as np
import unittest
import torch

# Add src to path for imports
import sys
sys.path.append('src')

from data_utils import load_data, get_or_generate_embeddings
from model_utils import SimpleNN, load_latest_model

class TestModel(unittest.TestCase):
    """Tests for the transaction category prediction model"""
    
    def setUp(self):
        """Set up test environment"""
        # Check if data files exist
        self.assertTrue(os.path.exists('data/ds_project_train_v1.csv'), "Training data file not found")
        self.assertTrue(os.path.exists('data/ds_project_test_v1.csv'), "Test data file not found")
        
        # Load a small sample of training data for testing
        self.train_data, _ = load_data()
        self.sample_data = self.train_data.sample(100, random_state=42)
        
    def test_data_format(self):
        """Test data format"""
        # Check training data columns
        self.assertIn('transaction_id', self.sample_data.columns, "transaction_id column missing")
        self.assertIn('amount', self.sample_data.columns, "amount column missing")
        self.assertIn('transaction', self.sample_data.columns, "transaction column missing")
        self.assertIn('category', self.sample_data.columns, "category column missing")
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['amount']), "amount should be numeric")
        self.assertTrue(pd.api.types.is_string_dtype(self.sample_data['transaction']), "transaction should be string")
        
    def test_embedding_generation(self):
        """Test that embeddings are generated correctly"""
        embedding_model_name = 'all-MiniLM-L6-v2'
        test_embeddings_path = 'embeddings/test_embeddings.npy'
        
        # Clean up previous test embeddings if they exist
        if os.path.exists(test_embeddings_path):
            os.remove(test_embeddings_path)
            
        embeddings = get_or_generate_embeddings(self.sample_data, embedding_model_name, test_embeddings_path)
        
        # Check that embeddings are created
        self.assertIsNotNone(embeddings, "Embeddings should not be None")
        self.assertEqual(embeddings.shape[0], len(self.sample_data), "Number of embeddings should match sample size")
        self.assertTrue(os.path.exists(test_embeddings_path), "Embedding file should be created")
        
        # Clean up
        os.remove(test_embeddings_path)

    def test_model_loading_if_exists(self):
        """Test model loading if a model exists"""
        if any(f.endswith('.pth') for f in os.listdir('models')):
            # Mock parameters for loading model
            # In a real scenario, these would be stored with the model
            input_size = 384  # for 'all-MiniLM-L6-v2'
            hidden_size = 512
            num_classes = len(self.train_data['category'].unique())
            
            model = load_latest_model(input_size, hidden_size, num_classes)
            
            # Check if it's a valid PyTorch model
            self.assertIsNotNone(model, "Model should be loaded")
            self.assertIsInstance(model, SimpleNN, "Loaded object should be a SimpleNN model")
            
    def test_predictions_format_if_exists(self):
        """Test predictions format if predictions.csv exists"""
        if os.path.exists('predictions.csv'):
            predictions = pd.read_csv('predictions.csv')
            
            # Check predictions columns
            self.assertIn('transaction_id', predictions.columns, "transaction_id column missing in predictions")
            self.assertIn('category', predictions.columns, "category column missing in predictions")
            
            # Check that there are no NaN values
            self.assertFalse(predictions.isnull().any().any(), "Predictions should not contain NaN values")

if __name__ == '__main__':
    unittest.main() 