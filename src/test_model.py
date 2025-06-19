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

    def test_single_transaction_prediction(self):
        """Test if the model returns an output for a single specific transaction."""
        model_dir = 'models'
        if not any(f.endswith('.pth') for f in os.listdir(model_dir)):
            self.skipTest(f"No models found in {model_dir}, skipping single prediction test.")
            return

        # Define the single transaction
        data = {
            'transaction_id': [67784],
            'amount': [57.74],
            'transaction': ["POS Purchase GOLF TOWN #41 C GOLF TOWN #41 C"]
        }
        single_transaction_df = pd.DataFrame(data)

        # Model parameters (consistent with other tests and common setup)
        input_size = 384  # For 'all-MiniLM-L6-v2'
        hidden_size = 512 # As used in test_model_loading_if_exists
        num_classes = 16
        
        model = load_latest_model(input_size, hidden_size, num_classes)
        self.assertIsNotNone(model, "Failed to load the model.")
        model.eval() # Set model to evaluation mode

        # Generate embeddings for the single transaction
        embedding_model_name = 'all-MiniLM-L6-v2'
        temp_embeddings_path = 'embeddings/temp_single_transaction_embeddings.npy'
        
        if os.path.exists(temp_embeddings_path):
            os.remove(temp_embeddings_path)

        embeddings_np = get_or_generate_embeddings(single_transaction_df, embedding_model_name, temp_embeddings_path)
        
        self.assertIsNotNone(embeddings_np, "Embeddings generation failed for the single transaction.")
        self.assertEqual(embeddings_np.shape[0], 1, "Should have one embedding for the single transaction.")
        self.assertEqual(embeddings_np.shape[1], input_size, f"Embedding dimension should be {input_size}.")

        # Clean up the temporary embedding file
        if os.path.exists(temp_embeddings_path):
            os.remove(temp_embeddings_path)

        # Convert embeddings to PyTorch tensor
        embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)

        # Make prediction
        with torch.no_grad(): # Disable gradient calculations for inference
            output = model(embeddings_tensor)

        # Assert output is not None and has expected shape
        self.assertIsNotNone(output, "Model did not return an output for the single transaction.")
        self.assertEqual(output.shape[0], 1, "Output batch size should be 1 for the single transaction.")
        self.assertEqual(output.shape[1], num_classes, f"Output should have {num_classes} scores (number of classes).")

if __name__ == '__main__':
    unittest.main() 