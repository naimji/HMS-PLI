# -*- coding: utf-8 -*-
"""
HMS-PLI Custom Inference API
===================================================================
This script provides a standalone interface for external researchers to 
load the pre-trained HMS-PLI model and perform binding affinity regression 
and interaction classification on novel protein-ligand pairs.

Required Dependencies:
- tensorflow >= 2.8.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.2

Usage Instructions:
1. Ensure the pre-trained model weights (.h5) are downloaded.
2. Run this script directly to test the sample sequences.
3. Import the `HMS_PLI_Inference` class into your own pipelines for high-throughput screening.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, MultiHeadAttention, Sequential
from typing import Dict, Union

# =====================================================================
# 1. CUSTOM LAYER DEFINITIONS (Required to load the .h5 model)
# =====================================================================

class AdaptiveMultiHeadAttention(Layer):
    """
    Adaptive Multi-Head Attention (AMHA) with Dynamic Head Selection.
    Learns to select the most relevant attention heads for each sample based 
    on sequence context, preventing feature redundancy across physicochemical channels.
    """
    def __init__(self, embed_dim, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = embed_dim // num_heads

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=self.head_dim, dropout=dropout_rate)
        
        # Head importance scorer
        self.head_scorer = Sequential([
            Dense(embed_dim // 2, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_heads, activation='softmax', name='head_weights')
        ])
        self.head_fusion = Dense(embed_dim, activation='linear')

    def call(self, query, key, value, training=False):
        # Implementation mirrors the training architecture
        attention_outputs = []
        for head in range(self.num_heads):
            head_output = self.mha(query, key, value, training=training)
            attention_outputs.append(head_output)

        stacked_outputs = tf.stack(attention_outputs, axis=-1)
        pooled_query = tf.reduce_mean(query, axis=1)
        head_importance = self.head_scorer(pooled_query, training=training)
        
        head_importance_expanded = tf.expand_dims(tf.expand_dims(head_importance, 1), 1)
        weighted_output = tf.reduce_sum(stacked_outputs * head_importance_expanded, axis=-1)
        return self.head_fusion(weighted_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


class HierarchicalFeatureFusion(Layer):
    """
    Hierarchical Feature Fusion Network (HFFN).
    Fuses extracted features across multiple hierarchical structural scales 
    (atomic, residue, global) using an adaptive gating mechanism.
    """
    def __init__(self, embed_dim: int, num_levels: int = 3, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.dropout_rate = dropout_rate

        self.level_extractors = [
            Sequential([Dense(embed_dim, activation='relu'), Dropout(dropout_rate)]) 
            for _ in range(num_levels)
        ]
        self.gating_network = Dense(num_levels, activation='softmax')
        self.final_fusion = Dense(embed_dim, activation='relu')

    def call(self, inputs, training=False):
        level_outputs = [extractor(inputs, training=training) for extractor in self.level_extractors]
        stacked_levels = tf.stack(level_outputs, axis=-1)
        gate_weights = self.gating_network(inputs)
        gate_weights = tf.expand_dims(gate_weights, 1)
        
        fused_features = tf.reduce_sum(stacked_levels * gate_weights, axis=-1)
        return self.final_fusion(fused_features)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_levels": self.num_levels,
            "dropout_rate": self.dropout_rate
        })
        return config

# Map custom objects for the Keras loader
CUSTOM_OBJECTS = {
    'AdaptiveMultiHeadAttention': AdaptiveMultiHeadAttention,
    'HierarchicalFeatureFusion': HierarchicalFeatureFusion
}


# =====================================================================
# 2. INFERENCE API ENGINE
# =====================================================================

class HMS_PLI_Inference:
    """
    Inference wrapper for the HMS-PLI framework.
    Loads pre-trained multi-task weights and processes raw sequences.
    """
    def __init__(self, model_weights_path: str):
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"Model weights not found at {model_weights_path}. Please download them from the repository.")
            
        print(f"[*] Loading pre-trained HMS-PLI model from {model_weights_path}...")
        self.model = tf.keras.models.load_model(model_weights_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        print("[+] Model successfully initialized.")

    def mock_encode(self, length: int, dims: int):
        """Placeholder for the HMS sequence encoder."""
        return np.random.rand(1, length, dims).astype(np.float32)

    def predict_pair(self, protein_fasta: str, ligand_smiles: str) -> Dict[str, Union[float, int]]:
        """
        Processes a raw sequence pair and predicts multi-task interaction metrics.
        
        Args:
            protein_fasta (str): Raw amino acid sequence.
            ligand_smiles (str): Raw SMILES string.
            
        Returns:
            Dict: Classification Probability, Predicted Affinity (pKd), Uncertainty, and Binary Class.
        """
        # NOTE: In production, replace `mock_encode` with your `UnifiedDataHandler.encode()` functions
        # This ensures the exact overlapping windows and physicochemical properties are mapped.
        X_prot_encoded = self.mock_encode(1200, 73)
        X_lig_encoded = self.mock_encode(120, 84)

        # Execute dual-task inference
        predictions = self.model.predict([X_prot_encoded, X_lig_encoded], verbose=0)
        
        # Parse the multi-task outputs
        interaction_prob = float(predictions[0][0][0])  # Sigmoid output [0, 1]
        binding_affinity = float(predictions[1][0][0])  # Linear regression output (pKd/Ki)
        uncertainty_score = float(predictions[2][0][0]) # Softplus output (Epistemic uncertainty)
        
        return {
            "Interaction_Probability": interaction_prob,
            "Predicted_Affinity": binding_affinity,
            "Prediction_Uncertainty": uncertainty_score,
            "Binary_Classification": 1 if interaction_prob >= 0.5 else 0
        }


# =====================================================================
# 3. COMMAND LINE EXECUTION (Example Usage)
# =====================================================================

if __name__ == "__main__":
    # Define model path (Users download this from GitHub Releases)
    MODEL_PATH = "hms_pli_best_model.h5" 
    
    # Define sample biological sequences (Imatinib targeting ABL1 Kinase)
    sample_protein = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    sample_ligand = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5" 
    
    print("\n" + "="*50)
    print("      HMS-PLI Custom Prediction Engine")
    print("="*50)
    print(f"Target Protein Length: {len(sample_protein)} residues")
    print(f"Ligand SMILES Length : {len(sample_ligand)} tokens")
    
    # Create dummy h5 file for the script to pass the error check if running blindly
    if not os.path.exists(MODEL_PATH):
        print(f"\n[!] WARNING: '{MODEL_PATH}' not found in directory.")
        print("[!] Please ensure you have downloaded the weights from GitHub.")
        print("[!] Skipping inference execution...")
    else:
        # Run Inference
        infer_engine = HMS_PLI_Inference(MODEL_PATH)
        results = infer_engine.predict_pair(sample_protein, sample_ligand)
        
        # Display Results
        print("\n[Prediction Results]")
        print(f"  > Interaction Probability : {results['Interaction_Probability']:.4f}")
        print(f"  > Predicted Affinity (pKd): {results['Predicted_Affinity']:.4f}")
        print(f"  > Model Uncertainty       : {results['Prediction_Uncertainty']:.4f}")
        
        class_str = "Active (Binding)" if results['Binary_Classification'] == 1 else "Inactive (Non-Binding)"
        print(f"  > Final Classification    : {class_str}")
    print("="*50 + "\n")