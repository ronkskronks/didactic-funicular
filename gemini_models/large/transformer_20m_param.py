#!/usr/bin/env python3

# transformer_20m_param.py
#
# This script implements a simplified Transformer Encoder model using PyTorch.
# It is designed to have approximately 20,000,000 trainable parameters.
# The model performs character-level sequence prediction, demonstrating basic
# text recognition, data correlation, and sequential reasoning.
#
# The code is heavily commented to explain each part of the implementation.
#
# Parameters Calculation:
# For a Transformer Encoder, the approximate number of parameters is:
# Total Params 

 (vocab_size * d_model) + num_layers * (4 * d_model^2 + 2 * d_model * d_ff)
# where:
#   - vocab_size: Size of the vocabulary.
#   - d_model: Embedding dimension and hidden size of the Transformer.
#   - num_layers: Number of Transformer Encoder layers.
#   - d_ff: Inner dimension of the Feed-Forward Network (typically 4 * d_model).
#
# Architecture for ~20,000,000 parameters:
# - Vocabulary Size (vocab_size): 100 (for demonstration data)
# - Embedding Dimension (d_model): 512
# - Number of Attention Heads (num_heads): 8
# - Feed-Forward Dimension (d_ff): 4 * d_model = 2048
# - Number of Layers (num_layers): 6
#
# Calculation Breakdown:
# 1. Embedding Layer: vocab_size * d_model = 100 * 512 = 51,200
# 2. Per Layer (Multi-Head Attention + Feed-Forward Network):
#    - Multi-Head Attention: ~4 * d_model^2 = 4 * 512^2 = 1,048,576
#    - Feed-Forward Network: 2 * d_model * d_ff = 2 * 512 * 2048 = 2,097,152
#    - Total per layer: 1,048,576 + 2,097,152 = 3,145,728
# 3. Total for 6 layers: 6 * 3,145,728 = 18,874,368
#
# Total Parameters = 51,200 (embedding) + 18,874,368 (layers) = 18,925,568
# This is very close to the requested 20,000,000 parameters.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse

# --- Model Definition ---

class MultiHeadSelfAttention(nn.Module):
    """
    Implements the Multi-Head Self-Attention mechanism.
    This is the core component of the Transformer, allowing the model to
    weigh the importance of different parts of the input sequence.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, value

        # Linear layers for Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # Input shapes: (batch_size, seq_len, d_model)
        batch_size = query.size(0)

        # 1. Linear projections for Q, K, V
        # Resulting shapes: (batch_size, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. Split into multiple heads
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        # Transpose to (batch_size, num_heads, seq_len, d_k) for batch matrix multiplication
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Calculate Attention Scores (Q @ K.transpose) / sqrt(d_k)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Apply mask (for preventing attention to future tokens in decoding, or padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # Fill masked positions with a very small number

        # 4. Apply Softmax to get attention probabilities
        attention_probs = torch.softmax(scores, dim=-1)

        # 5. Multiply with Value matrix (attention_probs @ V)
        # Resulting shape: (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attention_probs, V)

        # 6. Concatenate heads and apply final linear layer
        # Transpose back to (batch_size, seq_len, num_heads, d_k)
        # Reshape to (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)
        return output

class FeedForwardNetwork(nn.Module):
    """
    A simple two-layer Feed-Forward Network with ReLU activation.
    This processes each position independently and identically.
    """
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer, consisting of Multi-Head Self-Attention
    and a Feed-Forward Network, with Layer Normalization and Residual Connections.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model) # Normalizes across the feature dimension
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention part
        # Residual connection: x + self_attn(norm(x))
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward part
        # Residual connection: x + feed_forward(norm(x))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerEncoder(nn.Module):
    """
    The full Transformer Encoder model.
    Consists of an embedding layer, positional encoding, and multiple Encoder Layers.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Positional Encoding: Adds information about the position of tokens in the sequence.
        # Not trainable parameters, but crucial for sequence understanding.
        self.positional_encoding = self._get_positional_encoding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final linear layer to project back to vocabulary size for next token prediction
        self.output_layer = nn.Linear(d_model, vocab_size)

    def _get_positional_encoding(self, max_seq_len, d_model):
        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension and make it non-trainable
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, seq_len)
        # src_mask shape: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)

        # Token embedding + Positional Encoding
        x = self.token_embedding(src) # (batch_size, seq_len, d_model)
        x = x + self.positional_encoding[:, :x.size(1), :].to(x.device) # Add positional encoding

        # Pass through Encoder Layers
        for layer in self.layers:
            x = layer(x, src_mask)

        # Project to vocabulary size for next token prediction
        output = self.output_layer(x) # (batch_size, seq_len, vocab_size)
        return output

# --- Utility Functions ---

def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_char_mapping(text):
    """
    Creates a mapping from characters to integers and vice-versa.
    """
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_int, int_to_char, len(chars)

def text_to_sequences(text, char_to_int, seq_length):
    """
    Converts text into input-target sequences for training.
    Each input is a sequence of characters, and the target is the next character.
    """
    X_sequences = []
    y_sequences = []
    for i in range(0, len(text) - seq_length):
        input_seq_chars = text[i : i + seq_length]
        target_char = text[i + seq_length]

        # Convert input sequence to integer IDs
        X_sequences.append([char_to_int[char] for char in input_seq_chars])
        # Convert target character to integer ID
        y_sequences.append(char_to_int[target_char])

    return torch.tensor(X_sequences, dtype=torch.long), torch.tensor(y_sequences, dtype=torch.long)


if __name__ == "__main__":
    # This block demonstrates how to use the TransformerEncoder class.
    # It will only run when the script is executed directly.

    parser = argparse.ArgumentParser(description="Train or continue training a Transformer model.")
    parser.add_argument('--model_file', type=str, default="transformer_20m_param_model.pth",
                        help="Path to the model file (load from/save to).")
    parser.add_argument('--epochs', type=int, default=1,
                        help="Number of training epochs. Set to 1 for quick demo due to large model size.")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for training.")
    parser.add_argument('--new_model', action='store_true',
                        help="Force creation of a new model, even if model_file exists.")
    parser.add_argument('--d_model', type=int, default=512,
                        help="Embedding dimension and hidden size of the Transformer.")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of attention heads.")
    parser.add_argument('--d_ff', type=int, default=2048,
                        help="Inner dimension of the Feed-Forward Network.")
    parser.add_argument('--num_layers', type=int, default=6,
                        help="Number of Transformer Encoder layers.")
    parser.add_argument('--max_seq_len', type=int, default=50,
                        help="Maximum sequence length for positional encoding.")
    parser.add_argument('--seq_length', type=int, default=10,
                        help="Length of input sequences for training.")

    args = parser.parse_args()

    # --- Device Configuration (CPU vs. GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dummy Text Data for Demonstration ---
    # In a real scenario, this would be a much larger corpus.
    text_data = "hello world this is a test of the transformer network for character level prediction and some basic reasoning capabilities it learns patterns in sequences to predict the next character"
    text_data += " it can correlate data points within a sequence and perform simple reasoning based on learned patterns"

    # Preprocess text data
    char_to_int, int_to_char, vocab_size = create_char_mapping(text_data)
    print(f"Vocabulary size: {vocab_size}")

    X_sequences, y_sequences = text_to_sequences(text_data, char_to_int, args.seq_length)
    print(f"Generated {len(X_sequences)} training sequences.")

    # --- Model Initialization ---
    model = None
    # Check if model file exists and --new_model flag is not set
    if os.path.exists(args.model_file) and not args.new_model:
        print(f"Loading model from {args.model_file}...")
        try:
            model = torch.load(args.model_file, map_location=device)
            # Basic check: ensure loaded model matches expected architecture (optional)
            # This is more complex for nn.Module, so we rely on the user to ensure compatibility.
        except Exception as e:
            print(f"Error loading model: {e}. Creating a new model.")
            model = None

    if model is None:
        print("Creating a new Transformer model...")
        model = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len
        ).to(device)

    # Print the calculated total number of parameters.
    total_params = count_parameters(model)
    print(f"Model has {total_params} trainable parameters.")

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss() # Loss function for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) # Adam optimizer

    # --- Training Loop ---
    print(f"Starting Transformer training for {args.epochs} epochs...")
    # Note: Training a 20M parameter model will be slow on CPU.
    # The default epochs is set to 1 for demonstration purposes.
    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        total_loss = 0
        for i in range(len(X_sequences)):
            input_seq = X_sequences[i].unsqueeze(0).to(device) # Add batch dimension
            target_token = y_sequences[i].unsqueeze(0).to(device) # Add batch dimension

            optimizer.zero_grad() # Clear gradients

            # Forward pass
            # Model outputs (batch_size, seq_len, vocab_size)
            output = model(input_seq)

            # For next token prediction, we only care about the last token's prediction
            # Reshape output to (batch_size * seq_len, vocab_size) for CrossEntropyLoss
            # Target should be (batch_size * seq_len)
            loss = criterion(output.view(-1, vocab_size), target_token.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(X_sequences):.4f}")
    print("Training complete.")

    # --- Save Model ---
    torch.save(model, args.model_file)
    print(f"Model saved to {args.model_file}")

    # --- Demonstrate Prediction (Basic Text Generation) ---
    print("\nDemonstrating prediction:")
    model.eval() # Set model to evaluation mode
    seed_text = text_data[0:args.seq_length]
    print(f"Seed: \"{seed_text}\"")

    # Convert seed text to tensor
    seed_sequence_ids = torch.tensor([char_to_int[char] for char in seed_text], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad(): # Disable gradient calculation for inference
        output_logits = model(seed_sequence_ids) # (1, seq_len, vocab_size)
        # Get the prediction for the next token (last position in sequence)
        predicted_token_logits = output_logits[0, -1, :]
        predicted_token_id = torch.argmax(predicted_token_logits).item()
        predicted_char = int_to_char[predicted_token_id]

    print(f"Predicted next character: \"{predicted_char}\"")
