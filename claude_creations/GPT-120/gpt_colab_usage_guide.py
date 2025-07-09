# ============================================================================
# GOOGLE COLAB USAGE GUIDE FOR GPT-120M
# This file contains step-by-step instructions for using the GPT model in Colab
# Copy and paste these code blocks into separate Colab cells
# ============================================================================

# ============================================================================
# CELL 1: INSTALLATION AND IMPORTS
# Run this first to install required packages and import dependencies
# ============================================================================

# Install required packages (run this in Colab)
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install transformers datasets tqdm numpy pandas requests

# Import all necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import zipfile
import requests
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================================
# CELL 2: LOAD THE MODEL CODE
# Execute the main GPT model file (assuming you've uploaded it to Colab)
# ============================================================================

# If you've uploaded the gpt_120m_param.py file to Colab, run it
# %run gpt_120m_param.py

# Alternatively, if you want to define everything in the notebook,
# copy and paste the entire GPT model code here

# ============================================================================
# CELL 3: DATASET LOADING OPTIONS
# Choose one of these methods to load your dataset
# ============================================================================

# METHOD 1: Load from Google Drive
# First, mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Then load your dataset
# Replace 'your_dataset.txt' with your actual file path
# texts = DatasetLoader.load_from_drive('/content/drive/MyDrive/your_dataset.txt', 'txt')

# METHOD 2: Load from Hugging Face
# Example: Load WikiText-2 dataset
# texts = DatasetLoader.load_from_huggingface('wikitext', 'wikitext-2-raw-v1', 'text')

# METHOD 3: Download from URL
# Example: Download a text file from the internet
# texts = DatasetLoader.download_and_load('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

# METHOD 4: Use sample data for testing
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand human language.",
    "Deep learning models can learn complex patterns from data.",
    "Transformers have revolutionized the field of natural language processing.",
] * 200  # Repeat for more training data

print(f"Loaded {len(texts)} text samples")
print(f"Example text: {texts[0]}")

# ============================================================================
# CELL 4: MODEL CONFIGURATION
# Set up the model configuration with 120M parameters
# ============================================================================

# Create model configuration
config = GPTConfig(
    vocab_size=1000,      # Will be updated based on tokenizer
    n_embd=768,           # Embedding dimension
    n_head=12,            # Number of attention heads
    n_layer=12,           # Number of transformer layers
    block_size=512,       # Maximum sequence length
    dropout=0.1,          # Dropout probability
    learning_rate=6e-4,   # Learning rate
    weight_decay=0.1,     # Weight decay
    beta1=0.9,            # Adam beta1
    beta2=0.95            # Adam beta2
)

print("Model configuration:")
print(f"- Embedding dimension: {config.n_embd}")
print(f"- Number of heads: {config.n_head}")
print(f"- Number of layers: {config.n_layer}")
print(f"- Block size: {config.block_size}")
print(f"- Dropout: {config.dropout}")

# ============================================================================
# CELL 5: TOKENIZATION
# Create tokenizer and preprocess the data
# ============================================================================

# Create tokenizer from your dataset
print("Creating tokenizer...")
tokenizer = SimpleTokenizer(texts)
print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Sample tokens: {tokenizer.vocab[:20]}")

# Update configuration with actual vocabulary size
config.vocab_size = tokenizer.vocab_size

# Test tokenization
sample_text = "Hello, world!"
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print(f"Original: {sample_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

# ============================================================================
# CELL 6: DATA PREPROCESSING
# Convert text to training sequences
# ============================================================================

# Create data preprocessor
print("Preprocessing data...")
preprocessor = DataPreprocessor(tokenizer, config.block_size)
input_ids, target_ids = preprocessor.prepare_training_data(texts)

print(f"Input shape: {input_ids.shape}")
print(f"Target shape: {target_ids.shape}")
print(f"Number of training sequences: {len(input_ids)}")

# Create datasets and data loaders
dataset = TensorDataset(input_ids, target_ids)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Adjust batch size based on your GPU memory
batch_size = 8 if device == 'cuda' else 4

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training batches: {len(train_dataloader)}")
print(f"Validation batches: {len(val_dataloader)}")

# ============================================================================
# CELL 7: MODEL CREATION
# Create the GPT model with 120M parameters
# ============================================================================

# Create the model
print("Creating GPT model...")
model = GPT(config)

# The model will print its parameter count
print(f"Model created successfully!")
print(f"Model is on device: {next(model.parameters()).device}")

# Move model to GPU if available
model = model.to(device)
print(f"Model moved to device: {device}")

# ============================================================================
# CELL 8: TRAINING SETUP
# Set up the trainer and begin training
# ============================================================================

# Create trainer
print("Setting up trainer...")
trainer = GPTTrainer(model, config, device)

# Optional: Load from checkpoint if you have one
# trainer.load_checkpoint('checkpoint_step_1000.pth')

print("Trainer created successfully!")
print(f"Starting step: {trainer.step}")
print(f"Starting epoch: {trainer.epoch}")

# ============================================================================
# CELL 9: TRAINING LOOP
# Train the model (this will take time!)
# ============================================================================

# Training parameters
num_epochs = 5          # Number of epochs to train
save_every = 100        # Save checkpoint every N steps
validate_every = 50     # Validate every N steps

print(f"Starting training for {num_epochs} epochs...")
print(f"Training will save checkpoints every {save_every} steps")
print(f"Validation will run every {validate_every} steps")

# Start training
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_epochs=num_epochs,
    save_every=save_every,
    validate_every=validate_every
)

print("Training completed!")

# ============================================================================
# CELL 10: TEXT GENERATION
# Generate text using the trained model
# ============================================================================

# Switch to evaluation mode
model.eval()

# Function to generate text
def generate_text(prompt: str, max_tokens: int = 100, temperature: float = 0.8, top_k: int = 50):
    """Generate text using the trained model"""
    
    # Encode the prompt
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    # Generate tokens
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode and return
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    return generated_text

# Test text generation
print("Testing text generation...")

prompts = [
    "Machine learning is",
    "The future of AI",
    "Deep learning models",
    "Natural language processing"
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    generated = generate_text(prompt, max_tokens=50, temperature=0.8)
    print(f"Generated: {generated}")

# ============================================================================
# CELL 11: SAVE AND LOAD MODEL
# Save your trained model and load it later
# ============================================================================

# Save the complete model
def save_complete_model(model, tokenizer, config, filepath):
    """Save model, tokenizer, and config together"""
    save_data = {
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'tokenizer_char_to_id': tokenizer.char_to_id,
        'tokenizer_id_to_char': tokenizer.id_to_char,
        'config': config
    }
    torch.save(save_data, filepath)
    print(f"Complete model saved to {filepath}")

# Load the complete model
def load_complete_model(filepath, device='cuda'):
    """Load model, tokenizer, and config together"""
    data = torch.load(filepath, map_location=device)
    
    # Recreate tokenizer
    tokenizer = SimpleTokenizer.__new__(SimpleTokenizer)
    tokenizer.vocab = data['tokenizer_vocab']
    tokenizer.char_to_id = data['tokenizer_char_to_id']
    tokenizer.id_to_char = data['tokenizer_id_to_char']
    tokenizer.vocab_size = len(tokenizer.vocab)
    
    # Recreate model
    config = data['config']
    model = GPT(config)
    model.load_state_dict(data['model_state_dict'])
    model = model.to(device)
    
    return model, tokenizer, config

# Save your trained model
save_complete_model(model, tokenizer, config, 'gpt_120m_complete.pth')

# To load later:
# model, tokenizer, config = load_complete_model('gpt_120m_complete.pth', device)

# ============================================================================
# CELL 12: ADVANCED GENERATION TECHNIQUES
# More sophisticated text generation methods
# ============================================================================

def generate_with_nucleus_sampling(prompt: str, max_tokens: int = 100, temperature: float = 0.8, top_p: float = 0.9):
    """Generate text using nucleus (top-p) sampling"""
    
    model.eval()
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits for next token
            input_ids_cropped = input_ids[:, -config.block_size:]
            logits, _ = model(input_ids_cropped)
            logits = logits[:, -1, :] / temperature
            
            # Apply nucleus sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Find cutoff for nucleus
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Remove tokens outside nucleus
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].cpu().tolist())

# Test nucleus sampling
print("Testing nucleus sampling...")
for prompt in ["The future of technology", "Once upon a time"]:
    print(f"\nPrompt: '{prompt}'")
    generated = generate_with_nucleus_sampling(prompt, max_tokens=50, temperature=0.8, top_p=0.9)
    print(f"Generated: {generated}")

# ============================================================================
# CELL 13: TRAINING MONITORING
# Monitor training progress and visualize losses
# ============================================================================

import matplotlib.pyplot as plt

def plot_training_progress(trainer):
    """Plot training and validation losses"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    if trainer.train_losses:
        ax1.plot(trainer.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
    
    # Plot validation loss
    if trainer.val_losses:
        ax2.plot(trainer.val_losses)
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Validation Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot training progress
plot_training_progress(trainer)

# Print training statistics
print(f"Training steps completed: {trainer.step}")
print(f"Epochs completed: {trainer.epoch}")
print(f"Best validation loss: {trainer.best_loss:.4f}")
print(f"Final training loss: {trainer.train_losses[-1]:.4f}" if trainer.train_losses else "No training loss recorded")

# ============================================================================
# CELL 14: MEMORY MANAGEMENT
# Optimize memory usage for Colab
# ============================================================================

# Clear GPU memory
torch.cuda.empty_cache()
print("GPU memory cleared")

# Function to check memory usage
def check_memory():
    """Check current memory usage"""
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Current GPU memory: {current_memory:.2f} GB")
        print(f"Peak GPU memory: {max_memory:.2f} GB")
    else:
        print("CUDA not available")

check_memory()

# Tips for memory optimization:
print("\nMemory optimization tips:")
print("1. Reduce batch_size if you get out-of-memory errors")
print("2. Reduce block_size (sequence length) to save memory")
print("3. Use gradient checkpointing for very deep models")
print("4. Clear intermediate variables with 'del variable_name'")
print("5. Use mixed precision training (torch.cuda.amp)")

# ============================================================================
# CELL 15: FINAL DEMONSTRATION
# Complete example of using the trained model
# ============================================================================

print("=" * 60)
print("FINAL DEMONSTRATION OF GPT-120M")
print("=" * 60)

# Generate multiple samples with different settings
test_prompts = [
    "Artificial intelligence will",
    "In the future,",
    "Machine learning algorithms",
    "The development of neural networks"
]

print("\nGenerating text samples with different parameters:")
print("-" * 50)

for i, prompt in enumerate(test_prompts):
    print(f"\nSample {i+1}: '{prompt}'")
    
    # Generate with different temperatures
    for temp in [0.5, 0.8, 1.0]:
        generated = generate_text(prompt, max_tokens=30, temperature=temp, top_k=50)
        print(f"  Temperature {temp}: {generated}")

print("\n" + "=" * 60)
print("DEMONSTRATION COMPLETED!")
print("Your GPT-120M model is ready to use!")
print("=" * 60)

# ============================================================================
# TROUBLESHOOTING GUIDE
# Common issues and solutions
# ============================================================================

troubleshooting_guide = """
TROUBLESHOOTING GUIDE:

1. OUT OF MEMORY ERRORS:
   - Reduce batch_size (try 4 or 2)
   - Reduce block_size (try 256 or 128)
   - Use gradient checkpointing
   - Clear GPU memory: torch.cuda.empty_cache()

2. SLOW TRAINING:
   - Ensure you're using GPU (check device variable)
   - Reduce validation frequency
   - Use mixed precision training
   - Optimize data loading

3. POOR TEXT GENERATION:
   - Train for more epochs
   - Adjust temperature (0.7-1.0 usually works well)
   - Try different sampling methods (top-k, nucleus)
   - Ensure your training data is high quality

4. CHECKPOINT ISSUES:
   - Make sure to save checkpoints regularly
   - Check file paths are correct
   - Verify you have write permissions

5. COLAB DISCONNECTION:
   - Save checkpoints frequently
   - Use Google Drive to persist files
   - Consider using Colab Pro for longer runtimes

6. IMPORT ERRORS:
   - Install required packages: !pip install torch transformers datasets
   - Restart runtime if needed
   - Check Python version compatibility
"""

print(troubleshooting_guide)