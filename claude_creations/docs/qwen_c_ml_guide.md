# Qwen C ML Files - Complete Guide

## Overview
You've got three awesome handcrafted C ML implementations! Here's what each one does:

## 1. `tiny_nn.c` - Ultra Minimal Neural Network
**What it is:** A super basic neural network with just 5 parameters total
**Architecture:** 2 inputs → 1 hidden node → 1 output
**Purpose:** Learning the basics of forward propagation

### Key Components:
- **Structure:** Just weights and bias stored in `TinyNN` struct
- **Activation:** Sigmoid function for non-linearity
- **Function:** Tests XOR-like behavior with hardcoded weights
- **Parameters:** Only 5 total (ultra lightweight!)

### How to run:
```bash
cd /home/kronks/pinocchio/qwen
gcc tiny_nn.c -lm -o compiled_models/tiny_nn
./compiled_models/tiny_nn
```

---

## 2. `homemade_gpt.c` - From-Scratch GPT Implementation
**What it is:** A complete GPT transformer built without any external libraries
**Architecture:** Full transformer with attention, feed-forward, embeddings
**Purpose:** Understanding how GPT actually works under the hood

### Key Components:
- **Vocabulary:** 50 tokens (tiny for vintage PC compatibility)
- **Context:** 16 tokens maximum sequence length
- **Model Size:** 32-dimensional embeddings, 4 attention heads, 2 layers
- **Parameters:** ~50,000 total parameters

### Architecture Breakdown:
1. **Token Embeddings:** Maps vocabulary IDs to vector representations
2. **Position Embeddings:** Adds positional information using sinusoidal encoding
3. **Multi-Head Attention:** 4 heads with Query/Key/Value matrices (handcrafted!)
4. **Feed Forward:** Two-layer MLP with ReLU activation
5. **Residual Connections:** Skip connections for gradient flow
6. **Causal Masking:** Prevents looking at future tokens

### Custom Functions:
- `homemade_attention()`: Pure C attention mechanism with causal masking
- `homemade_feedforward()`: Two-layer MLP implementation
- `custom_softmax_element()`: Safe softmax that won't explode
- `custom_tanh()`: Approximation that won't break vintage hardware

### How to run:
```bash
gcc homemade_gpt.c -lm -o compiled_models/homemade_gpt
./compiled_models/homemade_gpt
```

### What it does:
- Initializes a GPT model with random weights
- Takes a sequence of token IDs as input
- Runs forward pass through all transformer layers
- Predicts the next token using greedy sampling

---

## 3. `trainable_gpt.c` - Complete Training System
**What it is:** A full GPT with training loop, tokenizer, and text generation
**Architecture:** Simplified transformer that can actually learn from data
**Purpose:** Complete end-to-end language modeling system

### Key Components:
- **Tokenizer:** Converts text to tokens, builds vocabulary dynamically
- **Training Loop:** Implements backpropagation and weight updates
- **Text Generation:** Produces new text after training
- **File I/O:** Loads training data from text files

### Architecture Details:
- **Vocabulary:** Up to 100 words (expandable)
- **Context:** 8 tokens (smaller for training efficiency)
- **Model:** 16-dim embeddings, 2 heads, 1 layer, 32-dim feed-forward
- **Learning:** Cross-entropy loss with simple gradient descent

### Training Process:
1. **Tokenization:** Splits text into words, builds vocabulary
2. **Forward Pass:** Runs input through transformer layers
3. **Loss Calculation:** Cross-entropy between prediction and target
4. **Backprop:** Updates weights using gradients (simplified version)
5. **Generation:** Uses trained model to predict next words

### How to run:
```bash
gcc trainable_gpt.c -lm -o compiled_models/trainable_gpt
./compiled_models/trainable_gpt data/sample_training_data.txt
```

### Training Data Format:
Put text in a file (one sentence per line). Example:
```
the cat sat on the mat
dogs are great pets
machine learning is fun
```

---

## Compilation and Usage Summary

### Compile all models:
```bash
cd /home/kronks/pinocchio/qwen
gcc tiny_nn.c -lm -o compiled_models/tiny_nn
gcc homemade_gpt.c -lm -o compiled_models/homemade_gpt  
gcc trainable_gpt.c -lm -o compiled_models/trainable_gpt
```

### Run them:
```bash
# Basic neural network test
./compiled_models/tiny_nn

# GPT inference test
./compiled_models/homemade_gpt

# Train and generate text
./compiled_models/trainable_gpt data/sample_training_data.txt
```

---

## Learning Path Recommendation

1. **Start with `tiny_nn.c`** - Understand basic forward pass
2. **Move to `homemade_gpt.c`** - See how attention and transformers work
3. **Finish with `trainable_gpt.c`** - Learn the complete training pipeline

---

## Key Insights

### What Makes These Special:
- **No Dependencies:** Pure C with just standard math library
- **Educational:** Every operation is explicit and visible
- **Lightweight:** Designed to run on modest hardware
- **Complete:** From tokenization to generation, everything included

### Architecture Choices:
- Small dimensions for fast computation
- Simplified attention (some shortcuts taken for clarity)
- Basic gradient updates (only output layer in trainable version)
- Greedy sampling for deterministic results

### Performance Notes:
- These are educational implementations, not optimized for speed
- Perfect for understanding concepts, not production use
- Memory usage is minimal and predictable
- Training is deliberately simple to avoid complexity

---

## Next Steps to Explore:
1. **Modify hyperparameters** (vocab size, dimensions, layers)
2. **Add more training data** and see how it affects generation
3. **Implement proper backprop** for all layers in trainable version
4. **Add different sampling methods** (temperature, top-k, etc.)
5. **Experiment with different activation functions**

These implementations give you complete control and understanding of every operation in a transformer-based language model!