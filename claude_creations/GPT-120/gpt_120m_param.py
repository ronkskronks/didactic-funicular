# ============================================================================
# GPT-120M: A 120 Million Parameter Transformer Language Model
# Designed specifically for Google Colab execution
# ============================================================================

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

# ============================================================================
# CONFIGURATION CLASS
# This dataclass holds all hyperparameters for our GPT model
# We use a dataclass to make configuration management clean and accessible
# ============================================================================

@dataclass
class GPTConfig:
    # Model architecture parameters
    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_embd: int = 768        # Embedding dimension (768 for balanced performance)
    n_head: int = 12         # Number of attention heads (must divide n_embd evenly)
    n_layer: int = 12        # Number of transformer layers
    block_size: int = 1024   # Maximum sequence length
    dropout: float = 0.1     # Dropout probability for regularization
    
    # Training parameters
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    def __post_init__(self):
        # Verify that embedding dimension is divisible by number of heads
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

# ============================================================================
# POSITIONAL ENCODING IMPLEMENTATION
# This implements learnable positional embeddings that help the model
# understand the position of tokens in a sequence (transformers have no
# inherent notion of position without this)
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Learnable positional embeddings for sequence position awareness.
    Unlike sinusoidal encodings, these are learned during training.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Create a learnable embedding table for positions
        # Each position gets its own embedding vector
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, embedding_dim)
        seq_len = x.size(1)
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        pos_ids = torch.arange(seq_len, device=x.device)
        
        # Get positional embeddings for these positions
        pos_embeddings = self.pos_emb(pos_ids)
        
        # Add positional information to token embeddings
        return x + pos_embeddings

# ============================================================================
# MULTI-HEAD ATTENTION MECHANISM
# This is the core of the transformer architecture - it allows the model
# to attend to different parts of the input sequence simultaneously
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with causal masking.
    This allows the model to focus on different aspects of the input
    simultaneously through multiple attention heads.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # Linear projections for queries, keys, and values
        # We compute all heads in parallel for efficiency
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Output projection to combine all heads
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Register causal mask as a buffer (not a parameter)
        # This ensures tokens can only attend to previous tokens
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate queries, keys, and values
        q = self.q_proj(x)  # (batch_size, seq_len, n_embd)
        k = self.k_proj(x)  # (batch_size, seq_len, n_embd)
        v = self.v_proj(x)  # (batch_size, seq_len, n_embd)
        
        # Reshape for multi-head attention
        # Split the embedding dimension across multiple heads
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Compute attention scores using scaled dot-product attention
        # Scale by sqrt(head_dim) to prevent vanishing gradients
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask to prevent looking at future tokens
        # Set future positions to negative infinity so they become 0 after softmax
        mask = self.causal_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Concatenate heads back together
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, embed_dim)
        
        # Apply output projection
        output = self.out_proj(attention_output)
        
        return output

# ============================================================================
# FEED-FORWARD NETWORK
# This is the position-wise feed-forward network that processes each position
# independently. It adds non-linearity and capacity to the model.
# ============================================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    This processes each position in the sequence independently,
    expanding the representation through a bottleneck architecture.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expand to 4x the embedding dimension (standard transformer practice)
        hidden_dim = 4 * config.n_embd
        
        # Two linear layers with GELU activation in between
        self.fc1 = nn.Linear(config.n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply first linear layer
        x = self.fc1(x)
        
        # Apply GELU activation (smoother than ReLU, preferred for transformers)
        x = F.gelu(x)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply second linear layer
        x = self.fc2(x)
        
        return x

# ============================================================================
# TRANSFORMER BLOCK
# This combines multi-head attention and feed-forward networks with
# residual connections and layer normalization - the core building block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of multi-head attention
    and feed-forward network with residual connections and layer normalization.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Layer normalization (applied before attention and feed-forward)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(config)
        
        # Feed-forward network
        self.feed_forward = FeedForward(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization architecture (norm before attention)
        # This is more stable than post-normalization
        
        # Multi-head attention with residual connection
        attention_output = self.attention(self.ln1(x))
        x = x + attention_output  # Residual connection
        
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(self.ln2(x))
        x = x + ff_output  # Residual connection
        
        return x

# ============================================================================
# MAIN GPT MODEL
# This is the complete GPT model that combines all components:
# token embeddings, positional encodings, transformer blocks, and output head
# ============================================================================

class GPT(nn.Module):
    """
    Complete GPT model with approximately 120M parameters.
    Architecture: Token Embedding + Positional Encoding + Transformer Blocks + Output Head
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings: convert token IDs to dense vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Positional encoding: add position information
        self.positional_encoding = PositionalEncoding(config)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head: project back to vocabulary size for next token prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights using a careful initialization scheme
        self.apply(self._init_weights)
        
        # Print model parameter count
        self.print_parameter_count()
    
    def _init_weights(self, module):
        """
        Initialize weights using a careful scheme for stable training.
        Different layer types get different initialization strategies.
        """
        if isinstance(module, nn.Linear):
            # Use normal distribution with std=0.02 for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use normal distribution for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer norm with ones for weight and zeros for bias
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def print_parameter_count(self):
        """
        Print the total number of parameters in the model.
        This helps verify we're close to our 120M parameter target.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Target: 120M parameters")
        print(f"Difference: {abs(total_params - 120_000_000):,}")
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the GPT model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length)
            targets: Optional target token IDs for loss calculation
            
        Returns:
            logits: Output logits of shape (batch_size, sequence_length, vocab_size)
            loss: Cross-entropy loss if targets provided, otherwise None
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert token IDs to embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Add positional information
        x = self.positional_encoding(token_embeddings)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply final layer normalization
        x = self.ln_f(x)
        
        # Project to vocabulary size for next token prediction
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy loss calculation
            # We predict the next token for each position
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1)
        
        return logits, loss
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Generate new tokens using the trained model.
        
        Args:
            input_ids: Starting token sequence
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top k tokens (if specified)
            
        Returns:
            Generated token sequence
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop input if it exceeds block size
                input_ids_cropped = input_ids[:, -self.config.block_size:]
                
                # Get logits for next token
                logits, _ = self(input_ids_cropped)
                
                # Focus on the last token's logits
                logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, indices, values)
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

# ============================================================================
# DATASET LOADING UTILITIES
# These functions provide multiple ways to load datasets in Google Colab:
# 1. From Google Drive (mounted)
# 2. From Hugging Face datasets
# 3. Direct download from URLs
# ============================================================================

class DatasetLoader:
    """
    Utility class for loading datasets in Google Colab environment.
    Supports multiple data sources and formats.
    """
    
    @staticmethod
    def mount_google_drive():
        """
        Mount Google Drive in Colab environment.
        This allows access to datasets stored in your Google Drive.
        """
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully!")
            return True
        except ImportError:
            print("Not running in Google Colab - Google Drive mounting not available")
            return False
        except Exception as e:
            print(f"Error mounting Google Drive: {e}")
            return False
    
    @staticmethod
    def load_from_drive(file_path: str, file_type: str = 'txt') -> List[str]:
        """
        Load dataset from Google Drive.
        
        Args:
            file_path: Path to file in Google Drive (e.g., '/content/drive/MyDrive/dataset.txt')
            file_type: Type of file ('txt', 'json', 'csv')
            
        Returns:
            List of text samples
        """
        try:
            if file_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Split by double newlines to separate documents
                    content = f.read()
                    samples = content.split('\n\n')
                    return [sample.strip() for sample in samples if sample.strip()]
            
            elif file_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Assume JSON contains a list of strings or objects with 'text' field
                    if isinstance(data, list):
                        if isinstance(data[0], str):
                            return data
                        elif isinstance(data[0], dict) and 'text' in data[0]:
                            return [item['text'] for item in data]
                    return []
            
            elif file_type == 'csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                # Assume there's a 'text' column
                if 'text' in df.columns:
                    return df['text'].tolist()
                else:
                    # Use the first column
                    return df.iloc[:, 0].tolist()
            
        except Exception as e:
            print(f"Error loading from Google Drive: {e}")
            return []
    
    @staticmethod
    def load_from_huggingface(dataset_name: str, split: str = 'train', text_field: str = 'text') -> List[str]:
        """
        Load dataset from Hugging Face datasets.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face
            split: Dataset split to load ('train', 'validation', 'test')
            text_field: Field name containing text data
            
        Returns:
            List of text samples
        """
        try:
            from datasets import load_dataset
            
            print(f"Loading {dataset_name} from Hugging Face...")
            dataset = load_dataset(dataset_name, split=split)
            
            # Extract text field
            texts = [item[text_field] for item in dataset]
            print(f"Loaded {len(texts)} samples")
            
            return texts
        
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            return []
    
    @staticmethod
    def download_and_load(url: str, filename: str = None) -> List[str]:
        """
        Download and load dataset from a URL.
        
        Args:
            url: URL to download from
            filename: Optional filename to save as
            
        Returns:
            List of text samples
        """
        try:
            if filename is None:
                filename = url.split('/')[-1]
            
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save file
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded {filename}")
            
            # Load based on file extension
            if filename.endswith('.txt'):
                return DatasetLoader.load_from_drive(filename, 'txt')
            elif filename.endswith('.json'):
                return DatasetLoader.load_from_drive(filename, 'json')
            elif filename.endswith('.csv'):
                return DatasetLoader.load_from_drive(filename, 'csv')
            elif filename.endswith('.zip'):
                # Extract and load first text file
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('.')
                    # Find first text file
                    for file in zip_ref.namelist():
                        if file.endswith('.txt'):
                            return DatasetLoader.load_from_drive(file, 'txt')
            
            return []
        
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return []

# ============================================================================
# TOKENIZATION AND DATA PREPROCESSING
# This handles converting raw text into token sequences that the model can process
# ============================================================================

class SimpleTokenizer:
    """
    Simple character-level tokenizer for demonstration purposes.
    In practice, you'd use a more sophisticated tokenizer like BPE or SentencePiece.
    """
    
    def __init__(self, texts: List[str]):
        # Create vocabulary from all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Sort for consistent ordering
        chars = sorted(list(all_chars))
        
        # Add special tokens
        self.vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + chars
        self.vocab_size = len(self.vocab)
        
        # Create mappings
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.id_to_char = {i: char for i, char in enumerate(self.vocab)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"First 20 tokens: {self.vocab[:20]}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        return [self.char_to_id.get(char, self.char_to_id['<unk>']) for char in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        return ''.join([self.id_to_char.get(id, '<unk>') for id in token_ids])

class DataPreprocessor:
    """
    Handles data preprocessing for GPT training.
    Converts raw text into training sequences.
    """
    
    def __init__(self, tokenizer: SimpleTokenizer, block_size: int = 1024):
        self.tokenizer = tokenizer
        self.block_size = block_size
    
    def prepare_training_data(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw texts into training sequences.
        
        Args:
            texts: List of text samples
            
        Returns:
            input_ids: Input token sequences
            target_ids: Target token sequences (shifted by 1)
        """
        all_token_ids = []
        
        print("Tokenizing texts...")
        for text in tqdm(texts):
            # Add beginning of sequence token
            tokens = [self.tokenizer.char_to_id['<bos>']]
            
            # Tokenize text
            tokens.extend(self.tokenizer.encode(text))
            
            # Add end of sequence token
            tokens.append(self.tokenizer.char_to_id['<eos>'])
            
            all_token_ids.extend(tokens)
        
        print(f"Total tokens: {len(all_token_ids)}")
        
        # Split into chunks of block_size + 1 (for input and target)
        sequences = []
        for i in range(0, len(all_token_ids) - self.block_size, self.block_size):
            seq = all_token_ids[i:i + self.block_size + 1]
            sequences.append(seq)
        
        print(f"Created {len(sequences)} training sequences")
        
        # Convert to tensors
        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        
        # Split into input and target
        input_ids = sequences_tensor[:, :-1]  # All but last token
        target_ids = sequences_tensor[:, 1:]   # All but first token
        
        return input_ids, target_ids

# ============================================================================
# TRAINING UTILITIES
# This section contains utilities for training the model including
# the trainer class, checkpointing, and training loop
# ============================================================================

class GPTTrainer:
    """
    Trainer class for GPT model with checkpointing and logging.
    """
    
    def __init__(self, model: GPT, config: GPTConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """
        Save model checkpoint with training state.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint and restore training state.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Single training step.
        """
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        logits, loss = self.model(input_ids, target_ids)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        self.step += 1
        
        return loss.item()
    
    def validate(self, val_dataloader) -> float:
        """
        Validation step.
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, loss = self.model(input_ids, target_ids)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs: int = 10, 
              save_every: int = 1000, validate_every: int = 500):
        """
        Main training loop.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training loop
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss:.4f}', 'step': self.step})
                
                # Save checkpoint
                if self.step % save_every == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.step}.pth')
                
                # Validation
                if val_dataloader and self.step % validate_every == 0:
                    val_loss = self.validate(val_dataloader)
                    self.val_losses.append(val_loss)
                    
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(f'checkpoint_step_{self.step}.pth', is_best=True)
                    
                    print(f"Step {self.step}: Val Loss = {val_loss:.4f}")
            
            # Record epoch statistics
            avg_epoch_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_epoch_loss)
            
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        print("Training completed!")

# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# This section shows how to use all the components together
# ============================================================================

def create_sample_dataset() -> List[str]:
    """
    Create a small sample dataset for demonstration.
    In practice, you'd load a much larger dataset.
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized the field of natural language processing.",
        "GPT models are autoregressive language models based on the transformer architecture.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models can generate coherent and contextually relevant text.",
        "Training neural networks requires careful tuning of hyperparameters.",
        "The field of AI is rapidly evolving with new breakthroughs every year."
    ]
    
    # Repeat to create more training data
    return sample_texts * 100

def main_demo():
    """
    Main demonstration function showing how to use the GPT model.
    This would typically be run in separate Colab cells.
    """
    print("=" * 50)
    print("GPT-120M DEMONSTRATION")
    print("=" * 50)
    
    # Step 1: Create configuration
    print("\n1. Creating model configuration...")
    config = GPTConfig(
        vocab_size=1000,  # Will be updated based on actual vocabulary
        n_embd=768,
        n_head=12,
        n_layer=12,
        block_size=256,   # Smaller for demo
        dropout=0.1
    )
    
    # Step 2: Load dataset
    print("\n2. Loading dataset...")
    # You can use any of these methods:
    
    # Option A: Load from Google Drive
    # DatasetLoader.mount_google_drive()
    # texts = DatasetLoader.load_from_drive('/content/drive/MyDrive/dataset.txt')
    
    # Option B: Load from Hugging Face
    # texts = DatasetLoader.load_from_huggingface('wikitext', 'wikitext-2-raw-v1')
    
    # Option C: Download from URL
    # texts = DatasetLoader.download_and_load('https://example.com/dataset.txt')
    
    # Option D: Use sample dataset
    texts = create_sample_dataset()
    
    print(f"Loaded {len(texts)} text samples")
    
    # Step 3: Create tokenizer
    print("\n3. Creating tokenizer...")
    tokenizer = SimpleTokenizer(texts)
    
    # Update config with actual vocabulary size
    config.vocab_size = tokenizer.vocab_size
    
    # Step 4: Preprocess data
    print("\n4. Preprocessing data...")
    preprocessor = DataPreprocessor(tokenizer, config.block_size)
    input_ids, target_ids = preprocessor.prepare_training_data(texts)
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(input_ids, target_ids)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Step 5: Create model
    print("\n5. Creating GPT model...")
    model = GPT(config)
    
    # Step 6: Create trainer
    print("\n6. Setting up trainer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = GPTTrainer(model, config, device)
    
    # Step 7: Train model
    print("\n7. Training model...")
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=2,  # Small for demo
        save_every=100,
        validate_every=50
    )
    
    # Step 8: Generate text
    print("\n8. Generating text...")
    model.eval()
    
    # Create a starting sequence
    start_text = "Machine learning"
    start_ids = tokenizer.encode(start_text)
    input_ids = torch.tensor([start_ids], dtype=torch.long).to(device)
    
    # Generate continuation
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    print(f"Generated text: {generated_text}")
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    main_demo()