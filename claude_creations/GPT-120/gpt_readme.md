# GPT-120M: 120 Million Parameter Language Model

A complete implementation of a GPT (Generative Pre-trained Transformer) model with approximately 120 million parameters, specifically designed for Google Colab execution.

## Features

- **120M Parameters**: Carefully designed architecture to achieve ~120 million parameters
- **Google Colab Optimized**: No command-line arguments, modular design for notebook execution
- **Multiple Dataset Sources**: Support for Google Drive, Hugging Face, and direct downloads
- **Comprehensive Training**: Complete training pipeline with checkpointing and validation
- **Text Generation**: Various generation methods including top-k and nucleus sampling
- **Educational**: Extensive comments explaining every component

## Architecture

- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **Maximum Sequence Length**: 512 (configurable)
- **Vocabulary Size**: Dynamic based on dataset

## Quick Start

### 1. Upload Files to Google Colab

Upload both `gpt_120m_param.py` and `gpt_colab_usage_guide.py` to your Colab environment.

### 2. Run the Model

```python
# Execute the main model file
%run gpt_120m_param.py

# Create configuration
config = GPTConfig(
    vocab_size=1000,  # Will be updated
    n_embd=768,
    n_head=12,
    n_layer=12,
    block_size=512,
    dropout=0.1
)

# Load your dataset (multiple options available)
texts = your_dataset_here

# Create tokenizer and preprocess
tokenizer = SimpleTokenizer(texts)
config.vocab_size = tokenizer.vocab_size
preprocessor = DataPreprocessor(tokenizer, config.block_size)
input_ids, target_ids = preprocessor.prepare_training_data(texts)

# Create model and trainer
model = GPT(config)
trainer = GPTTrainer(model, config, device='cuda')

# Train
trainer.train(train_dataloader, val_dataloader, num_epochs=5)

# Generate text
generated = model.generate(input_ids, max_new_tokens=50)
```

## Dataset Loading Options

### Option 1: Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
texts = DatasetLoader.load_from_drive('/content/drive/MyDrive/dataset.txt')
```

### Option 2: Hugging Face
```python
texts = DatasetLoader.load_from_huggingface('wikitext', 'wikitext-2-raw-v1')
```

### Option 3: Direct Download
```python
texts = DatasetLoader.download_and_load('https://example.com/dataset.txt')
```

## Training

The model supports:
- **Automatic Checkpointing**: Save model state during training
- **Validation**: Monitor performance on validation set
- **Loss Tracking**: Record training and validation losses
- **Resume Training**: Load from checkpoints to continue training

## Text Generation

Multiple generation methods available:
- **Top-k Sampling**: Sample from top k most likely tokens
- **Nucleus Sampling**: Sample from tokens within probability mass p
- **Temperature Control**: Adjust randomness of generation

## Memory Optimization

For Google Colab's memory constraints:
- Adjustable batch sizes
- Gradient checkpointing support
- Memory usage monitoring
- Optimization tips included

## File Structure

- `gpt_120m_param.py`: Main model implementation
- `gpt_colab_usage_guide.py`: Step-by-step Colab usage guide
- `gpt_readme.md`: This documentation file

## Model Components

### Core Architecture
- **GPT**: Main model class
- **TransformerBlock**: Individual transformer layers
- **MultiHeadAttention**: Attention mechanism
- **FeedForward**: Position-wise feed-forward networks
- **PositionalEncoding**: Learnable position embeddings

### Training Infrastructure
- **GPTTrainer**: Complete training loop with checkpointing
- **DataPreprocessor**: Text to token conversion
- **SimpleTokenizer**: Character-level tokenization (extendable)

### Utilities
- **DatasetLoader**: Multi-source dataset loading
- **Generation functions**: Various text generation methods
- **Memory management**: GPU memory optimization

## Parameter Count Verification

The model automatically calculates and displays its parameter count:
```
Total parameters: ~120,000,000
Trainable parameters: ~120,000,000
```

## Usage Tips

1. **Start Small**: Begin with a small dataset to test the pipeline
2. **Monitor Memory**: Use provided memory monitoring functions
3. **Save Frequently**: Enable regular checkpointing
4. **Validate Often**: Use validation to prevent overfitting
5. **Experiment**: Try different generation parameters

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (try 4 or 2)
- Reduce `block_size` (try 256 or 128)
- Use `torch.cuda.empty_cache()`

### Slow Training
- Ensure GPU is being used
- Reduce validation frequency
- Optimize data loading

### Poor Generation
- Train for more epochs
- Adjust temperature (0.7-1.0)
- Try different sampling methods
- Improve training data quality

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- tqdm
- matplotlib (for visualization)
- transformers (optional, for advanced tokenization)
- datasets (optional, for Hugging Face datasets)

## License

This implementation is provided for educational purposes. Feel free to modify and extend for your needs.

## Educational Value

This implementation prioritizes:
- **Clarity**: Every component is thoroughly commented
- **Understanding**: Mathematical concepts explained
- **Modularity**: Easy to modify and extend
- **Practical**: Real-world training considerations

Perfect for learning transformer architecture and implementing your own language models!