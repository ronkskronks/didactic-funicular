# Enhanced Trainable GPT - Usage Guide

## 🎛️ NOW YOUR C GPT IS TWEAKABLE AT RUNTIME!

No more recompiling to change parameters! This enhanced version is fully configurable.

## Quick Start

```bash
# Compile once
gcc enhanced_trainable_gpt.c -lm -o compiled_models/enhanced_gpt

# Train with defaults
./compiled_models/enhanced_gpt data/sample_training_data.txt

# Use custom config file
./compiled_models/enhanced_gpt -c gpt_config.txt data/sample_training_data.txt
```

## All Command Line Options

```bash
./compiled_models/enhanced_gpt [options] <training_file>

-c, --config FILE     Load configuration from file
-l, --load FILE       Load pre-trained model  
-s, --save FILE       Save model after training
-e, --epochs N        Number of training epochs
-r, --lr RATE         Learning rate
-t, --temp TEMP       Generation temperature
-g, --generate N      Number of tokens to generate
-i, --interactive     Start interactive mode
-v, --verbose         Verbose output
-h, --help            Show help
```

## Configuration File Format

Create a `config.txt` file:
```
max_vocab = 150
context_len = 12
d_model = 24
learning_rate = 0.008
epochs = 15
temperature = 0.8
```

## Example Workflows

### 1. Quick Experiment
```bash
# Small, fast model for testing
./compiled_models/enhanced_gpt -e 5 -r 0.02 -t 1.2 data/sample_training_data.txt
```

### 2. Save & Load Models  
```bash
# Train and save
./compiled_models/enhanced_gpt -s my_model.dat data/sample_training_data.txt

# Load and continue with interactive mode
./compiled_models/enhanced_gpt -l my_model.dat -i
```

### 3. Interactive Mode
```bash
# Enter interactive mode after training
./compiled_models/enhanced_gpt -i data/sample_training_data.txt

# Then in interactive mode:
> temp 0.5          # Set temperature
> gen 20            # Set generation length  
> hello world       # Generate from "hello world"
> quit              # Exit
```

### 4. Different Model Sizes
```bash
# Tiny model (fast)
echo "max_vocab = 50
context_len = 6  
d_model = 12
epochs = 5" > tiny_config.txt

./compiled_models/enhanced_gpt -c tiny_config.txt data/sample_training_data.txt

# Large model (slow but better)
echo "max_vocab = 300
context_len = 16
d_model = 32
d_ff = 64
epochs = 25" > large_config.txt

./compiled_models/enhanced_gpt -c large_config.txt data/sample_training_data.txt
```

## What's New vs Original

### ✅ Dynamic Configuration
- Runtime parameter tweaking
- Config files for easy experimentation
- Command line overrides

### ✅ Model Persistence
- Save trained models to binary files
- Load pre-trained models
- Continue training from checkpoints

### ✅ Interactive Mode
- Real-time text generation
- Dynamic temperature adjustment
- Live parameter tweaking

### ✅ Advanced Sampling
- Temperature-based sampling
- Greedy vs random generation
- Configurable output length

### ✅ Better Memory Management
- Dynamic allocation based on config
- Proper cleanup
- Scalable architecture

## Experimentation Ideas

1. **Architecture Search**: Try different combinations of d_model, layers, heads
2. **Learning Rate Schedules**: Start high, gradually lower
3. **Temperature Effects**: See how 0.1 vs 2.0 affects creativity
4. **Context Length**: How does 4 vs 16 tokens affect coherence?
5. **Vocabulary Size**: Trade-off between speed and expressiveness

## Performance Notes

- **Smaller models** (d_model=12, vocab=50): Train in seconds
- **Medium models** (d_model=24, vocab=150): Train in minutes  
- **Large models** (d_model=32+, vocab=300+): Train in tens of minutes

## File Organization

```
/home/kronks/pinocchio/qwen/
├── enhanced_trainable_gpt.c      # Source code
├── compiled_models/enhanced_gpt   # Compiled binary
├── gpt_config.txt                # Sample config
├── data/sample_training_data.txt  # Training data
└── models/                       # Saved models
    ├── my_model.dat
    └── experiment_1.dat
```

Now you can experiment with ML architectures in C without constant recompilation!