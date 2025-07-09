# MLC - Machine Learning in C
*Because Python wasn't challenging enough*

**Created by Gemini CLI**

A complete machine learning framework implemented in C. Because apparently someone thought "let's make neural networks harder by doing them in C with manual memory management."

## What's Here

### Core Implementation
- `neural_network.c` / `neural_network.h` - The main neural network engine
- `main.c` - Training and inference interface
- `Makefile` - Build system

### Trained Models
- `big_model.bin` - Large trained network
- `huge_model.bin` - Even larger network
- `serious_model.bin` - Production-ready model

### Executables
- `nn_trainer` - Training executable
- `trainer` - Alternative trainer

## Features
- **Pure C implementation** - No external ML libraries
- **Binary model format** - Efficient storage and loading
- **Makefile build system** - Just `make` and go
- **Multiple model sizes** - From big to huge to serious

## Building
```bash
cd MLC/
make
./nn_trainer
```

## Philosophy
"If you're going to suffer through C pointers and manual memory management, might as well do it while implementing backpropagation."

## Warning
This code contains:
- Manual memory allocation
- Pointer arithmetic
- Matrix operations without BLAS
- The tears of developers who chose the hard path

## Academic Value
Perfect for understanding:
- How neural networks work at the lowest level
- Why high-level frameworks exist
- The value of garbage collection
- What "close to the metal" really means

---
*"It's not about the destination, it's about appreciating the journey... and then never taking this route again"*