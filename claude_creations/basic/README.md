# Basic Claude Creations
*Where it all began*

**Created by Claude Code**

The humble beginnings of our C neural network journey. These implementations prove that you don't need fancy frameworks to build working AI - just patience, coffee, and a questionable amount of manual memory management.

## The Models

### `tiny_nn.c`
**Parameters:** 5 (yes, five)
**Purpose:** Learning the absolute basics

A neural network so small it fits in your head. Perfect for understanding forward propagation without getting lost in the details.

```bash
gcc tiny_nn.c -lm -o tiny_nn
./tiny_nn
```

### `simple_chatbot.c`
**Parameters:** ~3,000
**Purpose:** Your first conversational AI

A chatbot that actually responds to you. It's not Shakespeare, but it's trying its best.

```bash
gcc simple_chatbot.c -lm -o simple_chatbot
./simple_chatbot data/chatbot_training.txt
```

### `trainable_gpt.c`
**Parameters:** ~15,000
**Purpose:** A GPT you can actually train

A transformer implementation that you can feed your own data and watch it learn. It's like having a pet AI that gets smarter over time.

```bash
gcc trainable_gpt.c -lm -o trainable_gpt
./trainable_gpt your_training_data.txt
```

## Learning Path
1. Start with `tiny_nn.c` to understand the basics
2. Move to `simple_chatbot.c` for conversation
3. Graduate to `trainable_gpt.c` for real training

## Common Issues
- Segfaults (welcome to C!)
- Memory leaks (also welcome to C!)
- "Why didn't I just use Python?" (that's the spirit!)

## Success Stories
- Actually working neural networks
- Functioning chatbots
- Increased appreciation for garbage collection
- Deep understanding of matrix operations

---
*"Every expert was once a beginner who refused to give up"*